from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple
import os
import re
import time

import click
import pycountry
import requests
from loguru import logger

import utils
import const
from models import Issue


class ChronamMetadata(NamedTuple):
    """Newspaper-level metadata from the Chronicling America (LOC) API."""

    title: str
    city: str
    state: str
    country: str
    publisher: str
    year_start: int | None
    year_end: int | None
    access_restricted: bool | None
    language: str


@click.command("build")
@click.option(
    "--max-metadata-requests",
    type=int,
    default=8,
    help="Maximum number of parallel metadata APIs requests.",
)
@click.option(
    "--ignore-cache",
    is_flag=True,
    help="If set, bypasses cached metadata and re-fetches from the API.",
)
def build(max_metadata_requests: int, ignore_cache: bool = False):
    """
    Populates the database with a list of all available issues for all corpora.
    Lists archives via S3 pagination, then fetches newspaper-level metadata from the LOC API using a thread pool.
    Updates existing records with fresher metadata.
    """
    for corpus in const.CORPORA:
        filenames: list[str] = []
        filesizes: list[int] = []

        entries_to_create: list[Issue] = []
        entries_to_update: list[Issue] = []

        logger.info(f"Listing available issues for corpus {corpus} ...")
        filenames, filesizes = list_issues(corpus)
        logger.info(f"Found {len(filenames)} issue archives for corpus {corpus}")

        logger.info(f"Pulling metadata for {corpus} issues ...")
        with ThreadPoolExecutor(max_workers=max_metadata_requests) as executor:
            futures = []

            for i, filename in enumerate(filenames):
                filesize = filesizes[i]
                future = executor.submit(
                    pull_issue_metadata,
                    corpus,
                    filename,
                    filesize,
                    ignore_cache,
                )
                futures.append(future)

            for future in as_completed(futures):
                issue, is_new = future.result()

                if is_new:
                    entries_to_create.append(issue)
                else:
                    entries_to_update.append(issue)

        utils.process_db_write_batch(
            model=Issue,
            entries_to_create=entries_to_create,
            entries_to_update=entries_to_update,
            fields_to_update=[
                Issue.archive_size_bytes,
                Issue.title,
                Issue.city,
                Issue.state,
                Issue.country,
                Issue.publisher,
                Issue.year_start,
                Issue.year_end,
                Issue.loc_access_restricted,
                Issue.language,
            ],
        )

    logger.info("Ready")


def list_issues(corpus: str) -> tuple[list[str], list[int]]:
    """
    Lists all newspaper archives available on remote storage as well as their size in bytes.
    """
    if corpus not in const.CORPORA:
        raise Exception(
            f"Corpus {corpus} does not exist. Possible values: {', '.join(const.CORPORA)}"
        )

    filenames: list[str] = []
    filesizes: list[int] = []

    s3 = utils.get_s3_client(corpus)
    bucket_name = os.environ[f"{corpus}_S3_BUCKET_NAME"]

    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket_name):
        contents = page.get("Contents", [])

        for obj in contents:
            if obj.get("Key", None) is None:
                continue

            if not obj["Key"].endswith(".tar.gz"):
                continue

            filenames.append(obj["Key"])
            filesizes.append(obj["Size"])

    return filenames, filesizes


def pull_issue_metadata(
    corpus: str,
    filename: str,
    filesize: int,
    ignore_cache: bool = False,
) -> tuple[Issue, bool]:
    """
    Retrieves or creates an Issue record for the given archive file,
    populating it with metadata parsed from the filename and the LOC API.
    """
    existing = Issue.get_or_none((Issue.corpus == corpus) & (Issue.archive_filename == filename))
    is_new = existing is None
    issue = existing or Issue()

    issue.corpus = corpus
    issue.archive_filename = filename
    issue.archive_size_bytes = filesize

    if corpus == "BPL":
        if not re.match(r"^[a-z]+\d+_\d{10}\.tar\.gz$", filename):
            raise ValueError(
                f"Invalid BPL archive filename: '{filename}'. "
                "Expected format: {{lccn}}_{{YYYYMMDDEE}}.tar.gz."
            )

        stem = filename.removesuffix(".tar.gz")
        lccn, edition_slug = stem.split("_", maxsplit=1)

        issue.newspaper_id = lccn
        issue.newspaper_id_type = "lccn"
        issue.edition_slug = edition_slug
        issue.edition_slug_type = "YYYYMMDDEE"
        issue.year = int(edition_slug[0:4])
        issue.month = int(edition_slug[4:6])
        issue.day = int(edition_slug[6:8])
        issue.edition_number = int(edition_slug[8:10])

        metadata = _fetch_chronam_metadata(lccn, ignore_cache=ignore_cache)
        if metadata:
            issue.title = metadata.title
            issue.city = metadata.city
            issue.state = metadata.state
            issue.country = metadata.country
            issue.publisher = metadata.publisher
            issue.year_start = metadata.year_start
            issue.year_end = metadata.year_end
            issue.loc_access_restricted = metadata.access_restricted
            issue.language = metadata.language

    return issue, is_new


def _fetch_chronam_metadata(
    lccn: str,
    ignore_cache: bool = False,
) -> ChronamMetadata | None:
    """Fetches newspaper-level metadata from the LOC API, with disk caching."""
    with utils.get_cache() as cache:
        cache_key = f"chronam_metadata_{lccn}"

        cached = cache.get(cache_key) if not ignore_cache else None

        if cached is not None:
            logger.info(f"Metadata for {lccn} retrieved from cache.")
            return cached

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(f"https://www.loc.gov/item/{lccn}/?fo=json")
                if response.status_code == 404:
                    cache.set(cache_key, False)
                    return None
                response.raise_for_status()
                break
            except Exception:
                if attempt == max_retries:
                    raise
                logger.warning(
                    f"Metadata fetch for {lccn}: attempt {attempt}/{max_retries} failed."
                    " Retrying..."
                )
                time.sleep(attempt)

        item = response.json().get("item", {})

        # Location disambiguation: pick the value appearing earliest in item.location
        location = item.get("location", [])
        raw_city = _pick_chronam_location_value(item.get("location_city", []), location)
        raw_state = _pick_chronam_location_value(item.get("location_state", []), location)
        raw_country = _pick_chronam_location_value(item.get("location_country", []), location)

        # Normalize location fields
        city = raw_city.strip().title() if raw_city else ""
        try:
            state = pycountry.subdivisions.lookup(raw_state.strip()).code.split("-")[1]
        except LookupError:
            state = raw_state.strip().upper()[:2] if raw_state else ""
        try:
            country = pycountry.countries.lookup(raw_country.strip()).alpha_3
        except LookupError:
            country = raw_country.strip().upper()[:3] if raw_country else ""

        # Publication year range
        dates_pub = item.get("dates_of_publication", "")

        year_start = None
        year_end = None

        if dates_pub and "-" in dates_pub:
            parts = dates_pub.split("-")
            try:
                year_start = int(parts[0].strip())
            except ValueError:
                pass
            try:
                year_end = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else None
            except ValueError:
                pass

        # Publisher
        created_published = item.get("created_published", [])
        publisher = created_published[0] if created_published else ""

        # Language: take the first reported language, convert to ISO 639-3
        raw_languages = item.get("language", [])
        language = ""
        if raw_languages:
            try:
                language = pycountry.languages.lookup(raw_languages[0].strip()).alpha_3
            except LookupError:
                language = ""

        result = ChronamMetadata(
            title=item.get("title", ""),
            city=city,
            state=state,
            country=country,
            publisher=publisher,
            year_start=year_start,
            year_end=year_end,
            access_restricted=item.get("access_restricted"),
            language=language,
        )

        logger.info(f"Metadata for {lccn} retrieved from Chronam API.")

        cache.set(
            cache_key,
            result,
            expire=60 * 60 * 24 * 7,
        )

        return result


def _pick_chronam_location_value(values: list[str], location: list[str]) -> str:
    """
    Given a list of location values and the full item.location list, returns the value appearing earliest in the location list.
    Falls back to the first value.
    """
    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    best_value = values[0]
    best_index = len(location)

    for value in values:
        try:
            index = location.index(value.lower())
            if index < best_index:
                best_index = index
                best_value = value
        except ValueError:
            continue

    return best_value
