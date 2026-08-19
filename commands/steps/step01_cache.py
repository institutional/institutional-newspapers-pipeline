import os
import tarfile
import tempfile
import shutil
import io
import time
from pathlib import Path
from typing import NamedTuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

import click
from PIL import Image
from loguru import logger
import imagehash

import utils
from models import PipelineBatch, Issue, Scan
from const import (
    CPUS_LIMIT,
    MAX_S3_CONCURRENCY,
    SCAN_JPEG_QUALITY,
    SCAN_TEMP_DIR_PATH,
    STEP01_CACHE_MAX_CONSECUTIVE_FAILURES,
    STEP01_CACHE_FAILURE_PAUSE_MINUTES,
)


class ScanDownload(NamedTuple):
    issue_id: int
    corpus: str
    archive_filename: str
    scan_filename: str
    existing_cache_key: str | None
    raw_bytes_path: str


@click.command("step01-cache")
@click.option(
    "--pipeline-batch-id",
    type=int,
    required=True,
)
def step01_cache(pipeline_batch_id: int):
    """
    Pulls issues from remote storage, process them and add them to cache
    for the current pipeline batch.
    Creates `Scan` records if they don't already exist.

    Uses a three-phase pipeline:
    1. ThreadPool downloads archives from S3 and extracts raw bytes to disk (I/O-bound)
    2. ProcessPool runs image processing across all scans (CPU-bound)
    3. Sequential cache writes and DB bulk inserts (I/O-bound)

    Includes a circuit breaker: if STEP01_CACHE_MAX_CONSECUTIVE_FAILURES archives
    fail to download in a row, the step pauses for STEP01_CACHE_FAILURE_PAUSE_MINUTES
    and restarts from the beginning.
    """
    while True:
        should_restart = _run_step01_cache(pipeline_batch_id)
        if not should_restart:
            break

        logger.warning(
            f"Pausing for {STEP01_CACHE_FAILURE_PAUSE_MINUTES} minutes "
            f"before restarting step01_cache."
        )
        time.sleep(STEP01_CACHE_FAILURE_PAUSE_MINUTES * 60)


def _run_step01_cache(pipeline_batch_id: int) -> bool:
    """Runs the cache step. Returns True if the circuit breaker tripped and the step should be restarted after a pause."""
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    items = pipeline_batch.items

    issue_ids = [item.issue_id for item in items]
    all_scans = list(Scan.select(Scan, Issue).join(Issue).where(Scan.issue << issue_ids))

    scans_by_issue: dict[int, list[Scan]] = {}
    for scan in all_scans:
        scans_by_issue.setdefault(scan.issue_id, []).append(scan)

    SCAN_TEMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=SCAN_TEMP_DIR_PATH)

    try:
        # Phase 1: Download and extract (I/O-bound)
        scan_downloads: list[ScanDownload] = []
        circuit_breaker_tripped = False

        with ThreadPoolExecutor(max_workers=MAX_S3_CONCURRENCY) as executor:
            futures = {}

            for item in items:
                issue = item.issue
                existing_scans = scans_by_issue.get(issue.id, [])
                future = executor.submit(
                    _download_and_extract_issue,
                    issue,
                    existing_scans,
                    temp_dir,
                )
                futures[future] = issue

            consecutive_failures = 0

            for future in as_completed(futures):
                issue = futures[future]

                try:
                    downloads = future.result()
                    scan_downloads.extend(downloads)
                    consecutive_failures = 0

                    if downloads:
                        logger.info(
                            f"{issue.archive_filename} ({issue.corpus}): "
                            f"archive downloaded, {len(downloads)} scans extracted."
                        )
                except Exception:
                    consecutive_failures += 1
                    logger.debug(traceback.format_exc())
                    logger.error(
                        f"Could not download {issue.archive_filename} "
                        f"({issue.corpus}). Skipping."
                    )

                    if consecutive_failures >= STEP01_CACHE_MAX_CONSECUTIVE_FAILURES:
                        logger.warning(
                            f"{consecutive_failures} consecutive download failures. "
                            f"Circuit breaker triggered."
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        circuit_breaker_tripped = True
                        break

        if circuit_breaker_tripped:
            return True

        if not scan_downloads:
            return False

        # Phase 2: Process scans (CPU-bound, true parallelism via ProcessPool)
        chunk_size = max(1, len(scan_downloads) // CPUS_LIMIT)
        path_chunks = [
            [dl.raw_bytes_path for dl in scan_downloads[i : i + chunk_size]]
            for i in range(0, len(scan_downloads), chunk_size)
        ]

        all_results: list[tuple[bytes, int, int, str] | None] = []
        with ProcessPoolExecutor(max_workers=CPUS_LIMIT) as executor:
            for result_batch in executor.map(_process_scan_chunk, path_chunks):
                all_results.extend(result_batch)

        # Filter out scans that failed to process, keeping downloads aligned
        paired = [
            (dl, result) for dl, result in zip(scan_downloads, all_results) if result is not None
        ]
        if not paired:
            return False

        scan_downloads_ok, results_ok = zip(*paired)

        processed_by_archive: dict[tuple[str, str], int] = {}
        for download in scan_downloads_ok:
            key = (download.archive_filename, download.corpus)
            processed_by_archive[key] = processed_by_archive.get(key, 0) + 1

        for (archive_filename, corpus), count in processed_by_archive.items():
            logger.info(f"{archive_filename} ({corpus}): {count} scans processed.")

        # Phase 3: Write to cache and build Scan records
        cache = utils.get_cache()
        scans_to_create: list[Scan] = []

        for download, (jpeg_bytes, width, height, phash) in zip(scan_downloads_ok, results_ok):
            if download.existing_cache_key is not None:
                cache.set(download.existing_cache_key, jpeg_bytes)
            else:
                scan = Scan(
                    issue=download.issue_id,
                    scan_filename=download.scan_filename,
                    width=width,
                    height=height,
                    phash=phash,
                )
                scans_to_create.append(scan)
                cache.set(scan.cache_key, jpeg_bytes)

        if scans_to_create:
            utils.process_db_write_batch(Scan, entries_to_create=scans_to_create)

        # Log per-archive DB and re-cache summaries
        created_by_issue: dict[int, int] = {}
        recached_by_issue: dict[int, int] = {}

        for scan in scans_to_create:
            created_by_issue[scan.issue_id] = created_by_issue.get(scan.issue_id, 0) + 1

        for download in scan_downloads_ok:
            if download.existing_cache_key is not None:
                recached_by_issue[download.issue_id] = (
                    recached_by_issue.get(download.issue_id, 0) + 1
                )

        for item in items:
            issue = item.issue
            created = created_by_issue.get(issue.id, 0)
            recached = recached_by_issue.get(issue.id, 0)

            if created or recached:
                parts = []
                if created:
                    parts.append(f"{created} scans added to database")
                if recached:
                    parts.append(f"{recached} scans re-cached")

                logger.info(f"{issue.archive_filename} ({issue.corpus}): " f"{', '.join(parts)}.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return False


def _download_and_extract_issue(
    issue: Issue,
    existing_scans: list[Scan],
    temp_dir: str,
) -> list[ScanDownload]:
    """
    Downloads an issue's archive from S3 and extracts raw image bytes to disk.
    Skips scans that are already fully cached.
    """
    Image.MAX_IMAGE_PIXELS = None

    existing_scan_keys = {scan.scan_filename: scan.cache_key for scan in existing_scans}
    cache = utils.get_cache()
    cached_keys = {key for key in existing_scan_keys.values() if key in cache}

    if existing_scans and len(cached_keys) == len(existing_scans):
        logger.info(
            f"{issue.archive_filename} ({issue.corpus}): "
            f"{len(existing_scans)} scans already in cache. Skipping."
        )
        return []

    s3 = utils.get_s3_client(issue.corpus)
    bucket_name = os.getenv(f"{issue.corpus}_S3_BUCKET_NAME")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = s3.get_object(
                Bucket=bucket_name,
                Key=issue.archive_filename,
            )
            archive_bytes = response["Body"].read()
            break
        except Exception:
            if attempt == max_retries:
                raise
            logger.warning(
                f"{issue.archive_filename}: download attempt {attempt}/{max_retries} failed."
                " Retrying..."
            )
            time.sleep(attempt)

    downloads: list[ScanDownload] = []

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        members = sorted(tar.getmembers(), key=lambda m: m.name)

        for member in members:
            if not member.isfile():
                continue

            scan_filename = str(Path(member.name))

            if scan_filename in existing_scan_keys:
                if existing_scan_keys[scan_filename] in cached_keys:
                    logger.info(f"{existing_scan_keys[scan_filename]} already in cache. Skipping.")
                    continue

            with tar.extractfile(member) as fh:
                raw_bytes = fh.read()

            fd, temp_path = tempfile.mkstemp(dir=temp_dir)
            try:
                os.write(fd, raw_bytes)
            finally:
                os.close(fd)
            del raw_bytes

            downloads.append(
                ScanDownload(
                    issue_id=issue.id,
                    corpus=issue.corpus,
                    archive_filename=issue.archive_filename,
                    scan_filename=scan_filename,
                    existing_cache_key=existing_scan_keys.get(scan_filename),
                    raw_bytes_path=temp_path,
                )
            )

    del archive_bytes
    return downloads


def _process_scan_chunk(
    paths: list[str],
) -> list[tuple[bytes, int, int, str] | None]:
    """Worker function for ProcessPoolExecutor: processes a chunk of raw scan files.
    Returns None for scans that could not be decoded.
    """
    results: list[tuple[bytes, int, int, str] | None] = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            # os.remove(path)
            results.append(_process_and_encode(raw_bytes))
        except Exception:
            logger.debug(traceback.format_exc())
            logger.warning(f"Could not process scan at {path}. Skipping.")
            results.append(None)
    return results


def _process_and_encode(
    image_bytes: bytes,
) -> tuple[bytes, int, int, str]:
    """Processes raw image bytes and returns JPEG bytes, dimensions, and phash."""
    pil_image = utils.process_scan(image_bytes)

    jpeg_buffer = io.BytesIO()
    pil_image.save(jpeg_buffer, format="JPEG", quality=SCAN_JPEG_QUALITY)

    phash = str(imagehash.phash(pil_image))
    return jpeg_buffer.getvalue(), pil_image.width, pil_image.height, phash
