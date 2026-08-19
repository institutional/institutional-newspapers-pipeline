import os
import json
import time
import threading
import traceback
from pathlib import Path
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import click
import peewee
import pyarrow as pa
import pyarrow.parquet as pq
from boto3.s3.transfer import TransferConfig
from loguru import logger

import utils
import const
from utils import record_loaders
from models import Issue, Scan, Crop


class ThesauriTerm(TypedDict):
    term: str
    count: int


class ThesauriCategory(TypedDict):
    category: str
    terms: list[ThesauriTerm]


@click.command("export")
@click.argument("corpus", type=click.Choice(const.CORPORA))
@click.option(
    "--pre-cutoff-only/--no-pre-cutoff-only",
    default=True,
    help="Only export issues published before EXPORT_CUTOFF_YEAR.",
)
@click.option(
    "--build-workers",
    type=int,
    default=const.EXPORT_BUILD_WORKERS,
    help="Chunks built concurrently. Each divides CPUS_LIMIT for its internal image ProcessPool.",
)
@click.option(
    "--max-inflight-chunks",
    type=int,
    default=const.EXPORT_MAX_INFLIGHT_CHUNKS,
    help="Max built-but-not-yet-uploaded chunks allowed on disk at once.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip chunks already uploaded (with a matching manifest) to both destinations.",
)
@click.option(
    "--test-run",
    is_flag=True,
    default=False,
    help="Build and upload only the first chunk, then stop.",
)
def export(
    corpus: str,
    pre_cutoff_only: bool,
    build_workers: int,
    max_inflight_chunks: int,
    resume: bool,
    test_run: bool,
):
    """
    Prepares the releasable dataset as Parquet chunks (one row per scan) and uploads them to R2 and
    Hugging Face. Chunks are built and uploaded in parallel, and each is deleted from disk as soon
    as it is uploaded.

    Issues are packed whole into chunks so every archive is downloaded exactly once.
    """
    if corpus not in const.CORPUS_METADATA_SOURCE:
        logger.error(f"No metadata source configured for corpus '{corpus}'.")
        click.get_current_context().exit(1)
    metadata_source = const.CORPUS_METADATA_SOURCE[corpus]

    chunks = _plan_chunks(corpus, pre_cutoff_only)
    if not chunks:
        logger.error(f"No scans to export for corpus '{corpus}'.")
        click.get_current_context().exit(1)

    if test_run:
        chunks = chunks[:1]
        logger.info("Test run: only the first chunk will be processed.")

    build_workers = max(1, build_workers)
    logger.info(
        f"Planned {len(chunks)} chunk(s) for corpus '{corpus}'. "
        f"Building {build_workers} at a time (shared {const.CPUS_LIMIT}-worker image pool)."
    )

    s3 = utils.get_s3_client("RELEASE", max_pool_connections=const.EXPORT_S3_MAX_POOL_CONNECTIONS)
    transfer_config = _transfer_config()
    hf_api = _get_hf_api()
    hf_api.create_repo(repo_id=const.RELEASE_HF_DATASET, repo_type="dataset", exist_ok=True)

    r2_keys, hf_keys = _list_uploaded_keys(s3, hf_api, corpus)

    const.EXPORT_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # Bounds built-but-not-yet-uploaded chunks on disk; at least one slot of headroom per builder.
    semaphore = threading.Semaphore(max(max_inflight_chunks, build_workers))
    # Destinations still needing an upload, per chunk index (decided once, before building).
    needs_by_index: dict[int, tuple[bool, bool]] = {}
    build_futures = []
    failed_builds = 0
    failed_uploads: list[int] = []

    # Pass 1: classify every planned chunk before building, so the last already-complete shard is
    # known up front and can be force-rebuilt (see below).
    for index, issue_ids in enumerate(chunks):
        parquet_key = _chunk_key(corpus, index, "parquet")
        manifest_key = _chunk_key(corpus, index, "json")
        needs_by_index[index] = _resume_needs(
            s3, corpus, index, issue_ids, parquet_key, manifest_key, r2_keys, hf_keys, resume
        )

    # Resume fail-safe: a crash may have left the last shard truncated on either destination, so
    # rebuild and re-upload the highest already-complete chunk to both regardless of what's stored.
    complete_indices = [i for i, (r2, hf) in needs_by_index.items() if not r2 and not hf]
    force_index = max(complete_indices) if complete_indices else None
    if force_index is not None:
        needs_by_index[force_index] = (True, True)
        logger.info(
            f"Chunk {force_index:05d}: forcing rebuild of last complete shard (resume fail-safe)."
        )

    with (
        ProcessPoolExecutor(max_workers=const.CPUS_LIMIT) as image_pool,
        ThreadPoolExecutor(max_workers=build_workers) as build_pool,
        ThreadPoolExecutor(max_workers=const.MAX_S3_CONCURRENCY) as upload_pool,
    ):
        # Pass 2: build every chunk that still needs at least one destination.
        for index, issue_ids in enumerate(chunks):
            need_r2, need_hf = needs_by_index[index]
            if not need_r2 and not need_hf:
                logger.info(f"Chunk {index:05d}: already uploaded, skipping.")
                continue

            build_futures.append(
                build_pool.submit(
                    _build_chunk_guarded,
                    semaphore,
                    corpus,
                    index,
                    issue_ids,
                    metadata_source,
                    image_pool,
                )
            )

        # A single chunk must never abort the run: a multi-day export routinely sees transient 5xx
        # from HF and R2, and an escaping exception would also strand semaphore slots held by
        # built-but-unuploaded chunks, deadlocking the remaining builders.
        upload_futures: dict = {}
        for future in as_completed(build_futures):
            try:
                index, parquet_path, manifest_path = future.result()
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error("A chunk failed to build. Continuing; re-run with --resume to retry.")
                failed_builds += 1
                continue

            need_r2, need_hf = needs_by_index[index]
            upload_futures[
                upload_pool.submit(
                    _upload_and_cleanup,
                    s3,
                    hf_api,
                    parquet_path,
                    manifest_path,
                    _chunk_key(corpus, index, "parquet"),
                    _chunk_key(corpus, index, "json"),
                    need_r2,
                    need_hf,
                    index,
                    semaphore,
                    transfer_config,
                )
            ] = index

        for future in as_completed(upload_futures):
            index = upload_futures[future]
            try:
                future.result()
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error(f"Chunk {index:05d}: upload failed. Local files left in place.")
                failed_uploads.append(index)

    if failed_builds or failed_uploads:
        logger.error(
            f"{failed_builds} chunk(s) failed to build and {len(failed_uploads)} failed to upload "
            f"for corpus '{corpus}'. Re-run with --resume to retry."
        )
        if failed_uploads:
            logger.error(f"Failed upload indices: {sorted(failed_uploads)}")

    logger.info(f"Export complete. {len(chunks)} chunk(s) processed for corpus '{corpus}'.")


def _build_chunk_guarded(
    semaphore: threading.Semaphore,
    corpus: str,
    index: int,
    issue_ids: list[int],
    metadata_source: str,
    image_pool: ProcessPoolExecutor,
) -> tuple[int, Path, Path]:
    """Builds a chunk under the on-disk backpressure semaphore. Releases the semaphore on build
    failure; on success it is held until _upload_and_cleanup releases it after uploading.
    """
    semaphore.acquire()
    try:
        parquet_path, manifest_path = _build_chunk(
            corpus, index, issue_ids, metadata_source, image_pool
        )
    except Exception:
        semaphore.release()
        raise
    return index, parquet_path, manifest_path


def _plan_chunks(corpus: str, pre_cutoff_only: bool) -> list[list[int]]:
    """Returns issue-ID chunks, packing whole issues until a chunk reaches EXPORT_CHUNK_ROW_COUNT
    scans. Ordered by issue ID so chunk composition is deterministic across runs.
    """
    scan_count = peewee.fn.COUNT(Scan.id).alias("scan_count")
    query = Scan.select(Scan.issue, scan_count).join(Issue).where(Issue.corpus == corpus)
    if pre_cutoff_only:
        query = query.where(Issue.year < const.EXPORT_CUTOFF_YEAR)
    query = query.group_by(Scan.issue).order_by(Scan.issue)

    chunks: list[list[int]] = []
    current: list[int] = []
    current_scans = 0

    for row in query:
        issue_id = row.issue_id
        scan_count = row.scan_count

        if current and current_scans + scan_count > const.EXPORT_CHUNK_ROW_COUNT:
            chunks.append(current)
            current = []
            current_scans = 0

        current.append(issue_id)
        current_scans += scan_count

    if current:
        chunks.append(current)

    return chunks


def _build_chunk(
    corpus: str,
    index: int,
    issue_ids: list[int],
    metadata_source: str,
    image_pool: ProcessPoolExecutor,
) -> tuple[Path, Path]:
    """Builds one Parquet chunk (one row per scan) plus its JSON manifest. Returns their paths."""
    issues = list(Issue.select().where(Issue.id << issue_ids))
    issues_by_id = {issue.id: issue for issue in issues}

    scans = record_loaders.chunked_query(
        Scan.select(Scan, Issue).join(Issue), Scan.issue, issue_ids
    )
    scans_by_issue: dict[int, list[Scan]] = {}
    for scan in scans:
        scans_by_issue.setdefault(scan.issue_id, []).append(scan)

    scan_ids = [scan.id for scan in scans]
    crops = record_loaders.chunked_query(Crop.select(), Crop.scan, scan_ids)
    crops_by_scan: dict[int, list[Crop]] = {}
    for crop in crops:
        crops_by_scan.setdefault(crop.scan_id, []).append(crop)

    analysis = record_loaders.load_analysis_data([crop.id for crop in crops])

    images = record_loaders.fetch_and_process_scans(
        issues,
        scans_by_issue,
        image_format="WEBP",
        quality=const.SCAN_WEBP_QUALITY,
        image_pool=image_pool,
    )

    columns = _empty_columns()
    row_count = 0
    for issue_id in issue_ids:
        issue = issues_by_id[issue_id]
        issue_scans = sorted(scans_by_issue.get(issue_id, []), key=lambda s: s.scan_filename)
        for scan in issue_scans:
            image_bytes = images.get(scan.id)
            if image_bytes is None:
                logger.warning(f"Scan {scan.id} ({scan.scan_filename}) has no image. Skipping row.")
                continue
            scan_crops = sorted(
                crops_by_scan.get(scan.id, []), key=lambda c: (c.reading_order or 0, c.id)
            )
            _append_scan_row(
                columns, scan, scan_crops, analysis, issue, metadata_source, image_bytes
            )
            row_count += 1

    table = pa.table(columns, schema=_build_arrow_schema())
    table = table.replace_schema_metadata(_huggingface_schema_metadata())

    parquet_path = const.EXPORT_DIR_PATH / f"{corpus}-part-{index:05d}.parquet"
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        row_group_size=const.EXPORT_PARQUET_ROW_GROUP_SIZE,
        write_page_index=True,
    )

    manifest = {
        "corpus": corpus,
        "chunk_index": index,
        "issue_ids": issue_ids,
        "row_count": row_count,
    }
    manifest_path = const.EXPORT_DIR_PATH / f"{corpus}-part-{index:05d}.json"
    manifest_path.write_text(json.dumps(manifest))

    logger.info(f"Chunk {index:05d}: built {row_count} rows ({len(issue_ids)} issues).")
    return parquet_path, manifest_path


#
# Row building
#
def _append_scan_row(
    columns: dict[str, list],
    scan: Scan,
    scan_crops: list[Crop],
    analysis: dict[str, dict[int, object]],
    issue: Issue,
    metadata_source: str,
    image_bytes: bytes,
) -> None:
    """Appends one scan's data (scalars + per-crop lists) as a single row to the column buffers."""
    city, state, country, _ = utils.postprocess_locality(
        issue.city or "", issue.state or "", issue.country or ""
    )

    # Struct shape required by the Hugging Face Image feature (see _build_arrow_schema).
    columns["scan_image"].append({"bytes": image_bytes, "path": None})
    columns["corpus"].append(issue.corpus)
    columns["issue_id_src"].append(_issue_id(issue))
    columns["newspaper_id_src"].append(issue.newspaper_id)
    columns["newspaper_id_type_gen"].append(issue.newspaper_id_type)
    columns["page_number_gen"].append(_parse_page_number(scan.scan_filename))
    columns["scan_filename_src"].append(scan.scan_filename)
    columns["scan_width_src"].append(scan.width)
    columns["scan_height_src"].append(scan.height)
    columns["year_ext"].append(issue.year)
    columns["month_ext"].append(issue.month)
    columns["day_ext"].append(issue.day)
    columns["edition_ext"].append(issue.edition_number)
    columns["metadata_source_gen"].append(metadata_source)
    columns["city_gen"].append(city or None)
    columns["state_gen"].append(state or None)
    columns["country_gen"].append(country or None)
    columns["language_ext"].append(issue.language)

    crop_bbox: list = []
    crop_bbox_conf: list = []
    crop_vlm_ocr: list = []
    crop_tesseract_ocr: list = []
    crop_vlm_tokens: list = []
    crop_tesseract_tokens: list = []
    crop_text_analysis: list = []
    crop_classification: list = []
    crop_class_image: list = []
    crop_class_image_conf: list = []
    crop_class_text: list = []
    crop_class_text_conf: list = []
    crop_language: list = []
    crop_language_conf: list = []
    crop_ner_per: list = []
    crop_ner_per_conf: list = []
    crop_ner_loc: list = []
    crop_ner_loc_conf: list = []
    crop_ner_org: list = []
    crop_ner_org_conf: list = []
    crop_subject: list = []
    crop_subject_conf: list = []
    crop_chronam: list = []
    crop_text_emb: list = []
    crop_image_emb: list = []

    for crop in scan_crops:
        crop_id = crop.id
        crop_bbox.append(crop.bbox_xyxy)
        crop_bbox_conf.append(crop.confidence_score)

        ocr = analysis["ocr"].get(crop_id)
        vlm_text = None
        if ocr is not None and ocr.vlm_text:
            vlm_text, _ = utils.postprocess_vlm_text(ocr.vlm_text)
        crop_vlm_ocr.append(vlm_text)
        crop_tesseract_ocr.append(
            {
                "text": ocr.tesseract_text if ocr else None,
                "metadata": ocr.tesseract_metadata if ocr else None,
            }
        )

        token = analysis["token_count"].get(crop_id)
        crop_vlm_tokens.append(token.vlm_token_count if token else None)
        crop_tesseract_tokens.append(token.tesseract_token_count if token else None)

        text_analysis = analysis["text_analysis"].get(crop_id)
        crop_text_analysis.append(_text_analysis_dict(text_analysis) if text_analysis else None)

        cls = analysis["classification"].get(crop_id)
        crop_classification.append(cls.final_category if cls else None)
        crop_class_image.append(cls.image_category if cls else None)
        crop_class_image_conf.append(cls.image_confidence_score if cls else None)
        crop_class_text.append(cls.text_category if cls else None)
        crop_class_text_conf.append(cls.text_confidence_score if cls else None)

        lang = analysis["language"].get(crop_id)
        lang_code, lang_conf = _postprocess_language(lang, text_analysis, issue.language)
        crop_language.append(lang_code if lang else None)
        crop_language_conf.append(lang_conf if lang else None)

        ner = analysis["ner"].get(crop_id)
        crop_ner_per.append(ner.per_entities if ner else [])
        crop_ner_per_conf.append(ner.per_confidence_scores if ner else [])
        crop_ner_loc.append(ner.loc_entities if ner else [])
        crop_ner_loc_conf.append(ner.loc_confidence_scores if ner else [])
        crop_ner_org.append(ner.org_entities if ner else [])
        crop_ner_org_conf.append(ner.org_confidence_scores if ner else [])

        subject = analysis["subject"].get(crop_id)
        crop_subject.append(subject.ranked_labels if subject else [])
        crop_subject_conf.append(subject.scores if subject else [])

        chronam = analysis["chronam_thesauri_match"].get(crop_id)
        crop_chronam.append(_chronam_dict(chronam) if chronam else None)

        text_emb = analysis["text_static_embedding"].get(crop_id)
        crop_text_emb.append(text_emb.vlm_embedding if text_emb else None)
        image_emb = analysis["image_embedding"].get(crop_id)
        crop_image_emb.append(image_emb.embedding if image_emb else None)

    columns["crop_bbox_gen"].append(crop_bbox)
    columns["crop_bbox_conf_gen"].append(crop_bbox_conf)
    columns["crop_vlm_ocr_gen"].append(crop_vlm_ocr)
    columns["crop_tesseract_ocr_gen"].append(crop_tesseract_ocr)
    columns["crop_vlm_ocr_token_count_gen"].append(crop_vlm_tokens)
    columns["crop_tesseract_ocr_token_count_gen"].append(crop_tesseract_tokens)
    columns["crop_text_analysis_gen"].append(crop_text_analysis)
    columns["crop_classification_gen"].append(crop_classification)
    columns["crop_classification_image_only_gen"].append(crop_class_image)
    columns["crop_classification_image_only_conf_gen"].append(crop_class_image_conf)
    columns["crop_classification_text_only_gen"].append(crop_class_text)
    columns["crop_classification_text_only_conf_gen"].append(crop_class_text_conf)
    columns["crop_language_gen"].append(crop_language)
    columns["crop_language_conf_gen"].append(crop_language_conf)
    columns["crop_ner_per_gen"].append(crop_ner_per)
    columns["crop_ner_per_conf_gen"].append(crop_ner_per_conf)
    columns["crop_ner_loc_gen"].append(crop_ner_loc)
    columns["crop_ner_loc_conf_gen"].append(crop_ner_loc_conf)
    columns["crop_ner_org_gen"].append(crop_ner_org)
    columns["crop_ner_org_conf_gen"].append(crop_ner_org_conf)
    columns["crop_subject_gen"].append(crop_subject)
    columns["crop_subject_conf_gen"].append(crop_subject_conf)
    columns["crop_chronam_thesauri_matches_exp"].append(crop_chronam)
    columns["crop_text_embeddings"].append(crop_text_emb)
    columns["crop_image_embeddings"].append(crop_image_emb)


def _postprocess_language(lang, text_analysis, issue_language: str | None) -> tuple:
    """Returns (language_code, confidence) for a crop. The post-processed code is kept only when it
    actually differs from the detected code (in which case confidence is null); otherwise the
    detected code and its confidence pass through unchanged.
    """
    if lang is None or not lang.language_code:
        return (lang.language_code if lang else None, lang.confidence_score if lang else None)

    vlm_word_count = text_analysis.vlm_word_count if text_analysis else 0
    code, _ = utils.postprocess_language_detection(
        lang.language_code,
        lang.confidence_score or 0.0,
        vlm_word_count or 0,
        issue_language,
    )
    if code == lang.language_code:
        return (lang.language_code, lang.confidence_score)
    return (code, None)


def _text_analysis_dict(record) -> dict:
    return {
        "tesseract_tokenizability_score": record.tesseract_tokenizability_score,
        "tesseract_char_count": record.tesseract_char_count,
        "tesseract_word_count": record.tesseract_word_count,
        "tesseract_word_count_unique": record.tesseract_word_count_unique,
        "tesseract_word_type_token_ratio": record.tesseract_word_type_token_ratio,
        "tesseract_sentence_count": record.tesseract_sentence_count,
        "tesseract_sentence_count_unique": record.tesseract_sentence_count_unique,
        "vlm_tokenizability_score": record.vlm_tokenizability_score,
        "vlm_char_count": record.vlm_char_count,
        "vlm_word_count": record.vlm_word_count,
        "vlm_word_count_unique": record.vlm_word_count_unique,
        "vlm_word_type_token_ratio": record.vlm_word_type_token_ratio,
        "vlm_sentence_count": record.vlm_sentence_count,
        "vlm_sentence_count_unique": record.vlm_sentence_count_unique,
        "vlm_has_table": record.vlm_has_table,
        "vlm_has_markdown": record.vlm_has_markdown,
    }


def _chronam_dict(record) -> dict:
    return {
        "tesseract_matches": _thesauri_matches(record.tesseract_matches),
        "tesseract_match_count": record.tesseract_match_count,
        "tesseract_term_count": record.tesseract_term_count,
        "vlm_matches": _thesauri_matches(record.vlm_matches),
        "vlm_match_count": record.vlm_match_count,
        "vlm_term_count": record.vlm_term_count,
    }


def _thesauri_matches(matches: dict[str, dict[str, int]] | None) -> list[ThesauriCategory] | None:
    """Reshapes the stored {category: {term: count}} mapping into lists of structs, since Arrow map
    types have no equivalent in the Hugging Face `datasets` feature system.

    Preserves the null/empty distinction: None means no text was available, while an empty list
    means text was present but nothing matched.
    """
    if matches is None:
        return None
    return [
        ThesauriCategory(
            category=category,
            terms=[ThesauriTerm(term=term, count=count) for term, count in terms.items()],
        )
        for category, terms in matches.items()
    ]


def _issue_id(issue) -> str | None:
    """Composite issue identifier: newspaper ID joined with the edition slug when both are present.
    Falls back to whichever part exists, or None when neither does.
    """
    parts = [part for part in (issue.newspaper_id, issue.edition_slug) if part]
    return "_".join(parts) if parts else None


def _parse_page_number(scan_filename: str) -> int | None:
    """Infers a 1-based page number from the leading digits of a scan filename (e.g. '0001.jp2')."""
    import re

    match = re.match(r"^\D*(\d+)", Path(scan_filename).stem)
    return int(match.group(1)) if match else None


#
# Arrow schema
#
def _build_arrow_schema() -> pa.Schema:
    f32 = pa.float32()
    list_f32 = pa.list_(f32)
    list_list_f32 = pa.list_(list_f32)
    list_str = pa.list_(pa.string())
    list_list_str = pa.list_(list_str)
    list_int = pa.list_(pa.int32())

    word_struct = pa.struct(
        [("text", pa.string()), ("conf", f32), ("bbox_xyxy", pa.list_(pa.int32()))]
    )
    tesseract_struct = pa.list_(
        pa.struct([("text", pa.string()), ("metadata", pa.list_(word_struct))])
    )

    text_analysis_struct = pa.list_(
        pa.struct(
            [
                ("tesseract_tokenizability_score", f32),
                ("tesseract_char_count", pa.int32()),
                ("tesseract_word_count", pa.int32()),
                ("tesseract_word_count_unique", pa.int32()),
                ("tesseract_word_type_token_ratio", f32),
                ("tesseract_sentence_count", pa.int32()),
                ("tesseract_sentence_count_unique", pa.int32()),
                ("vlm_tokenizability_score", f32),
                ("vlm_char_count", pa.int32()),
                ("vlm_word_count", pa.int32()),
                ("vlm_word_count_unique", pa.int32()),
                ("vlm_word_type_token_ratio", f32),
                ("vlm_sentence_count", pa.int32()),
                ("vlm_sentence_count_unique", pa.int32()),
                ("vlm_has_table", pa.bool_()),
                ("vlm_has_markdown", pa.bool_()),
            ]
        )
    )

    # Arrow map types have no `datasets` equivalent, so category -> term -> count is expressed as
    # nested lists of structs (see _thesauri_matches).
    thesauri_matches = pa.list_(
        pa.struct(
            [
                ("category", pa.string()),
                ("terms", pa.list_(pa.struct([("term", pa.string()), ("count", pa.int32())]))),
            ]
        )
    )
    chronam_struct = pa.list_(
        pa.struct(
            [
                ("tesseract_matches", thesauri_matches),
                ("tesseract_match_count", pa.int32()),
                ("tesseract_term_count", pa.int32()),
                ("vlm_matches", thesauri_matches),
                ("vlm_match_count", pa.int32()),
                ("vlm_term_count", pa.int32()),
            ]
        )
    )

    return pa.schema(
        [
            ("scan_image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("corpus", pa.string()),
            ("issue_id_src", pa.string()),
            ("newspaper_id_src", pa.string()),
            ("newspaper_id_type_gen", pa.string()),
            ("page_number_gen", pa.int32()),
            ("scan_filename_src", pa.string()),
            ("scan_width_src", pa.int32()),
            ("scan_height_src", pa.int32()),
            ("year_ext", pa.int32()),
            ("month_ext", pa.int32()),
            ("day_ext", pa.int32()),
            ("edition_ext", pa.int32()),
            ("metadata_source_gen", pa.string()),
            ("city_gen", pa.string()),
            ("state_gen", pa.string()),
            ("country_gen", pa.string()),
            ("language_ext", pa.string()),
            ("crop_bbox_gen", list_list_f32),
            ("crop_bbox_conf_gen", list_f32),
            ("crop_vlm_ocr_gen", list_str),
            ("crop_tesseract_ocr_gen", tesseract_struct),
            ("crop_vlm_ocr_token_count_gen", list_int),
            ("crop_tesseract_ocr_token_count_gen", list_int),
            ("crop_text_analysis_gen", text_analysis_struct),
            ("crop_classification_gen", list_str),
            ("crop_classification_image_only_gen", list_str),
            ("crop_classification_image_only_conf_gen", list_f32),
            ("crop_classification_text_only_gen", list_str),
            ("crop_classification_text_only_conf_gen", list_f32),
            ("crop_language_gen", list_str),
            ("crop_language_conf_gen", list_f32),
            ("crop_ner_per_gen", list_list_str),
            ("crop_ner_per_conf_gen", list_list_f32),
            ("crop_ner_loc_gen", list_list_str),
            ("crop_ner_loc_conf_gen", list_list_f32),
            ("crop_ner_org_gen", list_list_str),
            ("crop_ner_org_conf_gen", list_list_f32),
            ("crop_subject_gen", list_list_str),
            ("crop_subject_conf_gen", list_list_f32),
            ("crop_chronam_thesauri_matches_exp", chronam_struct),
            ("crop_text_embeddings", list_list_f32),
            ("crop_image_embeddings", list_list_f32),
        ]
    )


def _empty_columns() -> dict[str, list]:
    return {field.name: [] for field in _build_arrow_schema()}


def _huggingface_schema_metadata() -> dict[bytes, bytes]:
    """Schema metadata that flags scan_image as a Hugging Face Image feature so the dataset viewer
    and `datasets` decode the embedded WEBP bytes. Other columns are inferred from the Arrow schema.
    """
    features = json.dumps({"info": {"features": {"scan_image": {"_type": "Image"}}}})
    return {b"huggingface": features.encode("utf-8")}


#
# Resume / upload
#
def _chunk_key(corpus: str, index: int, extension: str) -> str:
    return f"{corpus}-part-{index:05d}.{extension}"


def _list_uploaded_keys(s3, hf_api, corpus: str) -> tuple[set[str], set[str]]:
    """Lists object keys already present in R2 and Hugging Face under the corpus prefix."""
    prefix = f"{corpus}-part-"

    r2_keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=const.RELEASE_S3_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            r2_keys.add(obj["Key"])

    hf_keys = {
        key
        for key in hf_api.list_repo_files(repo_id=const.RELEASE_HF_DATASET, repo_type="dataset")
        if key.startswith(prefix)
    }

    return r2_keys, hf_keys


def _resume_needs(
    s3,
    corpus: str,
    index: int,
    issue_ids: list[int],
    parquet_key: str,
    manifest_key: str,
    r2_keys: set[str],
    hf_keys: set[str],
    resume: bool,
) -> tuple[bool, bool]:
    """Decides whether a chunk still needs uploading to R2 / HF.

    On R2 a chunk is done only when both its parquet and manifest are present and the manifest's
    issue set matches the currently planned chunk. The manifest is not uploaded to HF, so HF is
    done once its parquet is present. Without --resume, everything is rebuilt and re-uploaded.
    """
    if not resume:
        return True, True

    r2_has = parquet_key in r2_keys and manifest_key in r2_keys
    hf_has = parquet_key in hf_keys

    if r2_has and not _manifest_matches(s3, manifest_key, issue_ids):
        # Stored chunk no longer matches the planned issue set; rebuild for both destinations.
        return True, True

    return (not r2_has, not hf_has)


def _manifest_matches(s3, manifest_key: str, issue_ids: list[int]) -> bool:
    try:
        body = s3.get_object(Bucket=const.RELEASE_S3_BUCKET_NAME, Key=manifest_key)["Body"].read()
        return json.loads(body).get("issue_ids") == issue_ids
    except Exception:
        logger.debug(traceback.format_exc())
        return False


def _upload_and_cleanup(
    s3,
    hf_api,
    parquet_path: Path,
    manifest_path: Path,
    parquet_key: str,
    manifest_key: str,
    need_r2: bool,
    need_hf: bool,
    index: int,
    semaphore: threading.Semaphore,
    transfer_config: TransferConfig,
) -> None:
    """Uploads a chunk's parquet + manifest to the needed destinations (R2 and HF in parallel),
    then deletes the local files. Leaves files in place on failure so a later --resume can retry.
    """
    try:
        tasks = []
        if need_r2:
            tasks.append(
                (
                    "R2",
                    lambda: _upload_r2(
                        s3, parquet_path, manifest_path, parquet_key, manifest_key, transfer_config
                    ),
                )
            )
        if need_hf:
            tasks.append(("HF", lambda: _upload_hf(hf_api, parquet_path, parquet_key)))

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(_with_retries, task, f"chunk {index:05d} -> {name}"): name
                for name, task in tasks
            }
            for future in as_completed(futures):
                future.result()

        parquet_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        logger.info(
            f"Chunk {index:05d}: uploaded ({', '.join(name for name, _ in tasks)}) and removed locally."
        )
    finally:
        semaphore.release()


def _with_retries(task, description: str):
    """Retries a transfer with exponential backoff. Transient 5xx from R2 and Hugging Face are
    routine over a multi-day export, so a single failure must not cost the whole chunk.
    """
    for attempt in range(1, const.EXPORT_UPLOAD_MAX_RETRIES + 1):
        try:
            return task()
        except Exception:
            if attempt == const.EXPORT_UPLOAD_MAX_RETRIES:
                raise
            logger.debug(traceback.format_exc())
            backoff = const.EXPORT_UPLOAD_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                f"{description}: attempt {attempt}/{const.EXPORT_UPLOAD_MAX_RETRIES} failed."
                f" Retrying in {backoff}s..."
            )
            time.sleep(backoff)


def _upload_r2(
    s3,
    parquet_path: Path,
    manifest_path: Path,
    parquet_key: str,
    manifest_key: str,
    transfer_config: TransferConfig,
) -> None:
    s3.upload_file(
        str(parquet_path), const.RELEASE_S3_BUCKET_NAME, parquet_key, Config=transfer_config
    )
    s3.upload_file(
        str(manifest_path), const.RELEASE_S3_BUCKET_NAME, manifest_key, Config=transfer_config
    )


def _transfer_config() -> TransferConfig:
    """Multipart transfer settings for R2 uploads: raises concurrency and part size above the boto3
    defaults so multi-MB parquet chunks upload faster.
    """
    megabyte = 1024 * 1024
    return TransferConfig(
        multipart_threshold=const.EXPORT_S3_MULTIPART_THRESHOLD_MB * megabyte,
        multipart_chunksize=const.EXPORT_S3_MULTIPART_CHUNKSIZE_MB * megabyte,
        max_concurrency=const.EXPORT_S3_TRANSFER_CONCURRENCY,
        use_threads=True,
    )


def _upload_hf(hf_api, parquet_path: Path, parquet_key: str) -> None:
    """Uploads only the parquet to HF. The JSON manifest is resume bookkeeping kept on R2 only."""
    hf_api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_key,
        repo_id=const.RELEASE_HF_DATASET,
        repo_type="dataset",
    )


def _get_hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.getenv("HF_TOKEN"))
