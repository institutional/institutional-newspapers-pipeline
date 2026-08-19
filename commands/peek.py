import json
import base64
import random
from collections import Counter

import click
from loguru import logger

import utils
from utils import record_loaders
from models import (
    Issue,
    Scan,
    Crop,
    CropOCR,
    CropClassification,
    CropSubject,
    CropNER,
    CropLanguage,
    CropTokenCount,
    CropTextAnalysis,
    PipelineRun,
    PipelineBatch,
    PipelineBatchItem,
)
from const import (
    PEEK_DIR_PATH,
    SCAN_JPEG_QUALITY,
    OCR_TESSERACT_MAX_PIXELS,
    OCR_VLM_MAX_ASPECT_RATIO,
    OCR_VLM_MAX_PIXELS,
    OCR_VLM_MIN_PIXELS,
    OCR_VLM_SMART_RESIZE_FACTOR,
)
import const


@click.command("peek")
@click.option("--pipeline-run-id", type=int, default=None)
@click.option("--pipeline-batch-id", type=int, default=None)
@click.option("--limit", type=int, required=True)
def peek(pipeline_run_id: int | None, pipeline_batch_id: int | None, limit: int):
    """
    Exports all pipeline data for a random sample of issues as JSON files.
    Includes scan images as base64, all crop analysis records, and pipeline config metadata.
    Re-downloads missing scans from S3 if needed.

    Exactly one of --pipeline-run-id or --pipeline-batch-id must be provided.
    """
    if (pipeline_run_id is None) == (pipeline_batch_id is None):
        logger.error("Provide exactly one of --pipeline-run-id or --pipeline-batch-id.")
        click.get_current_context().exit(1)

    if pipeline_batch_id is not None:
        try:
            pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
        except PipelineBatch.DoesNotExist:
            logger.error(f"Pipeline batch {pipeline_batch_id} not found.")
            click.get_current_context().exit(1)
        pipeline_run = pipeline_batch.pipeline_run
    else:
        try:
            pipeline_run = PipelineRun.get(id=pipeline_run_id)
        except PipelineRun.DoesNotExist:
            logger.error(f"Pipeline run {pipeline_run_id} not found.")
            click.get_current_context().exit(1)

    # Query all distinct issue IDs, then filter for completeness
    batch_filter = (
        PipelineBatch.id == pipeline_batch_id
        if pipeline_batch_id is not None
        else PipelineBatch.pipeline_run == pipeline_run.id
    )
    candidate_issue_ids = [
        row.id
        for row in Issue.select(Issue.id)
        .join(PipelineBatchItem, on=(PipelineBatchItem.issue == Issue.id))
        .join(PipelineBatch, on=(PipelineBatchItem.pipeline_batch == PipelineBatch.id))
        .where(batch_filter)
        .distinct()
    ]

    complete_issue_ids, incomplete_details = _get_complete_issue_ids(candidate_issue_ids)

    if not complete_issue_ids:
        scope = (
            f"pipeline batch {pipeline_batch_id}"
            if pipeline_batch_id is not None
            else f"pipeline run {pipeline_run.id}"
        )
        logger.error(
            f"No complete issues found for {scope}. "
            f"({len(candidate_issue_ids)} candidates checked, "
            f"none had records in Scan and all Crop analysis tables.)"
        )
        click.get_current_context().exit(1)

    issues = list(Issue.select().where(Issue.id << list(complete_issue_ids)))
    logger.info(
        f"{len(issues)} of {len(candidate_issue_ids)} issues are complete "
        f"(have scans, crops, and all analysis records)."
    )

    if incomplete_details:
        table_counts: Counter[str] = Counter()
        for tables in incomplete_details.values():
            table_counts.update(tables)
        breakdown = ", ".join(
            f"{count} missing {table}" for table, count in table_counts.most_common()
        )
        logger.info(f"{len(incomplete_details)} incomplete issue(s): {breakdown}")
        for iid, tables in incomplete_details.items():
            logger.debug(f"  Issue {iid} missing: {', '.join(tables)}")

    random.shuffle(issues)
    issues = issues[:limit]
    logger.info(f"Selected {len(issues)} issues for export.")

    # Load all data
    issue_ids = [issue.id for issue in issues]

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

    crop_ids = [crop.id for crop in crops]

    analysis = record_loaders.load_analysis_data(crop_ids, PEEK_ANALYSIS_MODELS)

    # Ensure scan images are cached
    _ensure_scans_cached(issues, scans_by_issue)

    # Build metadata
    metadata = _build_metadata(pipeline_run)

    # Write JSON files
    cache = utils.get_cache()

    for issue in issues:
        issue_scans = scans_by_issue.get(issue.id, [])
        issue_scans.sort(key=lambda s: s.scan_filename)

        scans_data = []
        for scan in issue_scans:
            scan_crops = crops_by_scan.get(scan.id, [])
            scan_crops.sort(key=lambda c: (c.reading_order or 0, c.id))

            scan_image_bytes = cache.get(scan.cache_key)
            image_b64 = (
                base64.b64encode(scan_image_bytes).decode("ascii") if scan_image_bytes else None
            )

            crops_data = []
            for crop in scan_crops:
                crops_data.append(_build_crop_dict(crop, analysis, issue))

            scans_data.append(
                {
                    "id": scan.id,
                    "scan_filename": scan.scan_filename,
                    "width": scan.width,
                    "height": scan.height,
                    "phash": scan.phash,
                    "image_jpeg_base64": image_b64,
                    "crops": crops_data,
                }
            )

        output = {
            "metadata": metadata,
            "issue": _build_issue_dict(issue),
            "scans": scans_data,
        }

        if pipeline_batch_id is not None:
            filename = f"{issue.corpus}-batch{pipeline_batch_id}-{issue.id}.json"
        else:
            filename = f"{issue.corpus}-{pipeline_run.id}-{issue.id}.json"
        output_path = PEEK_DIR_PATH / filename
        with open(output_path, "w") as f:
            json.dump(output, f)

        logger.info(
            f"Exported {issue.archive_filename}: "
            f"{len(issue_scans)} scans, "
            f"{sum(len(crops_by_scan.get(s.id, [])) for s in issue_scans)} crops "
            f"→ {output_path.name}"
        )

    logger.info(f"Done. {len(issues)} files written to {PEEK_DIR_PATH}")


CROP_ANALYSIS_TABLES = (
    CropOCR,
    CropClassification,
    CropSubject,
    CropNER,
    CropLanguage,
    CropTokenCount,
    CropTextAnalysis,
)


def _get_complete_issue_ids(
    candidate_issue_ids: list[int],
) -> tuple[set[int], dict[int, list[str]]]:
    """Returns the subset of issue IDs that have scans, crops, and records in all
    crop analysis tables, plus a dict mapping each incomplete issue ID to the list
    of table names it is missing.
    """
    candidates = set(candidate_issue_ids)
    missing: dict[int, list[str]] = {}

    if not candidate_issue_ids:
        return set(), missing

    # Issues with at least one scan
    scan_rows = record_loaders.chunked_query(
        Scan.select(Scan.issue).distinct(), Scan.issue, candidate_issue_ids
    )
    issues_with_scans = {row.issue_id for row in scan_rows}
    for iid in candidates - issues_with_scans:
        missing.setdefault(iid, []).append("Scan")

    if not issues_with_scans:
        return set(), missing

    # Build crop_id → issue_id mapping via scans
    scan_rows = record_loaders.chunked_query(
        Scan.select(Scan.id, Scan.issue), Scan.issue, list(issues_with_scans)
    )
    scan_to_issue: dict[int, int] = {row.id: row.issue_id for row in scan_rows}
    scan_ids = list(scan_to_issue.keys())
    if not scan_ids:
        return set(), missing

    crop_rows = record_loaders.chunked_query(Crop.select(Crop.id, Crop.scan), Crop.scan, scan_ids)
    issues_with_crops = {scan_to_issue[row.scan_id] for row in crop_rows}
    for iid in issues_with_scans - issues_with_crops:
        missing.setdefault(iid, []).append("Crop")

    if not crop_rows:
        return set(), missing

    crop_to_issue: dict[int, int] = {row.id: scan_to_issue[row.scan_id] for row in crop_rows}
    crop_ids = list(crop_to_issue.keys())

    # Intersect: for each analysis table, find which issues have at least one record
    complete_issue_ids = set(crop_to_issue.values())
    for model in CROP_ANALYSIS_TABLES:
        rows = record_loaders.chunked_query(model.select(model.crop), model.crop, crop_ids)
        issues_in_table = {crop_to_issue[row.crop_id] for row in rows}
        for iid in complete_issue_ids - issues_in_table:
            missing.setdefault(iid, []).append(model.__name__)
        complete_issue_ids &= issues_in_table

    return complete_issue_ids, missing


# Peek dumps every analysis table except the embeddings (large vectors, not useful for inspection).
PEEK_ANALYSIS_MODELS = {
    key: model
    for key, model in record_loaders.ANALYSIS_MODELS.items()
    if key not in ("text_static_embedding", "image_embedding")
}


def _build_metadata(pipeline_run: PipelineRun) -> dict:
    return {
        "pipeline_run": {
            "id": pipeline_run.id,
            "corpus": pipeline_run.corpus,
            "items_total": pipeline_run.items_total,
            "items_per_batch": pipeline_run.items_per_batch,
            "batches_total": pipeline_run.batches_total,
            "created_date": str(pipeline_run.created_date),
        },
        "pipeline_config": {
            "segmentation": {
                "model": const.CROP_DETECTION_MODEL,
                "imgsz": const.CROP_DETECTION_IMGSZ,
                "conf": const.CROP_DETECTION_CONF,
                "iou": const.CROP_DETECTION_IOU,
                "max_det": const.CROP_DETECTION_MAX_DET,
            },
            "ocr": {
                "tesseract_max_pixels": const.OCR_TESSERACT_MAX_PIXELS,
                "vlm_model": const.OCR_VLM_MODEL,
                "vlm_model_context": const.OCR_VLM_MODEL_CONTEXT,
                "vlm_max_pixels": const.OCR_VLM_MAX_PIXELS,
                "vlm_min_pixels": const.OCR_VLM_MIN_PIXELS,
                "vlm_smart_resize_factor": const.OCR_VLM_SMART_RESIZE_FACTOR,
            },
            "classification": {
                "image_model": const.CROP_CLASSIFICATION_IMAGE_MODEL,
                "image_imgsz": const.CROP_CLASSIFICATION_IMAGE_IMGSZ,
                "text_model": const.CROP_CLASSIFICATION_TEXT_MODEL,
            },
            "ner": {
                "model": const.NER_FLAIR_MODEL,
                "conf": const.NER_FLAIR_CONF,
                "classes": list(const.NER_FLAIR_CLASSES),
            },
            "subject": {
                "model": const.SUBJECT_ZEROSHOT_MODEL,
                "conf": const.SUBJECT_MODEL_CONF,
                "classes": list(const.SUBJECT_CLASSES[0]),
            },
            "reading_order": {
                "column_width_percentile": const.READING_ORDER_COLUMN_WIDTH_PERCENTILE,
                "wide_crop_threshold_ratio": const.READING_ORDER_WIDE_CROP_THRESHOLD_RATIO,
                "hdbscan_min_cluster_size": const.READING_ORDER_HDBSCAN_MIN_CLUSTER_SIZE,
                "hdbscan_min_samples": const.READING_ORDER_HDBSCAN_MIN_SAMPLES,
            },
            "embeddings": {
                "static_text_model": const.STATIC_TEXT_EMBEDDING_MODEL,
                "image_model": const.IMAGE_EMBEDDING_MODEL,
            },
            "token_count": {
                "tiktoken_encoding": const.TOKEN_COUNT_TIKTOKEN_ENCODING,
            },
            "chronam_thesauri": {
                "dataset": const.CHRONAM_THESAURI_DATASET,
            },
        },
    }


def _build_issue_dict(issue: Issue) -> dict:
    pp_city, pp_state, pp_country, pp_locality_corrected = utils.postprocess_locality(
        issue.city or "",
        issue.state or "",
        issue.country or "",
    )

    return {
        "id": issue.id,
        "corpus": issue.corpus,
        "archive_filename": issue.archive_filename,
        "archive_size_bytes": issue.archive_size_bytes,
        "newspaper_id": issue.newspaper_id,
        "newspaper_id_type": issue.newspaper_id_type,
        "edition_slug": issue.edition_slug,
        "edition_slug_type": issue.edition_slug_type,
        "title": issue.title,
        "city": issue.city,
        "state": issue.state,
        "country": issue.country,
        "publisher": issue.publisher,
        "year": issue.year,
        "month": issue.month,
        "day": issue.day,
        "edition_number": issue.edition_number,
        "year_start": issue.year_start,
        "year_end": issue.year_end,
        "language": issue.language,
        "loc_access_restricted": issue.loc_access_restricted,
        "postprocessed_city": pp_city,
        "postprocessed_state": pp_state,
        "postprocessed_country": pp_country,
        "postprocessed_locality_corrected": pp_locality_corrected,
    }


def _build_crop_dict(crop: Crop, analysis: dict[str, dict[int, object]], issue: Issue) -> dict:
    crop_id = crop.id
    width = crop.width
    height = crop.height

    result: dict = {
        "id": crop_id,
        "bbox_xyxy": crop.bbox_xyxy,
        "confidence_score": crop.confidence_score,
        "reading_order": crop.reading_order,
        "width": width,
        "height": height,
        "resolution": _build_resolution_dict(width, height),
    }

    # OCR
    ocr = analysis["ocr"].get(crop_id)
    result["ocr"] = (
        {
            "tesseract_text": ocr.tesseract_text,
            "tesseract_metadata": ocr.tesseract_metadata,
            "vlm_text": ocr.vlm_text,
            "vlm_metadata": ocr.vlm_metadata,
        }
        if ocr
        else None
    )

    if ocr and ocr.vlm_text:
        pp_vlm_text, pp_vlm_text_modified = utils.postprocess_vlm_text(ocr.vlm_text)
    else:
        pp_vlm_text, pp_vlm_text_modified = None, False
    result["postprocessed_vlm_text"] = pp_vlm_text
    result["postprocessed_vlm_text_modified"] = pp_vlm_text_modified

    # Classification
    cls = analysis["classification"].get(crop_id)
    result["classification"] = (
        {
            "image_category": cls.image_category,
            "image_confidence_score": cls.image_confidence_score,
            "text_category": cls.text_category,
            "text_confidence_score": cls.text_confidence_score,
            "final_category": cls.final_category,
        }
        if cls
        else None
    )

    # Subject
    subj = analysis["subject"].get(crop_id)
    result["subject"] = (
        {
            "ranked_labels": subj.ranked_labels,
            "scores": subj.scores,
        }
        if subj
        else None
    )

    # NER
    ner = analysis["ner"].get(crop_id)
    result["ner"] = (
        {
            "per_entities": ner.per_entities,
            "per_confidence_scores": ner.per_confidence_scores,
            "loc_entities": ner.loc_entities,
            "loc_confidence_scores": ner.loc_confidence_scores,
            "org_entities": ner.org_entities,
            "org_confidence_scores": ner.org_confidence_scores,
        }
        if ner
        else None
    )

    # Language
    lang = analysis["language"].get(crop_id)
    result["language"] = (
        {
            "language_code": lang.language_code,
            "confidence_score": lang.confidence_score,
        }
        if lang
        else None
    )

    ta_for_lang = analysis["text_analysis"].get(crop_id)
    if lang and lang.language_code and ta_for_lang:
        pp_lang, pp_lang_overridden = utils.postprocess_language_detection(
            lang.language_code,
            lang.confidence_score or 0.0,
            ta_for_lang.vlm_word_count or 0,
            issue.language,
        )
    else:
        pp_lang, pp_lang_overridden = (lang.language_code if lang else None), False
    result["postprocessed_language_code"] = pp_lang
    result["postprocessed_language_overridden"] = pp_lang_overridden

    # Token count
    tc = analysis["token_count"].get(crop_id)
    result["token_count"] = (
        {
            "tesseract_token_count": tc.tesseract_token_count,
            "vlm_token_count": tc.vlm_token_count,
        }
        if tc
        else None
    )

    # Text analysis
    ta = analysis["text_analysis"].get(crop_id)
    result["text_analysis"] = (
        {
            "tesseract_tokenizability_score": ta.tesseract_tokenizability_score,
            "tesseract_char_count": ta.tesseract_char_count,
            "tesseract_word_count": ta.tesseract_word_count,
            "tesseract_word_count_unique": ta.tesseract_word_count_unique,
            "tesseract_word_type_token_ratio": ta.tesseract_word_type_token_ratio,
            "tesseract_sentence_count": ta.tesseract_sentence_count,
            "tesseract_sentence_count_unique": ta.tesseract_sentence_count_unique,
            "vlm_tokenizability_score": ta.vlm_tokenizability_score,
            "vlm_char_count": ta.vlm_char_count,
            "vlm_word_count": ta.vlm_word_count,
            "vlm_word_count_unique": ta.vlm_word_count_unique,
            "vlm_word_type_token_ratio": ta.vlm_word_type_token_ratio,
            "vlm_sentence_count": ta.vlm_sentence_count,
            "vlm_sentence_count_unique": ta.vlm_sentence_count_unique,
            "vlm_has_table": ta.vlm_has_table,
            "vlm_has_markdown": ta.vlm_has_markdown,
        }
        if ta
        else None
    )

    # ChronAm thesauri match
    ctm = analysis["chronam_thesauri_match"].get(crop_id)
    result["chronam_thesauri_match"] = (
        {
            "tesseract_matches": ctm.tesseract_matches,
            "tesseract_match_count": ctm.tesseract_match_count,
            "tesseract_term_count": ctm.tesseract_term_count,
            "vlm_matches": ctm.vlm_matches,
            "vlm_match_count": ctm.vlm_match_count,
            "vlm_term_count": ctm.vlm_term_count,
        }
        if ctm
        else None
    )

    return result


def _build_resolution_dict(width: int | None, height: int | None) -> dict | None:
    if width is None or height is None:
        return None

    return {
        "full": {
            "wxh": f"{width}x{height}",
            "mpx": round(width * height / 1_000_000, 4),
        },
        "tesseract_ocr": {
            "mpx": _compute_tesseract_mpx(width, height),
        },
        "vlm_ocr": {
            "mpx": _compute_vlm_mpx(width, height),
        },
    }


def _compute_tesseract_mpx(width: int, height: int) -> float:
    """Computes the resolution at which Tesseract would OCR this crop, in megapixels."""
    total_pixels = width * height
    if total_pixels > OCR_TESSERACT_MAX_PIXELS:
        scale = (OCR_TESSERACT_MAX_PIXELS / total_pixels) ** 0.5
        new_w = int(width * scale)
        new_h = int(height * scale)
    else:
        new_w, new_h = width, height
    return round(new_w * new_h / 1_000_000, 4)


def _compute_vlm_mpx(width: int, height: int) -> float:
    """Computes the resolution at which the VLM would OCR this crop, in megapixels."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    half_h, half_w = height // 2, width // 2
    if half_h * half_w >= OCR_VLM_MIN_PIXELS:
        target_h, target_w = half_h, half_w
    else:
        target_h, target_w = height, width

    if max(target_h, target_w) / min(target_h, target_w) > OCR_VLM_MAX_ASPECT_RATIO:
        return 0.0

    new_h, new_w = smart_resize(
        target_h,
        target_w,
        factor=OCR_VLM_SMART_RESIZE_FACTOR,
        max_pixels=OCR_VLM_MAX_PIXELS,
    )
    return round(new_h * new_w / 1_000_000, 4)


def _ensure_scans_cached(
    issues: list[Issue],
    scans_by_issue: dict[int, list[Scan]],
) -> None:
    """Re-downloads and caches scan images for any scans missing from the disk cache."""
    cache = utils.get_cache()

    missing_by_issue: dict[int, list[Scan]] = {}
    for issue in issues:
        for scan in scans_by_issue.get(issue.id, []):
            if scan.cache_key not in cache:
                missing_by_issue.setdefault(issue.id, []).append(scan)

    if not missing_by_issue:
        return

    total_missing = sum(len(v) for v in missing_by_issue.values())
    logger.info(f"{total_missing} scan(s) missing from cache. Downloading archives...")

    missing_issues = [issue for issue in issues if issue.id in missing_by_issue]
    scan_by_id = {scan.id: scan for scans in missing_by_issue.values() for scan in scans}

    processed = record_loaders.fetch_and_process_scans(
        missing_issues,
        missing_by_issue,
        image_format="JPEG",
        quality=SCAN_JPEG_QUALITY,
    )

    for scan_id, jpeg_bytes in processed.items():
        cache.set(scan_by_id[scan_id].cache_key, jpeg_bytes)

    logger.info(f"{len(processed)} scan(s) re-cached.")
