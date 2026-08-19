import io
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple, TypedDict

import click
from PIL import Image
from loguru import logger
from tesserocr import PyTessBaseAPI, RIL, get_languages

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop, CropOCR
from const import CPUS_LIMIT, DB_IN_CLAUSE_CHUNK_SIZE, OCR_TESSDATA_DIR_PATH, OCR_TESSERACT_MAX_PIXELS


class TesseractWordResult(TypedDict):
    text: str
    conf: float
    bbox_xyxy: list[int]


class OcrWorkItem(NamedTuple):
    crop_id: int
    cache_key: str
    lang: str


class OcrResult(NamedTuple):
    crop_id: int
    ocr_text: str | None
    ocr_metadata: list[TesseractWordResult] | None


@click.command("step03-crop-ocr-tesseract")
@click.option(
    "--pipeline-batch-id",
    type=int,
    required=True,
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="If set, will replace existing records.",
)
def step03_crop_ocr_tesseract(pipeline_batch_id: int, overwrite: bool = False):
    """
    Runs Tesseract OCR on every crop from the current pipeline batch.
    Stores full text and word-level bounding boxes in CropOCR records.

    Sorts crops by language to minimize Tesseract engine re-initialization, then splits them into small chunks across a ProcessPool for load balancing.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items that have already been processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_tesseract: set[tuple[int]] = set()
        for i in range(0, len(issue_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = issue_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            issues_with_tesseract.update(
                CropOCR.select(Scan.issue)
                .join(Crop)
                .join(Scan)
                .where(Scan.issue << chunk, CropOCR.tesseract_text.is_null(False))
                .distinct()
                .tuples()
            )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_tesseract
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    available_langs = set(get_languages(OCR_TESSDATA_DIR_PATH)[1])

    # Bulk-query all crops across all issues
    issue_ids = [item.issue_id for item in items_to_process]
    all_crops: list[Crop] = []
    for i in range(0, len(issue_ids), DB_IN_CLAUSE_CHUNK_SIZE):
        chunk = issue_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
        all_crops.extend(
            Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << chunk)
        )

    if not all_crops:
        logger.error("No crops found.")
        click.get_current_context().exit(1)
        return

    try:
        if overwrite:
            crop_ids = [crop.id for crop in all_crops]
            for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
                chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
                CropOCR.delete().where(CropOCR.crop << chunk).execute()

        # Build lightweight work items (no image data — workers read from cache)
        work_items: list[OcrWorkItem] = []
        for crop in all_crops:
            lang = _resolve_tesseract_lang(crop.scan.issue.language, available_langs)
            work_items.append(OcrWorkItem(crop_id=crop.id, cache_key=crop.cache_key, lang=lang))

        # Sort by language so each chunk is mostly one language, minimizing Tesseract re-init
        work_items.sort(key=lambda item: item.lang)

        # Split into many small chunks for better load balancing — idle workers pick up
        # the next chunk rather than waiting for the slowest single large chunk to finish.
        chunk_size = max(1, len(work_items) // (CPUS_LIMIT * 4))
        chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]

        all_results: list[OcrResult] = []
        with ProcessPoolExecutor(max_workers=CPUS_LIMIT) as executor:
            for result_batch in executor.map(_ocr_chunk, chunks):
                all_results.extend(result_batch)

        # Map crop_id -> issue_id for per-issue logging
        crop_id_to_issue_id: dict[int, int] = {crop.id: crop.scan.issue.id for crop in all_crops}

        # Build CropOCR records and track per-issue counts
        entries_to_create: list[CropOCR] = []
        issue_processed: dict[int, int] = defaultdict(int)
        issue_failed: dict[int, int] = defaultdict(int)

        for result in all_results:
            entries_to_create.append(
                CropOCR(
                    crop=result.crop_id,
                    tesseract_text=result.ocr_text,
                    tesseract_metadata=result.ocr_metadata,
                )
            )
            issue_id = crop_id_to_issue_id[result.crop_id]
            if result.ocr_text is not None:
                issue_processed[issue_id] += 1
            else:
                issue_failed[issue_id] += 1

        utils.process_db_write_batch(CropOCR, entries_to_create=entries_to_create)

        # Per-issue logging
        issue_map = {item.issue_id: item.issue for item in items_to_process}
        for issue_id, issue in issue_map.items():
            processed = issue_processed.get(issue_id, 0)
            failed = issue_failed.get(issue_id, 0)
            if processed or failed:
                logger.info(
                    f"{issue.archive_filename} ({issue.corpus}): "
                    f"{processed} crops OCR'd, {failed} failed"
                )

        total_processed = sum(issue_processed.values())
        total_failed = sum(issue_failed.values())
        logger.info(
            f"Tesseract OCR complete: {total_processed} crops processed, {total_failed} failed."
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Tesseract OCR processing failed. Exiting.")
        click.get_current_context().exit(1)


def _resolve_tesseract_lang(language: str | None, available_langs: set[str]) -> str:
    """Returns a Tesseract language code, falling back to 'eng' if unavailable."""
    if language and language in available_langs:
        return language
    return "eng"


def _ocr_chunk(chunk: list[OcrWorkItem]) -> list[OcrResult]:
    """Worker function: OCRs a chunk of crops in a subprocess."""
    cache = utils.get_cache()
    api: PyTessBaseAPI | None = None
    current_lang: str | None = None
    results: list[OcrResult] = []

    try:
        for item in chunk:
            try:
                jpeg_bytes = cache.get(item.cache_key)
                if jpeg_bytes is None:
                    raise RuntimeError(
                        f"Crop #{item.crop_id} not found in cache (key: {item.cache_key})."
                    )

                # Initialize or re-initialize Tesseract API on language change
                if api is None:
                    api = PyTessBaseAPI(path=OCR_TESSDATA_DIR_PATH, psm=1, oem=1, lang=item.lang)
                    current_lang = item.lang
                elif current_lang != item.lang:
                    api.Init(path=OCR_TESSDATA_DIR_PATH, lang=item.lang, oem=1, psm=1)
                    current_lang = item.lang

                ocr_text, ocr_metadata = _ocr_crop(api, jpeg_bytes)
                results.append(OcrResult(item.crop_id, ocr_text, ocr_metadata))

            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"Could not OCR crop #{item.crop_id}. Creating empty record.")
                results.append(OcrResult(item.crop_id, None, None))
    finally:
        if api is not None:
            api.End()

    return results


def _ocr_crop(api: PyTessBaseAPI, jpeg_bytes: bytes) -> tuple[str, list[TesseractWordResult]]:
    """Performs Tesseract OCR on JPEG bytes using the provided API instance."""
    pil_image = Image.open(io.BytesIO(jpeg_bytes))

    # Shrink to fit within the pixel budget while preserving aspect ratio
    inv_scale = 1.0
    total_pixels = pil_image.width * pil_image.height
    if total_pixels > OCR_TESSERACT_MAX_PIXELS:
        scale = (OCR_TESSERACT_MAX_PIXELS / total_pixels) ** 0.5
        inv_scale = 1.0 / scale
        new_w = int(pil_image.width * scale)
        new_h = int(pil_image.height * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    api.SetImage(pil_image)
    api.Recognize()

    ocr_text = api.GetUTF8Text()
    ocr_metadata: list[TesseractWordResult] = []

    ocr_iter = api.GetIterator()
    ocr_level = RIL.WORD

    while ocr_iter:
        text = ocr_iter.GetUTF8Text(ocr_level) or ""
        conf = ocr_iter.Confidence(ocr_level)
        bbox = ocr_iter.BoundingBox(ocr_level)

        ocr_metadata.append(TesseractWordResult(
            text=text,
            conf=conf,
            bbox_xyxy=[int(coord * inv_scale) for coord in bbox],
        ))

        if not ocr_iter.Next(ocr_level):
            break

    return (ocr_text, ocr_metadata)
