import os
import traceback

import click
from loguru import logger

import utils
from models import PipelineBatch, Issue, Scan, Crop, CropOCR
from models.crop_language import CropLanguage
from const import CPUS_LIMIT, DB_IN_CLAUSE_CHUNK_SIZE


@click.command("step12-crop-language")
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
def step12_crop_language(pipeline_batch_id: int, overwrite: bool = False):
    """
    Detects the primary language of CropOCR.vlm_text for every crop in the pipeline batch.
    Stores ISO 639-3 code and confidence score in CropLanguage.

    Uses lingua-py's built-in Rayon parallelism for batch detection. CPU-only, no GPU required.
    """
    from lingua import LanguageDetectorBuilder

    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items already processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_language = set(
            CropLanguage.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(Scan.issue << issue_ids)
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_language
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    issue_ids = [item.issue_id for item in items_to_process]

    # Load all crops for batch
    all_crops = list(
        Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << issue_ids)
    )

    if not all_crops:
        logger.error("No crops to process.")
        click.get_current_context().exit(1)
        return

    try:
        crop_ids = [crop.id for crop in all_crops]

        # Load CropOCR data in chunks
        ocr_by_crop_id: dict[int, CropOCR] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for ocr in CropOCR.select().where(CropOCR.crop << chunk):
                ocr_by_crop_id[ocr.crop_id] = ocr

        # In overwrite mode, find existing records to determine create vs update
        existing_crop_ids: set[int] = set()
        if overwrite:
            for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
                chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
                existing_crop_ids.update(
                    row[0]
                    for row in CropLanguage.select(CropLanguage.crop)
                    .where(CropLanguage.crop << chunk)
                    .tuples()
                )

        # Build text list for batch detection
        texts: list[str] = []
        has_text: list[bool] = []

        for crop_id in crop_ids:
            ocr = ocr_by_crop_id.get(crop_id)
            raw = ocr.vlm_text if ocr and ocr.vlm_text else None
            flat = utils.flatten_ocr_text(raw) if raw else None

            if flat and flat.strip():
                texts.append(flat)
                has_text.append(True)
            else:
                texts.append("")
                has_text.append(False)

        # Detect languages using lingua's built-in Rayon parallelism
        os.environ["RAYON_NUM_THREADS"] = str(CPUS_LIMIT)
        detector = LanguageDetectorBuilder.from_all_languages().build()
        confidence_values = detector.compute_language_confidence_values_in_parallel(texts)

        # Build entries
        entries_to_create: list[CropLanguage] = []
        entries_to_update: list[CropLanguage] = []

        for i, crop_id in enumerate(crop_ids):
            language_code: str | None = None
            confidence_score: float | None = None

            if has_text[i] and confidence_values[i]:
                best = confidence_values[i][0]
                language_code = best.language.iso_code_639_3.name
                confidence_score = round(float(best.value), 4)

            record = CropLanguage(
                crop=crop_id,
                language_code=language_code,
                confidence_score=confidence_score,
            )

            if crop_id in existing_crop_ids:
                entries_to_update.append(record)
            else:
                entries_to_create.append(record)

        logger.info(
            f"{len(crop_ids)} crops processed for language detection. "
            f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
        )

        utils.process_db_write_batch(
            model=CropLanguage,
            entries_to_create=entries_to_create,
            entries_to_update=entries_to_update,
            fields_to_update=[
                CropLanguage.language_code,
                CropLanguage.confidence_score,
            ],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Language detection failed. Exiting.")
        click.get_current_context().exit(1)
