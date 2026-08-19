import traceback

import tiktoken
import click
from loguru import logger

import utils
from models import PipelineBatch, Issue, Scan, Crop, CropOCR
from models.crop_token_count import CropTokenCount
from const import CPUS_LIMIT, TOKEN_COUNT_TIKTOKEN_ENCODING, DB_IN_CLAUSE_CHUNK_SIZE


@click.command("step11-crop-token-count")
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
def step11_crop_token_count(pipeline_batch_id: int, overwrite: bool = False):
    """
    Counts tokens in CropOCR.tesseract_text and CropOCR.vlm_text for every crop in the pipeline batch using tiktoken. Stores results in CropTokenCount.

    Uses tiktoken's built-in multi-threaded batch encoding. CPU-only, no GPU required.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items already processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_token_counts = set(
            CropTokenCount.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(Scan.issue << issue_ids)
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_token_counts
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
        logger.error("No crops found for the given items.")
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
                    for row in CropTokenCount.select(CropTokenCount.crop)
                    .where(CropTokenCount.crop << chunk)
                    .tuples()
                )

        # Build text lists for batch encoding
        tesseract_texts: list[str] = []
        vlm_texts: list[str] = []
        has_tesseract: list[bool] = []
        has_vlm: list[bool] = []

        for crop_id in crop_ids:
            ocr = ocr_by_crop_id.get(crop_id)
            tess = ocr.tesseract_text if ocr and ocr.tesseract_text else None
            vlm = ocr.vlm_text if ocr and ocr.vlm_text else None

            tesseract_texts.append(tess if tess else "")
            vlm_texts.append(vlm if vlm else "")
            has_tesseract.append(tess is not None)
            has_vlm.append(vlm is not None)

        # Count tokens using tiktoken's built-in parallelism
        enc = tiktoken.get_encoding(TOKEN_COUNT_TIKTOKEN_ENCODING)
        tesseract_tokens = enc.encode_batch(tesseract_texts, num_threads=CPUS_LIMIT)
        vlm_tokens = enc.encode_batch(vlm_texts, num_threads=CPUS_LIMIT)

        # Build entries
        entries_to_create: list[CropTokenCount] = []
        entries_to_update: list[CropTokenCount] = []

        for i, crop_id in enumerate(crop_ids):
            record = CropTokenCount(
                crop=crop_id,
                tesseract_token_count=len(tesseract_tokens[i]) if has_tesseract[i] else None,
                vlm_token_count=len(vlm_tokens[i]) if has_vlm[i] else None,
            )

            if crop_id in existing_crop_ids:
                entries_to_update.append(record)
            else:
                entries_to_create.append(record)

        logger.info(
            f"{len(crop_ids)} crops processed for token counting. "
            f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
        )

        utils.process_db_write_batch(
            model=CropTokenCount,
            entries_to_create=entries_to_create,
            entries_to_update=entries_to_update,
            fields_to_update=[
                CropTokenCount.tesseract_token_count,
                CropTokenCount.vlm_token_count,
            ],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Token counting failed. Exiting.")
        click.get_current_context().exit(1)
