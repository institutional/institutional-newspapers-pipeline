import traceback
from collections import defaultdict

import click
from loguru import logger

import utils
from models import PipelineBatch, Issue, Scan, Crop, CropClassification, CropOCR
from const import DB_IN_CLAUSE_CHUNK_SIZE


@click.command("step10-crop-reading-order")
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
def step10_crop_reading_order(pipeline_batch_id: int, overwrite: bool = False):
    """
    Computes reading order for crops on each scan using HDBSCAN column clustering.
    Updates crop.reading_order with 1-based positions.
    Pure DB + algorithm operation — no model loading or GPU required.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items already processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_reading_order = set(
            Crop.select(Scan.issue)
            .join(Scan)
            .where(
                Scan.issue << issue_ids,
                Crop.reading_order > 0,
            )
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_reading_order
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    issue_ids = [item.issue_id for item in items_to_process]

    # Load all crops with scan data
    all_crops = list(
        Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << issue_ids)
    )

    if not all_crops:
        logger.error("No crops found for the given items.")
        click.get_current_context().exit(1)
        return

    try:
        # Build lookup dicts for classification and OCR data
        crop_ids = [crop.id for crop in all_crops]

        classification_by_crop_id: dict[int, str | None] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for cls in CropClassification.select().where(CropClassification.crop << chunk):
                classification_by_crop_id[cls.crop_id] = cls.final_category

        ocr_by_crop_id: dict[int, str | None] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for ocr in CropOCR.select().where(CropOCR.crop << chunk):
                ocr_by_crop_id[ocr.crop_id] = ocr.vlm_text

        # Group crops by scan
        crops_by_scan_id: dict[int, list[Crop]] = defaultdict(list)
        scan_by_id: dict[int, Scan] = {}

        for crop in all_crops:
            crops_by_scan_id[crop.scan_id].append(crop)
            scan_by_id[crop.scan_id] = crop.scan

        # Compute reading order per scan
        entries_to_update: list[Crop] = []
        scans_processed = 0

        for scan_id, crops in crops_by_scan_id.items():
            scan = scan_by_id[scan_id]

            bboxes = [crop.bbox_xyxy for crop in crops]
            classifications = [classification_by_crop_id.get(crop.id) for crop in crops]

            texts: list[str | None] = []
            for crop in crops:
                raw = ocr_by_crop_id.get(crop.id)
                texts.append(utils.flatten_ocr_text(raw) if raw else None)

            ordered_indices = utils.get_reading_order(
                scan_width=scan.width,
                bboxes_xyxy=bboxes,
                classification=classifications,
                texts=texts,
            )

            # ordered_indices[position] = original crop list index; convert to 1-based
            for position, crop_list_index in enumerate(ordered_indices):
                crops[crop_list_index].reading_order = position + 1
                entries_to_update.append(crops[crop_list_index])

            scans_processed += 1

        logger.info(
            f"{scans_processed} scans processed. "
            f"{len(entries_to_update)} crops assigned reading order."
        )

        utils.process_db_write_batch(
            model=Crop,
            entries_to_update=entries_to_update,
            fields_to_update=[Crop.reading_order],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Reading order processing failed. Exiting.")
        click.get_current_context().exit(1)
