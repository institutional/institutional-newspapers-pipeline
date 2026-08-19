import traceback

import click
from loguru import logger

import utils
from models import PipelineBatch, Issue, Scan, Crop, CropClassification
from const import (
    CROP_CLASSIFICATION_FINAL_IMAGE_PHOTO_CONF,
    CROP_CLASSIFICATION_FINAL_IMAGE_EMPTY_CONF,
)


@click.command("step07-crop-classification-final")
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
def step07_crop_classification_final(pipeline_batch_id: int, overwrite: bool = False):
    """
    Combines text and image classification signals into a final category for each crop.
    Pure DB operation — no model loading or GPU required.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items
    issue_ids = [item.issue_id for item in all_items]

    all_classifications = list(
        CropClassification.select(CropClassification, Crop, Scan, Issue)
        .join(Crop)
        .join(Scan)
        .join(Issue)
        .where(Scan.issue << issue_ids)
    )

    if overwrite:
        records_to_process = all_classifications
    else:
        records_to_process = [cls for cls in all_classifications if cls.final_category is None]

        skipped = len(all_classifications) - len(records_to_process)
        if skipped:
            logger.info(f"{skipped} records already processed. Skipping (use --overwrite to redo).")

    if not records_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    try:
        entries_to_update: list[CropClassification] = []
        skipped_none = 0

        for cls in records_to_process:
            decision = _decide_final_category(cls)
            if decision is None:
                skipped_none += 1
                continue
            cls.final_category = decision
            entries_to_update.append(cls)

        logger.info(
            f"{len(entries_to_update)} crops assigned a final category. "
            f"{skipped_none} skipped (missing both text and image signals)."
        )

        utils.process_db_write_batch(
            model=CropClassification,
            entries_to_update=entries_to_update,
            fields_to_update=[CropClassification.final_category],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Final classification processing failed. Exiting.")
        click.get_current_context().exit(1)


def _decide_final_category(cls: CropClassification) -> str | None:
    """
    Combines text and image classification signals into a final category.
    Returns None if both signals are missing.
    """
    image_cat = cls.image_category
    image_conf = cls.image_confidence_score
    text_cat = cls.text_category
    text_conf = cls.text_confidence_score

    if image_cat is None and text_cat is None:
        return None

    if image_cat is None:
        return text_cat
    if text_cat is None:
        return image_cat

    # Default: modality with highest confidence
    decision = image_cat if (image_conf or 0.0) > (text_conf or 0.0) else text_cat

    # Override: image model's "Photograph or illustration" with sufficient confidence
    if (
        image_cat == "Photograph or illustration"
        and (image_conf or 0.0) > CROP_CLASSIFICATION_FINAL_IMAGE_PHOTO_CONF
    ):
        decision = image_cat

    # Override: image model's "Empty" with sufficient confidence
    if image_cat == "Empty" and (image_conf or 0.0) > CROP_CLASSIFICATION_FINAL_IMAGE_EMPTY_CONF:
        decision = image_cat

    # Override: if text is empty, image model takes precedence
    if text_cat == "Empty":
        decision = image_cat

    return decision
