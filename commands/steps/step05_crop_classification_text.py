import traceback

import click
from huggingface_hub import snapshot_download
from loguru import logger

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop, CropOCR, CropClassification
from const import (
    CPUS_LIMIT,
    CROP_CLASSIFICATION_TEXT_MODEL,
    DB_IN_CLAUSE_CHUNK_SIZE,
)


@click.command("step05-crop-classification-text")
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
def step05_crop_classification_text(pipeline_batch_id: int, overwrite: bool = False):
    """
    Uses a static text classifier to categorize each crop based on its VLM-extracted OCR text.
    Populates text_category and text_confidence_score in CropClassification records.

    Runs single-process batch inference on CPU — no GPU required.
    """
    from model2vec.inference import StaticModelPipeline

    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items that have already been processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_text_cls: set[tuple[int]] = set()
        for i in range(0, len(issue_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = issue_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            issues_with_text_cls.update(
                CropClassification.select(Scan.issue)
                .join(Crop)
                .join(Scan)
                .where(
                    Scan.issue << chunk,
                    CropClassification.text_category.is_null(False),
                )
                .distinct()
                .tuples()
            )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_text_cls
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    # Download and load model
    model_path = snapshot_download(CROP_CLASSIFICATION_TEXT_MODEL)
    model = StaticModelPipeline.from_pretrained(model_path)

    # Bulk DB reads
    item_ids = [item.id for item in items_to_process]
    items: list[PipelineBatchItem] = []
    for i in range(0, len(item_ids), DB_IN_CLAUSE_CHUNK_SIZE):
        chunk = item_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
        items.extend(
            PipelineBatchItem.select(PipelineBatchItem, Issue)
            .join(Issue)
            .where(PipelineBatchItem.id << chunk)
        )

    issue_ids = [item.issue_id for item in items]

    all_crops: list[Crop] = []
    for i in range(0, len(issue_ids), DB_IN_CLAUSE_CHUNK_SIZE):
        chunk = issue_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
        all_crops.extend(
            Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << chunk)
        )

    if not all_crops:
        logger.error("No crops found for the items to process.")
        click.get_current_context().exit(1)
        return

    try:
        crop_ids = [crop.id for crop in all_crops]

        ocr_by_crop_id: dict[int, CropOCR] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for ocr in CropOCR.select().where(CropOCR.crop << chunk):
                ocr_by_crop_id[ocr.crop_id] = ocr

        existing_cls_crop_ids: set[int] = set()
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            existing_cls_crop_ids.update(
                row[0]
                for row in CropClassification.select(CropClassification.crop)
                .where(CropClassification.crop << chunk)
                .tuples()
            )

        # Build parallel lists of crops and flattened texts
        all_texts: list[str] = []
        failed_indices: set[int] = set()

        for i, crop in enumerate(all_crops):
            try:
                ocr = ocr_by_crop_id.get(crop.id)
                raw_text = ocr.vlm_text if ocr and ocr.vlm_text else ""
                flat_text = utils.flatten_ocr_text(raw_text) if raw_text else ""
            except Exception:
                logger.warning(
                    f"Could not prepare text for crop #{crop.id}. "
                    f"Creating record with null values.\n{traceback.format_exc()}"
                )
                flat_text = ""
                failed_indices.add(i)
            all_texts.append(flat_text)

        # Split crops into those with and without text for inference
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []
        for i, text in enumerate(all_texts):
            if text.strip() and i not in failed_indices:
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Run model2vec inference on non-empty texts
        labels_by_index: dict[int, str] = {}
        confidences_by_index: dict[int, float] = {}

        if non_empty_texts:
            probas = model.predict_proba(non_empty_texts, batch_size=CPUS_LIMIT, max_length=None)
            label_indices = probas.argmax(axis=1)
            labels = model.classes_[label_indices]
            confidences = probas.max(axis=1)

            for j, idx in enumerate(non_empty_indices):
                labels_by_index[idx] = labels[j]
                confidences_by_index[idx] = float(confidences[j])

        # Build CropClassification records
        entries_to_create: list[CropClassification] = []
        entries_to_update: list[CropClassification] = []

        for i, crop in enumerate(all_crops):
            if i in labels_by_index:
                text_category = labels_by_index[i]
                text_confidence_score = confidences_by_index[i]
            elif i in failed_indices:
                text_category = None
                text_confidence_score = None
            else:
                text_category = "Empty"
                text_confidence_score = 1.0

            record = CropClassification(
                crop=crop.id,
                text_category=text_category,
                text_confidence_score=text_confidence_score,
            )

            if crop.id in existing_cls_crop_ids:
                entries_to_update.append(record)
            else:
                entries_to_create.append(record)

        empty_count = len(all_crops) - len(non_empty_texts)

        logger.info(
            f"{len(all_crops)} crops classified by text. "
            f"{len(non_empty_texts)} with text, {empty_count} empty. "
            f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
        )

        utils.process_db_write_batch(
            model=CropClassification,
            entries_to_create=entries_to_create,
            entries_to_update=entries_to_update,
            fields_to_update=[
                CropClassification.text_category,
                CropClassification.text_confidence_score,
            ],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Text classification processing failed. Exiting.")
        click.get_current_context().exit(1)
