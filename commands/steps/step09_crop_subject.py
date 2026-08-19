import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from loguru import logger

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop, CropOCR, CropSubject
from const import (
    CUDA_GPUS,
    SUBJECT_ZEROSHOT_MODEL,
    SUBJECT_CLASSES,
    SUBJECT_ZEROSHOT_BATCH_SIZE,
)


@click.command("step09-crop-subject")
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
def step09_crop_subject(pipeline_batch_id: int, overwrite: bool = False):
    """
    Uses a zero-shot classification model to detect subject labels from VLM OCR text for each crop.
    Populates ranked labels and scores in CropSubject records.
    Spins up 1 process per available CUDA GPU.

    Runs FP16 inference via the HuggingFace zero-shot-classification pipeline with batch prediction.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    if not CUDA_GPUS:
        logger.error("No CUDA devices available.")
        click.get_current_context().exit(1)

    # Skip items that have already been processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_subjects = set(
            CropSubject.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(
                Scan.issue << issue_ids,
                CropSubject.ranked_labels.is_null(False),
            )
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_subjects
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    # Split items across CUDA GPUs, balanced by crop count per item
    num_gpus = len(CUDA_GPUS)
    item_ids = [item.id for item in items_to_process]
    crop_weights = utils.get_crop_counts_by_item(item_ids)
    chunks = utils.distribute_to_gpus(item_ids, crop_weights, num_gpus)

    with ProcessPoolExecutor(max_workers=num_gpus, initializer=utils.get_db) as executor:
        futures = {}

        for gpu_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            device = CUDA_GPUS[gpu_index]
            future = executor.submit(_process_batch, chunk, device)
            futures[future] = device

        for future in as_completed(futures):
            device = futures[future]

            try:
                check = future.result()
                assert check
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error(f"Subject detection failed on {device}. Exiting.")
                click.get_current_context().exit(1)


def _process_batch(item_ids: list[int], device: str) -> bool:
    """Runs zero-shot subject classification for a subset of pipeline batch items on a single CUDA device."""
    from transformers import pipeline as hf_pipeline

    classifier = hf_pipeline(
        "zero-shot-classification",
        model=SUBJECT_ZEROSHOT_MODEL,
        dtype="float16",
        device=device,
    )

    non_empty_texts: list[str] = []
    non_empty_crop_ids: list[int] = []

    # Pre-fetch all DB data in bulk (4 queries instead of 4 per item)
    items = list(
        PipelineBatchItem.select(PipelineBatchItem, Issue)
        .join(Issue)
        .where(PipelineBatchItem.id << item_ids)
    )

    issue_ids = [item.issue_id for item in items]

    all_crops = list(
        Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << issue_ids)
    )

    crops_by_issue: dict[int, list[Crop]] = {}
    for crop in all_crops:
        crops_by_issue.setdefault(crop.scan.issue_id, []).append(crop)

    all_crop_ids = [crop.id for crop in all_crops]

    existing_subject_crop_ids: set[int] = set()
    if all_crop_ids:
        existing_subject_crop_ids = {
            row[0]
            for row in CropSubject.select(CropSubject.crop)
            .where(CropSubject.crop << all_crop_ids)
            .tuples()
        }

    ocr_by_crop_id: dict[int, CropOCR] = {}
    if all_crop_ids:
        for ocr in CropOCR.select().where(CropOCR.crop << all_crop_ids):
            ocr_by_crop_id[ocr.crop_id] = ocr

    for item in items:
        for crop in crops_by_issue.get(item.issue_id, []):
            try:
                ocr = ocr_by_crop_id.get(crop.id)
                raw_text = ocr.vlm_text if ocr and ocr.vlm_text else ""
                flat_text = utils.flatten_ocr_text(raw_text) if raw_text else ""
            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"Could not prepare text for crop #{crop.id}. Skipping.")
                flat_text = ""

            if flat_text.strip():
                non_empty_texts.append(flat_text)
                non_empty_crop_ids.append(crop.id)

    # Run inference on all collected texts
    result_by_crop_id: dict[int, dict] = {}

    if non_empty_texts:
        candidate_labels = list(SUBJECT_CLASSES[0])
        results = classifier(
            non_empty_texts,
            candidate_labels=candidate_labels,
            multi_label=False,
            batch_size=SUBJECT_ZEROSHOT_BATCH_SIZE,
        )

        if isinstance(results, dict):
            results = [results]

        for j, result in enumerate(results):
            result_by_crop_id[non_empty_crop_ids[j]] = result

    # Build CropSubject records
    entries_to_create: list[CropSubject] = []
    entries_to_update: list[CropSubject] = []

    for crop_id in all_crop_ids:
        if crop_id in result_by_crop_id:
            result = result_by_crop_id[crop_id]
            ranked_labels = result["labels"]
            scores = [float(s) for s in result["scores"]]
        else:
            ranked_labels = []
            scores = []

        record = CropSubject(
            crop=crop_id,
            ranked_labels=ranked_labels,
            scores=scores,
        )

        if crop_id in existing_subject_crop_ids:
            entries_to_update.append(record)
        else:
            entries_to_create.append(record)

    logger.info(
        f"{len(all_crop_ids)} crops processed for subject detection on {device}. "
        f"{len(non_empty_texts)} texts classified. "
        f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
    )

    # bulk_update doesn't work with ArrayField (peewee casts CASE as text, not text[]),
    # so delete-and-recreate instead.
    if entries_to_update:
        update_crop_ids = [entry.crop_id for entry in entries_to_update]
        CropSubject.delete().where(CropSubject.crop << update_crop_ids).execute()
        entries_to_create.extend(entries_to_update)
        entries_to_update.clear()

    utils.process_db_write_batch(
        model=CropSubject,
        entries_to_create=entries_to_create,
    )

    return True
