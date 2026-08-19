import io
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import NamedTuple

import click
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from loguru import logger
from ultralytics import YOLO

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop, CropClassification
from const import (
    CPUS_LIMIT,
    CUDA_GPUS,
    CROP_CLASSIFICATION_IMAGE_MODEL,
    CROP_CLASSIFICATION_IMAGE_IMGSZ,
    CROP_CLASSIFICATION_IMAGE_BATCH_SIZE,
    CROP_CLASSIFICATION_IMAGE_PREP_WORKERS,
)


class RawCrop(NamedTuple):
    """A crop loaded from cache, before image decoding."""

    crop: Crop
    jpeg_bytes: bytes
    has_existing_record: bool


@click.command("step06-crop-classification-image")
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
def step06_crop_classification_image(pipeline_batch_id: int, overwrite: bool = False):
    """
    Uses a YOLO image classifier to categorize each crop based on its visual content.
    Populates image_category and image_confidence_score in CropClassification records.
    Spins up 1 process per available CUDA GPU.

    Runs FP16 inference and applies classification transforms in the thread pool, passing pre-processed tensors to bypass ultralytics' single-threaded image preprocessing path.
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

        issues_with_image_cls = set(
            CropClassification.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(
                Scan.issue << issue_ids,
                CropClassification.image_category.is_null(False),
            )
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_image_cls
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    # Download model once in parent process so workers find it cached
    model_path = snapshot_download(CROP_CLASSIFICATION_IMAGE_MODEL)

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
            future = executor.submit(_process_batch, chunk, device, model_path, num_gpus)
            futures[future] = device

        for future in as_completed(futures):
            device = futures[future]

            try:
                check = future.result()
                assert check
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error(f"Image classification failed on {device}. Exiting.")
                click.get_current_context().exit(1)


def _process_batch(
    item_ids: list[int],
    device: str,
    model_path: str,
    num_gpus: int,
) -> bool:
    """
    Runs image classification for a subset of pipeline batch items on a single CUDA device.
    Uses triple-buffering: while the GPU runs inference on batch N, a thread pool fetches +
    decodes images for batch N+1, and a background thread writes batch N-1 results to DB.
    """
    model = YOLO(f"{model_path}/best.pt")
    transforms = model.model.transforms

    # Pre-fetch all DB data in bulk (3 queries instead of 3 per item)
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
    existing_cls_crop_ids: set[int] = set()
    if all_crop_ids:
        existing_cls_crop_ids = {
            row[0]
            for row in CropClassification.select(CropClassification.crop)
            .where(CropClassification.crop << all_crop_ids)
            .tuples()
        }

    # Pre-flatten all crops into a single list for batch-level processing
    crop_entries: list[tuple[Crop, bool]] = []
    for item in items:
        for crop in crops_by_issue.get(item.issue_id, []):
            crop_entries.append((crop, crop.id in existing_cls_crop_ids))

    batches = [
        crop_entries[i : i + CROP_CLASSIFICATION_IMAGE_BATCH_SIZE]
        for i in range(0, len(crop_entries), CROP_CLASSIFICATION_IMAGE_BATCH_SIZE)
    ]

    cache = utils.get_cache()
    prep_workers = CROP_CLASSIFICATION_IMAGE_PREP_WORKERS or max(1, CPUS_LIMIT // num_gpus)
    total_classified = 0

    # Triple-buffering: while GPU runs inference on one batch,
    # the thread pool fetches + decodes images for the next.
    # DB writes run in the background on a dedicated thread.
    pending_batch: list[RawCrop] | None = None
    pending_images: list[torch.Tensor] | None = None
    db_write_future: Future | None = None

    with (
        ThreadPoolExecutor(max_workers=prep_workers) as prep_executor,
        ThreadPoolExecutor(max_workers=1) as db_executor,
    ):
        for batch in batches:
            # Submit parallel cache reads + image decodes + transform for this batch
            fetch_futures = [
                prep_executor.submit(
                    _fetch_and_decode_crop, cache, crop, has_existing, transforms
                )
                for crop, has_existing in batch
            ]

            # While fetch runs, infer the previously prepped batch
            if pending_batch is not None:
                create, update = _classify_batch(pending_batch, pending_images, model, device)
                total_classified += len(pending_batch)

                if db_write_future is not None:
                    db_write_future.result()
                db_write_future = db_executor.submit(
                    utils.process_db_write_batch,
                    model=CropClassification,
                    entries_to_create=create,
                    entries_to_update=update,
                    fields_to_update=[
                        CropClassification.image_category,
                        CropClassification.image_confidence_score,
                    ],
                )

            # Collect fetch+decode+transform results for the current batch
            raw_crops: list[RawCrop] = []
            tensors: list[torch.Tensor] = []
            for future in fetch_futures:
                try:
                    raw_crop, tensor = future.result()
                    raw_crops.append(raw_crop)
                    tensors.append(tensor)
                except Exception:
                    logger.debug(traceback.format_exc())
                    logger.warning(f"Failed to fetch/decode crop on {device}. Skipping.")

            if raw_crops:
                pending_batch = raw_crops
                pending_images = tensors
            else:
                pending_batch = None
                pending_images = None

        # Flush the last pending batch
        if pending_batch is not None:
            create, update = _classify_batch(pending_batch, pending_images, model, device)
            total_classified += len(pending_batch)

            if db_write_future is not None:
                db_write_future.result()
            db_write_future = db_executor.submit(
                utils.process_db_write_batch,
                model=CropClassification,
                entries_to_create=create,
                entries_to_update=update,
                fields_to_update=[
                    CropClassification.image_category,
                    CropClassification.image_confidence_score,
                ],
            )

        # Wait for final DB write
        if db_write_future is not None:
            db_write_future.result()

    logger.info(f"{total_classified} crops classified by image on {device}.")

    return True


def _fetch_and_decode_crop(
    cache, crop: Crop, has_existing_record: bool, transforms
) -> tuple[RawCrop, torch.Tensor]:
    """Reads crop JPEG from cache, decodes it, and applies classification transforms."""
    jpeg_bytes = cache.get(crop.cache_key, retry=True)
    if jpeg_bytes is None:
        raise RuntimeError(
            f"Crop #{crop.id} not found in cache (key: {crop.cache_key}). Cannot proceed."
        )
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    tensor = transforms(image)
    return RawCrop(crop=crop, jpeg_bytes=jpeg_bytes, has_existing_record=has_existing_record), tensor


def _classify_batch(
    batch: list[RawCrop],
    tensors: list[torch.Tensor],
    model: YOLO,
    device: str,
) -> tuple[list[CropClassification], list[CropClassification]]:
    """Runs YOLO classification inference on a pre-transformed batch. Returns (to_create, to_update)."""
    entries_to_create: list[CropClassification] = []
    entries_to_update: list[CropClassification] = []

    stacked = torch.stack(tensors)
    results = model.predict(
        stacked,
        device=device,
        imgsz=CROP_CLASSIFICATION_IMAGE_IMGSZ,
        half=True,
        verbose=False,
    )

    for entry, result in zip(batch, results):
        try:
            class_name = model.names[result.probs.top1]
            confidence = float(result.probs.top1conf)

            record = CropClassification(
                crop=entry.crop.id,
                image_category=class_name,
                image_confidence_score=confidence,
            )

            if entry.has_existing_record:
                entries_to_update.append(record)
            else:
                entries_to_create.append(record)

        except Exception:
            logger.debug(traceback.format_exc())
            logger.warning(f"Failed to classify crop #{entry.crop.id}. Skipping.")

    return entries_to_create, entries_to_update
