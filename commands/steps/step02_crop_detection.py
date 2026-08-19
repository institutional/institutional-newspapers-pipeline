import io
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import NamedTuple

import click
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from loguru import logger
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop
from const import (
    CPUS_LIMIT,
    CUDA_GPUS,
    SCAN_JPEG_QUALITY,
    CROP_DETECTION_MODEL,
    CROP_DETECTION_IMGSZ,
    CROP_DETECTION_CONF,
    CROP_DETECTION_IOU,
    CROP_DETECTION_MAX_DET,
    CROP_DETECTION_BATCH_SIZE,
)


class RawScan(NamedTuple):
    """A scan loaded from cache, before image decoding."""

    scan: Scan
    jpeg_bytes: bytes
    pipeline_batch_item_id: int
    issue: Issue


@dataclass
class IssueProgress:
    """Tracks crop detection progress for a single issue, for logging."""

    issue: Issue
    total_scans: int
    processed_scans: int = 0
    total_crops: int = 0


@click.command("step02-crop-detection")
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
def step02_crop_detection(pipeline_batch_id: int, overwrite: bool = False):
    """
    Uses a YOLO object detection model to detect individual crops in each scan of the current batch. Spins up 1 process per available CUDA GPU.

    Runs FP16 inference and parallelizes LetterBox preprocessing across the thread pool to bypass ultralytics' single-threaded image preprocessing path.
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

        issues_with_crops = set(
            Crop.select(Scan.issue).join(Scan).where(Scan.issue << issue_ids).distinct().tuples()
        )

        items_to_process = [item for item in all_items if (item.issue_id,) not in issues_with_crops]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    # Download model once in parent process so workers find it cached
    model_path = snapshot_download(CROP_DETECTION_MODEL)

    # Split items across CUDA GPUs, balanced by scan count per issue
    num_gpus = len(CUDA_GPUS)
    item_ids = [item.id for item in items_to_process]
    issue_ids = [item.issue_id for item in items_to_process]
    scan_weights = utils.get_scan_counts_by_issue(issue_ids)
    item_weights = {item.id: scan_weights.get(item.issue_id, 1) for item in items_to_process}
    chunks = utils.distribute_to_gpus(item_ids, item_weights, num_gpus)

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
                logger.error(f"Crop detection failed on {device}. Exiting.")
                click.get_current_context().exit(1)


def _process_batch(
    item_ids: list[int],
    device: str,
    model_path: str,
    num_gpus: int,
) -> bool:
    """
    Runs crop detection for a subset of pipeline batch items on a single CUDA device.
    Uses double-buffering: while the GPU runs inference on batch N, a thread pool decodes
    images for batch N+1. Post-processing (crop extraction, JPEG encoding, cache writes)
    is also submitted to the thread pool.
    """
    model = YOLO(f"{model_path}/best.pt")
    cache = utils.get_cache()

    letterbox = LetterBox(
        CROP_DETECTION_IMGSZ,
        auto=False,
        stride=int(model.stride.max()),
    )

    raw_buffer: list[RawScan] = []
    issue_progress: dict[int, IssueProgress] = {}

    prep_workers = max(4, CPUS_LIMIT // num_gpus)

    # Pre-fetch all DB data in bulk (2 queries instead of 2 per item)
    items = list(
        PipelineBatchItem.select(PipelineBatchItem, Issue)
        .join(Issue)
        .where(PipelineBatchItem.id << item_ids)
    )

    issue_ids = [item.issue_id for item in items]

    all_scans = list(
        Scan.select(Scan, Issue).join(Issue).where(Scan.issue << issue_ids)
    )

    scans_by_issue: dict[int, list[Scan]] = {}
    for scan in all_scans:
        scans_by_issue.setdefault(scan.issue_id, []).append(scan)

    # Double-buffering: while GPU runs inference on one batch,
    # the thread pool decodes + preprocesses images for the next.
    pending_batch: list[RawScan] | None = None
    pending_futures: list[Future] | None = None

    with ThreadPoolExecutor(max_workers=prep_workers) as thread_pool:
        for item in items:
            issue: Issue = item.issue
            scans = scans_by_issue.get(issue.id, [])

            if not scans:
                continue

            # Register progress tracker before loading so it exists when batches drain
            issue_progress[issue.id] = IssueProgress(issue=issue, total_scans=0)

            loaded_count = 0

            for scan in scans:
                jpeg_bytes = cache.get(scan.cache_key)
                if jpeg_bytes is None:
                    logger.warning(
                        f"{issue.archive_filename}: scan {scan.scan_filename} "
                        f"not in cache. Skipping."
                    )
                    continue

                raw_buffer.append(
                    RawScan(
                        scan=scan,
                        jpeg_bytes=jpeg_bytes,
                        pipeline_batch_item_id=item.id,
                        issue=issue,
                    )
                )
                loaded_count += 1

                # Drain full batches as we go
                while len(raw_buffer) >= CROP_DETECTION_BATCH_SIZE:
                    batch = raw_buffer[:CROP_DETECTION_BATCH_SIZE]
                    raw_buffer = raw_buffer[CROP_DETECTION_BATCH_SIZE:]

                    # Submit per-image decode+preprocess futures in parallel
                    batch_futures = [
                        thread_pool.submit(
                            _decode_and_preprocess_scan, entry.jpeg_bytes, letterbox
                        )
                        for entry in batch
                    ]

                    if pending_futures is not None:
                        _process_buffered_batch(
                            pending_batch,
                            pending_futures,
                            model,
                            device,
                            cache,
                            issue_progress,
                            thread_pool,
                        )

                    pending_batch = batch
                    pending_futures = batch_futures

            if loaded_count == 0:
                del issue_progress[issue.id]
                continue

            issue_progress[issue.id].total_scans = loaded_count

            # Log if all scans were already processed during mid-loop drains
            progress = issue_progress[issue.id]
            if progress.processed_scans == progress.total_scans:
                logger.info(
                    f"{progress.issue.archive_filename} ({progress.issue.corpus}): "
                    f"{progress.total_crops} crops across "
                    f"{progress.total_scans} scans ({device})"
                )

        # Flush last pending batch
        if pending_futures is not None:
            _process_buffered_batch(
                pending_batch,
                pending_futures,
                model,
                device,
                cache,
                issue_progress,
                thread_pool,
            )

        # Flush remaining partial buffer
        if raw_buffer:
            remaining_futures = [
                thread_pool.submit(
                    _decode_and_preprocess_scan, entry.jpeg_bytes, letterbox
                )
                for entry in raw_buffer
            ]
            _process_buffered_batch(
                raw_buffer,
                remaining_futures,
                model,
                device,
                cache,
                issue_progress,
                thread_pool,
            )

    return True


def _decode_and_preprocess_scan(
    jpeg_bytes: bytes,
    letterbox: LetterBox,
) -> tuple[Image.Image, np.ndarray]:
    """Decodes JPEG bytes and applies LetterBox preprocessing. Runs in a thread pool."""
    pil_image = Image.open(io.BytesIO(jpeg_bytes))
    rgb_array = np.asarray(pil_image)
    bgr_array = np.ascontiguousarray(rgb_array[..., ::-1])
    preprocessed = letterbox(image=bgr_array)
    return pil_image, preprocessed


def _scale_boxes_to_original(
    xyxy: np.ndarray,
    original_width: int,
    original_height: int,
    imgsz: int = CROP_DETECTION_IMGSZ,
) -> np.ndarray:
    """Reverses LetterBox preprocessing to map bounding boxes back to original image coordinates."""
    r = min(imgsz / original_height, imgsz / original_width)
    pad_w = (imgsz - round(original_width * r)) / 2
    pad_h = (imgsz - round(original_height * r)) / 2
    scaled = xyxy.copy()
    scaled[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / r
    scaled[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / r
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, original_width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, original_height)
    return scaled


def _save_crops_for_scan(
    source_image: Image.Image,
    saved_crops: list[Crop],
    cache,
) -> None:
    """Crops, JPEG-encodes, and writes each detected crop to cache. Runs in a thread pool."""
    for saved_crop in saved_crops:
        x1, y1, x2, y2 = [int(v) for v in saved_crop.bbox_xyxy]
        crop_image = source_image.crop((x1, y1, x2, y2))

        jpeg_buffer = io.BytesIO()
        crop_image.save(jpeg_buffer, format="JPEG", quality=SCAN_JPEG_QUALITY)
        cache.set(saved_crop.cache_key, jpeg_buffer.getvalue())


def _process_buffered_batch(
    batch: list[RawScan],
    prep_futures: list[Future],
    model: YOLO,
    device: str,
    cache,
    issue_progress: dict[int, IssueProgress],
    thread_pool: ThreadPoolExecutor,
) -> None:
    """
    Runs YOLO inference on a batch of pre-decoded scans and post-processes each result.
    Crop extraction and cache writes are submitted to the thread pool.
    """
    # Collect parallel decode+preprocess results
    originals: list[Image.Image] = []
    preprocessed: list[np.ndarray] = []
    for future in prep_futures:
        pil_img, prepped = future.result()
        originals.append(pil_img)
        preprocessed.append(prepped)

    # Bulk delete existing crops for all scans in this batch
    scan_ids = [entry.scan.id for entry in batch]
    Crop.delete().where(Crop.scan << scan_ids).execute()

    results = model.predict(
        preprocessed,
        device=device,
        imgsz=CROP_DETECTION_IMGSZ,
        conf=CROP_DETECTION_CONF,
        iou=CROP_DETECTION_IOU,
        max_det=CROP_DETECTION_MAX_DET,
        half=True,
        verbose=False,
    )

    save_futures: list[Future] = []

    for entry, image, result in zip(batch, originals, results):
        progress = issue_progress[entry.issue.id]

        try:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                progress.processed_scans += 1
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            xyxy = _scale_boxes_to_original(xyxy, image.width, image.height)
            confs = boxes.conf.cpu().numpy()

            crops_to_create: list[Crop] = []
            for det_idx in range(len(xyxy)):
                bbox = xyxy[det_idx].tolist()
                width = int(bbox[2] - bbox[0])
                height = int(bbox[3] - bbox[1])
                if width <= 0 or height <= 0:
                    continue
                crop = Crop(
                    scan=entry.scan.id,
                    pipeline_batch_item=entry.pipeline_batch_item_id,
                    bbox_xyxy=bbox,
                    width=width,
                    height=height,
                    confidence_score=float(confs[det_idx]),
                    reading_order=0,
                )
                crops_to_create.append(crop)

            utils.process_db_write_batch(Crop, entries_to_create=crops_to_create)

            # Re-query crops to get database-assigned IDs for cache keys
            saved_crops = list(
                Crop.select(Crop, Scan, Issue)
                .join(Scan)
                .join(Issue)
                .where(Crop.scan == entry.scan.id)
            )

            save_futures.append(
                thread_pool.submit(_save_crops_for_scan, image, saved_crops, cache)
            )

            progress.total_crops += len(saved_crops)
            progress.processed_scans += 1

        except Exception:
            logger.debug(traceback.format_exc())
            logger.warning(
                f"Failed to post-process scan {entry.scan.scan_filename} "
                f"for {entry.issue.archive_filename} on {device}. Skipping scan."
            )
            progress.processed_scans += 1

        # Log when all scans for an issue have been processed
        if progress.processed_scans == progress.total_scans:
            logger.info(
                f"{progress.issue.archive_filename} ({progress.issue.corpus}): "
                f"{progress.total_crops} crops across "
                f"{progress.total_scans} scans ({device})"
            )

    # Wait for all crop-save futures to avoid unbounded memory growth
    for future in save_futures:
        future.result()
