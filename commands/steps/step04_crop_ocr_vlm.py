import io
import os
import socket
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TypedDict, NamedTuple

import click
from PIL import Image
from huggingface_hub import snapshot_download
from loguru import logger

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop, CropOCR
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from const import (
    CPUS_LIMIT,
    CUDA_GPUS,
    OCR_VLM_MODEL,
    OCR_VLM_MODEL_CONTEXT,
    OCR_VLM_MAX_MODEL_LEN,
    OCR_VLM_MAX_PIXELS,
    OCR_VLM_MIN_PIXELS,
    OCR_VLM_GPU_MEMORY_UTILIZATION,
    OCR_VLM_BATCH_SIZE,
    OCR_VLM_PREP_WORKERS,
    OCR_VLM_CHUNKED_PREFILL,
    OCR_VLM_PREFIX_CACHING,
    OCR_VLM_COMPILE_MM_ENCODER,
    OCR_VLM_FP8_KV_CACHE,
    OCR_VLM_MAX_ASPECT_RATIO,
    OCR_VLM_PROMPT,
    OCR_VLM_SMART_RESIZE_FACTOR,
)


class VlmInferenceMetadata(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class RawCrop(NamedTuple):
    """A crop loaded from cache, before VLM message preparation."""

    crop: Crop
    jpeg_bytes: bytes
    has_existing_record: bool


@click.command("step04-crop-ocr-vlm")
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
def step04_crop_ocr_vlm(pipeline_batch_id: int, overwrite: bool = False):
    """
    Uses a VLM to OCR every crop from the current pipeline batch.
    Stores full text and inference metadata in CropOCR records.
    Spins up 1 process per available CUDA GPU, each running a vLLM inference server.

    Uses double-buffering: a thread pool fetches and decodes images for the next batch while the GPU infers the current one. DB writes run on a background thread.
    """
    if not CUDA_GPUS:
        logger.error("No CUDA devices available.")
        click.get_current_context().exit(1)

    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items that have already been processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_vlm = set(
            CropOCR.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(Scan.issue << issue_ids, CropOCR.vlm_text.is_null(False))
            .distinct()
            .tuples()
        )

        items_to_process = [item for item in all_items if (item.issue_id,) not in issues_with_vlm]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    # Pre-download model in parent process so subprocesses find it cached
    snapshot_download(OCR_VLM_MODEL)

    # Split items across CUDA GPUs, balanced by crop count per item
    num_gpus = len(CUDA_GPUS)
    item_ids = [item.id for item in items_to_process]
    crop_weights = utils.get_crop_counts_by_item(item_ids)
    chunks = utils.distribute_to_gpus(item_ids, crop_weights, num_gpus)

    ports = _allocate_ports(num_gpus)

    with ProcessPoolExecutor(max_workers=num_gpus, initializer=utils.get_db) as executor:
        futures = {}

        for gpu_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            device = CUDA_GPUS[gpu_index]
            future = executor.submit(
                _process_gpu_batch, chunk, device, num_gpus, gpu_index, ports[gpu_index]
            )
            futures[future] = device

        for future in as_completed(futures):
            device = futures[future]

            try:
                check = future.result()
                assert check
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error(f"VLM OCR failed on {device}. Exiting.")
                click.get_current_context().exit(1)


def _allocate_ports(n: int) -> list[int]:
    """Binds N sockets simultaneously, guaranteeing distinct ports."""
    sockets = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            sockets.append(s)
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


def _process_gpu_batch(
    item_ids: list[int], device: str, num_gpus: int, gpu_index: int, master_port: int
) -> bool:
    """
    Runs VLM OCR for a subset of pipeline batch items on a single GPU.
    Loads the model once and reuses it across all batches.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")
    os.environ["MASTER_PORT"] = str(master_port)
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig
    import torch

    compilation_config = CompilationConfig(
        compile_mm_encoder=OCR_VLM_COMPILE_MM_ENCODER,
    )

    llm = LLM(
        model=OCR_VLM_MODEL,
        max_model_len=OCR_VLM_MAX_MODEL_LEN,
        gpu_memory_utilization=OCR_VLM_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_chunked_prefill=OCR_VLM_CHUNKED_PREFILL,
        enable_prefix_caching=OCR_VLM_PREFIX_CACHING,
        kv_cache_dtype="fp8" if OCR_VLM_FP8_KV_CACHE else "auto",
        compilation_config=compilation_config,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        # top_p=0.9,
        max_tokens=int(OCR_VLM_MODEL_CONTEXT * 1.5),
    )

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
    existing_ocr_crop_ids: set[int] = set()
    if all_crop_ids:
        existing_ocr_crop_ids = {
            row[0]
            for row in CropOCR.select(CropOCR.crop).where(CropOCR.crop << all_crop_ids).tuples()
        }

    # Pre-flatten all crops into a single list for batch-level processing
    crop_entries: list[tuple[Crop, bool]] = []
    for item in items:
        for crop in crops_by_issue.get(item.issue_id, []):
            crop_entries.append((crop, crop.id in existing_ocr_crop_ids))

    batches = [
        crop_entries[i : i + OCR_VLM_BATCH_SIZE]
        for i in range(0, len(crop_entries), OCR_VLM_BATCH_SIZE)
    ]

    total_crops = len(crop_entries)
    t0_gpu = time.perf_counter()

    cache = utils.get_cache()
    prep_workers = OCR_VLM_PREP_WORKERS or max(1, CPUS_LIMIT // num_gpus)

    # Double-buffering: while GPU runs inference on one batch,
    # the thread pool fetches + decodes images for the next.
    # DB writes run in the background on a dedicated thread.
    pending_batch: list[RawCrop] | None = None
    pending_images: list[Image.Image] | None = None
    db_write_future: Future | None = None

    with (
        ThreadPoolExecutor(max_workers=prep_workers) as prep_executor,
        ThreadPoolExecutor(max_workers=1) as db_executor,
    ):
        for batch in batches:
            # Submit parallel cache reads + image decodes for this batch
            fetch_futures = [
                prep_executor.submit(_fetch_and_decode_crop, cache, crop, has_existing)
                for crop, has_existing in batch
            ]

            # While fetch runs, infer the previously prepped batch
            if pending_batch is not None:
                create, update = _run_vlm_batch(
                    pending_batch, pending_images, llm, sampling_params, device
                )

                # Wait for previous DB write before submitting a new one
                if db_write_future is not None:
                    db_write_future.result()
                db_write_future = db_executor.submit(
                    utils.process_db_write_batch,
                    model=CropOCR,
                    entries_to_create=create,
                    entries_to_update=update,
                    fields_to_update=[CropOCR.vlm_text, CropOCR.vlm_metadata],
                )

            # Collect fetch+decode results for the current batch
            raw_crops: list[RawCrop] = []
            images: list[Image.Image] = []
            for future in fetch_futures:
                try:
                    raw_crop, image = future.result()
                    raw_crops.append(raw_crop)
                    images.append(image)
                except Exception:
                    logger.debug(traceback.format_exc())
                    logger.warning(f"Failed to fetch/decode crop on {device}. Skipping.")

            if raw_crops:
                pending_batch = raw_crops
                pending_images = images
            else:
                pending_batch = None
                pending_images = None

        # Flush the last pending batch
        if pending_batch is not None:
            create, update = _run_vlm_batch(
                pending_batch, pending_images, llm, sampling_params, device
            )
            if db_write_future is not None:
                db_write_future.result()
            db_write_future = db_executor.submit(
                utils.process_db_write_batch,
                model=CropOCR,
                entries_to_create=create,
                entries_to_update=update,
                fields_to_update=[CropOCR.vlm_text, CropOCR.vlm_metadata],
            )

        # Wait for final DB write
        if db_write_future is not None:
            db_write_future.result()

    elapsed_gpu = time.perf_counter() - t0_gpu
    logger.info(
        f"Finished processing {total_crops} crops on {device} in {elapsed_gpu:.1f}s "
        f"({total_crops / elapsed_gpu:.1f} crops/s)"
    )

    # Release GPU resources eagerly so the worker process exits quickly
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()

    return True


def _fetch_and_decode_crop(
    cache, crop: Crop, has_existing_record: bool
) -> tuple[RawCrop, Image.Image]:
    """Reads crop JPEG from cache and decodes it into a PIL Image."""
    jpeg_bytes = cache.get(crop.cache_key, retry=True)
    if jpeg_bytes is None:
        raise RuntimeError(
            f"Crop #{crop.id} not found in cache (key: {crop.cache_key}). Cannot proceed."
        )
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    # Shrink to 50% unless that would drop below OCR_VLM_MIN_PIXELS
    half_h, half_w = image.height // 2, image.width // 2
    if half_h * half_w >= OCR_VLM_MIN_PIXELS:
        target_h, target_w = half_h, half_w
    else:
        target_h, target_w = image.height, image.width

    aspect_ratio = max(target_h, target_w) / min(target_h, target_w)
    if aspect_ratio > OCR_VLM_MAX_ASPECT_RATIO:
        raise ValueError(
            f"Crop #{crop.id} aspect ratio too extreme ({aspect_ratio:.1f}:1). Skipping."
        )

    new_h, new_w = smart_resize(
        target_h,
        target_w,
        factor=OCR_VLM_SMART_RESIZE_FACTOR,
        max_pixels=OCR_VLM_MAX_PIXELS,
    )
    if (new_w, new_h) != image.size:
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    return RawCrop(crop=crop, jpeg_bytes=jpeg_bytes, has_existing_record=has_existing_record), image


def _run_vlm_batch(
    raw_batch: list[RawCrop],
    images: list[Image.Image],
    llm,
    sampling_params,
    device: str,
) -> tuple[list[CropOCR], list[CropOCR]]:
    """Runs VLM inference on a batch. Returns (entries_to_create, entries_to_update)."""
    entries_to_create: list[CropOCR] = []
    entries_to_update: list[CropOCR] = []

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": OCR_VLM_PROMPT},
                ],
            }
        ]
        for image in images
    ]

    t0 = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    logger.info(
        f"{len(raw_batch)} crops processed via VLM OCR ({device}) — "
        f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens in {elapsed:.1f}s"
    )

    for entry, output in zip(raw_batch, outputs):
        try:
            vlm_text = output.outputs[0].text.strip()
            vlm_metadata = VlmInferenceMetadata(
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(output.outputs[0].token_ids),
                finish_reason=output.outputs[0].finish_reason,
            )
        except Exception:
            logger.debug(traceback.format_exc())
            logger.warning(f"VLM inference failed for crop #{entry.crop.id} on {device}.")
            vlm_text = None
            vlm_metadata = None

        record = CropOCR(
            crop=entry.crop.id,
            vlm_text=vlm_text,
            vlm_metadata=vlm_metadata,
        )

        if entry.has_existing_record:
            entries_to_update.append(record)
        else:
            entries_to_create.append(record)

    return entries_to_create, entries_to_update
