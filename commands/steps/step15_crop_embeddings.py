import io
import multiprocessing
import traceback
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import NamedTuple

import click
from loguru import logger

import utils
from models import (
    PipelineBatch,
    PipelineBatchItem,
    Issue,
    Scan,
    Crop,
    CropOCR,
    CropTextStaticEmbedding,
    CropImageEmbedding,
)
from const import (
    CPUS_LIMIT,
    CUDA_GPUS,
    STATIC_TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_BATCH_SIZE,
    DB_IN_CLAUSE_CHUNK_SIZE,
)

_MP_CONTEXT = multiprocessing.get_context("forkserver")


class CropTextEmbeddingInput(NamedTuple):
    crop_id: int
    vlm_raw: str | None


class CropTextEmbeddingResult(NamedTuple):
    crop_id: int
    vlm_embedding: list[float] | None


class RawCrop(NamedTuple):
    """A crop loaded from cache, before image decoding."""

    crop: Crop
    jpeg_bytes: bytes


@click.command("step15-crop-embeddings")
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
def step15_crop_embeddings(pipeline_batch_id: int, overwrite: bool = False):
    """
    Generates text embeddings and image embeddings concurrently for every crop in the pipeline batch.

    Text embeddings run across a ProcessPool on CPU using a static embedding model. Image embeddings spin up 1 process per available CUDA GPU with double-buffering (thread pool decodes and preprocesses images for the next batch while the GPU infers the current one). Both run concurrently in a top-level thread pool.
    """
    results: dict[str, bool | None] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                StaticTextEmbedding.run, pipeline_batch_id, overwrite
            ): "text-embedding",
            executor.submit(ImageEmbedding.run, pipeline_batch_id, overwrite): "image-embedding",
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error(f"[{name}] Failed during concurrent embedding.")
                results[name] = None

    failed = [name for name, result in results.items() if result is None]

    if failed:
        logger.error(f"Embedding failed for: {', '.join(failed)}")
        click.get_current_context().exit(1)

    skipped = [name for name, result in results.items() if result is False]
    if skipped:
        logger.info(f"Nothing to process for: {', '.join(skipped)}")


class StaticTextEmbedding:
    """CPU-bound: generates static text embeddings using model2vec."""

    @staticmethod
    def run(pipeline_batch_id: int, overwrite: bool = False) -> bool:
        """Returns True on success, False if nothing to process. Raises on error."""
        pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
        all_items = pipeline_batch.items

        if overwrite:
            items_to_process = list(all_items)
        else:
            issue_ids = [item.issue_id for item in all_items]

            issues_with_embeddings = set(
                CropTextStaticEmbedding.select(Scan.issue)
                .join(Crop)
                .join(Scan)
                .where(
                    Scan.issue << issue_ids,
                    CropTextStaticEmbedding.vlm_embedding.is_null(False),
                )
                .distinct()
                .tuples()
            )

            items_to_process = [
                item for item in all_items if (item.issue_id,) not in issues_with_embeddings
            ]

            skipped = len(all_items) - len(items_to_process)
            if skipped:
                logger.info(
                    f"[text-embedding] {skipped} items already processed. "
                    f"Skipping (use --overwrite to redo)."
                )

        if not items_to_process:
            logger.warning("[text-embedding] No items to process.")
            return False

        issue_ids = [item.issue_id for item in items_to_process]

        all_crops = list(
            Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << issue_ids)
        )

        if not all_crops:
            logger.warning("[text-embedding] No crops found for the given items.")
            return False

        crop_ids = [crop.id for crop in all_crops]

        # Load CropOCR data in chunks
        ocr_by_crop_id: dict[int, CropOCR] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for ocr in CropOCR.select().where(CropOCR.crop << chunk):
                ocr_by_crop_id[ocr.crop_id] = ocr

        # Delete existing records so all new results go through bulk_create
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            CropTextStaticEmbedding.delete().where(CropTextStaticEmbedding.crop << chunk).execute()

        # Build worker inputs
        inputs: list[CropTextEmbeddingInput] = []
        for crop_id in crop_ids:
            ocr = ocr_by_crop_id.get(crop_id)
            vlm_raw = ocr.vlm_text if ocr and ocr.vlm_text else None
            inputs.append(CropTextEmbeddingInput(crop_id, vlm_raw))

        # Split into chunks and process with independent model-loading workers
        num_workers = CPUS_LIMIT // 2
        chunk_size = max(1, len(inputs) // num_workers)
        chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]

        all_results: list[CropTextEmbeddingResult] = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=_MP_CONTEXT) as pool:
            for result_batch in pool.map(StaticTextEmbedding._encode_chunk, chunks):
                all_results.extend(result_batch)

        # Build CropTextStaticEmbedding records
        entries: list[CropTextStaticEmbedding] = []
        non_empty_count = 0

        for result in all_results:
            if result.vlm_embedding is not None:
                non_empty_count += 1
            entries.append(
                CropTextStaticEmbedding(
                    crop=result.crop_id,
                    vlm_embedding=result.vlm_embedding,
                )
            )

        empty_count = len(crop_ids) - non_empty_count

        logger.info(
            f"[text-embedding] {len(crop_ids)} crops processed. "
            f"{non_empty_count} with text, {empty_count} empty/failed."
        )

        utils.process_db_write_batch(
            model=CropTextStaticEmbedding,
            entries_to_create=entries,
        )

        return True

    @staticmethod
    def _encode_chunk(chunk: list[CropTextEmbeddingInput]) -> list[CropTextEmbeddingResult]:
        """Worker: loads model, flattens text, encodes embeddings for a chunk of crops."""
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(STATIC_TEXT_EMBEDDING_MODEL)

        # Flatten and separate empty from non-empty
        flat_texts: list[str] = []
        flat_indices: list[int] = []
        for i, inp in enumerate(chunk):
            try:
                flat = utils.flatten_ocr_text(inp.vlm_raw) if inp.vlm_raw else ""
            except Exception:
                flat = ""
            if flat.strip():
                flat_texts.append(flat)
                flat_indices.append(i)

        # Encode non-empty texts (disable internal multiprocessing — we ARE the workers)
        embeddings_map: dict[int, list[float]] = {}
        if flat_texts:
            embeddings = model.encode(
                flat_texts,
                batch_size=1024,
                use_multiprocessing=False,
            )
            for j, idx in enumerate(flat_indices):
                embeddings_map[idx] = embeddings[j].tolist()

        return [
            CropTextEmbeddingResult(inp.crop_id, embeddings_map.get(i))
            for i, inp in enumerate(chunk)
        ]


class ImageEmbedding:
    """GPU-bound: generates image embeddings using DINOv2."""

    @staticmethod
    def run(pipeline_batch_id: int, overwrite: bool = False) -> bool:
        """Returns True on success, False if nothing to process. Raises on error."""
        from huggingface_hub import snapshot_download

        pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
        all_items = pipeline_batch.items

        if not CUDA_GPUS:
            raise RuntimeError("No CUDA devices available.")

        if overwrite:
            items_to_process = list(all_items)
        else:
            issue_ids = [item.issue_id for item in all_items]

            issues_with_embeddings = set(
                CropImageEmbedding.select(Scan.issue)
                .join(Crop)
                .join(Scan)
                .where(
                    Scan.issue << issue_ids,
                    CropImageEmbedding.embedding.is_null(False),
                )
                .distinct()
                .tuples()
            )

            items_to_process = [
                item for item in all_items if (item.issue_id,) not in issues_with_embeddings
            ]

            skipped = len(all_items) - len(items_to_process)
            if skipped:
                logger.info(
                    f"[image-embedding] {skipped} items already processed. "
                    f"Skipping (use --overwrite to redo)."
                )

        if not items_to_process:
            logger.warning("[image-embedding] No items to process.")
            return False

        # Download model once in parent so workers find it cached
        snapshot_download(IMAGE_EMBEDDING_MODEL)

        # Split items across CUDA GPUs, balanced by crop count per item
        num_gpus = len(CUDA_GPUS)
        item_ids = [item.id for item in items_to_process]
        crop_weights = utils.get_crop_counts_by_item(item_ids)
        chunks = utils.distribute_to_gpus(item_ids, crop_weights, num_gpus)

        with ProcessPoolExecutor(
            max_workers=num_gpus, initializer=utils.get_db, mp_context=_MP_CONTEXT,
        ) as pool:
            futures = {}

            for gpu_index, chunk in enumerate(chunks):
                if not chunk:
                    continue
                device = CUDA_GPUS[gpu_index]
                future = pool.submit(ImageEmbedding._process_batch, chunk, device, num_gpus)
                futures[future] = device

            for future in as_completed(futures):
                device = futures[future]
                try:
                    check = future.result()
                    assert check
                except Exception:
                    logger.debug(traceback.format_exc())
                    raise RuntimeError(f"Image embedding failed on {device}.")

        return True

    @staticmethod
    def _process_batch(
        item_ids: list[int],
        device: str,
        num_gpus: int,
    ) -> bool:
        """
        Generates image embeddings for a subset of pipeline batch items on a single CUDA device.
        Uses double-buffering: while the GPU runs inference on batch N, a thread pool decodes
        images and runs the HuggingFace processor for batch N+1.
        """
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image

        processor = AutoImageProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL, use_fast=True)
        model = AutoModel.from_pretrained(IMAGE_EMBEDDING_MODEL).to(device)
        model.eval()

        cache = utils.get_cache()

        raw_buffer: list[RawCrop] = []
        entries_to_create: list[CropImageEmbedding] = []
        total_embedded = 0

        prep_workers = max(4, CPUS_LIMIT // 2)

        # Pre-fetch all DB data in bulk
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

        # Delete existing records so all new results go through bulk_create
        all_crop_ids = [crop.id for crop in all_crops]
        for i in range(0, len(all_crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = all_crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            CropImageEmbedding.delete().where(CropImageEmbedding.crop << chunk).execute()

        # Double-buffering: while GPU runs inference on one batch,
        # the thread pool decodes images and runs the processor for the next.
        pending_batch: list[RawCrop] | None = None
        pending_future: Future | None = None

        with ThreadPoolExecutor(max_workers=prep_workers) as prep_executor:
            for item in items:
                for crop in crops_by_issue.get(item.issue_id, []):
                    jpeg_bytes = cache.get(crop.cache_key)
                    if jpeg_bytes is None:
                        logger.warning(
                            f"Crop #{crop.id} for {item.issue.archive_filename} "
                            f"not in cache. Skipping."
                        )
                        continue

                    raw_buffer.append(RawCrop(crop=crop, jpeg_bytes=jpeg_bytes))

                    # Drain full batches as we go
                    while len(raw_buffer) >= IMAGE_EMBEDDING_BATCH_SIZE:
                        batch = raw_buffer[:IMAGE_EMBEDDING_BATCH_SIZE]
                        raw_buffer = raw_buffer[IMAGE_EMBEDDING_BATCH_SIZE:]

                        batch_future = prep_executor.submit(
                            ImageEmbedding._load_and_preprocess_batch, batch, processor
                        )

                        # Run inference on the previously prepped batch while
                        # the thread pool decodes images for the current one.
                        if pending_future is not None:
                            ImageEmbedding._run_inference_batch(
                                pending_batch,
                                pending_future.result(),
                                model,
                                device,
                                entries_to_create,
                            )
                            total_embedded += len(pending_batch)

                        pending_batch = batch
                        pending_future = batch_future

            # Flush last pending batch
            if pending_future is not None:
                ImageEmbedding._run_inference_batch(
                    pending_batch,
                    pending_future.result(),
                    model,
                    device,
                    entries_to_create,
                )
                total_embedded += len(pending_batch)

            # Flush remaining partial buffer
            if raw_buffer:
                flush_inputs = ImageEmbedding._load_and_preprocess_batch(raw_buffer, processor)
                ImageEmbedding._run_inference_batch(
                    raw_buffer,
                    flush_inputs,
                    model,
                    device,
                    entries_to_create,
                )
                total_embedded += len(raw_buffer)

        logger.info(f"[image-embedding] {total_embedded} crops embedded on {device}.")

        utils.process_db_write_batch(
            model=CropImageEmbedding,
            entries_to_create=entries_to_create,
        )

        # Release GPU resources eagerly so the worker process exits quickly
        del model
        torch.cuda.empty_cache()

        return True

    @staticmethod
    def _load_and_preprocess_batch(
        batch: list[RawCrop],
        processor,
    ) -> dict[str, "torch.Tensor"]:
        """Decodes JPEG bytes and runs the image processor for a batch.
        Runs in a thread pool -- both PIL decode and the processor release the GIL.
        """
        from PIL import Image

        images = [
            Image.open(io.BytesIO(rc.jpeg_bytes)).convert("RGB").resize((448, 448)) for rc in batch
        ]
        return processor(images=images, return_tensors="pt")

    @staticmethod
    def _run_inference_batch(
        batch: list[RawCrop],
        inputs: dict[str, "torch.Tensor"],
        model,
        device: str,
        entries_to_create: list[CropImageEmbedding],
    ) -> None:
        """Runs DINOv2 inference on pre-processed inputs and builds DB records."""
        import torch

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token embedding (first token of last hidden state)
        embeddings = outputs.last_hidden_state[:, 0]

        for entry, embedding in zip(batch, embeddings):
            try:
                entries_to_create.append(
                    CropImageEmbedding(
                        crop=entry.crop.id,
                        embedding=embedding.cpu().tolist(),
                    )
                )
            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"Failed to embed crop #{entry.crop.id}. Skipping.")
