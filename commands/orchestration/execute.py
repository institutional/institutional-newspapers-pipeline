import os
import traceback
from datetime import datetime, timezone, timedelta
import time

import click
from loguru import logger

from models import PipelineRun, PipelineBatch
import commands.steps
import threading
import utils

import psutil

from const import (
    CPUS_LIMIT,
    MAX_S3_CONCURRENCY,
    CUDA_GPUS,
    NODE_NAME,
    PIPELINE_BATCH_TIMEOUT_SECONDS,
    RESOURCE_MONITOR_INTERVAL_SECONDS,
)


@click.command("execute")
@click.option(
    "--pipeline-run-id",
    type=int,
    help="Identifier of the pipeline run to launch.",
)
@click.option(
    "--force-pipeline-batch-id",
    type=int,
    required=False,
    default=None,
    help="If specificed, only focuses on the specified batch.",
)
@click.option(
    "--ignore-locks",
    is_flag=True,
    help="If set, will ignore batch locks (e.g, batch running on another machine).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="If set, runs the pipeline without updating batch status in the database.",
)
def execute(
    pipeline_run_id: int,
    force_pipeline_batch_id: int | None,
    ignore_locks: bool,
    dry_run: bool,
):
    """
    Executes a pipeline run.
    Runs all steps sequentially within each batch, advancing to the next batch on completion.
    Uses batch locking for safe multi-node execution and monitors RAM/VRAM/CPU usage in a background thread.
    """
    pipeline_run: PipelineRun = None
    pipeline_run_start = datetime.now(timezone.utc)
    has_crashed = False

    batches_processed = 0
    batches_skipped = 0
    batches_crashed = 0

    #
    # Check: Pipeline run status
    #
    try:
        pipeline_run = PipelineRun.get(id=pipeline_run_id)
    except:
        logger.error(f"Pipeline run {pipeline_run_id} does not exist.")
        click.get_current_context().exit(1)

    #
    # Check: CUDA-capable GPU availability
    #
    if not CUDA_GPUS:
        logger.debug(f"Available torch devices: {",".join(CUDA_GPUS())}")
        logger.error("The pipeline needs access to at least one CUDA-compatible GPU.")
        click.get_current_context().exit(1)

    #
    # Start of run
    #
    if dry_run:
        logger.warning("Dry run mode: batch status will NOT be persisted to the database.")

    log_progress(pipeline_run=pipeline_run)

    #
    # BATCH-LEVEL: steps that can be run on a subset of records
    # NOTE: We use this cache mechanism to run all steps on subset of pre-cached images
    #
    monitor = ResourceMonitor(interval=RESOURCE_MONITOR_INTERVAL_SECONDS)
    monitor.start()

    try:
        for pipeline_batch in pipeline_run.batches:
            pipeline_batch_id: int = pipeline_batch.id
            pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)  # Force refresh

            # Check if batch has not been excluded
            if force_pipeline_batch_id and pipeline_batch_id != force_pipeline_batch_id:
                logger.warning(
                    f"RUN #{pipeline_run_id} BATCH#{pipeline_batch_id} was excluded (--force-pipeline-batch-id)"
                )
                continue

            # Check whether batch is locked
            if (
                not dry_run
                and not ignore_locks
                and not check_pipeline_batch_availability(pipeline_batch)
            ):
                logger.warning(
                    f"RUN #{pipeline_run_id} BATCH#{pipeline_batch_id} is locked or complete. Skipping."
                )
                batches_skipped += 1
                continue

            if ignore_locks:
                logger.warning("Potential pipeline batch-level lock was ignored (--ignore-locks)")

            monitor.set_batch(pipeline_batch_id)

            # Run steps
            try:
                # Lock pipeline batch
                pipeline_batch.node_name = NODE_NAME
                pipeline_batch.started_date = datetime.now(timezone.utc)
                pipeline_batch.ended_date = None
                pipeline_batch.has_crashed = False

                if not dry_run:
                    pipeline_batch.save()

                log_progress(
                    step_name="",
                    pipeline_run=pipeline_run,
                    pipeline_batch=pipeline_batch,
                )

                has_crashed = False

                # Step 1: pull and process scans for issues in batch
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step01_cache,
                        step_fn_kwargs={"pipeline_batch_id": pipeline_batch_id},
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step01_cache took {time.perf_counter() - start:.2f}s")

                # Step 2: segmentation
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step02_crop_detection,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step02_crop_detection took {time.perf_counter() - start:.2f}s")

                # Step 3: Tesseract OCR
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step03_crop_ocr_tesseract,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step03_crop_ocr_tesseract took {time.perf_counter() - start:.2f}s")

                # Step 4: VLM OCR (dots.ocr)
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step04_crop_ocr_vlm,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step04_crop_ocr_vlm took {time.perf_counter() - start:.2f}s")

                # Step 5: Text-based crop classification
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step05_crop_classification_text,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(
                    f"step05_crop_classification_text took" f" {time.perf_counter() - start:.2f}s"
                )

                # Step 6: Image-based crop classification
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step06_crop_classification_image,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(
                    f"step06_crop_classification_image took" f" {time.perf_counter() - start:.2f}s"
                )

                # Step 7: Final crop classification
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step07_crop_classification_final,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(
                    f"step07_crop_classification_final took" f" {time.perf_counter() - start:.2f}s"
                )

                # Step 8: NER
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step08_crop_ner,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step08_crop_ner took {time.perf_counter() - start:.2f}s")

                # Step 9: Subject detection
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step09_crop_subject,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step09_crop_subject took {time.perf_counter() - start:.2f}s")

                # Step 10: Reading order
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step10_crop_reading_order,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step10_crop_reading_order took {time.perf_counter() - start:.2f}s")

                # Step 11: Token counting
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step11_crop_token_count,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step11_crop_token_count took {time.perf_counter() - start:.2f}s")

                # Step 12: Language detection
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step12_crop_language,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step12_crop_language took {time.perf_counter() - start:.2f}s")

                # Step 13: Text analysis
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step13_crop_text_analysis,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step13_crop_text_analysis took {time.perf_counter() - start:.2f}s")

                # Step 14: ChronAm thesauri matching
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step14_crop_chronam_thesauri_match,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(
                    f"step14_crop_chronam_thesauri_match took"
                    f" {time.perf_counter() - start:.2f}s"
                )

                # Step 15: Text embedding (CPU) + Image embedding (GPU) -- concurrent
                start = time.perf_counter()

                if not has_crashed:
                    has_crashed = not execute_batch_level_step(
                        step_fn=commands.steps.step15_crop_embeddings,
                        step_fn_kwargs={
                            "pipeline_batch_id": pipeline_batch_id,
                            "overwrite": True,
                        },
                        pipeline_run=pipeline_run,
                        pipeline_batch=pipeline_batch,
                        monitor=monitor,
                    )

                logger.info(f"step15_crop_embeddings took {time.perf_counter() - start:.2f}s")

            #
            # Exceptions catch-all
            #
            except Exception as err:
                has_crashed = True
                logger.debug(traceback.format_exc())
            #
            # Always unlock pipeline batch
            #
            finally:
                pipeline_batch.has_crashed = has_crashed
                pipeline_batch.ended_date = datetime.now(timezone.utc)

                if not dry_run:
                    pipeline_batch.save()

                log_progress(
                    pipeline_run=pipeline_run,
                    pipeline_batch=pipeline_batch,
                    time=pipeline_batch.ended_date - pipeline_batch.started_date,
                )

                if not has_crashed:
                    batches_processed += 1
                    logger.info("Clearing cache ...")
                    utils.get_cache().clear() # only if step succeeded
                else:
                    batches_crashed += 1
                    logger.error("Pipeline has crashed!")
                    break
    finally:
        monitor.stop()

    #
    # End of run
    #
    log_progress(
        step_name="",
        pipeline_run=pipeline_run,
        time=(datetime.now(timezone.utc) - pipeline_run_start),
    )

    logger.info(f"Batches processed: {batches_processed}")
    logger.info(f"Batches crashed: {batches_crashed}")
    logger.info(f"Batches skipped: {batches_skipped}")


def check_pipeline_batch_availability(pipeline_batch: PipelineBatch) -> bool:
    """
    Assesses whether a given batch is available to be processed.

    Returns True (meaning: batch can be processed) if:
    - Batch was never processed
    - Batch has crashed
    - Batch has started but has timed out
    """
    # Batch has no date
    if not pipeline_batch.started_date and not pipeline_batch.ended_date:
        return True

    # Batch has crashed
    if pipeline_batch.has_crashed:
        return True

    # Batch is running and hasn't timed out yet
    if pipeline_batch.started_date and not pipeline_batch.ended_date:
        batch_lifetime_seconds = (datetime.now(timezone.utc) - pipeline_batch.started_date).seconds

        if batch_lifetime_seconds > PIPELINE_BATCH_TIMEOUT_SECONDS:
            return True

    # If we reached that point, the batch is not available
    return False


def execute_batch_level_step(
    step_fn,
    step_fn_kwargs: dict,
    pipeline_run: PipelineRun,
    pipeline_batch: PipelineBatch,
    monitor: object | None = None,
) -> bool:
    """Executes a batch-level step."""
    start = datetime.now(timezone.utc)
    end = None
    step_fn_name = f"{step_fn}"
    success = True
    ctx = click.get_current_context()

    # Try to pretty print function name, depending on its nature
    try:
        step_fn_name = step_fn.name.split(".")[-1]
    except:
        step_fn_name = step_fn.__name__.split(".")[-1]

    if monitor:
        monitor.set_step(step_fn_name)

    log_progress(f"{step_fn_name}", pipeline_run, pipeline_batch)

    # Run step
    try:
        ctx.invoke(step_fn, **step_fn_kwargs)
    except KeyboardInterrupt as err:
        logger.error(f"{step_fn_name} was interrupted.")
        success = False
    except click.exceptions.Exit as err:
        if err.exit_code != 0:
            logger.error(f"{step_fn_name} exited with code {err.exit_code}.")
            success = False

    # Stats
    end = datetime.now(timezone.utc)
    log_progress(f"{step_fn_name}", pipeline_run, pipeline_batch, time=(end - start))

    return success


def log_progress(
    step_name: str = "",
    pipeline_run: PipelineRun = None,
    pipeline_batch: PipelineBatch = None,
    time: timedelta = None,
):
    """
    Prepares and writes a log line for a specific step.
    If `time` is provided, generates an end-of-step log.
    """
    message = "START " if not time else "END "

    if step_name:
        message += f"{step_name} "

    if pipeline_run:
        message += f"RUN#{pipeline_run.id} "

    if pipeline_batch:
        message += f"BATCH#{pipeline_batch.id} "

    if pipeline_batch and pipeline_batch.node_name:
        message += f"NODE#{pipeline_batch.node_name} "

    if time:
        message += f"- {time} "

    logger.info(message)


class ResourceMonitor:
    """Logs RAM, VRAM, and CPU usage at regular intervals from a background thread."""

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._current_step: str = ""
        self._current_batch: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_step(self, step_name: str) -> None:
        self._current_step = step_name

    def set_batch(self, batch_id: int) -> None:
        self._current_batch = batch_id

    def start(self) -> None:
        # Prime cpu_percent so the first real sample returns a meaningful value
        psutil.cpu_percent(interval=None)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            self._log_sample()

    def _log_sample(self) -> None:
        ram_pct = psutil.virtual_memory().percent
        cpu_pct = psutil.cpu_percent(interval=None)
        vram_percents = _get_vram_percents()
        vram_str = ",".join(f"{k}:{v:.1f}" for k, v in sorted(vram_percents.items()))

        logger.info(
            f"RESOURCE_SAMPLE step={self._current_step} batch={self._current_batch} "
            f"ram_pct={ram_pct:.1f} cpu_pct={cpu_pct:.1f} vram_pcts={vram_str}"
        )


def _get_vram_percents() -> dict[str, float]:
    """Query NVML for VRAM usage of GPUs listed in CUDA_VISIBLE_DEVICES."""
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            return {}
        gpu_indices = {int(idx.strip()) for idx in visible.split(",")}

        import pynvml

        pynvml.nvmlInit()
        percents: dict[str, float] = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if mem.total == 0:
                continue
            percents[str(idx)] = mem.used / mem.total * 100
        return percents
    except Exception:
        return {}
