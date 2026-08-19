import os

import click
from loguru import logger

from const import (
    CUDA_GPUS,
    OCR_VLM_MODEL,
    OCR_VLM_MODEL_CONTEXT,
    OCR_VLM_MAX_MODEL_LEN,
    OCR_VLM_GPU_MEMORY_UTILIZATION,
    OCR_VLM_CHUNKED_PREFILL,
    OCR_VLM_PREFIX_CACHING,
    OCR_VLM_COMPILE_MM_ENCODER,
    OCR_VLM_FP8_KV_CACHE,
)


@click.command("warmup-ocr-vlm")
def warmup_ocr_vlm():
    """
    Pre-downloads and compiles the OCR VLM model so step04 starts without warmup delay.
    Runs a dummy inference on the first available GPU to trigger torch.compile caching.
    """
    if not CUDA_GPUS:
        logger.error("No CUDA devices available.")
        click.get_current_context().exit(1)

    from huggingface_hub import snapshot_download

    logger.info(f"Downloading {OCR_VLM_MODEL} (if not already cached) ...")
    snapshot_download(OCR_VLM_MODEL)

    device = CUDA_GPUS[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig

    logger.info(f"Loading model on {device} with compilation enabled ...")
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
        max_tokens=int(OCR_VLM_MODEL_CONTEXT * 1.5),
    )

    # Run a dummy inference to trigger torch.compile tracing
    logger.info("Running dummy inference to warm up compiled graphs ...")
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": dummy_image},
                    {"type": "text", "text": "Extract the text content from this image."},
                ],
            }
        ]
    ]
    llm.chat(conversations, sampling_params, use_tqdm=False)

    logger.info("Warmup complete. Compiled graphs are cached for future runs.")
