import click
from loguru import logger

from const import (
    OCR_VLM_MODEL,
    CROP_DETECTION_MODEL,
    CROP_CLASSIFICATION_IMAGE_MODEL,
    CROP_CLASSIFICATION_TEXT_MODEL,
    SUBJECT_ZEROSHOT_MODEL,
    TOKEN_COUNT_TIKTOKEN_ENCODING,
    CHRONAM_THESAURI_DATASET,
    STATIC_TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
)


@click.command("cache-models")
def cache_models() -> None:
    """
    Downloads all models and datasets into local cache so they are available offline during pipeline execution.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    import tiktoken

    logger.info(f"Caching {OCR_VLM_MODEL} ...")
    snapshot_download(OCR_VLM_MODEL)

    logger.info(f"Caching {CROP_DETECTION_MODEL} ...")
    snapshot_download(CROP_DETECTION_MODEL)

    logger.info(f"Caching {CROP_CLASSIFICATION_IMAGE_MODEL} ...")
    snapshot_download(CROP_CLASSIFICATION_IMAGE_MODEL)

    logger.info(f"Caching {CROP_CLASSIFICATION_TEXT_MODEL} ...")
    snapshot_download(CROP_CLASSIFICATION_TEXT_MODEL)

    logger.info(f"Caching {SUBJECT_ZEROSHOT_MODEL} ...")
    snapshot_download(SUBJECT_ZEROSHOT_MODEL)

    logger.info(f"Caching tiktoken encoding {TOKEN_COUNT_TIKTOKEN_ENCODING} ...")
    tiktoken.get_encoding(TOKEN_COUNT_TIKTOKEN_ENCODING)

    logger.info(f"Caching {CHRONAM_THESAURI_DATASET} dataset ...")
    hf_hub_download(
        repo_id=CHRONAM_THESAURI_DATASET,
        filename="raw.jsonl",
        repo_type="dataset",
    )

    logger.info(f"Caching {STATIC_TEXT_EMBEDDING_MODEL} ...")
    snapshot_download(STATIC_TEXT_EMBEDDING_MODEL)

    logger.info(f"Caching {IMAGE_EMBEDDING_MODEL} ...")
    snapshot_download(IMAGE_EMBEDDING_MODEL)

    logger.info("All models and datasets cached.")
