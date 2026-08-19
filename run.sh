#!/usr/bin/env bash
set -euo pipefail

source .env

DRY_RUN=""
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN="--dry-run"
    fi
done

PIPELINE_RUN_ID="$1"
DATETIME=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${DATA_DIR_PATH}/logs"
LOG_FILE="${LOG_DIR}/pipeline-run-${PIPELINE_RUN_ID}-${DATETIME}.log"

mkdir -p "$LOG_DIR"

# Model caching
#uv run python pipeline.py system cache-models;

# VLM warmup
#uv run pipeline.py system warmup-ocr-vlm;

{
    echo "=== Environment configuration ==="
    echo "NODE_NAME=${NODE_NAME:-}"
    echo "CACHE_MAX_SIZE_IN_GB=${CACHE_MAX_SIZE_IN_GB:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "CPUS_LIMIT=${CPUS_LIMIT:-}"
    echo "MAX_S3_CONCURRENCY=${MAX_S3_CONCURRENCY:-}"
    echo "PIPELINE_BATCH_TIMEOUT_SECONDS=${PIPELINE_BATCH_TIMEOUT_SECONDS:-}"
    echo "CROP_DETECTION_BATCH_SIZE=${CROP_DETECTION_BATCH_SIZE:-}"
    echo "CROP_CLASSIFICATION_IMAGE_BATCH_SIZE=${CROP_CLASSIFICATION_IMAGE_BATCH_SIZE:-}"
    echo "OCR_VLM_BATCH_SIZE=${OCR_VLM_BATCH_SIZE:-}"
    echo "NER_FLAIR_BATCH_SIZE=${NER_FLAIR_BATCH_SIZE:-}"
    echo "SUBJECT_ZEROSHOT_BATCH_SIZE=${SUBJECT_ZEROSHOT_BATCH_SIZE:-}"
    echo "IMAGE_EMBEDDING_BATCH_SIZE=${IMAGE_EMBEDDING_BATCH_SIZE:-}"
    echo "================================="
    echo ""

    uv run pipeline.py --verbose orchestration execute --pipeline-run-id="$PIPELINE_RUN_ID" $DRY_RUN
} 2>&1 | tee "$LOG_FILE"
