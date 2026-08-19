from dotenv import load_dotenv

# Fallback: ensures env vars are available if const is imported outside pipeline.py
load_dotenv()

import os
from pathlib import Path
from datetime import datetime, timezone
import multiprocessing

from slugify import slugify

import utils

#
# Required env vars
#
REQUIRED_ENV_VARS = [
    "DATA_DIR_PATH",
    "CACHE_MAX_SIZE_IN_GB",
    "CORPORA",
    "PIPELINE_BATCH_TIMEOUT_SECONDS",
]
""" Lists project-wide required environment variables. """

#
# Node name (machine name)
#
NODE_NAME = os.environ.get("NODE_NAME", "main")

#
# Data directory
#
DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH", "data/")
CACHE_DIR_PATH = Path(DATA_DIR_PATH, "cache")
LOGS_DIR_PATH = Path(DATA_DIR_PATH, "logs")
DASHBOARD_DIR_PATH = Path(DATA_DIR_PATH, "dashboard")
SCAN_TEMP_DIR_PATH = Path(DATA_DIR_PATH, "scan_temp")
PEEK_DIR_PATH = Path(DATA_DIR_PATH, "peek")
DATABASE_DIR_PATH = Path(DATA_DIR_PATH, "database")
DATABASE_FILEPATH = Path(DATABASE_DIR_PATH, "database.db")
EXPORT_DIR_PATH = Path(DATA_DIR_PATH, "export")

#
# Templates directory
#
TEMPLATES_DIR_PATH = Path(Path(__file__).parent.parent, "templates")

#
# Corpora
#
CORPORA = os.environ.get("CORPORA", "").split(",")
""" List of all upstream newspaper corpora """


#
# Compute constraints
#
CPUS_LIMIT = int(os.getenv("CPUS_LIMIT", multiprocessing.cpu_count()))
CUDA_GPUS = [device for device in utils.get_compute_devices() if device.startswith("cuda:")]
MAX_S3_CONCURRENCY = int(os.getenv("MAX_S3_CONCURRENCY", 32))
PIPELINE_BATCH_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_BATCH_TIMEOUT_SECONDS"))
RESOURCE_MONITOR_INTERVAL_SECONDS = int(os.getenv("RESOURCE_MONITOR_INTERVAL_SECONDS", 10))


#
# Scan processing constants
#
SCAN_JPEG_QUALITY = 90
SCAN_WEBP_QUALITY = 70
SCAN_WEBP_METHOD = 2
"""
    libwebp encoder effort (0-6). Pillow defaults to 4, which spends ~14s of extra search per
    ~70 MP scan for a few percent of size: encoding dominated export CPU at that setting.
    Methods 0 and 1 are faster still but overflow libwebp's header partition on most scans above
    ~130 MP, so 2 is the lowest setting that encodes the whole corpus.
"""

SCAN_WEBP_METHOD_MAX_FALLBACK = 4
""" Highest effort to retry at when a WEBP encode fails; see record_loaders._encode_image. """
SCAN_CLAHE_CLIP_LIMIT = 1.5
SCAN_CLAHE_TILE_GRID_SIZE = (8, 8)
SCAN_AUTOCONTRAST_CUTOFF = 1.0

STEP01_CACHE_MAX_CONSECUTIVE_FAILURES = int(os.getenv("STEP01_CACHE_MAX_CONSECUTIVE_FAILURES", 5))
STEP01_CACHE_FAILURE_PAUSE_MINUTES = int(os.getenv("STEP01_CACHE_FAILURE_PAUSE_MINUTES", 30))


#
# OCR
#
OCR_VLM_MODEL = "rednote-hilab/dots.mocr"
OCR_VLM_MODEL_CONTEXT = 8192
OCR_VLM_MAX_MODEL_LEN = int(os.getenv("OCR_VLM_MAX_MODEL_LEN", OCR_VLM_MODEL_CONTEXT))
OCR_VLM_GPU_MEMORY_UTILIZATION = float(os.getenv("OCR_VLM_GPU_MEMORY_UTILIZATION", 0.95))
OCR_VLM_BATCH_SIZE = int(os.getenv("OCR_VLM_BATCH_SIZE", 4096))
OCR_VLM_PREP_WORKERS = int(os.getenv("OCR_VLM_PREP_WORKERS", 0))
OCR_VLM_MAX_PIXELS = int(os.getenv("OCR_VLM_MAX_PIXELS", 1_000_000))
OCR_VLM_MIN_PIXELS = int(os.getenv("OCR_VLM_MIN_PIXELS", 250_000))
OCR_VLM_CHUNKED_PREFILL = os.getenv("OCR_VLM_CHUNKED_PREFILL", "1") == "1"
OCR_VLM_PREFIX_CACHING = os.getenv("OCR_VLM_PREFIX_CACHING", "1") == "1"
OCR_VLM_COMPILE_MM_ENCODER = os.getenv("OCR_VLM_COMPILE_MM_ENCODER", "1") == "1"
OCR_VLM_FP8_KV_CACHE = os.getenv("OCR_VLM_FP8_KV_CACHE", "1") == "0"
OCR_VLM_PROMPT = os.getenv("OCR_VLM_PROMPT", "Extract the text content from this image.")
OCR_VLM_SMART_RESIZE_FACTOR = int(os.getenv("OCR_VLM_SMART_RESIZE_FACTOR", 28))
OCR_VLM_MAX_ASPECT_RATIO = 200

OCR_TESSERACT_MAX_PIXELS = int(os.getenv("OCR_TESSERACT_MAX_PIXELS", 1_500_000))
OCR_TESSDATA_DIR_PATH = os.getenv("OCR_TESSDATA_DIR_PATH", "/usr/local/share/tessdata")

#
# Crop detection (layout segmentation)
#
CROP_DETECTION_MODEL = "institutional/institutional-newspapers-segmenter-yolo26x"
CROP_DETECTION_IMGSZ = 960
CROP_DETECTION_CONF = 0.6
CROP_DETECTION_IOU = 0.15
CROP_DETECTION_MAX_DET = 500
CROP_DETECTION_BATCH_SIZE = int(os.getenv("CROP_DETECTION_BATCH_SIZE", 32))

#
# Crop classification
#
CROP_CLASSIFICATION_IMAGE_MODEL = (
    "institutional/institutional-newspapers-crop-classifier-image-yolo26m-cls"
)
CROP_CLASSIFICATION_IMAGE_IMGSZ = 768
CROP_CLASSIFICATION_IMAGE_CONF = 0.5
CROP_CLASSIFICATION_IMAGE_BATCH_SIZE = int(os.getenv("CROP_CLASSIFICATION_IMAGE_BATCH_SIZE", 128))
CROP_CLASSIFICATION_IMAGE_PREP_WORKERS = int(os.getenv("CROP_CLASSIFICATION_IMAGE_PREP_WORKERS", 0))

CROP_CLASSIFICATION_TEXT_MODEL = (
    "institutional/institutional-newspapers-crop-classifier-text-model2vec"
)
CROP_CLASSIFICATION_TEXT_CONF = 0.4

# Image model's "Photograph or illustration" prediction overrides text when above this threshold
CROP_CLASSIFICATION_FINAL_IMAGE_PHOTO_CONF = 0.5

# Image model's "Empty" prediction overrides text when above this threshold
CROP_CLASSIFICATION_FINAL_IMAGE_EMPTY_CONF = 0.4

#
# NER
#
NER_FLAIR_MODEL = "ner-fast"
NER_FLAIR_CONF = 0.85
NER_FLAIR_CLASSES = ("PER", "LOC", "ORG")
NER_FLAIR_MAX_SENTENCE_WORDS = int(os.getenv("NER_FLAIR_MAX_SENTENCE_WORDS", 400))
NER_FLAIR_MAX_SENTENCE_CHARS = int(os.getenv("NER_FLAIR_MAX_SENTENCE_CHARS", 3500))
NER_FLAIR_BATCH_SIZE = int(os.getenv("NER_FLAIR_BATCH_SIZE", 256))

#
# Subject detection
#
SUBJECT_ZEROSHOT_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
SUBJECT_MODEL_CONF = 0.1
SUBJECT_ZEROSHOT_BATCH_SIZE = int(os.getenv("SUBJECT_ZEROSHOT_BATCH_SIZE", 16))

SUBJECT_CLASSES = (
    (
        "Politics, International Relations & Government",
        "Business, Finance & Market Reports",
        "Crime, Law Enforcement & Courts",
        "Shipping, Transport & Travel Schedules",
        "Sports, Racing & Games",
        "Arts, Literature & Entertainment",
        "Science, Technology, Health & Agriculture",
        "Social News, Religion, Weddings & Deaths",
        "Opinion, Editorials & Letters",
        "Legal Notices, Auctions & Public Announcements",
        "Commercial Advertisements & Classifieds",
        "Mastheads, Page Headers & Printer Imprints",
    ),
)


#
# Token counting
#
TOKEN_COUNT_TIKTOKEN_ENCODING = os.getenv("TOKEN_COUNT_TIKTOKEN_ENCODING", "o200k_base")

#
# ChronAm Thesauri
#
CHRONAM_THESAURI_DATASET = "institutional/chronicling-america-thesauri"

#
# Static text embedding
#
STATIC_TEXT_EMBEDDING_MODEL = "minishlab/potion-multilingual-128M"

#
# Image embedding
#
IMAGE_EMBEDDING_MODEL = "facebook/dinov2-small"
IMAGE_EMBEDDING_BATCH_SIZE = int(os.getenv("IMAGE_EMBEDDING_BATCH_SIZE", 128))

#
# Reading order
#
READING_ORDER_COLUMN_WIDTH_PERCENTILE = 25
READING_ORDER_WIDE_CROP_THRESHOLD_RATIO = 1.4
READING_ORDER_HDBSCAN_MIN_CLUSTER_SIZE = 2
READING_ORDER_HDBSCAN_MIN_SAMPLES = 1

#
# Language detection post-processing
#
LANGUAGE_MIN_CONFIDENCE_SCORE = 0.50
""" Minimum Lingua confidence (0.0-1.0); below this the issue-level language wins. """

LANGUAGE_MIN_VLM_WORD_COUNT = 30
""" Minimum VLM word count; below this the issue-level language wins. """

#
# Database query chunking
#
DB_IN_CLAUSE_CHUNK_SIZE = 900
""" Max IDs per WHERE ... IN clause to avoid SQL parameter overflow. """

#
# Dataset export / release
#
EXPORT_CHUNK_ROW_COUNT = int(os.getenv("EXPORT_CHUNK_ROW_COUNT", 250))
""" Number of scans (rows) per exported Parquet chunk. """

EXPORT_BUILD_WORKERS = int(os.getenv("EXPORT_BUILD_WORKERS", 4))
"""
    Number of Parquet chunks built concurrently.
    Each builder's internal image ProcessPool is sized to CPUS_LIMIT // this, so the total process count stays within CPUS_LIMIT.
"""

EXPORT_MAX_INFLIGHT_CHUNKS = int(os.getenv("EXPORT_MAX_INFLIGHT_CHUNKS", 12))
"""
    Max built-but-not-yet-uploaded Parquet chunks allowed on disk at once.
    Also bounds concurrent uploads, since a builder holds its slot until its upload finishes: keep it
    well above EXPORT_BUILD_WORKERS so uploads overlap builds instead of serialising behind them.
"""

EXPORT_PARQUET_ROW_GROUP_SIZE = int(os.getenv("EXPORT_PARQUET_ROW_GROUP_SIZE", 10))
""" 
    Rows per Parquet row group. 
    Kept small so each row group stays well under the Hugging Face viewer's per-row-group scan limit, since embedded scan images make rows multi-megabyte. 
"""

EXPORT_IMAGE_TASK_SIZE = int(os.getenv("EXPORT_IMAGE_TASK_SIZE", 2))
"""
    Pages per task submitted to the shared image-processing ProcessPool during export.
    Many small tasks let work from concurrent chunk builders interleave across all cores.
    Since results are consumed in order, large tasks also make a chunk wait on its slowest task.
"""

EXPORT_S3_MAX_POOL_CONNECTIONS = int(os.getenv("EXPORT_S3_MAX_POOL_CONNECTIONS", 256))
"""
    botocore connection-pool size for the export clients. Kept >= MAX_S3_CONCURRENCY so concurrent download/upload threads don't serialize on a small pool.
    The download client is cached and shared across all chunk builders, and each concurrent upload opens up to EXPORT_S3_TRANSFER_CONCURRENCY connections of its own.
"""

EXPORT_UPLOAD_MAX_RETRIES = int(os.getenv("EXPORT_UPLOAD_MAX_RETRIES", 5))
""" Attempts per chunk upload before giving up and leaving the local files for a later --resume. """

EXPORT_UPLOAD_RETRY_BACKOFF_SECONDS = int(os.getenv("EXPORT_UPLOAD_RETRY_BACKOFF_SECONDS", 5))
""" Base delay for upload retry backoff; doubles each attempt. """

EXPORT_S3_MULTIPART_THRESHOLD_MB = int(os.getenv("EXPORT_S3_MULTIPART_THRESHOLD_MB", 64))
""" Object size (MB) above which export uploads switch to multipart. """

EXPORT_S3_MULTIPART_CHUNKSIZE_MB = int(os.getenv("EXPORT_S3_MULTIPART_CHUNKSIZE_MB", 64))
""" Part size (MB) for multipart export uploads. """

EXPORT_S3_TRANSFER_CONCURRENCY = int(os.getenv("EXPORT_S3_TRANSFER_CONCURRENCY", 8))
""" Per-upload multipart concurrency for export uploads. """

EXPORT_CUTOFF_YEAR = 1931
""" Issues published before this year are considered public domain. """

RELEASE_S3_BUCKET_NAME = os.getenv("RELEASE_S3_BUCKET_NAME")

RELEASE_HF_DATASET = os.getenv("RELEASE_HF_DATASET")

CORPUS_METADATA_SOURCE = {
    "BPL": "chronicling_america",
}
""" Maps each corpus to the provenance of its issue-level metadata. """


#
# Misc
#
DATETIME_SLUG = datetime_slug = slugify(
    datetime.now(timezone.utc).isoformat(sep=" ", timespec="minutes")
)
