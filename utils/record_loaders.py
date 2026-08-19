import os
import io
import time
import shutil
import tarfile
import tempfile
import functools
import traceback
from pathlib import Path
from typing import NamedTuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from PIL import Image
from loguru import logger

import utils
from models import (
    Issue,
    Scan,
    Crop,
    CropOCR,
    CropClassification,
    CropSubject,
    CropNER,
    CropLanguage,
    CropTokenCount,
    CropTextAnalysis,
    CropChronamThesauriMatch,
    CropTextStaticEmbedding,
    CropImageEmbedding,
)
from const import (
    DB_IN_CLAUSE_CHUNK_SIZE,
    MAX_S3_CONCURRENCY,
    CPUS_LIMIT,
    EXPORT_IMAGE_TASK_SIZE,
    EXPORT_S3_MAX_POOL_CONNECTIONS,
    SCAN_WEBP_METHOD,
    SCAN_WEBP_METHOD_MAX_FALLBACK,
)


def chunked_query(query, field, ids: list[int]) -> list:
    """Runs a query with a chunked WHERE...IN clause to avoid SQL parameter overflow."""
    results = []
    for i in range(0, len(ids), DB_IN_CLAUSE_CHUNK_SIZE):
        chunk = ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
        results.extend(list(query.where(field << chunk)))
    return results


# Crop analysis models keyed by a short name, used to bulk-load records per crop.
ANALYSIS_MODELS: dict[str, type] = {
    "ocr": CropOCR,
    "classification": CropClassification,
    "subject": CropSubject,
    "ner": CropNER,
    "language": CropLanguage,
    "token_count": CropTokenCount,
    "text_analysis": CropTextAnalysis,
    "chronam_thesauri_match": CropChronamThesauriMatch,
    "text_static_embedding": CropTextStaticEmbedding,
    "image_embedding": CropImageEmbedding,
}


def load_analysis_data(
    crop_ids: list[int],
    models: dict[str, type] = ANALYSIS_MODELS,
) -> dict[str, dict[int, object]]:
    """Bulk-loads crop analysis records into lookup dicts keyed by crop ID, per model."""
    analysis: dict[str, dict[int, object]] = {}
    for key, model in models.items():
        records = chunked_query(model.select(), model.crop, crop_ids)
        analysis[key] = {record.crop_id: record for record in records}
    return analysis


class _ScanRawBytes(NamedTuple):
    scan_id: int
    raw_bytes_path: str


@functools.lru_cache(maxsize=None)
def _download_client(corpus: str):
    """Returns a per-corpus S3 client reused across issue downloads. boto3 clients are thread-safe,
    so caching one avoids rebuilding a client (and its connection pool) for every issue.
    """
    return utils.get_s3_client(corpus, max_pool_connections=EXPORT_S3_MAX_POOL_CONNECTIONS)


def fetch_and_process_scans(
    issues: list[Issue],
    scans_by_issue: dict[int, list[Scan]],
    image_format: str,
    quality: int,
    max_workers: int = CPUS_LIMIT,
    image_pool: ProcessPoolExecutor | None = None,
) -> dict[int, bytes]:
    """Downloads each issue's archive exactly once, runs CLAHE+autocontrast on every page in a
    ProcessPool, and encodes to the given format. Returns processed image bytes keyed by scan ID.

    Splits I/O (downloads on threads) from CPU (image processing in processes) so neither blocks
    the other. Scans whose page could not be downloaded or decoded are omitted from the result.

    When `image_pool` is given it is reused (and its lifecycle is owned by the caller), so worker
    processes are spawned once and shared across many calls; several callers may submit to it
    concurrently. Otherwise a private pool of `max_workers` is created for this call alone.
    """
    target_scans = [scan for issue in issues for scan in scans_by_issue.get(issue.id, [])]
    if not target_scans:
        return {}

    temp_dir = tempfile.mkdtemp(dir=_scan_temp_root())

    try:
        # Phase 1: download archives and extract raw page bytes to disk (I/O-bound)
        extracted: list[_ScanRawBytes] = []
        with ThreadPoolExecutor(max_workers=MAX_S3_CONCURRENCY) as executor:
            futures = {
                executor.submit(
                    _download_and_extract_issue,
                    issue,
                    scans_by_issue.get(issue.id, []),
                    temp_dir,
                ): issue
                for issue in issues
            }
            for future in as_completed(futures):
                issue = futures[future]
                try:
                    extracted.extend(future.result())
                except Exception:
                    logger.debug(traceback.format_exc())
                    logger.error(
                        f"Could not download {issue.archive_filename} ({issue.corpus}). Skipping."
                    )

        if not extracted:
            return {}

        # Phase 2: process and encode all pages (CPU-bound, true parallelism via ProcessPool)
        # With a shared pool, fixed-size tasks let concurrent callers' work interleave across cores;
        # with a private pool, split evenly so each of its workers gets one task.
        if image_pool is not None:
            chunk_size = EXPORT_IMAGE_TASK_SIZE
        else:
            chunk_size = max(1, len(extracted) // max_workers)
        path_chunks = [
            [item.raw_bytes_path for item in extracted[i : i + chunk_size]]
            for i in range(0, len(extracted), chunk_size)
        ]
        chunk_args = [(paths, image_format, quality) for paths in path_chunks]

        all_results: list[bytes | None] = []
        if image_pool is not None:
            for result_batch in image_pool.map(_process_path_chunk, chunk_args):
                all_results.extend(result_batch)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for result_batch in executor.map(_process_path_chunk, chunk_args):
                    all_results.extend(result_batch)

        return {
            item.scan_id: encoded
            for item, encoded in zip(extracted, all_results)
            if encoded is not None
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _scan_temp_root() -> str:
    """Returns a temp root for extracted scan bytes, creating it as needed."""
    from const import SCAN_TEMP_DIR_PATH

    SCAN_TEMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    return str(SCAN_TEMP_DIR_PATH)


def _download_and_extract_issue(
    issue: Issue,
    scans: list[Scan],
    temp_dir: str,
) -> list[_ScanRawBytes]:
    """Downloads an issue's archive from S3 and extracts the raw bytes of its target scans."""
    Image.MAX_IMAGE_PIXELS = None

    scan_by_filename = {scan.scan_filename: scan for scan in scans}
    if not scan_by_filename:
        return []

    s3 = _download_client(issue.corpus)
    bucket_name = os.getenv(f"{issue.corpus}_S3_BUCKET_NAME")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = s3.get_object(Bucket=bucket_name, Key=issue.archive_filename)
            archive_bytes = response["Body"].read()
            break
        except Exception:
            if attempt == max_retries:
                raise
            logger.warning(
                f"{issue.archive_filename}: download attempt {attempt}/{max_retries} failed."
                " Retrying..."
            )
            time.sleep(attempt)

    extracted: list[_ScanRawBytes] = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not member.isfile():
                continue

            scan = scan_by_filename.get(str(Path(member.name)))
            if scan is None:
                continue

            with tar.extractfile(member) as file_handle:
                raw_bytes = file_handle.read()

            fd, temp_path = tempfile.mkstemp(dir=temp_dir)
            try:
                os.write(fd, raw_bytes)
            finally:
                os.close(fd)

            extracted.append(_ScanRawBytes(scan_id=scan.id, raw_bytes_path=temp_path))

    return extracted


def _process_path_chunk(args: tuple[list[str], str, int]) -> list[bytes | None]:
    """ProcessPool worker: processes a chunk of raw scan files and encodes them.
    Returns None for any scan that could not be decoded.
    """
    # Pin OpenCV to one thread per worker so the ProcessPool stays within CPUS_LIMIT cores
    # rather than each of CPUS_LIMIT workers fanning out across every core.
    import cv2

    cv2.setNumThreads(1)

    paths, image_format, quality = args

    results: list[bytes | None] = []
    for path in paths:
        try:
            with open(path, "rb") as file_handle:
                raw_bytes = file_handle.read()
            pil_image = utils.process_scan(raw_bytes)
            results.append(_encode_image(pil_image, image_format, quality, path))
        except Exception:
            logger.debug(traceback.format_exc())
            logger.warning(f"Could not process scan at {path}. Skipping.")
            results.append(None)
    return results


def _encode_image(pil_image, image_format: str, quality: int, path: str) -> bytes:
    """Encodes an image, retrying WEBP at higher encoder effort if the low-effort attempt fails.

    Low `method` values can trip libwebp's 512KB header-partition limit ("encoding error 6") on the
    largest, busiest scans, which the faster settings are otherwise worth using for. Retrying instead
    of giving up keeps that speed without silently dropping pages from the export.
    """
    if image_format != "WEBP":
        buffer = io.BytesIO()
        pil_image.save(buffer, format=image_format, quality=quality)
        return buffer.getvalue()

    for method in range(SCAN_WEBP_METHOD, SCAN_WEBP_METHOD_MAX_FALLBACK + 1):
        buffer = io.BytesIO()
        try:
            pil_image.save(buffer, format="WEBP", quality=quality, method=method)
            return buffer.getvalue()
        except ValueError:
            if method == SCAN_WEBP_METHOD_MAX_FALLBACK:
                raise
            logger.debug(traceback.format_exc())
            logger.warning(
                f"WEBP encode failed at method={method} for {path}"
                f" ({pil_image.width}x{pil_image.height}). Retrying at method={method + 1}."
            )

    raise AssertionError("unreachable: the final fallback attempt either returns or raises")
