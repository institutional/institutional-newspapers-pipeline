import re
import traceback
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor

import click
from loguru import logger

import utils
from models import PipelineBatch, Issue, Scan, Crop, CropOCR
from models.crop_language import CropLanguage
from models.crop_text_analysis import CropTextAnalysis
from const import CPUS_LIMIT, TOKEN_COUNT_TIKTOKEN_ENCODING, DB_IN_CLAUSE_CHUNK_SIZE

TABLE_HTML_RE = re.compile(r"<table", re.IGNORECASE)

TABLE_MARKDOWN_RE = re.compile(r"\|[\s\-:]+\|")

MARKDOWN_RE = re.compile(
    r"(^#{1,6}\s)" r"|(\*\*[^*]+\*\*)" r"|(^[-*+]\s)" r"|(```)",
    re.MULTILINE,
)


class CropTextInput(NamedTuple):
    crop_id: int
    tesseract_raw: str | None
    vlm_raw: str | None
    language_hint: str


class CropTextResult(NamedTuple):
    crop_id: int
    tesseract_tokenizability_score: float | None
    tesseract_char_count: int | None
    tesseract_word_count: int | None
    tesseract_word_count_unique: int | None
    tesseract_word_type_token_ratio: float | None
    tesseract_sentence_count: int | None
    tesseract_sentence_count_unique: int | None
    vlm_tokenizability_score: float | None
    vlm_char_count: int | None
    vlm_word_count: int | None
    vlm_word_count_unique: int | None
    vlm_word_type_token_ratio: float | None
    vlm_sentence_count: int | None
    vlm_sentence_count_unique: int | None
    vlm_has_table: bool | None
    vlm_has_markdown: bool | None


@click.command("step13-crop-text-analysis")
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
def step13_crop_text_analysis(pipeline_batch_id: int, overwrite: bool = False):
    """
    Computes text analysis metrics (word/sentence counts, tokenizability, table/markdown detection) on CropOCR.tesseract_text and CropOCR.vlm_text for every crop in the batch.
    Uses ICU (via PyICU) for language-aware word and sentence splitting.

    Parallelizes across a ProcessPool on CPU. No GPU required.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items already processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_analysis = set(
            CropTextAnalysis.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(Scan.issue << issue_ids)
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_analysis
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    issue_ids = [item.issue_id for item in items_to_process]

    # Load all crops for batch
    all_crops = list(
        Crop.select(Crop, Scan, Issue).join(Scan).join(Issue).where(Scan.issue << issue_ids)
    )

    if not all_crops:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    try:
        crop_ids = [crop.id for crop in all_crops]

        # Load CropOCR data in chunks
        ocr_by_crop_id: dict[int, CropOCR] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for ocr in CropOCR.select().where(CropOCR.crop << chunk):
                ocr_by_crop_id[ocr.crop_id] = ocr

        # Load CropLanguage data in chunks — convert ISO 639-3 to ISO 639-1 for ICU
        language_hint_by_crop_id: dict[int, str] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for lang_row in CropLanguage.select().where(CropLanguage.crop << chunk):
                language_hint_by_crop_id[lang_row.crop_id] = utils.iso639_3_to_1(
                    lang_row.language_code
                )

        # In overwrite mode, find existing records to determine create vs update
        existing_crop_ids: set[int] = set()
        if overwrite:
            for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
                chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
                existing_crop_ids.update(
                    row[0]
                    for row in CropTextAnalysis.select(CropTextAnalysis.crop)
                    .where(CropTextAnalysis.crop << chunk)
                    .tuples()
                )

        # Build worker inputs: pass raw text, flattening happens in worker processes
        inputs: list[CropTextInput] = []
        for crop_id in crop_ids:
            ocr = ocr_by_crop_id.get(crop_id)
            vlm_raw = ocr.vlm_text if ocr and ocr.vlm_text else None
            tess_raw = ocr.tesseract_text if ocr and ocr.tesseract_text else None
            language_hint = language_hint_by_crop_id.get(crop_id, "en")

            inputs.append(CropTextInput(crop_id, tess_raw, vlm_raw, language_hint))

        # Split into chunks and process in parallel
        chunk_size = max(1, len(inputs) // CPUS_LIMIT)
        chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]

        all_results: list[CropTextResult] = []
        with ProcessPoolExecutor(max_workers=CPUS_LIMIT) as executor:
            for result_batch in executor.map(_analyze_chunk, chunks):
                all_results.extend(result_batch)

        # Build database records
        entries_to_create: list[CropTextAnalysis] = []
        entries_to_update: list[CropTextAnalysis] = []

        for result in all_results:
            record = CropTextAnalysis(
                crop=result.crop_id,
                tesseract_tokenizability_score=result.tesseract_tokenizability_score,
                tesseract_char_count=result.tesseract_char_count,
                tesseract_word_count=result.tesseract_word_count,
                tesseract_word_count_unique=result.tesseract_word_count_unique,
                tesseract_word_type_token_ratio=result.tesseract_word_type_token_ratio,
                tesseract_sentence_count=result.tesseract_sentence_count,
                tesseract_sentence_count_unique=result.tesseract_sentence_count_unique,
                vlm_tokenizability_score=result.vlm_tokenizability_score,
                vlm_char_count=result.vlm_char_count,
                vlm_word_count=result.vlm_word_count,
                vlm_word_count_unique=result.vlm_word_count_unique,
                vlm_word_type_token_ratio=result.vlm_word_type_token_ratio,
                vlm_sentence_count=result.vlm_sentence_count,
                vlm_sentence_count_unique=result.vlm_sentence_count_unique,
                vlm_has_table=result.vlm_has_table,
                vlm_has_markdown=result.vlm_has_markdown,
            )

            if result.crop_id in existing_crop_ids:
                entries_to_update.append(record)
            else:
                entries_to_create.append(record)

        logger.info(
            f"{len(crop_ids)} crops processed for text analysis. "
            f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
        )

        utils.process_db_write_batch(
            model=CropTextAnalysis,
            entries_to_create=entries_to_create,
            entries_to_update=entries_to_update,
            fields_to_update=[
                CropTextAnalysis.tesseract_tokenizability_score,
                CropTextAnalysis.tesseract_char_count,
                CropTextAnalysis.tesseract_word_count,
                CropTextAnalysis.tesseract_word_count_unique,
                CropTextAnalysis.tesseract_word_type_token_ratio,
                CropTextAnalysis.tesseract_sentence_count,
                CropTextAnalysis.tesseract_sentence_count_unique,
                CropTextAnalysis.vlm_tokenizability_score,
                CropTextAnalysis.vlm_char_count,
                CropTextAnalysis.vlm_word_count,
                CropTextAnalysis.vlm_word_count_unique,
                CropTextAnalysis.vlm_word_type_token_ratio,
                CropTextAnalysis.vlm_sentence_count,
                CropTextAnalysis.vlm_sentence_count_unique,
                CropTextAnalysis.vlm_has_table,
                CropTextAnalysis.vlm_has_markdown,
            ],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Text analysis processing failed. Exiting.")
        click.get_current_context().exit(1)


def _analyze_chunk(chunk: list[CropTextInput]) -> list[CropTextResult]:
    """Worker function: analyzes a chunk of crops in a subprocess."""
    import tiktoken

    encoder = tiktoken.get_encoding(TOKEN_COUNT_TIKTOKEN_ENCODING)
    results: list[CropTextResult] = []

    for crop_input in chunk:
        tess_flat = (
            utils.flatten_ocr_text(crop_input.tesseract_raw) if crop_input.tesseract_raw else None
        )
        vlm_flat = utils.flatten_ocr_text(crop_input.vlm_raw) if crop_input.vlm_raw else None

        tess_metrics = _analyze_single_text(
            flat=tess_flat,
            language_hint=crop_input.language_hint,
            encoder=encoder,
        )
        vlm_metrics = _analyze_single_text(
            flat=vlm_flat,
            language_hint=crop_input.language_hint,
            encoder=encoder,
        )

        vlm_has_table = False
        vlm_has_markdown = False
        if crop_input.vlm_raw:
            vlm_has_table = bool(
                TABLE_HTML_RE.search(crop_input.vlm_raw)
                or TABLE_MARKDOWN_RE.search(crop_input.vlm_raw)
            )
            vlm_has_markdown = bool(MARKDOWN_RE.search(crop_input.vlm_raw))

        results.append(
            CropTextResult(
                crop_input.crop_id,
                *tess_metrics,
                *vlm_metrics,
                vlm_has_table,
                vlm_has_markdown,
            )
        )

    return results


def _analyze_single_text(
    flat: str | None,
    language_hint: str,
    encoder,
) -> tuple:
    """
    Computes 7 metrics for a single text source.
    Returns a tuple of (tokenizability_score, char_count, word_count, word_count_unique,
    word_type_token_ratio, sentence_count, sentence_count_unique).
    """
    if not flat or not flat.strip():
        return (None, None, None, None, None, None, None)

    clean_text = flat.replace("\u200b", "")

    # ICU-based language-aware word and sentence splitting
    words = utils.icu_word_tokenize(clean_text, language_hint)
    sentences = utils.icu_sentence_tokenize(clean_text, language_hint)

    words_lower = [word.lower() for word in words]
    sentences_lower = [sentence.lower().strip() for sentence in sentences]

    char_count = len(flat)
    word_count = len(words)
    word_count_unique = len(set(words_lower))
    word_type_token_ratio = (
        round(word_count_unique / word_count * 100, 2) if word_count > 0 else None
    )
    sentence_count = len(sentences)
    sentence_count_unique = len(set(sentences_lower))

    # Tokenizability: how efficiently words tokenize (1 word ≈ 1.25 tokens is ideal)
    tokenizability_score = None
    if words:
        total_word_tokens = sum(len(t) for t in encoder.encode_batch(words, num_threads=1))
        if total_word_tokens > 0:
            tokenizability_score = min(round(len(words) * 1.25 / total_word_tokens * 100, 2), 100.0)

    return (
        tokenizability_score,
        char_count,
        word_count,
        word_count_unique,
        word_type_token_ratio,
        sentence_count,
        sentence_count_unique,
    )
