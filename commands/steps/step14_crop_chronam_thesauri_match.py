import re
import json
import traceback
from typing import NamedTuple
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import click
from huggingface_hub import hf_hub_download
from loguru import logger

import utils
from models import PipelineBatch, Issue, Scan, Crop, CropOCR
from models.crop_chronam_thesauri_match import CropChronamThesauriMatch
from const import CPUS_LIMIT, CHRONAM_THESAURI_DATASET, DB_IN_CLAUSE_CHUNK_SIZE

# Module-level globals set before forking workers.
_THESAURI_PATTERN: re.Pattern | None = None
_TERM_TO_CATEGORY: dict[str, str] = {}


class CropThesauriInput(NamedTuple):
    crop_id: int
    tesseract_raw: str | None
    vlm_raw: str | None


class CropThesauriResult(NamedTuple):
    crop_id: int
    tesseract_matches: dict | None
    vlm_matches: dict | None
    tesseract_match_count: int | None
    tesseract_term_count: int | None
    vlm_match_count: int | None
    vlm_term_count: int | None


@click.command("step14-crop-chronam-thesauri-match")
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
def step14_crop_chronam_thesauri_match(pipeline_batch_id: int, overwrite: bool = False):
    """
    Detects Chronicling America thesauri terms in flattened OCR text (Tesseract and VLM) for every crop in the batch. Only processes crops from English-language, USA-origin issues.

    Compiles a single regex with longest-first alternation for greedy matching, then parallelizes across a ProcessPool on CPU.
    """
    global _THESAURI_PATTERN, _TERM_TO_CATEGORY

    # Load thesaurus terms and compile regex
    thesauri_path = hf_hub_download(
        repo_id=CHRONAM_THESAURI_DATASET,
        filename="raw.jsonl",
        repo_type="dataset",
    )

    term_to_category: dict[str, str] = {}
    with open(thesauri_path) as f:
        for line in f:
            row = json.loads(line)
            category = row["category"]
            term_to_category[row["keyword"].casefold()] = category
            for rt in row["related_terms"]:
                term_to_category[rt.casefold()] = category

    # Sort longest-first so the regex alternation prefers longer matches
    sorted_terms = sorted(term_to_category.keys(), key=len, reverse=True)
    pattern_str = r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b"
    _THESAURI_PATTERN = re.compile(pattern_str, re.IGNORECASE)
    _TERM_TO_CATEGORY = term_to_category

    logger.info(f"Loaded {len(term_to_category)} thesauri terms from {CHRONAM_THESAURI_DATASET}")

    # Load batch
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    # Skip items already processed unless --overwrite
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_matches = set(
            CropChronamThesauriMatch.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(Scan.issue << issue_ids)
            .distinct()
            .tuples()
        )

        items_to_process = [
            item for item in all_items if (item.issue_id,) not in issues_with_matches
        ]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    try:
        issue_ids = [item.issue_id for item in items_to_process]

        # Load crops — only from English-language, USA-origin issues
        all_crops = list(
            Crop.select(Crop, Scan, Issue)
            .join(Scan)
            .join(Issue)
            .where(
                Scan.issue << issue_ids,
                Issue.language == "eng",
                Issue.country == "USA",
            )
        )

        if not all_crops:
            logger.error("No eligible crops found (English + USA filter applied).")
            click.get_current_context().exit(1)
            return

        crop_ids = [crop.id for crop in all_crops]

        # Load CropOCR data in chunks
        ocr_by_crop_id: dict[int, CropOCR] = {}
        for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
            chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
            for ocr in CropOCR.select().where(CropOCR.crop << chunk):
                ocr_by_crop_id[ocr.crop_id] = ocr

        # In overwrite mode, find existing records to determine create vs update
        existing_crop_ids: set[int] = set()
        if overwrite:
            for i in range(0, len(crop_ids), DB_IN_CLAUSE_CHUNK_SIZE):
                chunk = crop_ids[i : i + DB_IN_CLAUSE_CHUNK_SIZE]
                existing_crop_ids.update(
                    row[0]
                    for row in CropChronamThesauriMatch.select(CropChronamThesauriMatch.crop)
                    .where(CropChronamThesauriMatch.crop << chunk)
                    .tuples()
                )

        # Build worker inputs: pass raw text, flattening happens in worker processes
        inputs: list[CropThesauriInput] = []
        for crop_id in crop_ids:
            ocr = ocr_by_crop_id.get(crop_id)
            tess_raw = ocr.tesseract_text if ocr and ocr.tesseract_text else None
            vlm_raw = ocr.vlm_text if ocr and ocr.vlm_text else None
            inputs.append(CropThesauriInput(crop_id, tess_raw, vlm_raw))

        # Chunk and process in parallel
        chunk_size = max(1, len(inputs) // CPUS_LIMIT)
        chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]

        all_results: list[CropThesauriResult] = []
        with ProcessPoolExecutor(max_workers=CPUS_LIMIT) as executor:
            for result_batch in executor.map(_match_chunk, chunks):
                all_results.extend(result_batch)

        # Build database records
        entries_to_create: list[CropChronamThesauriMatch] = []
        entries_to_update: list[CropChronamThesauriMatch] = []

        for result in all_results:
            record = CropChronamThesauriMatch(
                crop=result.crop_id,
                tesseract_matches=result.tesseract_matches,
                tesseract_match_count=result.tesseract_match_count,
                tesseract_term_count=result.tesseract_term_count,
                vlm_matches=result.vlm_matches,
                vlm_match_count=result.vlm_match_count,
                vlm_term_count=result.vlm_term_count,
            )

            if result.crop_id in existing_crop_ids:
                entries_to_update.append(record)
            else:
                entries_to_create.append(record)

        logger.info(
            f"{len(crop_ids)} crops processed for thesauri matching. "
            f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
        )

        utils.process_db_write_batch(
            model=CropChronamThesauriMatch,
            entries_to_create=entries_to_create,
            entries_to_update=entries_to_update,
            fields_to_update=[
                CropChronamThesauriMatch.tesseract_matches,
                CropChronamThesauriMatch.tesseract_match_count,
                CropChronamThesauriMatch.tesseract_term_count,
                CropChronamThesauriMatch.vlm_matches,
                CropChronamThesauriMatch.vlm_match_count,
                CropChronamThesauriMatch.vlm_term_count,
            ],
        )

    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Thesauri matching failed. Exiting.")
        click.get_current_context().exit(1)


def _match_chunk(chunk: list[CropThesauriInput]) -> list[CropThesauriResult]:
    """Worker: matches thesauri terms in a chunk of crops."""
    results: list[CropThesauriResult] = []

    for inp in chunk:
        tess_flat = utils.flatten_ocr_text(inp.tesseract_raw) if inp.tesseract_raw else None
        vlm_flat = utils.flatten_ocr_text(inp.vlm_raw) if inp.vlm_raw else None

        tess_matches, tess_match_count, tess_term_count = _match_text(tess_flat)
        vlm_matches, vlm_match_count, vlm_term_count = _match_text(vlm_flat)

        results.append(
            CropThesauriResult(
                crop_id=inp.crop_id,
                tesseract_matches=tess_matches,
                vlm_matches=vlm_matches,
                tesseract_match_count=tess_match_count,
                tesseract_term_count=tess_term_count,
                vlm_match_count=vlm_match_count,
                vlm_term_count=vlm_term_count,
            )
        )

    return results


def _match_text(
    flat: str | None,
) -> tuple[dict[str, dict[str, int]] | None, int | None, int | None]:
    """
    Matches thesauri terms in a single piece of flattened text.
    Returns (matches_by_category, total_match_count, distinct_term_count).
    """
    if not flat or not flat.strip():
        return (None, None, None)

    found = _THESAURI_PATTERN.findall(flat)
    if not found:
        return ({}, 0, 0)

    counts = Counter(term.casefold() for term in found)

    # Group by category
    by_category: dict[str, dict[str, int]] = defaultdict(dict)
    for term, count in counts.items():
        category = _TERM_TO_CATEGORY[term]
        by_category[category][term] = count

    match_count = sum(counts.values())
    term_count = len(counts)

    return (dict(by_category), match_count, term_count)
