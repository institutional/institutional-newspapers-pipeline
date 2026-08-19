import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from loguru import logger

import utils
from models import PipelineBatch, PipelineBatchItem, Issue, Scan, Crop, CropOCR, CropNER
from models.crop_language import CropLanguage
from const import (
    CUDA_GPUS,
    NER_FLAIR_MODEL,
    NER_FLAIR_CONF,
    NER_FLAIR_CLASSES,
    NER_FLAIR_BATCH_SIZE,
    NER_FLAIR_MAX_SENTENCE_WORDS,
    NER_FLAIR_MAX_SENTENCE_CHARS,
)


@click.command("step08-crop-ner")
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
def step08_crop_ner(pipeline_batch_id: int, overwrite: bool = False):
    """
    Uses a NER model to extract named entities from VLM OCR text for each crop.
    Populates per/loc/org entities and confidence scores in CropNER records.
    Spins up 1 process per available CUDA GPU.

    Applies ICU sentence tokenization before inference and runs batch prediction. Deduplicates entities per crop using case-insensitive matching, keeping the highest-confidence occurrence.
    """
    pipeline_batch = PipelineBatch.get(id=pipeline_batch_id)
    all_items = pipeline_batch.items

    if not CUDA_GPUS:
        logger.error("No CUDA devices available.")
        click.get_current_context().exit(1)

    # Skip items that have already been processed unless --overwrite is set
    if overwrite:
        items_to_process = list(all_items)
    else:
        issue_ids = [item.issue_id for item in all_items]

        issues_with_ner = set(
            CropNER.select(Scan.issue)
            .join(Crop)
            .join(Scan)
            .where(
                Scan.issue << issue_ids,
                CropNER.per_entities.is_null(False),
            )
            .distinct()
            .tuples()
        )

        items_to_process = [item for item in all_items if (item.issue_id,) not in issues_with_ner]

        skipped = len(all_items) - len(items_to_process)
        if skipped:
            logger.info(f"{skipped} items already processed. Skipping (use --overwrite to redo).")

    if not items_to_process:
        logger.error("No items to process.")
        click.get_current_context().exit(1)
        return

    # Split items across CUDA GPUs, balanced by crop count per item
    num_gpus = len(CUDA_GPUS)
    item_ids = [item.id for item in items_to_process]
    crop_weights = utils.get_crop_counts_by_item(item_ids)
    chunks = utils.distribute_to_gpus(item_ids, crop_weights, num_gpus)

    with ProcessPoolExecutor(max_workers=num_gpus, initializer=utils.get_db) as executor:
        futures = {}

        for gpu_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            device = CUDA_GPUS[gpu_index]
            future = executor.submit(_process_batch, chunk, device)
            futures[future] = device

        for future in as_completed(futures):
            device = futures[future]

            try:
                check = future.result()
                assert check
            except Exception:
                logger.debug(traceback.format_exc())
                logger.error(f"NER failed on {device}. Exiting.")
                click.get_current_context().exit(1)


def _process_batch(item_ids: list[int], device: str) -> bool:
    """Runs NER for a subset of pipeline batch items on a single CUDA device."""
    import flair
    import torch
    from flair.data import Sentence
    from flair.nn import Classifier

    flair.device = torch.device(device)
    model = Classifier.load(NER_FLAIR_MODEL)

    all_sentences: list[Sentence] = []
    sentence_to_crop_id: list[int] = []

    # Pre-fetch all DB data in bulk (4 queries instead of 4 per item)
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

    existing_ner_crop_ids: set[int] = set()
    if all_crop_ids:
        existing_ner_crop_ids = {
            row[0]
            for row in CropNER.select(CropNER.crop).where(CropNER.crop << all_crop_ids).tuples()
        }

    ocr_by_crop_id: dict[int, CropOCR] = {}
    if all_crop_ids:
        for ocr in CropOCR.select().where(CropOCR.crop << all_crop_ids):
            ocr_by_crop_id[ocr.crop_id] = ocr

    language_hint_by_crop_id: dict[int, str] = {}
    if all_crop_ids:
        for lang_row in CropLanguage.select().where(CropLanguage.crop << all_crop_ids):
            language_hint_by_crop_id[lang_row.crop_id] = utils.iso639_3_to_1(lang_row.language_code)

    for item in items:
        for crop in crops_by_issue.get(item.issue_id, []):
            try:
                ocr = ocr_by_crop_id.get(crop.id)
                raw_text = ocr.vlm_text if ocr and ocr.vlm_text else ""
                flat_text = utils.flatten_ocr_text(raw_text) if raw_text else ""
            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"Could not prepare text for crop #{crop.id}. Skipping.")
                flat_text = ""

            if not flat_text.strip():
                continue

            language_hint = language_hint_by_crop_id.get(crop.id, "en")
            sentences = utils.icu_sentence_tokenize(flat_text, language_hint)

            for sent_text in sentences:
                if len(sent_text) > NER_FLAIR_MAX_SENTENCE_CHARS:
                    sent_text = sent_text[:NER_FLAIR_MAX_SENTENCE_CHARS]
                words = sent_text.split()
                if len(words) <= NER_FLAIR_MAX_SENTENCE_WORDS:
                    all_sentences.append(Sentence(sent_text))
                    sentence_to_crop_id.append(crop.id)
                else:
                    # Safety: sub-split overly long sentences (e.g. unpunctuated OCR)
                    for i in range(0, len(words), NER_FLAIR_MAX_SENTENCE_WORDS):
                        sub_chunk = " ".join(words[i : i + NER_FLAIR_MAX_SENTENCE_WORDS])
                        if sub_chunk.strip():
                            all_sentences.append(Sentence(sub_chunk))
                            sentence_to_crop_id.append(crop.id)

    # Run inference on all collected sentences
    if all_sentences:
        model.predict(all_sentences, mini_batch_size=NER_FLAIR_BATCH_SIZE)

    # Harvest entities per crop, filtering by class, confidence, and length
    raw_entities: dict[int, list[tuple[str, str, float]]] = defaultdict(list)

    for i, sentence in enumerate(all_sentences):
        crop_id = sentence_to_crop_id[i]

        for entity in sentence.get_spans("ner"):
            if entity.tag not in NER_FLAIR_CLASSES:
                continue
            if entity.score < NER_FLAIR_CONF:
                continue
            if len(entity.text) < 2:
                continue
            raw_entities[crop_id].append((entity.text, entity.tag, float(entity.score)))

    # Deduplicate per crop: case-insensitive by (text_lower, tag), keeping highest confidence
    deduped: dict[int, dict[tuple[str, str], tuple[str, float]]] = {}

    for crop_id, entities in raw_entities.items():
        seen: dict[tuple[str, str], tuple[str, float]] = {}
        for text, tag, score in entities:
            key = (text.lower(), tag)
            if key not in seen or score > seen[key][1]:
                seen[key] = (text, score)
        deduped[crop_id] = seen

    # Build CropNER records
    entries_to_create: list[CropNER] = []
    entries_to_update: list[CropNER] = []

    for crop_id in all_crop_ids:
        per_entities: list[str] = []
        per_scores: list[float] = []
        loc_entities: list[str] = []
        loc_scores: list[float] = []
        org_entities: list[str] = []
        org_scores: list[float] = []

        if crop_id in deduped:
            for (_, tag), (text, score) in deduped[crop_id].items():
                if tag == "PER":
                    per_entities.append(text)
                    per_scores.append(score)
                elif tag == "LOC":
                    loc_entities.append(text)
                    loc_scores.append(score)
                elif tag == "ORG":
                    org_entities.append(text)
                    org_scores.append(score)

        record = CropNER(
            crop=crop_id,
            per_entities=per_entities,
            per_confidence_scores=per_scores,
            loc_entities=loc_entities,
            loc_confidence_scores=loc_scores,
            org_entities=org_entities,
            org_confidence_scores=org_scores,
        )

        if crop_id in existing_ner_crop_ids:
            entries_to_update.append(record)
        else:
            entries_to_create.append(record)

    logger.info(
        f"{len(all_crop_ids)} crops processed for NER on {device}. "
        f"{len(all_sentences)} sentences inferred. "
        f"({len(entries_to_create)} created, {len(entries_to_update)} updated)"
    )

    # bulk_update doesn't work with ArrayField (peewee casts CASE as text, not text[]),
    # so delete-and-recreate instead.
    if entries_to_update:
        update_crop_ids = [entry.crop_id for entry in entries_to_update]
        CropNER.delete().where(CropNER.crop << update_crop_ids).execute()
        entries_to_create.extend(entries_to_update)
        entries_to_update.clear()

    utils.process_db_write_batch(
        model=CropNER,
        entries_to_create=entries_to_create,
    )

    return True
