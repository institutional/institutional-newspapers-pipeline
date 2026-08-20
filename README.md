# 📰 Institutional Newspapers Pipeline

The [Institutional Data Initiative](https://institutional.org/)'s pipeline for extracting high quality data from raw newspaper scans.

Developed in collaboration with the [Boston Public Library](https://www.bpl.org/), this pipeline segments newspaper scans into individual crops and enriches each crop with OCR, crop-type classification, reading order, named entities, subjects, language detection, and pre-computed embeddings.

**More information:**
- 📄 [Full analysis available in our technical report](https://arxiv.org/abs/2608.18972)

**See also:**
- 🗂️ [Institutional Newspapers Collection](https://huggingface.co/collections/institutional/institutional-newspapers)
- 📚 [Institutional Newspapers: Boston Public Library](https://huggingface.co/datasets/institutional/institutional-newspapers-bpl) — the dataset this pipeline produced
- 🤖 [Segmentation model](https://huggingface.co/institutional/institutional-newspapers-segmenter-yolo26x)
- 🤖 [Crop-type image classifier](https://huggingface.co/institutional/institutional-newspapers-crop-classifier-image-yolo26m-cls)
- 🤖 [Crop-type text classifier](https://huggingface.co/institutional/institutional-newspapers-crop-classifier-text-model2vec)

---

## Summary
- [Getting started](#getting-started)
- [Pipeline overview](#pipeline-overview)
- [CLI: system](#cli-system)
- [CLI: orchestration](#cli-orchestration)
- [CLI: steps](#cli-steps)
- [CLI: analysis](#cli-analysis)
- [CLI: peek](#cli-peek)
- [CLI: export](#cli-export)
- [Adding corpora](#adding-corpora)
- [License](#license)

---

## Getting started

**Machine-level dependencies:**
- [uv](https://docs.astral.sh/uv/)
- [Tesseract 5](https://github.com/tesseract-ocr/tessdoc) with ["best" models](https://github.com/tesseract-ocr/tessdata_best)
- [sqlite](https://sqlite.org/)
- [rclone](https://rclone.org/) (only for the `backup-push.sh` / `backup-pull.sh` helper scripts)
- CUDA-capable GPU(s)

vLLM is only installed on Linux, so the VLM OCR step (04) requires a Linux host.

```bash
# Clone project
git clone https://github.com/institutional/institutional-newspapers-pipeline.git

# Install dependencies
# NOTE: Will attempt to install system-level dependencies on macOS and Debian-based systems.
bash install.sh

# Edit environment variables
nano .env # (or any text editor)

# Run commands
uv run pipeline.py command options

# Start pipeline run 
bash run.sh 1
```

**Typical workflow:**
```bash
# Beforehand: pull every model and dataset into the local cache ...
uv run pipeline.py system cache-models

# ... and pre-compile the encoder of the OCR VLM
uv run pipeline.py system warmup-ocr-vlm

# 1. Populate database with available issues from S3
uv run pipeline.py system build

# 2. Create a pipeline run with batches
uv run pipeline.py orchestration prepare --corpus=CORPUS --items-per-batch=100

# 3. Execute the pipeline run (with logging)
./run.sh <PIPELINE_RUN_ID>

# 4. Check status
uv run pipeline.py orchestration status

# 5. Analyze logs and generate dashboard
uv run pipeline.py analysis logs
uv run pipeline.py analysis dashboard --pipeline-run-id=<ID>

# 6. Export sample data for review
uv run pipeline.py peek --pipeline-run-id=<ID> --limit=10

# 7. Build and publish the releasable dataset
uv run pipeline.py export CORPUS
```

Every command comes with a `--help` option.

[👆 Back to the summary](#summary)

---

## Pipeline overview

The pipeline processes newspaper issues through 15 sequential steps. Each step operates on a batch of issues and writes results to a SQLite database. Every step is separate and interpretable, and can be re-run on its own.

Model names and thresholds for each step are defined in `const/__init__.py`.

| Step | Name | Description | Compute | Custom model / dataset |
|------|------|-------------|---------|------------------------|
| 01 | Cache | Download archives from S3, extract and preprocess scans | CPU | No |
| 02 | Crop detection | Layout segmentation using a YOLO26x object detection model | GPU | Yes |
| 03 | OCR (Tesseract) | Extract text and word-level bounding boxes via Tesseract | CPU | No |
| 04 | OCR (VLM) | Extract text using dots.mocr, served with vLLM | GPU | No |
| 05 | Classification (text) | Categorize crops using a Model2Vec static text classifier on VLM OCR text | CPU | Yes |
| 06 | Classification (image) | Categorize crops using a YOLO26m-cls image classifier | GPU | Yes |
| 07 | Classification (final) | Merge text and image classification signals | CPU | No |
| 08 | NER | Extract named entities (persons, locations, organizations) with Flair | GPU | No |
| 09 | Subject detection | Zero-shot topic classification on OCR text with ModernBERT | GPU | No |
| 10 | Reading order | HDBSCAN column clustering to determine crop reading order | CPU | No |
| 11 | Token count | Count tokens using tiktoken | CPU | No |
| 12 | Language detection | Detect primary language (ISO 639-3) using Lingua | CPU | No |
| 13 | Text analysis | Compute text metrics (word/sentence counts, tokenizability, etc.) | CPU | No |
| 14 | ChronAm thesauri match | Detect Chronicling America thesauri terms in OCR text | CPU | Yes |
| 15 | Embeddings | Generate text (Model2Vec) and image (DINOv2) embeddings concurrently | CPU + GPU | No |

[👆 Back to the summary](#summary)

---

## CLI: system

<details>
<summary><h3>system build</h3></summary>

Populates the database with a list of all available issues for all corpora. Lists archives via S3 pagination, then fetches newspaper-level metadata from the LOC API using a thread pool. Updates existing records with fresher metadata.

```bash
uv run pipeline.py system build
```

**Options:**
- `--max-metadata-requests` (default: 8): Maximum number of parallel metadata API requests.
- `--ignore-cache`: If set, bypasses cached metadata and re-fetches from the API.

</details>

<details>
<summary><h3>system clear-cache</h3></summary>

Clears the disk cache.

```bash
uv run pipeline.py system clear-cache
```

</details>

<details>
<summary><h3>system warmup-ocr-vlm</h3></summary>

Pre-downloads and compiles the OCR VLM model so step 04 starts without warmup delay. Runs a dummy inference on the first available GPU to trigger `torch.compile` caching.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py system warmup-ocr-vlm
```

</details>

<details>
<summary><h3>system cache-models</h3></summary>

Downloads all models and datasets into local cache so they are available offline during pipeline execution. Caches Hugging Face model snapshots, tiktoken encodings, and datasets.

```bash
uv run pipeline.py system cache-models
```

</details>

[👆 Back to the summary](#summary)

---

## CLI: orchestration

<details>
<summary><h3>orchestration prepare</h3></summary>

Creates a pipeline run and its batches. Splits issues into fixed-size batches and returns a run identifier that can be passed to `orchestration execute`.

```bash
uv run pipeline.py orchestration prepare --corpus=BPL --items-per-batch=100
```

**Options:**
- `--corpus` (required): Corpus to create the run for.
- `--offset`: Start at a specific position in the issue list (ordered by id asc).
- `--limit`: Limit the number of issues to include.
- `--items-per-batch` (default: 100, range: 10–1000): Number of issues per batch. The Boston Public Library run described in our technical report used 200.
- `--append-mode`: Only include issues that are not part of any other run.
- `--shuffle`: Randomly shuffle issues before applying `--offset`/`--limit` and splitting into batches.

This command asks for confirmation before creating the run.

</details>

<details>
<summary><h3>orchestration execute</h3></summary>

Executes a pipeline run. Runs all steps sequentially within each batch, advancing to the next batch on completion. Uses batch locking for safe multi-node execution and monitors RAM/VRAM/CPU usage in a background thread.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py orchestration execute --pipeline-run-id=1
```

**Options:**
- `--pipeline-run-id`: Identifier of the pipeline run to launch.
- `--force-pipeline-batch-id`: If specified, only processes the given batch.
- `--ignore-locks`: Ignore batch locks (e.g., batch running on another machine).
- `--dry-run`: Run the pipeline without updating batch status in the database.

Prefer `./run.sh <PIPELINE_RUN_ID>`, which wraps this command with logging and records the environment configuration that `analysis logs` later parses.

</details>

<details>
<summary><h3>orchestration status</h3></summary>

Lists all pipeline runs and their batches with current status, node assignment, and timing.

```bash
uv run pipeline.py orchestration status
```

</details>

[👆 Back to the summary](#summary)

---

## CLI: steps

Individual pipeline steps can be run directly for debugging or re-processing. Each step operates on a single pipeline batch.

<details>
<summary><h3>steps step01-cache</h3></summary>

Pulls issues from remote storage, processes them and adds them to cache for the current pipeline batch. Creates `Scan` records if they don't already exist.

Uses a three-phase pipeline:
1. ThreadPool downloads archives from S3 and extracts raw bytes to disk (I/O-bound)
2. ProcessPool runs image processing across all scans (CPU-bound)
3. Sequential cache writes and DB bulk inserts (I/O-bound)

```bash
uv run pipeline.py steps step01-cache --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.

</details>

<details>
<summary><h3>steps step02-crop-detection</h3></summary>

Uses a YOLO object detection model to detect individual crops in each scan of the current batch. Spins up 1 process per available CUDA GPU.

Runs FP16 inference and parallelizes LetterBox preprocessing across the thread pool to bypass ultralytics' single-threaded image preprocessing path.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py steps step02-crop-detection --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step03-crop-ocr-tesseract</h3></summary>

Runs Tesseract OCR on every crop from the current pipeline batch. Stores full text and word-level bounding boxes in CropOCR records.

Sorts crops by language to minimize Tesseract engine re-initialization, then splits them into small chunks across a ProcessPool for load balancing.

```bash
uv run pipeline.py steps step03-crop-ocr-tesseract --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step04-crop-ocr-vlm</h3></summary>

Uses a VLM to OCR every crop from the current pipeline batch. Stores full text and inference metadata in CropOCR records. Spins up 1 process per available CUDA GPU, each running a vLLM inference server.

Uses double-buffering: a thread pool fetches and decodes images for the next batch while the GPU infers the current one. DB writes run on a background thread.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py steps step04-crop-ocr-vlm --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step05-crop-classification-text</h3></summary>

Uses a static text classifier to categorize each crop based on its VLM-extracted OCR text. Populates text_category and text_confidence_score in CropClassification records.

Runs single-process batch inference on CPU — no GPU required.

```bash
uv run pipeline.py steps step05-crop-classification-text --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step06-crop-classification-image</h3></summary>

Uses a YOLO image classifier to categorize each crop based on its visual content. Populates image_category and image_confidence_score in CropClassification records. Spins up 1 process per available CUDA GPU.

Runs FP16 inference and applies classification transforms in the thread pool, passing pre-processed tensors to bypass ultralytics' single-threaded image preprocessing path.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py steps step06-crop-classification-image --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step07-crop-classification-final</h3></summary>

Combines text and image classification signals into a final category for each crop. Pure DB operation — no model loading or GPU required.

```bash
uv run pipeline.py steps step07-crop-classification-final --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step08-crop-ner</h3></summary>

Uses a NER model to extract named entities from VLM OCR text for each crop. Populates per/loc/org entities and confidence scores in CropNER records. Spins up 1 process per available CUDA GPU.

Applies ICU sentence tokenization before inference and runs batch prediction. Deduplicates entities per crop using case-insensitive matching, keeping the highest-confidence occurrence.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py steps step08-crop-ner --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step09-crop-subject</h3></summary>

Uses a zero-shot classification model to detect subject labels from VLM OCR text for each crop. Populates ranked labels and scores in CropSubject records. Spins up 1 process per available CUDA GPU.

Runs FP16 inference via the HuggingFace zero-shot-classification pipeline with batch prediction.

**Requirements:** CUDA GPU

```bash
uv run pipeline.py steps step09-crop-subject --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step10-crop-reading-order</h3></summary>

Computes reading order for crops on each scan using HDBSCAN column clustering. Updates `crop.reading_order` with 1-based positions. Pure DB + algorithm operation — no model loading or GPU required.

```bash
uv run pipeline.py steps step10-crop-reading-order --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step11-crop-token-count</h3></summary>

Counts tokens in Tesseract and VLM OCR text for every crop in the pipeline batch using tiktoken. CPU-only, no GPU required.

```bash
uv run pipeline.py steps step11-crop-token-count --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step12-crop-language</h3></summary>

Detects the primary language of VLM OCR text for every crop in the pipeline batch. Stores ISO 639-3 code and confidence score in CropLanguage. CPU-only, no GPU required.

```bash
uv run pipeline.py steps step12-crop-language --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step13-crop-text-analysis</h3></summary>

Computes text analysis metrics (word/sentence counts, tokenizability, table/markdown detection) on Tesseract and VLM OCR text for every crop in the batch. Uses ICU (via PyICU) for language-aware word and sentence splitting. CPU-only, no GPU required.

```bash
uv run pipeline.py steps step13-crop-text-analysis --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step14-crop-chronam-thesauri-match</h3></summary>

Detects Chronicling America thesauri terms in flattened OCR text (Tesseract and VLM) for every crop in the batch. Only processes crops from English-language, USA-origin issues.

Compiles a single regex with longest-first alternation for greedy matching, then parallelizes across a ProcessPool on CPU.

```bash
uv run pipeline.py steps step14-crop-chronam-thesauri-match --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

<details>
<summary><h3>steps step15-crop-embeddings</h3></summary>

Generates text embeddings and image embeddings concurrently for every crop in the pipeline batch.

Text embeddings run across a ProcessPool on CPU using a static embedding model. Image embeddings spin up 1 process per available CUDA GPU with double-buffering (thread pool decodes and preprocesses images for the next batch while the GPU infers the current one). Both run concurrently in a top-level thread pool.

**Requirements:** CUDA GPU (for image embeddings)

```bash
uv run pipeline.py steps step15-crop-embeddings --pipeline-batch-id=1
```

**Options:**
- `--pipeline-batch-id` (required): Pipeline batch to process.
- `--overwrite`: Replace existing records.

</details>

[👆 Back to the summary](#summary)

---

## CLI: analysis

<details>
<summary><h3>analysis logs</h3></summary>

Parses pipeline log files and generates self-contained HTML reports with step timing charts, resource usage graphs, and warnings/errors summaries.

```bash
uv run pipeline.py analysis logs
```

</details>

<details>
<summary><h3>analysis dashboard</h3></summary>

Queries the database for batch progress on a pipeline run and generates a self-contained HTML dashboard with progress bars, a batch timeline, duration charts, and record counts.

```bash
uv run pipeline.py analysis dashboard --pipeline-run-id=1
```

**Options:**
- `--pipeline-run-id` (required): Pipeline run ID to generate dashboard for.
- `--ignore-cache`: Bypass cached results and regenerate the dashboard.

</details>

[👆 Back to the summary](#summary)

---

## CLI: peek

<details>
<summary><h3>peek</h3></summary>

Exports all pipeline data for a random sample of issues as JSON files. Includes scan images as base64, all crop analysis records, and pipeline config metadata. Re-downloads missing scans from S3 if needed.

```bash
uv run pipeline.py peek --pipeline-run-id=1 --limit=10
```

**Options:**
- `--pipeline-run-id`: Pipeline run to export from.
- `--pipeline-batch-id`: Pipeline batch to export from. Exactly one of `--pipeline-run-id` or `--pipeline-batch-id` must be provided.
- `--limit` (required): Number of random issues to export.

</details>

[👆 Back to the summary](#summary)

---

## CLI: export

<details>
<summary><h3>export</h3></summary>

Builds the releasable dataset as Parquet chunks (one row per scan) and uploads them in parallel to R2 (S3-compatible) and Hugging Face. Each chunk is deleted from disk as soon as it is uploaded, so disk usage stays bounded regardless of corpus size.

```bash
uv run pipeline.py export BPL
```

**How it works:**

- **Chunking.** Issues are ordered by ID and packed whole into chunks until a chunk reaches `EXPORT_CHUNK_ROW_COUNT` scans (see `const`). Packing whole issues means every source archive is downloaded exactly once. Chunk composition is deterministic across runs. Chunks are named `{CORPUS}-part-NNNNN.parquet` and written with a row-group size of `EXPORT_PARQUET_ROW_GROUP_SIZE`, kept small because embedded scan images make rows multi-megabyte.
- **Row building.** For each scan, the issue/scan/crop records and all crop analysis records are bulk-loaded, the source archive is downloaded once and every page is re-processed (CLAHE + autocontrast) and encoded to WEBP (`SCAN_WEBP_QUALITY`). One Parquet row is emitted per scan, with per-crop analysis stored as nested list columns. The `scan_image` column is flagged as a Hugging Face `Image` feature so the dataset viewer decodes the embedded WEBP bytes.
- **Column naming.** Suffixes mark provenance: `_src` (from the source archive/filename), `_ext` (from external metadata), `_gen` (generated by the pipeline), `_exp` (experimental).
- **Bounded disk use.** At most `EXPORT_MAX_INFLIGHT_CHUNKS` built-but-not-yet-uploaded chunks sit on disk at once. Building runs ahead of uploading up to that limit; each chunk uploads to R2 and HF concurrently, then its local files are removed.
- **Resume.** Each chunk is uploaded alongside a JSON manifest recording its planned issue set (kept on R2 only). With `--resume`, a chunk is skipped only when its parquet and manifest are present and the manifest's issue set matches the currently planned chunk; otherwise it is rebuilt and re-uploaded. Failed uploads leave local files in place so a later `--resume` can retry.

**Requirements:** `RELEASE_S3_BUCKET_NAME`, `RELEASE_HF_DATASET`, and `HF_TOKEN` configured in `.env`; the corpus must have an entry in `CORPUS_METADATA_SOURCE` (see `const`).

**Options:**
- `--pre-cutoff-only/--no-pre-cutoff-only` (default: on): Only export issues published before `EXPORT_CUTOFF_YEAR` (public domain).
- `--build-workers` (default: `EXPORT_BUILD_WORKERS`): Chunks built concurrently. Each divides `CPUS_LIMIT` for its internal image ProcessPool.
- `--max-inflight-chunks` (default: `EXPORT_MAX_INFLIGHT_CHUNKS`): Max built-but-not-yet-uploaded chunks allowed on disk at once.
- `--resume`: Skip chunks already uploaded (with a matching manifest) to both destinations.
- `--test-run`: Build and upload only the first chunk, then stop.

</details>

[👆 Back to the summary](#summary)

---

## Adding corpora

### Archive storage

Each corpus must be stored on S3-compatible storage in its own dedicated bucket (the bucket must not contain anything else). The pipeline discovers archives by paginating through all `.tar.gz` objects in the bucket.

### Archive format

- Archives must be **`.tar.gz`** files containing scan images and nothing else.
- Supported image formats: **JP2**, **JPEG**, **TIFF**.
- **Scan filenames must sort alphabetically by page number**, since the pipeline determines page order by sorting archive members by name (e.g., `0001.jp2`, `0002.jp2`, ... or `page_001.tif`, `page_002.tif`, ...).
- **Archive filenames must contain key identifiers** for the newspaper and edition. These identifiers are corpus-specific and parsed in code (see below).

### Configuration

Add the corpus to `.env` (and `.env.example`):

```bash
# Add corpus name to the comma-separated list
CORPORA="BPL,YOURCORPUS"

# Add S3 credentials for the new corpus
YOURCORPUS_S3_BUCKET_NAME=""
YOURCORPUS_S3_ENDPOINT=""
YOURCORPUS_S3_REGION=""
YOURCORPUS_S3_ACCESS_KEY_ID=""
YOURCORPUS_S3_SECRET_ACCESS_KEY=""
```

### Code changes

Add an `elif` branch for the new corpus in `pull_issue_metadata()` in `commands/system/build.py`. This is where archive filenames are parsed into `Issue` fields.

Use the existing BPL branch as a reference:

```python
def pull_issue_metadata(
    corpus: str,
    filename: str,
    filesize: int,
    ignore_cache: bool = False,
) -> tuple[Issue, bool]:
    existing = Issue.get_or_none(
        (Issue.corpus == corpus) & (Issue.archive_filename == filename)
    )
    is_new = existing is None
    issue = existing or Issue()

    issue.corpus = corpus
    issue.archive_filename = filename
    issue.archive_size_bytes = filesize

    if corpus == "BPL":
        # BPL filename format: {lccn}_{YYYYMMDDEE}.tar.gz
        # ...

    elif corpus == "YOURCORPUS":
        # Validate and parse the archive filename
        if not re.match(r"^your-pattern-here\.tar\.gz$", filename):
            raise ValueError(f"Invalid YOURCORPUS archive filename: '{filename}'.")

        stem = filename.removesuffix(".tar.gz")

        # Extract identifiers from the filename
        issue.newspaper_id = ...          # Unique ID for the newspaper title
        issue.newspaper_id_type = "..."   # Type label (e.g., "lccn", "issn")
        issue.edition_slug = ...          # Unique ID for this specific edition
        issue.edition_slug_type = "..."   # Format description (e.g., "YYYYMMDDEE")

        # Parse date fields from the filename or edition slug
        issue.year = ...
        issue.month = ...
        issue.day = ...
        issue.edition_number = ...

        # Optionally fetch metadata from an external API
        # and populate title, city, state, country, publisher, language, etc.

    return issue, is_new
```

Once the branch is in place, run `uv run pipeline.py system build` to populate the database with the new corpus.

Two more things to adjust for a new corpus: add an entry to `CORPUS_METADATA_SOURCE` in `const/__init__.py`, which `export` requires, and review the post-processing rules in `utils/postprocess_locality.py`, which were written against Boston Public Library's metadata.

[👆 Back to the summary](#summary)

---

## License

This pipeline is released under the [GNU Affero General Public License v3.0](LICENSE).

The models and the dataset released alongside it carry their own terms; see the links at the top of this document.

[👆 Back to the summary](#summary)

---

## Cite 

```bibtext
@misc{cargnelutti2026institutionalnewspaperspipelinederiving,
      title={Institutional Newspapers Pipeline: Deriving billions of high quality tokens from historical newspapers}, 
      author={Matteo Cargnelutti and Catherine Brobston and Eben English and Jake Sadow and Kacie Bailey and Greg Leppert and Amanda Watson and Jessica Chapel and Jonathan Zittrain},
      year={2026},
      eprint={2608.18972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.18972}, 
}
```
