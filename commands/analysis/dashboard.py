import json
import random
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from math import ceil, log2
from pathlib import Path
from statistics import median, stdev
from string import Template

import click
import peewee
from loguru import logger

import const
import utils
from utils import dashboard_temp_tables
from models import (
    Crop,
    CropClassification,
    CropLanguage,
    CropTextAnalysis,
    CropTokenCount,
    Issue,
    PipelineBatch,
    PipelineBatchItem,
    PipelineRun,
    Scan,
)

_CROP_IN_RUN = peewee.SQL("(SELECT crop_id FROM _dash_crops)")
_CROP_PRE1931 = peewee.SQL("(SELECT crop_id FROM _dash_crops_pre1931)")

MAX_SCORES = 50_000
_CACHE_KEY_PREFIX = "dashboard_v2"
_CACHE_EXPIRE_SECONDS = 43_200  # 12 hours

STATUS_CSS = {
    "completed": "badge-success",
    "crashed": "badge-error",
    "running": "badge-warning",
    "pending": "badge-pending",
}

CHART_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be185d",
    "#65a30d",
    "#6d28d9",
    "#0d9488",
    "#d97706",
    "#4f46e5",
    "#059669",
    "#e11d48",
    "#7c3aed",
]

_STATE_ABBR_MAP: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}

_ABBR_SET = set(_STATE_ABBR_MAP.values())

_COUNTRY_NAME_MAP: dict[str, str] = {
    "united states": "United States of America",
    "usa": "United States of America",
    "us": "United States of America",
    "u.s.": "United States of America",
    "u.s.a.": "United States of America",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "great britain": "United Kingdom",
    "russia": "Russia",
    "russian federation": "Russia",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "north korea": "North Korea",
    "democratic people's republic of korea": "North Korea",
    "czech republic": "Czechia",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
}


@dataclass
class GeographyRow:
    city: str | None
    state: str | None
    country: str | None
    count: int


@dataclass
class CropTextStats:
    tesseract_total_tokens: int
    vlm_total_tokens: int
    tesseract_total_words: int
    vlm_total_words: int
    tesseract_total_sentences: int
    vlm_total_sentences: int
    tesseract_avg_wttr: float | None
    vlm_avg_wttr: float | None
    tesseract_median_wttr: float | None
    vlm_median_wttr: float | None
    vlm_crops_with_tables: int
    vlm_crops_with_markdown: int


@dataclass
class CropClassStats:
    agreement_count: int
    final_eq_image_count: int
    final_eq_text_count: int
    total: int


@dataclass
class DashboardData:
    run: PipelineRun
    batches: list[PipelineBatch]
    batch_statuses: dict[int, str]
    batch_durations: dict[int, float]
    batch_items_per_batch: dict[int, int]
    batch_crop_counts: dict[int, int]
    record_counts: OrderedDict[str, int]
    avg_batch_duration: float | None
    eta_seconds: float | None
    generated_at: datetime

    issue_count: int = 0
    scan_count: int = 0

    geography_counts: list[GeographyRow] = field(default_factory=list)

    scan_resolutions_mp: list[float] = field(default_factory=list)

    crop_confidence_scores: list[float] = field(default_factory=list)
    crops_per_scan: list[int] = field(default_factory=list)
    crops_per_scan_by_year: dict[int, list[int]] = field(default_factory=dict)
    scan_coverage_pcts: list[float] = field(default_factory=list)
    scan_coverage_by_year: dict[int, list[float]] = field(default_factory=dict)

    crops_by_language: list[tuple[str, int]] = field(default_factory=list)
    tokens_by_language: list[tuple[str, int]] = field(default_factory=list)

    crop_text_stats: CropTextStats | None = None
    tokens_over_time: dict[int, tuple[int, int]] = field(default_factory=dict)
    tokens_per_scan_over_time: dict[int, tuple[float, float]] = field(default_factory=dict)
    words_per_scan_over_time: dict[int, tuple[float, float]] = field(default_factory=dict)
    wttr_by_language: dict[str, list[float]] = field(default_factory=dict)

    crop_class_stats: CropClassStats | None = None
    crops_by_final_category: list[tuple[str, int]] = field(default_factory=list)
    category_over_time: dict[int, dict[str, int]] = field(default_factory=dict)
    category_area_over_time: dict[int, dict[str, float]] = field(default_factory=dict)
    category_individual_area_over_time: dict[int, dict[str, float]] = field(default_factory=dict)
    image_confidence_by_category: dict[str, list[float]] = field(default_factory=dict)
    text_confidence_by_category: dict[str, list[float]] = field(default_factory=dict)
    tokens_by_final_category: list[tuple[str, int]] = field(default_factory=list)
    tokens_per_category_over_time: dict[int, dict[str, int]] = field(default_factory=dict)

    top_labels: list[tuple[str, int]] = field(default_factory=list)
    label_confidence_scores: dict[str, list[float]] = field(default_factory=dict)

    ner_per_top: list[tuple[str, int]] = field(default_factory=list)
    ner_loc_top: list[tuple[str, int]] = field(default_factory=list)
    ner_org_top: list[tuple[str, int]] = field(default_factory=list)
    ner_type_totals: dict[str, int] = field(default_factory=dict)
    ner_per_scores: list[float] = field(default_factory=list)
    ner_loc_scores: list[float] = field(default_factory=list)
    ner_org_scores: list[float] = field(default_factory=list)

    chronam_term_totals: list[tuple[str, int]] = field(default_factory=list)
    chronam_terms_over_time: dict[int, dict[str, int]] = field(default_factory=dict)

    pre1931: "DashboardData | None" = None


@click.command("dashboard")
@click.option(
    "--pipeline-run-id",
    type=int,
    required=True,
    help="Pipeline run ID to generate dashboard for.",
)
@click.option(
    "--ignore-cache",
    is_flag=True,
    default=False,
    help="Bypass cached results and regenerate the dashboard.",
)
def dashboard(pipeline_run_id: int, ignore_cache: bool):
    """Generates a self-contained HTML dashboard with visualizations for a pipeline run."""
    cache = utils.get_cache()
    cache_key = f"{_CACHE_KEY_PREFIX}:{pipeline_run_id}"
    data_cache_key = f"{_CACHE_KEY_PREFIX}:data:{pipeline_run_id}"

    html: str | None = None
    if not ignore_cache:
        html = cache.get(cache_key)
        if html is not None:
            logger.info("Loaded dashboard from cache.")

    if html is None:
        data: DashboardData | None = None
        if not ignore_cache:
            data = cache.get(data_cache_key)
            if data is not None:
                logger.info("Loaded dashboard data from cache, re-rendering.")

        if data is None:
            try:
                run = PipelineRun.get_by_id(pipeline_run_id)
            except PipelineRun.DoesNotExist:
                logger.error(f"Pipeline run #{pipeline_run_id} not found.")
                return
            try:
                data = _load_dashboard_data(run)
            except Exception:
                logger.exception("Failed to load dashboard data.")
                raise
            cache.set(data_cache_key, data, expire=_CACHE_EXPIRE_SECONDS)

        html = _generate_html(data)
        cache.set(cache_key, html, expire=_CACHE_EXPIRE_SECONDS)

    filename = f"pipeline-run-{pipeline_run_id}-{const.DATETIME_SLUG}.html"
    output_path = Path(const.DASHBOARD_DIR_PATH, filename)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Dashboard saved: {output_path}")


def _load_dashboard_data(run: PipelineRun) -> DashboardData:
    """Collects all data needed for the dashboard report."""
    db = utils.get_db()

    batches = list(
        PipelineBatch.select()
        .where(PipelineBatch.pipeline_run == run.id)
        .order_by(PipelineBatch.id)
    )
    batch_ids = [b.id for b in batches]

    if batch_ids:
        dashboard_temp_tables.create_crop_temp_table(db, batch_ids)
        dashboard_temp_tables.create_crop_year_temp_table(db, batch_ids)

    try:
        data = _collect_dashboard_data(db, run, batches, batch_ids)
        if batch_ids:
            pre1931_count = dashboard_temp_tables.create_crop_pre1931_temp_table(db)
            if pre1931_count > 0:
                data.pre1931 = _collect_pre1931_data(db, data)
        return data
    finally:
        if batch_ids:
            dashboard_temp_tables.drop_temp_tables(db)


def _collect_dashboard_data(
    db: peewee.Database,
    run: PipelineRun,
    batches: list[PipelineBatch],
    batch_ids: list[int],
) -> DashboardData:
    """Runs all dashboard queries against the pre-populated temp tables."""
    batch_statuses = {b.id: _get_batch_status(b) for b in batches}

    batch_durations: dict[int, float] = {}
    for b in batches:
        if batch_statuses[b.id] == "completed" and b.started_date and b.ended_date:
            batch_durations[b.id] = (b.ended_date - b.started_date).total_seconds()

    batch_items_per_batch: dict[int, int] = {}
    if batch_ids:
        item_counts = (
            PipelineBatchItem.select(
                PipelineBatchItem.pipeline_batch,
                peewee.fn.COUNT(PipelineBatchItem.id).alias("count"),
            )
            .where(PipelineBatchItem.pipeline_batch.in_(batch_ids))
            .group_by(PipelineBatchItem.pipeline_batch)
        )
        for row in item_counts:
            batch_items_per_batch[row.pipeline_batch_id] = row.count

    batch_crop_counts: dict[int, int] = {}
    if batch_ids:
        crop_counts = (
            Crop.select(
                PipelineBatchItem.pipeline_batch,
                peewee.fn.COUNT(Crop.id).alias("count"),
            )
            .join(PipelineBatchItem)
            .where(PipelineBatchItem.pipeline_batch.in_(batch_ids))
            .group_by(PipelineBatchItem.pipeline_batch)
        )
        for row in crop_counts:
            batch_crop_counts[row.pipeline_batch_item.pipeline_batch_id] = row.count

    logger.debug("Loading record counts ...")
    record_counts = _load_record_counts(db) if batch_ids else OrderedDict()

    # Time estimation
    avg_batch_duration: float | None = None
    eta_seconds: float | None = None
    if batch_durations:
        avg_batch_duration = sum(batch_durations.values()) / len(batch_durations)
        now = datetime.now(timezone.utc)
        remaining = 0.0
        for b in batches:
            status = batch_statuses[b.id]
            if status == "pending":
                remaining += avg_batch_duration
            elif status == "running" and b.started_date:
                elapsed = (now - b.started_date).total_seconds()
                remaining += max(0.0, avg_batch_duration - elapsed)
        eta_seconds = remaining if remaining > 0 else None

    # Issue and Scan counts
    issue_count = 0
    scan_count = 0
    if batch_ids:
        issue_count = (
            PipelineBatchItem.select(peewee.fn.COUNT(PipelineBatchItem.issue.distinct()))
            .where(PipelineBatchItem.pipeline_batch.in_(batch_ids))
            .scalar()
        )
        scan_count = (
            Scan.select(peewee.fn.COUNT(Scan.id))
            .join(Issue)
            .join(
                PipelineBatchItem,
                on=(PipelineBatchItem.issue == Issue.id),
            )
            .where(PipelineBatchItem.pipeline_batch.in_(batch_ids))
            .scalar()
        )

    # Geography
    geography_counts: list[GeographyRow] = []
    if batch_ids:
        geo_query = (
            Issue.select(
                Issue.city,
                Issue.state,
                Issue.country,
                peewee.fn.COUNT(Issue.id).alias("count"),
            )
            .join(
                PipelineBatchItem,
                on=(PipelineBatchItem.issue == Issue.id),
            )
            .where(PipelineBatchItem.pipeline_batch.in_(batch_ids))
            .group_by(Issue.city, Issue.state, Issue.country)
            .order_by(peewee.fn.COUNT(Issue.id).desc())
        )
        for row in geo_query:
            geography_counts.append(
                GeographyRow(
                    city=row.city,
                    state=row.state,
                    country=row.country,
                    count=row.count,
                )
            )

    # Scan dimensions
    scan_resolutions_mp: list[float] = []
    if batch_ids:
        scan_dims = (
            Scan.select(Scan.width, Scan.height)
            .join(Issue)
            .join(
                PipelineBatchItem,
                on=(PipelineBatchItem.issue == Issue.id),
            )
            .where(PipelineBatchItem.pipeline_batch.in_(batch_ids))
            .where(Scan.width.is_null(False), Scan.height.is_null(False))
            .tuples()
        )
        for w, h in scan_dims:
            scan_resolutions_mp.append(w * h / 1_000_000)

    # Crop confidence scores
    crop_confidence_scores: list[float] = []
    if batch_ids:
        scores = (
            Crop.select(Crop.confidence_score)
            .where(Crop.id.in_(_CROP_IN_RUN))
            .where(Crop.confidence_score.is_null(False))
            .tuples()
        )
        crop_confidence_scores = [s[0] for s in scores]

    # Crops per scan (uses _dash_crops_year for year lookup)
    crops_per_scan: list[int] = []
    crops_per_scan_by_year: dict[int, list[int]] = {}
    if batch_ids:
        cps_rows = db.execute_sql(
            "SELECT c.scan_id, dcy.year, COUNT(c.id) AS n "
            "FROM crop c "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = c.id "
            "GROUP BY c.scan_id"
        ).fetchall()
        for _, year, n in cps_rows:
            crops_per_scan.append(n)
            if year is not None:
                crops_per_scan_by_year.setdefault(year, []).append(n)

    # Scan coverage by crop bounding box area
    scan_coverage_pcts: list[float] = []
    scan_coverage_by_year: dict[int, list[float]] = {}
    if batch_ids:
        scan_dims_map: dict[int, tuple[int, int]] = {}
        scan_dim_rows = db.execute_sql(
            "SELECT DISTINCT c.scan_id, s.width, s.height "
            "FROM crop c "
            "INNER JOIN scan s ON s.id = c.scan_id "
            "INNER JOIN _dash_crops dc ON dc.crop_id = c.id "
            "WHERE s.width IS NOT NULL AND s.height IS NOT NULL"
        ).fetchall()
        for scan_id, w, h in scan_dim_rows:
            scan_dims_map[scan_id] = (w, h)

        scan_year_map: dict[int, int | None] = {}
        scan_year_rows = db.execute_sql(
            "SELECT DISTINCT c.scan_id, dcy.year "
            "FROM crop c "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = c.id"
        ).fetchall()
        for scan_id, year in scan_year_rows:
            scan_year_map.setdefault(scan_id, year)

        scan_bbox_area: dict[int, float] = {}
        bbox_rows = (
            Crop.select(Crop.scan, Crop.bbox_xyxy)
            .where(Crop.id.in_(_CROP_IN_RUN))
            .where(Crop.bbox_xyxy.is_null(False))
            .tuples()
        )
        for scan_id, bbox in bbox_rows:
            if bbox and len(bbox) == 4 and scan_id in scan_dims_map:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                scan_bbox_area[scan_id] = scan_bbox_area.get(scan_id, 0.0) + area

        for scan_id, total_bbox_area in scan_bbox_area.items():
            w, h = scan_dims_map[scan_id]
            scan_area = w * h
            if scan_area > 0:
                pct = total_bbox_area / scan_area * 100
                scan_coverage_pcts.append(pct)
                year = scan_year_map.get(scan_id)
                if year is not None:
                    scan_coverage_by_year.setdefault(year, []).append(pct)

    # Crop language
    crops_by_language: list[tuple[str, int]] = []
    tokens_by_language: list[tuple[str, int]] = []
    if batch_ids:
        lang_query = (
            CropLanguage.select(
                CropLanguage.language_code,
                peewee.fn.COUNT(CropLanguage.crop).alias("count"),
            )
            .where(CropLanguage.crop.in_(_CROP_IN_RUN))
            .where(CropLanguage.language_code.is_null(False))
            .group_by(CropLanguage.language_code)
            .order_by(peewee.fn.COUNT(CropLanguage.crop).desc())
        )
        crops_by_language = [(row.language_code, row.count) for row in lang_query]

        tok_lang_query = (
            CropTokenCount.select(
                CropLanguage.language_code,
                peewee.fn.SUM(CropTokenCount.vlm_token_count).alias("total"),
            )
            .join(
                CropLanguage,
                on=(CropLanguage.crop == CropTokenCount.crop),
            )
            .where(CropTokenCount.crop.in_(_CROP_IN_RUN))
            .where(CropLanguage.language_code.is_null(False))
            .group_by(CropLanguage.language_code)
            .order_by(peewee.fn.SUM(CropTokenCount.vlm_token_count).desc())
            .dicts()
        )
        tokens_by_language = [
            (row["language_code"], int(row["total"] or 0)) for row in tok_lang_query
        ]

    logger.debug("Loading crop text stats ...")
    crop_text_stats = _load_crop_text_stats(db)

    # Tokens/words over time (uses _dash_crops_year)
    tokens_over_time: dict[int, tuple[int, int]] = {}
    tokens_per_scan_over_time: dict[int, tuple[float, float]] = {}
    words_per_scan_over_time: dict[int, tuple[float, float]] = {}
    if batch_ids:
        tok_time_rows = db.execute_sql(
            "SELECT dcy.year, "
            "SUM(ctc.tesseract_token_count), SUM(ctc.vlm_token_count) "
            "FROM crop_token_count ctc "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = ctc.crop_id "
            "WHERE dcy.year IS NOT NULL "
            "GROUP BY dcy.year ORDER BY dcy.year"
        ).fetchall()

        scan_counts_by_year: dict[int, int] = {}
        sc_rows = db.execute_sql(
            "SELECT i.year, COUNT(DISTINCT s.id) "
            "FROM scan s "
            "INNER JOIN issue i ON i.id = s.issue_id "
            "INNER JOIN pipeline_batch_item pbi ON pbi.issue_id = i.id "
            "WHERE pbi.pipeline_batch_id IN "
            "(SELECT DISTINCT pbi2.pipeline_batch_id "
            " FROM pipeline_batch_item pbi2 "
            " INNER JOIN crop c ON c.pipeline_batch_item_id = pbi2.id "
            " WHERE c.id IN (SELECT crop_id FROM _dash_crops)) "
            "AND i.year IS NOT NULL "
            "GROUP BY i.year"
        ).fetchall()
        for year, n_scans in sc_rows:
            scan_counts_by_year[year] = n_scans

        for year, tess, vlm in tok_time_rows:
            tess = int(tess or 0)
            vlm = int(vlm or 0)
            tokens_over_time[year] = (tess, vlm)
            n_scans = scan_counts_by_year.get(year, 1)
            tokens_per_scan_over_time[year] = (tess / n_scans, vlm / n_scans)

        word_time_rows = db.execute_sql(
            "SELECT dcy.year, "
            "SUM(cta.tesseract_word_count), SUM(cta.vlm_word_count) "
            "FROM crop_text_analysis cta "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = cta.crop_id "
            "WHERE dcy.year IS NOT NULL "
            "GROUP BY dcy.year ORDER BY dcy.year"
        ).fetchall()
        for year, tess, vlm in word_time_rows:
            tess = int(tess or 0)
            vlm = int(vlm or 0)
            n_scans = scan_counts_by_year.get(year, 1)
            words_per_scan_over_time[year] = (tess / n_scans, vlm / n_scans)

    # Word type-token ratio by language
    wttr_by_language: dict[str, list[float]] = {}
    if batch_ids:
        wttr_query = (
            CropTextAnalysis.select(
                CropLanguage.language_code,
                CropTextAnalysis.vlm_word_type_token_ratio,
            )
            .join(
                CropLanguage,
                on=(CropLanguage.crop == CropTextAnalysis.crop),
            )
            .where(CropTextAnalysis.crop.in_(_CROP_IN_RUN))
            .where(CropLanguage.language_code.is_null(False))
            .where(CropTextAnalysis.vlm_word_type_token_ratio.is_null(False))
            .tuples()
        )
        for lang, ratio in wttr_query:
            wttr_by_language.setdefault(lang, []).append(ratio)

    # Crop classification
    logger.debug("Loading crop classification stats ...")
    crop_class_stats, crops_by_final_category = _load_crop_classification_stats()
    category_over_time: dict[int, dict[str, int]] = {}
    image_confidence_by_category: dict[str, list[float]] = {}
    text_confidence_by_category: dict[str, list[float]] = {}
    if batch_ids:
        cat_time_rows = db.execute_sql(
            "SELECT dcy.year, cc.final_category, COUNT(cc.crop_id) "
            "FROM crop_classification cc "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = cc.crop_id "
            "WHERE dcy.year IS NOT NULL AND cc.final_category IS NOT NULL "
            "GROUP BY dcy.year, cc.final_category "
            "ORDER BY dcy.year"
        ).fetchall()
        for year, cat, n in cat_time_rows:
            category_over_time.setdefault(year, {})[cat] = n

        conf_query = (
            CropClassification.select(
                CropClassification.image_category,
                CropClassification.image_confidence_score,
                CropClassification.text_category,
                CropClassification.text_confidence_score,
            )
            .where(CropClassification.crop.in_(_CROP_IN_RUN))
            .tuples()
        )
        for img_cat, img_conf, txt_cat, txt_conf in conf_query:
            if img_cat and img_conf is not None:
                image_confidence_by_category.setdefault(img_cat, []).append(img_conf)
            if txt_cat and txt_conf is not None:
                text_confidence_by_category.setdefault(txt_cat, []).append(txt_conf)

    tokens_by_final_category: list[tuple[str, int]] = []
    if batch_ids:
        tok_cat_rows = db.execute_sql(
            "SELECT cc.final_category, COALESCE(SUM(ctc.vlm_token_count), 0) "
            "FROM crop_classification cc "
            "INNER JOIN crop_token_count ctc ON ctc.crop_id = cc.crop_id "
            "WHERE cc.crop_id IN (SELECT crop_id FROM _dash_crops) "
            "AND cc.final_category IS NOT NULL "
            "GROUP BY cc.final_category "
            "ORDER BY 2 DESC"
        ).fetchall()
        tokens_by_final_category = [(cat, int(total)) for cat, total in tok_cat_rows]

    tokens_per_category_over_time: dict[int, dict[str, int]] = {}
    if batch_ids:
        tok_cat_time_rows = db.execute_sql(
            "SELECT dcy.year, cc.final_category, "
            "COALESCE(SUM(ctc.vlm_token_count), 0) "
            "FROM crop_classification cc "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = cc.crop_id "
            "INNER JOIN crop_token_count ctc ON ctc.crop_id = cc.crop_id "
            "WHERE dcy.year IS NOT NULL "
            "AND cc.final_category IS NOT NULL "
            "GROUP BY dcy.year, cc.final_category "
            "ORDER BY dcy.year"
        ).fetchall()
        for year, cat, total in tok_cat_time_rows:
            tokens_per_category_over_time.setdefault(year, {})[cat] = int(total)

    # Category area coverage over time
    category_area_over_time: dict[int, dict[str, float]] = {}
    category_individual_area_over_time: dict[int, dict[str, float]] = {}
    if batch_ids:
        cat_area_rows = db.execute_sql(
            "SELECT c.scan_id, c.bbox_xyxy, cc.final_category, "
            "dcy.year, s.width, s.height "
            "FROM crop c "
            "INNER JOIN crop_classification cc ON cc.crop_id = c.id "
            "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = c.id "
            "INNER JOIN scan s ON s.id = c.scan_id "
            "WHERE c.bbox_xyxy IS NOT NULL "
            "AND cc.final_category IS NOT NULL "
            "AND dcy.year IS NOT NULL "
            "AND s.width IS NOT NULL AND s.height IS NOT NULL"
        ).fetchall()

        scan_cat_ratios: dict[tuple[int, int, str], list[float]] = {}
        for scan_id, bbox_json, cat, year, sw, sh in cat_area_rows:
            bbox = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            scan_area = sw * sh
            if scan_area <= 0:
                continue
            ratio = (x2 - x1) * (y2 - y1) / scan_area * 100
            scan_cat_ratios.setdefault((scan_id, year, cat), []).append(ratio)

        year_cat_scan_sums: dict[tuple[int, str], list[float]] = {}
        year_cat_individual: dict[tuple[int, str], list[float]] = {}
        for (scan_id, year, cat), ratios in scan_cat_ratios.items():
            yk = (year, cat)
            year_cat_scan_sums.setdefault(yk, []).append(sum(ratios))
            year_cat_individual.setdefault(yk, []).extend(ratios)

        for (year, cat), sums in year_cat_scan_sums.items():
            category_area_over_time.setdefault(year, {})[cat] = sum(sums) / len(sums)
        for (year, cat), vals in year_cat_individual.items():
            category_individual_area_over_time.setdefault(year, {})[cat] = sum(vals) / len(vals)

    logger.debug("Loading crop subject data ...")
    top_labels, label_confidence_scores = _load_crop_subject_data()
    logger.debug("Loading crop NER data ...")
    ner_data = _load_crop_ner_data()
    logger.debug("Loading chronam data ...")
    chronam_term_totals, chronam_terms_over_time = _load_chronam_data(db)

    return DashboardData(
        run=run,
        batches=batches,
        batch_statuses=batch_statuses,
        batch_durations=batch_durations,
        batch_items_per_batch=batch_items_per_batch,
        batch_crop_counts=batch_crop_counts,
        record_counts=record_counts,
        avg_batch_duration=avg_batch_duration,
        eta_seconds=eta_seconds,
        generated_at=datetime.now(timezone.utc),
        issue_count=issue_count,
        scan_count=scan_count,
        geography_counts=geography_counts,
        scan_resolutions_mp=scan_resolutions_mp,
        crop_confidence_scores=crop_confidence_scores,
        crops_per_scan=crops_per_scan,
        crops_per_scan_by_year=crops_per_scan_by_year,
        scan_coverage_pcts=scan_coverage_pcts,
        scan_coverage_by_year=scan_coverage_by_year,
        crops_by_language=crops_by_language,
        tokens_by_language=tokens_by_language,
        crop_text_stats=crop_text_stats,
        tokens_over_time=tokens_over_time,
        tokens_per_scan_over_time=tokens_per_scan_over_time,
        words_per_scan_over_time=words_per_scan_over_time,
        wttr_by_language=wttr_by_language,
        crop_class_stats=crop_class_stats,
        crops_by_final_category=crops_by_final_category,
        category_over_time=category_over_time,
        category_area_over_time=category_area_over_time,
        category_individual_area_over_time=category_individual_area_over_time,
        image_confidence_by_category=image_confidence_by_category,
        text_confidence_by_category=text_confidence_by_category,
        tokens_by_final_category=tokens_by_final_category,
        tokens_per_category_over_time=tokens_per_category_over_time,
        top_labels=top_labels,
        label_confidence_scores=label_confidence_scores,
        ner_per_top=ner_data["per_top"],
        ner_loc_top=ner_data["loc_top"],
        ner_org_top=ner_data["org_top"],
        ner_type_totals=ner_data["type_totals"],
        ner_per_scores=ner_data["per_scores"],
        ner_loc_scores=ner_data["loc_scores"],
        ner_org_scores=ner_data["org_scores"],
        chronam_term_totals=chronam_term_totals,
        chronam_terms_over_time=chronam_terms_over_time,
    )


def _collect_pre1931_data(db: peewee.Database, full_data: DashboardData) -> DashboardData:
    """Builds a pre-1931 filtered DashboardData from the full dataset."""
    crop_filter = _CROP_PRE1931
    crop_table = "_dash_crops_pre1931"

    # Category A: filter year-keyed dicts in Python (no re-query needed)
    pre_crops_per_scan_by_year = {
        y: v for y, v in full_data.crops_per_scan_by_year.items() if y < 1931
    }
    pre_scan_coverage_by_year = {
        y: v for y, v in full_data.scan_coverage_by_year.items() if y < 1931
    }
    pre_tokens_over_time = {
        y: v for y, v in full_data.tokens_over_time.items() if y < 1931
    }
    pre_tokens_per_scan_over_time = {
        y: v for y, v in full_data.tokens_per_scan_over_time.items() if y < 1931
    }
    pre_words_per_scan_over_time = {
        y: v for y, v in full_data.words_per_scan_over_time.items() if y < 1931
    }
    pre_category_over_time = {
        y: v for y, v in full_data.category_over_time.items() if y < 1931
    }
    pre_category_area_over_time = {
        y: v for y, v in full_data.category_area_over_time.items() if y < 1931
    }
    pre_category_individual_area_over_time = {
        y: v for y, v in full_data.category_individual_area_over_time.items() if y < 1931
    }
    pre_tokens_per_category_over_time = {
        y: v for y, v in full_data.tokens_per_category_over_time.items() if y < 1931
    }
    pre_chronam_terms_over_time = {
        y: v for y, v in full_data.chronam_terms_over_time.items() if y < 1931
    }

    # Derived flat lists from filtered year dicts
    pre_crops_per_scan: list[int] = []
    for vals in pre_crops_per_scan_by_year.values():
        pre_crops_per_scan.extend(vals)
    pre_scan_coverage_pcts: list[float] = []
    for vals in pre_scan_coverage_by_year.values():
        pre_scan_coverage_pcts.extend(vals)

    # Category B: re-query using _dash_crops_pre1931
    record_counts = _load_record_counts(db, crop_table)

    # Geography
    geography_counts: list[GeographyRow] = []
    geo_query = db.execute_sql(
        "SELECT i.city, i.state, i.country, COUNT(DISTINCT i.id) "
        "FROM issue i "
        "INNER JOIN scan s ON s.issue_id = i.id "
        "INNER JOIN crop c ON c.scan_id = s.id "
        f"INNER JOIN {crop_table} dc ON dc.crop_id = c.id "
        "GROUP BY i.city, i.state, i.country "
        "ORDER BY COUNT(DISTINCT i.id) DESC"
    ).fetchall()
    for city, state, country, count in geo_query:
        geography_counts.append(GeographyRow(city=city, state=state, country=country, count=count))

    # Scan resolutions
    scan_resolutions_mp: list[float] = []
    scan_dims = db.execute_sql(
        "SELECT DISTINCT s.id, s.width, s.height "
        "FROM scan s "
        "INNER JOIN crop c ON c.scan_id = s.id "
        f"INNER JOIN {crop_table} dc ON dc.crop_id = c.id "
        "WHERE s.width IS NOT NULL AND s.height IS NOT NULL"
    ).fetchall()
    for _, w, h in scan_dims:
        scan_resolutions_mp.append(w * h / 1_000_000)

    # Crop confidence scores
    crop_confidence_scores: list[float] = []
    scores = (
        Crop.select(Crop.confidence_score)
        .where(Crop.id.in_(crop_filter))
        .where(Crop.confidence_score.is_null(False))
        .tuples()
    )
    crop_confidence_scores = [s[0] for s in scores]

    # Language
    crops_by_language: list[tuple[str, int]] = []
    tokens_by_language: list[tuple[str, int]] = []
    lang_query = (
        CropLanguage.select(
            CropLanguage.language_code,
            peewee.fn.COUNT(CropLanguage.crop).alias("count"),
        )
        .where(CropLanguage.crop.in_(crop_filter))
        .where(CropLanguage.language_code.is_null(False))
        .group_by(CropLanguage.language_code)
        .order_by(peewee.fn.COUNT(CropLanguage.crop).desc())
    )
    crops_by_language = [(row.language_code, row.count) for row in lang_query]

    tok_lang_query = (
        CropTokenCount.select(
            CropLanguage.language_code,
            peewee.fn.SUM(CropTokenCount.vlm_token_count).alias("total"),
        )
        .join(CropLanguage, on=(CropLanguage.crop == CropTokenCount.crop))
        .where(CropTokenCount.crop.in_(crop_filter))
        .where(CropLanguage.language_code.is_null(False))
        .group_by(CropLanguage.language_code)
        .order_by(peewee.fn.SUM(CropTokenCount.vlm_token_count).desc())
        .dicts()
    )
    tokens_by_language = [
        (row["language_code"], int(row["total"] or 0)) for row in tok_lang_query
    ]

    # WTTR by language
    wttr_by_language: dict[str, list[float]] = {}
    wttr_query = (
        CropTextAnalysis.select(
            CropLanguage.language_code,
            CropTextAnalysis.vlm_word_type_token_ratio,
        )
        .join(CropLanguage, on=(CropLanguage.crop == CropTextAnalysis.crop))
        .where(CropTextAnalysis.crop.in_(crop_filter))
        .where(CropLanguage.language_code.is_null(False))
        .where(CropTextAnalysis.vlm_word_type_token_ratio.is_null(False))
        .tuples()
    )
    for lang, ratio in wttr_query:
        wttr_by_language.setdefault(lang, []).append(ratio)

    crop_text_stats = _load_crop_text_stats(db, crop_filter, crop_table)
    crop_class_stats, crops_by_final_category = _load_crop_classification_stats(crop_filter)

    # Confidence by category
    image_confidence_by_category: dict[str, list[float]] = {}
    text_confidence_by_category: dict[str, list[float]] = {}
    conf_query = (
        CropClassification.select(
            CropClassification.image_category,
            CropClassification.image_confidence_score,
            CropClassification.text_category,
            CropClassification.text_confidence_score,
        )
        .where(CropClassification.crop.in_(crop_filter))
        .tuples()
    )
    for img_cat, img_conf, txt_cat, txt_conf in conf_query:
        if img_cat and img_conf is not None:
            image_confidence_by_category.setdefault(img_cat, []).append(img_conf)
        if txt_cat and txt_conf is not None:
            text_confidence_by_category.setdefault(txt_cat, []).append(txt_conf)

    # Tokens by category
    tokens_by_final_category: list[tuple[str, int]] = []
    tok_cat_rows = db.execute_sql(
        "SELECT cc.final_category, COALESCE(SUM(ctc.vlm_token_count), 0) "
        "FROM crop_classification cc "
        "INNER JOIN crop_token_count ctc ON ctc.crop_id = cc.crop_id "
        f"WHERE cc.crop_id IN (SELECT crop_id FROM {crop_table}) "
        "AND cc.final_category IS NOT NULL "
        "GROUP BY cc.final_category "
        "ORDER BY 2 DESC"
    ).fetchall()
    tokens_by_final_category = [(cat, int(total)) for cat, total in tok_cat_rows]

    # Chronam totals (derive from filtered over-time data)
    chronam_term_counter: Counter = Counter()
    for year_terms in pre_chronam_terms_over_time.values():
        for term, count in year_terms.items():
            chronam_term_counter[term] += count
    chronam_term_totals = chronam_term_counter.most_common()

    top_labels, label_confidence_scores = _load_crop_subject_data(crop_filter)
    ner_data = _load_crop_ner_data(crop_filter)

    # Issue/scan counts for pre-1931
    issue_count = db.execute_sql(
        "SELECT COUNT(DISTINCT i.id) FROM issue i "
        "INNER JOIN scan s ON s.issue_id = i.id "
        "INNER JOIN crop c ON c.scan_id = s.id "
        f"INNER JOIN {crop_table} dc ON dc.crop_id = c.id"
    ).fetchone()[0]
    scan_count = db.execute_sql(
        "SELECT COUNT(DISTINCT s.id) FROM scan s "
        "INNER JOIN crop c ON c.scan_id = s.id "
        f"INNER JOIN {crop_table} dc ON dc.crop_id = c.id"
    ).fetchone()[0]

    return DashboardData(
        run=full_data.run,
        batches=full_data.batches,
        batch_statuses=full_data.batch_statuses,
        batch_durations=full_data.batch_durations,
        batch_items_per_batch=full_data.batch_items_per_batch,
        batch_crop_counts=full_data.batch_crop_counts,
        record_counts=record_counts,
        avg_batch_duration=full_data.avg_batch_duration,
        eta_seconds=full_data.eta_seconds,
        generated_at=full_data.generated_at,
        issue_count=issue_count,
        scan_count=scan_count,
        geography_counts=geography_counts,
        scan_resolutions_mp=scan_resolutions_mp,
        crop_confidence_scores=crop_confidence_scores,
        crops_per_scan=pre_crops_per_scan,
        crops_per_scan_by_year=pre_crops_per_scan_by_year,
        scan_coverage_pcts=pre_scan_coverage_pcts,
        scan_coverage_by_year=pre_scan_coverage_by_year,
        crops_by_language=crops_by_language,
        tokens_by_language=tokens_by_language,
        crop_text_stats=crop_text_stats,
        tokens_over_time=pre_tokens_over_time,
        tokens_per_scan_over_time=pre_tokens_per_scan_over_time,
        words_per_scan_over_time=pre_words_per_scan_over_time,
        wttr_by_language=wttr_by_language,
        crop_class_stats=crop_class_stats,
        crops_by_final_category=crops_by_final_category,
        category_over_time=pre_category_over_time,
        category_area_over_time=pre_category_area_over_time,
        category_individual_area_over_time=pre_category_individual_area_over_time,
        image_confidence_by_category=image_confidence_by_category,
        text_confidence_by_category=text_confidence_by_category,
        tokens_by_final_category=tokens_by_final_category,
        tokens_per_category_over_time=pre_tokens_per_category_over_time,
        top_labels=top_labels,
        label_confidence_scores=label_confidence_scores,
        ner_per_top=ner_data["per_top"],
        ner_loc_top=ner_data["loc_top"],
        ner_org_top=ner_data["org_top"],
        ner_type_totals=ner_data["type_totals"],
        ner_per_scores=ner_data["per_scores"],
        ner_loc_scores=ner_data["loc_scores"],
        ner_org_scores=ner_data["org_scores"],
        chronam_term_totals=chronam_term_totals,
        chronam_terms_over_time=pre_chronam_terms_over_time,
    )


def _load_record_counts(
    db: peewee.Database, crop_table: str = "_dash_crops"
) -> OrderedDict[str, int]:
    """Fetches all record counts in a single UNION ALL query."""
    tables = [
        ("Crop", "crop"),
        ("CropOCR", "crop_ocr"),
        ("CropClassification", "crop_classification"),
        ("CropSubject", "crop_subject"),
        ("CropNER", "crop_ner"),
        ("CropTextAnalysis", "crop_text_analysis"),
        ("CropTokenCount", "crop_token_count"),
        ("CropLanguage", "crop_language"),
        ("CropChronamThesauriMatch", "crop_chronam_thesauri_match"),
        ("CropTextStaticEmbedding", "crop_text_static_embedding"),
        ("CropImageEmbedding", "crop_image_embedding"),
    ]
    parts = []
    for label, table in tables:
        col = "id" if table == "crop" else "crop_id"
        parts.append(
            f"SELECT '{label}' AS name, COUNT(*) AS cnt "
            f"FROM {table} WHERE {col} IN "
            f"(SELECT crop_id FROM {crop_table})"
        )
    sql = " UNION ALL ".join(parts)
    cursor = db.execute_sql(sql)
    counts: OrderedDict[str, int] = OrderedDict()
    for name, cnt in cursor.fetchall():
        counts[name] = cnt
    return counts


def _compute_sql_median(
    db: peewee.Database,
    table: str,
    column: str,
    crop_table: str = "_dash_crops",
) -> float | None:
    """Computes the exact median of a column using SQL ORDER BY + OFFSET."""
    count_row = db.execute_sql(
        f"SELECT COUNT(*) FROM {table} "
        f"WHERE {column} IS NOT NULL "
        f"AND crop_id IN (SELECT crop_id FROM {crop_table})"
    ).fetchone()
    count = count_row[0]
    if count == 0:
        return None
    offset = (count - 1) // 2
    if count % 2 == 1:
        row = db.execute_sql(
            f"SELECT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL "
            f"AND crop_id IN (SELECT crop_id FROM {crop_table}) "
            f"ORDER BY {column} LIMIT 1 OFFSET {offset}"
        ).fetchone()
        return row[0]
    rows = db.execute_sql(
        f"SELECT {column} FROM {table} "
        f"WHERE {column} IS NOT NULL "
        f"AND crop_id IN (SELECT crop_id FROM {crop_table}) "
        f"ORDER BY {column} LIMIT 2 OFFSET {offset}"
    ).fetchall()
    return (rows[0][0] + rows[1][0]) / 2.0


def _load_crop_text_stats(
    db: peewee.Database, crop_filter: peewee.SQL = _CROP_IN_RUN, crop_table: str = "_dash_crops"
) -> CropTextStats | None:
    """Loads aggregate text statistics for crops matching the given filter."""
    agg = (
        CropTokenCount.select(
            peewee.fn.SUM(CropTokenCount.tesseract_token_count).alias("tess_tok"),
            peewee.fn.SUM(CropTokenCount.vlm_token_count).alias("vlm_tok"),
        )
        .where(CropTokenCount.crop.in_(crop_filter))
        .dicts()
        .first()
    )
    if not agg or agg["tess_tok"] is None:
        return None

    text_agg = (
        CropTextAnalysis.select(
            peewee.fn.SUM(CropTextAnalysis.tesseract_word_count).alias("tess_w"),
            peewee.fn.SUM(CropTextAnalysis.vlm_word_count).alias("vlm_w"),
            peewee.fn.SUM(CropTextAnalysis.tesseract_sentence_count).alias("tess_s"),
            peewee.fn.SUM(CropTextAnalysis.vlm_sentence_count).alias("vlm_s"),
            peewee.fn.AVG(CropTextAnalysis.tesseract_word_type_token_ratio).alias("tess_avg_wttr"),
            peewee.fn.AVG(CropTextAnalysis.vlm_word_type_token_ratio).alias("vlm_avg_wttr"),
            peewee.fn.SUM(
                peewee.Case(
                    None,
                    [(CropTextAnalysis.vlm_has_table == True, 1)],  # noqa: E712
                    0,
                )
            ).alias("vlm_tables"),
            peewee.fn.SUM(
                peewee.Case(
                    None,
                    [(CropTextAnalysis.vlm_has_markdown == True, 1)],  # noqa: E712
                    0,
                )
            ).alias("vlm_markdown"),
        )
        .where(CropTextAnalysis.crop.in_(crop_filter))
        .dicts()
        .first()
    )

    tess_median = _compute_sql_median(
        db, "crop_text_analysis", "tesseract_word_type_token_ratio", crop_table
    )
    vlm_median = _compute_sql_median(
        db, "crop_text_analysis", "vlm_word_type_token_ratio", crop_table
    )

    return CropTextStats(
        tesseract_total_tokens=int(agg["tess_tok"] or 0),
        vlm_total_tokens=int(agg["vlm_tok"] or 0),
        tesseract_total_words=int(text_agg["tess_w"] or 0),
        vlm_total_words=int(text_agg["vlm_w"] or 0),
        tesseract_total_sentences=int(text_agg["tess_s"] or 0),
        vlm_total_sentences=int(text_agg["vlm_s"] or 0),
        tesseract_avg_wttr=text_agg["tess_avg_wttr"],
        vlm_avg_wttr=text_agg["vlm_avg_wttr"],
        tesseract_median_wttr=tess_median,
        vlm_median_wttr=vlm_median,
        vlm_crops_with_tables=int(text_agg["vlm_tables"] or 0),
        vlm_crops_with_markdown=int(text_agg["vlm_markdown"] or 0),
    )


def _load_crop_classification_stats(
    crop_filter: peewee.SQL = _CROP_IN_RUN,
) -> tuple[CropClassStats | None, list[tuple[str, int]]]:
    """Loads classification agreement statistics and category distribution."""
    stats = (
        CropClassification.select(
            peewee.fn.COUNT(CropClassification.crop).alias("total"),
            peewee.fn.SUM(
                peewee.Case(
                    None,
                    [
                        (
                            CropClassification.image_category == CropClassification.text_category,
                            1,
                        )
                    ],
                    0,
                )
            ).alias("agreement"),
            peewee.fn.SUM(
                peewee.Case(
                    None,
                    [
                        (
                            CropClassification.final_category == CropClassification.image_category,
                            1,
                        )
                    ],
                    0,
                )
            ).alias("final_eq_image"),
            peewee.fn.SUM(
                peewee.Case(
                    None,
                    [
                        (
                            CropClassification.final_category == CropClassification.text_category,
                            1,
                        )
                    ],
                    0,
                )
            ).alias("final_eq_text"),
        )
        .where(CropClassification.crop.in_(crop_filter))
        .dicts()
        .first()
    )
    if not stats or not stats["total"]:
        return None, []

    class_stats = CropClassStats(
        agreement_count=int(stats["agreement"] or 0),
        final_eq_image_count=int(stats["final_eq_image"] or 0),
        final_eq_text_count=int(stats["final_eq_text"] or 0),
        total=int(stats["total"] or 0),
    )

    cat_query = (
        CropClassification.select(
            CropClassification.final_category,
            peewee.fn.COUNT(CropClassification.crop).alias("n"),
        )
        .where(CropClassification.crop.in_(crop_filter))
        .where(CropClassification.final_category.is_null(False))
        .group_by(CropClassification.final_category)
        .order_by(peewee.fn.COUNT(CropClassification.crop).desc())
    )
    crops_by_cat = [(row.final_category, row.n) for row in cat_query]

    return class_stats, crops_by_cat


def _load_crop_subject_data(
    crop_filter: peewee.SQL = _CROP_IN_RUN,
) -> tuple[list[tuple[str, int]], dict[str, list[float]]]:
    """Loads top subject labels and their confidence scores."""
    db = utils.get_db()
    label_counter: Counter = Counter()
    label_scores: dict[str, list[float]] = {}

    cursor = db.execute_sql(
        "SELECT json_extract(ranked_labels, '$[0]'), json_extract(scores, '$[0]') "
        f"FROM crop_subject WHERE crop_id IN {crop_filter.sql}"
    )

    _FETCH_SIZE = 50_000
    while True:
        rows = cursor.fetchmany(_FETCH_SIZE)
        if not rows:
            break
        for top_label, top_score in rows:
            if top_label and top_score is not None:
                label_counter[top_label] += 1
                label_scores.setdefault(top_label, []).append(top_score)

    top_labels = label_counter.most_common(30)
    top_label_names = {name for name, _ in top_labels}
    filtered_scores = {k: v for k, v in label_scores.items() if k in top_label_names}

    return top_labels, filtered_scores


def _load_crop_ner_data(crop_filter: peewee.SQL = _CROP_IN_RUN) -> dict:
    """Loads NER entity counts, top entities, and confidence scores."""
    db = utils.get_db()
    _FETCH_SIZE = 50_000

    per_counter: Counter = Counter()
    loc_counter: Counter = Counter()
    org_counter: Counter = Counter()
    per_scores: list[float] = []
    loc_scores: list[float] = []
    org_scores: list[float] = []

    cursor = db.execute_sql(
        "SELECT per_entities, per_confidence_scores, "
        "loc_entities, loc_confidence_scores, "
        "org_entities, org_confidence_scores "
        f"FROM crop_ner WHERE crop_id IN {crop_filter.sql}"
    )

    while True:
        rows = cursor.fetchmany(_FETCH_SIZE)
        if not rows:
            break
        for per_e, per_s, loc_e, loc_s, org_e, org_s in rows:
            if per_e:
                per_counter.update(json.loads(per_e))
            if per_s and len(per_scores) < MAX_SCORES:
                per_scores.extend(json.loads(per_s))
            if loc_e:
                loc_counter.update(json.loads(loc_e))
            if loc_s and len(loc_scores) < MAX_SCORES:
                loc_scores.extend(json.loads(loc_s))
            if org_e:
                org_counter.update(json.loads(org_e))
            if org_s and len(org_scores) < MAX_SCORES:
                org_scores.extend(json.loads(org_s))

    if len(per_scores) > MAX_SCORES:
        per_scores = random.sample(per_scores, MAX_SCORES)
    if len(loc_scores) > MAX_SCORES:
        loc_scores = random.sample(loc_scores, MAX_SCORES)
    if len(org_scores) > MAX_SCORES:
        org_scores = random.sample(org_scores, MAX_SCORES)

    return {
        "per_top": per_counter.most_common(20),
        "loc_top": loc_counter.most_common(20),
        "org_top": org_counter.most_common(20),
        "type_totals": {
            "PER": sum(per_counter.values()),
            "LOC": sum(loc_counter.values()),
            "ORG": sum(org_counter.values()),
        },
        "per_scores": per_scores,
        "loc_scores": loc_scores,
        "org_scores": org_scores,
    }


def _load_chronam_data(
    db: peewee.Database,
    crop_filter: peewee.SQL = _CROP_IN_RUN,
    crop_table: str = "_dash_crops",
) -> tuple[list[tuple[str, int]], dict[int, dict[str, int]]]:
    """Loads Chronam thesauri match data using the pre-populated year temp table."""
    _FETCH_SIZE = 10_000
    term_counter: Counter = Counter()
    terms_over_time: dict[int, Counter] = {}

    cursor = db.execute_sql(
        "SELECT cctm.vlm_matches, dcy.year "
        "FROM crop_chronam_thesauri_match cctm "
        f"INNER JOIN {crop_table} dc ON dc.crop_id = cctm.crop_id "
        "INNER JOIN _dash_crops_year dcy ON dcy.crop_id = cctm.crop_id "
        "WHERE cctm.vlm_matches IS NOT NULL"
    )

    while True:
        rows = cursor.fetchmany(_FETCH_SIZE)
        if not rows:
            break
        for vlm_matches_raw, year in rows:
            matches = (
                json.loads(vlm_matches_raw)
                if isinstance(vlm_matches_raw, str)
                else vlm_matches_raw
            )
            if not matches:
                continue
            for category_terms in matches.values():
                if not isinstance(category_terms, dict):
                    continue
                for term, count in category_terms.items():
                    term_counter[term] += count
                    if year is not None:
                        terms_over_time.setdefault(year, Counter())[term] += count

    term_totals = term_counter.most_common()
    terms_by_year = {
        year: dict(counter.most_common())
        for year, counter in sorted(terms_over_time.items())
    }

    return term_totals, terms_by_year


def _get_batch_status(batch: PipelineBatch) -> str:
    """Determines the current status of a pipeline batch."""
    if batch.has_crashed:
        return "crashed"
    if batch.started_date and batch.ended_date:
        return "completed"
    if batch.started_date and not batch.ended_date:
        return "running"
    return "pending"


def _format_seconds(seconds: float) -> str:
    """Formats a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _format_datetime(dt: datetime | None) -> str:
    """Formats a datetime as an ISO-style timestamp string."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _escape_html(text: str) -> str:
    """Escapes a string for safe inclusion in HTML."""
    return escape(str(text))


def _format_number(n: int | float) -> str:
    """Formats a number with thousands separators."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def _render_status_badge(status: str) -> str:
    """Renders an HTML badge for a batch status."""
    css_class = STATUS_CSS.get(status, "badge-pending")
    return f'<span class="badge {css_class}">{_escape_html(status.upper())}</span>'


def _get_overall_status(data: DashboardData) -> str:
    """Computes the overall pipeline run status from individual batch statuses."""
    statuses = set(data.batch_statuses.values())
    if "crashed" in statuses:
        return "crashed"
    if "running" in statuses:
        return "running"
    if all(s == "completed" for s in data.batch_statuses.values()):
        return "completed"
    if all(s == "pending" for s in data.batch_statuses.values()):
        return "pending"
    return "running"


def _compute_safe_stdev(values: list[float]) -> float:
    """Computes standard deviation, returning 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return stdev(values)


def _wrap_in_carousel(slides_html: list[str]) -> str:
    """Wraps a list of HTML content strings into a carousel with prev/next navigation."""
    if not slides_html:
        return ""
    if len(slides_html) == 1:
        return slides_html[0]
    slides = "\n".join(f'<div class="carousel-slide">{s}</div>' for s in slides_html)
    return f"""
    <div class="carousel">
        <div class="carousel-track">{slides}</div>
        <div class="carousel-nav">
            <button class="carousel-btn carousel-prev">&larr; Prev</button>
            <span class="carousel-indicator">1 / {len(slides_html)}</span>
            <button class="carousel-btn carousel-next">Next &rarr;</button>
        </div>
    </div>"""


def _svg_histogram(
    values: list[float],
    xlabel: str = "",
    ylabel: str = "Count",
    width: int = 800,
    height: int = 300,
    color: str = "#2563eb",
) -> str:
    """Renders a histogram as an inline SVG element."""
    if not values:
        return '<p class="no-data">No data available.</p>'

    n = len(values)
    num_bins = max(1, ceil(log2(n) + 1)) if n > 1 else 1
    num_bins = min(num_bins, 50)

    v_min, v_max = min(values), max(values)
    if v_min == v_max:
        v_max = v_min + 1

    bin_width = (v_max - v_min) / num_bins
    bins = [0] * num_bins
    for v in values:
        idx = min(int((v - v_min) / bin_width), num_bins - 1)
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    margin = {"top": 20, "right": 20, "bottom": 50, "left": 60}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]
    bar_w = chart_w / num_bins

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;max-width:{width}px;font-family:system-ui,sans-serif;">'
    ]

    # Bars
    for i, count in enumerate(bins):
        bar_h = (count / max_count * chart_h) if max_count > 0 else 0
        x = margin["left"] + i * bar_w
        y = margin["top"] + chart_h - bar_h
        lo = v_min + i * bin_width
        hi = v_min + (i + 1) * bin_width
        tip = f"{lo:.3f} – {hi:.3f}: {count:,}"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" opacity="0.85" '
            f'class="bar-interactive" data-tooltip="{_escape_html(tip)}">'
            f"<title>{tip}</title></rect>"
        )

    # Y-axis ticks
    for i in range(5):
        y = margin["top"] + chart_h - (i / 4) * chart_h
        val = int(max_count * i / 4)
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{val:,}</text>'
        )
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y:.1f}" '
            f'x2="{margin["left"] + chart_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-dasharray="3,3" />'
        )

    # X-axis labels
    for i in range(0, num_bins + 1, max(1, num_bins // 5)):
        x = margin["left"] + i * bar_w
        val = v_min + i * bin_width
        parts.append(
            f'<text x="{x:.1f}" y="{margin["top"] + chart_h + 20}" '
            f'text-anchor="middle" font-size="11" fill="#6b7280">{val:.2f}</text>'
        )

    # Axis labels
    parts.append(
        f'<text x="{width / 2}" y="{height - 5}" text-anchor="middle" '
        f'font-size="12" fill="#374151">{_escape_html(xlabel)}</text>'
    )
    parts.append(
        f'<text x="10" y="{height / 2}" text-anchor="middle" '
        f'font-size="12" fill="#374151" transform="rotate(-90,10,{height / 2})">'
        f"{_escape_html(ylabel)}</text>"
    )

    # Axes
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{margin["top"] + chart_h}" stroke="#9ca3af" />'
    )
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"] + chart_h}" '
        f'x2="{margin["left"] + chart_w}" y2="{margin["top"] + chart_h}" '
        f'stroke="#9ca3af" />'
    )

    parts.append("</svg>")
    return "\n".join(parts)


@dataclass
class LineSeries:
    label: str
    x_labels: list[str]
    values: list[float]
    color: str = "#2563eb"
    std_devs: list[float] | None = None


def _svg_line_chart(
    series_list: list[LineSeries],
    xlabel: str = "",
    ylabel: str = "",
    width: int = 800,
    height: int = 300,
) -> str:
    """Renders a multi-series line chart as an inline SVG element."""
    if not series_list or not series_list[0].values:
        return '<p class="no-data">No data available.</p>'

    all_vals = [v for s in series_list for v in s.values]
    if series_list[0].std_devs:
        all_vals += [
            v + sd for s in series_list if s.std_devs for v, sd in zip(s.values, s.std_devs)
        ]
    v_min = min(all_vals) if all_vals else 0
    v_max = max(all_vals) if all_vals else 1
    if v_min == v_max:
        v_max = v_min + 1
    v_min = min(v_min, 0)

    margin = {"top": 20, "right": 120, "bottom": 50, "left": 75}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    x_labels = series_list[0].x_labels
    n_points = len(x_labels)
    if n_points < 2:
        x_step = chart_w
    else:
        x_step = chart_w / (n_points - 1)

    def x_pos(i: int) -> float:
        return margin["left"] + i * x_step

    def y_pos(v: float) -> float:
        return margin["top"] + chart_h - (v - v_min) / (v_max - v_min) * chart_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;max-width:{width}px;font-family:system-ui,sans-serif;">'
    ]

    # Grid lines
    for i in range(5):
        y = margin["top"] + chart_h - (i / 4) * chart_h
        val = v_min + (v_max - v_min) * i / 4
        label = f"{val:,.0f}" if abs(val) >= 1 else f"{val:.2f}"
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y:.1f}" '
            f'x2="{margin["left"] + chart_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-dasharray="3,3" />'
        )
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{label}</text>'
        )

    # X-axis labels
    max_labels = 12
    step = max(1, n_points // max_labels)
    for i in range(0, n_points, step):
        parts.append(
            f'<text x="{x_pos(i):.1f}" y="{margin["top"] + chart_h + 20}" '
            f'text-anchor="middle" font-size="11" fill="#6b7280">'
            f"{_escape_html(x_labels[i])}</text>"
        )

    # Std dev band (if present)
    for idx, s in enumerate(series_list):
        if s.std_devs:
            upper_points = [
                f"{x_pos(i):.1f},{y_pos(s.values[i] + s.std_devs[i]):.1f}" for i in range(n_points)
            ]
            lower_points = [
                f"{x_pos(i):.1f},{y_pos(max(0, s.values[i] - s.std_devs[i])):.1f}"
                for i in range(n_points - 1, -1, -1)
            ]
            polygon = " ".join(upper_points + lower_points)
            parts.append(
                f'<polygon points="{polygon}" fill="{s.color}" opacity="0.12" '
                f'class="band-interactive" data-series="{idx}" />'
            )

    # Lines
    for idx, s in enumerate(series_list):
        points = " ".join(f"{x_pos(i):.1f},{y_pos(s.values[i]):.1f}" for i in range(n_points))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{s.color}" '
            f'stroke-width="2" class="line-interactive" data-series="{idx}" />'
        )
        for i in range(n_points):
            tip = f"{_escape_html(s.label)} | {_escape_html(x_labels[i])}: {s.values[i]:,.1f}"
            parts.append(
                f'<circle cx="{x_pos(i):.1f}" cy="{y_pos(s.values[i]):.1f}" '
                f'r="3" fill="{s.color}" class="point-interactive" '
                f'data-series="{idx}" data-tooltip="{tip}">'
                f"<title>{tip}</title></circle>"
            )

    # Legend
    for idx, s in enumerate(series_list):
        ly = margin["top"] + 15 + idx * 20
        lx = margin["left"] + chart_w + 10
        parts.append(
            f'<g class="legend-item" data-series="{idx}">'
            f'<rect x="{lx}" y="{ly - 6}" width="12" height="12" '
            f'fill="{s.color}" rx="2" />'
            f'<text x="{lx + 16}" y="{ly + 4}" font-size="11" '
            f'fill="#374151">{_escape_html(s.label)}</text></g>'
        )

    # Axis labels
    if xlabel:
        parts.append(
            f'<text x="{width / 2}" y="{height - 5}" text-anchor="middle" '
            f'font-size="12" fill="#374151">{_escape_html(xlabel)}</text>'
        )
    if ylabel:
        parts.append(
            f'<text x="10" y="{height / 2}" text-anchor="middle" font-size="12" '
            f'fill="#374151" transform="rotate(-90,10,{height / 2})">'
            f"{_escape_html(ylabel)}</text>"
        )

    # Axes
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{margin["top"] + chart_h}" stroke="#9ca3af" />'
    )
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"] + chart_h}" '
        f'x2="{margin["left"] + chart_w}" y2="{margin["top"] + chart_h}" '
        f'stroke="#9ca3af" />'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _svg_stacked_area(
    categories: list[str],
    year_data: dict[int, dict[str, int | float]],
    width: int = 800,
    height: int = 300,
) -> str:
    """Renders a normalized stacked area chart as an inline SVG element."""
    if not year_data:
        return '<p class="no-data">No data available.</p>'

    years = sorted(year_data.keys())
    n = len(years)
    if n < 2:
        return '<p class="no-data">Not enough data points for area chart.</p>'

    # Normalize each year to 100%
    normalized: list[dict[str, float]] = []
    for year in years:
        total = sum(year_data[year].values())
        if total > 0:
            row = {cat: year_data[year].get(cat, 0) / total * 100 for cat in categories}
            normalized.append(row)
        else:
            normalized.append({cat: 0.0 for cat in categories})

    margin = {"top": 20, "right": 160, "bottom": 50, "left": 60}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    def x_pos(i: int) -> float:
        return margin["left"] + i / (n - 1) * chart_w

    def y_pos(pct: float) -> float:
        return margin["top"] + chart_h - pct / 100 * chart_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;max-width:{width}px;font-family:system-ui,sans-serif;">'
    ]

    # Stack areas bottom-up
    for cat_idx in range(len(categories) - 1, -1, -1):
        cat = categories[cat_idx]
        color = CHART_COLORS[cat_idx % len(CHART_COLORS)]

        # Upper boundary: cumulative through cat_idx
        upper = []
        for i in range(n):
            cum = sum(normalized[i].get(categories[j], 0) for j in range(cat_idx + 1))
            upper.append((x_pos(i), y_pos(cum)))

        # Lower boundary: cumulative through cat_idx - 1
        lower = []
        for i in range(n - 1, -1, -1):
            cum = sum(normalized[i].get(categories[j], 0) for j in range(cat_idx))
            lower.append((x_pos(i), y_pos(cum)))

        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        tip = _escape_html(cat)
        parts.append(
            f'<polygon points="{points}" fill="{color}" opacity="0.75" '
            f'class="area-interactive" data-series="{cat_idx}" '
            f'data-tooltip="{tip}" />'
        )

    # X-axis labels
    max_labels = 12
    step = max(1, n // max_labels)
    for i in range(0, n, step):
        parts.append(
            f'<text x="{x_pos(i):.1f}" y="{margin["top"] + chart_h + 20}" '
            f'text-anchor="middle" font-size="11" fill="#6b7280">{years[i]}</text>'
        )

    # Y-axis labels
    for pct in (0, 25, 50, 75, 100):
        y = y_pos(pct)
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{pct}%</text>'
        )
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y:.1f}" '
            f'x2="{margin["left"] + chart_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-dasharray="3,3" />'
        )

    # Legend
    for idx, cat in enumerate(categories):
        ly = margin["top"] + 10 + idx * 18
        lx = margin["left"] + chart_w + 10
        color = CHART_COLORS[idx % len(CHART_COLORS)]
        label = cat[:25] + "..." if len(cat) > 28 else cat
        parts.append(
            f'<g class="legend-item" data-series="{idx}">'
            f'<rect x="{lx}" y="{ly - 5}" width="10" height="10" '
            f'fill="{color}" rx="2" />'
            f'<text x="{lx + 14}" y="{ly + 4}" font-size="10" '
            f'fill="#374151">{_escape_html(label)}</text></g>'
        )

    # Axes
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{margin["top"] + chart_h}" stroke="#9ca3af" />'
    )
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"] + chart_h}" '
        f'x2="{margin["left"] + chart_w}" y2="{margin["top"] + chart_h}" '
        f'stroke="#9ca3af" />'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _svg_horizontal_bars(
    items: list[tuple[str, int | float]],
    width: int = 800,
    bar_height: int = 26,
    color: str = "#2563eb",
    max_items: int = 30,
) -> str:
    """Renders a horizontal bar chart as an inline SVG element."""
    if not items:
        return '<p class="no-data">No data available.</p>'

    items = items[:max_items]
    max_val = max(v for _, v in items) if items else 1
    if max_val == 0:
        max_val = 1

    label_w = 260
    bar_gap = 4
    total_h = len(items) * (bar_height + bar_gap) + 10

    parts = [
        f'<svg viewBox="0 0 {width} {total_h}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;max-width:{width}px;font-family:system-ui,sans-serif;">'
    ]

    for i, (label, value) in enumerate(items):
        y = i * (bar_height + bar_gap)
        bar_w = (value / max_val) * (width - label_w - 20)
        display_label = label[:30] + "..." if len(str(label)) > 33 else str(label)

        parts.append(
            f'<text x="{label_w - 8}" y="{y + bar_height / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="12" fill="#374151">'
            f"{_escape_html(display_label)}</text>"
        )
        tip = f"{_escape_html(str(label))}: {_format_number(value)}"
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" '
            f'height="{bar_height}" fill="{color}" rx="3" opacity="0.85" '
            f'class="bar-interactive" data-tooltip="{tip}">'
            f"<title>{tip}</title></rect>"
        )
        if bar_w > 40:
            parts.append(
                f'<text x="{label_w + 8}" y="{y + bar_height / 2 + 4:.1f}" '
                f'font-size="11" fill="#fff" font-weight="500">'
                f"{_format_number(value)}</text>"
            )
        else:
            parts.append(
                f'<text x="{label_w + bar_w + 6}" y="{y + bar_height / 2 + 4:.1f}" '
                f'font-size="11" fill="#6b7280">{_format_number(value)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _render_leaflet_map(
    country_counts: dict[str, int],
    state_counts: dict[str, int],
    map_id: str = "geo-map",
) -> str:
    """Renders an interactive Leaflet.js choropleth map with global countries
    and an optional US states overlay."""
    if not country_counts and not state_counts:
        return '<p class="no-data">No geography data available.</p>'

    country_json = json.dumps(country_counts)
    state_json = json.dumps(state_counts)
    all_counts = list(country_counts.values()) + list(state_counts.values())
    max_count = max(all_counts) if all_counts else 1
    name_to_abbr = {name.title(): abbr for name, abbr in _STATE_ABBR_MAP.items()}
    name_to_abbr_json = json.dumps(name_to_abbr)

    return f"""
    <div id="{map_id}" class="leaflet-map"></div>
    <script>
    (function() {{
        var countryCounts = {country_json};
        var stateCounts = {state_json};
        var maxCount = {max_count};
        var nameToAbbr = {name_to_abbr_json};
        var el = document.getElementById('{map_id}');
        var map = L.map('{map_id}').setView([20, 0], 2);
        el._leafletMap = map;
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 8
        }}).addTo(map);
        function getColor(count, maxC) {{
            if (count === 0) return '#f3f4f6';
            var t = count / maxC;
            var r = Math.round(219 + (37 - 219) * t);
            var g = Math.round(234 + (99 - 234) * t);
            var b = Math.round(254 + (235 - 254) * t);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }}
        var countriesUrl =
            'https://raw.githubusercontent.com/datasets/geo-countries'
            + '/master/data/countries.geojson';
        var statesUrl =
            'https://raw.githubusercontent.com/PublicaMundi/MappingAPI'
            + '/master/data/geojson/us-states.json';
        var bounds = [];
        fetch(countriesUrl)
            .then(function(r) {{ return r.json(); }})
            .then(function(geojson) {{
                var countryMax = Math.max.apply(null,
                    Object.values(countryCounts).concat([1]));
                L.geoJSON(geojson, {{
                    style: function(feature) {{
                        var name = feature.properties.ADMIN || '';
                        var count = countryCounts[name] || 0;
                        return {{
                            fillColor: getColor(count, countryMax),
                            weight: 1, color: '#fff', fillOpacity: 0.75
                        }};
                    }},
                    onEachFeature: function(feature, layer) {{
                        var name = feature.properties.ADMIN || '';
                        var count = countryCounts[name] || 0;
                        layer.bindPopup(
                            '<b>' + name + '</b><br>'
                            + count.toLocaleString() + ' issues'
                        );
                        if (count > 0) bounds.push(layer.getBounds());
                    }}
                }}).addTo(map);
                if (Object.keys(stateCounts).length > 0) {{
                    fetch(statesUrl)
                        .then(function(r) {{ return r.json(); }})
                        .then(function(statesGeo) {{
                            var stateMax = Math.max.apply(null,
                                Object.values(stateCounts).concat([1]));
                            L.geoJSON(statesGeo, {{
                                style: function(feature) {{
                                    var name = feature.properties.name || '';
                                    var abbr = nameToAbbr[name] || '';
                                    var count = stateCounts[abbr] || 0;
                                    return {{
                                        fillColor: getColor(count, stateMax),
                                        weight: 1, color: '#ccc',
                                        fillOpacity: 0.8
                                    }};
                                }},
                                onEachFeature: function(feature, layer) {{
                                    var name = feature.properties.name || '';
                                    var abbr = nameToAbbr[name] || '';
                                    var count = stateCounts[abbr] || 0;
                                    layer.bindPopup(
                                        '<b>' + name + '</b><br>'
                                        + count.toLocaleString() + ' issues'
                                    );
                                }}
                            }}).addTo(map);
                            if (bounds.length > 0) {{
                                var combined = bounds[0];
                                for (var i = 1; i < bounds.length; i++)
                                    combined.extend(bounds[i]);
                                map.fitBounds(combined, {{padding: [20, 20]}});
                            }}
                        }});
                }} else if (bounds.length > 0) {{
                    var combined = bounds[0];
                    for (var i = 1; i < bounds.length; i++)
                        combined.extend(bounds[i]);
                    map.fitBounds(combined, {{padding: [20, 20]}});
                }}
            }});
    }})();
    </script>"""


def _render_global_filter_toggle() -> str:
    """Renders the global All issues / Pre-1931 issues toggle."""
    return (
        '<div class="global-filter">'
        '<button class="toggle-btn active" data-filter="all">All issues</button>'
        '<button class="toggle-btn" data-filter="pre1931">Pre-1931 issues</button>'
        "</div>"
    )


def _render_filtered_section(
    heading: str,
    all_content: str,
    pre1931_content: str | None,
    open_by_default: bool = False,
) -> str:
    """Wraps section content in a <details> with dual filter panels when pre-1931 data exists."""
    open_attr = " open" if open_by_default else ""
    if not pre1931_content:
        return (
            f'<details class="section"{open_attr}>'
            f"<summary><h2>{heading}</h2></summary>"
            f"{all_content}"
            f"</details>"
        )
    return (
        f'<details class="section"{open_attr}>'
        f"<summary><h2>{heading}</h2></summary>"
        f'<div class="filter-panel" data-filter="all">{all_content}</div>'
        f'<div class="filter-panel" data-filter="pre1931" style="display:none;">'
        f"{pre1931_content}</div>"
        f"</details>"
    )


def _render_header(data: DashboardData) -> str:
    """Renders the dashboard header with status badge and timestamp."""
    status = _get_overall_status(data)
    gen_time = _format_datetime(data.generated_at)
    return f"""
    <div class="header">
        <h1>Pipeline Run #{data.run.id} {_render_status_badge(status)}</h1>
        <p class="generated-at">Generated {gen_time} UTC</p>
    </div>"""


def _render_overview(data: DashboardData) -> str:
    """Renders the run overview section with progress bar and metadata."""
    status_counts: dict[str, int] = {}
    for s in data.batch_statuses.values():
        status_counts[s] = status_counts.get(s, 0) + 1

    status_cells = ""
    for s in ("completed", "running", "pending", "crashed"):
        count = status_counts.get(s, 0)
        if count > 0:
            status_cells += f" {_render_status_badge(s)} {count}"

    avg_display = _format_seconds(data.avg_batch_duration) if data.avg_batch_duration else "—"
    if data.eta_seconds is not None:
        eta_display = _format_seconds(data.eta_seconds)
    elif data.avg_batch_duration and all(s == "completed" for s in data.batch_statuses.values()):
        eta_display = "Done"
    else:
        eta_display = "Insufficient data"

    created = _format_datetime(data.run.created_date)
    total_items = sum(data.batch_items_per_batch.values())

    # Progress bar
    total_batches = len(data.batches)
    progress_svg = ""
    if total_batches > 0:
        bar_w = 800
        bar_h = 36
        segments = [
            ("completed", "#16a34a"),
            ("running", "#ca8a04"),
            ("pending", "#9ca3af"),
            ("crashed", "#dc2626"),
        ]
        svg_parts = ""
        x = 0.0
        for seg_status, fill in segments:
            seg_count = status_counts.get(seg_status, 0)
            if seg_count == 0:
                continue
            w = seg_count / total_batches * bar_w
            pct_label = f"{seg_count / total_batches * 100:.0f}%"
            svg_parts += (
                f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{bar_h}" '
                f'fill="{fill}" rx="3" />'
            )
            if w > 40:
                svg_parts += (
                    f'<text x="{x + w / 2:.1f}" y="{bar_h / 2 + 5}" '
                    f'text-anchor="middle" font-size="13" font-weight="600" fill="#fff">'
                    f"{seg_count} ({pct_label})</text>"
                )
            x += w

        completed = status_counts.get("completed", 0)
        pct = completed / total_batches * 100
        progress_svg = f"""
        <div style="padding: 16px 16px 0 16px;">
            <svg viewBox="0 0 {bar_w} {bar_h}" xmlns="http://www.w3.org/2000/svg"
                 style="width:100%; max-width:{bar_w}px; border-radius:6px; overflow:hidden;">
                {svg_parts}
            </svg>
            <p class="progress-text">{completed} of {total_batches} batches completed ({pct:.1f}%)</p>
        </div>"""

    return f"""
    <details class="section" open>
        <summary><h2>Run Overview</h2></summary>
        <table class="summary-table">
            <tr><td>Corpus</td><td>{_escape_html(data.run.corpus)}</td></tr>
            <tr><td>Created</td><td>{_escape_html(created)}</td></tr>
            <tr><td>Total Items</td><td>{total_items:,}</td></tr>
            <tr><td>Items Per Batch</td><td>{data.run.items_per_batch:,}</td></tr>
            <tr><td>Total Batches</td><td>{data.run.batches_total:,}</td></tr>
            <tr><td>Batch Status</td><td>{status_cells}</td></tr>
            <tr><td>Avg Batch Duration</td><td>{_escape_html(avg_display)}</td></tr>
            <tr><td>Est. Time to Completion</td><td>{_escape_html(eta_display)}</td></tr>
        </table>
        {progress_svg}
    </details>"""


def _render_batch_table(data: DashboardData) -> str:
    """Renders the batch details table."""
    rows = ""
    for b in data.batches:
        status = data.batch_statuses[b.id]
        duration = data.batch_durations.get(b.id)
        duration_display = _format_seconds(duration) if duration is not None else "—"
        items = data.batch_items_per_batch.get(b.id, 0)
        crops = data.batch_crop_counts.get(b.id, 0)
        node = b.node_name or "—"
        rows += f"""
            <tr>
                <td>#{b.id}</td>
                <td>{_escape_html(node)}</td>
                <td>{_render_status_badge(status)}</td>
                <td>{_format_datetime(b.started_date)}</td>
                <td>{_format_datetime(b.ended_date)}</td>
                <td>{_escape_html(duration_display)}</td>
                <td>{items:,}</td>
                <td>{crops:,}</td>
            </tr>"""

    return f"""
    <details class="section">
        <summary><h2>Batch Details</h2></summary>
        <div class="batch-table-container">
            <table class="batch-table">
                <thead>
                    <tr>
                        <th>Batch</th><th>Node</th><th>Status</th>
                        <th>Started</th><th>Ended</th><th>Duration</th>
                        <th>Items</th><th>Crops</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </details>"""


def _render_record_counts(data: DashboardData) -> str:
    """Renders the record counts section with table and bar chart."""
    if not data.record_counts:
        return ""
    all_content = _render_record_counts_content(data)
    pre1931_content = _render_record_counts_content(data.pre1931) if data.pre1931 else None
    return _render_filtered_section("Record Counts", all_content, pre1931_content)


def _render_record_counts_content(data: DashboardData) -> str:
    """Renders inner content for the record counts section."""
    if not data.record_counts:
        return '<p class="no-data">No record count data available.</p>'

    crop_total = data.record_counts.get("Crop", 0)

    rows = ""
    rows += f"""
        <tr>
            <td class="record-name">Issue</td>
            <td>{data.issue_count:,}</td>
            <td>—</td>
        </tr>
        <tr>
            <td class="record-name">Scan</td>
            <td>{data.scan_count:,}</td>
            <td>—</td>
        </tr>"""

    for name, count in data.record_counts.items():
        pct = (count / crop_total * 100) if crop_total > 0 else 0
        pct_display = f"{pct:.1f}%" if name != "Crop" else "—"
        rows += f"""
            <tr>
                <td class="record-name">{_escape_html(name)}</td>
                <td>{count:,}</td>
                <td>{pct_display}</td>
            </tr>"""

    all_items = [("Issue", data.issue_count), ("Scan", data.scan_count)] + list(
        data.record_counts.items()
    )
    max_count = max(v for _, v in all_items) if all_items else 1
    bars = ""
    for name, count in all_items:
        width_pct = (count / max_count * 100) if max_count > 0 else 0
        color = "#2563eb" if name in ("Crop", "Issue", "Scan") else "#60a5fa"
        bars += f"""
            <div class="chart-row">
                <div class="chart-label">{_escape_html(name)}</div>
                <div class="chart-bar-container">
                    <div class="chart-bar" style="width: {width_pct:.1f}%; background: {color}">
                        <span class="chart-value">{count:,}</span>
                    </div>
                </div>
            </div>"""

    return (
        f'<table class="record-table">'
        f"<thead><tr><th>Model</th><th>Count</th><th>% of Crops</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f'<div class="chart" style="margin-top: 12px;">{bars}</div>'
    )


def _render_geography(data: DashboardData) -> str:
    """Renders the geography section with global map and location table."""
    if not data.geography_counts:
        return ""
    all_content = _render_geography_content(data, map_id="geo-map-all")
    pre1931_content = (
        _render_geography_content(data.pre1931, map_id="geo-map-pre1931")
        if data.pre1931 and data.pre1931.geography_counts
        else None
    )
    return _render_filtered_section("Issues Geography", all_content, pre1931_content)


def _render_geography_content(data: DashboardData, map_id: str = "geo-map") -> str:
    """Renders inner content for the geography section."""
    country_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for row in data.geography_counts:
        if row.country:
            normalized = _normalize_country_name(row.country)
            country_counts[normalized] = country_counts.get(normalized, 0) + row.count
        if row.state:
            state_abbr = _convert_state_name_to_abbr(row.state)
            if state_abbr:
                state_counts[state_abbr] = state_counts.get(state_abbr, 0) + row.count

    map_html = _render_leaflet_map(country_counts, state_counts, map_id=map_id)

    table_rows = ""
    for row in data.geography_counts[:50]:
        table_rows += (
            f"<tr>"
            f"<td>{_escape_html(row.city or '—')}</td>"
            f"<td>{_escape_html(row.state or '—')}</td>"
            f"<td>{_escape_html(row.country or '—')}</td>"
            f"<td>{row.count:,}</td>"
            f"</tr>"
        )

    return (
        f'<div class="map-container">{map_html}</div>'
        f'<div class="batch-table-container">'
        f'<table class="stats-table">'
        f"<thead><tr><th>City</th><th>State</th><th>Country</th><th>Count</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table></div>"
    )


def _render_scan_dimensions(data: DashboardData) -> str:
    """Renders the scan dimensions statistics section."""
    if not data.scan_resolutions_mp:
        return ""
    all_content = _render_scan_dimensions_content(data)
    pre1931_content = (
        _render_scan_dimensions_content(data.pre1931)
        if data.pre1931 and data.pre1931.scan_resolutions_mp
        else None
    )
    return _render_filtered_section("Scan Dimensions", all_content, pre1931_content)


def _render_scan_dimensions_content(data: DashboardData) -> str:
    """Renders inner content for the scan dimensions section."""
    vals = data.scan_resolutions_mp
    if not vals:
        return '<p class="no-data">No scan dimension data available.</p>'

    avg_mp = sum(vals) / len(vals)
    med_mp = median(vals)
    min_mp = min(vals)
    max_mp = max(vals)

    hist_svg = _svg_histogram(vals, xlabel="Resolution (Megapixels)", ylabel="Count")

    return (
        f'<table class="stats-table">'
        f"<tr><td>Total Scans</td><td>{len(vals):,}</td></tr>"
        f"<tr><td>Average Resolution</td><td>{avg_mp:.2f} MP</td></tr>"
        f"<tr><td>Median Resolution</td><td>{med_mp:.2f} MP</td></tr>"
        f"<tr><td>Min Resolution</td><td>{min_mp:.2f} MP</td></tr>"
        f"<tr><td>Max Resolution</td><td>{max_mp:.2f} MP</td></tr>"
        f"</table>"
        f'<div class="subsection-header">Resolution Distribution</div>'
        f'<div class="chart-container">{hist_svg}</div>'
    )


def _render_crop_properties(data: DashboardData) -> str:
    """Renders the crop properties section with confidence and crops-per-scan stats."""
    crop_total = data.record_counts.get("Crop", 0)
    if crop_total == 0:
        return ""
    all_content = _render_crop_properties_content(data)
    pre1931_content = (
        _render_crop_properties_content(data.pre1931)
        if data.pre1931 and data.pre1931.record_counts.get("Crop", 0) > 0
        else None
    )
    return _render_filtered_section("Crop Properties", all_content, pre1931_content)


def _render_crop_properties_content(data: DashboardData) -> str:
    """Renders inner content for the crop properties section."""
    crop_total = data.record_counts.get("Crop", 0)
    if crop_total == 0:
        return '<p class="no-data">No crop data available.</p>'

    cps = data.crops_per_scan
    conf = data.crop_confidence_scores

    cps_avg = sum(cps) / len(cps) if cps else 0
    cps_min = min(cps) if cps else 0
    cps_max = max(cps) if cps else 0
    conf_avg = sum(conf) / len(conf) if conf else 0
    conf_med = median(conf) if conf else 0
    conf_min = min(conf) if conf else 0
    conf_max = max(conf) if conf else 0
    conf_std = _compute_safe_stdev(conf)

    hist_svg = _svg_histogram(conf, xlabel="Confidence Score", ylabel="Count")

    years = sorted(data.crops_per_scan_by_year.keys())
    cps_time_svg = ""
    if years:
        x_labels = [str(y) for y in years]
        cps_data = data.crops_per_scan_by_year
        avgs = [sum(cps_data[y]) / len(cps_data[y]) for y in years]
        sds = [_compute_safe_stdev([float(x) for x in cps_data[y]]) for y in years]
        series = LineSeries(
            label="Avg crops/scan",
            x_labels=x_labels,
            values=avgs,
            std_devs=sds,
        )
        cps_time_svg = _svg_line_chart(
            [series], xlabel="Year", ylabel="Avg crops per scan"
        )

    coverage_stats_html = ""
    coverage_hist_svg = ""
    coverage_time_svg = ""
    cvg = data.scan_coverage_pcts
    if cvg:
        cvg_avg = sum(cvg) / len(cvg)
        cvg_med = median(cvg)
        cvg_min = min(cvg)
        cvg_max = max(cvg)
        coverage_stats_html = (
            f'<div class="subsection-header">Scan Coverage by Crop Bounding Boxes</div>'
            f'<table class="stats-table">'
            f"<tr><td>Avg Coverage</td><td>{cvg_avg:.2f}%</td></tr>"
            f"<tr><td>Median Coverage</td><td>{cvg_med:.2f}%</td></tr>"
            f"<tr><td>Min Coverage</td><td>{cvg_min:.2f}%</td></tr>"
            f"<tr><td>Max Coverage</td><td>{cvg_max:.2f}%</td></tr>"
            f"</table>"
        )
        coverage_hist_svg = _svg_histogram(cvg, xlabel="Scan Coverage (%)", ylabel="Count")

        cvg_years = sorted(data.scan_coverage_by_year.keys())
        if cvg_years:
            cvg_x = [str(y) for y in cvg_years]
            cvg_data = data.scan_coverage_by_year
            cvg_avgs = [sum(cvg_data[y]) / len(cvg_data[y]) for y in cvg_years]
            cvg_sds = [_compute_safe_stdev(cvg_data[y]) for y in cvg_years]
            cvg_series = LineSeries(
                label="Avg scan coverage %",
                x_labels=cvg_x,
                values=cvg_avgs,
                std_devs=cvg_sds,
            )
            coverage_time_svg = _svg_line_chart(
                [cvg_series], xlabel="Year", ylabel="Avg scan coverage (%)"
            )

    return (
        f'<table class="stats-table">'
        f"<tr><td>Total Crops</td><td>{crop_total:,}</td></tr>"
        f"<tr><td>Average Crops per Scan</td><td>{cps_avg:.2f}</td></tr>"
        f"<tr><td>Min Crops per Scan</td><td>{cps_min:,}</td></tr>"
        f"<tr><td>Max Crops per Scan</td><td>{cps_max:,}</td></tr>"
        f"<tr><td>Avg Confidence Score</td><td>{conf_avg:.4f}</td></tr>"
        f"<tr><td>Median Confidence Score</td><td>{conf_med:.4f}</td></tr>"
        f"<tr><td>Std Dev Confidence Score</td><td>{conf_std:.4f}</td></tr>"
        f"<tr><td>Min Confidence Score</td><td>{conf_min:.4f}</td></tr>"
        f"<tr><td>Max Confidence Score</td><td>{conf_max:.4f}</td></tr>"
        f"</table>"
        f'<div class="subsection-header">Confidence Score Distribution</div>'
        f'<div class="chart-container">{hist_svg}</div>'
        f'<div class="subsection-header">Average Crops per Scan Over Time</div>'
        f'<div class="chart-container">{cps_time_svg}</div>'
        f"{coverage_stats_html}"
        f'<div class="subsection-header">Scan Coverage Distribution</div>'
        f'<div class="chart-container">{coverage_hist_svg}</div>'
        f'<div class="subsection-header">Average Scan Coverage Over Time</div>'
        f'<div class="chart-container">{coverage_time_svg}</div>'
    )


def _render_crop_language(data: DashboardData) -> str:
    """Renders the crop language distribution section."""
    if not data.crops_by_language:
        return ""
    all_content = _render_crop_language_content(data)
    pre1931_content = (
        _render_crop_language_content(data.pre1931)
        if data.pre1931 and data.pre1931.crops_by_language
        else None
    )
    return _render_filtered_section("Crop Language", all_content, pre1931_content)


def _render_crop_language_content(data: DashboardData) -> str:
    """Renders inner content for the crop language section."""
    lang_rows = ""
    for lang, count in data.crops_by_language:
        lang_rows += f"<tr><td>{_escape_html(lang)}</td><td>{count:,}</td></tr>"

    tok_rows = ""
    for lang, total in data.tokens_by_language:
        tok_rows += f"<tr><td>{_escape_html(lang)}</td><td>{total:,}</td></tr>"

    return (
        f'<div class="section-grid"><div>'
        f'<div class="subsection-header">Total Crops by Language</div>'
        f'<div class="batch-table-container">'
        f'<table class="stats-table">'
        f"<thead><tr><th>Language</th><th>Crops</th></tr></thead>"
        f"<tbody>{lang_rows}</tbody></table></div></div><div>"
        f'<div class="subsection-header">Total Tokens by Language (VLM OCR)</div>'
        f'<div class="batch-table-container">'
        f'<table class="stats-table">'
        f"<thead><tr><th>Language</th><th>VLM Tokens</th></tr></thead>"
        f"<tbody>{tok_rows}</tbody></table></div></div></div>"
    )


def _render_crop_text(data: DashboardData) -> str:
    """Renders the crop text statistics section with time-series charts."""
    if not data.crop_text_stats:
        return ""
    all_content = _render_crop_text_content(data)
    pre1931_content = (
        _render_crop_text_content(data.pre1931)
        if data.pre1931 and data.pre1931.crop_text_stats
        else None
    )
    return _render_filtered_section("Crop Text", all_content, pre1931_content)


def _render_crop_text_content(data: DashboardData) -> str:
    """Renders inner content for the crop text section."""
    stats = data.crop_text_stats
    if not stats:
        return '<p class="no-data">No text data available.</p>'

    tess_med_val = stats.tesseract_median_wttr
    tess_med = f"{tess_med_val:.4f}" if tess_med_val is not None else "—"
    vlm_med = f"{stats.vlm_median_wttr:.4f}" if stats.vlm_median_wttr is not None else "—"
    tess_avg = f"{stats.tesseract_avg_wttr:.4f}" if stats.tesseract_avg_wttr is not None else "—"
    vlm_avg = f"{stats.vlm_avg_wttr:.4f}" if stats.vlm_avg_wttr is not None else "—"

    years = sorted(data.tokens_over_time.keys())
    tokens_time_svg = ""
    if years:
        x_labels = [str(y) for y in years]
        tess_vals = [data.tokens_over_time[y][0] for y in years]
        vlm_vals = [data.tokens_over_time[y][1] for y in years]
        tokens_time_svg = _svg_line_chart(
            [
                LineSeries(
                    label="Tesseract", x_labels=x_labels,
                    values=[float(v) for v in tess_vals], color="#2563eb",
                ),
                LineSeries(
                    label="VLM", x_labels=x_labels,
                    values=[float(v) for v in vlm_vals], color="#dc2626",
                ),
            ],
            xlabel="Year", ylabel="Total Tokens",
        )

    tok_scan_years = sorted(data.tokens_per_scan_over_time.keys())
    tok_scan_svg = ""
    if tok_scan_years:
        x_labels = [str(y) for y in tok_scan_years]
        tess_tps = [data.tokens_per_scan_over_time[y][0] for y in tok_scan_years]
        vlm_tps = [data.tokens_per_scan_over_time[y][1] for y in tok_scan_years]
        tok_scan_svg = _svg_line_chart(
            [
                LineSeries(label="Tesseract", x_labels=x_labels, values=tess_tps, color="#2563eb"),
                LineSeries(label="VLM", x_labels=x_labels, values=vlm_tps, color="#dc2626"),
            ],
            xlabel="Year", ylabel="Tokens per Scan",
        )

    word_scan_years = sorted(data.words_per_scan_over_time.keys())
    word_scan_svg = ""
    if word_scan_years:
        x_labels = [str(y) for y in word_scan_years]
        tess_wps = [data.words_per_scan_over_time[y][0] for y in word_scan_years]
        vlm_wps = [data.words_per_scan_over_time[y][1] for y in word_scan_years]
        word_scan_svg = _svg_line_chart(
            [
                LineSeries(label="Tesseract", x_labels=x_labels, values=tess_wps, color="#2563eb"),
                LineSeries(label="VLM", x_labels=x_labels, values=vlm_wps, color="#dc2626"),
            ],
            xlabel="Year", ylabel="Words per Scan",
        )

    wttr_svg = ""
    if data.wttr_by_language:
        top_langs = sorted(
            data.wttr_by_language.keys(),
            key=lambda k: len(data.wttr_by_language[k]),
            reverse=True,
        )[:6]
        wttr_parts = []
        for i, lang in enumerate(top_langs):
            color = CHART_COLORS[i % len(CHART_COLORS)]
            svg = _svg_histogram(
                data.wttr_by_language[lang],
                xlabel=f"Word Type-Token Ratio ({lang})",
                color=color,
            )
            wttr_parts.append(
                f'<div class="subsection-header">{_escape_html(lang)}</div>'
                f'<div class="chart-container">{svg}</div>'
            )
        wttr_svg = _wrap_in_carousel(wttr_parts)

    return (
        f'<div class="section-grid"><div>'
        f'<div class="subsection-header">Tesseract</div>'
        f'<table class="stats-table">'
        f"<tr><td>Total Tokens</td><td>{stats.tesseract_total_tokens:,}</td></tr>"
        f"<tr><td>Total Words</td><td>{stats.tesseract_total_words:,}</td></tr>"
        f"<tr><td>Total Sentences</td><td>{stats.tesseract_total_sentences:,}</td></tr>"
        f"<tr><td>Avg Word Type-Token Ratio</td><td>{tess_avg}</td></tr>"
        f"<tr><td>Median Word Type-Token Ratio</td><td>{tess_med}</td></tr>"
        f"</table></div><div>"
        f'<div class="subsection-header">VLM</div>'
        f'<table class="stats-table">'
        f"<tr><td>Total Tokens</td><td>{stats.vlm_total_tokens:,}</td></tr>"
        f"<tr><td>Total Words</td><td>{stats.vlm_total_words:,}</td></tr>"
        f"<tr><td>Total Sentences</td><td>{stats.vlm_total_sentences:,}</td></tr>"
        f"<tr><td>Avg Word Type-Token Ratio</td><td>{vlm_avg}</td></tr>"
        f"<tr><td>Median Word Type-Token Ratio</td><td>{vlm_med}</td></tr>"
        f"<tr><td>Crops with Tables</td><td>{stats.vlm_crops_with_tables:,}</td></tr>"
        f"<tr><td>Crops with Markdown</td><td>{stats.vlm_crops_with_markdown:,}</td></tr>"
        f"</table></div></div>"
        f'<div class="subsection-header">Total Tokens Over Time</div>'
        f'<div class="chart-container">{tokens_time_svg}</div>'
        f'<div class="subsection-header">Total Tokens per Scan Over Time</div>'
        f'<div class="chart-container">{tok_scan_svg}</div>'
        f'<div class="subsection-header">Total Words per Scan Over Time</div>'
        f'<div class="chart-container">{word_scan_svg}</div>'
        f'<div class="subsection-header">Word Type-Token Ratio Distribution by Language</div>'
        f"{wttr_svg}"
    )


def _render_crop_classification(data: DashboardData) -> str:
    """Renders the crop classification section with agreement stats and confidence charts."""
    if not data.crop_class_stats or data.crop_class_stats.total == 0:
        return ""
    all_content = _render_crop_classification_content(data)
    has_pre1931 = (
        data.pre1931 and data.pre1931.crop_class_stats
        and data.pre1931.crop_class_stats.total > 0
    )
    pre1931_content = (
        _render_crop_classification_content(data.pre1931) if has_pre1931 else None
    )
    return _render_filtered_section("Crop Classification", all_content, pre1931_content)


def _render_crop_classification_content(data: DashboardData) -> str:
    """Renders inner content for the crop classification section."""
    stats = data.crop_class_stats
    if not stats or stats.total == 0:
        return '<p class="no-data">No classification data available.</p>'

    agreement_pct = stats.agreement_count / stats.total * 100
    img_pct = stats.final_eq_image_count / stats.total * 100
    txt_pct = stats.final_eq_text_count / stats.total * 100

    cat_bars = _svg_horizontal_bars(data.crops_by_final_category)
    tok_cat_bars = (
        _svg_horizontal_bars(data.tokens_by_final_category) if data.tokens_by_final_category else ""
    )

    all_cats = sorted({cat for cats in data.category_over_time.values() for cat in cats})
    stacked_svg = _svg_stacked_area(all_cats, data.category_over_time, width=900, height=350)

    tok_all_cats = sorted(
        {cat for cats in data.tokens_per_category_over_time.values() for cat in cats}
    )
    tok_stacked_svg = _svg_stacked_area(
        tok_all_cats, data.tokens_per_category_over_time, width=900, height=350
    )

    img_conf_parts = []
    for cat in sorted(data.image_confidence_by_category.keys()):
        scores = data.image_confidence_by_category[cat]
        if scores:
            svg = _svg_histogram(scores, xlabel=f"Image Confidence ({cat})", height=200)
            img_conf_parts.append(
                f'<div class="subsection-header">{_escape_html(cat)}</div>'
                f'<div class="chart-container">{svg}</div>'
            )
    img_conf_html = _wrap_in_carousel(img_conf_parts)

    txt_conf_parts = []
    for cat in sorted(data.text_confidence_by_category.keys()):
        scores = data.text_confidence_by_category[cat]
        if scores:
            svg = _svg_histogram(scores, xlabel=f"Text Confidence ({cat})", height=200)
            txt_conf_parts.append(
                f'<div class="subsection-header">{_escape_html(cat)}</div>'
                f'<div class="chart-container">{svg}</div>'
            )
    txt_conf_html = _wrap_in_carousel(txt_conf_parts)

    cat_area_stacked_svg = ""
    if data.category_area_over_time:
        area_cats = sorted({cat for yr in data.category_area_over_time.values() for cat in yr})
        cat_area_stacked_svg = _svg_stacked_area(
            area_cats, data.category_area_over_time, width=900, height=350
        )

    cat_indiv_line_svg = ""
    if data.category_individual_area_over_time:
        indiv_years = sorted(data.category_individual_area_over_time.keys())
        indiv_cats = sorted(
            {cat for yr in data.category_individual_area_over_time.values() for cat in yr}
        )
        indiv_x = [str(y) for y in indiv_years]
        indiv_series = []
        for idx, cat in enumerate(indiv_cats):
            vals = [
                data.category_individual_area_over_time.get(y, {}).get(cat, 0.0)
                for y in indiv_years
            ]
            indiv_series.append(
                LineSeries(
                    label=cat[:20], x_labels=indiv_x, values=vals,
                    color=CHART_COLORS[idx % len(CHART_COLORS)],
                )
            )
        cat_indiv_line_svg = _svg_line_chart(
            indiv_series, xlabel="Year", ylabel="Avg crop coverage (%)"
        )

    return (
        f'<table class="stats-table">'
        f"<tr><td>Total Classified</td><td>{stats.total:,}</td></tr>"
        f"<tr><td>Image/Text Agreement</td><td>{stats.agreement_count:,} ({agreement_pct:.1f}%)</td></tr>"
        f"<tr><td>Final = Image Category</td><td>{stats.final_eq_image_count:,} ({img_pct:.1f}%)</td></tr>"
        f"<tr><td>Final = Text Category</td><td>{stats.final_eq_text_count:,} ({txt_pct:.1f}%)</td></tr>"
        f"</table>"
        f'<div class="subsection-header">Distribution by Final Category</div>'
        f'<div class="chart-container">{cat_bars}</div>'
        f'<div class="subsection-header">Total VLM Tokens by Final Category</div>'
        f'<div class="chart-container">{tok_cat_bars}</div>'
        f'<div class="subsection-header">Crops per Category Over Time (Relative %)</div>'
        f'<div class="chart-container">{stacked_svg}</div>'
        f'<div class="subsection-header">Tokens per Category Over Time (Relative %)</div>'
        f'<div class="chart-container">{tok_stacked_svg}</div>'
        f'<div class="subsection-header">Image Confidence by Category</div>'
        f"{img_conf_html}"
        f'<div class="subsection-header">Text Confidence by Category</div>'
        f"{txt_conf_html}"
        f'<div class="subsection-header">Aggregate Category Area Coverage Over Time (Relative %)</div>'
        f'<div class="chart-container">{cat_area_stacked_svg}</div>'
        f'<div class="subsection-header">Individual Crop Area Coverage by Category Over Time</div>'
        f'<div class="chart-container">{cat_indiv_line_svg}</div>'
    )


def _render_crop_subject(data: DashboardData) -> str:
    """Renders the crop subject labels section."""
    if not data.top_labels:
        return ""
    all_content = _render_crop_subject_content(data)
    pre1931_content = (
        _render_crop_subject_content(data.pre1931)
        if data.pre1931 and data.pre1931.top_labels
        else None
    )
    return _render_filtered_section("Crop Subject", all_content, pre1931_content)


def _render_crop_subject_content(data: DashboardData) -> str:
    """Renders inner content for the crop subject section."""
    bars_svg = _svg_horizontal_bars(data.top_labels)

    conf_parts = []
    for label, _ in data.top_labels[:10]:
        scores = data.label_confidence_scores.get(label, [])
        if scores:
            svg = _svg_histogram(scores, xlabel=f"Confidence ({label[:30]})", height=200)
            display = label[:40] + "..." if len(label) > 43 else label
            conf_parts.append(
                f'<div class="subsection-header">{_escape_html(display)}</div>'
                f'<div class="chart-container">{svg}</div>'
            )
    conf_html = _wrap_in_carousel(conf_parts)

    return (
        f'<div class="subsection-header">Top Labels Across All Crops</div>'
        f'<div class="chart-container">{bars_svg}</div>'
        f'<div class="subsection-header">Confidence Score Distribution by Label</div>'
        f"{conf_html}"
    )


def _render_crop_ner(data: DashboardData) -> str:
    """Renders the NER entities section with top entities and confidence histograms."""
    if not data.ner_type_totals or sum(data.ner_type_totals.values()) == 0:
        return ""
    all_content = _render_crop_ner_content(data)
    has_pre1931 = (
        data.pre1931 and data.pre1931.ner_type_totals
        and sum(data.pre1931.ner_type_totals.values()) > 0
    )
    pre1931_content = (
        _render_crop_ner_content(data.pre1931) if has_pre1931 else None
    )
    return _render_filtered_section("Crop NER", all_content, pre1931_content)


def _render_crop_ner_content(data: DashboardData) -> str:
    """Renders inner content for the NER section."""
    totals_svg = _svg_horizontal_bars(
        [(k, v) for k, v in sorted(data.ner_type_totals.items(), key=lambda x: x[1], reverse=True)]
    )

    def _ner_table(entries: list[tuple[str, int]], label: str) -> str:
        if not entries:
            return ""
        rows = ""
        for entity, count in entries:
            rows += f"<tr><td>{_escape_html(entity)}</td><td>{count:,}</td></tr>"
        return (
            f'<div class="subsection-header">Top 20 {label}</div>'
            f'<div class="batch-table-container">'
            f'<table class="stats-table">'
            f"<thead><tr><th>Entity</th><th>Count</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    per_table = _ner_table(data.ner_per_top, "PER (Person)")
    loc_table = _ner_table(data.ner_loc_top, "LOC (Location)")
    org_table = _ner_table(data.ner_org_top, "ORG (Organization)")

    ner_conf_slides: list[str] = []
    if data.ner_per_scores:
        svg = _svg_histogram(data.ner_per_scores, xlabel="PER Confidence", height=200)
        ner_conf_slides.append(
            '<div class="subsection-header">PER Confidence Distribution</div>'
            f'<div class="chart-container">{svg}</div>'
        )
    if data.ner_loc_scores:
        svg = _svg_histogram(data.ner_loc_scores, xlabel="LOC Confidence", height=200)
        ner_conf_slides.append(
            '<div class="subsection-header">LOC Confidence Distribution</div>'
            f'<div class="chart-container">{svg}</div>'
        )
    if data.ner_org_scores:
        svg = _svg_histogram(data.ner_org_scores, xlabel="ORG Confidence", height=200)
        ner_conf_slides.append(
            '<div class="subsection-header">ORG Confidence Distribution</div>'
            f'<div class="chart-container">{svg}</div>'
        )
    ner_conf_html = _wrap_in_carousel(ner_conf_slides)

    return (
        f'<div class="subsection-header">Entity Type Totals</div>'
        f'<div class="chart-container">{totals_svg}</div>'
        f"{per_table}{loc_table}{org_table}"
        f'<div class="subsection-header">Confidence Distributions</div>'
        f"{ner_conf_html}"
    )


def _render_crop_chronam(data: DashboardData) -> str:
    """Renders the Chronam thesauri matches section with term frequency charts."""
    if not data.chronam_term_totals:
        return ""
    all_content = _render_crop_chronam_content(data)
    pre1931_content = (
        _render_crop_chronam_content(data.pre1931)
        if data.pre1931 and data.pre1931.chronam_term_totals
        else None
    )
    return _render_filtered_section("Crop Chronam Thesauri Match", all_content, pre1931_content)


def _render_crop_chronam_content(data: DashboardData) -> str:
    """Renders inner content for the chronam section."""
    rows = ""
    for term, count in data.chronam_term_totals[:50]:
        rows += f"<tr><td>{_escape_html(term)}</td><td>{count:,}</td></tr>"

    top_terms = [t for t, _ in data.chronam_term_totals[:10]]
    years = sorted(data.chronam_terms_over_time.keys())
    time_svg = ""
    if years and top_terms:
        x_labels = [str(y) for y in years]
        series = []
        for i, term in enumerate(top_terms):
            vals = [float(data.chronam_terms_over_time.get(y, {}).get(term, 0)) for y in years]
            series.append(
                LineSeries(
                    label=term[:20], x_labels=x_labels, values=vals,
                    color=CHART_COLORS[i % len(CHART_COLORS)],
                )
            )
        time_svg = _svg_line_chart(series, xlabel="Year", ylabel="Matches")

    return (
        f'<div class="subsection-header">Term Totals</div>'
        f'<div class="batch-table-container">'
        f'<table class="stats-table">'
        f"<thead><tr><th>Term</th><th>Count</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        f'<div class="subsection-header">Term Usage Over Time (Top 10)</div>'
        f'<div class="chart-container">{time_svg}</div>'
    )


def _normalize_country_name(name: str) -> str:
    """Normalizes a country name to match GeoJSON `properties.ADMIN` values."""
    mapped = _COUNTRY_NAME_MAP.get(name.strip().lower())
    if mapped:
        return mapped
    return name.strip().title()


def _convert_state_name_to_abbr(name: str) -> str | None:
    """Converts a US state name to its two-letter abbreviation."""
    if not name:
        return None
    upper = name.strip().upper()
    if upper in _ABBR_SET:
        return upper
    return _STATE_ABBR_MAP.get(name.strip().lower())


def _generate_html(data: DashboardData) -> str:
    """Assembles all rendered sections into a complete HTML document."""
    # Header and overview are fast and order-dependent
    prefix = [
        _render_header(data),
        _render_global_filter_toggle() if data.pre1931 else "",
        _render_overview(data),
        _render_batch_table(data),
    ]

    # Render remaining sections in parallel (each is independent and CPU-bound)
    render_fns = [
        _render_record_counts,
        _render_geography,
        _render_scan_dimensions,
        _render_crop_properties,
        _render_crop_language,
        _render_crop_text,
        _render_crop_classification,
        _render_crop_subject,
        _render_crop_ner,
        _render_crop_chronam,
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fn, data) for fn in render_fns]
        parallel_sections = [f.result() for f in futures]

    body = "\n".join(prefix + parallel_sections)
    return _wrap_html(f"Pipeline Run #{data.run.id}", body)


_DASHBOARD_TEMPLATE: Template | None = None


def _get_dashboard_template() -> Template:
    """Loads and caches the dashboard HTML template."""
    global _DASHBOARD_TEMPLATE
    if _DASHBOARD_TEMPLATE is None:
        _DASHBOARD_TEMPLATE = Template((const.TEMPLATES_DIR_PATH / "dashboard.html").read_text())
    return _DASHBOARD_TEMPLATE


def _wrap_html(title: str, body: str) -> str:
    """Wraps rendered body content in the dashboard HTML template."""
    return _get_dashboard_template().substitute(title=_escape_html(title), body=body)
