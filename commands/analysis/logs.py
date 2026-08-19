import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from string import Template

import click
from loguru import logger

import const


@dataclass
class StepInfo:
    name: str
    step_number: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    duration_display: str | None = None
    metrics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BatchInfo:
    batch_number: int
    node_name: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_display: str | None = None
    status: str = "running"
    steps: list[StepInfo] = field(default_factory=list)


@dataclass
class ResourceSample:
    timestamp: datetime
    step: str
    batch: int | None
    ram_pct: float
    cpu_pct: float
    vram_pcts: dict[str, float] = field(default_factory=dict)


@dataclass
class ParsedLog:
    """Represents a single pipeline run log file."""

    filename: str
    env_config: dict[str, str] = field(default_factory=dict)
    is_dry_run: bool = False
    run_number: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_display: str | None = None
    batches: list[BatchInfo] = field(default_factory=list)
    batches_processed: int | None = None
    batches_crashed: int | None = None
    batches_skipped: int | None = None
    resource_samples: list[ResourceSample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# Regex patterns
LOGURU_LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| (.+?) - (.+)"
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S.%f"

START_RUN_RE = re.compile(r"START RUN#(\d+)\s*$")
END_RUN_RE = re.compile(r"END RUN#(\d+) - (.+)$")
START_BATCH_RE = re.compile(r"START RUN#(\d+) BATCH#(\d+) NODE#(.+?)\s*$")
END_BATCH_RE = re.compile(r"END RUN#(\d+) BATCH#(\d+) NODE#(.+?) - (.+)$")
START_STEP_RE = re.compile(r"START (step\d+-[\w-]+) RUN#(\d+) BATCH#(\d+) NODE#(.+?)\s*$")
END_STEP_RE = re.compile(r"END (step\d+-[\w-]+) RUN#(\d+) BATCH#(\d+) NODE#(.+?) - (.+)$")
STEP_TOOK_RE = re.compile(r"(step\d+_\w+) took ([\d.]+)s")
STEP_NUMBER_RE = re.compile(r"step(\d+)")
BATCHES_PROCESSED_RE = re.compile(r"Batches processed: (\d+)")
BATCHES_CRASHED_RE = re.compile(r"Batches crashed: (\d+)")
BATCHES_SKIPPED_RE = re.compile(r"Batches skipped: (\d+)")
RESOURCE_SAMPLE_RE = re.compile(
    r"RESOURCE_SAMPLE step=([\w-]*) batch=(\d+|None) "
    r"ram_pct=([\d.]+) cpu_pct=([\d.]+) vram_pcts=([\S]*)"
)
ISSUE_SCANS_RE = re.compile(r".+? \(.+?\): archive downloaded, (\d+) scans extracted\.")
ISSUE_CACHED_RE = re.compile(r".+? \(.+?\): (\d+) scans already in cache\. Skipping\.")
CROP_COUNT_RE = re.compile(r"(\d+) crops across (\d+) scans")


def _parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, TIMESTAMP_FMT)


def _parse_vram_pcts(raw: str) -> dict[str, float]:
    """Parse '0:45.2,1:67.8' into {'0': 45.2, '1': 67.8}."""
    if not raw:
        return {}
    result: dict[str, float] = {}
    for pair in raw.split(","):
        idx, _, val = pair.partition(":")
        if val:
            result[idx] = float(val)
    return result


def _normalize_step_name(name: str) -> str:
    """Normalize step names: step01_cache -> step01-cache."""
    return name.replace("_", "-")


def _extract_step_number(name: str) -> int:
    match = STEP_NUMBER_RE.search(name)
    return int(match.group(1)) if match else 0


def _find_batch(parsed: ParsedLog, batch_number: int, node_name: str) -> BatchInfo:
    for batch in parsed.batches:
        if batch.batch_number == batch_number:
            return batch
    batch = BatchInfo(batch_number=batch_number, node_name=node_name)
    parsed.batches.append(batch)
    return batch


@click.command("logs")
def logs():
    """Parses pipeline log files and generates self-contained HTML reports with step timing charts, resource usage graphs, and warnings/errors summaries."""
    log_files = sorted(const.LOGS_DIR_PATH.glob("*.log"))

    if not log_files:
        logger.info(f"No .log files found in {const.LOGS_DIR_PATH}")
        return

    generated = 0
    skipped = 0

    for log_path in log_files:
        html_path = log_path.with_suffix(".html")

        if html_path.exists() and html_path.stat().st_mtime >= log_path.stat().st_mtime:
            skipped += 1
            continue

        parsed = parse_log_file(log_path)
        html = generate_html_report(parsed)
        html_path.write_text(html, encoding="utf-8")
        generated += 1
        logger.info(f"Generated report: {html_path.name}")

    logger.info(f"Done. {generated} report(s) generated, {skipped} skipped (up to date).")


def parse_log_file(filepath: Path) -> ParsedLog:
    """Parse a pipeline log file into structured data."""
    parsed = ParsedLog(filename=filepath.name)
    in_env_block = False
    current_step: StepInfo | None = None

    lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    for raw_line in lines:
        line = raw_line.rstrip()

        # Environment config block
        if line == "=== Environment configuration ===":
            in_env_block = True
            continue
        if line == "=================================":
            in_env_block = False
            continue
        if in_env_block:
            if "=" in line:
                key, _, value = line.partition("=")
                parsed.env_config[key.strip()] = value.strip()
            continue

        # Try parsing as a loguru line
        loguru_match = LOGURU_LINE_RE.match(line)
        if not loguru_match:
            continue

        timestamp_str, level, source, message = loguru_match.groups()
        timestamp = _parse_timestamp(timestamp_str)
        level = level.strip()
        message = message.strip()

        # Collect warnings and errors
        if level == "WARNING":
            parsed.warnings.append(message)
            if current_step:
                current_step.warnings.append(message)
        elif level == "ERROR":
            parsed.errors.append(message)
            if current_step:
                current_step.errors.append(message)

        # Dry run detection
        if "Dry run mode:" in message:
            parsed.is_dry_run = True

        # Run markers
        match = START_RUN_RE.search(message)
        if match:
            parsed.run_number = int(match.group(1))
            parsed.start_time = timestamp
            continue

        match = END_RUN_RE.search(message)
        if match:
            parsed.end_time = timestamp
            parsed.duration_display = match.group(2).strip()
            continue

        # Batch markers
        match = START_BATCH_RE.search(message)
        if match:
            batch = _find_batch(parsed, int(match.group(2)), match.group(3).strip())
            batch.start_time = timestamp
            continue

        match = END_BATCH_RE.search(message)
        if match:
            batch = _find_batch(parsed, int(match.group(2)), match.group(3).strip())
            batch.end_time = timestamp
            batch.duration_display = match.group(4).strip()
            batch.status = "success"
            continue

        # Step markers
        match = START_STEP_RE.search(message)
        if match:
            step_name = match.group(1)
            batch = _find_batch(parsed, int(match.group(3)), match.group(4).strip())
            current_step = StepInfo(
                name=step_name,
                step_number=_extract_step_number(step_name),
                start_time=timestamp,
            )
            batch.steps.append(current_step)
            continue

        match = END_STEP_RE.search(message)
        if match:
            step_name = match.group(1)
            batch = _find_batch(parsed, int(match.group(3)), match.group(4).strip())
            for step in batch.steps:
                if step.name == step_name:
                    step.end_time = timestamp
                    step.duration_display = match.group(5).strip()
                    break
            current_step = None
            continue

        # "took Xs" lines
        match = STEP_TOOK_RE.search(message)
        if match:
            normalized = _normalize_step_name(match.group(1))
            seconds = float(match.group(2))
            for batch in reversed(parsed.batches):
                for step in reversed(batch.steps):
                    if _normalize_step_name(step.name) == normalized:
                        step.duration_seconds = seconds
                        break
            continue

        # Batch summary counts
        match = BATCHES_PROCESSED_RE.search(message)
        if match:
            parsed.batches_processed = int(match.group(1))
            continue
        match = BATCHES_CRASHED_RE.search(message)
        if match:
            parsed.batches_crashed = int(match.group(1))
            continue
        match = BATCHES_SKIPPED_RE.search(message)
        if match:
            parsed.batches_skipped = int(match.group(1))
            continue

        # Resource monitoring samples
        match = RESOURCE_SAMPLE_RE.search(message)
        if match:
            batch_str = match.group(2)
            parsed.resource_samples.append(
                ResourceSample(
                    timestamp=timestamp,
                    step=match.group(1),
                    batch=int(batch_str) if batch_str != "None" else None,
                    ram_pct=float(match.group(3)),
                    cpu_pct=float(match.group(4)),
                    vram_pcts=_parse_vram_pcts(match.group(5)),
                )
            )
            continue

        # Step-specific metrics (INFO lines from step modules, not orchestration)
        if level == "INFO" and "commands.steps." in source and current_step:
            current_step.metrics.append(message)

    # Mark batches without END markers as pending (incomplete logs)
    for batch in parsed.batches:
        if batch.status == "running" and batch.start_time:
            batch.status = "pending"

    return parsed


def _format_seconds(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
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
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _esc(text: str) -> str:
    return escape(str(text))


def generate_html_report(parsed: ParsedLog) -> str:
    """Generate a self-contained HTML report from parsed log data."""
    sections = [
        _render_header(parsed),
        _render_env_section(parsed.env_config),
        _render_step_chart(parsed),
        _render_high_level_metrics(parsed),
        _render_resource_charts(parsed),
        _render_run_summary(parsed),
        _render_warnings_errors(parsed),
    ]
    body = "\n".join(sections)
    return _wrap_html(parsed.filename, body)


def _overall_status(parsed: ParsedLog) -> str:
    if parsed.batches_crashed and parsed.batches_crashed > 0:
        return "crashed"
    has_pending = False
    for batch in parsed.batches:
        if batch.status == "crashed":
            return "crashed"
        if batch.status == "pending":
            has_pending = True
    if has_pending:
        return "pending"
    return "success"


def _status_badge(status: str) -> str:
    css_class = {
        "success": "badge-success",
        "crashed": "badge-error",
        "pending": "badge-warning",
        "running": "badge-warning",
    }.get(status, "badge-warning")
    return f'<span class="badge {css_class}">{_esc(status.upper())}</span>'


def _render_header(parsed: ParsedLog) -> str:
    status = _overall_status(parsed)
    dry_run_html = ""
    if parsed.is_dry_run:
        dry_run_html = (
            '<div class="dry-run-banner">' "DRY RUN — batch status was not persisted" "</div>"
        )

    return f"""
    <div class="header">
        <h1>{_esc(parsed.filename)} {_status_badge(status)}</h1>
        {dry_run_html}
    </div>"""


def _render_env_section(env_config: dict[str, str]) -> str:
    if not env_config:
        return ""
    rows = "".join(
        f"<tr><td class='env-key'>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
        for k, v in env_config.items()
    )
    return f"""
    <details class="section">
        <summary><h2>Environment Configuration</h2></summary>
        <table class="env-table">{rows}</table>
    </details>"""


def _render_run_summary(parsed: ParsedLog) -> str:
    node_names = sorted({b.node_name for b in parsed.batches})
    run_label = f"#{parsed.run_number}" if parsed.run_number is not None else "—"

    summary_rows = f"""
        <tr><td>Run</td><td>{_esc(run_label)}</td></tr>
        <tr><td>Node(s)</td><td>{_esc(', '.join(node_names) if node_names else '—')}</td></tr>
        <tr><td>Start</td><td>{_format_datetime(parsed.start_time)}</td></tr>
        <tr><td>End</td><td>{_format_datetime(parsed.end_time)}</td></tr>
        <tr><td>Duration</td><td>{_esc(parsed.duration_display or '—')}</td></tr>"""

    if parsed.batches_processed is not None:
        summary_rows += f"""
        <tr><td>Batches processed</td><td>{parsed.batches_processed}</td></tr>
        <tr><td>Batches crashed</td><td>{parsed.batches_crashed or 0}</td></tr>
        <tr><td>Batches skipped</td><td>{parsed.batches_skipped or 0}</td></tr>"""

    batch_html = ""
    for batch in parsed.batches:
        batch_html += _render_batch_details(batch)

    return f"""
    <details class="section" open>
        <summary><h2>Run {_esc(run_label)}</h2></summary>
        <table class="summary-table">{summary_rows}</table>
        {batch_html}
    </details>"""


def _render_batch_details(batch: BatchInfo) -> str:
    step_rows = ""
    for step in batch.steps:
        duration_text = _format_seconds(step.duration_seconds) if step.duration_seconds else "—"
        warning_count = (
            f' <span class="warn-count">({len(step.warnings)} warn)</span>' if step.warnings else ""
        )

        # Collapsible log lines for step metrics
        logs_html = ""
        if step.metrics:
            metrics_items = "".join(f"<li>{_esc(m)}</li>" for m in step.metrics)
            logs_html = f"""
            <tr><td colspan="3" class="step-logs-cell">
                <details class="step-logs">
                    <summary>Logs ({len(step.metrics)})</summary>
                    <ul class="metrics-list">{metrics_items}</ul>
                </details>
            </td></tr>"""

        step_rows += f"""
            <tr>
                <td class="step-name">{_esc(step.name)}</td>
                <td>{_format_datetime(step.start_time)}</td>
                <td>{_esc(duration_text)}{warning_count}</td>
            </tr>{logs_html}"""

    return f"""
        <details class="batch-details">
            <summary>
                <strong>Batch #{batch.batch_number}</strong> — {_esc(batch.node_name)}
                {_status_badge(batch.status)}
                <span class="batch-duration">{_esc(batch.duration_display or '—')}</span>
            </summary>
            <table class="step-table">
                <thead>
                    <tr><th>Step</th><th>Start</th><th>Duration</th></tr>
                </thead>
                <tbody>{step_rows}</tbody>
            </table>
        </details>"""


def _collect_step_durations(parsed: ParsedLog) -> dict[str, list[float]]:
    """Collect durations per step name across all batches."""
    durations: dict[str, list[float]] = {}
    for batch in parsed.batches:
        for step in batch.steps:
            if step.duration_seconds is not None:
                durations.setdefault(step.name, []).append(step.duration_seconds)
    return durations


def _render_step_chart(parsed: ParsedLog) -> str:
    durations = _collect_step_durations(parsed)
    if not durations:
        return ""

    multiple_batches = len(parsed.batches) > 1
    sorted_steps = sorted(durations.items(), key=lambda x: _extract_step_number(x[0]))

    # Use total time across batches for chart proportions
    total_durations = {name: sum(vals) for name, vals in sorted_steps}
    max_duration = max(total_durations.values()) if total_durations else 1

    bars = ""
    for name, total in total_durations.items():
        width_pct = (total / max_duration * 100) if max_duration > 0 else 0
        vals = durations[name]
        if multiple_batches:
            avg = total / len(vals)
            label = (
                f"{_format_seconds(total)} total — {_format_seconds(avg)} avg ({len(vals)} batches)"
            )
        else:
            label = _format_seconds(total)
        bars += f"""
            <div class="chart-row">
                <div class="chart-label">{_esc(name)}</div>
                <div class="chart-bar-container">
                    <div class="chart-bar" style="width: {width_pct:.1f}%">
                        <span class="chart-value">{_esc(label)}</span>
                    </div>
                </div>
            </div>"""

    title = "Step Duration (total across batches)" if multiple_batches else "Step Duration"
    return f"""
    <details class="section" open>
        <summary><h2>{title}</h2></summary>
        <div class="chart">{bars}</div>
    </details>"""


def _render_high_level_metrics(parsed: ParsedLog) -> str:
    """Render a throughput metrics table based on batch-level stats."""
    rows: list[tuple[int, float, int, int, int]] = []  # batch#, duration_min, issues, scans, crops

    for batch in parsed.batches:
        if not batch.start_time or not batch.end_time:
            continue
        duration_min = (batch.end_time - batch.start_time).total_seconds() / 60
        if duration_min <= 0:
            continue

        issues = 0
        scans = 0
        crops = 0
        for step in batch.steps:
            if "step01" in step.name:
                for msg in step.metrics:
                    match = ISSUE_SCANS_RE.search(msg)
                    if match:
                        issues += 1
                        scans += int(match.group(1))
                        continue
                    match = ISSUE_CACHED_RE.search(msg)
                    if match:
                        issues += 1
                        scans += int(match.group(1))
            elif "step02" in step.name:
                for msg in step.metrics:
                    match = CROP_COUNT_RE.search(msg)
                    if match:
                        crops += int(match.group(1))

        rows.append((batch.batch_number, duration_min, issues, scans, crops))

    if not rows:
        return ""

    total_duration = sum(r[1] for r in rows)
    total_issues = sum(r[2] for r in rows)
    total_scans = sum(r[3] for r in rows)
    total_crops = sum(r[4] for r in rows)

    def _rate_cells(count: int, duration_min: float) -> str:
        per_min = count / duration_min if duration_min > 0 else 0
        per_hr = per_min * 60
        return f"<td>{per_min:.1f}</td><td>{per_hr:.0f}</td>"

    table_rows = ""
    for batch_num, dur, issues, scans, crops in rows:
        table_rows += (
            f"<tr><td>#{batch_num}</td><td>{_format_seconds(dur * 60)}</td>"
            f"<td>{issues}</td><td>{scans}</td><td>{crops}</td>"
            f"{_rate_cells(issues, dur)}"
            f"{_rate_cells(scans, dur)}"
            f"{_rate_cells(crops, dur)}</tr>"
        )

    table_rows += (
        f"<tr class='totals-row'><td>Total</td><td>{_format_seconds(total_duration * 60)}</td>"
        f"<td>{total_issues}</td><td>{total_scans}</td><td>{total_crops}</td>"
        f"{_rate_cells(total_issues, total_duration)}"
        f"{_rate_cells(total_scans, total_duration)}"
        f"{_rate_cells(total_crops, total_duration)}</tr>"
    )

    return f"""
    <details class="section" open>
        <summary><h2>High Level Metrics</h2></summary>
        <table class="hl-metrics-table">
            <thead><tr>
                <th>Batch</th><th>Duration</th>
                <th>Issues</th><th>Scans</th><th>Crops</th>
                <th>Issues/min</th><th>Issues/hr</th>
                <th>Scans/min</th><th>Scans/hr</th>
                <th>Crops/min</th><th>Crops/hr</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
    </details>"""


GPU_COLORS = [
    "#dc2626",
    "#ea580c",
    "#d97706",
    "#16a34a",
    "#0891b2",
    "#2563eb",
    "#7c3aed",
    "#c026d3",
]


def _compute_transition_dots(
    samples: list[ResourceSample],
) -> list[tuple[int, str]]:
    """Return (sample_index, label) for each batch+step transition.

    Label is the zero-padded step number (e.g. step01-cache = "01").
    """
    dots: list[tuple[int, str]] = []
    prev_key: tuple[int | None, str] | None = None
    for i, s in enumerate(samples):
        key = (s.batch, s.step)
        if key != prev_key and s.batch is not None and s.step:
            step_num = _extract_step_number(s.step)
            label = f"{step_num:02d}"
            dots.append((i, label))
            prev_key = key
    return dots


def _compute_hour_boundaries(
    samples: list[ResourceSample],
) -> list[tuple[datetime, datetime]]:
    """Split the sample time range into 3-hour windows."""
    t_min = samples[0].timestamp
    t_max = samples[-1].timestamp
    boundaries: list[tuple[datetime, datetime]] = []
    # Align to the start of the hour containing t_min
    window_start = t_min.replace(minute=0, second=0, microsecond=0)
    while window_start < t_max:
        window_end = window_start + timedelta(hours=3)
        boundaries.append((window_start, window_end))
        window_start = window_end
    return boundaries


def _render_time_series_svg(
    samples: list[ResourceSample],
    series: list[tuple[str, str, list[float]]],
    transition_dots: list[tuple[int, str]],
    window_start: datetime,
    window_end: datetime,
) -> str:
    """Render an SVG chart for the given 3-hour time window."""
    margin_left = 50
    margin_right = 20
    margin_top = 30
    margin_bottom = 40
    chart_w = 960
    chart_h = 200
    total_w = margin_left + chart_w + margin_right
    total_h = margin_top + chart_h + margin_bottom

    t_range = (window_end - window_start).total_seconds()

    def x_pos(ts: datetime) -> float:
        return margin_left + (ts - window_start).total_seconds() / t_range * chart_w

    def y_pos(pct: float) -> float:
        return margin_top + (1 - pct / 100) * chart_h

    # Filter samples to this window (include one before/after for line continuity)
    indices_in_window: list[int] = []
    for i, s in enumerate(samples):
        if s.timestamp >= window_start and s.timestamp <= window_end:
            indices_in_window.append(i)
    # Extend by one sample on each side for smooth line entry/exit
    if indices_in_window:
        first = indices_in_window[0]
        last = indices_in_window[-1]
        if first > 0:
            indices_in_window.insert(0, first - 1)
        if last < len(samples) - 1:
            indices_in_window.append(last + 1)
    elif samples:
        # No samples in window — find the surrounding pair for interpolation
        for i in range(len(samples) - 1):
            if samples[i].timestamp < window_start and samples[i + 1].timestamp > window_end:
                indices_in_window = [i, i + 1]
                break

    # Y-axis grid
    grid = ""
    for pct in (0, 25, 50, 75, 100):
        y = y_pos(pct)
        grid += (
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_w}" '
            f'y2="{y:.1f}" stroke="#ddd" stroke-dasharray="4,4" />\n'
            f'<text x="{margin_left - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{pct}%</text>\n'
        )

    # X-axis time labels — 30-minute ticks for a 3-hour window
    x_labels = ""
    tick_interval = 1800
    tick_s = tick_interval
    while tick_s < t_range:
        ts = window_start + timedelta(seconds=tick_s)
        x = x_pos(ts)
        time_label = ts.strftime("%H:%M")
        x_labels += (
            f'<line x1="{x:.1f}" y1="{margin_top + chart_h}" x2="{x:.1f}" '
            f'y2="{margin_top + chart_h + 5}" stroke="#999" />\n'
            f'<text x="{x:.1f}" y="{margin_top + chart_h + 18}" text-anchor="middle" '
            f'font-size="10" fill="#6b7280">{time_label}</text>\n'
        )
        tick_s += tick_interval

    # Window hour label on x-axis start
    start_label = window_start.strftime("%H:%M")
    x_labels += (
        f'<text x="{margin_left}" y="{margin_top + chart_h + 18}" text-anchor="middle" '
        f'font-size="10" fill="#6b7280">{start_label}</text>\n'
    )

    # Polylines (clipped to chart area)
    clip_id = f"clip-{id(series)}-{int(window_start.timestamp())}"
    clip_def = (
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="{margin_left}" y="{margin_top}" '
        f'width="{chart_w}" height="{chart_h}" />'
        f"</clipPath></defs>\n"
    )
    polylines = ""
    for label, color, values in series:
        if not indices_in_window:
            continue
        points = " ".join(
            f"{x_pos(samples[i].timestamp):.1f},{y_pos(values[i]):.1f}" for i in indices_in_window
        )
        polylines += (
            f'<polyline points="{points}" fill="none" '
            f'stroke="{color}" stroke-width="2" clip-path="url(#{clip_id})" />\n'
        )

    # Transition dots in this window
    dots_svg = ""
    label_r = 8
    window_dot_idx = 0
    for sample_i, label in transition_dots:
        ts = samples[sample_i].timestamp
        if ts < window_start or ts > window_end:
            continue
        cx = x_pos(ts)
        if len(series) == 1:
            cy = y_pos(series[0][2][sample_i])
        else:
            cy = y_pos(max(s[2][sample_i] for s in series))
        if window_dot_idx % 2 == 0:
            label_cy = cy - 12
        else:
            label_cy = cy + 14
        # Clamp label circle within chart vertical bounds
        label_cy = max(margin_top + label_r, min(margin_top + chart_h - label_r, label_cy))
        dots_svg += (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
            f'fill="#374151" stroke="white" stroke-width="1.5" />\n'
            f'<circle cx="{cx:.1f}" cy="{label_cy:.1f}" r="{label_r}" '
            f'fill="white" stroke="#374151" stroke-width="0.8" />\n'
            f'<text x="{cx:.1f}" y="{label_cy + 3:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#374151" font-weight="600">{label}</text>\n'
        )
        window_dot_idx += 1

    # Series legend
    pad = 5
    entry_widths = [len(label) * 7 + 26 for label, _, _ in series]
    total_legend_w = sum(entry_widths)
    legend_x = margin_left + chart_w - total_legend_w - pad
    legend_y = margin_top + 4
    legend = (
        f'<rect x="{legend_x}" y="{legend_y - pad}" '
        f'width="{total_legend_w + 2 * pad}" height="{pad + 4 + pad + 3}" '
        f'fill="white" rx="3" />\n'
    )
    lx = legend_x + pad
    for (label, color, _), ew in zip(series, entry_widths):
        legend += (
            f'<rect x="{lx}" y="{legend_y}" width="14" height="3" fill="{color}" />\n'
            f'<text x="{lx + 18}" y="{legend_y + 4}" font-size="11" '
            f'fill="#374151">{label}</text>\n'
        )
        lx += ew

    return (
        f'<svg class="resource-svg" viewBox="0 0 {total_w} {total_h}"'
        f' xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%; max-width:{total_w}px; height:{total_h}px;'
        f' font-family: -apple-system, BlinkMacSystemFont, sans-serif;">\n'
        f"  {clip_def}\n  {grid}\n  {x_labels}\n  {polylines}\n  {dots_svg}\n  {legend}\n"
        f'  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}"'
        f' y2="{margin_top + chart_h}" stroke="#999" />\n'
        f'  <line x1="{margin_left}" y1="{margin_top + chart_h}"'
        f' x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#999" />\n'
        f"</svg>"
    )


def _render_paginated_chart(
    chart_id: str,
    title: str,
    samples: list[ResourceSample],
    series: list[tuple[str, str, list[float]]],
    transition_dots: list[tuple[int, str]],
    hours: list[tuple[datetime, datetime]],
) -> str:
    """Render a paginated chart with one SVG per 3-hour window and left/right navigation."""
    pages = ""
    for page_idx, (w_start, w_end) in enumerate(hours):
        svg = _render_time_series_svg(samples, series, transition_dots, w_start, w_end)
        display = "block" if page_idx == 0 else "none"
        hour_label = w_start.strftime("%Y-%m-%d %H:%M")
        pages += (
            f'<div class="chart-page" data-page="{page_idx}" '
            f'style="display:{display}">'
            f'<div style="text-align:center;font-size:0.8rem;color:#6b7280;'
            f'padding:2px 0;">{hour_label} — {w_end.strftime("%H:%M")}</div>'
            f"{svg}</div>\n"
        )

    total_pages = len(hours)
    nav = (
        f'<div class="chart-nav" style="display:flex;align-items:center;'
        f'justify-content:center;gap:12px;padding:4px 0 8px;">'
        f'<button class="nav-btn nav-prev" title="Previous window" '
        f'style="border:1px solid #d1d5db;background:white;border-radius:6px;'
        f'padding:4px 12px;cursor:pointer;font-size:1rem;">'
        f"&#9664;</button>"
        f'<span class="nav-label" style="font-size:0.85rem;color:#374151;'
        f'min-width:80px;text-align:center;">1 / {total_pages}</span>'
        f'<button class="nav-btn nav-next" title="Next window" '
        f'style="border:1px solid #d1d5db;background:white;border-radius:6px;'
        f'padding:4px 12px;cursor:pointer;font-size:1rem;">'
        f"&#9654;</button>"
        f"</div>"
    )

    return (
        f'<div class="paginated-chart" id="{chart_id}">'
        f'<h3 style="padding:8px 16px 0;margin:0;font-size:0.95rem;">{title}</h3>'
        f'<div style="padding:0 16px;">{pages}</div>'
        f"{nav}</div>\n"
    )


def _render_resource_charts(parsed: ParsedLog) -> str:
    """Render three separate paginated SVG charts for RAM, VRAM, and CPU."""
    samples = parsed.resource_samples
    if not samples:
        return ""

    transition_dots = _compute_transition_dots(samples)
    windows = _compute_hour_boundaries(samples)

    # --- RAM ---
    ram_series: list[tuple[str, str, list[float]]] = [
        ("RAM", "#2563eb", [s.ram_pct for s in samples]),
    ]
    ram_html = _render_paginated_chart(
        "chart-ram", "RAM", samples, ram_series, transition_dots, windows
    )

    # --- VRAM (one line per GPU) ---
    gpu_indices = sorted({idx for s in samples for idx in s.vram_pcts})
    vram_html = ""
    if gpu_indices:
        vram_series: list[tuple[str, str, list[float]]] = []
        for i, idx in enumerate(gpu_indices):
            color = GPU_COLORS[i % len(GPU_COLORS)]
            vram_series.append((f"GPU {idx}", color, [s.vram_pcts.get(idx, 0.0) for s in samples]))
        vram_html = _render_paginated_chart(
            "chart-vram", "VRAM", samples, vram_series, transition_dots, windows
        )

    # --- CPU ---
    cpu_series: list[tuple[str, str, list[float]]] = [
        ("CPU", "#16a34a", [s.cpu_pct for s in samples]),
    ]
    cpu_html = _render_paginated_chart(
        "chart-cpu", "CPU", samples, cpu_series, transition_dots, windows
    )

    script = """
    <script>
    (function() {
      document.querySelectorAll('.paginated-chart').forEach(function(container) {
        var pages = container.querySelectorAll('.chart-page');
        var label = container.querySelector('.nav-label');
        var current = 0;
        var total = pages.length;

        function show(idx) {
          pages[current].style.display = 'none';
          current = idx;
          pages[current].style.display = 'block';
          label.textContent = (current + 1) + ' / ' + total;
        }

        container.querySelector('.nav-prev').addEventListener('click', function() {
          if (current > 0) show(current - 1);
        });
        container.querySelector('.nav-next').addEventListener('click', function() {
          if (current < total - 1) show(current + 1);
        });
      });
    })();
    </script>"""

    return f"""
    <details class="section" open>
        <summary><h2>Resource Usage</h2></summary>
        {ram_html}
        {vram_html}
        {cpu_html}
        {script}
    </details>"""


def _render_warnings_errors(parsed: ParsedLog) -> str:
    if not parsed.warnings and not parsed.errors:
        return ""

    total_count = len(parsed.warnings) + len(parsed.errors)

    # Group warnings and errors by step of origin
    step_issues: dict[str, dict[str, list[str]]] = {}
    for batch in parsed.batches:
        for step in batch.steps:
            if step.warnings or step.errors:
                entry = step_issues.setdefault(step.name, {"warnings": [], "errors": []})
                entry["warnings"].extend(step.warnings)
                entry["errors"].extend(step.errors)

    # Count attributed issues to find orphans
    attributed_warnings = sum(len(v["warnings"]) for v in step_issues.values())
    attributed_errors = sum(len(v["errors"]) for v in step_issues.values())
    orphan_warning_count = len(parsed.warnings) - attributed_warnings
    orphan_error_count = len(parsed.errors) - attributed_errors

    step_sections = ""
    for step_name in sorted(step_issues, key=_extract_step_number):
        issues = step_issues[step_name]
        step_total = len(issues["warnings"]) + len(issues["errors"])

        items = ""
        if issues["warnings"]:
            counter = Counter(issues["warnings"])
            items += "".join(
                f"<li><span class='warn-badge'>{count}x</span> {_esc(msg)}</li>"
                for msg, count in counter.most_common()
            )
        if issues["errors"]:
            counter = Counter(issues["errors"])
            items += "".join(
                f"<li class='error-item'><span class='error-badge'>{count}x</span> {_esc(msg)}</li>"
                for msg, count in counter.most_common()
            )

        step_sections += f"""
        <details>
            <summary><strong>{_esc(step_name)}</strong> ({step_total})</summary>
            <ul class="warn-list">{items}</ul>
        </details>"""

    # Orphan issues (outside any step boundary)
    if orphan_warning_count > 0 or orphan_error_count > 0:
        step_warning_counts: Counter[str] = Counter()
        step_error_counts: Counter[str] = Counter()
        for issues in step_issues.values():
            step_warning_counts.update(issues["warnings"])
            step_error_counts.update(issues["errors"])

        items = ""
        if orphan_warning_count > 0:
            all_warn_counter = Counter(parsed.warnings)
            for msg, count in all_warn_counter.most_common():
                remainder = count - step_warning_counts.get(msg, 0)
                if remainder > 0:
                    items += f"<li><span class='warn-badge'>{remainder}x</span> {_esc(msg)}</li>"
        if orphan_error_count > 0:
            all_err_counter = Counter(parsed.errors)
            for msg, count in all_err_counter.most_common():
                remainder = count - step_error_counts.get(msg, 0)
                if remainder > 0:
                    items += f"<li class='error-item'><span class='error-badge'>{remainder}x</span> {_esc(msg)}</li>"

        if items:
            orphan_total = orphan_warning_count + orphan_error_count
            step_sections += f"""
        <details>
            <summary><strong>Other</strong> ({orphan_total})</summary>
            <ul class="warn-list">{items}</ul>
        </details>"""

    return f"""
    <details class="section" open>
        <summary><h2>Warnings &amp; Errors ({total_count})</h2></summary>
        {step_sections}
    </details>"""


_LOGS_TEMPLATE: Template | None = None


def _get_logs_template() -> Template:
    global _LOGS_TEMPLATE
    if _LOGS_TEMPLATE is None:
        _LOGS_TEMPLATE = Template((const.TEMPLATES_DIR_PATH / "logs.html").read_text())
    return _LOGS_TEMPLATE


def _wrap_html(title: str, body: str) -> str:
    return _get_logs_template().substitute(title=_esc(title), body=body)
