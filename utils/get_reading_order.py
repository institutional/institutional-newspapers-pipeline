"""
Determine reading order for newspaper page crops using HDBSCAN column clustering.

Newspaper pages are laid out in columns, so reading order is primarily
left-to-right across columns and top-to-bottom within each column. The
algorithm estimates column structure by clustering the x-centers of narrow
"Content" crops via HDBSCAN, then assigns remaining crops (non-content and
wide/spanning elements) to the detected columns. A final left-edge bucketing
pass enforces strict L-R, T-B ordering within each bucket.

Steps:
  1. Classify crops as narrow or wide based on estimated column width.
  2. Cluster narrow Content crops into columns (HDBSCAN on x-centers).
  3. Assign narrow non-content crops to nearest column (by x-center distance).
  4. Assign wide crops to the overlapping column with the most content above.
  5. Order: left-to-right by column, top-to-bottom within columns.
  6. Post-process via left-edge bucketing for final L-R T-B enforcement.
     Visual elements (photographs, cartoons) are sorted by their top edge
     rather than y-center, so they appear before the content they illustrate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from const import (
    READING_ORDER_COLUMN_WIDTH_PERCENTILE,
    READING_ORDER_WIDE_CROP_THRESHOLD_RATIO,
    READING_ORDER_HDBSCAN_MIN_CLUSTER_SIZE,
    READING_ORDER_HDBSCAN_MIN_SAMPLES,
)


def get_reading_order(
    scan_width: int,
    bboxes_xyxy: list[list[float]],
    classification: list[str | None],
    texts: list[str | None] | None = None,
) -> list[int]:
    """Predict reading order for newspaper crop bounding boxes using HDBSCAN column clustering."""
    detector = ReadingOrderDetector()
    return detector.detect(scan_width, bboxes_xyxy, classification, texts)


class ReadingOrderDetector:
    """Main orchestrator for reading order detection using HDBSCAN column clustering."""

    def __init__(self):
        self._column_detector = ColumnDetector()
        self._postprocessor = ReadingOrderPostProcessor()

    def detect(
        self,
        scan_width: int,
        bboxes_xyxy: list[list[float]],
        classification: list[str | None],
        texts: list[str | None] | None = None,
    ) -> list[int]:
        """Detect reading order for the given bounding boxes and return ordered indices."""
        n_boxes = len(bboxes_xyxy)
        texts = texts or []

        if n_boxes == 0:
            return []
        if n_boxes == 1:
            return [0]
        if n_boxes == 2:
            order = self._sort_two_boxes(bboxes_xyxy)
            context = LayoutContext(
                scan_width=scan_width,
                bboxes_xyxy=bboxes_xyxy,
                classification=classification,
                texts=texts,
            )
            return self._postprocessor.postprocess(order, context)

        context = LayoutContext(
            scan_width=scan_width,
            bboxes_xyxy=bboxes_xyxy,
            classification=classification,
            texts=texts,
        )

        columns, wide_indices = self._column_detector.cluster_into_columns(context)

        if wide_indices and columns:
            columns = self._assign_wide_crops(context, columns, wide_indices)

        column_order = self._order_by_columns(context, columns)

        return self._postprocessor.postprocess(column_order, context)

    def _sort_two_boxes(self, bboxes_xyxy: list[list[float]]) -> list[int]:
        """Sort two boxes: same y-level sorts by x, otherwise by y."""
        y0, y1 = _xyxy_y_center(bboxes_xyxy[0]), _xyxy_y_center(bboxes_xyxy[1])
        x0, x1 = _xyxy_x_center(bboxes_xyxy[0]), _xyxy_x_center(bboxes_xyxy[1])

        if abs(y0 - y1) < abs(x0 - x1) * 0.5:
            return [0, 1] if x0 < x1 else [1, 0]
        return [0, 1] if y0 < y1 else [1, 0]

    def _assign_wide_crops(
        self,
        context: LayoutContext,
        columns: dict[int, list[int]],
        wide_indices: list[int],
    ) -> dict[int, list[int]]:
        """Assign wide crops to the overlapping column with the most content above."""
        columns = {k: list(v) for k, v in columns.items()}

        col_info = []
        for label, indices in columns.items():
            bboxes = [context.bboxes_xyxy[i] for i in indices]
            col_info.append(
                ColumnInfo(
                    label=label,
                    left=min(b[0] for b in bboxes),
                    right=max(b[2] for b in bboxes),
                    x_center=float(np.mean([_xyxy_x_center(b) for b in bboxes])),
                    indices=indices,
                )
            )

        col_info.sort(key=lambda c: c.x_center)

        for wide_idx in wide_indices:
            wide_bbox = context.bboxes_xyxy[wide_idx]
            wide_top = wide_bbox[1]
            wide_left, wide_right = wide_bbox[0], wide_bbox[2]

            overlapping_cols = []
            for col in col_info:
                if wide_left <= col.right and wide_right >= col.left:
                    content_above = sum(
                        1
                        for i in col.indices
                        if context.bboxes_xyxy[i][3] < wide_top
                    )
                    overlapping_cols.append((col.label, content_above, col.x_center))

            if overlapping_cols:
                best_col = max(overlapping_cols, key=lambda x: (x[1], -x[2]))[0]
                columns[best_col].append(wide_idx)
            elif col_info:
                columns[col_info[0].label].append(wide_idx)

        return columns

    def _order_by_columns(
        self,
        context: LayoutContext,
        columns: dict[int, list[int]],
    ) -> list[int]:
        """Order crops by columns: left-to-right, top-to-bottom within columns."""
        sorted_cols = sorted(
            columns.items(),
            key=lambda x: np.mean([context.x_centers[i] for i in x[1]]) if x[1] else 0,
        )

        result = []
        for _, crop_indices in sorted_cols:
            result.extend(sorted(crop_indices, key=lambda i: context.y_centers[i]))
        return result


class ColumnDetector:
    """Handles HDBSCAN-based column detection."""

    def _create_clusterer(self):
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=READING_ORDER_HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=READING_ORDER_HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
        )

    def cluster_into_columns(
        self, context: LayoutContext
    ) -> tuple[dict[int, list[int]], list[int]]:
        """Cluster crops into columns using HDBSCAN with classification-aware assignment."""
        narrow_content_indices = context.narrow_content_indices.tolist()
        narrow_non_content_indices = context.narrow_non_content_indices.tolist()
        wide_indices = context.wide_indices.tolist()

        # Cluster only Content crops first to establish column structure
        if len(narrow_content_indices) >= 2:
            content_x = context.x_centers[context.narrow_content_indices].reshape(-1, 1)

            labels = self._create_clusterer().fit_predict(content_x)
            columns = self._build_columns_from_labels(
                np.array(narrow_content_indices), content_x, labels
            )

            # Assign non-content crops to nearest column by x-center
            columns = self._assign_non_content_to_columns(
                context, columns, narrow_non_content_indices
            )
        elif len(context.narrow_indices) >= 2:
            # Fallback: cluster all narrow crops
            narrow_x = context.x_centers[context.narrow_mask].reshape(-1, 1)
            labels = self._create_clusterer().fit_predict(narrow_x)
            columns = self._build_columns_from_labels(context.narrow_indices, narrow_x, labels)
        else:
            columns = {0: list(range(context.n_boxes))}
            wide_indices = []

        return columns, wide_indices

    def _assign_non_content_to_columns(
        self,
        context: LayoutContext,
        columns: dict[int, list[int]],
        non_content_indices: list[int],
    ) -> dict[int, list[int]]:
        """Assign non-content crops to nearest column by x-center."""
        if not columns or not non_content_indices:
            return columns

        col_x_centers = {}
        for label, indices in columns.items():
            col_x_centers[label] = np.mean([context.x_centers[i] for i in indices])

        for idx in non_content_indices:
            x = context.x_centers[idx]
            nearest_col = min(col_x_centers.keys(), key=lambda c: abs(col_x_centers[c] - x))
            columns[nearest_col].append(idx)

        return columns

    def _build_columns_from_labels(
        self,
        narrow_indices: np.ndarray,
        narrow_x: np.ndarray,
        labels: np.ndarray,
    ) -> dict[int, list[int]]:
        """Build column dict from HDBSCAN labels, assigning noise to nearest cluster."""
        columns: dict[int, list[int]] = defaultdict(list)

        for local_idx, label in enumerate(labels):
            global_idx = narrow_indices[local_idx]

            if label == -1:
                label = self._find_nearest_cluster(narrow_x[local_idx], narrow_x, labels)

            columns[label].append(global_idx)

        return dict(columns)

    def _find_nearest_cluster(
        self, point: np.ndarray, all_points: np.ndarray, labels: np.ndarray
    ) -> int:
        """Find nearest non-noise cluster for a noise point."""
        valid_mask = labels >= 0
        if not np.any(valid_mask):
            return 0

        distances = np.abs(all_points[valid_mask].flatten() - point.flatten()[0])
        return int(labels[valid_mask][np.argmin(distances)])


class ReadingOrderPostProcessor:
    """Post-processing: L-R T-B enforcement using left-edge bucketing."""

    def postprocess(
        self,
        order: list[int],
        context: LayoutContext,
    ) -> list[int]:
        if len(order) <= 1:
            return order

        # More buckets for denser layouts
        n_crops = len(order)
        if n_crops > 80:
            bucket_count = 15
        elif n_crops > 40:
            bucket_count = 12
        else:
            bucket_count = 10

        # Check for lowercase starts (continuation signal) - used as tie-breaker
        starts_lowercase = set()
        if context.texts:
            for i, text in enumerate(context.texts):
                if text:
                    text = text.strip()
                    if text and text[0].islower():
                        starts_lowercase.add(i)

        def sort_key(idx: int) -> tuple[int, float, int]:
            x_left = context.bboxes_xyxy[idx][0]
            bucket = int(x_left / context.scan_width * bucket_count)

            # Visual elements use top edge to place them before the content they illustrate
            cls = context.classification[idx]
            if cls in ("Photograph or illustration", "Cartoon"):
                y = context.bboxes_xyxy[idx][1]
            else:
                y = context.y_centers[idx]

            continuation_penalty = 1 if idx in starts_lowercase else 0

            return (bucket, y, continuation_penalty)

        return sorted(order, key=sort_key)


@dataclass
class LayoutContext:
    """Immutable container for layout analysis inputs with cached computed properties."""

    scan_width: int
    bboxes_xyxy: list[list[float]]
    classification: list[str | None]
    texts: list[str | None] = field(default_factory=list)

    @cached_property
    def n_boxes(self) -> int:
        return len(self.bboxes_xyxy)

    @cached_property
    def widths(self) -> np.ndarray:
        return np.array([_xyxy_width(b) for b in self.bboxes_xyxy])

    @cached_property
    def x_centers(self) -> np.ndarray:
        return np.array([_xyxy_x_center(b) for b in self.bboxes_xyxy])

    @cached_property
    def y_centers(self) -> np.ndarray:
        return np.array([_xyxy_y_center(b) for b in self.bboxes_xyxy])

    @cached_property
    def estimated_column_width(self) -> float:
        return float(np.percentile(self.widths, READING_ORDER_COLUMN_WIDTH_PERCENTILE))

    @cached_property
    def wide_threshold(self) -> float:
        return self.estimated_column_width * READING_ORDER_WIDE_CROP_THRESHOLD_RATIO

    @cached_property
    def narrow_mask(self) -> np.ndarray:
        return self.widths <= self.wide_threshold

    @cached_property
    def narrow_indices(self) -> np.ndarray:
        return np.where(self.narrow_mask)[0]

    @cached_property
    def wide_indices(self) -> np.ndarray:
        return np.where(~self.narrow_mask)[0]

    @cached_property
    def content_mask(self) -> np.ndarray:
        return np.array([c == "Content" for c in self.classification])

    @cached_property
    def narrow_content_indices(self) -> np.ndarray:
        return np.where(self.narrow_mask & self.content_mask)[0]

    @cached_property
    def narrow_non_content_indices(self) -> np.ndarray:
        return np.where(self.narrow_mask & ~self.content_mask)[0]


@dataclass
class ColumnInfo:
    label: int
    left: float
    right: float
    x_center: float
    indices: list[int]


def _xyxy_x_center(bbox: list[float]) -> float:
    return (bbox[0] + bbox[2]) / 2


def _xyxy_y_center(bbox: list[float]) -> float:
    return (bbox[1] + bbox[3]) / 2


def _xyxy_width(bbox: list[float]) -> float:
    return bbox[2] - bbox[0]
