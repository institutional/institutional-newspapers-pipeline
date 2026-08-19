import heapq

import peewee

from models.scan import Scan
from models.crop import Crop


def distribute_to_gpus(
    item_ids: list[int],
    weights: dict[int, int],
    num_gpus: int,
) -> list[list[int]]:
    """Distributes items across GPUs using largest-first greedy bin packing.

    Assigns each item (heaviest first) to the GPU with the lowest cumulative weight,
    producing more balanced workloads than simple round-robin.
    """
    if num_gpus <= 0:
        return []

    # Sort items by weight descending — largest-first gives better bin packing
    sorted_ids = sorted(item_ids, key=lambda x: weights.get(x, 1), reverse=True)

    # Min-heap of (cumulative_weight, gpu_index)
    heap: list[tuple[int, int]] = [(0, i) for i in range(num_gpus)]
    chunks: list[list[int]] = [[] for _ in range(num_gpus)]

    for item_id in sorted_ids:
        weight = weights.get(item_id, 1)
        cumulative, gpu_index = heapq.heappop(heap)
        chunks[gpu_index].append(item_id)
        heapq.heappush(heap, (cumulative + weight, gpu_index))

    return chunks


def get_scan_counts_by_issue(issue_ids: list[int]) -> dict[int, int]:
    """Returns {issue_id: scan_count} for the given issues."""
    if not issue_ids:
        return {}

    rows = (
        Scan.select(Scan.issue, peewee.fn.COUNT(Scan.id).alias("cnt"))
        .where(Scan.issue << issue_ids)
        .group_by(Scan.issue)
        .tuples()
    )
    return {issue_id: cnt for issue_id, cnt in rows}


def get_crop_counts_by_item(item_ids: list[int]) -> dict[int, int]:
    """Returns {pipeline_batch_item_id: crop_count} for the given items."""
    if not item_ids:
        return {}

    rows = (
        Crop.select(Crop.pipeline_batch_item, peewee.fn.COUNT(Crop.id).alias("cnt"))
        .where(Crop.pipeline_batch_item << item_ids)
        .group_by(Crop.pipeline_batch_item)
        .tuples()
    )
    return {item_id: cnt for item_id, cnt in rows}
