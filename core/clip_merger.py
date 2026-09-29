"""Merge nearby clip candidates into coherent longer clips."""

import logging
from typing import List

from core.data_types import ClipCandidate

logger = logging.getLogger(__name__)


def merge_candidates(
    candidates: List[ClipCandidate],
    max_gap: float = 8.0,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    padding: float = 1.5,
) -> List[ClipCandidate]:
    """Merge nearby candidates into longer clips suitable for shorts.

    Args:
        candidates: Input candidates sorted by start time.
        max_gap: Max silence gap in seconds to merge two candidates.
        min_duration: Minimum clip duration; pad around if too short.
        max_duration: Maximum clip duration; split if too long.
        padding: Seconds to add before start & after end.

    Returns:
        Merged list of clip candidates.
    """
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c.start_time)
    groups: List[List[ClipCandidate]] = [[sorted_candidates[0]]]

    for candidate in sorted_candidates[1:]:
        last_group = groups[-1]
        last_end = last_group[-1].end_time

        if candidate.start_time - last_end <= max_gap:
            last_group.append(candidate)
        else:
            groups.append([candidate])

    merged: List[ClipCandidate] = []
    for group in groups:
        start = max(0.0, group[0].start_time - padding)
        end = group[-1].end_time + padding

        # Extend yang terlalu pendek
        if end - start < min_duration:
            end = start + min_duration

        # Potong yang kepanjangan
        if end - start > max_duration:
            end = start + max_duration

        best_score = max(c.score for c in group)
        text = " ".join(c.text for c in group if c.text).strip()
        volume = min(c.volume_dbfs for c in group if c.volume_dbfs is not None)

        merged.append(
            ClipCandidate(
                start_time=start,
                end_time=end,
                text=text,
                volume_dbfs=volume,
                score=best_score,
                reason=" | ".join(c.reason for c in group if c.reason)[:300],
                source="merged",
                metadata={
                    "merged_from": len(group),
                    "source_starts": [c.start_time for c in group],
                },
            )
        )

    logger.info("Merged %d candidates into %d clips", len(candidates), len(merged))
    return merged