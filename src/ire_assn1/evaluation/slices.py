from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

UserSlice = Literal["cold", "warm"]
PopularitySlice = Literal["head", "tail"]


def cold_warm_threshold(training_history_lengths: Sequence[int]) -> float:
    if not training_history_lengths:
        raise ValueError("training_history_lengths must not be empty")
    if any(length < 0 for length in training_history_lengths):
        raise ValueError("training history lengths must be non-negative")
    return float(statistics.median(training_history_lengths))


def user_slice(history_length: int, threshold: float) -> UserSlice:
    if history_length < 0 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("history length and threshold must be finite and non-negative")
    return "cold" if history_length <= threshold else "warm"


def head_article_ids(click_counts: Mapping[str, int], share: float = 0.8) -> frozenset[str]:
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    if any(count < 0 for count in click_counts.values()):
        raise ValueError("click counts must be non-negative")
    total = sum(click_counts.values())
    if total == 0:
        return frozenset()
    ordered = sorted(click_counts.items(), key=lambda item: (-item[1], item[0]))
    cutoff = total * share
    selected: set[str] = set()
    cumulative = 0
    for article_id, count in ordered:
        selected.add(article_id)
        cumulative += count
        if cumulative >= cutoff:
            break
    return frozenset(selected)


def article_slice(article_id: str, head_ids: frozenset[str]) -> PopularitySlice:
    return "head" if article_id in head_ids else "tail"


def target_slices(
    article_ids: Sequence[str], labels: Sequence[float], head_ids: frozenset[str]
) -> frozenset[PopularitySlice]:
    if len(article_ids) != len(labels):
        raise ValueError("article_ids and labels must have equal lengths")
    return frozenset(
        article_slice(article_id, head_ids)
        for article_id, label in zip(article_ids, labels, strict=True)
        if label > 0
    )
