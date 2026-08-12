from __future__ import annotations

import math
from collections.abc import Sequence


def ranked_order(
    article_ids: Sequence[str], labels: Sequence[float], scores: Sequence[float]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if len(article_ids) != len(labels) or len(article_ids) != len(scores):
        raise ValueError("article_ids, labels, and scores must have equal lengths")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    order = sorted(range(len(article_ids)), key=lambda index: (-scores[index], article_ids[index]))
    return (
        tuple(article_ids[index] for index in order),
        tuple(float(labels[index]) for index in order),
    )


def auc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal lengths")
    positives = [score for label, score in zip(labels, scores, strict=True) if label > 0]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label <= 0]
    if not positives or not negatives:
        return None
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                concordance += 1.0
            elif positive == negative:
                concordance += 0.5
    return concordance / (len(positives) * len(negatives))


def mrr(ranked_labels: Sequence[float]) -> float | None:
    for rank, label in enumerate(ranked_labels, start=1):
        if label > 0:
            return 1.0 / rank
    return None


def ndcg_at_k(ranked_labels: Sequence[float], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    if any(label < 0 or not math.isfinite(label) for label in ranked_labels):
        raise ValueError("labels must be finite and non-negative")
    ideal = sorted(ranked_labels, reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return None
    return _dcg(ranked_labels[:k]) / ideal_dcg


def recall_at_k(
    recommended_ids: Sequence[str],
    relevant_ids: Sequence[str] | set[str] | frozenset[str],
    k: int,
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return None
    retrieved = set(recommended_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def _dcg(labels: Sequence[float]) -> float:
    return sum((2.0**label - 1.0) / math.log2(rank + 1.0) for rank, label in enumerate(labels, 1))
