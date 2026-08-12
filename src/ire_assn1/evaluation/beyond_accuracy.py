from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def diversity_at_k(
    article_ids: Sequence[str], embeddings: Mapping[str, Sequence[float]], k: int
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    vectors = [
        _normalized_vector(embeddings[article_id])
        for article_id in article_ids[:k]
        if article_id in embeddings
    ]
    if len(vectors) < 2:
        return None
    distances = [
        1.0 - float(np.dot(vectors[left], vectors[right]))
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    return float(np.mean(np.asarray(distances, dtype=np.float64)))


def novelty_at_k(
    article_ids: Sequence[str],
    click_counts: Mapping[str, int],
    k: int,
    alpha: float = 1.0,
    catalog_size: int | None = None,
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    selected = tuple(article_ids[:k])
    if not selected:
        return None
    if any(count < 0 for count in click_counts.values()):
        raise ValueError("click counts must be non-negative")
    size = catalog_size if catalog_size is not None else len(click_counts)
    if size <= 0:
        raise ValueError("catalog_size must be positive")
    total = sum(click_counts.values())
    denominator = total + alpha * size
    values = [
        -math.log2((click_counts.get(article_id, 0) + alpha) / denominator)
        for article_id in selected
    ]
    return sum(values) / len(values)


def coverage_at_k(
    recommendations: Iterable[Sequence[str]], exposed_article_ids: Iterable[str], k: int
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    exposed = set(exposed_article_ids)
    if not exposed:
        return None
    recommended = {article_id for ranking in recommendations for article_id in ranking[:k]}
    return len(recommended & exposed) / len(exposed)


def _normalized_vector(vector: Sequence[float]) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("embeddings must be finite, non-empty vectors")
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("embedding norm must be non-zero")
    return value / norm
