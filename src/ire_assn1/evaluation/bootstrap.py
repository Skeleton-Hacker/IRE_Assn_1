from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from ire_assn1.evaluation.types import ConfidenceInterval


def clustered_bootstrap_interval(
    observations: Sequence[tuple[str, float]],
    samples: int = 1000,
    seed: int = 146,
    confidence: float = 0.95,
    statistic: Callable[[Sequence[float]], float] | None = None,
    bias_correct: bool = False,
) -> ConfidenceInterval | None:
    result = clustered_bootstrap(
        observations,
        samples=samples,
        seed=seed,
        confidence=confidence,
        statistic=statistic,
        bias_correct=bias_correct,
    )
    return result[0] if result is not None else None


def clustered_bootstrap[T](
    observations: Sequence[tuple[str, T]],
    samples: int = 1000,
    seed: int = 146,
    confidence: float = 0.95,
    statistic: Callable[[Sequence[T]], float] | None = None,
    bias_correct: bool = False,
) -> tuple[ConfidenceInterval, tuple[float, ...]] | None:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not observations:
        return None
    grouped: dict[str, list[T]] = defaultdict(list)
    for user_id, value in observations:
        if isinstance(value, int | float) and not np.isfinite(value):
            raise ValueError("bootstrap observations must be finite")
        grouped[user_id].append(value)
    user_ids = sorted(grouped)
    reducer = statistic if statistic is not None else cast(Callable[[Sequence[T]], float], _mean)
    original = reducer([value for _, value in observations])
    if not np.isfinite(original):
        raise ValueError("bootstrap statistic must be finite")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(user_ids, size=len(user_ids), replace=True)
        values = [value for user_id in sampled for value in grouped[str(user_id)]]
        estimate = reducer(values)
        if not np.isfinite(estimate):
            raise ValueError("bootstrap statistic must be finite")
        estimates[index] = estimate
    if bias_correct:
        estimates += original - float(np.mean(estimates))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return (
        ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            confidence=confidence,
            samples=samples,
            seed=seed,
        ),
        tuple(float(value) for value in estimates),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
