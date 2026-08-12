from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = 60
) -> tuple[tuple[str, float], ...]:
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings.values():
        seen: set[str] = set()
        for rank, article_id in enumerate(ranking, start=1):
            if article_id not in seen:
                scores[article_id] += 1.0 / (k + rank)
                seen.add(article_id)
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
