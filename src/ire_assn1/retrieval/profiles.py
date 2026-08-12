from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from ire_assn1.retrieval.types import Article, History, HistoryProfile, Impression, SearchHit


def eligible_article_ids(
    articles: Mapping[str, Article],
    impression_at: datetime,
    excluded_ids: Collection[str] = (),
) -> tuple[str, ...]:
    excluded = set(excluded_ids)
    return tuple(
        sorted(
            article_id
            for article_id, article in articles.items()
            if article.available_at <= impression_at and article_id not in excluded
        )
    )


def build_history_profile(
    history: History | None,
    impression_at: datetime,
    articles: Mapping[str, Article],
    history_length: int | None,
    represented_ids: Collection[str] | None = None,
) -> HistoryProfile:
    if history_length is not None and history_length <= 0:
        raise ValueError("History length must be positive or None")
    if history is None:
        return HistoryProfile(user_id="", article_ids=())
    allowed = None if represented_ids is None else set(represented_ids)
    timestamps = history.timestamps or (None,) * len(history.article_ids)
    selected: list[str] = []
    for article_id, clicked_at in zip(history.article_ids, timestamps, strict=True):
        article = articles.get(article_id)
        if article is None or article.available_at > impression_at:
            continue
        if clicked_at is not None and clicked_at > impression_at:
            continue
        if allowed is not None and article_id not in allowed:
            continue
        selected.append(article_id)
    if history_length is not None:
        selected = selected[-history_length:]
    return HistoryProfile(user_id=history.user_id, article_ids=tuple(selected))


@dataclass(frozen=True, slots=True)
class PopularityModel:
    counts: Mapping[str, int]

    @classmethod
    def from_impressions(cls, impressions: Iterable[Impression]) -> PopularityModel:
        counts: Counter[str] = Counter()
        for impression in impressions:
            if impression.source_split == "train":
                counts.update(impression.clicked_ids)
        return cls(counts=dict(counts))

    def rank(self, article_ids: Collection[str], k: int | None = None) -> tuple[SearchHit, ...]:
        if k is not None and k < 0:
            raise ValueError("k must be non-negative or None")
        ranked = sorted(
            set(article_ids), key=lambda article_id: (-self.score(article_id), article_id)
        )
        if k is not None:
            ranked = ranked[:k]
        return tuple(SearchHit(article_id, float(self.score(article_id))) for article_id in ranked)

    def score(self, article_id: str) -> int:
        return self.counts.get(article_id, 0)
