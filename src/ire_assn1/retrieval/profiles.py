from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime

from ire_assn1.retrieval.types import Article, History, HistoryProfile, Impression, SearchHit


class ArticleEligibility(Collection[str]):
    def __init__(
        self,
        articles: Mapping[str, Article],
        impression_at: datetime,
        excluded_ids: Collection[str] = (),
    ) -> None:
        self.articles = articles
        self.impression_at = impression_at
        self.excluded_ids = frozenset(excluded_ids)

    def __contains__(self, article_id: object) -> bool:
        if not isinstance(article_id, str) or article_id in self.excluded_ids:
            return False
        article = self.articles.get(article_id)
        return article is not None and article.available_at <= self.impression_at

    def __iter__(self) -> Iterator[str]:
        return (article_id for article_id in self.articles if article_id in self)

    def __len__(self) -> int:
        return sum(1 for _ in self)


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
    allowed = represented_ids
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
    ranked_ids: tuple[str, ...] = ()

    @classmethod
    def from_impressions(
        cls,
        impressions: Iterable[Impression],
        article_ids: Collection[str] = (),
    ) -> PopularityModel:
        counts: Counter[str] = Counter()
        for impression in impressions:
            if impression.source_split == "train":
                counts.update(impression.clicked_ids)
        ranked_ids = tuple(
            sorted(article_ids, key=lambda article_id: (-counts[article_id], article_id))
        )
        return cls(counts=dict(counts), ranked_ids=ranked_ids)

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

    def rank_matching(
        self,
        predicate: Callable[[str], bool],
        excluded_ids: Collection[str] = (),
        k: int | None = None,
    ) -> tuple[SearchHit, ...]:
        excluded = set(excluded_ids)
        hits: list[SearchHit] = []
        for article_id in self.ranked_ids:
            if article_id in excluded or not predicate(article_id):
                continue
            hits.append(SearchHit(article_id, float(self.score(article_id))))
            if k is not None and len(hits) == k:
                break
        return tuple(hits)
