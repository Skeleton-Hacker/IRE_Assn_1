from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Article:
    article_id: str
    title: str
    abstract: str
    available_at: datetime

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in (self.title, self.abstract) if part.strip())


@dataclass(frozen=True, slots=True)
class History:
    user_id: str
    article_ids: tuple[str, ...]
    timestamps: tuple[datetime | None, ...] = ()

    def __post_init__(self) -> None:
        if self.timestamps and len(self.article_ids) != len(self.timestamps):
            raise ValueError("History article IDs and timestamps must have equal lengths")


@dataclass(frozen=True, slots=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime
    candidate_ids: tuple[str, ...]
    clicked_ids: tuple[str, ...]
    labels: tuple[int, ...] = ()
    source_split: str = ""

    def __post_init__(self) -> None:
        if self.labels and len(self.candidate_ids) != len(self.labels):
            raise ValueError("Impression candidate IDs and labels must have equal lengths")


@dataclass(frozen=True, slots=True)
class HistoryProfile:
    user_id: str
    article_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    article_id: str
    score: float


@dataclass(frozen=True, slots=True)
class ScoredArticle:
    article_id: str
    score: float
    rank: int
    label: int | None
    position: int | None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    impression_id: str
    user_id: str
    timestamp: datetime
    source_split: str
    system: str
    mode: str
    used_fallback: bool
    articles: tuple[ScoredArticle, ...]


class Retriever(Protocol):
    @property
    def system(self) -> str: ...

    @property
    def represented_ids(self) -> frozenset[str]: ...

    def retrieve(
        self,
        profile_article_ids: Sequence[str],
        eligible_ids: Collection[str],
        k: int,
    ) -> Sequence[SearchHit]: ...

    def score(
        self,
        profile_article_ids: Sequence[str],
        candidate_ids: Collection[str],
    ) -> Mapping[str, float]: ...
