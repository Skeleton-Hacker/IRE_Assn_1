from __future__ import annotations

import importlib
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ire_assn1.retrieval.normalization import LanguageNormalizer
from ire_assn1.retrieval.types import Article, SearchHit


class _BM25Backend(Protocol):
    def index(self, corpus_tokens: list[list[str]], show_progress: bool = False) -> None: ...

    def retrieve(
        self,
        query_tokens: list[list[str]],
        corpus: Sequence[str],
        k: int,
        show_progress: bool = False,
    ) -> tuple[NDArray[np.str_], NDArray[np.float32]]: ...


class _BM25Constructor(Protocol):
    def __call__(self, *, k1: float, b: float) -> _BM25Backend: ...


def _load_bm25() -> _BM25Constructor:
    try:
        module = importlib.import_module("bm25s")
    except ImportError as error:
        raise RuntimeError("bm25s is required to build a lexical index") from error
    return cast(_BM25Constructor, module.__dict__["BM25"])


class BM25Retriever:
    def __init__(
        self,
        articles: Mapping[str, Article],
        language: str,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters are out of range")
        self.normalizer = LanguageNormalizer(language)
        tokenized = {
            article_id: list(self.normalizer.tokens(article.text))
            for article_id, article in articles.items()
            if article.text.strip()
        }
        self.article_ids = tuple(sorted(tokenized))
        self.documents = tokenized
        self._backend = _load_bm25()(k1=k1, b=b)
        if self.article_ids:
            self._backend.index(
                [self.documents[article_id] for article_id in self.article_ids],
                show_progress=True,
            )

    @property
    def system(self) -> str:
        return "bm25"

    @property
    def represented_ids(self) -> frozenset[str]:
        return frozenset(self.article_ids)

    @property
    def index_size_bytes(self) -> int:
        return sum(
            len(article_id.encode())
            + sum(len(token.encode()) for token in self.documents[article_id])
            for article_id in self.article_ids
        )

    def _query(self, profile_article_ids: Sequence[str]) -> list[str]:
        return [
            token
            for article_id in profile_article_ids
            for token in self.documents.get(article_id, ())
        ]

    def _all_scores(self, profile_article_ids: Sequence[str]) -> dict[str, float]:
        query = self._query(profile_article_ids)
        if not query or not self.article_ids:
            return {}
        documents, scores = self._backend.retrieve(
            [query],
            corpus=self.article_ids,
            k=len(self.article_ids),
            show_progress=True,
        )
        return {
            str(article_id): float(score)
            for article_id, score in zip(documents[0], scores[0], strict=True)
        }

    def retrieve(
        self,
        profile_article_ids: Sequence[str],
        eligible_ids: Collection[str],
        k: int,
    ) -> tuple[SearchHit, ...]:
        if k <= 0:
            return ()
        eligible = set(eligible_ids)
        hits = (
            SearchHit(article_id, score)
            for article_id, score in self._all_scores(profile_article_ids).items()
            if article_id in eligible
        )
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.article_id))[:k])

    def score(
        self,
        profile_article_ids: Sequence[str],
        candidate_ids: Collection[str],
    ) -> dict[str, float]:
        candidates = set(candidate_ids)
        return {
            article_id: score
            for article_id, score in self._all_scores(profile_article_ids).items()
            if article_id in candidates
        }
