from __future__ import annotations

import importlib
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ire_assn1.retrieval.normalization import LanguageNormalizer
from ire_assn1.retrieval.types import Article, EligibleArticle, SearchHit


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
        self.term_counts = {article_id: Counter(tokens) for article_id, tokens in tokenized.items()}
        self.document_lengths = {
            article_id: len(tokens) for article_id, tokens in tokenized.items()
        }
        self.average_document_length = (
            sum(self.document_lengths.values()) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized.values():
            document_frequency.update(set(tokens))
        self.document_frequency = document_frequency
        document_count = len(self.article_ids)
        self.inverse_document_frequency = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._backend = _load_bm25()(k1=k1, b=b)
        self.k1 = k1
        self.b = b
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

    def _search(
        self,
        profile_article_ids: Sequence[str],
        eligible_ids: EligibleArticle,
        k: int,
    ) -> tuple[SearchHit, ...]:
        query = self._query(profile_article_ids)
        if not query or not self.article_ids or k <= 0:
            return ()
        requested = min(len(self.article_ids), max(k, 1))
        while requested:
            documents, scores = self._backend.retrieve(
                [query],
                corpus=self.article_ids,
                k=requested,
                show_progress=False,
            )
            hits = tuple(
                SearchHit(str(article_id), float(score))
                for article_id, score in zip(documents[0], scores[0], strict=True)
                if str(article_id) in eligible_ids
            )
            if len(hits) >= k or requested == len(self.article_ids):
                return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.article_id))[:k])
            requested = min(len(self.article_ids), max(requested * 2, k))
        return ()

    def retrieve(
        self,
        profile_article_ids: Sequence[str],
        eligible_ids: EligibleArticle,
        k: int,
    ) -> tuple[SearchHit, ...]:
        if k <= 0:
            return ()
        return self._search(profile_article_ids, eligible_ids, k)

    def score(
        self,
        profile_article_ids: Sequence[str],
        candidate_ids: Collection[str],
    ) -> dict[str, float]:
        query_terms = set(self._query(profile_article_ids))
        if not query_terms or not self.article_ids or self.average_document_length == 0:
            return {
                article_id: 0.0 for article_id in set(candidate_ids) if article_id in self.documents
            }
        scores: dict[str, float] = {}
        for article_id in set(candidate_ids):
            term_counts = self.term_counts.get(article_id)
            if term_counts is None:
                continue
            document_length = self.document_lengths[article_id]
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                inverse_document_frequency = self.inverse_document_frequency[term]
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / self.average_document_length
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / denominator
            scores[article_id] = score
        return scores
