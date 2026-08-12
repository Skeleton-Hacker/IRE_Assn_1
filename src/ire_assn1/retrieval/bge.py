from __future__ import annotations

import importlib
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ire_assn1.retrieval.indexes import ExactFaissIndex, FaissIndex, normalize_vectors
from ire_assn1.retrieval.types import Article, SearchHit


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> NDArray[np.float32]: ...


class _SentenceTransformerConstructor(Protocol):
    def __call__(
        self, model_name_or_path: str, *, revision: str | None, device: str | None
    ) -> _SentenceTransformerModel: ...


class BGEEncoder:
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        revision: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self.model_name = model
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self._model: _SentenceTransformerModel | None = None

    def _load(self) -> _SentenceTransformerModel:
        if self._model is None:
            try:
                module = importlib.import_module("sentence_transformers")
            except ImportError as error:
                raise RuntimeError("sentence-transformers is required for BGE encoding") from error
            constructor = cast(
                _SentenceTransformerConstructor, module.__dict__["SentenceTransformer"]
            )
            self._model = constructor(
                self.model_name,
                revision=self.revision,
                device=self.device,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize_vectors(vectors)

    def encode_articles(
        self, articles: Mapping[str, Article]
    ) -> tuple[tuple[str, ...], NDArray[np.float32]]:
        article_ids = tuple(
            sorted(article_id for article_id, article in articles.items() if article.text)
        )
        vectors = self.encode([articles[article_id].text for article_id in article_ids])
        return article_ids, vectors


class DenseRetriever:
    def __init__(
        self,
        article_ids: Sequence[str],
        vectors: ArrayLike,
        index: FaissIndex | None = None,
    ) -> None:
        normalized = normalize_vectors(vectors)
        if len(article_ids) != len(normalized):
            raise ValueError("Article IDs and vectors must have equal lengths")
        if len(set(article_ids)) != len(article_ids):
            raise ValueError("Article IDs must be unique")
        self.article_ids = tuple(article_ids)
        self.vectors = normalized
        self.positions = {
            article_id: position for position, article_id in enumerate(self.article_ids)
        }
        self.index = index if index is not None else ExactFaissIndex(self.article_ids, self.vectors)

    @property
    def system(self) -> str:
        return "bge"

    @property
    def represented_ids(self) -> frozenset[str]:
        return frozenset(self.article_ids)

    @property
    def index_size_bytes(self) -> int:
        return self.index.size_bytes

    def profile_vector(self, profile_article_ids: Sequence[str]) -> NDArray[np.float32] | None:
        positions = [
            self.positions[article_id]
            for article_id in profile_article_ids
            if article_id in self.positions
        ]
        if not positions:
            return None
        mean = self.vectors[positions].mean(axis=0, keepdims=True)
        normalized = normalize_vectors(mean)[0]
        if not np.any(normalized):
            return None
        return normalized

    def retrieve(
        self,
        profile_article_ids: Sequence[str],
        eligible_ids: Collection[str],
        k: int,
    ) -> tuple[SearchHit, ...]:
        query = self.profile_vector(profile_article_ids)
        if query is None or k <= 0:
            return ()
        eligible = set(eligible_ids)
        requested = min(len(self.article_ids), max(k, 1))
        hits: tuple[SearchHit, ...] = ()
        while requested:
            hits = tuple(
                hit for hit in self.index.search(query, requested) if hit.article_id in eligible
            )
            if len(hits) >= k or requested == len(self.article_ids):
                break
            requested = min(len(self.article_ids), max(requested * 2, k))
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.article_id))[:k])

    def score(
        self,
        profile_article_ids: Sequence[str],
        candidate_ids: Collection[str],
    ) -> dict[str, float]:
        query = self.profile_vector(profile_article_ids)
        if query is None:
            return {}
        return {
            article_id: float(np.dot(query, self.vectors[self.positions[article_id]]))
            for article_id in candidate_ids
            if article_id in self.positions
        }
