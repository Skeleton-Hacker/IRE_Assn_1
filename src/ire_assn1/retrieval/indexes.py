from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ire_assn1.retrieval.types import SearchHit


class _FaissIndex(Protocol):
    hnsw: Any

    def add(self, vectors: NDArray[np.float32]) -> None: ...

    def search(
        self, queries: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...


def _load_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ImportError as error:
        raise RuntimeError("FAISS is required to build a semantic index") from error


def normalize_vectors(vectors: ArrayLike) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class FaissIndex:
    def __init__(self, article_ids: Sequence[str], vectors: ArrayLike) -> None:
        normalized = normalize_vectors(vectors)
        if len(article_ids) != len(normalized):
            raise ValueError("Article IDs and vectors must have equal lengths")
        if len(set(article_ids)) != len(article_ids):
            raise ValueError("Article IDs must be unique")
        self.article_ids = tuple(article_ids)
        self.vectors = normalized
        self._index = self._build(self.vectors.shape[1])
        if len(self.vectors):
            self._index.add(self.vectors)

    def _build(self, dimensions: int) -> _FaissIndex:
        raise NotImplementedError

    def search(self, query: ArrayLike, k: int) -> tuple[SearchHit, ...]:
        if k <= 0 or not self.article_ids:
            return ()
        query_matrix = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query_matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query and index dimensions must match")
        normalized_query = normalize_vectors(query_matrix)
        scores, positions = self._index.search(normalized_query, min(k, len(self.article_ids)))
        hits = [
            SearchHit(self.article_ids[int(position)], float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0
        ]
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.article_id)))

    @property
    def size_bytes(self) -> int:
        serialized = _load_faiss().serialize_index(self._index)
        return int(serialized.nbytes)


class ExactFaissIndex(FaissIndex):
    def _build(self, dimensions: int) -> _FaissIndex:
        return _load_faiss().IndexFlatIP(dimensions)


class HNSWFaissIndex(FaissIndex):
    def __init__(
        self,
        article_ids: Sequence[str],
        vectors: ArrayLike,
        m: int = 32,
        ef_search: int = 128,
    ) -> None:
        if m <= 0 or ef_search <= 0:
            raise ValueError("HNSW parameters must be positive")
        self.m = m
        self.ef_search = ef_search
        super().__init__(article_ids, vectors)

    def _build(self, dimensions: int) -> _FaissIndex:
        faiss = _load_faiss()
        index = faiss.IndexHNSWFlat(dimensions, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.ef_search
        return index
