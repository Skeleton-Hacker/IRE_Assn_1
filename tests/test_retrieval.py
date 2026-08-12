from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from ire_assn1.retrieval import (
    Article,
    BGEEncoder,
    DenseRetriever,
    History,
    Impression,
    LanguageNormalizer,
    PopularityModel,
    SearchHit,
    build_history_profile,
    evaluate_history_lengths,
    full_corpus_retrieval,
    impression_candidate_scoring,
    select_history_length,
)
from ire_assn1.retrieval.bm25 import BM25Retriever
from ire_assn1.retrieval.indexes import ExactFaissIndex, FaissIndex, HNSWFaissIndex
from ire_assn1.retrieval.types import Retriever

NOW = datetime(2026, 8, 12, 12, tzinfo=UTC)


class StaticRetriever:
    def __init__(self, scores: Mapping[str, float], represented: Collection[str]) -> None:
        self.scores = dict(scores)
        self._represented = frozenset(represented)

    @property
    def system(self) -> str:
        return "static"

    @property
    def represented_ids(self) -> frozenset[str]:
        return self._represented

    def retrieve(
        self,
        profile_article_ids: Sequence[str],
        eligible_ids: Collection[str],
        k: int,
    ) -> tuple[SearchHit, ...]:
        eligible = set(eligible_ids)
        return tuple(
            sorted(
                (
                    SearchHit(article_id, score)
                    for article_id, score in self.scores.items()
                    if article_id in eligible
                ),
                key=lambda hit: (-hit.score, hit.article_id),
            )[:k]
        )

    def score(
        self,
        profile_article_ids: Sequence[str],
        candidate_ids: Collection[str],
    ) -> dict[str, float]:
        candidates = set(candidate_ids)
        return {
            article_id: score
            for article_id, score in self.scores.items()
            if article_id in candidates
        }


class MatrixIndex:
    def __init__(self, article_ids: Sequence[str], vectors: NDArray[np.float32]) -> None:
        self.article_ids = tuple(article_ids)
        self.vectors = vectors

    def search(self, query: ArrayLike, k: int) -> tuple[SearchHit, ...]:
        scores = self.vectors @ np.asarray(query, dtype=np.float32)
        hits = [
            SearchHit(article_id, float(score))
            for article_id, score in zip(self.article_ids, scores, strict=True)
        ]
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.article_id))[:k])


@pytest.fixture
def articles() -> dict[str, Article]:
    return {
        "a": Article("a", "Alpha", "first", NOW - timedelta(days=3)),
        "b": Article("b", "Beta", "second", NOW - timedelta(days=2)),
        "c": Article("c", "Gamma", "third", NOW - timedelta(days=1)),
        "d": Article("d", "Delta", "fourth", NOW + timedelta(hours=1)),
        "e": Article("e", "", "", NOW - timedelta(days=1)),
    }


def test_language_normalization_preserves_danish_diacritics() -> None:
    normalizer = LanguageNormalizer("da")
    assert normalizer.tokens("På Øen er æbler OG blåbær") == ("øen", "æbler", "blåbær")


def test_language_normalization_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        LanguageNormalizer("fr")


def test_history_profile_is_recent_represented_and_serving_time_safe(
    articles: Mapping[str, Article],
) -> None:
    history = History(
        "u",
        ("a", "missing", "b", "d", "c"),
        (
            NOW - timedelta(days=3),
            NOW - timedelta(days=2),
            NOW - timedelta(days=2),
            NOW - timedelta(hours=1),
            NOW + timedelta(hours=2),
        ),
    )
    profile = build_history_profile(history, NOW, articles, 1, {"a", "b", "c", "d"})
    assert profile.article_ids == ("b",)


def test_full_corpus_fallback_excludes_future_and_read_articles(
    articles: Mapping[str, Article],
) -> None:
    impression = Impression("i", "u", NOW, (), ("c",), source_split="validation")
    history = History("u", ("a",), (NOW - timedelta(days=1),))
    retriever = StaticRetriever({}, articles)
    popularity = PopularityModel({"b": 4, "c": 4, "d": 10, "e": 1})
    result = full_corpus_retrieval(impression, history, articles, retriever, popularity, 5, 10)
    assert result.used_fallback
    assert [item.article_id for item in result.articles] == ["b", "c", "e"]
    assert [item.rank for item in result.articles] == [1, 2, 3]
    assert [item.label for item in result.articles] == [0, 1, 0]


def test_candidate_scoring_uses_primary_scores_then_missing_representation(
    articles: Mapping[str, Article],
) -> None:
    impression = Impression("i", "u", NOW, ("b", "c", "e", "d"), ("c",), (0, 1, 0, 0))
    history = History("u", ("a",))
    retriever = StaticRetriever({"b": 0.5, "c": 0.5}, {"a", "b", "c"})
    popularity = PopularityModel({"e": 20, "b": 2, "c": 1})
    result = impression_candidate_scoring(
        impression, history, articles, retriever, popularity, None
    )
    assert not result.used_fallback
    assert [item.article_id for item in result.articles] == ["b", "c", "e"]
    assert [item.position for item in result.articles] == [0, 1, 2]
    assert result.articles[2].score < result.articles[1].score


def test_dense_retriever_normalizes_profiles_and_scores() -> None:
    article_ids = ("a", "b", "c")
    vectors = np.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    index = cast(
        FaissIndex, MatrixIndex(article_ids, vectors / np.linalg.norm(vectors, axis=1)[:, None])
    )
    retriever = DenseRetriever(article_ids, vectors, index)
    scores = retriever.score(("a", "b"), {"a", "b", "c"})
    assert scores["c"] == pytest.approx(1.0)
    assert scores["a"] == pytest.approx(2**-0.5)
    assert [hit.article_id for hit in retriever.retrieve(("a",), {"b", "c"}, 2)] == ["c", "b"]


def test_bge_encoder_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("ire_assn1.retrieval.bge.importlib.import_module", unavailable)
    encoder = BGEEncoder()
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        encoder.encode(["text"])


def test_history_length_selection_prefers_configured_order_on_ties() -> None:
    assert select_history_length({5: 0.7, 10: 0.7, 20: 0.6, None: 0.7}) == 5


def test_history_length_evaluation_uses_validation_only(
    articles: Mapping[str, Article],
) -> None:
    impressions = (
        Impression("train", "u", NOW, (), ("b",), source_split="train"),
        Impression("validation", "u", NOW, (), ("c",), source_split="validation"),
    )
    histories = {
        ("u", "train"): History("u", ("a",)),
        ("u", "validation"): History("u", ("a",)),
    }
    retriever = cast(Retriever, StaticRetriever({"c": 1.0, "b": 0.5}, {"a", "b", "c"}))
    selected, scores = evaluate_history_lengths(
        impressions,
        histories,
        articles,
        retriever,
        PopularityModel.from_impressions(impressions),
        (5, None),
        k=1,
    )
    assert selected == 5
    assert scores == {5: 1.0, None: 1.0}


def test_popularity_counts_only_training_clicks() -> None:
    impressions = (
        Impression("a", "u", NOW, (), ("x",), source_split="train"),
        Impression("b", "u", NOW, (), ("y",), source_split="validation"),
    )
    popularity = PopularityModel.from_impressions(impressions)
    assert popularity.counts == {"x": 1}
    assert [hit.article_id for hit in popularity.rank({"x", "y"})] == ["x", "y"]


def test_bm25_backend_ranks_matching_article_first(
    articles: Mapping[str, Article],
) -> None:
    retriever = BM25Retriever(articles, "en")
    hits = retriever.retrieve(("a",), {"b", "c"}, 2)
    assert [hit.article_id for hit in hits] == ["b", "c"]
    assert retriever.index_size_bytes > 0


def test_faiss_indexes_return_exact_dense_order() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    for index in (
        ExactFaissIndex(("a", "b", "c"), vectors),
        HNSWFaissIndex(("a", "b", "c"), vectors),
    ):
        assert [hit.article_id for hit in index.search(vectors[0], 3)] == ["a", "c", "b"]
        assert index.size_bytes > 0
