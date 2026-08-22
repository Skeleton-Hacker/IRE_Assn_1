from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

from tqdm.auto import tqdm

from ire_assn1.retrieval.profiles import (
    ArticleEligibility,
    PopularityModel,
    build_history_profile,
)
from ire_assn1.retrieval.types import (
    Article,
    EligibleArticle,
    History,
    Impression,
    RetrievalResult,
    Retriever,
    ScoredArticle,
    SearchHit,
)


def _fallback_hits(
    available_ids: EligibleArticle,
    popularity: PopularityModel,
    k: int,
    all_article_ids: Collection[str] = (),
) -> tuple[SearchHit, ...]:
    if isinstance(available_ids, ArticleEligibility):
        if popularity.ranked_ids:
            return popularity.rank_matching(available_ids.__contains__, k=k)
        return popularity.rank(
            tuple(article_id for article_id in all_article_ids if article_id in available_ids), k
        )
    return popularity.rank(available_ids, k)


def _complete_hits(
    primary: Sequence[SearchHit],
    available_ids: EligibleArticle,
    popularity: PopularityModel,
    k: int,
    all_article_ids: Collection[str] = (),
) -> tuple[SearchHit, ...]:
    selected = {hit.article_id for hit in primary}
    completed = list(primary)
    if isinstance(available_ids, ArticleEligibility):
        if popularity.ranked_ids:
            missing = popularity.rank_matching(available_ids.__contains__, excluded_ids=selected)
        else:
            missing = popularity.rank(
                tuple(
                    article_id
                    for article_id in all_article_ids
                    if article_id not in selected and article_id in available_ids
                )
            )
    else:
        missing = popularity.rank(set(available_ids) - selected)
    minimum = min((hit.score for hit in primary), default=0.0)
    for offset, hit in enumerate(missing, start=1):
        completed.append(SearchHit(hit.article_id, minimum - float(offset)))
        if len(completed) == k:
            break
    return tuple(completed[:k])


def history_for_impression(
    histories: Mapping[tuple[str, str], History], impression: Impression
) -> History | None:
    return histories.get((impression.impression_id, impression.source_split)) or histories.get(
        (impression.user_id, impression.source_split)
    )


def full_corpus_retrieval(
    impression: Impression,
    history: History | None,
    articles: Mapping[str, Article],
    retriever: Retriever,
    popularity: PopularityModel,
    history_length: int | None,
    k: int,
) -> RetrievalResult:
    if k < 0:
        raise ValueError("k must be non-negative")
    profile = build_history_profile(
        history,
        impression.timestamp,
        articles,
        history_length,
        retriever.represented_ids,
    )
    read_ids = set(history.article_ids) if history is not None else set()

    available = ArticleEligibility(articles, impression.timestamp, read_ids)

    used_fallback = not profile.article_ids
    if used_fallback:
        hits = _fallback_hits(available, popularity, k, articles.keys())
    else:
        primary = tuple(retriever.retrieve(profile.article_ids, available, k))
        used_fallback = not primary
        hits = primary
        if len(hits) < k:
            hits = _complete_hits(hits, available, popularity, k, articles.keys())
    clicked = set(impression.clicked_ids)
    scored = tuple(
        ScoredArticle(
            article_id=hit.article_id,
            score=hit.score,
            rank=rank,
            label=int(hit.article_id in clicked),
            position=None,
        )
        for rank, hit in enumerate(hits, start=1)
    )
    return RetrievalResult(
        impression_id=impression.impression_id,
        user_id=impression.user_id,
        timestamp=impression.timestamp,
        source_split=impression.source_split,
        system=retriever.system,
        mode="full_corpus",
        used_fallback=used_fallback,
        articles=scored,
    )


def impression_candidate_scoring(
    impression: Impression,
    history: History | None,
    articles: Mapping[str, Article],
    retriever: Retriever,
    popularity: PopularityModel,
    history_length: int | None,
) -> RetrievalResult:
    profile = build_history_profile(
        history,
        impression.timestamp,
        articles,
        history_length,
        retriever.represented_ids,
    )
    positions = {
        article_id: position for position, article_id in enumerate(impression.candidate_ids)
    }
    labels = {
        article_id: impression.labels[position]
        for position, article_id in enumerate(impression.candidate_ids)
        if impression.labels
    }
    available = tuple(
        article_id
        for article_id in impression.candidate_ids
        if article_id in articles and articles[article_id].available_at <= impression.timestamp
    )
    used_fallback = not profile.article_ids
    if used_fallback:
        hits = _fallback_hits(available, popularity, len(available))
    else:
        scores = retriever.score(profile.article_ids, available)
        primary = tuple(
            sorted(
                (SearchHit(article_id, score) for article_id, score in scores.items()),
                key=lambda hit: (-hit.score, hit.article_id),
            )
        )
        used_fallback = not primary
        hits = _complete_hits(primary, available, popularity, len(available))
    scored = tuple(
        ScoredArticle(
            article_id=hit.article_id,
            score=hit.score,
            rank=rank,
            label=labels.get(hit.article_id, int(hit.article_id in impression.clicked_ids)),
            position=positions[hit.article_id],
        )
        for rank, hit in enumerate(hits, start=1)
    )
    return RetrievalResult(
        impression_id=impression.impression_id,
        user_id=impression.user_id,
        timestamp=impression.timestamp,
        source_split=impression.source_split,
        system=retriever.system,
        mode="impression_candidates",
        used_fallback=used_fallback,
        articles=scored,
    )


def select_history_length(
    scores: Mapping[int | None, float],
    candidates: Sequence[int | None] = (5, 10, 20, None),
) -> int | None:
    if not scores:
        raise ValueError("History-length scores cannot be empty")
    invalid = [length for length, score in scores.items() if not math.isfinite(score)]
    if invalid:
        raise ValueError(f"History-length scores must be finite: {invalid}")
    available = [length for length in candidates if length in scores]
    if not available:
        raise ValueError("No configured history length has a score")
    return max(available, key=lambda length: scores[length])


def evaluate_history_lengths(
    impressions: Iterable[Impression],
    histories: Mapping[tuple[str, str], History],
    articles: Mapping[str, Article],
    retriever: Retriever,
    popularity: PopularityModel,
    candidates: Sequence[int | None],
    k: int = 100,
) -> tuple[int | None, dict[int | None, float]]:
    validation = tuple(
        impression for impression in impressions if impression.source_split == "validation"
    )
    if not validation:
        raise ValueError("At least one validation impression is required")
    scores: dict[int | None, float] = {}
    for history_length in candidates:
        recalls: list[float] = []
        for impression in tqdm(
            validation,
            desc=f"{retriever.system} validation history={history_length}",
            unit="impression",
        ):
            relevant = set(impression.clicked_ids)
            if not relevant:
                continue
            result = full_corpus_retrieval(
                impression,
                history_for_impression(histories, impression),
                articles,
                retriever,
                popularity,
                history_length,
                k,
            )
            retrieved = {article.article_id for article in result.articles}
            recalls.append(len(relevant & retrieved) / len(relevant))
        scores[history_length] = sum(recalls) / len(recalls) if recalls else 0.0
    return select_history_length(scores, candidates), scores


def evaluate_history_lengths_stream(
    impressions: Callable[[], Iterable[Impression]],
    histories: Mapping[tuple[str, str], History],
    articles: Mapping[str, Article],
    retriever: Retriever,
    popularity: PopularityModel,
    candidates: Sequence[int | None],
    k: int = 100,
    max_impressions: int | None = None,
) -> tuple[int | None, dict[int | None, float]]:
    if max_impressions is not None and max_impressions <= 0:
        raise ValueError("max_impressions must be positive when provided")
    scores: dict[int | None, float] = {}
    for history_length in candidates:
        recalls: list[float] = []
        processed = 0
        for impression in tqdm(
            (value for value in impressions() if value.source_split == "validation"),
            desc=f"{retriever.system} validation history={history_length}",
            unit="impression",
        ):
            if max_impressions is not None and processed >= max_impressions:
                break
            processed += 1
            relevant = set(impression.clicked_ids)
            if not relevant:
                continue
            result = full_corpus_retrieval(
                impression,
                history_for_impression(histories, impression),
                articles,
                retriever,
                popularity,
                history_length,
                k,
            )
            retrieved = {article.article_id for article in result.articles}
            recalls.append(len(relevant & retrieved) / len(relevant))
        scores[history_length] = sum(recalls) / len(recalls) if recalls else 0.0
    return select_history_length(scores, candidates), scores
