from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

from ire_assn1.evaluation.config import EvaluationRunConfig
from ire_assn1.evaluation.types import EvaluationData, RankingRecord, RecommendationRecord
from ire_assn1.paths import project_path


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def _selected_history_length(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("history_length")
    if value is not None and not isinstance(value, int):
        raise ValueError(f"Invalid history length in {path}")
    return value


def _history_lengths(
    rows: Iterable[dict[str, Any]], selected: int | None
) -> tuple[dict[tuple[str, str], int], tuple[int, ...]]:
    lengths: dict[tuple[str, str], int] = {}
    training: list[int] = []
    for row in rows:
        source_split = str(row["source_split"])
        length = len(row.get("article_ids") or [])
        effective = min(length, selected) if selected is not None else length
        lengths[(str(row["user_id"]), source_split)] = effective
        if source_split == "train":
            training.append(length)
    return lengths, tuple(training)


def _group_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source_split") == "test":
            groups[str(row["impression_id"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (int(row["rank"]), str(row["article_id"])))
    return groups


def _rankings(
    groups: dict[str, list[dict[str, Any]]],
    histories: dict[tuple[str, str], int],
) -> tuple[RankingRecord, ...]:
    return tuple(
        RankingRecord(
            impression_id=impression_id,
            user_id=str(rows[0]["user_id"]),
            article_ids=tuple(str(row["article_id"]) for row in rows),
            labels=tuple(float(row["label"] or 0) for row in rows),
            scores=tuple(float(row["score"]) for row in rows),
            history_length=histories.get((str(rows[0]["user_id"]), "test"), 0),
        )
        for impression_id, rows in sorted(groups.items())
        if rows
    )


def _recommendations(
    groups: dict[str, list[dict[str, Any]]],
    histories: dict[tuple[str, str], int],
    relevant: dict[str, frozenset[str]],
) -> tuple[RecommendationRecord, ...]:
    return tuple(
        RecommendationRecord(
            impression_id=impression_id,
            user_id=str(rows[0]["user_id"]),
            article_ids=tuple(str(row["article_id"]) for row in rows),
            relevant_ids=relevant.get(impression_id, frozenset()),
            history_length=histories.get((str(rows[0]["user_id"]), "test"), 0),
        )
        for impression_id, rows in sorted(groups.items())
        if rows
    )


def _training_clicks(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("source_split") == "train" and int(row.get("label") or 0) > 0:
            counts[str(row["article_id"])] += int(row["label"])
    return dict(counts)


def _embeddings(path: Path, article_ids: set[str]) -> dict[str, tuple[float, ...]]:
    vectors_path = path / "article_embeddings.npy"
    ids_path = path / "article_embedding_ids.json"
    if not vectors_path.is_file() or not ids_path.is_file():
        return {}
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path, mmap_mode="r")
    if not isinstance(ids, list) or len(ids) != len(vectors):
        raise ValueError(f"Invalid embedding cache below {path}")
    return {
        str(article_id): tuple(float(value) for value in vectors[index])
        for index, article_id in enumerate(ids)
        if str(article_id) in article_ids
    }


def _future_popularity_diagnostics(
    groups: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, dict[str, float]]]:
    future_counts: Counter[str] = Counter(
        str(row["article_id"])
        for rows in groups.values()
        for row in rows
        if int(row["label"] or 0) > 0
    )
    rankings: dict[str, dict[str, tuple[str, ...]]] = {}
    labels: dict[str, dict[str, float]] = {}
    for impression_id, rows in sorted(groups.items()):
        model = tuple(str(row["article_id"]) for row in rows)
        future = tuple(
            sorted(model, key=lambda article_id: (-future_counts[article_id], article_id))
        )
        rankings[impression_id] = {"model": model, "future_popularity": future}
        labels[impression_id] = {str(row["article_id"]): float(row["label"] or 0) for row in rows}
    return rankings, labels


def load_retrieval_evaluation_data(run: EvaluationRunConfig) -> EvaluationData:
    processed = project_path("data") / "processed" / run.dataset / run.variant
    retrieval = project_path("data") / "retrieval" / run.dataset / run.variant / run.system
    bge_retrieval = project_path("data") / "retrieval" / run.dataset / run.variant / "bge"
    selected = _selected_history_length(retrieval / "selection.json")
    history_lengths, training_history_lengths = _history_lengths(
        _rows(processed / "histories.parquet"), selected
    )
    candidate_groups = _group_predictions(_rows(retrieval / "impression_candidates.parquet"))
    full_groups = _group_predictions(_rows(retrieval / "full_corpus.parquet"))
    candidates = _rows(processed / "candidates.parquet")
    impressions = _rows(processed / "impressions.parquet")
    relevant = {
        str(row["impression_id"]): frozenset(str(item) for item in row.get("clicked_ids") or [])
        for row in impressions
        if row.get("source_split") == "test"
    }
    rankings = _rankings(candidate_groups, history_lengths)
    recommendations = _recommendations(full_groups, history_lengths, relevant)
    exposed = {str(row["article_id"]) for row in candidates if row.get("source_split") == "test"}
    recommended = {
        article_id for record in recommendations for article_id in record.article_ids[:10]
    }
    diagnostic_rankings, diagnostic_labels = _future_popularity_diagnostics(candidate_groups)
    return EvaluationData(
        rankings=rankings,
        recommendations=recommendations,
        training_history_lengths=training_history_lengths,
        training_clicks=_training_clicks(candidates),
        embeddings=_embeddings(bge_retrieval, recommended),
        exposed_article_ids=frozenset(exposed),
        diagnostic_rankings=diagnostic_rankings,
        diagnostic_labels=diagnostic_labels,
    )
