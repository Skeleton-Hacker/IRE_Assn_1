from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ire_assn1.evaluation.beyond_accuracy import (
    coverage_at_k,
    diversity_at_k,
    novelty_at_k,
)
from ire_assn1.evaluation.bootstrap import clustered_bootstrap_interval
from ire_assn1.evaluation.config import (
    EvaluationConfig,
    EvaluationRunConfig,
    load_evaluation_config,
)
from ire_assn1.evaluation.ranking import auc, mrr, ndcg_at_k, ranked_order, recall_at_k
from ire_assn1.evaluation.rrf import reciprocal_rank_fusion
from ire_assn1.evaluation.serialization import write_evaluation_result
from ire_assn1.evaluation.slices import (
    cold_warm_threshold,
    head_article_ids,
    target_slices,
    user_slice,
)
from ire_assn1.evaluation.types import (
    DiagnosticResult,
    EvaluationData,
    EvaluationResult,
    MetricEstimate,
    RankingRecord,
    RecommendationRecord,
)


class RankingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impression_id: str
    user_id: str
    article_ids: list[str]
    labels: list[float]
    scores: list[float]
    history_length: int


class RecommendationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impression_id: str
    user_id: str
    article_ids: list[str]
    relevant_ids: list[str]
    history_length: int


class EvaluationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rankings: list[RankingPayload]
    recommendations: list[RecommendationPayload]
    training_history_lengths: list[int]
    training_clicks: dict[str, int]
    embeddings: dict[str, list[float]] = Field(default_factory=dict)
    exposed_article_ids: list[str] = Field(default_factory=list)
    diagnostic_rankings: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    diagnostic_labels: dict[str, dict[str, float]] = Field(default_factory=dict)


class UserRecord(Protocol):
    @property
    def user_id(self) -> str: ...


def evaluate(data: EvaluationData, run: EvaluationRunConfig) -> EvaluationResult:
    config = run.evaluation
    threshold = cold_warm_threshold(data.training_history_lengths)
    head_ids = head_article_ids(data.training_clicks, config.head_click_share)
    metrics: list[MetricEstimate] = []
    metrics.extend(_ranking_metrics(data.rankings, config, threshold, head_ids))
    metrics.extend(_recommendation_metrics(data.recommendations, config, threshold, head_ids))
    metrics.extend(_beyond_accuracy_metrics(data, config, threshold))
    diagnostics = _diagnostics(data.diagnostic_rankings, data.diagnostic_labels, config)
    return EvaluationResult(
        dataset=run.dataset,
        variant=run.variant,
        system=run.system,
        config_hash=run.resolved_hash,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def evaluate_from_config(config: str | Path | dict[str, object]) -> Path:
    run = load_evaluation_config(config)
    payload_path = (
        run.input_path if run.input_path.is_file() else run.input_path / "evaluation.json"
    )
    if payload_path.is_file():
        data = load_evaluation_data(payload_path)
    else:
        from ire_assn1.evaluation.adapters import load_retrieval_evaluation_data

        data = load_retrieval_evaluation_data(run)
    result = evaluate(data, run)
    destination = run.output_path / "evaluation.json"
    return write_evaluation_result(result, destination)


def load_evaluation_data(path: str | Path) -> EvaluationData:
    source = Path(path)
    payload_path = source if source.is_file() else source / "evaluation.json"
    if not payload_path.is_file():
        raise FileNotFoundError(f"Evaluation input not found: {payload_path}")
    payload = EvaluationPayload.model_validate_json(payload_path.read_text(encoding="utf-8"))
    return EvaluationData(
        rankings=tuple(
            RankingRecord(
                impression_id=item.impression_id,
                user_id=item.user_id,
                article_ids=tuple(item.article_ids),
                labels=tuple(item.labels),
                scores=tuple(item.scores),
                history_length=item.history_length,
            )
            for item in payload.rankings
        ),
        recommendations=tuple(
            RecommendationRecord(
                impression_id=item.impression_id,
                user_id=item.user_id,
                article_ids=tuple(item.article_ids),
                relevant_ids=frozenset(item.relevant_ids),
                history_length=item.history_length,
            )
            for item in payload.recommendations
        ),
        training_history_lengths=tuple(payload.training_history_lengths),
        training_clicks=dict(payload.training_clicks),
        embeddings={key: tuple(value) for key, value in payload.embeddings.items()},
        exposed_article_ids=frozenset(payload.exposed_article_ids),
        diagnostic_rankings={
            impression_id: {system: tuple(ids) for system, ids in rankings.items()}
            for impression_id, rankings in payload.diagnostic_rankings.items()
        },
        diagnostic_labels={
            impression_id: dict(labels)
            for impression_id, labels in payload.diagnostic_labels.items()
        },
    )


def _ranking_metrics(
    records: Sequence[RankingRecord],
    config: EvaluationConfig,
    threshold: float,
    head_ids: frozenset[str],
) -> list[MetricEstimate]:
    functions: list[tuple[str, Callable[[RankingRecord], float | None]]] = [
        ("auc", lambda record: auc(record.labels, record.scores)),
        ("mrr", _record_mrr),
    ]
    functions.extend(
        (f"ndcg_at_{k}", lambda record, cutoff=k: _record_ndcg(record, cutoff))
        for k in config.ranking_k
    )
    slices = _ranking_slices(records, threshold, head_ids)
    return [
        _estimate(name, selected, metric, config, slice_name)
        for slice_name, selected in slices.items()
        for name, metric in functions
    ]


def _recommendation_metrics(
    records: Sequence[RecommendationRecord],
    config: EvaluationConfig,
    threshold: float,
    head_ids: frozenset[str],
) -> list[MetricEstimate]:
    functions: list[tuple[str, Callable[[RecommendationRecord], float | None]]] = [
        (f"recall_at_{k}", _recall_metric(k)) for k in config.recall_k
    ]
    slices = _recommendation_slices(records, threshold, head_ids)
    return [
        _estimate(name, selected, metric, config, slice_name)
        for slice_name, selected in slices.items()
        for name, metric in functions
    ]


def _beyond_accuracy_metrics(
    data: EvaluationData, config: EvaluationConfig, threshold: float
) -> list[MetricEstimate]:
    k = config.beyond_accuracy_k
    slices: dict[str, tuple[RecommendationRecord, ...]] = {
        "overall": data.recommendations,
        "cold": tuple(
            record
            for record in data.recommendations
            if user_slice(record.history_length, threshold) == "cold"
        ),
        "warm": tuple(
            record
            for record in data.recommendations
            if user_slice(record.history_length, threshold) == "warm"
        ),
    }
    metrics: list[MetricEstimate] = []
    catalog_size = len(data.training_clicks)
    for slice_name, records in slices.items():
        metrics.append(
            _estimate(
                f"diversity_at_{k}",
                records,
                _diversity_metric(data.embeddings, k),
                config,
                slice_name,
            )
        )
        metrics.append(
            _estimate(
                f"novelty_at_{k}",
                records,
                _novelty_metric(data.training_clicks, k, config.novelty_alpha, catalog_size),
                config,
                slice_name,
            )
        )
        exposed = data.exposed_article_ids or frozenset(
            article_id for record in data.rankings for article_id in record.article_ids
        )
        value = coverage_at_k((record.article_ids for record in records), exposed, k)
        metrics.append(
            MetricEstimate(
                name=f"coverage_at_{k}",
                value=value,
                included=len(records) if value is not None else 0,
                excluded=0 if value is not None else len(records),
                slice_name=slice_name,
            )
        )
    return metrics


def _ranking_slices(
    records: Sequence[RankingRecord], threshold: float, head_ids: frozenset[str]
) -> dict[str, tuple[RankingRecord, ...]]:
    output = {
        "overall": tuple(records),
        "cold": tuple(
            record for record in records if user_slice(record.history_length, threshold) == "cold"
        ),
        "warm": tuple(
            record for record in records if user_slice(record.history_length, threshold) == "warm"
        ),
        "head": tuple(
            record
            for record in records
            if "head" in target_slices(record.article_ids, record.labels, head_ids)
        ),
        "tail": tuple(
            record
            for record in records
            if "tail" in target_slices(record.article_ids, record.labels, head_ids)
        ),
    }
    return output


def _recommendation_slices(
    records: Sequence[RecommendationRecord], threshold: float, head_ids: frozenset[str]
) -> dict[str, tuple[RecommendationRecord, ...]]:
    return {
        "overall": tuple(records),
        "cold": tuple(
            record for record in records if user_slice(record.history_length, threshold) == "cold"
        ),
        "warm": tuple(
            record for record in records if user_slice(record.history_length, threshold) == "warm"
        ),
        "head": tuple(record for record in records if record.relevant_ids & head_ids),
        "tail": tuple(record for record in records if record.relevant_ids - head_ids),
    }


def _record_mrr(record: RankingRecord) -> float | None:
    _, labels = ranked_order(record.article_ids, record.labels, record.scores)
    return mrr(labels)


def _record_ndcg(record: RankingRecord, k: int) -> float | None:
    _, labels = ranked_order(record.article_ids, record.labels, record.scores)
    return ndcg_at_k(labels, k)


def _recall_metric(k: int) -> Callable[[RecommendationRecord], float | None]:
    def metric(record: RecommendationRecord) -> float | None:
        return recall_at_k(record.article_ids, record.relevant_ids, k)

    return metric


def _diversity_metric(
    embeddings: dict[str, tuple[float, ...]], k: int
) -> Callable[[RecommendationRecord], float | None]:
    def metric(record: RecommendationRecord) -> float | None:
        return diversity_at_k(record.article_ids, embeddings, k)

    return metric


def _novelty_metric(
    click_counts: dict[str, int], k: int, alpha: float, catalog_size: int
) -> Callable[[RecommendationRecord], float | None]:
    def metric(record: RecommendationRecord) -> float | None:
        return novelty_at_k(record.article_ids, click_counts, k, alpha, catalog_size)

    return metric


def _estimate[RecordT: UserRecord](
    name: str,
    records: Sequence[RecordT],
    metric: Callable[[RecordT], float | None],
    config: EvaluationConfig,
    slice_name: str,
) -> MetricEstimate:
    observations: list[tuple[str, float]] = []
    for record in records:
        value = metric(record)
        if value is not None:
            observations.append((record.user_id, value))
    value = sum(item[1] for item in observations) / len(observations) if observations else None
    interval = clustered_bootstrap_interval(
        observations,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    return MetricEstimate(
        name=name,
        value=value,
        included=len(observations),
        excluded=len(records) - len(observations),
        slice_name=slice_name,
        interval=interval,
    )


def _diagnostics(
    rankings: dict[str, dict[str, tuple[str, ...]]],
    labels: dict[str, dict[str, float]],
    config: EvaluationConfig,
) -> list[DiagnosticResult]:
    if not rankings:
        return []
    fused_results = {
        impression_id: reciprocal_rank_fusion(systems, config.rrf_k)
        for impression_id, systems in sorted(rankings.items())
    }
    fused = {
        impression_id: [article_id for article_id, _ in result]
        for impression_id, result in fused_results.items()
    }
    observations: dict[str, list[float]] = {
        "auc": [],
        "mrr": [],
        **{f"ndcg_at_{k}": [] for k in config.ranking_k},
    }
    for impression_id, result in fused_results.items():
        if impression_id not in labels:
            continue
        article_ids = [article_id for article_id, _ in result]
        ordered_labels = [labels[impression_id].get(article_id, 0.0) for article_id in article_ids]
        auc_value = auc(ordered_labels, [score for _, score in result])
        mrr_value = mrr(ordered_labels)
        if auc_value is not None:
            observations["auc"].append(auc_value)
        if mrr_value is not None:
            observations["mrr"].append(mrr_value)
        for k in config.ranking_k:
            value = ndcg_at_k(ordered_labels, k)
            if value is not None:
                observations[f"ndcg_at_{k}"].append(value)
    metric_values = {
        name: sum(values) / len(values) if values else None for name, values in observations.items()
    }
    return [
        DiagnosticResult(
            name="future_popularity_rrf",
            rankings=fused,
            metrics=metric_values,
        )
    ]


def result_json(result: EvaluationResult) -> str:
    return json.dumps(result.model_dump(mode="json"), sort_keys=True)
