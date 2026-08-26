from __future__ import annotations

import json
import math
from pathlib import Path

from ire_assn1.evaluation.adapters import _history_lengths
from ire_assn1.evaluation.beyond_accuracy import (
    coverage_at_k,
    diversity_at_k,
    novelty_at_k,
)
from ire_assn1.evaluation.bootstrap import clustered_bootstrap_interval
from ire_assn1.evaluation.config import EvaluationConfig, EvaluationRunConfig
from ire_assn1.evaluation.harness import evaluate, evaluate_from_config
from ire_assn1.evaluation.ranking import auc, mrr, ndcg_at_k, ranked_order, recall_at_k
from ire_assn1.evaluation.rrf import reciprocal_rank_fusion
from ire_assn1.evaluation.serialization import (
    read_evaluation_result,
    write_evaluation_result,
)
from ire_assn1.evaluation.slices import (
    cold_warm_threshold,
    head_article_ids,
    target_slices,
    user_slice,
)
from ire_assn1.evaluation.types import (
    EvaluationData,
    RankingRecord,
    RecommendationRecord,
)


def test_ranking_metrics_and_deterministic_ties() -> None:
    article_ids, labels = ranked_order(["b", "a", "c"], [1, 0, 0], [0.5, 0.5, 0.1])
    assert article_ids == ("a", "b", "c")
    assert labels == (0.0, 1.0, 0.0)
    assert auc([1, 0, 0], [0.5, 0.5, 0.1]) == 0.75
    assert mrr(labels) == 0.5
    ndcg = ndcg_at_k(labels, 3)
    recall = recall_at_k(["a", "b", "c"], {"b", "d"}, 2)
    assert ndcg is not None and math.isclose(ndcg, 1.0 / math.log2(3.0))
    assert recall is not None and math.isclose(recall, 0.5)


def test_metric_exclusions() -> None:
    assert auc([1, 1], [0.2, 0.1]) is None
    assert auc([0, 0], [0.2, 0.1]) is None
    assert mrr([0, 0]) is None
    assert ndcg_at_k([0, 0], 2) is None
    assert recall_at_k(["a"], set(), 1) is None


def test_beyond_accuracy_metrics() -> None:
    embeddings = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (1.0, 1.0)}
    assert diversity_at_k(["a", "b"], embeddings, 2) == 1.0
    expected = (-math.log2(3 / 13) - math.log2(1 / 13)) / 2
    novelty = novelty_at_k(["a", "c"], {"a": 2, "b": 8, "c": 0}, 2)
    assert novelty is not None and math.isclose(novelty, expected)
    assert coverage_at_k([["a", "b"], ["b", "c"]], {"a", "b", "c", "d"}, 2) == 0.75


def test_slice_definitions() -> None:
    threshold = cold_warm_threshold([0, 2, 4, 8])
    assert threshold == 3.0
    assert user_slice(3, threshold) == "cold"
    assert user_slice(4, threshold) == "warm"
    head = head_article_ids({"b": 30, "a": 50, "c": 20}, 0.8)
    assert head == frozenset({"a", "b"})
    assert target_slices(["a", "c"], [1, 1], head) == frozenset({"head", "tail"})


def test_history_slice_lengths_are_not_model_truncated() -> None:
    rows = [
        {
            "impression_id": "i1",
            "user_id": "u1",
            "article_ids": ["a", "b", "c", "d"],
            "source_split": "test",
        },
        {
            "impression_id": None,
            "user_id": "u2",
            "article_ids": ["a", "b"],
            "source_split": "train",
        },
    ]
    lengths, training = _history_lengths(rows)
    assert lengths == {("i1", "test"): 4, ("u2", "train"): 2}
    assert training == (2,)


def test_clustered_bootstrap_is_deterministic() -> None:
    observations = [("u1", 1.0), ("u1", 0.0), ("u2", 0.25), ("u3", 0.75)]
    first = clustered_bootstrap_interval(observations, samples=200, seed=12)
    second = clustered_bootstrap_interval(observations, samples=200, seed=12)
    assert first == second
    assert first is not None
    assert first.lower <= 0.5 <= first.upper


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion({"retrieval": ["a", "b"], "future": ["b", "a"]}, k=60)
    assert [item[0] for item in fused] == ["a", "b"]
    assert math.isclose(fused[0][1], fused[1][1])


def test_harness_reports_slices_exclusions_and_diagnostics(tmp_path: Path) -> None:
    data = EvaluationData(
        rankings=(
            RankingRecord("i1", "u1", ("a", "c"), (1, 0), (0.9, 0.1), 1),
            RankingRecord("i2", "u2", ("b", "c"), (0, 0), (0.8, 0.2), 8),
        ),
        recommendations=(
            RecommendationRecord("i1", "u1", ("a", "b"), frozenset({"a"}), 1),
            RecommendationRecord("i2", "u2", ("c", "b"), frozenset({"c"}), 8),
        ),
        training_history_lengths=(1, 4),
        training_clicks={"a": 8, "b": 2, "c": 0},
        embeddings={"a": (1, 0), "b": (0, 1), "c": (1, 1)},
        exposed_article_ids=frozenset({"a", "b", "c"}),
        diagnostic_rankings={"i1": {"retrieval": ("a", "b"), "future": ("b", "a")}},
    )
    run = EvaluationRunConfig(
        dataset="synthetic",
        variant="fixture",
        system="fixture",
        seed=146,
        input_path=tmp_path,
        output_path=tmp_path,
        evaluation=EvaluationConfig(
            recall_k=(1, 2), ranking_k=(1, 2), bootstrap_samples=20, bootstrap_seed=9
        ),
        resolved_hash="abc",
    )
    result = evaluate(data, run)
    overall_auc = next(
        metric
        for metric in result.metrics
        if metric.name == "auc" and metric.slice_name == "overall"
    )
    assert overall_auc.value == 1.0
    assert overall_auc.included == 1
    assert overall_auc.excluded == 1
    assert {metric.slice_name for metric in result.metrics} >= {
        "overall",
        "cold",
        "warm",
        "head",
        "tail",
    }
    assert result.diagnostics[0].available_at_serving_time is False
    assert result.diagnostics[0].excluded_from_submissions is True
    coverage = next(
        metric
        for metric in result.metrics
        if metric.name == "coverage_at_10" and metric.slice_name == "overall"
    )
    assert coverage.interval is not None
    assert coverage.interval.samples == 20
    path = write_evaluation_result(result, tmp_path / "result.json")
    assert read_evaluation_result(path) == result


def test_coverage_bootstrap_resamples_recommendations_and_exposures(tmp_path: Path) -> None:
    rankings = []
    recommendations = []
    training_clicks = {}
    for index in range(20):
        article_ids = tuple(f"a{index}-{candidate}" for candidate in range(10))
        rankings.append(
            RankingRecord(
                f"i{index}",
                f"u{index}",
                article_ids,
                (1.0, *([0.0] * 9)),
                tuple(float(10 - candidate) for candidate in range(10)),
                index,
            )
        )
        recommendations.append(
            RecommendationRecord(
                f"i{index}",
                f"u{index}",
                (article_ids[0],),
                frozenset({article_ids[0]}),
                index,
            )
        )
        training_clicks.update(dict.fromkeys(article_ids, 1))
    data = EvaluationData(
        rankings=tuple(rankings),
        recommendations=tuple(recommendations),
        training_history_lengths=tuple(range(20)),
        training_clicks=training_clicks,
    )
    run = EvaluationRunConfig(
        dataset="synthetic",
        variant="fixture",
        system="fixture",
        seed=146,
        input_path=tmp_path,
        output_path=tmp_path,
        evaluation=EvaluationConfig(bootstrap_samples=100, bootstrap_seed=9),
        resolved_hash="abc",
    )
    result = evaluate(data, run)
    coverage = next(
        metric
        for metric in result.metrics
        if metric.name == "coverage_at_10" and metric.slice_name == "overall"
    )
    assert coverage.value == 0.1
    assert coverage.interval is not None
    assert coverage.interval.lower == 0.1
    assert coverage.interval.upper == 0.1


def test_config_driven_synthetic_harness(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    payload = {
        "rankings": [
            {
                "impression_id": "i1",
                "user_id": "u1",
                "article_ids": ["a", "b"],
                "labels": [1, 0],
                "scores": [1.0, 0.0],
                "history_length": 1,
            }
        ],
        "recommendations": [
            {
                "impression_id": "i1",
                "user_id": "u1",
                "article_ids": ["a", "b"],
                "relevant_ids": ["a"],
                "history_length": 1,
            }
        ],
        "training_history_lengths": [1],
        "training_clicks": {"a": 1, "b": 0},
        "embeddings": {"a": [1, 0], "b": [0, 1]},
        "exposed_article_ids": ["a", "b"],
    }
    (input_path / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        "recall_k: [1]\nranking_k: [1]\nbootstrap_samples: 10\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: synthetic",
                "variant: fixture",
                "system: fixture",
                f"input: {input_path}",
                f"output: {output_path}",
                f"evaluation: {evaluation_path}",
            ]
        ),
        encoding="utf-8",
    )
    destination = evaluate_from_config(config_path)
    result = read_evaluation_result(destination)
    assert destination == output_path / "evaluation.json"
    assert result.dataset == "synthetic"
    assert any(metric.name == "recall_at_1" for metric in result.metrics)
    bootstrap = json.loads((output_path / "bootstrap.json").read_text(encoding="utf-8"))
    assert bootstrap["samples"] == 10
    assert len(bootstrap["draws"]["overall.coverage_at_10"]) == 10
