import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ire_assn1.experiments import provenance
from ire_assn1.experiments.bundles import (
    _validate_benchmark,
    _validate_bootstrap,
    validate_metrics,
)
from ire_assn1.experiments.manifests import artifact, sha256_file
from ire_assn1.experiments.provenance import load_source_manifest, write_source_manifest


def test_artifact_hash(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("value\n", encoding="utf-8")
    record = artifact(path)
    assert record.sha256 == sha256_file(path)
    assert record.size_bytes == 6


def test_metric_validation() -> None:
    metrics = {
        "recall_at_50": 0.1,
        "recall_at_100": 0.2,
        "recall_at_200": 0.3,
        "auc": 0.5,
        "mrr": 0.4,
        "ndcg_at_5": 0.4,
        "ndcg_at_10": 0.5,
    }
    assert validate_metrics(metrics) == []


def test_metric_validation_accepts_information_and_cosine_distance_scales() -> None:
    metrics = {
        "recall_at_50": 0.1,
        "recall_at_100": 0.2,
        "recall_at_200": 0.3,
        "auc": 0.5,
        "mrr": 0.4,
        "ndcg_at_5": 0.4,
        "ndcg_at_10": 0.5,
        "novelty_at_10": 16.8,
        "diversity_at_10": 1.4,
    }
    assert validate_metrics(metrics) == []


def test_bootstrap_evidence_matches_reported_interval(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    evaluation = {
        "config_hash": "hash",
        "metrics": [
            {
                "name": "novelty_at_10",
                "slice_name": "overall",
                "value": 16.8,
                "interval": {
                    "lower": 16.8,
                    "upper": 16.8,
                    "confidence": 0.95,
                    "samples": 10,
                    "seed": 146,
                },
            }
        ],
    }
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    (tmp_path / "bootstrap.json").write_text(
        json.dumps(
            {
                "config_hash": "hash",
                "samples": 10,
                "seed": 146,
                "draws": {"overall.novelty_at_10": [16.8] * 10},
            }
        ),
        encoding="utf-8",
    )
    assert _validate_bootstrap(evaluation_path, evaluation) == []


def test_benchmark_evidence_is_complete_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    records = [
        {
            "index": "bm25",
            "index_parameters": {"k1": 1.2, "b": 0.75},
            "fraction": fraction,
            "repetition": repetition,
            "impressions": int(1000 * fraction),
            "population_impressions": 1000,
            "workload_impression_sha256": f"workload-{fraction}",
            "ranking_sha256": f"ranking-{fraction}",
            "rows": int(200000 * fraction),
            "elapsed_seconds": fraction,
            "throughput_impressions_per_second": 1000.0,
            "peak_rss_bytes": 1024,
            "rss_delta_bytes": 128,
            "peak_cuda_bytes": 0,
            "index_bytes": 4096,
        }
        for fraction in (0.25, 0.5, 1.0)
        for repetition in range(3)
    ]
    payload = {
        "schema_version": "1.0",
        "dataset": "mind",
        "variant": "small",
        "system": "bm25",
        "config_hash": "hash",
        "seed": 146,
        "workload_order": "first_test_impressions_in_feature_store_order",
        "top_k": 200,
        "hardware": {
            "platform": "test",
            "cpu_count": 1,
            "memory_bytes": 1024,
            "device": {"kind": "cpu"},
        },
        "records": records,
        "extrapolations": [{"index": "bm25", "factor": 10}],
    }
    pq.write_table(
        pa.table(
            {
                "dataset": ["mind"],
                "variant": ["small"],
                "system": ["bm25"],
                "index": ["bm25"],
                "impression_id": ["i1"],
                "article_id": ["a1"],
                "rank": [1],
                "score": [1.0],
            }
        ),
        tmp_path / "sample_predictions.parquet",
    )
    assert _validate_benchmark(path, payload) == []


def test_source_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    written = write_source_manifest(path, allow_dirty=True)
    loaded = load_source_manifest(written)

    assert loaded.git_sha
    assert loaded.tree_sha256
    assert loaded.files
    assert all("__pycache__" not in file.path for file in loaded.files)
    assert all(not file.path.endswith((".pyc", ".pyo")) for file in loaded.files)


def test_source_manifest_fallback_without_git(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    write_source_manifest(path, allow_dirty=True)

    def unavailable() -> provenance.SourceIdentity:
        raise provenance.SourceUnavailableError("git unavailable")

    monkeypatch.setattr(provenance, "git_identity", unavailable)
    identity = provenance.source_identity(path)

    assert identity.kind == "manifest"
    assert identity.git_sha
    assert identity.tree_sha256
