from pathlib import Path

from ire_assn1.experiments.bundles import validate_metrics
from ire_assn1.experiments.manifests import artifact, sha256_file


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
