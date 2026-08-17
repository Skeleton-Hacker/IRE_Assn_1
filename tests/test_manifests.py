from pathlib import Path

from ire_assn1.experiments import provenance
from ire_assn1.experiments.bundles import validate_metrics
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
