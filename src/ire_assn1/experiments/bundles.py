from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from ire_assn1.experiments.manifests import RunManifest, load_manifest, sha256_file
from ire_assn1.paths import ROOT, project_path

ALLOWED_NAMES = {
    "manifest.json",
    "metrics.json",
    "evaluation.json",
    "bootstrap.json",
    "benchmark.json",
    "sample_predictions.parquet",
    "resolved_config.json",
}
ALLOWED_SUFFIXES = {".png", ".pdf", ".svg", ".log"}


def bundle_members(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file() and (path.name in ALLOWED_NAMES or path.suffix in ALLOWED_SUFFIXES)
    )


def create_bundle(run_id: str) -> None:
    run_root = project_path("logs") / run_id
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    output = project_path("output") / "bundles" / f"{run_id}.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        members = bundle_members(run_root)
        for path in members:
            archive.add(path, arcname=path.relative_to(run_root))
        manifest = load_manifest(manifest_path)
        recorded = {
            project_path(item.path)
            for stage in manifest.stages
            for item in stage.outputs
            if Path(item.path).name in ALLOWED_NAMES or Path(item.path).suffix in ALLOWED_SUFFIXES
        }
        for path in sorted(recorded - set(members)):
            if path.is_file() and path != manifest_path:
                archive.add(path, arcname=path.relative_to(ROOT))
    digest = sha256_file(output)
    print(json.dumps({"bundle": str(output), "sha256": digest}, sort_keys=True))


def validate_metrics(metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "recall_at_50",
        "recall_at_100",
        "recall_at_200",
        "auc",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    }
    entries = metrics.get("metrics", metrics)
    if isinstance(entries, list):
        values = {
            str(item["name"]): item.get("value")
            for item in entries
            if isinstance(item, dict) and item.get("slice_name") == "overall"
        }
    elif isinstance(entries, dict):
        values = entries
    else:
        return ["Metrics payload has an invalid shape"]
    missing = required.difference(values)
    if missing:
        errors.append(f"Missing metrics: {', '.join(sorted(missing))}")
    for name, value in values.items():
        if isinstance(value, int | float) and (value < 0 or value > 1) and "count" not in name:
            errors.append(f"Metric outside [0, 1]: {name}")
    return errors


def validate_manifest(manifest: RunManifest) -> list[str]:
    errors: list[str] = []
    if manifest.git_dirty or not manifest.reportable:
        errors.append("Run is not reportable")
    failed = [stage.name for stage in manifest.stages if stage.state != "complete"]
    if failed:
        errors.append(f"Incomplete stages: {', '.join(failed)}")
    return errors


def validate_directory(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"valid": False, "errors": ["Missing manifest.json"]}
    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest)
    metric_paths = [*root.rglob("evaluation.json"), *root.rglob("metrics.json")]
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(metrics, dict):
            errors.extend(validate_metrics(metrics))
    bootstrap_path = root / "bootstrap.json"
    if bootstrap_path.exists():
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        if isinstance(bootstrap, dict):
            for metric, draws in bootstrap.items():
                if not isinstance(draws, list) or len(draws) != 1000:
                    errors.append(f"Expected 1000 bootstrap draws for {metric}")
    return {"valid": not errors, "run_id": manifest.run_id, "errors": errors}


def validate_bundle(path: Path) -> None:
    resolved = path.resolve()
    if resolved.is_dir():
        result = validate_directory(resolved)
    else:
        with tempfile.TemporaryDirectory() as directory:
            with tarfile.open(resolved, "r:gz") as archive:
                members = archive.getmembers()
                unsafe = any(
                    member.name.startswith("/") or ".." in Path(member.name).parts
                    for member in members
                )
                if unsafe:
                    raise ValueError("Unsafe bundle member path")
                archive.extractall(directory, filter="data")
            result = validate_directory(Path(directory))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise RuntimeError("Run validation failed")
