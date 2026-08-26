from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

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
    bounded = ("auc", "mrr", "ndcg_at_", "recall_at_", "coverage_at_")
    for name, value in values.items():
        if not isinstance(value, int | float):
            continue
        if not np.isfinite(value):
            errors.append(f"Metric is not finite: {name}")
        elif name.startswith(bounded) and not 0 <= value <= 1:
            errors.append(f"Metric outside [0, 1]: {name}")
        elif name.startswith("diversity_at_") and not 0 <= value <= 2:
            errors.append(f"Metric outside [0, 2]: {name}")
        elif name.startswith("novelty_at_") and value < 0:
            errors.append(f"Metric is negative: {name}")
    return errors


def _validate_bootstrap(evaluation_path: Path, metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    bootstrap_path = evaluation_path.with_name("bootstrap.json")
    if not bootstrap_path.is_file():
        return [f"Missing bootstrap evidence beside {evaluation_path}"]
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    draws = bootstrap.get("draws") if isinstance(bootstrap, dict) else None
    samples = bootstrap.get("samples") if isinstance(bootstrap, dict) else None
    if not isinstance(draws, dict) or not isinstance(samples, int):
        return [f"Invalid bootstrap evidence: {bootstrap_path}"]
    if bootstrap.get("config_hash") != metrics.get("config_hash"):
        errors.append(f"Bootstrap configuration does not match {evaluation_path}")
    entries = metrics.get("metrics")
    if not isinstance(entries, list):
        return errors
    for item in entries:
        if not isinstance(item, dict) or item.get("value") is None:
            continue
        interval = item.get("interval")
        key = f"{item.get('slice_name')}.{item.get('name')}"
        values = draws.get(key)
        if not isinstance(interval, dict):
            errors.append(f"Missing confidence interval for {key}")
            continue
        if interval.get("samples") != samples or interval.get("seed") != bootstrap.get("seed"):
            errors.append(f"Bootstrap parameters do not match confidence interval for {key}")
            continue
        if not isinstance(values, list) or len(values) != samples:
            errors.append(f"Expected {samples} bootstrap draws for {key}")
            continue
        confidence = float(interval.get("confidence", 0.95))
        tail = (1.0 - confidence) / 2.0
        lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
        if not np.isclose(lower, float(interval["lower"])) or not np.isclose(
            upper, float(interval["upper"])
        ):
            errors.append(f"Confidence interval does not match bootstrap draws for {key}")
    return errors


def _validate_benchmark(path: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "dataset",
        "variant",
        "system",
        "config_hash",
        "seed",
        "workload_order",
        "hardware",
        "top_k",
        "records",
        "extrapolations",
    }
    missing = required.difference(payload)
    if missing:
        errors.append(f"Missing benchmark metadata: {', '.join(sorted(missing))}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        errors.append("Benchmark records must be a non-empty list")
    else:
        record_required = {
            "index_parameters",
            "workload_impression_sha256",
            "ranking_sha256",
            "peak_rss_bytes",
            "peak_cuda_bytes",
            "index_bytes",
        }
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"Invalid benchmark record {index}")
                continue
            absent = record_required.difference(record)
            if absent:
                errors.append(f"Benchmark record {index} is missing: {', '.join(sorted(absent))}")
        complete_records = [
            record
            for record in records
            if isinstance(record, dict) and not record_required.difference(record)
        ]
        if len(complete_records) == len(records):
            fractions = {float(record["fraction"]) for record in complete_records}
            if fractions != {0.25, 0.5, 1.0}:
                errors.append("Benchmark workloads must use 25%, 50%, and 100% fractions")
            indexes = {str(record["index"]) for record in complete_records}
            if payload.get("system") == "bge" and indexes != {"exact", "hnsw"}:
                errors.append("Semantic benchmark must include exact and HNSW indexes")
            for index in indexes:
                for fraction in fractions:
                    selected = [
                        record
                        for record in complete_records
                        if record.get("index") == index
                        and float(record.get("fraction", -1)) == fraction
                    ]
                    repetitions = {int(record["repetition"]) for record in selected}
                    if repetitions != {0, 1, 2}:
                        errors.append(
                            f"Benchmark {index} at {fraction} must have three repetitions"
                        )
                    ranking_hashes = {str(record["ranking_sha256"]) for record in selected}
                    if len(ranking_hashes) != 1:
                        errors.append(
                            "Benchmark rankings changed across repetitions "
                            f"for {index} at {fraction}"
                        )
                    workload_hashes = {
                        str(record["workload_impression_sha256"]) for record in selected
                    }
                    if len(workload_hashes) != 1:
                        errors.append(
                            "Benchmark workload changed across repetitions "
                            f"for {index} at {fraction}"
                        )
                    for record in selected:
                        elapsed = float(record["elapsed_seconds"])
                        impressions = int(record["impressions"])
                        throughput = float(record["throughput_impressions_per_second"])
                        if elapsed <= 0 or not np.isclose(throughput, impressions / elapsed):
                            errors.append(f"Invalid timing evidence for {index} at {fraction}")
                        if int(record["peak_rss_bytes"]) <= 0 or int(record["index_bytes"]) <= 0:
                            errors.append(
                                f"Invalid memory or index evidence for {index} at {fraction}"
                            )
            for fraction in fractions:
                workload_hashes = {
                    str(record["workload_impression_sha256"])
                    for record in complete_records
                    if float(record.get("fraction", -1)) == fraction
                }
                if len(workload_hashes) != 1:
                    errors.append(f"Indexes used different workloads at {fraction}")
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict) or {
        "platform",
        "cpu_count",
        "memory_bytes",
        "device",
    }.difference(hardware):
        errors.append("Benchmark hardware identity is incomplete")
    extrapolations = payload.get("extrapolations")
    if not isinstance(extrapolations, list) or any(
        not isinstance(item, dict) or item.get("factor") != 10 for item in extrapolations
    ):
        errors.append("Benchmark must include a 10x extrapolation for each index")
    elif isinstance(records, list):
        indexes = {str(record["index"]) for record in records if isinstance(record, dict)}
        extrapolated = {str(item["index"]) for item in extrapolations}
        if extrapolated != indexes:
            errors.append("Benchmark 10x extrapolations do not cover every index")
    sample_path = path.with_name("sample_predictions.parquet")
    if not sample_path.is_file():
        errors.append(f"Missing fixed benchmark predictions beside {path}")
    else:
        parquet = pq.ParquetFile(sample_path)
        required_columns = {
            "dataset",
            "variant",
            "system",
            "index",
            "impression_id",
            "article_id",
            "rank",
            "score",
        }
        if parquet.metadata.num_rows == 0:
            errors.append(f"Fixed benchmark predictions are empty: {sample_path}")
        missing_columns = required_columns.difference(parquet.schema_arrow.names)
        if missing_columns:
            errors.append(
                f"Fixed benchmark predictions are missing: {', '.join(sorted(missing_columns))}"
            )
        if not missing_columns:
            table = pq.read_table(sample_path, columns=["index", "impression_id", "rank"])
            groups: dict[tuple[str, str], list[int]] = {}
            for row in table.to_pylist():
                key = (str(row["index"]), str(row["impression_id"]))
                groups.setdefault(key, []).append(int(row["rank"]))
            if any(sorted(ranks) != list(range(1, len(ranks) + 1)) for ranks in groups.values()):
                errors.append("Fixed benchmark predictions contain invalid rank permutations")
    return errors


def _validate_artifacts(root: Path, manifest: RunManifest) -> list[str]:
    errors: list[str] = []
    records = {
        record.path: record
        for stage in manifest.stages
        for record in stage.outputs
        if not Path(record.path).is_absolute()
        and (
            Path(record.path).name in ALLOWED_NAMES or Path(record.path).suffix in ALLOWED_SUFFIXES
        )
    }
    for relative, record in records.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing recorded artifact: {relative}")
        elif path.stat().st_size != record.size_bytes or sha256_file(path) != record.sha256:
            errors.append(f"Recorded artifact does not match manifest: {relative}")
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
    errors.extend(_validate_artifacts(root, manifest))
    metric_paths = [*root.rglob("evaluation.json"), *root.rglob("metrics.json")]
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(metrics, dict):
            errors.extend(validate_metrics(metrics))
            if metrics_path.name == "evaluation.json":
                errors.extend(_validate_bootstrap(metrics_path, metrics))
    for benchmark_path in root.rglob("benchmark.json"):
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if isinstance(benchmark, dict):
            errors.extend(_validate_benchmark(benchmark_path, benchmark))
    if not (root / "resolved_config.json").is_file():
        errors.append("Missing resolved_config.json")
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
