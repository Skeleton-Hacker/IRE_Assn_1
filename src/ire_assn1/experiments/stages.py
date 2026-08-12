from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import psutil

from ire_assn1.experiments.manifests import (
    ArtifactRecord,
    RunManifest,
    StageRecord,
    artifact,
    write_manifest,
)


def _artifacts(paths: Iterable[Path]) -> list[ArtifactRecord]:
    files = (
        nested
        for path in paths
        for nested in (
            sorted(
                child
                for child in path.rglob("*")
                if child.is_file()
                and (
                    child.name in {"manifest.json", ".complete.json", "selection.json"}
                    or (child.suffix == ".json" and child.stat().st_size < 16 * 1024 * 1024)
                )
            )
            if path.is_dir()
            else (path,)
        )
        if nested.is_file()
    )
    return [artifact(path) for path in files]


def run_stage[T](
    manifest: RunManifest,
    name: str,
    operation: Callable[[], T],
    inputs: Iterable[Path] = (),
    outputs: Iterable[Path] = (),
) -> T:
    started = time.perf_counter()
    stage = StageRecord(
        name=name,
        state="running",
        started_at=datetime.now(UTC),
        inputs=_artifacts(path for path in inputs if path.exists()),
    )
    manifest.stages.append(stage)
    write_manifest(manifest)
    try:
        result = operation()
        stage.state = "complete"
        stage.outputs = _artifacts(path for path in outputs if path.exists())
        return result
    except Exception as error:
        stage.state = "failed"
        stage.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        stage.elapsed_seconds = time.perf_counter() - started
        stage.finished_at = datetime.now(UTC)
        stage.peak_rss_bytes = psutil.Process().memory_info().rss
        try:
            torch = importlib.import_module("torch")

            stage.peak_cuda_bytes = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
        except ImportError:
            stage.peak_cuda_bytes = None
        write_manifest(manifest)
