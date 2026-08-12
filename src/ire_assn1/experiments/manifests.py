from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field

from ire_assn1.paths import ROOT, project_path
from ire_assn1.settings import config_hash


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    state: Literal["running", "complete", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_seconds: float | None = None
    peak_rss_bytes: int | None = None
    peak_cuda_bytes: int | None = None
    inputs: list[ArtifactRecord] = Field(default_factory=list)
    outputs: list[ArtifactRecord] = Field(default_factory=list)
    error: str | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    created_at: datetime
    git_sha: str
    git_dirty: bool
    reportable: bool
    config_sha256: str
    lock_sha256: str | None
    resolved_config: dict[str, Any]
    python_version: str
    platform: str
    cpu_count: int
    memory_bytes: int
    device: dict[str, Any]
    stages: list[StageRecord] = Field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> ArtifactRecord:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        relative = resolved
    return ArtifactRecord(
        path=str(relative),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def git_identity() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def device_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {"kind": "cpu"}
    try:
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            identity = {
                "kind": "cuda",
                "name": properties.name,
                "total_memory": properties.total_memory,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            }
    except ImportError:
        pass
    return identity


def create_manifest(
    name: str,
    variant: str,
    system: str,
    resolved_config: dict[str, Any],
    allow_dirty: bool = False,
) -> RunManifest:
    sha, dirty = git_identity()
    if dirty and not allow_dirty:
        raise RuntimeError("Official runs require a clean Git tree")
    timestamp = datetime.now(UTC)
    run_id = f"{name}-{variant}-{system}-{timestamp:%Y%m%dT%H%M%SZ}-{sha[:7]}"
    lock = project_path("pixi.lock")
    return RunManifest(
        run_id=run_id,
        created_at=timestamp,
        git_sha=sha,
        git_dirty=dirty,
        reportable=not dirty,
        config_sha256=config_hash(resolved_config),
        lock_sha256=sha256_file(lock) if lock.exists() else None,
        resolved_config=resolved_config,
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu_count=psutil.cpu_count() or 1,
        memory_bytes=psutil.virtual_memory().total,
        device=device_identity(),
    )


def write_manifest(manifest: RunManifest, root: Path | None = None) -> Path:
    run_root = root or project_path("logs") / manifest.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "manifest.json"
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
