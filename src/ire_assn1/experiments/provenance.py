from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ire_assn1.paths import ROOT, project_path

SOURCE_MANIFEST = ".ire-source.json"
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("SPEC.md"),
    Path("config"),
    Path("pixi.lock"),
    Path("pixi.toml"),
    Path("src"),
)


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    git_sha: str
    git_dirty: bool
    tree_sha256: str
    files: list[SourceFile] = Field(min_length=1)


@dataclass(frozen=True)
class SourceIdentity:
    git_sha: str
    git_dirty: bool
    tree_sha256: str
    kind: Literal["git", "manifest"]


class SourceUnavailableError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    paths: list[Path] = []
    for relative in SOURCE_PATHS:
        path = project_path(relative)
        if path.is_file():
            paths.append(relative)
        elif path.is_dir():
            paths.extend(
                candidate.relative_to(ROOT) for candidate in path.rglob("*") if candidate.is_file()
            )
        else:
            raise SourceUnavailableError(f"Required source path is missing: {relative}")
    return sorted(set(paths))


def _file_records(paths: list[Path]) -> list[SourceFile]:
    return [
        SourceFile(
            path=relative.as_posix(),
            sha256=sha256_file(project_path(relative)),
            size_bytes=project_path(relative).stat().st_size,
        )
        for relative in paths
    ]


def _tree_sha256(files: list[SourceFile]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(file.size_bytes).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_failure(
    error: FileNotFoundError | subprocess.CalledProcessError,
) -> SourceUnavailableError:
    if isinstance(error, subprocess.CalledProcessError):
        detail = (error.stderr or "").strip() or str(error)
    else:
        detail = str(error)
    return SourceUnavailableError(detail)


def git_identity() -> SourceIdentity:
    try:
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
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise _git_failure(error) from error
    files = _file_records(_source_paths())
    return SourceIdentity(sha, bool(status.strip()), _tree_sha256(files), "git")


def _manifest_path(path: str | Path = SOURCE_MANIFEST) -> Path:
    return project_path(path)


def load_source_manifest(path: str | Path = SOURCE_MANIFEST) -> SourceManifest:
    manifest_path = _manifest_path(path)
    if not manifest_path.exists():
        raise SourceUnavailableError(f"Source manifest is missing: {manifest_path}")
    try:
        manifest = SourceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SourceUnavailableError(f"Invalid source manifest: {manifest_path}") from error
    actual = _file_records(_source_paths())
    expected = {(file.path, file.sha256, file.size_bytes) for file in manifest.files}
    observed = {(file.path, file.sha256, file.size_bytes) for file in actual}
    if expected != observed or _tree_sha256(actual) != manifest.tree_sha256:
        raise SourceUnavailableError("Transferred source does not match the source manifest")
    return manifest


def source_identity(path: str | Path = SOURCE_MANIFEST) -> SourceIdentity:
    try:
        return git_identity()
    except SourceUnavailableError as git_error:
        try:
            manifest = load_source_manifest(path)
        except SourceUnavailableError as manifest_error:
            raise SourceUnavailableError(
                f"Git provenance unavailable ({git_error}); {manifest_error}"
            ) from manifest_error
        return SourceIdentity(
            manifest.git_sha,
            manifest.git_dirty,
            manifest.tree_sha256,
            "manifest",
        )


def write_source_manifest(
    path: str | Path = SOURCE_MANIFEST,
    *,
    allow_dirty: bool = False,
) -> Path:
    identity = git_identity()
    if identity.git_dirty and not allow_dirty:
        raise RuntimeError("Source manifests require a clean Git tree")
    files = _file_records(_source_paths())
    manifest = SourceManifest(
        git_sha=identity.git_sha,
        git_dirty=identity.git_dirty,
        tree_sha256=_tree_sha256(files),
        files=files,
    )
    destination = _manifest_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def provenance_report(path: str | Path = SOURCE_MANIFEST) -> dict[str, Any]:
    try:
        identity = git_identity()
        return {
            "available": True,
            "kind": identity.kind,
            "git_available": True,
            "git_clean": not identity.git_dirty,
            "git_sha": identity.git_sha,
            "tree_sha256": identity.tree_sha256,
            "source_manifest": False,
        }
    except SourceUnavailableError as git_error:
        try:
            manifest = load_source_manifest(path)
            return {
                "available": True,
                "kind": "manifest",
                "git_available": False,
                "git_clean": not manifest.git_dirty,
                "git_sha": manifest.git_sha,
                "tree_sha256": manifest.tree_sha256,
                "source_manifest": True,
            }
        except SourceUnavailableError as manifest_error:
            return {
                "available": False,
                "kind": None,
                "git_available": False,
                "git_clean": None,
                "git_error": str(git_error),
                "source_manifest": False,
                "source_manifest_error": str(manifest_error),
            }
