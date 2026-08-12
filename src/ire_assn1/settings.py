from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ire_assn1.paths import project_path


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: Path = Path("data")
    models: Path = Path("models")
    logs: Path = Path("logs")
    output: Path = Path("output")
    plots: Path = Path("plots")


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 146
    allow_dirty: bool = False


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved}")
    base: dict[str, Any] = {}
    extends = loaded.pop("extends", [])
    for parent in extends:
        base = deep_merge(base, load_mapping(parent))
    return deep_merge(base, loaded)


def load_config(path: str | Path) -> tuple[RootConfig, dict[str, Any]]:
    mapping = load_mapping(path)
    return RootConfig.model_validate(mapping), mapping


def config_hash(mapping: dict[str, Any]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
