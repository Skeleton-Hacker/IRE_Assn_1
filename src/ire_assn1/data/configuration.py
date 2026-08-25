from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ire_assn1.paths import project_path
from ire_assn1.settings import load_mapping


class MindDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["mind"]
    variant: Literal["small", "large"]
    language: Literal["en"]
    train_url: str
    validation_url: str
    test_url: str | None = None
    auth_env: str = Field(default="HF_TOKEN", min_length=1)
    train_archive: str
    validation_archive: str
    test_archive: str | None = None
    availability: Literal["first_seen"]
    validation_days: int = Field(default=1, ge=1)


class EbnerdDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ebnerd"]
    variant: Literal["demo", "small", "large"]
    language: Literal["da"]
    archive_url: str
    archive: str
    test_archive_url: str | None = None
    test_archive: str | None = None
    embedding_source: Literal["bert", "roberta", "contrastive"] = "bert"
    availability: Literal["published_at"]
    validation_days: int = Field(default=1, ge=1)


DatasetConfig = MindDatasetConfig | EbnerdDatasetConfig


def parse_dataset_config(mapping: dict[str, Any]) -> DatasetConfig:
    name = mapping.get("name")
    if name == "mind":
        return MindDatasetConfig.model_validate(mapping)
    if name == "ebnerd":
        return EbnerdDatasetConfig.model_validate(mapping)
    raise ValueError(f"Unsupported dataset: {name}")


def load_data_config(path: str | Path) -> tuple[Path, tuple[DatasetConfig, ...]]:
    mapping = load_mapping(path)
    raw_data_path = mapping.get("paths", {}).get("data", "data")
    if not isinstance(raw_data_path, str | Path):
        raise ValueError("paths.data must be a path")
    references = mapping.get("datasets")
    if references is None:
        dataset_mapping = {
            key: value for key, value in mapping.items() if key not in {"paths", "project"}
        }
        return project_path(raw_data_path), (parse_dataset_config(dataset_mapping),)
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise ValueError("datasets must be a list of configuration paths")
    datasets = tuple(parse_dataset_config(load_mapping(item)) for item in references)
    return project_path(raw_data_path), datasets
