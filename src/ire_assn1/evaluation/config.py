from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from ire_assn1.paths import project_path
from ire_assn1.settings import config_hash, load_mapping


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_k: tuple[PositiveInt, ...] = (50, 100, 200)
    ranking_k: tuple[PositiveInt, ...] = (5, 10)
    beyond_accuracy_k: PositiveInt = 10
    bootstrap_samples: PositiveInt = 1000
    bootstrap_seed: int = 146
    novelty_alpha: float = 1.0
    head_click_share: float = 0.8
    rrf_k: PositiveInt = 60

    @field_validator("recall_k", "ranking_k")
    @classmethod
    def unique_cutoffs(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("metric cutoffs must be non-empty and unique")
        return value

    @field_validator("novelty_alpha")
    @classmethod
    def positive_alpha(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("novelty_alpha must be positive")
        return value

    @field_validator("head_click_share")
    @classmethod
    def valid_share(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("head_click_share must be in (0, 1]")
        return value


class EvaluationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    variant: str
    system: str
    seed: int
    input_path: Path
    output_path: Path
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    resolved_hash: str


def load_evaluation_config(path: str | Path | Mapping[str, object]) -> EvaluationRunConfig:
    mapping = dict(path) if isinstance(path, Mapping) else load_mapping(path)
    evaluation_value = mapping.get("evaluation", {})
    evaluation_mapping = _evaluation_mapping(evaluation_value)
    project = mapping.get("project", {})
    seed = _integer(mapping.get("seed", _mapping_value(project, "seed", 146)), "seed")
    name = str(mapping.get("name", "unknown"))
    variant = str(mapping.get("variant", "unknown"))
    system = str(mapping.get("system", "unknown"))
    input_value = mapping.get("input", Path("data") / "processed" / name / variant)
    output_value = mapping.get("output", Path("output") / name / variant / system)
    return EvaluationRunConfig(
        dataset=name,
        variant=variant,
        system=system,
        seed=seed,
        input_path=project_path(str(input_value)),
        output_path=project_path(str(output_value)),
        evaluation=EvaluationConfig.model_validate(evaluation_mapping),
        resolved_hash=config_hash(mapping),
    )


def _evaluation_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        return load_mapping(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): item for key, item in mapping.items()}
    raise ValueError("evaluation must be a path or mapping")


def _mapping_value(value: object, key: str, default: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return mapping.get(key, default)
    return default


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value
