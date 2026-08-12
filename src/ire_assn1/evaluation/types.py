from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class RankingRecord:
    impression_id: str
    user_id: str
    article_ids: tuple[str, ...]
    labels: tuple[float, ...]
    scores: tuple[float, ...]
    history_length: int

    def __post_init__(self) -> None:
        size = len(self.article_ids)
        if size != len(self.labels) or size != len(self.scores):
            raise ValueError("article_ids, labels, and scores must have equal lengths")
        if self.history_length < 0:
            raise ValueError("history_length must be non-negative")
        if any(label < 0 or not math.isfinite(label) for label in self.labels):
            raise ValueError("labels must be finite and non-negative")
        if any(not math.isfinite(score) for score in self.scores):
            raise ValueError("scores must be finite")


@dataclass(frozen=True, slots=True)
class RecommendationRecord:
    impression_id: str
    user_id: str
    article_ids: tuple[str, ...]
    relevant_ids: frozenset[str]
    history_length: int

    def __post_init__(self) -> None:
        if self.history_length < 0:
            raise ValueError("history_length must be non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationData:
    rankings: tuple[RankingRecord, ...]
    recommendations: tuple[RecommendationRecord, ...]
    training_history_lengths: tuple[int, ...]
    training_clicks: dict[str, int]
    embeddings: dict[str, tuple[float, ...]] = field(default_factory=lambda: {})
    exposed_article_ids: frozenset[str] = field(default_factory=lambda: frozenset())
    diagnostic_rankings: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=lambda: {})
    diagnostic_labels: dict[str, dict[str, float]] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if any(length < 0 for length in self.training_history_lengths):
            raise ValueError("training history lengths must be non-negative")
        if any(count < 0 for count in self.training_clicks.values()):
            raise ValueError("training click counts must be non-negative")


class ConfidenceInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float
    upper: float
    confidence: float = 0.95
    samples: int
    seed: int


class MetricEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: float | None
    included: int
    excluded: int
    slice_name: str = "overall"
    interval: ConfidenceInterval | None = None


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    available_at_serving_time: Literal[False] = False
    excluded_from_submissions: Literal[True] = True
    rankings: dict[str, list[str]]
    metrics: dict[str, float | None] = Field(default_factory=lambda: {})


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset: str
    variant: str
    system: str
    config_hash: str
    metrics: list[MetricEstimate]
    diagnostics: list[DiagnosticResult] = Field(default_factory=lambda: [])
