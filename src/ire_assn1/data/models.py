from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa

SourceSplit = Literal["train", "validation", "test", "competition_test"]


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    name: Literal["mind", "ebnerd"]
    variant: str

    @property
    def prefix(self) -> str:
        return f"{self.name}:{self.variant}"

    def article_id(self, value: str | int) -> str:
        return f"{self.prefix}:article:{value}"

    def user_id(self, value: str | int) -> str:
        return f"{self.prefix}:user:{value}"

    def impression_id(self, value: str | int) -> str:
        return f"{self.prefix}:impression:{value}"

    def session_id(self, value: str | int) -> str:
        return f"{self.prefix}:session:{value}"


@dataclass(frozen=True, slots=True)
class DatasetTables:
    articles: pa.Table
    histories: pa.Table
    impressions: pa.Table
    candidates: pa.Table


@dataclass(frozen=True, slots=True)
class PreparationStats:
    article_conflicts: int
    article_duplicates: int
    validation_start: datetime


@dataclass(frozen=True, slots=True)
class PreparationResult:
    identity: DatasetIdentity
    tables: DatasetTables | None
    stats: PreparationStats
    output_dir: Path | None = None
    row_counts: Mapping[str, int] = field(default_factory=dict)


def normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(microsecond=value.microsecond)
    return value.astimezone(UTC).replace(tzinfo=None)
