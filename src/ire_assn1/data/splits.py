from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

from ire_assn1.data.models import SourceSplit, normalize_timestamp


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    validation_start: datetime
    validation_dates: tuple[date, ...]

    def assign_training(self, timestamp: datetime) -> SourceSplit:
        normalized = normalize_timestamp(timestamp)
        return "validation" if normalized >= self.validation_start else "train"


def official_temporal_split(
    training_timestamps: Iterable[datetime], validation_days: int = 1
) -> TemporalSplit:
    if validation_days < 1:
        raise ValueError("validation_days must be positive")
    unique_dates = sorted({normalize_timestamp(value).date() for value in training_timestamps})
    if len(unique_dates) <= validation_days:
        raise ValueError("Training data must contain an earlier day and all validation days")
    validation_dates = tuple(unique_dates[-validation_days:])
    return TemporalSplit(
        validation_start=datetime.combine(validation_dates[0], datetime.min.time()),
        validation_dates=validation_dates,
    )
