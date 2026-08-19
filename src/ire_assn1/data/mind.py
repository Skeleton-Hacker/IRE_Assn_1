from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ire_assn1.data.contracts import (
    ARTICLES_SCHEMA,
    CANDIDATES_SCHEMA,
    HISTORIES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    table_from_rows,
)
from ire_assn1.data.models import (
    DatasetIdentity,
    DatasetTables,
    PreparationResult,
    PreparationStats,
    SourceSplit,
)
from ire_assn1.data.splits import TemporalSplit, official_temporal_split

MIND_TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass(frozen=True, slots=True)
class MindArticle:
    article_id: str
    category: str | None
    subcategory: str | None
    title: str
    abstract: str
    entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MindBehavior:
    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    labels: tuple[int, ...]

    @property
    def clicked_ids(self) -> tuple[str, ...]:
        return tuple(
            article_id
            for article_id, label in zip(self.candidate_ids, self.labels, strict=True)
            if label == 1
        )


def _optional_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _parse_entities(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    loaded: Any = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError("MIND entities must be a JSON list")
    entities: list[str] = []
    seen: set[str] = set()
    for item in loaded:
        if not isinstance(item, dict):
            raise ValueError("MIND entity entries must be JSON objects")
        identifier = item.get("WikidataId") or item.get("Label")
        if identifier is not None:
            normalized = str(identifier)
            if normalized not in seen:
                seen.add(normalized)
                entities.append(normalized)
    return tuple(entities)


def _read_article_rows(path: Path) -> Iterable[tuple[int, list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            row = next(csv.reader([line], delimiter="\t"), [])
            if len(row) != 8:
                repaired: list[str] = []
                for value in row:
                    normalized = value.replace("\\\\t", "\t").replace("\\t", "\t")
                    repaired.extend(normalized.split("\t"))
                row = repaired
            yield line_number, row


def read_mind_articles(path: Path) -> tuple[MindArticle, ...]:
    articles: list[MindArticle] = []
    for line_number, row in _read_article_rows(path):
        if len(row) != 8:
            raise ValueError(
                f"Invalid MIND news row at {path}:{line_number}: expected 8 fields, got {len(row)}"
            )
        entities = _parse_entities(row[6]) + _parse_entities(row[7])
        articles.append(
            MindArticle(
                article_id=row[0],
                category=_optional_text(row[1]),
                subcategory=_optional_text(row[2]),
                title=row[3].strip(),
                abstract=row[4].strip(),
                entities=tuple(dict.fromkeys(entities)),
            )
        )
    return tuple(articles)


def read_mind_behaviors(path: Path) -> tuple[MindBehavior, ...]:
    behaviors: list[MindBehavior] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 5:
                raise ValueError(f"Invalid MIND behavior row at {path}:{line_number}")
            candidates: list[str] = []
            labels: list[int] = []
            for impression in row[4].split():
                article_id, separator, raw_label = impression.rpartition("-")
                if separator != "-" or raw_label not in {"0", "1"}:
                    raise ValueError(
                        f"Invalid MIND impression token at {path}:{line_number}: {impression}"
                    )
                candidates.append(article_id)
                labels.append(int(raw_label))
            behaviors.append(
                MindBehavior(
                    impression_id=row[0],
                    user_id=row[1],
                    timestamp=datetime.strptime(row[2], MIND_TIMESTAMP_FORMAT),
                    history=tuple(row[3].split()),
                    candidate_ids=tuple(candidates),
                    labels=tuple(labels),
                )
            )
    return tuple(behaviors)


def mind_first_seen(
    packages: Iterable[tuple[Sequence[MindBehavior], SourceSplit | TemporalSplit]],
) -> dict[str, tuple[datetime, SourceSplit]]:
    first_seen: dict[str, tuple[datetime, SourceSplit]] = {}
    priority = {"train": 0, "validation": 1, "test": 2}
    for behaviors, split_rule in packages:
        for behavior in behaviors:
            split = (
                split_rule.assign_training(behavior.timestamp)
                if isinstance(split_rule, TemporalSplit)
                else split_rule
            )
            observed = (*behavior.history, *behavior.candidate_ids, *behavior.clicked_ids)
            for article_id in observed:
                current = first_seen.get(article_id)
                candidate = (behavior.timestamp, split)
                if (
                    current is None
                    or candidate[0] < current[0]
                    or (
                        candidate[0] == current[0] and priority[candidate[1]] < priority[current[1]]
                    )
                ):
                    first_seen[article_id] = candidate
    return first_seen


def _find_single(root: Path, name: str) -> Path:
    matches = tuple(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _deduplicate_articles(
    sources: Iterable[Sequence[MindArticle]],
) -> tuple[dict[str, MindArticle], int, int]:
    articles: dict[str, MindArticle] = {}
    duplicates = 0
    conflicts = 0
    for source in sources:
        for article in source:
            existing = articles.get(article.article_id)
            if existing is None:
                articles[article.article_id] = article
            else:
                duplicates += 1
                conflicts += int(existing != article)
    return articles, duplicates, conflicts


def _normalize_behaviors(
    identity: DatasetIdentity,
    packages: Iterable[tuple[Sequence[MindBehavior], SourceSplit | TemporalSplit]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    histories: dict[tuple[str, SourceSplit], tuple[datetime, tuple[str, ...]]] = {}
    impressions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for behaviors, split_rule in packages:
        for behavior in behaviors:
            source_split = (
                split_rule.assign_training(behavior.timestamp)
                if isinstance(split_rule, TemporalSplit)
                else split_rule
            )
            user_id = identity.user_id(behavior.user_id)
            impression_id = identity.impression_id(behavior.impression_id)
            candidate_ids = [identity.article_id(value) for value in behavior.candidate_ids]
            clicked_ids = [identity.article_id(value) for value in behavior.clicked_ids]
            history_ids = tuple(identity.article_id(value) for value in behavior.history)
            history_key = (user_id, source_split)
            existing = histories.get(history_key)
            if existing is None or behavior.timestamp < existing[0]:
                histories[history_key] = (behavior.timestamp, history_ids)
            impressions.append(
                {
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "session_id": None,
                    "timestamp": behavior.timestamp,
                    "candidate_ids": candidate_ids,
                    "clicked_ids": clicked_ids,
                    "labels": list(behavior.labels),
                    "source_split": source_split,
                }
            )
            for position, (article_id, label) in enumerate(
                zip(candidate_ids, behavior.labels, strict=True)
            ):
                candidates.append(
                    {
                        "impression_id": impression_id,
                        "user_id": user_id,
                        "timestamp": behavior.timestamp,
                        "article_id": article_id,
                        "label": label,
                        "position": position,
                        "source_split": source_split,
                    }
                )
    history_rows = [
        {
            "user_id": user_id,
            "article_ids": list(article_ids),
            "timestamps": [None] * len(article_ids),
            "source_split": source_split,
        }
        for (user_id, source_split), (_, article_ids) in sorted(histories.items())
    ]
    return history_rows, impressions, candidates


def prepare_mind(
    train_root: Path,
    official_validation_root: Path,
    variant: str = "small",
    validation_days: int = 1,
) -> PreparationResult:
    identity = DatasetIdentity(name="mind", variant=variant)
    train_behaviors = read_mind_behaviors(_find_single(train_root, "behaviors.tsv"))
    test_behaviors = read_mind_behaviors(_find_single(official_validation_root, "behaviors.tsv"))
    temporal_split = official_temporal_split(
        (behavior.timestamp for behavior in train_behaviors), validation_days
    )
    packages: tuple[tuple[Sequence[MindBehavior], SourceSplit | TemporalSplit], ...] = (
        (train_behaviors, temporal_split),
        (test_behaviors, "test"),
    )
    availability = mind_first_seen(packages)
    train_articles = read_mind_articles(_find_single(train_root, "news.tsv"))
    test_articles = read_mind_articles(_find_single(official_validation_root, "news.tsv"))
    articles, duplicate_count, conflict_count = _deduplicate_articles(
        (train_articles, test_articles)
    )
    article_rows = [
        {
            "article_id": identity.article_id(article.article_id),
            "title": article.title,
            "abstract": article.abstract,
            "body": None,
            "category": article.category,
            "subcategory": article.subcategory,
            "entities": list(article.entities),
            "published_at": None,
            "available_at": availability[article.article_id][0],
            "source_split": availability[article.article_id][1],
        }
        for article in articles.values()
        if article.article_id in availability
    ]
    history_rows, impression_rows, candidate_rows = _normalize_behaviors(identity, packages)
    tables = DatasetTables(
        articles=table_from_rows(article_rows, ARTICLES_SCHEMA),
        histories=table_from_rows(history_rows, HISTORIES_SCHEMA),
        impressions=table_from_rows(impression_rows, IMPRESSIONS_SCHEMA),
        candidates=table_from_rows(candidate_rows, CANDIDATES_SCHEMA),
    )
    return PreparationResult(
        identity=identity,
        tables=tables,
        stats=PreparationStats(
            article_conflicts=conflict_count,
            article_duplicates=duplicate_count,
            validation_start=temporal_split.validation_start,
        ),
    )
