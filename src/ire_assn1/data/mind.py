from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from ire_assn1.data.store import write_feature_store_chunks

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
            row = next(csv.reader([line], delimiter="\t", quoting=csv.QUOTE_NONE), [])
            if len(row) != 8:
                repaired: list[str] = []
                for value in row:
                    normalized = value.replace("\\\\t", "\t").replace("\\t", "\t")
                    repaired.extend(normalized.split("\t"))
                row = repaired
            yield line_number, row


def read_mind_articles(path: Path) -> tuple[MindArticle, ...]:
    return tuple(_iter_mind_articles(path))


def _iter_mind_articles(path: Path) -> Iterable[MindArticle]:
    for line_number, row in _read_article_rows(path):
        if len(row) != 8:
            raise ValueError(
                f"Invalid MIND news row at {path}:{line_number}: expected 8 fields, got {len(row)}"
            )
        entities = _parse_entities(row[6]) + _parse_entities(row[7])
        yield MindArticle(
            article_id=row[0],
            category=_optional_text(row[1]),
            subcategory=_optional_text(row[2]),
            title=row[3].strip(),
            abstract=row[4].strip(),
            entities=tuple(dict.fromkeys(entities)),
        )


def _iter_mind_behaviors(path: Path) -> Iterable[MindBehavior]:
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
            yield MindBehavior(
                impression_id=row[0],
                user_id=row[1],
                timestamp=datetime.strptime(row[2], MIND_TIMESTAMP_FORMAT),
                history=tuple(row[3].split()),
                candidate_ids=tuple(candidates),
                labels=tuple(labels),
            )


def read_mind_behaviors(path: Path) -> tuple[MindBehavior, ...]:
    return tuple(_iter_mind_behaviors(path))


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


def _mind_split(behavior: MindBehavior, split_rule: SourceSplit | TemporalSplit) -> SourceSplit:
    return (
        split_rule.assign_training(behavior.timestamp)
        if isinstance(split_rule, TemporalSplit)
        else split_rule
    )


def _chunks[T](rows: Iterable[T], size: int = 8192) -> Iterable[list[T]]:
    chunk: list[T] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _mind_article_rows(
    paths: Sequence[Path],
    availability: Mapping[str, tuple[datetime, SourceSplit]],
    identity: DatasetIdentity,
    connection: sqlite3.Connection,
    stats: dict[str, int],
) -> Iterable[list[dict[str, object]]]:
    connection.execute(
        "CREATE TABLE articles ("
        "article_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    for path in paths:
        for article in _iter_mind_articles(path):
            if article.article_id not in availability:
                continue
            row = {
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
            payload = json.dumps(
                row,
                default=lambda value: value.isoformat() if isinstance(value, datetime) else value,
                sort_keys=True,
            )
            fingerprint = hashlib.sha256(payload.encode()).hexdigest()
            existing = connection.execute(
                "SELECT fingerprint FROM articles WHERE article_id = ?", (row["article_id"],)
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO articles(article_id, fingerprint, payload) VALUES (?, ?, ?)",
                    (row["article_id"], fingerprint, payload),
                )
            else:
                stats["article_duplicates"] += 1
                stats["article_conflicts"] += int(existing[0] != fingerprint)
        connection.commit()
    rows: list[dict[str, object]] = []
    for (payload,) in connection.execute("SELECT payload FROM articles ORDER BY rowid"):
        row = json.loads(payload)
        row["available_at"] = datetime.fromisoformat(row["available_at"])
        rows.append(row)
        if len(rows) == 8192:
            yield rows
            rows = []
    if rows:
        yield rows


def _mind_impression_rows(
    paths: Sequence[tuple[Path, SourceSplit | TemporalSplit]],
    identity: DatasetIdentity,
    connection: sqlite3.Connection,
) -> Iterable[list[dict[str, object]]]:
    def rows() -> Iterable[dict[str, object]]:
        for path, split_rule in paths:
            for behavior in _iter_mind_behaviors(path):
                source_split = _mind_split(behavior, split_rule)
                connection.execute(
                    """
                    INSERT INTO histories(user_id, source_split, timestamp, article_ids)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, source_split) DO UPDATE SET
                        timestamp = excluded.timestamp,
                        article_ids = excluded.article_ids
                    WHERE excluded.timestamp < histories.timestamp
                    """,
                    (
                        behavior.user_id,
                        source_split,
                        behavior.timestamp.isoformat(timespec="microseconds"),
                        json.dumps(behavior.history),
                    ),
                )
                yield {
                    "impression_id": identity.impression_id(behavior.impression_id),
                    "user_id": identity.user_id(behavior.user_id),
                    "session_id": None,
                    "timestamp": behavior.timestamp,
                    "candidate_ids": [
                        identity.article_id(value) for value in behavior.candidate_ids
                    ],
                    "clicked_ids": [identity.article_id(value) for value in behavior.clicked_ids],
                    "labels": list(behavior.labels),
                    "source_split": source_split,
                }
        connection.commit()

    return _chunks(rows())


def _mind_candidate_rows(
    paths: Sequence[tuple[Path, SourceSplit | TemporalSplit]],
    identity: DatasetIdentity,
) -> Iterable[list[dict[str, object]]]:
    rows = (
        {
            "impression_id": identity.impression_id(behavior.impression_id),
            "user_id": identity.user_id(behavior.user_id),
            "timestamp": behavior.timestamp,
            "article_id": identity.article_id(article_id),
            "label": label,
            "position": position,
            "source_split": _mind_split(behavior, split_rule),
        }
        for path, split_rule in paths
        for behavior in _iter_mind_behaviors(path)
        for position, (article_id, label) in enumerate(
            zip(behavior.candidate_ids, behavior.labels, strict=True)
        )
    )
    return _chunks(rows)


def _mind_history_rows(
    connection: sqlite3.Connection, identity: DatasetIdentity
) -> Iterable[list[dict[str, object]]]:
    def rows() -> Iterable[dict[str, object]]:
        cursor = connection.execute(
            "SELECT user_id, source_split, timestamp, article_ids "
            "FROM histories ORDER BY source_split, user_id"
        )
        for user_id, source_split, _, article_ids in cursor:
            values = json.loads(article_ids)
            yield {
                "user_id": identity.user_id(user_id),
                "article_ids": [identity.article_id(value) for value in values],
                "timestamps": [None] * len(values),
                "source_split": source_split,
            }

    return _chunks(rows())


def prepare_mind_streaming(
    train_root: Path,
    official_validation_root: Path,
    variant: str,
    validation_days: int,
    data_root: Path,
) -> PreparationResult:
    data_root.mkdir(parents=True, exist_ok=True)
    identity = DatasetIdentity(name="mind", variant=variant)
    train_behavior_path = _find_single(train_root, "behaviors.tsv")
    test_behavior_path = _find_single(official_validation_root, "behaviors.tsv")
    temporal_split = official_temporal_split(
        (behavior.timestamp for behavior in _iter_mind_behaviors(train_behavior_path)),
        validation_days,
    )
    paths: tuple[tuple[Path, SourceSplit | TemporalSplit], ...] = (
        (train_behavior_path, temporal_split),
        (test_behavior_path, "test"),
    )
    availability: dict[str, tuple[datetime, SourceSplit]] = {}
    priority = {"train": 0, "validation": 1, "test": 2}
    for path, split_rule in paths:
        for behavior in _iter_mind_behaviors(path):
            source_split = _mind_split(behavior, split_rule)
            for article_id in (*behavior.history, *behavior.candidate_ids, *behavior.clicked_ids):
                candidate = (behavior.timestamp, source_split)
                current = availability.get(article_id)
                if (
                    current is None
                    or candidate[0] < current[0]
                    or (
                        candidate[0] == current[0] and priority[candidate[1]] < priority[current[1]]
                    )
                ):
                    availability[article_id] = candidate
    article_stats = {"article_duplicates": 0, "article_conflicts": 0}
    with tempfile.NamedTemporaryFile(
        prefix="mind-histories-", suffix=".sqlite", dir=data_root, delete=False
    ) as temporary:
        database_path = Path(temporary.name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE histories ("
            "user_id TEXT NOT NULL, source_split TEXT NOT NULL, timestamp TEXT NOT NULL, "
            "article_ids TEXT NOT NULL, PRIMARY KEY (user_id, source_split))"
        )
        result = write_feature_store_chunks(
            identity,
            PreparationStats(
                article_conflicts=0,
                article_duplicates=0,
                validation_start=temporal_split.validation_start,
            ),
            {
                "articles": _mind_article_rows(
                    (
                        _find_single(train_root, "news.tsv"),
                        _find_single(official_validation_root, "news.tsv"),
                    ),
                    availability,
                    identity,
                    connection,
                    article_stats,
                ),
                "impressions": _mind_impression_rows(paths, identity, connection),
                "candidates": _mind_candidate_rows(paths, identity),
                "histories": _mind_history_rows(connection, identity),
            },
            data_root,
        )
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    return replace(
        result,
        stats=PreparationStats(
            article_conflicts=article_stats["article_conflicts"],
            article_duplicates=article_stats["article_duplicates"],
            validation_start=temporal_split.validation_start,
        ),
    )
