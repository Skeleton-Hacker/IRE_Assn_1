from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

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
    normalize_timestamp,
)
from ire_assn1.data.splits import TemporalSplit, official_temporal_split
from ire_assn1.data.store import write_feature_store_chunks

ARTICLE_COLUMNS = {
    "article_id": ("article_id",),
    "title": ("title",),
    "abstract": ("subtitle", "abstract"),
    "body": ("body",),
    "category": ("category_str", "category", "category_id"),
    "subcategory": ("subcategory", "subcategory_ids"),
    "entities": ("ner_clusters", "ner", "entities"),
    "published_at": ("published_time", "published_at", "time_published"),
}

BEHAVIOR_COLUMNS = {
    "impression_id": ("impression_id",),
    "user_id": ("user_id",),
    "session_id": ("session_id",),
    "timestamp": ("impression_time", "timestamp", "time"),
    "candidate_ids": ("article_ids_inview", "inview_article_ids"),
    "clicked_ids": ("article_ids_clicked", "clicked_article_ids"),
}

HISTORY_COLUMNS = {
    "user_id": ("user_id",),
    "article_ids": ("article_id_fixed", "article_ids", "article_ids_fixed"),
    "timestamps": (
        "impression_time_fixed",
        "timestamps",
        "impression_times_fixed",
    ),
}


def _resolve_columns(
    table: pa.Table,
    aliases: Mapping[str, Sequence[str]],
    optional: Iterable[str] = (),
) -> dict[str, str | None]:
    available = set(table.column_names)
    optional_set = set(optional)
    resolved: dict[str, str | None] = {}
    for normalized, candidates in aliases.items():
        match = next((candidate for candidate in candidates if candidate in available), None)
        if match is None and normalized not in optional_set:
            names = ", ".join(candidates)
            raise ValueError(f"Missing required EB-NeRD column, expected one of: {names}")
        resolved[normalized] = match
    return resolved


def _required(row: Mapping[str, Any], column: str | None, name: str) -> Any:
    if column is None or row.get(column) is None:
        raise ValueError(f"Missing required EB-NeRD value: {name}")
    return row[column]


def _values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return tuple(value)
    if hasattr(value, "tolist"):
        loaded = value.tolist()
        return tuple(loaded if isinstance(loaded, list) else [loaded])
    raise ValueError(f"Expected a list value, received {type(value)}")


def _datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"Expected datetime for {name}, received {type(value)}")
    return normalize_timestamp(value)


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _list_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list | tuple):
        normalized = [str(item) for item in value if item is not None]
        return "|".join(normalized) or None
    return _nullable_text(value)


def _source_split(
    published_at: datetime, temporal_split: TemporalSplit, test_start: datetime
) -> SourceSplit:
    if published_at >= test_start:
        return "test"
    return temporal_split.assign_training(published_at)


def _normalize_articles(
    table: pa.Table,
    identity: DatasetIdentity,
    temporal_split: TemporalSplit,
    test_start: datetime,
) -> tuple[list[dict[str, Any]], int, int]:
    columns = _resolve_columns(
        table, ARTICLE_COLUMNS, optional=("body", "category", "subcategory", "entities")
    )
    articles: dict[str, dict[str, Any]] = {}
    duplicates = 0
    conflicts = 0
    for row in table.to_pylist():
        raw_id = str(_required(row, columns["article_id"], "article_id"))
        published_at = _datetime(
            _required(row, columns["published_at"], "published_at"), "published_at"
        )
        entity_values = _values(row.get(columns["entities"])) if columns["entities"] else ()
        normalized = {
            "article_id": identity.article_id(raw_id),
            "title": str(_required(row, columns["title"], "title")).strip(),
            "abstract": str(_required(row, columns["abstract"], "abstract")).strip(),
            "body": _nullable_text(row.get(columns["body"])) if columns["body"] else None,
            "category": (
                _nullable_text(row.get(columns["category"])) if columns["category"] else None
            ),
            "subcategory": (
                _list_text(row.get(columns["subcategory"])) if columns["subcategory"] else None
            ),
            "entities": list(
                dict.fromkeys(str(item) for item in entity_values if item is not None)
            ),
            "published_at": published_at,
            "available_at": published_at,
            "source_split": _source_split(published_at, temporal_split, test_start),
        }
        existing = articles.get(raw_id)
        if existing is None:
            articles[raw_id] = normalized
        else:
            duplicates += 1
            conflicts += int(existing != normalized)
    return list(articles.values()), duplicates, conflicts


def _normalize_behavior_table(
    table: pa.Table,
    identity: DatasetIdentity,
    split_rule: SourceSplit | TemporalSplit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[SourceSplit, set[str]]]:
    columns = _resolve_columns(table, BEHAVIOR_COLUMNS, optional=("session_id", "clicked_ids"))
    impressions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    users: dict[SourceSplit, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for row in table.to_pylist():
        timestamp = _datetime(_required(row, columns["timestamp"], "timestamp"), "timestamp")
        source_split = (
            split_rule.assign_training(timestamp)
            if isinstance(split_rule, TemporalSplit)
            else split_rule
        )
        raw_user_id = str(_required(row, columns["user_id"], "user_id"))
        user_id = identity.user_id(raw_user_id)
        users[source_split].add(raw_user_id)
        raw_impression_id = str(_required(row, columns["impression_id"], "impression_id"))
        impression_id = identity.impression_id(raw_impression_id)
        raw_candidates = _values(_required(row, columns["candidate_ids"], "candidate_ids"))
        raw_clicks = _values(row.get(columns["clicked_ids"])) if columns["clicked_ids"] else ()
        clicked_raw_ids = {str(value) for value in raw_clicks}
        candidate_ids = [identity.article_id(value) for value in raw_candidates]
        clicked_ids = [
            identity.article_id(value) for value in raw_candidates if str(value) in clicked_raw_ids
        ]
        labels = [int(str(value) in clicked_raw_ids) for value in raw_candidates]
        raw_session = row.get(columns["session_id"]) if columns["session_id"] else None
        session_id = identity.session_id(raw_session) if raw_session is not None else None
        impressions.append(
            {
                "impression_id": impression_id,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": timestamp,
                "candidate_ids": candidate_ids,
                "clicked_ids": clicked_ids,
                "labels": labels,
                "source_split": source_split,
            }
        )
        for position, (article_id, label) in enumerate(zip(candidate_ids, labels, strict=True)):
            candidates.append(
                {
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "article_id": article_id,
                    "label": label,
                    "position": position,
                    "source_split": source_split,
                }
            )
    return impressions, candidates, users


def _read_histories(table: pa.Table) -> dict[str, tuple[tuple[str, ...], tuple[datetime, ...]]]:
    columns = _resolve_columns(table, HISTORY_COLUMNS)
    histories: dict[str, tuple[tuple[str, ...], tuple[datetime, ...]]] = {}
    for row in table.to_pylist():
        user_id = str(_required(row, columns["user_id"], "user_id"))
        article_ids = tuple(
            str(value) for value in _values(_required(row, columns["article_ids"], "article_ids"))
        )
        timestamps = tuple(
            _datetime(value, "history timestamp")
            for value in _values(_required(row, columns["timestamps"], "timestamps"))
        )
        if len(article_ids) != len(timestamps):
            raise ValueError(f"Mismatched EB-NeRD history lengths for user {user_id}")
        histories.setdefault(user_id, (article_ids, timestamps))
    return histories


def _normalize_histories(
    identity: DatasetIdentity,
    training_history: pa.Table,
    official_validation_history: pa.Table,
    training_users: Mapping[SourceSplit, set[str]],
    test_users: set[str],
) -> list[dict[str, Any]]:
    train_histories = _read_histories(training_history)
    test_histories = _read_histories(official_validation_history)
    rows: list[dict[str, Any]] = []
    targets: tuple[
        tuple[SourceSplit, set[str], Mapping[str, tuple[tuple[str, ...], tuple[datetime, ...]]]],
        ...,
    ] = (
        ("train", training_users["train"], train_histories),
        ("validation", training_users["validation"], train_histories),
        ("test", test_users, test_histories),
    )
    for source_split, users, source in targets:
        for raw_user_id in sorted(users):
            article_ids, timestamps = source.get(raw_user_id, ((), ()))
            rows.append(
                {
                    "user_id": identity.user_id(raw_user_id),
                    "article_ids": [identity.article_id(value) for value in article_ids],
                    "timestamps": list(timestamps),
                    "source_split": source_split,
                }
            )
    return rows


def _classify_behavior_path(path: Path) -> SourceSplit:
    parts = {part.lower() for part in path.parent.parts}
    if "train" in parts:
        return "train"
    if "validation" in parts or "val" in parts:
        return "test"
    raise ValueError(f"Cannot classify EB-NeRD behavior package: {path}")


def _discover_files(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    article_paths = tuple(root.rglob("articles.parquet"))
    if len(article_paths) != 1:
        raise FileNotFoundError(
            f"Expected one articles.parquet below {root}, found {len(article_paths)}"
        )
    behavior_paths = tuple(root.rglob("behaviors.parquet"))
    classified = {_classify_behavior_path(path): path for path in behavior_paths}
    if set(classified) != {"train", "test"}:
        raise FileNotFoundError(f"Expected train and validation behavior packages below {root}")
    train_behavior = classified["train"]
    test_behavior = classified["test"]
    train_history = train_behavior.with_name("history.parquet")
    test_history = test_behavior.with_name("history.parquet")
    if not train_history.is_file() or not test_history.is_file():
        raise FileNotFoundError("Expected history.parquet beside each EB-NeRD behaviors.parquet")
    return article_paths[0], train_behavior, train_history, test_behavior, test_history


def prepare_ebnerd(
    extracted_root: Path,
    variant: str,
    validation_days: int = 1,
) -> PreparationResult:
    identity = DatasetIdentity(name="ebnerd", variant=variant)
    article_path, train_behavior_path, train_history_path, test_behavior_path, test_history_path = (
        _discover_files(extracted_root)
    )
    train_behavior_table = pq.read_table(train_behavior_path)
    test_behavior_table = pq.read_table(test_behavior_path)
    train_columns = _resolve_columns(
        train_behavior_table, BEHAVIOR_COLUMNS, optional=("session_id", "clicked_ids")
    )
    test_columns = _resolve_columns(
        test_behavior_table, BEHAVIOR_COLUMNS, optional=("session_id", "clicked_ids")
    )
    train_timestamps = [
        _datetime(value, "timestamp")
        for value in train_behavior_table.column(str(train_columns["timestamp"])).to_pylist()
    ]
    test_timestamps = [
        _datetime(value, "timestamp")
        for value in test_behavior_table.column(str(test_columns["timestamp"])).to_pylist()
    ]
    if not test_timestamps:
        raise ValueError("Official EB-NeRD validation package is empty")
    temporal_split = official_temporal_split(train_timestamps, validation_days)
    train_impressions, train_candidates, training_users = _normalize_behavior_table(
        train_behavior_table, identity, temporal_split
    )
    test_impressions, test_candidates, test_users_by_split = _normalize_behavior_table(
        test_behavior_table, identity, "test"
    )
    article_rows, duplicate_count, conflict_count = _normalize_articles(
        pq.read_table(article_path), identity, temporal_split, min(test_timestamps)
    )
    history_rows = _normalize_histories(
        identity,
        pq.read_table(train_history_path),
        pq.read_table(test_history_path),
        training_users,
        test_users_by_split["test"],
    )
    tables = DatasetTables(
        articles=table_from_rows(article_rows, ARTICLES_SCHEMA),
        histories=table_from_rows(history_rows, HISTORIES_SCHEMA),
        impressions=table_from_rows([*train_impressions, *test_impressions], IMPRESSIONS_SCHEMA),
        candidates=table_from_rows([*train_candidates, *test_candidates], CANDIDATES_SCHEMA),
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


def _parquet_batches(path: Path, batch_size: int = 8192) -> Iterable[pa.Table]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield pa.Table.from_batches([batch], schema=parquet.schema_arrow)


def _parquet_columns(path: Path, aliases: Mapping[str, Sequence[str]]) -> dict[str, str]:
    available = set(pq.read_schema(path).names)
    resolved: dict[str, str] = {}
    for normalized, candidates in aliases.items():
        match = next((candidate for candidate in candidates if candidate in available), None)
        if match is None:
            names = ", ".join(candidates)
            raise ValueError(f"Missing required EB-NeRD column, expected one of: {names}")
        resolved[normalized] = match
    return resolved


def _parquet_timestamps(path: Path) -> Iterable[datetime]:
    column = _parquet_columns(path, {"timestamp": BEHAVIOR_COLUMNS["timestamp"]})["timestamp"]
    for table in _parquet_batches(path):
        yield from (_datetime(value, "timestamp") for value in table.column(column).to_pylist())


def _chunks[T](rows: Iterable[T], size: int = 8192) -> Iterable[list[T]]:
    chunk: list[T] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _concat[T](iterables: Iterable[Iterable[list[T]]]) -> Iterable[list[T]]:
    for rows in iterables:
        yield from rows


def _stream_ebnerd_articles(
    path: Path,
    identity: DatasetIdentity,
    temporal_split: TemporalSplit,
    test_start: datetime,
    connection: sqlite3.Connection,
    stats: dict[str, int],
) -> Iterable[list[dict[str, Any]]]:
    connection.execute(
        "CREATE TABLE articles ("
        "article_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    for table in _parquet_batches(path):
        rows, duplicates, conflicts = _normalize_articles(
            table, identity, temporal_split, test_start
        )
        stats["article_duplicates"] += duplicates
        stats["article_conflicts"] += conflicts
        for row in rows:
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
    rows: list[dict[str, Any]] = []
    for (payload,) in connection.execute("SELECT payload FROM articles ORDER BY rowid"):
        row = json.loads(payload)
        row["published_at"] = datetime.fromisoformat(row["published_at"])
        row["available_at"] = datetime.fromisoformat(row["available_at"])
        rows.append(row)
        if len(rows) == 8192:
            yield rows
            rows = []
    if rows:
        yield rows


def _stream_ebnerd_behaviors(
    path: Path,
    identity: DatasetIdentity,
    split_rule: SourceSplit | TemporalSplit,
    users: dict[SourceSplit, set[str]],
    kind: str,
) -> Iterable[list[dict[str, Any]]]:
    for table in _parquet_batches(path):
        impressions, candidates, batch_users = _normalize_behavior_table(
            table, identity, split_rule
        )
        for source_split, raw_users in batch_users.items():
            users[source_split].update(raw_users)
        rows = impressions if kind == "impressions" else candidates
        yield from _chunks(rows)


def _stream_ebnerd_histories(
    path: Path,
    identity: DatasetIdentity,
    source_splits: Sequence[SourceSplit],
    users: Mapping[SourceSplit, set[str]],
) -> Iterable[list[dict[str, Any]]]:
    seen: dict[SourceSplit, set[str]] = {source_split: set() for source_split in source_splits}
    for table in _parquet_batches(path):
        histories = _read_histories(table)
        rows: list[dict[str, Any]] = []
        for raw_user_id, (article_ids, timestamps) in histories.items():
            for source_split in source_splits:
                if raw_user_id not in users[source_split]:
                    continue
                if raw_user_id in seen[source_split]:
                    continue
                seen[source_split].add(raw_user_id)
                rows.append(
                    {
                        "user_id": identity.user_id(raw_user_id),
                        "article_ids": [identity.article_id(value) for value in article_ids],
                        "timestamps": list(timestamps),
                        "source_split": source_split,
                    }
                )
        yield from _chunks(rows)
    for source_split in source_splits:
        rows = [
            {
                "user_id": identity.user_id(raw_user_id),
                "article_ids": [],
                "timestamps": [],
                "source_split": source_split,
            }
            for raw_user_id in sorted(users[source_split] - seen[source_split])
        ]
        yield from _chunks(rows)


def prepare_ebnerd_streaming(
    extracted_root: Path,
    variant: str,
    validation_days: int,
    data_root: Path,
) -> PreparationResult:
    identity = DatasetIdentity(name="ebnerd", variant=variant)
    (
        article_path,
        train_behavior_path,
        train_history_path,
        test_behavior_path,
        test_history_path,
    ) = _discover_files(extracted_root)
    train_timestamps = _parquet_timestamps(train_behavior_path)
    test_start = min(_parquet_timestamps(test_behavior_path), default=None)
    if test_start is None:
        raise ValueError("Official EB-NeRD validation package is empty")
    temporal_split = official_temporal_split(train_timestamps, validation_days)
    paths: tuple[tuple[Path, SourceSplit | TemporalSplit], ...] = (
        (train_behavior_path, temporal_split),
        (test_behavior_path, "test"),
    )
    history_sources: tuple[tuple[Path, tuple[SourceSplit, ...]], ...] = (
        (train_history_path, ("train", "validation")),
        (test_history_path, ("test",)),
    )
    users: dict[SourceSplit, set[str]] = {"train": set(), "validation": set(), "test": set()}
    stats = {"article_duplicates": 0, "article_conflicts": 0}
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="ebnerd-prepare-", suffix=".sqlite", dir=data_root, delete=False
    ) as temporary:
        database_path = Path(temporary.name)
    connection = sqlite3.connect(database_path)
    try:
        result = write_feature_store_chunks(
            identity,
            PreparationStats(
                article_conflicts=0,
                article_duplicates=0,
                validation_start=temporal_split.validation_start,
            ),
            {
                "articles": _stream_ebnerd_articles(
                    article_path,
                    identity,
                    temporal_split,
                    test_start,
                    connection,
                    stats,
                ),
                "impressions": _concat(
                    (
                        _stream_ebnerd_behaviors(path, identity, split_rule, users, "impressions")
                        for path, split_rule in paths
                    )
                ),
                "candidates": _concat(
                    (
                        _stream_ebnerd_behaviors(path, identity, split_rule, users, "candidates")
                        for path, split_rule in paths
                    )
                ),
                "histories": _concat(
                    (
                        _stream_ebnerd_histories(
                            path,
                            identity,
                            source_splits,
                            users,
                        )
                        for path, source_splits in history_sources
                    )
                ),
            },
            data_root,
        )
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    result = replace(
        result,
        stats=PreparationStats(
            article_conflicts=stats["article_conflicts"],
            article_duplicates=stats["article_duplicates"],
            validation_start=temporal_split.validation_start,
        ),
    )
    return result
