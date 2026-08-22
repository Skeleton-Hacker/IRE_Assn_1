from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa

TIMESTAMP = pa.timestamp("us")
NULLABLE_TIMESTAMP_LIST = pa.list_(pa.field("item", TIMESTAMP, nullable=True))
STRING_LIST = pa.list_(pa.field("item", pa.string(), nullable=False))
LABEL_LIST = pa.list_(pa.field("item", pa.int8(), nullable=False))

ARTICLES_SCHEMA = pa.schema(
    [
        pa.field("article_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("abstract", pa.string(), nullable=False),
        pa.field("body", pa.string()),
        pa.field("category", pa.string()),
        pa.field("subcategory", pa.string()),
        pa.field("entities", STRING_LIST, nullable=False),
        pa.field("published_at", TIMESTAMP),
        pa.field("available_at", TIMESTAMP, nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
    ]
)

HISTORIES_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("article_ids", STRING_LIST, nullable=False),
        pa.field("timestamps", NULLABLE_TIMESTAMP_LIST, nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("impression_id", pa.string()),
    ]
)

IMPRESSIONS_SCHEMA = pa.schema(
    [
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string()),
        pa.field("timestamp", TIMESTAMP, nullable=False),
        pa.field("candidate_ids", STRING_LIST, nullable=False),
        pa.field("clicked_ids", STRING_LIST, nullable=False),
        pa.field("labels", LABEL_LIST, nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
    ]
)

CANDIDATES_SCHEMA = pa.schema(
    [
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", TIMESTAMP, nullable=False),
        pa.field("article_id", pa.string(), nullable=False),
        pa.field("label", pa.int8(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
    ]
)

RETRIEVAL_SCHEMA = pa.schema(
    [
        *CANDIDATES_SCHEMA,
        pa.field("system", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False),
    ]
)


def table_from_rows(rows: Iterable[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    return validate_table(table, schema)


def validate_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    if table.schema != schema:
        raise ValueError(f"Schema mismatch: expected {schema}, received {table.schema}")
    table.validate(full=True)
    for field in schema:
        if not field.nullable and table.column(field.name).null_count:
            raise ValueError(f"Null values in required field: {field.name}")
    return table
