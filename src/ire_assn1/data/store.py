from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ire_assn1.data.contracts import (
    ARTICLES_SCHEMA,
    CANDIDATES_SCHEMA,
    HISTORIES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    validate_table,
)
from ire_assn1.data.models import (
    DatasetIdentity,
    DatasetTables,
    PreparationResult,
    PreparationStats,
)

TABLE_SCHEMAS = {
    "articles": ARTICLES_SCHEMA,
    "histories": HISTORIES_SCHEMA,
    "impressions": IMPRESSIONS_SCHEMA,
    "candidates": CANDIDATES_SCHEMA,
}


def _write_table(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    validate_table(table, schema)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(result: PreparationResult) -> dict[str, object]:
    counts = (
        result.row_counts
        if result.tables is None
        else {name: getattr(result.tables, name).num_rows for name in TABLE_SCHEMAS}
    )
    return {
        "dataset": result.identity.name,
        "variant": result.identity.variant,
        "validation_start": result.stats.validation_start.isoformat(timespec="microseconds"),
        "article_duplicates": result.stats.article_duplicates,
        "article_conflicts": result.stats.article_conflicts,
        "rows": {name: counts[name] for name in TABLE_SCHEMAS},
        "files": {name: f"{name}.parquet" for name in TABLE_SCHEMAS},
    }


def write_feature_store(result: PreparationResult, data_root: Path) -> PreparationResult:
    if result.tables is None:
        raise ValueError("In-memory tables are required for write_feature_store")
    output_dir = data_root / "processed" / result.identity.name / result.identity.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, schema in TABLE_SCHEMAS.items():
        table = getattr(result.tables, name)
        if not isinstance(table, pa.Table):
            raise TypeError(f"Expected PyArrow table for {name}")
        _write_table(output_dir / f"{name}.parquet", table, schema)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            f"{json.dumps(_manifest(result), indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return replace(result, output_dir=output_dir)


def write_feature_store_chunks(
    identity: DatasetIdentity,
    stats: PreparationStats,
    chunks: Mapping[str, Iterable[Sequence[Mapping[str, object]]]],
    data_root: Path,
) -> PreparationResult:
    output_dir = data_root / "processed" / identity.name / identity.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    if set(chunks) != set(TABLE_SCHEMAS):
        raise ValueError("Streaming feature-store chunks must cover every table")
    for name, rows_iter in chunks.items():
        schema = TABLE_SCHEMAS[name]
        temporary = output_dir / f"{name}.parquet.tmp"
        writer: pq.ParquetWriter | None = None
        count = 0
        try:
            for rows in rows_iter:
                table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
                validate_table(table, schema)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                assert writer is not None
                writer.write_table(table)
                count += table.num_rows
            if writer is None:
                pq.write_table(pa.Table.from_pylist([], schema=schema), temporary)
            counts[name] = count
        finally:
            if writer is not None:
                writer.close()
            if temporary.is_file():
                os.replace(temporary, output_dir / f"{name}.parquet")
    result = PreparationResult(
        identity=identity,
        tables=None,
        stats=stats,
        output_dir=output_dir,
        row_counts=counts,
    )
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            f"{json.dumps(_manifest(result), indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def read_feature_store(path: Path) -> DatasetTables:
    tables = {
        name: validate_table(pq.read_table(path / f"{name}.parquet"), schema)
        for name, schema in TABLE_SCHEMAS.items()
    }
    return DatasetTables(
        articles=tables["articles"],
        histories=tables["histories"],
        impressions=tables["impressions"],
        candidates=tables["candidates"],
    )
