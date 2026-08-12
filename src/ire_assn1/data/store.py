from __future__ import annotations

import json
import os
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
from ire_assn1.data.models import DatasetTables, PreparationResult

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
    tables = result.tables
    return {
        "dataset": result.identity.name,
        "variant": result.identity.variant,
        "validation_start": result.stats.validation_start.isoformat(timespec="microseconds"),
        "article_duplicates": result.stats.article_duplicates,
        "article_conflicts": result.stats.article_conflicts,
        "rows": {
            "articles": tables.articles.num_rows,
            "histories": tables.histories.num_rows,
            "impressions": tables.impressions.num_rows,
            "candidates": tables.candidates.num_rows,
        },
        "files": {name: f"{name}.parquet" for name in TABLE_SCHEMAS},
    }


def write_feature_store(result: PreparationResult, data_root: Path) -> PreparationResult:
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
