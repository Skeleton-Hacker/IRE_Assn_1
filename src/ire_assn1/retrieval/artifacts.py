from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ire_assn1.paths import project_path

_ID_COLUMNS = ("article_id", "articleId", "id")
_SOURCE_ALIASES = {
    "bert": ("bert", "mbert", "multilingual"),
    "roberta": ("roberta", "xlm"),
    "contrastive": ("contrastive", "document_vector"),
}


@dataclass(frozen=True, slots=True)
class ProvidedEmbeddings:
    article_ids: tuple[str, ...]
    vectors: NDArray[np.float32]


def _is_vector(field: pa.Field) -> bool:
    field_type = field.type
    if (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    ):
        value_type = field_type.value_type
    else:
        return False
    return pa.types.is_floating(value_type)


def _source_matches(value: str, source: str) -> bool:
    normalized = value.lower()
    if source == "bert" and "roberta" in normalized:
        return False
    return any(alias in normalized for alias in _SOURCE_ALIASES[source])


def _path_matches(path: Path, source: str) -> bool:
    normalized = " ".join(part.lower() for part in path.parts)
    return _source_matches(normalized, source)


def _artifact_paths(
    data_root: Path,
    dataset: str,
    variant: str,
    mapping: Mapping[str, object],
) -> tuple[Path, ...]:
    configured = mapping.get("embedding_path")
    if configured is not None:
        path = project_path(str(configured))
        if path.is_file():
            return (path,)
        if path.is_dir():
            return tuple(sorted(path.rglob("*.parquet")))
        raise FileNotFoundError(f"Configured embedding path does not exist: {path}")

    root = data_root / "raw" / dataset / variant
    if not root.is_dir():
        raise FileNotFoundError(f"EB-NeRD raw data directory does not exist: {root}")
    artifacts = tuple(sorted(root.rglob("artifacts.parquet")))
    if artifacts:
        return artifacts
    return tuple(
        sorted(
            path
            for path in root.rglob("*.parquet")
            if "embedding" in path.name.lower() or "vector" in path.name.lower()
        )
    )


def _columns(path: Path, source: str, explicit_column: str | None) -> tuple[str, str]:
    schema = pq.ParquetFile(path).schema_arrow
    id_column = next((name for name in _ID_COLUMNS if name in schema.names), None)
    if id_column is None:
        raise ValueError(f"Embedding file has no article ID column: {path}")
    vector_fields = [field for field in schema if field.name != id_column and _is_vector(field)]
    if explicit_column is not None:
        if explicit_column not in schema.names:
            raise ValueError(f"Embedding column is missing from {path}: {explicit_column}")
        field = schema.field(explicit_column)
        if not _is_vector(field):
            raise ValueError(f"Embedding column is not a float vector: {path}:{explicit_column}")
        return id_column, explicit_column
    matching = [field for field in vector_fields if _source_matches(field.name, source)]
    if len(matching) == 1:
        return id_column, matching[0].name
    if len(vector_fields) == 1:
        source_named = any(
            alias in path.name.lower() for alias in ("bert", "roberta", "contrastive")
        )
        if source_named and not _path_matches(path, source):
            raise ValueError(f"Embedding file does not match source {source}: {path}")
        return id_column, vector_fields[0].name
    names = ", ".join(field.name for field in vector_fields)
    raise ValueError(f"Could not select {source} embedding from {path}; vector columns: {names}")


def _canonical_id(
    value: object,
    articles: Mapping[str, object],
    dataset: str,
    variant: str,
) -> str | None:
    raw = str(value)
    if raw in articles:
        return raw
    candidate = f"{dataset}:{variant}:article:{raw}"
    return candidate if candidate in articles else None


def load_provided_embeddings(
    data_root: Path,
    dataset: str,
    variant: str,
    articles: Mapping[str, object],
    mapping: Mapping[str, object],
) -> ProvidedEmbeddings:
    source = mapping.get("embedding_source", "bert")
    if not isinstance(source, str) or source not in _SOURCE_ALIASES:
        raise ValueError("embedding_source must be bert, roberta, or contrastive")
    explicit_column = mapping.get("embedding_column")
    if explicit_column is not None and not isinstance(explicit_column, str):
        raise ValueError("embedding_column must be a string when provided")
    paths = _artifact_paths(data_root, dataset, variant, mapping)
    if not paths:
        raise FileNotFoundError(f"No EB-NeRD embedding Parquet files found below {data_root}")

    records: dict[str, NDArray[np.float32]] = {}
    dimensions: int | None = None
    for path in paths:
        try:
            id_column, vector_column = _columns(path, source, explicit_column)
        except ValueError:
            if explicit_column is None and len(paths) > 1 and not _path_matches(path, source):
                continue
            raise
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=[id_column, vector_column], batch_size=8192):
            ids = batch.column(id_column).to_pylist()
            values = batch.column(vector_column).to_pylist()
            for raw_id, value in zip(ids, values, strict=True):
                article_id = _canonical_id(raw_id, articles, dataset, variant)
                if article_id is None or article_id in records:
                    continue
                vector = np.asarray(value, dtype=np.float32)
                if vector.ndim != 1 or vector.size == 0:
                    raise ValueError(f"Invalid embedding vector in {path} for article {raw_id}")
                if dimensions is None:
                    dimensions = int(vector.size)
                elif vector.size != dimensions:
                    raise ValueError(f"Inconsistent embedding dimensions in {path}")
                records[article_id] = vector
    if not records:
        raise ValueError(f"No supplied {source} embeddings matched processed article IDs")
    article_ids = tuple(sorted(records))
    vectors = np.stack([records[article_id] for article_id in article_ids]).astype(np.float32)
    return ProvidedEmbeddings(article_ids, vectors)
