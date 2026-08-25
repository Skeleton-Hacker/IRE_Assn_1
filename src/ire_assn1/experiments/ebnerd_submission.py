from __future__ import annotations

import importlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ire_assn1.paths import project_path

_ID_COLUMNS = ("article_id", "articleId", "id")
_USER_COLUMNS = ("user_id",)
_HISTORY_COLUMNS = ("article_id_fixed", "article_ids", "article_ids_fixed")
_IMPRESSION_COLUMNS = ("impression_id",)
_CANDIDATE_COLUMNS = ("article_ids_inview", "inview_article_ids")


def _single(root: Path, name: str) -> Path:
    paths = tuple(path for path in root.rglob(name) if "__MACOSX" not in path.parts)
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {name} below {root}, found {len(paths)}")
    return paths[0]


def _column(schema: pa.Schema, candidates: Sequence[str], kind: str) -> str:
    match = next((candidate for candidate in candidates if candidate in schema.names), None)
    if match is None:
        raise ValueError(f"Missing EB-NeRD {kind} column in {schema.names}")
    return match


def _is_vector(field: pa.Field) -> bool:
    field_type = field.type
    if not (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    ):
        return False
    return pa.types.is_floating(field_type.value_type)


def _embedding_file(mapping: Mapping[str, object], raw_root: Path) -> Path:
    configured = mapping.get("embedding_path")
    root = project_path(str(configured)) if configured is not None else raw_root / "embeddings"
    paths = tuple(
        path
        for path in root.rglob("*.parquet")
        if "__MACOSX" not in path.parts and "bert" in str(path).lower()
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one supplied BERT embedding Parquet below {root}, found {len(paths)}"
        )
    return paths[0]


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": parquet.metadata.num_rows,
    }


def _positive_integer(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _vector_values(array: pa.Array) -> NDArray[np.float32]:
    if not hasattr(array, "flatten"):
        raise ValueError("Embedding column must be a list array")
    offsets = np.asarray(array.offsets)
    lengths = np.diff(offsets)
    if len(lengths) == 0:
        return np.empty((0, 0), dtype=np.float32)
    if not np.all(lengths == lengths[0]) or lengths[0] <= 0:
        raise ValueError("Embedding vectors must have one fixed positive dimension")
    flattened = np.asarray(array.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    return flattened.reshape(len(lengths), int(lengths[0]))


def _load_embeddings(path: Path) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    id_column = _column(schema, _ID_COLUMNS, "article ID")
    vector_fields = [field for field in schema if field.name != id_column and _is_vector(field)]
    bert_fields = [field for field in vector_fields if "bert" in field.name.lower()]
    selected = bert_fields if bert_fields else vector_fields
    if len(selected) != 1:
        names = ", ".join(field.name for field in vector_fields)
        raise ValueError(f"Expected one BERT vector column in {path}, found: {names}")
    vector_column = selected[0].name
    batches = iter(parquet.iter_batches(columns=[id_column, vector_column], batch_size=8192))
    first = next(batches, None)
    if first is None:
        raise ValueError(f"Embedding file is empty: {path}")
    first_vectors = _vector_values(first.column(vector_column))
    rows = parquet.metadata.num_rows
    article_ids = np.empty(rows, dtype=np.int64)
    vectors = np.empty((rows, first_vectors.shape[1]), dtype=np.float32)
    offset = 0

    def consume(batch: pa.RecordBatch, values: NDArray[np.float32] | None = None) -> None:
        nonlocal offset
        matrix = values if values is not None else _vector_values(batch.column(vector_column))
        size = batch.num_rows
        article_ids[offset : offset + size] = np.asarray(
            batch.column(id_column).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        vectors[offset : offset + size] = matrix
        offset += size

    consume(first, first_vectors)
    progress = tqdm(total=rows, initial=first.num_rows, desc="load EB-NeRD BERT", unit="article")
    for batch in batches:
        consume(batch)
        progress.update(batch.num_rows)
    progress.close()
    if offset != rows or len(np.unique(article_ids)) != rows:
        raise ValueError("Supplied EB-NeRD embeddings have invalid article IDs")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return article_ids, vectors


def _article_lookup(article_ids: NDArray[np.int64]) -> NDArray[np.int32]:
    if np.any(article_ids < 0):
        raise ValueError("EB-NeRD article IDs must be non-negative")
    lookup = np.full(int(article_ids.max()) + 1, -1, dtype=np.int32)
    lookup[article_ids] = np.arange(len(article_ids), dtype=np.int32)
    return lookup


def _article_positions(
    article_ids: NDArray[np.int64], lookup: NDArray[np.int32]
) -> NDArray[np.int32]:
    positions = np.full(article_ids.shape, -1, dtype=np.int32)
    valid = (article_ids >= 0) & (article_ids < len(lookup))
    positions[valid] = lookup[article_ids[valid]]
    return positions


def _profile_cache(
    history_path: Path,
    embedding_path: Path,
    article_vectors: NDArray[np.float32],
    article_lookup: NDArray[np.int32],
    history_length: int,
    cache_root: Path,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    identifiers_path = cache_root / "user_ids.npy"
    profiles_path = cache_root / "user_profiles.npy"
    metadata_path = cache_root / "user_profiles.json"
    expected = {
        "history": _fingerprint(history_path),
        "embedding": _fingerprint(embedding_path),
        "history_length": history_length,
    }
    if identifiers_path.is_file() and profiles_path.is_file() and metadata_path.is_file():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual == expected:
            identifiers = np.load(identifiers_path, mmap_mode="r")
            profiles = np.load(profiles_path, mmap_mode="r")
            if profiles.shape == (len(identifiers), article_vectors.shape[1]):
                return identifiers, profiles

    parquet = pq.ParquetFile(history_path)
    schema = parquet.schema_arrow
    user_column = _column(schema, _USER_COLUMNS, "user ID")
    history_column = _column(schema, _HISTORY_COLUMNS, "history")
    rows = parquet.metadata.num_rows
    identifiers_temporary = identifiers_path.with_suffix(".npy.tmp")
    profiles_temporary = profiles_path.with_suffix(".npy.tmp")
    identifiers = np.lib.format.open_memmap(
        identifiers_temporary, mode="w+", dtype=np.int64, shape=(rows,)
    )
    profiles = np.lib.format.open_memmap(
        profiles_temporary,
        mode="w+",
        dtype=np.float32,
        shape=(rows, article_vectors.shape[1]),
    )
    offset = 0
    progress = tqdm(total=rows, desc="build EB-NeRD user profiles", unit="user")
    for batch in parquet.iter_batches(columns=[user_column, history_column], batch_size=8192):
        size = batch.num_rows
        identifiers[offset : offset + size] = np.asarray(
            batch.column(user_column).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        raw = np.full((size, history_length), -1, dtype=np.int64)
        for row, values in enumerate(batch.column(history_column).to_pylist()):
            selected = values[-history_length:] if values else ()
            if selected:
                raw[row, -len(selected) :] = np.asarray(selected, dtype=np.int64)
        sums = np.zeros((size, article_vectors.shape[1]), dtype=np.float32)
        for column in range(history_length):
            positions = _article_positions(raw[:, column], article_lookup)
            valid = positions >= 0
            sums[valid] += article_vectors[positions[valid]]
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        np.divide(sums, norms, out=sums, where=norms > 0)
        profiles[offset : offset + size] = sums
        offset += size
        progress.update(size)
    progress.close()
    identifiers.flush()
    profiles.flush()
    del identifiers, profiles
    if offset != rows:
        raise ValueError("EB-NeRD history row count changed during profile construction")
    os.replace(identifiers_temporary, identifiers_path)
    os.replace(profiles_temporary, profiles_path)
    metadata_path.write_text(
        f"{json.dumps(expected, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    loaded_identifiers = np.load(identifiers_path, mmap_mode="r")
    if len(np.unique(loaded_identifiers)) != len(loaded_identifiers):
        raise ValueError("EB-NeRD test histories contain duplicate user IDs")
    return loaded_identifiers, np.load(profiles_path, mmap_mode="r")


class _SemanticScorer:
    def __init__(self, vectors: NDArray[np.float32], device: str) -> None:
        self.vectors = vectors
        self.device = device
        self.torch: Any = None
        self.tensor: Any = None
        if device == "cuda":
            torch = importlib.import_module("torch")

            if not torch.cuda.is_available():
                raise RuntimeError("submission_device is cuda but CUDA is unavailable")
            self.torch = torch
            self.tensor = torch.from_numpy(vectors).to("cuda")
        elif device != "cpu":
            raise ValueError("submission_device must be cpu or cuda")

    def score(
        self,
        profiles: NDArray[np.float32],
        positions: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        valid = positions >= 0
        safe = np.where(valid, positions, 0)
        if self.device == "cuda":
            torch = self.torch
            profile_tensor = torch.from_numpy(np.asarray(profiles, dtype=np.float32)).to("cuda")
            position_tensor = torch.from_numpy(safe.astype(np.int64, copy=False)).to("cuda")
            scores = torch.einsum("bnd,bd->bn", self.tensor[position_tensor], profile_tensor)
            result = scores.cpu().numpy()
        else:
            result = np.einsum(
                "bnd,bd->bn",
                self.vectors[safe],
                np.asarray(profiles, dtype=np.float32),
                optimize=True,
            )
        result[~valid] = -np.inf
        return result


def _rank_batch(
    batch: pa.RecordBatch,
    columns: tuple[str, str, str],
    user_positions: Mapping[int, int],
    profiles: NDArray[np.float32],
    article_lookup: NDArray[np.int32],
    scorer: _SemanticScorer,
) -> list[str]:
    impression_column, user_column, candidate_column = columns
    impression_ids = batch.column(impression_column).to_pylist()
    user_ids = batch.column(user_column).to_pylist()
    candidates = batch.column(candidate_column).to_pylist()
    lengths = np.fromiter(
        (len(values) for values in candidates), dtype=np.int32, count=len(candidates)
    )
    width = int(lengths.max(initial=0))
    if width == 0:
        raise ValueError("EB-NeRD test impression has no in-view candidates")
    raw_candidates = np.full((len(candidates), width), -1, dtype=np.int64)
    for row, values in enumerate(candidates):
        raw_candidates[row, : len(values)] = np.asarray(values, dtype=np.int64)
    profile_rows = np.fromiter(
        (user_positions.get(int(user_id), -1) for user_id in user_ids),
        dtype=np.int64,
        count=len(user_ids),
    )
    batch_profiles = np.zeros((len(user_ids), profiles.shape[1]), dtype=np.float32)
    known_users = profile_rows >= 0
    batch_profiles[known_users] = profiles[profile_rows[known_users]]
    article_positions = _article_positions(raw_candidates, article_lookup)
    valid = np.arange(width)[None, :] < lengths[:, None]
    missing = valid & (article_positions < 0)
    if np.any(missing):
        article_id = int(raw_candidates[missing][0])
        raise ValueError(f"Supplied BERT embeddings do not cover candidate article {article_id}")
    scores = scorer.score(batch_profiles, article_positions)
    scores[~valid] = -np.inf
    ties = np.where(valid, raw_candidates, np.iinfo(np.int64).max)
    order = np.lexsort((ties, -scores), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(1, width + 1, dtype=np.int64), order.shape),
        axis=1,
    )
    return [
        f"{impression_id} [{','.join(map(str, ranks[row, :length]))}]\n"
        for row, (impression_id, length) in enumerate(zip(impression_ids, lengths, strict=True))
    ]


def _valid_chunk(path: Path, marker: Path, rows: int, key: Mapping[str, object]) -> bool:
    if not path.is_file() or not marker.is_file() or path.stat().st_size == 0:
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return payload == {"rows": rows, "key": key}


def write_ebnerd_semantic_submission(
    mapping: Mapping[str, object],
) -> tuple[Path, Path]:
    if mapping.get("name") != "ebnerd" or mapping.get("variant") != "large":
        raise ValueError("Direct semantic submission requires EB-NeRD large")
    data_root = project_path(str(mapping.get("data", "data")))
    raw_root = data_root / "raw" / "ebnerd" / "large"
    test_root = raw_root / "competition_test"
    behavior_path = _single(test_root, "behaviors.parquet")
    history_path = _single(test_root, "history.parquet")
    embedding_path = _embedding_file(mapping, raw_root)
    history_length = _positive_integer(mapping, "submission_history_length", 20)
    batch_size = _positive_integer(mapping, "submission_batch_size", 2048)
    device = str(mapping.get("submission_device", "cuda"))
    article_ids, article_vectors = _load_embeddings(embedding_path)
    lookup = _article_lookup(article_ids)
    cache_root = (
        data_root / "cache" / "ebnerd" / "large" / "bert_submission" / f"history_{history_length}"
    )
    user_ids, profiles = _profile_cache(
        history_path,
        embedding_path,
        article_vectors,
        lookup,
        history_length,
        cache_root,
    )
    user_positions = {int(user_id): position for position, user_id in enumerate(user_ids)}
    scorer = _SemanticScorer(article_vectors, device)
    parquet = pq.ParquetFile(behavior_path)
    schema = parquet.schema_arrow
    columns = (
        _column(schema, _IMPRESSION_COLUMNS, "impression ID"),
        _column(schema, _USER_COLUMNS, "user ID"),
        _column(schema, _CANDIDATE_COLUMNS, "in-view candidates"),
    )
    key = {
        "behavior": _fingerprint(behavior_path),
        "history": _fingerprint(history_path),
        "embedding": _fingerprint(embedding_path),
        "history_length": history_length,
    }
    chunk_root = cache_root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=parquet.metadata.num_rows,
        desc="EB-NeRD BERT submission",
        unit="impression",
    )
    chunk_paths: list[Path] = []
    written = 0
    for row_group in range(parquet.num_row_groups):
        rows = parquet.metadata.row_group(row_group).num_rows
        chunk = chunk_root / f"{row_group:05d}.txt"
        marker = chunk.with_suffix(".json")
        chunk_paths.append(chunk)
        if _valid_chunk(chunk, marker, rows, key):
            progress.update(rows)
            written += rows
            continue
        temporary = chunk.with_suffix(".txt.tmp")
        row_count = 0
        with temporary.open("w", encoding="utf-8") as output:
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=list(columns),
            ):
                lines = _rank_batch(
                    batch,
                    columns,
                    user_positions,
                    profiles,
                    lookup,
                    scorer,
                )
                output.writelines(lines)
                row_count += len(lines)
                progress.update(len(lines))
        if row_count != rows:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Row group {row_group} produced {row_count} predictions, expected {rows}"
            )
        os.replace(temporary, chunk)
        marker.write_text(
            f"{json.dumps({'rows': rows, 'key': key}, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        written += rows
    progress.close()
    if written != parquet.metadata.num_rows:
        raise ValueError(
            f"Submission produced {written} predictions, expected {parquet.metadata.num_rows}"
        )
    output = project_path(
        str(
            mapping.get(
                "submission",
                Path("output")
                / "submissions"
                / "ebnerd"
                / "large"
                / f"bert-h{history_length}"
                / "predictions.txt",
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(".txt.tmp")
    with temporary_output.open("wb") as destination:
        for chunk in chunk_paths:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    os.replace(temporary_output, output)
    archive = output.with_suffix(".zip")
    temporary_archive = archive.with_suffix(".zip.tmp")
    with ZipFile(temporary_archive, "w", compression=ZIP_DEFLATED) as destination:
        destination.write(output, arcname="predictions.txt")
    os.replace(temporary_archive, archive)
    print(output)
    print(archive)
    return output, archive
