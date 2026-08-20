from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ire_assn1.paths import project_path
from ire_assn1.retrieval.bge import BGEEncoder, DenseRetriever
from ire_assn1.retrieval.bm25 import BM25Retriever
from ire_assn1.retrieval.indexes import ExactFaissIndex, HNSWFaissIndex
from ire_assn1.retrieval.pipeline import (
    evaluate_history_lengths_stream,
    full_corpus_retrieval,
    impression_candidate_scoring,
)
from ire_assn1.retrieval.profiles import PopularityModel
from ire_assn1.retrieval.types import Article, History, Impression, RetrievalResult, Retriever
from ire_assn1.settings import load_mapping


@dataclass(frozen=True, slots=True)
class RetrievalRunSummary:
    dataset: str
    variant: str
    system: str
    history_length: int | None
    validation_scores: Mapping[str, float]
    full_corpus_rows: int
    candidate_rows: int
    output_directory: Path


def _value[T](mapping: Mapping[str, object], key: str, expected: type[T]) -> T:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise ValueError(f"Configuration field {key} must be {expected.__name__}")
    return value


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Configuration field {key} must be a string or null")
    return value


def _float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Configuration field {key} must be numeric")
    return float(value)


def _integer(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Configuration field {key} must be an integer")
    return value


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Configuration field {key} must be a list")
    return value


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"Invalid timestamp: {value!r}")


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise ValueError(f"Expected a list of identifiers, received {type(value).__name__}")


def _timestamps(value: object) -> tuple[datetime | None, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(None if item is None else _timestamp(item) for item in value)
    raise ValueError(f"Expected a list of timestamps, received {type(value).__name__}")


def _integers(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(int(cast(int, item)) for item in value)
    raise ValueError(f"Expected a list of labels, received {type(value).__name__}")


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Normalized retrieval input does not exist: {path}")
    return cast(list[dict[str, object]], pq.read_table(path).to_pylist())


def _iter_rows(path: Path) -> Iterable[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Normalized retrieval input does not exist: {path}")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        yield from cast(list[dict[str, object]], pa.Table.from_batches([batch]).to_pylist())


def _read_articles(path: Path) -> dict[str, Article]:
    articles: dict[str, Article] = {}
    for row in _read_rows(path):
        article_id = str(row["article_id"])
        articles[article_id] = Article(
            article_id=article_id,
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            available_at=_timestamp(row["available_at"]),
        )
    return articles


def _read_histories(path: Path) -> dict[tuple[str, str], History]:
    histories: dict[tuple[str, str], History] = {}
    for row in _read_rows(path):
        source_split = str(row.get("source_split") or "")
        history = History(
            user_id=str(row["user_id"]),
            article_ids=_strings(row.get("article_ids")),
            timestamps=_timestamps(row.get("timestamps")),
        )
        histories[(history.user_id, source_split)] = history
    return histories


def _read_impressions(path: Path) -> tuple[Impression, ...]:
    return tuple(_iter_impressions(path))


def _iter_impressions(path: Path) -> Iterable[Impression]:
    for row in _iter_rows(path):
        yield Impression(
            impression_id=str(row["impression_id"]),
            user_id=str(row["user_id"]),
            timestamp=_timestamp(row["timestamp"]),
            candidate_ids=_strings(row.get("candidate_ids")),
            clicked_ids=_strings(row.get("clicked_ids")),
            labels=_integers(row.get("labels")),
            source_split=str(row.get("source_split") or ""),
        )


def _history_lengths(mapping: Mapping[str, object]) -> tuple[int | None, ...]:
    lengths: list[int | None] = []
    for value in _sequence(mapping, "history_lengths"):
        if value is None:
            lengths.append(None)
        elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
            lengths.append(value)
        else:
            raise ValueError("History lengths must contain positive integers or null")
    if not lengths:
        raise ValueError("At least one history length is required")
    return tuple(lengths)


def _selected_history_length(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("history_length")
    if value is not None and not isinstance(value, int):
        raise ValueError(f"Invalid selected history length in {path}")
    return value


def _create_retriever(
    mapping: Mapping[str, object],
    articles: Mapping[str, Article],
    output_directory: Path,
) -> Retriever:
    system = _value(mapping, "system", str)
    if system == "bm25":
        return BM25Retriever(
            articles,
            _value(mapping, "language", str),
            k1=_float(mapping, "k1", 1.2),
            b=_float(mapping, "b", 0.75),
        )
    if system != "bge":
        raise ValueError(f"Unsupported retrieval system: {system}")
    model = str(mapping.get("model", "BAAI/bge-m3"))
    revision = _optional_string(mapping, "revision")
    digest = hashlib.sha256()
    for article_id, article in sorted(articles.items()):
        digest.update(article_id.encode())
        digest.update(b"\0")
        digest.update(article.text.encode())
        digest.update(b"\0")
    expected_cache = {
        "model": model,
        "revision": revision,
        "article_text_sha256": digest.hexdigest(),
    }
    embeddings_path = output_directory / "article_embeddings.npy"
    identifiers_path = output_directory / "article_embedding_ids.json"
    metadata_path = output_directory / "article_embeddings.json"
    cache_valid = (
        embeddings_path.is_file()
        and identifiers_path.is_file()
        and metadata_path.is_file()
        and json.loads(metadata_path.read_text(encoding="utf-8")) == expected_cache
    )
    if cache_valid:
        vectors = np.asarray(np.load(embeddings_path), dtype=np.float32)
        identifiers = json.loads(identifiers_path.read_text(encoding="utf-8"))
        if not isinstance(identifiers, list) or not all(
            isinstance(item, str) for item in identifiers
        ):
            raise ValueError("Cached article embedding identifiers are invalid")
        article_ids = tuple(cast(list[str], identifiers))
    else:
        encoder = BGEEncoder(
            model=model,
            revision=revision,
            batch_size=_integer(mapping, "batch_size", 32),
            device=_optional_string(mapping, "device"),
        )
        article_ids, vectors = encoder.encode_articles(articles)
        output_directory.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, vectors)
        identifiers_path.write_text(json.dumps(article_ids), encoding="utf-8")
        metadata_path.write_text(
            json.dumps(expected_cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    index_name = str(mapping.get("index", "exact"))
    if index_name == "exact":
        index = ExactFaissIndex(article_ids, vectors)
    elif index_name == "hnsw":
        index = HNSWFaissIndex(
            article_ids,
            vectors,
            m=_integer(mapping, "hnsw_m", 32),
            ef_search=_integer(mapping, "hnsw_ef_search", 128),
        )
    else:
        raise ValueError(f"Unsupported semantic index: {index_name}")
    return DenseRetriever(article_ids, vectors, index)


def _result_rows(results: Sequence[RetrievalResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for article in result.articles:
            rows.append(
                {
                    "impression_id": result.impression_id,
                    "user_id": result.user_id,
                    "timestamp": result.timestamp,
                    "article_id": article.article_id,
                    "label": article.label,
                    "position": article.position,
                    "source_split": result.source_split,
                    "system": result.system,
                    "score": article.score,
                    "rank": article.rank,
                    "mode": result.mode,
                    "used_fallback": result.used_fallback,
                }
            )
    return rows


def _write_results(path: Path, results: Iterable[RetrievalResult]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("impression_id", pa.string()),
            ("user_id", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("article_id", pa.string()),
            ("label", pa.int64()),
            ("position", pa.int64()),
            ("source_split", pa.string()),
            ("system", pa.string()),
            ("score", pa.float64()),
            ("rank", pa.int64()),
            ("mode", pa.string()),
            ("used_fallback", pa.bool_()),
        ]
    )
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for result in results:
            rows = _result_rows((result,))
            if not rows:
                continue
            if writer is None:
                writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            assert writer is not None
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            count += len(rows)
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=schema), temporary)
    finally:
        if writer is not None:
            writer.close()
        if temporary.is_file():
            os.replace(temporary, path)
    return count


def retrieve_from_config(config: str | Path | Mapping[str, object]) -> RetrievalRunSummary:
    mapping = dict(config) if isinstance(config, Mapping) else load_mapping(config)
    dataset = _value(mapping, "name", str)
    variant = _value(mapping, "variant", str)
    system = _value(mapping, "system", str)
    data_root = project_path(str(mapping.get("data", "data")))
    input_directory = project_path(
        str(mapping.get("input_directory", data_root / "processed" / dataset / variant))
    )
    output_directory = project_path(
        str(mapping.get("retrieval_output", data_root / "retrieval" / dataset / variant / system))
    )
    articles = _read_articles(input_directory / "articles.parquet")
    histories = _read_histories(input_directory / "histories.parquet")
    impressions_path = input_directory / "impressions.parquet"

    def impressions() -> Iterable[Impression]:
        return _iter_impressions(impressions_path)

    popularity = PopularityModel.from_impressions(impressions())
    retriever = _create_retriever(mapping, articles, output_directory)
    candidates = _history_lengths(mapping)
    selected, validation_scores = evaluate_history_lengths_stream(
        impressions,
        histories,
        articles,
        retriever,
        popularity,
        candidates,
        k=100,
    )
    full_rows = _write_results(
        output_directory / "full_corpus.parquet",
        (
            full_corpus_retrieval(
                impression,
                histories.get((impression.user_id, impression.source_split)),
                articles,
                retriever,
                popularity,
                selected,
                200,
            )
            for impression in tqdm(
                (value for value in impressions() if value.source_split != "train"),
                desc=f"{system} full-corpus retrieval",
                unit="impression",
            )
        ),
    )
    candidate_rows = _write_results(
        output_directory / "impression_candidates.parquet",
        (
            impression_candidate_scoring(
                impression,
                histories.get((impression.user_id, impression.source_split)),
                articles,
                retriever,
                popularity,
                selected,
            )
            for impression in tqdm(
                (value for value in impressions() if value.source_split != "train"),
                desc=f"{system} candidate scoring",
                unit="impression",
            )
        ),
    )
    serialized_scores = {
        "all" if key is None else str(key): value for key, value in validation_scores.items()
    }
    selection = {
        "dataset": dataset,
        "variant": variant,
        "system": system,
        "selection_metric": "recall_at_100",
        "history_length": selected,
        "validation_scores": serialized_scores,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RetrievalRunSummary(
        dataset=dataset,
        variant=variant,
        system=system,
        history_length=selected,
        validation_scores=serialized_scores,
        full_corpus_rows=full_rows,
        candidate_rows=candidate_rows,
        output_directory=output_directory,
    )
