from __future__ import annotations

import hashlib
import importlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any, Protocol, cast

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from ire_assn1.experiments.manifests import device_identity
from ire_assn1.experiments.resources import PeakRssSampler
from ire_assn1.paths import project_path
from ire_assn1.retrieval.pipeline import full_corpus_retrieval, history_for_impression
from ire_assn1.retrieval.profiles import PopularityModel
from ire_assn1.retrieval.runner import (
    _create_retriever,
    _iter_impressions,
    _read_articles,
    _read_histories,
    _selected_history_length,
)
from ire_assn1.retrieval.types import Article, History, Impression, Retriever
from ire_assn1.settings import config_hash, load_mapping


def _peak_cuda(reset: bool = False) -> int:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return 0
    return int(torch.cuda.max_memory_allocated())


class _IndexedRetriever(Protocol):
    @property
    def index_size_bytes(self) -> int: ...


def _benchmark_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, str | Path):
        return load_mapping(value)
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("benchmark must be a path or mapping")


def _index_bytes(retriever: Retriever) -> int:
    return cast(_IndexedRetriever, retriever).index_size_bytes


def _integer_parameter(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_parameter(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _index_parameters(index: str, mapping: Mapping[str, object]) -> dict[str, int | float | str]:
    if index == "hnsw":
        return {
            "metric": "inner_product",
            "m": _integer_parameter(mapping, "hnsw_m", 32),
            "ef_search": _integer_parameter(mapping, "hnsw_ef_search", 128),
        }
    if index == "exact":
        return {"metric": "inner_product", "implementation": "faiss_index_flat_ip"}
    return {
        "k1": _float_parameter(mapping, "k1", 1.2),
        "b": _float_parameter(mapping, "b", 0.75),
    }


def _workload_hash(impressions: Sequence[Impression]) -> str:
    digest = hashlib.sha256()
    for impression in impressions:
        digest.update(impression.impression_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_workload(
    impressions: Sequence[Impression],
    histories: Mapping[tuple[str, str], History],
    articles: Mapping[str, Article],
    retriever: Retriever,
    popularity: PopularityModel,
    selected: int | None,
    collect_samples: bool,
) -> tuple[int, str, list[dict[str, object]]]:
    digest = hashlib.sha256()
    samples: list[dict[str, object]] = []
    rows = 0
    for impression_number, impression in enumerate(impressions):
        result = full_corpus_retrieval(
            impression,
            history_for_impression(histories, impression),
            articles,
            retriever,
            popularity,
            selected,
            200,
        )
        digest.update(impression.impression_id.encode())
        digest.update(b"\0")
        rows += len(result.articles)
        for article in result.articles:
            digest.update(article.article_id.encode())
            digest.update(b"\0")
            digest.update(str(article.rank).encode())
            digest.update(b"\0")
            if collect_samples and impression_number < 20:
                samples.append(
                    {
                        "impression_id": impression.impression_id,
                        "article_id": article.article_id,
                        "rank": article.rank,
                        "score": article.score,
                    }
                )
    return rows, digest.hexdigest(), samples


def benchmark_from_config(config: Path | Mapping[str, object]) -> Path:
    mapping = dict(config) if isinstance(config, Mapping) else load_mapping(config)
    benchmark = _benchmark_mapping(mapping.get("benchmark", "config/benchmark.yaml"))
    name = str(mapping["name"])
    variant = str(mapping["variant"])
    system = str(mapping["system"])
    processed = project_path("data") / "processed" / name / variant
    retrieval = project_path("data") / "retrieval" / name / variant / system
    articles = _read_articles(processed / "articles.parquet")
    histories = _read_histories(processed / "histories.parquet")
    limit_value = benchmark.get("max_impressions", 1000)
    if not isinstance(limit_value, int) or isinstance(limit_value, bool) or limit_value <= 0:
        raise ValueError("benchmark.max_impressions must be a positive integer")
    impressions = tuple(
        islice(
            (
                impression
                for impression in _iter_impressions(processed / "impressions.parquet")
                if impression.source_split == "test"
            ),
            limit_value,
        )
    )
    popularity = PopularityModel.from_impressions(
        _iter_impressions(processed / "impressions.parquet")
    )
    selected = _selected_history_length(retrieval / "selection.json")
    indexes = benchmark.get("semantic_indexes", ["exact", "hnsw"]) if system == "bge" else ["bm25"]
    records: list[dict[str, Any]] = []
    sample_rows: list[dict[str, object]] = []
    for index_value in indexes:
        index = str(index_value)
        resolved = {**mapping, "index": index} if system == "bge" else mapping
        retriever = _create_retriever(resolved, articles, retrieval)
        for fraction_value in benchmark.get("fractions", [0.25, 0.5, 1.0]):
            fraction = float(fraction_value)
            size = max(1, round(len(impressions) * fraction))
            sample = impressions[:size]
            for repetition in range(int(benchmark.get("repetitions", 3))):
                _peak_cuda(reset=True)
                rss = PeakRssSampler()
                rss.start()
                started = time.perf_counter()
                try:
                    rows, ranking_hash, fixed = _run_workload(
                        sample,
                        histories,
                        articles,
                        retriever,
                        popularity,
                        selected,
                        collect_samples=fraction == 1.0 and repetition == 0,
                    )
                finally:
                    elapsed = time.perf_counter() - started
                    rss.stop()
                if fixed:
                    sample_rows.extend(
                        {
                            "dataset": name,
                            "variant": variant,
                            "system": system,
                            "index": index,
                            **row,
                        }
                        for row in fixed
                    )
                records.append(
                    {
                        "index": index,
                        "index_parameters": _index_parameters(index, mapping),
                        "fraction": fraction,
                        "repetition": repetition,
                        "impressions": size,
                        "population_impressions": len(impressions),
                        "workload_impression_sha256": _workload_hash(sample),
                        "ranking_sha256": ranking_hash,
                        "rows": rows,
                        "elapsed_seconds": elapsed,
                        "throughput_impressions_per_second": size / elapsed,
                        "peak_rss_bytes": rss.peak_bytes,
                        "rss_delta_bytes": max(0, rss.peak_bytes - rss.baseline_bytes),
                        "peak_cuda_bytes": _peak_cuda(),
                        "index_bytes": _index_bytes(retriever),
                    }
                )
    full = [record for record in records if record["fraction"] == 1.0]
    factor = int(benchmark.get("extrapolation_factor", 10))
    extrapolations = []
    for index in sorted({str(record["index"]) for record in full}):
        selected_records = [record for record in full if record["index"] == index]
        mean_elapsed = sum(float(record["elapsed_seconds"]) for record in selected_records) / len(
            selected_records
        )
        extrapolations.append(
            {
                "index": index,
                "factor": factor,
                "source_impressions": len(impressions),
                "projected_impressions": len(impressions) * factor,
                "mean_source_elapsed_seconds": mean_elapsed,
                "elapsed_seconds": factor * mean_elapsed,
                "assumption": "linear query-time scaling at fixed corpus and hardware",
            }
        )
    output = project_path(
        str(
            mapping.get(
                "benchmark_output",
                Path("output") / name / variant / system / "benchmark.json",
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = int(benchmark.get("seed", 146))
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": name,
                "variant": variant,
                "system": system,
                "config_hash": config_hash({**mapping, "benchmark": benchmark}),
                "seed": seed,
                "workload_order": "first_test_impressions_in_feature_store_order",
                "top_k": 200,
                "hardware": {
                    "platform": platform.platform(),
                    "cpu_count": psutil.cpu_count() or 1,
                    "memory_bytes": psutil.virtual_memory().total,
                    "device": device_identity(),
                },
                "records": records,
                "extrapolations": extrapolations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_path = output.parent / "sample_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(sample_rows), sample_path, compression="zstd")
    return output
