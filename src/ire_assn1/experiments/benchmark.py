from __future__ import annotations

import importlib
import json
import time
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Any, Protocol, cast

import psutil

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
from ire_assn1.retrieval.types import Retriever
from ire_assn1.settings import load_mapping


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
    for index in indexes:
        resolved = {**mapping, "index": index} if system == "bge" else mapping
        retriever = _create_retriever(resolved, articles, retrieval)
        for fraction_value in benchmark.get("fractions", [0.25, 0.5, 1.0]):
            fraction = float(fraction_value)
            size = max(1, round(len(impressions) * fraction))
            sample = impressions[:size]
            for repetition in range(int(benchmark.get("repetitions", 3))):
                process = psutil.Process()
                rss_before = process.memory_info().rss
                _peak_cuda(reset=True)
                started = time.perf_counter()
                rows = sum(
                    len(
                        full_corpus_retrieval(
                            impression,
                            history_for_impression(histories, impression),
                            articles,
                            retriever,
                            popularity,
                            selected,
                            200,
                        ).articles
                    )
                    for impression in sample
                )
                elapsed = time.perf_counter() - started
                records.append(
                    {
                        "index": str(index),
                        "fraction": fraction,
                        "repetition": repetition,
                        "impressions": size,
                        "population_impressions": len(impressions),
                        "rows": rows,
                        "elapsed_seconds": elapsed,
                        "throughput_impressions_per_second": size / elapsed,
                        "rss_delta_bytes": max(0, process.memory_info().rss - rss_before),
                        "peak_cuda_bytes": _peak_cuda(),
                        "index_bytes": _index_bytes(retriever),
                    }
                )
    full = [record for record in records if record["fraction"] == 1.0]
    factor = int(benchmark.get("extrapolation_factor", 10))
    extrapolations = [
        {
            "index": index,
            "factor": factor,
            "assumption": "linear query-time scaling at fixed corpus and hardware",
            "elapsed_seconds": factor
            * sum(float(record["elapsed_seconds"]) for record in full if record["index"] == index)
            / len([record for record in full if record["index"] == index]),
        }
        for index in sorted({str(record["index"]) for record in full})
    ]
    output = project_path(
        str(
            mapping.get(
                "benchmark_output",
                Path("output") / name / variant / system / "benchmark.json",
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"records": records, "extrapolations": extrapolations},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
