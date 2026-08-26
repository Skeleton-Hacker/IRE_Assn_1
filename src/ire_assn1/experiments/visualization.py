from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ire_assn1.paths import project_path
from ire_assn1.settings import load_mapping


def _overall(metrics: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [
        metric
        for metric in metrics
        if metric["slice_name"] == "overall" and str(metric["name"]).startswith(prefix)
    ]


def _seed(mapping: Mapping[str, object]) -> int:
    value = mapping.get("seed", 146)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer")
    return value


def recall_plot(metrics: list[dict[str, Any]], output: Path) -> None:
    selected = sorted(_overall(metrics, "recall_at_"), key=lambda item: int(item["name"][10:]))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [int(item["name"][10:]) for item in selected],
        [float(item["value"]) for item in selected],
        marker="o",
    )
    axis.set_xlabel("K")
    axis.set_ylabel("Recall")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def slice_plot(metrics: list[dict[str, Any]], output: Path) -> None:
    names = [
        "recall_at_100",
        "ndcg_at_10",
        "diversity_at_10",
        "novelty_at_10",
        "coverage_at_10",
    ]
    slices = ["overall", "cold", "warm", "head", "tail"]
    lookup = {(item["name"], item["slice_name"]): item for item in metrics}
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, name in zip(axes.flat, names, strict=False):
        selected = [lookup.get((name, slice_name)) for slice_name in slices]
        labels = [slice_name for slice_name, item in zip(slices, selected, strict=True) if item]
        items = [item for item in selected if item]
        values = [float(item["value"]) if item["value"] is not None else np.nan for item in items]
        lower = []
        upper = []
        for value, item in zip(values, items, strict=True):
            interval = item.get("interval")
            if interval is None or np.isnan(value):
                lower.append(0.0)
                upper.append(0.0)
            else:
                lower.append(max(0.0, value - float(interval["lower"])))
                upper.append(max(0.0, float(interval["upper"]) - value))
        axis.bar(labels, values, yerr=np.asarray([lower, upper]), capsize=3)
        axis.set_title(name)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylabel("Value")
    axes.flat[-1].remove()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def scaling_plot(records: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    indexes = sorted({str(record["index"]) for record in records})
    for index in indexes:
        selected = [record for record in records if record["index"] == index]
        fractions = sorted({float(record["fraction"]) for record in selected})
        elapsed = [
            float(
                np.mean(
                    [
                        record["elapsed_seconds"]
                        for record in selected
                        if record["fraction"] == fraction
                    ]
                )
            )
            for fraction in fractions
        ]
        axis.plot(fractions, elapsed, marker="o", label=index)
    axis.set_xlabel("Impression workload fraction")
    axis.set_ylabel("Elapsed seconds")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def memory_plot(records: list[dict[str, Any]], output: Path) -> None:
    labels = sorted({str(record["index"]) for record in records})
    rss = [
        max(int(record["peak_rss_bytes"]) for record in records if record["index"] == label) / 2**20
        for label in labels
    ]
    cuda = [
        max(int(record["peak_cuda_bytes"]) for record in records if record["index"] == label)
        / 2**20
        for label in labels
    ]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(positions - 0.2, rss, 0.4, label="Peak RSS")
    axis.bar(positions + 0.2, cuda, 0.4, label="Peak CUDA")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Memory MiB")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def embedding_plots(embeddings: Path, pca_output: Path, tsne_output: Path, seed: int) -> None:
    vectors = np.load(embeddings, mmap_mode="r")
    if len(vectors) < 3:
        return
    generator = np.random.default_rng(seed)
    indexes = generator.choice(len(vectors), size=min(5000, len(vectors)), replace=False)
    sample = np.asarray(vectors[indexes], dtype=np.float32)
    pca_values = PCA(n_components=2, random_state=seed).fit_transform(sample)
    tsne_values = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, len(sample) - 1),
    ).fit_transform(sample)
    for values, output in ((pca_values, pca_output), (tsne_values, tsne_output)):
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(values[:, 0], values[:, 1], s=3, alpha=0.5)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(output, dpi=180)
        plt.close(figure)


def plot_from_config(config: Path | Mapping[str, object]) -> Path:
    mapping = dict(config) if isinstance(config, Mapping) else load_mapping(config)
    name = str(mapping["name"])
    variant = str(mapping["variant"])
    system = str(mapping["system"])
    destination = project_path(str(mapping.get("plots", Path("plots") / name / variant / system)))
    destination.mkdir(parents=True, exist_ok=True)
    output = project_path(str(mapping.get("output", Path("output") / name / variant / system)))
    metrics_path = project_path(str(mapping.get("metrics", output / "evaluation.json")))
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        recall_plot(metrics, destination / "recall_at_k.png")
        slice_plot(metrics, destination / "sliced_metrics.png")
    benchmark_path = project_path(str(mapping.get("benchmark_output", output / "benchmark.json")))
    if benchmark_path.is_file():
        records = json.loads(benchmark_path.read_text(encoding="utf-8"))["records"]
        scaling_plot(records, destination / "scaling.png")
        memory_plot(records, destination / "memory.png")
    embeddings_path = mapping.get("embeddings")
    default_embeddings = (
        project_path("data") / "retrieval" / name / variant / "bge" / "article_embeddings.npy"
    )
    embeddings = project_path(str(embeddings_path)) if embeddings_path else default_embeddings
    if embeddings.is_file():
        embedding_plots(
            embeddings,
            destination / "embeddings_pca.png",
            destination / "embeddings_tsne.png",
            _seed(mapping),
        )
    return destination
