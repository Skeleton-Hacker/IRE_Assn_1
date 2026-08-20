from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ire_assn1.data.download import download_from_config
from ire_assn1.data.pipeline import prepare_competition_from_config, prepare_from_config
from ire_assn1.evaluation.harness import evaluate_from_config
from ire_assn1.experiments.benchmark import benchmark_from_config
from ire_assn1.experiments.manifests import create_manifest, write_manifest
from ire_assn1.experiments.stages import run_stage
from ire_assn1.experiments.visualization import plot_from_config
from ire_assn1.paths import project_path
from ire_assn1.retrieval.runner import retrieve_competition_from_config, retrieve_from_config
from ire_assn1.settings import load_mapping


def _paths(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Configuration field {key} must be a list of paths")
    return tuple(cast(list[str], value))


def _run_configs(mapping: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    evaluation = mapping.get("evaluation", "config/evaluation.yaml")
    benchmark = mapping.get("benchmark", "config/benchmark.yaml")
    return tuple(
        {
            **load_mapping(dataset_path),
            **load_mapping(retrieval_path),
            "evaluation": evaluation,
            "benchmark": benchmark,
        }
        for dataset_path in _paths(mapping, "datasets")
        for retrieval_path in _paths(mapping, "retrieval")
    )


def reproduce_from_config(config: Path, allow_dirty: bool = False) -> None:
    mapping = load_mapping(config)
    manifest = create_manifest("assignment", "final", "all", mapping, allow_dirty=allow_dirty)
    write_manifest(manifest)
    data_root = project_path("data")
    run_stage(
        manifest,
        "download",
        lambda: download_from_config(config),
        inputs=(config,),
        outputs=(data_root / "raw",),
    )
    run_stage(
        manifest,
        "prepare",
        lambda: prepare_from_config(config),
        inputs=(data_root / "raw",),
        outputs=(data_root / "processed",),
    )
    run_stage(
        manifest,
        "prepare-competition",
        lambda: prepare_competition_from_config(config),
        inputs=(data_root / "raw", data_root / "processed"),
        outputs=(data_root / "processed",),
    )
    run_configs = _run_configs(mapping)
    for run_config in run_configs:
        name = str(run_config["name"])
        variant = str(run_config["variant"])
        system = str(run_config["system"])
        retrieval_output = data_root / "retrieval" / name / variant / system
        run_stage(
            manifest,
            f"retrieve-{name}-{variant}-{system}",
            lambda resolved=run_config: retrieve_from_config(resolved),
            inputs=(data_root / "processed" / name / variant,),
            outputs=(retrieval_output,),
        )
        competition_input = data_root / "processed" / name / variant / "competition_test"
        if competition_input.is_dir():
            run_stage(
                manifest,
                f"retrieve-competition-{name}-{variant}-{system}",
                lambda resolved=run_config: retrieve_competition_from_config(resolved),
                inputs=(
                    data_root / "processed" / name / variant,
                    competition_input,
                    retrieval_output / "selection.json",
                ),
                outputs=(retrieval_output / "competition_test",),
            )
    for run_config in run_configs:
        name = str(run_config["name"])
        variant = str(run_config["variant"])
        system = str(run_config["system"])
        output = project_path("output") / name / variant / system
        run_stage(
            manifest,
            f"evaluate-{name}-{variant}-{system}",
            lambda resolved=run_config: evaluate_from_config(resolved),
            inputs=(
                data_root / "processed" / name / variant,
                data_root / "retrieval" / name / variant / system,
            ),
            outputs=(output,),
        )
        run_stage(
            manifest,
            f"benchmark-{name}-{variant}-{system}",
            lambda resolved=run_config: benchmark_from_config(resolved),
            inputs=(
                data_root / "processed" / name / variant,
                data_root / "retrieval" / name / variant / system,
            ),
            outputs=(output / "benchmark.json",),
        )
        plot_output = project_path("plots") / name / variant / system
        run_stage(
            manifest,
            f"plot-{name}-{variant}-{system}",
            lambda resolved=run_config: plot_from_config(resolved),
            inputs=(output,),
            outputs=(plot_output,),
        )
    write_manifest(manifest)
