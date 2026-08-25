from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ire_assn1.data.download import download_ebnerd_submission_assets, download_from_config
from ire_assn1.data.pipeline import prepare_competition_from_config, prepare_from_config
from ire_assn1.evaluation.harness import evaluate_from_config
from ire_assn1.experiments.benchmark import benchmark_from_config
from ire_assn1.experiments.manifests import (
    create_manifest,
    load_manifest,
    write_manifest,
)
from ire_assn1.experiments.stages import run_stage
from ire_assn1.experiments.submission import submit_from_config
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
    retrieval_limits = {
        key: mapping[key]
        for key in ("offline_impressions_limit", "history_selection_limit")
        if key in mapping
    }
    return tuple(
        {
            **load_mapping(dataset_path),
            **load_mapping(retrieval_path),
            **retrieval_limits,
            "evaluation": evaluation,
            "benchmark": benchmark,
        }
        for dataset_path in _paths(mapping, "datasets")
        for retrieval_path in _paths(mapping, "retrieval")
    )


def ebnerd_submission_from_config(
    config: Path, system: str, history_length: int | None = None
) -> None:
    if system not in {"bm25", "bge"}:
        raise ValueError("EB-NeRD submission system must be bm25 or bge")
    mapping = load_mapping(config)
    matches = tuple(
        run_config
        for run_config in _run_configs(mapping)
        if run_config.get("name") == "ebnerd"
        and run_config.get("variant") == "large"
        and run_config.get("system") == system
    )
    if len(matches) != 1:
        raise ValueError(f"No unique EB-NeRD large configuration found for {system}")
    run_config = {
        **matches[0],
        "submission_history_length": (
            history_length
            if history_length is not None
            else mapping.get("submission_history_length", 20)
        ),
        "submission_batch_size": mapping.get("submission_batch_size", 2048),
        "submission_device": mapping.get("submission_device", "cuda"),
    }
    if system == "bge":
        from ire_assn1.experiments.ebnerd_submission import write_ebnerd_semantic_submission

        download_ebnerd_submission_assets(config)
        write_ebnerd_semantic_submission(run_config)
        return
    retrieve_competition_from_config(run_config, submission_only=True)
    submit_from_config(run_config)


def _manifest_for_run(
    config: Path,
    mapping: dict[str, Any],
    allow_dirty: bool,
    resume_id: str | None,
):
    current = create_manifest("assignment", "final", "all", mapping, allow_dirty=allow_dirty)
    if resume_id is None:
        return current
    previous_path = project_path("logs") / resume_id / "manifest.json"
    previous = load_manifest(previous_path)
    if previous.git_sha != current.git_sha:
        raise RuntimeError("Resume manifest source revision does not match the current source")
    if previous.source_tree_sha256 != current.source_tree_sha256:
        raise RuntimeError("Resume manifest source tree does not match the current source")
    if previous.config_sha256 != current.config_sha256:
        raise RuntimeError("Resume manifest configuration does not match the current config")
    if previous.lock_sha256 != current.lock_sha256:
        raise RuntimeError("Resume manifest Pixi lock does not match the current lock")
    return previous


def reproduce_from_config(
    config: Path,
    allow_dirty: bool = False,
    resume_id: str | None = None,
) -> None:
    mapping = load_mapping(config)
    manifest = _manifest_for_run(config, mapping, allow_dirty, resume_id)
    write_manifest(manifest)
    data_root = project_path("data")
    run_stage(
        manifest,
        "download",
        lambda: download_from_config(config),
        inputs=(config,),
        outputs=(data_root / "raw",),
        resume=resume_id is not None,
    )
    run_stage(
        manifest,
        "prepare",
        lambda: prepare_from_config(config),
        inputs=(data_root / "raw",),
        outputs=(data_root / "processed",),
        resume=resume_id is not None,
    )
    run_stage(
        manifest,
        "prepare-competition",
        lambda: prepare_competition_from_config(config),
        inputs=(data_root / "raw", data_root / "processed"),
        outputs=(data_root / "processed",),
        resume=resume_id is not None,
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
            resume=resume_id is not None,
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
                resume=resume_id is not None,
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
            resume=resume_id is not None,
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
            resume=resume_id is not None,
        )
        plot_output = project_path("plots") / name / variant / system
        run_stage(
            manifest,
            f"plot-{name}-{variant}-{system}",
            lambda resolved=run_config: plot_from_config(resolved),
            inputs=(output,),
            outputs=(plot_output,),
            resume=resume_id is not None,
        )
    write_manifest(manifest)
