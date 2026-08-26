import json
from pathlib import Path

import pytest

from ire_assn1.experiments import runner
from ire_assn1.experiments.benchmark import _index_parameters
from ire_assn1.experiments.runner import _run_configs
from ire_assn1.experiments.visualization import plot_from_config
from ire_assn1.settings import load_mapping


def test_codabench_limits_reach_each_run_configuration() -> None:
    mapping = load_mapping(Path("config/codabench.yaml"))

    run_configs = _run_configs(mapping)

    assert len(run_configs) == 4
    assert all(run["offline_impressions_limit"] == 10000 for run in run_configs)
    assert all(run["history_selection_limit"] == 10000 for run in run_configs)


def test_benchmark_index_parameters_are_explicit() -> None:
    mapping = {"hnsw_m": 24, "hnsw_ef_search": 96, "k1": 1.2, "b": 0.75}
    assert _index_parameters("exact", mapping) == {
        "metric": "inner_product",
        "implementation": "faiss_index_flat_ip",
    }
    assert _index_parameters("hnsw", mapping) == {
        "metric": "inner_product",
        "m": 24,
        "ef_search": 96,
    }
    assert _index_parameters("bm25", mapping) == {"k1": 1.2, "b": 0.75}


def test_postprocess_configuration_skips_expensive_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzed = False

    def fail(*args: object) -> None:
        raise AssertionError("preparation and retrieval must be skipped")

    def analyze(*args: object) -> None:
        nonlocal analyzed
        analyzed = True

    monkeypatch.setattr(runner, "_manifest_for_run", lambda *args: object())
    monkeypatch.setattr(runner, "write_manifest", lambda *args: None)
    monkeypatch.setattr(runner, "_run_configs", lambda mapping: ())
    monkeypatch.setattr(runner, "_prepare_and_retrieve", fail)
    monkeypatch.setattr(runner, "_analyze", analyze)

    runner.reproduce_from_config(Path("config/offline-postprocess.yaml"))
    assert analyzed


def test_plot_uses_configured_output(tmp_path: Path) -> None:
    output = tmp_path / "custom-output"
    plots = tmp_path / "plots"
    output.mkdir()
    (output / "evaluation.json").write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "name": "recall_at_100",
                        "slice_name": "overall",
                        "value": 0.5,
                        "interval": {"lower": 0.4, "upper": 0.6},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plot_from_config(
        {
            "name": "synthetic",
            "variant": "fixture",
            "system": "fixture",
            "output": str(output),
            "plots": str(plots),
        }
    )
    assert (plots / "recall_at_k.png").is_file()
    assert (plots / "sliced_metrics.png").is_file()
