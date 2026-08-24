from pathlib import Path

from ire_assn1.experiments.runner import _run_configs
from ire_assn1.settings import load_mapping


def test_codabench_limits_reach_each_run_configuration() -> None:
    mapping = load_mapping(Path("config/codabench.yaml"))

    run_configs = _run_configs(mapping)

    assert len(run_configs) == 4
    assert all(run["offline_impressions_limit"] == 10000 for run in run_configs)
    assert all(run["history_selection_limit"] == 10000 for run in run_configs)
