from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ConfigOption = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


@app.command()
def doctor(config: ConfigOption = Path("config/base.yaml")) -> None:
    from ire_assn1.experiments.doctor import run_doctor

    run_doctor(config)


@app.command("source-manifest")
def source_manifest(
    output: Annotated[Path, typer.Option("--output", dir_okay=False)] = Path(".ire-source.json"),
    allow_dirty: bool = typer.Option(False),
) -> None:
    from ire_assn1.experiments.provenance import write_source_manifest

    print(write_source_manifest(output, allow_dirty=allow_dirty))


@app.command()
def download(config: ConfigOption = Path("config/base.yaml")) -> None:
    from ire_assn1.data.download import download_from_config

    download_from_config(config)


@app.command()
def prepare(config: ConfigOption = Path("config/base.yaml")) -> None:
    from ire_assn1.data.pipeline import prepare_from_config

    prepare_from_config(config)


@app.command()
def retrieve(config: ConfigOption) -> None:
    from ire_assn1.retrieval.runner import retrieve_from_config

    retrieve_from_config(config)


@app.command()
def evaluate(config: ConfigOption) -> None:
    from ire_assn1.evaluation.harness import evaluate_from_config

    evaluate_from_config(config)


@app.command()
def benchmark(config: ConfigOption) -> None:
    from ire_assn1.experiments.benchmark import benchmark_from_config

    benchmark_from_config(config)


@app.command()
def plot(config: ConfigOption) -> None:
    from ire_assn1.experiments.visualization import plot_from_config

    plot_from_config(config)


@app.command()
def submit(config: ConfigOption) -> None:
    from ire_assn1.experiments.submission import submit_from_config

    submit_from_config(config)


@app.command("submit-codabench")
def submit_codabench(config: ConfigOption = Path("config/codabench.yaml")) -> None:
    from ire_assn1.experiments.runner import _run_configs
    from ire_assn1.experiments.submission import submit_from_config
    from ire_assn1.settings import load_mapping

    for run_config in _run_configs(load_mapping(config)):
        submit_from_config(run_config)


@app.command("bundle-run")
def bundle_run(run_id: str) -> None:
    from ire_assn1.experiments.bundles import create_bundle

    create_bundle(run_id)


@app.command("validate-run")
def validate_run(path: Path) -> None:
    from ire_assn1.experiments.bundles import validate_bundle

    validate_bundle(path)


@app.command()
def reproduce(
    config: ConfigOption = Path("config/base.yaml"),
    allow_dirty: bool = typer.Option(False),
) -> None:
    from ire_assn1.experiments.runner import reproduce_from_config

    reproduce_from_config(config, allow_dirty=allow_dirty)
