# IRE Assignment 1

This repository implements lexical and semantic news retrieval for MIND-small and EB-NeRD.

## Local Verification

```bash
pixi install
pixi run check
pixi run eval-synthetic
```

Local verification uses committed synthetic fixtures and does not download datasets, language
models, or CUDA packages.

## HPC Setup

Create or enter the persistent scratch directory, clone the repository there, and enter an
interactive or batch GPU allocation before running GPU tasks.

```bash
pixi install -e gpu
pixi run -e gpu ire-assn1 doctor --config config/base.yaml
pixi run -e gpu download
pixi run -e gpu prepare
pixi run -e gpu reproduce
```

Individual stages are available through `pixi run ire-assn1 -- --help`. Official runs require a
clean Git tree and produce versioned manifests. Use `--allow-dirty` only for non-reportable
debugging runs.

## Artifact Layout

- `data/`: raw datasets, normalized stores, embeddings, indexes, and full predictions
- `models/`: model cache
- `logs/`: run manifests and compact logs
- `output/`: metric summaries, benchmark summaries, and submission files
- `plots/`: generated visualizations

Large and reproducible artifacts are ignored by Git. Compact final summaries, plots, manifests,
and leaderboard screenshots may be force-added after review.

## Returned HPC Runs

```bash
pixi run ire-assn1 -- bundle-run --run-id RUN_ID
pixi run ire-assn1 -- validate-run --path PATH_TO_BUNDLE
```

The `$eval` and `$bench` repository skills validate returned HPC evidence using the same Python
implementation used by CI.
