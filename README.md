# IRE Assignment 1

This repository implements lexical and semantic news retrieval for MIND-small and EB-NeRD.

## Local Verification

Ensure the `pixi` version installed is `0.76.1`

```bash
git config core.hooksPath .githooks
pixi install
pixi run check
pixi run eval-synthetic
pixi run source-manifest
```

Local verification uses committed synthetic fixtures and does not download datasets, language
models, or CUDA packages.

## HPC Setup

Generate `.ire-source.json` locally from the clean commit that will be evaluated, then transfer
the source tree and that file to the HPC. Do not clone, pull, or configure Git credentials on the
shared cluster. The source manifest is verified before an official run.

On the HPC, place the Pixi environment and package cache on persistent scratch. Set `SCRATCH` to
the cluster's persistent scratch path if it is not already defined. Create `.env` from
`.env.example` on the HPC and fill its TODO fields. `.env` is ignored, permission-tightened by the
setup script, and never included in source manifests.

```bash
cd ~/IRE/Assn_1
cp .env.example .env
${EDITOR:-vi} .env
bash scripts/setup_gnode.sh
pixi run -e gpu reproduce
```

Individual stages are available through `pixi run ire-assn1 -- --help`. The doctor command reports
whether Git or `.ire-source.json` provided provenance. Official runs require a verified clean
source manifest and produce versioned manifests. Use `--allow-dirty` only for non-reportable
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
pixi run ire-assn1 -- bundle-run RUN_ID
pixi run ire-assn1 -- validate-run PATH_TO_BUNDLE
```

The `$eval` and `$bench` repository skills validate returned HPC evidence using the same Python
implementation used by CI.
