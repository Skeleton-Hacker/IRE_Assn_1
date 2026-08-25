# IRE Assignment 1

This repository implements lexical and semantic news retrieval for MIND and EB-NeRD.

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

On the HPC, place the Pixi environment and package cache on persistent scratch. Set `SCRATCH_ROOT`
to the cluster's persistent scratch path. Create `.env` from
`.env.example` on the HPC and fill its TODO fields. `.env` is ignored, permission-tightened by the
setup script, and never included in source manifests.

```bash
cd ~/IRE/Assn_1
cp .env.example .env
${EDITOR:-vi} .env
# Run your per-node symlink.sh first if ~/.cache is not already on scratch.
bash scripts/setup_gnode.sh
set -a
. .env
set +a
pixi run -e gpu reproduce
```

The default `reproduce` command uses the small datasets for smoke testing. The Codabench workflow
must use the large datasets:

```bash
pixi run -e gpu reproduce-codabench
```

For a non-interactive four-day Slurm allocation, set the `#SBATCH -w` value in `sbatch.sh` to an
available node, then submit it from the repository root:

```bash
sbatch sbatch.sh
```

The batch script runs the per-node cache symlink setup, initializes the GPU environment, sources
the private `.env`, and runs the large Codabench configuration. Setup moves `data` to
`$DATA_ROOT` and links it back into the repository; `DATA_ROOT` defaults to
`$SCRATCH_ROOT/ire-assn1-data`. Set `IRE_CONFIG=config/base.yaml` before submission only when a
small smoke run is intended.

Downloads, archive extraction, embedding, indexing, validation, and retrieval display `tqdm`
progress where work is measurable. Cached stages can complete without showing work for that stage.
The large Codabench configuration limits offline diagnostics and history selection to 10,000
impressions to bound runtime and output size; competition candidate scoring remains complete.
The benchmark uses the first 1,000 test impressions in feature-store order for its 25%, 50%, and
100% workloads before the configured 10x extrapolation.

For the two EB-NeRD leaderboard submissions, run `sbatch sbatch_ebnerd_bert_h10.sh` and
`sbatch sbatch_ebnerd_bert_h20.sh`. Both use the supplied multilingual BERT system, download the
test and BERT archives when absent, and work directly from their raw Parquet files. Each history
variant has an isolated cache and output directory below `data` and `output`, so the jobs can run
concurrently and resume at source row-group boundaries. They do not rebuild the large offline
feature store. For MIND, use the existing BGE submission and package the already-scored BM25
competition candidates with `sbatch sbatch_mind_bm25_submission.sh`.

Individual stages are available through `pixi run ire-assn1 -- --help`. The doctor command reports
whether Git or `.ire-source.json` provided provenance. Official runs require a verified clean
source manifest and produce versioned manifests. Use `--allow-dirty` only for non-reportable
debugging runs. An interrupted run can resume after transferring the same source tree with:
`pixi run -e gpu ire-assn1 -- reproduce --config config/codabench.yaml --resume RUN_ID`.

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
