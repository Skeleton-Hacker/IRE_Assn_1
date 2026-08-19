# Assignment 1 Technical Specification

## Objective

Build reproducible lexical and semantic retrieval pipelines for MIND-small and EB-NeRD. Both
systems use article title and abstract. Article bodies are optional dataset fields and are excluded
from the primary comparison. The pipeline must rebuild raw data, feature stores, retrieval outputs,
evaluation summaries, benchmark evidence, visualizations, and Codabench submissions.

## Dataset Variants And Splits

Supported variants are `mind-small`, `ebnerd-demo`, and `ebnerd-small`. MIND-small and
EB-NeRD-small are final experiment variants. EB-NeRD-demo is the real-data smoke variant.

The official labeled validation package is the offline test period. The latest complete day in the
official training package is validation. Earlier official training impressions are training data.
Hidden competition test data is leaderboard-only. Supplied pre-impression histories are
authoritative and are not updated within a split.

EB-NeRD article availability is its publication timestamp. MIND availability is the earliest
observed timestamp across histories, candidate exposures, and clicks because publication time is
absent. Retrieval must exclude articles unavailable at an impression timestamp.

## Normalized Schemas

All identifiers are UTF-8 strings prefixed by dataset and variant. Timestamps are UTC-normalized
microsecond datetimes where source timezone information exists and consistently naive otherwise.

Articles contain `article_id`, `title`, `abstract`, nullable `body`, nullable `category`, nullable
`subcategory`, `entities`, nullable `published_at`, `available_at`, and `source_split`.

Histories contain `user_id`, ordered `article_ids`, ordered nullable `timestamps`, and
`source_split`.

Impressions contain `impression_id`, `user_id`, nullable `session_id`, `timestamp`, ordered
`candidate_ids`, ordered `clicked_ids`, ordered labels, and `source_split`.

Flat candidates contain `impression_id`, `user_id`, `timestamp`, `article_id`, `label`, `position`,
and `source_split`. Retrieval output extends this schema with `system`, `score`, and `rank`.

Schemas are written as Parquet and validated before use. Conflicting duplicate article records use
the earliest source record and emit conflict counts in the stage manifest.

## Retrieval

BM25 indexes normalized title and abstract with dataset-language lowercase tokenization and
stopword removal. Danish diacritics are preserved. Stemming is disabled. Primary parameters are
`k1=1.2` and `b=0.75`. Queries concatenate title and abstract from the most recent eligible clicked
articles.

BGE uses `BAAI/bge-m3` dense embeddings of title and abstract. Article vectors are L2-normalized.
User vectors are the normalized mean of eligible recent article vectors. Exact FAISS inner-product
search is primary. HNSW is benchmark-only.

History length is selected independently for each dataset and system from `5`, `10`, `20`, and
all using validation Recall@100. Primary systems use history only. Empty histories and missing
representations use training-only popularity followed by article ID as a deterministic tie-break.

Full-corpus candidate generation reports Recall@50, Recall@100, and Recall@200. Impression
ranking scores only supplied candidates and preserves no future information.

## Evaluation

Accuracy metrics are impression-level AUC, MRR, nDCG@5, and nDCG@10. Undefined impressions
are excluded per metric and exclusion counts are reported.

Diversity at 10 is mean pairwise cosine distance between BGE article vectors. Novelty at 10 is
mean self-information from training click popularity with Laplace smoothing alpha one. Coverage
at 10 divides unique recommended articles by the union of exposed candidate articles.

Cold and warm users are separated at the training median history length. Head articles are the
smallest popularity-ranked set accounting for 80 percent of training clicks; remaining articles are
tail. Metrics are reported for both slice families.

Confidence intervals use 1,000 percentile bootstrap samples clustered by user with a fixed seed.
A future-popularity RRF oracle with `k=60` is diagnostic, labeled unavailable at serving time, and
excluded from submissions.

## Benchmarking And Visualizations

Benchmarks use deterministic 25, 50, and 100 percent impression workloads. Each records wall
time, throughput, peak RSS, peak CUDA memory, and index size. Exact and HNSW semantic search
are compared. Ten-times-scale analysis extrapolates measured trends and states assumptions.

Generated visualizations include Recall@K, sliced metrics with confidence intervals, runtime and
memory scaling, PCA, and sampled t-SNE.

## Commands And Artifacts

The `ire-assn1` Typer CLI exposes `doctor`, `source-manifest`, `download`, `prepare`, `retrieve`,
`evaluate`, `benchmark`, `plot`, `submit`, `bundle-run`, `validate-run`, and `reproduce`. Pixi
exposes local verification and full CPU/GPU environments. `pixi run -e gpu reproduce` executes the
complete pipeline, while every stage remains independently runnable.

All dataset-derived artifacts reside below `data`. Model caches reside below `models`. Run records
reside below `logs`. Compact result summaries and submissions reside below `output`.
Visualizations reside below `plots`.

Official runs require a clean Git tree or a verified source manifest generated from one. Run
manifests include Git revision and status, provenance source kind, source tree hash, resolved
configuration and hash, Pixi lock hash, input and output checksums, dataset variant, model revision,
seeds, device information, timings, memory, and stage states. Cached stages are reusable only when
their code, configuration, and input identities match.

The `source-manifest` command writes the ignored `.ire-source.json` transfer artifact from a clean
Git checkout. It records the source revision and hashes of the code, configuration, specification,
Pixi manifest, and lockfile. A transferred HPC checkout may omit `.git` and use this manifest;
the pipeline verifies every recorded file before creating an official run manifest. `doctor`
reports Git or source-manifest provenance without requiring GitLab credentials. Missing or modified
provenance remains non-reportable and is never treated as a clean checkout.

MIND archives are downloaded from the gated Hugging Face repository using the read token named by
the dataset `auth_env` field, defaulting to `HF_TOKEN`. The token is read only from the process
environment and is never written to configuration, logs, manifests, or error messages.

The tracked `scripts/setup_gnode.sh` command is the HPC environment setup entry point. It reads the
ignored `.env`, requires `SCRATCH_ROOT` and `HF_TOKEN`, keeps datasets under `data`, links `.pixi`
and `models` to persistent scratch, installs the locked GPU environment, and runs `doctor`.

## Verification And Completion

Synthetic fixtures assert schemas, chronological splitting, article availability, no future-click
leakage, exact BM25 and dense ranks, cold-start fallback, deterministic ties, metric values, slices,
and bootstrap reproducibility. GitLab CI runs Ruff, formatting checks, Pyright, Pytest, and synthetic
end-to-end evaluation without real datasets, model downloads, or CUDA.

Each retrieval merge request requires CI plus an HPC smoke bundle. Final completion requires both
systems on both final datasets, all offline metrics and slices, benchmark evidence, generated
visualizations, two submissions per leaderboard, compact manifests, and reproducible commands.

Codabench prediction serialization remains deferred until current competition templates are
inspected. Full predictions remain ignored unless an explicit requirement demands otherwise.
