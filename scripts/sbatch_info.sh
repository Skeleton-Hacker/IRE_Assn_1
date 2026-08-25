#!/usr/bin/env bash
#SBATCH -J IRE_info
#SBATCH -c 1
#SBATCH --time=00:15:00
#SBATCH -w gnode092
#SBATCH -o logs/sbatch_info_%j.log
#SBATCH -e logs/sbatch_info_error_%j.log

set -Eeuo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Submit this script with: sbatch scripts/sbatch_info.sh\n' >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

printf 'hostname: %s\n' "$(hostname)"
printf 'repository: %s\n' "$REPO_ROOT"
printf 'data link: %s\n' "$(readlink -f data 2>/dev/null || printf 'unresolved')"

pixi run -e gpu python -c '
from pathlib import Path

import pyarrow.parquet as pq

root = Path("data/raw/ebnerd")
paths = sorted(root.rglob("*.parquet")) if root.exists() else []
if not paths:
    raise SystemExit(f"No parquet files found below {root}")

for path in paths:
    parquet = pq.ParquetFile(path)
    print(f"\n=== {path} ===")
    print(f"rows: {parquet.metadata.num_rows}")
    print(f"columns: {parquet.schema_arrow.names}")
    print(parquet.schema_arrow)
    batch = next(parquet.iter_batches(batch_size=1), None)
    if batch is None:
        continue
    row = batch.to_pylist()[0]
    for name, value in row.items():
        if isinstance(value, list):
            element_type = type(value[0]).__name__ if value else "empty"
            print(f"sample {name}: list length={len(value)} element_type={element_type}")
        else:
            print(f"sample {name}: type={type(value).__name__} value={str(value)[:160]}")
'
