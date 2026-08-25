#!/bin/bash
#SBATCH -J "IRE_EBNeRD_Submission"
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -w gnode092
#SBATCH -o ./logs/ebnerd_submission_%j.log
#SBATCH -e ./logs/ebnerd_submission_error_%j.log
#SBATCH --time="2-00:00:00"
#SBATCH --mail-user=yajat.rangnekar@research.iiit.ac.in
#SBATCH --mail-type=ALL

set -Eeuo pipefail
umask 077

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
ENV_FILE="$REPO_ROOT/.env"
SYSTEM="${IRE_SYSTEM:?Set IRE_SYSTEM to bm25 or bge}"

case "$SYSTEM" in
  bm25|bge) ;;
  *) printf 'IRE_SYSTEM must be bm25 or bge\n' >&2; exit 1 ;;
esac

cd "$REPO_ROOT"

if [[ ! -f "$HOME/symlink.sh" ]]; then
  printf 'Missing per-node cache setup script: %s\n' "$HOME/symlink.sh" >&2
  exit 1
fi

bash "$HOME/symlink.sh"
bash "$REPO_ROOT/scripts/setup_gnode.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing %s\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

: "${HF_TOKEN:?HF_TOKEN must be set in .env}"

export PYTHONUNBUFFERED=1
pixi run -e gpu doctor
CONFIG="${IRE_CONFIG:-config/ebnerd-submission.yaml}"

RAW_ROOT="$REPO_ROOT/data/raw/ebnerd/large"
PREDICTIONS="$REPO_ROOT/data/processed/ebnerd/large/competition_test/impressions.parquet"
EMBEDDING_FILE="$(find -L "$RAW_ROOT" -type f \( -name 'artifacts.parquet' -o -iname '*embedding*.parquet' -o -iname '*vector*.parquet' \) -print -quit 2>/dev/null || true)"
if [[ ! -s "$PREDICTIONS" || -z "$EMBEDDING_FILE" ]]; then
  pixi run -e gpu python -m ire_assn1 download --config "$CONFIG"
  pixi run -e gpu python -m ire_assn1 prepare --config "$CONFIG"
  pixi run -e gpu python -m ire_assn1 prepare-competition --config "$CONFIG"
fi

EMBEDDING_FILE="$(find -L "$RAW_ROOT" -type f \( -name 'artifacts.parquet' -o -iname '*embedding*.parquet' -o -iname '*vector*.parquet' \) -print -quit 2>/dev/null || true)"

if [[ ! -s "$PREDICTIONS" ]]; then
  printf 'Missing EB-NeRD competition feature store: %s\n' "$PREDICTIONS" >&2
  exit 1
fi

if [[ "$SYSTEM" == bge && -z "${EMBEDDING_FILE:-}" ]]; then
  printf 'Missing supplied EB-NeRD embedding artifacts below %s\n' "$RAW_ROOT" >&2
  exit 1
fi

exec pixi run -e gpu ire-assn1 -- ebnerd-submission --config "$CONFIG" --system "$SYSTEM"
