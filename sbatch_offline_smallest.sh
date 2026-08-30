#!/bin/bash
#SBATCH -J "IRE_OFFLINE_SMALLEST"
#SBATCH -c 9
#SBATCH -G 1
#SBATCH -w gnode092
#SBATCH -o ./logs/offline_smallest_%j.log
#SBATCH -e ./logs/offline_smallest_error_%j.log
#SBATCH --time="4-00:00:00"
#SBATCH --mail-user=yajat.rangnekar@research.iiit.ac.in
#SBATCH --mail-type=ALL

set -Eeuo pipefail
umask 077

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
ENV_FILE="$REPO_ROOT/.env"

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

exec pixi run -e gpu ire-assn1 -- reproduce --config config/offline-smallest.yaml
