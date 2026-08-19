#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing %s. Copy .env.example to .env and fill its TODO fields.\n' "$ENV_FILE" >&2
  exit 1
fi

chmod 600 "$ENV_FILE"
set -a
# TODO: Keep .env private; it contains the Hugging Face read token.
. "$ENV_FILE"
set +a

: "${SCRATCH_ROOT:?Set SCRATCH_ROOT in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"

if [[ "$SCRATCH_ROOT" == /path/to/* ]]; then
  printf 'Replace the SCRATCH_ROOT TODO in %s.\n' "$ENV_FILE" >&2
  exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
  printf 'pixi is not available in PATH. Install it in user space before setup.\n' >&2
  exit 1
fi

PIXI_ROOT="${PIXI_ROOT:-$SCRATCH_ROOT/ire-assn1-pixi}"
PIXI_ENV="$PIXI_ROOT/env"
PIXI_CACHE="$PIXI_ROOT/cache"
MODEL_ROOT="$PIXI_ROOT/models"

link_storage() {
  local link_path="$1"
  local target_path="$2"

  if [[ -L "$link_path" ]]; then
    local resolved_path
    resolved_path="$(readlink -f "$link_path")"
    if [[ -z "$resolved_path" || "$resolved_path" == "$REPO_ROOT"/* ]]; then
      printf 'Existing symlink resolves inside the repository: %s\n' "$link_path" >&2
      exit 1
    fi
    mkdir -p "$resolved_path"
    printf '%s\n' "$resolved_path"
    return
  fi

  if [[ -e "$link_path" ]]; then
    mv "$link_path" "${link_path}.incomplete-$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "$target_path"
  ln -s "$target_path" "$link_path"
  printf '%s\n' "$target_path"
}

PIXI_ENV="$(link_storage "$REPO_ROOT/.pixi" "$PIXI_ENV")"
MODEL_ROOT="$(link_storage "$REPO_ROOT/models" "$MODEL_ROOT")"

if [[ -L "$HOME/.cache" ]]; then
  CACHE_ROOT="$(readlink -f "$HOME/.cache")"
  if [[ -z "$CACHE_ROOT" || "$CACHE_ROOT" == "$REPO_ROOT"/* ]]; then
    printf 'Existing ~/.cache symlink resolves inside the repository.\n' >&2
    exit 1
  fi
  PIXI_CACHE="${PIXI_CACHE_DIR:-$CACHE_ROOT/pixi}"
else
  PIXI_CACHE="${PIXI_CACHE_DIR:-$PIXI_CACHE}"
fi

mkdir -p "$PIXI_CACHE"

export PIXI_CACHE_DIR="$PIXI_CACHE"
export HF_TOKEN

cd "$REPO_ROOT"
pixi install --locked -e gpu
pixi run -e gpu doctor
