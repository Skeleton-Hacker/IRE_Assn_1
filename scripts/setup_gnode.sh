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

mkdir -p "$PIXI_ENV" "$PIXI_CACHE" "$MODEL_ROOT"

link_storage() {
    local link_path="$1"
    local target_path="$2"

    if [[ -L "$link_path" ]]; then
        if [[ "$(readlink "$link_path")" != "$target_path" ]]; then
            printf 'Existing symlink has an unexpected target: %s\n' "$link_path" >&2
            exit 1
        fi
        return
    fi

    if [[ -e "$link_path" ]]; then
        mv "$link_path" "${link_path}.incomplete-$(date +%Y%m%d-%H%M%S)"
    fi
    ln -s "$target_path" "$link_path"
}

link_storage "$REPO_ROOT/.pixi" "$PIXI_ENV"
link_storage "$REPO_ROOT/models" "$MODEL_ROOT"

export PIXI_CACHE_DIR="$PIXI_CACHE"
export HF_TOKEN

cd "$REPO_ROOT"
pixi install --locked -e gpu
pixi run -e gpu doctor
