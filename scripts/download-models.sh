#!/usr/bin/env bash
# Author: Perijn
# Summary: Downloads the Halogen weights, the Ciru release and the retrieval models required by the isolated runtime.
# Usage: Copy .env.example to .env, configure paths, install `hf`, then run this script on the Strix Halo Linux host.
# The runtime inference network is intentionally internal: this provisioning step must complete before Compose starts.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/load-env.sh
source "$root/scripts/lib/load-env.sh"
load_deployment_env "$root/.env"

: "${MODELS_DIR:?Set MODELS_DIR in .env.}"
: "${ORNITH_MODEL_DIR:?Set ORNITH_MODEL_DIR in .env.}"
: "${HALOGEN_MODELS_DIR:?Set HALOGEN_MODELS_DIR in .env.}"
command -v hf >/dev/null || {
  printf '%s\n' 'Install huggingface_hub first: python3 -m pip install --user huggingface_hub'
  exit 1
}
mkdir -p "$MODELS_DIR" "$ORNITH_MODEL_DIR" "$HALOGEN_MODELS_DIR"

# The complete release is load-bearing: target/draft weights, custom kernels,
# runtime lockfiles, and supplied launchers must remain together.
hf download jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --local-dir "$ORNITH_MODEL_DIR"

runtime_root="$ORNITH_MODEL_DIR/installed-runtime"
runtime_env="sources/vllm-glm53-strix/runtime-env.sh"
if [[ ! -e "$runtime_root" ]]; then
  bash "$ORNITH_MODEL_DIR/runtime/INSTALL-ORNITH-RUNTIME.sh" "$runtime_root"
fi

# The supplied installer creates these links only on its first successful run.
# Repair an interrupted final-link step without overwriting its expensive runtime.
if ! test -f "$runtime_root/$runtime_env"; then
  printf 'Incomplete Ciru runtime: missing %s\n' "$runtime_root/$runtime_env" >&2
  exit 1
fi
ln -sfn sources/vllm-glm53-strix "$runtime_root/vllm"
ln -sfn sources/aiter-gfx1151 "$runtime_root/aiter"
ln -sfn "$runtime_env" "$runtime_root/runtime-env.sh"
if ! test -x "$runtime_root/venv/bin/python"; then
  printf 'Incomplete Ciru runtime: missing venv Python under %s\n' "$runtime_root" >&2
  exit 1
fi
runtime_vllm_init="$("$runtime_root/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/vllm/__init__.py"
if ! test -f "$runtime_vllm_init"; then
  printf 'Incomplete Ciru runtime: missing vendor vLLM at %s\n' "$runtime_vllm_init" >&2
  printf 'Move aside %s and rerun runtime/INSTALL-ORNITH-RUNTIME.sh.\n' "$runtime_root" >&2
  exit 1
fi

# uv creates the virtual environment's Python as an absolute symlink. Record
# its interpreter prefix so Compose can mount it at the same path in the router.
runtime_python="$(readlink -f "$runtime_root/venv/bin/python")"
runtime_python_root="$(dirname "$(dirname "$runtime_python")")"
if ! test -x "$runtime_python"; then
  printf 'Ciru runtime Python target is not executable: %s\n' "$runtime_python" >&2
  exit 1
fi
temporary_env="$(mktemp "$root/.env.XXXXXX")"
awk -v value="$runtime_python_root" '
  /^ORNITH_RUNTIME_PYTHON_ROOT=/ {
    print "ORNITH_RUNTIME_PYTHON_ROOT=" value
    found = 1
    next
  }
  { print }
  END {
    if (!found) print "ORNITH_RUNTIME_PYTHON_ROOT=" value
  }
' "$root/.env" > "$temporary_env"
mv "$temporary_env" "$root/.env"
chmod 600 "$root/.env"

# Halogen weights. The engine reads the trunk and picks up the quality sidecar by
# name beside it, so both have to be in $HALOGEN_MODELS_DIR with the exact names
# below, and it reads tokenizer/ from the same directory. The vision sidecar is
# the third required file now that compose enables HALOGEN_VISION_TOWER.
#
# These are fetched with an explicit --include list rather than a whole-repo pull so
# the variants this stack does not use (the overlay-speed sidecar, the standalone
# MTP head) stay off the host. Re-running is cheap and resumable, and it re-checks
# each file against the remote, which is what keeps the quality sidecar current --
# a pre-0.6.0 overlay silently costs the 8-bit draft-head projections, and before
# this script fetched it there was no automatic path to a refresh at all.
halogen_repo='peonist-ai/halogen-qwen3.8-flash-next'
halogen_artifacts=(
  'qwen38-flash-next-w4b.hgn'
  'qwen38-flash-next-w4b.overlay.hgn'
  'qwen38-flash-next-vision.hgn'
  'tokenizer/*'
)

# 115.5 GiB trunk, 2.4 GiB overlay, 0.84 GiB vision, tokenizer plus headroom for
# the partial-download file hf writes beside each target while it runs. Counting
# what is already on disk means a resumed run after an interruption does not trip
# the guard on space it has already spent.
halogen_need_kib=$((121 * 1048576))
halogen_present_kib=0
for artifact in "${halogen_artifacts[@]}"; do
  for path in "$HALOGEN_MODELS_DIR"/$artifact; do
    [[ -f "$path" ]] && halogen_present_kib=$(( halogen_present_kib + $(stat -c %s "$path") / 1024 ))
  done
done
halogen_free_kib="$(df -kP "$HALOGEN_MODELS_DIR" | awk 'END { print $4 }')"
if (( halogen_free_kib + halogen_present_kib < halogen_need_kib )); then
  printf 'Not enough space under %s for the Halogen set.\n' "$HALOGEN_MODELS_DIR" >&2
  printf '  needs about %s GiB; %s GiB free, %s GiB already downloaded.\n' \
    "$(awk -v k="$halogen_need_kib" 'BEGIN { printf "%.1f", k / 1048576 }')" \
    "$(awk -v k="$halogen_free_kib" 'BEGIN { printf "%.1f", k / 1048576 }')" \
    "$(awk -v k="$halogen_present_kib" 'BEGIN { printf "%.1f", k / 1048576 }')" >&2
  printf 'Free up space and rerun. Nothing was downloaded.\n' >&2
  exit 1
fi

hf download "$halogen_repo" --include "${halogen_artifacts[@]}" --local-dir "$HALOGEN_MODELS_DIR"

# The entrypoint reads tokenizer/tokenizer.json and cannot start without it, so
# check it here instead of letting a partial tokenizer surface as a boot failure.
test -f "$HALOGEN_MODELS_DIR/tokenizer/tokenizer.json" || {
  printf 'Halogen tokenizer missing: %s\n' "$HALOGEN_MODELS_DIR/tokenizer/tokenizer.json" >&2
  printf 'Rerun this script; if it persists, check that %s still ships tokenizer/.\n' "$halogen_repo" >&2
  exit 1
}

hf download Qwen/Qwen3-Embedding-4B-GGUF \
  Qwen3-Embedding-4B-Q6_K.gguf \
  --local-dir "$MODELS_DIR/qwen3-embedding"
hf download gpustack/bge-reranker-v2-m3-GGUF \
  bge-reranker-v2-m3-Q8_0.gguf \
  --local-dir "$MODELS_DIR/bge-reranker"

printf '%s\n' \
  'Downloaded static artifacts:' \
  '  - Ornith 1.5 Ciru Halo Agent release + pinned runtime' \
  '  - Halogen trunk, quality sidecar, vision sidecar and tokenizer' \
  '' \
  'Run scripts/validate.sh, then docker compose up -d --build. The LAN gateway is' \
  'plain HTTP; verify curl http://<server-ip>:8080/v1/models before sending documents.'
