#!/usr/bin/env bash
# Author: Perijn
# Summary: Downloads the Ciru release and every OCR/retrieval model required by the isolated runtime.
# Usage: Copy .env.example to .env, configure paths, install `hf`, then run this script on the Strix Halo Linux host.
# The runtime inference network is intentionally internal: this provisioning step must complete before Compose starts.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/load-env.sh
source "$root/scripts/lib/load-env.sh"
load_deployment_env "$root/.env"

: "${MODELS_DIR:?Set MODELS_DIR in .env.}"
: "${ORNITH_MODEL_DIR:?Set ORNITH_MODEL_DIR in .env.}"
command -v hf >/dev/null || {
  printf '%s\n' 'Install huggingface_hub first: python3 -m pip install --user huggingface_hub'
  exit 1
}
mkdir -p "$MODELS_DIR" "$ORNITH_MODEL_DIR"

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

hf download Qwen/Qwen3-Embedding-4B-GGUF \
  Qwen3-Embedding-4B-Q6_K.gguf \
  --local-dir "$MODELS_DIR/qwen3-embedding"
hf download gpustack/bge-reranker-v2-m3-GGUF \
  bge-reranker-v2-m3-Q8_0.gguf \
  --local-dir "$MODELS_DIR/bge-reranker"
hf download opendatalab/MinerU2.5-Pro-2605-1.2B \
  --local-dir "$MODELS_DIR/mineru/MinerU2.5-Pro-2605-1.2B"
hf download datalab-to/surya-ocr-2-gguf \
  surya-2.gguf \
  surya-2-mmproj.gguf \
  --local-dir "$MODELS_DIR/surya"

# PaddleOCR 3.x accepts these explicit local directories. Keeping its model
# acquisition here makes a clean runtime work without outbound Internet access.
hf download PaddlePaddle/PP-OCRv5_mobile_det \
  --local-dir "$MODELS_DIR/paddle/PP-OCRv5_mobile_det"
hf download PaddlePaddle/PP-OCRv5_mobile_rec \
  --local-dir "$MODELS_DIR/paddle/PP-OCRv5_mobile_rec"

printf '%s\n' \
  'Downloaded static artifacts:' \
  '  - Ornith 1.5 Ciru Halo Agent release + pinned runtime' \
  '  - MinerU2.5-Pro-2605-1.2B (Transformers primary)' \
  '  - Surya 2 GGUF + multimodal projector (Vulkan validator)' \
  '  - PP-OCRv5 mobile detector and recognizer (explicit local Paddle paths)' \
  '' \
  'Run scripts/validate.sh, then docker compose up -d --build. The LAN gateway is' \
  'plain HTTP; verify curl http://<server-ip>:8080/v1/models before sending documents.'
