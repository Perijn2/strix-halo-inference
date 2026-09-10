#!/usr/bin/env bash
# Author: Perijn
# Summary: Downloads the retrieval GGUF artifacts required by llama-swap.
# Usage: Install the user-supplied Ornith ROCMFP4 GGUF at the documented path, set MODELS_DIR, install `hf`, then run this script.
# The script deliberately does not download Ornith or Halogen weights because their selected artifacts are installed separately.
set -euo pipefail

: "${MODELS_DIR:?Set MODELS_DIR to the directory mounted at /models.}"
command -v hf >/dev/null || {
  printf '%s\n' 'Install huggingface_hub first: python3 -m pip install --user huggingface_hub'
  exit 1
}

ornith_path="$MODELS_DIR/ornith/Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf"
[[ -r "$ornith_path" ]] || {
  printf 'Install the selected Ornith GGUF first: %s\n' "$ornith_path" >&2
  exit 1
}

hf download Qwen/Qwen3-Embedding-4B-GGUF \
  Qwen3-Embedding-4B-Q6_K.gguf \
  --local-dir "$MODELS_DIR/qwen3-embedding"
hf download gpustack/bge-reranker-v2-m3-GGUF \
  bge-reranker-v2-m3-Q8_0.gguf \
  --local-dir "$MODELS_DIR/bge-reranker"
