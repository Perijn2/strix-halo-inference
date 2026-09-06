#!/usr/bin/env bash
# Author: Perijn
# Summary: Downloads the GGUF retrieval and Ornith artifacts expected by llama-swap.
# Usage: Set MODELS_DIR, install `hf` from huggingface_hub, then run this script once; Hugging Face resumes interrupted downloads.
# The script deliberately does not download Halogen weights because their destination and storage policy are configured separately.
set -euo pipefail

: "${MODELS_DIR:?Set MODELS_DIR to the directory mounted at /models.}"
command -v hf >/dev/null || {
  printf '%s\n' 'Install huggingface_hub first: python3 -m pip install --user huggingface_hub'
  exit 1
}

hf download ornith-ai/Ornith-1.5-35B-A3B-GGUF \
  Ornith-1.5-35B-Q4_K_M.gguf \
  --local-dir "$MODELS_DIR/ornith"
hf download Qwen/Qwen3-Embedding-4B-GGUF \
  Qwen3-Embedding-4B-Q6_K.gguf \
  --local-dir "$MODELS_DIR/qwen3-embedding"
hf download gpustack/bge-reranker-v2-m3-GGUF \
  bge-reranker-v2-m3-Q8_0.gguf \
  --local-dir "$MODELS_DIR/bge-reranker"
