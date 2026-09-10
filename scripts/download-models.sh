#!/usr/bin/env bash
# Author: Perijn
# Summary: Downloads the retrieval and OCR model artifacts required by llama-swap and the isolated OCR ensemble.
# Usage: Install the user-supplied Ornith ROCMFP4 GGUF at the documented path, set MODELS_DIR, install `hf`, then run this script.
# The script deliberately does not download Ornith or Halogen weights because their selected artifacts are installed separately.
#
# MinerU is downloaded as the original Transformers repository because the ensemble
# starts with mineru-vl-utils[transformers], not a GGUF. Surya is downloaded as
# the official GGUF plus multimodal projector for the dedicated Vulkan worker.
# PP-OCRv5's mobile detector/recognizer are fetched by PaddleOCR into the named
# ocr_cache volume on its first healthy start; Tesseract's English tessdata comes
# from the image's tesseract-ocr-eng Debian package.
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

# Primary structured-document parser. Keep the model's complete repository:
# config, processor, tokenizer, and safetensor shards are all load-bearing.
hf download opendatalab/MinerU2.5-Pro-2605-1.2B \
  --local-dir "$MODELS_DIR/mineru/MinerU2.5-Pro-2605-1.2B"

# Independent VLM checker. The ensemble's Surya client talks to the llama.cpp
# worker defined in Compose; the worker needs both files, not a Python checkpoint.
hf download datalab-to/surya-ocr-2-gguf \
  surya-2.gguf \
  surya-2-mmproj.gguf \
  --local-dir "$MODELS_DIR/surya"

cat <<'EOF'
Downloaded static artifacts:
  - MinerU2.5-Pro-2605-1.2B (Transformers primary)
  - Surya 2 GGUF + multimodal projector (Vulkan validator)

First OCR-sidecar start also downloads PP-OCRv5 mobile detector/recognizer into
Docker's named ocr_cache volume. Tesseract English data is built into the image.
Run `docker compose up -d --build`, then verify `docker compose ps` and
`curl -k https://localhost:8443/v1/models` before sending documents.
EOF