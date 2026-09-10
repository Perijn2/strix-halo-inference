# Author: Perijn
# Summary: Builds the OCR ensemble sidecar that runs MinerU, PP-OCRv5, Surya, and Tesseract behind one validated endpoint.
# Usage: Docker Compose builds this image for the ocr-ensemble service; llama-swap proxies to it and never imports this Python environment.
#
# Deliberately a separate image from llama-swap-halogen. The router image is a lean
# Vulkan llama.cpp plus Halogen stack; this one carries PyTorch, PaddlePaddle, and
# their model weights. Coupling them would make every router restart rebuild a
# multi-gigabyte ML environment.
#
# Device note: the default base is CPU PyTorch. gfx1151 has no upstream ROCm
# support, so a GPU build requires a TheRock nightly base image. Set OCR_DEVICE
# at runtime and rebuild from a ROCm base only once that build is validated; the
# application code reads the device from the environment either way.
FROM python:3.11-slim-bookworm

# tesseract-ocr is the classical engine with the largest lineage distance from the
# Qwen-derived VLMs. fonts-dejavu keeps rendered PDF page images legible.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-eng \
       fonts-dejavu-core \
       libgomp1 \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The default PyPI torch distribution pulls CUDA runtime wheels. The initial
# deployment is CPU-first, so install CPU-only torch before resolving the rest;
# MinerU and Paddle then see a compatible installed torch and do not fetch CUDA.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1

COPY docker/ocr-ensemble/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/ocr-ensemble/app/ /app/app

# Runs as a non-root user. Model weights are mounted read-only at /models. The
# pre-owned /audit-data directory seeds the named volume with writable ownership.
RUN useradd --create-home --uid 10001 ocr \
    && mkdir -p /audit-data \
    && chown -R ocr:ocr /app /audit-data
USER ocr

# One worker only. Multiple workers would each hold a full copy of every model and
# blow through the Compose memory cap immediately.
EXPOSE 8090
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
