# Author: Perijn
# Summary: Builds the Surya SDK client sidecar that calls the internal Vulkan Surya VLM worker.
# Usage: Docker Compose builds this image for surya-client; the OCR ensemble POSTs page images to it and never imports Surya directly.
#
# Surya and MinerU have an unsatisfiable Pillow constraint. This image is intentionally
# separate from docker/ocr-ensemble: it contains the Surya SDK's Pillow <11 stack,
# while the MinerU sidecar owns Pillow >=11.
FROM python:3.11-slim-bookworm

WORKDIR /app

# The client needs torch for Surya's SDK but never CUDA. Pin the CPU index so
# this AMD deployment does not download unused NVIDIA runtime wheels.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1

COPY docker/surya-client/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/surya-client/app/ /app/app

RUN useradd --create-home --uid 10002 surya \
    && chown -R surya:surya /app
USER surya

EXPOSE 8091
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
