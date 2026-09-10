<!--
Author: Perijn
Summary: Deploys a role-routed AI and validated OCR inference stack for a Strix Halo Ryzen AI Max+ 395 host.
Usage:
Core principle: Caddy exposes the unauthenticated API and UI on the configured listener, while llama-swap is the only inference router and presents stable role IDs to clients. `role/ocr` is a supervised forwarder to an isolated four-engine ensemble. Restrict host network access before exposing this listener.

Setup: Copy `.env.example` to `.env`, set host paths, run the generator to create the PostgreSQL password file and host group settings, download the documented model artifacts, then run `docker compose up -d`. PostgreSQL starts by default because it is the durable OCR audit ledger.

Workflow: Halogen Qwen handles orchestration, architecture, review, and general roles; Ciru Ornith handles implementation, testing, and documentation through its supplied custom vLLM/ROCm runtime. llama-swap starts and supervises each model on its first routed request. Embedding and reranking processes preload and remain available beside them. `role/ocr` runs MinerU, PP-OCRv5, Tesseract, and Surya for every page and routes disagreement to review.

API guide: Use `/v1/chat/completions` with `model` set to a stable `role/...` ID. Use `role/embed` through `/v1/embeddings`, `role/rerank` through `/v1/rerank`, and `role/ocr` with one base64 image_url content part. The llama-swap UI is available at `/ui`; its administrative API is unauthenticated.

Worked example: Run `cp .env.example .env`, configure paths, run `scripts/generate-secrets.sh`, run `scripts/download-models.sh` on the Strix Halo Linux host, start `docker compose up -d --build`, then call `curl -k https://localhost:8443/v1/models`. Send implementation work with `model: role/implementer`.
-->

# Strix Halo inference

Reliable Docker Compose inference for a dedicated Ryzen AI Max+ 395 / Radeon 8060S host.

## Roles

| Stable role | Routed model |
| --- | --- |
| `role/orchestrator` | Halogen Qwen, thinking/high |
| `role/architect` | Halogen Qwen, thinking/high |
| `role/reviewer` | Halogen Qwen, thinking/high |
| `role/implementer` | Ornith 1.5 Ciru Halo Agent, thinking |
| `role/tester` | Ornith 1.5 Ciru Halo Agent, thinking |
| `role/documenter` | Ornith 1.5 Ciru Halo Agent, fast |
| `role/embed` | Qwen3-Embedding-4B |
| `role/rerank` | BGE reranker v2-m3 |
| `role/ocr` | Four-engine validated OCR |

Retrieval preloads at startup. OCR is a separate, always-on, memory-capped sidecar; llama-swap owns only its local supervised forwarder and stable `role/ocr` ID. llama-swap starts Qwen and the Ciru Ornith runtime on their first role requests and keeps them loaded (`ttl: 0`) until an explicit llama-swap unload action. The deployed Ciru text profile limits each session to 131K tokens and admits up to six active agent sessions, while retaining its shared 44 GiB KV/state pool; provision the release's documented whole-host memory budget before enabling it.

## Start

```bash
cp .env.example .env
# Edit model paths in .env, then generate local credentials.
./scripts/generate-secrets.sh
docker compose up -d
# PostgreSQL starts by default for OCR audit history and may also store RAG metadata.
# Localhost uses Caddy's local certificate; use -k for local curl tests.
```

`scripts/generate-secrets.sh` writes the PostgreSQL password to the ignored `secrets/` directory and records numeric host `render`/`video` group IDs in `.env`. Run it with `--force` only when intentionally rotating the database credential.

## Model files

The `MODELS_DIR` mount must contain the retrieval and OCR artifacts below. `ORNITH_MODEL_DIR` is a separate mount containing the complete Ciru release and its installed runtime:

```text
qwen3-embedding/Qwen3-Embedding-4B-Q6_K.gguf
bge-reranker/bge-reranker-v2-m3-Q8_0.gguf
mineru/MinerU2.5-Pro-2605-1.2B/  # complete Hugging Face Transformers repository
surya/surya-2.gguf
surya/surya-2-mmproj.gguf
```

`HALOGEN_MODELS_DIR` is the flat model directory downloaded from the Halogen Qwen repository; it contains the `.hgn` checkpoint, quality overlay, and `tokenizer/` directory. `ORNITH_MODEL_DIR` must contain the complete [jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo) release, including `bundle/`, `runtime/`, and `installed-runtime/`. The included `INSTALL-ORNITH-RUNTIME.sh` creates `installed-runtime/`; its pinned vLLM/ROCm runtime and custom kernels are required—stock vLLM and the prior llama.cpp GGUF are incompatible.

Use [`scripts/download-models.sh`](scripts/download-models.sh) to fetch the complete Ciru release, install its pinned runtime, and fetch retrieval/OCR artifacts. Run it on the Linux Strix Halo host with sufficient disk space and memory; the published Ciru profile reports a 95.35 GiB peak whole-host measurement. Read [docs/operations.md](docs/operations.md) before operating Halogen.

## Validated OCR

`role/ocr` receives one rendered document page as a standard OpenAI `image_url` content part and returns the fused OCR text. The ensemble is not an LLM vote: MinerU2.5-Pro-2605 is the primary parser; PP-OCRv5 and Tesseract are non-generative validation engines; Surya 2 is an independent VLM structure check.

For a transient full-page audit—per-engine status, normalized bounding boxes, pairwise agreement, Consensus Entropy, ungrounded numeric tokens, and a typed review verdict—post `{ "image_b64": "...", "page_ref": "..." }` to `/upstream/ocr-ensemble/ocr`. For durable document review, POST an image or PDF as `{ "document_b64": "...", "filename": "...", "media_type": "application/pdf" }` to `/upstream/ocr-ensemble/documents`; it renders every page, audits it, and retains source/pages/audits for `OCR_AUDIT_RETENTION_DAYS` (90 by default). Treat every `review: true` response as a human-review queue item.

The initial deployment is CPU-first for the Python engines; Surya uses the existing Vulkan llama.cpp build through a compatibility-isolated SDK client. The default 6 GiB ensemble, 2 GiB SDK client, and 4 GiB Surya worker limits are a strict combined 12 GiB budget. Do not change `OCR_DEVICE`, `OCR_MINERU_BACKEND`, or any memory limit without following the validation steps in [docs/operations.md](docs/operations.md).

## OCR playground

After `docker compose up -d --build`, open `https://localhost:8443/ocr-playground/` and accept Caddy's local-certificate warning. Upload a PDF or image; the workspace renders/selects every page, stores the audit for 90 days, overlays normalized boxes from each OCR witness, and exposes a saved review history. The public gateway has a 32 MiB request cap, leaving roughly 24 MiB for a base64-encoded source upload.

## Ciru Ornith lifecycle

The old `Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` llama.cpp process is no longer used. `role/implementer`, `role/tester`, and `role/documenter` start the Ciru launcher from `/ornith/bundle/serve.sh`, preserving its custom quantization, native kernels, DFlash2 drafter, prefix cache, and OpenAI-compatible tool calling. `:think` and `:fast` variants reuse the same loaded Ciru process; they do not reload model weights.

## Qwen lifecycle

llama-swap owns the Halogen process. The first request for a Qwen role starts Halogen and waits for `/health`; cold loading can take minutes. It remains loaded until explicitly removed:

```bash
curl -k \
  -X POST https://localhost:8443/api/models/unload/qwen3.8-flash-next
```

The next Qwen role request starts it again. There is no idle TTL unload.

## Sources

- [llama-swap](https://github.com/mostlygeek/llama-swap)
- [Halogen Flash Server](https://github.com/peonist-ai/halogen-flash-server)
- [Laurent Zuijdwijk llama.cpp fork](https://github.com/LaurentZuijdwijk/llama.cpp)
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [pgvector](https://github.com/pgvector/pgvector)