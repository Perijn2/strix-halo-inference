<!--
Author: Perijn
Summary: Deploys a role-routed AI inference stack for a Strix Halo Ryzen AI Max+ 395 host.
Usage:
Core principle: Caddy exposes the unauthenticated API and UI on the configured listener, while llama-swap is the only inference router and presents stable role IDs to clients. Restrict host network access before exposing this listener.

Setup: Copy `.env.example` to `.env`, set host paths, run the generator to create the PostgreSQL password file and host group settings, download the documented model artifacts, then run `docker compose up -d`. PostgreSQL starts by default and may store RAG vectors and telemetry metadata.

Workflow: Halogen Qwen handles orchestration, architecture, review, and general roles; Ciru Ornith handles implementation, testing, and documentation through its supplied custom vLLM/ROCm runtime. llama-swap starts and supervises each model on its first routed request. Embedding and reranking processes preload and remain available beside them.

API guide: Use `/v1/chat/completions` with `model` set to a stable `role/...` ID. Use `role/embed` through `/v1/embeddings` and `role/rerank` through `/v1/rerank`. The llama-swap UI is available at `/ui`; its administrative API is unauthenticated.

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

llama-swap starts Qwen and Ciru Ornith lazily, without routing profiles or a swap matrix. All llama-swap children use `ttl: 0` and stay loaded until an explicit unload action. Ornith uses its vendor-required IU4 `agents64k` profile: 262K maximum model length, eight sequences, and a 44 GiB KV/state pool. It targets 80% of the 96 GiB GPU memory, so unload Qwen and other GPU workloads before loading Ornith on this host.

## Start

```bash
cp .env.example .env
# Edit model paths in .env, then generate local credentials.
./scripts/generate-secrets.sh
docker compose up -d
# PostgreSQL starts by default and may store RAG vectors and telemetry metadata.
# Caddy is HTTP-only for LAN use; restrict its unauthenticated port with a host firewall.
```

`scripts/generate-secrets.sh` writes the PostgreSQL password to the ignored `secrets/` directory and records numeric host `render`/`video` group IDs in `.env`. With an existing credential, `--force` performs a live `ALTER ROLE` through the running PostgreSQL service before atomically replacing the local secret. If that file was lost while `postgres_data` still exists, run `./scripts/generate-secrets.sh --recover`: it creates a replacement and reconciles the live role without deleting the audit ledger, even if `.env` is still missing its GPU-group or Ornith-runtime interpolation values.

## Model files

The `MODELS_DIR` mount must contain the retrieval artifacts below. `ORNITH_MODEL_DIR` is a separate mount containing the complete Ciru release and its installed runtime:

```text
qwen3-embedding/Qwen3-Embedding-4B-Q6_K.gguf
bge-reranker/bge-reranker-v2-m3-Q8_0.gguf
```

`HALOGEN_MODELS_DIR` is the flat model directory downloaded from the Halogen Qwen repository; it contains the `.hgn` checkpoint, the quality overlay, the vision tower sidecar (`qwen38-flash-next-vision.hgn`, required while `HALOGEN_VISION_TOWER` is on, and enforced by `scripts/validate.sh`), and the `tokenizer/` directory. `ORNITH_MODEL_DIR` must contain the complete [jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo) release, including `bundle/`, `runtime/`, and `installed-runtime/`. The included installer creates `installed-runtime/runtime-env.sh`; `scripts/download-models.sh` also records the uv interpreter root in `ORNITH_RUNTIME_PYTHON_ROOT` so Compose can make its absolute venv symlink available to the container. Its pinned vLLM/ROCm runtime and custom kernels are required—stock vLLM and the prior llama.cpp GGUF are incompatible.

Use [`scripts/download-models.sh`](scripts/download-models.sh) to fetch the complete Ciru release, install its pinned runtime, and fetch the retrieval artifacts. Run it on the Linux Strix Halo host with sufficient disk space and memory; the published Ciru profile reports a 95.35 GiB peak whole-host measurement. Read [docs/operations.md](docs/operations.md) before operating Halogen.

## Ciru Ornith lifecycle

The old `Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` llama.cpp process is no longer used. llama-swap starts the Ciru launcher from `/ornith/bundle/serve.sh` behind `ciru-model-proxy` on its first request, preserving its custom quantization, native kernels, DFlash2 drafter, prefix cache, and OpenAI-compatible tool calling. That release serves exactly one model name (`ciru-halo-agent`), so the proxy maps the accepted llama-swap IDs onto it and returns each caller's own ID in the reply; see [Ciru model-name normalization](docs/operations.md#ciru-model-name-normalization). The proxy also gives Ornith the activity telemetry Halogen has emitted natively since 0.7.0, deriving the same llama.cpp-compatible `timings` from wire clocks and the response's `usage` counts; see [Ciru PP/TG telemetry](docs/operations.md#ciru-pp-tg-telemetry). After it loads, `ttl: 0` keeps it resident beside Qwen until an explicit unload; `role/implementer`, `role/tester`, and `role/documenter` reuse that process. On unload the proxy stops and drains the whole engine process group so teardown completes before llama-swap proceeds; the transient `resource_tracker` semaphore notice that vLLM's abort-shutdown produces is explained in [Unload teardown and the resource tracker warning](docs/operations.md#unload-teardown-and-the-resource-tracker-warning).

## Qwen lifecycle

llama-swap starts Halogen on its first Qwen request and waits for its `/health` endpoint. Cold loading can take minutes. Qwen remains loaded until explicitly removed:

```bash
curl \
  -X POST http://<server-lan-ip>:8080/api/models/unload/qwen3.8-flash-next
```

The next Qwen role request starts it again. There is no idle TTL unload.

## Sources

- [llama-swap](https://github.com/mostlygeek/llama-swap)
- [Halogen Flash Server](https://github.com/peonist-ai/halogen-flash-server)
- [Laurent Zuijdwijk llama.cpp fork](https://github.com/LaurentZuijdwijk/llama.cpp)
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [pgvector](https://github.com/pgvector/pgvector)
