<!--
Author: Perijn
Summary: Deploys a role-routed AI inference stack for a Strix Halo Ryzen AI Max+ 395 host.

Usage:
Core principle: Caddy exposes one authenticated API, while llama-swap is the only inference router and presents stable role IDs to clients.

Setup: Copy `.env.example` to `.env`, set host paths and credentials, create the PostgreSQL password file, download the documented model artifacts, then run `docker compose up -d`. Add `--profile rag` to enable PostgreSQL with pgvector.

Workflow: The default `orchestration` profile sends planner, architect, reviewer, and general roles to Halogen Qwen. llama-swap starts Qwen on its first routed request and can unload it explicitly through its UI or API. Embedding and reranking processes preload and remain available beside Qwen. Switch to the `engineering` profile only when Ornith should receive implementation roles. Query `GET /api/profiles` and update the profile through `PUT /api/profiles/active`.

API guide: Use `/v1/chat/completions` with `model` set to a stable `role/...` ID. Use `role/embed` through `/v1/embeddings` and `role/rerank` through `/v1/rerank`. The llama-swap administrative API is protected by the same Caddy authentication.

Worked example: Run `cp .env.example .env`, replace its paths and Caddy password hash, start `docker compose up -d`, then call `curl -k -u inference:<password> https://localhost:8443/v1/models`. Select the engineering profile before sending a request with `model: role/implementer`.
-->

# Strix Halo inference

Reliable Docker Compose inference for a dedicated Ryzen AI Max+ 395 / Radeon 8060S host.

## Roles

| Stable role | Orchestration profile | Engineering profile |
| --- | --- | --- |
| `role/orchestrator` | Qwen, thinking/high | disabled |
| `role/architect` | Qwen, thinking/high | disabled |
| `role/reviewer` | Qwen, thinking/high | disabled |
| `role/implementer` | disabled | Ornith, thinking |
| `role/tester` | disabled | Ornith, thinking |
| `role/documenter` | disabled | Ornith, fast |
| `role/embed` | Qwen3-Embedding-4B | Qwen3-Embedding-4B |
| `role/rerank` | BGE reranker v2-m3 | BGE reranker v2-m3 |

Retrieval preloads at startup. llama-swap loads Qwen on its first role request and keeps it loaded (`ttl: 0`) until an explicit llama-swap unload action. Halogen has two 131K slots sharing a 262K total KV pool.

## Start

```bash
cp .env.example .env
# Edit model paths in .env, then generate local credentials.
./scripts/generate-secrets.sh
docker compose up -d
# Optional durable RAG storage:
docker compose --profile rag up -d
# Localhost uses Caddy's local certificate; use -k for local curl tests.
```

`scripts/generate-secrets.sh` writes Caddy and PostgreSQL passwords to the ignored `secrets/` directory and updates `.env` with the Caddy bcrypt hash and PostgreSQL password-file path. It never prints the passwords. Run it with `--force` only when intentionally rotating both credentials.

## Model files

The `MODELS_DIR` mount must contain:

```text
ornith/Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
qwen3-embedding/Qwen3-Embedding-4B-Q6_K.gguf
bge-reranker/bge-reranker-v2-m3-Q8_0.gguf
```

`HALOGEN_MODELS_DIR` is the flat model directory downloaded from the Halogen Qwen repository; it contains the `.hgn` checkpoint, quality overlay, and `tokenizer/` directory.

Install the selected Ornith ROCMFP4 STRIX LEAN GGUF at the path above, then use [`scripts/download-models.sh`](scripts/download-models.sh) to fetch retrieval GGUF artifacts. Read [docs/operations.md](docs/operations.md) before operating Halogen.

## Profile switching

```bash
curl -k -u "$CADDY_API_USER:$PASSWORD" https://localhost:8443/api/profiles
curl -k -u "$CADDY_API_USER:$PASSWORD" \
  -X PUT https://localhost:8443/api/profiles/active \
  -H 'content-type: application/json' \
  --data '{"name":"engineering"}'
```

The role profile changes routing and request parameters. `:think` and `:fast` variants reuse one loaded process; they do not reload model weights.

## Qwen lifecycle

llama-swap owns the Halogen process. The first request for a Qwen role starts Halogen and waits for `/health`; cold loading can take minutes. It remains loaded until explicitly removed:

```bash
curl -k -u "$CADDY_API_USER:$PASSWORD" \
  -X POST https://localhost:8443/api/models/unload/qwen3.8-flash-next
```

The next Qwen role request starts it again. There is no idle TTL unload.

## Sources

- [llama-swap](https://github.com/mostlygeek/llama-swap)
- [Halogen Flash Server](https://github.com/peonist-ai/halogen-flash-server)
- [Laurent Zuijdwijk llama.cpp fork](https://github.com/LaurentZuijdwijk/llama.cpp)
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [pgvector](https://github.com/pgvector/pgvector)
