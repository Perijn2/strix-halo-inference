<!--
Author: Perijn
Summary: Deploys a role-routed AI inference stack for a Strix Halo Ryzen AI Max+ 395 host.

Usage:
Core principle: Caddy exposes one authenticated API, while llama-swap is the only inference router and presents stable role IDs to clients.

Setup: Copy `.env.example` to `.env`, set host paths and credentials, create the PostgreSQL password file, download the documented model artifacts, then run `docker compose up -d`. Add `--profile rag` to enable PostgreSQL with pgvector.

Workflow: The default `orchestration` profile sends planner, architect, reviewer, and general roles to persistent Halogen Qwen. Embedding and reranking processes preload and remain available beside Qwen. Switch to the `engineering` profile only when Ornith should receive implementation roles. Query `GET /api/profiles` and update the profile through `PUT /api/profiles/active`.

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

Qwen and retrieval are always-hot in the normal orchestration mode. Halogen has two 131K slots sharing a 262K total KV pool. The Qwen service has no automatic TTL unload.

## Start

```bash
cp .env.example .env
# Edit .env, create its POSTGRES_PASSWORD_FILE, and replace the default Caddy hash.
docker compose up -d
# Optional durable RAG storage:
docker compose --profile rag up -d
# Localhost uses Caddy's local certificate; use -k for local curl tests.
```

Use a Caddy password hash rather than a clear-text password:

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'choose-a-password'
```

## Model files

The `MODELS_DIR` mount must contain:

```text
ornith/Ornith-1.5-35B-Q4_K_M.gguf
qwen3-embedding/Qwen3-Embedding-4B-Q6_K.gguf
bge-reranker/bge-reranker-v2-m3-Q8_0.gguf
```

`HALOGEN_MODELS_DIR` is the flat model directory downloaded from the Halogen Qwen repository; it contains the `.hgn` checkpoint, quality overlay, and `tokenizer/` directory.

Use [`scripts/download-models.sh`](scripts/download-models.sh) to fetch the GGUF artifacts. Read [docs/operations.md](docs/operations.md) before operating Halogen.

## Profile switching

```bash
curl -k -u "$CADDY_API_USER:$PASSWORD" https://localhost:8443/api/profiles
curl -k -u "$CADDY_API_USER:$PASSWORD" \
  -X PUT https://localhost:8443/api/profiles/active \
  -H 'content-type: application/json' \
  --data '{"name":"engineering"}'
```

The role profile changes routing and request parameters. `:think` and `:fast` variants of Ornith reuse one loaded process; they do not reload model weights.

## Sources

- [llama-swap](https://github.com/mostlygeek/llama-swap)
- [Halogen Flash Server](https://github.com/peonist-ai/halogen-flash-server)
- [Laurent Zuijdwijk llama.cpp fork](https://github.com/LaurentZuijdwijk/llama.cpp)
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [pgvector](https://github.com/pgvector/pgvector)
