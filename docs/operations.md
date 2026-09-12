<!--
Author: Perijn
Summary: Describes safe operation, profiling, rollback, and review handling for the inference and validated OCR stack.

Usage:
Core principle: Preserve Qwen, Ciru Ornith, and retrieval availability within the measured host-memory budget; treat OCR disagreement as a review signal, never as a license to silently rewrite source documents.

Setup: Configure `.env`, download verified model artifacts with `scripts/download-models.sh`, run `scripts/validate.sh`, build Compose, and confirm all service health endpoints.

Workflow: Check Caddy, llama-swap, ocr-ensemble, Surya, and PostgreSQL health. Send normal model requests to stable role IDs. Upload a PDF/image to the review workspace or `/upstream/ocr-ensemble/documents`; consume the rendered pages and structured audit only for review, not as a replacement for the original document.

API guide: `/health` verifies server readiness, `/v1/models` lists callable role IDs, `/upstream/ocr-ensemble/ocr` returns one transient audit, and `/upstream/ocr-ensemble/documents` creates a retained multi-page audit.

Worked example: Start the stack, wait for the OCR sidecar's first model load, upload a difficult PDF to `/ocr-playground/`, and review every page with `review: true` before releasing an extraction.
-->

# Operations

## Isolated test deployment

Use a distinct Compose project name and unused gateway port for a test stack, for example `CADDY_HTTP_PORT=18081 docker compose -p strix-halo-test up -d --build`. Do not run a full stable and full test model stack concurrently on one Strix Halo host: their combined unified-memory demand is unsafe. If Docker reports that a port is allocated, inspect the existing publisher with `docker ps --filter publish=18081` and choose another unused test port or stop only the disposable test project.

## Memory policy

Halogen uses two 131K-context slots and a 262K shared KV pool. `HALOGEN_HOST_RESERVE_GIB` is intentionally set to 28 GiB, leaving headroom for the persistent embedding and reranker workloads. Do not raise slots, context, or pool size without a fresh load and concurrency benchmark.

Qwen and Ciru Ornith start lazily on their first matching request, with no routing profiles, groups, or swap matrix. Each has `ttl: 0`, so a successfully loaded model remains resident until explicitly unloaded. Halogen and the Ciru runtime can take minutes to load. The vendor-required Ornith IU4 profile uses a 262K maximum model length, eight sequences, and a 44 GiB shared KV/state pool; it targets 80% of the 96 GiB GPU memory. Unload Qwen and other GPU workloads before loading Ornith on this host. Its published deployment measured a 95.35 GiB whole-host peak; do not run its full profile beside other large GPU models.

## Halogen PP/TG telemetry

Halogen reports its measured prefill and decode timings in its per-request `serve_api:` log ledger, not in the OpenAI response. `halogen-telemetry-proxy` is launched automatically between llama-swap and Halogen: it parses that ledger and adds llama.cpp-compatible `timings` to completed chat responses. llama-swap then records prompt-processing (PP) and token-generation (TG) speeds in Activity. The proxy preserves streaming and inserts its timing event immediately before the terminal SSE `[DONE]` event.

The timings are the engine's own prefill and decode-window measurements. Halogen does not expose a request ID in this ledger, so the proxy serializes decorated chat requests and only accepts ledger lines emitted after each request begins; this trades telemetry throughput for correct attribution. A missing PP/TG value means that no matching ledger line was available before the proxy's two-second wait; investigate the llama-swap upstream log stream rather than substituting HTTP wall-clock speed. Confirm the adapter after deployment with a non-streaming `role/general` chat request and `GET /api/metrics/activity?model=qwen3.8-flash-next`.

The OCR budget is deliberately separate and strict: `OCR_ENSEMBLE_MEMORY_LIMIT_GIB=6`, `OCR_SURYA_CLIENT_MEMORY_LIMIT_GIB=2`, and `OCR_SURYA_MEMORY_LIMIT_GIB=4` total 12 GiB. The Surya SDK client is separate because it requires Pillow <11 while MinerU requires Pillow >=11. The ensemble starts CPU-first (`OCR_DEVICE=cpu`) because upstream ROCm and PaddlePaddle do not validate `gfx1151`; Surya uses the existing Vulkan llama.cpp image. Do not change any cap, move MinerU to vLLM, or enable a Python GPU device until you measure all resident Qwen, Ornith, embedding, reranker, MinerU, and Surya memory together.

If you later move MinerU to vLLM, rebuild the OCR image from a tested TheRock-compatible base and set `OCR_VLLM_GPU_MEMORY_UTILIZATION` conservatively. vLLM’s available-memory accounting is not reliable on unified-memory hardware, so it must not be allowed to preallocate around the existing 28 GiB host reserve.

## OCR validation and review

`role/ocr` is a llama-swap role, but the fan-out and fusion happen in the `ocr-ensemble` sidecar. llama-swap starts a supervised local `socat` forwarder and proxies it; it cannot itself vote between models. The sidecar runs MinerU2.5-Pro-2605 as primary, PP-OCRv5 and Tesseract as classical checks, and calls the compatibility-isolated Surya SDK client, which calls the internal Surya worker for independent VLM validation.

The normal OpenAI endpoint returns only fused text. Submit a base64 page image to `POST /upstream/ocr-ensemble/ocr` when your caller needs the audit record. Persist the original page image and the audit response together; do not preserve extracted text without its source page and verdict.

| Verdict | Meaning | Required action |
| --- | --- | --- |
| `accept` | Every live engine agreed above the calibrated threshold and Consensus Entropy was low. | May proceed; retain the audit record. |
| `accept_weighted` | MinerU had at least one classical corroborator, but the page was not unanimous. | Spot-check according to your sampling policy. |
| `review_hallucination` | A mutually corroborating classical pair disagreed with MinerU, or MinerU/selected output emitted an ungrounded numeric value. | Highest-priority human review. |
| `review_hard_page` | PP-OCRv5 and Tesseract disagree strongly. | Review regardless of either VLM result. |
| `review_entropy` | Too little corroboration, high disagreement, or no engine returned text. | Review before release. |

The numeric grounding rule is intentionally conservative: it only treats a primary-model numeric token as strong evidence when the classical pair corroborates each other. A model-generated bounding box is not independent evidence; verify regulated values against a trusted PDF text layer or the original rendered page.

### Threshold calibration

The defaults in `.env.example` are starting points, not benchmark-derived guarantees. Before production:

1. Assemble a held-out corpus containing your actual difficult documents: faint scans, stamps, handwriting over print, skew, near-blank pages, tables, and multi-column layouts.
2. Record the raw per-engine text and the audit verdict for every page.
3. Label extraction faults against the original page or a trusted PDF text layer.
4. Tune only with a held-out split. Never tune against the same pages used to assert an error rate.
5. Choose a review capacity target, then measure fault discovery at that queue depth. Do not turn off a structural or numeric review rule to make the queue shorter.

## Model lifecycle

No llama-swap profiles, groups, or swap matrix restrict the configured Qwen, Ciru Ornith, retrieval, and OCR-forwarder processes; each starts on its first request and then remains loaded because its TTL is zero. Memory limits—not profile names—remain the practical resource boundary. `role/implementer`, `role/tester`, and `role/documenter` use the Ciru runtime directly. You can explicitly unload Qwen through `POST /api/models/unload/qwen3.8-flash-next` or Ciru Ornith through `POST /api/models/unload/ciru-ornith-1.5-halo-agent`; the next matching role request reloads it. Do not rely on an idle timeout.

### Cold-start health-check budget

llama-swap kills a child whose `checkEndpoint` never turns ready within `healthCheckTimeout` (default 120 s, minimum 15 s). Ciru Ornith's first load exceeds that window: its pinned `torch.compile`, a ~53 s profiling/warmup run, and CUDA-graph capture together take several minutes before `/health` reports ready. The signature is the engine log halting mid `Capturing CUDA graphs (decode, FULL)` with no Python traceback while the router restarts the child in a loop; a roughly fixed ~2-minute time-of-death means the startup timeout fired, not an out-of-memory kill. The Ciru model therefore sets `healthCheckTimeout: 900`. Once the AOT compile and warm caches exist in the `ornith_cache` volume reloads are much faster, but keep the extended budget so a cold cache after a fresh volume, a runtime/kernel update, or `docker compose down -v` still has room to finish. If you change the capture or warmup profile, keep this comfortably above the measured cold-load wall-clock time.

### Profile commands run without a shell

llama-swap shlex-splits a profile `cmd`, discards `#` lines, and execs `argv[0]` itself; it never spawns a shell. A leading `exec` is therefore looked up as a program literally named `exec` and the child dies with `exec: "exec": executable file not found in $PATH`, which is how the OCR forwarder was failing. The same applies to every other shell construct: `&&`, `||`, pipes, globs, and `$VAR` expansions are arguments rather than syntax. Only llama-swap's own substitution (`${PORT}`, macros) is applied. Write each profile command as one executable plus plain arguments; comment lines remain safe because they are stripped before splitting, which is why the Ciru `serve.sh` profile's comments do not break it.

### Ciru model-name normalization

vLLM validates the request body's `model` field against the single name its launch arguments advertise, and the Ciru release advertises `ciru-halo-agent` only. llama-swap resolves aliases for routing but forwards the body unchanged, so every other client-supplied ID is rejected by the engine with `The model ... does not exist` (404) before inference starts. Neither place can fix that upstream: the vendor release is mounted read-only at `/ornith` (only `bundle/cache` is writable), and Caddy's `request_body` directive can set a whole body but cannot rewrite one JSON field.

`ciru-model-proxy` closes the gap inside the llama-swap image, mirroring `halogen-telemetry-proxy`. The Ciru profile's `cmd` starts the proxy instead of `serve.sh`; the proxy launches the vendor launcher on the next free loopback port and stays in front of it:

- A `POST` whose JSON body names an accepted ID is rewritten to the served name on the way in, adding `X-Ciru-Model-Normalized: <requested id>` for traceability. Only that field changes: everything else, prompt text included, passes byte-for-byte, and a value quoting a model field inside a string cannot be mistaken for one.
- The reply's `model` field is rewritten back to the ID the caller used, for buffered JSON and for each streamed SSE event, so callers that assert on the response ID keep working.
- Any other name is forwarded unchanged and its 404 surfaces verbatim. Each distinct unmapped name logs one `ciru-model-proxy: unmapped model name ...` line into llama-swap's upstream log stream.
- The proxy recomputes request and response framing instead of copying it, because the rewritten name changes the payload length; reusing the inbound `Content-Length` truncates longer names and hangs shorter ones. Chunked request bodies are refused with 400 rather than silently mis-forwarded.
- The engine runs in its own process group, and `SIGTERM` escalates to `SIGKILL` after `--shutdown-grace` (30 s default) so the `EngineCore` child tree cannot survive as an orphan holding ~90 GiB of the unified pool.

`CIRU_MODEL_ALIASES` (Compose default `role/implementer,role/tester,role/documenter,ornith-1.5-35b-a3b`) is the accepted set, `CIRU_SERVED_MODEL` names what it maps to, and `CIRU_SERVE_SCRIPT` names the launcher command. Widen the set in `.env` and recreate the llama-swap container; no image rebuild is needed. Keep the alias list and the router's `aliases:` / `setParamsByID` keys in step — an ID that routes but is not accepted here fails the other way round.

**`ciru-ornith-1.5-halo-agent` is deliberately not an accepted body value.** That literal is the llama-swap model ID used by admin routes such as `POST /api/models/unload/ciru-ornith-1.5-halo-agent`, and sending it as the chat body's `model` still returns the engine's 404. If callers send it that way, append it to `CIRU_MODEL_ALIASES`; the router keeps its own ID regardless, so the two never collide.

Verify after deployment:

```bash
# accepted alias: 200, reply names the caller's own ID, upstream log shows the rewrite
curl -sS http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"role/implementer","messages":[{"role":"user","content":"say ok"}]}'

# unmapped value: the engine's 404 plus one warning line
docker compose logs llama-swap | grep 'ciru-model-proxy'

# teardown: no engine process may outlive the proxy
docker compose restart llama-swap
sleep 5 && docker compose exec llama-swap sh -c 'ps -eo pid,cmd | grep "[s]erve.sh" | wc -l'
```

Unit and end-to-end coverage lives in `docker/llama-swap-halogen/tests/test_ciru_model_proxy.py`; run it with `python -m unittest docker/llama-swap-halogen/tests/test_ciru_model_proxy.py`. The end-to-end cases drive the real handler against a fake engine, so the framing rules stay enforced without a GPU.

## Update policy

Pin and validate any production image digest after a successful soak test. Test llama.cpp fork updates with the PP512, PP2048, TG64, 64K-context, retrieval, OCR image request, review-verdict, and restart measurements before replacing the current image.

The OCR `requirements.txt` intentionally uses constrained version ranges until one complete target-host build resolves. After that first green build, freeze the resolved Python package versions, keep the lockfile or image digest with the corpus results, and rerun the calibration suite after any MinerU, PaddleOCR, Surya, Tesseract, or Consensus Entropy update.

## PostgreSQL and OCR retention

PostgreSQL starts by default. It is the durable OCR audit ledger and may also store RAG vectors/metadata; it does not participate in model routing. Original sources and rendered pages reside in the private `ocr_audit_data` Docker volume, while PostgreSQL stores document metadata and page audit JSON. The default durable endpoint limits sources to 24 MiB, PDFs to 100 pages, rendered pages to 40 million pixels each, and total retained artifacts to 2 GiB (`OCR_MAX_*` and `OCR_AUDIT_MAX_BYTES`). On every document submission, the ensemble removes records and artifact directories whose expiry has passed, then reconciles UUID artifact directories against live ledger IDs to clean any crash-orphaned data. `OCR_AUDIT_RETENTION_DAYS` defaults to 90; change it only with an explicit retention-policy decision.

Back up both PostgreSQL and the `ocr_audit_data` volume together if audit traceability matters. The browser workspace and unauthenticated Caddy listener expose retained source documents to anyone with network access to the stack; restrict that access before uploading sensitive files.

### Credential drift and repair

The Postgres image applies `POSTGRES_PASSWORD_FILE` only while it initializes an empty data directory. Replacing, restoring, or reissuing the secret after that first init never reaches the stored verifier, so the file and the volume silently disagree. `pg_isready` cannot see this—it authenticates nothing—so Compose keeps reporting the database healthy while `ocr-ensemble` crash-loops at startup with `FATAL: password authentication failed for user "inference"` and the playground's every request 499s behind it.

Make the live role agree with the file instead of recreating the volume, which would destroy the ledger:

```bash
./scripts/sync-postgres-secret.sh --dry-run   # report drift only, change nothing
./scripts/sync-postgres-secret.sh            # reconcile in place and converge ocr-ensemble
```

The script needs no prior credential because the image leaves `local all all trust` inside its own container, and it verifies through the container's own network address so it exercises the same `scram-sha-256` path the ensemble uses rather than the trusted loopback path. `scripts/generate-secrets.sh --force` stays the rotation path for as long as the current credential file still authenticates. If that file is missing, use the non-destructive bootstrap instead:

```bash
./scripts/generate-secrets.sh --recover
```

Recovery creates a replacement under `secrets/`, starts only postgres with a temporary Compose environment containing the real GPU group IDs and a runtime interpolation placeholder, reconciles the role, and then updates the real `.env`. It neither persists the placeholder nor removes `postgres_data`; run `scripts/download-models.sh` if the runtime is still absent, then run `docker compose up -d`. Do not use `docker compose down -v` unless the ledger is deliberately disposable.
