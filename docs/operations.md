<!--
Author: Perijn
Summary: Describes safe operation, profiling, rollback, and lifecycle handling for the inference stack.

Usage:
Core principle: Preserve Qwen, Ciru Ornith, and retrieval availability within the measured host-memory budget; the configured limits, not the model names, are the real resource boundary.

Setup: Configure `.env`, download verified model artifacts with `scripts/download-models.sh`, run `scripts/validate.sh`, build Compose, and confirm all service health endpoints.

Workflow: Check Caddy, llama-swap, and PostgreSQL health. Send every model request to a stable role ID and let llama-swap own the loading.

API guide: `/health` verifies server readiness, `/v1/models` lists callable role IDs, and `/v1/chat/completions`, `/v1/embeddings`, and `/v1/rerank` are all routed by role.

Worked example: Start the stack, wait for the first model load, send a `role/general` chat request, and confirm its measured timings reach `GET /api/metrics/activity`.
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


## Model lifecycle

No llama-swap profiles, groups, or swap matrix restrict the configured Qwen, Ciru Ornith, and retrieval processes; each starts on its first request and then remains loaded because its TTL is zero. Memory limits—not profile names—remain the practical resource boundary. `role/implementer`, `role/tester`, and `role/documenter` use the Ciru runtime directly. You can explicitly unload Qwen through `POST /api/models/unload/qwen3.8-flash-next` or Ciru Ornith through `POST /api/models/unload/ciru-ornith-1.5-halo-agent`; the next matching role request reloads it. Do not rely on an idle timeout.

### Cold-start health-check budget

llama-swap kills a child whose `checkEndpoint` never turns ready within `healthCheckTimeout` (default 120 s, minimum 15 s). Ciru Ornith's first load exceeds that window: its pinned `torch.compile`, a ~53 s profiling/warmup run, and CUDA-graph capture together take several minutes before `/health` reports ready. The signature is the engine log halting mid `Capturing CUDA graphs (decode, FULL)` with no Python traceback while the router restarts the child in a loop; a roughly fixed ~2-minute time-of-death means the startup timeout fired, not an out-of-memory kill. The Ciru model therefore sets `healthCheckTimeout: 900`. Once the AOT compile and warm caches exist in the `ornith_cache` volume reloads are much faster, but keep the extended budget so a cold cache after a fresh volume, a runtime/kernel update, or `docker compose down -v` still has room to finish. If you change the capture or warmup profile, keep this comfortably above the measured cold-load wall-clock time.

### Profile commands run without a shell

llama-swap shlex-splits a profile `cmd`, discards `#` lines, and execs `argv[0]` itself; it never spawns a shell. A leading `exec` is therefore looked up as a program literally named `exec` and the child dies with `exec: "exec": executable file not found in $PATH`. The same applies to every other shell construct: `&&`, `||`, pipes, globs, and `$VAR` expansions are arguments rather than syntax. Only llama-swap's own substitution (`${PORT}`, macros) is applied. Write each profile command as one executable plus plain arguments; comment lines remain safe because they are stripped before splitting, which is why the Ciru `serve.sh` profile's comments do not break it.

### Ciru model-name normalization

vLLM validates the request body's `model` field against the single name its launch arguments advertise, and the Ciru release advertises `ciru-halo-agent` only. llama-swap resolves aliases for routing but forwards the body unchanged, so every other client-supplied ID is rejected by the engine with `The model ... does not exist` (404) before inference starts. Neither place can fix that upstream: the vendor release is mounted read-only at `/ornith` (only `bundle/cache` is writable), and Caddy's `request_body` directive can set a whole body but cannot rewrite one JSON field.

`ciru-model-proxy` closes the gap inside the llama-swap image, mirroring `halogen-telemetry-proxy`. The Ciru profile's `cmd` starts the proxy instead of `serve.sh`; the proxy launches the vendor launcher on the next free loopback port and stays in front of it:

- A `POST` whose JSON body names an accepted ID is rewritten to the served name on the way in, adding `X-Ciru-Model-Normalized: <requested id>` for traceability. Only that field changes: everything else, prompt text included, passes byte-for-byte, and a value quoting a model field inside a string cannot be mistaken for one.
- The reply's `model` field is rewritten back to the ID the caller used, for buffered JSON and for each streamed SSE event, so callers that assert on the response ID keep working.
- Any other name is forwarded unchanged and its 404 surfaces verbatim. Each distinct unmapped name logs one `ciru-model-proxy: unmapped model name ...` line into llama-swap's upstream log stream.
- The proxy recomputes request and response framing instead of copying it, because the rewritten name changes the payload length; reusing the inbound `Content-Length` truncates longer names and hangs shorter ones. Chunked request bodies are refused with 400 rather than silently mis-forwarded.
- The engine runs in its own process group, and `SIGTERM` escalates to `SIGKILL` after `--shutdown-grace` (30 s default) so the `EngineCore` child tree cannot survive as an orphan holding ~90 GiB of the unified pool. The proxy then drains the group, waiting until no member is still alive before it exits, so llama-swap never sees the proxy gone while half-finished engine teardown still holds GPU state.

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

### Ciru PP/TG telemetry

The Ciru release publishes no per-request `serve_api:` ledger, so `ciru-model-proxy` derives llama.cpp-compatible `timings` from the wire clocks of each `/v1/chat/completions` response and attaches them exactly the way `halogen-telemetry-proxy` does: inline in the JSON reply, and as a timing event immediately before the terminal SSE `[DONE]`. Decorated responses carry `X-Ciru-Telemetry: wire`.

- For a stream the windows are exact: request-to-first-chunk is the queued-prefill window and first-chunk-to-last-chunk is the decode window, accurate to loopback overhead.
- A non-streaming reply cannot be split, so both windows report the whole end-to-end request; its `predicted_per_second` therefore understates decode speed conservatively. Use streaming for exact per-token rates.
- Token counts come only from the response's OpenAI `usage` block (`prompt_tokens`, `completion_tokens`, and `prompt_tokens_details.cached_tokens`). A stream must carry a usage event—request it with `stream_options: {"include_usage": true}`. Without real counts the proxy attaches no timings rather than inventing them, so a missing PP/TG value means the client omitted usage, not that the engine stalled.

Confirm after deployment with a streaming request and `GET /api/metrics/activity?model=ciru-ornith-1.5-halo-agent`.

### Unload teardown and the resource tracker warning

Unloading Ciru Ornith—as required before loading it beside Qwen on this host—drives the engine's request-abort shutdown, and the pinned vLLM hands Python 3.14's `multiprocessing.resource_tracker` one short-lived `/mp-…` semaphore that the tracker unlinks itself while it exits. The `resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown` UserWarning is that self-cleanup notice, not a fault: the semaphore is reclaimed and no leak persists. The proxy silences exactly that category by setting `PYTHONWARNINGS=ignore:resource_tracker:UserWarning` in the engine environment (all other warnings still reach llama-swap's log stream) and drains the process group so the tracker finishes its unlink before the proxy exits and llama-swap continues the swap.

## Update policy

Pin and validate any production image digest after a successful soak test. Test llama.cpp fork updates with the PP512, PP2048, TG64, 64K-context, retrieval, and restart measurements before replacing the current image.


## Credential drift and repair

The Postgres image applies `POSTGRES_PASSWORD_FILE` only while it initializes an empty data directory. Replacing, restoring, or reissuing the secret after that first init never reaches the stored verifier, so the file and the volume silently disagree. `pg_isready` cannot see this—it authenticates nothing—so Compose keeps reporting the database healthy while every real client fails at startup with `FATAL: password authentication failed for user "inference"`.

Make the live role agree with the file instead of recreating the volume, which would destroy the ledger:

```bash
./scripts/sync-postgres-secret.sh --dry-run   # report drift only, change nothing
./scripts/sync-postgres-secret.sh            # reconcile in place
```

The script needs no prior credential because the image leaves `local all all trust` inside its own container, and it verifies through the container's own network address so it exercises the same `scram-sha-256` path every client uses rather than the trusted loopback path. `scripts/generate-secrets.sh --force` stays the rotation path for as long as the current credential file still authenticates. If that file is missing, use the non-destructive bootstrap instead:

```bash
./scripts/generate-secrets.sh --recover
```

Recovery creates a replacement under `secrets/`, starts only postgres with a temporary Compose environment containing the real GPU group IDs and a runtime interpolation placeholder, reconciles the role, and then updates the real `.env`. It neither persists the placeholder nor removes `postgres_data`; run `scripts/download-models.sh` if the runtime is still absent, then run `docker compose up -d`. Do not use `docker compose down -v` unless the ledger is deliberately disposable.
