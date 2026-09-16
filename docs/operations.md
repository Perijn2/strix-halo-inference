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

Halogen uses two 131K-context slots and a 262K shared KV pool. `HALOGEN_HOST_RESERVE_GIB` is intentionally set to 28 GiB, leaving headroom for the persistent embedding and reranker workloads. Do not raise slots, context, or pool size without a fresh load and concurrency benchmark. At the 0.8.1 pin the prompt cache holds the pre-0.11 default of 8 entries; the 0.11 line moves that default to 12, roughly 0.44 GiB more host RAM with every entry holding its pool positions while it lives, so pin `HALOGEN_CACHE_ENTRIES` if this budget must hold exactly through the Step B bump.

Qwen and Ciru Ornith start lazily on their first matching request, with no routing profiles, groups, or swap matrix. Each has `ttl: 0`, so a successfully loaded model remains resident until explicitly unloaded. Halogen and the Ciru runtime can take minutes to load. The vendor-required Ornith IU4 profile uses a 262K maximum model length, eight sequences, and a 44 GiB shared KV/state pool; it targets 80% of the 96 GiB GPU memory. Unload Qwen and other GPU workloads before loading Ornith on this host. Its published deployment measured a 95.35 GiB whole-host peak; do not run its full profile beside other large GPU models.

## Halogen PP/TG telemetry

Halogen publishes llama.cpp-compatible `timings` itself since 0.7.0: every response carries `prompt_n`, `predicted_n`, `prompt_ms`, `predicted_ms`, `prompt_per_second`, `predicted_per_second`, `cache_n`, `draft_n`, and `draft_n_accepted`, inline in a non-streaming body and on the final frames of a stream. With the base image pinned at 0.8.1, `halogen-launch` execs the Halogen API directly behind llama-swap with nothing in the request path, and the router reads those fields into the prompt-processing (PP) and token-generation (TG) rates and the draft count shown in Activity.

Do not put a timing sidecar back on this path. The previous `halogen-telemetry-proxy` replaced the whole `timings` object with its own, so it discarded `draft_n` and `draft_n_accepted`; it reported `prompt_n` as the entire prompt while deriving the rate from the uncached tokens, which is the semantics upstream corrected in 0.8.0; and it serialized every chat request behind one lock with a two-second ledger wait, capping the two configured KV slots at a single in-flight chat. The version bump is only a win if that wrapper comes out with it.

The `serve_api:` ledger still prints in the Halogen log and remains how you read a slow turn directly, but nothing parses it now. `HALOGEN_TIMING_PROXY=1` restores the old wrapper without a rebuild; it exists only as the Step A rollback lever and is removed at Step B.

Halogen also answers `GET /metrics` in llama-server metric names since 0.8.0 (`llamacpp:prompt_tokens_total` and its siblings, plus the `halogen:` counters), on the loopback port llama-swap passed as `HALOGEN_API_PORT`; that is reachable inside the container, not through Caddy.

Confirm the routed path with a non-streaming `role/general` chat request and `GET /api/metrics/activity?model=qwen3.8-flash-next`: PP and TG must appear without any sidecar between the router and Halogen.

### Halogen version ladder

The base image moves in two measured steps rather than one, because seventeen releases jumped at once cannot be attributed on a host where a single cold load costs minutes.

**Step A — current pin, 0.8.1.** The wire fixes and the native timings: keep-alive raised from 5 s to 300 s (0.5.7); the 10-second SSE keepalive that stops a Node/undici client hanging at 300 s, and the accepted `developer` role (0.6.1); a conversation that merely mentions the image placeholder served as text instead of poisoning every later turn (0.6.2); the 64-deep lookup-table read that takes a cold 32k prompt from 46–52 s over its usual time down to about 1.3 s (0.6.3); native response `timings` (0.7.0); `GET /metrics` and enforced JSON schema (0.8.0); and a refused grammar request ending in a 400 instead of dropping the connection (0.8.1). 0.6.0 also adds prompt lookup beside the MTP head — three-token chains verified in a single step, measured at +15% on coding-agent turns with thinking off and +13% with it on, greedy requests only, `HALOGEN_PLD=0` to turn off — and it raises the speculative verify reservation from two rows to four, about 0.2 GiB of device memory that Step A itself adds over the 0.5.6 baseline and that the KV-pool fit accounts for. Verify the pin with `docker compose logs llama-swap | grep 'halogen-flash-server'` — the first line of every mode names the release — and `/health` on the Halogen port reports `version.match`.

**Step B — pending, 0.11.1.** The agent-cache and thinking-control work: the two retained conversation-history entries that stop a recap or side turn evicting the point the real turn continues from, region-based KV-pool eviction that never drops the entry a request just matched, and the `kv pool: no room` line that finally explains a cold turn (0.8.1 through 0.11.0); the thinking answer room; the harness thinking-control aliases; and the fix for the engine crash on a text-image-text-image order (0.11.1). Three decisions to make before taking it:

- `HALOGEN_CACHE_ENTRIES` rises from a default of 8 to 12 — roughly 0.44 GiB more host RAM, and every entry holds the pool positions its content covers for as long as it exists, which matters against the 262144-position pool with only two slots. Set it explicitly to keep the present footprint.
- The answer room applies to the roles as configured, since they send `enable_thinking` and a `reasoning_effort` but no budget: thinking closes early to leave `max(1024, 15% of max_tokens)` for the answer. Set `HALOGEN_THINKING_ANSWER_ROOM=0` if any role must think all the way to `max_tokens`.
- The 0.11 pre-flight check warns once more than 10 GiB of host RAM is already in use before the server starts. With the embedding and reranker models resident this fires by design; it resizes nothing.

**Vision stays off until Step B.** Halogen has read images since 0.5.0, but only with `HALOGEN_VISION_TOWER` pointed at the vision sidecar; unset, the server is text-only and an image is a 400 naming the flag. We keep it unset at 0.8.1 on purpose: until 0.10.1, a text request that resumed from the prompt cache on a slot whose previous request carried an image inherited that request's image position table and its new tokens were rotated by it, so the same greedy turn diverged from a fresh run about 70 tokens in. That bug only exists on a server with a tower, and this stack mixes text and image traffic across two slots. To turn it on after the bump: `hf download peonist-ai/halogen-qwen3.8-flash-next qwen38-flash-next-vision.hgn --local-dir "$HALOGEN_MODELS_DIR"` (0.84 GiB; the read-only mount blocks the image's own self-fetch, same as the quality sidecar), set `HALOGEN_VISION_TOWER=1` to look beside the checkpoint, and confirm `/health` reports `vision.enabled`. Budget for it: one uncovered image costs about 5.5, 11.8 or 25.3 seconds at 1280x800, 1920x1080 or 2560x1440, and 4K costs about 105 seconds without reading better than 1440p, which is why `HALOGEN_VISION_MAX_PIXELS` defaults to 2560x1440. Only `data:` URLs or bare base64 are accepted — fetching an `http(s)` image is refused by design — and `json_schema` with an image is a 400 since 0.8.0. The gateway's 32 MB `request_body` cap in `config/caddy/Caddyfile` leaves room for roughly a 24 MB image as base64, so no edge change is needed.

**Required before either step is trusted: refresh the quality sidecar.** 0.6.0 put the MTP draft head's own projections at 8 bits in `qwen38-flash-next-w4b.overlay.hgn`, lifting draft acceptance on prose from 51% to 59% — about 4% of decode at 1,500 tokens of context, within noise on code — and grew the file from 2.31 to 2.40 GiB. The image notices a pre-0.6.0 sidecar and re-fetches it only when `HALOGEN_DOWNLOAD` is set and the models volume is writable; `/halogen-models` is mounted read-only here, so that path cannot run and the entrypoint reports what is missing instead. Refresh it on the host before the soak: `hf download peonist-ai/halogen-qwen3.8-flash-next qwen38-flash-next-w4b.overlay.hgn --local-dir "$HALOGEN_MODELS_DIR"`. Upstream's README file listing still prints 2.31 GiB for that file; the current sidecar is 2.40.

**Also new at 0.8.1.** The entrypoint validates every `HALOGEN_*` value through `serve_api.py --check-defaults` before anything loads, so a typo that 0.5.6 accepted silently now fails the boot explicitly. Upstream also dropped `seccomp:unconfined` at 0.6.1 after measuring the image starting and serving without it (#8), which makes this stack's `security_opt` entry a candidate to remove. Test that on its own rather than bundling it with the bump: upstream measured the Halogen image alone, and the Ciru runtime shares this container.


## Model lifecycle

No llama-swap profiles, groups, or swap matrix restrict the configured Qwen, Ciru Ornith, and retrieval processes; each starts on its first request and then remains loaded because its TTL is zero. Memory limits—not profile names—remain the practical resource boundary. `role/implementer`, `role/tester`, and `role/documenter` use the Ciru runtime directly. You can explicitly unload Qwen through `POST /api/models/unload/qwen3.8-flash-next` or Ciru Ornith through `POST /api/models/unload/ciru-ornith-1.5-halo-agent`; the next matching role request reloads it. Do not rely on an idle timeout.

### Cold-start health-check budget

llama-swap kills a child whose `checkEndpoint` never turns ready within `healthCheckTimeout` (default 120 s, minimum 15 s). Ciru Ornith's first load exceeds that window: its pinned `torch.compile`, a ~53 s profiling/warmup run, and CUDA-graph capture together take several minutes before `/health` reports ready. The signature is the engine log halting mid `Capturing CUDA graphs (decode, FULL)` with no Python traceback while the router restarts the child in a loop; a roughly fixed ~2-minute time-of-death means the startup timeout fired, not an out-of-memory kill. The Ciru model therefore sets `healthCheckTimeout: 900`. Once the AOT compile and warm caches exist in the `ornith_cache` volume reloads are much faster, but keep the extended budget so a cold cache after a fresh volume, a runtime/kernel update, or `docker compose down -v` still has room to finish. If you change the capture or warmup profile, keep this comfortably above the measured cold-load wall-clock time.

### Profile commands run without a shell

llama-swap shlex-splits a profile `cmd`, discards `#` lines, and execs `argv[0]` itself; it never spawns a shell. A leading `exec` is therefore looked up as a program literally named `exec` and the child dies with `exec: "exec": executable file not found in $PATH`. The same applies to every other shell construct: `&&`, `||`, pipes, globs, and `$VAR` expansions are arguments rather than syntax. Only llama-swap's own substitution (`${PORT}`, macros) is applied. Write each profile command as one executable plus plain arguments; comment lines remain safe because they are stripped before splitting, which is why the Ciru `serve.sh` profile's comments do not break it.

### Ciru model-name normalization

vLLM validates the request body's `model` field against the single name its launch arguments advertise, and the Ciru release advertises `ciru-halo-agent` only. llama-swap resolves aliases for routing but forwards the body unchanged, so every other client-supplied ID is rejected by the engine with `The model ... does not exist` (404) before inference starts. Neither place can fix that upstream: the vendor release is mounted read-only at `/ornith` (only `bundle/cache` is writable), and Caddy's `request_body` directive can set a whole body but cannot rewrite one JSON field.

`ciru-model-proxy` closes the gap inside the llama-swap image. It is a sidecar in the same shape as the pre-0.7.0 `halogen-telemetry-proxy`, and unlike Halogen it has nothing newer to fall back on: the Ciru runtime publishes neither response timings nor a version handshake, so this wrapper stays in the request path. The Ciru profile's `cmd` starts the proxy instead of `serve.sh`; the proxy launches the vendor launcher on the next free loopback port and stays in front of it:

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

The Ciru release publishes no per-request `serve_api:` ledger, so `ciru-model-proxy` derives llama.cpp-compatible `timings` from the wire clocks of each `/v1/chat/completions` response and attaches them in the same llama.cpp shape Halogen has emitted natively since 0.7.0: inline in the JSON reply, and as a timing event immediately before the terminal SSE `[DONE]`. Decorated responses carry `X-Ciru-Telemetry: wire`.

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
