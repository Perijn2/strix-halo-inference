<!--
Author: Perijn
Summary: Describes safe operation, profiling, and rollback for the inference stack.

Usage:
Core principle: preserve Qwen and retrieval availability in orchestration mode; change modes deliberately before sending implementation work to Ornith.

Setup: Configure `.env`, download verified model artifacts, run `scripts/validate.sh`, and start Compose.

Workflow: Check Caddy, llama-swap, and Halogen health. Keep the orchestration profile active for Qwen/RAG work. Select engineering only for Ornith work, then return to orchestration after it completes. Investigate logs and health before restarting a model service.

API guide: `/health` verifies individual server readiness, `/v1/models` lists callable role IDs, and `/api/profiles` reports or changes llama-swap routing state.

Worked example: Start the default stack, query `/v1/models` through authenticated Caddy, switch the active profile to `engineering`, submit `role/implementer`, then restore `orchestration`.
-->

# Operations

## Memory policy

Halogen uses two 131K-context slots and a 262K shared KV pool. `HALOGEN_HOST_RESERVE_GIB` is intentionally set to 28 GiB, leaving headroom for the persistent embedding and reranker workloads. Do not raise slots, context, or pool size without a fresh load and concurrency benchmark.

The default mode is persistent: Qwen has no inactivity-based unload. Halogen’s startup can take minutes because it loads a large checkpoint; its Compose health check allows twenty minutes before counting failures.

## Switching modes

The active llama-swap profile is an API routing policy, not a background scheduler. Select `engineering` before implementation work. If Ornith cannot coexist with Qwen on the measured host, stop Halogen deliberately before starting large Ornith sessions; do not rely on an idle timeout.

## Update policy

Pin and validate any production image digest after a successful soak test. Test llama.cpp fork updates with the PP512, PP2048, TG64, 64K-context, retrieval, and restart measurements before replacing the current image.

## PostgreSQL

PostgreSQL is optional and starts only with the `rag` profile. It stores durable vectors and metadata; it does not participate in model routing. Back up PostgreSQL with the normal PostgreSQL backup tooling before upgrades.
