#!/usr/bin/env bash
# Author: Perijn
# Summary: Starts the original Halogen entrypoint as a llama-swap managed child process.
# Usage:
#   Core principle: Halogen 0.7.0 and newer publish llama.cpp-shaped ``timings`` on
#   every response itself, so the Halogen API runs directly behind llama-swap with
#   nothing in the request path. This launcher only locates the entrypoint and hands
#   it the port llama-swap manages.
#   Setup: llama-swap sets HALOGEN_API_PORT to its assigned ${PORT} and invokes
#   ``halogen-launch all``; the entrypoint starts Halogen's engine and OpenAI API as
#   one supervised process tree bound to that port.
#   Workflow: Resolve the entrypoint, then exec it in place so llama-swap signals and
#   reaps the Halogen tree directly. Any other mode is forwarded verbatim.
#   API guide: The started server answers the OpenAI API and ``/health`` on
#   127.0.0.1:${HALOGEN_API_PORT}; the response body carries the ``timings``
#   llama-swap records as PP and TG in Activity.
#   Worked example: ``HALOGEN_API_PORT=33841 halogen-launch all`` serves the
#   combined engine and API on 127.0.0.1:33841 for the router to health-check.
#   Fallback: ``HALOGEN_TIMING_PROXY=1`` restores the pre-0.7.0 arrangement, where
#   ``halogen-telemetry-proxy`` sits in front and synthesises timings from Halogen's
#   ``serve_api:`` log ledger. Keep it only as the Step A rollback lever while the
#   0.8.1 soak runs; it serialises chat requests, so remove it at Step B.
# The launcher accepts the published image layout and the source-tree layout to make the packaged image upgrade failure explicit.
set -euo pipefail

for entrypoint in /usr/local/bin/entrypoint.sh /entrypoint.sh /halogen/deploy/entrypoint.sh; do
  if [[ -x "$entrypoint" ]]; then
    if [[ "${1:-all}" == "all" ]]; then
      : "${HALOGEN_API_PORT:?HALOGEN_API_PORT must be set by llama-swap}"
      if [[ "${HALOGEN_TIMING_PROXY:-0}" == "1" ]]; then
        exec halogen-telemetry-proxy --entrypoint "$entrypoint" --port "$HALOGEN_API_PORT"
      fi
      exec "$entrypoint" all
    fi
    exec "$entrypoint" "$@"
  fi
done

printf '%s\n' 'Halogen entrypoint was not found in the image.' >&2
exit 127
