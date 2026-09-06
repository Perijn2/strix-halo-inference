#!/usr/bin/env bash
# Author: Perijn
# Summary: Starts the original Halogen entrypoint as a llama-swap managed child process.
# Usage: llama-swap invokes `halogen-launch all`; the selected entrypoint starts Halogen's engine and OpenAI API as one supervised process tree.
# The launcher accepts the published image layout and the source-tree layout to make the packaged image upgrade failure explicit.
set -euo pipefail

for entrypoint in /usr/local/bin/entrypoint.sh /entrypoint.sh /halogen/deploy/entrypoint.sh; do
  if [[ -x "$entrypoint" ]]; then
    exec "$entrypoint" "$@"
  fi
done

printf '%s\n' 'Halogen entrypoint was not found in the image.' >&2
exit 127
