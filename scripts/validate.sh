#!/usr/bin/env bash
# Author: Perijn
# Summary: Validates deployment configuration, the Halogen and Ciru model artifacts, and the Ciru runtime prerequisites.
# Usage: Create .env, run scripts/generate-secrets.sh and scripts/download-models.sh, then run this script before deployment.
# Values are parsed as data rather than evaluated, so .env cannot execute shell code during validation.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
# shellcheck source=lib/load-env.sh
source "$root/scripts/lib/load-env.sh"
load_deployment_env "$root/.env"

: "${POSTGRES_PASSWORD_FILE:?POSTGRES_PASSWORD_FILE is required in .env.}"
: "${ORNITH_MODEL_DIR:?ORNITH_MODEL_DIR is required in .env.}"
: "${MODELS_DIR:?MODELS_DIR is required in .env.}"
test -r "$POSTGRES_PASSWORD_FILE" || {
  printf 'Cannot read POSTGRES_PASSWORD_FILE: %s\n' "$POSTGRES_PASSWORD_FILE"
  exit 1
}
test -f "$ORNITH_MODEL_DIR/bundle/serve.sh" || {
  printf 'Missing Ciru launcher: %s/bundle/serve.sh\n' "$ORNITH_MODEL_DIR"
  exit 1
}
test -x "$ORNITH_MODEL_DIR/installed-runtime/venv/bin/python" || {
  printf 'Missing pinned Ciru runtime Python: run scripts/download-models.sh\n' >&2
  exit 1
}
runtime_vllm_init="$("$ORNITH_MODEL_DIR/installed-runtime/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/vllm/__init__.py"
test -f "$runtime_vllm_init" || {
  printf 'Incomplete Ciru runtime: missing vendor vLLM at %s\n' "$runtime_vllm_init" >&2
  exit 1
}
test -f "$ORNITH_MODEL_DIR/installed-runtime/runtime-env.sh" || {
  printf 'Incomplete Ciru runtime: missing installed-runtime/runtime-env.sh; run scripts/download-models.sh\n' >&2
  exit 1
}
: "${ORNITH_RUNTIME_PYTHON_ROOT:?run scripts/download-models.sh to set ORNITH_RUNTIME_PYTHON_ROOT}"
test -d "$ORNITH_RUNTIME_PYTHON_ROOT" || {
  printf 'Missing Ciru interpreter directory: %s\n' "$ORNITH_RUNTIME_PYTHON_ROOT" >&2
  exit 1
}
expected_python_root="$(dirname "$(dirname "$(readlink -f "$ORNITH_MODEL_DIR/installed-runtime/venv/bin/python")")")"
test "$ORNITH_RUNTIME_PYTHON_ROOT" = "$expected_python_root" || {
  printf 'ORNITH_RUNTIME_PYTHON_ROOT does not match the Ciru venv interpreter\n' >&2
  exit 1
}

# Halogen artifacts. The entrypoint's own checks for the quality sidecar warn and
# carry on, so a stale sidecar is easy to miss here; check it in the gate instead.
: "${HALOGEN_MODELS_DIR:?HALOGEN_MODELS_DIR is required in .env.}"
halogen_ckpt="$HALOGEN_MODELS_DIR/qwen38-flash-next-w4b.hgn"
test -f "$halogen_ckpt" || {
  printf 'Missing Halogen checkpoint: %s\n' "$halogen_ckpt" >&2
  exit 1
}
# The sidecar path is derived from the checkpoint name by the image entrypoint as
# "<checkpoint>.overlay.hgn" beside it. 0.6.0 grew it from 2.31 to 2.40 GiB by
# adding the 8-bit draft-head projections (~4% of decode on prose), so anything
# under 2.35 GiB under this name predates that and is silently costing the gap.
halogen_overlay="${halogen_ckpt%.hgn}.overlay.hgn"
overlay_min=$((2350 * 1073741824 / 1000))
if [[ ! -f "$halogen_overlay" ]]; then
  printf 'WARN: no quality sidecar at %s; the bare checkpoint loses the 8-bit draft-head projections.\n' "$halogen_overlay" >&2
elif (( $(stat -c %s "$halogen_overlay") < overlay_min )); then
  printf 'WARN: %s predates 0.6.0 (no 8-bit draft-head entries). Refresh it with:\n      hf download peonist-ai/halogen-qwen3.8-flash-next %s --local-dir %s\n' \
    "$halogen_overlay" "$(basename "$halogen_overlay")" "$HALOGEN_MODELS_DIR" >&2
fi
# Compose enables the vision tower, and the models mount is read-only, so the
# entrypoint cannot fetch the sidecar itself the way HALOGEN_DOWNLOAD would on a
# writable volume. A missing tower therefore fails here at preflight rather than at
# boot, which is the only place this repo can still catch it.
halogen_vision="$HALOGEN_MODELS_DIR/qwen38-flash-next-vision.hgn"
test -f "$halogen_vision" || {
  printf 'Missing Halogen vision sidecar: %s\n    Compose enables it with HALOGEN_VISION_TOWER=1 but the models mount is read-only,\n    so it has to be on the host already. Fetch it with:\n      hf download peonist-ai/halogen-qwen3.8-flash-next %s --local-dir %s\n    Or set HALOGEN_VISION_TOWER to 0 in compose.yaml to run text-only.\n' \
    "$halogen_vision" "$(basename "$halogen_vision")" "$HALOGEN_MODELS_DIR" >&2
  exit 1
}
# 0.84 GiB delivered. A file far under that is an LFS pointer or a truncated
# download, which the engine would only reject after loading the 115 GiB trunk.
vision_min=$((700 * 1048576))
if (( $(stat -c %s "$halogen_vision") < vision_min )); then
  printf 'WARN: %s is under %s MiB and is probably not the real sidecar. Re-download it with:\n      hf download peonist-ai/halogen-qwen3.8-flash-next %s --local-dir %s\n' \
    "$(basename "$halogen_vision")" 700 "$halogen_vision" "$HALOGEN_MODELS_DIR" >&2
fi

# The disk prompt cache is gated by the filesystem, not by the flag. The engine
# opens the cache with O_DIRECT and refuses a tmpfs or an overlay; after refusing
# it carries on serving from memory only, so the stack starts, answers, and
# reports healthy while the restart-survival feature that was switched on does
# nothing at all. Nothing else in this stack surfaces that, so probe it directly.
#
# The probe runs in this stack's own image against a throwaway volume on the
# default driver. Inside the container because a host user in the docker group
# cannot write to /var/lib/docker/volumes and would see a permission error where
# there is no storage fault, and with a page-aligned mmap buffer because that is
# the shape of open the engine makes -- a plain bytes buffer is not aligned and
# fails with EINVAL on storage that is perfectly capable. The volume is created
# and removed here, so a probe never touches a real conversation cache.
cache_probe_image='strix-halo/llama-swap-halogen:local'
if ! docker image inspect "$cache_probe_image" >/dev/null 2>&1; then
  printf 'NOTE: the direct-I/O probe was skipped because %s is not built yet.\n' "$cache_probe_image" >&2
  printf '      Build it (docker compose build) and rerun before trusting the disk cache.\n' >&2
else
  cache_probe_vol="halogen-direct-io-probe.$$"
  docker volume create "$cache_probe_vol" >/dev/null
  if probe_out="$(docker run --rm --entrypoint python3 -v "$cache_probe_vol:/probe" "$cache_probe_image" -c '
import mmap, os
buf = mmap.mmap(-1, 4096)
fd = os.open("/probe/probe.bin", os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
os.write(fd, buf)
os.close(fd)
os.unlink("/probe/probe.bin")
' 2>&1)"; then
    printf 'Direct I/O is accepted on the cache volume driver.\n'
  else
    printf 'The prompt cache cannot work on this host storage: direct I/O was refused.\n' >&2
    printf '  %s\n' "$probe_out" >&2
    printf '  compose.yaml sets HALOGEN_CACHE_DIR, so without direct I/O the engine keeps the\n' >&2
    printf '  whole cache in memory: a restart re-prefills every conversation and the 64 GiB\n' >&2
    printf '  budget is never touched. The usual causes are a Docker data-root on tmpfs, or a\n' >&2
    printf '  cache path sitting on an overlay mount.\n' >&2
    printf '  Move the Docker data-root onto real storage, or drop HALOGEN_CACHE_DIR to make\n' >&2
    printf '  the memory-only cache a decision rather than an accident.\n' >&2
    docker volume rm "$cache_probe_vol" >/dev/null 2>&1 || true
    exit 1
  fi
  docker volume rm "$cache_probe_vol" >/dev/null 2>&1 || true
fi

# The vendor launcher does not implement --dry-run; syntax-check it instead of
# accidentally starting an inference server during preflight validation.
bash -n "$ORNITH_MODEL_DIR/bundle/serve.sh"
bash -n "$ORNITH_MODEL_DIR/bundle/packaging/serve.sh"

docker compose --profile rag config >/dev/null
printf '%s\n' 'Compose configuration and offline model prerequisites are valid.'
