#!/usr/bin/env bash
# Author: Perijn
# Summary: Validates deployment configuration, required model artifacts, and Ciru runtime prerequisites.
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
for model_dir in \
  "$MODELS_DIR/paddle/PP-OCRv5_mobile_det" \
  "$MODELS_DIR/paddle/PP-OCRv5_mobile_rec"; do
  test -d "$model_dir" || {
    printf 'Missing provisioned PP-OCRv5 model directory: %s\n' "$model_dir"
    exit 1
  }
done

# The vendor launcher does not implement --dry-run; syntax-check it instead of
# accidentally starting an inference server during preflight validation.
bash -n "$ORNITH_MODEL_DIR/bundle/serve.sh"
bash -n "$ORNITH_MODEL_DIR/bundle/packaging/serve.sh"

docker compose --profile rag config >/dev/null
printf '%s\n' 'Compose configuration and offline model prerequisites are valid.'
