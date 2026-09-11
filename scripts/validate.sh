#!/usr/bin/env bash
# Author: Perijn
# Summary: Validates deployment configuration, required model artifacts, and the exact Ciru launch command.
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
test -x "$ORNITH_MODEL_DIR/bundle/serve.sh" || {
  printf 'Missing Ciru launcher: %s/bundle/serve.sh\n' "$ORNITH_MODEL_DIR"
  exit 1
}
test -x "$ORNITH_MODEL_DIR/installed-runtime/venv/bin/python" || {
  printf 'Missing pinned Ciru runtime: run runtime/INSTALL-ORNITH-RUNTIME.sh in %s\n' "$ORNITH_MODEL_DIR"
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

# Validate the same intentional memory-control arguments Compose gives Ciru.
ORNITH_RUNTIME_ROOT="$ORNITH_MODEL_DIR/installed-runtime" \
  bash "$ORNITH_MODEL_DIR/bundle/serve.sh" --host 127.0.0.1 --port 8080 \
  --context 131072 --max-seqs 6 --dry-run

docker compose --profile rag config >/dev/null
printf '%s\n' 'Compose configuration and offline model prerequisites are valid.'
