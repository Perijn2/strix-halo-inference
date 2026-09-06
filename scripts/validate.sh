#!/usr/bin/env bash
# Author: Perijn
# Summary: Validates the Compose topology and checks its required deployment configuration.
# Usage: Create .env and its referenced PostgreSQL password file, then run this script before deployment.
# The command expands every Compose profile to detect invalid interpolation and schema errors without starting services.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

test -f .env || { printf '%s\n' 'Missing .env; copy .env.example and configure it.'; exit 1; }
# shellcheck disable=SC1091
source .env
: "${POSTGRES_PASSWORD_FILE:?POSTGRES_PASSWORD_FILE is required}"
test -r "$POSTGRES_PASSWORD_FILE" || { printf 'Cannot read POSTGRES_PASSWORD_FILE: %s\n' "$POSTGRES_PASSWORD_FILE"; exit 1; }

docker compose --profile rag config >/dev/null
printf '%s\n' 'Compose configuration is valid.'
