#!/usr/bin/env bash
# Author: Perijn
# Summary: Validates the Compose topology and checks its required deployment configuration.
# Usage: Create `.env`, run `scripts/generate-secrets.sh`, then run this script before deployment.
# The command reads the PostgreSQL password-file setting without evaluating .env, so Compose-escaped bcrypt hashes remain safe.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

test -f .env || { printf '%s\n' 'Missing .env; copy .env.example and configure it.'; exit 1; }
postgres_password_file="$(awk -F= '/^POSTGRES_PASSWORD_FILE=/ { print substr($0, index($0, "=") + 1); exit }' .env)"
[[ -n "$postgres_password_file" ]] || { printf '%s\n' 'POSTGRES_PASSWORD_FILE is required in .env.'; exit 1; }
test -r "$postgres_password_file" || { printf 'Cannot read POSTGRES_PASSWORD_FILE: %s\n' "$postgres_password_file"; exit 1; }

docker compose config -q
printf '%s\n' 'Compose configuration is valid.'
