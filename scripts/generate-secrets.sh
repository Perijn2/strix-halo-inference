#!/usr/bin/env bash
# Author: Perijn
# Summary: Generates local Caddy and PostgreSQL credentials and writes their Compose configuration.
# Usage: Create `.env` from `.env.example`, then run this script once; use `--force` only to rotate both credentials.
# The script writes clear-text secrets to the ignored `secrets/` directory, updates .env with a bcrypt hash and password-file path, and never prints either password.
set -euo pipefail

force=false
if [[ "${1:-}" == "--force" ]]; then
  force=true
elif [[ $# -ne 0 ]]; then
  printf '%s\n' 'Usage: scripts/generate-secrets.sh [--force]' >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$root/.env"
secrets_dir="$root/secrets"
caddy_password_file="$secrets_dir/caddy_password"
postgres_password_file="$secrets_dir/postgres_password"

[[ -f "$env_file" ]] || {
  printf '%s\n' 'Missing .env; copy .env.example to .env first.' >&2
  exit 1
}
command -v openssl >/dev/null || {
  printf '%s\n' 'openssl is required to generate credentials.' >&2
  exit 1
}
command -v docker >/dev/null || {
  printf '%s\n' 'docker is required to generate the Caddy password hash.' >&2
  exit 1
}

if [[ "$force" == false && ( -e "$caddy_password_file" || -e "$postgres_password_file" ) ]]; then
  printf '%s\n' 'Credential files already exist; use --force to rotate both credentials.' >&2
  exit 1
fi

umask 077
mkdir -p "$secrets_dir"
caddy_password="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48)"
postgres_password="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48)"
printf '%s' "$caddy_password" > "$caddy_password_file"
printf '%s' "$postgres_password" > "$postgres_password_file"
caddy_hash="$(docker run --rm caddy:2-alpine caddy hash-password --plaintext "$caddy_password")"

temporary_env="$(mktemp "$root/.env.XXXXXX")"
awk -v hash="$caddy_hash" -v postgres_file="$postgres_password_file" '
  /^CADDY_API_PASSWORD_HASH=/ {
    print "CADDY_API_PASSWORD_HASH=" hash
    found_hash = 1
    next
  }
  /^POSTGRES_PASSWORD_FILE=/ {
    print "POSTGRES_PASSWORD_FILE=" postgres_file
    found_postgres_file = 1
    next
  }
  { print }
  END {
    if (!found_hash) print "CADDY_API_PASSWORD_HASH=" hash
    if (!found_postgres_file) print "POSTGRES_PASSWORD_FILE=" postgres_file
  }
' "$env_file" > "$temporary_env"
mv "$temporary_env" "$env_file"
chmod 600 "$env_file" "$caddy_password_file" "$postgres_password_file"
printf '%s\n' 'Generated local Caddy and PostgreSQL credentials in the ignored secrets directory.'
printf '%s\n' 'Read secrets/caddy_password locally when configuring an API client.'
