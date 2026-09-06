#!/usr/bin/env bash
# Author: Perijn
# Summary: Generates local Caddy and PostgreSQL credentials and records host GPU group IDs for Compose.
# Usage: Create `.env` from `.env.example`, then run this script; use `--force` only to rotate both credential files.
# The script stores clear-text secrets in the ignored `secrets/` directory, escapes the Caddy bcrypt hash for Compose, and never prints either password.
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
command -v getent >/dev/null || {
  printf '%s\n' 'getent is required to resolve host GPU group IDs.' >&2
  exit 1
}

render_gid="$(getent group render | awk -F: 'NR == 1 { print $3 }')"
video_gid="$(getent group video | awk -F: 'NR == 1 { print $3 }')"
[[ -n "$render_gid" && -n "$video_gid" ]] || {
  printf '%s\n' 'Host render and video groups are required for /dev/dri access.' >&2
  exit 1
}

umask 077
mkdir -p "$secrets_dir"
if [[ "$force" == true ]]; then
  rm -f "$caddy_password_file" "$postgres_password_file"
fi

if [[ -e "$caddy_password_file" || -e "$postgres_password_file" ]]; then
  [[ -r "$caddy_password_file" && -r "$postgres_password_file" ]] || {
    printf '%s\n' 'Credential files are incomplete; use --force to rotate both credentials.' >&2
    exit 1
  }
  caddy_password="$(<"$caddy_password_file")"
else
  caddy_password="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48)"
  postgres_password="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48)"
  printf '%s' "$caddy_password" > "$caddy_password_file"
  printf '%s' "$postgres_password" > "$postgres_password_file"
fi

caddy_hash="$(docker run --rm caddy:2-alpine caddy hash-password --plaintext "$caddy_password")"
compose_caddy_hash="${caddy_hash//\$/\$\$}"

temporary_env="$(mktemp "$root/.env.XXXXXX")"
awk -v hash="$compose_caddy_hash" -v postgres_file="$postgres_password_file" \
    -v render_gid="$render_gid" -v video_gid="$video_gid" '
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
  /^RENDER_GID=/ {
    print "RENDER_GID=" render_gid
    found_render_gid = 1
    next
  }
  /^VIDEO_GID=/ {
    print "VIDEO_GID=" video_gid
    found_video_gid = 1
    next
  }
  { print }
  END {
    if (!found_hash) print "CADDY_API_PASSWORD_HASH=" hash
    if (!found_postgres_file) print "POSTGRES_PASSWORD_FILE=" postgres_file
    if (!found_render_gid) print "RENDER_GID=" render_gid
    if (!found_video_gid) print "VIDEO_GID=" video_gid
  }
' "$env_file" > "$temporary_env"
mv "$temporary_env" "$env_file"
chmod 600 "$env_file" "$caddy_password_file" "$postgres_password_file"
printf '%s\n' 'Configured local credentials and numeric render/video group IDs.'
printf '%s\n' 'Read secrets/caddy_password locally when configuring an API client.'
