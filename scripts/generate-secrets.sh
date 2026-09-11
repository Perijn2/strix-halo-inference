#!/usr/bin/env bash
# Author: Perijn
# Summary: Generates the PostgreSQL credential, safely rotates a live database credential, and records host GPU group IDs.
# Usage: Create .env from .env.example and run this script; use --force only while the initialized postgres service is running.
# The script never replaces an existing password file until PostgreSQL has accepted the new credential.
set -euo pipefail

force=false
if [[ "${1:-}" == "--force" ]]; then
  force=true
elif [[ $# -ne 0 ]]; then
  printf '%s\n' 'Usage: scripts/generate-secrets.sh [--force]' >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/load-env.sh
source "$root/scripts/lib/load-env.sh"
env_file="$root/.env"
secrets_dir="$root/secrets"
postgres_password_file="$secrets_dir/postgres_password"
load_deployment_env "$env_file"

command -v openssl >/dev/null || {
  printf '%s\n' 'openssl is required to generate the PostgreSQL credential.' >&2
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
if [[ "$force" == true && -e "$postgres_password_file" ]]; then
  command -v docker >/dev/null || {
    printf '%s\n' '--force needs Docker Compose and a running initialized postgres service.' >&2
    exit 1
  }
  old_password="$(<"$postgres_password_file")"
  postgres_user="${POSTGRES_USER:-inference}"
  postgres_db="${POSTGRES_DB:-inference}"
  new_password="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48)"
  temporary_password="$(mktemp "$secrets_dir/.postgres_password.XXXXXX")"
  printf '%s' "$new_password" > "$temporary_password"
  if ! docker compose -f "$root/compose.yaml" ps -q postgres | grep -q .; then
    rm -f "$temporary_password"
    printf '%s\n' '--force refused: start the initialized postgres service before rotating its credential.' >&2
    exit 1
  fi
  if ! docker compose -f "$root/compose.yaml" exec -T \
    -e PGPASSWORD="$old_password" postgres \
    psql -v ON_ERROR_STOP=1 -U "$postgres_user" -d "$postgres_db" \
    -v role="$postgres_user" -v new_password="$new_password" \
    -c "ALTER ROLE :\"role\" PASSWORD :'new_password';"; then
    rm -f "$temporary_password"
    printf '%s\n' '--force did not change the local secret because PostgreSQL rejected the rotation.' >&2
    exit 1
  fi
  mv -f "$temporary_password" "$postgres_password_file"
elif [[ ! -e "$postgres_password_file" ]]; then
  postgres_password="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48)"
  printf '%s' "$postgres_password" > "$postgres_password_file"
fi

temporary_env="$(mktemp "$root/.env.XXXXXX")"
awk -v postgres_file="$postgres_password_file" -v render_gid="$render_gid" -v video_gid="$video_gid" '
  /^POSTGRES_PASSWORD_FILE=/ { print "POSTGRES_PASSWORD_FILE=" postgres_file; found_postgres_file = 1; next }
  /^RENDER_GID=/ { print "RENDER_GID=" render_gid; found_render_gid = 1; next }
  /^VIDEO_GID=/ { print "VIDEO_GID=" video_gid; found_video_gid = 1; next }
  { print }
  END {
    if (!found_postgres_file) print "POSTGRES_PASSWORD_FILE=" postgres_file
    if (!found_render_gid) print "RENDER_GID=" render_gid
    if (!found_video_gid) print "VIDEO_GID=" video_gid
  }
' "$env_file" > "$temporary_env"
mv "$temporary_env" "$env_file"
chmod 600 "$env_file" "$postgres_password_file"
printf '%s\n' 'Configured the PostgreSQL credential and numeric render/video group IDs.'
