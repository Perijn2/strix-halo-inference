#!/usr/bin/env bash
# Author: Perijn
# Summary: Generates, rotates, or recovers the PostgreSQL credential and records host GPU group IDs.
# Usage: Run with no flag for first initialization, --force to rotate a known live credential, or --recover to replace a missing credential without discarding postgres_data.
# The script never replaces an existing password file until PostgreSQL has accepted the new credential.
# Recovery creates a temporary Compose environment with real GPU groups and a harmless
# runtime placeholder, so it can start and repair only postgres even when a partial
# .env has not yet been completed by scripts/download-models.sh.
set -euo pipefail

force=false
recover=false
case "${1:-}" in
  --force) force=true ;;
  --recover) recover=true ;;
  '') ;;
  *)
    printf '%s\n' 'Usage: scripts/generate-secrets.sh [--force|--recover]' >&2
    exit 2
    ;;
esac

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

new_password() {
  "$(command -v openssl)" rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-48
}

rewrite_env() {
  local destination="${1:?pass destination path}"
  local secret_path="${2:?pass PostgreSQL secret path}"
  local runtime_override="${3:-}"

  awk -v postgres_file="$secret_path" -v render_gid="$render_gid" \
    -v video_gid="$video_gid" -v runtime_override="$runtime_override" '
    /^POSTGRES_PASSWORD_FILE=/ { print "POSTGRES_PASSWORD_FILE=" postgres_file; found_postgres_file = 1; next }
    /^RENDER_GID=/ { print "RENDER_GID=" render_gid; found_render_gid = 1; next }
    /^VIDEO_GID=/ { print "VIDEO_GID=" video_gid; found_video_gid = 1; next }
    runtime_override != "" && /^ORNITH_RUNTIME_PYTHON_ROOT=/ {
      print "ORNITH_RUNTIME_PYTHON_ROOT=" runtime_override
      found_runtime = 1
      next
    }
    { print }
    END {
      if (!found_postgres_file) print "POSTGRES_PASSWORD_FILE=" postgres_file
      if (!found_render_gid) print "RENDER_GID=" render_gid
      if (!found_video_gid) print "VIDEO_GID=" video_gid
      if (runtime_override != "" && !found_runtime) print "ORNITH_RUNTIME_PYTHON_ROOT=" runtime_override
    }
  ' "$env_file" > "$destination"
}

create_recovery_env() {
  local secret_path="${1:?pass temporary secret path}"
  local runtime_placeholder="${ORNITH_RUNTIME_PYTHON_ROOT:-/tmp}"
  local recovery_env

  recovery_env="$(mktemp "$root/.postgres-recovery.env.XXXXXX")"
  rewrite_env "$recovery_env" "$secret_path" "$runtime_placeholder"
  printf '%s' "$recovery_env"
}

recover_missing_credential() {
  local password temporary_password recovery_env

  command -v docker >/dev/null || {
    printf '%s\n' '--recover needs Docker Compose to start and repair postgres.' >&2
    exit 1
  }

  password="$(new_password)"
  temporary_password="$(mktemp "$secrets_dir/.postgres_password.XXXXXX")"
  printf '%s' "$password" > "$temporary_password"
  recovery_env="$(create_recovery_env "$temporary_password")"

  if ! docker compose --env-file "$recovery_env" -f "$root/compose.yaml" up -d postgres; then
    rm -f "$temporary_password" "$recovery_env"
    printf '%s\n' '--recover could not start postgres with the temporary recovery environment.' >&2
    exit 1
  fi

  # The temporary environment makes the just-created secret visible to the
  # reconciler without persisting a /tmp runtime fallback into the real .env.
  if ! ENV_FILE="$recovery_env" "$root/scripts/sync-postgres-secret.sh" --no-restart; then
    rm -f "$temporary_password" "$recovery_env"
    printf '%s\n' '--recover left the real .env and password file unchanged because PostgreSQL rejected the reconciliation.' >&2
    exit 1
  fi

  mv -f "$temporary_password" "$postgres_password_file"
  rm -f "$recovery_env"
}

if [[ "$recover" == true ]]; then
  recover_missing_credential
elif [[ "$force" == true && -e "$postgres_password_file" ]]; then
  command -v docker >/dev/null || {
    printf '%s\n' '--force needs Docker Compose and a running initialized postgres service.' >&2
    exit 1
  }

  old_password="$(<"$postgres_password_file")"
  postgres_user="${POSTGRES_USER:-inference}"
  postgres_db="${POSTGRES_DB:-inference}"
  password="$(new_password)"
  temporary_password="$(mktemp "$secrets_dir/.postgres_password.XXXXXX")"
  printf '%s' "$password" > "$temporary_password"
  compose_env="$(create_recovery_env "$postgres_password_file")"

  if ! docker compose --env-file "$compose_env" -f "$root/compose.yaml" ps -q postgres | grep -q .; then
    rm -f "$temporary_password" "$compose_env"
    printf '%s\n' '--force refused: start the initialized postgres service before rotating its credential.' >&2
    exit 1
  fi
  if ! docker compose --env-file "$compose_env" -f "$root/compose.yaml" exec -T \
    -e PGPASSWORD="$old_password" postgres \
    psql -v ON_ERROR_STOP=1 -U "$postgres_user" -d "$postgres_db" \
    -v role="$postgres_user" -v new_password="$password" \
    -c "ALTER ROLE :\"role\" PASSWORD :'new_password';"; then
    rm -f "$temporary_password" "$compose_env"
    printf '%s\n' '--force did not change the local secret because PostgreSQL rejected the rotation.' >&2
    exit 1
  fi
  mv -f "$temporary_password" "$postgres_password_file"
  rm -f "$compose_env"
elif [[ "$force" == true ]]; then
  printf '%s\n' '--force requires the existing password file. Use --recover when that file is missing.' >&2
  exit 1
elif [[ ! -e "$postgres_password_file" ]]; then
  # Minting a credential that an already-initialized data directory never adopted
  # is exactly how a secret file drifts away from the stored verifier. Refuse it
  # here rather than leave the operator a healthy-looking pg_isready over a database
  # that rejects every real client.
  if command -v docker >/dev/null 2>&1 &&
    {
      docker compose -f "$root/compose.yaml" ps -a -q postgres 2>/dev/null | grep -q . ||
        docker volume ls --filter name=postgres_data -q 2>/dev/null | grep -q .
    }; then
    printf '%s\n' 'Refusing to mint a new credential: a postgres container or postgres_data volume already exists here, and its stored verifier is unknown to the missing secret file.' >&2
    printf '%s\n' 'Run scripts/generate-secrets.sh --recover to create a new credential and reconcile it without discarding the audit ledger.' >&2
    exit 1
  fi
  password="$(new_password)"
  printf '%s' "$password" > "$postgres_password_file"
fi

temporary_env="$(mktemp "$root/.env.XXXXXX")"
rewrite_env "$temporary_env" "$postgres_password_file"
mv "$temporary_env" "$env_file"
chmod 600 "$env_file" "$postgres_password_file"
printf '%s\n' 'Configured the PostgreSQL credential and numeric render/video group IDs.'
