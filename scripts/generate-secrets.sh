#!/usr/bin/env bash
# Author: Perijn
# Summary: Generates, rotates, or recovers the PostgreSQL credential, records host GPU group IDs, and restores the runtime Python root and file ownership a wiped .env loses.
# Usage: Run with no flag for first initialization, --force to rotate a known live credential, or --recover to replace a missing credential without discarding postgres_data.
# The script never replaces an existing password file until PostgreSQL has accepted the new credential.
# Recovery never depends on the rest of the stack. Compose interpolates every service before it
# selects one, so the repair builds a throwaway environment that satisfies each variable
# compose.yaml requires without a default. A blank or partial .env, a host without the render
# group, or a model runtime that was never downloaded cannot hold the database repair hostage.
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
compose_file="${COMPOSE_FILE:-$root/compose.yaml}"
secrets_dir="$root/secrets"
postgres_password_file="$secrets_dir/postgres_password"
sync_script="$root/scripts/sync-postgres-secret.sh"

# Every temporary artifact lives in one directory that leaves with the process, so a
# failed run cannot strand a half-written .env or a stray credential in the repo.
sandbox="$(mktemp -d "${TMPDIR:-/tmp}/strix-postgres-recovery.XXXXXX")"
trap 'rm -rf "$sandbox"' EXIT

die() {
  printf '%s\n' "$1" >&2
  exit "${2:-1}"
}

# A repair run under sudo must not strand what it creates as root-owned: the
# unprivileged docker compose that follows cannot read a root-owned .env, and the
# reflex repair (chmod 777) then exposes the deployment config to every local
# account. Whenever SUDO_* names the invoking user, hand each path back to them.
hand_to_caller() {
  [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" ]] || return 0
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    chown "$SUDO_UID:$SUDO_GID" "$path" 2>/dev/null || true
  done
}

# A deleted .env is a reason to recover, not a reason to refuse: seed it from the
# committed example so the repair has somewhere to write the new credential back.
if [[ ! -f "$env_file" ]]; then
  [[ -f "$root/.env.example" ]] || die "Neither $env_file nor $root/.env.example exists; there is nothing to recover into."
  cp "$root/.env.example" "$env_file"
  hand_to_caller "$env_file"
  printf '%s\n' "Seeded $env_file from .env.example; review it once the recovery finishes."
fi
load_deployment_env "$env_file"

# Compose refuses to interpolate a file containing an unsatisfied ${VAR:?...}, and it
# does that for the whole file rather than for the service being started. Reading the
# required set from compose.yaml keeps this honest when a service adds a variable;
# a hard-coded list is how the recovery path dead-locked in the first place.
required_compose_vars() {
  awk '{
    line = $0
    while (match(line, /\$\{[A-Za-z_][A-Za-z0-9_]*:\?/)) {
      print substr(line, RSTART + 2, RLENGTH - 4)
      line = substr(line, RSTART + RLENGTH)
    }
  }' "$compose_file" | sort -u
}

# The host GPU groups belong to the services recovery never starts. Resolve them when
# the host can offer them and fall back to whatever .env already carries; a missing
# render group must not be able to block a credential repair.
render_gid=''
video_gid=''
if command -v getent >/dev/null 2>&1; then
  render_gid="$(getent group render | awk -F: 'NR == 1 { print $3 }')" || render_gid=''
  video_gid="$(getent group video | awk -F: 'NR == 1 { print $3 }')" || video_gid=''
fi
[[ -n "$render_gid" ]] || render_gid="${RENDER_GID:-}"
[[ -n "$video_gid" ]] || video_gid="${VIDEO_GID:-}"

# Stand-ins for the recovery environment only. Compose rejects a group_add list
# whose entries are equal, so the two placeholders must never collide even when the
# host can supply neither group.
placeholder_render_gid="$render_gid"
placeholder_video_gid="$video_gid"
[[ "$placeholder_render_gid" =~ ^[0-9]+$ ]] || placeholder_render_gid="$(id -g 2>/dev/null || printf '0')"
[[ "$placeholder_render_gid" =~ ^[0-9]+$ ]] || placeholder_render_gid='0'
[[ "$placeholder_video_gid" =~ ^[0-9]+$ ]] || placeholder_video_gid="$placeholder_render_gid"
if [[ "$placeholder_render_gid" == "$placeholder_video_gid" ]]; then
  placeholder_video_gid=$(( (placeholder_render_gid % 65533) + 1 ))
fi

# The credential directory is created before anything is minted so a first run on a
# fresh checkout cannot fail halfway and leave the password written elsewhere.
umask 077
mkdir -p "$secrets_dir"
hand_to_caller "$secrets_dir"

new_password() {
  local password=''
  if command -v openssl >/dev/null 2>&1; then
    password="$(openssl rand -base64 48 2>/dev/null | tr -dc 'A-Za-z0-9' | cut -c1-48 || true)"
  fi
  # A host without openssl, or with one that fails, still gets a real credential
  # rather than a hard stop in the middle of a recovery.
  if [[ -z "$password" ]]; then
    password="$(head -c 96 /dev/urandom 2>/dev/null | base64 2>/dev/null | tr -dc 'A-Za-z0-9' | cut -c1-48 || true)"
  fi
  [[ -n "$password" ]] || die 'Password generation produced no entropy; refusing to write a weak credential.'
  printf '%s' "$password"
}

# Rewrites KEY=VALUE lines from a overrides file, appending keys the source lacks.
# Values are matched on the key only, so a value containing "=" survives intact.
apply_overrides() {
  local source_file="${1:?pass the file to read}" destination="${2:?pass the destination}" overrides_file="${3:?pass the KEY=VALUE overrides}"
  awk -v ovfile="$overrides_file" '
    BEGIN {
      count = 0
      while ((getline line < ovfile) > 0) {
        if (line == "") continue
        eq = index(line, "=")
        if (eq < 2) continue
        key = substr(line, 1, eq - 1)
        if (key !~ /^[A-Za-z_][A-Za-z0-9_]*$/) continue
        count++
        order[count] = key
        value[key] = substr(line, eq + 1)
      }
      close(ovfile)
    }
    {
      eq = index($0, "=")
      if (eq > 1) {
        key = substr($0, 1, eq - 1)
        if (key ~ /^[A-Za-z_][A-Za-z0-9_]*$/ && (key in value)) {
          print key "=" value[key]
          seen[key] = 1
          next
        }
      }
      print
    }
    END {
      for (i = 1; i <= count; i++) {
        if (!(order[i] in seen)) print order[i] "=" value[order[i]]
      }
    }
  ' "$source_file" > "$destination"
}

# Builds a Compose environment in which every required variable resolves. POSTGRES_PASSWORD_FILE
# always comes from the caller; every other required variable is stood in for only when it is
# missing or blank. A stand-in never reaches the real .env: paths point at a throwaway directory
# and group IDs at the caller's.
build_compose_env() {
  local secret_path="${1:?pass the postgres secret to use}" destination="${2:?pass the destination env path}"
  local overrides="$sandbox/overrides.env" placeholder_dir="$sandbox/placeholder" var current
  mkdir -p "$placeholder_dir"
  : > "$overrides"
  while IFS= read -r var; do
    # The caller always owns this one. Recovery mints a credential at a temporary
    # path that .env does not name yet, and the path .env does name is the one that
    # went missing. Interpolating the stale path would make Compose fail to mount
    # the secret before postgres exists to be repaired, which is the exact state
    # --recover is for.
    if [[ "$var" == POSTGRES_PASSWORD_FILE ]]; then
      printf '%s=%s\n' "$var" "$secret_path" >> "$overrides"
      continue
    fi
    current="${!var:-}"
    [[ -n "$current" ]] && continue
    case "$var" in
      RENDER_GID) printf '%s=%s\n' "$var" "$placeholder_render_gid" >> "$overrides" ;;
      VIDEO_GID) printf '%s=%s\n' "$var" "$placeholder_video_gid" >> "$overrides" ;;
      *) printf '%s=%s\n' "$var" "$placeholder_dir" >> "$overrides" ;;
    esac
  done < <(required_compose_vars)
  apply_overrides "$env_file" "$destination" "$overrides"
}

# Runs Compose against a throwaway environment. load_deployment_env already exported
# the real .env into this process, and process variables outrank --env-file, so each
# variable the temporary file supplies is unset from the process first.
compose_with() {
  local env_path="${1:?pass the Compose environment to use}"
  shift
  local -a unsets=()
  while IFS= read -r var; do
    unsets+=(-u "$var")
  done < <(required_compose_vars)
  env ${unsets[@]+"${unsets[@]}"} docker compose --env-file "$env_path" -f "$compose_file" "$@"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "$1"
}

postgres_container_exists() {
  local env_path="${1:?pass the Compose environment}"
  compose_with "$env_path" ps -a -q postgres 2>/dev/null | grep -q .
}

postgres_volume_exists() {
  local env_path="${1:?pass the Compose environment}" project=''
  project="$(compose_with "$env_path" config --project-name 2>/dev/null || true)"
  if [[ -z "$project" ]]; then
    project="$(basename "$root" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')"
  fi
  docker volume ls --format '{{.Name}}' 2>/dev/null | grep -qx "${project}_postgres_data"
}

# First initialization must not mint a credential over a data directory whose
# stored verifier is unknown: that is exactly how a secret file drifts away from
# the ledger and leaves pg_isready reporting healthy over a database that rejects
# every real client. The probe runs through the throwaway environment so a blank
# .env cannot make it silently pass, which is how the old guard failed open.
first_initialization() {
  local password temporary_password probe_env
  password="$(new_password)"
  temporary_password="$sandbox/postgres_password"
  printf '%s' "$password" > "$temporary_password"
  chmod 600 "$temporary_password"
  probe_env="$sandbox/probe.env"
  build_compose_env "$temporary_password" "$probe_env"

  if command -v docker >/dev/null 2>&1 &&
    { postgres_container_exists "$probe_env" || postgres_volume_exists "$probe_env"; }; then
    printf '%s\n' 'Refusing to mint a new credential: a postgres container or postgres_data volume already exists here, and its stored verifier is unknown to the missing secret file.' >&2
    printf '%s\n' 'Run scripts/generate-secrets.sh --recover to create a new credential and reconcile it without discarding the audit ledger.' >&2
    exit 1
  fi
  mv -f "$temporary_password" "$postgres_password_file"
}

# Recovery repairs the live role through the container's trusted local socket, so it
# needs no prior credential, and it never recreates the data directory. The new
# password is only promoted once the network probe accepts it, which is the same
# scram path every consuming service uses.
recover_missing_credential() {
  require_docker '--recover needs Docker Compose to start and repair postgres.'

  local password temporary_password recovery_env
  password="$(new_password)"
  temporary_password="$sandbox/postgres_password"
  printf '%s' "$password" > "$temporary_password"
  chmod 600 "$temporary_password"
  recovery_env="$sandbox/recovery.env"
  build_compose_env "$temporary_password" "$recovery_env"

  if ! postgres_container_exists "$recovery_env" && ! postgres_volume_exists "$recovery_env"; then
    printf '%s\n' 'Note: no postgres container or postgres_data volume was found here, so this initializes a fresh database.'
  fi

  if ! compose_with "$recovery_env" up -d postgres; then
    die '--recover could not start postgres with the temporary recovery environment.'
  fi

  if ! ENV_FILE="$recovery_env" COMPOSE_FILE="$compose_file" bash "$sync_script" --no-restart; then
    die '--recover left the real .env and password file unchanged because PostgreSQL rejected the reconciliation.'
  fi

  mv -f "$temporary_password" "$postgres_password_file"
}

rotate_live_credential() {
  require_docker '--force needs Docker Compose and a running initialized postgres service.'
  [[ -e "$postgres_password_file" ]] || die '--force requires the existing password file. Use --recover when that file is missing.'

  local old_password password temporary_password compose_env
  old_password="$(<"$postgres_password_file")"
  postgres_user="${POSTGRES_USER:-inference}"
  postgres_db="${POSTGRES_DB:-inference}"
  password="$(new_password)"
  temporary_password="$sandbox/postgres_password"
  printf '%s' "$password" > "$temporary_password"
  chmod 600 "$temporary_password"
  compose_env="$sandbox/rotate.env"
  build_compose_env "$postgres_password_file" "$compose_env"

  if ! compose_with "$compose_env" ps -q postgres | grep -q .; then
    die '--force refused: start the initialized postgres service before rotating its credential.'
  fi
  if ! compose_with "$compose_env" exec -T \
    -e PGPASSWORD="$old_password" postgres \
    psql -v ON_ERROR_STOP=1 -U "$postgres_user" -d "$postgres_db" \
    -v role="$postgres_user" -v new_password="$password" \
    -c "ALTER ROLE :\"role\" PASSWORD :'new_password';"; then
    die '--force did not change the local secret because PostgreSQL rejected the rotation.'
  fi
  mv -f "$temporary_password" "$postgres_password_file"
}

if [[ "$recover" == true ]]; then
  recover_missing_credential
elif [[ "$force" == true ]]; then
  rotate_live_credential
elif [[ ! -e "$postgres_password_file" ]]; then
  first_initialization
fi

# The router mounts the uv-resolved interpreter prefix at its host path, so the
# variable must name the real tree. Recovery can re-derive it from an installed
# runtime with the same resolution scripts/download-models.sh and scripts/
# validate.sh perform, so a wiped .env does not force an expensive re-download
# before the stack can start.
detect_runtime_python_root() {
  local model_dir="${ORNITH_MODEL_DIR:-}" runtime_root runtime_python candidate
  [[ -n "$model_dir" ]] || return 1
  runtime_root="$model_dir/installed-runtime"
  [[ -x "$runtime_root/venv/bin/python" ]] || return 1
  runtime_python="$(readlink -f "$runtime_root/venv/bin/python" 2>/dev/null)" || return 1
  [[ -n "$runtime_python" && -x "$runtime_python" ]] || return 1
  candidate="$(dirname "$(dirname "$runtime_python")")"
  [[ -d "$candidate" ]] || return 1
  printf '%s' "$candidate"
}

# Only real values are written back. A placeholder exists to satisfy Compose during
# the repair, never to become the deployment configuration.
final_overrides="$sandbox/final.env"
: > "$final_overrides"
printf 'POSTGRES_PASSWORD_FILE=%s\n' "$postgres_password_file" >> "$final_overrides"
if [[ -n "$render_gid" ]]; then
  printf 'RENDER_GID=%s\n' "$render_gid" >> "$final_overrides"
fi
if [[ -n "$video_gid" ]]; then
  printf 'VIDEO_GID=%s\n' "$video_gid" >> "$final_overrides"
fi

# An empty ORNITH_RUNTIME_PYTHON_ROOT is the difference between a recovered
# database and a stack that still cannot start: Compose refuses to interpolate
# the llama-swap mount without it. Re-derive it when the runtime is installed;
# never overwrite a value the deployment already carries.
runtime_python_root="${ORNITH_RUNTIME_PYTHON_ROOT:-}"
if [[ -z "$runtime_python_root" ]]; then
  runtime_python_root="$(detect_runtime_python_root || true)"
fi
if [[ -n "$runtime_python_root" ]]; then
  printf 'ORNITH_RUNTIME_PYTHON_ROOT=%s\n' "$runtime_python_root" >> "$final_overrides"
fi

temporary_env="$sandbox/env.new"
apply_overrides "$env_file" "$temporary_env" "$final_overrides"
mv "$temporary_env" "$env_file"
chmod 600 "$env_file" "$postgres_password_file"
hand_to_caller "$env_file" "$secrets_dir" "$postgres_password_file"

if [[ -z "$render_gid" || -z "$video_gid" ]]; then
  printf '%s\n' 'Warning: RENDER_GID and/or VIDEO_GID are still unset, so the /dev/dri services will not start. Set the host render and video group IDs in .env when this host has them.' >&2
fi
if [[ -n "$runtime_python_root" && -z "${ORNITH_RUNTIME_PYTHON_ROOT:-}" ]]; then
  printf '%s\n' "Re-derived ORNITH_RUNTIME_PYTHON_ROOT=$runtime_python_root from the installed runtime."
elif [[ -z "$runtime_python_root" ]]; then
  printf '%s\n' 'Warning: ORNITH_RUNTIME_PYTHON_ROOT is unset and no installed runtime was found under ORNITH_MODEL_DIR/installed-runtime, so the llama-swap router will not start. Run scripts/download-models.sh to provision it.' >&2
fi
printf '%s\n' 'Configured the PostgreSQL credential and numeric render/video group IDs.'
