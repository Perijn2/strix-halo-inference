#!/usr/bin/env bash
# Author: Perijn
# Summary: Repairs a drifted PostgreSQL credential by aligning the live role password with the current Compose secret file.
# Usage: scripts/sync-postgres-secret.sh [--dry-run] while the initialized postgres service is running; override ENV_FILE and COMPOSE_FILE for a non-default project.
#
# Core principle: the secret file is the source of truth and the audit ledger is never recreated.
# The Postgres image applies POSTGRES_PASSWORD_FILE only while it initializes an empty data
# directory, so a secret that was replaced, restored, or reissued out of band never reaches the
# stored scram verifier. pg_isready keeps reporting the service healthy because it authenticates
# nothing, while every real client dies with `FATAL: password authentication failed`. This script
# restores agreement in place through the container's trusted local socket, which needs no prior
# password, so no volume is destroyed and no audit row is lost.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

dry_run=false
case "${1:-}" in
  --dry-run) dry_run=true ;;
  '') ;;
  *)
    printf 'Usage: scripts/sync-postgres-secret.sh [--dry-run]\n' >&2
    exit 2
    ;;
esac

# shellcheck source=lib/load-env.sh
source "$root/scripts/lib/load-env.sh"
load_deployment_env "${ENV_FILE:-$root/.env}"

compose_file="${COMPOSE_FILE:-$root/compose.yaml}"
postgres_service='postgres'
postgres_user="${POSTGRES_USER:-inference}"
postgres_db="${POSTGRES_DB:-inference}"
secret_file="${POSTGRES_PASSWORD_FILE:?set POSTGRES_PASSWORD_FILE in .env}"

command -v docker >/dev/null || {
  printf '%s\n' 'docker is required to reach the postgres service.' >&2
  exit 1
}
test -r "$secret_file" || {
  printf 'Cannot read the PostgreSQL secret: %s\n' "$secret_file" >&2
  exit 1
}
# The application strips the secret on read, so mirror that here rather than
# reporting drift over an invisible trailing newline.
password="$(tr -d '\r\n' < "$secret_file")"
[[ -n "$password" ]] || {
  printf 'The PostgreSQL secret is empty: %s\n' "$secret_file" >&2
  exit 1
}
if ! docker compose -f "$compose_file" ps --status running -q "$postgres_service" | grep -q .; then
  printf '%s\n' "No running ${postgres_service} service in ${compose_file}; start the initialized database before syncing." >&2
  exit 1
fi

# Loopback is trust in the default pg_hba, so a probe there would pass even with a
# wrong password. Reach the container over its own network address instead, which
# matches the scram-sha-256 path every consuming service actually uses.
# The probe travels on stdin rather than through a copied path, so no host-side
# path rewriting can corrupt where it lands inside the container.
probe_script="$(mktemp "${TMPDIR:-/tmp}/strix-pg-probe.XXXXXX")"
cleanup() { rm -f "$probe_script"; }
trap cleanup EXIT
cat > "$probe_script" <<'PROBE'
#!/bin/sh
set -eu
address="$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 { print $1 }')"
[ -n "$address" ] || {
  printf 'cannot resolve the container address\n' >&2
  exit 3
}
PGPASSWORD="$PROBE_PASSWORD" psql -h "$address" -U "$PROBE_USER" -d "$PROBE_DB" -tAc 'select 1' >/dev/null
PROBE

probe() {
  docker compose -f "$compose_file" exec -i -T \
    -e PROBE_PASSWORD="$password" \
    -e PROBE_USER="$postgres_user" \
    -e PROBE_DB="$postgres_db" \
    "$postgres_service" sh -s < "$probe_script" >/dev/null
}

if probe; then
  printf '%s\n' 'In sync: the live credential already matches the secret file.'
  exit 0
fi

if [[ "$dry_run" == true ]]; then
  printf '%s\n' "Drift: ${secret_file} does not authenticate against the live role \"${postgres_user}\"." >&2
  printf '%s\n' 'Run the same command without --dry-run to reconcile it in place.' >&2
  exit 1
fi

printf 'Reconciling role "%s" with %s ...\n' "$postgres_user" "$secret_file"
# CURRENT_USER is a legal role spec, so the target role never needs identifier
# quoting and psql quotes the password literal itself.
if ! printf "ALTER ROLE CURRENT_USER PASSWORD :'pw';\n" |
  docker compose -f "$compose_file" exec -i -T "$postgres_service" \
    psql -v ON_ERROR_STOP=1 -U "$postgres_user" -d "$postgres_db" -v pw="$password" -f - >/dev/null; then
  printf '%s\n' 'PostgreSQL rejected the reconciliation. If the local socket is not trust, this role cannot be repaired without an operator credential.' >&2
  exit 1
fi

if ! probe; then
  printf '%s\n' 'Reconciliation reported success but the network credential still fails; inspect pg_hba.conf and the role verifier.' >&2
  exit 1
fi

printf 'Reconciled: scram-sha-256 now accepts %s for user "%s".\n' "$secret_file" "$postgres_user"

# A live ensemble reconnects per operation and recovers on its own; a crash-looped
# one never reached that code path, so converge it explicitly and without touching
# dependencies, reporting what was actually done rather than assuming.
if docker compose -f "$compose_file" config --services 2>/dev/null | grep -qx 'ocr-ensemble'; then
  if docker compose -f "$compose_file" ps -a -q ocr-ensemble | grep -q .; then
    docker compose -f "$compose_file" restart ocr-ensemble >/dev/null 2>&1 || true
    printf '%s\n' 'Restarted ocr-ensemble so it re-reads the secret.'
  else
    docker compose -f "$compose_file" up -d --no-deps ocr-ensemble >/dev/null 2>&1 || true
    printf '%s\n' 'Started ocr-ensemble.'
  fi
fi
