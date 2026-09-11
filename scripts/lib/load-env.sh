#!/usr/bin/env bash
# Author: Perijn
# Summary: Reads simple deployment values from the project .env file without evaluating shell code.
# Usage: Source from a provisioning script, call load_deployment_env, then read the required named variables.
# This intentionally accepts only KEY=VALUE entries. Quotes and shell expansions are not interpreted.

load_deployment_env() {
  local env_file="${1:?pass the .env file path}"
  [[ -f "$env_file" ]] || {
    printf 'Missing .env: %s\n' "$env_file" >&2
    return 1
  }

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# || -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || {
      printf 'Unsupported .env entry (expected KEY=VALUE): %s\n' "$line" >&2
      return 1
    }
    local key="${BASH_REMATCH[1]}"
    local value="${BASH_REMATCH[2]}"
    if [[ "$value" == *'$'* || "$value" == *'`'* ]]; then
      printf 'Shell expansion is not allowed in .env value: %s\n' "$key" >&2
      return 1
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
}
