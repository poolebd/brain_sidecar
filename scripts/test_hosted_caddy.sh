#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

export BRAIN_SIDECAR_CADDY_HOSTED_URL="${BRAIN_SIDECAR_CADDY_HOSTED_URL:-https://notes.shoalstone.net/}"
export BRAIN_SIDECAR_CADDY_EXPECTED_IP="${BRAIN_SIDECAR_CADDY_EXPECTED_IP:-192.168.86.45}"
export BRAIN_SIDECAR_CADDY_BASIC_USER="${BRAIN_SIDECAR_CADDY_BASIC_USER:-portal}"

if [[ -z "${BRAIN_SIDECAR_CADDY_BASIC_PASSWORD:-}" ]]; then
  echo "Set BRAIN_SIDECAR_CADDY_BASIC_PASSWORD in the environment or .env before running the hosted Caddy smoke." >&2
  exit 2
fi

npm --prefix "$ROOT_DIR/ui" run test:e2e:hosted
