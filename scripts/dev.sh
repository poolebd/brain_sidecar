#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi

. .venv/bin/activate
pip install -e ".[gpu,recall,dev]"

if [[ ! -d ui/node_modules ]]; then
  npm --prefix ui install
fi

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT_DIR}/.env"
  set +a
fi

. "${ROOT_DIR}/scripts/gpu-env.sh"

export BRAIN_SIDECAR_DATA_DIR="${BRAIN_SIDECAR_DATA_DIR:-${ROOT_DIR}/runtime}"
export BRAIN_SIDECAR_TEST_MODE_ENABLED="${BRAIN_SIDECAR_TEST_MODE_ENABLED:-1}"

python -m uvicorn brain_sidecar.server.app:create_app \
  --factory \
  --host "${BRAIN_SIDECAR_HOST:-127.0.0.1}" \
  --port "${BRAIN_SIDECAR_PORT:-8765}" &
BACKEND_PID="$!"

npm --prefix ui run dev -- --host "${BRAIN_SIDECAR_UI_HOST:-127.0.0.1}" --port "${BRAIN_SIDECAR_UI_PORT:-8766}" &
UI_PID="$!"

cleanup() {
  kill "${BACKEND_PID}" "${UI_PID}" 2>/dev/null || true
}
trap cleanup EXIT

wait -n "${BACKEND_PID}" "${UI_PID}"
