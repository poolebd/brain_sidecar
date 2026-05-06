#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi

. .venv/bin/activate

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT_DIR}/.env"
  set +a
fi

export BRAIN_SIDECAR_ASR_BACKEND="${BRAIN_SIDECAR_ASR_BACKEND:-nemotron_streaming}"
export BRAIN_SIDECAR_NEMOTRON_CHUNK_MS="${BRAIN_SIDECAR_NEMOTRON_CHUNK_MS:-160}"
export BRAIN_SIDECAR_NEMOTRON_DTYPE="${BRAIN_SIDECAR_NEMOTRON_DTYPE:-float32}"

PYTHON_EXTRAS="gpu,recall,dev"
if [[ "$BRAIN_SIDECAR_ASR_BACKEND" == "nemotron_streaming" ]]; then
  PYTHON_EXTRAS="$PYTHON_EXTRAS,nemotron"
fi
pip install -e ".[$PYTHON_EXTRAS]"

if [[ "$BRAIN_SIDECAR_ASR_BACKEND" == "nemotron_streaming" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("nemo.collections.asr") else 1)
PY
  then
    pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
  fi
fi

if [[ ! -d ui/node_modules ]]; then
  npm --prefix ui install
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
