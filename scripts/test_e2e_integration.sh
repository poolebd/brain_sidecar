#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_WAV="${BRAIN_SIDECAR_E2E_FIXTURE_WAV:-$ROOT_DIR/runtime/test-audio/osr-us-female-harvard.wav}"

if [[ ! -f "$FIXTURE_WAV" ]]; then
  echo "Missing fixture WAV: $FIXTURE_WAV" >&2
  echo "Run ./scripts/fetch_test_audio.sh or set BRAIN_SIDECAR_E2E_FIXTURE_WAV." >&2
  exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing Python virtualenv at $ROOT_DIR/.venv." >&2
  echo "Create it with: python3.11 -m venv .venv && . .venv/bin/activate && pip install -e '.[gpu,recall,dev]'" >&2
  exit 1
fi

export BRAIN_SIDECAR_E2E_FIXTURE_WAV="$FIXTURE_WAV"
export BRAIN_SIDECAR_DATA_DIR="${BRAIN_SIDECAR_DATA_DIR:-$ROOT_DIR/runtime/e2e-integration}"
export BRAIN_SIDECAR_ASR_PRIMARY_MODEL="${BRAIN_SIDECAR_ASR_PRIMARY_MODEL:-tiny.en}"
export BRAIN_SIDECAR_ASR_FALLBACK_MODEL="${BRAIN_SIDECAR_ASR_FALLBACK_MODEL:-tiny.en}"
export BRAIN_SIDECAR_ASR_BEAM_SIZE="${BRAIN_SIDECAR_ASR_BEAM_SIZE:-1}"
export BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS="${BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS:-999}"
export BRAIN_SIDECAR_WEB_CONTEXT_ENABLED="${BRAIN_SIDECAR_WEB_CONTEXT_ENABLED:-0}"
export BRAIN_SIDECAR_TEST_MODE_ENABLED="${BRAIN_SIDECAR_TEST_MODE_ENABLED:-1}"

. "$ROOT_DIR/.venv/bin/activate"
. "$ROOT_DIR/scripts/gpu-env.sh"

cd "$ROOT_DIR/ui"
npx playwright test --config=playwright.integration.config.ts "$@"
