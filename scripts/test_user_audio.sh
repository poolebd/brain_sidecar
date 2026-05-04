#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_AUDIO="${BRAIN_SIDECAR_USER_TEST_SOURCE:-/home/bp/Nextcloud2/_library/_Sargent-&-Lundy/Apr 2 at 12-00.m4a}"
RUN_DIR="${BRAIN_SIDECAR_USER_TEST_DIR:-$ROOT_DIR/runtime/user-tests/apr-2-1200}"
ARTIFACT_DIR="$RUN_DIR/artifacts"
DATA_DIR="$RUN_DIR/data"
MAX_SECONDS="${BRAIN_SIDECAR_USER_TEST_MAX_SECONDS:-}"

if [[ ! -f "$SOURCE_AUDIO" ]]; then
  echo "Missing source audio: $SOURCE_AUDIO" >&2
  exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing Python virtualenv at $ROOT_DIR/.venv." >&2
  echo "Create it with: python3.11 -m venv .venv && . .venv/bin/activate && pip install -e '.[gpu,recall,dev]'" >&2
  exit 1
fi

if [[ "${BRAIN_SIDECAR_USER_TEST_KEEP_DATA:-0}" != "1" ]]; then
  rm -rf "$ARTIFACT_DIR" "$DATA_DIR"
fi
mkdir -p "$ARTIFACT_DIR" "$DATA_DIR"

if [[ -n "$MAX_SECONDS" ]]; then
  FIXTURE_WAV="$RUN_DIR/input-${MAX_SECONDS}s.wav"
  ffmpeg -y -i "$SOURCE_AUDIO" -t "$MAX_SECONDS" -ac 1 -ar 16000 -sample_fmt s16 "$FIXTURE_WAV" >/dev/null 2>&1
else
  FIXTURE_WAV="$RUN_DIR/input.wav"
  if [[ ! -f "$FIXTURE_WAV" || "$SOURCE_AUDIO" -nt "$FIXTURE_WAV" ]]; then
    ffmpeg -y -i "$SOURCE_AUDIO" -ac 1 -ar 16000 -sample_fmt s16 "$FIXTURE_WAV" >/dev/null 2>&1
  fi
fi

DURATION_SECONDS="$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$FIXTURE_WAV")"

export BRAIN_SIDECAR_E2E_FIXTURE_WAV="$FIXTURE_WAV"
export BRAIN_SIDECAR_USER_TEST_ARTIFACT_DIR="$ARTIFACT_DIR"
export BRAIN_SIDECAR_USER_TEST_DURATION_SECONDS="$DURATION_SECONDS"
export BRAIN_SIDECAR_DATA_DIR="$DATA_DIR"
export BRAIN_SIDECAR_HOST="${BRAIN_SIDECAR_HOST:-127.0.0.1}"
export BRAIN_SIDECAR_PORT="${BRAIN_SIDECAR_PORT:-8775}"
export PW_API_BASE="${PW_API_BASE:-http://127.0.0.1:8775}"
export PW_UI_BASE="${PW_UI_BASE:-http://127.0.0.1:8776}"
export VITE_API_BASE="$PW_API_BASE"
export VITE_ENABLE_FIXTURE_AUDIO=1
export VITE_DEFAULT_FIXTURE_AUDIO="$FIXTURE_WAV"
export BRAIN_SIDECAR_ASR_PRIMARY_MODEL="${BRAIN_SIDECAR_ASR_PRIMARY_MODEL:-medium.en}"
export BRAIN_SIDECAR_ASR_FALLBACK_MODEL="${BRAIN_SIDECAR_ASR_FALLBACK_MODEL:-small.en}"
export BRAIN_SIDECAR_ASR_BEAM_SIZE="${BRAIN_SIDECAR_ASR_BEAM_SIZE:-5}"
export BRAIN_SIDECAR_ASR_INITIAL_PROMPT="${BRAIN_SIDECAR_ASR_INITIAL_PROMPT:-Preferred spellings for this validation: Online Generator Monitoring; OGM; T.A. Smith; 500kV breaker replacement; relay modernization; CT; PT; PG&E gas mains; SaskPower workforce planning; cost drivers; outage window; commissioning tests.}"
export BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS="${BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS:-3}"
export BRAIN_SIDECAR_DISABLE_LIVE_EMBEDDINGS="${BRAIN_SIDECAR_DISABLE_LIVE_EMBEDDINGS:-1}"
export BRAIN_SIDECAR_USER_TEST_SCREENSHOT_INTERVAL_MS="${BRAIN_SIDECAR_USER_TEST_SCREENSHOT_INTERVAL_MS:-90000}"
export BRAIN_SIDECAR_WEB_CONTEXT_ENABLED="${BRAIN_SIDECAR_WEB_CONTEXT_ENABLED:-0}"
export BRAIN_SIDECAR_TEST_MODE_ENABLED="${BRAIN_SIDECAR_TEST_MODE_ENABLED:-1}"

. "$ROOT_DIR/.venv/bin/activate"
. "$ROOT_DIR/scripts/gpu-env.sh"

echo "User audio fixture: $FIXTURE_WAV"
echo "Duration seconds: $DURATION_SECONDS"
echo "Artifacts: $ARTIFACT_DIR"

cd "$ROOT_DIR/ui"
npx playwright test --config=playwright.user-audio.config.ts "$@"
