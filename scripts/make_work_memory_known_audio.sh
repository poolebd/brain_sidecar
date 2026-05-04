#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT_DIR/runtime/user-tests/work-memory-known/source.wav}"
VOICE="${BRAIN_SIDECAR_FIXTURE_VOICE:-en-us}"
SPEED="${BRAIN_SIDECAR_FIXTURE_VOICE_SPEED:-125}"
GAP="${BRAIN_SIDECAR_FIXTURE_WORD_GAP:-8}"

if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "Missing espeak-ng. Install it or provide BRAIN_SIDECAR_USER_TEST_SOURCE manually." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

TEXT="This is a controlled Brain Sidecar validation source. Topic one: Online Generator Monitoring at T A Smith. Think about hydrogen cooled generators, flux probes, alarm interpretation, and maintenance decisions. Topic two: five hundred kilovolt breaker replacement at T A Smith. The conversation should recall outage windows, acceptance criteria, commissioning tests, warranty terms, and operating risk. Topic three: transmission relay modernization. Use current transformer and potential transformer inputs, relay settings, trip matrix, breaker failure logic, and nuisance trip diagnostics. Topic four: P G and E gas mains replacement cost model. We need cost drivers, soil, terrain, G I S data, random forest, and scenario analysis. Topic five: SaskPower workforce planning. Connect workload, capacity, productivity, backlog risk, and phase handoff assumptions. Topic six: current technology practice. Ask what current best practices for vector database indexing are. End of known source."

espeak-ng -v "$VOICE" -s "$SPEED" -g "$GAP" -w "$OUT" "$TEXT"

if command -v ffprobe >/dev/null 2>&1; then
  DURATION="$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$OUT")"
  echo "Wrote known work-memory audio: $OUT (${DURATION}s)"
else
  echo "Wrote known work-memory audio: $OUT"
fi
