#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/runtime/test-audio}"
RAW_DIR="${OUT_DIR}/raw"

mkdir -p "${OUT_DIR}" "${RAW_DIR}"

download() {
  local url="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    curl -L --fail --retry 3 --connect-timeout 20 --output "${path}" "${url}"
  fi
}

convert_wav() {
  local input="$1"
  local output="$2"
  shift 2
  ffmpeg -hide_banner -loglevel error -y "$@" -i "${input}" -ac 1 -ar 16000 -sample_fmt s16 "${output}"
}

download "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav" \
  "${RAW_DIR}/OSR_us_000_0010_8k.wav"
convert_wav "${RAW_DIR}/OSR_us_000_0010_8k.wav" "${OUT_DIR}/osr-us-female-harvard.wav"

download "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0030_8k.wav" \
  "${RAW_DIR}/OSR_us_000_0030_8k.wav"
convert_wav "${RAW_DIR}/OSR_us_000_0030_8k.wav" "${OUT_DIR}/osr-us-male-harvard.wav"

download "https://commons.wikimedia.org/wiki/Special:Redirect/file/Alexander_Graham_Bell%27s_Voice.ogg" \
  "${RAW_DIR}/alexander-graham-bell-voice.ogg"
convert_wav "${RAW_DIR}/alexander-graham-bell-voice.ogg" "${OUT_DIR}/bell-public-domain.wav"

download "https://commons.wikimedia.org/wiki/Special:Redirect/file/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg" \
  "${RAW_DIR}/jfk-rice-moon.ogg"
convert_wav "${RAW_DIR}/jfk-rice-moon.ogg" "${OUT_DIR}/jfk-moon-30s.wav" -ss 0 -t 30

cat > "${OUT_DIR}/README.md" <<'EOF'
# Brain Sidecar Test Audio

These files are converted to mono 16 kHz PCM WAV for Brain Sidecar fixture tests.

## Files

- `osr-us-female-harvard.wav`: Open Speech Repository, American English Harvard sentences, source `OSR_us_000_0010_8k.wav`.
- `osr-us-male-harvard.wav`: Open Speech Repository, American English Harvard sentences, source `OSR_us_000_0030_8k.wav`.
- `bell-public-domain.wav`: Wikimedia Commons, "Alexander Graham Bell's Voice.ogg", public domain.
- `jfk-moon-30s.wav`: Wikimedia Commons, first 30 seconds of JFK Rice University moon speech, public domain U.S. government work.

Open Speech Repository asks that the source be identified as "Open Speech Repository".
EOF

echo "Wrote fixtures to ${OUT_DIR}"
for wav in "${OUT_DIR}"/*.wav; do
  printf '%s\t' "$(basename "${wav}")"
  ffprobe -hide_banner -loglevel error \
    -show_entries stream=codec_name,sample_rate,channels,duration \
    -of compact=p=0:nk=1 \
    "${wav}"
done
