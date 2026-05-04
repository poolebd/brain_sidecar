# Brain Sidecar

Brain Sidecar is an all-local live audio companion for transcription, topic notes,
action capture, and recall. The first UI is a local browser app, but the core is
headless and event-driven so a desktop GUI can reuse the same engine later.

## What v1 Does

- Captures a USB mic through `ffmpeg`.
- Transcribes locally with Faster-Whisper on CUDA only.
- Generates rolling notes with local Ollama.
- Searches prior sessions and selected local files with local embeddings.
- Surfaces fast work-memory parallels from a pre-indexed local career/project archive.
- Learns a local voice profile from guided phrase corrections to bias ASR spellings.
- Stores text artifacts only. Raw audio stays in memory and is discarded.
- Fails visibly if GPU transcription is unavailable.

## Architecture

- `brain_sidecar/core`: UI-independent engine for devices, capture, ASR, notes,
  recall, storage, and typed events.
- `brain_sidecar/server`: FastAPI adapter exposing HTTP and SSE.
- `ui`: Vite React client for the first local web interface.

The GUI expansion seam is the typed event bus plus local HTTP API. A later Tauri,
Electron, Qt, or native GUI can either call the FastAPI daemon or embed the core
Python package directly.

## Setup

```bash
cd /home/bp/project/brain-sidecar
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[gpu,recall,dev]"
npm --prefix ui install
```

Make sure the local Ollama models exist:

```bash
ollama pull phi3:mini
ollama pull embeddinggemma
```

If Faster-Whisper cannot find CUDA libraries after installing the GPU extras,
launch through `scripts/dev.sh` or source `scripts/gpu-env.sh`; it exports the
`LD_LIBRARY_PATH` used by the pip-installed NVIDIA CUDA libraries.

## Run

```bash
./scripts/dev.sh
```

Backend: `http://127.0.0.1:8765`

UI: `http://127.0.0.1:8766`

## GPU Contract

The app requires:

- NVIDIA GPU visible to `nvidia-smi`.
- Faster-Whisper/CTranslate2 CUDA runtime.
- Ollama running locally, with the configured chat model on GPU.

ASR is configured with `device="cuda"` and `compute_type="float16"`. If
`medium.en` cannot load because of VRAM pressure, the app may load `small.en`,
but still on CUDA only. CPU fallback is intentionally not implemented.

## Speed And Quality Tuning

The live pipeline is split into capture, transcription, and post-processing
tasks. Capture keeps reading audio while CUDA ASR and Ollama work in the
background. If transcription falls behind, stale audio windows are dropped and
the UI shows the dropped-window count so the sidecar stays real-time.

Useful knobs:

- `BRAIN_SIDECAR_ASR_BEAM_SIZE`: higher can improve transcript quality, lower is faster.
- `BRAIN_SIDECAR_ASR_VAD_MIN_SILENCE_MS`: lower splits speech faster, higher creates steadier segments, default `300`.
- `BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS`: audio window before ASR emits a transcript, default `3.4`.
- `BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS`: overlap between ASR windows, default `0.8`.
- `BRAIN_SIDECAR_AUDIO_CHUNK_MS`: capture chunk size, default `250`.
- `BRAIN_SIDECAR_ASR_NO_SPEECH_THRESHOLD`: reject segments Faster-Whisper marks as likely silence, default `0.6`.
- `BRAIN_SIDECAR_ASR_LOG_PROB_THRESHOLD`: reject low-confidence ASR segments, default `-1.0`.
- `BRAIN_SIDECAR_ASR_COMPRESSION_RATIO_THRESHOLD`: reject repetitive/compressed ASR spans, default `2.4`.
- `BRAIN_SIDECAR_ASR_MIN_AUDIO_RMS`: skip near-silent audio windows before ASR, default `0.006`.
- `BRAIN_SIDECAR_ASR_MIN_FREE_VRAM_MB`: free VRAM target before loading Faster-Whisper, default `3500`.
- `BRAIN_SIDECAR_ASR_UNLOAD_OLLAMA_ON_START`: stop GPU-resident Ollama models before ASR load when VRAM is tight, default `true`.
- `BRAIN_SIDECAR_ASR_GPU_FREE_TIMEOUT_SECONDS`: wait budget after unloading Ollama, default `10`.
- `BRAIN_SIDECAR_OLLAMA_KEEP_ALIVE`: Ollama `keep_alive` sent by app chat/embed calls, default `10m` so the smaller chat model can stay warm beside ASR.
- `BRAIN_SIDECAR_TRANSCRIPTION_QUEUE_SIZE`: number of pending audio windows before stale windows are dropped, default `8`.
- `BRAIN_SIDECAR_DEDUPE_SIMILARITY_THRESHOLD`: suppresses repeated text from overlapping windows.
- `BRAIN_SIDECAR_ASR_INITIAL_PROMPT`: optional vocabulary/context hint for names, projects, or jargon.
- `BRAIN_SIDECAR_VOICE_PROFILE_REQUIRED_PHRASES`: accepted enrollment phrases before profile hints activate.
- `BRAIN_SIDECAR_VOICE_ENROLLMENT_PHRASE_SECONDS`: fixed recording window for each guided voice phrase.
- `BRAIN_SIDECAR_WEB_CONTEXT_ENABLED`: set to `true` to allow selective Brave web context for explicit current-tech questions.
- `BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY`: Brave Search API key; without this, web context stays disabled.
- `BRAIN_SIDECAR_WEB_CONTEXT_MIN_INTERVAL_SECONDS`: cooldown between live web lookups, default `90`.
- `BRAIN_SIDECAR_WEB_CONTEXT_TIMEOUT_SECONDS`: hard Brave request timeout, default `3.0`.
- `BRAIN_SIDECAR_WEB_CONTEXT_MAX_RESULTS`: maximum Brave Web Search results used per note, default `3`.

Web context uses Brave Web Search only. It sends a compact sanitized query, never
raw transcript windows, and publishes results as live-only Topic Notes. These
notes are not stored in SQLite, embedded, or added to recall indexes.

## Work Memory

The Work Memory layer is for current-role recall support, not recruiting. It
uses high-signal historical documents under `/home/bp/Nextcloud2/Job Hunting`
as a canonical timeline, then correlates evidence from
`/home/bp/Nextcloud2/_library/_shoalstone/past work/` into compact project
cards.

Live conversations do not scan folders, parse files, or call an extra model for
this feature. Reindexing is explicit and can use embeddings offline; real-time
matching uses the precomputed project cards so recall stays cheap.

Current-employer/client material is treated as guardrail context by default.
Medical, disability, clearance, HR/admin, executables, archives, and large media
are excluded or metadata-only.

Useful endpoints:

```bash
curl http://127.0.0.1:8765/api/work-memory/status
curl -X POST http://127.0.0.1:8765/api/work-memory/reindex -H 'Content-Type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8765/api/work-memory/search -H 'Content-Type: application/json' -d '{"query":"generator monitoring failure modes"}'
```

## Voice Profile

Open the Web UI and use **Voice Profile** to train preferred spellings on your
voice. The guided flow records one short phrase at a time, transcribes it
locally, lets you correct the text, and stores only correction/profile artifacts.
Raw enrollment audio is discarded after transcription.

Once enough phrases are accepted, the profile becomes ready and is used as a
dynamic Faster-Whisper prompt for live sessions. This is ASR personalization, not
full acoustic fine-tuning, so it is fast, local, and reversible.

## Useful Checks

```bash
curl http://127.0.0.1:8765/api/health/gpu
curl http://127.0.0.1:8765/api/devices
```

## Test Audio Fixtures

Fetch small speech snippets and convert them to the app's mono 16 kHz WAV
fixture format:

```bash
./scripts/fetch_test_audio.sh
```

The converted clips land in `runtime/test-audio/`, which is intentionally
ignored by Git. Fixture playback is test-only; set
`BRAIN_SIDECAR_TEST_MODE_ENABLED=1` before passing `fixture_wav` to
`POST /api/sessions/{id}/start`.

To prove the ASR runtime can bind to CUDA without downloading the larger v1
model, run:

```bash
BRAIN_SIDECAR_ASR_PRIMARY_MODEL=tiny.en BRAIN_SIDECAR_ASR_FALLBACK_MODEL=tiny.en python - <<'PY'
import asyncio
from brain_sidecar.config import load_settings
from brain_sidecar.core.transcription import FasterWhisperTranscriber

async def main():
    transcriber = FasterWhisperTranscriber(load_settings())
    await transcriber.load()
    print("loaded CUDA model:", transcriber.model_size)

asyncio.run(main())
PY
```

## Tests

```bash
. .venv/bin/activate
pytest -q
npm --prefix ui run test:e2e
```

Run the live Playwright integration lane against the real backend, Vite UI,
GPU-backed Faster-Whisper runtime, and fixture WAV capture with:

```bash
./scripts/test_e2e_integration.sh
```

The integration script defaults to `runtime/test-audio/osr-us-female-harvard.wav`
for clearer speech, uses `tiny.en` for fast local validation, and sources the
same GPU library path helper used by `scripts/dev.sh`.

Run the user-style validation lane against the polished UI, GPU ASR, wide
screenshots, and a controlled work-memory audio source with:

```bash
./scripts/make_work_memory_known_audio.sh
BRAIN_SIDECAR_USER_TEST_SOURCE=/home/bp/project/brain-sidecar/runtime/user-tests/work-memory-known/source.wav \
BRAIN_SIDECAR_USER_TEST_DIR=/home/bp/project/brain-sidecar/runtime/user-tests/work-memory-known \
BRAIN_SIDECAR_USER_TEST_SCREENSHOT_INTERVAL_MS=20000 \
BRAIN_SIDECAR_EXPECT_WORK_MEMORY=1 \
BRAIN_SIDECAR_EXPECT_RESPONSE_TERMS="Online Generator Monitoring,T.A. Smith,500kV breaker,relay modernization,PG&E gas mains,SaskPower workforce planning" \
BRAIN_SIDECAR_EXPECT_TERM_MIN_HITS=4 \
npm --prefix ui run test:user-audio
```

The report and screenshots land in `runtime/user-tests/work-memory-known/artifacts/`.
