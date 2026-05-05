# Brain Sidecar

Brain Sidecar is an all-local live audio companion for transcription, topic notes,
action capture, and recall. The first UI is a local browser app, but the core is
headless and event-driven so a desktop GUI can reuse the same engine later.

## What v1 Does

- Captures a USB mic through `ffmpeg`.
- Transcribes locally with Faster-Whisper on CUDA only.
- Generates meeting-intelligence sidecar cards with local Ollama.
- Searches prior sessions and selected local files with local embeddings.
- Surfaces fast work-memory parallels from a pre-indexed local career/project archive.
- Learns local Speaker Identity embeddings so BP-owned follow-ups can be distinguished from other speakers.
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
- `BRAIN_SIDECAR_OLLAMA_CHAT_TIMEOUT_SECONDS`: local note/card synthesis timeout, default `20`.
- `BRAIN_SIDECAR_OLLAMA_EMBED_TIMEOUT_SECONDS`: local embedding timeout, default `12`.
- `BRAIN_SIDECAR_TRANSCRIPTION_QUEUE_SIZE`: number of pending audio windows before stale windows are dropped, default `8`.
- `BRAIN_SIDECAR_PARTIAL_TRANSCRIPTS_ENABLED`: publish guarded provisional transcript previews, default `false`. Preview ASR is opt-in because final windows must never wait behind preview work.
- `BRAIN_SIDECAR_PARTIAL_WINDOW_SECONDS`: preview ASR window length, capped to the final ASR window, default `2.0`.
- `BRAIN_SIDECAR_PARTIAL_MIN_INTERVAL_SECONDS`: minimum spacing between preview ASR jobs, default `2.0`.
- `BRAIN_SIDECAR_DEDUPE_SIMILARITY_THRESHOLD`: suppresses repeated text from overlapping windows.
- `BRAIN_SIDECAR_ASR_INITIAL_PROMPT`: optional vocabulary/context hint for names, projects, or jargon.
- `BRAIN_SIDECAR_SPEAKER_ENROLLMENT_SAMPLE_SECONDS`: USB-mic recording window for each Speaker Identity sample, default `8`.
- `BRAIN_SIDECAR_SPEAKER_IDENTITY_LABEL`: display label for BP's own voice, default `BP`.
- `BRAIN_SIDECAR_SPEAKER_RETAIN_RAW_ENROLLMENT_AUDIO`: must remain `false` unless explicitly changing the privacy contract; default `false`.
- `BRAIN_SIDECAR_RECALL_MIN_SCORE`: minimum passive live recall score, default `0.58`.
- `BRAIN_SIDECAR_RECALL_MAX_LIVE_HITS`: max passive live recall cards, default `4`.
- `BRAIN_SIDECAR_RECALL_PREFER_SUMMARIES`: prefer saved-session summaries over raw transcript snippets when scores are comparable, default `true`.
- `BRAIN_SIDECAR_WORK_MEMORY_JOB_HISTORY_ROOT`: configured job-history root, default `/home/bp/Nextcloud2/Job Hunting`.
- `BRAIN_SIDECAR_WORK_MEMORY_PAST_WORK_ROOT`: configured past-work root, default `/home/bp/Nextcloud2/_library/_shoalstone/past work`.
- `BRAIN_SIDECAR_WORK_MEMORY_PAS_ROOT`: optional distinct PAS archive root.
- `BRAIN_SIDECAR_WORK_MEMORY_SEARCH_TIMEOUT_SECONDS`: live work-memory search budget, default `1.5`.
- `BRAIN_SIDECAR_WEB_CONTEXT_ENABLED`: set to `true` to allow selective Brave web context for explicit current-tech questions.
- `BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY`: Brave Search API key; without this, web context stays disabled.
- `BRAIN_SIDECAR_WEB_CONTEXT_MIN_INTERVAL_SECONDS`: cooldown between live web lookups, default `90`.
- `BRAIN_SIDECAR_WEB_CONTEXT_TIMEOUT_SECONDS`: hard Brave request timeout, default `3.0`.
- `BRAIN_SIDECAR_WEB_CONTEXT_MAX_RESULTS`: maximum Brave Web Search results used per note, default `3`.

Web context uses Brave Web Search only. It sends a compact sanitized query, never
raw transcript windows, and publishes results as live-only sidecar cards. Skipped
searches carry a reason such as `web_disabled`, `brave_key_missing`,
`cooldown_active`, `query_empty_after_sanitization`,
`query_looked_private_or_internal`, `query_lacked_public_current_technical_signals`,
`request_timeout`, or `no_results`. Web cards are not stored in SQLite,
embedded, or added to recall indexes.

## Live Meeting Workflow

Use **Listen** for temporary, listen-only meetings and **Record** only when the
transcript should become future recall material. The main view is organized for
a quick meeting glance:

- Live Transcript shows final transcript lines and, when explicitly enabled,
  gray provisional partials.
- Contribution Lane shows the best current "say this," "ask this," risk,
  follow-up, memory, work-memory, and web cards.
- Work Notes keeps the broader action/decision/question/context buckets.
- Tools keeps operational status, GPU details, indexing, and speaker identity
  controls away from the transcript.

Capture is server-mic-only in the product path. `server_device` records from the
best healthy ALSA capture device on this computer, using `plughw` so mono USB
mics can open cleanly. The UI does not pin a particular brand or ALSA card; it
shows the auto-selected server mic plus Guided Auto controls for input boost,
speech sensitivity, reset, and mic-test playback. `fixture` is available only in
explicit test mode, and browser microphone streaming is rejected to keep capture
local to the trusted machine and avoid remote/browser mic drift.

## Event Contract

SSE session streams emit compatibility events plus the normalized sidecar card
contract:

- `transcript_partial`: provisional ASR preview. It has the same core payload as
  final transcript events, sets `is_final: false`, is never persisted, and is
  replaced/reconciled by overlapping final lines in the UI. Preview metrics are
  exposed as `partial_windows_enqueued`, `partial_windows_skipped_busy`,
  `partial_windows_skipped_queue`, `partial_windows_dropped_for_final`, and
  `partial_asr_duration_ms`.
- `transcript_final`: authoritative transcript segment. It is persisted only
  when the active session is in saved/recording mode.
- `note_update` and `recall_hit`: existing compatibility events consumed by the
  UI and external clients.
- `sidecar_card`: normalized meeting-assistant card with category, title, body,
  optional suggested say/ask text, `why_now`, priority, confidence,
  `source_segment_ids`, `source_type`, sources/citations, `card_key`,
  ephemeral/expiry metadata, and `raw_audio_retained: false`.
- `audio_status` and `gpu_status`: operational state such as queue depth,
  dropped windows, preview metrics, event-drop counts, ASR settings, and GPU
  pressure. SSE streams send heartbeats and keep a bounded replay buffer for
  reconnects using `Last-Event-ID`.

Raw audio is never written to disk. Temporary/listen-only sessions do not store
transcript text, note cards, embeddings, diarization rows, work-memory recall
events, memory summaries, or web context. Web cards remain sanitized,
Brave-backed, live-only, and ephemeral.

## Work Memory

The Work Memory layer is for PAS/past-work recall support during meetings. It
uses high-signal historical documents under the configured job-history root as a
canonical timeline, then correlates evidence from the configured past-work and
optional PAS roots into compact project cards.

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

## Speaker Identity

Open the Web UI and use **Speaker Identity** to teach the sidecar BP's voice from
single-speaker USB-mic samples. The flow stores local speaker embeddings and
discarded-audio quality metadata only; raw enrollment audio is not retained.
Run **Test Mic** in the Input controls before recording samples. The mic check
captures a short in-memory sample from the selected USB device, exposes an
in-browser playback preview, and reports peak level, clipping, usable speech,
quality, and a plain recommendation without writing audio to disk. Enrollment
finalization uses only the current training
attempt and can ignore an outlier sample when enough consistent single-speaker
speech remains, so a failed attempt should not poison the next one.

When ready, transcript events can label high-confidence BP speech with
`speaker_role: "user"` and the configured label. Meeting-intelligence cards use
that metadata conservatively: BP-owned first-person commitments can become
follow-ups, other-speaker commitments do not become BP actions, and unknown or
low-confidence speakers are worded cautiously.

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
