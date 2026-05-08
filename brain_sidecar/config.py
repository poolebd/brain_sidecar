from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from brain_sidecar.core.asr import (
    ASR_BACKEND_NEMOTRON_STREAMING,
    validate_asr_backend,
    validate_nemotron_chunk_ms,
    validate_nemotron_device,
    validate_nemotron_dtype,
)

_DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_DOTENV_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@+-]+$")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = _env(name, default).strip().lower()
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {name}={value!r}; expected one of: {allowed_values}.")
    return value


def _load_dotenv(path: Path | None = None) -> None:
    path = path or _DEFAULT_ENV_PATH
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or name.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _configured_env_path() -> Path:
    return Path(os.environ.get("BRAIN_SIDECAR_ENV_PATH", str(_DEFAULT_ENV_PATH))).expanduser()


def _dotenv_literal(value: str) -> str:
    if _DOTENV_SAFE_VALUE_RE.match(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def update_dotenv_value(name: str, value: str, path: Path | None = None) -> None:
    env_path = (path or _configured_env_path()).expanduser()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    prefix_pattern = re.compile(rf"^(\s*(?:export\s+)?{re.escape(name)}\s*=).*$")
    replacement = f"{name}={_dotenv_literal(value)}"
    if not env_path.exists():
        env_path.write_text(f"{replacement}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        match = prefix_pattern.match(line)
        if match:
            lines[index] = f"{match.group(1)}{_dotenv_literal(value)}"
            break
    else:
        lines.append(replacement)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    host: str
    port: int
    asr_primary_model: str
    asr_fallback_model: str
    asr_compute_type: str
    ollama_host: str
    ollama_chat_model: str
    ollama_embed_model: str
    env_path: Path = _DEFAULT_ENV_PATH
    asr_device: str = "cuda"
    ollama_chat_fallback_model: str = ""
    ollama_chat_min_free_vram_mb: int = 0
    ollama_chat_host: str = ""
    ollama_embed_host: str = ""
    ollama_chat_keep_alive: str = ""
    ollama_embed_keep_alive: str = ""
    asr_backend: str = ASR_BACKEND_NEMOTRON_STREAMING
    nemotron_model_id: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_chunk_ms: int = 160
    nemotron_device: str = "cuda"
    nemotron_dtype: str = "float32"
    nemotron_hf_token: str = ""
    nemotron_local_files_only: bool = False
    streaming_partials_enabled: bool = True
    streaming_stable_final_chunks: int = 3
    streaming_min_final_words: int = 10
    streaming_min_final_seconds: float = 2.8
    audio_sample_rate: int = 16_000
    audio_chunk_ms: int = 250
    transcription_window_seconds: float = 3.4
    transcription_overlap_seconds: float = 0.8
    transcription_queue_size: int = 8
    postprocess_queue_size: int = 8
    notes_every_segments: int = 3
    min_segment_chars: int = 8
    dedupe_recent_segments: int = 18
    dedupe_similarity_threshold: float = 0.88
    asr_beam_size: int = 5
    asr_vad_min_silence_ms: int = 300
    asr_condition_on_previous_text: bool = False
    asr_initial_prompt: str | None = None
    asr_no_speech_threshold: float = 0.6
    asr_log_prob_threshold: float = -1.0
    asr_compression_ratio_threshold: float = 2.4
    asr_min_audio_rms: float = 0.006
    partial_transcripts_enabled: bool = False
    partial_window_seconds: float = 2.0
    partial_min_interval_seconds: float = 2.0
    ollama_chat_timeout_seconds: float = 20.0
    ollama_embed_timeout_seconds: float = 12.0
    work_memory_search_timeout_seconds: float = 1.5
    sidecar_quality_gate_enabled: bool = True
    sidecar_min_evidence_segments: int = 2
    sidecar_duplicate_window_seconds: float = 120.0
    sidecar_generic_clarify_window_seconds: float = 300.0
    sidecar_max_cards_per_5min: int = 8
    sidecar_max_cards_per_generation_pass: int = 3
    energy_lens_enabled: bool = True
    energy_lens_min_confidence: str = "medium"
    energy_lens_max_keywords: int = 6
    energy_lens_max_cards_per_pass: int = 1
    asr_min_free_vram_mb: int = 3500
    asr_unload_ollama_on_start: bool = False
    asr_gpu_free_timeout_seconds: float = 10.0
    speaker_enrollment_sample_seconds: float = 8.0
    speaker_identity_label: str = "BP"
    speaker_retain_raw_enrollment_audio: bool = False
    disable_live_embeddings: bool = False
    ollama_keep_alive: str = "30m"
    web_context_enabled: bool = False
    brave_search_api_key: str = ""
    web_context_min_interval_seconds: float = 90.0
    web_context_timeout_seconds: float = 3.0
    web_context_max_results: int = 3
    recall_min_score: float = 0.58
    recall_max_live_hits: int = 4
    recall_prefer_summaries: bool = True
    work_memory_job_history_root: Path = Path("/home/bp/Nextcloud2/Job Hunting")
    work_memory_past_work_root: Path = Path("/home/bp/Nextcloud2/_library/_shoalstone/past work")
    work_memory_pas_root: Path | None = None
    test_mode_enabled: bool = False
    test_audio_run_dir: Path = Path("runtime/test-mode-runs")


def load_settings() -> Settings:
    env_path = _configured_env_path()
    _load_dotenv(env_path)
    cwd_runtime = Path.cwd() / "runtime"
    asr_backend = validate_asr_backend(_env("BRAIN_SIDECAR_ASR_BACKEND", ASR_BACKEND_NEMOTRON_STREAMING))
    ollama_host = _env("BRAIN_SIDECAR_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    ollama_keep_alive = _env("BRAIN_SIDECAR_OLLAMA_KEEP_ALIVE", "30m")
    transcription_window_seconds = max(
        1.0,
        float(_env("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")),
    )
    transcription_overlap_seconds = min(
        max(0.0, float(_env("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", "0.8"))),
        max(0.0, transcription_window_seconds - 0.1),
    )
    return Settings(
        env_path=env_path,
        data_dir=Path(_env("BRAIN_SIDECAR_DATA_DIR", str(cwd_runtime))).expanduser(),
        host=_env("BRAIN_SIDECAR_HOST", "127.0.0.1"),
        port=int(_env("BRAIN_SIDECAR_PORT", "8765")),
        asr_primary_model=_env("BRAIN_SIDECAR_ASR_PRIMARY_MODEL", "medium.en"),
        asr_fallback_model=_env("BRAIN_SIDECAR_ASR_FALLBACK_MODEL", "small.en"),
        asr_compute_type=_env("BRAIN_SIDECAR_ASR_COMPUTE_TYPE", "float16"),
        asr_device=_env_choice("BRAIN_SIDECAR_ASR_DEVICE", "cuda", {"cuda", "cpu"}),
        ollama_host=ollama_host,
        ollama_chat_model=_env("BRAIN_SIDECAR_OLLAMA_CHAT_MODEL", "phi3:mini"),
        ollama_chat_fallback_model=_env("BRAIN_SIDECAR_OLLAMA_CHAT_FALLBACK_MODEL", ""),
        ollama_chat_min_free_vram_mb=max(0, int(_env("BRAIN_SIDECAR_OLLAMA_CHAT_MIN_FREE_VRAM_MB", "0"))),
        ollama_embed_model=_env("BRAIN_SIDECAR_OLLAMA_EMBED_MODEL", "embeddinggemma"),
        ollama_chat_host=_env("BRAIN_SIDECAR_OLLAMA_CHAT_HOST", ollama_host).rstrip("/"),
        ollama_embed_host=_env("BRAIN_SIDECAR_OLLAMA_EMBED_HOST", ollama_host).rstrip("/"),
        ollama_chat_keep_alive=_env("BRAIN_SIDECAR_OLLAMA_CHAT_KEEP_ALIVE", ollama_keep_alive),
        ollama_embed_keep_alive=_env("BRAIN_SIDECAR_OLLAMA_EMBED_KEEP_ALIVE", "0"),
        asr_backend=asr_backend,
        nemotron_model_id=_env("BRAIN_SIDECAR_NEMOTRON_MODEL_ID", "nvidia/nemotron-speech-streaming-en-0.6b"),
        nemotron_chunk_ms=validate_nemotron_chunk_ms(int(_env("BRAIN_SIDECAR_NEMOTRON_CHUNK_MS", "160"))),
        nemotron_device=validate_nemotron_device(_env("BRAIN_SIDECAR_NEMOTRON_DEVICE", "cuda")),
        nemotron_dtype=validate_nemotron_dtype(_env("BRAIN_SIDECAR_NEMOTRON_DTYPE", "float32")),
        nemotron_hf_token=_env("BRAIN_SIDECAR_NEMOTRON_HF_TOKEN", ""),
        nemotron_local_files_only=_env_bool("BRAIN_SIDECAR_NEMOTRON_LOCAL_FILES_ONLY", False),
        streaming_partials_enabled=_env_bool(
            "BRAIN_SIDECAR_STREAMING_PARTIALS_ENABLED",
            asr_backend == ASR_BACKEND_NEMOTRON_STREAMING,
        ),
        streaming_stable_final_chunks=max(1, int(_env("BRAIN_SIDECAR_STREAMING_STABLE_FINAL_CHUNKS", "3"))),
        streaming_min_final_words=max(4, int(_env("BRAIN_SIDECAR_STREAMING_MIN_FINAL_WORDS", "10"))),
        streaming_min_final_seconds=max(
            1.0,
            float(_env("BRAIN_SIDECAR_STREAMING_MIN_FINAL_SECONDS", "2.8")),
        ),
        audio_chunk_ms=max(50, int(_env("BRAIN_SIDECAR_AUDIO_CHUNK_MS", "250"))),
        transcription_window_seconds=transcription_window_seconds,
        transcription_overlap_seconds=transcription_overlap_seconds,
        transcription_queue_size=max(1, int(_env("BRAIN_SIDECAR_TRANSCRIPTION_QUEUE_SIZE", "8"))),
        postprocess_queue_size=max(1, int(_env("BRAIN_SIDECAR_POSTPROCESS_QUEUE_SIZE", "8"))),
        notes_every_segments=max(1, int(_env("BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS", "3"))),
        min_segment_chars=max(1, int(_env("BRAIN_SIDECAR_MIN_SEGMENT_CHARS", "8"))),
        dedupe_recent_segments=max(1, int(_env("BRAIN_SIDECAR_DEDUPE_RECENT_SEGMENTS", "18"))),
        dedupe_similarity_threshold=float(_env("BRAIN_SIDECAR_DEDUPE_SIMILARITY_THRESHOLD", "0.88")),
        asr_beam_size=int(_env("BRAIN_SIDECAR_ASR_BEAM_SIZE", "5")),
        asr_vad_min_silence_ms=int(_env("BRAIN_SIDECAR_ASR_VAD_MIN_SILENCE_MS", "300")),
        asr_condition_on_previous_text=_env_bool("BRAIN_SIDECAR_ASR_CONDITION_ON_PREVIOUS_TEXT", False),
        asr_initial_prompt=os.environ.get("BRAIN_SIDECAR_ASR_INITIAL_PROMPT"),
        asr_no_speech_threshold=float(_env("BRAIN_SIDECAR_ASR_NO_SPEECH_THRESHOLD", "0.6")),
        asr_log_prob_threshold=float(_env("BRAIN_SIDECAR_ASR_LOG_PROB_THRESHOLD", "-1.0")),
        asr_compression_ratio_threshold=float(_env("BRAIN_SIDECAR_ASR_COMPRESSION_RATIO_THRESHOLD", "2.4")),
        asr_min_audio_rms=max(0.0, float(_env("BRAIN_SIDECAR_ASR_MIN_AUDIO_RMS", "0.006"))),
        partial_transcripts_enabled=_env_bool("BRAIN_SIDECAR_PARTIAL_TRANSCRIPTS_ENABLED", False),
        partial_window_seconds=min(
            transcription_window_seconds,
            max(0.5, float(_env("BRAIN_SIDECAR_PARTIAL_WINDOW_SECONDS", "2.0"))),
        ),
        partial_min_interval_seconds=max(
            0.5,
            float(_env("BRAIN_SIDECAR_PARTIAL_MIN_INTERVAL_SECONDS", "2.0")),
        ),
        asr_min_free_vram_mb=max(0, int(_env("BRAIN_SIDECAR_ASR_MIN_FREE_VRAM_MB", "3500"))),
        asr_unload_ollama_on_start=_env_bool("BRAIN_SIDECAR_ASR_UNLOAD_OLLAMA_ON_START", False),
        asr_gpu_free_timeout_seconds=max(
            0.0,
            float(_env("BRAIN_SIDECAR_ASR_GPU_FREE_TIMEOUT_SECONDS", "10")),
        ),
        ollama_chat_timeout_seconds=max(
            0.5,
            float(_env("BRAIN_SIDECAR_OLLAMA_CHAT_TIMEOUT_SECONDS", "20")),
        ),
        ollama_embed_timeout_seconds=max(
            0.5,
            float(_env("BRAIN_SIDECAR_OLLAMA_EMBED_TIMEOUT_SECONDS", "12")),
        ),
        work_memory_search_timeout_seconds=max(
            0.1,
            float(_env("BRAIN_SIDECAR_WORK_MEMORY_SEARCH_TIMEOUT_SECONDS", "1.5")),
        ),
        sidecar_quality_gate_enabled=_env_bool("BRAIN_SIDECAR_SIDECAR_QUALITY_GATE_ENABLED", True),
        sidecar_min_evidence_segments=max(1, int(_env("BRAIN_SIDECAR_SIDECAR_MIN_EVIDENCE_SEGMENTS", "2"))),
        sidecar_duplicate_window_seconds=max(
            1.0,
            float(_env("BRAIN_SIDECAR_SIDECAR_DUPLICATE_WINDOW_SECONDS", "120")),
        ),
        sidecar_generic_clarify_window_seconds=max(
            1.0,
            float(_env("BRAIN_SIDECAR_SIDECAR_GENERIC_CLARIFY_WINDOW_SECONDS", "300")),
        ),
        sidecar_max_cards_per_5min=max(1, int(_env("BRAIN_SIDECAR_SIDECAR_MAX_CARDS_PER_5MIN", "8"))),
        sidecar_max_cards_per_generation_pass=max(
            1,
            int(_env("BRAIN_SIDECAR_SIDECAR_MAX_CARDS_PER_GENERATION_PASS", "3")),
        ),
        energy_lens_enabled=_env_bool("BRAIN_SIDECAR_ENERGY_LENS_ENABLED", True),
        energy_lens_min_confidence=_env("BRAIN_SIDECAR_ENERGY_LENS_MIN_CONFIDENCE", "medium").strip().lower(),
        energy_lens_max_keywords=max(1, int(_env("BRAIN_SIDECAR_ENERGY_LENS_MAX_KEYWORDS", "6"))),
        energy_lens_max_cards_per_pass=max(1, int(_env("BRAIN_SIDECAR_ENERGY_LENS_MAX_CARDS_PER_PASS", "1"))),
        speaker_enrollment_sample_seconds=max(
            2.0,
            float(_env("BRAIN_SIDECAR_SPEAKER_ENROLLMENT_SAMPLE_SECONDS", "8.0")),
        ),
        speaker_identity_label=_env("BRAIN_SIDECAR_SPEAKER_IDENTITY_LABEL", "BP"),
        speaker_retain_raw_enrollment_audio=_env_bool("BRAIN_SIDECAR_SPEAKER_RETAIN_RAW_ENROLLMENT_AUDIO", False),
        disable_live_embeddings=_env_bool("BRAIN_SIDECAR_DISABLE_LIVE_EMBEDDINGS", False),
        ollama_keep_alive=ollama_keep_alive,
        web_context_enabled=_env_bool("BRAIN_SIDECAR_WEB_CONTEXT_ENABLED", False),
        brave_search_api_key=_env("BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY", ""),
        web_context_min_interval_seconds=max(
            0.0,
            float(_env("BRAIN_SIDECAR_WEB_CONTEXT_MIN_INTERVAL_SECONDS", "90")),
        ),
        web_context_timeout_seconds=max(
            0.1,
            float(_env("BRAIN_SIDECAR_WEB_CONTEXT_TIMEOUT_SECONDS", "3.0")),
        ),
        web_context_max_results=max(1, int(_env("BRAIN_SIDECAR_WEB_CONTEXT_MAX_RESULTS", "3"))),
        recall_min_score=max(0.0, min(1.0, float(_env("BRAIN_SIDECAR_RECALL_MIN_SCORE", "0.58")))),
        recall_max_live_hits=max(0, int(_env("BRAIN_SIDECAR_RECALL_MAX_LIVE_HITS", "4"))),
        recall_prefer_summaries=_env_bool("BRAIN_SIDECAR_RECALL_PREFER_SUMMARIES", True),
        work_memory_job_history_root=Path(
            _env("BRAIN_SIDECAR_WORK_MEMORY_JOB_HISTORY_ROOT", "/home/bp/Nextcloud2/Job Hunting")
        ).expanduser(),
        work_memory_past_work_root=Path(
            _env(
                "BRAIN_SIDECAR_WORK_MEMORY_PAST_WORK_ROOT",
                "/home/bp/Nextcloud2/_library/_shoalstone/past work",
            )
        ).expanduser(),
        work_memory_pas_root=(
            Path(os.environ["BRAIN_SIDECAR_WORK_MEMORY_PAS_ROOT"]).expanduser()
            if os.environ.get("BRAIN_SIDECAR_WORK_MEMORY_PAS_ROOT")
            else None
        ),
        test_mode_enabled=_env_bool("BRAIN_SIDECAR_TEST_MODE_ENABLED", False),
        test_audio_run_dir=Path(
            _env("BRAIN_SIDECAR_TEST_AUDIO_RUN_DIR", str(cwd_runtime / "test-mode-runs"))
        ).expanduser(),
    )
