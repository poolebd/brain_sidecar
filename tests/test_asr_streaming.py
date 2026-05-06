from __future__ import annotations

import asyncio
import builtins
import sys
from pathlib import Path

import pytest

import brain_sidecar.config as config
from brain_sidecar.config import Settings, load_settings
from brain_sidecar.core.asr import (
    ASR_BACKEND_NEMOTRON_STREAMING,
    StreamingAsrEvent,
)
from brain_sidecar.core.asr_factory import create_asr_backend
from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.events import EVENT_TRANSCRIPT_FINAL, EVENT_TRANSCRIPT_PARTIAL, SidecarEvent
from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.nemotron_streaming import (
    StablePrefixFinalizer,
    StreamingPcmChunker,
    NemotronStreamingTranscriber,
)
from brain_sidecar.core.session import ActiveSession, AudioWindow, InMemoryPcmRingBuffer, SessionManager


class DummyCapture(AudioCapture):
    async def chunks(self):
        if False:
            yield b""

    async def stop(self) -> None:
        return None


class FakeStreamingSession:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def accept_pcm16(self, pcm: bytes, start_offset_s: float) -> list[StreamingAsrEvent]:
        self.calls += 1
        return [
            StreamingAsrEvent(
                kind="partial",
                text="Confirm the relay",
                start_s=start_offset_s,
                end_s=start_offset_s + 0.56,
                model="fake-nemotron",
            ),
            StreamingAsrEvent(
                kind="final",
                text="Confirm the relay settings.",
                start_s=start_offset_s,
                end_s=start_offset_s + 0.56,
                model="fake-nemotron",
            ),
        ]

    async def flush(self, final_offset_s: float | None = None) -> list[StreamingAsrEvent]:
        return []

    async def close(self) -> None:
        self.closed = True


class FakeStreamingBackend:
    backend_name = ASR_BACKEND_NEMOTRON_STREAMING
    streaming_supported = True
    model_size = "fake-nemotron"
    last_error = None

    def __init__(self) -> None:
        self.session = FakeStreamingSession()

    async def load(self) -> None:
        return None

    async def open_stream(self, session_id: str, start_offset_s: float = 0.0) -> FakeStreamingSession:
        return self.session


def make_settings(tmp_path: Path, *, backend: str = "faster_whisper") -> Settings:
    return Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
        asr_backend=backend,
        disable_live_embeddings=True,
    )


def active_session(session_id: str, *, save_transcript: bool) -> ActiveSession:
    return ActiveSession(
        id=session_id,
        capture=DummyCapture(),
        window_queue=asyncio.Queue(maxsize=4),
        postprocess_queue=asyncio.Queue(maxsize=4),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
        save_transcript=save_transcript,
        asr_backend=ASR_BACKEND_NEMOTRON_STREAMING,
        asr_model="fake-nemotron",
        streaming_chunk_ms=560,
        pcm_ring_buffer=InMemoryPcmRingBuffer(sample_rate=16_000, seconds=5.0),
    )


def test_asr_backend_config_defaults_to_faster_whisper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.delenv("BRAIN_SIDECAR_ASR_BACKEND", raising=False)

    settings = load_settings()

    assert settings.asr_backend == "faster_whisper"
    assert settings.nemotron_chunk_ms == 160


def test_asr_backend_config_accepts_nemotron(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_ASR_BACKEND", "nemotron_streaming")
    monkeypatch.setenv("BRAIN_SIDECAR_NEMOTRON_CHUNK_MS", "1120")

    settings = load_settings()

    assert settings.asr_backend == "nemotron_streaming"
    assert settings.nemotron_chunk_ms == 1120
    assert settings.nemotron_dtype == "float32"
    assert settings.streaming_partials_enabled is True


def test_asr_backend_config_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_ASR_BACKEND", "cloud")

    with pytest.raises(ValueError, match="Unsupported BRAIN_SIDECAR_ASR_BACKEND"):
        load_settings()

    monkeypatch.setenv("BRAIN_SIDECAR_ASR_BACKEND", "nemotron_streaming")
    monkeypatch.setenv("BRAIN_SIDECAR_NEMOTRON_CHUNK_MS", "640")
    with pytest.raises(ValueError, match="Unsupported BRAIN_SIDECAR_NEMOTRON_CHUNK_MS"):
        load_settings()

    monkeypatch.setenv("BRAIN_SIDECAR_NEMOTRON_CHUNK_MS", "560")
    monkeypatch.setenv("BRAIN_SIDECAR_NEMOTRON_DTYPE", "bfloat16")
    with pytest.raises(ValueError, match="Unsupported BRAIN_SIDECAR_NEMOTRON_DTYPE"):
        load_settings()


def test_asr_factory_uses_lazy_nemotron_backend(tmp_path: Path) -> None:
    sys.modules.pop("nemo", None)
    sys.modules.pop("nemo.collections.asr", None)

    backend = create_asr_backend(make_settings(tmp_path, backend="nemotron_streaming"))

    assert isinstance(backend, NemotronStreamingTranscriber)
    assert "nemo.collections.asr" not in sys.modules


def test_streaming_chunker_reframes_pcm_and_flushes_remainder() -> None:
    chunker = StreamingPcmChunker(sample_rate=16_000, chunk_ms=80)
    chunks = chunker.accept(b"\1" * 4000, 1.0)

    assert len(chunks) == 1
    assert len(chunks[0].pcm) == 2560
    assert chunks[0].start_offset_s == 1.0
    assert chunks[0].end_offset_s == pytest.approx(1.08)

    remainder = chunker.flush()
    assert len(remainder) == 1
    assert len(remainder[0].pcm) == 1440
    assert remainder[0].final is True


def test_stable_prefix_finalizer_emits_partials_and_nonduplicated_finals() -> None:
    finalizer = StablePrefixFinalizer(stable_chunks=2, partials_enabled=True)

    first = finalizer.accept_text("Confirm the relay", start_s=0.0, end_s=0.56, model="fake")
    second = finalizer.accept_text("Confirm the relay settings", start_s=0.56, end_s=1.12, model="fake")
    flushed = finalizer.flush(final_offset_s=1.5, model="fake")

    assert [event.kind for event in first] == ["partial"]
    assert [event.text for event in second if event.kind == "final"] == ["Confirm the relay"]
    assert [event.text for event in flushed] == ["settings"]


def test_nemotron_load_failure_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("brain_sidecar.core.nemotron_streaming.prepare_asr_gpu", lambda _settings: None)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("nemo"):
            raise ImportError("missing nemo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    transcriber = NemotronStreamingTranscriber(make_settings(tmp_path, backend="nemotron_streaming"))

    with pytest.raises(RuntimeError, match="NVIDIA NeMo ASR"):
        transcriber._load_sync()


def test_streaming_session_path_publishes_partial_and_final_without_raw_audio(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path, backend="nemotron_streaming"))
    manager.transcriber = FakeStreamingBackend()  # type: ignore[assignment]
    session = manager.storage.create_session("streaming")
    active = active_session(session.id, save_transcript=True)
    active.pcm_ring_buffer.append(b"\1" * 17_920, 0.0)
    manager._active[session.id] = active

    partial_collector = event_loop.create_task(collect_event(manager, session.id, EVENT_TRANSCRIPT_PARTIAL))
    final_collector = event_loop.create_task(collect_event(manager, session.id, EVENT_TRANSCRIPT_FINAL))
    task = event_loop.create_task(
        manager._run_streaming_transcription_loop(session.id, active.window_queue, active.postprocess_queue)
    )
    active.window_queue.put_nowait(AudioWindow(b"\1" * 17_920, 0.0))
    active.window_queue.put_nowait(AudioWindow(b"", 0.56, final=True))

    partial = event_loop.run_until_complete(asyncio.wait_for(partial_collector, timeout=1.0))
    final = event_loop.run_until_complete(asyncio.wait_for(final_collector, timeout=1.0))
    event_loop.run_until_complete(asyncio.wait_for(task, timeout=1.0))

    assert partial.payload["is_final"] is False
    assert partial.payload["raw_audio_retained"] is False
    assert final.payload["text"] == "Confirm the relay settings."
    assert final.payload["asr_backend"] == "nemotron_streaming"
    assert final.payload["raw_audio_retained"] is False
    assert manager.storage.recent_segments(session.id)[0].text == "Confirm the relay settings."


def test_listen_only_streaming_final_does_not_persist(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path, backend="nemotron_streaming"))
    session = manager.storage.create_session("listen-only")
    active = active_session(session.id, save_transcript=False)
    manager._active[session.id] = active

    batch = event_loop.run_until_complete(
        manager._handle_streaming_asr_event(
            session.id,
            active,
            StreamingAsrEvent(
                kind="final",
                text="Do not persist this.",
                start_s=0.0,
                end_s=1.0,
                model="fake-nemotron",
            ),
            asr_duration_ms=12.0,
        )
    )

    assert [segment.text for segment in batch.segments] == ["Do not persist this."]
    assert manager.storage.recent_segments(session.id) == []


async def collect_event(manager: SessionManager, session_id: str, event_type: str) -> SidecarEvent:
    async for event in manager.bus.subscribe(session_id):
        if event.type == event_type:
            return event
    raise AssertionError("subscription ended")
