from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from uuid import uuid4
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain_sidecar.config import load_settings, update_dotenv_value
from brain_sidecar.core.devices import list_audio_devices
from brain_sidecar.core.gpu import read_gpu_status
from brain_sidecar.core.meeting_contract import normalize_meeting_contract
from brain_sidecar.core.session import SessionManager
from brain_sidecar.core.test_mode import TestModeService

MAX_INPUT_FILE_BYTES = 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
BROWSER_STREAM_REMOVED = (
    "Browser microphone capture has been removed. Use server_device for capture from the local server microphone."
)
FIXTURE_TEST_MODE_REQUIRED = "Fixture audio is only available when recorded audio test mode is enabled."
SSE_HEARTBEAT = ": heartbeat\n\n"
OLLAMA_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}$")


def encode_sse_event(event) -> str:
    data = json.dumps(event.to_dict(), ensure_ascii=False)
    return f"id: {event.id}\nevent: {event.type}\ndata: {data}\n\n"


class CreateSessionRequest(BaseModel):
    title: str | None = None


class UpdateSessionRequest(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=180)]


class MicTuningRequest(BaseModel):
    auto_level: bool = True
    input_gain_db: float = Field(default=0.0, ge=-12.0, le=24.0)
    speech_sensitivity: Literal["quiet", "normal", "noisy"] = "normal"


class MeetingContractRequest(BaseModel):
    goal: str | None = None
    mode: Literal["quiet", "balanced", "assertive"] = "quiet"
    reminders: list[str] = Field(default_factory=list)


class StartSessionRequest(BaseModel):
    device_id: str | None = None
    fixture_wav: str | None = None
    audio_source: str | None = None
    save_transcript: bool = True
    mic_tuning: MicTuningRequest | None = None
    meeting_contract: MeetingContractRequest | None = None


class LibraryRootRequest(BaseModel):
    path: Annotated[str, Field(min_length=1)]


class RecallSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    limit: int = Field(default=8, ge=1, le=24)


class WorkMemoryReindexRequest(BaseModel):
    roots: list[str] | None = None
    embed: bool = True


class WorkMemorySearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    limit: int = Field(default=5, ge=1, le=12)


class WebContextSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    session_id: str | None = None


class SidecarQueryRequest(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    session_id: str | None = None


class SpeakerSampleRecordRequest(BaseModel):
    device_id: str | None = None
    fixture_wav: str | None = None
    audio_source: str | None = None
    mic_tuning: MicTuningRequest | None = None


class MicTestRequest(BaseModel):
    device_id: str | None = None
    fixture_wav: str | None = None
    audio_source: str | None = None
    seconds: float = Field(default=3.0, ge=1.0, le=8.0)
    mic_tuning: MicTuningRequest | None = None


class SpeakerFeedbackRequest(BaseModel):
    session_id: Annotated[str, Field(min_length=1)]
    segment_id: str | None = None
    old_label: str = ""
    new_label: Annotated[str, Field(min_length=1)]
    feedback_type: Annotated[str, Field(min_length=1)] = "correct_label"


class TestAudioPrepareRequest(BaseModel):
    source_path: Annotated[str, Field(min_length=1)]
    max_seconds: float | None = Field(default=None, gt=0)
    expected_terms: list[str] = Field(default_factory=list, max_length=64)


class TestModeReportRequest(BaseModel):
    report: dict[str, Any] = Field(default_factory=dict)


class OllamaChatModelRequest(BaseModel):
    model: Annotated[str, Field(min_length=1, max_length=160)]


def _safe_upload_name(filename: str | None) -> str:
    original = Path(filename or "input-file").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._")
    return (safe or "input-file")[:160]


def _audio_source_for_request(audio_source: str | None, fixture_wav: Path | None, *, test_mode_enabled: bool) -> str:
    selected = audio_source or ("fixture" if fixture_wav else "server_device")
    if selected == "browser_stream":
        raise HTTPException(status_code=400, detail=BROWSER_STREAM_REMOVED)
    if selected == "fixture" or fixture_wav is not None:
        if not test_mode_enabled:
            raise HTTPException(status_code=403, detail=FIXTURE_TEST_MODE_REQUIRED)
        return "fixture"
    if selected != "server_device":
        raise HTTPException(status_code=400, detail=f"Unsupported audio_source: {selected}")
    return "server_device"


def _mic_tuning_payload(tuning: MicTuningRequest | None) -> dict[str, object] | None:
    if tuning is None:
        return None
    if hasattr(tuning, "model_dump"):
        return tuning.model_dump()
    return tuning.dict()


def _meeting_contract_payload(contract: MeetingContractRequest | None) -> dict[str, object]:
    if contract is None:
        return normalize_meeting_contract().to_dict()
    payload = contract.model_dump() if hasattr(contract, "model_dump") else contract.dict()
    return normalize_meeting_contract(payload).to_dict()


def _normalize_ollama_model_name(model: str) -> str:
    clean_model = model.strip()
    if not OLLAMA_MODEL_NAME_RE.fullmatch(clean_model):
        raise HTTPException(status_code=400, detail="Ollama model names may only use letters, numbers, _, -, ., :, and /.")
    return clean_model


def _ollama_models_payload(settings, models: list[dict[str, Any]], *, error: str | None = None) -> dict:
    selected = settings.ollama_chat_model
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        name = str(model.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        size = model.get("size")
        normalized.append(
            {
                "name": name,
                "size": size if isinstance(size, int) else None,
                "digest": model.get("digest") if isinstance(model.get("digest"), str) else None,
                "modified_at": model.get("modified_at") if isinstance(model.get("modified_at"), str) else None,
                "cloud": name.endswith(":cloud") or size is None,
                "current": name == selected,
            }
        )
    if selected and selected not in seen:
        normalized.insert(
            0,
            {
                "name": selected,
                "size": None,
                "digest": None,
                "modified_at": None,
                "cloud": selected.endswith(":cloud"),
                "current": True,
                "configured": True,
            },
        )
    return {
        "selected_chat_model": selected,
        "chat_host": settings.ollama_chat_host or settings.ollama_host,
        "models": normalized,
        "error": error,
    }


async def _ollama_reachability(manager: SessionManager, chat_host: str, embed_host: str) -> tuple[bool, bool]:
    if chat_host == embed_host:
        reachable = await asyncio.to_thread(manager.ollama.host_reachable, chat_host)
        return reachable, reachable
    chat_task = asyncio.to_thread(manager.ollama.host_reachable, chat_host)
    embed_task = asyncio.to_thread(manager.ollama.host_reachable, embed_host)
    chat_reachable, embed_reachable = await asyncio.gather(chat_task, embed_task)
    return bool(chat_reachable), bool(embed_reachable)


def create_app() -> FastAPI:
    settings = load_settings()
    manager = SessionManager(settings)
    test_mode = TestModeService(settings)

    app = FastAPI(
        title="Brain Sidecar",
        version="0.1.0",
        description="Local live audio transcription, notes, and recall sidecar.",
    )
    app.state.manager = manager
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8766",
            "http://localhost:8766",
            "http://127.0.0.1:8776",
            "http://localhost:8776",
            "https://notes.shoalstone.net",
        ],
        allow_origin_regex=r"https?://([A-Za-z0-9.-]+|\[[0-9A-Fa-f:]+\]):(8766|8776)",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "Brain Sidecar",
            "status": "ready",
            "docs": "/docs",
            "gpu_health": "/api/health/gpu",
        }

    @app.get("/api/devices")
    async def devices() -> dict:
        device_infos = list_audio_devices(probe=True)
        devices = [device.to_dict() for device in device_infos]
        selected_device = next((device for device in device_infos if device.healthy), None)
        return {
            "devices": devices,
            "selected_device": selected_device.to_dict() if selected_device else None,
            "server_mic_available": selected_device is not None,
            "selection_reason": selected_device.selection_reason if selected_device else "No healthy server microphone detected.",
        }

    @app.get("/api/health/gpu")
    async def gpu_health() -> dict:
        status = read_gpu_status().to_dict()
        status["required"] = settings.asr_device == "cuda"
        manager = app.state.manager
        status.update(manager._asr_status_fields(None))
        status["asr_primary_model"] = settings.asr_primary_model
        status["asr_fallback_model"] = settings.asr_fallback_model
        status["asr_device"] = settings.asr_device
        status["asr_backend"] = settings.asr_backend
        status["energy_lens_enabled"] = settings.energy_lens_enabled
        status["energy_lens_keyword_count"] = getattr(manager.energy_detector, "keyword_count", 0)
        status["asr_beam_size"] = settings.asr_beam_size
        status["asr_vad_min_silence_ms"] = settings.asr_vad_min_silence_ms
        status["asr_no_speech_threshold"] = settings.asr_no_speech_threshold
        status["asr_log_prob_threshold"] = settings.asr_log_prob_threshold
        status["asr_compression_ratio_threshold"] = settings.asr_compression_ratio_threshold
        status["asr_min_audio_rms"] = settings.asr_min_audio_rms
        status["partial_transcripts_enabled"] = settings.partial_transcripts_enabled
        status["partial_window_seconds"] = settings.partial_window_seconds
        status["partial_min_interval_seconds"] = settings.partial_min_interval_seconds
        status["ollama_chat_timeout_seconds"] = settings.ollama_chat_timeout_seconds
        status["ollama_embed_timeout_seconds"] = settings.ollama_embed_timeout_seconds
        status["asr_min_free_vram_mb"] = settings.asr_min_free_vram_mb
        status["asr_unload_ollama_on_start"] = settings.asr_unload_ollama_on_start
        status["asr_gpu_free_timeout_seconds"] = settings.asr_gpu_free_timeout_seconds
        status["audio_chunk_ms"] = settings.audio_chunk_ms
        status["server_audio_source"] = "auto_detected_server_device"
        status["transcription_window_seconds"] = settings.transcription_window_seconds
        status["transcription_overlap_seconds"] = settings.transcription_overlap_seconds
        status["transcription_queue_size"] = settings.transcription_queue_size
        status["speaker_enrollment_sample_seconds"] = settings.speaker_enrollment_sample_seconds
        status["speaker_identity_label"] = settings.speaker_identity_label
        status["speaker_retain_raw_enrollment_audio"] = settings.speaker_retain_raw_enrollment_audio
        status["dedupe_similarity_threshold"] = settings.dedupe_similarity_threshold
        status["ollama_chat_model"] = settings.ollama_chat_model
        status["ollama_chat_fallback_model"] = settings.ollama_chat_fallback_model
        status["ollama_chat_min_free_vram_mb"] = settings.ollama_chat_min_free_vram_mb
        status["ollama_embed_model"] = settings.ollama_embed_model
        status["ollama_host"] = settings.ollama_host
        status["ollama_chat_host"] = settings.ollama_chat_host or settings.ollama_host
        status["ollama_embed_host"] = settings.ollama_embed_host or settings.ollama_host
        status["ollama_keep_alive"] = settings.ollama_keep_alive
        status["ollama_chat_keep_alive"] = settings.ollama_chat_keep_alive or settings.ollama_keep_alive
        status["ollama_embed_keep_alive"] = settings.ollama_embed_keep_alive or settings.ollama_keep_alive
        status["ollama_chat_reachable"], status["ollama_embed_reachable"] = await _ollama_reachability(
            app.state.manager,
            status["ollama_chat_host"],
            status["ollama_embed_host"],
        )
        status["web_context_enabled"] = settings.web_context_enabled
        status["web_context_configured"] = bool(settings.brave_search_api_key.strip())
        status["recall_min_score"] = settings.recall_min_score
        status["recall_max_live_hits"] = settings.recall_max_live_hits
        status["recall_prefer_summaries"] = settings.recall_prefer_summaries
        status["notes_every_segments"] = settings.notes_every_segments
        status["sidecar_quality_gate_enabled"] = settings.sidecar_quality_gate_enabled
        status["sidecar_min_evidence_segments"] = settings.sidecar_min_evidence_segments
        status["sidecar_max_cards_per_5min"] = settings.sidecar_max_cards_per_5min
        status["sidecar_max_cards_per_generation_pass"] = settings.sidecar_max_cards_per_generation_pass
        status["energy_lens_max_cards_per_pass"] = settings.energy_lens_max_cards_per_pass
        status["work_memory_job_history_root"] = str(settings.work_memory_job_history_root)
        status["work_memory_past_work_root"] = str(settings.work_memory_past_work_root)
        status["work_memory_pas_root"] = str(settings.work_memory_pas_root) if settings.work_memory_pas_root else None
        status["test_mode_enabled"] = settings.test_mode_enabled
        status["test_audio_run_dir"] = str(settings.test_audio_run_dir)
        return status

    async def _list_ollama_models() -> tuple[list[dict[str, Any]], str | None]:
        chat_host = settings.ollama_chat_host or settings.ollama_host
        try:
            return await asyncio.to_thread(app.state.manager.ollama.list_models, chat_host), None
        except Exception as exc:
            return [], str(exc)

    @app.get("/api/models/ollama")
    async def list_ollama_models() -> dict:
        models, error = await _list_ollama_models()
        return _ollama_models_payload(settings, models, error=error)

    @app.post("/api/models/ollama/chat")
    async def select_ollama_chat_model(request: OllamaChatModelRequest) -> dict:
        model = _normalize_ollama_model_name(request.model)
        models, error = await _list_ollama_models()
        available_names = {str(item.get("name") or "").strip() for item in models}
        available_names.discard("")
        if available_names and model not in available_names:
            raise HTTPException(status_code=400, detail=f"Ollama model is not available on the chat host: {model}")
        if not available_names and model != settings.ollama_chat_model:
            raise HTTPException(status_code=503, detail=error or "Ollama model list is unavailable.")

        object.__setattr__(settings, "ollama_chat_model", model)
        update_dotenv_value("BRAIN_SIDECAR_OLLAMA_CHAT_MODEL", model, settings.env_path)
        return _ollama_models_payload(settings, models, error=error)

    @app.post("/api/input-files")
    async def upload_input_file(file: UploadFile = File(...)) -> dict:
        upload_dir = settings.data_dir / "input-files"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_upload_name(file.filename)
        target = upload_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}-{safe_name}"
        size = 0
        try:
            with target.open("wb") as output:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_INPUT_FILE_BYTES:
                        target.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="Input file is larger than 1 GB.")
                    output.write(chunk)
        finally:
            await file.close()

        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Input file is empty.")
        return {"path": str(target), "filename": safe_name, "size_bytes": size}

    @app.post("/api/test-mode/audio/prepare")
    async def prepare_test_audio(request: TestAudioPrepareRequest) -> dict:
        if not settings.test_mode_enabled:
            raise HTTPException(status_code=403, detail="Recorded audio test mode is disabled.")
        try:
            prepared = await asyncio.to_thread(
                test_mode.prepare_audio,
                Path(request.source_path),
                max_seconds=request.max_seconds,
                expected_terms=request.expected_terms,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return prepared.to_dict()

    @app.post("/api/test-mode/runs/{run_id}/report")
    async def save_test_report(run_id: str, request: TestModeReportRequest) -> dict:
        if not settings.test_mode_enabled:
            raise HTTPException(status_code=403, detail="Recorded audio test mode is disabled.")
        try:
            return await asyncio.to_thread(test_mode.write_report, run_id, request.report)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/web-context/search")
    async def search_web_context(request: WebContextSearchRequest) -> dict:
        return await manager.search_web_context(request.query, session_id=request.session_id)

    @app.post("/api/sidecar/query")
    async def sidecar_query(request: SidecarQueryRequest) -> dict:
        try:
            return await manager.ask_sidecar(request.query, session_id=request.session_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions")
    async def create_session(request: CreateSessionRequest) -> dict:
        return await manager.create_session(request.title)

    @app.get("/api/sessions")
    async def list_sessions(
        limit: int = 50,
        include_empty: bool = True,
        status: str | None = None,
        query: str | None = None,
    ) -> dict:
        return {
            "sessions": await asyncio.to_thread(
                manager.storage.list_sessions,
                limit=max(1, min(limit, 200)),
                include_empty=include_empty,
                status=status,
                query=query,
            )
        }

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str) -> dict:
        try:
            return await asyncio.to_thread(manager.storage.session_detail, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from exc

    @app.patch("/api/sessions/{session_id}")
    async def update_session(session_id: str, request: UpdateSessionRequest) -> dict:
        try:
            return await asyncio.to_thread(manager.storage.update_session_title, session_id, request.title)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/start")
    async def start_session(session_id: str, request: StartSessionRequest) -> dict:
        fixture_wav = Path(request.fixture_wav).expanduser().resolve() if request.fixture_wav else None
        if fixture_wav and not fixture_wav.exists():
            raise HTTPException(status_code=400, detail=f"Fixture WAV does not exist: {fixture_wav}")
        audio_source = _audio_source_for_request(
            request.audio_source,
            fixture_wav,
            test_mode_enabled=settings.test_mode_enabled,
        )
        try:
            await manager.start_session(
                session_id,
                device_id=request.device_id,
                fixture_wav=fixture_wav,
                audio_source=audio_source,
                save_transcript=request.save_transcript,
                mic_tuning=_mic_tuning_payload(request.mic_tuning),
                meeting_contract=_meeting_contract_payload(request.meeting_contract),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "running",
            "session_id": session_id,
            "audio_source": audio_source,
            "save_transcript": request.save_transcript,
            "raw_audio_retained": False,
            "meeting_contract": _meeting_contract_payload(request.meeting_contract),
        }

    @app.websocket("/api/sessions/{session_id}/audio-stream")
    async def browser_audio_stream(session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.close(code=1008, reason=BROWSER_STREAM_REMOVED)

    @app.post("/api/sessions/{session_id}/stop")
    async def stop_session(session_id: str) -> dict:
        await manager.stop_session(session_id)
        return {"status": "stopped", "session_id": session_id, "raw_audio_retained": False}

    async def _event_stream(session_id: str | None, last_event_id: str | None):
        async def stream():
            queue = await manager.bus.subscribe_queue(session_id, replay_after_id=last_event_id)
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield SSE_HEARTBEAT
                        continue
                    yield encode_sse_event(event)
            finally:
                await manager.bus.unsubscribe_queue(queue, session_id)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(
        session_id: str,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        return await _event_stream(session_id, last_event_id)

    @app.get("/api/events")
    async def events(last_event_id: str | None = Header(default=None, alias="Last-Event-ID")) -> StreamingResponse:
        return await _event_stream(None, last_event_id)

    @app.get("/api/speaker/status")
    async def speaker_status() -> dict:
        return await manager.speaker_identity_status()

    @app.post("/api/microphone/test")
    async def microphone_test(request: MicTestRequest) -> dict:
        fixture_wav = Path(request.fixture_wav).expanduser().resolve() if request.fixture_wav else None
        if fixture_wav and not fixture_wav.exists():
            raise HTTPException(status_code=400, detail=f"Fixture WAV does not exist: {fixture_wav}")
        audio_source = _audio_source_for_request(
            request.audio_source,
            fixture_wav,
            test_mode_enabled=settings.test_mode_enabled,
        )
        try:
            return await manager.test_microphone(
                device_id=request.device_id,
                fixture_wav=fixture_wav,
                audio_source=audio_source,
                seconds=request.seconds,
                mic_tuning=_mic_tuning_payload(request.mic_tuning),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/speaker/enrollments")
    async def create_speaker_enrollment() -> dict:
        return await manager.create_speaker_enrollment()

    @app.get("/api/speaker/enrollments/{enrollment_id}")
    async def get_speaker_enrollment(enrollment_id: str) -> dict:
        try:
            return await manager.speaker_enrollment(enrollment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/speaker/enrollments/{enrollment_id}/record")
    async def record_speaker_sample(enrollment_id: str, request: SpeakerSampleRecordRequest) -> dict:
        fixture_wav = Path(request.fixture_wav).expanduser().resolve() if request.fixture_wav else None
        if fixture_wav and not fixture_wav.exists():
            raise HTTPException(status_code=400, detail=f"Fixture WAV does not exist: {fixture_wav}")
        audio_source = _audio_source_for_request(
            request.audio_source,
            fixture_wav,
            test_mode_enabled=settings.test_mode_enabled,
        )
        try:
            return await manager.record_speaker_enrollment_sample(
                enrollment_id,
                device_id=request.device_id,
                fixture_wav=fixture_wav,
                audio_source=audio_source,
                mic_tuning=_mic_tuning_payload(request.mic_tuning),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/api/speaker/enrollments/{enrollment_id}/audio-stream")
    async def browser_speaker_audio_stream(enrollment_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.close(code=1008, reason=BROWSER_STREAM_REMOVED)

    @app.post("/api/speaker/enrollments/{enrollment_id}/finalize")
    async def finalize_speaker_enrollment(enrollment_id: str) -> dict:
        try:
            return await manager.finalize_speaker_enrollment(enrollment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/speaker/profiles/self/recalibrate")
    async def recalibrate_speaker_profile() -> dict:
        try:
            return await manager.recalibrate_speaker_profile()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/speaker/profiles/self/reset")
    async def reset_speaker_profile() -> dict:
        return await manager.reset_speaker_profile()

    @app.post("/api/speaker/feedback")
    async def speaker_feedback(request: SpeakerFeedbackRequest) -> dict:
        return await manager.apply_speaker_feedback(
            session_id=request.session_id,
            segment_id=request.segment_id,
            old_label=request.old_label,
            new_label=request.new_label,
            feedback_type=request.feedback_type,
        )

    @app.api_route("/api/voice/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def legacy_voice_removed(path: str) -> dict:
        raise HTTPException(
            status_code=410,
            detail="Legacy ASR phrase-correction voice training has been removed from the active product path. Use /api/speaker/*.",
        )

    @app.post("/api/library/roots")
    async def add_library_root(request: LibraryRootRequest) -> dict:
        path = Path(request.path).expanduser().resolve()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
        return await manager.add_library_root(path)

    @app.get("/api/library/roots")
    async def list_library_roots() -> dict:
        return {"roots": [str(path) for path in manager.storage.library_roots()]}

    @app.get("/api/library/chunks")
    async def list_library_chunks(
        limit: int = 200,
        query: str | None = None,
        source_path: str | None = None,
    ) -> dict:
        return {
            "sources": manager.storage.document_chunk_sources(limit=limit, query=query),
            "chunks": manager.storage.document_chunks(source_path=source_path, query=query, limit=limit),
        }

    @app.post("/api/library/reindex")
    async def reindex_library() -> dict:
        try:
            report = await manager.recall.reindex_roots()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.to_dict()

    @app.post("/api/recall/search")
    async def recall_search(request: RecallSearchRequest) -> dict:
        try:
            hits = await manager.recall.search(request.query, limit=request.limit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"hits": [hit.to_dict() for hit in hits]}

    @app.get("/api/work-memory/status")
    async def work_memory_status() -> dict:
        return manager.work_memory.status()

    @app.get("/api/work-memory/projects")
    async def work_memory_projects() -> dict:
        projects = manager.storage.work_memory_projects()
        for project in projects:
            project["evidence"] = manager.storage.work_memory_evidence(project["id"], limit=4)
        return {"projects": projects}

    @app.get("/api/work-memory/sources")
    async def work_memory_sources(limit: int = 80) -> dict:
        return {"sources": manager.storage.work_memory_sources(limit=max(1, min(limit, 200)))}

    @app.post("/api/work-memory/reindex")
    async def work_memory_reindex(request: WorkMemoryReindexRequest | None = None) -> dict:
        roots = None
        if request and request.roots:
            roots = [Path(root).expanduser().resolve() for root in request.roots]
            missing = [str(root) for root in roots if not root.exists()]
            if missing:
                raise HTTPException(status_code=400, detail=f"Work memory root does not exist: {missing[0]}")
        try:
            report = await manager.work_memory.reindex(roots=roots, embed=True if request is None else request.embed)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.to_dict()

    @app.post("/api/work-memory/search")
    async def work_memory_search(request: WorkMemorySearchRequest) -> dict:
        try:
            cards = manager.work_memory.search(request.query, limit=request.limit, manual=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"cards": [card.to_dict() for card in cards]}

    @app.on_event("shutdown")
    async def shutdown() -> None:
        active_ids = list(manager._active.keys())
        await asyncio.gather(*(manager.stop_session(session_id) for session_id in active_ids), return_exceptions=True)

    return app


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "brain_sidecar.server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
