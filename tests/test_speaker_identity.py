from __future__ import annotations

import base64
import io
import math
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain_sidecar.config import Settings
from brain_sidecar.core.speaker_identity import (
    SpeakerBackendStatus,
    SpeakerIdentityService,
    analyze_pcm16,
    calibrate_threshold,
    cosine_similarity,
    l2_normalize,
)
from brain_sidecar.core.storage import Storage
from brain_sidecar.server.app import create_app


class QueueSpeakerBackend:
    model_name = "queue-speaker-test"
    model_version = "test-v1"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = list(vectors)

    def status(self) -> SpeakerBackendStatus:
        return SpeakerBackendStatus(True, self.model_name, self.model_version, "cpu")

    def embed_pcm16(self, pcm: bytes, sample_rate: int = 16_000) -> list[float]:
        if not self.vectors:
            raise RuntimeError("No queued speaker embedding remains.")
        return l2_normalize(self.vectors.pop(0))


def make_settings(tmp_path: Path) -> Settings:
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
        transcription_queue_size=1,
        postprocess_queue_size=1,
        speaker_enrollment_sample_seconds=8.0,
    )


def test_embedding_math_and_threshold_are_conservative() -> None:
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([2.0, 0.0], [1.0, 0.0]) == 1.0
    assert calibrate_threshold([0.91, 0.89, 0.93], [0.72, 0.74]) >= 0.82


def test_speaker_enrollment_creates_centroid_without_raw_audio(tmp_path: Path) -> None:
    service = speaker_service(
        tmp_path,
        [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.99, 0.01, 0.0]],
    )
    enrollment = service.start_enrollment()
    pcm = tone_pcm(seconds=8.0)

    for _ in range(3):
        service.add_enrollment_sample(enrollment["id"], pcm)

    finalized = service.finalize_enrollment(enrollment["id"])
    status = finalized["profile"]

    assert status["ready"] is True
    assert status["profile"]["display_name"] == "BP"
    assert status["embedding_count"] == 3
    assert status["raw_audio_retained"] is False
    assert finalized["centroid_embedding_id"]
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob("*.pcm"))


def test_speaker_enrollment_finalizes_current_attempt_without_old_bad_samples(tmp_path: Path) -> None:
    service = speaker_service(
        tmp_path,
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.99, 0.01, 0.0]],
    )
    pcm = tone_pcm(seconds=8.0)
    old_enrollment = service.start_enrollment()
    service.add_enrollment_sample(old_enrollment["id"], pcm)

    enrollment = service.start_enrollment()
    service.add_enrollment_sample(enrollment["id"], pcm)
    service.add_enrollment_sample(enrollment["id"], pcm)
    finalized = service.finalize_enrollment(enrollment["id"])

    assert finalized["profile"]["ready"] is True
    assert finalized["profile"]["embedding_count"] == 2
    assert finalized["pruned"]["embeddings"] == 1


def test_speaker_enrollment_ignores_outlier_when_consistent_speech_remains(tmp_path: Path) -> None:
    service = speaker_service(
        tmp_path,
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0]],
    )
    enrollment = service.start_enrollment()
    pcm = tone_pcm(seconds=8.0)

    for _ in range(3):
        service.add_enrollment_sample(enrollment["id"], pcm)

    finalized = service.finalize_enrollment(enrollment["id"])

    assert finalized["profile"]["ready"] is True
    assert finalized["profile"]["embedding_count"] == 2
    assert len(finalized["ignored_embedding_ids"]) == 1
    assert finalized["raw_audio_retained"] is False


def test_audio_quality_counts_steady_training_speech_not_only_loud_peaks() -> None:
    quality = analyze_pcm16(pulsed_training_pcm(seconds=8.0))

    assert quality.duration_seconds == pytest.approx(8.0, abs=0.05)
    assert quality.usable_speech_seconds >= 7.5
    assert "too_much_silence" not in quality.issues


def test_audio_quality_ignores_low_level_room_noise() -> None:
    quality = analyze_pcm16(tone_pcm(seconds=3.0, amplitude=0.006))

    assert quality.usable_speech_seconds == 0
    assert "very_low_volume" in quality.issues


def test_runtime_labels_matching_cluster_as_bp_only_above_threshold(tmp_path: Path) -> None:
    service = speaker_service(
        tmp_path,
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    enrollment = service.start_enrollment()
    pcm = tone_pcm(seconds=8.0)
    for _ in range(3):
        service.add_enrollment_sample(enrollment["id"], pcm)
    service.finalize_enrollment(enrollment["id"])

    me = service.label_segment(
        session_id="ses-test",
        segment_id="seg-me",
        pcm=pcm,
        start_ms=0,
        end_ms=1600,
        transcript_text="I am speaking now.",
    )
    other = service.label_segment(
        session_id="ses-test",
        segment_id="seg-other",
        pcm=pcm,
        start_ms=1800,
        end_ms=3500,
        transcript_text="Someone else is speaking.",
    )

    assert me.display_speaker_label == "BP"
    assert me.speaker_role == "user"
    assert me.match_score is not None and me.match_score >= 0.82
    assert other.display_speaker_label == "Speaker 1"
    assert other.matched_profile_id is None
    assert other.speaker_role == "other"


def test_anonymous_speaker_numbering_is_stable_within_session(tmp_path: Path) -> None:
    service = speaker_service(
        tmp_path,
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01],
        ],
    )
    enrollment = service.start_enrollment()
    pcm = tone_pcm(seconds=8.0)
    for _ in range(3):
        service.add_enrollment_sample(enrollment["id"], pcm)
    service.finalize_enrollment(enrollment["id"])

    first = service.label_segment(
        session_id="ses-stable",
        segment_id="seg-a",
        pcm=pcm,
        start_ms=0,
        end_ms=1800,
    )
    second = service.label_segment(
        session_id="ses-stable",
        segment_id="seg-b",
        pcm=pcm,
        start_ms=2200,
        end_ms=3900,
    )

    assert first.display_speaker_label == "Speaker 1"
    assert second.display_speaker_label == "Speaker 1"


def test_bad_legacy_asr_corrections_can_be_quarantined(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.connect()
    profile = storage.ensure_voice_profile()
    enrollment = storage.create_voice_enrollment(["test"], profile["id"])
    phrase = enrollment["phrases"][0]
    storage.save_voice_phrase_transcript(enrollment["id"], phrase["id"], "transcription")
    storage.accept_voice_phrase(
        enrollment["id"],
        phrase["id"],
        "transposition",
        [],
        {"artifact_type": "test"},
    )

    assert storage.accepted_voice_phrase_count() == 1
    removed = storage.quarantine_voice_corrections([("transcription", "transposition")])

    assert removed == 1
    assert storage.voice_corrections() == []
    assert storage.accepted_voice_phrase_count() == 0


def test_speaker_status_api_exposes_new_profile_and_legacy_voice_api_is_gone(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    status = client.get("/api/speaker/status")
    assert status.status_code == 200
    assert status.json()["profile"]["id"] == "self_bp"
    assert status.json()["raw_audio_retained"] is False

    legacy = client.get("/api/voice/profile")
    assert legacy.status_code == 410


def test_microphone_test_endpoint_reports_quality_without_raw_audio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_MODE_ENABLED", "true")
    app = create_app()
    wav_path = tmp_path / "mic-check.wav"
    write_wav(wav_path, tone_pcm(seconds=3.0))

    response = TestClient(app).post(
        "/api/microphone/test",
        json={
            "audio_source": "fixture",
            "fixture_wav": str(wav_path),
            "seconds": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_audio_retained"] is False
    assert payload["quality"]["usable_speech_seconds"] > 0
    assert payload["recommendation"]["status"] in {"good", "noisy"}
    assert payload["playback_audio"]["mime_type"] == "audio/wav"
    assert payload["playback_audio"]["duration_seconds"] == pytest.approx(3.0, abs=0.05)
    preview = base64.b64decode(payload["playback_audio"]["data_base64"])
    with wave.open(io.BytesIO(preview), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert wav.getsampwidth() == 2
        assert wav.getnframes() > 0


def speaker_service(tmp_path: Path, vectors: list[list[float]]) -> SpeakerIdentityService:
    storage = Storage(tmp_path)
    storage.connect()
    return SpeakerIdentityService(storage, make_settings(tmp_path), backend=QueueSpeakerBackend(vectors))


def tone_pcm(
    *,
    seconds: float,
    sample_rate: int = 16_000,
    frequency: float = 220.0,
    amplitude: float = 0.25,
) -> bytes:
    frames = int(seconds * sample_rate)
    samples = bytearray()
    for index in range(frames):
        value = int(amplitude * 32767 * math.sin(2.0 * math.pi * frequency * (index / sample_rate)))
        samples.extend(value.to_bytes(2, byteorder="little", signed=True))
    return bytes(samples)


def pulsed_training_pcm(*, seconds: float, sample_rate: int = 16_000) -> bytes:
    frames = int(seconds * sample_rate)
    samples = bytearray()
    pulse_centers = [0.8, 1.6, 2.5, 3.4, 4.2, 5.2, 6.3]
    for index in range(frames):
        t = index / sample_rate
        pulse = sum(math.exp(-((t - center) / 0.08) ** 2) for center in pulse_centers)
        amplitude = 0.018 + min(0.08, 0.06 * pulse)
        value = int(amplitude * 32767 * math.sin(2.0 * math.pi * 190.0 * t))
        samples.extend(value.to_bytes(2, byteorder="little", signed=True))
    return bytes(samples)


def write_wav(path: Path, pcm: bytes, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
