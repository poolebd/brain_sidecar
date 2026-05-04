from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from brain_sidecar.config import Settings
from brain_sidecar.core.models import new_id
from brain_sidecar.core.storage import Storage


SELF_PROFILE_ID = "self_bp"
SELF_DISPLAY_NAME = "BP"
SPEECH_SAMPLE_RATE = 16_000
MIN_ENROLLMENT_SPEECH_SECONDS = 15.0
TARGET_ENROLLMENT_SPEECH_SECONDS = 20.0
MIN_RUNTIME_MATCH_SECONDS = 0.8
DEFAULT_SELF_THRESHOLD = 0.82
DEFAULT_CLUSTER_THRESHOLD = 0.72
MIN_SELF_MARGIN = 0.04


@dataclass(frozen=True)
class AudioQuality:
    duration_seconds: float
    usable_speech_seconds: float
    speech_fraction: float
    rms: float
    peak: float
    clipping_fraction: float
    quality_score: float
    issues: list[str]
    speech_regions: list[tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "usable_speech_seconds": self.usable_speech_seconds,
            "speech_fraction": self.speech_fraction,
            "rms": self.rms,
            "peak": self.peak,
            "clipping_fraction": self.clipping_fraction,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "speech_regions": [
                {"start_sample": start, "end_sample": end}
                for start, end in self.speech_regions
            ],
        }


@dataclass(frozen=True)
class SpeakerBackendStatus:
    available: bool
    model_name: str
    model_version: str
    device: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SpeakerLabelResult:
    diarization_speaker_id: str | None
    display_speaker_label: str | None
    matched_profile_id: str | None
    match_confidence: float | None
    match_score: float | None
    speaker_role: str
    low_confidence: bool = False
    reason: str | None = None

    def transcript_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "speaker_role": self.speaker_role,
        }
        if self.display_speaker_label:
            payload["speaker_label"] = self.display_speaker_label
        if self.match_confidence is not None:
            payload["speaker_confidence"] = self.match_confidence
        if self.match_score is not None:
            payload["speaker_match_score"] = self.match_score
        if self.diarization_speaker_id:
            payload["diarization_speaker_id"] = self.diarization_speaker_id
        if self.low_confidence:
            payload["speaker_low_confidence"] = True
        if self.reason:
            payload["speaker_match_reason"] = self.reason
        return payload


@dataclass
class _SpeakerCluster:
    cluster_id: str
    display_label: str
    centroid: list[float]
    speech_seconds: float
    first_seen_ms: int
    updated_at: float = field(default_factory=time.time)


class SpeakerEmbeddingBackend(Protocol):
    @property
    def model_name(self) -> str:
        ...

    @property
    def model_version(self) -> str:
        ...

    def status(self) -> SpeakerBackendStatus:
        ...

    def embed_pcm16(self, pcm: bytes, sample_rate: int = SPEECH_SAMPLE_RATE) -> list[float]:
        ...


class SpeechBrainSpeakerBackend:
    """Lazy SpeechBrain ECAPA backend.

    The class is safe to instantiate when SpeechBrain/PyTorch are not installed;
    it reports unavailable status and raises only when embedding is requested.
    """

    model_name = "speechbrain/spkrec-ecapa-voxceleb"
    model_version = "speechbrain-ecapa-voxceleb"

    def __init__(self, *, preferred_device: str | None = None) -> None:
        self.preferred_device = preferred_device
        self._classifier = None
        self._device: str | None = None
        self._load_error: str | None = None

    def status(self) -> SpeakerBackendStatus:
        if self._classifier is not None:
            return SpeakerBackendStatus(True, self.model_name, self.model_version, self._device or "cpu")
        missing = [
            package
            for package in ("torch", "torchaudio", "speechbrain")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            return SpeakerBackendStatus(
                False,
                self.model_name,
                self.model_version,
                self.preferred_device or "auto",
                reason=f"Missing optional speaker packages: {', '.join(missing)}.",
            )
        if self._load_error:
            return SpeakerBackendStatus(
                False,
                self.model_name,
                self.model_version,
                self.preferred_device or "auto",
                reason=self._load_error,
            )
        return SpeakerBackendStatus(True, self.model_name, self.model_version, self.preferred_device or "auto")

    def embed_pcm16(self, pcm: bytes, sample_rate: int = SPEECH_SAMPLE_RATE) -> list[float]:
        self._load()
        assert self._classifier is not None
        import torch

        audio = pcm16_to_float32(pcm)
        if sample_rate != SPEECH_SAMPLE_RATE:
            raise RuntimeError("Speaker identity expects mono 16 kHz PCM16 audio.")
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self._device or "cpu")
        with torch.no_grad():
            embedding = self._classifier.encode_batch(wav)
        vector = embedding.squeeze().detach().cpu().numpy().astype(float).tolist()
        return l2_normalize(vector)

    def _load(self) -> None:
        if self._classifier is not None:
            return
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as exc:
            self._load_error = f"Could not import SpeechBrain speaker backend: {exc}"
            raise RuntimeError(self._load_error) from exc

        if self.preferred_device:
            device = self.preferred_device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_name,
                run_opts={"device": device},
            )
            self._device = device
        except Exception as exc:
            self._load_error = f"Could not load SpeechBrain ECAPA model: {exc}"
            raise RuntimeError(self._load_error) from exc


class DeterministicSpeakerBackend:
    """Small deterministic backend for tests and local dry runs."""

    model_name = "deterministic-test-speaker"
    model_version = "test-v1"

    def status(self) -> SpeakerBackendStatus:
        return SpeakerBackendStatus(True, self.model_name, self.model_version, "cpu")

    def embed_pcm16(self, pcm: bytes, sample_rate: int = SPEECH_SAMPLE_RATE) -> list[float]:
        samples = pcm16_to_float32(pcm)
        if samples.size == 0:
            return [1.0, 0.0, 0.0, 0.0]
        mean = float(np.mean(samples))
        rms = float(np.sqrt(np.mean(samples * samples)))
        sign_balance = float(np.mean(samples > 0))
        peak = float(np.max(np.abs(samples)))
        return l2_normalize([mean, rms, sign_balance, peak])


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(value / norm) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return float(sum(a * b for a, b in zip(l2_normalize(left), l2_normalize(right), strict=True)))


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    values = [0.0] * dim
    count = 0
    for vector in vectors:
        if len(vector) != dim:
            continue
        normalized = l2_normalize(vector)
        values = [left + right for left, right in zip(values, normalized, strict=True)]
        count += 1
    if count == 0:
        return []
    return l2_normalize([value / count for value in values])


def pairwise_scores(vectors: list[list[float]]) -> list[float]:
    scores: list[float] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1:]:
            scores.append(cosine_similarity(left, right))
    return scores


def calibrate_threshold(same_speaker_scores: list[float], negative_scores: list[float] | None = None) -> float:
    if not same_speaker_scores:
        return DEFAULT_SELF_THRESHOLD
    same_sorted = sorted(same_speaker_scores)
    low_same = same_sorted[max(0, int(len(same_sorted) * 0.15) - 1)]
    threshold = max(DEFAULT_SELF_THRESHOLD, low_same - 0.08)
    if negative_scores:
        high_negative = sorted(negative_scores)[min(len(negative_scores) - 1, int(len(negative_scores) * 0.95))]
        threshold = max(threshold, high_negative + 0.05)
    return round(min(0.95, max(0.72, threshold)), 4)


def confidence_from_score(score: float, threshold: float) -> float:
    if score < threshold:
        return max(0.0, min(0.84, score / max(0.01, threshold) * 0.84))
    room = max(0.01, 1.0 - threshold)
    return round(min(0.99, 0.86 + ((score - threshold) / room) * 0.13), 4)


def analyze_pcm16(pcm: bytes, sample_rate: int = SPEECH_SAMPLE_RATE) -> AudioQuality:
    samples = pcm16_to_float32(pcm)
    duration_seconds = float(samples.size / sample_rate) if sample_rate else 0.0
    if samples.size == 0:
        return AudioQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ["no_audio"], [])

    abs_samples = np.abs(samples)
    peak = float(np.max(abs_samples))
    rms = float(np.sqrt(np.mean(samples * samples)))
    clipping_fraction = float(np.mean(abs_samples >= 0.98))
    regions = energy_speech_regions(samples, sample_rate)
    usable = sum((end - start) / sample_rate for start, end in regions)
    speech_fraction = usable / duration_seconds if duration_seconds > 0 else 0.0

    issues: list[str] = []
    if duration_seconds < 1.0:
        issues.append("too_short")
    if usable < 0.5:
        issues.append("too_little_speech")
    if rms < 0.008:
        issues.append("very_low_volume")
    if peak > 0.98 or clipping_fraction > 0.01:
        issues.append("clipping")
    if speech_fraction < 0.25 and duration_seconds >= 2.0:
        issues.append("too_much_silence")

    quality = 1.0
    if "too_little_speech" in issues:
        quality -= 0.45
    if "very_low_volume" in issues:
        quality -= 0.2
    if "clipping" in issues:
        quality -= 0.3
    if "too_much_silence" in issues:
        quality -= 0.15
    quality -= min(0.2, max(0.0, 0.45 - speech_fraction) * 0.25)
    return AudioQuality(
        duration_seconds=round(duration_seconds, 4),
        usable_speech_seconds=round(float(usable), 4),
        speech_fraction=round(float(speech_fraction), 4),
        rms=round(float(rms), 6),
        peak=round(float(peak), 6),
        clipping_fraction=round(float(clipping_fraction), 6),
        quality_score=round(max(0.0, min(1.0, quality)), 4),
        issues=issues,
        speech_regions=regions,
    )


def energy_speech_regions(samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    frame_size = max(1, int(sample_rate * 0.03))
    if samples.size < frame_size:
        return [(0, int(samples.size))] if float(np.sqrt(np.mean(samples * samples))) >= 0.008 else []

    frame_rms: list[float] = []
    frame_ranges: list[tuple[int, int]] = []
    for start in range(0, samples.size, frame_size):
        end = min(samples.size, start + frame_size)
        frame = samples[start:end]
        frame_rms.append(float(np.sqrt(np.mean(frame * frame))))
        frame_ranges.append((start, end))

    arr = np.asarray(frame_rms, dtype=np.float32)
    threshold = max(0.006, float(np.percentile(arr, 70)) * 0.38, float(np.median(arr)) * 1.5)
    active = arr >= threshold
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if bool(is_active) and start is None:
            start = frame_ranges[index][0]
        if start is not None and (not bool(is_active) or index == len(active) - 1):
            end = frame_ranges[index][1] if bool(is_active) and index == len(active) - 1 else frame_ranges[index][0]
            if end - start >= int(sample_rate * 0.18):
                regions.append((start, end))
            start = None

    if not regions:
        whole_rms = float(np.sqrt(np.mean(samples * samples)))
        if whole_rms >= 0.008:
            return [(0, int(samples.size))]
        return []

    merged: list[tuple[int, int]] = []
    max_gap = int(sample_rate * 0.2)
    for start, end in regions:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def speech_only_pcm(pcm: bytes, quality: AudioQuality) -> bytes:
    if not quality.speech_regions:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2")
    pieces = [samples[start:end] for start, end in quality.speech_regions if end > start]
    if not pieces:
        return pcm
    return np.concatenate(pieces).astype("<i2").tobytes()


class SpeakerIdentityService:
    def __init__(
        self,
        storage: Storage,
        settings: Settings,
        *,
        backend: SpeakerEmbeddingBackend | None = None,
    ) -> None:
        self.storage = storage
        self.settings = settings
        self.backend = backend or SpeechBrainSpeakerBackend()
        self.storage.ensure_speaker_profile(
            SELF_PROFILE_ID,
            display_name=self.self_display_name,
            kind="self",
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
        )
        self._session_clusters: dict[str, list[_SpeakerCluster]] = {}

    def status(self) -> dict[str, Any]:
        profile = self.storage.ensure_speaker_profile(
            SELF_PROFILE_ID,
            display_name=self.self_display_name,
            kind="self",
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
        )
        embeddings = self.storage.speaker_embeddings(SELF_PROFILE_ID, sources=("enrollment", "corrected_segment"))
        centroid_row = self.storage.speaker_centroid_embedding(SELF_PROFILE_ID)
        total_speech = sum(float(row.get("duration_seconds") or 0.0) for row in embeddings)
        same_scores = pairwise_scores([row["vector"] for row in embeddings])
        threshold = float(profile.get("threshold") or DEFAULT_SELF_THRESHOLD)
        ready = bool(profile["active"]) and centroid_row is not None and total_speech >= MIN_ENROLLMENT_SPEECH_SECONDS
        needs_recalibration = len(embeddings) >= 2 and same_scores and min(same_scores) < max(0.55, threshold - 0.18)
        enrollment_status = (
            "ready"
            if ready and not needs_recalibration
            else "needs_recalibration"
            if centroid_row is not None and needs_recalibration
            else "needs_more_audio"
            if total_speech > 0
            else "not_enrolled"
        )
        return {
            "profile": {
                "id": profile["id"],
                "display_name": profile["display_name"],
                "kind": profile["kind"],
                "active": bool(profile["active"]),
                "threshold": threshold,
                "embedding_model": profile["embedding_model"],
                "embedding_model_version": profile["embedding_model_version"],
                "created_at": profile["created_at"],
                "updated_at": profile["updated_at"],
            },
            "backend": self.backend.status().to_dict(),
            "enrollment_status": enrollment_status,
            "ready": ready and not needs_recalibration,
            "needs_recalibration": needs_recalibration,
            "usable_speech_seconds": round(total_speech, 3),
            "target_speech_seconds": TARGET_ENROLLMENT_SPEECH_SECONDS,
            "minimum_speech_seconds": MIN_ENROLLMENT_SPEECH_SECONDS,
            "embedding_count": len(embeddings),
            "centroid_vector_dim": len(centroid_row["vector"]) if centroid_row else None,
            "same_speaker_scores": {
                "count": len(same_scores),
                "min": round(min(same_scores), 4) if same_scores else None,
                "mean": round(float(sum(same_scores) / len(same_scores)), 4) if same_scores else None,
            },
            "quality_score": self._profile_quality(embeddings, same_scores, total_speech),
            "raw_audio_retained": False,
            "recent_segments": self.storage.recent_diarization_segments(limit=12),
        }

    def start_enrollment(self) -> dict[str, Any]:
        enrollment = self.storage.create_speaker_enrollment(SELF_PROFILE_ID)
        return self._enrollment_payload(enrollment)

    def enrollment(self, enrollment_id: str) -> dict[str, Any]:
        return self._enrollment_payload(self.storage.speaker_enrollment(enrollment_id))

    def add_enrollment_sample(
        self,
        enrollment_id: str,
        pcm: bytes,
        *,
        sample_rate: int = SPEECH_SAMPLE_RATE,
        source: str = "enrollment",
    ) -> dict[str, Any]:
        enrollment = self.storage.speaker_enrollment(enrollment_id)
        if enrollment["profile_id"] != SELF_PROFILE_ID:
            raise ValueError("Only the self speaker profile can be trained from this flow.")
        quality = analyze_pcm16(pcm, sample_rate)
        if quality.usable_speech_seconds < 0.5:
            raise ValueError("Not enough usable speech was detected. Record a single-speaker sample again.")
        if "clipping" in quality.issues:
            raise ValueError("The sample is clipping. Lower the input gain or move slightly farther from the mic.")
        if quality.quality_score < 0.35:
            raise ValueError("The sample quality is too low for speaker identity enrollment.")
        clean_pcm = speech_only_pcm(pcm, quality)
        vector = self.backend.embed_pcm16(clean_pcm, sample_rate)
        embedding = self.storage.add_speaker_embedding(
            profile_id=SELF_PROFILE_ID,
            vector=l2_normalize(vector),
            source=source,
            quality_score=quality.quality_score,
            duration_seconds=quality.usable_speech_seconds,
            sample_rate=sample_rate,
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
            metadata={
                "quality": quality.to_dict(),
                "raw_audio_retained": False,
                "enrollment_id": enrollment_id,
            },
        )
        sample = self.storage.add_speaker_enrollment_sample(
            enrollment_id=enrollment_id,
            profile_id=SELF_PROFILE_ID,
            embedding_id=embedding["id"],
            duration_seconds=quality.duration_seconds,
            usable_speech_seconds=quality.usable_speech_seconds,
            quality_score=quality.quality_score,
            issues=quality.issues,
            metadata={"raw_audio_retained": False},
        )
        return {
            "sample": sample,
            "embedding": {
                "id": embedding["id"],
                "vector_dim": embedding["vector_dim"],
                "quality_score": embedding["quality_score"],
                "duration_seconds": embedding["duration_seconds"],
            },
            "quality": quality.to_dict(),
            "profile": self.status(),
            "raw_audio_retained": False,
        }

    def finalize_enrollment(self, enrollment_id: str) -> dict[str, Any]:
        enrollment = self.storage.speaker_enrollment(enrollment_id)
        if enrollment["profile_id"] != SELF_PROFILE_ID:
            raise ValueError("Only the self speaker profile can be finalized from this flow.")
        embeddings = self.storage.speaker_embeddings(SELF_PROFILE_ID, sources=("enrollment", "corrected_segment"))
        total_speech = sum(float(row.get("duration_seconds") or 0.0) for row in embeddings)
        if total_speech < MIN_ENROLLMENT_SPEECH_SECONDS:
            raise ValueError(
                f"Need at least {MIN_ENROLLMENT_SPEECH_SECONDS:.0f}s of usable single-speaker speech; "
                f"currently have {total_speech:.1f}s."
            )
        vectors = [row["vector"] for row in embeddings]
        same_scores = pairwise_scores(vectors)
        if same_scores and min(same_scores) < 0.52:
            raise ValueError("Enrollment samples are inconsistent; one sample may contain another speaker or bad audio.")
        threshold = calibrate_threshold(same_scores)
        centroid_vector = centroid(vectors)
        centroid_row = self.storage.add_speaker_embedding(
            profile_id=SELF_PROFILE_ID,
            vector=centroid_vector,
            source="centroid",
            quality_score=self._profile_quality(embeddings, same_scores, total_speech),
            duration_seconds=total_speech,
            sample_rate=SPEECH_SAMPLE_RATE,
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
            metadata={
                "same_speaker_scores": same_scores,
                "threshold": threshold,
                "raw_audio_retained": False,
                "source_embedding_count": len(embeddings),
            },
        )
        self.storage.update_speaker_profile(
            SELF_PROFILE_ID,
            active=True,
            threshold=threshold,
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
            notes="Self speaker identity profile calibrated from enrollment embeddings.",
        )
        self.storage.set_speaker_enrollment_status(enrollment_id, "completed")
        return {
            "profile": self.status(),
            "centroid_embedding_id": centroid_row["id"],
            "threshold": threshold,
            "same_speaker_scores": same_scores,
            "raw_audio_retained": False,
        }

    def recalibrate(self) -> dict[str, Any]:
        embeddings = self.storage.speaker_embeddings(SELF_PROFILE_ID, sources=("enrollment", "corrected_segment"))
        if len(embeddings) < 2:
            raise ValueError("At least two speaker embeddings are required to recalibrate.")
        vectors = [row["vector"] for row in embeddings]
        same_scores = pairwise_scores(vectors)
        threshold = calibrate_threshold(same_scores)
        centroid_vector = centroid(vectors)
        centroid_row = self.storage.add_speaker_embedding(
            profile_id=SELF_PROFILE_ID,
            vector=centroid_vector,
            source="centroid",
            quality_score=self._profile_quality(embeddings, same_scores, sum(row["duration_seconds"] for row in embeddings)),
            duration_seconds=sum(row["duration_seconds"] for row in embeddings),
            sample_rate=SPEECH_SAMPLE_RATE,
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
            metadata={"same_speaker_scores": same_scores, "threshold": threshold, "raw_audio_retained": False},
        )
        self.storage.update_speaker_profile(SELF_PROFILE_ID, threshold=threshold, active=True)
        return {"profile": self.status(), "centroid_embedding_id": centroid_row["id"], "threshold": threshold}

    def reset_profile(self) -> dict[str, Any]:
        removed = self.storage.reset_speaker_profile(SELF_PROFILE_ID)
        self.storage.ensure_speaker_profile(
            SELF_PROFILE_ID,
            display_name=self.self_display_name,
            kind="self",
            embedding_model=self.backend.model_name,
            embedding_model_version=self.backend.model_version,
        )
        self._session_clusters.clear()
        return {"removed": removed, "profile": self.status(), "raw_audio_retained": False}

    def apply_feedback(
        self,
        *,
        session_id: str,
        segment_id: str | None,
        old_label: str,
        new_label: str,
        feedback_type: str,
    ) -> dict[str, Any]:
        row = self.storage.add_speaker_label_feedback(
            session_id=session_id,
            segment_id=segment_id,
            old_label=old_label,
            new_label=new_label,
            feedback_type=feedback_type,
            applied_to_training=False,
            metadata={"note": "Feedback is recorded for recalibration; raw audio is not retained."},
        )
        return {"feedback": row, "profile": self.status()}

    def label_segment(
        self,
        *,
        session_id: str,
        segment_id: str,
        pcm: bytes,
        start_ms: int,
        end_ms: int,
        transcript_text: str | None = None,
        persist: bool = False,
    ) -> SpeakerLabelResult:
        status = self.status()
        centroid_row = self.storage.speaker_centroid_embedding(SELF_PROFILE_ID)
        backend_status = self.backend.status()
        if not status["ready"] or centroid_row is None or not backend_status.available:
            result = SpeakerLabelResult(
                diarization_speaker_id=None,
                display_speaker_label=None,
                matched_profile_id=None,
                match_confidence=None,
                match_score=None,
                speaker_role="unknown",
                reason="speaker_profile_not_ready" if not status["ready"] else "speaker_backend_unavailable",
            )
            self._persist_label(session_id, segment_id, start_ms, end_ms, transcript_text, result, persist)
            return result

        quality = analyze_pcm16(pcm, SPEECH_SAMPLE_RATE)
        if quality.usable_speech_seconds < MIN_RUNTIME_MATCH_SECONDS:
            result = SpeakerLabelResult(
                diarization_speaker_id=None,
                display_speaker_label="Unknown speaker",
                matched_profile_id=None,
                match_confidence=0.0,
                match_score=None,
                speaker_role="unknown",
                low_confidence=True,
                reason="too_little_speech_for_identity",
            )
            self._persist_label(session_id, segment_id, start_ms, end_ms, transcript_text, result, persist)
            return result

        vector = self.backend.embed_pcm16(speech_only_pcm(pcm, quality), SPEECH_SAMPLE_RATE)
        threshold = float(status["profile"]["threshold"] or DEFAULT_SELF_THRESHOLD)
        self_score = cosine_similarity(vector, centroid_row["vector"])
        anonymous_scores = [
            cosine_similarity(vector, cluster.centroid)
            for cluster in self._session_clusters.get(session_id, [])
        ]
        best_anon = max(anonymous_scores) if anonymous_scores else None
        margin = self_score - (best_anon if best_anon is not None else 0.0)
        if self_score >= threshold and margin >= MIN_SELF_MARGIN:
            result = SpeakerLabelResult(
                diarization_speaker_id="SELF",
                display_speaker_label=self.self_display_name,
                matched_profile_id=SELF_PROFILE_ID,
                match_confidence=confidence_from_score(self_score, threshold),
                match_score=round(self_score, 4),
                speaker_role="user",
            )
            self._persist_label(session_id, segment_id, start_ms, end_ms, transcript_text, result, persist)
            return result
        if self_score >= threshold - 0.04:
            result = SpeakerLabelResult(
                diarization_speaker_id=None,
                display_speaker_label="Unknown speaker",
                matched_profile_id=None,
                match_confidence=confidence_from_score(self_score, threshold),
                match_score=round(self_score, 4),
                speaker_role="unknown",
                low_confidence=True,
                reason="ambiguous_self_match",
            )
            self._persist_label(session_id, segment_id, start_ms, end_ms, transcript_text, result, persist)
            return result

        cluster = self._anonymous_cluster(session_id, vector, quality.usable_speech_seconds, start_ms)
        result = SpeakerLabelResult(
            diarization_speaker_id=cluster.cluster_id,
            display_speaker_label=cluster.display_label,
            matched_profile_id=None,
            match_confidence=0.0,
            match_score=round(self_score, 4),
            speaker_role="other",
            reason="below_self_threshold",
        )
        self._persist_label(session_id, segment_id, start_ms, end_ms, transcript_text, result, persist)
        return result

    def _anonymous_cluster(
        self,
        session_id: str,
        vector: list[float],
        speech_seconds: float,
        first_seen_ms: int,
    ) -> _SpeakerCluster:
        clusters = self._session_clusters.setdefault(session_id, [])
        best: _SpeakerCluster | None = None
        best_score = -1.0
        for cluster in clusters:
            score = cosine_similarity(vector, cluster.centroid)
            if score > best_score:
                best = cluster
                best_score = score
        if best is not None and best_score >= DEFAULT_CLUSTER_THRESHOLD:
            total = best.speech_seconds + speech_seconds
            weighted = [
                ((left * best.speech_seconds) + (right * speech_seconds)) / max(0.001, total)
                for left, right in zip(best.centroid, l2_normalize(vector), strict=True)
            ]
            best.centroid = l2_normalize(weighted)
            best.speech_seconds = total
            best.updated_at = time.time()
            return best
        next_index = len(clusters) + 1
        cluster = _SpeakerCluster(
            cluster_id=f"SPEAKER_{next_index - 1:02d}",
            display_label=f"Speaker {next_index}",
            centroid=l2_normalize(vector),
            speech_seconds=speech_seconds,
            first_seen_ms=first_seen_ms,
        )
        clusters.append(cluster)
        clusters.sort(key=lambda item: item.first_seen_ms)
        for index, item in enumerate(clusters, start=1):
            item.display_label = f"Speaker {index}"
        return cluster

    @property
    def self_display_name(self) -> str:
        value = getattr(self.settings, "speaker_identity_label", SELF_DISPLAY_NAME)
        return value.strip() or SELF_DISPLAY_NAME

    def _persist_label(
        self,
        session_id: str,
        segment_id: str,
        start_ms: int,
        end_ms: int,
        transcript_text: str | None,
        result: SpeakerLabelResult,
        persist: bool,
    ) -> None:
        if not persist:
            return
        self.storage.add_diarization_segment(
            session_id=session_id,
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            diarization_speaker_id=result.diarization_speaker_id,
            display_speaker_label=result.display_speaker_label,
            matched_profile_id=result.matched_profile_id,
            match_confidence=result.match_confidence,
            match_score=result.match_score,
            transcript_text=transcript_text,
            is_overlap=False,
            finalized=True,
            metadata={
                "speaker_role": result.speaker_role,
                "low_confidence": result.low_confidence,
                "reason": result.reason,
            },
        )

    def _enrollment_payload(self, enrollment: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": enrollment["id"],
            "profile_id": enrollment["profile_id"],
            "status": enrollment["status"],
            "started_at": enrollment["started_at"],
            "completed_at": enrollment["completed_at"],
            "samples": self.storage.speaker_enrollment_samples(enrollment["id"]),
            "profile": self.status(),
            "raw_audio_retained": False,
        }

    def _profile_quality(
        self,
        embeddings: list[dict[str, Any]],
        same_scores: list[float],
        total_speech: float,
    ) -> float:
        if not embeddings:
            return 0.0
        duration_score = min(1.0, total_speech / TARGET_ENROLLMENT_SPEECH_SECONDS)
        sample_quality = sum(float(row.get("quality_score") or 0.0) for row in embeddings) / len(embeddings)
        consistency = 0.7 if not same_scores else max(0.0, min(1.0, (sum(same_scores) / len(same_scores) - 0.45) / 0.5))
        return round((duration_score * 0.35) + (sample_quality * 0.35) + (consistency * 0.3), 4)
