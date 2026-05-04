from pathlib import Path
from types import SimpleNamespace

import numpy as np

from brain_sidecar.config import Settings
from brain_sidecar.core.transcription import (
    FasterWhisperTranscriber,
    audio_rms,
    clean_transcript_text,
    is_signal_segment,
    is_signal_text,
)


def test_clean_transcript_text_compacts_whitespace() -> None:
    assert clean_transcript_text(" hello\n  local\tbrain ") == "hello local brain"


def test_is_signal_text_rejects_short_or_non_word_fragments() -> None:
    assert not is_signal_text(" uh ", min_chars=4)
    assert not is_signal_text("...", min_chars=2)
    assert is_signal_text("clear sentence", min_chars=4)


def test_is_signal_segment_rejects_weak_asr_metadata() -> None:
    thresholds = {
        "min_chars": 4,
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
    }

    assert is_signal_segment(
        SimpleNamespace(text="clear sentence", no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.1),
        **thresholds,
    )
    assert not is_signal_segment(
        SimpleNamespace(text="Thanks for watching", no_speech_prob=0.95, avg_logprob=-0.2, compression_ratio=1.1),
        **thresholds,
    )
    assert not is_signal_segment(
        SimpleNamespace(text="clear sentence", no_speech_prob=0.1, avg_logprob=-1.6, compression_ratio=1.1),
        **thresholds,
    )
    assert not is_signal_segment(
        SimpleNamespace(text="clear sentence", no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=3.1),
        **thresholds,
    )


def test_audio_rms_identifies_silence() -> None:
    assert audio_rms(np.zeros(1600, dtype=np.float32)) == 0.0
    assert audio_rms(np.ones(1600, dtype=np.float32) * 0.01) > 0.009


def test_audio_rms_ignores_isolated_spikes() -> None:
    audio = np.zeros(1600, dtype=np.float32)
    audio[10] = 1.0

    assert audio_rms(audio) == 0.0


def test_transcriber_passes_quality_thresholds_and_filters_segments(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    transcriber = FasterWhisperTranscriber(settings)
    model = FakeWhisperModel()
    transcriber._model = model
    transcriber.model_size = "fake.en"

    pcm = (np.ones(16_000, dtype=np.int16) * 1000).tobytes()
    result = transcriber._transcribe_sync(pcm, 10.0)

    assert model.kwargs["no_speech_threshold"] == 0.42
    assert model.kwargs["log_prob_threshold"] == -0.8
    assert model.kwargs["compression_ratio_threshold"] == 2.1
    assert [span.text for span in result.spans] == ["good words"]
    assert result.spans[0].start_s == 10.25


def test_transcriber_skips_near_silent_windows_before_model_call(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(make_settings(tmp_path))
    model = FakeWhisperModel()
    transcriber._model = model
    transcriber.model_size = "fake.en"

    result = transcriber._transcribe_sync(b"\0" * 32_000, 0.0)

    assert result.spans == []
    assert model.kwargs == {}


class FakeWhisperModel:
    def __init__(self) -> None:
        self.kwargs = {}

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return [
            SimpleNamespace(
                start=0.25,
                end=1.0,
                text=" good words ",
                no_speech_prob=0.1,
                avg_logprob=-0.2,
                compression_ratio=1.0,
            ),
            SimpleNamespace(
                start=1.1,
                end=1.6,
                text="Thanks for watching",
                no_speech_prob=0.95,
                avg_logprob=-0.2,
                compression_ratio=1.0,
            ),
        ], SimpleNamespace(language="en")


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
        asr_no_speech_threshold=0.42,
        asr_log_prob_threshold=-0.8,
        asr_compression_ratio_threshold=2.1,
        asr_min_audio_rms=0.003,
    )
