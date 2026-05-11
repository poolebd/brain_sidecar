from __future__ import annotations

import asyncio
import sys
import types
import wave
from pathlib import Path

import brain_sidecar.config as config
from brain_sidecar.config import load_settings
from brain_sidecar.core import review_asr
from brain_sidecar.core.asr import TranscribedSpan
from brain_sidecar.core.review_asr import NemoReviewAsr, _dedupe_overlapping_spans, _spans_from_plain_text


class FakeNemoModel:
    def __init__(self, *, oom_once: bool = False, oom_on_single: bool = False) -> None:
        self.oom_once = oom_once
        self.oom_on_single = oom_on_single
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        paths,
        *,
        batch_size: int,
        return_hypotheses: bool,
        timestamps: bool = True,
        num_workers: int = 0,
        verbose: bool = False,
    ):
        path_list = [Path(path) for path in paths]
        self.calls.append({"paths": [path.name for path in path_list], "batch_size": batch_size})
        if self.oom_once and (batch_size > 1 or self.oom_on_single):
            self.oom_once = False
            raise RuntimeError("CUDA out of memory")
        return [
            {
                "text": f"text from {path.stem}",
                "timestamp": {
                    "segment": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "segment": f"text from {path.stem}",
                        }
                    ]
                },
            }
            for path in path_list
        ]


class FakeNemoTimestepModel:
    def transcribe(
        self,
        paths,
        *,
        batch_size: int,
        return_hypotheses: bool,
        timestamps: bool = True,
        num_workers: int = 0,
        verbose: bool = False,
    ):
        return [
            {
                "text": "alpha beta gamma delta",
                "timestep": {
                    "word": [
                        {"word": "alpha", "start": 0.0, "end": 0.4},
                        {"word": "beta", "start": 0.5, "end": 0.9},
                        {"word": "gamma", "start": 1.0, "end": 1.4},
                        {"word": "delta", "start": 1.5, "end": 1.9},
                    ]
                },
            }
        ]


class FakeLoadedNemoModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class FakeCudaRuntime:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def test_nemo_review_asr_prepares_gpu_before_cuda_load(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_DEVICE", "cuda")
    install_fake_torch(monkeypatch, available=True)
    load_calls: list[tuple[str, str]] = []
    prepare_calls: list[bool] = []

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *, model_name: str, map_location: str):
            load_calls.append((model_name, map_location))
            return FakeLoadedNemoModel()

    install_fake_nemo(monkeypatch, FakeASRModel)

    def fake_prepare(_settings, *, force_unload: bool = False):
        prepare_calls.append(force_unload)

    monkeypatch.setattr(review_asr, "prepare_asr_gpu", fake_prepare)

    settings = load_settings()
    backend = NemoReviewAsr(settings)
    backend._load_sync()

    assert prepare_calls == [False]
    assert load_calls == [(settings.review_asr_model, "cuda")]
    assert backend._loaded_device == "cuda"


def test_nemo_review_asr_skips_gpu_preparation_for_cpu_load(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_DEVICE", "cpu")
    install_fake_torch(monkeypatch, available=True)
    load_calls: list[tuple[str, str]] = []

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *, model_name: str, map_location: str):
            load_calls.append((model_name, map_location))
            return FakeLoadedNemoModel()

    install_fake_nemo(monkeypatch, FakeASRModel)
    monkeypatch.setattr(
        review_asr,
        "prepare_asr_gpu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected GPU preparation")),
    )

    settings = load_settings()
    backend = NemoReviewAsr(settings)
    backend._load_sync()

    assert load_calls == [(settings.review_asr_model, "cpu")]
    assert backend._loaded_device == "cpu"


def test_nemo_review_asr_retries_cuda_load_after_gpu_cleanup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_DEVICE", "cuda")
    fake_torch = install_fake_torch(monkeypatch, available=True)
    load_calls: list[str] = []
    prepare_calls: list[bool] = []

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *, model_name: str, map_location: str):
            load_calls.append(map_location)
            if len(load_calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return FakeLoadedNemoModel()

    install_fake_nemo(monkeypatch, FakeASRModel)

    def fake_prepare(_settings, *, force_unload: bool = False):
        prepare_calls.append(force_unload)

    monkeypatch.setattr(review_asr, "prepare_asr_gpu", fake_prepare)

    settings = load_settings()
    backend = NemoReviewAsr(settings)
    backend._load_sync()

    assert prepare_calls == [False, True]
    assert load_calls == ["cuda", "cuda"]
    assert fake_torch.cuda.empty_cache_calls == 1
    assert backend._loaded_device == "cuda"


def test_nemo_review_asr_chunks_batches_offsets_and_cleans_up(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_BATCH_SIZE", "2")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_CHUNK_SECONDS", "30")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_CHUNK_OVERLAP_SECONDS", "5")
    wav_path = tmp_path / "audio.wav"
    write_silent_wav(wav_path, duration_s=70.0)

    settings = load_settings()
    model = FakeNemoModel()
    backend = NemoReviewAsr(settings)
    backend._model = model
    backend.model_size = settings.review_asr_model

    result, messages = asyncio.run(transcribe_with_progress(backend, wav_path))

    assert [call["batch_size"] for call in model.calls] == [2, 1]
    assert [round(span.start_s, 1) for span in result.spans] == [0.0, 25.0, 50.0]
    assert [round(span.end_s, 1) for span in result.spans] == [1.0, 26.0, 51.0]
    assert [span.text for span in result.spans] == [
        "text from chunk_0000",
        "text from chunk_0001",
        "text from chunk_0002",
    ]
    assert any(message == "Prepared 3 ASR chunks." for message, _ in messages)
    assert any(message == "Transcribing chunks 1-2/3." for message, _ in messages)
    assert not list(tmp_path.glob("audio-chunks-*"))


def test_nemo_review_asr_retries_oom_batches_one_chunk_at_a_time(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_BATCH_SIZE", "2")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_CHUNK_SECONDS", "30")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_CHUNK_OVERLAP_SECONDS", "0")
    wav_path = tmp_path / "oom.wav"
    write_silent_wav(wav_path, duration_s=50.0)

    settings = load_settings()
    model = FakeNemoModel(oom_once=True)
    backend = NemoReviewAsr(settings)
    backend._model = model
    backend.model_size = settings.review_asr_model

    result, messages = asyncio.run(transcribe_with_progress(backend, wav_path))

    assert [call["batch_size"] for call in model.calls] == [2, 1, 1]
    assert len(result.spans) == 2
    assert any("retrying the current ASR batch" in message for message, _ in messages)
    assert not list(tmp_path.glob("oom-chunks-*"))


def test_nemo_review_asr_reloads_and_retries_single_chunk_cuda_oom(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_CHUNK_SECONDS", "30")
    monkeypatch.setenv("BRAIN_SIDECAR_REVIEW_ASR_CHUNK_OVERLAP_SECONDS", "0")
    wav_path = tmp_path / "single-oom.wav"
    write_silent_wav(wav_path, duration_s=10.0)

    settings = load_settings()
    model = FakeNemoModel(oom_once=True, oom_on_single=True)
    backend = NemoReviewAsr(settings)
    backend._model = model
    backend._loaded_device = "cuda"
    backend.model_size = settings.review_asr_model
    reload_calls = 0

    async def fake_reload_after_cuda_oom() -> None:
        nonlocal reload_calls
        reload_calls += 1

    monkeypatch.setattr(backend, "_reload_after_cuda_oom", fake_reload_after_cuda_oom)

    result, messages = asyncio.run(transcribe_with_progress(backend, wav_path))

    assert reload_calls == 1
    assert [call["batch_size"] for call in model.calls] == [1, 1]
    assert len(result.spans) == 1
    assert any("reloading NeMo ASR" in message for message, _ in messages)
    assert not list(tmp_path.glob("single-oom-chunks-*"))


def test_nemo_review_asr_reads_ctc_timestep_words(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    wav_path = tmp_path / "timestep.wav"
    write_silent_wav(wav_path, duration_s=3.0)

    settings = load_settings()
    backend = NemoReviewAsr(settings)
    backend._model = FakeNemoTimestepModel()
    backend.model_size = settings.review_asr_model

    result = backend._transcribe_sync(wav_path)

    assert result.spans
    assert result.spans[0].text == "alpha beta gamma delta"
    assert result.spans[0].start_s == 0.0
    assert result.spans[0].end_s == 1.9


def test_plain_text_fallback_splits_long_unpunctuated_chunks() -> None:
    text = " ".join(f"word{index}" for index in range(70))

    spans = _spans_from_plain_text(text, duration=42.0)

    assert len(spans) == 3
    assert all(span.end_s > span.start_s for span in spans)
    assert spans[0].text.startswith("word0 word1")
    assert spans[-1].text.endswith("word69")


def test_dedupe_overlapping_spans_keeps_later_repeated_content_outside_overlap() -> None:
    spans = [
        TranscribedSpan(start_s=0.0, end_s=1.0, text="Repeat this boundary line."),
        TranscribedSpan(start_s=3.0, end_s=4.0, text="Repeat this boundary line."),
        TranscribedSpan(start_s=8.5, end_s=9.5, text="Repeat this boundary line."),
    ]

    result = _dedupe_overlapping_spans(spans, overlap_seconds=3.0)

    assert [(span.start_s, span.text) for span in result] == [
        (0.0, "Repeat this boundary line."),
        (8.5, "Repeat this boundary line."),
    ]


async def transcribe_with_progress(
    backend: NemoReviewAsr,
    wav_path: Path,
) -> tuple[object, list[tuple[str, int]]]:
    messages: list[tuple[str, int]] = []

    async def progress(message: str, percent: int) -> None:
        messages.append((message, percent))

    return await backend.transcribe_file(wav_path, progress=progress), messages


def write_silent_wav(path: Path, *, duration_s: float) -> None:
    sample_rate = 16_000
    frames = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0" * frames * 2)


def install_fake_torch(monkeypatch, *, available: bool):
    torch_module = types.ModuleType("torch")
    torch_module.cuda = FakeCudaRuntime(available=available)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    return torch_module


def install_fake_nemo(monkeypatch, asr_model_class) -> None:
    nemo_module = types.ModuleType("nemo")
    collections_module = types.ModuleType("nemo.collections")
    asr_module = types.ModuleType("nemo.collections.asr")
    models_module = types.ModuleType("nemo.collections.asr.models")
    nemo_module.__path__ = []
    collections_module.__path__ = []
    asr_module.__path__ = []
    models_module.ASRModel = asr_model_class
    nemo_module.collections = collections_module
    collections_module.asr = asr_module
    asr_module.models = models_module
    monkeypatch.setitem(sys.modules, "nemo", nemo_module)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections_module)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_module)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models_module)
