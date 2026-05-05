from __future__ import annotations

from brain_sidecar.core.asr_aliases import extract_supported_aliases
from brain_sidecar.eval.models import EvalTranscriptEvent
from brain_sidecar.eval.noise import apply_default_noise, corrupt_terms, duplicate_overlap


def event() -> EvalTranscriptEvent:
    return EvalTranscriptEvent(
        id="seg_1",
        session_id="eval_noise",
        start_s=0.0,
        end_s=4.0,
        text="Siemens needs the SLD and six to eight hours from Greg.",
        expected_topics=["siemens", "sld", "hours"],
    )


def test_noise_is_deterministic_with_seed() -> None:
    first = apply_default_noise([event()], seed=11)
    second = apply_default_noise([event()], seed=11)

    assert [item.to_json() for item in first] == [item.to_json() for item in second]


def test_overlap_noise_increases_input_segments() -> None:
    noisy = [duplicate_overlap(event()), event()]

    assert len(noisy) == 2
    assert noisy[0].start_s > event().start_s


def test_noise_mapping_does_not_modify_expected_terms() -> None:
    corrupted = corrupt_terms(event(), {"Siemens": ["demons"]}, seed=1)

    assert "demons" in corrupted.text
    assert corrupted.expected_topics == ["siemens", "sld", "hours"]


def test_photo_peak_noise_does_not_create_supported_topic() -> None:
    corrupted = corrupt_terms(event(), {"six to eight hours": ["photo peak"]}, seed=1)

    assert "photo peak" in corrupted.text
    assert "hours" not in extract_supported_aliases(corrupted.text)
