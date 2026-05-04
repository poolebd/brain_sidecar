from brain_sidecar.core.transcription import clean_transcript_text, is_signal_text


def test_clean_transcript_text_compacts_whitespace() -> None:
    assert clean_transcript_text(" hello\n  local\tbrain ") == "hello local brain"


def test_is_signal_text_rejects_short_or_non_word_fragments() -> None:
    assert not is_signal_text(" uh ", min_chars=4)
    assert not is_signal_text("...", min_chars=2)
    assert is_signal_text("clear sentence", min_chars=4)
