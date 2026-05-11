from __future__ import annotations

import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from brain_sidecar.core.models import TranscriptSegment


AMI_ES2002A_AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Array1-01.wav"
AMI_MANUAL_ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


@dataclass(frozen=True)
class AmiFixture:
    meeting_id: str
    audio_path: Path
    annotations_zip: Path
    transcript_segments: list[TranscriptSegment]


def ensure_ami_es2002a_fixture(cache_root: Path, *, download: bool) -> AmiFixture:
    fixture_dir = cache_root / "ami" / "ES2002a"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    audio_path = fixture_dir / "ES2002a.Array1-01.wav"
    annotations_zip = fixture_dir / "ami_public_manual_1.6.2.zip"
    if download:
        _download_once(AMI_ES2002A_AUDIO_URL, audio_path)
        _download_once(AMI_MANUAL_ANNOTATIONS_URL, annotations_zip)
    if not audio_path.exists() or not annotations_zip.exists():
        missing = []
        if not audio_path.exists():
            missing.append(str(audio_path))
        if not annotations_zip.exists():
            missing.append(str(annotations_zip))
        raise FileNotFoundError(
            "AMI ES2002a fixture is not cached. Set BRAIN_SIDECAR_DOWNLOAD_OPEN_SOURCE_FIXTURES=1 "
            f"to download it. Missing: {', '.join(missing)}"
        )
    return AmiFixture(
        meeting_id="ES2002a",
        audio_path=audio_path,
        annotations_zip=annotations_zip,
        transcript_segments=parse_ami_words_zip(annotations_zip, "ES2002a", max_segments=180),
    )


def parse_ami_words_zip(annotations_zip: Path, meeting_id: str, *, max_segments: int = 180) -> list[TranscriptSegment]:
    words: list[tuple[float, float, str, str]] = []
    with zipfile.ZipFile(annotations_zip) as archive:
        names = [
            name
            for name in archive.namelist()
            if f"/words/{meeting_id}." in name and name.endswith(".words.xml")
        ]
        if not names:
            names = [
                name
                for name in archive.namelist()
                if f"{meeting_id}." in name and name.endswith(".words.xml")
            ]
        for name in sorted(names):
            speaker = _speaker_from_words_path(name)
            with archive.open(name) as handle:
                root = ElementTree.parse(handle).getroot()
            for elem in root.iter():
                if _local_name(elem.tag) != "w":
                    continue
                text = _clean_ami_word(elem.text or "")
                if not text:
                    continue
                start = _float_attr(elem, "starttime")
                end = _float_attr(elem, "endtime")
                if start is None or end is None or end < start:
                    continue
                words.append((start, end, text, speaker))
    words.sort(key=lambda item: (item[0], item[1], item[3]))
    return _ami_words_to_segments(words, meeting_id=meeting_id, max_segments=max_segments)


def _download_once(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    temp_target = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=90) as response, temp_target.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    temp_target.replace(target)


def _ami_words_to_segments(
    words: list[tuple[float, float, str, str]],
    *,
    meeting_id: str,
    max_segments: int,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    current: list[tuple[float, float, str, str]] = []
    for word in words:
        if current and (
            word[3] != current[-1][3]
            or word[0] - current[-1][1] > 1.1
            or len(current) >= 26
        ):
            segments.append(_segment_from_ami_words(meeting_id, len(segments), current))
            current = []
            if len(segments) >= max_segments:
                break
        current.append(word)
    if current and len(segments) < max_segments:
        segments.append(_segment_from_ami_words(meeting_id, len(segments), current))
    return segments


def _segment_from_ami_words(
    meeting_id: str,
    index: int,
    words: list[tuple[float, float, str, str]],
) -> TranscriptSegment:
    text = _join_words([word[2] for word in words])
    return TranscriptSegment(
        id=f"{meeting_id}_seg_{index:04d}",
        session_id=f"open_source_{meeting_id}",
        start_s=words[0][0],
        end_s=words[-1][1],
        text=text,
        is_final=True,
        speaker_label=words[0][3],
        speaker_role="participant",
        speaker_confidence=1.0,
        source_segment_ids=[f"{meeting_id}_seg_{index:04d}"],
    )


def _join_words(words: list[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_ami_word(text: str) -> str:
    text = text.strip()
    if text in {"", "<blank>", "None"}:
        return ""
    return text


def _float_attr(elem: ElementTree.Element, name: str) -> float | None:
    value = elem.attrib.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _speaker_from_words_path(path: str) -> str:
    match = re.search(r"\.([A-Z])\.words\.xml$", path)
    return f"Speaker {match.group(1)}" if match else "Speaker"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
