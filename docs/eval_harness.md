# Offline Sidecar Evaluation Harness

The eval harness replays local transcript fixtures through Brain Sidecar's
production transcript consolidation and note-quality gate. It is for tuning card
quality without waiting for real meetings, live audio, GPU ASR, Ollama, or
network access.

## Local Fixture Replay

```bash
python -m brain_sidecar.eval.replay \
  --fixture tests/fixtures/eval/energy_consulting_seed.jsonl \
  --output runtime/eval/latest_report.json \
  --markdown runtime/eval/latest_report.md
```

The committed fixtures are small JSONL files. Each line is one event:
`transcript`, `memory`, `candidate_card`, or `expectation`. Pipeline replay uses
deterministic transcript-grounded candidates plus fixture candidates, then runs
the production `NoteQualityGate`. Gate-only replay uses only fixture-provided
candidate cards:

```bash
python -m brain_sidecar.eval.replay \
  --mode gate-only \
  --fixture tests/fixtures/eval/energy_consulting_seed.jsonl \
  --output runtime/eval/gate_only.json
```

## Synthetic Noise

```bash
python -m brain_sidecar.eval.replay \
  --noise default \
  --fixture tests/fixtures/eval/energy_consulting_seed.jsonl \
  --output runtime/eval/noisy_report.json \
  --markdown runtime/eval/noisy_report.md
```

Noise transforms are deterministic and evaluation-only: overlap duplicates,
fragments, filler words, punctuation loss, and narrow ASR-like corruptions such
as Siemens to "demons", NERC to "narc", and SLD to "s l d". They are not used by
production note generation and are not a glossary.

## Synthetic Energy Fixtures

```bash
python -m brain_sidecar.eval.fixtures synth-energy \
  --count 25 \
  --seed 7 \
  --output runtime/eval/synthetic_energy.jsonl
```

The generated snippets are synthetic evaluation data, not meeting records and
not generation priors.

## Local Corpus Imports

The harness can normalize local plain-text and JSONL transcripts:

```bash
python -m brain_sidecar.eval.corpus import \
  --source-type plain-text \
  --input path/to/transcript.txt \
  --output runtime/eval/plain_text.jsonl

python -m brain_sidecar.eval.corpus import \
  --source-type jsonl \
  --input path/to/source.jsonl \
  --output runtime/eval/source_normalized.jsonl
```

Public/open meeting corpora such as AMI, ICSI, QMSum, MeetingBank, DialogSum,
and SAMSum should be acquired separately by the user under their licenses. The
repo does not download or vendor those corpora, and CI does not depend on them.
Source-specific importers can be added once a local corpus layout is present.

## Metrics

Reports include segment counts, collapsed final segments, generated/accepted and
suppressed card counts, memory-card count, cards per simulated time, source ID
coverage, evidence quote coverage, unsupported-topic violations, memory leakage,
identity-claim violations, duplicate cards, generic clarification counts,
expected topic hits/misses, suppression reasons, and recommendations.

The energy fixture is expected to pass with 4-12 accepted current-meeting cards,
100% source ID coverage, 100% evidence quote coverage, no unsupported topics, no
memory leakage, and all required consulting topics hit.

## Limitations

The default pipeline path is deterministic and intentionally conservative. It
does not measure live ASR latency, microphone behavior, Ollama quality, or UI
readability. Its job is to make quality-gate and replay regressions repeatable.
