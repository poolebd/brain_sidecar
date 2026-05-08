# Energy Lens Implementation

Source artifact: `docs/research/energy-keyword-framework.md`, copied from `/home/bp/Downloads/deep-research-report.md`.

The generated lexicon lives at `brain_sidecar/data/energy_keywords.json` and is regenerated with:

```bash
python scripts/build_energy_keyword_lexicon.py
```

Use `python scripts/build_energy_keyword_lexicon.py --check` in CI or review to confirm the JSON is current.

## Model

The lens uses the report categories: commercial, service, operations, technical, technology, market, sustainability, finance, and regulatory. Report weights are treated only as heuristic probabilities that a phrase indicates an energy-consulting conversation. They are not search volume, factual confidence, or sales priority.

The runtime detector only reads final transcript segments from the current meeting window. It returns a compact frame with confidence, top categories, top keywords, evidence segment IDs, and a short evidence quote. Raw audio is never retained.

## Acronyms

Ambiguous acronyms such as PPA, REC, EUI, ROI, NPV, DER, BESS, BAS, and RCx require context. They become usable only when an expanded phrase is nearby, another non-acronym energy anchor appears in the same or adjacent final segment, or the acronym appears inside an explicit report phrase such as `virtual PPA` or `ASHRAE Level 2 audit`.

## Cards

The deterministic Energy Consulting Agent emits sparse clarification cards for:

- billing, tariffs, and load profiles
- audits, benchmarking, and EUI
- commissioning, controls, and analytics
- procurement, PPAs, RECs, and EACs
- demand response, flexibility, storage, and DERs
- sustainability and carbon accounting
- business case, incentives, and project finance
- commercial scoping

Cards stay transcript-backed, include source IDs and evidence quotes, and still pass the normal Sidecar quality gate.

## Privacy

The energy frame is ephemeral runtime status. It is not a database table, and keyword hits are not persisted. Listen-only sessions do not persist transcript text, notes, embeddings, summaries, or keyword artifacts.
