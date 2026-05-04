# Speaker Identity Architecture

Brain Sidecar now separates four concerns that were previously blurred by the old voice-training UI:

- ASR transcribes words with Faster-Whisper.
- VAD/audio quality analysis rejects silence, clipping, very short samples, and low-volume enrollment audio.
- Speaker verification stores BP's speaker embeddings and a calibrated cosine threshold.
- Speaker identification labels only the matching cluster as `BP`; non-matching voices remain anonymous as `Speaker 1`, `Speaker 2`, and so on.

The local implementation prefers SpeechBrain ECAPA-TDNN embeddings through the optional `speaker` dependency group. If `torch`, `torchaudio`, or `speechbrain` are missing, `/api/speaker/status` reports the backend as unavailable instead of pretending enrollment worked. Tests use a deterministic backend so threshold, no-false-BP, and stable-anonymous-speaker behavior are covered without network/model downloads.

Raw enrollment audio is not retained. The database stores speaker profile rows, individual embeddings, a centroid embedding, enrollment quality metadata, thresholds, diarization segment labels, and feedback. Legacy ASR phrase-correction data is not used for speaker identity and `/api/voice/*` now returns `410 Gone`; the admin tool can quarantine known bad corrections after first backing up the DB.

Runtime diarization in this smallest complete version is embedding-based per-ASR-segment clustering. It does not yet run a full pyannote/NeMo diarization model, so overlapping speakers and sub-segment speaker turns remain a known limitation. The design keeps the storage/API boundary ready for a full diarization backend once dependencies are installed and model access is confirmed.

Operational commands:

```bash
python -m brain_sidecar.tools.speaker_admin evaluate-profile
python -m brain_sidecar.tools.speaker_admin reset-profile --backup
python -m brain_sidecar.tools.speaker_admin quarantine-legacy-asr-corrections --backup
```

To enable the real local embedding backend:

```bash
python -m pip install -e ".[speaker]"
```
