from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from brain_sidecar.core.models import NoteCard, SessionRecord, TranscriptSegment, new_id


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_search_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _review_steps_for_payload(status: str, phase: str, message: str, progress: int) -> list[dict[str, Any]]:
    specs = [
        ("conditioning", "Conditioning"),
        ("transcribing", "High-accuracy ASR"),
        ("reviewing", "Segment review"),
        ("evaluating", "Meeting output"),
        ("completed", "Validation"),
    ]
    active_index = 0
    if status in {"queued", "paused_for_live"}:
        active_index = -1
    elif status == "running_asr":
        active_index = 1 if phase == "asr" else 0
    elif status == "running_cards":
        active_index = 2
    elif status == "running_summary":
        active_index = 3
    elif status in {"completed_awaiting_validation", "approved", "discarded"}:
        active_index = 4
    elif status in {"canceled", "error"}:
        active_index = min(4, max(0, progress // 25))

    steps: list[dict[str, Any]] = []
    for index, (key, label) in enumerate(specs):
        if active_index < 0:
            step_status = "pending"
            step_progress = 0
            step_message = "Queued." if index == 0 and status == "queued" else ""
        elif index < active_index:
            step_status = "completed"
            step_progress = 100
            step_message = "Completed."
        elif index == active_index:
            if status == "error":
                step_status = "error"
            elif status == "canceled":
                step_status = "canceled"
            elif status in {"completed_awaiting_validation", "approved", "discarded"}:
                step_status = "completed"
            else:
                step_status = "running"
            step_progress = max(0, min(100, progress))
            step_message = message
        else:
            step_status = "pending"
            step_progress = 0
            step_message = ""
        steps.append({
            "key": key,
            "label": label,
            "status": step_status,
            "message": step_message,
            "progress": step_progress,
        })
    return steps


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "brain_sidecar.sqlite3"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("pragma journal_mode=WAL")
        self._conn.execute("pragma busy_timeout=5000")
        self.init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                create table if not exists sessions (
                  id text primary key,
                  title text not null,
                  status text not null,
                  started_at real not null,
                  ended_at real
                );
                create table if not exists transcript_segments (
                  id text primary key,
                  session_id text not null,
                  start_s real not null,
                  end_s real not null,
                  text text not null,
                  is_final integer not null,
                  created_at real not null
                );
                create index if not exists idx_transcript_session on transcript_segments(session_id, created_at);
                create table if not exists schema_migrations (
                  name text primary key,
                  applied_at real not null
                );
                create table if not exists note_cards (
                  id text primary key,
                  session_id text not null,
                  kind text not null,
                  title text not null,
                  body text not null,
                  source_segment_ids text not null,
                  created_at real not null,
                  evidence_quote text not null default '',
                  owner text,
                  due_date text,
                  missing_info text
                );
                create index if not exists idx_notes_session on note_cards(session_id, created_at);
                create table if not exists library_roots (
                  id text primary key,
                  path text not null unique,
                  created_at real not null
                );
                create table if not exists document_chunks (
                  id text primary key,
                  source_path text not null,
                  chunk_index integer not null,
                  text text not null,
                  metadata text not null,
                  updated_at real not null
                );
                create unique index if not exists idx_chunks_source on document_chunks(source_path, chunk_index);
                create table if not exists embedding_records (
                  id text primary key,
                  source_type text not null,
                  source_id text not null,
                  text text not null,
                  metadata text not null,
                  vector_json text not null,
                  updated_at real not null
                );
                create index if not exists idx_embeddings_source on embedding_records(source_type, source_id);
                create table if not exists voice_profiles (
                  id text primary key,
                  name text not null,
                  active integer not null,
                  created_at real not null,
                  updated_at real not null
                );
                create table if not exists voice_enrollments (
                  id text primary key,
                  profile_id text not null,
                  status text not null,
                  started_at real not null,
                  completed_at real
                );
                create table if not exists voice_enrollment_phrases (
                  id text primary key,
                  enrollment_id text not null,
                  profile_id text not null,
                  phrase_index integer not null,
                  prompt text not null,
                  transcript_text text,
                  corrected_text text,
                  preferred_terms text not null,
                  profile_artifact_json text,
                  accepted integer not null,
                  recorded_at real,
                  accepted_at real
                );
                create index if not exists idx_voice_phrases_profile on voice_enrollment_phrases(profile_id, accepted_at);
                create table if not exists voice_corrections (
                  id text primary key,
                  profile_id text not null,
                  enrollment_phrase_id text,
                  heard_text text not null,
                  corrected_text text not null,
                  preferred_terms text not null,
                  profile_artifact_json text not null,
                  created_at real not null
                );
                create index if not exists idx_voice_corrections_profile on voice_corrections(profile_id, created_at);
                create table if not exists speaker_profiles (
                  id text primary key,
                  display_name text not null,
                  kind text not null,
                  active integer not null,
                  created_at real not null,
                  updated_at real not null,
                  embedding_model text,
                  embedding_model_version text,
                  threshold real,
                  notes text
                );
                create table if not exists speaker_embeddings (
                  id text primary key,
                  profile_id text not null,
                  vector_json text not null,
                  vector_dim integer not null,
                  source text not null,
                  quality_score real not null,
                  duration_seconds real not null,
                  sample_rate integer not null,
                  embedding_model text not null,
                  embedding_model_version text not null,
                  created_at real not null,
                  metadata_json text not null
                );
                create index if not exists idx_speaker_embeddings_profile on speaker_embeddings(profile_id, source, created_at);
                create table if not exists speaker_enrollments (
                  id text primary key,
                  profile_id text not null,
                  status text not null,
                  started_at real not null,
                  completed_at real
                );
                create index if not exists idx_speaker_enrollments_profile on speaker_enrollments(profile_id, started_at);
                create table if not exists speaker_enrollment_samples (
                  id text primary key,
                  enrollment_id text not null,
                  profile_id text not null,
                  embedding_id text,
                  duration_seconds real not null,
                  usable_speech_seconds real not null,
                  quality_score real not null,
                  issues_json text not null,
                  created_at real not null,
                  metadata_json text not null
                );
                create index if not exists idx_speaker_samples_enrollment on speaker_enrollment_samples(enrollment_id, created_at);
                create table if not exists diarization_sessions (
                  id text primary key,
                  audio_source_id text,
                  created_at real not null,
                  diarization_model text,
                  diarization_model_version text,
                  asr_model text,
                  asr_model_version text,
                  status text not null,
                  metadata_json text not null
                );
                create table if not exists diarization_segments (
                  id text primary key,
                  session_id text not null,
                  transcript_segment_id text,
                  start_ms integer not null,
                  end_ms integer not null,
                  diarization_speaker_id text,
                  display_speaker_label text,
                  matched_profile_id text,
                  match_confidence real,
                  match_score real,
                  transcript_text text,
                  is_overlap integer not null,
                  finalized integer not null,
                  created_at real not null,
                  metadata_json text not null
                );
                create index if not exists idx_diarization_segments_session on diarization_segments(session_id, start_ms);
                create table if not exists speaker_label_feedback (
                  id text primary key,
                  session_id text not null,
                  segment_id text,
                  old_label text,
                  new_label text not null,
                  feedback_type text not null,
                  created_at real not null,
                  applied_to_training integer not null,
                  metadata_json text not null
                );
                create index if not exists idx_speaker_feedback_session on speaker_label_feedback(session_id, created_at);
                create table if not exists session_memory_summaries (
                  session_id text primary key,
                  title text not null,
                  summary text not null,
                  topics_json text not null,
                  decisions_json text not null,
                  actions_json text not null,
                  unresolved_questions_json text not null,
                  entities_json text not null,
                  lessons_json text not null,
                  source_segment_ids_json text not null,
                  review_standard text not null default '',
                  key_points_json text not null default '[]',
                  projects_json text not null default '[]',
                  project_workstreams_json text not null default '[]',
                  technical_findings_json text not null default '[]',
                  portfolio_rollup_json text not null default '{}',
                  review_metrics_json text not null default '{}',
                  risks_json text not null default '[]',
                  coverage_notes_json text not null default '[]',
                  reference_context_json text not null default '[]',
                  context_diagnostics_json text not null default '{}',
                  created_at real not null,
                  updated_at real not null
                );
                create table if not exists review_jobs (
                  id text primary key,
                  source text not null,
                  title text not null,
                  status text not null,
                  validation_status text not null,
                  phase text,
                  progress_pct real not null default 0,
                  message text,
                  queue_position integer,
                  session_id text,
                  live_id text,
                  source_filename text,
                  temporary_audio_path text,
                  temporary_audio_retained integer not null default 0,
                  audio_deleted_at real,
                  audio_delete_error text,
                  cleanup_status text,
                  result_json text,
                  error_message text,
                  created_at real not null,
                  updated_at real not null,
                  started_at real,
                  completed_at real,
                  approved_at real,
                  discarded_at real,
                  canceled_at real
                );
                create index if not exists idx_review_jobs_status on review_jobs(status, created_at);
                create index if not exists idx_review_jobs_updated on review_jobs(updated_at);
                create table if not exists company_refs (
                  id text primary key,
                  canonical_name text not null,
                  entity_type text not null default 'company',
                  domain text not null default '',
                  description text not null,
                  website text,
                  metadata_json text not null default '{}',
                  sources_json text not null default '[]',
                  active integer not null default 1,
                  updated_at real not null
                );
                create table if not exists company_ref_aliases (
                  id text primary key,
                  ref_id text not null,
                  alias text not null,
                  normalized_alias text not null,
                  alias_type text not null,
                  weight real not null default 1.0,
                  requires_context integer not null default 0,
                  context_terms_json text not null default '[]',
                  negative_terms_json text not null default '[]'
                );
                create index if not exists idx_company_ref_aliases_normalized on company_ref_aliases(normalized_alias);
                create index if not exists idx_company_ref_aliases_ref on company_ref_aliases(ref_id);
                create index if not exists idx_company_refs_active on company_refs(active);
                """
            )
            self._ensure_column("voice_corrections", "quarantined", "integer not null default 0")
            self._ensure_column("voice_corrections", "quarantine_reason", "text")
            self._apply_migrations()
            self.conn.commit()

    def _apply_migrations(self) -> None:
        self._ensure_column("sessions", "save_transcript", "integer")
        self._ensure_column("transcript_segments", "speaker_role", "text")
        self._ensure_column("transcript_segments", "speaker_label", "text")
        self._ensure_column("transcript_segments", "speaker_confidence", "real")
        self._ensure_column("transcript_segments", "speaker_match_score", "real")
        self._ensure_column("transcript_segments", "speaker_match_reason", "text")
        self._ensure_column("transcript_segments", "speaker_low_confidence", "integer")
        self._ensure_column("transcript_segments", "diarization_speaker_id", "text")
        self._ensure_column("note_cards", "evidence_quote", "text not null default ''")
        self._ensure_column("note_cards", "owner", "text")
        self._ensure_column("note_cards", "due_date", "text")
        self._ensure_column("note_cards", "missing_info", "text")
        self._ensure_column("session_memory_summaries", "key_points_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "projects_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "review_standard", "text not null default ''")
        self._ensure_column("session_memory_summaries", "project_workstreams_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "technical_findings_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "portfolio_rollup_json", "text not null default '{}'")
        self._ensure_column("session_memory_summaries", "review_metrics_json", "text not null default '{}'")
        self._ensure_column("session_memory_summaries", "risks_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "coverage_notes_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "reference_context_json", "text not null default '[]'")
        self._ensure_column("session_memory_summaries", "context_diagnostics_json", "text not null default '{}'")
        self._record_migration("20260507_session_retention")
        self._record_migration("20260504_transcript_speaker_fields")
        self._record_migration("20260504_session_memory_summaries")
        self._record_migration("20260505_note_evidence_fields")
        self._record_migration("20260510_review_jobs")
        self._record_migration("20260510_review_summary_fields")
        self._record_migration("20260510_review_summary_context_fields")
        self._record_migration("20260510_energy_consultant_review_standard")
        self._record_migration("20260510_review_terminal_summary_fields")

    def _record_migration(self, name: str) -> None:
        self.conn.execute(
            "insert or ignore into schema_migrations(name, applied_at) values (?, ?)",
            (name, time.time()),
        )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.conn.execute(f"pragma table_info({table})").fetchall()
        }
        if column not in columns:
            self.conn.execute(f"alter table {table} add column {column} {definition}")

    def ensure_voice_profile(self, profile_id: str = "default", name: str = "My Voice") -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert or ignore into voice_profiles(id, name, active, created_at, updated_at)
                values (?, ?, ?, ?, ?)
                """,
                (profile_id, name, 1, now, now),
            )
            row = self.conn.execute("select * from voice_profiles where id = ?", (profile_id,)).fetchone()
            self.conn.commit()
        return dict(row)

    def set_voice_profile_active(self, active: bool, profile_id: str = "default") -> dict[str, Any]:
        self.ensure_voice_profile(profile_id)
        with self._lock:
            self.conn.execute(
                "update voice_profiles set active = ?, updated_at = ? where id = ?",
                (1 if active else 0, time.time(), profile_id),
            )
            row = self.conn.execute("select * from voice_profiles where id = ?", (profile_id,)).fetchone()
            self.conn.commit()
        return dict(row)

    def create_voice_enrollment(
        self,
        phrases: list[str],
        profile_id: str = "default",
    ) -> dict[str, Any]:
        self.ensure_voice_profile(profile_id)
        enrollment_id = new_id("venr")
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into voice_enrollments(id, profile_id, status, started_at, completed_at)
                values (?, ?, ?, ?, ?)
                """,
                (enrollment_id, profile_id, "created", now, None),
            )
            for index, prompt in enumerate(phrases):
                self.conn.execute(
                    """
                    insert into voice_enrollment_phrases(
                      id, enrollment_id, profile_id, phrase_index, prompt, transcript_text,
                      corrected_text, preferred_terms, profile_artifact_json, accepted,
                      recorded_at, accepted_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id("vphr"),
                        enrollment_id,
                        profile_id,
                        index,
                        prompt,
                        None,
                        None,
                        "[]",
                        None,
                        0,
                        None,
                        None,
                    ),
                )
            self.conn.commit()
        return self.voice_enrollment(enrollment_id)

    def voice_enrollment(self, enrollment_id: str) -> dict[str, Any]:
        enrollment = self.conn.execute(
            "select * from voice_enrollments where id = ?",
            (enrollment_id,),
        ).fetchone()
        if enrollment is None:
            raise KeyError(f"Voice enrollment not found: {enrollment_id}")
        return {
            **dict(enrollment),
            "phrases": self.voice_enrollment_phrases(enrollment_id),
        }

    def voice_enrollment_phrases(self, enrollment_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from voice_enrollment_phrases
            where enrollment_id = ?
            order by phrase_index
            """,
            (enrollment_id,),
        ).fetchall()
        return [self._voice_phrase_row_to_dict(row) for row in rows]

    def voice_enrollment_phrase(self, enrollment_id: str, phrase_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """
            select * from voice_enrollment_phrases
            where enrollment_id = ? and id = ?
            """,
            (enrollment_id, phrase_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Voice enrollment phrase not found: {phrase_id}")
        return self._voice_phrase_row_to_dict(row)

    def save_voice_phrase_transcript(self, enrollment_id: str, phrase_id: str, transcript_text: str) -> dict[str, Any]:
        with self._lock:
            self.conn.execute(
                """
                update voice_enrollment_phrases
                set transcript_text = ?, corrected_text = coalesce(corrected_text, ?), recorded_at = ?
                where enrollment_id = ? and id = ?
                """,
                (transcript_text, transcript_text, time.time(), enrollment_id, phrase_id),
            )
            self.conn.commit()
        return self.voice_enrollment_phrase(enrollment_id, phrase_id)

    def accept_voice_phrase(
        self,
        enrollment_id: str,
        phrase_id: str,
        corrected_text: str,
        preferred_terms: list[str],
        profile_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        phrase = self.voice_enrollment_phrase(enrollment_id, phrase_id)
        now = time.time()
        terms_json = json.dumps(preferred_terms)
        artifact_json = json.dumps(profile_artifact, sort_keys=True)
        heard_text = phrase.get("transcript_text") or ""
        with self._lock:
            self.conn.execute(
                """
                update voice_enrollment_phrases
                set corrected_text = ?, preferred_terms = ?, profile_artifact_json = ?,
                    accepted = 1, accepted_at = ?
                where enrollment_id = ? and id = ?
                """,
                (corrected_text, terms_json, artifact_json, now, enrollment_id, phrase_id),
            )
            self.conn.execute(
                """
                insert into voice_corrections(
                  id, profile_id, enrollment_phrase_id, heard_text, corrected_text,
                  preferred_terms, profile_artifact_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("vcorr"),
                    phrase["profile_id"],
                    phrase_id,
                    heard_text,
                    corrected_text,
                    terms_json,
                    artifact_json,
                    now,
                ),
            )
            accepted_count = self.conn.execute(
                """
                select count(*) as accepted_count
                from voice_enrollment_phrases
                where enrollment_id = ? and accepted = 1
                """,
                (enrollment_id,),
            ).fetchone()["accepted_count"]
            total_count = self.conn.execute(
                """
                select count(*) as total_count
                from voice_enrollment_phrases
                where enrollment_id = ?
                """,
                (enrollment_id,),
            ).fetchone()["total_count"]
            if accepted_count >= total_count:
                self.conn.execute(
                    "update voice_enrollments set status = ?, completed_at = ? where id = ?",
                    ("completed", now, enrollment_id),
                )
            self.conn.execute(
                "update voice_profiles set updated_at = ? where id = ?",
                (now, phrase["profile_id"]),
            )
            self.conn.commit()
        return self.voice_enrollment_phrase(enrollment_id, phrase_id)

    def voice_corrections(self, profile_id: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from voice_corrections
            where profile_id = ? and coalesce(quarantined, 0) = 0
            order by created_at desc
            limit ?
            """,
            (profile_id, limit),
        ).fetchall()
        return [
            {
                **dict(row),
                "preferred_terms": json.loads(row["preferred_terms"]),
                "profile_artifact": json.loads(row["profile_artifact_json"]),
            }
            for row in rows
        ]

    def quarantine_voice_corrections(
        self,
        pairs: list[tuple[str, str]],
        *,
        profile_id: str = "default",
        reason: str = "quarantined by speaker identity migration",
    ) -> int:
        if not pairs:
            return 0
        with self._lock:
            count = 0
            for heard, corrected in pairs:
                cursor = self.conn.execute(
                    """
                    update voice_corrections
                    set quarantined = 1, quarantine_reason = ?
                    where profile_id = ?
                      and lower(trim(heard_text)) = lower(trim(?))
                      and lower(trim(corrected_text)) = lower(trim(?))
                    """,
                    (reason, profile_id, heard, corrected),
                )
                count += cursor.rowcount
            self.conn.commit()
        return count

    def accepted_voice_phrase_count(self, profile_id: str = "default") -> int:
        row = self.conn.execute(
            """
            select count(*) as accepted_count
            from voice_enrollment_phrases phrase
            where profile_id = ? and accepted = 1
              and not exists (
                select 1 from voice_corrections correction
                where correction.enrollment_phrase_id = phrase.id
                  and coalesce(correction.quarantined, 0) = 1
              )
            """,
            (profile_id,),
        ).fetchone()
        return int(row["accepted_count"])

    def ensure_speaker_profile(
        self,
        profile_id: str,
        *,
        display_name: str,
        kind: str,
        embedding_model: str | None = None,
        embedding_model_version: str | None = None,
        threshold: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert or ignore into speaker_profiles(
                  id, display_name, kind, active, created_at, updated_at,
                  embedding_model, embedding_model_version, threshold, notes
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    display_name,
                    kind,
                    0,
                    now,
                    now,
                    embedding_model,
                    embedding_model_version,
                    threshold,
                    notes,
                ),
            )
            row = self.conn.execute("select * from speaker_profiles where id = ?", (profile_id,)).fetchone()
            self.conn.commit()
        return self._speaker_profile_row_to_dict(row)

    def update_speaker_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None = None,
        active: bool | None = None,
        embedding_model: str | None = None,
        embedding_model_version: str | None = None,
        threshold: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        existing = self.conn.execute("select * from speaker_profiles where id = ?", (profile_id,)).fetchone()
        if existing is None:
            raise KeyError(f"Speaker profile not found: {profile_id}")
        values = {
            "display_name": display_name if display_name is not None else existing["display_name"],
            "active": (1 if active else 0) if active is not None else existing["active"],
            "embedding_model": embedding_model if embedding_model is not None else existing["embedding_model"],
            "embedding_model_version": (
                embedding_model_version
                if embedding_model_version is not None
                else existing["embedding_model_version"]
            ),
            "threshold": threshold if threshold is not None else existing["threshold"],
            "notes": notes if notes is not None else existing["notes"],
        }
        with self._lock:
            self.conn.execute(
                """
                update speaker_profiles
                set display_name = ?, active = ?, embedding_model = ?,
                    embedding_model_version = ?, threshold = ?, notes = ?, updated_at = ?
                where id = ?
                """,
                (
                    values["display_name"],
                    values["active"],
                    values["embedding_model"],
                    values["embedding_model_version"],
                    values["threshold"],
                    values["notes"],
                    time.time(),
                    profile_id,
                ),
            )
            row = self.conn.execute("select * from speaker_profiles where id = ?", (profile_id,)).fetchone()
            self.conn.commit()
        return self._speaker_profile_row_to_dict(row)

    def create_speaker_enrollment(self, profile_id: str) -> dict[str, Any]:
        enrollment_id = new_id("spenr")
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into speaker_enrollments(id, profile_id, status, started_at, completed_at)
                values (?, ?, ?, ?, ?)
                """,
                (enrollment_id, profile_id, "created", now, None),
            )
            self.conn.commit()
        return self.speaker_enrollment(enrollment_id)

    def speaker_enrollment(self, enrollment_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "select * from speaker_enrollments where id = ?",
            (enrollment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Speaker enrollment not found: {enrollment_id}")
        return dict(row)

    def set_speaker_enrollment_status(self, enrollment_id: str, status: str) -> dict[str, Any]:
        completed_at = time.time() if status == "completed" else None
        with self._lock:
            self.conn.execute(
                """
                update speaker_enrollments
                set status = ?, completed_at = coalesce(?, completed_at)
                where id = ?
                """,
                (status, completed_at, enrollment_id),
            )
            self.conn.commit()
        return self.speaker_enrollment(enrollment_id)

    def add_speaker_enrollment_sample(
        self,
        *,
        enrollment_id: str,
        profile_id: str,
        embedding_id: str,
        duration_seconds: float,
        usable_speech_seconds: float,
        quality_score: float,
        issues: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        sample_id = new_id("spsmp")
        with self._lock:
            self.conn.execute(
                """
                insert into speaker_enrollment_samples(
                  id, enrollment_id, profile_id, embedding_id, duration_seconds,
                  usable_speech_seconds, quality_score, issues_json, created_at, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    enrollment_id,
                    profile_id,
                    embedding_id,
                    duration_seconds,
                    usable_speech_seconds,
                    quality_score,
                    json.dumps(issues),
                    time.time(),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
            self.conn.commit()
        row = self.conn.execute(
            "select * from speaker_enrollment_samples where id = ?",
            (sample_id,),
        ).fetchone()
        return self._speaker_sample_row_to_dict(row)

    def speaker_enrollment_samples(self, enrollment_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from speaker_enrollment_samples
            where enrollment_id = ?
            order by created_at
            """,
            (enrollment_id,),
        ).fetchall()
        return [self._speaker_sample_row_to_dict(row) for row in rows]

    def add_speaker_embedding(
        self,
        *,
        profile_id: str,
        vector: list[float],
        source: str,
        quality_score: float,
        duration_seconds: float,
        sample_rate: int,
        embedding_model: str,
        embedding_model_version: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        embedding_id = new_id("spemb")
        vector_json = json.dumps([float(value) for value in vector])
        with self._lock:
            if source == "centroid":
                self.conn.execute(
                    "delete from speaker_embeddings where profile_id = ? and source = ?",
                    (profile_id, "centroid"),
                )
            self.conn.execute(
                """
                insert into speaker_embeddings(
                  id, profile_id, vector_json, vector_dim, source, quality_score,
                  duration_seconds, sample_rate, embedding_model, embedding_model_version,
                  created_at, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding_id,
                    profile_id,
                    vector_json,
                    len(vector),
                    source,
                    float(quality_score),
                    float(duration_seconds),
                    int(sample_rate),
                    embedding_model,
                    embedding_model_version,
                    time.time(),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
            self.conn.commit()
        row = self.conn.execute("select * from speaker_embeddings where id = ?", (embedding_id,)).fetchone()
        return self._speaker_embedding_row_to_dict(row)

    def speaker_embeddings(
        self,
        profile_id: str,
        *,
        sources: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if sources:
            placeholders = ",".join("?" for _ in sources)
            rows = self.conn.execute(
                f"""
                select * from speaker_embeddings
                where profile_id = ? and source in ({placeholders})
                order by created_at
                """,
                (profile_id, *sources),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                select * from speaker_embeddings
                where profile_id = ?
                order by created_at
                """,
                (profile_id,),
            ).fetchall()
        return [self._speaker_embedding_row_to_dict(row) for row in rows]

    def speaker_embedding(self, embedding_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "select * from speaker_embeddings where id = ?",
            (embedding_id,),
        ).fetchone()
        return self._speaker_embedding_row_to_dict(row) if row is not None else None

    def prune_speaker_enrollment_artifacts(
        self,
        *,
        profile_id: str,
        keep_enrollment_id: str,
        keep_embedding_ids: list[str],
    ) -> dict[str, int]:
        placeholders = ",".join("?" for _ in keep_embedding_ids) or "null"
        embedding_params: tuple[Any, ...] = (profile_id, *keep_embedding_ids)
        sample_params: tuple[Any, ...] = (profile_id, keep_enrollment_id, *keep_embedding_ids)
        with self._lock:
            embedding_count = self.conn.execute(
                f"""
                select count(*) as count from speaker_embeddings
                where profile_id = ? and source = 'enrollment' and id not in ({placeholders})
                """,
                embedding_params,
            ).fetchone()["count"]
            sample_count = self.conn.execute(
                f"""
                select count(*) as count from speaker_enrollment_samples
                where profile_id = ? and (enrollment_id != ? or embedding_id not in ({placeholders}))
                """,
                sample_params,
            ).fetchone()["count"]
            self.conn.execute(
                f"""
                delete from speaker_embeddings
                where profile_id = ? and source = 'enrollment' and id not in ({placeholders})
                """,
                embedding_params,
            )
            self.conn.execute(
                f"""
                delete from speaker_enrollment_samples
                where profile_id = ? and (enrollment_id != ? or embedding_id not in ({placeholders}))
                """,
                sample_params,
            )
            self.conn.execute(
                """
                delete from speaker_enrollments
                where profile_id = ? and id != ? and status != 'completed'
                """,
                (profile_id, keep_enrollment_id),
            )
            self.conn.commit()
        return {"embeddings": int(embedding_count), "samples": int(sample_count)}

    def speaker_centroid_embedding(self, profile_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            select * from speaker_embeddings
            where profile_id = ? and source = 'centroid'
            order by created_at desc
            limit 1
            """,
            (profile_id,),
        ).fetchone()
        return self._speaker_embedding_row_to_dict(row) if row is not None else None

    def reset_speaker_profile(self, profile_id: str) -> dict[str, int]:
        with self._lock:
            embedding_count = self.conn.execute(
                "select count(*) as count from speaker_embeddings where profile_id = ?",
                (profile_id,),
            ).fetchone()["count"]
            sample_count = self.conn.execute(
                "select count(*) as count from speaker_enrollment_samples where profile_id = ?",
                (profile_id,),
            ).fetchone()["count"]
            enrollment_count = self.conn.execute(
                "select count(*) as count from speaker_enrollments where profile_id = ?",
                (profile_id,),
            ).fetchone()["count"]
            self.conn.execute("delete from speaker_embeddings where profile_id = ?", (profile_id,))
            self.conn.execute("delete from speaker_enrollment_samples where profile_id = ?", (profile_id,))
            self.conn.execute("delete from speaker_enrollments where profile_id = ?", (profile_id,))
            self.conn.execute(
                """
                update speaker_profiles
                set active = 0, threshold = null, notes = null, updated_at = ?
                where id = ?
                """,
                (time.time(), profile_id),
            )
            self.conn.commit()
        return {
            "embeddings": int(embedding_count),
            "samples": int(sample_count),
            "enrollments": int(enrollment_count),
        }

    def add_diarization_session(
        self,
        *,
        session_id: str,
        audio_source_id: str | None,
        diarization_model: str,
        diarization_model_version: str,
        asr_model: str | None,
        asr_model_version: str | None,
        status: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert or ignore into diarization_sessions(
                  id, audio_source_id, created_at, diarization_model,
                  diarization_model_version, asr_model, asr_model_version,
                  status, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    audio_source_id,
                    now,
                    diarization_model,
                    diarization_model_version,
                    asr_model,
                    asr_model_version,
                    status,
                    json.dumps(metadata, sort_keys=True),
                ),
            )
            self.conn.commit()
        row = self.conn.execute("select * from diarization_sessions where id = ?", (session_id,)).fetchone()
        return dict(row)

    def add_diarization_segment(
        self,
        *,
        session_id: str,
        segment_id: str,
        start_ms: int,
        end_ms: int,
        diarization_speaker_id: str | None,
        display_speaker_label: str | None,
        matched_profile_id: str | None,
        match_confidence: float | None,
        match_score: float | None,
        transcript_text: str | None,
        is_overlap: bool,
        finalized: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        row_id = new_id("dseg")
        with self._lock:
            self.conn.execute(
                """
                insert into diarization_segments(
                  id, session_id, transcript_segment_id, start_ms, end_ms,
                  diarization_speaker_id, display_speaker_label, matched_profile_id,
                  match_confidence, match_score, transcript_text, is_overlap,
                  finalized, created_at, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    session_id,
                    segment_id,
                    int(start_ms),
                    int(end_ms),
                    diarization_speaker_id,
                    display_speaker_label,
                    matched_profile_id,
                    match_confidence,
                    match_score,
                    transcript_text,
                    1 if is_overlap else 0,
                    1 if finalized else 0,
                    time.time(),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
            self.conn.commit()
        row = self.conn.execute("select * from diarization_segments where id = ?", (row_id,)).fetchone()
        return self._diarization_segment_row_to_dict(row)

    def recent_diarization_segments(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from diarization_segments
            order by created_at desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        return [self._diarization_segment_row_to_dict(row) for row in rows]

    def add_speaker_label_feedback(
        self,
        *,
        session_id: str,
        segment_id: str | None,
        old_label: str | None,
        new_label: str,
        feedback_type: str,
        applied_to_training: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        feedback_id = new_id("spfb")
        with self._lock:
            self.conn.execute(
                """
                insert into speaker_label_feedback(
                  id, session_id, segment_id, old_label, new_label,
                  feedback_type, created_at, applied_to_training, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    session_id,
                    segment_id,
                    old_label,
                    new_label,
                    feedback_type,
                    time.time(),
                    1 if applied_to_training else 0,
                    json.dumps(metadata, sort_keys=True),
                ),
            )
            self.conn.commit()
        row = self.conn.execute("select * from speaker_label_feedback where id = ?", (feedback_id,)).fetchone()
        return {
            **dict(row),
            "applied_to_training": bool(row["applied_to_training"]),
            "metadata": json.loads(row["metadata_json"]),
        }

    def create_session(self, title: str) -> SessionRecord:
        record = SessionRecord(id=new_id("ses"), title=title, status="created")
        with self._lock:
            self.conn.execute(
                "insert into sessions(id, title, status, started_at, ended_at) values (?, ?, ?, ?, ?)",
                (record.id, record.title, record.status, record.started_at, record.ended_at),
            )
            self.conn.commit()
        return record

    def set_session_status(
        self,
        session_id: str,
        status: str,
        ended_at: float | None = None,
        *,
        save_transcript: bool | None = None,
    ) -> None:
        save_value = None if save_transcript is None else 1 if save_transcript else 0
        with self._lock:
            if save_value is None:
                self.conn.execute(
                    "update sessions set status = ?, ended_at = coalesce(?, ended_at) where id = ?",
                    (status, ended_at, session_id),
                )
            else:
                self.conn.execute(
                    "update sessions set status = ?, ended_at = coalesce(?, ended_at), save_transcript = ? where id = ?",
                    (status, ended_at, save_value, session_id),
                )
            self.conn.commit()

    def update_session_title(self, session_id: str, title: str) -> dict[str, Any]:
        clean_title = " ".join(title.split()).strip()
        if not clean_title:
            raise ValueError("Session title cannot be empty.")
        with self._lock:
            updated = self.conn.execute(
                "update sessions set title = ? where id = ?",
                (clean_title[:180], session_id),
            )
            if updated.rowcount == 0:
                raise KeyError(session_id)
            self.conn.commit()
        return self.session_metadata(session_id)

    def list_sessions(
        self,
        *,
        limit: int = 50,
        include_empty: bool = True,
        status: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("s.status = ?")
            params.append(status)
        if query:
            where.append("lower(s.title) like ?")
            params.append(f"%{query.lower()}%")
        if not include_empty:
            where.append("(coalesce(t.transcript_count, 0) > 0 or coalesce(n.note_count, 0) > 0 or m.session_id is not null)")
        where_sql = f"where {' and '.join(where)}" if where else ""
        rows = self.conn.execute(
            f"""
            select
              s.*,
              coalesce(t.transcript_count, 0) as transcript_count,
              coalesce(n.note_count, 0) as note_count,
              m.session_id is not null as summary_exists
            from sessions s
            left join (
              select session_id, count(*) as transcript_count
              from transcript_segments
              group by session_id
            ) t on t.session_id = s.id
            left join (
              select session_id, count(*) as note_count
              from note_cards
              group by session_id
            ) n on n.session_id = s.id
            left join session_memory_summaries m on m.session_id = s.id
            {where_sql}
            order by coalesce(s.ended_at, s.started_at) desc
            limit ?
            """,
            (*params, max(1, min(limit, 200))),
        ).fetchall()
        return [self._session_row_to_dict(row) for row in rows]

    def session_metadata(self, session_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            """
            select
              s.*,
              coalesce(t.transcript_count, 0) as transcript_count,
              coalesce(n.note_count, 0) as note_count,
              m.session_id is not null as summary_exists
            from sessions s
            left join (
              select session_id, count(*) as transcript_count
              from transcript_segments
              group by session_id
            ) t on t.session_id = s.id
            left join (
              select session_id, count(*) as note_count
              from note_cards
              group by session_id
            ) n on n.session_id = s.id
            left join session_memory_summaries m on m.session_id = s.id
            where s.id = ?
            """,
            (session_id,),
        ).fetchall()
        if not rows:
            raise KeyError(session_id)
        return self._session_row_to_dict(rows[0])

    def session_detail(self, session_id: str) -> dict[str, Any]:
        metadata = self.session_metadata(session_id)
        saved = metadata["save_transcript"] is not False
        return {
            **metadata,
            "transcript_segments": self.session_transcript_segments(session_id) if saved else [],
            "note_cards": self.session_note_cards(session_id) if saved else [],
            "summary": self.session_memory_summary(session_id) if saved else None,
            "transcript_redacted": not saved,
            "raw_audio_retained": False,
        }

    def session_transcript_segments(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from transcript_segments
            where session_id = ?
            order by start_s, created_at
            """,
            (session_id,),
        ).fetchall()
        return [self._transcript_row_to_segment(row).to_dict() for row in rows]

    def session_note_cards(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from note_cards
            where session_id = ?
            order by created_at
            """,
            (session_id,),
        ).fetchall()
        return [self._note_row_to_dict(row) for row in rows]

    def session_memory_summary(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "select * from session_memory_summaries where session_id = ?",
            (session_id,),
        ).fetchone()
        return self._session_memory_summary_row_to_dict(row) if row else None

    def create_review_job(
        self,
        *,
        job_id: str,
        source: str,
        title: str,
        status: str,
        validation_status: str,
        source_filename: str | None,
        temporary_audio_path: Path,
        live_id: str | None = None,
        message: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into review_jobs(
                  id, source, title, status, validation_status, phase, progress_pct,
                  message, queue_position, session_id, live_id, source_filename,
                  temporary_audio_path, temporary_audio_retained, audio_deleted_at,
                  audio_delete_error, cleanup_status, result_json, error_message,
                  created_at, updated_at, started_at, completed_at, approved_at,
                  discarded_at, canceled_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    source,
                    title,
                    status,
                    validation_status,
                    status,
                    0.0,
                    message,
                    None,
                    None,
                    live_id,
                    source_filename,
                    str(temporary_audio_path),
                    1,
                    None,
                    None,
                    "retained",
                    None,
                    None,
                    now,
                    now,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            self.conn.commit()
        self.refresh_review_queue_positions()
        return self.review_job(job_id)

    def list_review_jobs(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if status:
            where = "where status = ?"
            params.append(status)
        with self._lock:
            rows = self.conn.execute(
                f"""
                select * from review_jobs
                {where}
                order by
                  case when status in ('queued', 'paused_for_live') then 0
                       when status like 'running_%' then 1
                       when status = 'completed_awaiting_validation' then 2
                       else 3 end,
                  coalesce(queue_position, 999999),
                  updated_at desc
                limit ?
                """,
                (*params, max(1, min(int(limit), 200))),
            ).fetchall()
        return [self._review_job_row_to_dict(row) for row in rows]

    def review_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute("select * from review_jobs where id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._review_job_row_to_dict(row)

    def latest_review_job(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                """
                select * from review_jobs
                order by
                  case when status not in ('approved', 'discarded', 'canceled', 'error') then 0 else 1 end,
                  updated_at desc
                limit 1
                """
            ).fetchone()
        return self._review_job_row_to_dict(row) if row is not None else None

    def refresh_review_queue_positions(self) -> None:
        with self._lock:
            rows = self.conn.execute(
                """
                select id from review_jobs
                where status in ('queued', 'paused_for_live')
                order by created_at
                """
            ).fetchall()
            for index, row in enumerate(rows, start=1):
                self.conn.execute(
                    "update review_jobs set queue_position = ? where id = ?",
                    (index, row["id"]),
                )
            self.conn.execute(
                "update review_jobs set queue_position = null where status not in ('queued', 'paused_for_live')"
            )
            self.conn.commit()

    def claim_next_review_job(self) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            row = self.conn.execute(
                """
                select * from review_jobs
                where status in ('queued', 'paused_for_live')
                order by created_at
                limit 1
                """
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                """
                update review_jobs
                set status = ?, phase = ?, progress_pct = ?, message = ?,
                    queue_position = null, started_at = coalesce(started_at, ?),
                    updated_at = ?
                where id = ? and status in ('queued', 'paused_for_live')
                """,
                ("running_asr", "asr", 5.0, "Preparing high-accuracy Review ASR.", now, now, row["id"]),
            )
            self.conn.commit()
        self.refresh_review_queue_positions()
        return self.review_job(str(row["id"]))

    def update_review_job_progress(
        self,
        job_id: str,
        *,
        status: str,
        phase: str | None,
        progress_pct: float,
        message: str = "",
        error_message: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                update review_jobs
                set status = ?, phase = ?, progress_pct = ?, message = ?,
                    error_message = ?, updated_at = ?
                where id = ?
                """,
                (status, phase, float(max(0.0, min(100.0, progress_pct))), message, error_message, now, job_id),
            )
            self.conn.commit()
        self.refresh_review_queue_positions()
        return self.review_job(job_id)

    def complete_review_job(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                update review_jobs
                set status = ?, validation_status = ?, phase = ?, progress_pct = ?,
                    message = ?, result_json = ?, completed_at = ?, updated_at = ?
                where id = ?
                """,
                (
                    "completed_awaiting_validation",
                    "pending",
                    "awaiting_validation",
                    100.0,
                    "Review output ready for validation.",
                    json.dumps(result, sort_keys=True),
                    now,
                    now,
                    job_id,
                ),
            )
            self.conn.commit()
        return self.review_job(job_id)

    def approve_review_job(self, job_id: str, session_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            current = self.review_job(job_id)
            if current["status"] == "approved":
                return current
            if current["status"] != "completed_awaiting_validation":
                raise ValueError(f"Cannot approve review job in status {current['status']}.")
            self.conn.execute(
                """
                update review_jobs
                set status = ?, validation_status = ?, session_id = ?, approved_at = ?,
                    phase = ?, message = ?, updated_at = ?
                where id = ?
                """,
                ("approved", "approved", session_id, now, "approved", "Review approved and saved.", now, job_id),
            )
            self.conn.commit()
        return self.review_job(job_id)

    def discard_review_job(self, job_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            current = self.review_job(job_id)
            if current["status"] == "discarded":
                return current
            if current["status"] not in {"queued", "paused_for_live", "completed_awaiting_validation", "error"}:
                raise ValueError(f"Cannot discard review job in status {current['status']}.")
            self.conn.execute(
                """
                update review_jobs
                set status = ?, validation_status = ?, discarded_at = ?,
                    phase = ?, message = ?, updated_at = ?
                where id = ?
                """,
                ("discarded", "discarded", now, "discarded", "Review discarded.", now, job_id),
            )
            self.conn.commit()
        self.refresh_review_queue_positions()
        return self.review_job(job_id)

    def cancel_review_job(self, job_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            current = self.review_job(job_id)
            if current["status"] == "canceled":
                return current
            if current["status"] not in {"queued", "paused_for_live"}:
                raise ValueError(f"Cannot cancel review job in status {current['status']}.")
            self.conn.execute(
                """
                update review_jobs
                set status = ?, validation_status = ?, canceled_at = ?,
                    phase = ?, message = ?, updated_at = ?
                where id = ?
                """,
                ("canceled", "canceled", now, "canceled", "Review canceled.", now, job_id),
            )
            self.conn.commit()
        self.refresh_review_queue_positions()
        return self.review_job(job_id)

    def error_review_job(self, job_id: str, message: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                update review_jobs
                set status = ?, phase = ?, progress_pct = ?, message = ?,
                    error_message = ?, completed_at = coalesce(completed_at, ?), updated_at = ?
                where id = ?
                """,
                ("error", "error", 100.0, message, message, now, now, job_id),
            )
            self.conn.commit()
        return self.review_job(job_id)

    def mark_review_audio_deleted(self, job_id: str, deleted_at: float | None = None) -> dict[str, Any]:
        now = deleted_at or time.time()
        with self._lock:
            self.conn.execute(
                """
                update review_jobs
                set temporary_audio_retained = 0, audio_deleted_at = ?,
                    audio_delete_error = null, cleanup_status = ?, updated_at = ?
                where id = ?
                """,
                (now, "deleted", now, job_id),
            )
            self.conn.commit()
        return self.review_job(job_id)

    def mark_review_audio_cleanup_error(self, job_id: str, error: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                update review_jobs
                set temporary_audio_retained = 1, audio_delete_error = ?,
                    cleanup_status = ?, updated_at = ?
                where id = ?
                """,
                (error, "error", now, job_id),
            )
            self.conn.commit()
        return self.review_job(job_id)

    def add_transcript_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self.conn.execute(
                """
                insert or ignore into transcript_segments
                (
                  id, session_id, start_s, end_s, text, is_final, created_at,
                  speaker_role, speaker_label, speaker_confidence,
                  speaker_match_score, speaker_match_reason, speaker_low_confidence,
                  diarization_speaker_id
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.id,
                    segment.session_id,
                    segment.start_s,
                    segment.end_s,
                    segment.text,
                    1 if segment.is_final else 0,
                    segment.created_at,
                    segment.speaker_role,
                    segment.speaker_label,
                    segment.speaker_confidence,
                    segment.speaker_match_score,
                    segment.speaker_match_reason,
                    None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
                    segment.diarization_speaker_id,
                ),
            )
            self.conn.commit()

    def upsert_transcript_segment(self, segment: TranscriptSegment, *, replaces_segment_id: str | None = None) -> None:
        with self._lock:
            target_id = replaces_segment_id or segment.id
            updated = self.conn.execute(
                """
                update transcript_segments
                set id = ?, start_s = ?, end_s = ?, text = ?, is_final = ?,
                    created_at = ?, speaker_role = ?, speaker_label = ?,
                    speaker_confidence = ?, speaker_match_score = ?,
                    speaker_match_reason = ?, speaker_low_confidence = ?,
                    diarization_speaker_id = ?
                where session_id = ? and id = ?
                """,
                (
                    segment.id,
                    segment.start_s,
                    segment.end_s,
                    segment.text,
                    1 if segment.is_final else 0,
                    segment.created_at,
                    segment.speaker_role,
                    segment.speaker_label,
                    segment.speaker_confidence,
                    segment.speaker_match_score,
                    segment.speaker_match_reason,
                    None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
                    segment.diarization_speaker_id,
                    segment.session_id,
                    target_id,
                ),
            )
            if updated.rowcount == 0 and target_id != segment.id:
                updated = self.conn.execute(
                    """
                    update transcript_segments
                    set start_s = ?, end_s = ?, text = ?, is_final = ?,
                        created_at = ?, speaker_role = ?, speaker_label = ?,
                        speaker_confidence = ?, speaker_match_score = ?,
                        speaker_match_reason = ?, speaker_low_confidence = ?,
                        diarization_speaker_id = ?
                    where session_id = ? and id = ?
                    """,
                    (
                        segment.start_s,
                        segment.end_s,
                        segment.text,
                        1 if segment.is_final else 0,
                        segment.created_at,
                        segment.speaker_role,
                        segment.speaker_label,
                        segment.speaker_confidence,
                        segment.speaker_match_score,
                        segment.speaker_match_reason,
                        None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
                        segment.diarization_speaker_id,
                        segment.session_id,
                        segment.id,
                    ),
                )
            if updated.rowcount == 0:
                self.conn.execute(
                    """
                    insert into transcript_segments
                    (
                      id, session_id, start_s, end_s, text, is_final, created_at,
                      speaker_role, speaker_label, speaker_confidence,
                      speaker_match_score, speaker_match_reason, speaker_low_confidence,
                      diarization_speaker_id
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        segment.id,
                        segment.session_id,
                        segment.start_s,
                        segment.end_s,
                        segment.text,
                        1 if segment.is_final else 0,
                        segment.created_at,
                        segment.speaker_role,
                        segment.speaker_label,
                        segment.speaker_confidence,
                        segment.speaker_match_score,
                        segment.speaker_match_reason,
                        None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
                        segment.diarization_speaker_id,
                    ),
                )
            self.conn.commit()

    def recent_segments(self, session_id: str, limit: int = 12) -> list[TranscriptSegment]:
        rows = self.conn.execute(
            """
            select * from transcript_segments
            where session_id = ?
            order by created_at desc
            limit ?
            """,
            (session_id, limit),
        ).fetchall()
        return [self._transcript_row_to_segment(row) for row in reversed(rows)]

    def add_note(self, note: NoteCard) -> None:
        with self._lock:
            self.conn.execute(
                """
                insert into note_cards(
                  id, session_id, kind, title, body, source_segment_ids, created_at,
                  evidence_quote, owner, due_date, missing_info
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.session_id,
                    note.kind,
                    note.title,
                    note.body,
                    json.dumps(note.source_segment_ids),
                    note.created_at,
                    note.evidence_quote,
                    note.owner,
                    note.due_date,
                    note.missing_info,
                ),
            )
            self.conn.commit()

    def replace_session_notes(self, session_id: str, notes: list[NoteCard]) -> None:
        with self._lock:
            self.conn.execute("delete from note_cards where session_id = ?", (session_id,))
            for note in notes:
                self.conn.execute(
                    """
                    insert into note_cards(
                      id, session_id, kind, title, body, source_segment_ids, created_at,
                      evidence_quote, owner, due_date, missing_info
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note.id,
                        note.session_id,
                        note.kind,
                        note.title,
                        note.body,
                        json.dumps(note.source_segment_ids),
                        note.created_at,
                        note.evidence_quote,
                        note.owner,
                        note.due_date,
                        note.missing_info,
                    ),
                )
            self.conn.commit()

    def add_library_root(self, path: Path) -> str:
        root_id = new_id("root")
        with self._lock:
            self.conn.execute(
                "insert or ignore into library_roots(id, path, created_at) values (?, ?, ?)",
                (root_id, str(path), time.time()),
            )
            row = self.conn.execute("select id from library_roots where path = ?", (str(path),)).fetchone()
            self.conn.commit()
        return str(row["id"])

    def library_roots(self) -> list[Path]:
        rows = self.conn.execute("select path from library_roots order by created_at").fetchall()
        return [Path(row["path"]) for row in rows]

    def document_chunk_sources(
        self,
        *,
        limit: int = 200,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if query:
            where = "where lower(source_path) like ? or lower(text) like ?"
            needle = f"%{query.lower()}%"
            params.extend([needle, needle])
        rows = self.conn.execute(
            f"""
            select
              source_path,
              count(*) as chunk_count,
              max(updated_at) as updated_at,
              min(chunk_index) as first_chunk_index
            from document_chunks
            {where}
            group by source_path
            order by updated_at desc
            limit ?
            """,
            (*params, max(1, min(limit, 500))),
        ).fetchall()
        return [
            {
                "source_path": row["source_path"],
                "title": Path(row["source_path"]).name,
                "chunk_count": int(row["chunk_count"]),
                "updated_at": row["updated_at"],
                "first_chunk_index": row["first_chunk_index"],
            }
            for row in rows
        ]

    def document_chunks(
        self,
        *,
        source_path: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if source_path:
            where.append("source_path = ?")
            params.append(source_path)
        if query:
            where.append("(lower(source_path) like ? or lower(text) like ?)")
            needle = f"%{query.lower()}%"
            params.extend([needle, needle])
        where_sql = f"where {' and '.join(where)}" if where else ""
        rows = self.conn.execute(
            f"""
            select * from document_chunks
            {where_sql}
            order by source_path, chunk_index
            limit ?
            """,
            (*params, max(1, min(limit, 500))),
        ).fetchall()
        return [self._document_chunk_row_to_dict(row) for row in rows]

    def upsert_document_chunk(self, source_path: Path, chunk_index: int, text: str, metadata: dict[str, Any]) -> str:
        chunk_id = new_id("chunk")
        with self._lock:
            self.conn.execute(
                """
                insert into document_chunks(id, source_path, chunk_index, text, metadata, updated_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(source_path, chunk_index) do update set
                  text = excluded.text,
                  metadata = excluded.metadata,
                  updated_at = excluded.updated_at
                """,
                (chunk_id, str(source_path), chunk_index, text, json.dumps(metadata), time.time()),
            )
            row = self.conn.execute(
                "select id from document_chunks where source_path = ? and chunk_index = ?",
                (str(source_path), chunk_index),
            ).fetchone()
            self.conn.commit()
        return str(row["id"])

    def upsert_embedding(
        self,
        source_type: str,
        source_id: str,
        text: str,
        metadata: dict[str, Any],
        vector: list[float],
    ) -> None:
        embedding_id = f"{source_type}:{source_id}"
        with self._lock:
            self.conn.execute(
                """
                insert into embedding_records(id, source_type, source_id, text, metadata, vector_json, updated_at)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  text = excluded.text,
                  metadata = excluded.metadata,
                  vector_json = excluded.vector_json,
                  updated_at = excluded.updated_at
                """,
                (
                    embedding_id,
                    source_type,
                    source_id,
                    text,
                    json.dumps(metadata),
                    json.dumps(vector),
                    time.time(),
                ),
            )
            self.conn.commit()

    def embedding_records(self) -> Iterable[dict[str, Any]]:
        rows = self.conn.execute("select * from embedding_records").fetchall()
        for row in rows:
            yield {
                "id": row["id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "vector": json.loads(row["vector_json"]),
                "updated_at": row["updated_at"],
            }

    def embedding_marker(self) -> tuple[int, float]:
        row = self.conn.execute(
            "select count(*) as count, coalesce(max(updated_at), 0) as updated_at from embedding_records"
        ).fetchone()
        return int(row["count"]), float(row["updated_at"])

    def upsert_company_refs(self, refs: Iterable[Any]) -> int:
        count = 0
        with self._lock:
            for ref in refs:
                ref_id = str(_item_value(ref, "id") or "").strip()
                canonical_name = str(_item_value(ref, "canonical_name") or "").strip()
                description = str(_item_value(ref, "description") or "").strip()
                if not ref_id or not canonical_name or not description:
                    continue
                updated_at = float(_item_value(ref, "updated_at") or time.time())
                self.conn.execute(
                    """
                    insert into company_refs(
                      id, canonical_name, entity_type, domain, description, website,
                      metadata_json, sources_json, active, updated_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(id) do update set
                      canonical_name = excluded.canonical_name,
                      entity_type = excluded.entity_type,
                      domain = excluded.domain,
                      description = excluded.description,
                      website = excluded.website,
                      metadata_json = excluded.metadata_json,
                      sources_json = excluded.sources_json,
                      active = excluded.active,
                      updated_at = excluded.updated_at
                    """,
                    (
                        ref_id,
                        canonical_name,
                        str(_item_value(ref, "entity_type") or "company").strip() or "company",
                        str(_item_value(ref, "domain") or "").strip(),
                        description,
                        _item_value(ref, "website"),
                        json.dumps(_item_value(ref, "metadata") or {}, sort_keys=True),
                        json.dumps(_item_value(ref, "sources") or []),
                        1 if bool(_item_value(ref, "active", True)) else 0,
                        updated_at,
                    ),
                )
                self.conn.execute("delete from company_ref_aliases where ref_id = ?", (ref_id,))
                for alias in list(_item_value(ref, "aliases") or []):
                    alias_text = str(_item_value(alias, "alias") or "").strip()
                    normalized_alias = str(_item_value(alias, "normalized_alias") or "").strip()
                    if not alias_text or not normalized_alias:
                        continue
                    self.conn.execute(
                        """
                        insert into company_ref_aliases(
                          id, ref_id, alias, normalized_alias, alias_type, weight,
                          requires_context, context_terms_json, negative_terms_json
                        )
                        values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(_item_value(alias, "id") or new_id("calias")),
                            ref_id,
                            alias_text,
                            normalized_alias,
                            str(_item_value(alias, "alias_type") or "name").strip() or "name",
                            float(_item_value(alias, "weight") or 1.0),
                            1 if bool(_item_value(alias, "requires_context", False)) else 0,
                            json.dumps(_item_value(alias, "context_terms") or []),
                            json.dumps(_item_value(alias, "negative_terms") or []),
                        ),
                    )
                count += 1
            self.conn.commit()
        return count

    def company_refs(self, active_only: bool = True) -> list[dict[str, Any]]:
        where = "where active = 1" if active_only else ""
        rows = self.conn.execute(
            f"""
            select *
            from company_refs
            {where}
            order by lower(canonical_name)
            """
        ).fetchall()
        refs = [self._company_ref_row_to_dict(row) for row in rows]
        aliases_by_ref = self._company_aliases_by_ref(active_only=active_only)
        for ref in refs:
            ref["aliases"] = aliases_by_ref.get(ref["id"], [])
        return refs

    def company_ref_aliases(self, active_only: bool = True) -> list[dict[str, Any]]:
        if active_only:
            rows = self.conn.execute(
                """
                select a.*
                from company_ref_aliases a
                join company_refs r on r.id = a.ref_id
                where r.active = 1
                order by lower(a.normalized_alias), lower(a.alias)
                """
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                select *
                from company_ref_aliases
                order by lower(normalized_alias), lower(alias)
                """
            ).fetchall()
        return [self._company_alias_row_to_dict(row) for row in rows]

    def company_ref_status(self) -> dict[str, Any]:
        row = self.conn.execute(
            """
            select
              count(*) as ref_count,
              sum(case when active = 1 then 1 else 0 end) as active_ref_count,
              coalesce(max(updated_at), 0) as latest_updated_at
            from company_refs
            """
        ).fetchone()
        alias_row = self.conn.execute(
            """
            select count(*) as alias_count
            from company_ref_aliases a
            join company_refs r on r.id = a.ref_id
            where r.active = 1
            """
        ).fetchone()
        return {
            "ref_count": int(row["ref_count"] or 0),
            "active_ref_count": int(row["active_ref_count"] or 0),
            "active_alias_count": int(alias_row["alias_count"] or 0),
            "latest_updated_at": float(row["latest_updated_at"] or 0.0),
        }

    def search_company_refs(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        clean_query = _normalize_search_text(query)
        tokens = [
            token for token in clean_query.split()
            if len(token) >= 2 and token not in {"the", "and", "for", "what", "does", "who", "company"}
        ]
        needles = [clean_query, *tokens] if clean_query else []
        if not needles:
            return []
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for ref in self.company_refs(active_only=True):
            canonical = _normalize_search_text(ref["canonical_name"])
            aliases = [_normalize_search_text(alias["alias"]) for alias in ref.get("aliases", [])]
            haystack = " ".join([
                canonical,
                _normalize_search_text(ref["domain"]),
                _normalize_search_text(ref["description"]),
                *aliases,
            ])
            score = 0
            if clean_query == canonical or clean_query in aliases:
                score = 100
            elif any(needle and needle in aliases for needle in needles):
                score = 90
            elif any(needle and needle in canonical for needle in needles):
                score = 80
            elif any(needle and needle in haystack for needle in needles):
                score = 60
            if score:
                scored.append((score, canonical, ref))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[: max(1, min(int(limit), 50))]]

    def upsert_session_memory_summary(
        self,
        *,
        session_id: str,
        title: str,
        summary: str,
        topics: list[str],
        decisions: list[str],
        actions: list[str],
        unresolved_questions: list[str],
        entities: list[str],
        lessons: list[str],
        source_segment_ids: list[str],
        review_standard: str | None = None,
        key_points: list[str] | None = None,
        projects: list[str] | None = None,
        project_workstreams: list[dict[str, Any]] | None = None,
        technical_findings: list[dict[str, Any]] | None = None,
        portfolio_rollup: dict[str, Any] | None = None,
        review_metrics: dict[str, Any] | None = None,
        risks: list[str] | None = None,
        coverage_notes: list[str] | None = None,
        reference_context: list[dict[str, Any]] | None = None,
        context_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into session_memory_summaries(
                  session_id, title, summary, topics_json, decisions_json, actions_json,
                  unresolved_questions_json, entities_json, lessons_json,
                  source_segment_ids_json, review_standard, key_points_json, projects_json,
                  project_workstreams_json, technical_findings_json, portfolio_rollup_json,
                  review_metrics_json, risks_json,
                  coverage_notes_json, reference_context_json, context_diagnostics_json,
                  created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(session_id) do update set
                  title = excluded.title,
                  summary = excluded.summary,
                  topics_json = excluded.topics_json,
                  decisions_json = excluded.decisions_json,
                  actions_json = excluded.actions_json,
                  unresolved_questions_json = excluded.unresolved_questions_json,
                  entities_json = excluded.entities_json,
                  lessons_json = excluded.lessons_json,
                  source_segment_ids_json = excluded.source_segment_ids_json,
                  review_standard = excluded.review_standard,
                  key_points_json = excluded.key_points_json,
                  projects_json = excluded.projects_json,
                  project_workstreams_json = excluded.project_workstreams_json,
                  technical_findings_json = excluded.technical_findings_json,
                  portfolio_rollup_json = excluded.portfolio_rollup_json,
                  review_metrics_json = excluded.review_metrics_json,
                  risks_json = excluded.risks_json,
                  coverage_notes_json = excluded.coverage_notes_json,
                  reference_context_json = excluded.reference_context_json,
                  context_diagnostics_json = excluded.context_diagnostics_json,
                  updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    title,
                    summary,
                    json.dumps(topics),
                    json.dumps(decisions),
                    json.dumps(actions),
                    json.dumps(unresolved_questions),
                    json.dumps(entities),
                    json.dumps(lessons),
                    json.dumps(source_segment_ids),
                    review_standard or "",
                    json.dumps(key_points or []),
                    json.dumps(projects or []),
                    json.dumps(project_workstreams or []),
                    json.dumps(technical_findings or []),
                    json.dumps(portfolio_rollup or {}),
                    json.dumps(review_metrics or {}),
                    json.dumps(risks or []),
                    json.dumps(coverage_notes or []),
                    json.dumps(reference_context or []),
                    json.dumps(context_diagnostics or {}),
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def session_memory_summaries(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select *
            from session_memory_summaries
            order by updated_at desc
            """
        ).fetchall()
        return [self._session_memory_summary_row_to_dict(row) for row in rows]

    def sqlite_runtime_settings(self) -> dict[str, Any]:
        journal_mode = self.conn.execute("pragma journal_mode").fetchone()[0]
        busy_timeout = self.conn.execute("pragma busy_timeout").fetchone()[0]
        return {"journal_mode": journal_mode, "busy_timeout": int(busy_timeout)}

    def _session_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        transcript_count = int(row["transcript_count"])
        note_count = int(row["note_count"])
        summary_exists = bool(row["summary_exists"])
        raw_save = row["save_transcript"]
        save_transcript = None if raw_save is None else bool(raw_save)
        if save_transcript is True:
            retention = "saved"
        elif save_transcript is False:
            retention = "temporary"
        elif transcript_count or note_count or summary_exists:
            retention = "saved"
        else:
            retention = "empty"
        return {
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "save_transcript": save_transcript,
            "retention": retention,
            "transcript_count": transcript_count,
            "note_count": note_count,
            "summary_exists": summary_exists,
            "raw_audio_retained": False,
        }

    def _review_job_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = None
        result_json = row["result_json"]
        if result_json:
            try:
                result = json.loads(result_json)
            except json.JSONDecodeError:
                result = None
        temporary_audio_retained = bool(row["temporary_audio_retained"])
        payload = {
            "id": row["id"],
            "job_id": row["id"],
            "source": row["source"],
            "title": row["title"],
            "status": row["status"],
            "validation_status": row["validation_status"],
            "phase": row["phase"],
            "progress_pct": float(row["progress_pct"] or 0.0),
            "progress_percent": int(round(float(row["progress_pct"] or 0.0))),
            "message": row["message"] or "",
            "queue_position": row["queue_position"],
            "session_id": row["session_id"],
            "live_id": row["live_id"],
            "source_filename": row["source_filename"],
            "filename": row["source_filename"] or "",
            "temporary_audio_path": row["temporary_audio_path"],
            "temporary_audio_retained": temporary_audio_retained,
            "raw_audio_retained": temporary_audio_retained,
            "audio_deleted_at": row["audio_deleted_at"],
            "audio_delete_error": row["audio_delete_error"],
            "cleanup_status": row["cleanup_status"] or ("retained" if temporary_audio_retained else "deleted"),
            "result": result,
            "error": row["error_message"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "approved_at": row["approved_at"],
            "discarded_at": row["discarded_at"],
            "canceled_at": row["canceled_at"],
            "active": row["status"] not in {"approved", "discarded", "canceled", "error", "completed_awaiting_validation"},
        }
        if result:
            payload["clean_segments"] = result.get("clean_transcript") or []
            payload["meeting_cards"] = result.get("meeting_cards") or []
            payload["summary"] = result.get("summary")
            diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
            payload["duration_seconds"] = diagnostics.get("duration_seconds")
            payload["raw_segment_count"] = diagnostics.get("raw_segment_count", len(payload["clean_segments"]))
            payload["corrected_segment_count"] = diagnostics.get("corrected_segment_count", len(payload["clean_segments"]))
            payload["asr_backend"] = diagnostics.get("asr_backend")
            payload["asr_model"] = diagnostics.get("asr_model")
            payload["diagnostics"] = diagnostics
        else:
            payload["clean_segments"] = []
            payload["meeting_cards"] = []
            payload["summary"] = None
            payload["duration_seconds"] = None
            payload["raw_segment_count"] = 0
            payload["corrected_segment_count"] = 0
            payload["asr_backend"] = None
            payload["asr_model"] = None
            payload["diagnostics"] = {}
        payload["steps"] = _review_steps_for_payload(
            str(payload["status"]),
            str(payload.get("phase") or ""),
            str(payload.get("message") or ""),
            int(payload["progress_percent"]),
        )
        return payload

    def _transcript_row_to_segment(self, row: sqlite3.Row) -> TranscriptSegment:
        return TranscriptSegment(
            id=row["id"],
            session_id=row["session_id"],
            start_s=row["start_s"],
            end_s=row["end_s"],
            text=row["text"],
            is_final=bool(row["is_final"]),
            created_at=row["created_at"],
            speaker_role=row["speaker_role"],
            speaker_label=row["speaker_label"],
            speaker_confidence=row["speaker_confidence"],
            speaker_match_score=row["speaker_match_score"],
            speaker_match_reason=row["speaker_match_reason"],
            speaker_low_confidence=(
                bool(row["speaker_low_confidence"])
                if row["speaker_low_confidence"] is not None
                else None
            ),
            diarization_speaker_id=row["diarization_speaker_id"],
        )

    def _note_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "kind": row["kind"],
            "title": row["title"],
            "body": row["body"],
            "source_segment_ids": json.loads(row["source_segment_ids"]),
            "created_at": row["created_at"],
            "evidence_quote": row["evidence_quote"],
            "owner": row["owner"],
            "due_date": row["due_date"],
            "missing_info": row["missing_info"],
            "raw_audio_retained": False,
        }

    def _document_chunk_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source_path": row["source_path"],
            "title": Path(row["source_path"]).name,
            "chunk_index": int(row["chunk_index"]),
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
            "updated_at": row["updated_at"],
            "raw_audio_retained": False,
        }

    def _company_ref_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "canonical_name": row["canonical_name"],
            "entity_type": row["entity_type"],
            "domain": row["domain"],
            "description": row["description"],
            "website": row["website"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "sources": json.loads(row["sources_json"] or "[]"),
            "active": bool(row["active"]),
            "updated_at": float(row["updated_at"]),
            "raw_audio_retained": False,
        }

    def _company_alias_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "ref_id": row["ref_id"],
            "alias": row["alias"],
            "normalized_alias": row["normalized_alias"],
            "alias_type": row["alias_type"],
            "weight": float(row["weight"]),
            "requires_context": bool(row["requires_context"]),
            "context_terms": json.loads(row["context_terms_json"] or "[]"),
            "negative_terms": json.loads(row["negative_terms_json"] or "[]"),
        }

    def _company_aliases_by_ref(self, *, active_only: bool) -> dict[str, list[dict[str, Any]]]:
        aliases: dict[str, list[dict[str, Any]]] = {}
        for alias in self.company_ref_aliases(active_only=active_only):
            aliases.setdefault(alias["ref_id"], []).append(alias)
        return aliases

    def _session_memory_summary_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "review_standard": row["review_standard"] or "",
            "title": row["title"],
            "summary": row["summary"],
            "key_points": json.loads(row["key_points_json"] or "[]"),
            "topics": json.loads(row["topics_json"]),
            "projects": json.loads(row["projects_json"] or "[]"),
            "project_workstreams": json.loads(row["project_workstreams_json"] or "[]"),
            "technical_findings": json.loads(row["technical_findings_json"] or "[]"),
            "portfolio_rollup": json.loads(row["portfolio_rollup_json"] or "{}"),
            "review_metrics": json.loads(row["review_metrics_json"] or "{}"),
            "decisions": json.loads(row["decisions_json"]),
            "actions": json.loads(row["actions_json"]),
            "unresolved_questions": json.loads(row["unresolved_questions_json"]),
            "risks": json.loads(row["risks_json"] or "[]"),
            "entities": json.loads(row["entities_json"]),
            "lessons": json.loads(row["lessons_json"]),
            "coverage_notes": json.loads(row["coverage_notes_json"] or "[]"),
            "reference_context": json.loads(row["reference_context_json"] or "[]"),
            "context_diagnostics": json.loads(row["context_diagnostics_json"] or "{}"),
            "source_segment_ids": json.loads(row["source_segment_ids_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _voice_phrase_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            **dict(row),
            "preferred_terms": json.loads(row["preferred_terms"]),
            "profile_artifact": json.loads(row["profile_artifact_json"])
            if row["profile_artifact_json"]
            else None,
            "accepted": bool(row["accepted"]),
        }

    def _speaker_profile_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            **dict(row),
            "active": bool(row["active"]),
            "threshold": float(row["threshold"]) if row["threshold"] is not None else None,
        }

    def _speaker_embedding_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "profile_id": row["profile_id"],
            "vector": json.loads(row["vector_json"]),
            "vector_dim": int(row["vector_dim"]),
            "source": row["source"],
            "quality_score": float(row["quality_score"]),
            "duration_seconds": float(row["duration_seconds"]),
            "sample_rate": int(row["sample_rate"]),
            "embedding_model": row["embedding_model"],
            "embedding_model_version": row["embedding_model_version"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
        }

    def _speaker_sample_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "enrollment_id": row["enrollment_id"],
            "profile_id": row["profile_id"],
            "embedding_id": row["embedding_id"],
            "duration_seconds": float(row["duration_seconds"]),
            "usable_speech_seconds": float(row["usable_speech_seconds"]),
            "quality_score": float(row["quality_score"]),
            "issues": json.loads(row["issues_json"]),
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
            "raw_audio_retained": False,
        }

    def _diarization_segment_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "segment_id": row["transcript_segment_id"],
            "start_ms": int(row["start_ms"]),
            "end_ms": int(row["end_ms"]),
            "diarization_speaker_id": row["diarization_speaker_id"],
            "display_speaker_label": row["display_speaker_label"],
            "matched_profile_id": row["matched_profile_id"],
            "match_confidence": float(row["match_confidence"]) if row["match_confidence"] is not None else None,
            "match_score": float(row["match_score"]) if row["match_score"] is not None else None,
            "transcript_text": row["transcript_text"],
            "is_overlap": bool(row["is_overlap"]),
            "finalized": bool(row["finalized"]),
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
        }
