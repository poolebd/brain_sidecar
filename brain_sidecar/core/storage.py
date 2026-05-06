from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from brain_sidecar.core.models import NoteCard, SessionRecord, TranscriptSegment, new_id


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
                create table if not exists work_memory_sources (
                  id text primary key,
                  path text not null unique,
                  source_group text not null,
                  sensitivity text not null,
                  status text not null,
                  title text not null,
                  content_hash text not null,
                  metadata text not null,
                  disabled integer not null,
                  created_at real not null,
                  updated_at real not null
                );
                create index if not exists idx_work_memory_sources_group on work_memory_sources(source_group, status);
                create table if not exists work_memory_projects (
                  id text primary key,
                  project_key text not null unique,
                  title text not null,
                  organization text not null,
                  date_range text not null,
                  role text not null,
                  domain text not null,
                  summary text not null,
                  lessons_json text not null,
                  triggers_json text not null,
                  source_group text not null,
                  confidence real not null,
                  updated_at real not null
                );
                create index if not exists idx_work_memory_projects_group on work_memory_projects(source_group);
                create table if not exists work_memory_evidence (
                  id text primary key,
                  project_id text not null,
                  source_id text,
                  source_path text not null,
                  snippet text not null,
                  artifact_type text not null,
                  weight real not null,
                  created_at real not null
                );
                create index if not exists idx_work_memory_evidence_project on work_memory_evidence(project_id, weight);
                create table if not exists work_memory_recall_events (
                  id text primary key,
                  session_id text,
                  project_id text not null,
                  query text not null,
                  score real not null,
                  reason text not null,
                  created_at real not null
                );
                create index if not exists idx_work_memory_recall_session on work_memory_recall_events(session_id, created_at);
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
                  created_at real not null,
                  updated_at real not null
                );
                """
            )
            self._ensure_column("voice_corrections", "quarantined", "integer not null default 0")
            self._ensure_column("voice_corrections", "quarantine_reason", "text")
            self._apply_migrations()
            self.conn.commit()

    def _apply_migrations(self) -> None:
        self._ensure_column("transcript_segments", "speaker_role", "text")
        self._ensure_column("transcript_segments", "speaker_label", "text")
        self._ensure_column("transcript_segments", "speaker_confidence", "real")
        self._ensure_column("transcript_segments", "speaker_match_reason", "text")
        self._ensure_column("transcript_segments", "speaker_low_confidence", "integer")
        self._ensure_column("note_cards", "evidence_quote", "text not null default ''")
        self._ensure_column("note_cards", "owner", "text")
        self._ensure_column("note_cards", "due_date", "text")
        self._ensure_column("note_cards", "missing_info", "text")
        self._record_migration("20260504_transcript_speaker_fields")
        self._record_migration("20260504_session_memory_summaries")
        self._record_migration("20260505_note_evidence_fields")

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

    def set_session_status(self, session_id: str, status: str, ended_at: float | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "update sessions set status = ?, ended_at = coalesce(?, ended_at) where id = ?",
                (status, ended_at, session_id),
            )
            self.conn.commit()

    def add_transcript_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self.conn.execute(
                """
                insert or ignore into transcript_segments
                (
                  id, session_id, start_s, end_s, text, is_final, created_at,
                  speaker_role, speaker_label, speaker_confidence,
                  speaker_match_reason, speaker_low_confidence
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    segment.speaker_match_reason,
                    None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
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
                    speaker_confidence = ?, speaker_match_reason = ?,
                    speaker_low_confidence = ?
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
                    segment.speaker_match_reason,
                    None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
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
                        speaker_confidence = ?, speaker_match_reason = ?,
                        speaker_low_confidence = ?
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
                        segment.speaker_match_reason,
                        None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
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
                      speaker_match_reason, speaker_low_confidence
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        segment.speaker_match_reason,
                        None if segment.speaker_low_confidence is None else 1 if segment.speaker_low_confidence else 0,
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
        return [
            TranscriptSegment(
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
                speaker_match_reason=row["speaker_match_reason"],
                speaker_low_confidence=(
                    bool(row["speaker_low_confidence"])
                    if row["speaker_low_confidence"] is not None
                    else None
                ),
            )
            for row in reversed(rows)
        ]

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
    ) -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into session_memory_summaries(
                  session_id, title, summary, topics_json, decisions_json, actions_json,
                  unresolved_questions_json, entities_json, lessons_json,
                  source_segment_ids_json, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def clear_work_memory(self) -> None:
        with self._lock:
            self.conn.execute("delete from work_memory_recall_events")
            self.conn.execute("delete from work_memory_evidence")
            self.conn.execute("delete from work_memory_projects")
            self.conn.execute("delete from work_memory_sources")
            self.conn.execute("delete from embedding_records where source_type = ?", ("work_memory_project",))
            self.conn.commit()

    def upsert_work_memory_source(
        self,
        *,
        path: str,
        source_group: str,
        sensitivity: str,
        status: str,
        title: str,
        content_hash: str,
        metadata: dict[str, Any],
        disabled: bool,
    ) -> str:
        source_id = new_id("wsrc")
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into work_memory_sources(
                  id, path, source_group, sensitivity, status, title, content_hash,
                  metadata, disabled, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(path) do update set
                  source_group = excluded.source_group,
                  sensitivity = excluded.sensitivity,
                  status = excluded.status,
                  title = excluded.title,
                  content_hash = excluded.content_hash,
                  metadata = excluded.metadata,
                  disabled = excluded.disabled,
                  updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    path,
                    source_group,
                    sensitivity,
                    status,
                    title,
                    content_hash,
                    json.dumps(metadata),
                    1 if disabled else 0,
                    now,
                    now,
                ),
            )
            row = self.conn.execute("select id from work_memory_sources where path = ?", (path,)).fetchone()
            self.conn.commit()
        return str(row["id"])

    def upsert_work_memory_project(
        self,
        *,
        key: str,
        title: str,
        organization: str,
        date_range: str,
        role: str,
        domain: str,
        summary: str,
        lessons: list[str],
        triggers: list[str],
        source_group: str,
        confidence: float,
    ) -> str:
        project_id = new_id("wproj")
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into work_memory_projects(
                  id, project_key, title, organization, date_range, role, domain,
                  summary, lessons_json, triggers_json, source_group, confidence, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(project_key) do update set
                  title = excluded.title,
                  organization = excluded.organization,
                  date_range = excluded.date_range,
                  role = excluded.role,
                  domain = excluded.domain,
                  summary = excluded.summary,
                  lessons_json = excluded.lessons_json,
                  triggers_json = excluded.triggers_json,
                  source_group = excluded.source_group,
                  confidence = excluded.confidence,
                  updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    key,
                    title,
                    organization,
                    date_range,
                    role,
                    domain,
                    summary,
                    json.dumps(lessons),
                    json.dumps(triggers),
                    source_group,
                    confidence,
                    now,
                ),
            )
            row = self.conn.execute("select id from work_memory_projects where project_key = ?", (key,)).fetchone()
            self.conn.commit()
        return str(row["id"])

    def add_work_memory_evidence(
        self,
        *,
        project_id: str,
        source_id: str | None,
        source_path: str,
        snippet: str,
        artifact_type: str,
        weight: float,
    ) -> str:
        evidence_id = new_id("wevd")
        with self._lock:
            self.conn.execute(
                """
                insert into work_memory_evidence(
                  id, project_id, source_id, source_path, snippet, artifact_type, weight, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (evidence_id, project_id, source_id, source_path, snippet, artifact_type, weight, time.time()),
            )
            self.conn.commit()
        return evidence_id

    def work_memory_projects(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from work_memory_projects
            order by date_range, title
            """
        ).fetchall()
        return [self._work_memory_project_row_to_dict(row) for row in rows]

    def work_memory_sources(self, limit: int = 120) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from work_memory_sources
            order by updated_at desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        return [self._work_memory_source_row_to_dict(row) for row in rows]

    def work_memory_evidence(self, project_id: str, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            select * from work_memory_evidence
            where project_id = ?
            order by weight desc, created_at desc
            limit ?
            """,
            (project_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def add_work_memory_recall_event(
        self,
        *,
        session_id: str | None,
        project_id: str,
        query: str,
        score: float,
        reason: str,
    ) -> str:
        event_id = new_id("wrec")
        with self._lock:
            self.conn.execute(
                """
                insert into work_memory_recall_events(id, session_id, project_id, query, score, reason, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, session_id, project_id, query, score, reason, time.time()),
            )
            self.conn.commit()
        return event_id

    def work_memory_summary(self) -> dict[str, Any]:
        project_count = self.conn.execute("select count(*) as count from work_memory_projects").fetchone()["count"]
        evidence_count = self.conn.execute("select count(*) as count from work_memory_evidence").fetchone()["count"]
        source_count = self.conn.execute("select count(*) as count from work_memory_sources").fetchone()["count"]
        disabled_count = self.conn.execute(
            "select count(*) as count from work_memory_sources where disabled = 1"
        ).fetchone()["count"]
        latest_row = self.conn.execute("select max(updated_at) as latest from work_memory_sources").fetchone()
        rows = self.conn.execute(
            """
            select source_group, count(*) as count
            from work_memory_sources
            group by source_group
            order by source_group
            """
        ).fetchall()
        return {
            "projects": int(project_count),
            "evidence": int(evidence_count),
            "sources": int(source_count),
            "disabled_sources": int(disabled_count),
            "latest_index_at": latest_row["latest"],
            "source_groups": {row["source_group"]: int(row["count"]) for row in rows},
        }

    def _work_memory_project_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project_key": row["project_key"],
            "title": row["title"],
            "organization": row["organization"],
            "date_range": row["date_range"],
            "role": row["role"],
            "domain": row["domain"],
            "summary": row["summary"],
            "lessons": json.loads(row["lessons_json"]),
            "triggers": json.loads(row["triggers_json"]),
            "source_group": row["source_group"],
            "confidence": float(row["confidence"]),
            "updated_at": row["updated_at"],
        }

    def _work_memory_source_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "path": row["path"],
            "source_group": row["source_group"],
            "sensitivity": row["sensitivity"],
            "status": row["status"],
            "title": row["title"],
            "content_hash": row["content_hash"],
            "metadata": json.loads(row["metadata"]),
            "disabled": bool(row["disabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _session_memory_summary_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "title": row["title"],
            "summary": row["summary"],
            "topics": json.loads(row["topics_json"]),
            "decisions": json.loads(row["decisions_json"]),
            "actions": json.loads(row["actions_json"]),
            "unresolved_questions": json.loads(row["unresolved_questions_json"]),
            "entities": json.loads(row["entities_json"]),
            "lessons": json.loads(row["lessons_json"]),
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
