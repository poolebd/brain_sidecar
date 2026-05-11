export type TranscriptLine = {
  id: string;
  text: string;
  start_s: number;
  end_s: number;
  source_segment_ids?: string[];
  asr_model?: string;
  created_at?: number;
};

export type SavedTranscriptSegment = TranscriptLine & {
  is_final?: boolean;
  speaker_role?: string;
  speaker_label?: string;
  speaker_confidence?: number;
  speaker_match_score?: number;
  speaker_match_reason?: string;
  speaker_low_confidence?: boolean;
  diarization_speaker_id?: string;
};

export type SummaryProjectWorkstream = {
  project?: string;
  client_site?: string;
  status?: string;
  decisions?: string[];
  actions?: string[];
  risks?: string[];
  open_questions?: string[];
  owners?: string[];
  next_checkpoint?: string;
  source_segment_ids?: string[];
};

export type SummaryTechnicalFinding = {
  topic?: string;
  question?: string;
  assumptions?: string[];
  methods?: string[];
  findings?: string[];
  recommendations?: string[];
  risks?: string[];
  data_gaps?: string[];
  reference_context?: string[];
  confidence?: string;
  source_segment_ids?: string[];
};

export type SummaryReferenceContext = {
  kind?: string;
  title?: string;
  body?: string;
  citation?: string;
  query?: string;
  source_segment_ids?: string[];
};

export type SummaryPortfolioRollup = {
  bp_next_actions?: string[];
  open_loops?: string[];
  cross_project_dependencies?: string[];
  risk_posture?: string;
  source_segment_ids?: string[];
};

export type SummaryReviewMetrics = {
  workstream_count?: number;
  action_count?: number;
  risk_count?: number;
  technical_finding_count?: number;
  source_count?: number;
  source_coverage?: number;
  time_span_coverage?: number;
  context_kinds?: string[];
};

export type SessionMemorySummary = {
  session_id: string;
  review_standard?: string;
  title: string;
  summary: string;
  key_points?: string[];
  topics: string[];
  projects?: string[];
  project_workstreams?: SummaryProjectWorkstream[];
  technical_findings?: SummaryTechnicalFinding[];
  portfolio_rollup?: SummaryPortfolioRollup;
  review_metrics?: SummaryReviewMetrics;
  decisions: string[];
  actions: string[];
  unresolved_questions: string[];
  risks?: string[];
  entities: string[];
  lessons: string[];
  coverage_notes?: string[];
  reference_context?: SummaryReferenceContext[];
  context_diagnostics?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
  source_segment_ids: string[];
  created_at: number;
  updated_at: number;
};

export type ReviewStep = {
  key: string;
  label: string;
  status: "pending" | "running" | "completed" | "error" | "canceled" | string;
  message: string;
  progress: number;
  detail?: string;
  current?: number | null;
  total?: number | null;
  unit?: string;
  elapsed_seconds?: number;
};

export type ReviewJobResponse = {
  id?: string;
  job_id: string;
  source?: "live" | "upload" | string;
  title: string;
  filename: string;
  status: "queued" | "paused_for_live" | "running_asr" | "running_cards" | "running_summary" | "completed_awaiting_validation" | "approved" | "discarded" | "canceled" | "error" | "conditioning" | "transcribing" | "reviewing" | "evaluating" | "completed" | string;
  validation_status?: "pending" | "approved" | "discarded" | "canceled" | string;
  phase?: string | null;
  progress_pct?: number;
  message?: string;
  queue_position?: number | null;
  error?: string | null;
  save_result: boolean;
  session_id?: string | null;
  live_id?: string | null;
  duration_seconds?: number | null;
  raw_segment_count: number;
  corrected_segment_count: number;
  progress_percent: number;
  asr_backend?: string | null;
  asr_model?: string | null;
  steps: ReviewStep[];
  clean_segments: SavedTranscriptSegment[];
  meeting_cards: Record<string, unknown>[];
  summary: SessionMemorySummary | null;
  diagnostics?: Record<string, unknown>;
  raw_audio_retained: boolean;
  temporary_audio_retained?: boolean;
  audio_deleted_at?: number | null;
  audio_delete_error?: string | null;
  cleanup_status?: string | null;
  created_at: number;
  updated_at: number;
  elapsed_seconds?: number;
  active?: boolean;
};
