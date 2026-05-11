import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode, RefObject } from "react";
import {
  buildLiveFieldRows,
  buildConsultingBriefMarkdown,
  buildSidecarDisplayCards,
  buildSidecarQualityMetrics,
  displaySourceLabel,
  getTranscriptDisplayLabel,
  groupMeetingOutputCards,
  isCompanyReferenceCard,
  isCurrentMeetingCard,
  isManualCard,
  normalizeDisplayText,
  qualityLabelForCard,
  type LiveFieldRow,
  type MeetingOutputBucket,
  type SidecarDisplayCard,
  type SidecarPriority,
  type TranscriptEvent,
  type TranscriptSource,
} from "./presentation";
import { LiveMeetingWorkbench, liveWorkbenchModeLabel, SessionRuntimeStrip, SessionSelectorBar, type LiveWorkbenchMode, type SessionRuntimeItem } from "./live/LiveShell";
import { compareReviewJobs, isReviewDefaultSelectable, isReviewTerminal, MeetingSummaryBrief, ReviewPage, reviewJobId } from "./review/ReviewPage";
import { buildMeetingCaptureMarkdown, buildMeetingCaptureViewModel } from "./review/meetingCapture";
import { FilePickAction } from "./shared/FilePickAction";
import type { ReviewJobResponse, SavedTranscriptSegment, SessionMemorySummary, TranscriptLine } from "./types";

type ThemeName = "neutral" | "midnight" | "canopy" | "ember";
type AudioSource = "server_device" | "fixture";
type AppPage = "live" | "review" | "sessions" | "tools" | "models" | "references";
type ToolView = "voice" | "speaker" | "test" | "debug" | "appearance";
type ReferenceView = "overview" | "roots" | "documents";

type MemorySelection =
  | { view: "roots"; id: string; root: string }
  | { view: "documents"; id: string; source: LibraryChunkSource };

const API_BASE = import.meta.env.VITE_API_BASE ?? defaultApiBase();
const ENABLE_FIXTURE_AUDIO = import.meta.env.VITE_ENABLE_FIXTURE_AUDIO === "1";
const DEFAULT_FIXTURE_AUDIO = import.meta.env.VITE_DEFAULT_FIXTURE_AUDIO ?? "";
const DEFAULT_THEME: ThemeName = "midnight";
const APP_PAGE_STORAGE_KEY = "brain-sidecar-active-page-v1";
const THEME_STORAGE_KEY = "brain-sidecar-theme-v2";
const DEBUG_METADATA_STORAGE_KEY = "brain-sidecar-debug-metadata-v1";
const MIC_TUNING_STORAGE_KEY = "brain-sidecar-mic-tuning-v1";
const MEETING_FOCUS_STORAGE_KEY = "brain-sidecar-meeting-focus-v1";
const REVIEW_LAST_JOB_STORAGE_KEY = "brain-sidecar-last-review-job-v1";
const DEFAULT_CONTEXT_CARD_LIMIT = 3;
const DEFAULT_MEETING_GOAL = "Offload meeting obligations, questions, risks, follow-ups, useful context, and post-call synthesis for BP.";
const TEST_AUDIO_ACCEPT = "audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.mp4,.webm";
const DEFAULT_TEST_MAX_SECONDS = "60";
const TEST_PLAYBACK_AUTOSTOP_GRACE_MS = 3500;
const TEST_PLAYBACK_START_TIMEOUT_MS = 12_000;

const MEETING_REMINDER_OPTIONS = [
  { key: "owners", label: "Owners", reminder: "Track owners and ownership ambiguity." },
  { key: "questions", label: "Open questions", reminder: "Track open questions and clarifications." },
  { key: "risks", label: "Risks/dependencies", reminder: "Track risks, dependencies, and blocked paths." },
  { key: "contributions", label: "Say-this contributions", reminder: "Suggest concise say-this contributions only when grounded." },
  { key: "brief", label: "Post-call brief", reminder: "Prepare a post-call brief with evidence." },
] as const;

const MEETING_OUTPUT_SECTIONS: { bucket: MeetingOutputBucket; title: string; empty: string }[] = [
  { bucket: "actions", title: "Actions", empty: "No actions yet." },
  { bucket: "decisions", title: "Decisions", empty: "No decisions yet." },
  { bucket: "questions", title: "Questions / Clarifications", empty: "No open questions yet." },
  { bucket: "risks", title: "Risks", empty: "No risks yet." },
  { bucket: "notes", title: "Other Notes", empty: "No other meeting notes yet." },
];

const SPEAKER_TRAINING_PROMPTS = [
  "This is BP speaking at my normal desk setup. I want Brain Sidecar to recognize my voice during work calls.",
  "Today I need to track decisions, action items, open questions, and anything I promise to follow up on.",
  "If someone else is speaking, the transcript should not label them as me. My voice should be labeled BP only when confidence is high.",
  "I am speaking a little quieter now, like I might during a call while thinking through a project issue.",
  "Now I am speaking at a slightly stronger volume, still naturally, from the same microphone position I use during the workday.",
];

const THEMES: { value: ThemeName; label: string }[] = [
  { value: "neutral", label: "Clinical Blue" },
  { value: "midnight", label: "Night Lab" },
  { value: "canopy", label: "Canopy" },
  { value: "ember", label: "Ember" },
];

type DeviceInfo = {
  id: string;
  label: string;
  driver: string;
  ffmpeg_input: string;
  hardware_id?: string;
  healthy?: boolean;
  in_use?: boolean;
  score?: number;
  selection_reason?: string;
};

type DeviceListResponse = {
  devices: DeviceInfo[];
  selected_device?: DeviceInfo | null;
  server_mic_available?: boolean;
  selection_reason?: string;
};

type SpeechSensitivity = "quiet" | "normal" | "noisy";
type MeetingMode = "quiet" | "balanced" | "assertive";
type WebContextMode = "default" | "on" | "off";
type MeetingReminderKey = typeof MEETING_REMINDER_OPTIONS[number]["key"];

type MicTuning = {
  auto_level: boolean;
  input_gain_db: number;
  speech_sensitivity: SpeechSensitivity;
};

type MeetingFocusState = {
  goal: string;
  mode: MeetingMode;
  webContextMode: WebContextMode;
  reminders: Record<MeetingReminderKey, boolean>;
};

type MeetingContractPayload = {
  goal: string;
  mode: MeetingMode;
  reminders: string[];
};

type MeetingDiagnostics = {
  generated_candidate_count: number;
  accepted_count: number;
  suppressed_count: number;
  top_suppression_reasons: { reason: string; count: number }[];
  evidence_quote_coverage: number;
  source_id_coverage: number;
};

type EnergyFrame = {
  active: boolean;
  score: number;
  confidence: "low" | "medium" | "high";
  categories: string[];
  keywords: string[];
  evidenceSegmentIds: string[];
  evidenceQuote: string;
  summaryLabel: string;
};

type EventEnvelope = {
  id: string;
  type: string;
  session_id: string | null;
  at: number;
  payload: Record<string, unknown>;
};

type NoteSource = {
  title: string;
  url?: string;
  path?: string;
};

type NoteCard = {
  id: string;
  kind: string;
  title: string;
  body: string;
  source_segment_ids: string[];
  created_at?: number;
  ephemeral?: boolean;
  source_type?: string;
  sources?: NoteSource[];
  evidence_quote?: string;
  owner?: string;
  due_date?: string;
  missing_info?: string;
  confidence?: number;
  priority?: string;
  why_now?: string;
  suggested_say?: string;
  suggested_ask?: string;
  citations?: string[];
};

type SessionSummary = {
  id: string;
  title: string;
  status: string;
  started_at: number;
  ended_at?: number | null;
  save_transcript?: boolean | null;
  retention?: "saved" | "temporary" | "empty";
  transcript_count: number;
  note_count: number;
  summary_exists: boolean;
  raw_audio_retained?: boolean;
};

type SessionDetail = SessionSummary & {
  transcript_segments: SavedTranscriptSegment[];
  note_cards: NoteCard[];
  summary: SessionMemorySummary | null;
  transcript_redacted?: boolean;
};

type RecallHit = {
  source_type: string;
  source_id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
  source_segment_ids?: string[];
  pinned?: boolean;
  explicitly_requested?: boolean;
  priority?: string;
};

type LibraryChunkSource = {
  source_path: string;
  title: string;
  chunk_count: number;
  updated_at: number;
  first_chunk_index: number;
};

type LibraryChunk = {
  id: string;
  source_path: string;
  title: string;
  chunk_index: number;
  text: string;
  metadata: Record<string, unknown>;
  updated_at: number;
};

type TestModePrepared = {
  run_id: string;
  source_path: string;
  fixture_wav: string;
  duration_seconds: number;
  artifact_dir: string;
  report_path: string;
  expected_terms: string[];
};

type TestModeReport = {
  run_id: string;
  session_id: string;
  source_path: string;
  fixture_wav: string;
  duration_seconds: number;
  started_at: string;
  completed_at: string;
  status: string;
  transcript_segments: number;
  note_cards: number;
  recall_hits: number;
  queue_depth: number;
  dropped_windows: number;
  reference_context_cards: number;
  error_count: number;
  expected_terms: string[];
  matched_expected_terms: string[];
  issues: string[];
  report_path?: string;
};

type TestModeSnapshot = {
  transcript: TranscriptLine[];
  notes: NoteCard[];
  recall: RecallHit[];
  sidecarCards: SidecarDisplayCard[];
  pipeline: PipelineState;
  errors: string[];
  status: string;
  testExpectedTerms: string;
};

type GpuHealth = {
  nvidia_available?: boolean;
  name?: string;
  memory_total_mb?: number;
  memory_used_mb?: number;
  memory_free_mb?: number;
  gpu_pressure?: string;
  gpu_processes?: { pid: number; process_name: string; used_memory_mb: number }[];
  asr_cuda_available?: boolean;
  asr_cuda_error?: string;
  asr_backend?: string;
  asr_model?: string | null;
  asr_dtype?: string;
  asr_device?: string;
  asr_backend_error?: string | null;
  asr_primary_model?: string;
  asr_fallback_model?: string;
  asr_min_free_vram_mb?: number;
  asr_unload_ollama_on_start?: boolean;
  asr_vad_min_silence_ms?: number;
  audio_chunk_ms?: number;
  transcription_window_seconds?: number;
  transcription_overlap_seconds?: number;
  ollama_gpu_models?: string[];
  ollama_chat_model?: string;
  ollama_embed_model?: string;
  ollama_chat_host?: string;
  ollama_embed_host?: string;
  ollama_chat_reachable?: boolean;
  ollama_embed_reachable?: boolean;
  ollama_keep_alive?: string;
  ollama_chat_keep_alive?: string;
  ollama_embed_keep_alive?: string;
  ollama_chat_timeout_seconds?: number;
  ollama_embed_timeout_seconds?: number;
  asr_beam_size?: number;
  transcription_queue_size?: number;
  partial_transcripts_enabled?: boolean;
  partial_window_seconds?: number;
  partial_min_interval_seconds?: number;
  speaker_enrollment_sample_seconds?: number;
  speaker_identity_label?: string;
  speaker_retain_raw_enrollment_audio?: boolean;
  dedupe_similarity_threshold?: number;
  web_context_enabled?: boolean;
  web_context_configured?: boolean;
  test_mode_enabled?: boolean;
  sidecar_quality_gate_enabled?: boolean;
  test_audio_run_dir?: string;
};

type OllamaModelOption = {
  name: string;
  size?: number | null;
  digest?: string | null;
  modified_at?: string | null;
  cloud?: boolean;
  current?: boolean;
  configured?: boolean;
};

type OllamaModelsResponse = {
  selected_chat_model: string;
  chat_host: string;
  models: OllamaModelOption[];
  error?: string | null;
};

type PipelineState = {
  queueDepth: number;
  droppedWindows: number;
  lastAudioRms?: number;
  silentWindows: number;
  asrEmptyWindows: number;
  finalSegmentsReplaced: number;
  asrDurationMs?: number;
  partialAsrDurationMs?: number;
};

type SpeakerStatus = {
  profile: {
    id: string;
    display_name: string;
    kind: string;
    active: boolean;
    threshold: number | null;
    embedding_model?: string;
    embedding_model_version?: string;
    created_at?: number;
    updated_at?: number;
  };
  backend: {
    available: boolean;
    model_name: string;
    model_version: string;
    device: string;
    reason?: string | null;
  };
  enrollment_status: "not_enrolled" | "needs_more_audio" | "ready" | "needs_recalibration";
  ready: boolean;
  needs_recalibration: boolean;
  usable_speech_seconds: number;
  target_speech_seconds: number;
  minimum_speech_seconds: number;
  embedding_count: number;
  centroid_vector_dim: number | null;
  same_speaker_scores: {
    count: number;
    min: number | null;
    mean: number | null;
  };
  quality_score: number;
  raw_audio_retained: boolean;
  recent_segments?: {
    id: string;
    session_id?: string;
    segment_id?: string | null;
    display_speaker_label?: string | null;
    match_confidence?: number | null;
    match_score?: number | null;
    transcript_text?: string | null;
  }[];
};

type SpeakerEnrollmentSample = {
  id: string;
  enrollment_id: string;
  duration_seconds: number;
  usable_speech_seconds: number;
  quality_score: number;
  issues: string[];
  raw_audio_retained: boolean;
};

type SpeakerEnrollment = {
  id: string;
  profile_id: string;
  status: string;
  started_at: number;
  completed_at: number | null;
  samples: SpeakerEnrollmentSample[];
  profile: SpeakerStatus;
  raw_audio_retained: boolean;
};

type MicTestResult = {
  device?: DeviceInfo | null;
  audio_source: AudioSource;
  quality: {
    duration_seconds: number;
    usable_speech_seconds: number;
    speech_fraction: number;
    rms: number;
    peak: number;
    clipping_fraction: number;
    quality_score: number;
    issues: string[];
  };
  recommendation: {
    status: "good" | "too_quiet" | "too_loud" | "noisy";
    title: string;
    detail: string;
  };
  mic_tuning?: MicTuning;
  suggested_tuning?: (MicTuning & { reason?: string }) | null;
  playback_audio?: {
    mime_type: string;
    data_base64: string;
    duration_seconds: number;
  } | null;
  raw_audio_retained: boolean;
};

type ManualWebResponse = {
  query: string;
  enabled: boolean;
  configured: boolean;
  note: NoteCard | null;
  skip_reason?: string | null;
  cards?: Record<string, unknown>[];
};

type ManualSidecarResponse = {
  query: string;
  sanitized_web_query?: string | null;
  web_skip_reason?: string | null;
  cards: Record<string, unknown>[];
  sections: Record<string, Record<string, unknown>[]>;
  raw_audio_retained: boolean;
};

type InputFileUploadResponse = {
  path: string;
  filename: string;
  size_bytes: number;
};

type SystemLog = {
  id: string;
  seq: number;
  at: number;
  level: "status" | "warning" | "error";
  title: string;
  message: string;
  metadata?: Record<string, unknown>;
};

type SystemLogInput = Omit<SystemLog, "id" | "seq" | "at"> & {
  id?: string;
  at?: number;
};

type BackendItemOptions = {
  origin?: "live" | "manual" | "unpaired";
  sourceSegmentIds?: string[];
};

type FetchJsonResult<T> = {
  ok: boolean;
  payload: T;
};

export function App() {
  const [activePage, setActivePage] = useState<AppPage>(() => {
    if (typeof window === "undefined") {
      return "live";
    }
    const storedPage = window.localStorage.getItem(APP_PAGE_STORAGE_KEY);
    if (storedPage === "memory") {
      return "references";
    }
    return isAppPage(storedPage) ? storedPage : "live";
  });
  const [theme, setTheme] = useState<ThemeName>(() => {
    if (typeof window === "undefined") {
      return DEFAULT_THEME;
    }
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY) as ThemeName | null;
    return storedTheme && THEMES.some((item) => item.value === storedTheme) ? storedTheme : DEFAULT_THEME;
  });
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [deviceNotice, setDeviceNotice] = useState("");
  const [audioSource, setAudioSource] = useState<AudioSource>(() => defaultAudioSource());
  const [micTuning, setMicTuning] = useState<MicTuning>(() => loadMicTuning());
  const [meetingFocus, setMeetingFocus] = useState<MeetingFocusState>(() => loadMeetingFocus());
  const [meetingFocusExpanded, setMeetingFocusExpanded] = useState(false);
  const [activeMeetingContract, setActiveMeetingContract] = useState<MeetingContractPayload | null>(null);
  const [meetingDiagnostics, setMeetingDiagnostics] = useState<MeetingDiagnostics | null>(null);
  const [energyFrame, setEnergyFrame] = useState<EnergyFrame | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [liveId, setLiveId] = useState<string | null>(null);
  const [currentSessionTitle, setCurrentSessionTitle] = useState("New meeting");
  const [sessionCatalog, setSessionCatalog] = useState<SessionSummary[]>([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [showAllSessions, setShowAllSessions] = useState(false);
  const [sessionsBusy, setSessionsBusy] = useState("");
  const [selectedPastSessionId, setSelectedPastSessionId] = useState<string | null>(null);
  const [selectedPastSession, setSelectedPastSession] = useState<SessionDetail | null>(null);
  const [sessionTitleDraft, setSessionTitleDraft] = useState("");
  const [reviewTitle, setReviewTitle] = useState("");
  const [reviewJob, setReviewJob] = useState<ReviewJobResponse | null>(null);
  const [reviewJobs, setReviewJobs] = useState<ReviewJobResponse[]>([]);
  const [reviewBusy, setReviewBusy] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [toolView, setToolView] = useState<ToolView>("voice");
  const [status, setStatus] = useState("idle");
  const [gpu, setGpu] = useState<GpuHealth>({});
  const [ollamaModels, setOllamaModels] = useState<OllamaModelsResponse | null>(null);
  const [ollamaModelBusy, setOllamaModelBusy] = useState("");
  const [ollamaModelNotice, setOllamaModelNotice] = useState("");
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [notes, setNotes] = useState<NoteCard[]>([]);
  const [recall, setRecall] = useState<RecallHit[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [pipeline, setPipeline] = useState<PipelineState>({
    queueDepth: 0,
    droppedWindows: 0,
    silentWindows: 0,
    asrEmptyWindows: 0,
    finalSegmentsReplaced: 0,
  });
  const [fixturePath, setFixturePath] = useState(DEFAULT_FIXTURE_AUDIO);
  const [testSourcePath, setTestSourcePath] = useState(DEFAULT_FIXTURE_AUDIO);
  const [testMaxSeconds, setTestMaxSeconds] = useState(DEFAULT_TEST_MAX_SECONDS);
  const [testExpectedTerms, setTestExpectedTerms] = useState("");
  const [testPrepared, setTestPrepared] = useState<TestModePrepared | null>(null);
  const [testBusy, setTestBusy] = useState("");
  const [testPlaybackStartedAtMs, setTestPlaybackStartedAtMs] = useState<number | null>(null);
  const [testPlaybackBaseElapsedMs, setTestPlaybackBaseElapsedMs] = useState(0);
  const [testPlaybackNow, setTestPlaybackNow] = useState(Date.now());
  const [liveTestExpanded, setLiveTestExpanded] = useState(false);
  const [fileUploadBusy, setFileUploadBusy] = useState("");
  const [testReport, setTestReport] = useState<TestModeReport | null>(null);
  const [libraryPath, setLibraryPath] = useState("");
  const [libraryRoots, setLibraryRoots] = useState<string[]>([]);
  const [librarySources, setLibrarySources] = useState<LibraryChunkSource[]>([]);
  const [libraryChunks, setLibraryChunks] = useState<LibraryChunk[]>([]);
  const [selectedLibrarySourcePath, setSelectedLibrarySourcePath] = useState<string | null>(null);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryBusy, setMemoryBusy] = useState("");
  const [manualQuery, setManualQuery] = useState("");
  const [manualQueryBusy, setManualQueryBusy] = useState(false);
  const [speakerStatus, setSpeakerStatus] = useState<SpeakerStatus | null>(null);
  const [speakerEnrollment, setSpeakerEnrollment] = useState<SpeakerEnrollment | null>(null);
  const [speakerBusy, setSpeakerBusy] = useState("");
  const [micTest, setMicTest] = useState<MicTestResult | null>(null);
  const [micTestBusy, setMicTestBusy] = useState(false);
  const [micPlaybackUrl, setMicPlaybackUrl] = useState("");
  const [speakerRecordingStartedAt, setSpeakerRecordingStartedAt] = useState<number | null>(null);
  const [speakerRecordingNow, setSpeakerRecordingNow] = useState(Date.now());
  const [transcriptEvents, setTranscriptEvents] = useState<TranscriptEvent[]>([]);
  const [sidecarCards, setSidecarCards] = useState<SidecarDisplayCard[]>([]);
  const [dismissedCardKeys, setDismissedCardKeys] = useState<Set<string>>(() => new Set());
  const [expandedContextCards, setExpandedContextCards] = useState<Set<string>>(() => new Set());
  const [highlightedSourceIds, setHighlightedSourceIds] = useState<Set<string>>(() => new Set());
  const [systemLogs, setSystemLogs] = useState<SystemLog[]>([]);
  const [showAllContext, setShowAllContext] = useState(false);
  const [briefVisible, setBriefVisible] = useState(false);
  const [showDebugMetadata, setShowDebugMetadata] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.localStorage.getItem(DEBUG_METADATA_STORAGE_KEY) === "true";
  });
  const [followLive, setFollowLive] = useState(true);
  const [unseenItems, setUnseenItems] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const speakerEventSourceRef = useRef<EventSource | null>(null);
  const testStartedAtRef = useRef<string | null>(null);
  const testSessionIdRef = useRef<string | null>(null);
  const activeTestPreparedRef = useRef<TestModePrepared | null>(null);
  const testAutoStopTimerRef = useRef<number | null>(null);
  const testSnapshotRef = useRef<TestModeSnapshot>({
    transcript: [],
    notes: [],
    recall: [],
    sidecarCards: [],
    pipeline: {
      queueDepth: 0,
      droppedWindows: 0,
      silentWindows: 0,
      asrEmptyWindows: 0,
      finalSegmentsReplaced: 0,
    },
    errors: [],
    status: "idle",
    testExpectedTerms: "",
  });
  const eventSeqRef = useRef(0);
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null);
  const reviewTranscriptScrollRef = useRef<HTMLDivElement | null>(null);
  const liveEndRef = useRef<HTMLDivElement | null>(null);
  const followLiveRef = useRef(true);
  const micPlaybackUrlRef = useRef<string | null>(null);
  const evidenceHighlightTimerRef = useRef<number | null>(null);
  const captureWarningRef = useRef("");

  useEffect(() => {
    refreshDevices();
    refreshGpu();
    refreshSpeakerStatus();
    refreshSessionCatalog();
    attachSpeakerEvents();
  }, []);

  useEffect(() => {
    window.localStorage.setItem(APP_PAGE_STORAGE_KEY, activePage);
    if (activePage === "sessions") {
      refreshSessionCatalog();
    }
    if (activePage === "references") {
      refreshMemoryPage();
    }
    if (activePage === "models") {
      refreshGpu();
      refreshOllamaModels();
    }
    if (activePage === "review" && reviewBusy !== "uploading") {
      void refreshReviewJobs();
    }
  }, [activePage]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(DEBUG_METADATA_STORAGE_KEY, String(showDebugMetadata));
  }, [showDebugMetadata]);

  useEffect(() => {
    window.localStorage.setItem(MIC_TUNING_STORAGE_KEY, JSON.stringify(micTuning));
  }, [micTuning]);

  useEffect(() => {
    window.localStorage.setItem(MEETING_FOCUS_STORAGE_KEY, JSON.stringify(meetingFocus));
  }, [meetingFocus]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      speakerEventSourceRef.current?.close();
      clearTestAutoStopTimer();
      if (evidenceHighlightTimerRef.current) {
        window.clearTimeout(evidenceHighlightTimerRef.current);
      }
      if (micPlaybackUrlRef.current) {
        URL.revokeObjectURL(micPlaybackUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    testSnapshotRef.current = {
      transcript,
      notes,
      recall,
      sidecarCards,
      pipeline,
      errors,
      status,
      testExpectedTerms,
    };
  }, [errors, notes, pipeline, recall, sidecarCards, status, testExpectedTerms, transcript]);

  useEffect(() => {
    if (speakerBusy !== "recording") {
      return;
    }
    const timer = window.setInterval(() => setSpeakerRecordingNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [speakerBusy]);

  useEffect(() => {
    if (testBusy !== "playing" || testPlaybackStartedAtMs === null) {
      return;
    }
    setTestPlaybackNow(Date.now());
    const timer = window.setInterval(() => setTestPlaybackNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [testBusy, testPlaybackStartedAtMs]);

  useEffect(() => {
    if (!reviewJob || isReviewTerminal(reviewJob.status)) {
      return;
    }
    const timer = window.setTimeout(() => {
      void refreshReviewJob(reviewJob.job_id);
      void refreshReviewJobs();
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [reviewJob?.job_id, reviewJob?.status]);

  const gpuLabel = useMemo(() => {
    if (!gpu.nvidia_available) return "GPU not visible";
    const total = gpu.memory_total_mb ?? 0;
    const used = gpu.memory_used_mb ?? 0;
    return `${gpu.name ?? "NVIDIA GPU"} · ${used}/${total} MB`;
  }, [gpu]);

  const vramPercent = useMemo(() => {
    if (!gpu.memory_total_mb || !gpu.memory_used_mb) return 0;
    return Math.min(100, Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100));
  }, [gpu.memory_total_mb, gpu.memory_used_mb]);
  const usefulContextCards = useMemo(
    () => buildSidecarDisplayCards(
      sidecarCards.filter((card) => !dismissedCardKeys.has(card.cardKey ?? card.id)),
      transcriptEvents,
      { maxCards: Number.POSITIVE_INFINITY },
    ),
    [dismissedCardKeys, sidecarCards, transcriptEvents],
  );
  const currentMeetingCards = useMemo(
    () => usefulContextCards.filter(isCurrentMeetingCard),
    [usefulContextCards],
  );
  const otherContextCards = useMemo(
    () => usefulContextCards.filter((card) => !isCurrentMeetingCard(card) && !isManualCard(card)),
    [usefulContextCards],
  );
  const manualSourceCards = useMemo(
    () => usefulContextCards.filter(isManualCard).slice(0, 12),
    [usefulContextCards],
  );
  const liveCards = useMemo(
    () => usefulContextCards.filter((card) => isCurrentMeetingCard(card) || isManualCard(card)),
    [usefulContextCards],
  );
  const liveFieldRows = useMemo(
    () => buildLiveFieldRows(transcriptEvents, liveCards),
    [transcriptEvents, liveCards],
  );
  const liveFieldScrollKey = useMemo(
    () => liveFieldRows.map((row) => (
      `${row.id}:${row.transcript?.id ?? ""}:${row.cards.map((card) => `${card.id}:${expandedContextCards.has(card.id) ? "open" : "closed"}`).join(",")}`
    )).join("|"),
    [expandedContextCards, liveFieldRows],
  );
  useLayoutEffect(() => {
    if (!followLiveRef.current) {
      return;
    }
    scrollFieldToLive("auto");
    const timer = window.setTimeout(() => {
      if (followLiveRef.current) {
        scrollFieldToLive("auto");
      }
    }, 40);
    return () => window.clearTimeout(timer);
  }, [liveFieldScrollKey]);
  const visibleMeetingCards = showAllContext
    ? currentMeetingCards
    : currentMeetingCards.slice(0, DEFAULT_CONTEXT_CARD_LIMIT);
  const visibleOtherContextCards = showAllContext
    ? otherContextCards
    : otherContextCards.slice(0, DEFAULT_CONTEXT_CARD_LIMIT);
  const meetingOutputGroups = useMemo(() => groupMeetingOutputCards(visibleMeetingCards), [visibleMeetingCards]);
  const qualityMetrics = useMemo(() => buildSidecarQualityMetrics(usefulContextCards), [usefulContextCards]);
  const consultingBriefMarkdown = useMemo(
    () => buildConsultingBriefMarkdown(usefulContextCards, "Consulting Brief", activeMeetingContract ?? undefined),
    [activeMeetingContract, usefulContextCards],
  );
  const reviewTranscriptEvents = useMemo(
    () => (reviewJob?.clean_segments ?? []).map((segment) => ({
      id: `review-transcript-${segment.id}`,
      segmentId: segment.id,
      sourceSegmentIds: segment.source_segment_ids ?? [segment.id],
      text: segment.text,
      at: payloadTimeMs(segment.created_at) ?? Date.now(),
      source: "microphone" as const,
      speakerRole: speakerRoleFromPayload(segment.speaker_role),
      speakerLabel: segment.speaker_label,
      speakerConfidence: segment.speaker_confidence,
      startedAt: payloadTimeMs(segment.start_s),
      endedAt: payloadTimeMs(segment.end_s),
      isFinal: true,
    })),
    [reviewJob?.clean_segments],
  );
  const reviewCards = useMemo(
    () => {
      if (!reviewJob) {
        return [];
      }
      return reviewJob.meeting_cards
        .map((payload) => displayCardForSidecarPayload(payload, {
          id: `review-card-${String(payload.id ?? "")}`,
          type: "sidecar_card",
          session_id: reviewJob.session_id ?? reviewJob.job_id,
          at: reviewJob.updated_at,
          payload,
        }))
        .filter((card): card is SidecarDisplayCard => Boolean(card));
    },
    [reviewJob],
  );
  const reviewMeetingGroups = useMemo(() => groupMeetingOutputCards(reviewCards), [reviewCards]);
  const reviewCapture = useMemo(
    () => reviewJob ? buildMeetingCaptureViewModel(reviewJob, reviewCards) : null,
    [reviewCards, reviewJob],
  );
  const reviewCaptureMarkdown = useMemo(
    () => reviewCapture ? buildMeetingCaptureMarkdown(reviewCapture) : "",
    [reviewCapture],
  );

  const asrBackendLabel = "Faster-Whisper";
  const asrModelLabel = gpu.asr_model ?? gpu.asr_primary_model ?? "small.en";
  const asrDeviceLabel = gpu.asr_device ?? "cpu";
  const asrDeviceStatusLabel = asrDeviceLabel === "cpu"
    ? "CPU ASR"
    : gpu.asr_cuda_available ? "CUDA ready" : gpu.asr_cuda_error ?? "CUDA check";
  const asrCadenceLabel = gpu.partial_transcripts_enabled
    ? `Preview ${formatSeconds(gpu.partial_window_seconds ?? 0)}`
    : `Final ${formatSeconds(gpu.transcription_window_seconds ?? 0)}`;
  const asrTimingLabel = pipeline.asrDurationMs != null
    ? `${pipeline.asrDurationMs.toFixed(0)} ms final`
    : pipeline.partialAsrDurationMs != null
      ? `${pipeline.partialAsrDurationMs.toFixed(0)} ms preview`
      : "ASR idle";
  const testModeVisible = Boolean(gpu.test_mode_enabled);
  const webStatusLabel = gpu.web_context_enabled
    ? gpu.web_context_configured ? "web ready" : "web key missing"
    : gpu.web_context_configured ? "web selectable" : "web off";
  const ollamaChatHostLabel = formatOllamaHostLabel(gpu.ollama_chat_host);
  const ollamaEmbedHostLabel = formatOllamaHostLabel(gpu.ollama_embed_host);
  const ollamaChatReachabilityLabel = gpu.ollama_chat_reachable == null
    ? "chat check"
    : gpu.ollama_chat_reachable ? "chat ready" : "chat offline";
  const ollamaEmbedReachabilityLabel = gpu.ollama_embed_reachable == null
    ? "embed check"
    : gpu.ollama_embed_reachable ? "embed ready" : "embed offline";
  const speakerEnrollmentLabel = speakerStatus?.profile.display_name ?? "BP";
  const speakerStatusLabel = !speakerStatus
    ? "loading"
    : speakerStatus.enrollment_status.replace(/_/g, " ");
  const speakerSampleSeconds = gpu.speaker_enrollment_sample_seconds ?? 8;
  const speakerRecordingElapsedSeconds = speakerRecordingStartedAt
    ? Math.max(0, (speakerRecordingNow - speakerRecordingStartedAt) / 1000)
    : 0;
  const speakerRecordingProgress = Math.min(100, Math.round((speakerRecordingElapsedSeconds / speakerSampleSeconds) * 100));
  const speakerUsableSpeech = speakerStatus?.usable_speech_seconds ?? 0;
  const speakerTargetSpeech = speakerStatus?.target_speech_seconds ?? 20;
  const speakerMinimumSpeech = speakerStatus?.minimum_speech_seconds ?? 15;
  const speakerNeededSpeech = Math.max(0, speakerMinimumSpeech - speakerUsableSpeech);
  const speakerProgressPercent = Math.min(100, Math.round((speakerUsableSpeech / speakerTargetSpeech) * 100));
  const speakerCanFinalize = speakerUsableSpeech >= speakerMinimumSpeech;
  const speakerSampleCount = speakerEnrollment?.samples.length ?? speakerStatus?.embedding_count ?? 0;
  const speakerProgress = `${Math.min(speakerTargetSpeech, speakerUsableSpeech).toFixed(1)}s/${speakerTargetSpeech.toFixed(0)}s`;
  const speakerStage = !speakerStatus?.backend.available
    ? "setup"
    : speakerStatus.ready
      ? "ready"
      : speakerEnrollment
        ? speakerCanFinalize ? "finish" : "record"
        : "start";
  const speakerReadinessTitle = !speakerStatus
    ? "Checking speaker identity"
    : speakerStatus.ready
      ? `${speakerEnrollmentLabel} identity ready`
      : speakerStatus.enrollment_status === "needs_recalibration"
        ? "Needs recalibration"
        : speakerUsableSpeech > 0
          ? `${Math.max(0, speakerMinimumSpeech - speakerUsableSpeech).toFixed(1)}s more usable speech needed`
          : "Not enrolled";
  const speakerReadinessDetail = !speakerStatus
    ? "Loading local embedding status."
    : !speakerStatus.backend.available
      ? speakerStatus.backend.reason ?? "Speaker embedding backend is unavailable."
      : speakerStatus.ready
        ? `Transcripts can label your matching voice as ${speakerEnrollmentLabel}; non-matches stay Other speaker or Unknown.`
        : `Record only ${speakerEnrollmentLabel} so transcripts can identify your voice. Raw enrollment audio is discarded.`;
  const speakerTrainingButtonLabel = speakerBusy === "starting"
    ? "Starting..."
    : speakerUsableSpeech > 0
      ? "Start Over"
      : `Set Up ${speakerEnrollmentLabel} Label`;
  const speakerRecordButtonLabel = speakerBusy === "recording"
    ? `Recording ${Math.min(speakerSampleSeconds, speakerRecordingElapsedSeconds).toFixed(1)}s`
    : `Record ${speakerSampleSeconds.toFixed(0)}s Sample`;
  const micQualityPercent = Math.round((micTest?.quality.quality_score ?? 0) * 100);
  const micPeakPercent = Math.round((micTest?.quality.peak ?? 0) * 100);
  const micUsableSeconds = micTest?.quality.usable_speech_seconds ?? 0;
  const micTestTone = micTest?.recommendation.status === "good"
    ? "good"
    : micTest
      ? "warning"
      : "";
  const speakerNextActionLabel = speakerStage === "setup"
    ? "Speaker model unavailable"
    : speakerStage === "ready"
      ? `${speakerEnrollmentLabel} transcript label is ready`
      : speakerStage === "start"
        ? `Set up ${speakerEnrollmentLabel} transcript label`
        : speakerStage === "finish"
          ? `Finish and use ${speakerEnrollmentLabel} label`
          : `Record ${Math.max(1, Math.ceil(speakerNeededSpeech / Math.max(1, speakerSampleSeconds)))} more sample${Math.ceil(speakerNeededSpeech / Math.max(1, speakerSampleSeconds)) === 1 ? "" : "s"}`;
  const captureStarting = ["starting", "freeing_gpu", "loading_asr"].includes(status);
  const captureActive = ["listening", "catching_up", "paused"].includes(status);
  const selectedServerMic = devices.find((device) => device.id === selectedDevice) ?? null;
  const serverMicInUse = Boolean(selectedServerMic?.in_use);
  const missingServerDevice = audioSource === "server_device" && (!selectedDevice || (serverMicInUse && !captureActive));
  const captureStartDisabled = captureStarting || captureActive || missingServerDevice;
  const captureStopDisabled = !(sessionId || liveId) || status === "idle" || status === "stopped";
  const serverMicStatus = selectedServerMic
    ? `${selectedServerMic.label}${selectedServerMic.in_use ? " (in use)" : selectedServerMic.healthy === false ? " (not usable)" : ""}`
    : "No healthy server microphone detected";
  const inputReadinessTitle = missingServerDevice
    ? "Missing mic"
    : micTest && micTest.recommendation.status !== "good"
      ? "Needs tuning"
      : "Ready";
  const inputReadinessDetail = missingServerDevice
    ? serverMicStatus
    : `${serverMicStatus} · gain ${micTuning.input_gain_db > 0 ? "+" : ""}${micTuning.input_gain_db.toFixed(0)} dB · ${micTuning.speech_sensitivity}`;
  const playbackActive = ["playing", "starting-playback", "pausing", "paused", "resuming"].includes(testBusy);
  const playbackPaused = testBusy === "paused" || status === "paused";
  const captureModeLabel = playbackActive || audioSource === "fixture"
    ? "Recorded audio"
    : "Server mic";
  const sessionRetentionStatus = "Ephemeral Live";
  const activeRetentionStatus = "Queued for Review";
  const gpuFreeLabel = gpu.memory_free_mb != null && gpu.memory_total_mb != null
    ? `${gpu.memory_free_mb}/${gpu.memory_total_mb} MB free`
    : "VRAM check";
  const captureStatusSummary = missingServerDevice
    ? "No healthy server mic"
    : `${captureModeLabel} · ${captureActive ? activeRetentionStatus : sessionRetentionStatus}`;
  const topRuntimeSummary = captureActive || captureStarting
    ? captureStatusSummary
    : `${asrBackendLabel} · ${gpuFreeLabel}`;
  const topRuntimeDetail = captureActive || captureStarting
    ? `${asrBackendLabel} · ${gpuFreeLabel}`
    : captureStatusSummary;
  const headerQueueLabel = pipeline.queueDepth > 0 || captureActive || captureStarting
    ? `Queue ${pipeline.queueDepth}`
    : "";
  const contextEmptyTitle = captureStarting
    ? "Starting context"
    : captureActive
      ? "Live reference"
      : "Context standby";
  const contextEmptyBody = captureActive && currentMeetingCards.length === 0
    ? "No grounded cards yet. Unsupported/noisy cards are being suppressed."
    : captureActive || captureStarting
      ? "Actions, decisions, questions, and useful context will appear here."
      : "Start listening, record a transcript, or run a query.";
  const captureIdleButtonLabel = audioSource === "fixture"
    ? "Start Playback"
    : "Start Live";
  const captureActiveLabel = playbackPaused ? "Paused" : playbackActive ? "Playing..." : "Live Running";
  const captureButtonLabel = status === "freeing_gpu"
    ? "Freeing GPU..."
    : status === "loading_asr" || status === "starting"
      ? "Loading ASR..."
      : captureActive
        ? captureActiveLabel
        : captureIdleButtonLabel;
  const playbackButtonLabel = testBusy === "starting-playback" || status === "freeing_gpu" || status === "loading_asr"
    ? "Starting Playback..."
    : testBusy === "playing"
      ? "Playing..."
      : testBusy === "paused"
        ? "Resume"
      : "Start Playback";
  const prepareButtonLabel = testBusy === "preparing" ? "Preparing..." : testPrepared ? "Prepared" : "Prepare Audio";
  const liveTestUploadDisabled = captureActive || captureStarting || Boolean(fileUploadBusy) || Boolean(testBusy);
  const liveTestConfigDisabled = liveTestUploadDisabled;
  const liveTestToggleDisabled = Boolean(fileUploadBusy) || testBusy === "preparing";
  const liveTestStatus = fileUploadBusy === "test-audio"
    ? "uploading"
    : testBusy || (testPrepared ? "ready" : "idle");
  const liveTestStateLabel = liveTestStatus === "uploading"
    ? "Uploading"
    : liveTestStatus === "preparing"
      ? "Preparing"
      : liveTestStatus === "starting-playback"
        ? "Starting"
        : liveTestStatus === "playing"
          ? "Playing"
          : liveTestStatus === "paused"
            ? "Paused"
            : liveTestStatus === "pausing"
              ? "Pausing"
              : liveTestStatus === "resuming"
                ? "Resuming"
                : testReport
                  ? "Report ready"
                  : testPrepared
                    ? "Ready"
                    : "Idle";
  const liveTestStateDetail = liveTestStatus === "uploading"
    ? "Uploading the file to local test storage."
    : liveTestStatus === "preparing"
      ? "Converting audio into a local WAV fixture."
      : liveTestStatus === "starting-playback"
        ? "Starting the live pipeline with recorded audio."
        : liveTestStatus === "playing"
          ? "Feeding recorded audio through the live transcript."
          : liveTestStatus === "paused"
            ? "Playback is paused; resume to continue the transcript."
            : testReport
              ? "Playback finished and the local report is ready."
              : testPrepared
                ? "Audio is prepared. Start playback when ready."
                : "Upload a common audio file to run it through Live.";
  const canResetLiveTest = Boolean(testPrepared || testReport || testBusy || fileUploadBusy === "test-audio");
  const testPlaybackDurationSeconds = testPrepared?.duration_seconds ?? 0;
  const testPlaybackElapsedSeconds = testBusy === "playing" && testPlaybackStartedAtMs !== null
    ? Math.min(
        testPlaybackDurationSeconds || Number.POSITIVE_INFINITY,
        Math.max(0, (testPlaybackBaseElapsedMs + testPlaybackNow - testPlaybackStartedAtMs) / 1000),
      )
    : playbackPaused
      ? Math.min(testPlaybackDurationSeconds || Number.POSITIVE_INFINITY, Math.max(0, testPlaybackBaseElapsedMs / 1000))
    : 0;
  const testPlaybackProgress = testPlaybackDurationSeconds > 0
    ? Math.max(0, Math.min(100, (testPlaybackElapsedSeconds / testPlaybackDurationSeconds) * 100))
    : 0;
  const testPlaybackRemainingSeconds = testPlaybackDurationSeconds > 0
    ? Math.max(0, testPlaybackDurationSeconds - testPlaybackElapsedSeconds)
    : 0;
  const testPlaybackMessage = transcript.length > 0
    ? `${transcript.length} heard ${pluralize("line", transcript.length)} so far. Cards will land when there is enough evidence.`
    : pipeline.asrEmptyWindows > 0
      ? "Audio is moving; ASR has not found speech in the recent windows yet."
      : "Audio is playing through the live pipeline. Waiting for the first transcript window.";
  const statusTone = status === "listening" || status === "catching_up"
    ? "live"
    : status === "error"
      ? "danger"
      : captureStarting
        ? "busy"
        : "idle";
  const liveWorkbenchMode: LiveWorkbenchMode = briefVisible
    ? "ended"
    : playbackActive || playbackPaused
      ? "replay"
      : liveTestExpanded || Boolean(testBusy) || Boolean(testPrepared) || fileUploadBusy === "test-audio"
        ? "dev-test"
        : "live";

  async function refreshDevices() {
    const response = await fetch(`${API_BASE}/api/devices`);
    const payload = await response.json() as DeviceListResponse;
    const nextDevices = payload.devices ?? [];
    setDevices(nextDevices);
    const selected = payload.selected_device ?? nextDevices.find((device: DeviceInfo) => device.healthy !== false) ?? null;
    setDeviceNotice(selected
      ? payload.selection_reason || selected.selection_reason || "Server microphone auto-selected."
      : payload.selection_reason || "No healthy server microphone detected. Connect a mic, then refresh.");
    setSelectedDevice(selected?.id ?? "");
  }

  async function refreshGpu() {
    const response = await fetch(`${API_BASE}/api/health/gpu`);
    setGpu(await response.json());
  }

  async function refreshOllamaModels() {
    setOllamaModelBusy((current) => current || "loading");
    try {
      const response = await fetch(`${API_BASE}/api/models/ollama`);
      if (!response.ok) {
        setOllamaModelNotice("Model list unavailable");
        return;
      }
      const payload = await response.json() as OllamaModelsResponse;
      setOllamaModels(payload);
      if (payload.error) {
        setOllamaModelNotice("Model list degraded");
      } else if (ollamaModelNotice === "Model list unavailable" || ollamaModelNotice === "Model list degraded") {
        setOllamaModelNotice("");
      }
    } catch {
      setOllamaModelNotice("Model list unavailable");
    } finally {
      setOllamaModelBusy((current) => current === "loading" ? "" : current);
    }
  }

  async function selectOllamaChatModel(model: string) {
    const cleanModel = model.trim();
    if (!cleanModel || cleanModel === (gpu.ollama_chat_model ?? ollamaModels?.selected_chat_model)) {
      return;
    }
    setOllamaModelBusy("saving");
    setOllamaModelNotice("");
    try {
      const response = await fetch(`${API_BASE}/api/models/ollama/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: cleanModel }),
      });
      const payload = await response.json() as OllamaModelsResponse & { detail?: string };
      if (!response.ok) {
        setOllamaModelNotice(payload.detail || "Model update failed");
        return;
      }
      setOllamaModels(payload);
      setGpu((current) => ({
        ...current,
        ollama_chat_model: payload.selected_chat_model,
        ollama_chat_host: payload.chat_host,
      }));
      setOllamaModelNotice(`Saved ${payload.selected_chat_model} to .env`);
      await refreshGpu();
    } catch {
      setOllamaModelNotice("Model update failed");
    } finally {
      setOllamaModelBusy("");
    }
  }

  async function refreshSpeakerStatus() {
    const response = await fetch(`${API_BASE}/api/speaker/status`);
    if (!response.ok) return;
    setSpeakerStatus(await response.json());
  }

  async function refreshSessionCatalog() {
    setSessionsBusy((current) => current || "loading");
    const query = sessionSearch.trim();
    const url = new URL(`${API_BASE}/api/sessions`, window.location.origin);
    url.searchParams.set("include_empty", "true");
    url.searchParams.set("limit", "80");
    if (query) {
      url.searchParams.set("query", query);
    }
    const response = await fetch(url);
    setSessionsBusy((current) => current === "loading" ? "" : current);
    if (!response.ok) {
      return;
    }
    const payload = await response.json() as { sessions?: SessionSummary[] };
    setSessionCatalog(payload.sessions ?? []);
  }

  async function loadPastSession(id: string) {
    setSessionsBusy("detail");
    const response = await fetch(`${API_BASE}/api/sessions/${id}`);
    setSessionsBusy("");
    if (!response.ok) {
      return;
    }
    const payload = await response.json() as SessionDetail;
    setSelectedPastSessionId(id);
    setSelectedPastSession(payload);
    setSessionTitleDraft(payload.title);
  }

  async function updatePastSessionTitle() {
    if (!selectedPastSession || !sessionTitleDraft.trim()) {
      return;
    }
    setSessionsBusy("saving-title");
    const response = await fetch(`${API_BASE}/api/sessions/${selectedPastSession.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: sessionTitleDraft.trim() }),
    });
    setSessionsBusy("");
    if (!response.ok) {
      return;
    }
    const metadata = await response.json() as SessionSummary;
    setSelectedPastSession((current) => current ? { ...current, ...metadata } : current);
    setSessionTitleDraft(metadata.title);
    setSessionCatalog((current) => current.map((session) => session.id === metadata.id ? metadata : session));
    if (metadata.id === sessionId) {
      setCurrentSessionTitle(metadata.title);
    }
  }

  async function createSession(title?: string, signal?: AbortSignal) {
    const requestedTitle = (title ?? currentSessionTitle.trim()) || null;
    const response = await fetch(`${API_BASE}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: requestedTitle }),
      signal,
    });
    const payload = await response.json();
    setSessionId(payload.id);
    const nextTitle = String(payload.title ?? requestedTitle ?? "New meeting");
    setCurrentSessionTitle(nextTitle);
    setTranscript([]);
    setNotes([]);
    setRecall([]);
    setErrors([]);
    setPipeline({
      queueDepth: 0,
      droppedWindows: 0,
      silentWindows: 0,
      asrEmptyWindows: 0,
      finalSegmentsReplaced: 0,
    });
    captureWarningRef.current = "";
    setActiveMeetingContract(null);
    setMeetingDiagnostics(null);
    setEnergyFrame(null);
    setTranscriptEvents([]);
    setSidecarCards([]);
    setDismissedCardKeys(new Set());
    setExpandedContextCards(new Set());
    setSystemLogs([]);
    setShowAllContext(false);
    setUnseenItems(0);
    followLiveRef.current = true;
    setFollowLive(true);
    attachEvents(payload.id);
    void refreshSessionCatalog().catch(() => {
      appendSystemLog({
        level: "warning",
        title: "Session list refresh skipped",
        message: "The meeting session was created, but the saved-session list could not refresh yet.",
      });
    });
    return payload.id as string;
  }

  async function newLiveSession() {
    if (captureActive || captureStarting) {
      return;
    }
    setSessionId(null);
    setLiveId(null);
    setCurrentSessionTitle("New meeting");
    setTranscript([]);
    setNotes([]);
    setRecall([]);
    setErrors([]);
    setPipeline({
      queueDepth: 0,
      droppedWindows: 0,
      silentWindows: 0,
      asrEmptyWindows: 0,
      finalSegmentsReplaced: 0,
    });
    captureWarningRef.current = "";
    setActiveMeetingContract(null);
    setMeetingDiagnostics(null);
    setEnergyFrame(null);
    setTranscriptEvents([]);
    setSidecarCards([]);
    setDismissedCardKeys(new Set());
    setExpandedContextCards(new Set());
    setSystemLogs([]);
    setShowAllContext(false);
    setUnseenItems(0);
    followLiveRef.current = true;
    setFollowLive(true);
    setStatus("idle");
    setActivePage("live");
  }

  async function updateLiveSessionTitle(titleOverride?: string) {
    const cleanTitle = (titleOverride ?? currentSessionTitle).trim();
    if (!sessionId || !cleanTitle || captureActive || captureStarting) {
      return;
    }
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: cleanTitle }),
    });
    if (!response.ok) {
      return;
    }
    const metadata = await response.json() as SessionSummary;
    setCurrentSessionTitle(metadata.title);
    setSessionCatalog((current) => current.map((session) => session.id === metadata.id ? metadata : session));
    if (selectedPastSession?.id === metadata.id) {
      setSelectedPastSession((current) => current ? { ...current, ...metadata } : current);
      setSessionTitleDraft(metadata.title);
    }
  }

  function selectLiveSession(id: string) {
    if (captureActive || captureStarting) {
      return;
    }
    if (id === "new") {
      setSessionId(null);
      setCurrentSessionTitle("New meeting");
        return;
    }
    const selected = sessionCatalog.find((session) => session.id === id);
    if (!selected) {
      return;
    }
    setSessionId(selected.id);
    setCurrentSessionTitle(selected.title);
    setTranscript([]);
    setNotes([]);
    setRecall([]);
    setTranscriptEvents([]);
    setSidecarCards([]);
    setDismissedCardKeys(new Set());
    setExpandedContextCards(new Set());
    attachEvents(selected.id);
  }

  async function refreshMemoryPage() {
    setMemoryBusy((current) => current || "loading");
    await Promise.all([
      refreshLibraryRoots(),
      refreshLibraryChunks(selectedLibrarySourcePath ?? undefined),
    ]);
    setMemoryBusy((current) => current === "loading" ? "" : current);
  }

  async function refreshLibraryRoots() {
    const response = await fetch(`${API_BASE}/api/library/roots`);
    if (!response.ok) {
      return;
    }
    const payload = await response.json() as { roots?: string[] };
    setLibraryRoots(payload.roots ?? []);
  }

  async function refreshLibraryChunks(sourcePath?: string) {
    const url = new URL(`${API_BASE}/api/library/chunks`, window.location.origin);
    url.searchParams.set("limit", "120");
    if (memoryQuery.trim()) {
      url.searchParams.set("query", memoryQuery.trim());
    }
    if (sourcePath) {
      url.searchParams.set("source_path", sourcePath);
    }
    const response = await fetch(url);
    if (!response.ok) {
      return;
    }
    const payload = await response.json() as { sources?: LibraryChunkSource[]; chunks?: LibraryChunk[] };
    setLibrarySources(payload.sources ?? []);
    setLibraryChunks(payload.chunks ?? []);
  }

  async function searchMemory() {
    const query = memoryQuery.trim();
    if (!query) {
      await refreshLibraryChunks(selectedLibrarySourcePath ?? undefined);
      return;
    }
    setMemoryBusy("searching");
    await refreshLibraryChunks(selectedLibrarySourcePath ?? undefined);
    setMemoryBusy("");
  }

  function attachEvents(id: string, kind: "session" | "live" = "session") {
    eventSourceRef.current?.close();
    const source = new EventSource(`${API_BASE}/api/${kind === "live" ? "live" : "sessions"}/${id}/events`);
    source.onmessage = handleEventMessage;
    for (const type of [
      "audio_status",
      "transcript_partial",
      "transcript_final",
      "note_update",
      "recall_hit",
      "sidecar_card",
      "gpu_status",
      "error",
    ]) {
      source.addEventListener(type, handleEventMessage);
    }
    eventSourceRef.current = source;
  }

  function attachSpeakerEvents() {
    speakerEventSourceRef.current?.close();
    const source = new EventSource(`${API_BASE}/api/events`);
    source.addEventListener("speaker_profile_update", (event) => {
      const envelope = JSON.parse((event as MessageEvent).data) as EventEnvelope;
      setSpeakerStatus(envelope.payload as unknown as SpeakerStatus);
    });
    speakerEventSourceRef.current = source;
  }

  function handleEventMessage(event: MessageEvent) {
    const envelope = JSON.parse(event.data) as EventEnvelope;
    const payload = envelope.payload;
    if (envelope.type === "audio_status") {
      const nextStatus = String(payload.status ?? "listening");
      setStatus(nextStatus);
      syncMeetingStatusFromPayload(payload);
      if ("memory_free_mb" in payload || "gpu_pressure" in payload || "ollama_gpu_models" in payload) {
        setGpu((current) => ({ ...current, ...payload }));
      }
      setPipeline((current) => ({
        ...current,
        queueDepth: Number(payload.queue_depth ?? current.queueDepth),
        droppedWindows: Number(payload.dropped_windows ?? current.droppedWindows),
        lastAudioRms: numberOrUndefined(payload.last_audio_rms) ?? current.lastAudioRms,
        silentWindows: Number(payload.silent_windows ?? current.silentWindows),
        asrEmptyWindows: Number(payload.asr_empty_windows ?? current.asrEmptyWindows),
        finalSegmentsReplaced: Number(payload.final_segments_replaced ?? current.finalSegmentsReplaced),
        asrDurationMs: numberOrUndefined(payload.asr_duration_ms) ?? current.asrDurationMs,
        partialAsrDurationMs: numberOrUndefined(payload.partial_asr_duration_ms) ?? current.partialAsrDurationMs,
      }));
      const captureWarning = captureWarningFromPayload(payload);
      if (captureWarning && captureWarning.id !== captureWarningRef.current) {
        captureWarningRef.current = captureWarning.id;
        appendSystemLog({
          level: "warning",
          title: captureWarning.title,
          message: captureWarning.message,
          metadata: payload,
        });
      }
      if (nextStatus === "freeing_gpu") {
        appendSystemLog({
          level: "status",
          title: "Freeing GPU for ASR",
          message: `Stopping GPU-resident Ollama models before ${asrBackendNameFromPayload(payload)} loads. Free VRAM: ${payload.memory_free_mb ?? "unknown"} MB.`,
          metadata: payload,
        });
      }
      if (nextStatus === "loading_asr") {
        appendSystemLog({
          level: "status",
          title: "Loading ASR",
          message: `${asrBackendNameFromPayload(payload)} is loading on ${payload.asr_device ?? "cpu"}. Target free VRAM: ${payload.asr_min_free_vram_mb ?? "unknown"} MB.`,
          metadata: payload,
        });
      }
      if (nextStatus === "catching_up") {
        appendSystemLog({
          level: "warning",
          title: "Capture falling behind",
          message: `${payload.dropped_windows ?? "Some"} audio windows dropped while ASR caught up.`,
          metadata: payload,
        });
      }
    }
    if (envelope.type === "gpu_status") {
      syncMeetingStatusFromPayload(payload);
      setGpu((current) => ({ ...current, ...payload }));
    }
    if (envelope.type === "transcript_partial") {
      setPipeline((current) => ({
        ...current,
        queueDepth: Number(payload.queue_depth ?? current.queueDepth),
        droppedWindows: Number(payload.dropped_windows ?? current.droppedWindows),
        lastAudioRms: numberOrUndefined(payload.last_audio_rms) ?? current.lastAudioRms,
        silentWindows: Number(payload.silent_windows ?? current.silentWindows),
        asrEmptyWindows: Number(payload.asr_empty_windows ?? current.asrEmptyWindows),
        finalSegmentsReplaced: Number(payload.final_segments_replaced ?? current.finalSegmentsReplaced),
        asrDurationMs: numberOrUndefined(payload.asr_duration_ms) ?? current.asrDurationMs,
        partialAsrDurationMs: numberOrUndefined(payload.partial_asr_duration_ms) ?? current.partialAsrDurationMs,
      }));
      appendTranscriptEvent(transcriptEventFromPayload(payload, envelope));
    }
    if (envelope.type === "transcript_final") {
      syncMeetingStatusFromPayload(payload);
      setPipeline((current) => ({
        ...current,
        queueDepth: Number(payload.queue_depth ?? current.queueDepth),
        droppedWindows: Number(payload.dropped_windows ?? current.droppedWindows),
        lastAudioRms: numberOrUndefined(payload.last_audio_rms) ?? current.lastAudioRms,
        silentWindows: Number(payload.silent_windows ?? current.silentWindows),
        asrEmptyWindows: Number(payload.asr_empty_windows ?? current.asrEmptyWindows),
        finalSegmentsReplaced: Number(payload.final_segments_replaced ?? current.finalSegmentsReplaced),
        asrDurationMs: numberOrUndefined(payload.asr_duration_ms) ?? current.asrDurationMs,
        partialAsrDurationMs: numberOrUndefined(payload.partial_asr_duration_ms) ?? current.partialAsrDurationMs,
      }));
      const line = {
        id: String(payload.id),
        text: String(payload.text),
        start_s: Number(payload.start_s),
        end_s: Number(payload.end_s),
        source_segment_ids: parseStringList(payload.source_segment_ids),
        asr_model: payload.asr_model ? String(payload.asr_model) : undefined,
        created_at: payload.created_at ? Number(payload.created_at) : undefined,
      };
      const replacesSegmentId = stringOrUndefined(payload.replaces_segment_id);
      setTranscript((current) => replaceTranscriptLine(current, line, replacesSegmentId));
      appendTranscriptEvent(transcriptEventFromPayload(payload, envelope));
    }
    if (envelope.type === "sidecar_card") {
      const card = displayCardForSidecarPayload(payload, envelope);
      if (card) {
        appendSidecarCard(card);
      }
    }
    if (envelope.type === "note_update") {
      const note = noteFromPayload(payload);
      if (!note) {
        return;
      }
      setNotes((current) => [note, ...current]);
      appendSidecarCard(displayCardForNote(note));
    }
    if (envelope.type === "recall_hit") {
      const hit = recallHitFromPayload(payload);
      appendSidecarCard(displayCardForRecall(hit));
      setRecall((current) => [hit, ...current.slice(0, 11)]);
    }
    if (envelope.type === "error") {
      const message = String(payload.message ?? "Unknown error");
      if (payload.fatal !== false) {
        setStatus("error");
      }
      setErrors((current) => [message, ...current.slice(0, 4)]);
      appendSystemLog({
        level: "error",
        title: "Runtime issue",
        message,
        metadata: payload,
      });
    }
  }

  function syncMeetingStatusFromPayload(payload: Record<string, unknown>) {
    const nextContract = meetingContractFromPayload(payload.meeting_contract);
    if (nextContract) {
      setActiveMeetingContract(nextContract);
    }
    const nextDiagnostics = meetingDiagnosticsFromPayload(payload.meeting_diagnostics);
    if (nextDiagnostics) {
      setMeetingDiagnostics(nextDiagnostics);
    }
    if ("energy_lens_active" in payload || "energy_lens_confidence" in payload) {
      setEnergyFrame(energyFrameFromPayload(payload));
    }
  }

  function pushError(message: string) {
    setErrors((current) => [message, ...current.slice(0, 4)]);
    appendSystemLog({
      level: "error",
      title: "Runtime issue",
      message,
    });
  }

  async function uploadInputFile(file: File, target: string): Promise<string | null> {
    const body = new FormData();
    body.append("file", file);
    setFileUploadBusy(target);
    const result = await fetchJson<InputFileUploadResponse>(`${API_BASE}/api/input-files`, {
      method: "POST",
      body,
    });
    setFileUploadBusy("");
    if (!result.ok) {
      const detail = (result.payload as { detail?: string }).detail ?? "File upload failed.";
      pushError(detail);
      return null;
    }
    appendSystemLog({
      level: "status",
      title: "Input file uploaded",
      message: `${result.payload.filename} (${humanFileSize(result.payload.size_bytes)})`,
      metadata: result.payload as unknown as Record<string, unknown>,
    });
    return result.payload.path;
  }

  async function handleInputFilePick(
    event: ChangeEvent<HTMLInputElement>,
    target: string,
    onPath: (path: string) => void,
  ) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) {
      return;
    }
    const path = await uploadInputFile(file, target);
    if (path) {
      onPath(path);
    }
  }

  async function handleReviewFilePick(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) {
      return;
    }
    const body = new FormData();
    body.append("file", file);
    body.append("source", "upload");
    if (reviewTitle.trim()) {
      body.append("title", reviewTitle.trim());
    }
    setReviewBusy("uploading");
    setReviewError("");
    setCurrentReviewJob(null);
    const result = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs`, {
      method: "POST",
      body,
    });
    setReviewBusy("");
    if (!result.ok) {
      const detail = (result.payload as { detail?: string }).detail ?? "Review upload failed.";
      setReviewError(detail);
      pushError(detail);
      return;
    }
    setCurrentReviewJob(result.payload);
    void refreshReviewJobs();
  }

  function startNewReview() {
    setReviewError("");
    setReviewTitle("");
    setCurrentReviewJob(null);
  }

  function setCurrentReviewJob(job: ReviewJobResponse | null) {
    if (job?.id && !job.job_id) {
      job.job_id = job.id;
    }
    setReviewJob(job);
    if (job) {
      setReviewJobs((current) => {
        const jobId = reviewJobId(job);
        const next = [job, ...current.filter((candidate) => reviewJobId(candidate) !== jobId)];
        return next.sort(compareReviewJobs);
      });
    }
    if (typeof window === "undefined") {
      return;
    }
    if (job?.job_id) {
      window.localStorage.setItem(REVIEW_LAST_JOB_STORAGE_KEY, job.job_id);
    } else {
      window.localStorage.removeItem(REVIEW_LAST_JOB_STORAGE_KEY);
    }
  }

  async function refreshReviewJobs() {
    setReviewBusy((current) => current || "loading-list");
    const result = await fetchJson<{ jobs?: ReviewJobResponse[] }>(`${API_BASE}/api/review/jobs?limit=50`, {
      method: "GET",
    });
    setReviewBusy((current) => current === "loading-list" ? "" : current);
    if (!result.ok) {
      return;
    }
    const jobs = (result.payload.jobs ?? []).map((job) => ({ ...job, job_id: reviewJobId(job) }));
    setReviewJobs(jobs);
    if (!reviewJob && jobs.length > 0) {
      const selected = jobs.find(isReviewDefaultSelectable);
      if (selected) {
        void refreshReviewJob(reviewJobId(selected));
      }
    } else if (reviewJob) {
      const updated = jobs.find((job) => reviewJobId(job) === reviewJobId(reviewJob));
      if (updated && updated.status !== reviewJob.status) {
        void refreshReviewJob(reviewJobId(updated));
      }
    }
  }

  async function refreshLatestReviewJob() {
    setReviewBusy((current) => current || "loading-latest");
    const latest = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs/latest`, {
      method: "GET",
    });
    if (latest.ok) {
      setCurrentReviewJob(latest.payload);
      setReviewBusy("");
      return;
    }
    const storedJobId = typeof window !== "undefined"
      ? window.localStorage.getItem(REVIEW_LAST_JOB_STORAGE_KEY)
      : "";
    if (storedJobId) {
      const stored = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs/${encodeURIComponent(storedJobId)}`, {
        method: "GET",
      });
      if (stored.ok) {
        setCurrentReviewJob(stored.payload);
      }
    }
    setReviewBusy("");
  }

  async function refreshReviewJob(jobId: string) {
    const result = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs/${encodeURIComponent(jobId)}`, {
      method: "GET",
    });
    if (!result.ok) {
      const detail = (result.payload as { detail?: string }).detail ?? "Review status unavailable.";
      setReviewError(detail);
      return;
    }
    const wasApproved = reviewJob?.status === "approved";
    setCurrentReviewJob(result.payload);
    if (!wasApproved && result.payload.status === "approved") {
      void refreshSessionCatalog();
      if (result.payload.session_id) {
        void loadPastSession(result.payload.session_id);
      }
    }
  }

  async function approveReviewJob() {
    if (!reviewJob || reviewJob.status !== "completed_awaiting_validation") {
      return;
    }
    setReviewBusy("approving");
    const result = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs/${encodeURIComponent(reviewJobId(reviewJob))}/approve`, {
      method: "POST",
    });
    setReviewBusy("");
    if (!result.ok) {
      const detail = (result.payload as { detail?: string }).detail ?? "Review approval failed.";
      setReviewError(detail);
      pushError(detail);
      return;
    }
    setCurrentReviewJob(result.payload);
    void refreshReviewJobs();
    if (result.payload.session_id) {
      void refreshSessionCatalog();
    }
  }

  async function discardReviewJob() {
    if (!reviewJob || !["queued", "paused_for_live", "completed_awaiting_validation", "error"].includes(reviewJob.status)) {
      return;
    }
    setReviewBusy("discarding");
    const result = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs/${encodeURIComponent(reviewJobId(reviewJob))}/discard`, {
      method: "POST",
    });
    setReviewBusy("");
    if (!result.ok) {
      const detail = (result.payload as { detail?: string }).detail ?? "Review discard failed.";
      setReviewError(detail);
      pushError(detail);
      return;
    }
    setCurrentReviewJob(result.payload);
    void refreshReviewJobs();
  }

  async function cancelReviewJob() {
    if (!reviewJob || isReviewTerminal(reviewJob.status)) {
      return;
    }
    setReviewBusy("canceling");
    const result = await fetchJson<ReviewJobResponse>(`${API_BASE}/api/review/jobs/${encodeURIComponent(reviewJobId(reviewJob))}/cancel`, {
      method: "POST",
    });
    setReviewBusy("");
    if (!result.ok) {
      const detail = (result.payload as { detail?: string }).detail ?? "Review cancel failed.";
      setReviewError(detail);
      pushError(detail);
      return;
    }
    setCurrentReviewJob(result.payload);
  }

  function appendTranscriptEvent(event: TranscriptEvent) {
    setTranscriptEvents((current) => {
      if (!event.isFinal && current.some((candidate) => transcriptFinalAlreadyCoversPartial(candidate, event))) {
        return current;
      }
      if (!event.isFinal && transcriptWordCount(event.text) < 2) {
        return current;
      }
      const replacementIds = transcriptReplacementIds(event);
      const withoutSameId = current.filter((candidate) => !transcriptReplacementIds(candidate).some((id) => replacementIds.includes(id)));
      const reconciled = withoutSameId.filter((candidate) => {
        if (!transcriptEventsOverlap(candidate, event)) {
          return true;
        }
        if (event.isFinal) {
          return candidate.isFinal;
        }
        return candidate.isFinal;
      });
      return [...reconciled, event].slice(-240);
    });
    if (!followLiveRef.current) {
      setUnseenItems((current) => current + 1);
    }
  }

  function appendSidecarCard(card: SidecarDisplayCard) {
    const key = card.cardKey ?? card.id;
    if (dismissedCardKeys.has(key) && !card.pinned) {
      return;
    }
    setSidecarCards((current) => {
      const existing = current.find((candidate) => (candidate.cardKey ?? candidate.id) === key);
      const pinned = Boolean(existing?.pinned || card.pinned);
      const nextCard = { ...card, pinned };
      const withoutSame = current.filter((candidate) => {
        const candidateKey = candidate.cardKey ?? candidate.id;
        if (candidateKey === key) {
          return false;
        }
        return !areCardsEquivalent(candidate, nextCard);
      });
      return [nextCard, ...withoutSame].slice(0, 160);
    });
  }

  function togglePinnedCard(card: SidecarDisplayCard) {
    const key = card.cardKey ?? card.id;
    setSidecarCards((current) => current.map((candidate) => (
      (candidate.cardKey ?? candidate.id) === key
        ? { ...candidate, pinned: !candidate.pinned }
        : candidate
    )));
  }

  function dismissCard(card: SidecarDisplayCard) {
    const key = card.cardKey ?? card.id;
    setDismissedCardKeys((current) => new Set(current).add(key));
    setSidecarCards((current) => current.filter((candidate) => (candidate.cardKey ?? candidate.id) !== key));
  }

  async function copyCardSuggestion(card: SidecarDisplayCard) {
    const text = card.suggestedSay ?? card.suggestedAsk ?? card.evidenceQuote ?? card.summary;
    if (!text) {
      return;
    }
    try {
      await navigator.clipboard?.writeText(text);
      appendSystemLog({
        level: "status",
        title: "Copied suggestion",
        message: text,
      });
    } catch {
      appendSystemLog({
        level: "warning",
        title: "Copy unavailable",
        message: text,
      });
    }
  }

  async function copyConsultingBrief() {
    try {
      await navigator.clipboard?.writeText(consultingBriefMarkdown);
      appendSystemLog({
        level: "status",
        title: "Copied consulting brief",
        message: "Meeting brief copied as Markdown.",
      });
    } catch {
      appendSystemLog({
        level: "warning",
        title: "Copy unavailable",
        message: consultingBriefMarkdown,
      });
    }
  }

  function appendSystemLog(item: SystemLogInput) {
    eventSeqRef.current += 1;
    const nextItem: SystemLog = {
      ...item,
      id: item.id ?? `system-${eventSeqRef.current}`,
      seq: eventSeqRef.current,
      at: item.at ?? Date.now(),
    };
    setSystemLogs((current) => [nextItem, ...current].slice(0, 80));
  }

  function setFollowState(value: boolean) {
    followLiveRef.current = value;
    setFollowLive(value);
    if (value) {
      setUnseenItems(0);
    }
  }

  function handleFieldScroll() {
    const node = transcriptScrollRef.current;
    if (!node) return;
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    setFollowState(distanceFromBottom < 96);
  }

  function scrollFieldToLive(behavior: ScrollBehavior = "auto") {
    const node = transcriptScrollRef.current;
    if (!node) return;
    const end = liveEndRef.current;
    if (end) {
      end.scrollIntoView({ behavior, block: "end" });
    }
    node.scrollTo({ top: Math.max(0, node.scrollHeight - node.clientHeight), behavior });
    setFollowState(true);
  }

  function toggleExpandedContextCard(id: string) {
    setExpandedContextCards((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function focusEvidenceIds(idsInput: string[]) {
    const ids = new Set(idsInput.filter(Boolean));
    if (ids.size === 0) {
      return;
    }
    setHighlightedSourceIds(ids);
    if (evidenceHighlightTimerRef.current) {
      window.clearTimeout(evidenceHighlightTimerRef.current);
    }
    evidenceHighlightTimerRef.current = window.setTimeout(() => {
      setHighlightedSourceIds(new Set());
      evidenceHighlightTimerRef.current = null;
    }, 2400);

    const node = transcriptScrollRef.current;
    if (!node) {
      return;
    }
    const targets = Array.from(node.querySelectorAll<HTMLElement>("[data-transcript-id], [data-segment-id]"))
      .filter((element) => {
        const sourceIds = (element.dataset.sourceSegmentIds ?? "").split(" ").filter(Boolean);
        return ids.has(element.dataset.transcriptId ?? "")
          || ids.has(element.dataset.segmentId ?? "")
          || sourceIds.some((sourceId) => ids.has(sourceId));
      });
    targets[0]?.scrollIntoView({ behavior: "smooth", block: "center" });
    setFollowState(false);
  }

  function focusReviewEvidenceIds(idsInput: string[]) {
    const ids = new Set(idsInput.filter(Boolean));
    if (ids.size === 0) {
      return;
    }
    setHighlightedSourceIds(ids);
    if (evidenceHighlightTimerRef.current) {
      window.clearTimeout(evidenceHighlightTimerRef.current);
    }
    evidenceHighlightTimerRef.current = window.setTimeout(() => {
      setHighlightedSourceIds(new Set());
      evidenceHighlightTimerRef.current = null;
    }, 2400);
    const node = reviewTranscriptScrollRef.current;
    if (!node) {
      return;
    }
    const targets = Array.from(node.querySelectorAll<HTMLElement>("[data-transcript-id], [data-source-segment-ids]"))
      .filter((element) => {
        const sourceIds = (element.dataset.sourceSegmentIds ?? "").split(" ").filter(Boolean);
        return ids.has(element.dataset.transcriptId ?? "") || sourceIds.some((sourceId) => ids.has(sourceId));
      });
    targets[0]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function jumpToEvidence(card: SidecarDisplayCard) {
    if ((card.sourceSegmentIds ?? []).filter(Boolean).length === 0) {
      return;
    }
    setExpandedContextCards((current) => new Set(current).add(card.id));
    focusEvidenceIds(card.sourceSegmentIds ?? []);
  }

  async function start(options: {
    fixtureWav?: string;
    testRun?: TestModePrepared;
    freshSession?: boolean;
    signal?: AbortSignal;
  } = {}): Promise<boolean> {
    if (captureStarting || captureActive) {
      return false;
    }
    const selectedSource: AudioSource = options.fixtureWav ? "fixture" : audioSource;
    const fixtureWav = options.fixtureWav
      ?? (selectedSource === "fixture" && fixturePath.trim() ? fixturePath.trim() : null);
    if (!fixtureWav && missingServerDevice) {
      return false;
    }
    const useSessionCompatibilityPath = Boolean(options.testRun || fixtureWav || audioSource === "fixture");
    let id: string;
    try {
      if (useSessionCompatibilityPath) {
        id = options.freshSession ? await createSession(undefined, options.signal) : sessionId ?? (await createSession(undefined, options.signal));
      } else {
        id = liveId ?? "";
      }
    } catch (error) {
      const aborted = error instanceof DOMException && error.name === "AbortError";
      const detail = error instanceof Error && error.message ? ` ${error.message}` : "";
      const message = aborted
        ? "Recorded audio start timed out before playback began."
        : `Recorded audio could not create a playback session.${detail}`;
      setStatus("error");
      if (options.testRun) {
        setTestBusy("");
        activeTestPreparedRef.current = null;
        setTestPlaybackStartedAtMs(null);
        clearTestAutoStopTimer();
      }
      setActiveMeetingContract(null);
      setErrors((current) => [message, ...current.slice(0, 4)]);
      appendSystemLog({
        level: "error",
        title: aborted ? "Recorded audio did not start" : "Start failed",
        message,
      });
      return false;
    }
    const contractForStart = buildMeetingContractPayload(meetingFocus);
    setActiveMeetingContract(contractForStart);
    setMeetingDiagnostics(null);
    setEnergyFrame(null);
    setMeetingFocusExpanded(false);
    setStatus("starting");
    setBriefVisible(false);
    appendSystemLog({
      level: "status",
      title: "Starting capture",
      message: `${fixtureWav ? "Recorded audio playback" : "Live microphone capture"} is starting. ${
        useSessionCompatibilityPath ? "Recorded test audio is starting." : "Live transcript and cards are ephemeral; audio will be queued for Review validation on stop."
      }`,
    });
    let response: Response;
    try {
      if (useSessionCompatibilityPath) {
        response = await fetch(`${API_BASE}/api/sessions/${id}/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            device_id: null,
            fixture_wav: fixtureWav,
            audio_source: fixtureWav ? "fixture" : "server_device",
            save_transcript: false,
            mic_tuning: micTuning,
            meeting_contract: contractForStart,
            web_context_enabled: webContextEnabledForStart(meetingFocus),
          }),
          signal: options.signal,
        });
      } else {
        response = await fetch(`${API_BASE}/api/live/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            device_id: null,
            audio_source: "server_device",
            title: currentSessionTitle.trim() || "Live meeting",
            mic_tuning: micTuning,
            meeting_contract: contractForStart,
            web_context_enabled: webContextEnabledForStart(meetingFocus),
          }),
          signal: options.signal,
        });
      }
    } catch (error) {
      const aborted = error instanceof DOMException && error.name === "AbortError";
      const message = aborted
        ? "Recorded audio start timed out before playback began."
        : "Start failed while contacting the local runtime.";
      setStatus("error");
      if (options.testRun) {
        setTestBusy("");
        activeTestPreparedRef.current = null;
        setTestPlaybackStartedAtMs(null);
        clearTestAutoStopTimer();
      }
      setActiveMeetingContract(null);
      setErrors((current) => [message, ...current.slice(0, 4)]);
      appendSystemLog({
        level: "error",
        title: aborted ? "Recorded audio did not start" : "Start failed",
        message,
      });
      return false;
    }
    if (!response.ok) {
      const payload = await response.json();
      setStatus("error");
      if (options.testRun) {
        setTestBusy("");
        activeTestPreparedRef.current = null;
        setTestPlaybackStartedAtMs(null);
        clearTestAutoStopTimer();
      }
      setActiveMeetingContract(null);
      setErrors((current) => [payload.detail ?? "Start failed", ...current]);
      appendSystemLog({
        level: "error",
        title: "Start failed",
        message: payload.detail ?? "Start failed",
        metadata: payload,
      });
      return false;
    }
    const startPayload = await response.json().catch(() => null);
    if (!useSessionCompatibilityPath) {
      const nextLiveId = String(startPayload?.live_id ?? "");
      if (nextLiveId) {
        setLiveId(nextLiveId);
        setSessionId(null);
        id = nextLiveId;
        attachEvents(nextLiveId, "live");
      }
      if (startPayload?.title) {
        setCurrentSessionTitle(String(startPayload.title));
      }
    }
    const normalizedContract = meetingContractFromPayload(startPayload?.meeting_contract);
    if (normalizedContract) {
      setActiveMeetingContract(normalizedContract);
    }
    if (options.testRun) {
      const startedAtMs = Date.now();
      testStartedAtRef.current = new Date(startedAtMs).toISOString();
      testSessionIdRef.current = id;
      activeTestPreparedRef.current = options.testRun;
      setTestPlaybackStartedAtMs(startedAtMs);
      setTestPlaybackBaseElapsedMs(0);
      setTestPlaybackNow(startedAtMs);
      scheduleTestAutoStop(options.testRun, id, options.testRun.duration_seconds * 1000);
      setTestBusy("playing");
    }
    setStatus((current) => current === "error" ? current : "listening");
    return true;
  }

  async function stop(sessionIdOverride?: string) {
    const stoppingLiveId = !sessionIdOverride && liveId && !activeTestPreparedRef.current ? liveId : null;
    const stoppedSessionId = sessionIdOverride ?? sessionId;
    if (!stoppingLiveId && !stoppedSessionId) return;
    clearTestAutoStopTimer();
    const snapshot = testSnapshotRef.current;
    let stopPayload: Record<string, unknown> | null = null;
    if (stoppingLiveId) {
      const response = await fetch(`${API_BASE}/api/live/${encodeURIComponent(stoppingLiveId)}/stop`, { method: "POST" });
      stopPayload = await response.json().catch(() => null);
      if (response.ok && stopPayload?.review_job_id) {
        appendSystemLog({
          level: "status",
          title: "Queued for Review",
          message: `Temporary audio queued as ${String(stopPayload.review_job_id)}${stopPayload.queue_position ? ` · position ${stopPayload.queue_position}` : ""}.`,
          metadata: stopPayload,
        });
        void refreshReviewJobs();
        void refreshReviewJob(String(stopPayload.review_job_id));
      }
    } else if (stoppedSessionId) {
      await fetch(`${API_BASE}/api/sessions/${stoppedSessionId}/stop`, { method: "POST" });
    }
    setStatus("stopped");
    appendSystemLog({
      level: "status",
      title: "Capture stopped",
      message: stoppingLiveId
        ? `${snapshot.transcript.length} heard lines cleared from Live; temporary audio is in the Review queue.`
        : `${snapshot.transcript.length} heard lines shown live; transcript text was not saved.`,
    });
    setBriefVisible(!stoppingLiveId);
    if (stoppingLiveId) {
      eventSourceRef.current?.close();
      setLiveId(null);
      setTranscript([]);
      setNotes([]);
      setRecall([]);
      setTranscriptEvents([]);
      setSidecarCards([]);
      setDismissedCardKeys(new Set());
      setExpandedContextCards(new Set());
      setErrors([]);
      setPipeline({
        queueDepth: 0,
        droppedWindows: 0,
        silentWindows: 0,
        asrEmptyWindows: 0,
        finalSegmentsReplaced: 0,
      });
      followLiveRef.current = true;
      setFollowLive(true);
    }
    setTestPlaybackStartedAtMs(null);
    setTestPlaybackBaseElapsedMs(0);
    await refreshSessionCatalog();
    const preparedForReport = activeTestPreparedRef.current ?? testPrepared;
    const testWasActive = Boolean(
      liveTestExpanded
      || playbackActive
      || playbackPaused
      || activeTestPreparedRef.current
      || testSessionIdRef.current
      || ["playing", "paused", "pausing", "resuming", "starting-playback"].includes(testBusy),
    );
    if (preparedForReport && stoppedSessionId) {
      await saveTestReport(stoppedSessionId, preparedForReport, testStartedAtRef.current ?? new Date().toISOString());
    } else if (testWasActive) {
      setTestBusy("");
      testStartedAtRef.current = null;
      testSessionIdRef.current = null;
      activeTestPreparedRef.current = null;
      clearTestAutoStopTimer();
    }
  }

  function clearTestAutoStopTimer() {
    if (testAutoStopTimerRef.current) {
      window.clearTimeout(testAutoStopTimerRef.current);
      testAutoStopTimerRef.current = null;
    }
  }

  function scheduleTestAutoStop(prepared: TestModePrepared, sessionIdToStop: string, remainingMs?: number) {
    clearTestAutoStopTimer();
    const durationMs = Math.max(1000, remainingMs ?? prepared.duration_seconds * 1000);
    testAutoStopTimerRef.current = window.setTimeout(() => {
      testAutoStopTimerRef.current = null;
      void stop(sessionIdToStop);
    }, durationMs + TEST_PLAYBACK_AUTOSTOP_GRACE_MS);
  }

  async function prepareTestAudio(sourcePathOverride?: string): Promise<TestModePrepared | null> {
    const sourcePath = (sourcePathOverride ?? testSourcePath).trim();
    if (!sourcePath) return null;
    const maxSeconds = testMaxSeconds.trim() ? Number(testMaxSeconds.trim()) : null;
    if (maxSeconds !== null && (!Number.isFinite(maxSeconds) || maxSeconds <= 0)) {
      setErrors((current) => ["Max seconds must be a positive number.", ...current.slice(0, 4)]);
      return null;
    }
    setTestBusy("preparing");
    setTestPlaybackStartedAtMs(null);
    setTestPlaybackBaseElapsedMs(0);
    setTestPrepared(null);
    setTestReport(null);
    const response = await fetch(`${API_BASE}/api/test-mode/audio/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_path: sourcePath,
        max_seconds: maxSeconds,
        expected_terms: parseExpectedTerms(testExpectedTerms),
      }),
    });
    const payload = await response.json();
    setTestBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Could not prepare recorded audio", ...current.slice(0, 4)]);
      return null;
    }
    const prepared = payload as TestModePrepared;
    setTestPrepared(prepared);
    setTestExpectedTerms(prepared.expected_terms.join(", "));
    appendSystemLog({
      level: "status",
      title: "Audio prepared",
      message: `${prepared.duration_seconds.toFixed(1)}s ready for playback. ${prepared.expected_terms.length} expected terms.`,
      metadata: prepared as unknown as Record<string, unknown>,
    });
    return prepared;
  }

  async function startTestPlayback() {
    if (!testPrepared) return;
    await startPreparedTestPlayback(testPrepared);
  }

  async function startPreparedTestPlayback(prepared: TestModePrepared): Promise<void> {
    setTestReport(null);
    setTestBusy("starting-playback");
    const controller = new AbortController();
    const startTimer = window.setTimeout(() => {
      controller.abort();
    }, TEST_PLAYBACK_START_TIMEOUT_MS);
    try {
      const started = await start({
        fixtureWav: prepared.fixture_wav,
        testRun: prepared,
        freshSession: true,
        signal: controller.signal,
      });
      if (!started) {
        setTestBusy("");
      }
    } finally {
      window.clearTimeout(startTimer);
    }
  }

  function currentTestPlaybackElapsedMs(): number {
    if (testBusy === "playing" && testPlaybackStartedAtMs !== null) {
      return Math.max(0, testPlaybackBaseElapsedMs + Date.now() - testPlaybackStartedAtMs);
    }
    return Math.max(0, testPlaybackBaseElapsedMs);
  }

  async function pauseTestPlayback() {
    if (!sessionId || testBusy !== "playing") {
      return;
    }
    const elapsedMs = currentTestPlaybackElapsedMs();
    setTestBusy("pausing");
    clearTestAutoStopTimer();
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/pause`, { method: "POST" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({})) as { detail?: string };
      setErrors((current) => [payload.detail ?? "Could not pause recorded audio playback.", ...current.slice(0, 4)]);
      setTestBusy("playing");
      const preparedForResume = activeTestPreparedRef.current ?? testPrepared;
      if (preparedForResume) {
        scheduleTestAutoStop(
          preparedForResume,
          sessionId,
          Math.max(1000, preparedForResume.duration_seconds * 1000 - elapsedMs),
        );
      }
      return;
    }
    setTestPlaybackBaseElapsedMs(elapsedMs);
    setTestPlaybackStartedAtMs(null);
    setTestBusy("paused");
    setStatus("paused");
  }

  async function resumeTestPlayback() {
    const activePrepared = activeTestPreparedRef.current ?? testPrepared;
    if (!sessionId || !activePrepared || testBusy !== "paused") {
      return;
    }
    setTestBusy("resuming");
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/resume`, { method: "POST" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({})) as { detail?: string };
      setErrors((current) => [payload.detail ?? "Could not resume recorded audio playback.", ...current.slice(0, 4)]);
      setTestBusy("paused");
      setStatus("paused");
      return;
    }
    const resumedAtMs = Date.now();
    setTestPlaybackStartedAtMs(resumedAtMs);
    setTestPlaybackNow(resumedAtMs);
    scheduleTestAutoStop(
      activePrepared,
      sessionId,
      Math.max(1000, activePrepared.duration_seconds * 1000 - testPlaybackBaseElapsedMs),
    );
    setTestBusy("playing");
    setStatus("listening");
  }

  async function restartTestPlayback() {
    const preparedForRestart = activeTestPreparedRef.current ?? testPrepared;
    const sessionToStop = testSessionIdRef.current ?? sessionId;
    if (!preparedForRestart) {
      return;
    }
    if (playbackActive && sessionToStop) {
      await stop(sessionToStop);
    }
    await startPreparedTestPlayback(preparedForRestart);
  }

  async function resetTestPlayback() {
    const sessionToStop = testSessionIdRef.current ?? sessionId;
    if (playbackActive && sessionToStop) {
      await stop(sessionToStop);
    }
    clearTestAutoStopTimer();
    testStartedAtRef.current = null;
    testSessionIdRef.current = null;
    activeTestPreparedRef.current = null;
    setTestBusy("");
    setTestPrepared(null);
    setTestReport(null);
    setTestPlaybackStartedAtMs(null);
    setTestPlaybackBaseElapsedMs(0);
    setLiveTestExpanded(false);
  }

  async function handleLiveTestFilePick(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file || liveTestUploadDisabled) {
      return;
    }
    setLiveTestExpanded(true);
    const path = await uploadInputFile(file, "test-audio");
    if (!path) {
      return;
    }
    setTestSourcePath(path);
    const prepared = await prepareTestAudio(path);
    if (prepared) {
      await startPreparedTestPlayback(prepared);
    }
  }

  async function saveTestReport(sessionIdForReport: string, prepared: TestModePrepared, startedAt: string) {
    const report = buildTestModeReport(sessionIdForReport, prepared, startedAt);
    const response = await fetch(`${API_BASE}/api/test-mode/runs/${prepared.run_id}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ report }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Could not save recorded audio report", ...current.slice(0, 4)]);
      setTestReport(report);
    } else {
      setTestReport({ ...report, report_path: String(payload.report_path ?? prepared.report_path) });
    }
    setTestBusy("");
    testStartedAtRef.current = null;
    testSessionIdRef.current = null;
    activeTestPreparedRef.current = null;
    setTestPlaybackStartedAtMs(null);
    clearTestAutoStopTimer();
  }

  function buildTestModeReport(sessionIdForReport: string, prepared: TestModePrepared, startedAt: string): TestModeReport {
    const snapshot = testSnapshotRef.current;
    const expectedTerms = prepared.expected_terms.length > 0
      ? prepared.expected_terms
      : parseExpectedTerms(snapshot.testExpectedTerms);
    const responseText = [
      snapshot.transcript.map((line) => line.text).join("\n"),
      snapshot.notes.map((note) => `${note.title}\n${note.body}`).join("\n"),
      snapshot.recall.map((hit) => hit.text).join("\n"),
      snapshot.sidecarCards.map((card) => `${card.title}\n${card.summary}\n${card.whyRelevant ?? ""}`).join("\n"),
    ].filter(Boolean).join("\n");
    return {
      run_id: prepared.run_id,
      session_id: sessionIdForReport,
      source_path: prepared.source_path,
      fixture_wav: prepared.fixture_wav,
      duration_seconds: prepared.duration_seconds,
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      status: snapshot.status,
      transcript_segments: snapshot.transcript.length,
      note_cards: snapshot.notes.length,
      recall_hits: snapshot.recall.length,
      queue_depth: snapshot.pipeline.queueDepth,
      dropped_windows: snapshot.pipeline.droppedWindows,
      reference_context_cards: snapshot.sidecarCards.filter((card) => !isCurrentMeetingCard(card) && !isManualCard(card)).length,
      error_count: snapshot.errors.length,
      expected_terms: expectedTerms,
      matched_expected_terms: matchExpectedTerms(responseText, expectedTerms),
      issues: snapshot.errors.length > 0 ? ["UI surfaced one or more errors during playback."] : [],
    };
  }

  async function addLibraryRoot() {
    if (!libraryPath.trim()) return;
    const response = await fetch(`${API_BASE}/api/library/roots`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: libraryPath.trim() }),
    });
    if (!response.ok) {
      const payload = await response.json();
      setErrors((current) => [payload.detail ?? "Could not add library root", ...current]);
      return;
    }
    appendSystemLog({
      level: "status",
      title: "File root added",
      message: libraryPath.trim(),
    });
    setLibraryPath("");
    await refreshLibraryRoots();
  }

  async function reindex() {
    setStatus("indexing");
    const response = await fetch(`${API_BASE}/api/library/reindex`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Reindex failed", ...current]);
      return;
    }
    setStatus(`indexed ${payload.chunks_indexed ?? 0} chunks`);
    appendSystemLog({
      level: "status",
      title: "Files indexed",
      message: `${payload.chunks_indexed ?? 0} chunks indexed from ${payload.roots_indexed ?? "configured"} roots.`,
      metadata: payload,
    });
    await refreshLibraryChunks(selectedLibrarySourcePath ?? undefined);
  }

  async function searchRecall() {
    const query = manualQuery.trim();
    if (!query || manualQueryBusy) return;
    setManualQueryBusy(true);
    setManualQuery("");
    appendTranscriptEvent(typedTranscriptEvent(query));

    const result = await fetchJson<ManualSidecarResponse>(`${API_BASE}/api/sidecar/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    setManualQueryBusy(false);
    if (!result.ok) {
      appendSystemLog({
        level: "error",
        title: "Sidecar query failed",
        message: String((result.payload as { detail?: string }).detail ?? "Sidecar query failed."),
        metadata: result.payload as unknown as Record<string, unknown>,
      });
      return;
    }
    const payload = result.payload;
    for (const rawCard of payload.cards ?? []) {
      const card = displayCardForSidecarPayload(rawCard, {
        id: `manual-${Date.now()}`,
        type: "sidecar_card",
        session_id: sessionId,
        at: Date.now() / 1000,
        payload: rawCard,
      });
      if (card) {
        appendSidecarCard({ ...card, explicitlyRequested: true });
      }
    }
    if (payload.web_skip_reason) {
      appendSystemLog({
        level: "status",
        title: "Web context skipped",
        message: payload.web_skip_reason.replace(/_/g, " "),
        metadata: payload as unknown as Record<string, unknown>,
      });
    }
  }

  async function testMicrophone() {
    setMicTestBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/microphone/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: null,
          audio_source: fixturePath.trim() && audioSource === "fixture" ? "fixture" : "server_device",
          fixture_wav: fixturePath.trim() && audioSource === "fixture" ? fixturePath.trim() : null,
          seconds: 3,
          mic_tuning: micTuning,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setErrors((current) => [payload.detail ?? "Microphone test failed", ...current]);
        return;
      }
      setMicTest(payload);
      if (payload.suggested_tuning && micTuning.auto_level) {
        setMicTuning(normalizeMicTuning(payload.suggested_tuning));
      }
      replaceMicPlaybackUrl(micTestPlaybackUrl(payload));
      appendSystemLog({
        level: payload.recommendation?.status === "good" ? "status" : "warning",
        title: payload.recommendation?.title ?? "Microphone check complete",
        message: payload.recommendation?.detail ?? "Mic test returned.",
        metadata: micTestLogMetadata(payload),
      });
    } catch (error) {
      pushError(error instanceof Error ? error.message : "Microphone test failed");
    } finally {
      setMicTestBusy(false);
    }
  }

  function replaceMicPlaybackUrl(nextUrl: string) {
    if (micPlaybackUrlRef.current) {
      URL.revokeObjectURL(micPlaybackUrlRef.current);
    }
    micPlaybackUrlRef.current = nextUrl || null;
    setMicPlaybackUrl(nextUrl);
  }

  async function startSpeakerEnrollment() {
    setSpeakerBusy("starting");
    const response = await fetch(`${API_BASE}/api/speaker/enrollments`, { method: "POST" });
    const payload = await response.json();
    setSpeakerBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Could not start speaker identity training", ...current]);
      return;
    }
    setSpeakerEnrollment(payload);
    setSpeakerStatus(payload.profile);
    setActivePage("tools");
    appendSystemLog({
      level: "status",
      title: "Speaker identity training started",
      message: "Record single-speaker samples. Raw audio is discarded after embeddings are created.",
      metadata: payload,
    });
  }

  async function recordSpeakerSample() {
    if (!speakerEnrollment) return;
    setSpeakerBusy("recording");
    const startedAt = Date.now();
    setSpeakerRecordingStartedAt(startedAt);
    setSpeakerRecordingNow(startedAt);
    try {
      const response = await fetch(`${API_BASE}/api/speaker/enrollments/${speakerEnrollment.id}/record`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: null,
          audio_source: fixturePath.trim() && audioSource === "fixture" ? "fixture" : "server_device",
          fixture_wav: fixturePath.trim() && audioSource === "fixture" ? fixturePath.trim() : null,
          mic_tuning: micTuning,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setErrors((current) => [payload.detail ?? "Speaker sample recording failed", ...current]);
        return;
      }
      mergeSpeakerSample(payload.sample);
      setSpeakerStatus(payload.profile);
      appendSystemLog({
        level: "status",
        title: "Speaker sample embedded",
        message: `${payload.sample.usable_speech_seconds?.toFixed?.(1) ?? "0"}s usable speech, quality ${Math.round((payload.sample.quality_score ?? 0) * 100)}%.`,
        metadata: payload,
      });
    } catch (error) {
      pushError(error instanceof Error ? error.message : "Speaker sample recording failed");
    } finally {
      setSpeakerBusy("");
      setSpeakerRecordingStartedAt(null);
    }
  }

  async function finalizeSpeakerEnrollment() {
    if (!speakerEnrollment) return;
    setSpeakerBusy("finalizing");
    const response = await fetch(`${API_BASE}/api/speaker/enrollments/${speakerEnrollment.id}/finalize`, { method: "POST" });
    const payload = await response.json();
    setSpeakerBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Speaker enrollment finalization failed", ...current]);
      return;
    }
    setSpeakerStatus(payload.profile);
    setSpeakerEnrollment((current) => current ? { ...current, status: "completed", profile: payload.profile } : current);
    appendSystemLog({
      level: "status",
      title: "Speaker identity ready",
      message: `${speakerEnrollmentLabel} threshold set to ${payload.threshold}.`,
      metadata: payload,
    });
  }

  async function recalibrateSpeakerProfile() {
    setSpeakerBusy("recalibrating");
    const response = await fetch(`${API_BASE}/api/speaker/profiles/self/recalibrate`, { method: "POST" });
    const payload = await response.json();
    setSpeakerBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Speaker recalibration failed", ...current]);
      return;
    }
    setSpeakerStatus(payload.profile);
    appendSystemLog({
      level: "status",
      title: "Speaker threshold recalibrated",
      message: `New threshold: ${payload.threshold}.`,
      metadata: payload,
    });
  }

  async function resetSpeakerProfile() {
    setSpeakerBusy("resetting");
    const response = await fetch(`${API_BASE}/api/speaker/profiles/self/reset`, { method: "POST" });
    const payload = await response.json();
    setSpeakerBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Speaker reset failed", ...current]);
      return;
    }
    setSpeakerEnrollment(null);
    setSpeakerStatus(payload.profile);
    appendSystemLog({
      level: "status",
      title: "Speaker identity reset",
      message: "Deleted learned embeddings and cleared calibration.",
      metadata: payload,
    });
  }

  async function submitSpeakerFeedback(
    segment: NonNullable<SpeakerStatus["recent_segments"]>[number] | null,
    newLabel: string,
    feedbackType: string,
  ) {
    const response = await fetch(`${API_BASE}/api/speaker/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: segment?.session_id ?? liveId ?? sessionId ?? "manual-review",
        segment_id: segment?.segment_id ?? segment?.id ?? null,
        old_label: segment?.display_speaker_label ?? "",
        new_label: newLabel,
        feedback_type: feedbackType,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Speaker feedback failed", ...current]);
      return;
    }
    setSpeakerStatus(payload.profile);
    appendSystemLog({
      level: "status",
      title: "Speaker feedback saved",
      message: `${feedbackType.replace(/_/g, " ")} → ${newLabel}`,
      metadata: payload,
    });
  }

  function mergeSpeakerSample(sample: SpeakerEnrollmentSample) {
    setSpeakerEnrollment((current) => {
      if (!current) return current;
      return {
        ...current,
        samples: [sample, ...current.samples.filter((candidate) => candidate.id !== sample.id)],
      };
    });
  }

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        status={status}
        asrBackendLabel={asrBackendLabel}
        gpuFreeLabel={gpuFreeLabel}
        retentionLabel={captureActive ? activeRetentionStatus : sessionRetentionStatus}
        onNavigate={setActivePage}
      />
      <div className="app-main">
      <header className="top-bar">
        <div className="top-runtime-context" aria-label="Runtime summary">
          <span>Runtime</span>
          <strong>{topRuntimeSummary}</strong>
          <small>{topRuntimeDetail}</small>
        </div>
        <div className="top-actions">
          <span className={`status-pill ${statusTone}`} aria-label="Runtime status">
            <span aria-hidden="true" />
            {status}
          </span>
          <span className="retention-pill" aria-label="Live retention status">
            {captureActive ? "Review handoff armed" : sessionRetentionStatus}
          </span>
          {headerQueueLabel && <span className="queue-pill" aria-label="Capture queue status">{headerQueueLabel}</span>}
          <button className="primary" onClick={() => start()} disabled={captureStartDisabled}>
            {captureButtonLabel}
          </button>
          {!captureStopDisabled && (
            <button className="danger" onClick={() => stop()}>
              {liveId ? "Stop & Queue" : "Stop"}
            </button>
          )}
        </div>
      </header>

      {activePage === "live" && (
        <LiveMeetingWorkbench
          mode={liveWorkbenchMode}
          active={captureActive || captureStarting || playbackActive || playbackPaused}
          transcriptPane={(
            <LiveTranscriptPane
              setupHeader={(
                <section className="live-cockpit-header" aria-label="Live cockpit setup">
                  <SessionSelectorBar
                    title={currentSessionTitle}
                    status={status}
                    disabled={captureActive || captureStarting}
                    onTitleChange={setCurrentSessionTitle}
                  />
                  <div className={`cockpit-control-row ${testModeVisible ? "with-test" : ""}`}>
                    <CaptureControlBar
                      statusSummary={captureStatusSummary}
                      warning={missingServerDevice ? serverMicStatus : ""}
                    />
                    <MeetingFocusControl
                      value={meetingFocus}
                      expanded={meetingFocusExpanded}
                      activeContract={activeMeetingContract}
                      captureActive={captureActive || captureStarting}
                      webContextConfigured={gpu.web_context_configured === true}
                      webContextGlobalEnabled={gpu.web_context_enabled === true}
                      onChange={setMeetingFocus}
                      onEdit={() => setMeetingFocusExpanded(true)}
                      onDone={() => setMeetingFocusExpanded(false)}
                    />
                    <ReplayTestDrawer
                      visible={testModeVisible}
                      expanded={liveTestExpanded}
                      stateDetail={liveTestStateDetail}
                      stateLabel={liveTestStateLabel}
                      stateStatus={liveTestStatus}
                      canReset={canResetLiveTest}
                      resetDisabled={fileUploadBusy === "test-audio" || testBusy === "preparing" || testBusy === "starting-playback"}
                      toggleDisabled={liveTestToggleDisabled}
                      testSourcePath={testSourcePath}
                      fileBusy={fileUploadBusy === "test-audio"}
                      uploadDisabled={liveTestUploadDisabled}
                      maxSeconds={testMaxSeconds}
                      expectedTerms={testExpectedTerms}
                      configDisabled={liveTestConfigDisabled}
                      prepared={testPrepared}
                      report={testReport}
                      playbackActive={playbackActive}
                      playbackPaused={playbackPaused}
                      testBusy={testBusy}
                      captureActive={captureActive}
                      captureStarting={captureStarting}
                      playbackElapsedSeconds={testPlaybackElapsedSeconds}
                      onToggle={() => setLiveTestExpanded((current) => !current)}
                      onReset={() => void resetTestPlayback()}
                      onFilePick={handleLiveTestFilePick}
                      onMaxSecondsChange={setTestMaxSeconds}
                      onExpectedTermsChange={setTestExpectedTerms}
                      onStartPlayback={startTestPlayback}
                      onPausePlayback={pauseTestPlayback}
                      onResumePlayback={resumeTestPlayback}
                      onRestartPlayback={restartTestPlayback}
                      onStopPlayback={() => stop(testSessionIdRef.current ?? sessionId ?? undefined)}
                      stopPlaybackDisabled={!playbackActive || !sessionId}
                    />
                  </div>
                </section>
              )}
              runtimeStrip={(
                <SessionRuntimeStrip
                  mode={liveWorkbenchMode}
                  items={liveWorkbenchMode === "live"
                    ? [
                        {
                          label: "State",
                          value: captureActive || captureStarting ? "Live" : "Ready",
                          tone: captureActive || captureStarting ? "good" : "normal",
                        },
                        { label: "Input", value: captureModeLabel },
                        ...(pipeline.queueDepth > 0
                          ? [{ label: "Queue", value: String(pipeline.queueDepth), tone: "warning" as const }]
                          : []),
                        { label: "Sidecar", value: `${currentMeetingCards.length} returns` },
                      ]
                    : [
                        {
                          label: "State",
                          value: liveWorkbenchMode === "ended"
                            ? "Ended"
                            : liveWorkbenchMode === "replay"
                              ? "Replay"
                              : liveWorkbenchMode === "dev-test"
                                ? "Test"
                                : captureActive || captureStarting ? "Live" : "Ready",
                          tone: captureActive || captureStarting ? "good" : "normal",
                        },
                        { label: "Input", value: captureModeLabel },
                        { label: "ASR", value: asrTimingLabel },
                        { label: "Queue", value: String(pipeline.queueDepth), tone: pipeline.queueDepth > 0 ? "warning" : "normal" },
                        { label: "Sidecar", value: `${currentMeetingCards.length} returns` },
                      ]}
                />
              )}
              transcriptCount={transcript.length}
              currentCardCount={currentMeetingCards.length}
              asrDeviceStatusLabel={asrDeviceStatusLabel}
              errors={errors}
              playbackCue={(testBusy === "playing" || playbackPaused) && testPrepared ? (
                <LivePlaybackCue
                  durationSeconds={testPrepared.duration_seconds}
                  elapsedSeconds={testPlaybackElapsedSeconds}
                  progress={testPlaybackProgress}
                  remainingSeconds={testPlaybackRemainingSeconds}
                  message={testPlaybackMessage}
                  paused={playbackPaused}
                />
              ) : null}
              rows={liveFieldRows}
              expandedCards={expandedContextCards}
              highlightedSourceIds={highlightedSourceIds}
              showDebugMetadata={showDebugMetadata}
              active={captureActive || captureStarting}
              followLive={followLive}
              unseenItems={unseenItems}
              manualQuery={manualQuery}
              manualQueryBusy={manualQueryBusy}
              manualSourceCards={manualSourceCards}
              transcriptScrollRef={transcriptScrollRef}
              liveEndRef={liveEndRef}
              onScroll={handleFieldScroll}
              onJumpToLive={() => scrollFieldToLive("smooth")}
              onToggleCard={toggleExpandedContextCard}
              onJumpToEvidence={jumpToEvidence}
              onPin={togglePinnedCard}
              onDismiss={dismissCard}
              onCopy={copyCardSuggestion}
              onManualQueryChange={setManualQuery}
              onManualSearch={searchRecall}
            />
          )}
          sidecarPane={liveWorkbenchMode === "ended" ? (
            <PostCallMeetingOutput
              groups={meetingOutputGroups}
              usefulCount={currentMeetingCards.length}
              rawCount={sidecarCards.length}
              qualityMetrics={qualityMetrics}
              meetingDiagnostics={meetingDiagnostics}
              qualityGateActive={gpu.sidecar_quality_gate_enabled !== false}
              loading={manualQueryBusy}
              emptyTitle={contextEmptyTitle}
              emptyBody={contextEmptyBody}
              contextCards={visibleOtherContextCards}
              manualCards={manualSourceCards}
              showAll={showAllContext}
              canShowMore={
                currentMeetingCards.length > DEFAULT_CONTEXT_CARD_LIMIT
                || otherContextCards.length > DEFAULT_CONTEXT_CARD_LIMIT
              }
              onToggleShowAll={() => setShowAllContext((value) => !value)}
              expandedCards={expandedContextCards}
              showDebugMetadata={showDebugMetadata}
              onToggle={toggleExpandedContextCard}
              onJumpToEvidence={jumpToEvidence}
              onPin={togglePinnedCard}
              onDismiss={dismissCard}
              onCopy={copyCardSuggestion}
              contract={activeMeetingContract}
              energyFrame={energyFrame}
              showBrief={briefVisible}
              briefMarkdown={consultingBriefMarkdown}
              currentCount={currentMeetingCards.length}
              contextCount={otherContextCards.length}
              onCopyBrief={copyConsultingBrief}
            />
          ) : (
            <LiveModelReturnsPane
              mode={liveWorkbenchMode}
              status={status}
              gpu={gpu}
              pipeline={pipeline}
              transcriptCount={transcript.length}
              currentCards={currentMeetingCards}
              contextCards={visibleOtherContextCards}
              manualCards={manualSourceCards}
              groups={meetingOutputGroups}
              qualityMetrics={qualityMetrics}
              meetingDiagnostics={meetingDiagnostics}
              qualityGateActive={gpu.sidecar_quality_gate_enabled !== false}
              rawCount={sidecarCards.length}
              systemLogs={systemLogs}
              contract={activeMeetingContract}
              energyFrame={energyFrame}
              showAll={showAllContext}
              canShowMore={
                currentMeetingCards.length > DEFAULT_CONTEXT_CARD_LIMIT
                || otherContextCards.length > DEFAULT_CONTEXT_CARD_LIMIT
              }
              onToggleShowAll={() => setShowAllContext((value) => !value)}
              expandedCards={expandedContextCards}
              showDebugMetadata={showDebugMetadata}
              onToggle={toggleExpandedContextCard}
              onJumpToEvidence={jumpToEvidence}
              onPin={togglePinnedCard}
              onDismiss={dismissCard}
              onCopy={copyCardSuggestion}
            />
          )}
        />
      )}

      {activePage === "review" && (
        <ReviewPage
          title={reviewTitle}
          job={reviewJob}
          jobs={reviewJobs}
          busy={reviewBusy}
          error={reviewError}
          transcriptEvents={reviewTranscriptEvents}
          cards={reviewCards}
          capture={reviewCapture}
          groups={reviewMeetingGroups}
          expandedCards={expandedContextCards}
          highlightedSourceIds={highlightedSourceIds}
          showDebugMetadata={showDebugMetadata}
          transcriptScrollRef={reviewTranscriptScrollRef}
          onTitleChange={setReviewTitle}
          onFilePick={handleReviewFilePick}
          onNewReview={startNewReview}
          onRecoverLatest={() => void refreshReviewJobs()}
          onCancel={() => void cancelReviewJob()}
          onApprove={() => void approveReviewJob()}
          onDiscard={() => void discardReviewJob()}
          onSelectJob={(job) => void refreshReviewJob(reviewJobId(job))}
          onToggleCard={toggleExpandedContextCard}
          onJumpToEvidence={(card) => focusReviewEvidenceIds(card.sourceSegmentIds ?? [])}
          onJumpToEvidenceIds={focusReviewEvidenceIds}
          onCopyCard={copyCardSuggestion}
          onCopyTranscript={() => navigator.clipboard?.writeText((reviewJob?.clean_segments ?? []).map((segment) => segment.text).join("\n"))}
          onCopyCapture={() => navigator.clipboard?.writeText(reviewCaptureMarkdown)}
          onOpenSession={() => {
            if (!reviewJob?.session_id) {
              return;
            }
            setActivePage("sessions");
            void loadPastSession(reviewJob.session_id);
          }}
        />
      )}

      {activePage === "sessions" && (
        <SessionsPage
          sessions={sessionCatalog}
          selectedSession={selectedPastSession}
          selectedSessionId={selectedPastSessionId}
          search={sessionSearch}
          showAll={showAllSessions}
          busy={sessionsBusy}
          titleDraft={sessionTitleDraft}
          highlightedSourceIds={highlightedSourceIds}
          onSearchChange={setSessionSearch}
          onShowAllChange={setShowAllSessions}
          onRefresh={refreshSessionCatalog}
          onSelect={loadPastSession}
          onTitleDraftChange={setSessionTitleDraft}
          onSaveTitle={updatePastSessionTitle}
          onHighlightSources={setHighlightedSourceIds}
          onCreateSavedSession={() => {
            setActivePage("review");
          }}
          onGoLive={() => setActivePage("live")}
        />
      )}

      {activePage === "models" && (
        <ModelsPage
          gpu={gpu}
          pipeline={pipeline}
          gpuLabel={gpuLabel}
          vramPercent={vramPercent}
          gpuFreeLabel={gpuFreeLabel}
          asrBackendLabel={asrBackendLabel}
          asrModelLabel={asrModelLabel}
          asrCadenceLabel={asrCadenceLabel}
          asrTimingLabel={asrTimingLabel}
          ollamaChatHostLabel={ollamaChatHostLabel}
          ollamaEmbedHostLabel={ollamaEmbedHostLabel}
          ollamaChatReachabilityLabel={ollamaChatReachabilityLabel}
          ollamaEmbedReachabilityLabel={ollamaEmbedReachabilityLabel}
          ollamaModels={ollamaModels}
          ollamaModelBusy={ollamaModelBusy}
          ollamaModelNotice={ollamaModelNotice}
          onRefresh={refreshGpu}
          onRefreshOllamaModels={refreshOllamaModels}
          onSelectOllamaChatModel={selectOllamaChatModel}
        />
      )}

      {activePage === "references" && (
        <ReferencesPage
          libraryPath={libraryPath}
          libraryRoots={libraryRoots}
          librarySources={librarySources}
          libraryChunks={libraryChunks}
          selectedLibrarySourcePath={selectedLibrarySourcePath}
          memoryQuery={memoryQuery}
          memoryBusy={memoryBusy}
          webContextEnabled={gpu.web_context_enabled === true}
          webContextConfigured={gpu.web_context_configured === true}
          onLibraryPathChange={setLibraryPath}
          onAddLibraryRoot={addLibraryRoot}
          onReindexLibrary={reindex}
          onSelectLibrarySource={(sourcePath) => {
            setSelectedLibrarySourcePath(sourcePath);
            refreshLibraryChunks(sourcePath);
          }}
          onMemoryQueryChange={setMemoryQuery}
          onSearch={searchMemory}
          onRefresh={refreshMemoryPage}
        />
      )}

      {activePage === "tools" && (
        <main className="page tools-page" aria-label="Tools">
          <div className="utility-header page-header">
            <div>
              <p className="label">Tools</p>
              <h1>Tools</h1>
              <span>Calibrate input, train BP speaker identity, run test audio, inspect logs, and adjust appearance.</span>
            </div>
          </div>
          <nav className="utility-tabs" aria-label="Tool sections">
            {([
              ["voice", "Voice & Input"],
              ["speaker", "Speaker Identity"],
              ["test", "Test Mode"],
              ["debug", "Logs & Debug"],
              ["appearance", "Appearance"],
            ] as [ToolView, string][]).map(([view, label]) => (
              <button
                key={view}
                type="button"
                className={toolView === view ? "active" : ""}
                aria-current={toolView === view ? "page" : undefined}
                onClick={() => setToolView(view)}
              >
                {label}
              </button>
            ))}
          </nav>

          {toolView === "voice" && (
          <section className="utility-section" id="capture" aria-label="Voice & Input">
            <div className="utility-section-heading">
              <h3>Voice & Input</h3>
              <span>{devices.length || "No"} devices</span>
            </div>
            <div className={`tool-readiness-card ${inputReadinessTitle === "Ready" ? "ready" : "warning"}`} aria-label="Input readiness">
              <div>
                <p className="label">Input readiness</p>
                <strong>{inputReadinessTitle}</strong>
                <span>{inputReadinessDetail}</span>
              </div>
              <button className="primary subtle" onClick={testMicrophone} disabled={micTestBusy || audioSource !== "server_device" || missingServerDevice}>
                {micTestBusy ? "Testing..." : "Test Mic"}
              </button>
            </div>
            {ENABLE_FIXTURE_AUDIO && (
              <label className="field field-wide">
                <span>Audio source</span>
                <select
                  aria-label="Audio source"
                  value={audioSource}
                  onChange={(event) => setAudioSource(event.target.value as AudioSource)}
                >
                  <option value="server_device">Server microphone</option>
                  <option value="fixture">Fixture WAV</option>
                </select>
              </label>
            )}
            <div className={`server-mic-card ${missingServerDevice ? "warning" : "ready"}`} aria-label="Server microphone">
              <div>
                <p className="label">Server microphone</p>
                <strong>{serverMicStatus}</strong>
                <span>{deviceNotice || "Auto-selected on this computer. Browser microphones are disabled."}</span>
              </div>
              <button className="secondary" onClick={refreshDevices}>Refresh</button>
            </div>
            <div className="compact-tool mic-tuning" role="region" aria-label="Guided mic tuning">
              <div className="tuning-header">
                <p className="label">Guided Auto</p>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={micTuning.auto_level}
                    onChange={(event) => setMicTuning((current) => ({ ...current, auto_level: event.target.checked }))}
                  />
                  <span>Auto Level</span>
                </label>
              </div>
              <label className="field field-wide">
                <span>Input Boost</span>
                <input
                  aria-label="Input Boost"
                  type="range"
                  min="-12"
                  max="12"
                  step="1"
                  value={micTuning.input_gain_db}
                  onChange={(event) => setMicTuning((current) => normalizeMicTuning({ ...current, input_gain_db: Number(event.target.value) }))}
                />
                <small>{micTuning.input_gain_db > 0 ? "+" : ""}{micTuning.input_gain_db.toFixed(0)} dB</small>
              </label>
              <div className="mode-control compact sensitivity-control" role="group" aria-label="Speech Sensitivity">
                {(["quiet", "normal", "noisy"] as SpeechSensitivity[]).map((level) => (
                  <button
                    key={level}
                    type="button"
                    aria-pressed={micTuning.speech_sensitivity === level}
                    className={micTuning.speech_sensitivity === level ? "active" : ""}
                    onClick={() => setMicTuning((current) => ({ ...current, speech_sensitivity: level }))}
                  >
                    <span>{capitalize(level)}</span>
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button className="secondary" onClick={() => setMicTuning(defaultMicTuning())}>Reset Auto</button>
                {micTest?.suggested_tuning && (
                  <button className="secondary" onClick={() => setMicTuning(normalizeMicTuning(micTest.suggested_tuning ?? undefined))}>
                    Apply Suggested
                  </button>
                )}
              </div>
              {micTuning.input_gain_db >= 10 && (
                <p className="tool-note warning-text">High boost can clip loud rooms or speaker audio. Run Test Mic before relying on it.</p>
              )}
              {micTest?.suggested_tuning?.reason && <p className="tool-note">{micTest.suggested_tuning.reason}</p>}
            </div>
            {deviceNotice && <p className={`tool-note ${missingServerDevice ? "warning-text" : ""}`} role="status">{deviceNotice}</p>}
            <div className={`mic-test-card ${micTestTone}`} aria-label="Microphone check">
              <div>
                <strong>{micTest?.recommendation.title ?? "Mic check"}</strong>
                <span>
                  {micTest
                    ? micTest.recommendation.detail
                    : "Run this before speaker training to confirm level, clipping, and usable single-speaker audio."}
                </span>
              </div>
              {micTest && (
                <div className="mic-meter" aria-label="Microphone test result">
                  <div><span style={{ width: `${Math.min(100, micPeakPercent)}%` }} /></div>
                  <small>{micUsableSeconds.toFixed(1)}s speech · quality {micQualityPercent}% · peak {micPeakPercent}%</small>
                </div>
              )}
              {micPlaybackUrl && (
                <div className="mic-playback" aria-label="Microphone test playback">
                  <strong>Recorded preview</strong>
                  <audio controls src={micPlaybackUrl} preload="metadata" />
                  <small>Raw audio is not saved.</small>
                </div>
              )}
              {micTest?.quality.issues.length ? (
                <div className="chip-row">
                  {micTest.quality.issues.map((issue) => <span key={issue} className="chip warning">{issue}</span>)}
                </div>
              ) : null}
            </div>
            {ENABLE_FIXTURE_AUDIO && (
              <div className="compact-tool" role="region" aria-label="Fixture audio test controls">
                <div className="field">
                  <span>Fixture WAV path</span>
                  <div className="path-picker">
                    <input
                      aria-label="Fixture WAV path"
                      value={fixturePath}
                      onChange={(event) => setFixturePath(event.target.value)}
                      placeholder="/absolute/path/to/mono-16khz.wav"
                    />
                    <FilePickAction
                      label="Browse fixture WAV"
                      actionLabel="Upload WAV"
                      accept=".wav,audio/wav,audio/x-wav"
                      path={fixturePath}
                      busy={fileUploadBusy === "fixture"}
                      disabled={Boolean(fileUploadBusy)}
                      onChange={(event) => handleInputFilePick(event, "fixture", setFixturePath)}
                    />
                  </div>
                </div>
              </div>
            )}
          </section>
          )}

          {toolView === "test" && testModeVisible && (
            <section className="utility-section" id="test-mode" role="region" aria-label="Recorded audio test">
              <div className="utility-section-heading">
                <h3>Recorded Audio Test</h3>
                <span>{testBusy || (testPrepared ? "ready" : "idle")}</span>
              </div>
              <div className="test-mode-grid">
                <div className="field field-wide">
                  <span>Source audio path</span>
                  <div className="path-picker">
                    <input
                      aria-label="Source audio path"
                      value={testSourcePath}
                      onChange={(event) => setTestSourcePath(event.target.value)}
                      placeholder="/absolute/path/to/recording.m4a"
                    />
                    <FilePickAction
                      label="Browse source audio"
                      actionLabel="Upload audio"
                      accept={TEST_AUDIO_ACCEPT}
                      path={testSourcePath}
                      busy={fileUploadBusy === "test-audio"}
                      disabled={Boolean(fileUploadBusy) || testBusy === "preparing" || captureActive || captureStarting}
                      onChange={(event) => handleInputFilePick(event, "test-audio", setTestSourcePath)}
                    />
                  </div>
                </div>
                <label className="field">
                  <span>Max seconds</span>
                  <input
                    aria-label="Max seconds"
                    inputMode="decimal"
                    value={testMaxSeconds}
                    onChange={(event) => setTestMaxSeconds(event.target.value)}
                    placeholder="optional"
                  />
                </label>
                <label className="field field-wide">
                  <span>Expected terms</span>
                  <input
                    aria-label="Expected terms"
                    value={testExpectedTerms}
                    onChange={(event) => setTestExpectedTerms(event.target.value)}
                    placeholder="Comma-separated validation terms"
                  />
                </label>
              </div>
              <div className="button-row">
                <button className="secondary" onClick={() => prepareTestAudio()} disabled={testBusy === "preparing" || captureActive || captureStarting}>
                  {prepareButtonLabel}
                </button>
                <button className="primary subtle" onClick={startTestPlayback} disabled={!testPrepared || Boolean(testBusy) || captureActive || captureStarting}>
                  {playbackButtonLabel}
                </button>
              </div>
              {testPrepared && (
                <div className="test-mode-summary" aria-label="Prepared audio">
                  <span><strong>{testPrepared.duration_seconds.toFixed(1)}s</strong> duration</span>
                  <span><strong>{testPrepared.expected_terms.length}</strong> expected terms</span>
                  {testBusy === "playing" && <span>Auto-stops at end</span>}
                  <small>{testPrepared.fixture_wav}</small>
                </div>
              )}
              {testReport && (
                <div className="test-mode-summary" aria-label="Recorded audio report">
                  <span><strong>{testReport.transcript_segments}</strong> heard lines</span>
                  <span><strong>{testReport.matched_expected_terms.length}/{testReport.expected_terms.length}</strong> terms matched</span>
                  <small>{testReport.report_path ?? testPrepared?.report_path}</small>
                </div>
              )}
            </section>
          )}

          {toolView === "test" && !testModeVisible && (
            <section className="utility-section" aria-label="Recorded audio test unavailable">
              <div className="empty-panel">
                <h2>Test mode is off</h2>
                <p>Recorded playback controls appear here when test mode is enabled in the local runtime.</p>
              </div>
            </section>
          )}

          {toolView === "speaker" && (
          <section className="utility-section" id="speaker-identity" aria-label="Speaker identity" role="region">
            <div className="utility-section-heading">
              <h3>Speaker Identity</h3>
              <span>{speakerStatusLabel}</span>
            </div>
            <div className={`tool-readiness-card ${speakerStatus?.ready ? "ready" : "warning"}`} aria-label="BP label readiness">
              <div>
                <p className="label">BP label readiness</p>
                <strong>{speakerReadinessTitle}</strong>
                <span>{speakerReadinessDetail}</span>
              </div>
              {!speakerEnrollment && !speakerStatus?.ready && (
                <button
                  className="primary subtle"
                  onClick={startSpeakerEnrollment}
                  disabled={Boolean(speakerBusy) || speakerStatus?.backend.available === false || missingServerDevice}
                >
                  {speakerTrainingButtonLabel}
                </button>
              )}
            </div>

            <div className={`speaker-training-card ${speakerStage}`} aria-label="Speaker training next step">
              <div className="speaker-training-status">
                <span>Next</span>
                <strong>{speakerNextActionLabel}</strong>
                <small>{speakerReadinessDetail}</small>
              </div>
              <div className="speaker-progress-meter" aria-label={`Speaker identity progress ${speakerProgress}`}>
                <div>
                  <span style={{ width: `${speakerProgressPercent}%` }} />
                </div>
                <small>
                  {speakerUsableSpeech.toFixed(1)}s usable · {speakerCanFinalize ? "ready to finish" : `${speakerNeededSpeech.toFixed(1)}s needed`}
                </small>
              </div>
              <div className="speaker-step-list" aria-label="Speaker training steps">
                <span className={speakerStage !== "setup" && speakerStage !== "start" ? "done" : speakerStage === "start" ? "active" : ""}>1 Start</span>
                <span className={speakerStage === "record" ? "active" : speakerStage === "finish" || speakerStage === "ready" ? "done" : ""}>2 Record</span>
                <span className={speakerStage === "finish" ? "active" : speakerStage === "ready" ? "done" : ""}>3 Finish</span>
              </div>
            </div>

            {speakerEnrollment && !speakerStatus?.ready && (
              <div className="speaker-record-flow">
                <div className="speaker-record-prompt" aria-label="Speaker recording prompt">
                  <strong>Record only {speakerEnrollmentLabel}</strong>
                  <span>Use your normal mic position. Speak naturally for the full timer. Stop if another voice enters; this verifies BP, not separate meeting speakers.</span>
                </div>
                <div className="speaker-training-prompts" aria-label="Suggested speaker training prompts">
                  <div className="speaker-training-prompts-head">
                    <strong>Say one of these</strong>
                    <span>Use a different line for each sample.</span>
                  </div>
                  <ol>
                    {SPEAKER_TRAINING_PROMPTS.map((prompt) => (
                      <li key={prompt}>{prompt}</li>
                    ))}
                  </ol>
                </div>
                {speakerBusy === "recording" && (
                  <div className="voice-recording" aria-label="Speaker recording progress">
                    <div><span style={{ width: `${speakerRecordingProgress}%` }} /></div>
                    <small>
                      {Math.min(speakerSampleSeconds, speakerRecordingElapsedSeconds).toFixed(1)}s / {speakerSampleSeconds.toFixed(0)}s
                    </small>
                  </div>
                )}
                <div className="button-row">
                  <button
                    className={speakerCanFinalize ? "secondary" : "primary"}
                    onClick={recordSpeakerSample}
                    disabled={speakerBusy === "recording" || speakerStatus?.backend.available === false || missingServerDevice}
                  >
                    {speakerRecordButtonLabel}
                  </button>
                  <button
                    className="primary subtle"
                    onClick={finalizeSpeakerEnrollment}
                    disabled={Boolean(speakerBusy) || speakerUsableSpeech < speakerMinimumSpeech}
                  >
                    {speakerBusy === "finalizing" ? "Finalizing..." : `Finish and Use ${speakerEnrollmentLabel}`}
                  </button>
                </div>
                {speakerEnrollment.samples.length > 0 && (
                  <div className="speaker-sample-list" aria-label="Speaker enrollment samples">
                    <p className="label">{speakerEnrollment.samples.length} sample{speakerEnrollment.samples.length === 1 ? "" : "s"} recorded</p>
                    <div className="chip-row">
                      {speakerEnrollment.samples.slice(0, 6).map((sample) => (
                        <span key={sample.id} className={`chip ${sample.issues.length ? "warning" : ""}`}>
                          {sample.usable_speech_seconds.toFixed(1)}s usable / {sample.duration_seconds.toFixed(1)}s recorded · {Math.round(sample.quality_score * 100)}%{sample.issues.length ? " · check" : ""}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {speakerStatus?.ready && (
              <div className="speaker-ready-card" aria-label="Speaker identity ready">
                <strong>{speakerEnrollmentLabel} transcript label is ready</strong>
                <span>Live and Review can label high-confidence {speakerEnrollmentLabel} speech; every non-match stays Other speaker or Unknown speaker.</span>
              </div>
            )}

            <details className="speaker-maintenance">
              <summary>Advanced speaker controls</summary>
              <div className="speaker-stats" aria-label="Speaker profile metrics">
                <span><strong>{speakerStatus?.embedding_count ?? 0}</strong> embeddings</span>
                <span><strong>{Math.round((speakerStatus?.quality_score ?? 0) * 100)}%</strong> quality</span>
                <span><strong>{speakerStatus?.profile.threshold?.toFixed(2) ?? "none"}</strong> threshold</span>
              </div>
              <div className="button-row">
                <button
                  className="secondary"
                  onClick={startSpeakerEnrollment}
                  disabled={Boolean(speakerBusy) || speakerStatus?.backend.available === false || missingServerDevice}
                >
                  {speakerUsableSpeech > 0 ? "Start Over" : `Set Up ${speakerEnrollmentLabel} Label`}
                </button>
                <button className="secondary" onClick={recalibrateSpeakerProfile} disabled={Boolean(speakerBusy) || (speakerStatus?.embedding_count ?? 0) < 2}>
                  Recalibrate Threshold
                </button>
                <button className="secondary danger-subtle" onClick={resetSpeakerProfile} disabled={Boolean(speakerBusy)}>
                  Delete Learned Embeddings
                </button>
              </div>
            </details>

            {(speakerStatus?.recent_segments ?? []).length > 0 && (
              <details className="voice-preview">
                <summary>Review speaker recognition</summary>
                <div className="speaker-review-list">
                  {speakerStatus?.recent_segments?.slice(0, 6).map((segment) => (
                    <div key={segment.id} className="speaker-review-row">
                      <strong>{segment.display_speaker_label ?? "Unknown"}</strong>
                      <span>{Math.round((segment.match_confidence ?? 0) * 100)}%</span>
                      <p>{segment.transcript_text ?? ""}</p>
                      <div className="speaker-feedback-controls">
                        <button type="button" className="secondary" onClick={() => submitSpeakerFeedback(segment, speakerEnrollmentLabel, "this_was_me")}>
                          This was me
                        </button>
                        <button type="button" className="secondary" onClick={() => submitSpeakerFeedback(segment, "Unknown speaker", "this_was_not_me")}>
                          This was not me
                        </button>
                        <button type="button" className="secondary" onClick={() => submitSpeakerFeedback(segment, segment.display_speaker_label ?? "Unknown speaker", "bad_training_data")}>
                          Mark bad training data
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </section>
          )}

          {toolView === "debug" && (
          <section className="utility-section" aria-label="Logs & Debug" role="region">
            <div className="utility-section-heading">
              <h3>Logs & Debug</h3>
              <span>{gpu.asr_cuda_available ? "ready" : "check"}</span>
            </div>
            <p className="gpu-name">{gpuLabel}</p>
            <div className="vram-bar" aria-label={`VRAM usage ${vramPercent}%`}>
              <span style={{ width: `${vramPercent}%` }} />
            </div>
            <div className="chip-row debug-key-status" aria-label="Debug status summary">
              <span className="chip">{gpu.asr_cuda_available ? "CUDA ready" : gpu.asr_cuda_error ?? "CUDA check"}</span>
              <span className="chip">ASR {asrBackendLabel}</span>
              <span className="chip">{asrDeviceStatusLabel}</span>
              <span className="chip">{gpuFreeLabel}</span>
              <span className="chip">{webStatusLabel}</span>
            </div>
            <details className="debug-diagnostics">
              <summary>Runtime diagnostics</summary>
              <div className="chip-row">
                <span className="chip">{asrModelLabel}</span>
                <span className="chip">{asrCadenceLabel}</span>
                <span className="chip">{asrTimingLabel}</span>
                <span className="chip">
                  window {formatSeconds(gpu.transcription_window_seconds ?? 5)} / overlap {formatSeconds(gpu.transcription_overlap_seconds ?? 0.75)}
                </span>
                <span className="chip">Chat {gpu.ollama_chat_model ?? "qwen3.5:397b-cloud"} @ {ollamaChatHostLabel}</span>
                <span className={gpu.ollama_chat_reachable === false ? "chip warning" : "chip"}>{ollamaChatReachabilityLabel}</span>
                <span className="chip">Embed {gpu.ollama_embed_model ?? "embeddinggemma"} @ {ollamaEmbedHostLabel}</span>
                <span className={gpu.ollama_embed_reachable === false ? "chip warning" : "chip"}>{ollamaEmbedReachabilityLabel}</span>
                {(gpu.ollama_gpu_models ?? []).length === 0 ? (
                  <span className="chip muted">Ollama idle</span>
                ) : (
                  gpu.ollama_gpu_models?.map((model) => <span key={model} className="chip">{model}</span>)
                )}
                <span className="chip">keep chat {gpu.ollama_chat_keep_alive ?? gpu.ollama_keep_alive ?? "30m"}</span>
                <span className="chip">keep embed {gpu.ollama_embed_keep_alive ?? "0"}</span>
                <span className="chip">dedupe {gpu.dedupe_similarity_threshold ?? "?"}</span>
              </div>
            </details>
            <label className="toggle-row">
              <input
                aria-label="Show debug metadata"
                type="checkbox"
                checked={showDebugMetadata}
                onChange={(event) => setShowDebugMetadata(event.target.checked)}
              />
              Show debug metadata
            </label>
            <div className="system-log" aria-label="System logs">
              <strong>System logs</strong>
              {systemLogs.length === 0 ? (
                <p className="tool-note">No system logs yet.</p>
              ) : (
                <div className="system-log-list">
                  {systemLogs.map((log) => (
                    <article key={`${log.id}-${log.seq}`} className={`system-log-entry ${log.level}`}>
                      <div>
                        <strong>{log.title}</strong>
                        <time>{formatClock(log.at)}</time>
                      </div>
                      <p>{log.message}</p>
                      {showDebugMetadata && log.metadata && <pre>{formatDebugValue(log.metadata)}</pre>}
                    </article>
                  ))}
                </div>
              )}
            </div>
          </section>
          )}

          {toolView === "appearance" && (
          <section className="utility-section" aria-label="Appearance" role="region">
            <div className="utility-section-heading">
              <h3>Appearance</h3>
              <span>{THEMES.find((item) => item.value === theme)?.label ?? theme}</span>
            </div>
            <label className="field">
              <span>Theme</span>
              <select aria-label="Theme" value={theme} onChange={(event) => setTheme(event.target.value as ThemeName)}>
                {THEMES.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </label>
            <div className="theme-swatch-row" aria-label="Theme swatches">
              {THEMES.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  className={`theme-swatch ${theme === item.value ? "active" : ""}`}
                  aria-label={`${item.label} theme`}
                  onClick={() => setTheme(item.value)}
                >
                  <span />
                  <strong>{item.label}</strong>
                </button>
              ))}
            </div>
          </section>
            )}
        </main>
      )}
      </div>
    </div>
  );
}

function Sidebar({
  activePage,
  status,
  asrBackendLabel,
  gpuFreeLabel,
  retentionLabel,
  onNavigate,
}: {
  activePage: AppPage;
  status: string;
  asrBackendLabel: string;
  gpuFreeLabel: string;
  retentionLabel: string;
  onNavigate: (page: AppPage) => void;
}) {
  const items: { page: AppPage; label: string; short: string }[] = [
    { page: "live", label: "Live", short: "Live" },
    { page: "review", label: "Review", short: "Rev" },
    { page: "sessions", label: "Sessions", short: "Log" },
    { page: "tools", label: "Tools", short: "Tool" },
    { page: "models", label: "Models", short: "AI" },
    { page: "references", label: "References", short: "Refs" },
  ];
  return (
    <aside className="sidebar" aria-label="Primary navigation">
      <div className="sidebar-brand">
        <img className="brand-mark" src="/favicon.svg" alt="" aria-hidden="true" />
        <div>
          <strong>Brain Sidecar</strong>
          <small>Dross cockpit</small>
        </div>
      </div>
      <nav className="sidebar-nav" aria-label="Primary navigation">
        {items.map((item) => (
          <button
            key={item.page}
            type="button"
            className={`sidebar-item ${activePage === item.page ? "sidebar-item-active" : ""}`}
            aria-current={activePage === item.page ? "page" : undefined}
            onClick={() => onNavigate(item.page)}
          >
            <span aria-hidden="true">{item.short}</span>
            {item.label}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <button
          type="button"
          className="sidebar-status-button"
          aria-label="Open capture details"
          onClick={() => onNavigate("live")}
        >
          <span className="status-badge">{status}</span>
          <small>{retentionLabel}</small>
        </button>
        <button
          type="button"
          className="sidebar-status-button"
          aria-label="Open model details"
          onClick={() => onNavigate("models")}
        >
          <span>{asrBackendLabel}</span>
          <small>{gpuFreeLabel}</small>
        </button>
      </div>
    </aside>
  );
}

function sessionHasUserValue(session: SessionSummary): boolean {
  if (session.retention === "saved") {
    return true;
  }
  if ((session.transcript_count ?? 0) > 0 || (session.note_count ?? 0) > 0 || session.summary_exists) {
    return true;
  }
  return !["created", "stopped"].includes((session.status ?? "").toLowerCase());
}

function SessionsPage({
  sessions,
  selectedSession,
  selectedSessionId,
  search,
  showAll,
  busy,
  titleDraft,
  highlightedSourceIds,
  onSearchChange,
  onShowAllChange,
  onRefresh,
  onSelect,
  onTitleDraftChange,
  onSaveTitle,
  onHighlightSources,
  onCreateSavedSession,
  onGoLive,
}: {
  sessions: SessionSummary[];
  selectedSession: SessionDetail | null;
  selectedSessionId: string | null;
  search: string;
  showAll: boolean;
  busy: string;
  titleDraft: string;
  highlightedSourceIds: Set<string>;
  onSearchChange: (value: string) => void;
  onShowAllChange: (value: boolean) => void;
  onRefresh: () => void;
  onSelect: (id: string) => void;
  onTitleDraftChange: (value: string) => void;
  onSaveTitle: () => void;
  onHighlightSources: (ids: Set<string>) => void;
  onCreateSavedSession: () => void;
  onGoLive: () => void;
}) {
  const transcriptText = selectedSession?.transcript_segments.map((segment) => segment.text).join("\n") ?? "";
  const briefText = selectedSession?.summary?.summary ?? selectedSession?.note_cards.map((card) => `${card.title}\n${card.body}`).join("\n\n") ?? "";
  const cleanSearch = search.trim().toLowerCase();
  const visibleSessions = showAll ? sessions : sessions.filter(sessionHasUserValue);
  const filteredSessions = cleanSearch
    ? visibleSessions.filter((session) => (
      session.title.toLowerCase().includes(cleanSearch)
      || session.status.toLowerCase().includes(cleanSearch)
      || (session.retention ?? "").toLowerCase().includes(cleanSearch)
    ))
    : visibleSessions;
  const selectedTitleDirty = Boolean(
    selectedSession
    && titleDraft.trim()
    && titleDraft.trim() !== selectedSession.title.trim(),
  );
  if (sessions.length === 0) {
    return (
      <main className="page sessions-page" aria-label="Sessions">
        <header className="page-header">
          <div>
            <p className="label">Saved Work</p>
            <h1>Sessions</h1>
            <span>Browse saved transcripts, cards, and post-call summaries.</span>
          </div>
          {busy === "loading" && <span className="inline-loading">Refreshing saved sessions</span>}
        </header>
        <section className="empty-state session-empty-state" aria-label="No saved sessions">
          <h2>No saved sessions yet</h2>
          <p>Approve a completed Review job to save transcript text, Meeting Output cards, and summaries.</p>
          <div className="button-row centered">
            <button className="primary subtle" type="button" onClick={onCreateSavedSession}>
              Open Review
            </button>
            <button className="secondary" type="button" onClick={onGoLive}>
              Go to Live
            </button>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="page sessions-page" aria-label="Sessions">
      <header className="page-header">
        <div>
          <p className="label">Saved Work</p>
          <h1>Sessions</h1>
          <span>Browse saved transcripts, cards, and post-call summaries.</span>
        </div>
        <div className="page-header-actions">
          {busy === "loading" && <span className="inline-loading">Refreshing saved sessions</span>}
          <button className="secondary" type="button" onClick={() => onShowAllChange(!showAll)}>
            {showAll ? "Hide Empty" : "All Sessions"}
          </button>
          <button className="secondary" type="button" onClick={onRefresh} disabled={busy === "loading"}>Refresh</button>
        </div>
      </header>
      <div className="split-pane">
        <section className="session-list" aria-label="Session list">
          <label className="field field-wide">
            <span>Search sessions</span>
            <input
              aria-label="Search sessions"
              value={search}
              onChange={(event) => onSearchChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") onRefresh();
              }}
              placeholder="meeting title"
            />
          </label>
          {filteredSessions.length === 0 ? (
            <p className="empty-note">{showAll ? "No sessions match this filter." : "No saved or useful sessions match. Use All Sessions for empty temporary runs."}</p>
          ) : (
            filteredSessions.map((session) => (
              <button
                key={session.id}
                type="button"
                className={`session-list-item ${selectedSessionId === session.id ? "active" : ""}`}
                onClick={() => onSelect(session.id)}
              >
                <strong>{session.title}</strong>
                <span>{formatDateTime(session.started_at)}</span>
                <small className="chip-row">
                  <span className="status-badge">{session.status}</span>
                  <span className="status-badge">{session.retention ?? "empty"}</span>
                  <span className="status-badge">{session.transcript_count} lines</span>
                  <span className="status-badge">{session.note_count} cards</span>
                </small>
              </button>
            ))
          )}
        </section>
        <section className="session-detail" aria-label="Session detail">
          {!selectedSession ? (
            <div className="empty-panel">
              <h2>Select a session</h2>
              <p>Saved transcripts, Meeting Output cards, and summaries open here.</p>
            </div>
          ) : (
            <>
              <div className="session-title-row">
                <label className="field field-wide">
                  <span>Title</span>
                  <input
                    aria-label="Session title"
                    value={titleDraft}
                    onChange={(event) => onTitleDraftChange(event.target.value)}
                  />
                </label>
                {selectedTitleDirty && (
                  <button className="secondary" type="button" onClick={onSaveTitle} disabled={busy === "saving-title"}>
                    {busy === "saving-title" ? "Saving..." : "Save Title"}
                  </button>
                )}
              </div>
              <div className="chip-row">
                <span className="chip">{selectedSession.status}</span>
                <span className="chip">{selectedSession.retention}</span>
                <span className="chip">{selectedSession.transcript_count} transcript lines</span>
                <span className="chip">{selectedSession.note_count} cards</span>
                <span className="chip">raw audio discarded</span>
              </div>
              {selectedSession.transcript_redacted ? (
                <div className="empty-panel">
                  <h3>This session was listen-only</h3>
                  <p>Transcript text was not persisted for this temporary session.</p>
                </div>
              ) : (
                <div className="session-content-grid">
                  <section>
                    <div className="section-heading-row">
                      <h3>Transcript</h3>
                      <button className="secondary" type="button" disabled={!transcriptText} onClick={() => navigator.clipboard?.writeText(transcriptText)}>Copy transcript</button>
                    </div>
                    <div className="session-transcript scroll">
                      {selectedSession.transcript_segments.length === 0 ? (
                        <p className="empty-note">No transcript was saved for this session.</p>
                      ) : selectedSession.transcript_segments.map((segment) => {
                        const ids = new Set([segment.id, ...(segment.source_segment_ids ?? [])]);
                        const highlighted = Array.from(ids).some((id) => highlightedSourceIds.has(id));
                        return (
                          <article
                            key={segment.id}
                            className={`saved-transcript-row ${highlighted ? "highlight" : ""}`}
                            data-transcript-id={segment.id}
                          >
                            <span>{formatSeconds(segment.start_s)}-{formatSeconds(segment.end_s)}</span>
                            <strong>{segment.speaker_label ?? segment.speaker_role ?? "Speaker"}</strong>
                            <p>{segment.text}</p>
                          </article>
                        );
                      })}
                    </div>
                  </section>
                  <section>
                    <div className="section-heading-row">
                      <h3>Cards & Brief</h3>
                      <button className="secondary" type="button" disabled={!briefText} onClick={() => navigator.clipboard?.writeText(briefText)}>Copy brief</button>
                    </div>
                    {selectedSession.summary && <MeetingSummaryBrief summary={selectedSession.summary} compact />}
                    <div className="session-card-list">
                      {selectedSession.note_cards.length === 0 ? (
                        <p className="empty-note">No Meeting Output cards were saved for this session.</p>
                      ) : selectedSession.note_cards.map((card) => (
                        <article key={card.id} className="session-note-card">
                          <span className="status-badge">{card.kind}</span>
                          <strong>{card.title}</strong>
                          <p>{card.body}</p>
                          {card.evidence_quote && <blockquote>{card.evidence_quote}</blockquote>}
                          {(card.source_segment_ids ?? []).length > 0 && (
                            <button className="secondary" type="button" onClick={() => onHighlightSources(new Set(card.source_segment_ids))}>
                              Highlight evidence
                            </button>
                          )}
                        </article>
                      ))}
                    </div>
                  </section>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function ModelsPage({
  gpu,
  pipeline,
  gpuLabel,
  vramPercent,
  gpuFreeLabel,
  asrBackendLabel,
  asrModelLabel,
  asrCadenceLabel,
  asrTimingLabel,
  ollamaChatHostLabel,
  ollamaEmbedHostLabel,
  ollamaChatReachabilityLabel,
  ollamaEmbedReachabilityLabel,
  ollamaModels,
  ollamaModelBusy,
  ollamaModelNotice,
  onRefresh,
  onRefreshOllamaModels,
  onSelectOllamaChatModel,
}: {
  gpu: GpuHealth;
  pipeline: PipelineState;
  gpuLabel: string;
  vramPercent: number;
  gpuFreeLabel: string;
  asrBackendLabel: string;
  asrModelLabel: string;
  asrCadenceLabel: string;
  asrTimingLabel: string;
  ollamaChatHostLabel: string;
  ollamaEmbedHostLabel: string;
  ollamaChatReachabilityLabel: string;
  ollamaEmbedReachabilityLabel: string;
  ollamaModels: OllamaModelsResponse | null;
  ollamaModelBusy: string;
  ollamaModelNotice: string;
  onRefresh: () => void;
  onRefreshOllamaModels: () => void;
  onSelectOllamaChatModel: (model: string) => void;
}) {
  const asrDevice = gpu.asr_device ?? "cpu";
  const hearingReady = !gpu.asr_backend_error && (asrDevice === "cpu" || gpu.asr_cuda_available !== false);
  const thinkingReady = gpu.ollama_chat_reachable !== false;
  const memoryReady = gpu.ollama_embed_reachable !== false;
  const runtimeReady = hearingReady && thinkingReady && memoryReady;
  const readinessTone = runtimeReady ? "ready" : hearingReady || thinkingReady || memoryReady ? "warning" : "danger";
  const readinessPill = runtimeReady ? "Ready" : hearingReady || thinkingReady || memoryReady ? "Degraded" : "Setup needed";
  const readinessTitle = runtimeReady
    ? "Dross runtime ready"
    : hearingReady && !thinkingReady
      ? "Dross hearing ready; thinking offline"
      : thinkingReady && !hearingReady
        ? "Dross thinking ready; hearing needs setup"
        : "Dross setup needed";
  const embeddingKeepalive = gpu.ollama_embed_keep_alive ?? "0";
  const embeddingKeepaliveLabel = embeddingKeepalive === "0" ? "not resident (0)" : embeddingKeepalive;
  const residentChatModels = gpu.ollama_gpu_models ?? [];
  const chatModel = gpu.ollama_chat_model ?? "qwen3.5:397b-cloud";
  const selectedChatModel = ollamaModels?.selected_chat_model ?? chatModel;
  const modelOptions = (ollamaModels?.models.length ? ollamaModels.models : [{ name: selectedChatModel, current: true }])
    .reduce<OllamaModelOption[]>((options, model) => {
      if (!model.name || options.some((option) => option.name === model.name)) {
        return options;
      }
      return [...options, model];
    }, []);
  if (selectedChatModel && !modelOptions.some((model) => model.name === selectedChatModel)) {
    modelOptions.unshift({ name: selectedChatModel, current: true, configured: true });
  }
  const sameGpuInterpretation = runtimeReady
    ? `Runtime path looks healthy: ${asrBackendLabel} handles hearing on ${asrDevice}, ${selectedChatModel} is ${residentChatModels.includes(selectedChatModel) ? "resident" : "reachable"}, and embeddings ${embeddingKeepalive === "0" ? "load on demand" : "stay resident"}.`
    : "Health checks show which part of Dross needs attention before a meeting.";
  return (
    <main className="page models-page" aria-label="Models">
      <header className="page-header">
        <div>
          <p className="label">Runtime</p>
          <h1>Models</h1>
          <span>Local model residency and runtime model selection.</span>
        </div>
        <button className="secondary" type="button" onClick={onRefresh}>Refresh status</button>
      </header>
      <section className={`readiness-summary ${readinessTone}`} aria-label="Dross readiness summary">
        <div>
          <p className="label">Readiness</p>
          <h2>{readinessTitle}</h2>
          <div className="readiness-kv" aria-label="Dross readiness details">
            <span><strong>Hearing</strong>{hearingReady ? `${asrBackendLabel} · ${asrCadenceLabel}` : (gpu.asr_backend_error || gpu.asr_cuda_error || "ASR check needed")}</span>
            <span><strong>Thinking</strong>{thinkingReady ? `${selectedChatModel} · ${ollamaChatHostLabel}` : "chat offline"}</span>
            <span><strong>Memory</strong>{memoryReady ? `${gpu.ollama_embed_model ?? "embeddinggemma"} · ${ollamaEmbedHostLabel}` : "embeddings offline"}</span>
            <span><strong>VRAM</strong>{gpuFreeLabel}</span>
          </div>
          <p className="readiness-interpretation">{sameGpuInterpretation}</p>
        </div>
        <span className={`readiness-pill ${readinessTone}`}>{readinessPill}</span>
      </section>
      <div className="models-grid">
        <ModelCard title="Hearing / ASR" status={gpu.asr_backend_error ? "error" : "ready"}>
          <RuntimeRow label="Backend" value={asrBackendLabel} />
          <RuntimeRow label="Model" value={asrModelLabel} />
          <RuntimeRow label="Device" value={asrDevice} />
          <RuntimeRow label="Cadence" value={asrCadenceLabel} />
          <RuntimeRow label="Last ASR" value={asrTimingLabel} />
          <RuntimeRow label="Dtype" value={gpu.asr_dtype ?? "configured in .env"} />
          {gpu.asr_backend_error && <p className="warning-text">{gpu.asr_backend_error}</p>}
        </ModelCard>
        <ModelCard title="Thinking / Chat" status={gpu.ollama_chat_reachable === false ? "offline" : "ready"}>
          <label className="field model-select-field">
            Ollama model
            <select
              aria-label="Ollama chat model"
              value={selectedChatModel}
              disabled={ollamaModelBusy === "saving" || modelOptions.length === 0}
              onChange={(event) => onSelectOllamaChatModel(event.target.value)}
            >
              {modelOptions.map((model) => (
                <option key={model.name} value={model.name}>
                  {formatOllamaModelOption(model)}
                </option>
              ))}
            </select>
          </label>
          <div className="model-selector-meta">
            <small>{ollamaModelBusy === "loading" ? "Loading models" : `${modelOptions.length} available`}</small>
            <button className="secondary" type="button" onClick={onRefreshOllamaModels} disabled={ollamaModelBusy !== ""}>
              Refresh list
            </button>
          </div>
          {ollamaModelNotice && <p className={ollamaModelNotice.includes("failed") || ollamaModelNotice.includes("unavailable") ? "warning-text" : "muted-text"}>{ollamaModelNotice}</p>}
          <RuntimeRow label="Active" value={selectedChatModel} />
          <RuntimeRow label="Host" value={ollamaChatHostLabel} />
          <RuntimeRow label="Reachability" value={ollamaChatReachabilityLabel} />
          <RuntimeRow label="Keepalive" value={gpu.ollama_chat_keep_alive ?? gpu.ollama_keep_alive ?? "30m"} />
          <RuntimeRow label="Timeout" value={`${gpu.ollama_chat_timeout_seconds ?? 20}s`} />
        </ModelCard>
        <ModelCard title="Memory / Embeddings" status={gpu.ollama_embed_reachable === false ? "offline" : "ready"}>
          <RuntimeRow label="Model" value={gpu.ollama_embed_model ?? "embeddinggemma"} />
          <RuntimeRow label="Host" value={ollamaEmbedHostLabel} />
          <RuntimeRow label="Reachability" value={ollamaEmbedReachabilityLabel} />
          <RuntimeRow label="Keepalive" value={embeddingKeepaliveLabel} />
          <RuntimeRow label="Timeout" value={`${gpu.ollama_embed_timeout_seconds ?? 12}s`} />
        </ModelCard>
        <ModelCard title="GPU Residency" status={gpu.gpu_pressure ?? "check"}>
          <RuntimeRow label="Device" value={gpuLabel} />
          <div className="vram-bar" aria-label={`VRAM usage ${vramPercent}%`}><span style={{ width: `${vramPercent}%` }} /></div>
          <RuntimeRow label="Free VRAM" value={gpuFreeLabel} />
          <RuntimeRow label="Ollama models" value={gpu.ollama_gpu_models?.join(", ") || "idle"} />
          <RuntimeRow label="Unload on ASR" value={gpu.asr_unload_ollama_on_start ? "yes" : "no"} />
          <RuntimeRow label="Transcript revisions" value={String(pipeline.finalSegmentsReplaced)} />
        </ModelCard>
      </div>
      <details className="copy-snippet config-recipes" aria-label="Config recipes">
        <summary>
          <h2>Config recipes</h2>
          <span>Restart required after .env changes</span>
        </summary>
        {[
          ["CPU ASR fallback", "Use when CUDA is unstable or local GPU headroom should stay free.", "BRAIN_SIDECAR_ASR_DEVICE=cpu\nBRAIN_SIDECAR_ASR_COMPUTE_TYPE=int8"],
          ["CUDA ASR opt-in", "Use only when the GPU is stable and you want Faster-Whisper on CUDA.", "BRAIN_SIDECAR_ASR_DEVICE=cuda\nBRAIN_SIDECAR_ASR_COMPUTE_TYPE=float16"],
          ["Unload Ollama before ASR", "Use when CUDA ASR needs more free VRAM than resident chat models leave available.", "BRAIN_SIDECAR_ASR_UNLOAD_OLLAMA_ON_START=true"],
          ["Split chat to LAN", "Use when a LAN Ollama host should carry chat while this machine keeps hearing local.", "BRAIN_SIDECAR_OLLAMA_CHAT_HOST=http://192.168.86.219:11434"],
        ].map(([label, detail, snippet]) => (
          <button key={label} className="secondary" type="button" onClick={() => navigator.clipboard?.writeText(snippet)}>
            <span>{label}</span>
            <small>{detail}</small>
            <code>{snippet}</code>
            <small>Set in .env · restart required</small>
          </button>
        ))}
      </details>
    </main>
  );
}

function ModelCard({ title, status, children }: { title: string; status: string; children: ReactNode }) {
  return (
    <article className="model-card">
      <div className="section-heading-row">
        <h2>{title}</h2>
        <span className="status-badge">{status}</span>
      </div>
      <div className="model-card-body">{children}</div>
    </article>
  );
}

function formatOllamaModelOption(model: OllamaModelOption): string {
  if (model.cloud) {
    return `${model.name} · cloud`;
  }
  if (model.size && model.size > 0) {
    return `${model.name} · ${formatBytes(model.size)}`;
  }
  return model.configured ? `${model.name} · configured` : model.name;
}

function formatBytes(bytes: number): string {
  const gib = bytes / (1024 ** 3);
  if (gib >= 1) {
    return `${gib.toFixed(gib >= 10 ? 0 : 1)} GB`;
  }
  const mib = bytes / (1024 ** 2);
  return `${Math.max(1, Math.round(mib))} MB`;
}

function RuntimeRow({ label, value }: { label: string; value: string }) {
  return (
    <span className="runtime-row">
      <small>{label}</small>
      {" "}
      <strong>{value}</strong>
    </span>
  );
}

function ReferencesPage({
  libraryPath,
  libraryRoots,
  librarySources,
  libraryChunks,
  selectedLibrarySourcePath,
  memoryQuery,
  memoryBusy,
  webContextEnabled,
  webContextConfigured,
  onLibraryPathChange,
  onAddLibraryRoot,
  onReindexLibrary,
  onSelectLibrarySource,
  onMemoryQueryChange,
  onSearch,
  onRefresh,
}: {
  libraryPath: string;
  libraryRoots: string[];
  librarySources: LibraryChunkSource[];
  libraryChunks: LibraryChunk[];
  selectedLibrarySourcePath: string | null;
  memoryQuery: string;
  memoryBusy: string;
  webContextEnabled: boolean;
  webContextConfigured: boolean;
  onLibraryPathChange: (value: string) => void;
  onAddLibraryRoot: () => void;
  onReindexLibrary: () => void;
  onSelectLibrarySource: (sourcePath: string) => void;
  onMemoryQueryChange: (value: string) => void;
  onSearch: () => void;
  onRefresh: () => void;
}) {
  const [memoryView, setMemoryView] = useState<ReferenceView>("overview");
  const [selectedMemoryItem, setSelectedMemoryItem] = useState<MemorySelection | null>(null);
  const cleanQuery = memoryQuery.trim().toLowerCase();
  const visibleRoots = cleanQuery
    ? libraryRoots.filter((root) => root.toLowerCase().includes(cleanQuery))
    : libraryRoots;
  const visibleDocuments = cleanQuery
    ? librarySources.filter((source) => (
      source.title.toLowerCase().includes(cleanQuery)
      || source.source_path.toLowerCase().includes(cleanQuery)
    ))
    : librarySources;
  const selectedDocument = selectedMemoryItem?.view === "documents"
    ? selectedMemoryItem.source
    : librarySources.find((source) => source.source_path === selectedLibrarySourcePath) ?? null;
  const selectedRoot = selectedMemoryItem?.view === "roots" ? selectedMemoryItem.root : null;
  const searchReady = memoryQuery.trim().length > 0;
  const selectedRootIndexedCount = selectedRoot
    ? librarySources.filter((source) => source.source_path.startsWith(selectedRoot)).length
    : 0;
  const categoryItems: { view: ReferenceView; label: string; count: number; disabled?: boolean }[] = [
    { view: "overview", label: "Overview", count: librarySources.length },
    { view: "roots", label: "Library Roots", count: libraryRoots.length },
    { view: "documents", label: "Documents", count: librarySources.length },
  ];
  const activeCategoryLabel = categoryItems.find((item) => item.view === memoryView)?.label ?? "Overview";
  const memoryPaneTitle = memoryView === "overview"
    ? "Index overview"
    : memoryView === "roots"
      ? "Library roots"
      : "Documents";

  function activateMemoryView(view: ReferenceView) {
    setMemoryView(view);
    if (view === "overview") {
      setSelectedMemoryItem(null);
      return;
    }
    if (view === "roots" && visibleRoots[0]) {
      setSelectedMemoryItem({ view, id: `root:${visibleRoots[0]}`, root: visibleRoots[0] });
      return;
    }
    if (view === "documents" && visibleDocuments[0]) {
      setSelectedMemoryItem({ view, id: visibleDocuments[0].source_path, source: visibleDocuments[0] });
      onSelectLibrarySource(visibleDocuments[0].source_path);
      return;
    }
    setSelectedMemoryItem(null);
  }

  function runMemorySearch() {
    if (!memoryQuery.trim()) {
      return;
    }
    onSearch();
  }

  function clearMemorySearch() {
    onMemoryQueryChange("");
    setSelectedMemoryItem(null);
    setMemoryView("overview");
  }

  return (
    <main className="page memory-page" aria-label="References">
      <header className="page-header">
        <div>
          <p className="label">Index</p>
          <h1>References</h1>
          <span>{librarySources.length} indexed documents / {libraryRoots.length} roots</span>
        </div>
        <div className="page-header-actions">
          {memoryBusy === "loading" && <span className="inline-loading">Refreshing references</span>}
          <button className="secondary" type="button" onClick={onRefresh} disabled={memoryBusy === "loading"}>Refresh</button>
        </div>
      </header>
      <div className="memory-explorer explorer-shell">
        <aside className="memory-tree explorer-nav scroll" aria-label="Reference categories">
          <form
            className="memory-search-form"
            role="search"
            onSubmit={(event) => {
              event.preventDefault();
              runMemorySearch();
            }}
          >
            <label className="field field-wide">
              <span>Search references</span>
              <input
                aria-label="Search references"
                value={memoryQuery}
                onChange={(event) => onMemoryQueryChange(event.target.value)}
                placeholder="document, standard, phrase"
              />
            </label>
            <p className="memory-search-note" aria-label="Reference search behavior">
              Type filters the open category. Search loads matching EE reference chunks.
            </p>
            <div className="memory-search-actions">
              <button className="secondary" type="submit" disabled={!memoryQuery.trim() || memoryBusy === "searching"}>
                {memoryBusy === "searching" ? "Searching..." : "Search"}
              </button>
              {searchReady && (
                <button className="secondary" type="button" onClick={clearMemorySearch}>
                  Clear
                </button>
              )}
            </div>
          </form>
          <nav className="memory-category-nav" aria-label="Reference category navigation">
            {categoryItems.map((item) => (
              <button
                key={item.view}
                type="button"
                className={`memory-category-item ${memoryView === item.view ? "active" : ""} ${item.disabled ? "disabled" : ""}`}
                aria-current={memoryView === item.view ? "page" : undefined}
                disabled={item.disabled}
                onClick={() => activateMemoryView(item.view)}
              >
                <span>{item.label}</span>
                <strong>{item.count}</strong>
              </button>
            ))}
          </nav>
        </aside>

        <section className="memory-list-pane scroll" aria-label="Reference items">
          <div className="memory-pane-head">
            <div>
              <p className="label">{activeCategoryLabel}</p>
              <h2>{memoryPaneTitle}</h2>
            </div>
            <div className="button-row">
              <button className="secondary" type="button" onClick={onReindexLibrary}>Reindex References</button>
            </div>
          </div>

          {memoryView === "overview" && (
            <>
              <div className="memory-overview" aria-label="Reference index summary">
                <span><strong>{libraryRoots.length}</strong> library roots</span>
                <span><strong>{librarySources.length}</strong> documents</span>
                <span><strong>{librarySources.reduce((total, source) => total + source.chunk_count, 0)}</strong> chunks</span>
                <span><strong>{webContextEnabled ? "on" : "off"}</strong> Brave</span>
                <span><strong>{webContextConfigured ? "ready" : "missing"}</strong> web key</span>
              </div>
              <div className="memory-short-list" aria-label="Recent references">
                <div className="section-heading-row">
                  <h3>Recent references</h3>
                  <span>{Math.min(4, librarySources.length)} shown</span>
                </div>
                {librarySources.slice(0, 4).map((source) => ({
                  id: `doc:${source.source_path}`,
                  title: source.title,
                  meta: `${source.chunk_count} chunks · ${PathHint(source.source_path)}`,
                  onClick: () => {
                    setMemoryView("documents");
                    setSelectedMemoryItem({ view: "documents", id: source.source_path, source });
                    onSelectLibrarySource(source.source_path);
                  },
                })).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className="memory-item-row"
                    onClick={item.onClick}
                  >
                    <strong>{item.title}</strong>
                    <span>{item.meta}</span>
                  </button>
                ))}
                {librarySources.length === 0 && (
                  <div className="empty-panel">
                    <h2>No indexed references yet</h2>
                    <p>Add the EE reference root or reindex the configured roots.</p>
                  </div>
                )}
              </div>
            </>
          )}

          {memoryView === "roots" && (
            <>
              <div className="memory-root-add" aria-label="Add library root">
                <label className="field field-wide">
                  <span>Library root path</span>
                  <input
                    aria-label="Library root path"
                    value={libraryPath}
                    onChange={(event) => onLibraryPathChange(event.target.value)}
                    placeholder="/path/to/notes-or-docs"
                  />
                </label>
                <button className="primary subtle" type="button" onClick={onAddLibraryRoot} disabled={!libraryPath.trim()}>
                  Add Root
                </button>
              </div>
              {visibleRoots.length === 0 ? (
                <div className="empty-panel">
                  <h2>No library roots yet</h2>
                  <p>Add a local path to make files available for document indexing.</p>
                </div>
              ) : visibleRoots.map((root) => (
                <button
                  key={root}
                  type="button"
                  className={`memory-item-row ${selectedRoot === root ? "active" : ""}`}
                  onClick={() => setSelectedMemoryItem({ view: "roots", id: `root:${root}`, root })}
                >
                  <strong>{PathLabel(root)}</strong>
                  <span>{PathHint(root)}</span>
                </button>
              ))}
            </>
          )}

          {memoryView === "documents" && (
            visibleDocuments.length === 0 ? (
              <div className="empty-panel">
                <h2>No indexed documents yet</h2>
                <p>Add a Library Root or reindex the current roots to inspect file chunks here.</p>
              </div>
            ) : visibleDocuments.map((source) => (
              <button
                key={source.source_path}
                type="button"
                className={`memory-item-row ${selectedDocument?.source_path === source.source_path ? "active" : ""}`}
                onClick={() => {
                  setSelectedMemoryItem({ view: "documents", id: source.source_path, source });
                  onSelectLibrarySource(source.source_path);
                }}
              >
                <strong>{source.title}</strong>
                <span>{source.chunk_count} chunks · updated {formatDateTime(source.updated_at)}</span>
                <small>{PathHint(source.source_path)}</small>
              </button>
            ))
          )}

        </section>

        <section className="memory-detail explorer-detail scroll" aria-label="Reference detail">
          {memoryView === "overview" && (
            <div className="memory-detail-stack">
              <div className="empty-panel">
                <h2>Reference index status</h2>
                <p>Local EE references ground technical analysis. Brave adds sanitized, live-only public context when enabled and configured.</p>
              </div>
              <div className="chip-row">
                {libraryRoots.slice(0, 6).map((root) => <span key={root} className="chip">{PathLabel(root)}</span>)}
                <span className={webContextEnabled && webContextConfigured ? "chip" : "chip warning"}>
                  Brave {webContextEnabled ? webContextConfigured ? "ready" : "key missing" : "off"}
                </span>
              </div>
            </div>
          )}

          {memoryView !== "overview" && !selectedMemoryItem && (
            <div className="empty-panel">
              <h2>
                {memoryView === "roots" ? "No root selected"
                  : "No document selected"}
              </h2>
              <p>Select a row to inspect paths, chunks, and indexing context for this category.</p>
            </div>
          )}

          {selectedRoot && (
            <div className="memory-detail-stack">
              <p className="label">Library Root</p>
              <h2>{PathLabel(selectedRoot)}</h2>
              <code>{selectedRoot}</code>
              <div className="chip-row">
                <span className="chip">{selectedRootIndexedCount} indexed documents</span>
              </div>
              <div className="button-row">
                <button className="secondary" type="button" onClick={onReindexLibrary}>Reindex References</button>
              </div>
            </div>
          )}

          {selectedDocument && (
            <div className="memory-detail-stack">
              <p className="label">Indexed Document</p>
              <h2>{selectedDocument.title}</h2>
              <code>{selectedDocument.source_path}</code>
              <div className="chip-row">
                <span className="chip">{selectedDocument.chunk_count} chunks</span>
                <span className="chip">first chunk {selectedDocument.first_chunk_index}</span>
              </div>
              <div className="library-chunk-list memory-chunk-list" aria-label="Document chunks">
                {libraryChunks.length === 0 ? <p className="empty-note">No chunks loaded for this document yet.</p> : libraryChunks.map((chunk) => (
                  <article key={chunk.id} className="library-chunk-card">
                    <span className="status-badge">chunk {chunk.chunk_index}</span>
                    <strong>{chunk.title}</strong>
                    <p>{chunk.text}</p>
                  </article>
                ))}
              </div>
            </div>
          )}

        </section>
      </div>
    </main>
  );
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="metric-card" aria-label={`${label} metric`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function LiveTranscriptPane({
  setupHeader,
  runtimeStrip,
  transcriptCount,
  currentCardCount,
  asrDeviceStatusLabel,
  errors,
  playbackCue,
  rows,
  expandedCards,
  highlightedSourceIds,
  showDebugMetadata,
  active,
  followLive,
  unseenItems,
  manualQuery,
  manualQueryBusy,
  manualSourceCards,
  transcriptScrollRef,
  liveEndRef,
  onScroll,
  onJumpToLive,
  onToggleCard,
  onJumpToEvidence,
  onPin,
  onDismiss,
  onCopy,
  onManualQueryChange,
  onManualSearch,
}: {
  setupHeader: ReactNode;
  runtimeStrip: ReactNode;
  transcriptCount: number;
  currentCardCount: number;
  asrDeviceStatusLabel: string;
  errors: string[];
  playbackCue: ReactNode;
  rows: LiveFieldRow[];
  expandedCards: Set<string>;
  highlightedSourceIds: Set<string>;
  showDebugMetadata: boolean;
  active: boolean;
  followLive: boolean;
  unseenItems: number;
  manualQuery: string;
  manualQueryBusy: boolean;
  manualSourceCards: SidecarDisplayCard[];
  transcriptScrollRef: RefObject<HTMLDivElement | null>;
  liveEndRef: RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  onJumpToLive: () => void;
  onToggleCard: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
  onManualQueryChange: (value: string) => void;
  onManualSearch: () => void;
}) {
  return (
    <section className="transcript-pane live-transcript-pane" aria-label="Live transcript pane">
      {setupHeader}
      {runtimeStrip}
      <div className="field-toolbar live-field-toolbar">
        <div>
          <p className="label">Live</p>
          <h1>Conversation</h1>
        </div>
        <div className="field-toolbar-status">
          <span>{transcriptCount} heard</span>
          <span>{currentCardCount} returns</span>
          <span>{asrDeviceStatusLabel}</span>
        </div>
      </div>

      {errors.length > 0 && (
        <section className="errors" role="alert">
          {errors.map((error, index) => (
            <p key={`${error}-${index}`}>{error}</p>
          ))}
        </section>
      )}

      <div
        id="transcript"
        className="transcript-scroll live-field-scroll scroll"
        role="feed"
        aria-label="Live field"
        ref={transcriptScrollRef as RefObject<HTMLDivElement>}
        onScroll={onScroll}
        onWheel={() => window.setTimeout(onScroll, 0)}
      >
        {playbackCue}
        <LiveFieldPane
          rows={rows}
          expandedCards={expandedCards}
          highlightedSourceIds={highlightedSourceIds}
          showDebugMetadata={showDebugMetadata}
          onToggle={onToggleCard}
          onJumpToEvidence={onJumpToEvidence}
          onPin={onPin}
          onDismiss={onDismiss}
          onCopy={onCopy}
          active={active}
        />
        <div className="live-field-end" ref={liveEndRef as RefObject<HTMLDivElement>} aria-hidden="true" />
      </div>

      {!followLive && (
        <button className="jump-live" onClick={onJumpToLive}>
          Jump to live{unseenItems > 0 ? ` (${unseenItems})` : ""}
        </button>
      )}

      <form
        className="query-bar"
        role="search"
        onSubmit={(event) => {
          event.preventDefault();
          onManualSearch();
        }}
      >
        <input
          aria-label="Unified query"
          value={manualQuery}
          onChange={(event) => onManualQueryChange(event.target.value)}
          placeholder="Ask Sidecar"
        />
        <button className="primary subtle" disabled={manualQueryBusy || !manualQuery.trim()}>
          {manualQueryBusy ? "Searching..." : "Search"}
        </button>
      </form>

      <ManualSourceSections cards={manualSourceCards} />
    </section>
  );
}

function ReplayTestDrawer({
  visible,
  expanded,
  stateDetail,
  stateLabel,
  stateStatus,
  canReset,
  resetDisabled,
  toggleDisabled,
  testSourcePath,
  fileBusy,
  uploadDisabled,
  maxSeconds,
  expectedTerms,
  configDisabled,
  prepared,
  report,
  playbackActive,
  playbackPaused,
  testBusy,
  captureActive,
  captureStarting,
  playbackElapsedSeconds,
  onToggle,
  onReset,
  onFilePick,
  onMaxSecondsChange,
  onExpectedTermsChange,
  onStartPlayback,
  onPausePlayback,
  onResumePlayback,
  onRestartPlayback,
  onStopPlayback,
  stopPlaybackDisabled,
}: {
  visible: boolean;
  expanded: boolean;
  stateDetail: string;
  stateLabel: string;
  stateStatus: string;
  canReset: boolean;
  resetDisabled: boolean;
  toggleDisabled: boolean;
  testSourcePath: string;
  fileBusy: boolean;
  uploadDisabled: boolean;
  maxSeconds: string;
  expectedTerms: string;
  configDisabled: boolean;
  prepared: TestModePrepared | null;
  report: TestModeReport | null;
  playbackActive: boolean;
  playbackPaused: boolean;
  testBusy: string;
  captureActive: boolean;
  captureStarting: boolean;
  playbackElapsedSeconds: number;
  onToggle: () => void;
  onReset: () => void;
  onFilePick: (event: ChangeEvent<HTMLInputElement>) => void;
  onMaxSecondsChange: (value: string) => void;
  onExpectedTermsChange: (value: string) => void;
  onStartPlayback: () => void;
  onPausePlayback: () => void;
  onResumePlayback: () => void;
  onRestartPlayback: () => void;
  onStopPlayback: () => void;
  stopPlaybackDisabled: boolean;
}) {
  if (!visible) {
    return null;
  }
  return (
    <section className={`live-test-panel ${expanded ? "expanded" : "collapsed"}`} role="region" aria-label="Live audio test">
      <div className="live-test-head">
        <div>
          <span className="label">Replay / Test</span>
          <strong>Recorded audio</strong>
          <small>{stateDetail}</small>
        </div>
        <div className="live-test-head-actions">
          <span className={`live-test-state ${stateStatus}`}>{stateLabel}</span>
          {canReset && (
            <button className="secondary compact-button" type="button" onClick={onReset} disabled={resetDisabled}>
              Reset
            </button>
          )}
          <button
            className={`secondary compact-button live-test-toggle ${expanded ? "active" : ""}`}
            type="button"
            aria-expanded={expanded}
            onClick={onToggle}
            disabled={toggleDisabled}
          >
            {expanded ? "Hide" : "Test"}
          </button>
        </div>
      </div>
      {expanded && (
        <div className="live-test-grid">
          <FilePickAction
            label="Test audio upload"
            actionLabel="Upload and play"
            accept={TEST_AUDIO_ACCEPT}
            path={testSourcePath}
            busy={fileBusy}
            disabled={uploadDisabled}
            onChange={onFilePick}
          />
          <label className="field">
            <span>Max seconds</span>
            <input
              aria-label="Live test max seconds"
              inputMode="decimal"
              value={maxSeconds}
              onChange={(event) => onMaxSecondsChange(event.target.value)}
              placeholder="optional"
              disabled={configDisabled}
            />
          </label>
          <label className="field field-wide">
            <span>Expected terms</span>
            <input
              aria-label="Live test expected terms"
              value={expectedTerms}
              onChange={(event) => onExpectedTermsChange(event.target.value)}
              placeholder="optional validation terms"
              disabled={configDisabled}
            />
          </label>
        </div>
      )}
      {prepared && (
        <div className="test-mode-summary" aria-label="Live prepared audio">
          <span><strong>{prepared.duration_seconds.toFixed(1)}s</strong> duration</span>
          <span><strong>{prepared.expected_terms.length}</strong> expected terms</span>
          {(testBusy === "playing" || playbackPaused) && (
            <span>
              <strong>{formatPlaybackClock(playbackElapsedSeconds)}</strong>
              {" / "}
              {formatPlaybackClock(prepared.duration_seconds)}
            </span>
          )}
          <small>{prepared.fixture_wav}</small>
        </div>
      )}
      {(prepared || playbackActive) && (
        <div className="test-playback-controls" role="group" aria-label="Recorded audio controls">
          <button
            className="primary compact-button"
            type="button"
            onClick={onStartPlayback}
            disabled={!prepared || Boolean(testBusy) || captureActive || captureStarting}
          >
            Start playback
          </button>
          <button className="secondary compact-button" type="button" onClick={onPausePlayback} disabled={testBusy !== "playing"}>
            Pause
          </button>
          <button className="secondary compact-button" type="button" onClick={onResumePlayback} disabled={testBusy !== "paused"}>
            Resume
          </button>
          <button
            className="secondary compact-button"
            type="button"
            onClick={onRestartPlayback}
            disabled={!prepared || ["preparing", "starting-playback", "pausing", "resuming"].includes(testBusy) || captureStarting}
          >
            Restart
          </button>
          <button className="danger compact-button" type="button" onClick={onStopPlayback} disabled={stopPlaybackDisabled}>
            Stop playback
          </button>
        </div>
      )}
      {report && (
        <div className="test-mode-summary" aria-label="Live recorded audio report">
          <span><strong>{report.transcript_segments}</strong> heard lines</span>
          <span><strong>{report.matched_expected_terms.length}/{report.expected_terms.length}</strong> terms matched</span>
          <small>{report.report_path ?? prepared?.report_path}</small>
        </div>
      )}
    </section>
  );
}

function LiveModelReturnsPane({
  mode,
  status,
  gpu,
  pipeline,
  transcriptCount,
  currentCards,
  contextCards,
  manualCards,
  groups,
  qualityMetrics,
  meetingDiagnostics,
  qualityGateActive,
  rawCount,
  systemLogs,
  contract,
  energyFrame,
  showAll,
  canShowMore,
  onToggleShowAll,
  expandedCards,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onPin,
  onDismiss,
  onCopy,
}: {
  mode: LiveWorkbenchMode;
  status: string;
  gpu: GpuHealth;
  pipeline: PipelineState;
  transcriptCount: number;
  currentCards: SidecarDisplayCard[];
  contextCards: SidecarDisplayCard[];
  manualCards: SidecarDisplayCard[];
  groups: Record<MeetingOutputBucket, SidecarDisplayCard[]>;
  qualityMetrics: ReturnType<typeof buildSidecarQualityMetrics>;
  meetingDiagnostics: MeetingDiagnostics | null;
  qualityGateActive: boolean;
  rawCount: number;
  systemLogs: SystemLog[];
  contract: MeetingContractPayload | null;
  energyFrame: EnergyFrame | null;
  showAll: boolean;
  canShowMore: boolean;
  onToggleShowAll: () => void;
  expandedCards: Set<string>;
  showDebugMetadata: boolean;
  onToggle: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
}) {
  const intelligenceOnly = mode === "live" || mode === "replay" || mode === "dev-test";
  const visibleSections = MEETING_OUTPUT_SECTIONS.filter((section) => groups[section.bucket].length > 0);
  const hasCards = visibleSections.length > 0 || contextCards.length > 0 || manualCards.length > 0;
  const latestLogs = systemLogs.slice(0, 3);
  const statusTiles: Array<{ label: string; value: string; detail: string; tone?: "normal" | "good" | "warning" }> = mode === "live"
    ? [
        {
          label: "Ollama",
          value: gpu.ollama_chat_model ?? "chat model",
          detail: formatOllamaHostLabel(gpu.ollama_chat_host),
        },
        {
          label: "Sidecar",
          value: `${currentCards.length} grounded`,
          detail: `${rawCount} raw · queue ${pipeline.queueDepth}`,
          tone: currentCards.length > 0 ? "good" : pipeline.queueDepth > 0 ? "warning" : "normal",
        },
      ]
    : [
        {
          label: "Ollama",
          value: gpu.ollama_chat_model ?? "chat model",
          detail: formatOllamaHostLabel(gpu.ollama_chat_host),
        },
        {
          label: "Runtime",
          value: status,
          detail: `Queue ${pipeline.queueDepth}`,
          tone: pipeline.queueDepth > 0 ? "warning" : "normal",
        },
        {
          label: "Transcript",
          value: `${transcriptCount} heard`,
          detail: pipeline.asrDurationMs != null ? `${pipeline.asrDurationMs.toFixed(0)} ms final` : "ASR window",
        },
        {
          label: "Sidecar",
          value: `${rawCount} raw`,
          detail: `${currentCards.length} grounded`,
          tone: currentCards.length > 0 ? "good" : "normal",
        },
      ];
  return (
    <section className="context-pane live-model-returns-pane" id="recall" aria-label="Sidecar Intelligence">
      <div className="field-toolbar context-toolbar">
        <div>
          <p className="label">Sidecar</p>
          <h1>Sidecar Intelligence</h1>
        </div>
        <div className="field-toolbar-status">
          <span>{liveWorkbenchModeLabel(mode)}</span>
          <span>{currentCards.length} current</span>
          <span>{qualityGateActive ? "Quality gate on" : "Quality gate off"}</span>
        </div>
      </div>

      {!intelligenceOnly && (
        <div className="model-return-status-grid" aria-label="Model return status">
          {statusTiles.map((tile) => (
            <StatusTile
              key={tile.label}
              label={tile.label}
              value={tile.value}
              detail={tile.detail}
              tone={tile.tone}
            />
          ))}
        </div>
      )}

      {!intelligenceOnly && <ContractActiveBadge contract={contract} />}
      <EnergyLensBadge frame={energyFrame} />

      <div className="context-card-list live-return-list scroll" aria-label="Sidecar returns">
        {!intelligenceOnly && (
          <QualityStrip
            metrics={qualityMetrics}
            diagnostics={meetingDiagnostics}
            rawCount={rawCount}
            qualityGateActive={qualityGateActive}
            showDebugMetadata={showDebugMetadata}
          />
        )}

        {!intelligenceOnly && !hasCards && (
          <ContextQuietEmpty
            title="Awaiting grounded returns"
            body="Ollama and Sidecar returns will appear here when transcript evidence is strong enough."
          />
        )}

        {visibleSections.map((section) => (
          <section key={section.bucket} className="work-note-section meeting-output-section live-return-section" aria-label={`${section.title} summary`}>
            <div className="work-note-section-head">
              <h2>{section.title}</h2>
              <span>{groups[section.bucket].length}</span>
            </div>
            <div className="context-preview-list">
              {groups[section.bucket].map((card) => (
                <ContextCard
                  key={`live-return-${section.bucket}-${card.id}`}
                  card={card}
                  compact
                  expanded={expandedCards.has(card.id)}
                  showDebugMetadata={showDebugMetadata}
                  onToggle={() => onToggle(card.id)}
                  onJumpToEvidence={() => onJumpToEvidence(card)}
                  onPin={() => onPin(card)}
                  onDismiss={() => onDismiss(card)}
                  onCopy={() => onCopy(card)}
                />
              ))}
            </div>
          </section>
        ))}

        {canShowMore && (
          <button className="secondary show-more-context" onClick={onToggleShowAll}>
            {showAll ? "Show less" : "Show more"}
          </button>
        )}

        {contextCards.length > 0 && (
          <details className="session-context-preview context-output reference-context-output" aria-label="Reference context">
            <summary>
              <span>Reference context</span>
              <strong>{contextCards.length}</strong>
            </summary>
            <div className="context-preview-list">
              {contextCards.map((card) => (
                <ContextCard
                  key={`live-context-${card.id}`}
                  card={card}
                  compact
                  expanded={expandedCards.has(card.id)}
                  showDebugMetadata={showDebugMetadata}
                  onToggle={() => onToggle(card.id)}
                  onJumpToEvidence={() => onJumpToEvidence(card)}
                  onPin={() => onPin(card)}
                  onDismiss={() => onDismiss(card)}
                  onCopy={() => onCopy(card)}
                />
              ))}
            </div>
          </details>
        )}

        {manualCards.length > 0 && (
          <details className="session-context-preview manual-output" aria-label="Manual query cards">
            <summary>
              <span>Manual query</span>
              <strong>{manualCards.length}</strong>
            </summary>
            <div className="context-preview-list">
              {manualCards.map((card) => (
                <ContextCard
                  key={`live-manual-${card.id}`}
                  card={card}
                  compact
                  expanded={expandedCards.has(card.id)}
                  showDebugMetadata={showDebugMetadata}
                  onToggle={() => onToggle(card.id)}
                  onJumpToEvidence={() => onJumpToEvidence(card)}
                  onPin={() => onPin(card)}
                  onDismiss={() => onDismiss(card)}
                  onCopy={() => onCopy(card)}
                />
              ))}
            </div>
          </details>
        )}

        {!intelligenceOnly && latestLogs.length > 0 && (
          <details className="session-context-preview live-return-log" aria-label="Runtime return log">
            <summary>
              <span>Runtime events</span>
              <strong>{latestLogs.length}</strong>
            </summary>
            <div className="live-return-log-list">
              {latestLogs.map((item) => (
                <article key={item.id} className={`live-return-log-item ${item.level}`}>
                  <strong>{item.title}</strong>
                  <p>{item.message}</p>
                </article>
              ))}
            </div>
          </details>
        )}
      </div>
    </section>
  );
}

function PostCallMeetingOutput({
  groups,
  usefulCount,
  rawCount,
  qualityMetrics,
  meetingDiagnostics,
  qualityGateActive,
  loading,
  emptyTitle,
  emptyBody,
  contextCards,
  manualCards,
  showAll,
  canShowMore,
  onToggleShowAll,
  expandedCards,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onPin,
  onDismiss,
  onCopy,
  contract,
  energyFrame,
  showBrief,
  briefMarkdown,
  currentCount,
  contextCount,
  onCopyBrief,
}: {
  groups: Record<MeetingOutputBucket, SidecarDisplayCard[]>;
  usefulCount: number;
  rawCount: number;
  qualityMetrics: ReturnType<typeof buildSidecarQualityMetrics>;
  meetingDiagnostics: MeetingDiagnostics | null;
  qualityGateActive: boolean;
  loading: boolean;
  emptyTitle: string;
  emptyBody: string;
  contextCards: SidecarDisplayCard[];
  manualCards: SidecarDisplayCard[];
  showAll: boolean;
  canShowMore: boolean;
  onToggleShowAll: () => void;
  expandedCards: Set<string>;
  showDebugMetadata: boolean;
  onToggle: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
  contract: MeetingContractPayload | null;
  energyFrame: EnergyFrame | null;
  showBrief: boolean;
  briefMarkdown: string;
  currentCount: number;
  contextCount: number;
  onCopyBrief: () => void;
}) {
  return (
    <section className="context-pane post-call-output-pane" id="recall" aria-label="Meeting Output">
      <div className="field-toolbar context-toolbar">
        <div>
          <p className="label">Post-call</p>
          <h1>Meeting Output</h1>
        </div>
        <div className="field-toolbar-status">
          <span>{usefulCount} current</span>
          <span>{qualityGateActive ? "Quality gate on" : "Quality gate off"}</span>
        </div>
      </div>
      <ContractActiveBadge contract={contract} />
      <EnergyLensBadge frame={energyFrame} />
      <SessionContextPane
        groups={groups}
        usefulCount={usefulCount}
        rawCount={rawCount}
        qualityMetrics={qualityMetrics}
        meetingDiagnostics={meetingDiagnostics}
        qualityGateActive={qualityGateActive}
        captureActive={false}
        loading={loading}
        emptyTitle={emptyTitle}
        emptyBody={emptyBody}
        contextCards={contextCards}
        manualCards={manualCards}
        showAll={showAll}
        canShowMore={canShowMore}
        onToggleShowAll={onToggleShowAll}
        expandedCards={expandedCards}
        showDebugMetadata={showDebugMetadata}
        onToggle={onToggle}
        onJumpToEvidence={onJumpToEvidence}
        onPin={onPin}
        onDismiss={onDismiss}
        onCopy={onCopy}
      />
      {showBrief && (
        <ConsultingBriefPane
          markdown={briefMarkdown}
          currentCount={currentCount}
          contextCount={contextCount}
          onCopy={onCopyBrief}
        />
      )}
    </section>
  );
}

function StatusTile({
  label,
  value,
  detail,
  tone = "normal",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "normal" | "good" | "warning";
}) {
  return (
    <article className={`status-tile ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function CaptureControlBar({
  statusSummary,
  warning,
}: {
  statusSummary: string;
  warning: string;
}) {
  return (
    <section className={`capture-control-bar ${warning ? "warning" : ""}`} aria-label="Capture controls">
      <div className="capture-control-summary" aria-label="Selected capture mode">
        <strong>{statusSummary}</strong>
        {warning && <span>{warning}</span>}
      </div>
    </section>
  );
}

function MeetingFocusControl({
  value,
  expanded,
  activeContract,
  captureActive,
  webContextConfigured,
  webContextGlobalEnabled,
  onChange,
  onEdit,
  onDone,
}: {
  value: MeetingFocusState;
  expanded: boolean;
  activeContract: MeetingContractPayload | null;
  captureActive: boolean;
  webContextConfigured: boolean;
  webContextGlobalEnabled: boolean;
  onChange: (value: MeetingFocusState) => void;
  onEdit: () => void;
  onDone: () => void;
}) {
  const summaryContract = captureActive && activeContract ? activeContract : buildMeetingContractPayload(value);
  const reminderSummary = captureActive && activeContract
    ? `${activeContract.reminders.length} ${pluralize("reminder", activeContract.reminders.length)}`
    : summarizeMeetingReminders(value);
  const goalSummary = captureActive && activeContract
    ? compactContractGoal(activeContract.goal)
    : compactFocusGoal(value.goal);
  return (
    <section className={`meeting-focus-control ${expanded ? "expanded" : ""}`} aria-label="Meeting Focus">
      <div className="meeting-focus-bar">
        <div className="meeting-focus-summary" aria-label="Meeting Focus summary">
          <span>Focus</span>
          <strong>{capitalize(summaryContract.mode)}</strong>
          <span>{webContextSummary(value.webContextMode, webContextConfigured, webContextGlobalEnabled)}</span>
          <span>{reminderSummary}</span>
          <p>{goalSummary}</p>
        </div>
        <button
          type="button"
          className="secondary compact-button"
          aria-expanded={expanded}
          disabled={captureActive}
          onClick={expanded ? onDone : onEdit}
        >
          {expanded ? "Done" : "Edit"}
        </button>
      </div>

      {captureActive && (
        <p className="meeting-focus-note">Active contract is locked for this session.</p>
      )}

      {expanded && !captureActive && (
        <MeetingFocusEditor
          value={value}
          webContextConfigured={webContextConfigured}
          webContextGlobalEnabled={webContextGlobalEnabled}
          onChange={onChange}
          onDone={onDone}
        />
      )}
    </section>
  );
}

function MeetingFocusEditor({
  value,
  webContextConfigured,
  webContextGlobalEnabled,
  onChange,
  onDone,
}: {
  value: MeetingFocusState;
  webContextConfigured: boolean;
  webContextGlobalEnabled: boolean;
  onChange: (value: MeetingFocusState) => void;
  onDone: () => void;
}) {
  const webModes: { mode: WebContextMode; label: string }[] = [
    { mode: "default", label: webContextGlobalEnabled ? "Default on" : "Default off" },
    { mode: "on", label: "On" },
    { mode: "off", label: "Off" },
  ];
  return (
    <div className="meeting-focus-editor">
      <label className="field meeting-focus-goal">
        <span>Meeting Focus</span>
        <textarea
          aria-label="Meeting Focus goal"
          value={value.goal}
          onChange={(event) => onChange({ ...value, goal: event.target.value })}
          placeholder="What should Sidecar help offload? owners, questions, risks, follow-ups..."
        />
      </label>
      <div className="meeting-focus-options">
        <div className="mode-control compact meeting-mode-control" role="group" aria-label="Meeting focus mode">
          {(["quiet", "balanced", "assertive"] as MeetingMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              aria-pressed={value.mode === mode}
              className={value.mode === mode ? "active" : ""}
              onClick={() => onChange({ ...value, mode })}
            >
              <span>{capitalize(mode)}</span>
            </button>
          ))}
        </div>
        <div className="meeting-web-context-control">
          <span>Brave web</span>
          <div className="mode-control compact" role="group" aria-label="Brave web context">
            {webModes.map(({ mode, label }) => (
              <button
                key={mode}
                type="button"
                aria-pressed={value.webContextMode === mode}
                className={value.webContextMode === mode ? "active" : ""}
                disabled={mode === "on" && !webContextConfigured}
                onClick={() => onChange({ ...value, webContextMode: mode })}
              >
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="meeting-reminder-grid" aria-label="Meeting focus reminders">
          {MEETING_REMINDER_OPTIONS.map((option) => (
            <label key={option.key} className="toggle-row">
              <input
                type="checkbox"
                checked={value.reminders[option.key]}
                onChange={(event) => onChange({
                  ...value,
                  reminders: { ...value.reminders, [option.key]: event.target.checked },
                })}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
        <div className="meeting-focus-editor-actions">
          <button type="button" className="primary subtle" onClick={onDone}>Done</button>
        </div>
      </div>
    </div>
  );
}

function ContractActiveBadge({ contract }: { contract: MeetingContractPayload | null }) {
  if (!contract) {
    return null;
  }
  return (
    <details className="contract-badge contract-active-badge" aria-label="Contract active">
      <summary>
        <span>Contract active · {capitalize(contract.mode)} · {contract.reminders.length} {pluralize("reminder", contract.reminders.length)}</span>
      </summary>
      <div className="contract-details">
        <p>{contract.goal}</p>
        <div className="chip-row">
          {contract.reminders.map((reminder) => <span key={reminder} className="chip">{reminder}</span>)}
        </div>
      </div>
    </details>
  );
}

function EnergyLensBadge({ frame }: { frame: EnergyFrame | null }) {
  if (!frame?.active || frame.confidence === "low") {
    return null;
  }
  const categoryLabel = frame.categories.slice(0, 2).join(" + ");
  return (
    <details className="contract-badge energy-lens-badge" aria-label="Energy lens">
      <summary>
        <span>Energy lens{categoryLabel ? ` · ${categoryLabel}` : ""}</span>
      </summary>
      <div className="contract-details">
        <p>Detected from current transcript evidence.</p>
        <div className="chip-row">
          <span className="chip">{frame.confidence}</span>
          {frame.categories.slice(0, 3).map((category) => <span key={category} className="chip">{category}</span>)}
          {frame.keywords.slice(0, 6).map((keyword) => <span key={keyword} className="chip">{keyword}</span>)}
        </div>
        {frame.evidenceQuote && <p className="energy-lens-evidence">{frame.evidenceQuote}</p>}
      </div>
    </details>
  );
}

function LivePlaybackCue({
  durationSeconds,
  elapsedSeconds,
  progress,
  remainingSeconds,
  message,
  paused,
}: {
  durationSeconds: number;
  elapsedSeconds: number;
  progress: number;
  remainingSeconds: number;
  message: string;
  paused: boolean;
}) {
  return (
    <section className="live-playback-cue" aria-label="Recorded audio playback">
      <div className="live-playback-head">
        <div>
          <span className="label">Recorded audio</span>
          <strong>{paused ? "Paused" : "Playing through Live"}</strong>
        </div>
        <span>
          {formatPlaybackClock(elapsedSeconds)} / {formatPlaybackClock(durationSeconds)}
        </span>
      </div>
      <div className="live-playback-bar" aria-label={`Playback progress ${Math.round(progress)}%`}>
        <span style={{ width: `${progress}%` }} />
      </div>
      <p>{message}</p>
      <small>{formatPlaybackClock(remainingSeconds)} remaining · {paused ? "resume to continue feeding audio" : "transcript and cards appear as ASR windows complete"}</small>
    </section>
  );
}

function LiveFieldPane({
  rows,
  expandedCards,
  highlightedSourceIds,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onPin,
  onDismiss,
  onCopy,
  active,
}: {
  rows: LiveFieldRow[];
  expandedCards: Set<string>;
  highlightedSourceIds: Set<string>;
  showDebugMetadata: boolean;
  onToggle: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
  active: boolean;
}) {
  if (rows.length === 0) {
    return (
      <div className="live-empty-hint" aria-label="Live field empty state">
        {active
          ? "Listening... grounded cards will appear when there is enough evidence."
          : "Ready. Start listening or ask Sidecar."}
      </div>
    );
  }
  return (
    <>
      {rows.map((row) => (
        <article
          key={row.id}
          className={`field-pair-row ${row.transcript ? "" : "unpaired"} ${row.cards.length ? "has-sidecar" : "transcript-only"}`}
          aria-label={row.transcript ? "Transcript and sidecar row" : "Sidecar context row"}
        >
          {row.transcript && (
            <div className="field-lane transcript-lane">
              <TranscriptBubble event={row.transcript} highlighted={transcriptHighlighted(row.transcript, highlightedSourceIds)} />
            </div>
          )}
          {row.cards.length > 0 && (
            <div className="field-lane backend-lane">
              {row.cards.map((card) => (
                <ContextCard
                  key={`live-${card.id}`}
                  card={card}
                  expanded={expandedCards.has(card.id)}
                  showDebugMetadata={showDebugMetadata}
                  onToggle={() => onToggle(card.id)}
                  onJumpToEvidence={() => onJumpToEvidence(card)}
                  onPin={() => onPin(card)}
                  onDismiss={() => onDismiss(card)}
                  onCopy={() => onCopy(card)}
                />
              ))}
            </div>
          )}
        </article>
      ))}
    </>
  );
}

function TranscriptBubble({ event, highlighted = false }: { event: TranscriptEvent; highlighted?: boolean }) {
  const label = getTranscriptDisplayLabel(event);
  return (
    <article
      className={`transcript-bubble ${event.source} ${event.isFinal ? "final" : "interim"} ${highlighted ? "evidence-highlight" : ""}`}
      aria-label={`${label} transcript item`}
      data-transcript-id={event.id}
      data-segment-id={event.segmentId}
      data-source-segment-ids={(event.sourceSegmentIds ?? []).join(" ")}
    >
      <div className="transcript-bubble-head">
        <span>{label}</span>
        <div>
          {!event.isFinal && <small>Preview</small>}
          <time>{formatClock(event.at)}</time>
        </div>
      </div>
      <p>{event.text}</p>
    </article>
  );
}

function transcriptHighlighted(event: TranscriptEvent, highlightedSourceIds: Set<string>): boolean {
  return highlightedSourceIds.has(event.id)
    || Boolean(event.segmentId && highlightedSourceIds.has(event.segmentId))
    || (event.sourceSegmentIds ?? []).some((segmentId) => highlightedSourceIds.has(segmentId));
}

function SessionContextPane({
  groups,
  usefulCount,
  rawCount,
  qualityMetrics,
  meetingDiagnostics,
  qualityGateActive,
  captureActive,
  loading,
  emptyTitle,
  emptyBody,
  contextCards,
  manualCards,
  showAll,
  canShowMore,
  onToggleShowAll,
  expandedCards,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onPin,
  onDismiss,
  onCopy,
}: {
  groups: Record<MeetingOutputBucket, SidecarDisplayCard[]>;
  usefulCount: number;
  rawCount: number;
  qualityMetrics: ReturnType<typeof buildSidecarQualityMetrics>;
  meetingDiagnostics: MeetingDiagnostics | null;
  qualityGateActive: boolean;
  captureActive: boolean;
  loading: boolean;
  emptyTitle: string;
  emptyBody: string;
  contextCards: SidecarDisplayCard[];
  manualCards: SidecarDisplayCard[];
  showAll: boolean;
  canShowMore: boolean;
  onToggleShowAll: () => void;
  expandedCards: Set<string>;
  showDebugMetadata: boolean;
  onToggle: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
}) {
  const cardCount = MEETING_OUTPUT_SECTIONS.reduce((count, section) => count + groups[section.bucket].length, 0);
  const visibleSections = MEETING_OUTPUT_SECTIONS.filter((section) => groups[section.bucket].length > 0);
  const hasSupplementalCards = contextCards.length > 0 || manualCards.length > 0;
  const quietTitle = captureActive && cardCount === 0 ? "Silent by design" : emptyTitle;
  const quietBody = captureActive && cardCount === 0
    ? "No grounded cards yet. Unsupported/noisy cards are being suppressed."
    : emptyBody;
  return (
    <div className="context-card-list session-context-list scroll" aria-label="Meeting output">
      <QualityStrip
        metrics={qualityMetrics}
        diagnostics={meetingDiagnostics}
        rawCount={rawCount}
        qualityGateActive={qualityGateActive}
        showDebugMetadata={showDebugMetadata}
      />

      {loading && cardCount === 0 && !hasSupplementalCards && (
        <ContextQuietEmpty title="Searching" body="Looking for useful work context." />
      )}

      {!loading && cardCount === 0 && !hasSupplementalCards && (
        <ContextQuietEmpty title={quietTitle} body={quietBody} />
      )}

      {visibleSections.map((section) => (
        <section key={section.bucket} className="work-note-section meeting-output-section" aria-label={`${section.title} summary`}>
          <div className="work-note-section-head">
            <h2>{section.title}</h2>
            <span>{groups[section.bucket].length}</span>
          </div>
          <div className="context-preview-list">
            {groups[section.bucket].map((card) => (
              <ContextCard
                key={`meeting-${section.bucket}-${card.id}`}
                card={card}
                compact
                expanded={expandedCards.has(card.id)}
                showDebugMetadata={showDebugMetadata}
                onToggle={() => onToggle(card.id)}
                onJumpToEvidence={() => onJumpToEvidence(card)}
                onPin={() => onPin(card)}
                onDismiss={() => onDismiss(card)}
                onCopy={() => onCopy(card)}
              />
            ))}
          </div>
        </section>
      ))}

      {canShowMore && (
        <button className="secondary show-more-context" onClick={onToggleShowAll}>
          {showAll ? "Show less" : "Show more"}
        </button>
      )}

      {contextCards.length > 0 && (
        <details className="session-context-preview context-output reference-context-output" aria-label="Reference context">
          <summary>
            <span>Reference context</span>
            <strong>{contextCards.length}</strong>
          </summary>
          <div className="context-preview-list">
            {contextCards.map((card) => (
              <ContextCard
                key={`context-${card.id}`}
                card={card}
                compact
                expanded={expandedCards.has(card.id)}
                showDebugMetadata={showDebugMetadata}
                onToggle={() => onToggle(card.id)}
                onJumpToEvidence={() => onJumpToEvidence(card)}
                onPin={() => onPin(card)}
                onDismiss={() => onDismiss(card)}
                onCopy={() => onCopy(card)}
              />
            ))}
          </div>
        </details>
      )}

      {manualCards.length > 0 && (
        <details className="session-context-preview manual-output" aria-label="Manual query cards">
          <summary>
            <span>Manual query</span>
            <strong>{manualCards.length}</strong>
          </summary>
          <div className="context-preview-list">
            {manualCards.map((card) => (
              <ContextCard
                key={`manual-${card.id}`}
                card={card}
                compact
                expanded={expandedCards.has(card.id)}
                showDebugMetadata={showDebugMetadata}
                onToggle={() => onToggle(card.id)}
                onJumpToEvidence={() => onJumpToEvidence(card)}
                onPin={() => onPin(card)}
                onDismiss={() => onDismiss(card)}
                onCopy={() => onCopy(card)}
              />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function QualityStrip({
  metrics,
  diagnostics,
  rawCount,
  qualityGateActive,
  showDebugMetadata,
}: {
  metrics: ReturnType<typeof buildSidecarQualityMetrics>;
  diagnostics: MeetingDiagnostics | null;
  rawCount: number;
  qualityGateActive: boolean;
  showDebugMetadata: boolean;
}) {
  return (
    <details className="quality-strip-details" aria-label="Meeting output details">
      <summary>Details</summary>
      <section className="quality-strip" aria-label="Sidecar quality status">
        <span><strong>{metrics.acceptedCurrentCount}</strong> current</span>
        <span><strong>{metrics.contextCount}</strong> context</span>
        <span><strong>{metrics.manualCount}</strong> manual</span>
        {diagnostics && (
          <>
            <span><strong>{diagnostics.generated_candidate_count}</strong> generated</span>
            <span><strong>{diagnostics.suppressed_count}</strong> suppressed</span>
          </>
        )}
        <span><strong>{formatPercent(metrics.evidenceQuoteCoverage)}</strong> evidence</span>
        <span><strong>{formatPercent(metrics.sourceIdCoverage)}</strong> sources</span>
        <span>{qualityGateActive ? "Quality gate active" : "Quality gate off"}</span>
        <span className="quiet">{rawCount} raw</span>
        {showDebugMetadata && diagnostics?.top_suppression_reasons.length ? (
          <span className="quiet">Reasons: {diagnostics.top_suppression_reasons.map((item) => `${item.reason} ${item.count}`).join("; ")}</span>
        ) : null}
      </section>
    </details>
  );
}

function ConsultingBriefPane({
  markdown,
  currentCount,
  contextCount,
  onCopy,
}: {
  markdown: string;
  currentCount: number;
  contextCount: number;
  onCopy: () => void;
}) {
  return (
    <section className="consulting-brief" aria-label="Consulting brief">
      <div className="consulting-brief-head">
        <div>
          <p className="label">Post-call</p>
          <h2>Consulting brief</h2>
        </div>
        <button className="secondary" type="button" onClick={onCopy}>Copy Markdown</button>
      </div>
      <p className="tool-note">
        Built from {currentCount} current-meeting cards and {contextCount} reference/web context cards.
      </p>
      <pre>{markdown}</pre>
    </section>
  );
}

function ManualSourceSections({ cards }: { cards: SidecarDisplayCard[] }) {
  if (cards.length === 0) {
    return null;
  }
  const isTechnicalReference = (card: SidecarDisplayCard) => (
    card.sourceType === "local_file" || card.sourceType === "document_chunk" || card.sourceType === "file"
  );
  const sections = [
    { title: "Company refs", cards: cards.filter(isCompanyReferenceCard) },
    { title: "Technical references", cards: cards.filter(isTechnicalReference) },
    { title: "Prior transcript", cards: cards.filter((card) => !isCompanyReferenceCard(card) && !isTechnicalReference(card) && (card.sourceType === "saved_transcript" || card.sourceType === "session" || card.category === "memory")) },
    { title: "Web", cards: cards.filter((card) => card.category === "web" || card.sourceType === "brave_web") },
    { title: "Suggested meeting contribution", cards: cards.filter((card) => card.category === "contribution") },
  ].filter((section) => section.cards.length > 0);
  if (sections.length === 0) {
    return null;
  }
  return (
    <section className="manual-source-sections" aria-label="Manual query source sections">
      {sections.map((section) => (
        <div key={section.title}>
          <strong>{section.title}</strong>
          <span>{section.cards.length}</span>
        </div>
      ))}
    </section>
  );
}

function ContextCard({
  card,
  compact = false,
  expanded,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onPin,
  onDismiss,
  onCopy,
}: {
  card: SidecarDisplayCard;
  compact?: boolean;
  expanded: boolean;
  showDebugMetadata: boolean;
  onToggle: () => void;
  onJumpToEvidence: () => void;
  onPin: () => void;
  onDismiss: () => void;
  onCopy: () => void;
}) {
  const hasEvidence = Boolean(card.evidenceQuote?.trim());
  const hasSourceSegments = (card.sourceSegmentIds?.length ?? 0) > 0;
  const hasNormalDetails = Boolean(
    hasEvidence
    || hasSourceSegments
    || card.whyRelevant
    || card.missingInfo
    || (card.sources && card.sources.length > 0)
    || (card.citations && card.citations.length > 0),
  );
  const hasDebugDetails = showDebugMetadata && Boolean(card.rawText || card.debugMetadata);
  const hasDetails = hasNormalDetails || hasDebugDetails;
  const suggestion = card.suggestedSay ?? card.suggestedAsk;
  const sourceLabel = card.sourceLabel ?? card.source;
  const qualityLabel = card.qualityLabel ?? "";
  return (
    <article
      className={`context-card field-bubble ${compact ? "compact" : ""} ${card.provisional ? "provisional" : ""} ${confidenceClass(card)} ${card.category}`}
      aria-label={`${categoryLabel(card.category)} context card`}
    >
      <div className="context-card-head">
        <span>{categoryLabel(card.category)}</span>
        <div className="card-actions">
          <time>{formatClock(card.at)}</time>
          <button type="button" className="icon-button" aria-label={card.pinned ? "Unpin card" : "Pin card"} onClick={onPin}>
            {card.pinned ? "Pinned" : "Pin"}
          </button>
          <button type="button" className="icon-button" aria-label="Dismiss card" onClick={onDismiss}>Dismiss</button>
        </div>
      </div>
      <h3>{card.title}</h3>
      <div className="context-card-meta">
        {sourceLabel && <span className="source-chip">{sourceLabel}</span>}
        {qualityLabel && <span className="quality-chip">{qualityLabel}</span>}
        {card.owner && <span>Owner: {card.owner}</span>}
        {card.dueDate && <span>Due: {card.dueDate}</span>}
        {card.date && <span>{card.date}</span>}
      </div>
      <p>{card.summary}</p>
      {suggestion && (
        <div className="suggestion-row">
          <strong>{card.suggestedAsk ? "Ask this" : "Say this"}</strong>
          <span>{suggestion}</span>
          <button type="button" className="link-button" onClick={onCopy}>Copy</button>
        </div>
      )}
      {card.whyRelevant && <p className="context-card-why">{card.whyRelevant}</p>}
      {hasDetails && (
        <>
          <div className="card-detail-actions">
            <button className="link-button" onClick={onToggle}>
              {expanded ? "Hide details" : hasEvidence ? "Evidence" : "Details"}
            </button>
            {hasSourceSegments && (
              <button className="link-button" onClick={onJumpToEvidence}>
                Jump to source{card.sourceSegmentIds!.length > 1 ? ` (${card.sourceSegmentIds!.length})` : ""}
              </button>
            )}
          </div>
          {expanded && (
            <div className="field-details">
              {hasEvidence && (
                <blockquote className="evidence-quote" aria-label={`Evidence for ${card.title}`}>
                  {card.evidenceQuote}
                </blockquote>
              )}
              {hasSourceSegments && (
                <div className="source-segment-list" aria-label={`Source segments for ${card.title}`}>
                  {card.sourceSegmentIds!.slice(0, 6).map((segmentId) => (
                    <button key={segmentId} type="button" className="source-segment-chip" onClick={onJumpToEvidence}>
                      {segmentId}
                    </button>
                  ))}
                </div>
              )}
              {card.whyRelevant && (
                <p className="detail-note"><strong>Why now:</strong> {card.whyRelevant}</p>
              )}
              {card.missingInfo && (
                <p className="missing-info"><strong>Missing info:</strong> {card.missingInfo}</p>
              )}
              {card.sources && card.sources.length > 0 && (
                <div className="note-sources" aria-label={`Sources for ${card.title}`}>
                  {card.sources.map((source) => (
                    source.url ? (
                      <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
                        {source.title}
                      </a>
                    ) : (
                      <small key={source.path ?? source.title}>{source.title}</small>
                    )
                  ))}
                </div>
              )}
              {card.citations && card.citations.length > 0 && (
                <div className="citation-list">
                  {card.citations.slice(0, 3).map((citation) => <small key={citation}>{citation}</small>)}
                </div>
              )}
              {hasDebugDetails && (
                <details className="debug-details" open>
                  <summary>Debug metadata</summary>
                  <dl>
                    {debugEntries(card).map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>{formatDebugValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                  {card.rawText && <pre>{card.rawText}</pre>}
                </details>
              )}
            </div>
          )}
        </>
      )}
    </article>
  );
}

function displayCardForSidecarPayload(payload: Record<string, unknown>, envelope: EventEnvelope): SidecarDisplayCard | null {
  const title = String(payload.title ?? "").trim();
  const body = String(payload.body ?? "").trim();
  if (!title || !body) {
    return null;
  }
  const category = sidecarCategoryFromPayload(payload.category);
  const at = payloadTimeMs(payload.created_at ?? envelope.at) ?? Date.now();
  const sourceSegmentIds = parseStringList(payload.source_segment_ids);
  const sourceType = String(payload.source_type ?? "transcript");
  const explicitlyRequested = Boolean(payload.explicitly_requested) || String(payload.card_key ?? "").startsWith("manual:");
  const cardKey = payload.card_key ? String(payload.card_key) : undefined;
  const suggestionSay = stringOrUndefined(payload.suggested_say);
  const suggestionAsk = stringOrUndefined(payload.suggested_ask);
  const confidence = numberOrUndefined(payload.confidence);
  const draftCard = {
    sourceType,
    sourceSegmentIds,
    evidenceQuote: stringOrUndefined(payload.evidence_quote),
    explicitlyRequested,
    confidence,
    category,
  } as SidecarDisplayCard;
  return {
    id: `sidecar-${String(payload.id)}`,
    cardKey,
    category,
    title: readableTitle(title, categoryLabel(category)),
    summary: body,
    whyRelevant: stringOrUndefined(payload.why_now),
    source: sourceLabel(sourceType, explicitlyRequested),
    sourceType,
    sourceLabel: sourceLabel(sourceType, explicitlyRequested),
    qualityLabel: qualityLabelForCard(draftCard),
    at,
    score: confidence,
    confidence,
    rawText: body,
    sourceSegmentIds,
    evidenceQuote: stringOrUndefined(payload.evidence_quote),
    owner: stringOrUndefined(payload.owner),
    dueDate: stringOrUndefined(payload.due_date),
    missingInfo: stringOrUndefined(payload.missing_info),
    suggestedSay: suggestionSay,
    suggestedAsk: suggestionAsk,
    priority: priorityFromValue(payload.priority),
    sources: parseNoteSources(payload.sources),
    citations: parseStringList(payload.citations),
    ephemeral: Boolean(payload.ephemeral),
    expiresAt: payloadTimeMs(payload.expires_at),
    provisional: Boolean(payload.provisional),
    explicitlyRequested,
    debugMetadata: {
      sourceType,
      sourceId: payload.id,
      backendLabel: payload.category,
      confidence: payload.confidence,
      cardKey,
      sourceSegmentIds,
      sanitizedQuery: payload.sanitized_query,
      freshness: payload.freshness,
      rawAudioRetained: payload.raw_audio_retained,
      evidenceQuote: payload.evidence_quote,
      owner: payload.owner,
      dueDate: payload.due_date,
      missingInfo: payload.missing_info,
    },
  };
}

function transcriptEventFromPayload(payload: Record<string, unknown>, envelope: EventEnvelope): TranscriptEvent {
  const source = transcriptSourceFromPayload(payload.source);
  const at = payloadTimeMs(payload.created_at ?? envelope.at) ?? Date.now();
  return {
    id: `transcript-${String(payload.id)}`,
    segmentId: String(payload.id),
    replacesSegmentId: stringOrUndefined(payload.replaces_segment_id),
    sourceSegmentIds: parseStringList(payload.source_segment_ids),
    text: String(payload.text ?? ""),
    at,
    source,
    speakerRole: speakerRoleFromPayload(payload.speaker_role),
    speakerLabel: payload.speaker_label ? String(payload.speaker_label) : undefined,
    speakerConfidence: numberOrUndefined(payload.speaker_confidence),
    startedAt: payloadTimeMs(payload.started_at ?? payload.start_s),
    endedAt: payloadTimeMs(payload.ended_at ?? payload.end_s),
    isFinal: payload.is_final !== false,
  };
}

function typedTranscriptEvent(text: string): TranscriptEvent {
  const at = Date.now();
  return {
    id: `typed-${at}`,
    segmentId: `typed-${at}`,
    text,
    at,
    source: "typed",
    speakerRole: "user",
    speakerConfidence: 1,
    isFinal: true,
  };
}

function displayCardForNote(note: NoteCard, options: BackendItemOptions = {}): SidecarDisplayCard {
  const sourceSegmentIds = options.sourceSegmentIds ?? note.source_segment_ids;
  const sourceType = note.source_type ?? "transcript";
  const explicitlyRequested = options.origin === "manual";
  const category = note.ephemeral || note.source_type === "brave_web"
    ? "web"
    : sidecarCategoryFromPayload(["action", "decision", "question", "risk", "clarification", "contribution"].includes(note.kind) ? note.kind : "note");
  const at = payloadTimeMs(note.created_at) ?? Date.now();
  const confidence = note.confidence;
  const draftCard = {
    sourceType,
    sourceSegmentIds,
    evidenceQuote: note.evidence_quote,
    explicitlyRequested,
    confidence,
    category,
  } as SidecarDisplayCard;
  return {
    id: `note-${note.id}-${options.origin ?? "live"}`,
    cardKey: `note:${category}:${note.title.toLowerCase().trim()}`,
    category,
    title: readableTitle(note.title, category === "web" ? "Web context" : "Note"),
    summary: note.body,
    whyRelevant: noteReason(note, options.origin),
    at,
    rawText: note.body,
    sourceSegmentIds,
    evidenceQuote: note.evidence_quote,
    owner: note.owner,
    dueDate: note.due_date,
    missingInfo: note.missing_info,
    source: sourceLabel(sourceType, explicitlyRequested),
    sourceType,
    sourceLabel: sourceLabel(sourceType, explicitlyRequested),
    qualityLabel: qualityLabelForCard(draftCard),
    confidence,
    score: confidence,
    explicitlyRequested,
    priority: priorityFromValue(note.priority ?? (note.kind === "action" || note.kind === "decision" ? "high" : "normal")),
    suggestedSay: note.suggested_say,
    suggestedAsk: note.suggested_ask,
    sources: note.sources,
    citations: note.citations,
    debugMetadata: {
      sourceType,
      sourceId: note.id,
      backendLabel: note.kind,
      sourceSegmentIds,
      evidenceQuote: note.evidence_quote,
      owner: note.owner,
      dueDate: note.due_date,
      missingInfo: note.missing_info,
    },
  };
}

function displayCardForRecall(hit: RecallHit, options: BackendItemOptions = {}): SidecarDisplayCard {
  const sourceSegmentIds = options.sourceSegmentIds ?? hit.source_segment_ids ?? [];
  const sourceType = hit.source_type;
  const explicitlyRequested = options.origin === "manual" || Boolean(hit.explicitly_requested ?? hit.metadata.explicitly_requested);
  const title = readableTitle(String(hit.metadata.title ?? ""), "Relevant memory");
  const at = Date.now();
  const confidence = hit.score;
  const draftCard = {
    sourceType,
    sourceSegmentIds,
    explicitlyRequested,
    confidence,
    category: "memory",
  } as SidecarDisplayCard;
  return {
    id: `recall-${sourceType}-${hit.source_id}-${at}`,
    cardKey: `recall:${sourceType}:${hit.source_id}`,
    category: "memory",
    title,
    summary: hit.text,
    whyRelevant: recallReason(hit, options.origin),
    source: sourceLabel(sourceType, explicitlyRequested),
    sourceType,
    sourceLabel: sourceLabel(sourceType, explicitlyRequested),
    qualityLabel: qualityLabelForCard(draftCard),
    at,
    score: hit.score,
    confidence: hit.score,
    rawText: hit.text,
    sourceSegmentIds,
    pinned: Boolean(hit.pinned ?? hit.metadata.pinned),
    explicitlyRequested,
    priority: priorityFromValue(hit.priority ?? hit.metadata.priority),
    debugMetadata: {
      sourceType,
      sourceId: hit.source_id,
      score: hit.score,
      retrievalReason: hit.metadata.reason ?? hit.metadata.retrieval_reason,
      metadata: hit.metadata,
      sourceSegmentIds,
    },
  };
}

function areCardsEquivalent(left: SidecarDisplayCard, right: SidecarDisplayCard): boolean {
  const leftKey = left.cardKey ?? "";
  const rightKey = right.cardKey ?? "";
  if (leftKey && rightKey && leftKey === rightKey) {
    return true;
  }
  const leftArtifact = String(left.debugMetadata?.sourceId ?? "");
  const rightArtifact = String(right.debugMetadata?.sourceId ?? "");
  const leftSource = String(left.debugMetadata?.sourceType ?? "");
  const rightSource = String(right.debugMetadata?.sourceType ?? "");
  if (leftArtifact && rightArtifact && leftArtifact === rightArtifact && leftSource === rightSource) {
    return true;
  }
  const leftText = normalizeDisplayText(`${left.title} ${left.summary}`);
  const rightText = normalizeDisplayText(`${right.title} ${right.summary}`);
  return Boolean(leftText && rightText && leftText === rightText);
}

function categoryLabel(category: SidecarDisplayCard["category"]): string {
  if (category === "web") return "Web";
  if (category === "action") return "Follow-up";
  if (category === "decision") return "Decision";
  if (category === "question") return "Ask this";
  if (category === "risk") return "Watch risk";
  if (category === "clarification") return "Clarify";
  if (category === "contribution") return "Say this";
  if (category === "status") return "Status";
  if (category === "memory") return "Memory";
  if (category === "reference") return "Reference";
  return "Note";
}

function confidenceClass(card: SidecarDisplayCard): string {
  const confidence = card.confidence ?? card.score;
  if (confidence === undefined) {
    return "";
  }
  if (confidence >= 0.82) {
    return "high-confidence";
  }
  if (confidence < 0.58) {
    return "low-confidence";
  }
  return "";
}

function debugEntries(card: SidecarDisplayCard): [string, unknown][] {
  const metadata = card.debugMetadata ?? {};
  const entries: [string, unknown][] = [
    ["source type", metadata.sourceType],
    ["source ID", metadata.sourceId],
    ["score", metadata.score ?? card.score],
    ["confidence", metadata.confidence ?? card.confidence],
    ["retrieval reason", metadata.retrievalReason],
    ["backend label", metadata.backendLabel],
    ["card key", metadata.cardKey ?? card.cardKey],
    ["source segments", metadata.sourceSegmentIds],
  ];
  return entries.filter(([, value]) => value !== undefined && value !== null && value !== "");
}

function formatDebugValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(value >= 10 ? 0 : 3).replace(/\.?0+$/, "") : String(value);
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function readableTitle(value: string, fallback: string): string {
  const title = value.trim();
  if (!title || title === "transcript_segment") {
    return fallback;
  }
  return title.replace(/_/g, " ");
}

function noteReason(note: NoteCard, origin?: BackendItemOptions["origin"]): string {
  if (note.why_now?.trim()) {
    return note.why_now.trim();
  }
  if (origin === "manual") {
    return "Returned for the typed search.";
  }
  if (note.ephemeral || note.source_type === "brave_web") {
    return "Current web context for the conversation.";
  }
  if (note.kind === "action") {
    return "Actionable note from recent speech.";
  }
  return "Useful note from recent speech.";
}

function recallReason(hit: RecallHit, origin?: BackendItemOptions["origin"]): string {
  const reason = hit.metadata.reason ?? hit.metadata.retrieval_reason;
  if (typeof reason === "string" && reason.trim()) {
    return reason.trim();
  }
  if (origin === "manual") {
    return "Returned for the typed search.";
  }
  return "Retrieved because it overlaps with the recent transcript.";
}

function sourceLabel(sourceType: string, explicitlyRequested = false): string {
  return displaySourceLabel(sourceType, explicitlyRequested);
}

function transcriptSourceFromPayload(value: unknown): TranscriptSource {
  return String(value ?? "").toLowerCase() === "typed" ? "typed" : "microphone";
}

function speakerRoleFromPayload(value: unknown): TranscriptEvent["speakerRole"] {
  const role = String(value ?? "").toLowerCase();
  if (role === "user") return "user";
  if (role === "other" || role === "assistant" || role === "speaker") return "other";
  return "unknown";
}

function priorityFromValue(value: unknown): SidecarPriority {
  const priority = String(value ?? "normal").toLowerCase();
  if (priority === "high" || priority === "low") {
    return priority;
  }
  return "normal";
}

function sidecarCategoryFromPayload(value: unknown): SidecarDisplayCard["category"] {
  const category = String(value ?? "note").toLowerCase();
  if (
    [
      "action",
      "decision",
      "question",
      "risk",
      "clarification",
      "contribution",
      "memory",
      "reference",
      "web",
      "status",
      "note",
    ].includes(category)
  ) {
    return category as SidecarDisplayCard["category"];
  }
  return "note";
}

function numberOrUndefined(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function safeInteger(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.round(number)) : 0;
}

function safeRatio(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
}

function stringOrUndefined(value: unknown): string | undefined {
  const text = String(value ?? "").trim();
  return text ? text : undefined;
}

function arrayOfStrings(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
}

function asrBackendNameFromPayload(payload: Record<string, unknown>): string {
  void payload;
  return "Faster-Whisper";
}

function replaceTranscriptLine(
  current: TranscriptLine[],
  line: TranscriptLine,
  replacesSegmentId?: string,
): TranscriptLine[] {
  const replacementIds = [line.id, replacesSegmentId, ...(line.source_segment_ids ?? [])]
    .filter((value): value is string => Boolean(value));
  const index = current.findIndex((candidate) => (
    replacementIds.includes(candidate.id)
    || (candidate.source_segment_ids ?? []).some((segmentId) => replacementIds.includes(segmentId))
  ));
  if (index === -1) {
    return [...current, line];
  }
  const next = [...current];
  next[index] = line;
  return next;
}

function transcriptReplacementIds(event: TranscriptEvent): string[] {
  return [
    event.id,
    event.segmentId,
    event.replacesSegmentId,
    ...(event.sourceSegmentIds ?? []),
  ].filter(Boolean) as string[];
}

function captureWarningFromPayload(payload: Record<string, unknown>): { id: string; title: string; message: string } | null {
  const warning = isRecord(payload.capture_warning) ? payload.capture_warning : null;
  const reason = warning ? String(warning.reason ?? "") : "";
  if (reason) {
    return {
      id: `${reason}:${payload.silent_windows ?? 0}:${payload.asr_empty_windows ?? 0}`,
      title: reason === "audio_too_quiet" ? "Capture is too quiet" : "ASR is not hearing speech",
      message: String(warning?.message ?? "Capture is active, but ASR is not producing transcript text."),
    };
  }
  const silentWindows = Number(payload.silent_windows ?? 0);
  const emptyWindows = Number(payload.asr_empty_windows ?? 0);
  if (silentWindows >= 3) {
    return {
      id: `audio_too_quiet:${silentWindows}`,
      title: "Capture is too quiet",
      message: `Recent audio windows are below the speech threshold. Last RMS: ${formatRms(payload.last_audio_rms)}.`,
    };
  }
  if (emptyWindows >= 3) {
    return {
      id: `asr_no_spans:${emptyWindows}`,
      title: "ASR is not hearing speech",
      message: "Capture is active, but ASR has emitted no transcript spans for several windows.",
    };
  }
  return null;
}

function formatRms(value: unknown): string {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "unknown";
}

function transcriptEventsOverlap(left: TranscriptEvent, right: TranscriptEvent): boolean {
  if (left.source !== "microphone" || right.source !== "microphone") {
    return false;
  }
  if (left.startedAt === undefined || left.endedAt === undefined || right.startedAt === undefined || right.endedAt === undefined) {
    return false;
  }
  return Math.max(left.startedAt, right.startedAt) <= Math.min(left.endedAt, right.endedAt);
}

function transcriptFinalAlreadyCoversPartial(candidate: TranscriptEvent, event: TranscriptEvent): boolean {
  return candidate.isFinal
    && transcriptEventsOverlap(candidate, event)
    && transcriptTextContains(candidate.text, event.text);
}

function transcriptTextContains(left: string, right: string): boolean {
  const leftText = normalizeTranscriptText(left);
  const rightText = normalizeTranscriptText(right);
  return Boolean(leftText && rightText && leftText.includes(rightText));
}

function normalizeTranscriptText(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").replace(/[.,!?;:"']/g, "").trim();
}

function transcriptWordCount(value: string): number {
  return normalizeTranscriptText(value).split(" ").filter(Boolean).length;
}

function payloadTimeMs(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  if (typeof value === "string" && Number.isNaN(Number(value))) {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return undefined;
  }
  return numeric > 10_000_000_000 ? numeric : numeric * 1000;
}

function noteFromPayload(value: unknown): NoteCard | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const payload = value as Record<string, unknown>;
  return {
    id: String(payload.id),
    kind: String(payload.kind),
    title: String(payload.title),
    body: String(payload.body),
    source_segment_ids: parseStringList(payload.source_segment_ids),
    created_at: numberOrUndefined(payload.created_at),
    ephemeral: Boolean(payload.ephemeral),
    source_type: payload.source_type ? String(payload.source_type) : undefined,
    sources: parseNoteSources(payload.sources),
    evidence_quote: stringOrUndefined(payload.evidence_quote),
    owner: stringOrUndefined(payload.owner),
    due_date: stringOrUndefined(payload.due_date),
    missing_info: stringOrUndefined(payload.missing_info),
    confidence: numberOrUndefined(payload.confidence),
    priority: payload.priority ? String(payload.priority) : undefined,
    why_now: stringOrUndefined(payload.why_now),
    suggested_say: stringOrUndefined(payload.suggested_say),
    suggested_ask: stringOrUndefined(payload.suggested_ask),
    citations: parseStringList(payload.citations),
  };
}

function recallHitFromPayload(payload: Record<string, unknown>): RecallHit {
  return {
    source_type: String(payload.source_type),
    source_id: String(payload.source_id),
    text: String(payload.text),
    score: Number(payload.score),
    metadata: isRecord(payload.metadata) ? payload.metadata : {},
    source_segment_ids: parseStringList(payload.source_segment_ids),
    pinned: Boolean(payload.pinned),
    explicitly_requested: Boolean(payload.explicitly_requested),
    priority: payload.priority ? String(payload.priority) : undefined,
  };
}

function parseStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => String(item).trim())
    .filter(Boolean)
    .slice(0, 24);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function parseNoteSources(value: unknown): NoteSource[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item): NoteSource | null => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const record = item as Record<string, unknown>;
      const title = String(record.title ?? "").trim();
      const url = String(record.url ?? "").trim();
      const path = String(record.path ?? "").trim();
      if (!title || (!url && !path)) {
        return null;
      }
      return { title, url: url || undefined, path: path || undefined };
    })
    .filter((item): item is NoteSource => item !== null)
    .slice(0, 3);
}

function parseExpectedTerms(value: string): string[] {
  const seen = new Set<string>();
  return value
    .split(",")
    .map((term) => term.trim())
    .filter((term) => {
      const key = normalizeForMatch(term);
      if (!key || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    })
    .slice(0, 64);
}

function matchExpectedTerms(text: string, terms: string[]): string[] {
  const normalizedText = normalizeForMatch(text);
  return terms.filter((term) => normalizedText.includes(normalizeForMatch(term)));
}

function normalizeForMatch(text: string): string {
  return text
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/\bt\.?\s*a\.?\s+smith\b/g, "ta smith")
    .replace(/\bfive\s+hundred\s+(?:k\s*v|kilovolt|kv)\b/g, "500kv")
    .replace(/\b500\s*k\s*v\b/g, "500kv")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  );
}

function ContextQuietEmpty({ title, body }: { title: string; body: string }) {
  return (
    <div className="context-quiet-empty" aria-live="polite">
      <strong>{title}</strong>
      <span>{body}</span>
    </div>
  );
}

function formatClock(value: number): string {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDateTime(value: number | null | undefined): string {
  if (!value) {
    return "unknown time";
  }
  return new Date(value * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSeconds(value: number): string {
  return `${Number(value).toFixed(1)}s`;
}

function formatSecondsFromMs(value: number | undefined): string {
  return formatSeconds((value ?? 0) / 1000);
}

function formatPlaybackClock(value: number): string {
  const totalSeconds = Math.max(0, Math.round(Number.isFinite(value) ? value : 0));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function PathLabel(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? path;
}

function PathHint(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  if (parts.length <= 1) {
    return path;
  }
  return parts.slice(Math.max(0, parts.length - 3), -1).join(" / ");
}

function isAppPage(value: string | null): value is AppPage {
  return value === "live" || value === "review" || value === "sessions" || value === "tools" || value === "models" || value === "references";
}

function formatOllamaHostLabel(host?: string): string {
  if (!host) {
    return "local";
  }
  try {
    const url = new URL(host);
    if (url.hostname === "127.0.0.1" || url.hostname === "localhost") {
      return "local";
    }
    return url.port && url.port !== "11434" ? `${url.hostname}:${url.port}` : url.hostname;
  } catch {
    return host.replace(/^https?:\/\//, "").replace(/\/$/, "") || "local";
  }
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function pluralize(label: string, count: number): string {
  return count === 1 ? label : `${label}s`;
}

function compactFocusGoal(goal: string): string {
  const trimmed = goal.trim();
  return compactContractGoal(trimmed && trimmed !== DEFAULT_MEETING_GOAL
    ? trimmed
    : "Offload owners, questions, risks, follow-ups");
}

function compactContractGoal(goal: string): string {
  const trimmed = goal.trim() || "Offload owners, questions, risks, follow-ups";
  return trimmed.length > 96 ? `${trimmed.slice(0, 93).trim()}...` : trimmed;
}

function summarizeMeetingReminders(focus: MeetingFocusState): string {
  const selected = MEETING_REMINDER_OPTIONS
    .filter((option) => focus.reminders[option.key])
    .map((option) => reminderShortLabel(option.label));
  if (selected.length === 0) {
    return "No reminders";
  }
  if (selected.length <= 3) {
    return selected.join(", ");
  }
  return `${selected.length} reminders`;
}

function reminderShortLabel(label: string): string {
  if (label === "Open questions") {
    return "Questions";
  }
  if (label === "Risks/dependencies") {
    return "Risks";
  }
  if (label === "Say-this contributions") {
    return "Say-this";
  }
  if (label === "Post-call brief") {
    return "Brief";
  }
  return label;
}

function defaultApiBase(): string {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:8765";
  }
  const host = window.location.hostname;
  if (isLocalBrowserHost(host)) {
    return `http://${host}:8765`;
  }
  if (window.location.port === "8766" || window.location.port === "8776") {
    return `${window.location.protocol}//${host}:8765`;
  }
  return "";
}

function defaultAudioSource(): AudioSource {
  if (ENABLE_FIXTURE_AUDIO && DEFAULT_FIXTURE_AUDIO) {
    return "fixture";
  }
  return "server_device";
}

function defaultMicTuning(): MicTuning {
  return {
    auto_level: true,
    input_gain_db: 0,
    speech_sensitivity: "normal",
  };
}

function defaultMeetingFocus(): MeetingFocusState {
  return {
    goal: "",
    mode: "quiet",
    webContextMode: "default",
    reminders: MEETING_REMINDER_OPTIONS.reduce((result, option) => ({
      ...result,
      [option.key]: true,
    }), {} as Record<MeetingReminderKey, boolean>),
  };
}

function loadMicTuning(): MicTuning {
  if (typeof window === "undefined") {
    return defaultMicTuning();
  }
  try {
    const stored = window.localStorage.getItem(MIC_TUNING_STORAGE_KEY);
    return normalizeMicTuning(stored ? JSON.parse(stored) : undefined);
  } catch {
    return defaultMicTuning();
  }
}

function loadMeetingFocus(): MeetingFocusState {
  if (typeof window === "undefined") {
    return defaultMeetingFocus();
  }
  try {
    const stored = window.localStorage.getItem(MEETING_FOCUS_STORAGE_KEY);
    return normalizeMeetingFocus(stored ? JSON.parse(stored) : undefined);
  } catch {
    return defaultMeetingFocus();
  }
}

function normalizeMicTuning(value?: Partial<MicTuning> | null): MicTuning {
  const sensitivity = value?.speech_sensitivity;
  const inputGain = Number(value?.input_gain_db ?? 0);
  return {
    auto_level: value?.auto_level ?? true,
    input_gain_db: Math.max(-12, Math.min(12, Number.isFinite(inputGain) ? inputGain : 0)),
    speech_sensitivity: sensitivity === "quiet" || sensitivity === "noisy" ? sensitivity : "normal",
  };
}

function normalizeMeetingFocus(value?: Partial<MeetingFocusState> | null): MeetingFocusState {
  const fallback = defaultMeetingFocus();
  const mode = value?.mode === "balanced" || value?.mode === "assertive" ? value.mode : "quiet";
  const webContextMode = (
    value?.webContextMode === "on" || value?.webContextMode === "off"
      ? value.webContextMode
      : "default"
  );
  const reminders = { ...fallback.reminders };
  if (value?.reminders && typeof value.reminders === "object") {
    for (const option of MEETING_REMINDER_OPTIONS) {
      reminders[option.key] = value.reminders[option.key] !== false;
    }
  }
  return {
    goal: String(value?.goal ?? "").slice(0, 420),
    mode,
    webContextMode,
    reminders,
  };
}

function buildMeetingContractPayload(focus: MeetingFocusState): MeetingContractPayload {
  const reminders = MEETING_REMINDER_OPTIONS
    .filter((option) => focus.reminders[option.key])
    .map((option) => option.reminder);
  return {
    goal: focus.goal.trim() || DEFAULT_MEETING_GOAL,
    mode: focus.mode,
    reminders,
  };
}

function webContextEnabledForStart(focus: MeetingFocusState): boolean | undefined {
  if (focus.webContextMode === "on") {
    return true;
  }
  if (focus.webContextMode === "off") {
    return false;
  }
  return undefined;
}

function webContextSummary(
  mode: WebContextMode,
  configured: boolean,
  globalEnabled: boolean,
): string {
  if (!configured) {
    return "Brave key missing";
  }
  if (mode === "on") {
    return "Brave on";
  }
  if (mode === "off") {
    return "Brave off";
  }
  return globalEnabled ? "Brave default on" : "Brave default off";
}

function meetingContractFromPayload(value: unknown): MeetingContractPayload | null {
  if (!isRecord(value)) {
    return null;
  }
  const mode = String(value.mode ?? "quiet").toLowerCase();
  return {
    goal: String(value.goal ?? DEFAULT_MEETING_GOAL).trim() || DEFAULT_MEETING_GOAL,
    mode: mode === "balanced" || mode === "assertive" ? mode : "quiet",
    reminders: parseStringList(value.reminders),
  };
}

function meetingDiagnosticsFromPayload(value: unknown): MeetingDiagnostics | null {
  if (!isRecord(value)) {
    return null;
  }
  const reasons = Array.isArray(value.top_suppression_reasons)
    ? value.top_suppression_reasons
      .filter(isRecord)
      .map((item) => ({
        reason: String(item.reason ?? ""),
        count: Number(item.count ?? 0),
      }))
      .filter((item) => item.reason && Number.isFinite(item.count))
    : [];
  return {
    generated_candidate_count: safeInteger(value.generated_candidate_count),
    accepted_count: safeInteger(value.accepted_count),
    suppressed_count: safeInteger(value.suppressed_count),
    top_suppression_reasons: reasons,
    evidence_quote_coverage: safeRatio(value.evidence_quote_coverage),
    source_id_coverage: safeRatio(value.source_id_coverage),
  };
}

function energyFrameFromPayload(payload: Record<string, unknown>): EnergyFrame | null {
  const active = Boolean(payload.energy_lens_active);
  const confidence = energyConfidenceFromPayload(payload.energy_lens_confidence);
  if (!active || confidence === "low") {
    return null;
  }
  return {
    active,
    score: Number(payload.energy_lens_score ?? 0),
    confidence,
    categories: parseStringList(payload.energy_lens_categories),
    keywords: parseStringList(payload.energy_lens_keywords),
    evidenceSegmentIds: parseStringList(payload.energy_lens_evidence_segment_ids),
    evidenceQuote: String(payload.energy_lens_evidence_quote ?? ""),
    summaryLabel: String(payload.energy_lens_summary_label ?? "Energy lens"),
  };
}

function energyConfidenceFromPayload(value: unknown): EnergyFrame["confidence"] {
  return value === "high" || value === "medium" || value === "low" ? value : "low";
}

function capitalize(value: string): string {
  return value ? value[0].toUpperCase() + value.slice(1) : value;
}

function isLocalBrowserHost(host: string): boolean {
  return host === "127.0.0.1" || host === "localhost" || host === "::1";
}

function humanFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function micTestPlaybackUrl(payload: MicTestResult): string {
  const playback = payload.playback_audio;
  if (!playback?.data_base64) {
    return "";
  }
  try {
    const binary = window.atob(playback.data_base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return URL.createObjectURL(new Blob([bytes], { type: playback.mime_type || "audio/wav" }));
  } catch {
    return "";
  }
}

function micTestLogMetadata(payload: MicTestResult): Record<string, unknown> {
  return {
    ...payload,
    playback_audio: payload.playback_audio
      ? {
          mime_type: payload.playback_audio.mime_type,
          duration_seconds: payload.playback_audio.duration_seconds,
          data_base64_bytes: payload.playback_audio.data_base64.length,
        }
      : null,
  };
}

async function fetchJson<T>(url: string, init: RequestInit): Promise<FetchJsonResult<T>> {
  try {
    const response = await fetch(url, init);
    const payload = await response.json();
    return { ok: response.ok, payload };
  } catch (error) {
    return {
      ok: false,
      payload: { detail: error instanceof Error ? error.message : "Request failed" } as T,
    };
  }
}
