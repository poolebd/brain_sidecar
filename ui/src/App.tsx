import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  buildSidecarDisplayCards,
  getTranscriptDisplayLabel,
  groupWorkCards,
  normalizeDisplayText,
  type SidecarDisplayCard,
  type SidecarPriority,
  type TranscriptEvent,
  type TranscriptSource,
  type WorkCardBucket,
} from "./presentation";

type ThemeName = "neutral" | "midnight" | "canopy" | "ember";
type AudioSource = "server_device" | "fixture";
type SessionRetention = "temporary" | "saved";

const API_BASE = import.meta.env.VITE_API_BASE ?? defaultApiBase();
const ENABLE_FIXTURE_AUDIO = import.meta.env.VITE_ENABLE_FIXTURE_AUDIO === "1";
const DEFAULT_FIXTURE_AUDIO = import.meta.env.VITE_DEFAULT_FIXTURE_AUDIO ?? "";
const DEFAULT_THEME: ThemeName = "midnight";
const THEME_STORAGE_KEY = "brain-sidecar-theme-v2";
const DEBUG_METADATA_STORAGE_KEY = "brain-sidecar-debug-metadata-v1";
const SESSION_RETENTION_STORAGE_KEY = "brain-sidecar-session-retention-v1";
const DEFAULT_CONTEXT_CARD_LIMIT = 3;

const WORK_NOTE_SECTIONS: { bucket: WorkCardBucket; title: string; empty: string }[] = [
  { bucket: "actions", title: "Actions", empty: "No actions yet." },
  { bucket: "decisions", title: "Decisions", empty: "No decisions yet." },
  { bucket: "questions", title: "Questions", empty: "No open questions yet." },
  { bucket: "context", title: "Relevant Context", empty: "No useful context yet." },
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
  preferred?: boolean;
};

type DeviceListResponse = {
  devices: DeviceInfo[];
  preferred_device_configured?: boolean;
  preferred_device_available?: boolean;
  preferred_device_id?: string | null;
  preferred_device_match?: string | null;
  preferred_device_label?: string | null;
};

type EventEnvelope = {
  id: string;
  type: string;
  session_id: string | null;
  at: number;
  payload: Record<string, unknown>;
};

type TranscriptLine = {
  id: string;
  text: string;
  start_s: number;
  end_s: number;
  asr_model?: string;
  created_at?: number;
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

type WorkMemoryStatus = {
  projects: number;
  evidence: number;
  sources: number;
  disabled_sources: number;
  latest_index_at: number | null;
  source_groups: Record<string, number>;
  default_roots: string[];
  guardrail: string;
};

type WorkMemoryCard = {
  project_id: string;
  title: string;
  organization: string;
  date_range: string;
  score: number;
  confidence: number;
  reason: string;
  lesson: string;
  citations: string[];
  text: string;
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
  work_parallel_rows: number;
  error_count: number;
  expected_terms: string[];
  matched_expected_terms: string[];
  issues: string[];
  report_path?: string;
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
  ollama_keep_alive?: string;
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
  test_audio_run_dir?: string;
};

type PipelineState = {
  queueDepth: number;
  droppedWindows: number;
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
  const [sessionRetention, setSessionRetention] = useState<SessionRetention>(() => {
    if (typeof window === "undefined") {
      return "temporary";
    }
    const storedRetention = window.localStorage.getItem(SESSION_RETENTION_STORAGE_KEY);
    return storedRetention === "saved" ? "saved" : "temporary";
  });
  const [activeSessionRetention, setActiveSessionRetention] = useState<SessionRetention | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [gpu, setGpu] = useState<GpuHealth>({});
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [notes, setNotes] = useState<NoteCard[]>([]);
  const [recall, setRecall] = useState<RecallHit[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [pipeline, setPipeline] = useState<PipelineState>({ queueDepth: 0, droppedWindows: 0 });
  const [fixturePath, setFixturePath] = useState(DEFAULT_FIXTURE_AUDIO);
  const [testSourcePath, setTestSourcePath] = useState(DEFAULT_FIXTURE_AUDIO);
  const [testMaxSeconds, setTestMaxSeconds] = useState("");
  const [testExpectedTerms, setTestExpectedTerms] = useState("");
  const [testPrepared, setTestPrepared] = useState<TestModePrepared | null>(null);
  const [testBusy, setTestBusy] = useState("");
  const [fileUploadBusy, setFileUploadBusy] = useState("");
  const [testReport, setTestReport] = useState<TestModeReport | null>(null);
  const [libraryPath, setLibraryPath] = useState("");
  const [manualQuery, setManualQuery] = useState("");
  const [manualQueryBusy, setManualQueryBusy] = useState(false);
  const [workMemoryStatus, setWorkMemoryStatus] = useState<WorkMemoryStatus | null>(null);
  const [workMemoryMatches, setWorkMemoryMatches] = useState<WorkMemoryCard[]>([]);
  const [workMemoryBusy, setWorkMemoryBusy] = useState("");
  const [speakerStatus, setSpeakerStatus] = useState<SpeakerStatus | null>(null);
  const [speakerEnrollment, setSpeakerEnrollment] = useState<SpeakerEnrollment | null>(null);
  const [speakerBusy, setSpeakerBusy] = useState("");
  const [micTest, setMicTest] = useState<MicTestResult | null>(null);
  const [micTestBusy, setMicTestBusy] = useState(false);
  const [speakerRecordingStartedAt, setSpeakerRecordingStartedAt] = useState<number | null>(null);
  const [speakerRecordingNow, setSpeakerRecordingNow] = useState(Date.now());
  const [transcriptEvents, setTranscriptEvents] = useState<TranscriptEvent[]>([]);
  const [sidecarCards, setSidecarCards] = useState<SidecarDisplayCard[]>([]);
  const [dismissedCardKeys, setDismissedCardKeys] = useState<Set<string>>(() => new Set());
  const [expandedContextCards, setExpandedContextCards] = useState<Set<string>>(() => new Set());
  const [systemLogs, setSystemLogs] = useState<SystemLog[]>([]);
  const [showAllContext, setShowAllContext] = useState(false);
  const [showDebugMetadata, setShowDebugMetadata] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.localStorage.getItem(DEBUG_METADATA_STORAGE_KEY) === "true";
  });
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [followLive, setFollowLive] = useState(true);
  const [unseenItems, setUnseenItems] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const speakerEventSourceRef = useRef<EventSource | null>(null);
  const testStartedAtRef = useRef<string | null>(null);
  const testSessionIdRef = useRef<string | null>(null);
  const eventSeqRef = useRef(0);
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null);
  const followLiveRef = useRef(true);

  useEffect(() => {
    refreshDevices();
    refreshGpu();
    refreshSpeakerStatus();
    refreshWorkMemoryStatus();
    attachSpeakerEvents();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(DEBUG_METADATA_STORAGE_KEY, String(showDebugMetadata));
  }, [showDebugMetadata]);

  useEffect(() => {
    window.localStorage.setItem(SESSION_RETENTION_STORAGE_KEY, sessionRetention);
  }, [sessionRetention]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      speakerEventSourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!followLiveRef.current) {
      return;
    }
    window.requestAnimationFrame(() => scrollFieldToLive("smooth"));
  }, [transcriptEvents]);

  useEffect(() => {
    if (speakerBusy !== "recording") {
      return;
    }
    const timer = window.setInterval(() => setSpeakerRecordingNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [speakerBusy]);

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
  const contributionCards = useMemo(
    () => buildSidecarDisplayCards(
      sidecarCards.filter((card) => !dismissedCardKeys.has(card.cardKey ?? card.id)),
      transcriptEvents,
      { maxCards: 5 },
    ),
    [dismissedCardKeys, sidecarCards, transcriptEvents],
  );
  const manualSourceCards = useMemo(
    () => sidecarCards.filter((card) => card.explicitlyRequested && !dismissedCardKeys.has(card.cardKey ?? card.id)).slice(0, 12),
    [dismissedCardKeys, sidecarCards],
  );
  const visibleContextCards = showAllContext
    ? usefulContextCards
    : usefulContextCards.slice(0, DEFAULT_CONTEXT_CARD_LIMIT);
  const visibleWorkCards = useMemo(() => groupWorkCards(visibleContextCards), [visibleContextCards]);

  const latestTranscript = transcript[transcript.length - 1]?.text ?? "No transcript yet.";
  const workMemoryLabel = workMemoryStatus
    ? `${workMemoryStatus.projects} projects / ${workMemoryStatus.sources} sources`
    : "not indexed";
  const testModeVisible = Boolean(gpu.test_mode_enabled);
  const webStatusLabel = gpu.web_context_enabled
    ? gpu.web_context_configured ? "web ready" : "web key missing"
    : "web off";
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
        ? `Transcripts can label your matching voice as ${speakerEnrollmentLabel}; other voices stay Speaker 1, Speaker 2, or Unknown.`
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
  const captureActive = ["listening", "catching_up"].includes(status);
  const missingServerDevice = audioSource === "server_device" && !selectedDevice;
  const captureStartDisabled = captureStarting || captureActive || missingServerDevice;
  const captureStopDisabled = !sessionId || status === "idle" || status === "stopped";
  const activeRetention = activeSessionRetention ?? sessionRetention;
  const captureModeLabel = audioSource === "fixture"
    ? "Recording playback"
    : "USB mic";
  const sessionRetentionStatus = sessionRetention === "saved" ? "Saved transcript" : "Not saved";
  const sessionRetentionDetail = sessionRetention === "saved"
    ? "Transcript text and notes will be available for future recall. Raw audio is not saved."
    : "Ephemeral: transcript text and notes stay on screen only. Raw audio is not saved.";
  const contextEmptyTitle = captureStarting
    ? "Starting"
    : captureActive
      ? activeRetention === "saved" ? "Recording" : "Listening"
      : "Ready";
  const contextEmptyBody = captureActive || captureStarting
    ? "Actions, decisions, questions, and useful context will appear here."
    : "Start listening, record a transcript, or run a query.";
  const captureIdleButtonLabel = audioSource === "fixture"
    ? "Start Playback"
    : sessionRetention === "saved"
      ? "Start Recording"
      : "Start Listening";
  const captureActiveLabel = activeRetention === "saved" ? "Recording" : "Listening";
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
      : "Start Playback";
  const prepareButtonLabel = testBusy === "preparing" ? "Preparing..." : testPrepared ? "Prepared" : "Prepare Audio";
  const gpuFreeLabel = gpu.memory_free_mb != null && gpu.memory_total_mb != null
    ? `${gpu.memory_free_mb}/${gpu.memory_total_mb} MB free`
    : "VRAM check";
  const statusTone = status === "listening" || status === "catching_up"
    ? "live"
    : status === "error"
      ? "danger"
      : captureStarting
        ? "busy"
        : "idle";

  async function refreshDevices() {
    const response = await fetch(`${API_BASE}/api/devices`);
    const payload = await response.json() as DeviceListResponse;
    const nextDevices = payload.devices ?? [];
    setDevices(nextDevices);
    const preferred = nextDevices.find((device: DeviceInfo) => device.preferred);
    const preferredMissing = Boolean(payload.preferred_device_configured && !payload.preferred_device_available);
    setDeviceNotice(preferredMissing
      ? `${payload.preferred_device_label || "Preferred microphone"} is not connected. Reconnect it, then refresh.`
      : "");
    setSelectedDevice((current) => {
      if (preferred?.id) {
        return preferred.id;
      }
      if (preferredMissing) {
        return "";
      }
      if (current && nextDevices.some((device: DeviceInfo) => device.id === current)) {
        return current;
      }
      return nextDevices[0]?.id ?? "";
    });
  }

  async function refreshGpu() {
    const response = await fetch(`${API_BASE}/api/health/gpu`);
    setGpu(await response.json());
  }

  async function refreshSpeakerStatus() {
    const response = await fetch(`${API_BASE}/api/speaker/status`);
    if (!response.ok) return;
    setSpeakerStatus(await response.json());
  }

  async function refreshWorkMemoryStatus() {
    const response = await fetch(`${API_BASE}/api/work-memory/status`);
    if (!response.ok) return;
    setWorkMemoryStatus(await response.json());
  }

  async function createSession() {
    const response = await fetch(`${API_BASE}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const payload = await response.json();
    setSessionId(payload.id);
    setTranscript([]);
    setNotes([]);
    setRecall([]);
    setWorkMemoryMatches([]);
    setErrors([]);
    setPipeline({ queueDepth: 0, droppedWindows: 0 });
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
    return payload.id as string;
  }

  function attachEvents(id: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`${API_BASE}/api/sessions/${id}/events`);
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
      if ("memory_free_mb" in payload || "gpu_pressure" in payload || "ollama_gpu_models" in payload) {
        setGpu((current) => ({ ...current, ...payload }));
      }
      setPipeline((current) => ({
        queueDepth: Number(payload.queue_depth ?? current.queueDepth),
        droppedWindows: Number(payload.dropped_windows ?? current.droppedWindows),
      }));
      if (nextStatus === "freeing_gpu") {
        appendSystemLog({
          level: "status",
          title: "Freeing GPU for ASR",
          message: `Stopping GPU-resident Ollama models before Faster-Whisper loads. Free VRAM: ${payload.memory_free_mb ?? "unknown"} MB.`,
          metadata: payload,
        });
      }
      if (nextStatus === "loading_asr") {
        appendSystemLog({
          level: "status",
          title: "Loading ASR",
          message: `Faster-Whisper is loading on CUDA. Target free VRAM: ${payload.asr_min_free_vram_mb ?? "unknown"} MB.`,
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
      setGpu((current) => ({ ...current, ...payload }));
    }
    if (envelope.type === "transcript_partial") {
      setPipeline((current) => ({
        ...current,
        queueDepth: Number(payload.queue_depth ?? current.queueDepth),
        droppedWindows: Number(payload.dropped_windows ?? current.droppedWindows),
      }));
      appendTranscriptEvent(transcriptEventFromPayload(payload, envelope));
    }
    if (envelope.type === "transcript_final") {
      setPipeline((current) => ({
        ...current,
        queueDepth: Number(payload.queue_depth ?? current.queueDepth),
        droppedWindows: Number(payload.dropped_windows ?? current.droppedWindows),
      }));
      const line = {
        id: String(payload.id),
        text: String(payload.text),
        start_s: Number(payload.start_s),
        end_s: Number(payload.end_s),
        asr_model: payload.asr_model ? String(payload.asr_model) : undefined,
        created_at: payload.created_at ? Number(payload.created_at) : undefined,
      };
      setTranscript((current) => [...current, line]);
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
      if (hit.source_type === "work_memory_project") {
        const card = workCardFromHit(hit);
        setWorkMemoryMatches((current) => [
          card,
            ...current.filter((candidate) => candidate.project_id !== hit.source_id).slice(0, 5),
        ]);
        appendSidecarCard(displayCardForWorkCard(card, {
          origin: hit.source_segment_ids?.length ? "live" : "unpaired",
          sourceSegmentIds: hit.source_segment_ids,
        }));
      } else {
        appendSidecarCard(displayCardForRecall(hit));
      }
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

  function appendTranscriptEvent(event: TranscriptEvent) {
    setTranscriptEvents((current) => {
      const withoutSameId = current.filter((candidate) => candidate.id !== event.id);
      const reconciled = event.isFinal
        ? withoutSameId.filter((candidate) => candidate.isFinal || !transcriptEventsOverlap(candidate, event))
        : withoutSameId;
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
    const text = card.suggestedSay ?? card.suggestedAsk;
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
    node.scrollTo({ top: node.scrollHeight, behavior });
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

  async function start(options: {
    fixtureWav?: string;
    testRun?: TestModePrepared;
    freshSession?: boolean;
  } = {}): Promise<boolean> {
    if (captureStartDisabled) {
      return false;
    }
    const id = options.freshSession ? await createSession() : sessionId ?? (await createSession());
    const selectedSource: AudioSource = options.fixtureWav ? "fixture" : audioSource;
    const fixtureWav = options.fixtureWav
      ?? (selectedSource === "fixture" && fixturePath.trim() ? fixturePath.trim() : null);
    const retentionForStart = sessionRetention;
    setActiveSessionRetention(retentionForStart);
    setStatus("starting");
    appendSystemLog({
      level: "status",
      title: "Starting capture",
      message: `${fixtureWav ? "Recorded audio playback" : "Live microphone capture"} is starting. ${
        retentionForStart === "saved" ? "Transcript text will be saved." : "Transcript text will stay live only."
      }`,
    });
    const response = await fetch(`${API_BASE}/api/sessions/${id}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_id: fixtureWav ? null : selectedDevice || null,
        fixture_wav: fixtureWav,
        audio_source: fixtureWav ? "fixture" : "server_device",
        save_transcript: retentionForStart === "saved",
      }),
    });
    if (!response.ok) {
      const payload = await response.json();
      setStatus("error");
      if (options.testRun) {
        setTestBusy("");
      }
      setActiveSessionRetention(null);
      setErrors((current) => [payload.detail ?? "Start failed", ...current]);
      appendSystemLog({
        level: "error",
        title: "Start failed",
        message: payload.detail ?? "Start failed",
        metadata: payload,
      });
      return false;
    }
    if (options.testRun) {
      testStartedAtRef.current = new Date().toISOString();
      testSessionIdRef.current = id;
      setTestBusy("playing");
    }
    setStatus((current) => current === "error" ? current : "listening");
    return true;
  }

  async function stop() {
    if (!sessionId) return;
    const stoppedSessionId = sessionId;
    const stoppedRetention = activeSessionRetention ?? sessionRetention;
    await fetch(`${API_BASE}/api/sessions/${sessionId}/stop`, { method: "POST" });
    setStatus("stopped");
    appendSystemLog({
      level: "status",
      title: "Capture stopped",
      message: stoppedRetention === "saved"
        ? `${transcript.length} heard lines saved as transcript text.`
        : `${transcript.length} heard lines shown live; transcript text was not saved.`,
    });
    setActiveSessionRetention(null);
    if (testPrepared && testStartedAtRef.current && testSessionIdRef.current === stoppedSessionId) {
      await saveTestReport(stoppedSessionId, testPrepared, testStartedAtRef.current);
    }
  }

  async function prepareTestAudio() {
    if (!testSourcePath.trim()) return;
    const maxSeconds = testMaxSeconds.trim() ? Number(testMaxSeconds.trim()) : null;
    if (maxSeconds !== null && (!Number.isFinite(maxSeconds) || maxSeconds <= 0)) {
      setErrors((current) => ["Max seconds must be a positive number.", ...current.slice(0, 4)]);
      return;
    }
    setTestBusy("preparing");
    setTestPrepared(null);
    setTestReport(null);
    const response = await fetch(`${API_BASE}/api/test-mode/audio/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_path: testSourcePath.trim(),
        max_seconds: maxSeconds,
        expected_terms: parseExpectedTerms(testExpectedTerms),
      }),
    });
    const payload = await response.json();
    setTestBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Could not prepare recorded audio", ...current.slice(0, 4)]);
      return;
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
  }

  async function startTestPlayback() {
    if (!testPrepared) return;
    setTestReport(null);
    setTestBusy("starting-playback");
    const started = await start({ fixtureWav: testPrepared.fixture_wav, testRun: testPrepared, freshSession: true });
    if (!started) {
      setTestBusy("");
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
  }

  function buildTestModeReport(sessionIdForReport: string, prepared: TestModePrepared, startedAt: string): TestModeReport {
    const expectedTerms = prepared.expected_terms.length > 0
      ? prepared.expected_terms
      : parseExpectedTerms(testExpectedTerms);
    const responseText = [
      transcript.map((line) => line.text).join("\n"),
      notes.map((note) => `${note.title}\n${note.body}`).join("\n"),
      recall.map((hit) => hit.text).join("\n"),
      workMemoryMatches.map((card) => card.text).join("\n"),
      sidecarCards.map((card) => `${card.title}\n${card.summary}\n${card.whyRelevant ?? ""}`).join("\n"),
    ].filter(Boolean).join("\n");
    return {
      run_id: prepared.run_id,
      session_id: sessionIdForReport,
      source_path: prepared.source_path,
      fixture_wav: prepared.fixture_wav,
      duration_seconds: prepared.duration_seconds,
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      status,
      transcript_segments: transcript.length,
      note_cards: notes.length,
      recall_hits: recall.length,
      queue_depth: pipeline.queueDepth,
      dropped_windows: pipeline.droppedWindows,
      work_parallel_rows: workMemoryMatches.length,
      error_count: errors.length,
      expected_terms: expectedTerms,
      matched_expected_terms: matchExpectedTerms(responseText, expectedTerms),
      issues: errors.length > 0 ? ["UI surfaced one or more errors during playback."] : [],
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
    await refreshWorkMemoryStatus();
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

  async function reindexWorkMemory() {
    setWorkMemoryBusy("indexing");
    const response = await fetch(`${API_BASE}/api/work-memory/reindex`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const payload = await response.json();
    setWorkMemoryBusy("");
    if (!response.ok) {
      setErrors((current) => [payload.detail ?? "Work memory reindex failed", ...current]);
      return;
    }
    await refreshWorkMemoryStatus();
    setStatus(`work memory indexed ${payload.projects_indexed ?? 0} projects`);
    appendSystemLog({
      level: "status",
      title: "Work memory indexed",
      message: `${payload.projects_indexed ?? 0} projects refreshed.`,
      metadata: payload,
    });
  }

  async function testMicrophone() {
    setMicTestBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/microphone/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: selectedDevice || null,
          audio_source: fixturePath.trim() && audioSource === "fixture" ? "fixture" : "server_device",
          fixture_wav: fixturePath.trim() && audioSource === "fixture" ? fixturePath.trim() : null,
          seconds: 3,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setErrors((current) => [payload.detail ?? "Microphone test failed", ...current]);
        return;
      }
      setMicTest(payload);
      appendSystemLog({
        level: payload.recommendation?.status === "good" ? "status" : "warning",
        title: payload.recommendation?.title ?? "Microphone check complete",
        message: payload.recommendation?.detail ?? "Mic test returned.",
        metadata: payload,
      });
    } catch (error) {
      pushError(error instanceof Error ? error.message : "Microphone test failed");
    } finally {
      setMicTestBusy(false);
    }
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
    setDrawerOpen(true);
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
          device_id: selectedDevice || null,
          audio_source: fixturePath.trim() && audioSource === "fixture" ? "fixture" : "server_device",
          fixture_wav: fixturePath.trim() && audioSource === "fixture" ? fixturePath.trim() : null,
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
        session_id: segment?.session_id ?? sessionId ?? "manual-review",
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
    <div className="app-frame">
      <header className="top-bar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <strong>Brain Sidecar</strong>
            <small>{webStatusLabel}</small>
          </div>
        </div>
        <div className="top-metrics" aria-label="Session metrics">
          <Metric label="Transcript" value={String(transcript.length)} detail="heard lines" />
          <Metric label="Notes" value={String(notes.length)} detail="cards" />
          <Metric label="Recall" value={String(recall.length)} detail={workMemoryStatus ? `${workMemoryStatus.projects} work` : "hits"} />
          <Metric label="Queue" value={String(pipeline.queueDepth)} detail={`beam ${gpu.asr_beam_size ?? "?"}`} />
        </div>
        <div className="top-actions">
          <span className={`status-pill ${statusTone}`} aria-label="Runtime status">
            <span aria-hidden="true" />
            {status}
          </span>
          <span className={`retention-pill ${activeRetention}`} aria-label="Session save mode status">
            {captureActive ? (activeRetention === "saved" ? "Recording" : "Listening") : sessionRetentionStatus}
          </span>
          <button className="secondary" onClick={() => setDrawerOpen(true)}>Tools</button>
          <button className="primary" onClick={() => start()} disabled={captureStartDisabled}>
            {captureButtonLabel}
          </button>
          <button className="danger" onClick={stop} disabled={captureStopDisabled}>
            Stop
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <section className="transcript-pane" aria-label="Live transcript pane">
          <div className="field-toolbar">
            <div>
              <p className="label">Live Transcript</p>
              <h1>Live transcript</h1>
            </div>
            <div className="field-toolbar-status">
              <span>{gpu.asr_cuda_available ? "CUDA ready" : gpu.asr_cuda_error ?? "GPU check"}</span>
              <span>{gpuFreeLabel}</span>
            </div>
          </div>

          <section className="capture-strip" aria-label="Capture controls">
            <div className="mode-control" role="group" aria-label="Session save mode">
              <button
                type="button"
                aria-pressed={sessionRetention === "temporary"}
                className={sessionRetention === "temporary" ? "active" : ""}
                disabled={captureStarting || captureActive}
                onClick={() => setSessionRetention("temporary")}
              >
                <span>Listen</span>
                <small>Ephemeral</small>
              </button>
              <button
                type="button"
                aria-pressed={sessionRetention === "saved"}
                className={sessionRetention === "saved" ? "active" : ""}
                disabled={captureStarting || captureActive}
                onClick={() => setSessionRetention("saved")}
              >
                <span>Record</span>
                <small>Saved transcript</small>
              </button>
            </div>
            <div className="capture-strip-summary" aria-label="Selected capture mode">
              <strong>{captureModeLabel} · {captureActive ? (activeRetention === "saved" ? "Recording" : "Listening") : sessionRetentionStatus}</strong>
              <span>{sessionRetentionDetail}</span>
            </div>
          </section>

          {errors.length > 0 && (
            <section className="errors" role="alert">
              {errors.map((error, index) => (
                <p key={`${error}-${index}`}>{error}</p>
              ))}
            </section>
          )}

          <div
            id="transcript"
            className="transcript-scroll scroll"
            role="feed"
            aria-label="Live transcript"
            ref={transcriptScrollRef}
            onScroll={handleFieldScroll}
          >
            <TranscriptPane events={transcriptEvents} />
          </div>

          <ContributionLane
            cards={contributionCards}
            expandedCards={expandedContextCards}
            showDebugMetadata={showDebugMetadata}
            onToggle={toggleExpandedContextCard}
            onPin={togglePinnedCard}
            onDismiss={dismissCard}
            onCopy={copyCardSuggestion}
          />

          {!followLive && (
            <button className="jump-live" onClick={() => scrollFieldToLive("smooth")}>
              Jump to live{unseenItems > 0 ? ` (${unseenItems})` : ""}
            </button>
          )}

          <form
            className="query-bar"
            role="search"
            onSubmit={(event) => {
              event.preventDefault();
              searchRecall();
            }}
          >
            <input
              aria-label="Unified query"
              value={manualQuery}
              onChange={(event) => setManualQuery(event.target.value)}
              placeholder="Ask Sidecar anything"
            />
            <button className="primary subtle" disabled={manualQueryBusy || !manualQuery.trim()}>
              {manualQueryBusy ? "Searching..." : "Search"}
            </button>
          </form>

          <ManualSourceSections cards={manualSourceCards} />

          <section className="latest-card" aria-label="Latest heard" aria-live="polite">
            <p className="label">Latest heard</p>
            <blockquote>{latestTranscript}</blockquote>
          </section>
        </section>

        <section className="context-pane" id="recall" aria-label="Work Notes">
          <div className="field-toolbar context-toolbar">
            <div>
              <p className="label">Sidecar</p>
              <h1>Work Notes</h1>
            </div>
            <div className="field-toolbar-status">
              <span>{workMemoryLabel}</span>
              <span>{usefulContextCards.length} useful</span>
            </div>
          </div>
          <WorkNotesPane
            groups={visibleWorkCards}
            usefulCount={usefulContextCards.length}
            rawCount={sidecarCards.length}
            loading={manualQueryBusy}
            emptyTitle={contextEmptyTitle}
            emptyBody={contextEmptyBody}
            expandedCards={expandedContextCards}
            showDebugMetadata={showDebugMetadata}
            onToggle={toggleExpandedContextCard}
            onPin={togglePinnedCard}
            onDismiss={dismissCard}
            onCopy={copyCardSuggestion}
          />
          {usefulContextCards.length > DEFAULT_CONTEXT_CARD_LIMIT && (
            <button className="secondary show-more-context" onClick={() => setShowAllContext((value) => !value)}>
              {showAllContext ? "Show less" : `Show more (${usefulContextCards.length - DEFAULT_CONTEXT_CARD_LIMIT})`}
            </button>
          )}
        </section>

        <aside className={`tool-drawer ${drawerOpen ? "open" : ""}`} aria-label="Tools drawer">
          <div className="drawer-header">
            <div>
              <p className="label">Tools</p>
              <h2>Session controls</h2>
            </div>
            <button className="secondary" onClick={() => setDrawerOpen(false)}>Close</button>
          </div>

          <section className="drawer-section" id="capture">
            <div className="drawer-section-heading">
              <h3>Input</h3>
              <span>{devices.length || "No"} devices</span>
            </div>
            <label className="field field-wide">
              <span>Audio source</span>
              <select
                aria-label="Audio source"
                value={audioSource}
                onChange={(event) => setAudioSource(event.target.value as AudioSource)}
              >
                <option value="server_device">Server USB microphone</option>
                {ENABLE_FIXTURE_AUDIO && <option value="fixture">Fixture WAV</option>}
              </select>
            </label>
            <div className="compact-tool">
              <p className="label">Transcript Retention</p>
              <div className="mode-control compact" role="group" aria-label="Transcript retention">
                <button
                  type="button"
                  aria-pressed={sessionRetention === "temporary"}
                  className={sessionRetention === "temporary" ? "active" : ""}
                  disabled={captureStarting || captureActive}
                  onClick={() => setSessionRetention("temporary")}
                >
                  <span>Listen</span>
                  <small>Ephemeral</small>
                </button>
                <button
                  type="button"
                  aria-pressed={sessionRetention === "saved"}
                  className={sessionRetention === "saved" ? "active" : ""}
                  disabled={captureStarting || captureActive}
                  onClick={() => setSessionRetention("saved")}
                >
                  <span>Record</span>
                  <small>Saved transcript</small>
                </button>
              </div>
              <p className="tool-note">{sessionRetentionDetail}</p>
            </div>
            <label className="field field-wide">
              <span>USB microphone</span>
              <select
                aria-label="USB microphone"
                value={selectedDevice}
                onChange={(event) => setSelectedDevice(event.target.value)}
                disabled={audioSource !== "server_device"}
              >
                <option value="">No USB mic found yet</option>
                {devices.map((device) => (
                  <option key={device.id} value={device.id}>
                    {device.label}
                  </option>
                ))}
              </select>
            </label>
            {deviceNotice && <p className="tool-note warning-text" role="status">{deviceNotice}</p>}
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
              {micTest?.quality.issues.length ? (
                <div className="chip-row">
                  {micTest.quality.issues.map((issue) => <span key={issue} className="chip warning">{issue}</span>)}
                </div>
              ) : null}
            </div>
            <div className="button-row">
              <button className="secondary" onClick={refreshDevices}>Refresh</button>
              <button className="secondary" onClick={testMicrophone} disabled={micTestBusy || audioSource !== "server_device" || missingServerDevice}>
                {micTestBusy ? "Testing..." : "Test Mic"}
              </button>
              <button className="primary" onClick={() => start()} disabled={captureStartDisabled}>
                {captureButtonLabel}
              </button>
              <button className="danger" onClick={stop} disabled={captureStopDisabled}>
                End Capture
              </button>
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
                    <input
                      aria-label="Browse fixture WAV"
                      className="file-picker"
                      type="file"
                      accept=".wav,audio/wav,audio/x-wav"
                      disabled={Boolean(fileUploadBusy)}
                      onChange={(event) => handleInputFilePick(event, "fixture", setFixturePath)}
                    />
                  </div>
                </div>
              </div>
            )}
          </section>

          {testModeVisible && (
            <section className="drawer-section" id="test-mode" role="region" aria-label="Recorded audio test">
              <div className="drawer-section-heading">
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
                    <input
                      aria-label="Browse source audio"
                      className="file-picker"
                      type="file"
                      accept="audio/*,.wav,.mp3,.m4a,.mp4,.aac,.flac,.ogg,.webm"
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
                <button className="secondary" onClick={prepareTestAudio} disabled={testBusy === "preparing" || captureActive || captureStarting}>
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

          <section className="drawer-section" id="speaker-identity" aria-label="Speaker identity">
            <div className="drawer-section-heading">
              <h3>Speaker Identity</h3>
              <span>{speakerStatusLabel}</span>
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

            {!speakerEnrollment && !speakerStatus?.ready && (
              <button
                className="primary speaker-primary-action"
                onClick={startSpeakerEnrollment}
                disabled={Boolean(speakerBusy) || speakerStatus?.backend.available === false || missingServerDevice}
              >
                {speakerTrainingButtonLabel}
              </button>
            )}

            {speakerEnrollment && !speakerStatus?.ready && (
              <div className="speaker-record-flow">
                <div className="speaker-record-prompt" aria-label="Speaker recording prompt">
                  <strong>Record only {speakerEnrollmentLabel}</strong>
                  <span>Use your normal mic position. Speak naturally for the full timer. Stop if another voice enters; this is for speaker labels, not word accuracy.</span>
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
                          {sample.usable_speech_seconds.toFixed(1)}s · {Math.round(sample.quality_score * 100)}%{sample.issues.length ? " · check" : ""}
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
                <span>New transcript segments can be labeled {speakerEnrollmentLabel}; other voices stay Speaker 1, Speaker 2, or Unknown.</span>
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
            <details className="voice-preview">
              <summary>Review learned speaker profile</summary>
              <div className="speaker-review-list">
                <div className="speaker-review-row">
                  <strong>{speakerStatus?.profile.display_name ?? "BP"}</strong>
                  <span>{speakerStatus?.backend.device ?? "device"}</span>
                  <p>{speakerStatus?.backend.model_name ?? "No speaker model loaded"}</p>
                  <div className="speaker-feedback-controls">
                    <button type="button" className="secondary" onClick={() => submitSpeakerFeedback(null, "merge speakers", "merge_speakers")}>
                      Merge speakers
                    </button>
                    <button type="button" className="secondary" onClick={() => submitSpeakerFeedback(null, "split speaker", "split_speaker")}>
                      Split speaker
                    </button>
                    <button type="button" className="secondary" onClick={() => submitSpeakerFeedback(null, speakerEnrollmentLabel, "rename_speaker")}>
                      Rename speaker
                    </button>
                  </div>
                </div>
              </div>
            </details>
            <div className="button-row">
              <button className="secondary danger-subtle" onClick={resetSpeakerProfile} disabled={Boolean(speakerBusy)}>
                Delete Learned Embeddings
              </button>
            </div>
          </section>

          <section className="drawer-section" id="indexes">
            <div className="drawer-section-heading">
              <h3>Indexes</h3>
              <span>{workMemoryLabel}</span>
            </div>
            <div className="memory-overview" aria-label="Work memory index summary">
              <span><strong>{workMemoryStatus?.projects ?? 0}</strong> projects</span>
              <span><strong>{workMemoryStatus?.evidence ?? 0}</strong> citations</span>
              <span><strong>{workMemoryStatus?.disabled_sources ?? 0}</strong> guarded</span>
            </div>
            <div className="field">
              <span>File root</span>
              <div className="path-picker">
                <input
                  aria-label="File root"
                  value={libraryPath}
                  onChange={(event) => setLibraryPath(event.target.value)}
                  placeholder="/path/to/notes-or-docs"
                />
                <input
                  aria-label="Browse index file"
                  className="file-picker"
                  type="file"
                  accept=".txt,.md,.markdown,.pdf,.docx,.json,.csv,.rtf"
                  disabled={Boolean(fileUploadBusy) || workMemoryBusy === "indexing"}
                  onChange={(event) => handleInputFilePick(event, "index-file", setLibraryPath)}
                />
              </div>
            </div>
            <div className="button-row">
              <button className="secondary" onClick={addLibraryRoot}>Add Files</button>
              <button className="secondary" onClick={reindex}>Reindex Files</button>
              <button className="secondary" onClick={reindexWorkMemory} disabled={workMemoryBusy === "indexing"}>
                {workMemoryBusy === "indexing" ? "Indexing..." : "Index Work"}
              </button>
            </div>
          </section>

          <section className="drawer-section">
            <div className="drawer-section-heading">
              <h3>System</h3>
              <span>{gpu.asr_cuda_available ? "ready" : "check"}</span>
            </div>
            <p className="gpu-name">{gpuLabel}</p>
            <div className="vram-bar" aria-label={`VRAM usage ${vramPercent}%`}>
              <span style={{ width: `${vramPercent}%` }} />
            </div>
            <div className="chip-row">
              <span className="chip">ASR {gpu.asr_primary_model ?? "medium.en"}</span>
              <span className="chip">
                window {formatSeconds(gpu.transcription_window_seconds ?? 5)} / overlap {formatSeconds(gpu.transcription_overlap_seconds ?? 0.75)}
              </span>
              <span className="chip">Ollama {gpu.ollama_chat_model ?? "phi3:mini"}</span>
              {(gpu.ollama_gpu_models ?? []).length === 0 ? (
                <span className="chip muted">Ollama idle</span>
              ) : (
                gpu.ollama_gpu_models?.map((model) => <span key={model} className="chip">{model}</span>)
              )}
              <span className="chip">keep {gpu.ollama_keep_alive ?? "10m"}</span>
              <span className="chip">dedupe {gpu.dedupe_similarity_threshold ?? "?"}</span>
            </div>
            <label className="field">
              <span>Theme</span>
              <select aria-label="Theme" value={theme} onChange={(event) => setTheme(event.target.value as ThemeName)}>
                {THEMES.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </label>
            <label className="toggle-row">
              <input
                aria-label="Show debug metadata"
                type="checkbox"
                checked={showDebugMetadata}
                onChange={(event) => setShowDebugMetadata(event.target.checked)}
              />
              Show debug metadata
            </label>
            {showDebugMetadata && (
              <details className="system-log" open>
                <summary>System logs</summary>
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
                        {log.metadata && <pre>{formatDebugValue(log.metadata)}</pre>}
                      </article>
                    ))}
                  </div>
                )}
              </details>
            )}
          </section>
        </aside>
      </main>
    </div>
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

function TranscriptPane({ events }: { events: TranscriptEvent[] }) {
  if (events.length === 0) {
    return <Empty title="Ready" body="Start listening or run a query." />;
  }
  return (
    <>
      {events.map((event) => {
        const label = getTranscriptDisplayLabel(event);
        return (
          <article
            key={event.id}
            className={`transcript-bubble ${event.source} ${event.isFinal ? "final" : "interim"}`}
            aria-label={`${label} transcript item`}
          >
            <div className="transcript-bubble-head">
              <span>{label}</span>
              <div>
                {!event.isFinal && <small>Interim</small>}
                <time>{formatClock(event.at)}</time>
              </div>
            </div>
            <p>{event.text}</p>
          </article>
        );
      })}
    </>
  );
}

function WorkNotesPane({
  groups,
  usefulCount,
  rawCount,
  loading,
  emptyTitle,
  emptyBody,
  expandedCards,
  showDebugMetadata,
  onToggle,
  onPin,
  onDismiss,
  onCopy,
}: {
  groups: Record<WorkCardBucket, SidecarDisplayCard[]>;
  usefulCount: number;
  rawCount: number;
  loading: boolean;
  emptyTitle: string;
  emptyBody: string;
  expandedCards: Set<string>;
  showDebugMetadata: boolean;
  onToggle: (id: string) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
}) {
  const cardCount = WORK_NOTE_SECTIONS.reduce((count, section) => count + groups[section.bucket].length, 0);
  if (loading && cardCount === 0) {
    return <Empty title="Searching" body="Looking for useful work context." />;
  }
  if (cardCount === 0 && rawCount > 0 && usefulCount === 0) {
    return <Empty title="No Useful Context" body="Recent backend hits were echoes, duplicates, or too weak to show." />;
  }
  if (cardCount === 0) {
    return <Empty title={emptyTitle} body={emptyBody} />;
  }
  return (
    <div className="context-card-list work-notes-list scroll" aria-label="Work note cards">
      {WORK_NOTE_SECTIONS.map((section) => (
        <section key={section.bucket} className="work-note-section" aria-label={`${section.title} work notes`}>
          <div className="work-note-section-head">
            <h2>{section.title}</h2>
            <span>{groups[section.bucket].length}</span>
          </div>
          {groups[section.bucket].length === 0 ? (
            <p className="work-note-empty">{section.empty}</p>
          ) : (
            groups[section.bucket].map((card) => (
              <ContextCard
                key={card.id}
                card={card}
                expanded={expandedCards.has(card.id)}
                showDebugMetadata={showDebugMetadata}
                onToggle={() => onToggle(card.id)}
                onPin={() => onPin(card)}
                onDismiss={() => onDismiss(card)}
                onCopy={() => onCopy(card)}
              />
            ))
          )}
        </section>
      ))}
    </div>
  );
}

function ContributionLane({
  cards,
  expandedCards,
  showDebugMetadata,
  onToggle,
  onPin,
  onDismiss,
  onCopy,
}: {
  cards: SidecarDisplayCard[];
  expandedCards: Set<string>;
  showDebugMetadata: boolean;
  onToggle: (id: string) => void;
  onPin: (card: SidecarDisplayCard) => void;
  onDismiss: (card: SidecarDisplayCard) => void;
  onCopy: (card: SidecarDisplayCard) => void;
}) {
  if (cards.length === 0) {
    return null;
  }
  return (
    <section className="contribution-lane" aria-label="Contribution Lane">
      <div className="contribution-lane-head">
        <div>
          <p className="label">Contribution Lane</p>
          <h2>Say, ask, watch</h2>
        </div>
        <span>{cards.length}</span>
      </div>
      <div className="contribution-card-row">
        {cards.map((card) => (
          <ContextCard
            key={`lane-${card.id}`}
            card={card}
            compact
            expanded={expandedCards.has(card.id)}
            showDebugMetadata={showDebugMetadata}
            onToggle={() => onToggle(card.id)}
            onPin={() => onPin(card)}
            onDismiss={() => onDismiss(card)}
            onCopy={() => onCopy(card)}
          />
        ))}
      </div>
    </section>
  );
}

function ManualSourceSections({ cards }: { cards: SidecarDisplayCard[] }) {
  if (cards.length === 0) {
    return null;
  }
  const sections = [
    { title: "Prior transcript", cards: cards.filter((card) => String(card.debugMetadata?.sourceType) === "saved_transcript" || card.category === "memory") },
    { title: "PAS / past work", cards: cards.filter((card) => card.category === "work_memory" || String(card.debugMetadata?.sourceType) === "work_memory") },
    { title: "Current public web", cards: cards.filter((card) => card.category === "web" || String(card.debugMetadata?.sourceType) === "brave_web") },
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
  onPin,
  onDismiss,
  onCopy,
}: {
  card: SidecarDisplayCard;
  compact?: boolean;
  expanded: boolean;
  showDebugMetadata: boolean;
  onToggle: () => void;
  onPin: () => void;
  onDismiss: () => void;
  onCopy: () => void;
}) {
  const hasNormalDetails = Boolean((card.sources && card.sources.length > 0) || (card.citations && card.citations.length > 0));
  const hasDebugDetails = showDebugMetadata && Boolean(card.rawText || card.debugMetadata);
  const hasDetails = hasNormalDetails || hasDebugDetails;
  const suggestion = card.suggestedSay ?? card.suggestedAsk;
  return (
    <article
      className={`context-card field-bubble ${compact ? "compact" : ""} ${card.provisional ? "provisional" : ""} ${confidenceClass(card)} ${card.category === "work" || card.category === "work_memory" ? "memory" : card.category}`}
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
      {card.source || card.date ? (
        <p className="context-card-source">{[card.source, card.date].filter(Boolean).join(" / ")}</p>
      ) : null}
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
          <button className="link-button" onClick={onToggle}>
            {expanded ? "Hide details" : "Show details"}
          </button>
          {expanded && (
            <div className="field-details">
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

function workCardFromHit(hit: RecallHit): WorkMemoryCard {
  const metadata = hit.metadata;
  const citations = Array.isArray(metadata.citations)
    ? metadata.citations.map((item) => String(item))
    : [];
  return {
    project_id: hit.source_id,
    title: String(metadata.title ?? "Prior project"),
    organization: String(metadata.organization ?? "work history"),
    date_range: String(metadata.date_range ?? ""),
    score: hit.score,
    confidence: Number(metadata.confidence ?? hit.score),
    reason: String(metadata.reason ?? ""),
    lesson: String(metadata.lesson ?? ""),
    citations,
    text: hit.text,
  };
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
  const cardKey = payload.card_key ? String(payload.card_key) : undefined;
  const suggestionSay = stringOrUndefined(payload.suggested_say);
  const suggestionAsk = stringOrUndefined(payload.suggested_ask);
  return {
    id: `sidecar-${String(payload.id)}`,
    cardKey,
    category,
    title: readableTitle(title, categoryLabel(category)),
    summary: body,
    whyRelevant: stringOrUndefined(payload.why_now),
    source: sourceLabel(sourceType),
    at,
    score: numberOrUndefined(payload.confidence),
    confidence: numberOrUndefined(payload.confidence),
    rawText: body,
    sourceSegmentIds,
    suggestedSay: suggestionSay,
    suggestedAsk: suggestionAsk,
    priority: priorityFromValue(payload.priority),
    sources: parseNoteSources(payload.sources),
    citations: parseStringList(payload.citations),
    ephemeral: Boolean(payload.ephemeral),
    expiresAt: payloadTimeMs(payload.expires_at),
    provisional: Boolean(payload.provisional),
    explicitlyRequested: Boolean(payload.explicitly_requested) || String(cardKey ?? "").startsWith("manual:"),
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
    },
  };
}

function transcriptEventFromPayload(payload: Record<string, unknown>, envelope: EventEnvelope): TranscriptEvent {
  const source = transcriptSourceFromPayload(payload.source);
  const at = payloadTimeMs(payload.created_at ?? envelope.at) ?? Date.now();
  return {
    id: `transcript-${String(payload.id)}`,
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
  const category = note.ephemeral || note.source_type === "brave_web"
    ? "web"
    : sidecarCategoryFromPayload(["action", "decision", "question", "risk", "clarification", "contribution"].includes(note.kind) ? note.kind : "note");
  const at = payloadTimeMs(note.created_at) ?? Date.now();
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
    explicitlyRequested: options.origin === "manual",
    priority: note.kind === "action" || note.kind === "decision" ? "high" : "normal",
    sources: note.sources,
    debugMetadata: {
      sourceType: note.source_type ?? note.kind,
      sourceId: note.id,
      backendLabel: note.kind,
      sourceSegmentIds,
    },
  };
}

function displayCardForRecall(hit: RecallHit, options: BackendItemOptions = {}): SidecarDisplayCard {
  const sourceSegmentIds = options.sourceSegmentIds ?? hit.source_segment_ids ?? [];
  const sourceType = hit.source_type;
  const title = readableTitle(String(hit.metadata.title ?? ""), "Relevant memory");
  const at = Date.now();
  return {
    id: `recall-${sourceType}-${hit.source_id}-${at}`,
    cardKey: `recall:${sourceType}:${hit.source_id}`,
    category: "memory",
    title,
    summary: hit.text,
    whyRelevant: recallReason(hit, options.origin),
    source: sourceLabel(sourceType),
    at,
    score: hit.score,
    rawText: hit.text,
    sourceSegmentIds,
    pinned: Boolean(hit.pinned ?? hit.metadata.pinned),
    explicitlyRequested: options.origin === "manual" || Boolean(hit.explicitly_requested ?? hit.metadata.explicitly_requested),
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

function displayCardForWorkCard(card: WorkMemoryCard, options: BackendItemOptions = {}): SidecarDisplayCard {
  const sourceSegmentIds = options.sourceSegmentIds ?? [];
  const at = Date.now();
  return {
    id: `work-${card.project_id}-${at}`,
    cardKey: `work:${card.project_id}`,
    category: "work_memory",
    title: readableTitle(card.title, "Prior work"),
    summary: card.lesson || card.text,
    whyRelevant: card.reason ? `Similar pattern: ${card.reason}` : "Relevant prior work pattern.",
    source: card.organization,
    date: card.date_range,
    at,
    score: card.score,
    rawText: card.text,
    citations: card.citations,
    sourceSegmentIds,
    explicitlyRequested: options.origin === "manual",
    priority: card.confidence >= 0.9 ? "high" : "normal",
    debugMetadata: {
      sourceType: "work_memory_project",
      sourceId: card.project_id,
      score: card.score,
      confidence: card.confidence,
      retrievalReason: card.reason,
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
  if (category === "work" || category === "work_memory") return "Work memory";
  if (category === "action") return "Follow-up";
  if (category === "decision") return "Decision";
  if (category === "question") return "Ask this";
  if (category === "risk") return "Watch risk";
  if (category === "clarification") return "Clarify";
  if (category === "contribution") return "Say this";
  if (category === "status") return "Status";
  if (category === "memory") return "Memory";
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

function sourceLabel(sourceType: string): string {
  if (sourceType === "file" || sourceType === "document_chunk") return "Prior notes";
  if (sourceType === "session") return "Prior session";
  if (sourceType === "work_memory_project") return "Work memory";
  return "Local memory";
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
      "work",
      "work_memory",
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

function stringOrUndefined(value: unknown): string | undefined {
  const text = String(value ?? "").trim();
  return text ? text : undefined;
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

function formatClock(value: number): string {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatSeconds(value: number): string {
  return `${Number(value).toFixed(1)}s`;
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
