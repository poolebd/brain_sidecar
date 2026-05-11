import type { Page, Route } from "@playwright/test";

type MockEvent = {
  type: string;
  payload: Record<string, unknown>;
};

type MockSpeakerStatus = {
  profile: {
    id: string;
    display_name: string;
    kind: string;
    active: boolean;
    threshold: number | null;
    embedding_model: string;
    embedding_model_version: string;
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
  same_speaker_scores: { count: number; min: number | null; mean: number | null };
  quality_score: number;
  raw_audio_retained: boolean;
  recent_segments: unknown[];
};

const API_ORIGIN = "http://127.0.0.1:8765";
const MOCK_MIC_WAV_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAF8FlQp7D+sTwxflGjsdsh4/H90ekB1iG2MYqhRTEIALVgb7AJn7WPZh8dns4+ie5SHjgeHJ4ADhIuIo5APnnOrY7pfztfgK/m0Dtwi/DV4ScRbYGXscRB4nHxwfJB5GHJEZGBb4EU4NPgjwAo39O/gk83DuQeq45vDj/uHx4NDgnuFS4+LlOek97dHx0PYW/HgB0Qb1C74QCBWxGJ4buB3wHjwfmR4OHaUacBeJEw4PHwrjBIP/Jfr19Bjwtevs59vkmeI44cDgOOGZ4tvk7Oe16xjw9fQl+oP/4wQfCg4PiRNwF6UaDh2ZHjwf8B64HZ4bsRgIFb4Q9QvRBngBFvzQ9tHxPe056eLlUuOe4dDg8eD+4fDjuOZB6nDuJPM7+I398AI+CE4N+BEYFpEZRhwkHhwfJx9EHnsc2BlxFg==";

type MockDevice = {
  id: string;
  label: string;
  driver: string;
  ffmpeg_input: string;
  hardware_id: string;
  healthy: boolean;
  in_use?: boolean;
  score: number;
  selection_reason: string;
};

type MockDeviceResponse = {
  devices: MockDevice[];
  selected_device?: MockDevice | null;
  server_mic_available?: boolean;
  selection_reason?: string;
};

type MockApiOptions = {
  testModeEnabled?: boolean;
  testPreparedDurationSeconds?: number;
  startSessionResponse?: { status?: number; body?: Record<string, unknown> };
  abortSessionListAfterCreate?: boolean;
  devicesResponse?: MockDeviceResponse;
  gpuHealthResponse?: Record<string, unknown>;
  sessionsResponse?: Record<string, unknown>[];
  latestReviewStatus?: string;
  reviewUsefulnessStatus?: string;
  reviewUsefulnessScore?: number;
  reviewUsefulnessFlags?: string[];
  reviewQueueScenario?: "single" | "two_jobs";
  reviewJobError?: string;
  reviewCleanupError?: string;
  longReviewEvidence?: boolean;
  reviewTypographyStress?: boolean;
};

export const mockDevices: MockDevice[] = [
  {
    id: "usb-blue",
    label: "Blue Yeti USB Microphone",
    driver: "alsa",
    ffmpeg_input: "plughw:1,0",
    hardware_id: "1111:2222",
    healthy: true,
    score: 95,
    selection_reason: "USB capture",
  },
  {
    id: "desk-array",
    label: "Desk Array Microphone",
    driver: "alsa",
    ffmpeg_input: "plughw:2,0",
    hardware_id: "3333:4444",
    healthy: true,
    score: 78,
    selection_reason: "USB capture",
  },
];

export async function installMockEventSource(page: Page) {
  await page.addInitScript(() => {
    class MockEventSource extends EventTarget {
      static instances: MockEventSource[] = [];

      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 1;
      url: string;

      constructor(url: string) {
        super();
        this.url = url;
        MockEventSource.instances.push(this);
      }

      close() {
        this.readyState = 2;
      }

      emit(type: string, payload: Record<string, unknown>) {
        const event = new MessageEvent(type, {
          data: JSON.stringify({
            id: `${type}-event`,
            type,
            session_id: "session-123",
            at: Date.now() / 1000,
            payload,
          }),
        });
        if (type === "message") {
          this.onmessage?.(event);
        }
        this.dispatchEvent(event);
      }
    }

    Object.defineProperty(window, "EventSource", {
      configurable: true,
      writable: true,
      value: MockEventSource,
    });

    Object.defineProperty(window, "__brainSidecarEmit", {
      configurable: true,
      writable: true,
      value: (type: string, payload: Record<string, unknown>) => {
        const source = [...MockEventSource.instances]
          .reverse()
          .find((candidate) => candidate.url.includes("/api/live/") || candidate.url.includes("/api/sessions/"))
          ?? MockEventSource.instances.at(-1);
        if (!source) {
          throw new Error("No mock EventSource instance is available.");
        }
        source.emit(type, payload);
      },
    });

    Object.defineProperty(window, "__brainSidecarEventSourceUrls", {
      configurable: true,
      get: () => MockEventSource.instances.map((source) => source.url),
    });
  });
}

export async function mockApi(page: Page, options: MockApiOptions = {}) {
  const testModeEnabled = options.testModeEnabled ?? false;
  const devicesResponse: MockDeviceResponse = options.devicesResponse ?? {
    devices: mockDevices,
    selected_device: mockDevices[0],
    server_mic_available: true,
    selection_reason: "USB capture",
  };
  let speakerStatus: MockSpeakerStatus = {
    profile: {
      id: "self_bp",
      display_name: "BP",
      kind: "self",
      active: false,
      threshold: null,
      embedding_model: "speechbrain/spkrec-ecapa-voxceleb",
      embedding_model_version: "speechbrain-ecapa-voxceleb",
    },
    backend: {
      available: true,
      model_name: "speechbrain/spkrec-ecapa-voxceleb",
      model_version: "speechbrain-ecapa-voxceleb",
      device: "cuda",
    },
    enrollment_status: "not_enrolled",
    ready: false,
    needs_recalibration: false,
    usable_speech_seconds: 0,
    target_speech_seconds: 120,
    minimum_speech_seconds: 60,
    embedding_count: 0,
    centroid_vector_dim: null,
    same_speaker_scores: { count: 0, min: null, mean: null },
    quality_score: 0,
    raw_audio_retained: false,
    recent_segments: [],
  };
  const savedSession = {
    id: "session-saved-1",
    title: "Saved model planning call",
    status: "stopped",
    started_at: 1_768_000_000,
    ended_at: 1_768_000_300,
    save_transcript: true,
    retention: "saved",
    transcript_count: 2,
    note_count: 1,
    summary_exists: true,
    raw_audio_retained: false,
  };
  const reviewSavedSession = {
    id: "session-review-1",
    title: "Review Apollo call",
    status: "stopped",
    started_at: 1_768_000_090,
    ended_at: 1_768_000_130,
    save_transcript: true,
    retention: "saved",
    transcript_count: 2,
    note_count: 2,
    summary_exists: true,
    raw_audio_retained: false,
  };
  let ollamaChatModel = String(options.gpuHealthResponse?.ollama_chat_model ?? "qwen3.5:397b-cloud");
  const ollamaModels = [
    { name: "qwen3.5:397b-cloud", size: null, cloud: true, current: true },
    { name: "gpt-oss:120b-cloud", size: null, cloud: true, current: false },
    { name: "phi3:mini", size: 2_200_000_000, cloud: false, current: false },
    { name: "qwen3.5:9b", size: 6_600_000_000, cloud: false, current: false },
  ];
  let createdSessionCount = 0;
  let reviewPollCount = 0;
  let reviewJobCreated = Boolean(options.latestReviewStatus);
  let reviewApproved = false;
  let reviewDiscarded = false;
  let reviewCanceled = false;
  const reviewJobPayload = (status: string) => {
    const stressCopy = options.reviewTypographyStress;
    const stressTerm = "InterconnectionTariffSensitivityReviewPacketAlphaBravoCharlieDelta";
    const usefulnessStatus = options.reviewUsefulnessStatus ?? "passed";
    const usefulnessFlags = options.reviewUsefulnessFlags ?? [];
    const usefulnessScore = options.reviewUsefulnessScore ?? (usefulnessStatus === "low_usefulness" || usefulnessStatus === "needs_repair" ? 0.42 : usefulnessStatus === "repaired" ? 0.78 : 0.91);
    const normalizedStatus = status === "completed"
      ? "completed_awaiting_validation"
      : status === "transcribing"
        ? "running_asr"
        : status;
    const completed = ["completed_awaiting_validation", "approved", "discarded"].includes(normalizedStatus);
    const ready = completed || normalizedStatus === "approved";
    const transcribing = normalizedStatus === "running_asr";
    const queued = normalizedStatus === "queued";
    const canceled = normalizedStatus === "canceled";
    const approved = normalizedStatus === "approved";
    const discarded = normalizedStatus === "discarded";
    const steps = [
      {
        key: "conditioning",
        label: "Conditioning",
        status: queued ? "pending" : completed || transcribing || approved || discarded || canceled ? "completed" : "running",
        message: queued ? "" : completed || transcribing || approved || discarded || canceled ? "Audio conditioned for ASR." : "Converting audio for transcription.",
        progress: queued ? 0 : completed || transcribing || approved || discarded || canceled ? 100 : 35,
      },
      {
        key: "transcribing",
        label: "High-accuracy ASR",
        status: completed || approved || discarded ? "completed" : canceled ? "canceled" : transcribing ? "running" : "pending",
        message: completed || approved || discarded ? "2 transcript segments." : canceled ? "Canceled." : transcribing ? "Transcribing chunks 1-2/4." : "",
        progress: completed || approved || discarded ? 100 : transcribing || canceled ? 58 : 0,
        current: transcribing ? 2 : undefined,
        total: transcribing ? 4 : undefined,
        unit: transcribing ? "chunk" : undefined,
        elapsed_seconds: transcribing ? 4.5 : undefined,
      },
      {
        key: "reviewing",
        label: "Segment review",
        status: completed || approved || discarded ? "completed" : "pending",
        message: completed || approved || discarded ? "2 clean segments." : "",
        progress: completed || approved || discarded ? 100 : 0,
      },
      {
        key: "evaluating",
        label: "Meeting output",
        status: completed || approved || discarded ? "completed" : "pending",
        message: completed || approved || discarded ? "2 meeting cards." : "",
        progress: completed || approved || discarded ? 100 : 0,
      },
      {
        key: "completed",
        label: "Done",
        status: completed || approved || discarded ? "completed" : "pending",
        message: completed || approved || discarded ? "Review output ready for validation." : "",
        progress: completed || approved || discarded ? 100 : 0,
      },
    ];
    const cleanSegments = ready ? [
      {
          id: "review-seg-1",
        session_id: approved ? "session-review-1" : "reviewjob-123",
        start_s: 0,
        end_s: 7.2,
          text: stressCopy
            ? `BP will send the RFI log to Greg by Monday after reconciling ${stressTerm} against the utility-bill baseline.`
            : "BP will send the RFI log to Greg by Monday.",
        is_final: true,
        created_at: 1_768_000_100,
        speaker_role: "user",
        speaker_label: "BP",
        speaker_confidence: 0.96,
        speaker_match_score: 0.94,
        diarization_speaker_id: "SELF",
        source_segment_ids: ["review-seg-1"],
      },
      {
        id: "review-seg-2",
        session_id: approved ? "session-review-1" : "reviewjob-123",
        start_s: 7.4,
        end_s: 14.1,
          text: stressCopy
            ? `Sunil owns the Siemens document review path for generator-metering-retrofit-study-${stressTerm}.`
            : "Sunil owns the Siemens document review path.",
        is_final: true,
        created_at: 1_768_000_110,
        speaker_role: "other",
        speaker_label: "Other speaker",
        speaker_confidence: 0,
        speaker_match_score: 0.12,
        speaker_match_reason: "below_self_threshold",
        diarization_speaker_id: "SPEAKER_00",
        source_segment_ids: ["review-seg-2"],
      },
    ] : [];
    if (ready && options.longReviewEvidence) {
      cleanSegments.push(...Array.from({ length: 28 }, (_, index) => {
        const segmentNumber = index + 3;
        return {
          id: `review-seg-${segmentNumber}`,
          session_id: approved ? "session-review-1" : "reviewjob-123",
          start_s: 14.5 + index * 5,
          end_s: 18.5 + index * 5,
          text: stressCopy
            ? `Extended evidence row ${segmentNumber} keeps Apollo ${stressTerm}-${segmentNumber} grounded in owner, risk, tariff follow-up, metering, controls, commissioning, and procurement details.`
            : `Extended evidence row ${segmentNumber} keeps the Apollo review grounded in owner, risk, and tariff follow-up details.`,
          is_final: true,
          created_at: 1_768_000_120 + index,
          speaker_role: index % 2 === 0 ? "user" : "other",
          speaker_label: index % 2 === 0 ? "BP" : "Other speaker",
          speaker_confidence: index % 2 === 0 ? 0.92 : 0.2,
          speaker_match_score: index % 2 === 0 ? 0.9 : 0.18,
          diarization_speaker_id: index % 2 === 0 ? "SELF" : "SPEAKER_00",
          source_segment_ids: [`review-seg-${segmentNumber}`],
        };
      }));
    }
    const meetingCards = ready ? [
      {
        id: "review-card-action",
        session_id: approved ? "session-review-1" : "reviewjob-123",
        category: "action",
        title: stressCopy ? `Send RFI log for ${stressTerm}` : "Send RFI log",
        body: stressCopy
          ? `BP owns sending the RFI log to Greg by Monday with the ${stressTerm} assumptions attached.`
          : "BP owns sending the RFI log to Greg by Monday.",
        why_now: stressCopy
          ? `The corrected transcript contains an explicit owner, timing, and ${stressTerm} reference.`
          : "The corrected transcript contains an explicit owner and timing.",
        suggested_say: stressCopy
          ? `I will send the RFI log to Greg by Monday with the ${stressTerm} assumptions attached.`
          : "I will send the RFI log to Greg by Monday.",
        priority: "high",
        confidence: 0.88,
        source_segment_ids: ["review-seg-1"],
        source_type: "transcript",
        evidence_quote: stressCopy
          ? `BP will send the RFI log to Greg by Monday after reconciling ${stressTerm}.`
          : "BP will send the RFI log to Greg by Monday.",
        owner: "BP",
        due_date: "Monday",
        raw_audio_retained: false,
      },
      {
        id: "review-card-review-path",
        session_id: approved ? "session-review-1" : "reviewjob-123",
        category: "decision",
        title: stressCopy ? `Review path owner for ${stressTerm}` : "Review path owner",
        body: stressCopy
          ? `Sunil owns the Siemens document review path for generator-metering-retrofit-study-${stressTerm}.`
          : "Sunil owns the Siemens document review path.",
        why_now: stressCopy
          ? "The corrected transcript identifies the reviewer and the long technical reference path."
          : "The corrected transcript identifies the reviewer.",
        priority: "normal",
        confidence: 0.81,
        source_segment_ids: ["review-seg-2"],
        source_type: "transcript",
        evidence_quote: null,
        raw_audio_retained: false,
      },
    ] : [];
    if (ready && options.longReviewEvidence) {
      meetingCards.push(...Array.from({ length: 20 }, (_, index) => {
        const segmentNumber = index + 3;
        return {
          id: `review-card-extra-${segmentNumber}`,
          session_id: approved ? "session-review-1" : "reviewjob-123",
          category: index % 3 === 0 ? "risk" : index % 3 === 1 ? "action" : "question",
          title: stressCopy ? `Extended review card ${segmentNumber} ${stressTerm}` : `Extended review card ${segmentNumber}`,
          body: stressCopy
            ? `Track Apollo evidence row ${segmentNumber} and ${stressTerm} without letting dense supporting cards push the primary meeting summary out of view.`
            : `Track Apollo evidence row ${segmentNumber} without letting supporting cards push the primary meeting summary out of view.`,
          why_now: "Long evidence coverage is available for scroll behavior validation.",
          priority: index % 2 === 0 ? "normal" : "high",
          confidence: 0.72 + (index % 4) * 0.03,
          source_segment_ids: [`review-seg-${segmentNumber}`],
          source_type: "transcript",
          evidence_quote: stressCopy
            ? `Extended evidence row ${segmentNumber} keeps Apollo ${stressTerm} grounded in owner and tariff evidence.`
            : `Extended evidence row ${segmentNumber} keeps the Apollo review grounded.`,
          raw_audio_retained: false,
        };
      }));
    }
    const jobError = normalizedStatus === "error" ? (options.reviewJobError ?? "Review cleanup failed after transcription.") : null;
    const cleanupError = options.reviewCleanupError ?? null;
    return {
      id: "reviewjob-123",
      job_id: "reviewjob-123",
      source: "upload",
      title: stressCopy ? `Review Apollo ${stressTerm} handoff call` : "Review Apollo call",
      source_filename: "apollo-call.webm",
      filename: "apollo-call.webm",
      status: normalizedStatus,
      validation_status: approved ? "approved" : discarded ? "discarded" : canceled ? "canceled" : "pending",
      phase: approved ? "approved" : discarded ? "discarded" : canceled ? "canceled" : completed ? "awaiting_validation" : transcribing ? "transcribing" : "queued",
      progress_pct: completed || approved || discarded ? 100 : transcribing ? 41 : queued ? 0 : 5,
      message: approved
        ? "Approved and saved to Sessions."
        : discarded
          ? "Discarded without creating a Session."
          : canceled
            ? "Review canceled."
            : completed
              ? "Review output ready for validation."
              : transcribing
                ? "Transcribing chunks 1-2/4."
                : "Queued for Review.",
      queue_position: queued ? 1 : null,
      error: jobError,
      save_result: false,
      session_id: approved ? "session-review-1" : null,
      duration_seconds: ready ? 18.2 : null,
      raw_segment_count: cleanSegments.length,
      corrected_segment_count: cleanSegments.length,
      progress_percent: completed || approved || discarded ? 100 : transcribing ? 41 : queued ? 0 : 5,
      asr_backend: "nemo",
      asr_model: "stt_en_fastconformer_ctc_xxlarge",
      steps,
      clean_segments: cleanSegments,
      meeting_cards: meetingCards,
      summary: ready ? {
        session_id: approved ? "session-review-1" : "reviewjob-123",
        title: stressCopy ? `Apollo ${stressTerm} document handoff` : "Apollo document handoff",
        summary: stressCopy
          ? `The review centered on Apollo follow-up for ${stressTerm}: BP owns sending the RFI log to Greg by Monday, and Sunil owns the Siemens document review path across tariff, metering, controls, commissioning, and procurement workstreams.`
          : "The review centered on Apollo follow-up: BP owns sending the RFI log to Greg by Monday, and Sunil owns the Siemens document review path.",
        review_standard: "energy_consultant_v1",
        key_points: stressCopy
          ? [`BP has the immediate RFI-log sendout for ${stressTerm}.`, "Sunil is the named owner for Siemens document review across tariff and controls references."]
          : ["BP has the immediate RFI-log sendout.", "Sunil is the named owner for Siemens document review."],
        topics: stressCopy ? ["review", "siemens", "monday", stressTerm] : ["review", "siemens", "monday"],
        projects: stressCopy ? [`Apollo ${stressTerm}`, "Siemens document review"] : ["Apollo", "Siemens document review"],
        portfolio_rollup: {
          bp_next_actions: stressCopy ? [`BP will send the RFI log to Greg by Monday with ${stressTerm} attached.`] : ["BP will send the RFI log to Greg by Monday."],
          open_loops: stressCopy ? [`The handoff depends on ${stressTerm} reaching Greg on time without losing tariff assumptions.`] : ["The handoff depends on the RFI log reaching Greg on time."],
          cross_project_dependencies: stressCopy ? [`Apollo: ${stressTerm} depends on the RFI log reaching Greg on time.`] : ["Apollo: The handoff depends on the RFI log reaching Greg on time."],
          risk_posture: "watch",
          source_segment_ids: ["review-seg-1", "review-seg-2"],
        },
        review_metrics: {
          workstream_count: 2,
          action_count: 1,
          risk_count: 1,
          technical_finding_count: 1,
          source_count: 2,
          source_coverage: 1,
          time_span_coverage: 1,
          context_kinds: ["energy_lens", "ee_reference", "brave_web"],
        },
        project_workstreams: [
          {
            project: stressCopy ? `Apollo ${stressTerm}` : "Apollo",
            status: "active",
            actions: stressCopy ? [`BP will send the RFI log to Greg by Monday with ${stressTerm}.`] : ["BP will send the RFI log to Greg by Monday."],
            risks: stressCopy ? [`The handoff depends on ${stressTerm} reaching Greg on time.`] : ["The handoff depends on the RFI log reaching Greg on time."],
            open_questions: ["Confirm Greg's preferred handoff format."],
            owners: ["BP"],
            next_checkpoint: stressCopy ? `Monday RFI sendout for ${stressTerm}` : "Monday RFI sendout",
            source_segment_ids: ["review-seg-1"],
          },
          {
            project: "Siemens document review",
            status: "decision_made",
            decisions: ["Sunil owns the Siemens document review path."],
            owners: ["Sunil"],
            source_segment_ids: ["review-seg-2"],
          },
        ],
        technical_findings: [
          {
            topic: stressCopy ? `Siemens document review path ${stressTerm}` : "Siemens document review path",
            question: stressCopy ? `Which reference path should guide the Siemens document review for ${stressTerm}?` : "Which reference path should guide the Siemens document review?",
            assumptions: stressCopy ? [`Breaker coordination, relay timing, tariff schedules, and ${stressTerm} references are relevant to the review path.`] : ["Breaker coordination and relay timing references are relevant to the review path."],
            methods: stressCopy ? [`Compare Siemens document review needs against EE Index references, current public review practices, and ${stressTerm}.`] : ["Compare Siemens document review needs against EE Index references and current public review practices."],
            findings: stressCopy ? [`The transcript identifies Sunil as the review owner and preserves ${stressTerm} context for interpretation.`] : ["The transcript identifies Sunil as the review owner and preserves EE reference context for interpretation."],
            recommendations: stressCopy ? [`Use the EE Index, current public context, and ${stressTerm} while reviewing the Siemens documents.`] : ["Use the EE Index and current public context while reviewing the Siemens documents."],
            risks: stressCopy ? [`The review could miss current guidance if it ignores ${stressTerm}.`] : ["The review could miss current guidance if it ignores the reference context."],
            reference_context: stressCopy ? ["DOE Electrical Science Volume 1", "Current public web context", stressTerm] : ["DOE Electrical Science Volume 1", "Current public web context"],
            confidence: "medium",
            source_segment_ids: ["review-seg-2"],
          },
        ],
        decisions: ["Sunil owns the Siemens document review path."],
        actions: stressCopy ? [`BP will send the RFI log to Greg by Monday with ${stressTerm}.`] : ["BP will send the RFI log to Greg by Monday."],
        unresolved_questions: ["Confirm Greg's preferred handoff format."],
        risks: stressCopy ? [`The handoff depends on ${stressTerm} reaching Greg on time.`] : ["The handoff depends on the RFI log reaching Greg on time."],
        entities: ["BP", "Greg", "Sunil", "Siemens"],
        lessons: [],
        coverage_notes: stressCopy ? [`Summary uses corrected transcript segments and ${stressTerm} evidence.`] : ["Summary uses both corrected transcript segments."],
        reference_context: [
          {
            kind: "energy_lens",
            title: stressCopy ? `Energy Lens: operations + market ${stressTerm}` : "Energy Lens: operations + market",
            body: stressCopy ? `Utility bill analysis, tariff review, and ${stressTerm} were treated as the energy-consulting context for the meeting.` : "Utility bill analysis and tariff review were treated as the energy-consulting context for the meeting.",
            citation: "",
            source_segment_ids: ["review-seg-1"],
          },
          {
            kind: "ee_reference",
            title: "DOE Electrical Science Volume 1",
            body: "Breaker coordination and relay timing references informed the technical interpretation.",
            citation: "/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science.pdf",
            source_segment_ids: ["review-seg-2"],
          },
          {
            kind: "brave_web",
            title: "Current public web context",
            body: "Recent public guidance was checked for current review practices.",
            citation: "https://example.com/current-review-practices",
            source_segment_ids: ["review-seg-2"],
          },
        ],
        context_diagnostics: {
          energy_lens: "included",
          ee_reference_hits: 1,
          web_context_hits: 1,
        },
        diagnostics: {
          usefulness_status: usefulnessStatus,
          usefulness_score: usefulnessScore,
          usefulness_flags: usefulnessFlags,
        },
        source_segment_ids: ["review-seg-1", "review-seg-2"],
        created_at: 1_768_000_120,
        updated_at: 1_768_000_120,
      } : null,
      raw_audio_retained: true,
      diagnostics: ready ? {
        speaker_identity: {
          ready: true,
          enrollment_status: "ready",
          profile_label: "BP",
          backend_available: true,
          bp_segment_count: 1,
          other_segment_count: 1,
          unknown_segment_count: 0,
          low_confidence_count: 0,
          match_score_min: 0.12,
          match_score_mean: 0.53,
          match_score_max: 0.94,
        },
        usefulness_status: usefulnessStatus,
        usefulness_score: usefulnessScore,
        usefulness_flags: usefulnessFlags,
      } : {},
      temporary_audio_retained: !(approved || discarded || canceled),
      audio_deleted_at: approved || discarded || canceled ? 1_768_000_140 : null,
      audio_delete_error: cleanupError,
      cleanup_status: cleanupError ? "cleanup_failed" : approved || discarded || canceled ? "deleted" : "retained_for_validation",
      created_at: 1_768_000_090,
      updated_at: completed || approved || discarded ? 1_768_000_130 : 1_768_000_092,
      completed_at: completed || approved || discarded ? 1_768_000_130 : null,
      approved_at: approved ? 1_768_000_140 : null,
      discarded_at: discarded ? 1_768_000_140 : null,
      canceled_at: canceled ? 1_768_000_140 : null,
      elapsed_seconds: completed || approved || discarded ? 40 : transcribing ? 8 : 2,
      active: !completed && !approved && !discarded && !canceled,
    };
  };
  const currentReviewStatus = () => {
    if (reviewApproved) return "approved";
    if (reviewDiscarded) return "discarded";
    if (reviewCanceled) return "canceled";
    if (options.latestReviewStatus) return options.latestReviewStatus;
    if (!reviewJobCreated) return null;
    return reviewPollCount >= 2 ? "completed_awaiting_validation" : reviewPollCount > 0 ? "running_asr" : "queued";
  };
  const secondaryReviewJobPayload = () => ({
    ...reviewJobPayload("running_asr"),
    id: "reviewjob-456",
    job_id: "reviewjob-456",
    title: "Delta site tariff review",
    source_filename: "delta-site-call.webm",
    filename: "delta-site-call.webm",
    message: "Transcribing tariff review chunks.",
    queue_position: null,
    created_at: 1_768_000_080,
    updated_at: 1_768_000_125,
  });

  await page.route(`${API_ORIGIN}/api/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (method === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: corsHeaders(),
      });
      return;
    }

    if (method === "GET" && path === "/api/devices") {
      await json(route, devicesResponse);
      return;
    }

    if (method === "GET" && path === "/api/health/gpu") {
      await json(route, {
        nvidia_available: true,
        name: "NVIDIA RTX 4090",
        memory_total_mb: 24_576,
        memory_used_mb: 6_144,
        memory_free_mb: 18_432,
        gpu_pressure: "ok",
        gpu_processes: [],
        asr_cuda_available: true,
        asr_backend: "faster_whisper",
        asr_device: "cpu",
        asr_model: "small.en",
        asr_dtype: "int8",
        asr_backend_error: null,
        asr_primary_model: "small.en",
        asr_fallback_model: "tiny.en",
        asr_min_free_vram_mb: 3500,
        asr_unload_ollama_on_start: false,
        ollama_gpu_models: [],
        ollama_chat_model: ollamaChatModel,
        ollama_embed_model: "embeddinggemma",
        ollama_host: "http://127.0.0.1:11434",
        ollama_chat_host: "http://127.0.0.1:11434",
        ollama_embed_host: "http://127.0.0.1:11434",
        ollama_chat_reachable: true,
        ollama_embed_reachable: true,
        ollama_keep_alive: "30m",
        ollama_chat_keep_alive: "30m",
        ollama_embed_keep_alive: "0",
        asr_beam_size: 5,
        asr_vad_min_silence_ms: 300,
        asr_no_speech_threshold: 0.6,
        asr_log_prob_threshold: -1.0,
        asr_compression_ratio_threshold: 2.4,
        asr_min_audio_rms: 0.006,
        audio_chunk_ms: 250,
        transcription_window_seconds: 3.4,
        transcription_overlap_seconds: 0.8,
        transcription_queue_size: 8,
        partial_transcripts_enabled: true,
        partial_window_seconds: 2.0,
        partial_min_interval_seconds: 2.0,
        speaker_enrollment_sample_seconds: 8,
        speaker_enrollment_minimum_seconds: 60,
        speaker_enrollment_target_seconds: 120,
        speaker_identity_label: "BP",
        speaker_retain_raw_enrollment_audio: false,
        dedupe_similarity_threshold: 0.88,
        web_context_enabled: true,
        web_context_configured: true,
        sidecar_quality_gate_enabled: true,
        energy_lens_enabled: true,
        energy_lens_keyword_count: 121,
        test_mode_enabled: testModeEnabled,
        test_audio_run_dir: "/tmp/brain-sidecar-tests",
        ...(options.gpuHealthResponse ?? {}),
      });
      return;
    }

    if (method === "GET" && path === "/api/models/ollama") {
      await json(route, {
        selected_chat_model: ollamaChatModel,
        chat_host: "http://127.0.0.1:11434",
        models: ollamaModels.map((model) => ({ ...model, current: model.name === ollamaChatModel })),
        error: null,
      });
      return;
    }

    if (method === "POST" && path === "/api/models/ollama/chat") {
      const payload = request.postDataJSON() as { model?: string };
      ollamaChatModel = String(payload.model ?? ollamaChatModel);
      await json(route, {
        selected_chat_model: ollamaChatModel,
        chat_host: "http://127.0.0.1:11434",
        models: ollamaModels.map((model) => ({ ...model, current: model.name === ollamaChatModel })),
        error: null,
      });
      return;
    }

    if (method === "GET" && path === "/api/speaker/status") {
      await json(route, speakerStatus);
      return;
    }

    if (method === "POST" && path === "/api/microphone/test") {
      await json(route, {
        device: mockDevices[0],
        audio_source: "server_device",
        quality: {
          duration_seconds: 3,
          usable_speech_seconds: 2.6,
          speech_fraction: 0.87,
          rms: 0.041,
          peak: 0.42,
          clipping_fraction: 0,
          quality_score: 0.86,
          issues: [],
        },
        recommendation: {
          status: "good",
          title: "Mic check looks good",
          detail: "Use this setup for speaker samples: one voice, normal meeting distance, steady speech.",
        },
        mic_tuning: {
          auto_level: true,
          input_gain_db: 0,
          speech_sensitivity: "normal",
        },
        suggested_tuning: {
          auto_level: true,
          input_gain_db: 0,
          speech_sensitivity: "normal",
          reason: "Current mic tuning looks usable.",
        },
        playback_audio: {
          mime_type: "audio/wav",
          data_base64: MOCK_MIC_WAV_BASE64,
          duration_seconds: 3,
        },
        raw_audio_retained: false,
      });
      return;
    }

    if (method === "POST" && path === "/api/speaker/enrollments") {
      await json(route, {
        id: "spenr-123",
        profile_id: "self_bp",
        status: "created",
        started_at: 1_768_000_000,
        completed_at: null,
        samples: [],
        profile: speakerStatus,
        raw_audio_retained: false,
      });
      return;
    }

    if (method === "POST" && path === "/api/speaker/enrollments/spenr-123/record") {
      const sample = {
        id: "spsmp-1",
        enrollment_id: "spenr-123",
        profile_id: "self_bp",
        duration_seconds: 8,
        usable_speech_seconds: 7.2,
        quality_score: 0.88,
        issues: [],
        raw_audio_retained: false,
      };
      speakerStatus = {
        ...speakerStatus,
        enrollment_status: "needs_more_audio",
        usable_speech_seconds: 7.2,
        embedding_count: 1,
        quality_score: 0.44,
      };
      await json(route, {
        sample,
        embedding: { id: "spemb-1", vector_dim: 192, quality_score: 0.88, duration_seconds: 7.2 },
        profile: speakerStatus,
        raw_audio_retained: false,
      });
      return;
    }

    if (method === "POST" && path === "/api/speaker/enrollments/spenr-123/finalize") {
      speakerStatus = {
        ...speakerStatus,
        profile: { ...speakerStatus.profile, active: true, threshold: 0.82 },
        enrollment_status: "ready",
        ready: true,
        usable_speech_seconds: 65.1,
        embedding_count: 9,
        quality_score: 0.91,
        centroid_vector_dim: 192,
      };
      await json(route, {
        profile: speakerStatus,
        centroid_embedding_id: "spemb-centroid",
        threshold: 0.82,
        raw_audio_retained: false,
      });
      return;
    }

    if (method === "POST" && path === "/api/speaker/profiles/self/recalibrate") {
      speakerStatus = {
        ...speakerStatus,
        profile: { ...speakerStatus.profile, threshold: 0.84 },
      };
      await json(route, { profile: speakerStatus, threshold: 0.84 });
      return;
    }

    if (method === "POST" && path === "/api/speaker/profiles/self/reset") {
      speakerStatus = {
        ...speakerStatus,
        profile: { ...speakerStatus.profile, active: false, threshold: null },
        enrollment_status: "not_enrolled",
        ready: false,
        usable_speech_seconds: 0,
        embedding_count: 0,
        quality_score: 0,
        centroid_vector_dim: null,
      };
      await json(route, { profile: speakerStatus, removed: { embeddings: 3, samples: 3, enrollments: 1 } });
      return;
    }

    if (method === "POST" && path === "/api/speaker/feedback") {
      await json(route, {
        feedback: {
          id: "spfb-1",
          ...(JSON.parse(request.postData() ?? "{}") as Record<string, unknown>),
          applied_to_training: false,
        },
        profile: speakerStatus,
      });
      return;
    }

    if (method === "GET" && path === "/api/sessions") {
      if (options.abortSessionListAfterCreate && createdSessionCount > 0) {
        await route.abort("failed");
        return;
      }
      await json(route, { sessions: options.sessionsResponse ?? (reviewApproved ? [reviewSavedSession, savedSession] : [savedSession]) });
      return;
    }

    if (method === "POST" && path === "/api/sessions") {
      createdSessionCount += 1;
      const body = JSON.parse(request.postData() ?? "{}") as { title?: string | null };
      await json(route, {
        id: "session-123",
        title: body.title || "New meeting",
        status: "created",
        started_at: 1_768_000_500,
      });
      return;
    }

    if (method === "POST" && path === "/api/live/start") {
      const body = JSON.parse(request.postData() ?? "{}") as { title?: string | null };
      await json(route, {
        status: "running",
        live_id: "live-123",
        title: body.title || "Live meeting",
        audio_source: "server_device",
        raw_audio_retained: false,
        temporary_audio_retained: true,
        meeting_contract: body.meeting_contract ?? null,
      });
      return;
    }

    if (method === "POST" && path === "/api/live/live-123/stop") {
      reviewJobCreated = true;
      reviewApproved = false;
      reviewDiscarded = false;
      reviewCanceled = false;
      reviewPollCount = 0;
      await json(route, {
        status: "queued",
        live_id: "live-123",
        review_job_id: "reviewjob-123",
        queue_position: 1,
        temporary_audio_retained: true,
        raw_audio_retained: true,
        message: "Temporary audio queued for Review validation.",
      });
      return;
    }

    if (method === "POST" && path === "/api/review/jobs") {
      reviewPollCount = 0;
      reviewJobCreated = true;
      reviewApproved = false;
      reviewDiscarded = false;
      reviewCanceled = false;
      await json(route, reviewJobPayload("queued"));
      return;
    }

    if (method === "GET" && path === "/api/review/jobs") {
      const status = currentReviewStatus();
      const jobs = status ? [reviewJobPayload(status)] : [];
      if (options.reviewQueueScenario === "two_jobs") {
        jobs.push(secondaryReviewJobPayload());
      }
      await json(route, { jobs });
      return;
    }

    if (method === "GET" && path === "/api/review/jobs/latest") {
      if (options.latestReviewStatus) {
        await json(route, reviewJobPayload(options.latestReviewStatus));
        return;
      }
      const status = currentReviewStatus();
      if (!status) {
        await json(route, { detail: "No review jobs have been started." }, 404);
        return;
      }
      await json(route, reviewJobPayload(status));
      return;
    }

    if (method === "POST" && path === "/api/review/jobs/reviewjob-123/cancel") {
      reviewCanceled = true;
      await json(route, reviewJobPayload("canceled"));
      return;
    }

    if (method === "POST" && path === "/api/review/jobs/reviewjob-123/approve") {
      reviewApproved = true;
      reviewDiscarded = false;
      reviewCanceled = false;
      reviewPollCount = 3;
      await json(route, reviewJobPayload("approved"));
      return;
    }

    if (method === "POST" && path === "/api/review/jobs/reviewjob-123/discard") {
      reviewApproved = false;
      reviewDiscarded = true;
      reviewCanceled = false;
      reviewPollCount = 3;
      await json(route, reviewJobPayload("discarded"));
      return;
    }

    if (method === "GET" && path === "/api/review/jobs/reviewjob-123") {
      reviewPollCount += 1;
      await json(route, reviewJobPayload(currentReviewStatus() ?? "queued"));
      return;
    }

    if (method === "GET" && path === "/api/review/jobs/reviewjob-456") {
      await json(route, secondaryReviewJobPayload());
      return;
    }

    if (method === "GET" && path === "/api/sessions/session-review-1") {
      const completed = reviewJobPayload("approved");
      await json(route, {
        ...reviewSavedSession,
        transcript_segments: completed.clean_segments,
        note_cards: completed.meeting_cards.map((card) => ({
          id: card.id,
          session_id: card.session_id,
          kind: card.category,
          title: card.title,
          body: card.body,
          source_segment_ids: card.source_segment_ids,
          evidence_quote: card.evidence_quote,
          owner: card.owner,
          due_date: card.due_date,
          created_at: 1_768_000_130,
        })),
        summary: completed.summary,
        transcript_redacted: false,
      });
      return;
    }

    if (method === "GET" && path === "/api/sessions/session-saved-1") {
      await json(route, {
        ...savedSession,
        transcript_segments: [
          {
            id: "seg-1",
            session_id: "session-saved-1",
            start_s: 0,
            end_s: 4,
            text: "We should keep Faster-Whisper on CPU and use cloud chat.",
            is_final: true,
            created_at: 1_768_000_010,
            speaker_label: "BP",
            source_segment_ids: ["seg-1"],
          },
          {
            id: "seg-2",
            session_id: "session-saved-1",
            start_s: 4,
            end_s: 8,
            text: "Temporary sessions should not persist transcript text.",
            is_final: true,
            created_at: 1_768_000_020,
            speaker_label: "Other speaker",
            source_segment_ids: ["seg-2"],
          },
        ],
        note_cards: [
          {
            id: "note-1",
            session_id: "session-saved-1",
            kind: "decision",
            title: "CPU ASR default",
            body: "Faster-Whisper on CPU plus cloud chat is the normal stable pair.",
            source_segment_ids: ["seg-1"],
            evidence_quote: "Faster-Whisper on CPU",
            created_at: 1_768_000_040,
          },
        ],
        summary: {
          session_id: "session-saved-1",
          title: "Model planning brief",
          summary: "The call settled on CPU Faster-Whisper and cloud chat defaults.",
          topics: ["faster-whisper", "cloud chat"],
          decisions: ["Use CPU ASR defaults."],
          actions: [],
          unresolved_questions: [],
          entities: ["Faster-Whisper"],
          lessons: [],
          source_segment_ids: ["seg-1"],
          created_at: 1_768_000_060,
          updated_at: 1_768_000_060,
        },
        transcript_redacted: false,
      });
      return;
    }

    if (method === "PATCH" && path === "/api/sessions/session-saved-1") {
      const body = JSON.parse(request.postData() ?? "{}") as { title?: string };
      await json(route, { ...savedSession, title: body.title ?? savedSession.title });
      return;
    }

    if (method === "PATCH" && path === "/api/sessions/session-123") {
      const body = JSON.parse(request.postData() ?? "{}") as { title?: string };
      await json(route, {
        id: "session-123",
        title: body.title ?? "New meeting",
        status: "created",
        started_at: 1_768_000_500,
        save_transcript: null,
        retention: "empty",
        transcript_count: 0,
        note_count: 0,
        summary_exists: false,
        raw_audio_retained: false,
      });
      return;
    }

    if (method === "POST" && path === "/api/sessions/session-123/start") {
      if (options.startSessionResponse) {
        await json(route, options.startSessionResponse.body ?? { ok: false }, options.startSessionResponse.status ?? 500);
      } else {
        await json(route, { ok: true });
      }
      return;
    }

    if (method === "POST" && path === "/api/sessions/session-123/stop") {
      await json(route, { ok: true });
      return;
    }

    if (method === "POST" && path === "/api/sessions/session-123/pause") {
      await json(route, { status: "paused", session_id: "session-123", raw_audio_retained: false });
      return;
    }

    if (method === "POST" && path === "/api/sessions/session-123/resume") {
      await json(route, { status: "listening", session_id: "session-123", raw_audio_retained: false });
      return;
    }

    if (method === "POST" && path === "/api/input-files") {
      await json(route, {
        path: "/tmp/brain-sidecar-tests/input-files/uploaded-input.dat",
        filename: "uploaded-input.dat",
        size_bytes: 12,
      });
      return;
    }

    if (method === "POST" && path === "/api/test-mode/audio/prepare") {
      const body = JSON.parse(request.postData() ?? "{}") as {
        source_path?: string;
        expected_terms?: string[];
      };
      await json(route, {
        run_id: "testrun-123",
        source_path: body.source_path ?? "/tmp/source.m4a",
        fixture_wav: "/tmp/brain-sidecar-tests/testrun-123/input.wav",
        duration_seconds: options.testPreparedDurationSeconds ?? 12.5,
        artifact_dir: "/tmp/brain-sidecar-tests/testrun-123",
        report_path: "/tmp/brain-sidecar-tests/testrun-123/report.json",
        expected_terms: body.expected_terms ?? [],
      });
      return;
    }

    if (method === "POST" && path === "/api/test-mode/runs/testrun-123/report") {
      await json(route, {
        run_id: "testrun-123",
        report_path: "/tmp/brain-sidecar-tests/testrun-123/report.json",
      });
      return;
    }

    if (method === "POST" && path === "/api/library/roots") {
      await json(route, { id: "root-1" });
      return;
    }

    if (method === "GET" && path === "/api/library/roots") {
      await json(route, { roots: ["/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering"] });
      return;
    }

    if (method === "GET" && path === "/api/library/chunks") {
      await json(route, {
        sources: [
          {
            source_path: "/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe-hdbk-1011-92-vol-1-electrical-science.pdf",
            title: "DOE Electrical Science Volume 1",
            chunk_count: 3,
            updated_at: 1_768_000_000,
            first_chunk_index: 0,
          },
        ],
        chunks: [
          {
            id: "chunk-1",
            source_path: "/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe-hdbk-1011-92-vol-1-electrical-science.pdf",
            title: "DOE Electrical Science Volume 1",
            chunk_index: 0,
            text: "Circuit protection references describe breaker coordination, fault current, and relay timing.",
            metadata: {},
            updated_at: 1_768_000_000,
          },
        ],
      });
      return;
    }

    if (method === "POST" && path === "/api/library/reindex") {
      await json(route, { chunks_indexed: 42 });
      return;
    }

    if (method === "POST" && path === "/api/recall/search") {
      await json(route, {
        hits: [
          {
            source_type: "file",
            source_id: "notes.md",
            text: "Prior planning note about the Apollo rollout.",
            score: 0.93,
            metadata: {},
          },
        ],
      });
      return;
    }

    if (method === "POST" && path === "/api/sidecar/query") {
      const body = JSON.parse(request.postData() ?? "{}") as { query?: string };
      await json(route, {
        query: body.query ?? "",
        sanitized_web_query: body.query ?? "",
        web_skip_reason: null,
        raw_audio_retained: false,
        sections: {
          prior_transcript: [
            {
              id: "manual-recall-card",
              session_id: "session-123",
              category: "memory",
              title: "Apollo planning note",
              body: "Prior planning note about the Apollo rollout.",
              why_now: "Returned for the typed search.",
              priority: "high",
              confidence: 0.93,
              source_segment_ids: [],
              source_type: "saved_transcript",
              sources: [],
              citations: [],
              ephemeral: true,
              card_key: "manual:recall:file:notes",
              explicitly_requested: true,
              raw_audio_retained: false,
            },
          ],
          technical_references: [
            {
              id: "manual-reference-card",
              session_id: "session-123",
              category: "memory",
              title: "DOE Electrical Science Volume 1",
              body: "Circuit protection references describe breaker coordination, fault current, and relay timing.",
              suggested_say: "The DOE reference points back to breaker coordination and relay timing as the grounding concepts.",
              why_now: "Returned for the typed technical reference lookup.",
              priority: "high",
              confidence: 0.86,
              source_segment_ids: [],
              source_type: "local_file",
              sources: [],
              citations: ["/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe-hdbk-1011-92-vol-1-electrical-science.pdf"],
              ephemeral: true,
              card_key: "manual:reference:doe-electrical-science",
              explicitly_requested: true,
              raw_audio_retained: false,
            },
          ],
          current_public_web: [
            {
              id: "manual-web-card",
              session_id: "session-123",
              category: "web",
              title: `Web context: ${body.query ?? "manual query"}`,
              body: "I found a few current references: current API docs and release guidance.",
              why_now: "Returned for the typed web lookup.",
              priority: "high",
              confidence: 0.78,
              source_segment_ids: [],
              source_type: "brave_web",
              sources: [{ title: "Current API docs", url: "https://example.com/current-api-docs" }],
              citations: ["https://example.com/current-api-docs"],
              ephemeral: true,
              card_key: "manual:web:apollo-rollout",
              explicitly_requested: true,
              raw_audio_retained: false,
            },
          ],
          suggested_meeting_contribution: [
            {
              id: "manual-contribution-card",
              session_id: "session-123",
              category: "contribution",
              title: "Suggested meeting contribution",
              body: "Use the reference and current web result as grounded context.",
              suggested_say: "It may help to check the rollout against the reference criteria and the current release guidance.",
              why_now: "BP explicitly asked Sidecar.",
              priority: "high",
              confidence: 0.8,
              source_segment_ids: [],
              source_type: "local_file",
              sources: [],
              citations: [],
              ephemeral: true,
              card_key: "manual:contribution:apollo-rollout",
              explicitly_requested: true,
              raw_audio_retained: false,
            },
          ],
        },
        cards: [
          {
            id: "manual-recall-card",
            session_id: "session-123",
            category: "memory",
            title: "Apollo planning note",
            body: "Prior planning note about the Apollo rollout.",
            why_now: "Returned for the typed search.",
            priority: "high",
            confidence: 0.93,
            source_segment_ids: [],
            source_type: "saved_transcript",
            sources: [],
            citations: [],
            ephemeral: true,
            card_key: "manual:recall:file:notes",
            explicitly_requested: true,
            raw_audio_retained: false,
          },
          {
            id: "manual-reference-card",
            session_id: "session-123",
            category: "memory",
            title: "DOE Electrical Science Volume 1",
            body: "Circuit protection references describe breaker coordination, fault current, and relay timing.",
            suggested_say: "The DOE reference points back to breaker coordination and relay timing as the grounding concepts.",
            why_now: "Returned for the typed technical reference lookup.",
            priority: "high",
            confidence: 0.86,
            source_segment_ids: [],
            source_type: "local_file",
            sources: [],
            citations: ["/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe-hdbk-1011-92-vol-1-electrical-science.pdf"],
            ephemeral: true,
            card_key: "manual:reference:doe-electrical-science",
            explicitly_requested: true,
            raw_audio_retained: false,
          },
          {
            id: "manual-web-card",
            session_id: "session-123",
            category: "web",
            title: `Web context: ${body.query ?? "manual query"}`,
            body: "I found a few current references: current API docs and release guidance.",
            why_now: "Returned for the typed web lookup.",
            priority: "high",
            confidence: 0.78,
            source_segment_ids: [],
            source_type: "brave_web",
            sources: [{ title: "Current API docs", url: "https://example.com/current-api-docs" }],
            citations: ["https://example.com/current-api-docs"],
            ephemeral: true,
            card_key: "manual:web:apollo-rollout",
            explicitly_requested: true,
            raw_audio_retained: false,
          },
          {
            id: "manual-contribution-card",
            session_id: "session-123",
            category: "contribution",
            title: "Suggested meeting contribution",
            body: "Use the reference and current web result as grounded context.",
            suggested_say: "It may help to check the rollout against the reference criteria and the current release guidance.",
            why_now: "BP explicitly asked Sidecar.",
            priority: "high",
            confidence: 0.8,
            source_segment_ids: [],
            source_type: "local_file",
            sources: [],
            citations: [],
            ephemeral: true,
            card_key: "manual:contribution:apollo-rollout",
            explicitly_requested: true,
            raw_audio_retained: false,
          },
        ],
      });
      return;
    }

    if (method === "POST" && path === "/api/web-context/search") {
      const body = JSON.parse(request.postData() ?? "{}") as { query?: string; session_id?: string | null };
      await json(route, {
        query: body.query ?? "",
        enabled: true,
        configured: true,
        note: {
	          id: "manual-web-note",
	          kind: "topic",
	          title: `Web context: ${body.query ?? "manual query"}`,
	          body: "I found a few current references: current API docs and release guidance.",
	          ephemeral: true,
	          source_type: "brave_web",
	          source_segment_ids: [],
	          sources: [
            { title: "Current API docs", url: "https://example.com/current-api-docs" },
          ],
        },
      });
      return;
    }

    await route.fulfill({
      status: 404,
      headers: corsHeaders(),
      contentType: "application/json",
      body: JSON.stringify({ detail: `No mock registered for ${method} ${path}` }),
    });
  });
}

export async function emitSessionEvent(page: Page, event: MockEvent) {
  await page.evaluate(
    ({ type, payload }) => {
      window.__brainSidecarEmit(type, payload);
    },
    event,
  );
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    headers: corsHeaders(),
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "content-type",
      "access-control-allow-methods": "GET,POST,PATCH,OPTIONS",
  };
}

declare global {
  interface Window {
    __brainSidecarEmit: (type: string, payload: Record<string, unknown>) => void;
    __brainSidecarEventSourceUrls: string[];
  }
}
