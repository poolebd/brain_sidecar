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
  devicesResponse?: MockDeviceResponse;
  gpuHealthResponse?: Record<string, unknown>;
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
          .find((candidate) => candidate.url.includes("/api/sessions/"))
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
    target_speech_seconds: 20,
    minimum_speech_seconds: 15,
    embedding_count: 0,
    centroid_vector_dim: null,
    same_speaker_scores: { count: 0, min: null, mean: null },
    quality_score: 0,
    raw_audio_retained: false,
    recent_segments: [],
  };

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
        asr_backend: "nemotron_streaming",
        asr_streaming_supported: true,
        asr_streaming_chunk_ms: 160,
        nemotron_model_id: "nvidia/nemotron-speech-streaming-en-0.6b",
        nemotron_loaded: false,
        asr_model: "nvidia/nemotron-speech-streaming-en-0.6b",
        asr_dtype: "float32",
        asr_backend_error: null,
        asr_primary_model: "medium.en",
        asr_fallback_model: "small.en",
        asr_min_free_vram_mb: 3500,
        asr_unload_ollama_on_start: true,
        ollama_gpu_models: ["llama3.1"],
        ollama_chat_model: "llama3.1",
        ollama_keep_alive: "0s",
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
        streaming_partials_enabled: true,
        streaming_stable_final_chunks: 3,
        speaker_enrollment_sample_seconds: 8,
        speaker_identity_label: "BP",
        speaker_retain_raw_enrollment_audio: false,
        dedupe_similarity_threshold: 0.88,
        web_context_enabled: true,
        web_context_configured: true,
        sidecar_quality_gate_enabled: true,
        test_mode_enabled: testModeEnabled,
        test_audio_run_dir: "/tmp/brain-sidecar-tests",
        ...(options.gpuHealthResponse ?? {}),
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
        usable_speech_seconds: 20.1,
        embedding_count: 3,
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

    if (method === "GET" && path === "/api/work-memory/status") {
      await json(route, {
        projects: 21,
        evidence: 88,
        sources: 412,
        disabled_sources: 14,
        latest_index_at: 1_768_000_000,
        source_groups: {
          opc_history: 180,
          consulting_history: 160,
          navy_history: 72,
        },
        default_roots: ["/home/bp/Nextcloud2/Job Hunting"],
        guardrail: "Current employer/client material is guardrail context only.",
      });
      return;
    }

    if (method === "POST" && path === "/api/sessions") {
      await json(route, { id: "session-123" });
      return;
    }

    if (method === "POST" && path === "/api/sessions/session-123/start") {
      await json(route, { ok: true });
      return;
    }

    if (method === "POST" && path === "/api/sessions/session-123/stop") {
      await json(route, { ok: true });
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
        duration_seconds: 12.5,
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

    if (method === "POST" && path === "/api/library/reindex") {
      await json(route, { chunks_indexed: 42 });
      return;
    }

    if (method === "POST" && path === "/api/work-memory/reindex") {
      await json(route, {
        projects_indexed: 21,
        evidence_indexed: 88,
        sources_indexed: 412,
      });
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
          pas_past_work: [
            {
              id: "manual-work-card",
              session_id: "session-123",
              category: "work_memory",
              title: "Online Generator Monitoring - T.A. Smith",
              body: "Monitoring is only useful when measurements map to failure modes and decisions.",
              suggested_say: "I saw a similar pattern at T.A. Smith: map measurements to the decisions they change.",
              why_now: "shared signals around generator monitoring, failure modes",
              priority: "high",
              confidence: 0.86,
              source_segment_ids: [],
              source_type: "work_memory",
              sources: [],
              citations: ["/home/bp/Nextcloud2/_library/_shoalstone/past work/OPC/2021 OGM/OGMS spec.txt"],
              ephemeral: true,
              card_key: "manual:work:wproj-ogm",
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
              body: "Use the prior planning note and OGM lesson as grounded context.",
              suggested_say: "It may help to map this rollout signal to the decision it changes.",
              why_now: "BP explicitly asked Sidecar.",
              priority: "high",
              confidence: 0.8,
              source_segment_ids: [],
              source_type: "work_memory",
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
            id: "manual-work-card",
            session_id: "session-123",
            category: "work_memory",
            title: "Online Generator Monitoring - T.A. Smith",
            body: "Monitoring is only useful when measurements map to failure modes and decisions.",
            suggested_say: "I saw a similar pattern at T.A. Smith: map measurements to the decisions they change.",
            why_now: "shared signals around generator monitoring, failure modes",
            priority: "high",
            confidence: 0.86,
            source_segment_ids: [],
            source_type: "work_memory",
            sources: [],
            citations: ["/home/bp/Nextcloud2/_library/_shoalstone/past work/OPC/2021 OGM/OGMS spec.txt"],
            ephemeral: true,
            card_key: "manual:work:wproj-ogm",
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
            body: "Use the prior planning note and OGM lesson as grounded context.",
            suggested_say: "It may help to map this rollout signal to the decision it changes.",
            why_now: "BP explicitly asked Sidecar.",
            priority: "high",
            confidence: 0.8,
            source_segment_ids: [],
            source_type: "work_memory",
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

    if (method === "POST" && path === "/api/work-memory/search") {
      await json(route, {
        cards: [
          {
            project_id: "wproj-ogm",
            title: "Online Generator Monitoring - T.A. Smith",
            organization: "Oglethorpe Power",
            date_range: "Mar 2019 - June 2021",
            score: 0.86,
            confidence: 0.92,
            reason: "shared signals around generator monitoring, failure modes",
            lesson: "Monitoring is only useful when measurements map to failure modes and decisions.",
            citations: ["/home/bp/Nextcloud2/_library/_shoalstone/past work/OPC/2021 OGM/OGMS spec.txt"],
            text: "This conversation is reminiscent of Online Generator Monitoring - T.A. Smith (Oglethorpe Power, Mar 2019 - June 2021). Similarity: shared signals around generator monitoring, failure modes. Useful reminder: Monitoring is only useful when measurements map to failure modes and decisions.",
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
    "access-control-allow-methods": "GET,POST,OPTIONS",
  };
}

declare global {
  interface Window {
    __brainSidecarEmit: (type: string, payload: Record<string, unknown>) => void;
    __brainSidecarEventSourceUrls: string[];
  }
}
