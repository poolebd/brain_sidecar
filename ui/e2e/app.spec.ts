import { expect, test, type Page } from "@playwright/test";
import { emitSessionEvent, installMockEventSource, mockApi, mockDevices } from "./mocks";

test.beforeEach(async ({ page }) => {
  await installMockEventSource(page);
  await mockApi(page);
  await page.goto("/");
});

test("loads mocked device and GPU state", async ({ page }) => {
  await expect(page.getByRole("heading", { name: "Live field" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Meeting Output" })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "midnight");
  await expect(page.getByLabel("Meeting Focus summary")).toContainText("Focus");
  await expect(page.getByLabel("Meeting Focus goal")).toHaveCount(0);
  await expect(page.getByLabel("Live field empty state")).toContainText("Ready. Start listening or ask Sidecar.");
  const liveBox = await page.locator("#transcript").boundingBox();
  expect(liveBox).not.toBeNull();
  expect(liveBox!.y).toBeLessThan(220);
  await expect(page.getByRole("search")).toBeVisible();
  const toolsButton = page.getByRole("button", { name: "Tools" });
  await expect(toolsButton).toHaveAttribute("aria-expanded", "false");
  await toolsButton.click();
  await expect(toolsButton).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByLabel("Audio source")).toHaveCount(0);
  await expect(page.getByLabel("USB microphone")).toHaveCount(0);
  await expect(page.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone");
  await expect(page.getByLabel("Guided mic tuning")).toContainText("Auto Level");
  await expect(page.getByLabel("Tools drawer").getByRole("button", { name: "Close" })).toHaveCount(0);
  await openDrawerRegion(page, "System");
  await expect(page.getByText("NVIDIA RTX 4090 · 6144/24576 MB")).toBeVisible();
  await expect(page.getByLabel("VRAM usage 25%")).toBeVisible();
  await expect(page.getByText("CUDA ready")).toBeVisible();
  await expect(page.getByLabel("Tools drawer").getByText("21 projects / 412 sources")).toBeVisible();
  await toolsButton.click();
  await expect(toolsButton).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByLabel("Tools drawer")).not.toBeVisible();
  await toolsButton.click();
  await page.keyboard.press("Escape");
  await expect(toolsButton).toHaveAttribute("aria-expanded", "false");
});

test("clamps stale high mic boost and warns before capture", async ({ page }) => {
  await page.evaluate(() => {
    window.localStorage.setItem("brain-sidecar-mic-tuning-v1", JSON.stringify({
      auto_level: true,
      input_gain_db: 24,
      speech_sensitivity: "normal",
    }));
  });
  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();

  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByLabel("Input Boost")).toHaveValue("12");
  await expect(drawer.getByLabel("Guided mic tuning")).toContainText("+12 dB");
  await expect(drawer.getByLabel("Guided mic tuning")).toContainText("High boost can clip");
});

test("restores Meeting Focus controls from localStorage", async ({ page }) => {
  await page.evaluate(() => {
    window.localStorage.setItem("brain-sidecar-meeting-focus-v1", JSON.stringify({
      goal: "Track owners for the outage review.",
      mode: "balanced",
      reminders: {
        owners: true,
        questions: false,
        risks: true,
        contributions: false,
        brief: true,
      },
    }));
  });
  await page.reload();

  const focus = page.getByLabel("Meeting Focus");
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Balanced");
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Track owners for the outage review.");
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveCount(0);
  await focus.getByRole("button", { name: "Edit" }).click();
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveValue("Track owners for the outage review.");
  await expect(focus.getByRole("button", { name: "Balanced" })).toHaveAttribute("aria-pressed", "true");
  await expect(focus.getByLabel("Open questions")).not.toBeChecked();
  await expect(focus.getByLabel("Risks/dependencies")).toBeChecked();
  await focus.getByLabel("Meeting Focus goal").fill("Track owners and risk decisions.");
  await focus.getByRole("button", { name: "Assertive" }).click();
  await focus.getByLabel("Open questions").check();
  await focus.locator(".meeting-focus-editor").getByRole("button", { name: "Done" }).click();
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveCount(0);
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Assertive");
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Track owners and risk decisions.");

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await page.getByRole("button", { name: "Start Listening" }).click();
  const body = JSON.parse((await startRequest).postData() ?? "{}");
  expect(body.meeting_contract).toMatchObject({
    goal: "Track owners and risk decisions.",
    mode: "assertive",
  });
  expect(body.meeting_contract.reminders).toEqual([
    "Track owners and ownership ambiguity.",
    "Track open questions and clarifications.",
    "Track risks, dependencies, and blocked paths.",
    "Prepare a post-call brief with evidence.",
  ]);
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveCount(0);
});

test("shows active USB mic capture as in use instead of missing", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    devicesResponse: {
      devices: [{ ...mockDevices[0], in_use: true, selection_reason: "USB capture; USB mic is in use by active capture" }],
      selected_device: { ...mockDevices[0], in_use: true, selection_reason: "USB capture; USB mic is in use by active capture" },
      server_mic_available: true,
      selection_reason: "USB capture; USB mic is in use by active capture",
    },
  });
  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();

  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone (in use)");
  await expect(drawer.getByLabel("Server microphone")).toContainText("USB mic is in use by active capture");
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Listening" })).toBeDisabled();
});

test("hides recorded audio test controls when test mode is disabled", async ({ page }) => {
  await expect(page.getByRole("region", { name: "Recorded audio test" })).toHaveCount(0);
});

test("starts a mocked session and renders SSE updates", async ({ page }) => {
  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));

  await page.getByRole("button", { name: "Start Listening" }).click();
  const request = await startRequest;
  const requestBody = JSON.parse(request.postData() ?? "{}");
  expect(requestBody).toMatchObject({
    device_id: null,
    audio_source: "server_device",
    save_transcript: false,
    mic_tuning: {
      auto_level: true,
      input_gain_db: 0,
      speech_sensitivity: "normal",
    },
  });
  expect(requestBody.meeting_contract).toMatchObject({
    mode: "quiet",
    goal: expect.stringContaining("Offload meeting obligations"),
    reminders: expect.arrayContaining(["Track owners and ownership ambiguity."]),
  });
  await expect(page.getByLabel("Session save mode status")).toContainText("Listening");
  await expect(page.getByLabel("Contract active")).toContainText("Contract active");
  const meetingOutput = page.getByRole("region", { name: "Meeting Output" });
  await expect(meetingOutput).toContainText("Silent by design");
  await expect(meetingOutput).toContainText("No grounded cards yet. Unsupported/noisy cards are being suppressed.");
  await expect(meetingOutput).not.toContainText("No actions yet.");
  await expect(page.getByLabel("Live field empty state")).toContainText("Listening... grounded cards will appear when there is enough evidence.");
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/sessions/session-123/events",
  );
  expect(await page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/events",
  );

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: { status: "listening", queue_depth: 3, dropped_windows: 1 },
  });
  await emitSessionEvent(page, {
    type: "transcript_partial",
    payload: {
      id: "line-1-preview",
      text: "We need to follow up with Alex",
      start_s: 1.2,
      end_s: 3.0,
      is_final: false,
      asr_model: "whisper-large-v3",
      queue_depth: 2,
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  await expect(page.getByLabel("Mic transcript item").filter({ hasText: "Alex" })).toContainText("Preview");
  await emitSessionEvent(page, {
    type: "transcript_partial",
    payload: {
      id: "line-1-preview-2",
      text: "We need to follow up with the platform team",
      start_s: 1.3,
      end_s: 3.2,
      is_final: false,
      asr_model: "whisper-large-v3",
      queue_depth: 2,
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  await expect(page.getByLabel("Mic transcript item").filter({ hasText: "platform team" })).toContainText("Preview");
  await expect(page.locator("#transcript")).not.toContainText("Alex");
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "line-1",
      text: "We need to follow up with the platform team tomorrow.",
      start_s: 1.2,
      end_s: 4.8,
      asr_model: "whisper-large-v3",
      queue_depth: 2,
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "line-1",
      replaces_segment_id: "line-1",
      source_segment_ids: ["line-1", "line-1b"],
      text: "We need to follow up with the platform team tomorrow before the readiness check.",
      start_s: 1.2,
      end_s: 5.4,
      asr_model: "whisper-large-v3",
      queue_depth: 2,
      final_segments_replaced: 1,
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
	  await emitSessionEvent(page, {
	    type: "note_update",
	    payload: {
	      id: "note-1",
	      kind: "action",
	      title: "Platform follow-up",
	      body: "Schedule a short sync about deployment readiness.",
	      source_segment_ids: ["line-1b"],
        evidence_quote: "follow up with the platform team tomorrow",
	    },
	  });
	  await emitSessionEvent(page, {
	    type: "recall_hit",
	    payload: {
      source_type: "session",
      source_id: "previous-session",
	      text: "A previous meeting mentioned deployment readiness risks.",
	      score: 0.91,
	      metadata: {},
	      source_segment_ids: ["line-1b"],
	    },
	  });
  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "card-1",
      session_id: "session-123",
      category: "contribution",
      title: "Name the release owner",
      body: "The release thread needs one owner before the deployment readiness check.",
      suggested_say: "It sounds like we should name one release owner before the readiness check.",
      why_now: "Recent speech mentioned platform follow-up and deployment readiness.",
      priority: "high",
      confidence: 0.92,
      source_segment_ids: ["line-1b"],
      source_type: "transcript",
      evidence_quote: "follow up with the platform team tomorrow",
      card_key: "contribution:release-owner",
      ephemeral: true,
      raw_audio_retained: false,
    },
  });

  await expect(page.getByLabel("Runtime status")).toContainText("listening");
  await expect(page.getByText("1 heard").first()).toBeVisible();
  await expect(page.locator("#transcript")).not.toContainText("Preview");
  const transcriptItem = page.getByLabel("Mic transcript item").filter({ hasText: "platform team tomorrow" });
  await expect(transcriptItem).toHaveCount(1);
  await expect(transcriptItem).toContainText("before the readiness check");
  await expect(transcriptItem).not.toContainText("You");
  const liveRow = page.getByRole("article", { name: "Transcript and sidecar row" }).filter({ hasText: "platform team tomorrow" });
  await expect(liveRow.getByRole("heading", { name: "Platform follow-up" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Actions summary" })).toContainText("Platform follow-up");
  await expect(page.getByRole("region", { name: "Decisions summary" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Questions / Clarifications summary" })).toHaveCount(0);
  await expect(liveRow).toContainText("Name the release owner");
  await expect(liveRow).toContainText("It sounds like we should name one release owner");
  const contributionCard = liveRow.getByLabel("Say this context card").filter({ hasText: "Name the release owner" });
  await expect(contributionCard).toContainText("Current meeting");
  await expect(contributionCard).toContainText("Evidence-backed");
  await contributionCard.getByRole("button", { name: "Evidence" }).click();
  await expect(contributionCard.getByLabel("Evidence for Name the release owner")).toContainText("platform team tomorrow");
  await contributionCard.getByRole("button", { name: "Jump to source" }).click();
  await expect(transcriptItem).toHaveClass(/evidence-highlight/);
  const contextSection = page.getByLabel("Context cards");
  await contextSection.getByText("Context / past transcript / web").click();
  await expect(contextSection).toContainText("A previous meeting mentioned deployment readiness risks.");
  await expect(page.locator("#transcript").getByText("A previous meeting mentioned deployment readiness risks.")).toHaveCount(0);
  await expect(page.getByLabel("Capture queue status")).toContainText("Queue 2");
  await expect(page.locator("#transcript")).not.toContainText("whisper-large-v3");
  await expect(page.locator("#transcript")).not.toContainText("score 0.91");
  await expect(page.locator("#transcript")).not.toContainText("previous-session");
  const transcriptBox = await page.locator("#transcript").boundingBox();
  const backendBox = await page.locator("#recall").boundingBox();
  expect(transcriptBox).not.toBeNull();
  expect(backendBox).not.toBeNull();
  expect(backendBox!.x).toBeGreaterThan(transcriptBox!.x);
});

test("makes saved transcript recording an explicit capture mode", async ({ page }) => {
  const modeGroup = page.getByRole("group", { name: "Session save mode" });
  await expect(modeGroup.getByRole("button", { name: /Listen/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel("Session save mode status")).toContainText("Not saved");

  await modeGroup.getByRole("button", { name: /Record/ }).click();
  await expect(modeGroup.getByRole("button", { name: /Record/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel("Session save mode status")).toContainText("Saved transcript");
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Recording" })).toBeVisible();

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));

  await page.getByRole("banner").getByRole("button", { name: "Start Recording" }).click();
  const request = await startRequest;
  const body = JSON.parse(request.postData() ?? "{}");
  expect(body.save_transcript).toBe(true);
  await expect(page.getByLabel("Session save mode status")).toContainText("Recording");
  await expect(modeGroup.getByRole("button", { name: /Listen/ })).toBeDisabled();
});

test("shows speaker identity controls instead of legacy ASR voice training", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  const speaker = await openDrawerRegion(page, "Speaker identity");
  await expect(speaker).toBeVisible();
  await expect(speaker).toContainText("not enrolled");
  await expect(speaker).toContainText("0.0s usable");
  await expect(speaker).toContainText("15.0s needed");
  await expect(speaker.getByLabel("Speaker training steps")).toContainText("1 Start");
  await expect(page.getByRole("region", { name: "Voice profile" })).toHaveCount(0);
  await expect(page.getByText("ASR guidance preview")).toHaveCount(0);
  await expect(page.getByLabel("Guided mic tuning")).toContainText("Input Boost");
  await page.getByLabel("Speech Sensitivity").getByRole("button", { name: "Quiet" }).click();
  const micRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/microphone/test")
  ));
  await page.getByRole("button", { name: "Test Mic" }).click();
  expect(JSON.parse((await micRequest).postData() ?? "{}")).toMatchObject({
    device_id: null,
    audio_source: "server_device",
    fixture_wav: null,
    seconds: 3,
    mic_tuning: {
      auto_level: true,
      input_gain_db: 0,
      speech_sensitivity: "quiet",
    },
  });
  await expect(page.getByLabel("Microphone check")).toContainText("Mic check looks good");
  await expect(page.getByLabel("Microphone check")).toContainText("2.6s speech");
  const playback = page.getByLabel("Microphone test playback");
  await expect(playback).toBeVisible();
  await expect(playback.locator("audio")).toHaveAttribute("src", /^blob:/);
  await expect(playback).toContainText("Raw audio is not saved");
  await speaker.getByText("Advanced speaker controls").click();
  await expect(speaker).toContainText("embeddings");
  await speaker.getByText("Review learned speaker profile").click();
  await expect(speaker.getByRole("button", { name: "Merge speakers" })).toBeVisible();
  await expect(speaker.getByRole("button", { name: "Split speaker" })).toBeVisible();
  await expect(speaker.getByRole("button", { name: "Rename speaker" })).toBeVisible();

  await speaker.getByRole("button", { name: "Set Up BP Label" }).first().click();
  await expect(speaker.getByRole("button", { name: "Record 8s Sample" })).toBeVisible();
  await expect(speaker.getByLabel("Speaker recording prompt")).toContainText("Record only BP");
  await expect(speaker.getByLabel("Speaker recording prompt")).toContainText("speaker labels");
  await expect(speaker.getByLabel("Suggested speaker training prompts")).toContainText("Say one of these");
  await expect(speaker.getByLabel("Suggested speaker training prompts")).toContainText("normal desk setup");
  await expect(speaker.getByLabel("Suggested speaker training prompts")).toContainText("should not label them as me");
  const recordRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/speaker/enrollments/spenr-123/record")
  ));
  await speaker.getByRole("button", { name: "Record 8s Sample" }).click();
  expect(JSON.parse((await recordRequest).postData() ?? "{}")).toMatchObject({
    device_id: null,
    audio_source: "server_device",
    fixture_wav: null,
    mic_tuning: {
      auto_level: true,
      input_gain_db: 0,
      speech_sensitivity: "normal",
    },
  });
  await expect(speaker).toContainText("7.2s");
});

test("renders high-confidence BP speaker labels from transcript events", async ({ page }) => {
  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "bp-line-1",
      text: "This is my part of the meeting.",
      start_s: 0,
      end_s: 2,
      speaker_role: "user",
      speaker_label: "BP",
      speaker_confidence: 0.94,
    },
  });

  const transcriptItem = page.getByLabel("BP transcript item").filter({ hasText: "my part of the meeting" });
  await expect(transcriptItem).toBeVisible();
  await expect(transcriptItem).not.toContainText("You");
});

test("supports contribution card copy, pin, and dismiss controls", async ({ page }) => {
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as unknown as { __copiedSuggestion?: string }).__copiedSuggestion = text;
        },
      },
    });
  });
  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "card-copy",
      session_id: "session-123",
      category: "question",
      title: "Clarify rollback timing",
      body: "The rollback timing is still ambiguous.",
      suggested_ask: "What is the rollback cutoff if validation fails?",
      why_now: "The current thread mentions readiness but not the fallback cutoff.",
      priority: "high",
      confidence: 0.91,
      source_segment_ids: [],
      source_type: "transcript",
      card_key: "question:rollback-cutoff",
      ephemeral: true,
      raw_audio_retained: false,
    },
  });

  const liveField = page.locator("#transcript");
  const card = liveField.getByLabel("Ask this context card").filter({ hasText: "Clarify rollback timing" });
  await expect(card).toBeVisible();
  await card.getByRole("button", { name: "Copy" }).click();
  await expect.poll(() => page.evaluate(() => (window as unknown as { __copiedSuggestion?: string }).__copiedSuggestion)).toBe(
    "What is the rollback cutoff if validation fails?",
  );

  await card.getByRole("button", { name: "Pin card" }).click();
  await expect(card.getByRole("button", { name: "Unpin card" })).toBeVisible();
  await card.getByRole("button", { name: "Dismiss card" }).click();
  await expect(liveField.getByLabel("Ask this context card").filter({ hasText: "Clarify rollback timing" })).toHaveCount(0);
});

test("shows GPU cleanup and ASR loading progress in capture controls", async ({ page }) => {
  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: {
      status: "freeing_gpu",
      memory_free_mb: 1024,
      memory_total_mb: 24576,
      gpu_pressure: "critical",
      ollama_gpu_models: ["llama3.1"],
      asr_min_free_vram_mb: 3500,
    },
  });

  await expect(page.getByRole("banner").getByRole("button", { name: "Freeing GPU..." })).toBeDisabled();
  await expect(page.locator("#transcript")).not.toContainText("Freeing GPU for ASR");

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: {
      status: "loading_asr",
      memory_free_mb: 4200,
      memory_total_mb: 24576,
      gpu_pressure: "ok",
      ollama_gpu_models: [],
      asr_min_free_vram_mb: 3500,
    },
  });

  await expect(page.getByRole("banner").getByRole("button", { name: "Loading ASR..." })).toBeDisabled();
  await expect(page.locator("#transcript")).not.toContainText("Loading ASR");

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: { status: "listening", queue_depth: 0, dropped_windows: 0 },
  });
  await expect(page.getByRole("banner").getByRole("button", { name: "Listening" })).toBeDisabled();
});

test("limits context cards and reveals raw metadata only in debug mode", async ({ page }) => {
  await page.getByRole("button", { name: "Start Listening" }).click();
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "debug-line-1",
      text: "Apollo rollout needs platform readiness context.",
      start_s: 0,
      end_s: 2,
    },
  });

  for (let index = 1; index <= 4; index += 1) {
    await emitSessionEvent(page, {
      type: "recall_hit",
      payload: {
        source_type: "session",
        source_id: `debug-source-${index}`,
        text: [
          "Prior session contains rollout readiness context for platform launch.",
          "Architecture notes describe capacity review before Apollo deployment.",
          "Release notes mention stakeholder signoff for service migration.",
          "Planning notes capture owner mapping for support escalation.",
        ][index - 1],
        score: 0.9 - index * 0.01,
        metadata: { title: `Readiness memory ${index}`, reason: `distinct readiness overlap ${index}` },
        source_segment_ids: ["debug-line-1"],
      },
    });
  }

  const debugRow = page.getByRole("article", { name: "Transcript and sidecar row" }).filter({ hasText: "Apollo rollout needs platform readiness context." });
  await expect(debugRow.locator(".context-card")).toHaveCount(0);
  const contextSection = page.getByLabel("Context cards");
  await contextSection.getByText("Context / past transcript / web").click();
  await expect(contextSection.locator(".context-card")).toHaveCount(3);
  await expect(page.locator("#transcript")).not.toContainText("debug-source-1");
  await expect(page.locator("#transcript")).not.toContainText("0.89");

  await page.getByRole("button", { name: "Tools" }).click();
  await openDrawerRegion(page, "System");
  await page.getByLabel("Show debug metadata").check();
  await page.getByRole("button", { name: "Tools" }).click();
  await contextSection.locator(".context-card").first().getByRole("button", { name: "Details" }).click();
  await expect(page.locator("#recall")).toContainText("source ID");
  await expect(page.locator("#recall")).toContainText(/debug-source-/);
  await expect(page.locator("#recall")).toContainText("score");
});

test("surfaces mocked SSE errors and supports stop", async ({ page }) => {
  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await emitSessionEvent(page, {
    type: "error",
    payload: { message: "Microphone disconnected" },
  });

  await expect(page.getByRole("alert")).toContainText("Microphone disconnected");
  await expect(page.getByLabel("Runtime status")).toContainText("error");

  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("stopped");
});

test("shows copyable post-call brief from evidence-backed cards", async ({ page }) => {
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as unknown as { __copiedBrief?: string }).__copiedBrief = text;
        },
      },
    });
  });
  await page.getByRole("button", { name: "Start Listening" }).click();

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "brief-line-1",
      text: "Send the Siemens comments by Monday.",
      start_s: 0,
      end_s: 3,
    },
  });
  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "brief-card-1",
      session_id: "session-123",
      category: "action",
      title: "Send by Monday",
      body: "Send the Siemens comments by Monday.",
      why_now: "The meeting set a Monday target.",
      priority: "high",
      confidence: 0.93,
      source_segment_ids: ["brief-line-1"],
      source_type: "transcript",
      evidence_quote: "Send the Siemens comments by Monday.",
      due_date: "Monday",
      card_key: "action:send-by-monday",
    },
  });
  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "brief-memory-1",
      session_id: "session-123",
      category: "work_memory",
      title: "Relay distractor",
      body: "Past CT/PT relay settings work.",
      why_now: "Injected memory distractor.",
      priority: "normal",
      confidence: 0.76,
      source_segment_ids: [],
      source_type: "work_memory_project",
      citations: ["eval-memory:relay"],
      card_key: "memory:relay",
    },
  });

  await page.getByRole("button", { name: "Stop" }).click();
  const brief = page.getByLabel("Consulting brief");
  await expect(brief).toBeVisible();
  await expect(brief).toContainText("## Meeting Goal");
  await expect(brief).toContainText("## Contract Reminders");
  await expect(brief).toContainText("## Suggested Follow-up Language");
  await expect(brief).toContainText("## Evidence Index");
  await expect(brief).toContainText("Evidence: \"Send the Siemens comments by Monday.\"");
  await expect(brief).toContainText("## Relevant Work Memory");
  await brief.getByRole("button", { name: "Copy Markdown" }).click();
  await expect.poll(() => page.evaluate(() => (window as unknown as { __copiedBrief?: string }).__copiedBrief)).toContain(
    "Source segments: brief-line-1",
  );
});

test("renders ephemeral web context notes in the context pane", async ({ page }) => {
  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "web-line-1",
      text: "What are current best practices for vector database indexing?",
      start_s: 0,
      end_s: 3.5,
    },
  });
  await emitSessionEvent(page, {
    type: "note_update",
    payload: {
      id: "web-note-1",
      kind: "topic",
      title: "Web context: vector database indexing",
      body: "I found a few current references: Brave Search API docs describe request parameters.",
      source_segment_ids: ["web-line-1"],
      ephemeral: true,
      source_type: "brave_web",
      sources: [
        { title: "Brave Search API docs", url: "https://api-dashboard.search.brave.com/app/documentation/web-search/get-started" },
      ],
    },
  });

  const field = page.locator("#recall");
  await field.getByLabel("Context cards").getByText("Context / past transcript / web").click();
  const webItem = field.getByLabel("Web context card").filter({ hasText: "vector database indexing" });
  await expect(webItem).toBeVisible();
  await expect(page.getByLabel("Mic transcript item")).toContainText("current best practices");
  await expect(webItem).not.toContainText("live only");
  await webItem.getByRole("button", { name: "Details" }).click();
  await expect(webItem.getByRole("link", { name: "Brave Search API docs" })).toHaveAttribute(
    "href",
    "https://api-dashboard.search.brave.com/app/documentation/web-search/get-started",
  );
});

test("prepares recorded audio and starts playback from the GUI", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { testModeEnabled: true });
  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();

  const testPanel = await openDrawerRegion(page, "Recorded audio test");
  await expect(testPanel).toBeVisible();

  await testPanel.getByLabel("Browse source audio").setInputFiles({
    name: "current-role-recording.m4a",
    mimeType: "audio/mp4",
    buffer: Buffer.from("fake audio"),
  });
  await expect(testPanel.getByLabel("Source audio path")).toHaveValue(
    "/tmp/brain-sidecar-tests/input-files/uploaded-input.dat",
  );
  await testPanel.getByLabel("Max seconds").fill("12");
  await testPanel.getByLabel("Expected terms").fill("Online Generator Monitoring, T.A. Smith");
  await testPanel.getByRole("button", { name: "Prepare Audio" }).click();

  await expect(testPanel.getByLabel("Prepared audio")).toContainText("12.5s");
  await expect(testPanel.getByLabel("Prepared audio")).toContainText("2");

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await testPanel.getByRole("button", { name: "Start Playback" }).click();
  const request = await startRequest;
  expect(JSON.parse(request.postData() ?? "{}")).toMatchObject({
    device_id: null,
    fixture_wav: "/tmp/brain-sidecar-tests/testrun-123/input.wav",
  });

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: { status: "listening", queue_depth: 0, dropped_windows: 0 },
  });
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "test-line-1",
      text: "Online Generator Monitoring at T.A. Smith needs validation.",
      start_s: 0,
      end_s: 4,
      queue_depth: 0,
    },
  });
  await expect(page.getByText("1 heard").first()).toBeVisible();

  const reportRequest = page.waitForRequest((candidate) => (
    candidate.method() === "POST" && candidate.url().endsWith("/api/test-mode/runs/testrun-123/report")
  ));
  await page.getByRole("button", { name: "Stop" }).click();
  const report = JSON.parse((await reportRequest).postData() ?? "{}").report;

  expect(report).toMatchObject({
    run_id: "testrun-123",
    transcript_segments: 1,
    expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
    matched_expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
  });
  await expect(testPanel.getByLabel("Recorded audio report")).toContainText("1");
  await expect(testPanel.getByLabel("Recorded audio report")).toContainText("2/2");
  await expect(testPanel.getByLabel("Recorded audio report")).toContainText("report.json");
});

test("locks live capture to the auto-selected server microphone", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByLabel("Audio source")).toHaveCount(0);
  await expect(drawer.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone");
  await expect(drawer.getByLabel("Browser microphone status")).toHaveCount(0);

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await drawer.getByRole("button", { name: "Start Listening" }).click();
  const request = await startRequest;

  expect(JSON.parse(request.postData() ?? "{}")).toMatchObject({
    device_id: null,
    fixture_wav: null,
    audio_source: "server_device",
    mic_tuning: {
      auto_level: true,
      input_gain_db: 0,
      speech_sensitivity: "normal",
    },
  });
});

test("blocks capture and speaker training when no healthy server microphone is available", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    devicesResponse: {
      devices: [{ ...mockDevices[1], healthy: false, selection_reason: "probe failed" }],
      selected_device: null,
      server_mic_available: false,
      selection_reason: "No healthy server microphone detected.",
    },
  });
  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();

  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByLabel("USB microphone")).toHaveCount(0);
  await expect(drawer.getByLabel("Server microphone")).toContainText("No healthy server microphone detected");
  await expect(drawer.getByRole("status")).toContainText("No healthy server microphone detected");
  await expect(drawer.getByRole("button", { name: "Test Mic" })).toBeDisabled();
  await expect(drawer.getByRole("button", { name: "Start Listening" })).toBeDisabled();
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Listening" })).toBeDisabled();
  const speaker = await openDrawerRegion(page, "Speaker identity");
  await expect(speaker.getByRole("button", { name: "Set Up BP Label" }).first()).toBeDisabled();
});

test("uses mocked library and recall APIs", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  const indexes = await openDrawerRegion(page, "Indexes");
  await indexes.getByLabel("Browse index file").setInputFiles({
    name: "project-history.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("# Project history"),
  });
  await expect(indexes.getByLabel("File root")).toHaveValue("/tmp/brain-sidecar-tests/input-files/uploaded-input.dat");
  await indexes.getByRole("button", { name: "Add Files" }).click();
  await expect(indexes.getByLabel("File root")).toHaveValue("");

  await indexes.getByRole("button", { name: "Reindex Files" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("indexed 42 chunks");

  await page.getByRole("button", { name: "Tools" }).click();
  await page.getByPlaceholder("Ask Sidecar anything").fill("apollo rollout");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Prior transcript");
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Work memory");
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Web");
  await expect(page.locator("#transcript").getByText("Prior planning note about the Apollo rollout.")).toBeVisible();
  await expect(page.locator("#transcript").getByText("Online Generator Monitoring - T.A. Smith", { exact: true })).toBeVisible();
  await expect(page.locator("#transcript").getByText("Web context: apollo rollout")).toBeVisible();
  await expect(page.locator("#transcript")).toContainText("failure modes");
  await expect(page.getByLabel("You transcript item")).toContainText("apollo rollout");
  await expect(page.locator("#transcript")).toContainText("Prior planning note about the Apollo rollout.");
  await expect(page.locator("#transcript")).not.toContainText("notes.md");
  await expect(page.locator("#transcript")).not.toContainText("score 0.93");

  await page.getByRole("button", { name: "Tools" }).click();
  const reopenedIndexes = await openDrawerRegion(page, "Indexes");
  await reopenedIndexes.getByRole("button", { name: "Index Work" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("work memory indexed 21 projects");
});

test("keeps transcript and context readable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "mobile-line-1",
      text: "The mobile layout should keep transcript and sidecar notes readable.",
      start_s: 0,
      end_s: 2,
    },
  });
  await emitSessionEvent(page, {
    type: "note_update",
    payload: {
      id: "mobile-note-1",
      kind: "context",
      title: "Readable sidecar",
      body: "Keep the backend bubble close to the transcript without overlap.",
      source_segment_ids: ["mobile-line-1"],
    },
  });

  await expect(page.locator("#transcript")).toContainText("mobile layout");
  await expect(page.locator("#recall").getByRole("heading", { name: "Readable sidecar" })).toBeVisible();
  const transcriptBox = await page.locator("#transcript").boundingBox();
  const backendBox = await page.locator("#recall").boundingBox();
  expect(transcriptBox).not.toBeNull();
  expect(backendBox).not.toBeNull();
  expect(backendBox!.y).toBeGreaterThan(transcriptBox!.y);
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("persists theme selection across reloads", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  let system = await openDrawerRegion(page, "System");
  await system.getByLabel("Theme").selectOption("canopy");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "canopy");

  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();
  system = await openDrawerRegion(page, "System");
  await expect(system.getByLabel("Theme")).toHaveValue("canopy");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "canopy");
});

test("keeps core capture controls usable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });

  await expect(page.getByRole("heading", { name: "Live field" })).toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  await expect(page.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone");
  await expect(page.getByLabel("Guided mic tuning")).toBeVisible();
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Listening" })).toBeVisible();
});

async function openDrawerRegion(page: Page, name: string) {
  const region = page.getByRole("region", { name });
  await expect(region).toBeVisible();
  const isOpen = await region.evaluate((element) => (
    element instanceof HTMLDetailsElement ? element.open : true
  ));
  if (!isOpen) {
    await region.locator("summary").first().click();
  }
  return region;
}
