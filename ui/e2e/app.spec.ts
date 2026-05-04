import { expect, test } from "@playwright/test";
import { emitSessionEvent, installMockEventSource, mockApi, mockDevices } from "./mocks";

test.beforeEach(async ({ page }) => {
  await installMockEventSource(page);
  await mockApi(page);
  await page.goto("/");
});

test("loads mocked device and GPU state", async ({ page }) => {
  await expect(page.getByRole("heading", { name: "Live transcript" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Work Notes" })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "midnight");
  await expect(page.getByRole("search")).toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  await expect(page.getByLabel("Audio source")).toHaveValue("server_device");
  await expect(page.getByLabel("Audio source")).not.toContainText("browser");
  await expect(page.getByLabel("USB microphone")).toHaveValue("usb-blue");
  await expect(page.getByLabel("USB microphone").locator("option:checked")).toHaveText("Blue Yeti USB Microphone");
  await expect(page.getByText("NVIDIA RTX 4090 · 6144/24576 MB")).toBeVisible();
  await expect(page.getByLabel("VRAM usage 25%")).toBeVisible();
  await expect(page.getByText("CUDA ready")).toBeVisible();
  await expect(page.getByLabel("Tools drawer").getByText("21 projects / 412 sources")).toBeVisible();
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
  expect(JSON.parse(request.postData() ?? "{}").save_transcript).toBe(false);
  await expect(page.getByLabel("Session save mode status")).toContainText("Listening");
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
      text: "We need to follow up with the platform team",
      start_s: 1.2,
      end_s: 3.0,
      is_final: false,
      asr_model: "whisper-large-v3",
      queue_depth: 1,
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  await expect(page.getByLabel("Mic transcript item").filter({ hasText: "platform team" })).toContainText("Interim");
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
	    type: "note_update",
	    payload: {
	      id: "note-1",
	      kind: "action",
	      title: "Platform follow-up",
	      body: "Schedule a short sync about deployment readiness.",
	      source_segment_ids: ["line-1"],
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
	      source_segment_ids: ["line-1"],
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
      source_segment_ids: ["line-1"],
      source_type: "transcript",
      card_key: "contribution:release-owner",
      ephemeral: true,
      raw_audio_retained: false,
    },
  });

  await expect(page.getByLabel("Runtime status")).toContainText("listening");
  await expect(page.getByLabel("Transcript metric")).toContainText("1");
  await expect(page.getByLabel("Latest heard")).toContainText("platform team tomorrow");
  await expect(page.locator("#transcript")).not.toContainText("Interim");
  const transcriptItem = page.getByLabel("Mic transcript item").filter({ hasText: "platform team tomorrow" });
  await expect(transcriptItem).toBeVisible();
  await expect(transcriptItem).not.toContainText("You");
  await expect(page.locator("#recall").getByRole("heading", { name: "Platform follow-up" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Actions work notes" })).toContainText("Platform follow-up");
  await expect(page.getByRole("region", { name: "Decisions work notes" })).toContainText("No decisions yet.");
  await expect(page.getByRole("region", { name: "Questions work notes" })).toContainText("No open questions yet.");
  await expect(page.getByRole("region", { name: "Relevant Context work notes" })).toContainText("previous meeting");
  await expect(page.getByRole("region", { name: "Contribution Lane" })).toContainText("Name the release owner");
  await expect(page.getByRole("region", { name: "Contribution Lane" })).toContainText("It sounds like we should name one release owner");
  await expect(page.locator("#recall").getByText("A previous meeting mentioned deployment readiness risks.")).toBeVisible();
  await expect(page.getByLabel("Queue metric")).toContainText("2");
  await expect(page.locator("#transcript")).not.toContainText("whisper-large-v3");
  await expect(page.locator("#recall")).not.toContainText("score 0.91");
  await expect(page.locator("#recall")).not.toContainText("previous-session");
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
  const speaker = page.getByRole("region", { name: "Speaker identity" });
  await expect(speaker).toBeVisible();
  await expect(speaker).toContainText("not enrolled");
  await expect(speaker).toContainText("0.0s usable");
  await expect(speaker).toContainText("15.0s needed");
  await expect(speaker.getByLabel("Speaker training steps")).toContainText("1 Start");
  await expect(page.getByRole("region", { name: "Voice profile" })).toHaveCount(0);
  await expect(page.getByText("ASR guidance preview")).toHaveCount(0);
  const micRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/microphone/test")
  ));
  await page.getByRole("button", { name: "Test Mic" }).click();
  expect(JSON.parse((await micRequest).postData() ?? "{}")).toMatchObject({
    device_id: "usb-blue",
    audio_source: "server_device",
    fixture_wav: null,
    seconds: 3,
  });
  await expect(page.getByLabel("Microphone check")).toContainText("Mic check looks good");
  await expect(page.getByLabel("Microphone check")).toContainText("2.6s speech");
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
    device_id: "usb-blue",
    audio_source: "server_device",
    fixture_wav: null,
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

  const lane = page.getByRole("region", { name: "Contribution Lane" });
  const card = lane.getByLabel("Ask this context card").filter({ hasText: "Clarify rollback timing" });
  await expect(card).toBeVisible();
  await card.getByRole("button", { name: "Copy" }).click();
  await expect.poll(() => page.evaluate(() => (window as unknown as { __copiedSuggestion?: string }).__copiedSuggestion)).toBe(
    "What is the rollback cutoff if validation fails?",
  );

  await card.getByRole("button", { name: "Pin card" }).click();
  await expect(card.getByRole("button", { name: "Unpin card" })).toBeVisible();
  await card.getByRole("button", { name: "Dismiss card" }).click();
  await expect(lane).toHaveCount(0);
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

  await expect(page.locator("#recall .context-card")).toHaveCount(3);
  await expect(page.locator("#recall")).not.toContainText("debug-source-1");
  await expect(page.locator("#recall")).not.toContainText("0.89");

  await page.getByRole("button", { name: "Tools" }).click();
  await page.getByLabel("Show debug metadata").check();
  await page.getByRole("button", { name: "Close" }).click();
  await page.locator("#recall .context-card").first().getByRole("button", { name: "Show details" }).click();
  await expect(page.locator("#recall")).toContainText("source ID");
  await expect(page.locator("#recall")).toContainText("debug-source-1");
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
  const webItem = field.getByLabel("Web context card").filter({ hasText: "vector database indexing" });
  await expect(webItem).toBeVisible();
  await expect(page.getByLabel("Mic transcript item")).toContainText("current best practices");
  await expect(webItem).not.toContainText("live only");
  await webItem.getByRole("button", { name: "Show details" }).click();
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

  const testPanel = page.getByRole("region", { name: "Recorded audio test" });
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
  await expect(page.getByLabel("Transcript metric")).toContainText("1");

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

test("locks live capture to the local USB microphone", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByLabel("Audio source")).toHaveValue("server_device");
  await expect(drawer.getByLabel("Audio source").locator("option")).toHaveText(["Server USB microphone"]);
  await expect(drawer.getByLabel("Browser microphone status")).toHaveCount(0);

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await drawer.getByRole("button", { name: "Start Listening" }).click();
  const request = await startRequest;

  expect(JSON.parse(request.postData() ?? "{}")).toMatchObject({
    device_id: "usb-blue",
    fixture_wav: null,
    audio_source: "server_device",
  });
});

test("blocks capture and speaker training when the preferred USB microphone is missing", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    devicesResponse: {
      devices: [{ ...mockDevices[1], preferred: false }],
      preferred_device_configured: true,
      preferred_device_available: false,
      preferred_device_match: "291a:3369",
      preferred_device_label: "Anker PowerConf C200 microphone",
    },
  });
  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();

  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByLabel("USB microphone")).toHaveValue("");
  await expect(drawer.getByRole("status")).toContainText("Anker PowerConf C200 microphone is not connected");
  await expect(drawer.getByRole("button", { name: "Test Mic" })).toBeDisabled();
  await expect(drawer.getByRole("button", { name: "Start Listening" })).toBeDisabled();
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Listening" })).toBeDisabled();
  await expect(drawer.getByRole("region", { name: "Speaker identity" }).getByRole("button", { name: "Set Up BP Label" }).first()).toBeDisabled();
});

test("uses mocked library and recall APIs", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  await page.getByLabel("Browse index file").setInputFiles({
    name: "project-history.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("# Project history"),
  });
  await expect(page.getByLabel("File root")).toHaveValue("/tmp/brain-sidecar-tests/input-files/uploaded-input.dat");
  await page.getByRole("button", { name: "Add Files" }).click();
  await expect(page.getByLabel("File root")).toHaveValue("");

  await page.getByRole("button", { name: "Reindex Files" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("indexed 42 chunks");

  await page.getByRole("button", { name: "Close" }).click();
  await page.getByPlaceholder("Ask Sidecar anything").fill("apollo rollout");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Prior transcript");
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("PAS / past work");
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Current public web");
  await page.getByRole("button", { name: /Show more/ }).click();
  await expect(page.locator("#recall").getByText("Prior planning note about the Apollo rollout.")).toBeVisible();
  await expect(page.locator("#recall").getByText("Online Generator Monitoring - T.A. Smith", { exact: true })).toBeVisible();
  await expect(page.locator("#recall").getByText("Web context: apollo rollout")).toBeVisible();
  await expect(page.locator("#recall")).toContainText("failure modes");
  await expect(page.getByLabel("You transcript item")).toContainText("apollo rollout");
  await expect(page.locator("#recall")).toContainText("Prior planning note about the Apollo rollout.");
  await expect(page.locator("#recall")).not.toContainText("notes.md");
  await expect(page.locator("#recall")).not.toContainText("score 0.93");

  await page.getByRole("button", { name: "Tools" }).click();
  await page.getByRole("button", { name: "Index Work" }).click();
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
  await page.getByLabel("Theme").selectOption("canopy");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "canopy");

  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();
  await expect(page.getByLabel("Theme")).toHaveValue("canopy");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "canopy");
});

test("keeps core capture controls usable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });

  await expect(page.getByRole("heading", { name: "Live transcript" })).toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  await page.getByLabel("USB microphone").selectOption("desk-array");
  await expect(page.getByLabel("USB microphone")).toHaveValue("desk-array");
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Listening" })).toBeVisible();
});
