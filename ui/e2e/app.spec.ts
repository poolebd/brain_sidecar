import { expect, test, type Page } from "@playwright/test";
import { emitSessionEvent, installMockEventSource, mockApi, mockDevices } from "./mocks";

test.beforeEach(async ({ page }) => {
  await installMockEventSource(page);
  await mockApi(page);
  await page.goto("/");
});

test("loads mocked device and GPU state", async ({ page }) => {
  await expect(page.getByRole("heading", { name: "Conversation" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Sidecar Intelligence" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Meeting Output" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Sidecar Intelligence" })).not.toContainText("Silent by design");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "midnight");
  await expect(page.getByLabel("Meeting Focus summary")).toContainText("Focus");
  await expect(page.getByLabel("Meeting Focus goal")).toHaveCount(0);
  await expect(page.getByLabel("Live field empty state")).toContainText("Ready. Start listening or ask Sidecar.");
  await expect(page.getByLabel("Energy lens")).toHaveCount(0);
  await expect(page.getByRole("banner").getByRole("button", { name: "Stop" })).toHaveCount(0);
  const liveBox = await page.locator("#transcript").boundingBox();
  expect(liveBox).not.toBeNull();
  expect(liveBox!.y).toBeLessThan(410);
  await expect(page.getByRole("search")).toBeVisible();
  const genericToolNoun = new RegExp("Utilit" + "(y|ies)");
  await expect(page.locator("body")).not.toContainText(genericToolNoun);
  await expect(page.locator("body")).not.toContainText(/Meeting utilities|Transcript Retention|Indexes|Browse index file/);
  const toolsButton = page.getByRole("button", { name: "Tools" });
  await toolsButton.click();
  await expect(page.getByRole("main", { name: "Tools" })).toBeVisible();
  await expect(page.getByRole("main", { name: "Tools" }).getByRole("heading", { name: "Tools" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Tool sections" })).toContainText("Voice & Input");
  await expect(page.getByLabel("Audio source")).toHaveCount(0);
  await expect(page.getByLabel("USB microphone")).toHaveCount(0);
  await expect(page.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone");
  await expect(page.getByLabel("Input readiness")).toContainText("Ready");
  await expect(page.getByLabel("Guided mic tuning")).toContainText("Auto Level");
  await expect(page.getByRole("main", { name: "Tools" }).getByRole("button", { name: "Close" })).toHaveCount(0);
  const debug = await openToolSection(page, "Logs & Debug");
  await expect(debug.getByText("NVIDIA RTX 4090 · 6144/24576 MB")).toBeVisible();
  await expect(debug.getByLabel("VRAM usage 25%")).toBeVisible();
  await expect(debug.getByText("CUDA ready")).toBeVisible();
  await expect(debug.locator("details.debug-diagnostics")).toHaveJSProperty("open", false);
  await expect(page.getByRole("main", { name: "Tools" })).not.toContainText("Transcript Retention");
  await expect(page.getByRole("main", { name: "Tools" })).not.toContainText("Indexes");
  await page.getByRole("button", { name: "Live", exact: true }).click();
  await expect(page.getByRole("main", { name: "Tools" })).not.toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  await expect(page.getByRole("main", { name: "Tools" })).toBeVisible();

  await page.getByLabel("Open model details").click();
  await expect(page.getByRole("main", { name: "Models" })).toBeVisible();
  await page.getByLabel("Open capture details").click();
  await expect(page.getByRole("main", { name: "Live" })).toBeVisible();
});

test("navigates the app shell and browses saved sessions", async ({ page }) => {
  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  for (const label of ["Live", "Review", "Sessions", "Tools", "Models", "References"]) {
    await expect(nav.getByRole("button", { name: label })).toBeVisible();
  }

  await nav.getByRole("button", { name: "Sessions" }).click();
  await expect(nav.getByRole("button", { name: "Sessions" })).toHaveAttribute("aria-current", "page");
  const sessionsPage = page.getByRole("main", { name: "Sessions" });
  await expect(sessionsPage).toBeVisible();
  await sessionsPage.getByRole("button", { name: /Saved model planning call/ }).click();
  await expect(sessionsPage.getByLabel("Session detail")).toContainText("Faster-Whisper on CPU");
  await expect(sessionsPage.getByLabel("Session detail")).toContainText("CPU ASR default");

  const patchTitle = page.waitForRequest((request) => (
    request.method() === "PATCH" && request.url().endsWith("/api/sessions/session-saved-1")
  ));
  await sessionsPage.getByLabel("Session title").fill("Renamed model planning call");
  await sessionsPage.getByRole("button", { name: "Save Title" }).click();
  expect(JSON.parse((await patchTitle).postData() ?? "{}")).toMatchObject({
    title: "Renamed model planning call",
  });

  await nav.getByRole("button", { name: "Live", exact: true }).click();
  await expect(page.getByRole("main", { name: "Live" })).toBeVisible();
  await expect(page.getByText("Open", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save Title" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "New Session" })).toHaveCount(0);
  await expect(page.getByLabel("Live run")).toContainText("Ephemeral");
  await expect(page.getByLabel("Live retention", { exact: true })).toContainText("Audio temporary until validation");
  await page.getByLabel("Live session title").fill("Live shell check");
  const startLive = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/start")
  ));
  await page.getByRole("button", { name: "Start Live" }).click();
  expect(JSON.parse((await startLive).postData() ?? "{}")).toMatchObject({
    title: "Live shell check",
    audio_source: "server_device",
  });
  await expect(page.getByLabel("Live retention status")).toContainText("Review handoff armed");
});

test("Review upload creates a job without saving raw results", async ({ page }) => {
  const review = await openReviewPage(page);
  await expect(review.getByLabel("Review retention")).toContainText("Manual approval required");
  await review.getByLabel("Review title").fill("Review Apollo call");

  const requestPromise = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/review/jobs")
  ));
  await review.getByLabel("Review audio upload").setInputFiles({
    name: "apollo-call.webm",
    mimeType: "audio/webm",
    buffer: Buffer.from("fake audio"),
  });
  const body = (await requestPromise).postData() ?? "";
  expect(body).not.toContain("name=\"save_result\"");

  await expect(review.getByRole("region", { name: "Review summary validation" })).toContainText("Needs validation");
  await expect(review.getByRole("group", { name: "Recorded audio controls" })).toHaveCount(0);
});

test("completed Review presents one Meeting Summary before sources", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const review = await openCompletedReview(page);

  const summaryValidation = review.getByRole("region", { name: "Review summary validation" });
  await expect(summaryValidation).toContainText("Needs validation");
  await expect(summaryValidation.getByRole("button", { name: "Approve" })).toBeVisible();
  await expect(summaryValidation.getByRole("button", { name: "Copy summary" })).toBeVisible();
  await expect(review.getByRole("region", { name: "Review status" })).toHaveCount(0);
  await expect(review.getByLabel("Review runtime details")).toHaveCount(0);
  await expect(review.getByRole("progressbar", { name: "Review overall progress" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Review progress" })).toHaveCount(0);
  await expect(review.getByText("More context")).toHaveCount(0);
  await expect(review.getByRole("button", { name: "Details" })).toHaveCount(0);
  await expect(review.getByText(/more thread|more technical/i)).toHaveCount(0);
  const summaryFirstOrder = await review.evaluate(() => {
    const toolbar = document.querySelector('[aria-label="Review summary validation"]');
    const summary = document.querySelector('[aria-label="Meeting Summary"]');
    const summaryParagraph = document.querySelector('[aria-label="Summary"]');
    const keyNotes = document.querySelector('[aria-label="Notes"]');
    const actions = document.querySelector('[aria-label="Follow-ups"]');
    const decisions = document.querySelector('[aria-label="Decisions"]');
    const questions = document.querySelector('[aria-label="Open questions"]');
    const risks = document.querySelector('[aria-label="Risks / concerns"]');
    const technical = document.querySelector('[aria-label="Technical Findings"]');
    const transcriptDetails = document.querySelector('[aria-label="Sources / transcript"]');
    const validationCards = document.querySelector('[aria-label="Validation Evidence Cards"]');
    const workspace = document.querySelector('[aria-label="Review workspace"]');
    const summaryNodes = [summary, summaryParagraph, keyNotes, decisions, actions, questions, risks];
    return {
      summaryPresent: Boolean(summary),
      validationCardsVisible: Boolean(validationCards),
      toolbarBeforeSummary: Boolean(toolbar && summary && (toolbar.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING)),
      sectionsInOrder: summaryNodes.every((node, index, nodes) => (
        Boolean(node) && (index === 0 || Boolean(nodes[index - 1] && (nodes[index - 1]!.compareDocumentPosition(node!) & Node.DOCUMENT_POSITION_FOLLOWING)))
      )),
      technicalHidden: !technical,
      transcriptClosed: transcriptDetails instanceof HTMLDetailsElement ? !transcriptDetails.open : false,
      workspaceHasColumns: workspace ? getComputedStyle(workspace).gridTemplateColumns.split(" ").filter(Boolean).length > 0 : false,
    };
  });
  expect(summaryFirstOrder).toEqual({
    summaryPresent: true,
    validationCardsVisible: false,
    toolbarBeforeSummary: true,
    sectionsInOrder: true,
    technicalHidden: true,
    transcriptClosed: true,
    workspaceHasColumns: true,
  });
  await expect(review.getByRole("heading", { name: "Meeting Summary" })).toBeVisible();
  await expect(review.getByRole("heading", { name: "Meeting Capture" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Meeting Summary" })).toContainText("Review Apollo call");
  await expect(review.getByRole("region", { name: "Summary", exact: true })).toContainText("RFI log");
  await expect(review.getByRole("region", { name: "Notes", exact: true })).not.toContainText("captured");
  await expect(review.getByRole("region", { name: "Follow-ups" })).toContainText("RFI log");
  await expect(review.getByRole("region", { name: "Decisions" })).toContainText("Sunil owns the Siemens document review path");
  await expect(review.getByRole("region", { name: "Open questions" })).toContainText("Confirm Greg's preferred handoff format");
  await expect(review.getByRole("region", { name: "Risks / concerns" })).toContainText("handoff depends");
  await expect(review.getByRole("region", { name: "Technical Findings" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Parking Lot / Low Confidence" })).toHaveCount(0);
  await expect(review.getByText("Validation Evidence")).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Evidence Quality" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Meeting Follow-up" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Project / Site Threads" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Technical Review" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Review summary validation" })).not.toContainText("...");
  await expect(review.getByLabel("Reference Context")).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Clean Transcript" })).toHaveCount(0);
  await expect(review.getByText("Transcript Evidence")).toBeHidden();
  await expect(review.getByRole("region", { name: "Validation Evidence Cards" })).toHaveCount(0);
  await review.getByLabel("Sources / transcript").locator("summary").first().click();
  await expect(review.getByRole("region", { name: "Clean Transcript" })).toContainText("BP will send the RFI log to Greg by Monday.");
  await expect(review.getByRole("region", { name: "Clean Transcript" })).toContainText("Other speaker");
});

test("Approve saves a completed Review as a Session and deletes temporary audio", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const review = await openCompletedReview(page);

  await review.getByRole("button", { name: "Approve" }).click();
  await expect(review.getByRole("region", { name: "Review summary validation" })).toContainText("Saved to Sessions");
  await expect(review.getByRole("region", { name: "Review status" })).toHaveCount(0);
  await review.getByRole("button", { name: "Open Session" }).click();
  await expect(page.getByRole("main", { name: "Sessions" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Review Apollo call" })).toBeVisible();
});

test("Discard removes validation actions and never exposes Open Session", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const review = await openCompletedReview(page);

  await review.getByRole("button", { name: "Discard" }).click();
  await expect(review.getByRole("button", { name: "Approve" })).toHaveCount(0);
  await expect(review.getByRole("button", { name: "Discard" })).toHaveCount(0);
  await expect(review.getByRole("button", { name: "Open Session" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Review status" })).toContainText("Discarded");
  await expect(review.getByRole("region", { name: "Review status" })).toContainText("Audio deleted");
  await expect(review.getByRole("button", { name: "Copy summary" })).toBeDisabled();
});

test("New audio resets the workspace and exposes upload controls", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const review = await openCompletedReview(page);

  await expect(review.getByRole("button", { name: "New audio" })).toBeVisible();
  await review.getByRole("button", { name: "New audio" }).click();
  await expect(review.getByLabel("Review audio upload")).toBeVisible();
  await expect(review.getByRole("region", { name: "No review loaded" })).toContainText("Upload audio");
  await expect(review.getByRole("region", { name: "Review summary validation" })).toHaveCount(0);
});

test("Review action buttons hold disabled states while requests are in flight", async ({ page }) => {
  const review = await openReviewPage(page);
  const uploadGate = deferred();
  const uploadUrl = "http://127.0.0.1:8765/api/review/jobs";
  await page.route(uploadUrl, async (route) => {
    await uploadGate.promise;
    await route.fallback();
  });
  const upload = review.getByLabel("Review audio upload").setInputFiles({
    name: "apollo-call.webm",
    mimeType: "audio/webm",
    buffer: Buffer.from("fake audio"),
  });
  await expect(review.getByLabel("Review title")).toBeDisabled();
  await expect(review.getByLabel("Review audio upload")).toBeDisabled();
  uploadGate.resolve();
  await upload;
  await page.unroute(uploadUrl);

  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const approveReview = await openCompletedReview(page);
  const approveGate = deferred();
  const approveUrl = "http://127.0.0.1:8765/api/review/jobs/reviewjob-123/approve";
  await page.route(approveUrl, async (route) => {
    await approveGate.promise;
    await route.fallback();
  });
  const approve = approveReview.getByRole("button", { name: "Approve" }).click();
  await expect(approveReview.getByRole("button", { name: "Approving..." })).toBeDisabled();
  approveGate.resolve();
  await approve;
  await page.unroute(approveUrl);

  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const discardReview = await openCompletedReview(page);
  const discardGate = deferred();
  const discardUrl = "http://127.0.0.1:8765/api/review/jobs/reviewjob-123/discard";
  await page.route(discardUrl, async (route) => {
    await discardGate.promise;
    await route.fallback();
  });
  const discard = discardReview.getByRole("button", { name: "Discard" }).click();
  await expect(discardReview.getByRole("button", { name: "Discarding..." })).toBeDisabled();
  discardGate.resolve();
  await discard;
  await page.unroute(discardUrl);

  await remockApi(page, { latestReviewStatus: "queued" });
  const cancelReview = await openReviewPage(page);
  const cancelGate = deferred();
  const cancelUrl = "http://127.0.0.1:8765/api/review/jobs/reviewjob-123/cancel";
  await page.route(cancelUrl, async (route) => {
    await cancelGate.promise;
    await route.fallback();
  });
  const cancel = cancelReview.getByRole("button", { name: "Cancel" }).click();
  await expect(cancelReview.getByRole("button", { name: "Canceling..." })).toBeDisabled();
  cancelGate.resolve();
  await cancel;
  await page.unroute(cancelUrl);
});

test("Review queue opens, closes, and selects another job without losing active styling", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation", reviewQueueScenario: "two_jobs" });
  const review = await openCompletedReview(page);
  const queueToggle = review.getByText("Queue (2)");

  await queueToggle.click();
  const queue = review.getByRole("region", { name: "Review queue" });
  await expect(queue).toBeVisible();
  await expect(queue).toContainText("Apollo call");
  await expect(queue).toContainText("Delta site tariff review");
  const queueBox = await queue.boundingBox();
  const viewport = page.viewportSize();
  expect(queueBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  expect(queueBox!.x).toBeGreaterThanOrEqual(0);
  expect(queueBox!.x + queueBox!.width).toBeLessThanOrEqual(viewport!.width + 1);

  await queue.getByRole("button", { name: /Delta site tariff review/ }).click();
  await expect(review.getByRole("button", { name: /Delta site tariff review/ })).toHaveClass(/active/);
  await expect(review.getByRole("region", { name: "Review status" })).toContainText("delta-site-call.webm");

  await queueToggle.click();
  await expect(review.getByRole("region", { name: "Review queue" })).toHaveCount(0);
  await queueToggle.click();
  await queue.getByRole("button", { name: /Apollo call/ }).click();
  await expect(review.getByRole("region", { name: "Review summary validation" })).toContainText("Needs validation");
});

test("Review copy, evidence, and source-jump actions work", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation", longReviewEvidence: true });
  await installClipboardStub(page);
  const review = await openCompletedReview(page);

  await review.getByRole("button", { name: "Copy summary" }).click();
  const brief = await clipboardText(page);
  expect(brief).toContain("# Meeting Summary");
  expect(brief).toContain("## Summary");
  expect(brief).toContain("## Notes");
  expect(brief).toContain("## Decisions");
  expect(brief).toContain("## Follow-ups");
  expect(brief).toContain("## Open questions");
  expect(brief).toContain("## Risks / concerns");
  expect(brief).not.toContain("## Key Notes");
  expect(brief).not.toContain("## Action Items / Follow-ups");
  expect(brief).not.toContain("## Executive Summary");
  expect(brief).not.toContain("## Reference Context");
  expect(brief).not.toContain("## Evidence Cards");
  expect(brief).not.toContain("## Technical Findings");
  expect(brief).not.toContain("## Source Notes");
  const copiedBullets = brief.split("\n").filter((line) => line.startsWith("- "));
  expect(new Set(copiedBullets).size).toBe(copiedBullets.length);
  expect(brief).toContain("RFI log");

  await expect(review.getByRole("region", { name: "Clean Transcript" })).toHaveCount(0);
  await review.getByLabel("Sources / transcript").locator("summary").first().click();
  await review.getByRole("button", { name: "Copy transcript" }).click();
  expect(await clipboardText(page)).toContain("Extended evidence row 30 keeps the Apollo review grounded");
  await review.getByLabel("Sources / transcript").locator("summary").first().click();

  await expect(review.getByRole("region", { name: "Validation Evidence Cards" })).toHaveCount(0);
  await review.getByRole("region", { name: "Follow-ups" }).getByRole("button", { name: /source/i }).first().click();
  await expect(review.getByLabel("Sources / transcript")).toHaveJSProperty("open", true);
  await expect(review.locator('[data-transcript-id="review-seg-1"]')).toHaveClass(/highlight/);
});

test("Review decision toolbar stays visible while transcript sources toggle", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation" });
  const review = await openCompletedReview(page);

  await expect(review.getByRole("region", { name: "Review status" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Review progress" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Review summary validation" })).toContainText("Needs validation");
  await expect(review.getByText("Status and audio retention")).toHaveCount(0);
  await expect(review.getByText("Processing details")).toHaveCount(0);
  await expect(review.getByText("Upload another file")).toHaveCount(0);
  await expect(review.getByRole("button", { name: "New audio" })).toBeVisible();
  await expect(review.getByLabel("Review runtime details")).toHaveCount(0);
  await expect(review.getByRole("button", { name: "Details" })).toHaveCount(0);
  await expect(review.getByRole("region", { name: "Follow-ups" }).getByRole("button", { name: /source/i }).first()).toBeVisible();

  const transcriptDetails = review.getByLabel("Sources / transcript");
  await expect(transcriptDetails).toHaveJSProperty("open", false);
  await transcriptDetails.locator("summary").first().click();
  await expect(transcriptDetails).toHaveJSProperty("open", true);
  await expect(review.getByRole("region", { name: "Clean Transcript" })).toContainText("BP will send the RFI log to Greg by Monday.");
  await expect(review.getByRole("region", { name: "Review summary validation" })).toContainText("Needs validation");
  await transcriptDetails.locator("summary").first().click();
  await expect(transcriptDetails).toHaveJSProperty("open", false);
});

test("Review source panel scrolls and item evidence highlights transcript rows", async ({ page }) => {
  await remockApi(page, { latestReviewStatus: "completed_awaiting_validation", longReviewEvidence: true });
  const review = await openCompletedReview(page);

  const transcriptDetails = review.getByLabel("Sources / transcript");
  await expect(transcriptDetails).toHaveJSProperty("open", false);
  await transcriptDetails.locator("summary").first().click();
  await expect(transcriptDetails).toHaveJSProperty("open", true);

  const scrollState = await review.evaluate(() => {
    const transcript = document.querySelector<HTMLElement>(".review-transcript-scroll");
    const pageY = window.scrollY;
    if (!transcript) {
      return { transcriptScrollable: false, transcriptMoved: false, pageStable: false };
    }
    transcript.scrollTop = transcript.scrollHeight;
    return {
      transcriptScrollable: transcript.scrollHeight > transcript.clientHeight,
      transcriptMoved: transcript.scrollTop > 0,
      pageStable: window.scrollY === pageY,
    };
  });
  expect(scrollState).toEqual({
    transcriptScrollable: true,
    transcriptMoved: true,
    pageStable: true,
  });

  const transcriptScroll = review.locator(".review-transcript-scroll");
  await transcriptScroll.evaluate((element) => { element.scrollTop = element.scrollHeight; });
  const beforeHeight = await page.evaluate(() => document.documentElement.scrollHeight);
  await review.getByRole("region", { name: "Follow-ups" }).getByRole("button", { name: /source/i }).first().click();
  await expect(review.locator('[data-transcript-id="review-seg-1"]')).toHaveClass(/highlight/);
  await expect.poll(() => transcriptScroll.evaluate((element) => element.scrollTop)).toBeLessThan(120);
  const afterHeight = await page.evaluate(() => document.documentElement.scrollHeight);
  expect(afterHeight - beforeHeight).toBeLessThan(500);
  await expect(review.getByRole("region", { name: "Meeting Summary" })).toBeVisible();

  for (const size of [
    { width: 1440, height: 900 },
    { width: 1024, height: 760 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(size);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
});

test("Review typography fits and layout flexes across dense content", async ({ page }) => {
  await remockApi(page, {
    latestReviewStatus: "completed_awaiting_validation",
    longReviewEvidence: true,
    reviewQueueScenario: "two_jobs",
    reviewTypographyStress: true,
  });
  const review = await openCompletedReview(page);

  for (const size of [
    { width: 1440, height: 900 },
    { width: 1280, height: 820 },
    { width: 1024, height: 760 },
    { width: 900, height: 760 },
    { width: 768, height: 900 },
    { width: 480, height: 844 },
    { width: 390, height: 844 },
    { width: 320, height: 844 },
  ]) {
    await page.setViewportSize(size);
    const transcriptDetails = review.getByLabel("Sources / transcript");
    const transcriptIsOpen = await transcriptDetails.evaluate((element) => (
      element instanceof HTMLDetailsElement ? element.open : false
    ));
    if (transcriptIsOpen) {
      await transcriptDetails.locator("summary").first().click();
    }
    await expect(review.getByRole("region", { name: "Review summary validation" })).toBeVisible();
    await expect(review.getByRole("region", { name: "Meeting Summary" })).toBeVisible();
    await expect(review.getByRole("region", { name: "Clean Transcript" })).toHaveCount(0);

    const queueMenu = review.getByLabel("Review queue menu");
    await queueMenu.locator("summary").first().click();
    await expect(review.getByRole("region", { name: "Review queue" })).toBeVisible();
    const queuePlacement = await review.evaluate(() => {
      const queue = document.querySelector('[aria-label="Review queue"]');
      const summary = document.querySelector('[aria-label="Review summary validation"]');
      const queueBox = queue?.getBoundingClientRect();
      const summaryBox = summary?.getBoundingClientRect();
      return Boolean(queueBox && summaryBox && queueBox.bottom <= summaryBox.top + 1);
    });
    expect(queuePlacement).toBe(true);
    await expectReviewTextFit(page);
    await expectReviewSummaryTextVisible(page);
    await expectReviewControlTargetsAndFocus(page);
    await queueMenu.locator("summary").first().click();

    await expectReviewSummaryHierarchy(page);
    await expect(review.getByRole("region", { name: "Review summary validation" })).not.toContainText("...");

    await transcriptDetails.locator("summary").first().click();
    await expect(review.getByRole("region", { name: "Clean Transcript" })).toBeVisible();
    const independentScroll = await review.evaluate(() => {
      const transcript = document.querySelector<HTMLElement>(".review-transcript-scroll");
      if (!transcript) {
        return { transcriptScrollable: false };
      }
      return {
        transcriptScrollable: transcript.scrollHeight > transcript.clientHeight,
      };
    });
    expect(independentScroll).toEqual({
      transcriptScrollable: true,
    });

    const heightBeforeInlineEvidence = await page.evaluate(() => document.documentElement.scrollHeight);
    await expect(review.getByRole("button", { name: "Details" })).toHaveCount(0);
    await expect(review.getByRole("region", { name: "Follow-ups" }).getByRole("button", { name: /source/i }).first()).toBeVisible();
    await expectReviewTextFit(page);
    await expectReviewSummaryTextVisible(page);
    await expectReviewControlTargetsAndFocus(page);
    const heightAfterInlineEvidence = await page.evaluate(() => document.documentElement.scrollHeight);
    expect(heightAfterInlineEvidence - heightBeforeInlineEvidence).toBeLessThan(200);
  }

  await page.addStyleTag({
    content: `
      .review-page p,
      .review-page li,
      .review-page span,
      .review-page strong,
      .review-page small,
      .review-page button,
      .review-page h1,
      .review-page h2,
      .review-page h3 {
        letter-spacing: 0.12em !important;
        line-height: 1.5 !important;
        word-spacing: 0.16em !important;
      }

      .review-page p {
        margin-bottom: 2em !important;
      }
    `,
  });

  for (const size of [
    { width: 1024, height: 760 },
    { width: 480, height: 844 },
    { width: 390, height: 844 },
    { width: 320, height: 844 },
  ]) {
    await page.setViewportSize(size);
    await expectReviewTextFit(page);
    await expectReviewTextSpacingFit(page);
    await expectReviewSummaryTextVisible(page);
  }
});

test("queued and running Review jobs show compact progress without completed-summary actions", async ({ page }) => {
  for (const status of ["queued", "running_asr"]) {
    await remockApi(page, { latestReviewStatus: status });
    const review = await openReviewPage(page);
    await expect(review.getByRole("region", { name: "Review status" })).toBeVisible();
    await expect(review.getByRole("region", { name: "Review summary validation" })).toHaveCount(0);
    await expect(review.getByRole("button", { name: "Approve" })).toHaveCount(0);
    await expect(review.getByRole("button", { name: "Copy summary" })).toHaveCount(0);
    await expect(review.getByRole("region", { name: "Review progress" })).toBeVisible();
    await expect(review.getByRole("region", { name: "Clean Transcript" })).toHaveCount(0);
  }
});

test("review status calls out low-usefulness summaries before approval", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    latestReviewStatus: "completed_awaiting_validation",
    reviewUsefulnessStatus: "low_usefulness",
    reviewUsefulnessFlags: ["generic_topic_soup", "unsupported_technical_findings"],
  });
  await page.goto("/");

  await page.getByRole("button", { name: "Review" }).click();
  const review = page.getByRole("main", { name: "Review" });
  await review.getByText("Queue (1)").click();
  await review.getByRole("button", { name: /Apollo/ }).click();

  const decisionToolbar = review.getByRole("region", { name: "Review summary validation" });
  await expect(decisionToolbar).toContainText("Needs validation");
  await expect(decisionToolbar).toContainText("2 usefulness flags");
  await expect(review.getByRole("region", { name: "Review status" })).toHaveCount(0);
  await expect(review.getByRole("button", { name: "Approve" })).toBeVisible();
});

test("review error and cleanup-failed states do not create a Session action", async ({ page }) => {
  await remockApi(page, {
    latestReviewStatus: "error",
    reviewJobError: "Review ASR failed before validation.",
    reviewCleanupError: "Temporary audio cleanup failed.",
  });
  const review = await openReviewPage(page);
  await review.getByText("Queue (1)").click();
  await review.getByRole("button", { name: /Apollo call/ }).click();

  await expect(review.getByRole("alert")).toContainText("Review ASR failed before validation.");
  const status = review.getByRole("region", { name: "Review status" });
  await expect(status).toContainText("Review blocked");
  await expect(status).toContainText("Cleanup failed");
  await expect(status).toContainText("Temporary audio cleanup failed.");
  await expect(review.getByRole("button", { name: "Open Session" })).toHaveCount(0);
});

test("review page does not auto-open stale terminal jobs", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { latestReviewStatus: "discarded" });
  await page.goto("/");

  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  await nav.getByRole("button", { name: "Review" }).click();
  const review = page.getByRole("main", { name: "Review" });
  await expect(review).toBeVisible();
  await expect(review.getByRole("region", { name: "No review loaded" })).toContainText("Upload audio");
  await expect(review.getByLabel("Review audio upload")).toBeVisible();
  await review.getByText("Queue (1)").click();
  await expect(review.getByRole("region", { name: "Review queue" })).toContainText("Apollo call");
  await expect(review.getByRole("region", { name: "Review workspace" })).toHaveCount(0);
});

test("recovers the latest review job and supports cancel", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { latestReviewStatus: "queued" });
  await page.reload();
  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  await nav.getByRole("button", { name: "Review" }).click();
  const review = page.getByRole("main", { name: "Review" });

  await expect(review.getByRole("region", { name: "Review status" })).toContainText("apollo-call.webm");
  await expect(review.getByRole("region", { name: "Review status" })).toContainText("Queued");
  await review.getByRole("button", { name: "Cancel" }).click();
  await expect(review.getByRole("region", { name: "Review status" })).toContainText("Review canceled");
});

test("shows a saved-session empty state with a Review CTA", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { sessionsResponse: [] });
  await page.reload();

  await page.getByRole("button", { name: "Sessions" }).click();
  const sessionsPage = page.getByRole("main", { name: "Sessions" });
  await expect(sessionsPage).toContainText("No saved sessions yet");
  await expect(sessionsPage).toContainText("Approve a completed Review job");
  await expect(sessionsPage.locator(".split-pane")).toHaveCount(0);
  await sessionsPage.getByRole("button", { name: "Open Review" }).click();
  await expect(page.getByRole("main", { name: "Review" })).toBeVisible();
});

test("sessions hide empty temporary runs unless All Sessions is enabled", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    sessionsResponse: [
      {
        id: "temp-empty",
        title: "Empty temporary run",
        status: "stopped",
        started_at: 1_768_000_050,
        ended_at: 1_768_000_060,
        save_transcript: false,
        retention: "temporary",
        transcript_count: 0,
        note_count: 0,
        summary_exists: false,
        raw_audio_retained: false,
      },
      {
        id: "saved-valuable",
        title: "Useful saved meeting",
        status: "stopped",
        started_at: 1_768_000_070,
        ended_at: 1_768_000_120,
        save_transcript: true,
        retention: "saved",
        transcript_count: 4,
        note_count: 1,
        summary_exists: true,
        raw_audio_retained: false,
      },
    ],
  });
  await page.reload();
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Sessions" }).click();

  const sessions = page.getByRole("main", { name: "Sessions" });
  await expect(sessions.getByRole("button", { name: "Useful saved meeting" })).toBeVisible();
  await expect(sessions.getByRole("button", { name: "Empty temporary run" })).toHaveCount(0);
  await sessions.getByRole("button", { name: "All Sessions" }).click();
  await expect(sessions.getByRole("button", { name: "Empty temporary run" })).toBeVisible();
});

test("shows model residency and memory management pages", async ({ page }) => {
  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  await nav.getByRole("button", { name: "Models" }).click();
  const modelsPage = page.getByRole("main", { name: "Models" });
  await expect(modelsPage).toBeVisible();
  await expect(modelsPage.getByLabel("Dross readiness summary")).toContainText("Dross runtime ready");
  await expect(modelsPage.getByLabel("Dross readiness summary")).toContainText("Ready");
  await expect(modelsPage.getByLabel("Dross readiness summary")).toContainText("Runtime path looks healthy");
  await expect(modelsPage.getByLabel("Dross readiness summary")).not.toContainText("3/3");
  await expect(modelsPage).toContainText("Faster-Whisper");
  await expect(modelsPage).toContainText("small.en");
  await expect(modelsPage).toContainText("CPU");
  await expect(modelsPage).toContainText("qwen3.5:397b-cloud");
  await expect(modelsPage).toContainText("embeddinggemma");
  await expect(modelsPage).toContainText("Keepalive 30m");
  await expect(modelsPage).toContainText("Keepalive not resident (0)");
  await expect(modelsPage).toContainText("GPU Residency");
  const modelSelector = modelsPage.getByLabel("Ollama chat model");
  await expect(modelSelector).toHaveValue("qwen3.5:397b-cloud");
  await modelSelector.selectOption("gpt-oss:120b-cloud");
  await expect(modelSelector).toHaveValue("gpt-oss:120b-cloud");
  await expect(modelsPage).toContainText("Saved gpt-oss:120b-cloud to .env");
  await expect(modelsPage.getByLabel("Dross readiness summary")).toContainText("gpt-oss:120b-cloud");
  await expect(modelsPage.getByLabel("Config recipes")).toHaveJSProperty("open", false);

  await nav.getByRole("button", { name: "References" }).click();
  const memoryPage = page.getByRole("main", { name: "References" });
  await expect(memoryPage).toBeVisible();
  const memoryCategories = page.getByRole("navigation", { name: "Reference category navigation" });
  for (const label of ["Overview", "Library Roots", "Documents"]) {
    await expect(memoryCategories.getByRole("button", { name: new RegExp(label) })).toBeVisible();
  }
  await expect(memoryCategories.getByRole("button", { name: /Search Results/ })).toHaveCount(0);
  await expect(memoryPage.getByLabel("Reference search behavior")).toContainText("Search loads matching EE reference chunks");
  await expect(memoryPage.getByLabel("Browse index file")).toHaveCount(0);
  await expect(memoryPage.getByRole("button", { name: "Add Root" })).toHaveCount(0);
  await memoryCategories.getByRole("button", { name: /Documents/ }).click();
  await expect(memoryPage.getByLabel("Reference items")).toContainText("DOE Electrical Science Volume 1");
  await expect(memoryPage.getByRole("button", { name: "Add Root" })).toHaveCount(0);
  await memoryPage.getByRole("button", { name: /DOE Electrical Science Volume 1/ }).click();
  await expect(memoryPage.getByLabel("Reference detail")).toContainText("breaker coordination");
  await expect(memoryPage.getByLabel("Reference detail")).toContainText("runtime/reference/electrical-engineering");
  await expect(memoryPage.locator("details.metadata-details")).toHaveCount(0);

  const searchRequest = page.waitForRequest((request) => (
    request.method() === "GET" && request.url().includes("/api/library/chunks") && request.url().includes("query=doe")
  ));
  await memoryPage.getByLabel("Search references").fill("doe");
  await memoryPage.getByRole("button", { name: "Search", exact: true }).click();
  await searchRequest;
  await expect(memoryPage.getByLabel("Reference items")).toContainText("DOE Electrical Science Volume 1");

  await memoryCategories.getByRole("button", { name: /Library Roots/ }).click();
  const addRootRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/library/roots")
  ));
  await memoryPage.getByLabel("Library root path").fill("/tmp/brain-sidecar-notes");
  await memoryPage.getByRole("button", { name: "Add Root" }).click();
  expect(JSON.parse((await addRootRequest).postData() ?? "{}")).toMatchObject({
    path: "/tmp/brain-sidecar-notes",
  });

  const reindexLibrary = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/library/reindex")
  ));
  await memoryPage.getByRole("button", { name: "Reindex References" }).click();
  await reindexLibrary;
  await expect(page.getByLabel("Runtime status")).toContainText("indexed 42 chunks");
});

test("shows Faster-Whisper ASR status compactly", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    gpuHealthResponse: {
      asr_backend: "faster_whisper",
      asr_device: "cpu",
      asr_model: "small.en",
      asr_primary_model: "small.en",
      asr_fallback_model: "tiny.en",
      asr_dtype: "int8",
    },
  });
  await page.reload();
  await page.getByRole("button", { name: "Models" }).click();

  const models = page.getByRole("main", { name: "Models" });
  await expect(models).toContainText("Faster-Whisper");
  await expect(models).toContainText("Preview 2.0s");
  await expect(models).toContainText("ready");
  await expect(models).toContainText("qwen3.5:397b-cloud");
  await expect(models).toContainText("embeddinggemma");
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

  const drawer = page.getByRole("main", { name: "Tools" });
  await expect(drawer.getByLabel("Input Boost")).toHaveValue("12");
  await expect(drawer.getByLabel("Guided mic tuning")).toContainText("+12 dB");
  await expect(drawer.getByLabel("Guided mic tuning")).toContainText("High boost can clip");
});

test("restores Meeting Focus controls from localStorage", async ({ page }) => {
  await page.evaluate(() => {
    window.localStorage.setItem("brain-sidecar-meeting-focus-v1", JSON.stringify({
      goal: "Track owners for the outage review.",
      mode: "balanced",
      webContextMode: "off",
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
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Brave off");
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveCount(0);
  await focus.getByRole("button", { name: "Edit" }).click();
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveValue("Track owners for the outage review.");
  await expect(focus.getByRole("button", { name: "Balanced" })).toHaveAttribute("aria-pressed", "true");
  const braveControl = focus.getByRole("group", { name: "Brave web context" });
  await expect(braveControl.getByRole("button", { name: "Off", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(focus.getByLabel("Open questions")).not.toBeChecked();
  await expect(focus.getByLabel("Risks/dependencies")).toBeChecked();
  await focus.getByLabel("Meeting Focus goal").fill("Track owners and risk decisions.");
  await focus.getByRole("button", { name: "Assertive" }).click();
  await braveControl.getByRole("button", { name: "On", exact: true }).click();
  await focus.getByLabel("Open questions").check();
  await focus.locator(".meeting-focus-editor").getByRole("button", { name: "Done" }).click();
  await expect(focus.getByLabel("Meeting Focus goal")).toHaveCount(0);
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Assertive");
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Brave on");
  await expect(focus.getByLabel("Meeting Focus summary")).toContainText("Track owners and risk decisions.");

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/start")
  ));
  await page.getByRole("button", { name: "Start Live" }).click();
  const body = JSON.parse((await startRequest).postData() ?? "{}");
  expect(body.meeting_contract).toMatchObject({
    goal: "Track owners and risk decisions.",
    mode: "assertive",
  });
  expect(body.web_context_enabled).toBe(true);
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

  const drawer = page.getByRole("main", { name: "Tools" });
  await expect(drawer.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone (in use)");
  await expect(drawer.getByLabel("Server microphone")).toContainText("USB mic is in use by active capture");
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Live" })).toBeDisabled();
});

test("hides recorded audio test controls when test mode is disabled", async ({ page }) => {
  await expect(page.getByRole("main", { name: "Live" }).getByRole("button", { name: "Test" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Recorded audio test" })).toHaveCount(0);
});

test("runs recorded audio from the Live page Test button", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { testModeEnabled: true });
  await page.reload();

  await page.getByRole("main", { name: "Live" }).getByRole("button", { name: "Test" }).click();
  const testPanel = page.getByRole("region", { name: "Live audio test" });
  await expect(testPanel).toContainText("Recorded audio");
  await expect(testPanel.locator(".file-action")).toContainText("Upload and play");
  await expect(testPanel.getByLabel("Live test max seconds")).toHaveValue("60");
  await testPanel.getByLabel("Live test max seconds").fill("12");
  await testPanel.getByLabel("Live test expected terms").fill("Online Generator Monitoring, T.A. Smith");

  const prepareRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/test-mode/audio/prepare")
  ));
  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await testPanel.getByLabel("Test audio upload").setInputFiles({
    name: "current-role-recording.webm",
    mimeType: "audio/webm",
    buffer: Buffer.from("fake audio"),
  });

  const prepareBody = JSON.parse((await prepareRequest).postData() ?? "{}");
  expect(prepareBody).toMatchObject({
    source_path: "/tmp/brain-sidecar-tests/input-files/uploaded-input.dat",
    max_seconds: 12,
    expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
  });
  const startBody = JSON.parse((await startRequest).postData() ?? "{}");
  expect(startBody).toMatchObject({
    device_id: null,
    fixture_wav: "/tmp/brain-sidecar-tests/testrun-123/input.wav",
    audio_source: "fixture",
    save_transcript: false,
  });
  await expect(testPanel.getByLabel("Live prepared audio")).toContainText("12.5s");
  await expect(testPanel.getByLabel("Recorded audio controls")).toContainText("Pause");
  await expect(page.getByLabel("Runtime summary")).toContainText("Recorded audio");
  await expect(page.getByLabel("Recorded audio playback")).toContainText("Playing through Live");
  await expect(page.getByLabel("Recorded audio playback")).toContainText("Waiting for the first transcript window");

  const pauseRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/pause")
  ));
  await testPanel.getByRole("button", { name: "Pause" }).click();
  await pauseRequest;
  await expect(page.getByLabel("Recorded audio playback")).toContainText("Paused");

  const resumeRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/resume")
  ));
  await testPanel.getByRole("button", { name: "Resume" }).click();
  await resumeRequest;
  await expect(page.getByLabel("Recorded audio playback")).toContainText("Playing through Live");

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: { status: "listening", queue_depth: 0, dropped_windows: 0 },
  });
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "live-test-line-1",
      text: "Online Generator Monitoring at T.A. Smith needs validation.",
      start_s: 0,
      end_s: 4,
      queue_depth: 0,
    },
  });
  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "live-test-card-1",
      session_id: "session-123",
      category: "action",
      title: "Validate generator monitoring",
      body: "Confirm the monitoring validation path from the recorded test conversation.",
      why_now: "Current transcript evidence mentioned generator monitoring validation.",
      priority: "high",
      confidence: 0.86,
      source_segment_ids: ["live-test-line-1"],
      source_type: "transcript",
      evidence_quote: "Online Generator Monitoring at T.A. Smith needs validation.",
      card_key: "live-test:generator-monitoring",
      raw_audio_retained: false,
    },
  });

  await expect(page.locator("#transcript")).toContainText("Online Generator Monitoring at T.A. Smith needs validation.");
  await expect(page.getByRole("region", { name: "Sidecar Intelligence" })).toContainText("Validate generator monitoring");

  const reportRequest = page.waitForRequest((candidate) => (
    candidate.method() === "POST" && candidate.url().endsWith("/api/test-mode/runs/testrun-123/report")
  ));
  await page.getByRole("banner").getByRole("button", { name: "Stop" }).click();
  const report = JSON.parse((await reportRequest).postData() ?? "{}").report;
  expect(report).toMatchObject({
    run_id: "testrun-123",
    transcript_segments: 1,
    expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
    matched_expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
  });
  await expect(testPanel.getByLabel("Live recorded audio report")).toContainText("2/2");
});

test("shows recoverable recorded audio start errors instead of looking hung", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, {
    testModeEnabled: true,
    startSessionResponse: {
      status: 500,
      body: { detail: "Fixture playback could not start" },
    },
  });
  await page.reload();

  await page.getByRole("main", { name: "Live" }).getByRole("button", { name: "Test" }).click();
  const testPanel = page.getByRole("region", { name: "Live audio test" });
  await testPanel.getByLabel("Test audio upload").setInputFiles({
    name: "bad-start.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("fake audio"),
  });

  await expect(page.getByRole("alert")).toContainText("Fixture playback could not start");
  await expect(testPanel.locator(".live-test-state")).toContainText("Ready");
  await expect(testPanel.getByRole("button", { name: "Reset" })).toBeEnabled();
});

test("starts recorded audio even when saved-session refresh fails", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { testModeEnabled: true, abortSessionListAfterCreate: true });
  await page.reload();

  await page.getByRole("main", { name: "Live" }).getByRole("button", { name: "Test" }).click();
  const testPanel = page.getByRole("region", { name: "Live audio test" });
  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await testPanel.getByLabel("Test audio upload").setInputFiles({
    name: "current-role-recording.webm",
    mimeType: "audio/webm",
    buffer: Buffer.from("fake audio"),
  });

  await startRequest;
  await expect(page.getByLabel("Recorded audio playback")).toContainText("Playing through Live");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("auto-stops recorded audio from the Live Test panel when the prepared clip ends", async ({ page }) => {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, { testModeEnabled: true, testPreparedDurationSeconds: 0.1 });
  await page.reload();

  await page.getByRole("main", { name: "Live" }).getByRole("button", { name: "Test" }).click();
  const testPanel = page.getByRole("region", { name: "Live audio test" });
  await testPanel.getByLabel("Live test expected terms").fill("tariff");

  const reportRequest = page.waitForRequest((candidate) => (
    candidate.method() === "POST" && candidate.url().endsWith("/api/test-mode/runs/testrun-123/report")
  ));
  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/sessions/session-123/start")
  ));
  await testPanel.getByLabel("Test audio upload").setInputFiles({
    name: "short-meeting.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("fake audio"),
  });
  await startRequest;

  await emitSessionEvent(page, {
    type: "audio_status",
    payload: { status: "listening", queue_depth: 0, dropped_windows: 0 },
  });
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "auto-stop-line-1",
      text: "Can we confirm the tariff analysis owner?",
      start_s: 0,
      end_s: 2,
      queue_depth: 0,
    },
  });

  const report = JSON.parse((await reportRequest).postData() ?? "{}").report;
  expect(report).toMatchObject({
    run_id: "testrun-123",
    transcript_segments: 1,
    expected_terms: ["tariff"],
    matched_expected_terms: ["tariff"],
  });
  await expect(testPanel.getByLabel("Live recorded audio report")).toContainText("1");
  await expect(testPanel.getByLabel("Live recorded audio report")).toContainText("1/1");
});

test("starts ephemeral Live and renders SSE updates", async ({ page }) => {
  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/start")
  ));

  await page.getByRole("button", { name: "Start Live" }).click();
  const request = await startRequest;
  const requestBody = JSON.parse(request.postData() ?? "{}");
  expect(requestBody).toMatchObject({
    device_id: null,
    audio_source: "server_device",
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
  expect(requestBody.save_transcript).toBeUndefined();
  await expect(page.getByLabel("Live retention status")).toContainText("Review handoff armed");
  await expect(page.getByLabel("Contract active")).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Meeting Output" })).toHaveCount(0);
  const returnsRail = page.getByRole("region", { name: "Sidecar Intelligence" });
  await expect(returnsRail).toContainText("Sidecar Intelligence");
  await expect(returnsRail).not.toContainText("Awaiting grounded returns");
  await expect(returnsRail).not.toContainText("Ollama");
  await expect(returnsRail).not.toContainText("Silent by design");
  await expect(page.getByLabel("Live field empty state")).toContainText("Listening... grounded cards will appear when there is enough evidence.");
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/live/live-123/events",
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
  await expect(page.locator("#transcript")).toContainText("platform team tomorrow");
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
  const contextSection = page.getByLabel("Reference context");
  await contextSection.getByText("Reference context").click();
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

test("keeps transcript echoes out of the Sidecar Intelligence rail", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/live/live-123/events",
  );

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "tariff-signal-line",
      text: "Can we confirm who owns the tariff analysis by Monday?",
      start_s: 0.0,
      end_s: 4.0,
      asr_model: "faster-whisper",
      raw_audio_retained: false,
    },
  });

  const intelligenceRail = page.getByRole("region", { name: "Sidecar Intelligence" });
  await expect(page.getByRole("region", { name: "Live Signals" })).toHaveCount(0);
  await expect(intelligenceRail).not.toContainText("Clarification cue");
  await expect(intelligenceRail).not.toContainText("Action cue");
  await expect(page.locator("#transcript")).toContainText("Can we confirm who owns the tariff analysis by Monday?");

  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "signal-card-1",
      session_id: "session-123",
      category: "clarification",
      title: "Tariff inputs",
      body: "Recent transcript points to utility billing or tariff analysis.",
      suggested_ask: "Can we get the rate schedule before pricing?",
      why_now: "Current transcript evidence mentioned tariff analysis.",
      priority: "normal",
      confidence: 0.84,
      source_segment_ids: ["tariff-signal-line"],
      source_type: "transcript",
      evidence_quote: "tariff analysis by Monday",
      card_key: "signal:tariff-inputs",
      raw_audio_retained: false,
    },
  });

  await expect(intelligenceRail).toContainText("Tariff inputs");
  await expect(intelligenceRail).toContainText("Can we get the rate schedule before pricing?");
  await expect(intelligenceRail).toContainText("Current transcript evidence mentioned tariff analysis.");
  await expect(page.getByRole("region", { name: "Live Signals" })).toHaveCount(0);
});

test("shows energy lens badge only from supported final transcript evidence", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/live/live-123/events",
  );

  await emitSessionEvent(page, {
    type: "transcript_partial",
    payload: {
      id: "energy-preview",
      text: "utility bill analysis and tariff analysis",
      start_s: 0.0,
      end_s: 1.2,
      is_final: false,
      asr_model: "faster-whisper",
      raw_audio_retained: false,
    },
  });
  await expect(page.getByLabel("Energy lens")).toHaveCount(0);

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "ppa-only",
      text: "PPA",
      start_s: 1.2,
      end_s: 2.0,
      asr_model: "faster-whisper",
      energy_lens_active: false,
      energy_lens_confidence: "low",
      raw_audio_retained: false,
    },
  });
  await expect(page.getByLabel("Energy lens")).toHaveCount(0);

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "energy-line",
      text: "We need utility bill analysis and tariff analysis before pricing this site.",
      start_s: 2.0,
      end_s: 5.0,
      asr_model: "faster-whisper",
      energy_lens_active: true,
      energy_lens_score: 1.86,
      energy_lens_confidence: "high",
      energy_lens_categories: ["operations", "market"],
      energy_lens_keywords: ["utility bill analysis", "tariff analysis"],
      energy_lens_evidence_segment_ids: ["energy-line"],
      energy_lens_evidence_quote: "We need utility bill analysis and tariff analysis before pricing this site.",
      energy_lens_summary_label: "Energy lens: operations + market",
      raw_audio_retained: false,
    },
  });

  const badge = page.getByLabel("Energy lens");
  await expect(badge).toContainText("Energy lens");
  await expect(badge).toHaveJSProperty("open", false);
  await badge.getByText("Energy lens").click();
  await expect(badge).toContainText("Detected from current transcript evidence");
  await expect(badge).toContainText("operations");
  await expect(badge).toContainText("utility bill analysis");
  await expect(badge).toContainText("tariff analysis");
});

test("renders energy-generated cards with evidence jump", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/live/live-123/events",
  );

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "energy-card-line",
      text: "We need utility bill analysis and tariff analysis before pricing this site.",
      start_s: 0.0,
      end_s: 4.0,
      asr_model: "faster-whisper",
      energy_lens_active: true,
      energy_lens_score: 1.86,
      energy_lens_confidence: "high",
      energy_lens_categories: ["operations", "market"],
      energy_lens_keywords: ["utility bill analysis", "tariff analysis"],
      energy_lens_evidence_segment_ids: ["energy-card-line"],
      energy_lens_evidence_quote: "We need utility bill analysis and tariff analysis before pricing this site.",
      energy_lens_summary_label: "Energy lens: operations + market",
      raw_audio_retained: false,
    },
  });
  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "energy-card-1",
      session_id: "session-123",
      category: "clarification",
      title: "Tariff inputs",
      body: "Recent transcript points to utility billing or tariff analysis.",
      suggested_ask: "Can we get the rate schedule, 12 months of bills, and interval data?",
      why_now: "Energy lens: operations + market is active from current transcript evidence.",
      priority: "high",
      confidence: 0.84,
      source_segment_ids: ["energy-card-line"],
      source_type: "transcript",
      evidence_quote: "utility bill analysis and tariff analysis",
      card_key: "energy:billing-tariff:energy-card-line",
      raw_audio_retained: false,
    },
  });

  const transcriptItem = page.getByLabel("Mic transcript item").filter({ hasText: "utility bill analysis" });
  const card = page.getByRole("feed", { name: "Live field" }).getByLabel("Clarify context card").filter({ hasText: "Tariff inputs" });
  await expect(card).toContainText("Evidence-backed");
  await expect(card).toContainText("Ask this");
  await card.getByRole("button", { name: "Evidence" }).click();
  await expect(card.getByLabel("Evidence for Tariff inputs")).toContainText("utility bill analysis");
  await card.getByRole("button", { name: "Jump to source" }).click();
  await expect(transcriptItem).toHaveClass(/evidence-highlight/);
});

test("auto-scrolls the live field as transcript and sidecar rows grow", async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 700 });
  await page.getByRole("button", { name: "Start Live" }).click();
  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/live/live-123/events",
  );

  for (let index = 0; index < 100; index += 1) {
    await emitSessionEvent(page, {
      type: "transcript_final",
      payload: {
        id: `scroll-line-${index}`,
        text: `Live auto-scroll test line ${index} with enough words to create a real transcript bubble.`,
        start_s: index,
        end_s: index + 0.8,
        asr_model: "faster-whisper",
        transcript_retention: "temporary",
        raw_audio_retained: false,
      },
    });
  }

  const liveField = page.locator("#transcript");
  await expect(liveField).toContainText("Live auto-scroll test line 99");
  await expect.poll(() => liveField.evaluate((element) => (
    element.scrollHeight - element.scrollTop - element.clientHeight
  ))).toBeLessThan(8);

  await liveField.hover();
  await page.mouse.wheel(0, -2400);
  await expect.poll(() => liveField.evaluate((element) => (
    element.scrollHeight - element.scrollTop - element.clientHeight
  ))).toBeGreaterThan(96);
  await expect(page.getByRole("button", { name: /Jump to live/ })).toBeVisible();

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "scroll-line-100",
      text: "Live auto-scroll test line 100 should not pull the reader away while history is being reviewed.",
      start_s: 100,
      end_s: 100.8,
      asr_model: "faster-whisper",
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  await expect(liveField).toContainText("Live auto-scroll test line 100");
  await expect(page.getByRole("button", { name: /Jump to live/ })).toBeVisible();
  await expect.poll(() => liveField.evaluate((element) => (
    element.scrollHeight - element.scrollTop - element.clientHeight
  ))).toBeGreaterThan(96);

  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "scroll-card",
      session_id: "session-123",
      category: "action",
      title: "Late context card",
      body: "This card arrives after the transcript row and should not leave the live field stuck above the bottom.",
      suggested_say: "Let's keep this visible at the bottom.",
      why_now: "It belongs to the newest transcript row.",
      priority: "high",
      confidence: 0.88,
      source_segment_ids: ["scroll-line-100"],
      source_type: "transcript",
      evidence_quote: "Live auto-scroll test line 100",
      raw_audio_retained: false,
    },
  });

  await expect(liveField).toContainText("Late context card");
  await expect(page.getByRole("button", { name: /Jump to live/ })).toBeVisible();
  await page.getByRole("button", { name: /Jump to live/ }).click();
  await expect.poll(() => liveField.evaluate((element) => (
    element.scrollHeight - element.scrollTop - element.clientHeight
  ))).toBeLessThan(8);
});

test("keeps showing newer ASR previews that overlap older finals", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
  await expect.poll(() => page.evaluate(() => window.__brainSidecarEventSourceUrls)).toContain(
    "http://127.0.0.1:8765/api/live/live-123/events",
  );

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "stream-final-1",
      text: "Earlier committed speech is already visible.",
      start_s: 0.0,
      end_s: 4.0,
      asr_model: "faster-whisper",
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  await emitSessionEvent(page, {
    type: "transcript_partial",
    payload: {
      id: "stream-preview-single",
      text: "newer",
      start_s: 0.48,
      end_s: 11.8,
      is_final: false,
      asr_model: "faster-whisper",
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });
  const transcript = page.locator("#transcript");
  await expect(transcript).not.toContainText("newer");

  await emitSessionEvent(page, {
    type: "transcript_partial",
    payload: {
      id: "stream-preview-1",
      text: "newer phrase is arriving",
      start_s: 0.48,
      end_s: 12.0,
      is_final: false,
      asr_model: "faster-whisper",
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });

  await expect(transcript).toContainText("Earlier committed speech");
  await expect(page.getByLabel("Mic transcript item").filter({ hasText: "newer phrase is arriving" })).toContainText("Preview");

  await emitSessionEvent(page, {
    type: "transcript_partial",
    payload: {
      id: "stream-preview-2",
      text: "newer phrase is still moving",
      start_s: 0.48,
      end_s: 12.4,
      is_final: false,
      asr_model: "faster-whisper",
      transcript_retention: "temporary",
      raw_audio_retained: false,
    },
  });

  await expect(page.getByLabel("Mic transcript item").filter({ hasText: "newer phrase is still moving" })).toContainText("Preview");
  await expect(transcript).not.toContainText("newer phrase is arriving");
});

test("keeps normal Live ephemeral and queues Review on stop", async ({ page }) => {
  await expect(page.getByRole("group", { name: "Session save mode" })).toHaveCount(0);
  await expect(page.getByLabel("Live retention status")).toContainText("Ephemeral Live");

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/start")
  ));
  await page.getByRole("banner").getByRole("button", { name: "Start Live" }).click();
  const body = JSON.parse((await startRequest).postData() ?? "{}");
  expect(body.save_transcript).toBeUndefined();
  await expect(page.getByLabel("Live retention status")).toContainText("Review handoff armed");

  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "ephemeral-line-1",
      text: "This live transcript should clear after queueing.",
      start_s: 0,
      end_s: 2,
    },
  });
  await expect(page.locator("#transcript")).toContainText("This live transcript should clear after queueing.");

  const stopRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/live-123/stop")
  ));
  await page.getByRole("banner").getByRole("button", { name: "Stop & Queue" }).click();
  await stopRequest;
  await expect(page.locator("#transcript")).not.toContainText("This live transcript should clear after queueing.");
  await page.getByRole("button", { name: "Review" }).click();
  const reviewMain = page.getByRole("main", { name: "Review" });
  const queueToggle = reviewMain.getByText(/^Queue \(\d+\)$/);
  if (await queueToggle.count()) {
    await queueToggle.first().click();
  }
  await expect(reviewMain.getByRole("region", { name: "Review queue" })).toContainText("Apollo call");
});

test("shows speaker identity controls instead of legacy ASR voice training", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
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

  const speaker = await openToolSection(page, "Speaker Identity", "Speaker identity");
  await expect(speaker).toBeVisible();
  await expect(speaker).toContainText("not enrolled");
  await expect(speaker).toContainText("0.0s usable");
  await expect(speaker).toContainText("60.0s needed");
  await expect(speaker.getByLabel("Speaker training steps")).toContainText("1 Start");
  await expect(page.getByRole("region", { name: "Voice profile" })).toHaveCount(0);
  await expect(page.getByText("ASR guidance preview")).toHaveCount(0);
  await expect(speaker.locator("details.speaker-maintenance")).toHaveJSProperty("open", false);
  await expect(speaker.getByRole("button", { name: "Delete Learned Embeddings" })).toHaveCount(0);
  await speaker.getByText("Advanced speaker controls").click();
  await expect(speaker).toContainText("embeddings");
  await expect(speaker.getByRole("button", { name: "Delete Learned Embeddings" })).toBeVisible();
  await expect(speaker.getByText("Review learned speaker profile")).toHaveCount(0);
  await expect(speaker.getByRole("button", { name: "Merge speakers" })).toHaveCount(0);
  await expect(speaker.getByRole("button", { name: "Split speaker" })).toHaveCount(0);
  await expect(speaker.getByRole("button", { name: "Rename speaker" })).toHaveCount(0);

  await speaker.getByRole("button", { name: "Set Up BP Label" }).first().click();
  await expect(speaker.getByRole("button", { name: "Record 8s Sample" })).toBeVisible();
  await expect(speaker.getByLabel("Speaker recording prompt")).toContainText("Record only BP");
  await expect(speaker.getByLabel("Speaker recording prompt")).toContainText("verifies BP");
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
  await page.getByRole("button", { name: "Start Live" }).click();
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
  await page.getByRole("button", { name: "Start Live" }).click();
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
  await page.getByRole("button", { name: "Start Live" }).click();
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
  await expect(page.getByRole("banner").getByRole("button", { name: "Live Running" })).toBeDisabled();
});

test("limits context cards and reveals raw metadata only in debug mode", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
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
  const contextSection = page.getByLabel("Reference context");
  await contextSection.getByText("Reference context").click();
  await expect(contextSection.locator(".context-card")).toHaveCount(3);
  await expect(page.locator("#transcript")).not.toContainText("debug-source-1");
  await expect(page.locator("#transcript")).not.toContainText("0.89");

  await page.getByRole("button", { name: "Tools" }).click();
  await openToolSection(page, "Logs & Debug");
  await page.getByLabel("Show debug metadata").check();
  await page.getByRole("button", { name: "Live", exact: true }).click();
  const refreshedContextSection = page.getByLabel("Reference context");
  await openDetailsByLabel(page, "Reference context");
  await refreshedContextSection.locator(".context-card").first().getByRole("button", { name: "Details" }).click();
  await expect(page.locator("#recall")).toContainText("source ID");
  await expect(page.locator("#recall")).toContainText(/debug-source-/);
  await expect(page.locator("#recall")).toContainText("score");
});

test("surfaces mocked SSE errors and supports stop", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);

  await emitSessionEvent(page, {
    type: "error",
    payload: { message: "Microphone disconnected" },
  });

  await expect(page.getByRole("alert")).toContainText("Microphone disconnected");
  await expect(page.getByLabel("Runtime status")).toContainText("error");

  await page.getByRole("banner").getByRole("button", { name: "Stop & Queue" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("stopped");
});

test("stop and queue keeps post-call persistence out of Live", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();

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
      id: "brief-reference-1",
      session_id: "session-123",
      category: "memory",
      title: "DOE relay reference",
      body: "CT/PT relay settings should be checked against the technical reference.",
      why_now: "Injected reference context.",
      priority: "normal",
      confidence: 0.76,
      source_segment_ids: [],
      source_type: "local_file",
      citations: ["runtime/reference/electrical-engineering/doe.pdf"],
      card_key: "reference:relay",
    },
  });

  await expect(page.locator("#transcript")).toContainText("Send the Siemens comments by Monday.");
  const stopRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/live-123/stop")
  ));
  await page.getByRole("banner").getByRole("button", { name: "Stop & Queue" }).click();
  await stopRequest;
  await expect(page.getByRole("region", { name: "Meeting Output" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Sidecar Intelligence" })).toBeVisible();
  await expect(page.locator("#transcript")).not.toContainText("Send the Siemens comments by Monday.");
  await page.getByRole("button", { name: "Review" }).click();
  const reviewMain = page.getByRole("main", { name: "Review" });
  const queueToggle = reviewMain.getByText(/^Queue \(\d+\)$/);
  if (await queueToggle.count()) {
    await queueToggle.first().click();
  }
  await expect(reviewMain.getByRole("region", { name: "Review queue" })).toContainText("Apollo call");
});

test("renders ephemeral web context notes in the context pane", async ({ page }) => {
  await page.getByRole("button", { name: "Start Live" }).click();
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
  await field.getByLabel("Reference context").getByText("Reference context").click();
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

  const testPanel = await openToolSection(page, "Test Mode", "Recorded audio test");
  await expect(testPanel).toBeVisible();
  await expect(testPanel.locator(".file-action")).toContainText("Upload audio");
  await expect(testPanel.locator("input.file-picker")).toHaveCount(0);
  await expect(testPanel).not.toContainText("No file chosen");

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
  await page.getByRole("button", { name: "Live", exact: true }).click();
  await expect(page.locator("#transcript")).toContainText("Online Generator Monitoring at T.A. Smith needs validation.");

  const reportRequest = page.waitForRequest((candidate) => (
    candidate.method() === "POST" && candidate.url().endsWith("/api/test-mode/runs/testrun-123/report")
  ));
  await page.getByRole("banner").getByRole("button", { name: "Stop" }).click();
  const report = JSON.parse((await reportRequest).postData() ?? "{}").report;

  expect(report).toMatchObject({
    run_id: "testrun-123",
    transcript_segments: 1,
    expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
    matched_expected_terms: ["Online Generator Monitoring", "T.A. Smith"],
  });
  await page.getByRole("button", { name: "Tools" }).click();
  const reopenedTestPanel = await openToolSection(page, "Test Mode", "Recorded audio test");
  await expect(reopenedTestPanel.getByLabel("Recorded audio report")).toContainText("1");
  await expect(reopenedTestPanel.getByLabel("Recorded audio report")).toContainText("2/2");
  await expect(reopenedTestPanel.getByLabel("Recorded audio report")).toContainText("report.json");
});

test("locks live capture to the auto-selected server microphone", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  const drawer = page.getByRole("main", { name: "Tools" });
  await expect(drawer.getByLabel("Audio source")).toHaveCount(0);
  await expect(drawer.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone");
  await expect(drawer.getByLabel("Browser microphone status")).toHaveCount(0);

  const startRequest = page.waitForRequest((request) => (
    request.method() === "POST" && request.url().endsWith("/api/live/start")
  ));
  await page.getByRole("banner").getByRole("button", { name: "Start Live" }).click();
  const request = await startRequest;

  expect(JSON.parse(request.postData() ?? "{}")).toMatchObject({
    device_id: null,
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

  const drawer = page.getByRole("main", { name: "Tools" });
  await expect(drawer.getByLabel("USB microphone")).toHaveCount(0);
  await expect(drawer.getByLabel("Server microphone")).toContainText("No healthy server microphone detected");
  await expect(drawer.getByRole("status")).toContainText("No healthy server microphone detected");
  await expect(drawer.getByRole("button", { name: "Test Mic" })).toBeDisabled();
  await expect(drawer.getByRole("button", { name: "Start Live" })).toHaveCount(0);
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Live" })).toBeDisabled();
  const speaker = await openToolSection(page, "Speaker Identity", "Speaker identity");
  await expect(speaker.getByRole("button", { name: "Set Up BP Label" }).first()).toBeDisabled();
});

test("uses mocked library and recall APIs", async ({ page }) => {
  await page.getByRole("button", { name: "References" }).click();
  const memoryPage = page.getByRole("main", { name: "References" });
  const categories = page.getByRole("navigation", { name: "Reference category navigation" });
  await expect(memoryPage.getByLabel("Browse index file")).toHaveCount(0);
  await categories.getByRole("button", { name: /Library Roots/ }).click();
  await memoryPage.getByLabel("Library root path").fill("/tmp/brain-sidecar-tests/input-files/uploaded-input.dat");
  await memoryPage.getByRole("button", { name: "Add Root" }).click();
  await expect(memoryPage.getByLabel("Library root path")).toHaveValue("");

  await memoryPage.getByLabel("Reference items").getByRole("button", { name: "Reindex References" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("indexed 42 chunks");

  await page.getByRole("button", { name: "Live", exact: true }).click();
  await page.getByPlaceholder("Ask Sidecar").fill("apollo rollout");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Prior transcript");
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Technical references");
  await expect(page.getByRole("region", { name: "Manual query source sections" })).toContainText("Web");
  await expect(page.locator("#transcript").getByText("Prior planning note about the Apollo rollout.")).toBeVisible();
  await expect(page.locator("#transcript").getByText("DOE Electrical Science Volume 1", { exact: true })).toBeVisible();
  await expect(page.locator("#transcript").getByText("Web context: apollo rollout")).toBeVisible();
  await expect(page.locator("#transcript")).toContainText("breaker coordination");
  await expect(page.getByLabel("You transcript item")).toContainText("apollo rollout");
  await expect(page.locator("#transcript")).toContainText("Prior planning note about the Apollo rollout.");
  await expect(page.locator("#transcript")).not.toContainText("notes.md");
  await expect(page.locator("#transcript")).not.toContainText("score 0.93");
});

test("keeps transcript and context readable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Start Live" }).click();
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

test("keeps cockpit pages compact at desktop and laptop widths", async ({ page }) => {
  for (const size of [
    { width: 1920, height: 1008 },
    { width: 1440, height: 900 },
  ]) {
    await page.setViewportSize(size);
    await page.getByRole("button", { name: "Live", exact: true }).click();
    await expect(page.getByRole("banner").getByRole("button", { name: "Start Live" })).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

    await page.getByRole("button", { name: "Tools" }).click();
    await expect(page.getByRole("main", { name: "Tools" })).toBeVisible();
    await expect(page.getByLabel("Voice & Input")).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

    await page.getByRole("button", { name: "Models" }).click();
    await expect(page.getByRole("main", { name: "Models" })).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

    await page.getByRole("button", { name: "References" }).click();
    await expect(page.getByLabel("Reference categories")).toBeVisible();
    await expect(page.getByLabel("Reference items")).toBeVisible();
    await expect(page.getByLabel("Reference detail")).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
});

test("persists theme selection across reloads", async ({ page }) => {
  await page.getByRole("button", { name: "Tools" }).click();
  let appearance = await openToolSection(page, "Appearance");
  await appearance.getByRole("combobox", { name: "Theme" }).selectOption("canopy");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "canopy");

  await page.reload();
  await page.getByRole("button", { name: "Tools" }).click();
  appearance = await openToolSection(page, "Appearance");
  await expect(appearance.getByRole("combobox", { name: "Theme" })).toHaveValue("canopy");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "canopy");
});

test("keeps core capture controls usable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });

  await expect(page.getByRole("heading", { name: "Conversation" })).toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  await expect(page.getByLabel("Server microphone")).toContainText("Blue Yeti USB Microphone");
  await expect(page.getByLabel("Guided mic tuning")).toBeVisible();
  await expect(page.getByRole("banner").getByRole("button", { name: "Start Live" })).toBeVisible();
});

async function openToolSection(page: Page, tabName: string, regionName = tabName) {
  await page.getByRole("navigation", { name: "Tool sections" }).getByRole("button", { name: tabName }).click();
  const region = page.getByRole("region", { name: regionName });
  await expect(region).toBeVisible();
  return region;
}

async function openReviewPage(page: Page) {
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Review" }).click();
  const review = page.getByRole("main", { name: "Review" });
  await expect(review).toBeVisible();
  return review;
}

async function openCompletedReview(page: Page) {
  const review = await openReviewPage(page);
  const queueToggle = review.getByText(/^Queue \(\d+\)$/);
  if (await queueToggle.count()) {
    await queueToggle.first().click();
  }
  await review.getByRole("button", { name: /Apollo/ }).click();
  await expect(review.getByRole("region", { name: "Review summary validation" })).toContainText("Needs validation");
  const queueMenu = review.getByLabel("Review queue menu");
  if (await queueMenu.count()) {
    const queueOpen = await queueMenu.evaluate((element) => (
      element instanceof HTMLDetailsElement ? element.open : false
    ));
    if (queueOpen) {
      await queueMenu.locator("summary").first().click();
    }
  }
  return review;
}

async function remockApi(page: Page, options: Parameters<typeof mockApi>[1] = {}) {
  await page.unroute("http://127.0.0.1:8765/api/**");
  await mockApi(page, options);
  await page.goto("/");
}

async function installClipboardStub(page: Page) {
  await page.evaluate(() => {
    type ClipboardWindow = typeof window & { __brainSidecarCopiedText?: string };
    (window as ClipboardWindow).__brainSidecarCopiedText = "";
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as ClipboardWindow).__brainSidecarCopiedText = text;
        },
      },
    });
  });
}

async function clipboardText(page: Page) {
  return page.evaluate(() => (
    (window as typeof window & { __brainSidecarCopiedText?: string }).__brainSidecarCopiedText ?? ""
  ));
}

async function expectReviewTextFit(page: Page) {
  const failures = await page.evaluate(() => {
    const root = document.querySelector<HTMLElement>('[aria-label="Review"].review-page');
    if (!root) {
      return ["Review page root was not found."];
    }
    const result: string[] = [];
    const tolerance = 2;
    const describe = (element: Element) => {
      const selector = element instanceof HTMLElement
        ? element.className || element.getAttribute("aria-label") || element.tagName.toLowerCase()
        : element.tagName.toLowerCase();
      const text = (element.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 90);
      return `${selector}${text ? `: ${text}` : ""}`;
    };
    const isVisible = (element: Element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      let parent = element.parentElement;
      while (parent) {
        if (parent instanceof HTMLDetailsElement && !parent.open) {
          const summary = parent.querySelector("summary");
          if (summary !== element && !summary?.contains(element)) {
            return false;
          }
        }
        parent = parent.parentElement;
      }
      return rect.width > 1 && rect.height > 1 && style.visibility !== "hidden" && style.display !== "none";
    };
    const targets = [
      ".review-summary-toolbar",
      ".meeting-summary-notes",
      ".meeting-summary-notes *",
      ".meeting-summary-note-item",
      ".meeting-summary-brief",
      ".meeting-summary-brief *",
      ".meeting-summary-hero",
      ".meeting-summary-brief-section",
      ".meeting-summary-brief-item",
      ".meeting-summary-brief-item-meta span",
      ".meeting-summary-brief-group h4",
      ".meeting-summary-metrics span",
      ".meeting-summary-workstream-list article",
      ".meeting-summary-technical-list article",
      ".meeting-summary-context-list article",
      ".review-queue-panel",
      ".review-queue-item",
      ".review-summary-actions button",
      ".review-status-chips .chip",
      ".review-output-panel",
      ".review-output-panel .context-card",
      ".review-transcript-panel",
      ".review-transcript-panel .saved-transcript-row",
      ".review-page button",
      ".review-page .chip",
      ".review-page summary",
      ".file-action span",
      ".file-action small",
      ".source-segment-chip",
    ];
    const wrapTargets = [
      ".review-page button",
      ".review-page .chip",
      ".review-page summary",
      ".review-queue-item span",
      ".review-queue-item strong",
      ".review-queue-item small",
      ".meeting-summary-notes-hero h2",
      ".meeting-summary-notes-hero p",
      ".meeting-summary-note-item span",
      ".meeting-summary-empty-line",
      ".meeting-summary-hero h2",
      ".meeting-summary-hero p",
      ".meeting-summary-evidence span",
      ".meeting-summary-brief-item",
      ".meeting-summary-brief-item-meta span",
      ".meeting-summary-brief-group h4",
      ".meeting-summary-metrics span",
      ".meeting-summary-keypoints span",
      ".meeting-summary-row-head strong",
      ".meeting-summary-row-note",
      ".meeting-summary-mini-list li",
      ".meeting-summary-context-list strong",
      ".context-card h3",
      ".context-card p",
      ".file-action span",
      ".file-action small",
      ".saved-transcript-row p",
      ".source-segment-chip",
    ];
    const fontTargets = [
      ".review-summary-toolbar",
      ".meeting-summary-notes",
      ".meeting-summary-notes-section",
      ".meeting-summary-note-item",
      ".meeting-summary-brief",
      ".meeting-summary-brief-section",
      ".meeting-summary-brief-item",
      ".meeting-summary-workstream-list article",
      ".meeting-summary-technical-list article",
      ".review-queue-item",
      ".context-card",
      ".saved-transcript-row",
      ".review-page button",
      ".review-page .chip",
      ".review-page summary",
    ];
    if (document.documentElement.scrollWidth > window.innerWidth + tolerance) {
      result.push(`Document overflow ${document.documentElement.scrollWidth}px > ${window.innerWidth}px`);
    }
    for (const selector of targets) {
      for (const element of Array.from(root.querySelectorAll<HTMLElement>(selector)).filter(isVisible)) {
        const rect = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        if (element.scrollWidth > element.clientWidth + tolerance && !["auto", "scroll"].includes(style.overflowX)) {
          result.push(`Horizontal text overflow in ${selector} (${element.scrollWidth}px > ${element.clientWidth}px): ${describe(element)}`);
        }
        if (rect.left < -tolerance || rect.right > window.innerWidth + tolerance) {
          result.push(`Element escapes viewport in ${selector} (${Math.round(rect.left)}..${Math.round(rect.right)} of ${window.innerWidth}): ${describe(element)}`);
        }
      }
    }
    for (const selector of wrapTargets) {
      for (const element of Array.from(root.querySelectorAll<HTMLElement>(selector)).filter(isVisible)) {
        const style = window.getComputedStyle(element);
        const wrapsWords = ["anywhere", "break-word"].includes(style.overflowWrap) || ["break-word", "break-all"].includes(style.wordBreak);
        if (style.whiteSpace === "nowrap") {
          result.push(`No wrapping allowed in ${selector}: ${describe(element)}`);
        }
        if (!wrapsWords) {
          result.push(`Long-word wrapping is not enabled in ${selector}: ${describe(element)}`);
        }
        if (selector.startsWith(".meeting-summary-") && style.overflowWrap === "anywhere") {
          result.push(`Meeting summary text uses overflow-wrap:anywhere in ${selector}: ${describe(element)}`);
        }
      }
    }
    for (const selector of fontTargets) {
      for (const element of Array.from(root.querySelectorAll<HTMLElement>(selector)).filter(isVisible)) {
        const style = window.getComputedStyle(element);
        const fontSize = Number.parseFloat(style.fontSize);
        const lineHeight = style.lineHeight === "normal" ? fontSize * 1.2 : Number.parseFloat(style.lineHeight);
        if (fontSize < 10) {
          result.push(`Font too small in ${selector} (${style.fontSize}): ${describe(element)}`);
        }
        if (Number.isFinite(lineHeight) && lineHeight < fontSize * 1.05) {
          result.push(`Line height too tight in ${selector} (${style.lineHeight} for ${style.fontSize}): ${describe(element)}`);
        }
      }
    }
    return result;
  });
  expect(failures).toEqual([]);
}

async function expectReviewSummaryHierarchy(page: Page) {
  const result = await page.evaluate(() => {
    const summaryPrimary = document.querySelector('[aria-label="Review summary validation"]');
    const transcriptDetails = document.querySelector('[aria-label="Sources / transcript"]');
    const transcript = document.querySelector('[aria-label="Clean Transcript"]');
    const hierarchyLabels = ["Meeting Summary", "Summary", "Notes", "Decisions", "Follow-ups", "Open questions", "Risks / concerns"];
    const hierarchyNodes = hierarchyLabels.map((label) => document.querySelector(`[aria-label="${label}"]`));
    const hierarchyIndexes = hierarchyNodes.map((node) => node ? Array.from(document.querySelectorAll("[aria-label]")).indexOf(node) : -1);
    const summaryBox = summaryPrimary?.getBoundingClientRect();
    const transcriptDetailsBox = transcriptDetails?.getBoundingClientRect();
    const transcriptBox = transcript?.getBoundingClientRect();
    const transcriptOpen = transcriptDetails instanceof HTMLDetailsElement ? transcriptDetails.open : false;
    return {
      hierarchyPresent: hierarchyIndexes.every((index) => index >= 0),
      hierarchyInOrder: hierarchyIndexes.every((index, itemIndex, indexes) => itemIndex === 0 || indexes[itemIndex - 1] < index),
      validationCardsHidden: !document.querySelector('[aria-label="Validation Evidence Cards"]'),
      summaryBeforeSources: Boolean(summaryPrimary && transcriptDetails && (summaryPrimary.compareDocumentPosition(transcriptDetails) & Node.DOCUMENT_POSITION_FOLLOWING)),
      hierarchyBeforeSources: hierarchyNodes.every((node) => Boolean(node && transcriptDetails && (node.compareDocumentPosition(transcriptDetails) & Node.DOCUMENT_POSITION_FOLLOWING))),
      transcriptHiddenByDefault: !transcriptOpen,
      hierarchyBeforeTranscriptWhenOpen: !transcriptOpen || hierarchyNodes.every((node) => Boolean(node && transcript && (node.compareDocumentPosition(transcript) & Node.DOCUMENT_POSITION_FOLLOWING))),
      visuallyBeforeEvidence: Boolean(summaryBox && transcriptDetailsBox && summaryBox.top <= (transcriptOpen && transcriptBox ? Math.min(transcriptDetailsBox.top, transcriptBox.top) : transcriptDetailsBox.top)),
    };
  });
  expect(result).toEqual({
    hierarchyPresent: true,
    hierarchyInOrder: true,
    validationCardsHidden: true,
    summaryBeforeSources: true,
    hierarchyBeforeSources: true,
    transcriptHiddenByDefault: true,
    hierarchyBeforeTranscriptWhenOpen: true,
    visuallyBeforeEvidence: true,
  });
}

async function expectReviewSummaryTextVisible(page: Page) {
  const failures = await page.evaluate(() => {
    const root = document.querySelector<HTMLElement>('[aria-label="Review"].review-page');
    if (!root) {
      return ["Review page root was not found."];
    }
    const result: string[] = [];
    const tolerance = 2;
    const describe = (element: Element) => {
      const selector = element instanceof HTMLElement
        ? element.className || element.getAttribute("aria-label") || element.tagName.toLowerCase()
        : element.tagName.toLowerCase();
      const text = (element.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 90);
      return `${selector}${text ? `: ${text}` : ""}`;
    };
    const isVisible = (element: Element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      let parent = element.parentElement;
      while (parent) {
        if (parent instanceof HTMLDetailsElement && !parent.open) {
          const summary = parent.querySelector("summary");
          if (summary !== element && !summary?.contains(element)) {
            return false;
          }
        }
        parent = parent.parentElement;
      }
      return rect.width > 1 && rect.height > 1 && style.visibility !== "hidden" && style.display !== "none";
    };
    const selectors = [
      ".review-summary-toolbar h2",
      ".review-summary-toolbar span",
      ".meeting-summary-notes-hero h2",
      ".meeting-summary-notes-hero p",
      ".meeting-summary-notes-section h3",
      ".meeting-summary-note-item span",
      ".meeting-summary-empty-line",
      ".meeting-summary-hero h2",
      ".meeting-summary-hero p",
      ".meeting-summary-evidence span",
      ".meeting-summary-rollup *",
      ".meeting-summary-workstreams *",
      ".meeting-summary-technical *",
      ".meeting-summary-metrics span",
      ".meeting-summary-keypoints span",
    ];
    for (const selector of selectors) {
      for (const element of Array.from(root.querySelectorAll<HTMLElement>(selector)).filter(isVisible)) {
        const style = window.getComputedStyle(element);
        if (["hidden", "clip"].includes(style.overflowX) && element.scrollWidth > element.clientWidth + tolerance) {
          result.push(`Summary text clips horizontally in ${selector}: ${describe(element)}`);
        }
        if (["hidden", "clip"].includes(style.overflowY) && element.scrollHeight > element.clientHeight + tolerance) {
          result.push(`Summary text clips vertically in ${selector}: ${describe(element)}`);
        }
        if (selector === ".meeting-summary-note-item span") {
          const lineHeight = style.lineHeight === "normal" ? Number.parseFloat(style.fontSize) * 1.2 : Number.parseFloat(style.lineHeight);
          const text = (element.textContent ?? "").trim();
          const hasUsefulWord = /\b[A-Za-z0-9-]{8,}\b/.test(text);
          if (hasUsefulWord && element.getBoundingClientRect().width < 120 && element.getBoundingClientRect().height > lineHeight * 3.5) {
            result.push(`Summary note appears squeezed into narrow vertical wrapping: ${describe(element)}`);
          }
          if (style.overflowWrap === "anywhere") {
            result.push(`Summary note uses overflow-wrap:anywhere: ${describe(element)}`);
          }
        }
      }
    }
    return result;
  });
  expect(failures).toEqual([]);
}

async function expectReviewControlTargetsAndFocus(page: Page) {
  const failures = await page.evaluate(() => {
    const root = document.querySelector<HTMLElement>('[aria-label="Review"].review-page');
    if (!root) {
      return ["Review page root was not found."];
    }
    const result: string[] = [];
    const tolerance = 2;
    const minimumTarget = 24;
    const originalWindowScroll = { x: window.scrollX, y: window.scrollY };
    const originalRootScroll = root.scrollTop;
    const describe = (element: Element) => {
      const selector = element instanceof HTMLElement
        ? element.className || element.getAttribute("aria-label") || element.tagName.toLowerCase()
        : element.tagName.toLowerCase();
      const text = (element.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 90);
      return `${selector}${text ? `: ${text}` : ""}`;
    };
    const isVisible = (element: Element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      let parent = element.parentElement;
      while (parent) {
        if (parent instanceof HTMLDetailsElement && !parent.open) {
          const summary = parent.querySelector("summary");
          if (summary !== element && !summary?.contains(element)) {
            return false;
          }
        }
        parent = parent.parentElement;
      }
      return rect.width > 1 && rect.height > 1 && style.visibility !== "hidden" && style.display !== "none";
    };
    const targetControls = Array.from(root.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, .file-action'
    )).filter(isVisible);
    for (const element of targetControls) {
      const targetBox = element.classList.contains("file-picker-native")
        ? element.closest<HTMLElement>(".file-action")?.getBoundingClientRect()
        : element.getBoundingClientRect();
      if (!targetBox) {
        continue;
      }
      if (targetBox.width + tolerance < minimumTarget || targetBox.height + tolerance < minimumTarget) {
        result.push(`Review control target is below ${minimumTarget}px (${Math.round(targetBox.width)}x${Math.round(targetBox.height)}): ${describe(element)}`);
      }
    }

    const focusableControls = Array.from(root.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])'
    )).filter(isVisible);
    for (const element of focusableControls) {
      element.focus({ preventScroll: false });
      const focusBoxElement = element.classList.contains("file-picker-native")
        ? element.closest<HTMLElement>(".file-action") ?? element
        : element;
      focusBoxElement.scrollIntoView({ block: "nearest", inline: "nearest" });
      const rect = focusBoxElement.getBoundingClientRect();
      if (rect.top < -tolerance || rect.left < -tolerance || rect.bottom > window.innerHeight + tolerance || rect.right > window.innerWidth + tolerance) {
        result.push(`Focused Review control is not fully visible (${Math.round(rect.left)},${Math.round(rect.top)} ${Math.round(rect.width)}x${Math.round(rect.height)}): ${describe(element)}`);
        continue;
      }
      const pointX = Math.max(0, Math.min(window.innerWidth - 1, rect.left + rect.width / 2));
      const pointY = Math.max(0, Math.min(window.innerHeight - 1, rect.top + rect.height / 2));
      const topElement = document.elementFromPoint(pointX, pointY);
      const focusVisible = Boolean(topElement && (
        element === topElement
        || element.contains(topElement)
        || topElement.contains(element)
        || focusBoxElement === topElement
        || focusBoxElement.contains(topElement)
        || topElement.contains(focusBoxElement)
      ));
      if (!focusVisible) {
        result.push(`Focused Review control is obscured by ${topElement ? describe(topElement) : "nothing"}: ${describe(element)}`);
      }
    }
    root.scrollTop = originalRootScroll;
    window.scrollTo(originalWindowScroll.x, originalWindowScroll.y);
    return result.slice(0, 30);
  });
  expect(failures).toEqual([]);
}

async function expectReviewTextSpacingFit(page: Page) {
  const failures = await page.evaluate(() => {
    const root = document.querySelector<HTMLElement>('[aria-label="Review"].review-page');
    if (!root) {
      return ["Review page root was not found."];
    }
    const result: string[] = [];
    const tolerance = 2;
    const describe = (element: Element) => {
      const selector = element instanceof HTMLElement
        ? element.className || element.getAttribute("aria-label") || element.tagName.toLowerCase()
        : element.tagName.toLowerCase();
      const text = (element.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 90);
      return `${selector}${text ? `: ${text}` : ""}`;
    };
    const isVisible = (element: Element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      let parent = element.parentElement;
      while (parent) {
        if (parent instanceof HTMLDetailsElement && !parent.open) {
          const summary = parent.querySelector("summary");
          if (summary !== element && !summary?.contains(element)) {
            return false;
          }
        }
        parent = parent.parentElement;
      }
      return rect.width > 1 && rect.height > 1 && style.visibility !== "hidden" && style.display !== "none";
    };
    const selectors = [
      ".review-summary-toolbar h2",
      ".review-summary-toolbar span",
      ".meeting-summary-notes-hero h2",
      ".meeting-summary-notes-hero p",
      ".meeting-summary-note-item span",
      ".meeting-summary-hero h2",
      ".meeting-summary-hero p",
      ".meeting-summary-keypoints span",
      ".meeting-summary-metrics span",
      ".meeting-summary-evidence span",
      ".meeting-summary-mini-list li",
      ".meeting-summary-row-note",
      ".meeting-summary-context-list strong",
      ".review-step small",
      ".context-card h3",
      ".context-card p",
      ".saved-transcript-row p",
      ".source-segment-chip",
    ];
    for (const selector of selectors) {
      for (const element of Array.from(root.querySelectorAll<HTMLElement>(selector)).filter(isVisible)) {
        const style = window.getComputedStyle(element);
        const hidesX = ["hidden", "clip"].includes(style.overflowX);
        const hidesY = ["hidden", "clip"].includes(style.overflowY);
        if (hidesX && element.scrollWidth > element.clientWidth + tolerance) {
          result.push(`Text spacing clips horizontally in ${selector}: ${describe(element)}`);
        }
        if (hidesY && element.scrollHeight > element.clientHeight + tolerance) {
          result.push(`Text spacing clips vertically in ${selector}: ${describe(element)}`);
        }
      }
    }
    return result;
  });
  expect(failures).toEqual([]);
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function countOccurrences(value: string, needle: string): number {
  return value.split(needle).length - 1;
}

async function openDetailsByLabel(page: Page, name: string) {
  const region = page.getByLabel(name);
  await expect(region).toBeVisible();
  const isOpen = await region.evaluate((element) => (
    element instanceof HTMLDetailsElement ? element.open : true
  ));
  if (!isOpen) {
    await region.locator("summary").first().click();
  }
  return region;
}
