import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const apiBase = process.env.PW_API_BASE ?? "http://127.0.0.1:8775";
const fixtureWav = process.env.BRAIN_SIDECAR_E2E_FIXTURE_WAV ?? "";
const artifactDir = process.env.BRAIN_SIDECAR_USER_TEST_ARTIFACT_DIR
  ?? path.resolve("../runtime/user-tests/apr-2-1200/artifacts");
const durationSeconds = Number(process.env.BRAIN_SIDECAR_USER_TEST_DURATION_SECONDS ?? 300);
const screenshotIntervalMs = Number(process.env.BRAIN_SIDECAR_USER_TEST_SCREENSHOT_INTERVAL_MS ?? 90_000);
const maxStallMs = Number(process.env.BRAIN_SIDECAR_USER_TEST_MAX_TRANSCRIPT_STALL_MS ?? 240_000);
const expectReferenceContext = process.env.BRAIN_SIDECAR_EXPECT_REFERENCE_CONTEXT === "1";
const expectedResponseTerms = parseExpectedTerms(process.env.BRAIN_SIDECAR_EXPECT_RESPONSE_TERMS ?? "");
const expectedTranscriptTerms = parseExpectedTerms(process.env.BRAIN_SIDECAR_EXPECT_TRANSCRIPT_TERMS ?? "");
const expectedCardTerms = parseExpectedTerms(process.env.BRAIN_SIDECAR_EXPECT_CARD_TERMS ?? "");
const expectedResponseTermMinHits = expectedMinHits(
  process.env.BRAIN_SIDECAR_EXPECT_TERM_MIN_HITS,
  expectedResponseTerms,
);
const expectedTranscriptTermMinHits = expectedMinHits(
  process.env.BRAIN_SIDECAR_EXPECT_TRANSCRIPT_TERM_MIN_HITS,
  expectedTranscriptTerms,
);
const expectedCardTermMinHits = expectedMinHits(
  process.env.BRAIN_SIDECAR_EXPECT_CARD_TERM_MIN_HITS,
  expectedCardTerms,
);
const expectedNoteCardsMin = Number(process.env.BRAIN_SIDECAR_EXPECT_NOTE_CARDS_MIN ?? "0");
const expectEnergyLens = parseExpectedBoolean(process.env.BRAIN_SIDECAR_EXPECT_ENERGY_LENS);

type Checkpoint = {
  at_s: number;
  screenshot: string;
  screenshots: string[];
  status: string;
  transcript_segments: number;
  note_cards: number;
  reference_context_cards: number;
  queue_depth: number;
  dropped_windows: number;
  reference_context_text_chars: number;
  error_count: number;
  latest_heard_chars: number;
  meeting_output_cards: number;
  transcript_text_chars: number;
  meeting_output_text_chars: number;
  energy_lens_visible: boolean;
  energy_lens_text_chars: number;
};

type Report = {
  source_fixture: string;
  api_base: string;
  duration_seconds: number;
  screenshot_interval_ms: number;
  started_at: string;
  completed_at?: string;
  health?: Record<string, unknown>;
  checkpoints: Checkpoint[];
  final?: Omit<Checkpoint, "screenshot" | "screenshots">;
  expected_response_terms?: string[];
  matched_response_terms?: string[];
  expected_transcript_terms?: string[];
  matched_transcript_terms?: string[];
  expected_card_terms?: string[];
  matched_card_terms?: string[];
  expected_note_cards_min?: number;
  expected_energy_lens?: boolean;
  combined_transcript_text?: string;
  meeting_output_text?: string;
  energy_lens_text?: string;
  note_card_count?: number;
  energy_lens_visible?: boolean;
  response_text_chars?: number;
  transcript_text_chars?: number;
  meeting_output_text_chars?: number;
  issues: string[];
};

test("plays current-role audio through the real UI and validates response quality", async ({ page, request }) => {
  test.setTimeout(Math.max(180_000, Math.ceil(durationSeconds * 1000) + 420_000));

  fs.mkdirSync(artifactDir, { recursive: true });
  expect(fixtureWav, "BRAIN_SIDECAR_E2E_FIXTURE_WAV must be set").toBeTruthy();
  expect(fs.existsSync(fixtureWav), `Fixture WAV does not exist: ${fixtureWav}`).toBeTruthy();

  const healthResponse = await request.get(`${apiBase}/api/health/gpu`);
  expect(healthResponse.ok()).toBeTruthy();
  const health = await healthResponse.json();
  expect(health.nvidia_available).toBe(true);
  expect(health.asr_cuda_available, health.asr_cuda_error ?? "ASR CUDA should be available").toBe(true);

  const report: Report = {
    source_fixture: fixtureWav,
    api_base: apiBase,
    duration_seconds: durationSeconds,
    screenshot_interval_ms: screenshotIntervalMs,
    started_at: new Date().toISOString(),
    health,
    checkpoints: [],
    issues: [],
  };
  if (expectedResponseTerms.length > 0) {
    report.expected_response_terms = expectedResponseTerms;
  }
  if (expectedTranscriptTerms.length > 0) {
    report.expected_transcript_terms = expectedTranscriptTerms;
  }
  if (expectedCardTerms.length > 0) {
    report.expected_card_terms = expectedCardTerms;
  }
  if (expectedNoteCardsMin > 0) {
    report.expected_note_cards_min = expectedNoteCardsMin;
  }
  if (expectEnergyLens) {
    report.expected_energy_lens = true;
  }

  writeReport(report);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Conversation" })).toBeVisible();
  await expect(page.getByRole("feed", { name: "Live field" })).toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  const toolsPage = page.getByRole("main", { name: "Tools" });
  const fixtureControls = toolsPage.getByRole("region", { name: "Fixture audio test controls" });
  await expect(fixtureControls).toBeVisible();
  await fixtureControls.getByLabel("Fixture WAV path").fill(fixtureWav);
  await page.getByRole("button", { name: "Live" }).click();
  await captureCheckpoint(page, report, "00-start");

  await page.getByRole("banner").getByRole("button", { name: "Start Playback" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|loading_asr|listening/, { timeout: 30_000 });
  await expect(page.getByLabel("Runtime status")).toContainText("listening", { timeout: 90_000 });

  await waitForTranscriptProgress(page, 0, 180_000);
  await captureCheckpoint(page, report, "01-first-transcript");

  const startedAt = Date.now();
  const stopAt = startedAt + Math.ceil(durationSeconds * 1000) + 20_000;
  let lastTranscriptCount = await readMetric(page, "Transcript");
  let lastTranscriptAt = Date.now();
  let checkpointIndex = 2;

  while (Date.now() < stopAt) {
    const waitMs = Math.min(screenshotIntervalMs, Math.max(1000, stopAt - Date.now()));
    await page.waitForTimeout(waitMs);
    const currentTranscriptCount = await readMetric(page, "Transcript");
    if (currentTranscriptCount > lastTranscriptCount) {
      lastTranscriptCount = currentTranscriptCount;
      lastTranscriptAt = Date.now();
    }

    await captureCheckpoint(page, report, `${String(checkpointIndex).padStart(2, "0")}-periodic`);
    checkpointIndex += 1;

    const errorCount = await page.getByRole("alert").count();
    if (errorCount > 0) {
      report.issues.push("UI surfaced one or more error alerts during playback.");
      await captureCheckpoint(page, report, `${String(checkpointIndex).padStart(2, "0")}-error`);
      break;
    }

    if (Date.now() - lastTranscriptAt > maxStallMs) {
      report.issues.push(`Transcript stalled for more than ${Math.round(maxStallMs / 1000)} seconds.`);
      await captureCheckpoint(page, report, `${String(checkpointIndex).padStart(2, "0")}-stall`);
      break;
    }
  }

  await page.getByRole("banner").getByRole("button", { name: /Stop|Stop & Queue|Stop Playback/ }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("stopped", { timeout: 30_000 });
  await expandMeetingOutputDetails(page);
  const finalCheckpoint = await captureCheckpoint(page, report, `${String(checkpointIndex).padStart(2, "0")}-final`);
  report.final = withoutScreenshot(finalCheckpoint);
  const transcriptText = await combinedTranscriptText(page);
  const meetingOutputText = await currentMeetingOutputText(page);
  const cardText = await currentMeetingCardText(page);
  const energyLensText = await visibleEnergyLensText(page);
  const responseText = await combinedResponseText(page);
  report.combined_transcript_text = transcriptText.trim();
  report.meeting_output_text = meetingOutputText.trim();
  report.energy_lens_text = energyLensText.trim();
  report.note_card_count = await meetingOutputCardCount(page);
  report.energy_lens_visible = await energyLensBadgeVisible(page);
  report.response_text_chars = responseText.trim().length;
  report.transcript_text_chars = transcriptText.trim().length;
  report.meeting_output_text_chars = meetingOutputText.trim().length;
  report.matched_response_terms = matchExpectedTerms(responseText, expectedResponseTerms);
  report.matched_transcript_terms = matchExpectedTerms(transcriptText, expectedTranscriptTerms);
  report.matched_card_terms = matchExpectedTerms(cardText, expectedCardTerms);
  if (report.matched_response_terms.length < expectedResponseTermMinHits) {
    report.issues.push(
      `Expected at least ${expectedResponseTermMinHits} response terms, matched ${report.matched_response_terms.length}.`,
    );
  }
  if (report.matched_transcript_terms.length < expectedTranscriptTermMinHits) {
    report.issues.push(
      `Expected at least ${expectedTranscriptTermMinHits} transcript terms, matched ${report.matched_transcript_terms.length}.`,
    );
  }
  if (expectedNoteCardsMin > 0 && report.note_card_count < expectedNoteCardsMin) {
    report.issues.push(`Expected at least ${expectedNoteCardsMin} Meeting Output cards, found ${report.note_card_count}.`);
  }
  if (report.matched_card_terms.length < expectedCardTermMinHits) {
    report.issues.push(
      `Expected at least ${expectedCardTermMinHits} Meeting Output card terms, matched ${report.matched_card_terms.length}.`,
    );
  }
  if (expectEnergyLens && !report.energy_lens_visible) {
    report.issues.push("Expected the Energy lens badge to be visible.");
  }
  if (finalCheckpoint.dropped_windows !== 0) {
    report.issues.push(`Expected zero dropped windows, found ${finalCheckpoint.dropped_windows}.`);
  }
  if (finalCheckpoint.error_count !== 0) {
    report.issues.push(`Expected zero UI alerts, found ${finalCheckpoint.error_count}.`);
  }
  report.completed_at = new Date().toISOString();
  writeReport(report);

  expect(report.issues, `User-audio issues: ${report.issues.join("; ")}`).toEqual([]);
  expect(finalCheckpoint.transcript_segments).toBeGreaterThan(0);
  expect(finalCheckpoint.latest_heard_chars).toBeGreaterThan(8);
  if (expectReferenceContext) {
    expect(finalCheckpoint.reference_context_cards).toBeGreaterThan(0);
    expect(finalCheckpoint.reference_context_text_chars).toBeGreaterThan(80);
  }
});

async function waitForTranscriptProgress(page: Page, previousCount: number, timeoutMs: number): Promise<void> {
  await page.waitForFunction(
    ({ count }) => {
      return document.querySelectorAll("#transcript .transcript-bubble").length > count;
    },
    { count: previousCount },
    { timeout: timeoutMs },
  );
}

async function captureCheckpoint(page: Page, report: Report, label: string): Promise<Checkpoint> {
  const elapsedSeconds = secondsSince(report.started_at);
  const safeLabel = `${label}-${String(Math.round(elapsedSeconds)).padStart(5, "0")}s`;
  const screenshotPath = path.join(artifactDir, `${safeLabel}.png`);
  const checkpoint: Checkpoint = {
    at_s: Math.round(elapsedSeconds),
    screenshot: screenshotPath,
    screenshots: await captureScreenshotSet(page, safeLabel, screenshotPath),
    status: await runtimeStatus(page),
    transcript_segments: await readMetric(page, "Transcript"),
    note_cards: await readMetric(page, "Notes"),
    reference_context_cards: await referenceContextCardCount(page),
    queue_depth: await readMetric(page, "Queue"),
    dropped_windows: await droppedWindows(page),
    reference_context_text_chars: await referenceContextTextCharCount(page),
    error_count: await page.getByRole("alert").count(),
    latest_heard_chars: await latestHeardCharCount(page),
    meeting_output_cards: await meetingOutputCardCount(page),
    transcript_text_chars: (await combinedTranscriptText(page)).trim().length,
    meeting_output_text_chars: (await currentMeetingOutputText(page)).trim().length,
    energy_lens_visible: await energyLensBadgeVisible(page),
    energy_lens_text_chars: (await visibleEnergyLensText(page)).trim().length,
  };
  report.checkpoints.push(checkpoint);
  writeReport(report);
  return checkpoint;
}

async function captureScreenshotSet(page: Page, safeLabel: string, topPath: string): Promise<string[]> {
  await page.evaluate(() => window.scrollTo(0, 0));
  await captureWidePage(page, topPath);
  const transcriptPath = path.join(artifactDir, `${safeLabel}-transcript.png`);
  await page.locator("#transcript").scrollIntoViewIfNeeded();
  await page.locator("#transcript").screenshot({ path: transcriptPath });
  const meetingOutputPath = path.join(artifactDir, `${safeLabel}-meeting-output.png`);
  const meetingOutput = page.locator("#recall");
  if ((await meetingOutput.count()) > 0 && await meetingOutput.isVisible()) {
    await meetingOutput.scrollIntoViewIfNeeded();
    await meetingOutput.screenshot({ path: meetingOutputPath });
  } else {
    await page.screenshot({ path: meetingOutputPath, fullPage: false });
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  return [topPath, transcriptPath, meetingOutputPath];
}

async function captureWidePage(page: Page, screenshotPath: string): Promise<void> {
  try {
    await page.screenshot({ path: screenshotPath, fullPage: true });
  } catch {
    await page.screenshot({ path: screenshotPath, fullPage: false });
  }
}

async function runtimeStatus(page: Page): Promise<string> {
  const text = await page.getByLabel("Runtime status").innerText();
  return text.replace(/\s+/g, " ").trim();
}

async function readMetric(page: Page, label: string): Promise<number> {
  const legacyMetric = page.getByLabel(`${label} metric`);
  if (await legacyMetric.count()) {
    const text = await legacyMetric.locator("strong").innerText();
    return Number(text.trim()) || 0;
  }

  if (label === "Transcript") {
    return numberFromText(await page.locator(".live-field-toolbar .field-toolbar-status").innerText(), /(\d+)\s+heard/i);
  }
  if (label === "Notes") {
    return numberFromText(await page.locator(".context-toolbar .field-toolbar-status").innerText(), /(\d+)\s+current/i);
  }
  if (label === "Queue") {
    const queue = page.getByLabel("Capture queue status");
    if (await queue.count()) {
      return numberFromText(await queue.innerText(), /queue\s+(\d+)/i);
    }
  }
  return 0;
}

async function droppedWindows(page: Page): Promise<number> {
  const signal = page.locator(".signal-ring small");
  if ((await signal.count()) === 0) {
    return 0;
  }
  const text = await signal.innerText();
  return Number(text.match(/\d+/)?.[0] ?? "0");
}

async function latestHeardCharCount(page: Page): Promise<number> {
  const latestBubble = page.locator("#transcript .transcript-bubble p").last();
  if ((await latestBubble.count()) === 0) {
    return 0;
  }
  const text = await latestBubble.innerText();
  return text.trim().length;
}

async function referenceContextCardCount(page: Page): Promise<number> {
  return page.locator("#recall .reference-context-output .context-card").count();
}

async function referenceContextTextCharCount(page: Page): Promise<number> {
  const referenceContext = page.locator("#recall .reference-context-output");
  if ((await referenceContext.count()) === 0) {
    return 0;
  }
  const text = await referenceContext.evaluate((element) => element.textContent ?? "");
  return text.trim().length;
}

async function expandMeetingOutputDetails(page: Page): Promise<void> {
  const detailButtons = page.locator("#recall .meeting-output-section button.link-button");
  const count = await detailButtons.count();
  for (let index = 0; index < count; index += 1) {
    const button = detailButtons.nth(index);
    if (!(await button.isVisible())) {
      continue;
    }
    const label = (await button.innerText()).trim();
    if (label === "Evidence" || label === "Details") {
      await button.click();
    }
  }
}

async function combinedTranscriptText(page: Page): Promise<string> {
  return page.locator("#transcript .transcript-bubble p").evaluateAll((elements) =>
    elements.map((element) => element.textContent ?? "").join("\n"),
  );
}

async function currentMeetingOutputText(page: Page): Promise<string> {
  return page.locator("#recall .meeting-output-section").evaluateAll((elements) =>
    elements.map((element) => element.textContent ?? "").join("\n"),
  );
}

async function currentMeetingCardText(page: Page): Promise<string> {
  return page.locator("#recall .meeting-output-section .context-card").evaluateAll((elements) =>
    elements.map((element) => element.textContent ?? "").join("\n"),
  );
}

async function meetingOutputCardCount(page: Page): Promise<number> {
  return page.locator("#recall .meeting-output-section .context-card").count();
}

async function energyLensBadgeVisible(page: Page): Promise<boolean> {
  const badge = page.locator('[aria-label="Energy lens"]').first();
  return (await badge.count()) > 0 && await badge.isVisible();
}

async function visibleEnergyLensText(page: Page): Promise<string> {
  const badge = page.locator('[aria-label="Energy lens"]').first();
  if ((await badge.count()) === 0 || !(await badge.isVisible())) {
    return "";
  }
  return badge.innerText();
}

async function combinedResponseText(page: Page): Promise<string> {
  return page.locator("#transcript, #recall").evaluateAll((elements) =>
    elements.map((element) => element.textContent ?? "").join("\n"),
  );
}

function parseExpectedTerms(value: string): string[] {
  return value
    .split(",")
    .map((term) => term.trim())
    .filter(Boolean);
}

function expectedMinHits(value: string | undefined, terms: string[]): number {
  const parsed = Number(value ?? terms.length);
  const minHits = Number.isFinite(parsed) ? parsed : terms.length;
  return Math.min(terms.length, Math.max(0, minHits));
}

function parseExpectedBoolean(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes((value ?? "").trim().toLowerCase());
}

function matchExpectedTerms(text: string, terms: string[]): string[] {
  const normalizedText = normalizeForMatch(text);
  return terms.filter((term) => normalizedText.includes(normalizeForMatch(term)));
}

function normalizeForMatch(text: string): string {
  return text
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/\bp\s*g\s+(?:and\s+)?e\b/g, "pg e")
    .replace(/\bpg\s+and\s+e\b/g, "pg e")
    .replace(/\bpge\b/g, "pg e")
    .replace(/\bt\.?\s*a\.?\s+smith\b/g, "ta smith")
    .replace(/\bfive\s+hundred\s+(?:k\s*v|kilovolt|kv)\b/g, "500kv")
    .replace(/\b500\s*k\s*v\b/g, "500kv")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function numberFromText(text: string, pattern: RegExp): number {
  return Number(text.match(pattern)?.[1] ?? "0") || 0;
}

function secondsSince(isoDate: string): number {
  return (Date.now() - Date.parse(isoDate)) / 1000;
}

function withoutScreenshot(checkpoint: Checkpoint): Omit<Checkpoint, "screenshot" | "screenshots"> {
  const { screenshot: _screenshot, screenshots: _screenshots, ...rest } = checkpoint;
  return rest;
}

function writeReport(report: Report): void {
  fs.mkdirSync(artifactDir, { recursive: true });
  fs.writeFileSync(path.join(artifactDir, "report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf-8");
}
