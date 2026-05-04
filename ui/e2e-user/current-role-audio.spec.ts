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
const expectWorkMemory = process.env.BRAIN_SIDECAR_EXPECT_WORK_MEMORY === "1";
const expectedResponseTerms = (process.env.BRAIN_SIDECAR_EXPECT_RESPONSE_TERMS ?? "")
  .split(",")
  .map((term) => term.trim())
  .filter(Boolean);
const expectedResponseTermMinHitsRaw = Number(
  process.env.BRAIN_SIDECAR_EXPECT_TERM_MIN_HITS ?? expectedResponseTerms.length,
);
const expectedResponseTermMinHits = Math.min(
  expectedResponseTerms.length,
  Number.isFinite(expectedResponseTermMinHitsRaw) ? expectedResponseTermMinHitsRaw : expectedResponseTerms.length,
);

type Checkpoint = {
  at_s: number;
  screenshot: string;
  screenshots: string[];
  status: string;
  transcript_segments: number;
  note_cards: number;
  recall_hits: number;
  queue_depth: number;
  dropped_windows: number;
  work_parallel_rows: number;
  memory_text_chars: number;
  error_count: number;
  latest_heard_chars: number;
};

type Report = {
  source_fixture: string;
  api_base: string;
  duration_seconds: number;
  screenshot_interval_ms: number;
  started_at: string;
  completed_at?: string;
  health?: Record<string, unknown>;
  work_memory_index?: Record<string, unknown>;
  checkpoints: Checkpoint[];
  final?: Omit<Checkpoint, "screenshot" | "screenshots">;
  expected_response_terms?: string[];
  matched_response_terms?: string[];
  response_text_chars?: number;
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

  const indexResponse = await request.post(`${apiBase}/api/work-memory/reindex`, {
    data: { embed: false },
    timeout: 120_000,
  });
  const indexPayload = await indexResponse.json();
  expect(indexResponse.ok(), `Work memory reindex failed: ${JSON.stringify(indexPayload)}`).toBeTruthy();
  report.work_memory_index = indexPayload;
  writeReport(report);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Live transcript" })).toBeVisible();
  await page.getByRole("button", { name: "Tools" }).click();
  const drawer = page.getByLabel("Tools drawer");
  await expect(drawer.getByRole("region", { name: "Fixture audio test controls" })).toBeVisible();
  await drawer.getByLabel("Fixture WAV path").fill(fixtureWav);
  await captureCheckpoint(page, report, "00-start");

  await drawer.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/, { timeout: 30_000 });
  await expect(page.getByLabel("Runtime status")).toContainText("listening", { timeout: 90_000 });
  await drawer.getByRole("button", { name: "Close" }).click();

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

  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText("stopped", { timeout: 30_000 });
  const finalCheckpoint = await captureCheckpoint(page, report, `${String(checkpointIndex).padStart(2, "0")}-final`);
  report.final = withoutScreenshot(finalCheckpoint);
  const responseText = await combinedResponseText(page);
  report.response_text_chars = responseText.trim().length;
  report.matched_response_terms = matchExpectedTerms(responseText, expectedResponseTerms);
  if (report.matched_response_terms.length < expectedResponseTermMinHits) {
    report.issues.push(
      `Expected at least ${expectedResponseTermMinHits} response terms, matched ${report.matched_response_terms.length}.`,
    );
  }
  report.completed_at = new Date().toISOString();
  writeReport(report);

  expect(report.issues, `User-audio issues: ${report.issues.join("; ")}`).toEqual([]);
  expect(finalCheckpoint.transcript_segments).toBeGreaterThan(0);
  expect(finalCheckpoint.latest_heard_chars).toBeGreaterThan(8);
  if (expectWorkMemory) {
    expect(finalCheckpoint.work_parallel_rows).toBeGreaterThan(0);
    expect(finalCheckpoint.memory_text_chars).toBeGreaterThan(80);
  }
});

async function waitForTranscriptProgress(page: Page, previousCount: number, timeoutMs: number): Promise<void> {
  await page.waitForFunction(
    ({ label, count }) => {
      const metric = [...document.querySelectorAll<HTMLElement>(".metric-card")]
        .find((element) => element.getAttribute("aria-label") === `${label} metric`);
      const value = Number(metric?.querySelector("strong")?.textContent ?? "0");
      return value > count;
    },
    { label: "Transcript", count: previousCount },
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
    recall_hits: await readMetric(page, "Recall"),
    queue_depth: await readMetric(page, "Queue"),
    dropped_windows: await droppedWindows(page),
    work_parallel_rows: await page.locator(".field-bubble.memory").count(),
    memory_text_chars: await memoryTextCharCount(page),
    error_count: await page.getByRole("alert").count(),
    latest_heard_chars: await latestHeardCharCount(page),
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
  const memoryPath = path.join(artifactDir, `${safeLabel}-memory.png`);
  const recall = page.locator("#recall");
  if ((await recall.count()) > 0 && await recall.isVisible()) {
    await recall.scrollIntoViewIfNeeded();
    await recall.screenshot({ path: memoryPath });
  } else {
    await page.screenshot({ path: memoryPath, fullPage: false });
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  return [topPath, transcriptPath, memoryPath];
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
  const text = await page.getByLabel(`${label} metric`).locator("strong").innerText();
  return Number(text.trim()) || 0;
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
  const text = await page.getByLabel("Latest heard").locator("blockquote").innerText();
  return text.trim().length;
}

async function memoryTextCharCount(page: Page): Promise<number> {
  const recall = page.locator("#recall");
  if ((await recall.count()) === 0 || !(await recall.isVisible())) {
    return 0;
  }
  const text = await recall.innerText();
  return text.trim().length;
}

async function combinedResponseText(page: Page): Promise<string> {
  return page.locator("#transcript, #recall").evaluateAll((elements) =>
    elements.map((element) => element.textContent ?? "").join("\n"),
  );
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
