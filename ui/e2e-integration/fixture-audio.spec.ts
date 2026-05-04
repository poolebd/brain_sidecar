import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const apiBase = process.env.PW_API_BASE ?? "http://127.0.0.1:8765";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixtureWav = process.env.BRAIN_SIDECAR_E2E_FIXTURE_WAV
  ?? path.join(repoRoot, "runtime", "test-audio", "osr-us-female-harvard.wav");

type SidecarEvent = {
  id: string;
  type: string;
  session_id: string | null;
  at: number;
  payload: Record<string, unknown>;
};

test("serves the real UI and backend health endpoint", async ({ page, request }) => {
  const health = await request.get(`${apiBase}/api/health/gpu`);
  expect(health.ok()).toBeTruthy();

  const payload = await health.json();
  expect(payload.required).toBe(true);
  expect(payload.nvidia_available).toBe(true);
  expect(payload.asr_cuda_available, payload.asr_cuda_error ?? "ASR CUDA should be available").toBe(true);
  expect(payload.asr_primary_model).toBeTruthy();
  expect(payload.asr_fallback_model).toBeTruthy();
  expect(payload.transcription_queue_size).toBeGreaterThan(0);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Live transcript" })).toBeVisible();
  await expect(page.getByRole("search")).toBeVisible();
});

test("streams fixture audio through the real API and SSE lane", async ({ page, request }) => {
  const sessionId = await createSession(request);

  await page.goto("/");

  try {
    const start = await request.post(`${apiBase}/api/sessions/${sessionId}/start`, {
      data: { fixture_wav: fixtureWav },
    });
    const startPayload = await start.json();
    expect(start.ok(), `Fixture start failed: ${JSON.stringify(startPayload)}`).toBeTruthy();
    expect(startPayload).toMatchObject({ status: "running", session_id: sessionId });

    await openBrowserSse(page, sessionId);

    const transcript = await waitForBrowserSseEvent(page, "transcript_final", 150_000);
    expect(transcript.session_id).toBe(sessionId);
    expect(transcript.payload.id).toBeTruthy();
    expect(transcript.payload.asr_model).toBe("tiny.en");
    expect(Number(transcript.payload.start_s)).toBeGreaterThanOrEqual(0);
    expect(Number(transcript.payload.end_s)).toBeGreaterThan(Number(transcript.payload.start_s));
    expect(String(transcript.payload.text).trim().length).toBeGreaterThanOrEqual(8);
  } finally {
    await page.evaluate(() => window.__brainSidecarIntegrationCloseSse?.());
    await request.post(`${apiBase}/api/sessions/${sessionId}/stop`);
  }
});

test("rejects a missing fixture WAV path", async ({ request }) => {
  const sessionId = await createSession(request);
  const missingFixture = path.join(repoRoot, "runtime", "test-audio", "does-not-exist.wav");

  const start = await request.post(`${apiBase}/api/sessions/${sessionId}/start`, {
    data: { fixture_wav: missingFixture },
  });

  expect(start.status()).toBe(400);
  await expect(start.json()).resolves.toMatchObject({
    detail: expect.stringContaining("Fixture WAV does not exist"),
  });
});

async function createSession(request: APIRequestContext): Promise<string> {
  const response = await request.post(`${apiBase}/api/sessions`, {
    data: { title: "Playwright fixture-audio integration" },
  });
  expect(response.ok()).toBeTruthy();

  const payload = await response.json();
  expect(payload.id).toBeTruthy();
  return String(payload.id);
}

async function openBrowserSse(page: Page, sessionId: string): Promise<void> {
  await page.evaluate(
    ({ apiBaseUrl, id }) => new Promise<void>((resolve) => {
      const events: SidecarEvent[] = [];
      const source = new EventSource(`${apiBaseUrl}/api/sessions/${id}/events`);
      let settled = false;

      const markReady = () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      };

      source.onopen = markReady;
      source.onerror = () => {
        window.__brainSidecarIntegrationLastSseError = `SSE reconnect for session ${id}`;
        window.setTimeout(markReady, 100);
      };
      window.setTimeout(markReady, 500);

      for (const type of ["audio_status", "transcript_final", "error"]) {
        source.addEventListener(type, (event) => {
          events.push(JSON.parse((event as MessageEvent).data) as SidecarEvent);
        });
      }

      window.__brainSidecarIntegrationEvents = events;
      window.__brainSidecarIntegrationCloseSse = () => source.close();
    }),
    { apiBaseUrl: apiBase, id: sessionId },
  );
}

async function waitForBrowserSseEvent(
  page: Page,
  type: string,
  timeoutMs: number,
  payloadStatus?: string,
): Promise<SidecarEvent> {
  await page.waitForFunction(
    ({ eventType, status }) => window.__brainSidecarIntegrationEvents?.some((event) => (
      event.type === eventType && (!status || event.payload.status === status)
    )),
    { eventType: type, status: payloadStatus ?? "" },
    { timeout: timeoutMs },
  );

  return page.evaluate(({ eventType, status }) => {
    const event = window.__brainSidecarIntegrationEvents?.find((candidate) => (
      candidate.type === eventType && (!status || candidate.payload.status === status)
    ));
    if (!event) {
      throw new Error(`No ${eventType} event was captured.`);
    }
    return event;
  }, { eventType: type, status: payloadStatus ?? "" });
}

declare global {
  interface Window {
    __brainSidecarIntegrationEvents?: SidecarEvent[];
    __brainSidecarIntegrationCloseSse?: () => void;
    __brainSidecarIntegrationLastSseError?: string;
  }
}
