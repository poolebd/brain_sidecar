import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const uiDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(uiDir, "..");
const apiBase = process.env.PW_API_BASE ?? "http://127.0.0.1:8775";
const uiBase = process.env.PW_UI_BASE ?? "http://127.0.0.1:8776";
const apiUrl = new URL(apiBase);
const uiUrl = new URL(uiBase);
const apiPort = apiUrl.port || (apiUrl.protocol === "https:" ? "443" : "80");
const uiPort = uiUrl.port || (uiUrl.protocol === "https:" ? "443" : "80");
const timeoutMs = Number(process.env.BRAIN_SIDECAR_USER_TEST_TIMEOUT_MS ?? 2_700_000);
const backendStartup = [
  `cd ${shellQuote(repoRoot)}`,
  ". .venv/bin/activate",
  ". scripts/gpu-env.sh",
  "python -m brain_sidecar",
].join(" && ");

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

export default defineConfig({
  testDir: "./e2e-user",
  timeout: timeoutMs,
  expect: {
    timeout: 20_000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: uiBase,
    trace: "retain-on-failure",
    viewport: { width: 1920, height: 1200 },
    deviceScaleFactor: 1,
    launchOptions: {
      args: ["--disable-gpu", "--disable-dev-shm-usage"],
    },
  },
  webServer: [
    {
      command: `bash -lc ${shellQuote(backendStartup)}`,
      url: `${apiBase}/api/health/gpu`,
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        BRAIN_SIDECAR_HOST: apiUrl.hostname,
        BRAIN_SIDECAR_PORT: apiPort,
        BRAIN_SIDECAR_DATA_DIR:
          process.env.BRAIN_SIDECAR_DATA_DIR ?? path.join(repoRoot, "runtime", "user-tests", "apr-2-1200", "data"),
        BRAIN_SIDECAR_ASR_PRIMARY_MODEL: process.env.BRAIN_SIDECAR_ASR_PRIMARY_MODEL ?? "medium.en",
        BRAIN_SIDECAR_ASR_FALLBACK_MODEL: process.env.BRAIN_SIDECAR_ASR_FALLBACK_MODEL ?? "small.en",
        BRAIN_SIDECAR_ASR_BEAM_SIZE: process.env.BRAIN_SIDECAR_ASR_BEAM_SIZE ?? "5",
        BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS: process.env.BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS ?? "3",
        BRAIN_SIDECAR_TEST_MODE_ENABLED: process.env.BRAIN_SIDECAR_TEST_MODE_ENABLED ?? "1",
      },
    },
    {
      command: `npm run dev -- --host ${uiUrl.hostname} --port ${uiPort} --strictPort`,
      url: uiBase,
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        VITE_API_BASE: apiBase,
        VITE_ENABLE_FIXTURE_AUDIO: "1",
        VITE_DEFAULT_FIXTURE_AUDIO: process.env.BRAIN_SIDECAR_E2E_FIXTURE_WAV ?? "",
      },
    },
  ],
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1920, height: 1200 },
        deviceScaleFactor: 1,
        launchOptions: {
          args: ["--disable-gpu", "--disable-dev-shm-usage"],
        },
      },
    },
  ],
});
