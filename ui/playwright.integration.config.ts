import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const uiDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(uiDir, "..");
const apiBase = process.env.PW_API_BASE ?? "http://127.0.0.1:8765";
const uiBase = process.env.PW_UI_BASE ?? "http://127.0.0.1:8766";
const apiUrl = new URL(apiBase);
const uiUrl = new URL(uiBase);
const apiPort = apiUrl.port || (apiUrl.protocol === "https:" ? "443" : "80");
const uiPort = uiUrl.port || (uiUrl.protocol === "https:" ? "443" : "80");
const reuseServers = process.env.PW_REUSE_SERVERS === "1" && !process.env.CI;
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
  testDir: "./e2e-integration",
  timeout: 180_000,
  expect: {
    timeout: 15_000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: uiBase,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `bash -lc ${shellQuote(backendStartup)}`,
      url: `${apiBase}/api/health/gpu`,
      reuseExistingServer: reuseServers,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        BRAIN_SIDECAR_HOST: apiUrl.hostname,
        BRAIN_SIDECAR_PORT: apiPort,
        BRAIN_SIDECAR_DATA_DIR:
          process.env.BRAIN_SIDECAR_DATA_DIR ?? path.join(repoRoot, "runtime", "e2e-integration"),
        BRAIN_SIDECAR_ASR_PRIMARY_MODEL: process.env.BRAIN_SIDECAR_ASR_PRIMARY_MODEL ?? "tiny.en",
        BRAIN_SIDECAR_ASR_FALLBACK_MODEL: process.env.BRAIN_SIDECAR_ASR_FALLBACK_MODEL ?? "tiny.en",
        BRAIN_SIDECAR_ASR_BEAM_SIZE: process.env.BRAIN_SIDECAR_ASR_BEAM_SIZE ?? "1",
        BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS: process.env.BRAIN_SIDECAR_NOTES_EVERY_SEGMENTS ?? "999",
        BRAIN_SIDECAR_TEST_MODE_ENABLED: process.env.BRAIN_SIDECAR_TEST_MODE_ENABLED ?? "1",
      },
    },
    {
      command: `npm run dev -- --host ${uiUrl.hostname} --port ${uiPort} --strictPort`,
      url: uiBase,
      reuseExistingServer: reuseServers,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        VITE_API_BASE: apiBase,
      },
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
