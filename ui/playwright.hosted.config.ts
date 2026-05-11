import { defineConfig, devices } from "@playwright/test";

const hostedUrl = process.env.BRAIN_SIDECAR_CADDY_HOSTED_URL ?? "https://notes.shoalstone.net/";
const basicUser = process.env.BRAIN_SIDECAR_CADDY_BASIC_USER ?? "portal";
const basicPassword = process.env.BRAIN_SIDECAR_CADDY_BASIC_PASSWORD ?? "";

export default defineConfig({
  testDir: "./e2e-hosted",
  timeout: 45_000,
  expect: {
    timeout: 8_000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: hostedUrl,
    httpCredentials: basicPassword
      ? {
          username: basicUser,
          password: basicPassword,
        }
      : undefined,
    ignoreHTTPSErrors: true,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
