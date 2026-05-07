import { mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { emitSessionEvent, installMockEventSource, mockApi } from "./mocks";

const screenshotDir = path.resolve(process.cwd(), "../docs/screenshots/meeting-cockpit");
const captureScreenshots = process.env.BRAIN_SIDECAR_CAPTURE_SCREENSHOTS === "1";

test("captures meeting cockpit page screenshots", async ({ page }) => {
  if (!captureScreenshots) {
    test.info().annotations.push({
      type: "screenshot regeneration",
      description: "Set BRAIN_SIDECAR_CAPTURE_SCREENSHOTS=1 to update committed screenshots.",
    });
    expect(captureScreenshots).toBe(false);
    return;
  }

  mkdirSync(screenshotDir, { recursive: true });
  await page.setViewportSize({ width: 1440, height: 900 });
  await installMockEventSource(page);
  await mockApi(page, { testModeEnabled: true });
  await page.goto("/");

  await page.getByRole("button", { name: "Start Listening" }).click();
  await expect(page.getByLabel("Runtime status")).toContainText(/starting|listening/);
  await emitSessionEvent(page, {
    type: "transcript_final",
    payload: {
      id: "shot-line-1",
      text: "We need a clean meeting cockpit with the live transcript paired to useful sidecar context.",
      start_s: 0,
      end_s: 4,
    },
  });
  await emitSessionEvent(page, {
    type: "sidecar_card",
    payload: {
      id: "shot-card-1",
      session_id: "session-123",
      category: "action",
      title: "Keep the cockpit readable",
      body: "Use paired lanes so transcript and sidecar context stay readable during the meeting.",
      why_now: "The live discussion called out the cockpit layout.",
      priority: "high",
      confidence: 0.92,
      source_segment_ids: ["shot-line-1"],
      source_type: "transcript",
      evidence_quote: "live transcript paired to useful sidecar context",
      raw_audio_retained: false,
    },
  });
  await page.screenshot({ path: path.join(screenshotDir, "live.png"), fullPage: true, animations: "disabled" });

  await page.getByRole("button", { name: "Sessions" }).click();
  await page.getByRole("button", { name: /Saved model planning call/ }).click();
  await expect(page.getByRole("main", { name: "Sessions" })).toContainText("Same-GPU default");
  await page.screenshot({ path: path.join(screenshotDir, "sessions.png"), fullPage: true, animations: "disabled" });

  await page.getByRole("button", { name: "Tools" }).click();
  await expect(page.getByRole("main", { name: "Tools" })).toContainText("Voice & Input");
  await page.screenshot({ path: path.join(screenshotDir, "tools.png"), fullPage: true, animations: "disabled" });
  await page.screenshot({ path: path.join(screenshotDir, "tools-voice-input.png"), fullPage: true, animations: "disabled" });
  await page.getByRole("navigation", { name: "Tool sections" }).getByRole("button", { name: "Speaker Identity" }).click();
  await expect(page.getByLabel("BP label readiness")).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, "tools-speaker-identity.png"), fullPage: true, animations: "disabled" });
  await page.getByRole("navigation", { name: "Tool sections" }).getByRole("button", { name: "Test Mode" }).click();
  await expect(page.getByRole("region", { name: "Recorded audio test" })).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, "tools-test-mode.png"), fullPage: true, animations: "disabled" });
  await page.getByRole("navigation", { name: "Tool sections" }).getByRole("button", { name: "Logs & Debug" }).click();
  await expect(page.getByRole("region", { name: "Logs & Debug" })).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, "tools-logs-debug.png"), fullPage: true, animations: "disabled" });
  await page.getByRole("navigation", { name: "Tool sections" }).getByRole("button", { name: "Appearance" }).click();
  await expect(page.getByRole("region", { name: "Appearance" })).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, "tools-appearance.png"), fullPage: true, animations: "disabled" });

  await page.getByRole("button", { name: "Models" }).click();
  await expect(page.getByRole("main", { name: "Models" })).toContainText("Dross runtime ready");
  await page.screenshot({ path: path.join(screenshotDir, "models.png"), fullPage: true, animations: "disabled" });

  await page.getByRole("button", { name: "Memory" }).click();
  await page.getByRole("navigation", { name: "Memory category navigation" }).getByRole("button", { name: /Documents/ }).click();
  await page.getByRole("button", { name: /OGMS spec.txt/ }).click();
  await expect(page.getByRole("main", { name: "Memory" })).toContainText("Monitoring signals are useful when they map to decisions.");
  await page.screenshot({ path: path.join(screenshotDir, "memory.png"), fullPage: true, animations: "disabled" });
});
