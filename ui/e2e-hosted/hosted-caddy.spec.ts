import dns from "node:dns/promises";
import net from "node:net";
import { expect, test } from "@playwright/test";

const hostedUrl = process.env.BRAIN_SIDECAR_CADDY_HOSTED_URL ?? "https://notes.shoalstone.net/";
const expectedCaddyIp = process.env.BRAIN_SIDECAR_CADDY_EXPECTED_IP ?? "192.168.86.45";
const basicPassword = process.env.BRAIN_SIDECAR_CADDY_BASIC_PASSWORD ?? "";
const hasHostedCredentials = basicPassword.trim().length > 0;

test.describe("local Caddy hosted path", () => {
  test("preflight confirms notes.shoalstone.net is the local route", async () => {
    test.skip(!hasHostedCredentials, "Set BRAIN_SIDECAR_CADDY_BASIC_PASSWORD to run the hosted Caddy smoke.");

    const hostname = new URL(hostedUrl).hostname;
    const addresses = await dns.lookup(hostname, { all: true });
    expect(addresses.map((address) => address.address)).toContain(expectedCaddyIp);

    await expect(listenerReady(expectedCaddyIp, 443), `${expectedCaddyIp}:443 should be the local Caddy listener`).resolves.toBe(true);
    await expect(listenerReady("127.0.0.1", 8765), "FastAPI backend should be listening for Caddy").resolves.toBe(true);
    await expect(listenerReady("127.0.0.1", 8766), "Vite UI should be listening for Caddy").resolves.toBe(true);
  });

  test("renders the app and Review through local Caddy", async ({ page }) => {
    test.skip(!hasHostedCredentials, "Set BRAIN_SIDECAR_CADDY_BASIC_PASSWORD to run the hosted Caddy smoke.");

    const response = await page.goto("/", { waitUntil: "domcontentloaded" });
    expect(response, "Caddy should return a browser response").not.toBeNull();
    expect(response!.status(), "Basic auth failed; 401 is only the boundary, not a passed smoke").not.toBe(401);
    expect(response!.ok(), `Unexpected hosted response ${response!.status()}`).toBe(true);

    await expect(page.getByRole("main", { name: "Live" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Conversation" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Sidecar Intelligence" })).toBeVisible();

    await page.getByRole("button", { name: "Review" }).click();
    const review = page.getByRole("main", { name: "Review" });
    await expect(review).toBeVisible();
    await expect(review.getByLabel("Review audio upload")).toBeVisible();
    await expect(review.getByLabel("Review retention")).toContainText("Manual approval required");

    const health = await page.evaluate(async () => {
      const healthResponse = await fetch("/api/health/gpu");
      return {
        status: healthResponse.status,
        payload: await healthResponse.json(),
      };
    });
    expect(health.status).toBe(200);
    expect(health.payload).toMatchObject({
      nvidia_available: expect.any(Boolean),
      ollama_chat_reachable: expect.any(Boolean),
      energy_lens_enabled: expect.any(Boolean),
    });
  });
});

function listenerReady(host: string, port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host, port });
    const finish = (ready: boolean) => {
      socket.destroy();
      resolve(ready);
    };
    socket.setTimeout(1500);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
  });
}
