import { expect, test, type Page } from "@playwright/test";

import { storageStateFor } from "./support";

/**
 * Launches the step-length smoke preset on Local CPU and watches its training log stream in over SSE.
 * Needs the worker running and a pipeline interpreter (PIPELINE_PYTHON); the run is cancelled at the end so
 * the suite does not leave a CPU job behind.
 */
test.use({ storageState: storageStateFor("ml_engineer") });

async function launchSmokeRun(page: Page, name: string): Promise<void> {
  await page.goto("/runs/new?skill=g1-step-length&preset=smoke");
  await page.getByRole("radio", { name: /Smoke test/ }).check();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("radio", { name: /Local CPU/ }).check();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByLabel(/Run name/).fill(name);
  await page.getByRole("button", { name: "Launch run" }).click();
  await expect(page).toHaveURL(/\/runs\/[0-9a-f-]{36}/);
  await expect(page.getByRole("heading", { level: 1, name })).toBeVisible();
}

test("an ML engineer launches the smoke preset on Local CPU and sees live logs", async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const name = `e2e-smoke-${Date.now().toString(36)}`;
  await launchSmokeRun(page, name);

  // Lines arrive over the event stream while the page stays open: no reload anywhere below.
  const log = page.getByRole("log", { name: "Run logs" });
  await expect(log.locator("[data-index]").first()).toBeVisible({ timeout: 180_000 });
  const footer = page.getByText(/^[\d,]+ lines/);
  const shown = async () => Number((await footer.textContent())?.match(/^[\d,]+/)?.[0].replaceAll(",", ""));
  const first = await shown();
  await expect.poll(shown, { timeout: 120_000 }).toBeGreaterThan(first);
  // Follow stays on while lines stream in (row measurement must not switch it off).
  await expect(page.getByRole("switch", { name: "Follow" })).toBeChecked();
  await expect(page.getByText(/Train \(Brax PPO\)/).first()).toBeVisible();

  await page.getByRole("button", { name: "Cancel run" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Cancel run" }).click();
  await expect(page.getByText("Cancelled").first()).toBeVisible({ timeout: 60_000 });
});
