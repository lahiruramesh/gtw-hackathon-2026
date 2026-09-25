import { expect, test } from "@playwright/test";

import { proxyStatus, storageStateFor } from "./support";

test.use({ storageState: storageStateFor("viewer") });

test.describe("a viewer is read-only", () => {
  test("sees runs, gates and evaluations but no actions", async ({ page }) => {
    await page.goto("/runs");
    await expect(page.getByRole("heading", { level: 1, name: "Runs" })).toBeVisible();
    await expect(page.getByRole("link", { name: "New run" })).toHaveCount(0);

    await page.getByRole("link", { name: "g1-steplength-v1" }).click();
    await expect(page.getByRole("heading", { level: 1, name: "g1-steplength-v1" })).toBeVisible();
    for (const action of ["Cancel run", "Retry", "Approve", "Reject"]) {
      await expect(page.getByRole("button", { name: action })).toHaveCount(0);
    }
    await page.getByRole("tab", { name: "Gate" }).click();
    await expect(page.getByText("Step-length error < 3 cm (unseen engine)")).toBeVisible();
    await page.getByRole("tab", { name: "Checkpoints" }).click();
    await expect(page.getByRole("button", { name: /Evaluate|Warm start/ })).toHaveCount(0);
  });

  test("cannot launch, sync skills or change compute targets", async ({ page }) => {
    await page.goto("/runs/new");
    await expect(page.getByText("You don't have access")).toBeVisible();

    await page.goto("/skills");
    await expect(page.getByRole("button", { name: "Sync from repository" })).toHaveCount(0);

    await page.goto("/compute");
    await expect(page.getByRole("heading", { level: 1, name: "Compute" })).toBeVisible();
    for (const action of ["Add target", "Edit", "Set credentials", "Replace credentials", "Check health"]) {
      await expect(page.getByRole("button", { name: action })).toHaveCount(0);
    }
  });

  test("the API rejects every mutation with 403", async ({ page }) => {
    const targets = (await (await page.request.get("/api/backend/compute-targets")).json()) as { id: string }[];
    const runs = (await (await page.request.get("/api/backend/runs?limit=1")).json()) as { items: { id: string }[] };
    const targetId = targets[0]!.id;
    const runId = runs.items[0]!.id;
    const attempts: [string, string, unknown?][] = [
      ["POST", "/runs", { skill_id: "g1-step-length", preset_id: "smoke", params: {}, compute_target_id: targetId }],
      ["POST", `/runs/${runId}/cancel`],
      ["POST", `/runs/${runId}/review`, { decision: "approve" }],
      ["POST", `/runs/${runId}/launch-decision`, { decision: "approve" }],
      ["POST", "/skills/sync"],
      ["PATCH", `/compute-targets/${targetId}`, { description: "changed by a viewer" }],
      ["POST", `/compute-targets/${targetId}/check`],
      ["POST", "/audit-events", { action: "user.create", entity_type: "user", detail: {} }],
    ];
    for (const [method, path, body] of attempts) {
      expect(await proxyStatus(page, method, path, body), `${method} ${path}`).toBe(403);
    }
  });
});
