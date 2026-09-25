import { expect, test } from "@playwright/test";

import { proxyStatus, storageStateFor, type TestRole } from "./support";

const NAV = [
  "Dashboard",
  "Skills",
  "Runs",
  "Compare",
  "Approvals",
  "Learning methods",
  "Compute",
  "Users",
  "Audit log",
];

/** Sidebar entries each role sees (shared/permissions.json); everything else must be hidden. */
const VISIBLE: Record<TestRole, readonly string[]> = {
  admin: NAV,
  ml_engineer: ["Dashboard", "Skills", "Runs", "Compare", "Approvals", "Learning methods", "Compute"],
  operator: ["Dashboard", "Skills", "Runs", "Compare", "Learning methods", "Compute"],
  safety_reviewer: ["Dashboard", "Skills", "Runs", "Compare", "Approvals", "Learning methods", "Compute"],
  viewer: ["Dashboard", "Skills", "Runs", "Compare", "Learning methods", "Compute"],
};

const CAN_LAUNCH: Record<TestRole, boolean> = {
  admin: true,
  ml_engineer: true,
  operator: true,
  safety_reviewer: false,
  viewer: false,
};

for (const role of Object.keys(VISIBLE) as TestRole[]) {
  test.describe(`as ${role}`, () => {
    test.use({ storageState: storageStateFor(role) });

    test("the sidebar shows only the pages the role can use", async ({ page }) => {
      await page.goto("/");
      const sidebar = page.locator("[data-sidebar=sidebar]");
      for (const label of NAV) {
        const link = sidebar.getByRole("link", { name: label, exact: true });
        if (VISIBLE[role].includes(label)) await expect(link).toBeVisible();
        else await expect(link).toHaveCount(0);
      }
      await expect(page.getByRole("link", { name: "New run" })).toHaveCount(CAN_LAUNCH[role] ? 1 : 0);
    });

    if (role === "admin") return;

    test("admin pages are closed", async ({ page }) => {
      for (const path of ["/admin/users", "/admin/audit"]) {
        await page.goto(path);
        await expect(page.getByText("You don't have access")).toBeVisible();
      }
      expect(await proxyStatus(page, "GET", "/audit-events")).toBe(403);
    });
  });
}

test.describe("as operator", () => {
  test.use({ storageState: storageStateFor("operator") });

  test("launches presets only: custom parameters are locked and rejected by the API", async ({ page }) => {
    await page.goto("/runs/new?skill=g1-step-length");
    await expect(page.getByRole("radio", { name: /Custom parameters/ })).toBeDisabled();
    await expect(page.getByText(/Operators launch approved presets/)).toBeVisible();

    const targets = (await (await page.request.get("/api/backend/compute-targets")).json()) as {
      id: string;
      kind: string;
    }[];
    const local = targets.find((target) => target.kind === "local_cpu")!;
    const custom = {
      skill_id: "g1-step-length",
      params: { smoke: true, timesteps: 40000 },
      compute_target_id: local.id,
    };
    expect(await proxyStatus(page, "POST", "/runs", custom)).toBe(403);
    expect(await proxyStatus(page, "POST", "/skills/sync")).toBe(403);
  });
});
