import { expect, test } from "@playwright/test";

const PAGES = [
  { path: "/", heading: "Dashboard" },
  { path: "/skills", heading: "Skills" },
  { path: "/runs", heading: "Runs" },
  { path: "/runs/new", heading: "New run" },
  { path: "/compare", heading: "Compare runs" },
  { path: "/approvals", heading: "Approvals" },
  { path: "/methods", heading: "Learning methods" },
  { path: "/compute", heading: "Compute" },
  { path: "/admin/users", heading: "Users" },
  { path: "/admin/audit", heading: "Audit log" },
];

for (const { path, heading } of PAGES) {
  test(`${path} renders for an admin`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    await expect(page.getByText("Something went wrong")).toHaveCount(0);
    // No horizontal page scroll at phone width.
    await page.setViewportSize({ width: 375, height: 812 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });
}

test("the proxy rejects paths that try to leave /api/v1", async ({ request }) => {
  const response = await request.get("/api/backend/runs%2F..%2Fadmin");
  expect(response.status()).toBe(400);
});
