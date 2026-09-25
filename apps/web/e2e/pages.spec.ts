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

test("?next= after sign-in never leaves the site", async ({ page, baseURL }) => {
  for (const next of ["/%09/evil.example", "//evil.example", "/%0a/evil.example", "https://evil.example"]) {
    const response = await page.goto(`/login?next=${next}`);
    expect(response?.status(), next).toBeLessThan(500);
    await expect(page).toHaveURL(`${baseURL}/`);
  }
});

test("the browser cannot mint API tokens", async ({ request }) => {
  expect((await request.get("/api/auth/token")).status()).toBe(404);
  expect((await request.get("/api/auth/jwks")).status()).toBe(200); // the API still verifies with it
});

test("the proxy refuses state-changing requests from other sites", async ({ request }) => {
  const response = await request.post("/api/backend/skills/sync", {
    headers: { origin: "https://evil.example", "sec-fetch-site": "cross-site", "content-type": "text/plain" },
  });
  expect(response.status()).toBe(403);
});

test("the proxy rejects paths that try to leave /api/v1", async ({ request }) => {
  const response = await request.get("/api/backend/runs%2F..%2Fadmin");
  expect(response.status()).toBe(400);
});
