import { expect, test } from "@playwright/test";

test.describe("signed out", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("redirects to the login page and keeps the destination", async ({ page }) => {
    await page.goto("/runs");
    await expect(page).toHaveURL(/\/login\?next=%2Fruns/);
    await expect(page.getByText("Ask an admin to create one")).toBeVisible();
  });

  test("a stale session cookie still returns the user to the page they asked for", async ({
    page,
    context,
    baseURL,
  }) => {
    await context.addCookies([{ name: "better-auth.session_token", value: "stale.value", url: baseURL! }]);
    await page.goto("/runs/new?skill=g1-step-length");
    await expect(page).toHaveURL(/\/login\?next=%2Fruns%2Fnew%3Fskill%3Dg1-step-length$/);
  });

  test("rejects a wrong password", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Email").fill(process.env.ADMIN_EMAIL ?? "nobody@skf.local");
    await page.getByLabel("Password").fill("definitely-not-the-password");
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByText(/Wrong email or password|Too many attempts/)).toBeVisible();
  });

  test("the API proxy refuses requests without a session", async ({ request }) => {
    const response = await request.get("/api/backend/runs");
    expect(response.status()).toBe(401);
  });
});
