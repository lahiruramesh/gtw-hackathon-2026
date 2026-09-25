import { expect, test } from "@playwright/test";

test.describe("signed out", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("redirects to the login page and keeps the destination", async ({ page }) => {
    await page.goto("/runs");
    await expect(page).toHaveURL(/\/login\?next=%2Fruns/);
    await expect(page.getByText("Ask an admin for an account")).toBeVisible();
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
