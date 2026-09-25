import { randomBytes } from "node:crypto";
import { readFileSync } from "node:fs";

import type { Page } from "@playwright/test";

/** One account per non-admin role, created through /admin/users by `users.setup.ts`. */
export const ROLE_USERS = [
  { role: "ml_engineer", label: "ML engineer", email: "e2e-ml-engineer@skf.local", name: "E2E ML engineer" },
  { role: "operator", label: "Operator", email: "e2e-operator@skf.local", name: "E2E Operator" },
  {
    role: "safety_reviewer",
    label: "Safety reviewer",
    email: "e2e-safety-reviewer@skf.local",
    name: "E2E Safety reviewer",
  },
  { role: "viewer", label: "Viewer", email: "e2e-viewer@skf.local", name: "E2E Viewer" },
] as const;

export type TestRole = "admin" | (typeof ROLE_USERS)[number]["role"];

export const AUTH_DIR = "e2e/.auth";
/** Passwords of the role accounts, rotated on every setup run. Gitignored, mode 0600. */
export const USERS_FILE = `${AUTH_DIR}/users.json`;

export function storageStateFor(role: TestRole): string {
  return `${AUTH_DIR}/${role}.json`;
}

export function newPassword(): string {
  return randomBytes(18).toString("base64url");
}

export function roleCredentials(role: Exclude<TestRole, "admin">): { email: string; password: string } {
  const users = JSON.parse(readFileSync(USERS_FILE, "utf8")) as Record<string, { email: string; password: string }>;
  const user = users[role];
  if (!user) throw new Error(`No e2e account for ${role}; run the setup project first`);
  return user;
}

/**
 * Signs in through the login form. Better Auth allows 3 sign-ins per 10 s per client, and the setup signs in
 * five accounts back to back, so a rate-limited attempt waits out the window and tries again.
 */
export async function signIn(page: Page, email: string, password: string): Promise<void> {
  for (let attempt = 0; attempt < 4; attempt++) {
    await page.goto("/login");
    await page.getByLabel("Email").fill(email);
    await page.getByLabel("Password").fill(password);
    await page.getByRole("button", { name: "Sign in" }).click();
    const outcome = await Promise.race([
      page
        .getByRole("heading", { level: 1, name: "Dashboard" })
        .waitFor()
        .then(() => "signed-in" as const),
      page
        .getByText("Too many attempts")
        .waitFor()
        .then(() => "rate-limited" as const),
    ]);
    if (outcome === "signed-in") return;
    await page.waitForTimeout(10_000);
  }
  throw new Error(`Could not sign in as ${email}`);
}

/** Status of the API call a same-origin proxy request produced (the browser session supplies the JWT). */
export async function proxyStatus(page: Page, method: string, path: string, body?: unknown): Promise<number> {
  const response = await page.request.fetch(`/api/backend${path}`, {
    method,
    data: body,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
  });
  return response.status();
}
