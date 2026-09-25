import { existsSync } from "node:fs";

import { defineConfig, devices } from "@playwright/test";

import { storageStateFor } from "./e2e/support";

// Local runs reuse the seeded admin from .env.local; CI passes ADMIN_EMAIL / ADMIN_PASSWORD directly.
if (existsSync(".env.local")) process.loadEnvFile(".env.local");

/**
 * End-to-end tests against a running stack (web on E2E_BASE_URL, API behind it).
 * Start them with `pnpm dev -p 3100` (and the API) first; nothing here launches servers.
 */
export default defineConfig({
  testDir: "e2e",
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3100",
    trace: "retain-on-failure",
  },
  // Specs run as the admin unless they `test.use({ storageState: storageStateFor(role) })`.
  projects: [
    { name: "admin-session", testMatch: /auth\.setup\.ts/ },
    { name: "role-accounts", testMatch: /users\.setup\.ts/, dependencies: ["admin-session"] },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], storageState: storageStateFor("admin") },
      dependencies: ["role-accounts"],
    },
  ],
});
