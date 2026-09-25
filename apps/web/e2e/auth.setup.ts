import { mkdirSync } from "node:fs";

import { test as setup } from "@playwright/test";

import { AUTH_DIR, signIn, storageStateFor } from "./support";

const email = process.env.ADMIN_EMAIL;
const password = process.env.ADMIN_PASSWORD;

setup("sign in as the seeded admin", async ({ page }) => {
  setup.skip(!email || !password, "ADMIN_EMAIL and ADMIN_PASSWORD are required");
  mkdirSync(AUTH_DIR, { recursive: true, mode: 0o700 });
  await signIn(page, email!, password!);
  await page.context().storageState({ path: storageStateFor("admin") });
});
