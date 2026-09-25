import { writeFileSync } from "node:fs";

import { expect, test as setup } from "@playwright/test";

import { newPassword, ROLE_USERS, signIn, storageStateFor, USERS_FILE } from "./support";

setup.use({ storageState: storageStateFor("admin") });

setup("admin creates one account per role on /admin/users", async ({ page, browser }) => {
  const credentials: Record<string, { email: string; password: string }> = {};
  await page.goto("/admin/users");
  await expect(page.getByRole("heading", { level: 1, name: "Users" })).toBeVisible();

  for (const user of ROLE_USERS) {
    const password = newPassword();
    const row = page.getByRole("row").filter({ hasText: user.email });
    if ((await row.count()) === 0) {
      await page.getByRole("button", { name: "Add user" }).click();
      const dialog = page.getByRole("dialog", { name: "Add user" });
      await dialog.getByLabel("Name").fill(user.name);
      await dialog.getByLabel("Email").fill(user.email);
      await dialog.getByLabel("Temporary password").fill(password);
      await dialog.getByRole("combobox").click();
      await page.getByRole("option", { name: user.label, exact: true }).click();
      await dialog.getByRole("button", { name: "Create user" }).click();
      await expect(page.getByText(`Created ${user.email}`)).toBeVisible();
    } else {
      // Accounts survive between runs; rotate the password so this run knows it.
      await row.getByRole("button", { name: `Actions for ${user.email}` }).click();
      await page.getByRole("menuitem", { name: "Set password" }).click();
      const dialog = page.getByRole("dialog", { name: "Set a new password" });
      await dialog.getByLabel("New password").fill(password);
      await dialog.getByRole("button", { name: "Set password" }).click();
      await expect(dialog).toBeHidden();
    }
    await expect(page.getByRole("row").filter({ hasText: user.email })).toContainText(user.label);
    credentials[user.role] = { email: user.email, password };
  }
  writeFileSync(USERS_FILE, JSON.stringify(credentials, null, 2), { mode: 0o600 });

  for (const user of ROLE_USERS) {
    const context = await browser.newContext({ storageState: { cookies: [], origins: [] } });
    const rolePage = await context.newPage();
    await signIn(rolePage, user.email, credentials[user.role]!.password);
    await context.storageState({ path: storageStateFor(user.role) });
    await context.close();
  }
});
