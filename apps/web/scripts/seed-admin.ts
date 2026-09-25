/**
 * Creates the first admin from ADMIN_EMAIL / ADMIN_PASSWORD. Idempotent: an existing user with
 * that email is promoted to admin (if needed) and otherwise left alone.
 * Usage: pnpm seed:admin
 */
import { z } from "zod";

import { getAuth } from "@/lib/auth";
import { MIN_PASSWORD_LENGTH } from "@/lib/password-policy";

const seedEnv = z.object({
  ADMIN_EMAIL: z.email(),
  ADMIN_PASSWORD: z.string().min(MIN_PASSWORD_LENGTH, `must be at least ${MIN_PASSWORD_LENGTH} characters`),
  ADMIN_NAME: z.string().min(1).default("Administrator"),
});

async function main() {
  const env = seedEnv.parse(process.env);
  const email = env.ADMIN_EMAIL.toLowerCase();
  const auth = getAuth();
  const context = await auth.$context;

  const existing = await context.internalAdapter.findUserByEmail(email);
  if (existing) {
    // The internal adapter's user type does not include plugin fields such as `role`.
    const role = "role" in existing.user ? existing.user.role : undefined;
    if (role !== "admin") {
      await context.internalAdapter.updateUser(existing.user.id, { role: "admin" });
      console.log(`Promoted ${email} to admin.`);
    } else {
      console.log(`Admin ${email} already exists.`);
    }
    return;
  }

  // Server-side call without request headers: allowed even though public sign-up is disabled.
  await auth.api.createUser({
    body: { email, password: env.ADMIN_PASSWORD, name: env.ADMIN_NAME, role: "admin" },
  });
  console.log(`Created admin ${email}.`);
}

main()
  .catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  })
  .finally(() => process.exit());
