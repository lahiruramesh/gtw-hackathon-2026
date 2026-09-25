/**
 * Creates or updates Better Auth's tables in the `auth` schema (the pg pool's search_path).
 * Usage: pnpm auth:migrate
 */
import { getAuth } from "@/lib/auth";

async function main() {
  const context = await getAuth().$context;
  await context.runMigrations();
  console.log("Better Auth schema is up to date.");
}

main()
  .catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  })
  .finally(() => process.exit());
