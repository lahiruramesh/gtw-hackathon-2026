import { z } from "zod";

const serverEnvSchema = z.object({
  DATABASE_URL: z.string().startsWith("postgres", "must be a postgres:// or postgresql:// URL"),
  BETTER_AUTH_SECRET: z.string().min(32, "must be at least 32 characters"),
  BETTER_AUTH_URL: z.url(),
  API_INTERNAL_URL: z.url(),
  AUTH_ISSUER: z.string().min(1).default("skf-skill-studio"),
  AUTH_AUDIENCE: z.string().min(1).default("skf-api"),
});

export type ServerEnv = z.infer<typeof serverEnvSchema>;

let cached: ServerEnv | undefined;

/**
 * Validated server environment. Parsed lazily on first use so `next build` can run without
 * runtime secrets (the Docker image is built before they exist).
 */
export function serverEnv(): ServerEnv {
  if (cached) return cached;
  const parsed = serverEnvSchema.safeParse(process.env);
  if (!parsed.success) {
    const problems = parsed.error.issues.map((issue) => `  ${issue.path.join(".")}: ${issue.message}`);
    throw new Error(`Invalid environment:\n${problems.join("\n")}`);
  }
  cached = parsed.data;
  return cached;
}
