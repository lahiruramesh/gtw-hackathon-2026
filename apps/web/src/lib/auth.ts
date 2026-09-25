import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { admin, jwt } from "better-auth/plugins";
import { adminAc, userAc } from "better-auth/plugins/admin/access";
import { Pool } from "pg";

import { serverEnv } from "@/lib/env";
import { MIN_PASSWORD_LENGTH } from "@/lib/password-policy";
import { DEFAULT_ROLE, ROLES, type RoleId } from "@/lib/permissions";

const DAY_SECONDS = 60 * 60 * 24;

// Only `admin` may use the Better Auth admin API; every other app role gets user statements.
const authRoles = Object.fromEntries(ROLES.map((role) => [role.id, role.id === "admin" ? adminAc : userAc])) as Record<
  RoleId,
  typeof adminAc | typeof userAc
>;

function createAuth() {
  const env = serverEnv();
  return betterAuth({
    appName: "SKF Skill Studio",
    baseURL: env.BETTER_AUTH_URL,
    secret: env.BETTER_AUTH_SECRET,
    database: new Pool({
      connectionString: env.DATABASE_URL,
      options: "-c search_path=auth",
    }),
    emailAndPassword: {
      enabled: true,
      disableSignUp: true,
      minPasswordLength: MIN_PASSWORD_LENGTH,
    },
    session: {
      expiresIn: 7 * DAY_SECONDS,
      updateAge: DAY_SECONDS,
    },
    rateLimit: {
      enabled: true,
      storage: "database",
    },
    plugins: [
      admin({
        defaultRole: DEFAULT_ROLE,
        adminRoles: ["admin"],
        roles: authRoles,
      }),
      jwt({
        jwks: { keyPairConfig: { alg: "EdDSA", crv: "Ed25519" } },
        jwt: {
          issuer: env.AUTH_ISSUER,
          audience: env.AUTH_AUDIENCE,
          expirationTime: "15m",
          definePayload: ({ user }) => ({ email: user.email, name: user.name, role: user.role }),
        },
        // Tokens are minted server-side for the API only; never hand them to the browser.
        disableSettingJwtHeader: true,
      }),
      nextCookies(),
    ],
  });
}

export type Auth = ReturnType<typeof createAuth>;

const globalForAuth = globalThis as unknown as { skfAuth?: Auth };

/** The Better Auth instance, created on first use (keeps one pg pool across dev reloads). */
export function getAuth(): Auth {
  globalForAuth.skfAuth ??= createAuth();
  return globalForAuth.skfAuth;
}
