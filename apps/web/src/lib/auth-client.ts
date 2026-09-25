import { createAuthClient } from "better-auth/react";

/** Browser-side Better Auth client: sign in and sign out only (admin actions run on the server). */
export const authClient = createAuthClient();
