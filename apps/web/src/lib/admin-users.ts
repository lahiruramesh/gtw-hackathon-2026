import "server-only";

import { isAPIError } from "better-auth/api";
import { headers } from "next/headers";

import { toErrorInfo, type Result } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { AuditEventInput } from "@/lib/api/types";
import { getAuth } from "@/lib/auth";
import { getViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export interface ManagedUser {
  id: string;
  name: string;
  email: string;
  role: string;
  banned: boolean;
  banReason: string | null;
  createdAt: string;
}

export interface UserActionOutcome {
  /** False when the change succeeded but the audit event could not be written. */
  audited: boolean;
}

function authErrorResult(error: unknown): Result<never> {
  if (isAPIError(error)) {
    const body = error.body as { code?: string; message?: string } | undefined;
    return {
      ok: false,
      error: { status: error.statusCode, code: body?.code ?? "auth_error", message: body?.message ?? error.message },
    };
  }
  return { ok: false, error: toErrorInfo(error) };
}

const FORBIDDEN: Result<never> = {
  ok: false,
  error: { status: 403, code: "forbidden", message: "Only admins can manage users." },
};

/** Records a user-management action in the API's audit log (SPEC §9.2 POST /audit-events). */
async function recordAudit(event: AuditEventInput): Promise<boolean> {
  try {
    await apiFetch("/audit-events", { method: "POST", body: event });
    return true;
  } catch {
    return false;
  }
}

interface ManageUserOptions<T> {
  operation: (requestHeaders: Headers) => Promise<T>;
  /** Built from the operation's result. Must never contain passwords. */
  audit: (result: T) => AuditEventInput;
  /** Refuse when the admin targets their own account (e.g. demoting or banning themselves). */
  notSelf?: { userId: string; verb: string };
}

/** Runs one Better Auth admin operation as the signed-in admin, then audits it. */
export async function manageUser<T>({
  operation,
  audit,
  notSelf,
}: ManageUserOptions<T>): Promise<Result<UserActionOutcome>> {
  const viewer = await getViewer();
  if (!viewer || !can(viewer, "user:manage")) return FORBIDDEN;
  if (notSelf && notSelf.userId === viewer.id) {
    return {
      ok: false,
      error: { status: 409, code: "conflict", message: `You can't ${notSelf.verb} your own account.` },
    };
  }
  let result: T;
  try {
    result = await operation(await headers());
  } catch (error) {
    return authErrorResult(error);
  }
  return { ok: true, data: { audited: await recordAudit(audit(result)) } };
}

export async function listManagedUsers(): Promise<Result<ManagedUser[]>> {
  try {
    const { users } = await getAuth().api.listUsers({
      query: { limit: 500, sortBy: "createdAt", sortDirection: "desc" },
      headers: await headers(),
    });
    return {
      ok: true,
      data: users.map((user) => ({
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role ?? "",
        banned: user.banned === true,
        banReason: user.banReason ?? null,
        createdAt: new Date(user.createdAt).toISOString(),
      })),
    };
  } catch (error) {
    return authErrorResult(error);
  }
}
