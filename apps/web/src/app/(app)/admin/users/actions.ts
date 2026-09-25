"use server";

import { refresh } from "next/cache";
import { z } from "zod";

import { manageUser, type UserActionOutcome } from "@/lib/admin-users";
import type { Result } from "@/lib/api/errors";
import { getAuth } from "@/lib/auth";
import { MIN_PASSWORD_LENGTH } from "@/lib/password-policy";
import { ROLE_IDS } from "@/lib/permissions";

const userId = z.string().min(1).max(200);
const password = z.string().min(MIN_PASSWORD_LENGTH, `Use at least ${MIN_PASSWORD_LENGTH} characters`).max(128);
const role = z.enum(ROLE_IDS);

const createUserSchema = z.object({
  name: z.string().trim().min(1, "Enter a name").max(100),
  email: z.email("Enter a valid email address"),
  password,
  role,
});

export type CreateUserInput = z.infer<typeof createUserSchema>;

function invalid(error: z.ZodError): Result<never> {
  return {
    ok: false,
    error: { status: 422, code: "validation_error", message: error.issues[0]?.message ?? "Invalid input" },
  };
}

async function done(result: Promise<Result<UserActionOutcome>>): Promise<Result<UserActionOutcome>> {
  const outcome = await result;
  if (outcome.ok) refresh();
  return outcome;
}

export async function createUser(input: CreateUserInput): Promise<Result<UserActionOutcome>> {
  const parsed = createUserSchema.safeParse(input);
  if (!parsed.success) return invalid(parsed.error);
  const { name, email, role: newRole } = parsed.data;
  return done(
    manageUser({
      operation: (headers) =>
        getAuth().api.createUser({ body: { ...parsed.data, email: email.toLowerCase() }, headers }),
      audit: ({ user }) => ({
        action: "user.create",
        entity_type: "user",
        entity_id: user.id,
        detail: { email, name, role: newRole },
      }),
    }),
  );
}

export async function setUserRole(id: string, newRole: string): Promise<Result<UserActionOutcome>> {
  const parsed = z.object({ id: userId, role }).safeParse({ id, role: newRole });
  if (!parsed.success) return invalid(parsed.error);
  return done(
    manageUser({
      operation: (headers) => getAuth().api.setRole({ body: { userId: id, role: parsed.data.role }, headers }),
      audit: () => ({
        action: "user.set_role",
        entity_type: "user",
        entity_id: id,
        detail: { role: parsed.data.role },
      }),
      notSelf: { userId: id, verb: "change the role of" },
    }),
  );
}

export async function banUser(id: string, reason: string): Promise<Result<UserActionOutcome>> {
  const parsed = z.object({ id: userId, reason: z.string().trim().max(500) }).safeParse({ id, reason });
  if (!parsed.success) return invalid(parsed.error);
  return done(
    manageUser({
      operation: (headers) =>
        getAuth().api.banUser({ body: { userId: id, banReason: parsed.data.reason || undefined }, headers }),
      audit: () => ({
        action: "user.ban",
        entity_type: "user",
        entity_id: id,
        detail: { reason: parsed.data.reason || null },
      }),
      notSelf: { userId: id, verb: "ban" },
    }),
  );
}

export async function unbanUser(id: string): Promise<Result<UserActionOutcome>> {
  const parsed = userId.safeParse(id);
  if (!parsed.success) return invalid(parsed.error);
  return done(
    manageUser({
      operation: (headers) => getAuth().api.unbanUser({ body: { userId: id }, headers }),
      audit: () => ({ action: "user.unban", entity_type: "user", entity_id: id, detail: {} }),
    }),
  );
}

export async function setUserPassword(id: string, newPassword: string): Promise<Result<UserActionOutcome>> {
  const parsed = z.object({ id: userId, password }).safeParse({ id, password: newPassword });
  if (!parsed.success) return invalid(parsed.error);
  return done(
    manageUser({
      operation: (headers) =>
        getAuth().api.setUserPassword({ body: { userId: id, newPassword: parsed.data.password }, headers }),
      audit: () => ({ action: "user.set_password", entity_type: "user", entity_id: id, detail: {} }),
    }),
  );
}

export async function revokeUserSessions(id: string): Promise<Result<UserActionOutcome>> {
  const parsed = userId.safeParse(id);
  if (!parsed.success) return invalid(parsed.error);
  return done(
    manageUser({
      operation: (headers) => getAuth().api.revokeUserSessions({ body: { userId: id }, headers }),
      audit: () => ({ action: "user.revoke_sessions", entity_type: "user", entity_id: id, detail: {} }),
      notSelf: { userId: id, verb: "sign out every session of" },
    }),
  );
}
