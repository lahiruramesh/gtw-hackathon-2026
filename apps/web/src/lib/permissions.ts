import { z } from "zod";

import permissionsFile from "@shared/permissions.json";

/** Every permission string the UI gates on. Must match shared/permissions.json exactly. */
export const PERMISSIONS = [
  "skill:read",
  "skill:write",
  "run:read",
  "run:create_preset",
  "run:create_custom",
  "run:cancel_any",
  "run:approve_launch",
  "run:evaluate",
  "release:review",
  "compute:read",
  "compute:write",
  "audit:read",
  "user:manage",
] as const;

export type Permission = (typeof PERMISSIONS)[number];

const roleSchema = z.object({
  label: z.string(),
  description: z.string(),
  permissions: z.array(z.enum(PERMISSIONS)),
});

const fileSchema = z.object({
  roles: z.record(z.string(), roleSchema),
  default_role: z.string(),
});

// Parsed at import so a permission added to the shared file without UI support fails loudly.
const parsed = fileSchema.parse(permissionsFile);

export type RoleId = keyof typeof permissionsFile.roles;

export interface RoleDefinition {
  id: RoleId;
  label: string;
  description: string;
  permissions: readonly Permission[];
}

export const ROLES: readonly RoleDefinition[] = Object.entries(parsed.roles).map(([id, role]) => ({
  id: id as RoleId,
  ...role,
}));

export const ROLE_IDS = ROLES.map((role) => role.id) as [RoleId, ...RoleId[]];

export const DEFAULT_ROLE = parsed.default_role as RoleId;

export function isRoleId(value: unknown): value is RoleId {
  return typeof value === "string" && ROLE_IDS.includes(value as RoleId);
}

export function roleDefinition(role: string | null | undefined): RoleDefinition | undefined {
  return ROLES.find((definition) => definition.id === role);
}

export function roleLabel(role: string | null | undefined): string {
  return roleDefinition(role)?.label ?? role ?? "Unknown";
}

export function permissionsFor(role: string | null | undefined): ReadonlySet<Permission> {
  return new Set(roleDefinition(role)?.permissions ?? []);
}
