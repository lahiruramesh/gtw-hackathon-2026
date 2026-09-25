import type { Permission } from "@/lib/permissions";

/** Permission check usable on both server and client (UI gating only; the API enforces). */
export function can(viewer: { permissions: readonly Permission[] }, permission: Permission): boolean {
  return viewer.permissions.includes(permission);
}

export function canAny(viewer: { permissions: readonly Permission[] }, permissions: readonly Permission[]): boolean {
  return permissions.some((permission) => viewer.permissions.includes(permission));
}
