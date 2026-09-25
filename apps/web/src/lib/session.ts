import "server-only";

import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { cache } from "react";

import { getAuth } from "@/lib/auth";
import { permissionsFor, roleLabel, type Permission } from "@/lib/permissions";

/** The signed-in user as the UI sees them. Plain data so it can be passed to client components. */
export interface Viewer {
  id: string;
  name: string;
  email: string;
  role: string;
  roleLabel: string;
  permissions: Permission[];
}

export const getViewer = cache(async (): Promise<Viewer | null> => {
  // Reading the request first marks the route dynamic before the auth instance (and env) is touched.
  const requestHeaders = await headers();
  const session = await getAuth().api.getSession({ headers: requestHeaders });
  if (!session) return null;
  const role = session.user.role ?? "";
  return {
    id: session.user.id,
    name: session.user.name,
    email: session.user.email,
    role,
    roleLabel: roleLabel(role),
    permissions: [...permissionsFor(role)],
  };
});

export async function requireViewer(): Promise<Viewer> {
  const viewer = await getViewer();
  if (!viewer) redirect("/login");
  return viewer;
}
