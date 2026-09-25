import { getSessionCookie } from "better-auth/cookies";
import { NextResponse, type NextRequest } from "next/server";

import { loginPath, REQUEST_PATH_HEADER } from "@/lib/login-path";

/**
 * Optimistic gate: sends visitors without a session cookie to /login. Real checks happen on the
 * server per page (`requireViewer`) and in the API; the path header lets those send a user whose
 * cookie turned out to be stale back to the page they asked for.
 */
export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  if (getSessionCookie(request)) {
    const headers = new Headers(request.headers);
    headers.set(REQUEST_PATH_HEADER, `${pathname}${search}`);
    return NextResponse.next({ request: { headers } });
  }

  if (pathname.startsWith("/api/")) {
    return Response.json(
      { error: { code: "unauthenticated", message: "Sign in to continue", details: {} } },
      { status: 401 },
    );
  }
  return NextResponse.redirect(new URL(loginPath(`${pathname}${search}`), request.url));
}

export const config = {
  matcher: ["/((?!login|api/auth|_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
