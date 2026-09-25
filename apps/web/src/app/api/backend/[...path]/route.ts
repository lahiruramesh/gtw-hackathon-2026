import type { NextRequest } from "next/server";

import { FORWARDED_REQUEST_HEADERS, filterResponseHeaders, resolveUpstreamPath } from "@/lib/api/proxy-path";
import { getAuth } from "@/lib/auth";
import { serverEnv } from "@/lib/env";

type Context = RouteContext<"/api/backend/[...path]">;

function errorResponse(status: number, code: string, message: string): Response {
  return Response.json({ error: { code, message, details: {} } }, { status });
}

/**
 * Same-origin proxy from the browser to the API: attaches the user's short-lived JWT and streams
 * the upstream body through unbuffered (required for server-sent events).
 */
async function proxy(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  const upstreamPath = resolveUpstreamPath(path, new URL(request.url).pathname);
  if (!upstreamPath) return errorResponse(400, "invalid_path", "Invalid API path");

  let token: string;
  try {
    ({ token } = await getAuth().api.getToken({ headers: request.headers }));
  } catch {
    return errorResponse(401, "unauthenticated", "Sign in again to continue");
  }

  const upstreamUrl = new URL(`/api/v1/${upstreamPath}`, serverEnv().API_INTERNAL_URL);
  upstreamUrl.search = request.nextUrl.search;

  const headers = new Headers({ Authorization: `Bearer ${token}` });
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
      signal: request.signal,
      cache: "no-store",
      redirect: "manual",
    });
  } catch {
    if (request.signal.aborted) return new Response(null, { status: 499 });
    return errorResponse(503, "api_unreachable", "The Skill Studio API is not reachable.");
  }

  const responseHeaders = filterResponseHeaders(upstream.headers);
  if (responseHeaders.get("content-type")?.startsWith("text/event-stream")) {
    responseHeaders.set("cache-control", "no-cache, no-transform");
    responseHeaders.set("x-accel-buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export { proxy as DELETE, proxy as GET, proxy as PATCH, proxy as POST, proxy as PUT };
