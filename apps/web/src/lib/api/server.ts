import "server-only";

import { headers } from "next/headers";

import { ApiError, errorFromResponse, unreachableError } from "@/lib/api/errors";
import { withQuery, type Query } from "@/lib/api/query";
import { getAuth } from "@/lib/auth";
import { serverEnv } from "@/lib/env";
import { redirectToLogin } from "@/lib/session";

const REQUEST_TIMEOUT_MS = 15_000;

export interface ApiRequestOptions {
  method?: "GET" | "POST" | "PATCH" | "PUT" | "DELETE";
  query?: Query;
  body?: unknown;
}

/**
 * Short-lived API token for the signed-in user. Sends the user to sign in (and back to this page)
 * when there is no valid session.
 */
export async function getApiToken(requestHeaders: Headers): Promise<string> {
  try {
    const { token } = await getAuth().api.getToken({ headers: requestHeaders });
    return token;
  } catch {
    return redirectToLogin();
  }
}

export function apiUrl(path: string): URL {
  return new URL(`/api/v1${path}`, serverEnv().API_INTERNAL_URL);
}

/**
 * Calls the Skill Studio API as the signed-in user. Server components and server actions only.
 * Never cached: every response is per-user.
 */
export async function apiFetch<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const token = await getApiToken(await headers());
  const hasBody = options.body !== undefined;

  let response: Response;
  try {
    response = await fetch(apiUrl(withQuery(path, options.query)), {
      method: options.method ?? "GET",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token}`,
        ...(hasBody ? { "Content-Type": "application/json" } : {}),
      },
      body: hasBody ? JSON.stringify(options.body) : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw unreachableError();
  }

  if (!response.ok) {
    if (response.status === 502 || response.status === 504) throw unreachableError();
    throw await errorFromResponse(response);
  }
  if (response.status === 204) return undefined as T;
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError({ status: 502, code: "bad_response", message: "The API returned an invalid response." });
  }
}
