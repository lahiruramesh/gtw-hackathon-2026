import { errorFromResponse, unreachableError } from "@/lib/api/errors";
import { withQuery, type Query } from "@/lib/api/query";

/** Browser path of the same-origin API proxy (app/api/backend/[...path]). */
export function proxyPath(path: string, query?: Query): string {
  return withQuery(`/api/backend${path}`, query);
}

interface ClientRequestOptions {
  method?: "GET" | "POST" | "PATCH" | "PUT" | "DELETE";
  query?: Query;
  body?: unknown;
  signal?: AbortSignal;
}

/** Client components' API access, through the proxy that attaches the user's token. */
export async function clientFetch<T>(path: string, options: ClientRequestOptions = {}): Promise<T> {
  const hasBody = options.body !== undefined;
  let response: Response;
  try {
    response = await fetch(proxyPath(path, options.query), {
      method: options.method ?? "GET",
      headers: { Accept: "application/json", ...(hasBody ? { "Content-Type": "application/json" } : {}) },
      body: hasBody ? JSON.stringify(options.body) : undefined,
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw unreachableError();
  }
  if (!response.ok) throw await errorFromResponse(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}
