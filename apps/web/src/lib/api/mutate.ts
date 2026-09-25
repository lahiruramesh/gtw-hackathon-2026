import "server-only";

import { refresh } from "next/cache";

import { toResult, type Result } from "@/lib/api/errors";
import { apiFetch, type ApiRequestOptions } from "@/lib/api/server";

/** Server-action helper: calls the API and, on success, refreshes the client router's data. */
export async function mutate<T>(path: string, options: ApiRequestOptions): Promise<Result<T>> {
  const result = await toResult(apiFetch<T>(path, options));
  if (result.ok) refresh();
  return result;
}

/** Encodes one path segment; server actions receive ids from the client and must not trust them. */
export function segment(value: string): string {
  return encodeURIComponent(value);
}
