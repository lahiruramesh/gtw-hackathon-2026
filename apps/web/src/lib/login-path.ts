import { safeNextPath } from "@/lib/safe-redirect";

/**
 * Request header the proxy sets to the page's path and query, so server code that finds the
 * session expired can send the user back there after signing in.
 */
export const REQUEST_PATH_HEADER = "x-skf-path";

/** The sign-in page, returning to `next` (a same-origin path) afterwards. */
export function loginPath(next: string | null | undefined): string {
  const destination = safeNextPath(next);
  return destination === "/" ? "/login" : `/login?next=${encodeURIComponent(destination)}`;
}
