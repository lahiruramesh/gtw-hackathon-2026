const SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const ENCODED_SEPARATOR = /%(2f|5c|2e|00)/i;
const MAX_SEGMENTS = 8;

/**
 * Validates the catch-all segments of `/api/backend/[...path]` and returns the API path below
 * `/api/v1/`, or null when the request could escape that prefix. `rawPathname` is the undecoded
 * request path, used to reject encoded slashes and dots that decoded segments would hide.
 */
export function resolveUpstreamPath(segments: readonly string[], rawPathname: string): string | null {
  if (segments.length === 0 || segments.length > MAX_SEGMENTS) return null;
  if (ENCODED_SEPARATOR.test(rawPathname)) return null;
  if (!segments.every((segment) => SEGMENT.test(segment))) return null;
  return segments.join("/");
}

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

// fetch() already decoded the body, so length and encoding no longer describe what we send; the
// upstream server banner is not the browser's business.
const STRIPPED_RESPONSE_HEADERS = new Set([
  ...HOP_BY_HOP,
  "set-cookie",
  "content-encoding",
  "content-length",
  "server",
]);

export const FORWARDED_REQUEST_HEADERS = ["accept", "content-type", "last-event-id"] as const;

export function filterResponseHeaders(upstream: Headers): Headers {
  const result = new Headers();
  upstream.forEach((value, key) => {
    if (!STRIPPED_RESPONSE_HEADERS.has(key.toLowerCase())) result.set(key, value);
  });
  return result;
}

/** Largest request body the proxy forwards (JSON commands and forms; the API never takes uploads). */
export const MAX_REQUEST_BODY_BYTES = 1024 * 1024;

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/**
 * A second line of defence against cross-site request forgery for state-changing calls, beyond
 * SameSite cookies (which let sibling subdomains through): the request must come from our own origin.
 */
export function isForeignRequest(method: string, requestHeaders: Headers, appOrigin: string): boolean {
  if (SAFE_METHODS.has(method)) return false;
  const site = requestHeaders.get("sec-fetch-site");
  if (site === "cross-site" || site === "same-site") return true;
  const origin = requestHeaders.get("origin");
  return origin !== null && origin !== appOrigin;
}

/** Reads at most `limit` bytes of a request body; null when it is larger. */
export async function readLimitedBody(
  body: ReadableStream<Uint8Array> | null,
  limit: number,
): Promise<Uint8Array<ArrayBuffer> | null> {
  const chunks: Uint8Array[] = [];
  let size = 0;
  if (body) {
    const reader = body.getReader();
    for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) {
      size += chunk.value.byteLength;
      if (size > limit) {
        await reader.cancel();
        return null;
      }
      chunks.push(chunk.value);
    }
  }
  const result = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}
