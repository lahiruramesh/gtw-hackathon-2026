const PARSE_BASE = "http://skf.invalid";

function hasControlOrBackslash(value: string): boolean {
  for (const char of value) {
    const code = char.charCodeAt(0);
    if (code < 0x20 || code === 0x7f || char === "\\") return true;
  }
  return false;
}

/**
 * Only same-origin paths are allowed as post-login destinations. URL parsing strips tabs and
 * newlines, so `/\t/evil.example` would otherwise resolve to another host: such values, and any
 * value that parses to a different origin, fall back to "/".
 */
export function safeNextPath(value: string | null | undefined): string {
  if (!value || !value.startsWith("/") || hasControlOrBackslash(value)) return "/";
  let url: URL;
  try {
    url = new URL(value, PARSE_BASE);
  } catch {
    return "/";
  }
  if (url.origin !== PARSE_BASE) return "/";
  return `${url.pathname}${url.search}${url.hash}`;
}
