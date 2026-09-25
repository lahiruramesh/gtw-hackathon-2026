export interface TextPart {
  text: string;
  match: boolean;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Splits text around case-insensitive occurrences of `query`, for <mark> highlighting. */
export function splitHighlight(text: string, query: string): TextPart[] {
  if (!query) return [{ text, match: false }];
  const pattern = new RegExp(`(${escapeRegExp(query)})`, "gi");
  return text
    .split(pattern)
    .filter((part) => part !== "")
    .map((part) => ({ text: part, match: part.toLowerCase() === query.toLowerCase() }));
}

export function includesIgnoreCase(text: string, query: string): boolean {
  return text.toLowerCase().includes(query.toLowerCase());
}
