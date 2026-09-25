export type QueryValue = string | number | boolean | null | undefined | readonly string[];
export type Query = Record<string, QueryValue>;

/** Serialises a query object, dropping empty values and joining arrays with commas (the API's list format). */
export function toSearchParams(query: Query | undefined): URLSearchParams {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      if (value.length > 0) params.set(key, value.join(","));
      continue;
    }
    params.set(key, String(value));
  }
  return params;
}

export function withQuery(path: string, query: Query | undefined): string {
  const search = toSearchParams(query).toString();
  return search ? `${path}?${search}` : path;
}
