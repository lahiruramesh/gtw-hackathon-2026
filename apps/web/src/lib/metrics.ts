import type { MetricsResponse } from "@/lib/api/types";

/** Default chart selection: the skill's primary metric first, then its other declared keys. */
export function defaultMetricKeys(
  available: readonly string[],
  primary: string,
  declared: readonly string[],
  limit = 4,
): string[] {
  const ordered = [primary, ...declared.filter((key) => key !== primary)];
  const chosen = ordered.filter((key) => available.includes(key));
  return (chosen.length > 0 ? chosen : available).slice(0, limit);
}

/** Primary first, then the order the user picked them in. */
export function orderSelection(selected: readonly string[], primary: string): string[] {
  return [...selected].sort((a, b) => Number(b === primary) - Number(a === primary));
}

/** Appends one live `metric` event to a metrics response (no-op for steps already present). */
export function appendMetricPoint(
  current: MetricsResponse,
  step: number,
  values: Record<string, number>,
): MetricsResponse {
  const series = [...current.series];
  const keys = new Set(current.keys);
  for (const [key, value] of Object.entries(values)) {
    const index = series.findIndex((item) => item.key === key);
    if (index === -1) {
      series.push({ key, points: [[step, value]] });
      keys.add(key);
      continue;
    }
    const existing = series[index]!;
    if (existing.points.some(([pointStep]) => pointStep === step)) continue;
    series[index] = { key, points: [...existing.points, [step, value]] };
  }
  return { keys: [...keys], series };
}
