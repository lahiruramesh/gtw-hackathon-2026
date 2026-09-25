export interface ChartSeries {
  /** Must be a CSS identifier: it becomes the `--color-<id>` variable. */
  id: string;
  label: string;
  points: readonly (readonly [number, number])[];
}

export type ChartRow = { step: number } & Record<string, number | null>;

/** Joins series on `step` into recharts rows; a series without a value at a step gets null. */
export function mergeSeries(series: readonly ChartSeries[]): ChartRow[] {
  const rows = new Map<number, ChartRow>();
  for (const { id, points } of series) {
    for (const [step, value] of points) {
      const row = rows.get(step) ?? ({ step } as ChartRow);
      row[id] = Number.isFinite(value) ? value : null;
      rows.set(step, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.step - b.step);
}

export const CHART_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];
