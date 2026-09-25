/**
 * Turns an evaluation summary (arbitrary JSON written by a skill's summarizer) into displayable
 * parts: scalar tiles, a step-length tracking heatmap, the stairs by-height table, generic tables
 * and key/value blocks. The API stores summaries as JSONB, which does not keep key order, so every
 * purpose-built view fixes its own column order.
 */

type Scalar = number | string | boolean | null;
type Row = Record<string, unknown>;

export interface EvalScalar {
  key: string;
  value: Scalar;
}

export interface EvalTable {
  key: string;
  columns: string[];
  rows: { id: string; cells: Row }[];
}

export interface EvalDetails {
  key: string;
  entries: EvalScalar[];
}

export interface EvalHeatmap {
  xKey: string;
  yKey: string;
  valueLabel: string;
  xs: number[];
  ys: number[];
  /** Mean value per `${x}|${y}` cell (cm). */
  values: Record<string, number>;
  /** Share of fallen episodes per cell, when the rows say whether the robot fell. */
  fallRates: Record<string, number>;
}

/** One step height of a stairs strict test (skills/stairs/summarize.py `by_height`). */
export interface StairHeight {
  riseCm: number;
  runs: number;
  crossed: number;
  fell: number;
  stable: boolean | null;
  maxTiltDeg: number | null;
  minPelvisM: number | null;
  /** The certified height: the highest rise at which every crossing stayed stable. */
  certified: boolean;
}

export interface EvalView {
  scalars: EvalScalar[];
  heatmap: EvalHeatmap | null;
  heights: StairHeight[] | null;
  tables: EvalTable[];
  details: EvalDetails[];
  rest: Record<string, unknown>;
}

const X_KEYS = ["step_req", "step_cmd", "step_length", "step"];
const Y_KEYS = ["vx", "speed"];
/** Value columns in preference order, with the factor that converts them to centimetres. */
const VALUE_KEYS: [string, number][] = [
  ["step_abs_err_cm", 1],
  ["step_err_cm", 1],
  ["step_abs_err_m", 100],
  ["step_err", 100],
];

function isRecord(value: unknown): value is Row {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isScalar(value: unknown): value is Scalar {
  return value === null || ["number", "string", "boolean"].includes(typeof value);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function cellKey(x: number, y: number): string {
  return `${x}|${y}`;
}

function toHeatmap(rows: Row[]): EvalHeatmap | null {
  const first = rows[0];
  if (!first) return null;
  const xKey = X_KEYS.find((key) => typeof first[key] === "number");
  const yKey = Y_KEYS.find((key) => typeof first[key] === "number");
  const value = VALUE_KEYS.find(([key]) => typeof first[key] === "number");
  if (!xKey || !yKey || !value) return null;
  const [valueKey, factor] = value;

  const samples = new Map<string, { values: number[]; fell: boolean[] }>();
  for (const row of rows) {
    const x = row[xKey];
    const y = row[yKey];
    const v = row[valueKey];
    if (typeof x !== "number" || typeof y !== "number") continue;
    const cell = samples.get(cellKey(x, y)) ?? { values: [], fell: [] };
    if (typeof v === "number" && Number.isFinite(v)) cell.values.push(v * factor);
    if (typeof row.fell === "boolean") cell.fell.push(row.fell);
    samples.set(cellKey(x, y), cell);
  }
  const values: Record<string, number> = {};
  const fallRates: Record<string, number> = {};
  for (const [key, cell] of samples) {
    if (cell.values.length > 0) values[key] = mean(cell.values);
    if (cell.fell.length > 0) fallRates[key] = cell.fell.filter(Boolean).length / cell.fell.length;
  }
  const unique = (key: string) =>
    [...new Set(rows.map((row) => row[key]).filter((v): v is number => typeof v === "number"))].sort((a, b) => a - b);
  return { xKey, yKey, valueLabel: "Step-length error (cm)", xs: unique(xKey), ys: unique(yKey), values, fallRates };
}

const optionalNumber = (value: unknown) => (typeof value === "number" ? value : null);

function toHeights(rows: Row[], certifiedCm: unknown): StairHeight[] | null {
  if (!rows.every((row) => typeof row.rise_cm === "number" && typeof row.runs === "number")) return null;
  return rows
    .map((row) => ({
      riseCm: row.rise_cm as number,
      runs: row.runs as number,
      crossed: optionalNumber(row.crossed) ?? 0,
      fell: optionalNumber(row.fell) ?? 0,
      stable: typeof row.stable === "boolean" ? row.stable : null,
      maxTiltDeg: optionalNumber(row.max_tilt_deg),
      minPelvisM: optionalNumber(row.min_pelvis_m),
      certified: typeof certifiedCm === "number" && Math.abs((row.rise_cm as number) - certifiedCm) < 1e-6,
    }))
    .sort((a, b) => a.riseCm - b.riseCm);
}

function toTable(key: string, value: unknown): EvalTable | null {
  let entries: [string, Row][];
  if (Array.isArray(value) && value.length > 0 && value.every(isRecord)) {
    entries = value.map((row, index) => [String(row.case ?? row.name ?? index + 1), row]);
  } else if (isRecord(value) && Object.keys(value).length > 0 && Object.values(value).every(isRecord)) {
    entries = Object.entries(value as Record<string, Row>);
  } else {
    return null;
  }
  const keyed = !Array.isArray(value);
  const columns = [
    ...new Set(entries.flatMap(([, row]) => Object.keys(row).filter((column) => isScalar(row[column])))),
  ];
  if (columns.length === 0) return null;
  return {
    key,
    columns: keyed ? ["case", ...columns.filter((column) => column !== "case")] : columns,
    rows: entries.map(([id, row]) => ({ id, cells: keyed ? { case: id, ...row } : row })),
  };
}

export function describeEvaluation(summary: Record<string, unknown>): EvalView {
  const view: EvalView = { scalars: [], heatmap: null, heights: null, tables: [], details: [], rest: {} };
  for (const [key, value] of Object.entries(summary)) {
    if (isScalar(value)) {
      view.scalars.push({ key, value });
      continue;
    }
    if (key === "grid" && Array.isArray(value) && value.every(isRecord) && !view.heatmap) {
      view.heatmap = toHeatmap(value);
      if (view.heatmap) continue;
    }
    if (key === "by_height" && Array.isArray(value) && value.every(isRecord) && !view.heights) {
      view.heights = toHeights(value, summary.certified_cm);
      if (view.heights) continue;
    }
    const table = toTable(key, value);
    if (table) {
      view.tables.push(table);
      continue;
    }
    if (isRecord(value) && Object.values(value).every(isScalar)) {
      view.details.push({
        key,
        entries: Object.entries(value).map(([entry, v]) => ({ key: entry, value: v as Scalar })),
      });
      continue;
    }
    view.rest[key] = value;
  }
  return view;
}

/** Display rule for evaluation numbers: rates as percentages, lengths with units. */
export function formatEvalValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value !== "number") return Array.isArray(value) ? value.join(" to ") : String(value);
  if (!Number.isFinite(value)) return "—";
  if (/rate$|_rate_|ratio/.test(key)) return `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`;
  if (key.endsWith("_cm")) return `${value.toFixed(1)} cm`;
  if (key.endsWith("_m")) return `${value.toFixed(3)} m`;
  if (key.endsWith("_deg")) return `${value.toFixed(1)}°`;
  if (Number.isInteger(value)) return value.toLocaleString("en-GB");
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3);
}
