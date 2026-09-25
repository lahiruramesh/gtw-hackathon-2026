const EMPTY = "—";

const LOCALE = "en-GB";

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return value.toLocaleString(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Adaptive precision for metric values of unknown magnitude (rewards, errors, rates). */
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1000) return formatCompact(value);
  if (Number.isInteger(value)) return formatNumber(value);
  if (magnitude >= 100) return formatNumber(value, 0);
  if (magnitude >= 1) return formatNumber(value, 2);
  return String(Number(value.toPrecision(3)));
}

/** 202_000_000 → "202M", 12_500 → "12.5k". */
export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  // en-GB abbreviates millions as "m"; the upper-case form is the convention for step counts.
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatPercent(ratio: number | null | undefined, digits = 0): string {
  if (ratio === null || ratio === undefined || !Number.isFinite(ratio)) return EMPTY;
  return `${formatNumber(ratio * 100, digits)}%`;
}

export function formatGpuHours(hours: number | null | undefined): string {
  if (hours === null || hours === undefined || !Number.isFinite(hours)) return EMPTY;
  return `${formatNumber(hours, hours < 10 ? 1 : 0)} GPU-h`;
}

export function formatCost(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  // en-US prints "$", where en-GB would print "US$".
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return EMPTY;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${formatNumber(value, unit === 0 || value >= 100 ? 0 : 1)} ${units[unit]}`;
}

/** 5700 → "1 h 35 min", 42 → "42 s". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return EMPTY;
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest === 0 ? `${hours} h` : `${hours} h ${rest} min`;
}

export function formatMinutes(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined) return EMPTY;
  return formatDuration(minutes * 60);
}

export function secondsBetween(start: string | null, end: string | null, now: number = Date.now()): number | null {
  if (!start) return null;
  const from = Date.parse(start);
  const to = end ? Date.parse(end) : now;
  if (Number.isNaN(from) || Number.isNaN(to)) return null;
  return Math.max(0, (to - from) / 1000);
}

export function shortSha(sha: string | null | undefined): string {
  return sha ? sha.slice(0, 7) : EMPTY;
}

/** Deterministic UTC rendering used on the server and as the hydration fallback. */
export function formatDateTimeUtc(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return EMPTY;
  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

export function formatDateTimeLocal(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  const sameYear = date.getFullYear() === now.getFullYear();
  return date.toLocaleString(LOCALE, {
    day: "numeric",
    month: "short",
    ...(sameYear ? {} : { year: "numeric" }),
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatRelative(iso: string, now: number = Date.now()): string {
  const seconds = (Date.parse(iso) - now) / 1000;
  const format = new Intl.RelativeTimeFormat(LOCALE, { numeric: "auto" });
  const abs = Math.abs(seconds);
  if (abs < 60) return format.format(Math.round(seconds), "second");
  if (abs < 3600) return format.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return format.format(Math.round(seconds / 3600), "hour");
  return format.format(Math.round(seconds / 86400), "day");
}

/** "eval/episode_reward" → "Episode reward". */
export function humanizeKey(key: string): string {
  const last = key.split("/").pop() ?? key;
  const words = last
    .replace(/^episode_/, "")
    .replace(/[_-]+/g, " ")
    .trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return EMPTY;
  if (typeof value === "number") return Number.isInteger(value) ? formatNumber(value) : formatMetric(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}
