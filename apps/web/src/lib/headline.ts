import type { HeadlineValue } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

/** The API applies each headline's scale; the unit tells us how to print it. */
export function formatHeadline(item: HeadlineValue, digits = 1): string {
  if (item.value === null) return "—";
  const precision = item.unit === "%" ? 0 : Number.isInteger(item.value) ? 0 : digits;
  const number = formatNumber(item.value, precision);
  if (!item.unit) return number;
  return item.unit === "%" ? `${number}%` : `${number} ${item.unit}`;
}
