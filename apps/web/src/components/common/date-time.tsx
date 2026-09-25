"use client";

import { useSyncExternalStore } from "react";

import { formatDateTimeLocal, formatDateTimeUtc, formatRelative } from "@/lib/format";

const subscribeNever = () => () => {};

/** Renders a timestamp in the viewer's time zone; the server renders a deterministic UTC value. */
export function DateTime({ value, relative = false }: { value: string | null | undefined; relative?: boolean }) {
  const isClient = useSyncExternalStore(
    subscribeNever,
    () => true,
    () => false,
  );
  if (!value) return <span className="text-muted-foreground">—</span>;
  const label = !isClient ? formatDateTimeUtc(value) : relative ? formatRelative(value) : formatDateTimeLocal(value);
  return (
    <time
      dateTime={value}
      title={isClient ? new Date(value).toLocaleString() : undefined}
      className="tabular whitespace-nowrap"
    >
      {label}
    </time>
  );
}
