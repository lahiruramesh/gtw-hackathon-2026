import { cn } from "cn";

import { statusStyle, type StatusDomain, type StatusTone } from "@/lib/status";

const TONE_CLASSES: Record<StatusTone, string> = {
  success: "border-status-success/30 bg-status-success/10 text-status-success",
  warning: "border-status-warning/30 bg-status-warning/10 text-status-warning",
  danger: "border-status-danger/30 bg-status-danger/10 text-status-danger",
  info: "border-status-info/30 bg-status-info/10 text-status-info",
  neutral: "border-border bg-muted text-muted-foreground",
};

const DOT_CLASSES: Record<StatusTone, string> = {
  success: "bg-status-success",
  warning: "bg-status-warning",
  danger: "bg-status-danger",
  info: "bg-status-info",
  neutral: "bg-status-neutral",
};

interface StatusBadgeProps {
  domain: StatusDomain;
  status: string;
  /** Names what the status belongs to, e.g. "Sim" gives "Sim passed". */
  prefix?: string;
  className?: string;
}

/** The single mapping from run, stage, gate, review and health states to colour and label. */
export function StatusBadge({ domain, status, prefix, className }: StatusBadgeProps) {
  const style = statusStyle(domain, status);
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1.5 rounded-full border px-2 text-xs font-medium whitespace-nowrap",
        TONE_CLASSES[style.tone],
        className,
      )}
    >
      <span className="relative flex size-1.5" aria-hidden>
        {style.active && (
          <span
            className={cn(
              "absolute inline-flex size-full animate-ping rounded-full opacity-60",
              DOT_CLASSES[style.tone],
            )}
          />
        )}
        <span className={cn("relative inline-flex size-1.5 rounded-full", DOT_CLASSES[style.tone])} />
      </span>
      {prefix ? `${prefix} ${style.label.toLowerCase()}` : style.label}
    </span>
  );
}

export function StatusDot({ domain, status }: { domain: StatusDomain; status: string }) {
  const style = statusStyle(domain, status);
  return (
    <span className={cn("inline-block size-2 shrink-0 rounded-full", DOT_CLASSES[style.tone])} title={style.label} />
  );
}
