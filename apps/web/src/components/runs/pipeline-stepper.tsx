"use client";

import { CheckIcon, CircleIcon, ExternalLinkIcon, LoaderIcon, MinusIcon, XIcon, type LucideIcon } from "lucide-react";

import { StatusBadge } from "@/components/common/status-badge";
import { useRunLive } from "@/components/runs/run-live-provider";
import { Progress } from "@/components/ui/progress";
import { useNow } from "@/hooks/use-now";
import type { GateVerdict, Stage, StageStatus } from "@/lib/api/types";
import { formatCost, formatDuration, formatPercent, secondsBetween } from "@/lib/format";
import { gateStageVerdict } from "@/lib/status";
import { cn } from "cn";

const STATUS_ICONS: Partial<Record<StageStatus, LucideIcon>> = {
  succeeded: CheckIcon,
  failed: XIcon,
  cancelled: MinusIcon,
  skipped: MinusIcon,
  queued: LoaderIcon,
  provisioning: LoaderIcon,
  running: LoaderIcon,
  collecting: LoaderIcon,
};

const ICON_TONES: Partial<Record<StageStatus, string>> = {
  succeeded: "bg-status-success/15 text-status-success",
  failed: "bg-status-danger/15 text-status-danger",
  running: "bg-status-info/15 text-status-info",
  provisioning: "bg-status-info/15 text-status-info",
  collecting: "bg-status-info/15 text-status-info",
};

function StageStep({ stage, gateVerdict, now }: { stage: Stage; gateVerdict: GateVerdict | null; now: number | null }) {
  const verdict = gateStageVerdict(stage, gateVerdict);
  const shownStatus: StageStatus = verdict === "fail" ? "failed" : stage.status;
  const Icon = STATUS_ICONS[shownStatus] ?? CircleIcon;
  const spinning = Icon === LoaderIcon;
  const running = stage.started_at !== null && stage.finished_at === null;
  const seconds =
    running && now === null ? null : secondsBetween(stage.started_at, stage.finished_at, now ?? undefined);
  return (
    <li className="flex min-w-0 flex-1 gap-3 rounded-md border p-3">
      <span
        className={cn(
          "mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground",
          ICON_TONES[shownStatus],
        )}
        aria-hidden
      >
        <Icon className={cn("size-3.5", spinning && "animate-spin [animation-duration:2s]")} />
      </span>
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="line-clamp-2 text-sm font-medium break-words" title={stage.key}>
            {stage.title}
          </span>
          {stage.attempt > 1 && <span className="text-xs text-muted-foreground">attempt {stage.attempt}</span>}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {verdict ? (
            <span title="The gate check ran; this is its verdict">
              <StatusBadge domain="gate" status={verdict} />
            </span>
          ) : (
            <StatusBadge domain="stage" status={stage.status} />
          )}
          <span className="tabular text-muted-foreground">{formatDuration(seconds)}</span>
          {stage.gpu_seconds > 0 && (
            <span className="tabular text-muted-foreground">
              {formatDuration(stage.gpu_seconds)} GPU · {formatCost(stage.cost)}
            </span>
          )}
        </div>
        {stage.progress !== null && stage.status === "running" && (
          <div className="flex items-center gap-2">
            <Progress value={stage.progress * 100} className="h-1.5" aria-label={`${stage.title} progress`} />
            <span className="tabular text-xs text-muted-foreground">{formatPercent(stage.progress)}</span>
          </div>
        )}
        {stage.message && (
          <p className="truncate text-xs text-muted-foreground" title={stage.message}>
            {stage.message}
          </p>
        )}
        {stage.error && <p className="text-xs break-words text-status-danger">{stage.error}</p>}
        {stage.external_url && (
          <a
            href={stage.external_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
          >
            Open in provider <ExternalLinkIcon className="size-3" />
          </a>
        )}
      </div>
    </li>
  );
}

export function PipelineStepper() {
  const { run } = useRunLive();
  const now = useNow();
  if (run.stages.length === 0) return null;
  return (
    <ol className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3" aria-label="Pipeline stages">
      {run.stages.map((stage) => (
        <StageStep key={stage.id} stage={stage} gateVerdict={run.gate_verdict} now={now} />
      ))}
    </ol>
  );
}
