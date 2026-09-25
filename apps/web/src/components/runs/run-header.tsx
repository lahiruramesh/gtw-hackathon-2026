"use client";

import Link from "next/link";

import { DateTime } from "@/components/common/date-time";
import { StatusBadge } from "@/components/common/status-badge";
import { COMPUTE_KIND_LABELS } from "@/components/compute/target-kind";
import { RunActions } from "@/components/runs/run-actions";
import { useRunLive } from "@/components/runs/run-live-provider";
import { Badge } from "@/components/ui/badge";
import { formatCost, formatNumber, shortSha } from "@/lib/format";
import type { ConnectionState } from "@/lib/run-events";

const CONNECTION_LABELS: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  live: "Live",
  reconnecting: "Reconnecting…",
  closed: "",
};

export function RunHeader({ viewerId }: { viewerId: string }) {
  const { run, connection } = useRunLive();
  return (
    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
      <div className="min-w-0 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="truncate text-xl font-semibold tracking-tight">{run.name}</h1>
          <StatusBadge domain="run" status={run.status} />
          {run.imported && <Badge variant="outline">Imported</Badge>}
          {connection !== "closed" && (
            <span className="text-xs text-muted-foreground" aria-live="polite">
              {CONNECTION_LABELS[connection]}
            </span>
          )}
        </div>
        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <div>
            <dt className="sr-only">Skill</dt>
            <dd>
              <Link
                href={`/skills/${run.skill_id}`}
                className="underline-offset-4 hover:text-foreground hover:underline"
              >
                {run.skill_name}
              </Link>
              {run.preset_id ? ` · preset ${run.preset_id}` : " · custom parameters"}
            </dd>
          </div>
          <div>
            <dt className="sr-only">Compute target</dt>
            <dd>
              {run.compute_target
                ? `${run.compute_target.name} (${COMPUTE_KIND_LABELS[run.compute_target.kind]})`
                : "No compute target"}
            </dd>
          </div>
          {run.git_sha && (
            <div>
              <dt className="sr-only">Commit</dt>
              <dd className="font-mono" title={run.git_sha}>
                {shortSha(run.git_sha)}
              </dd>
            </div>
          )}
          <div>
            <dt className="sr-only">Created</dt>
            <dd>
              {run.created_by.name} · <DateTime value={run.created_at} />
            </dd>
          </div>
          <div>
            <dt className="sr-only">Usage</dt>
            <dd className="tabular">
              {formatNumber(run.gpu_hours, 2)} GPU-h · {formatCost(run.cost)}
            </dd>
          </div>
        </dl>
        {run.error && <p className="text-sm text-status-danger">{run.error}</p>}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        <RunActions viewerId={viewerId} />
      </div>
    </div>
  );
}
