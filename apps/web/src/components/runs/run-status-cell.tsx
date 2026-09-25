import { StatusBadge } from "@/components/common/status-badge";
import type { RunSummary } from "@/lib/api/types";
import { formatPercent } from "@/lib/format";
import { isRunActive } from "@/lib/status";

export function RunStatusCell({ run }: { run: RunSummary }) {
  const stage = run.current_stage;
  return (
    <div className="flex flex-col items-start gap-1">
      <StatusBadge domain="run" status={run.status} />
      {stage && isRunActive(run.status) && (
        <span className="text-xs text-muted-foreground">
          {stage.title}
          {stage.progress !== null && ` · ${formatPercent(stage.progress)}`}
        </span>
      )}
    </div>
  );
}
