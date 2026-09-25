import { StatusBadge } from "@/components/common/status-badge";
import { COMPUTE_KIND_LABELS } from "@/components/compute/target-kind";
import { Card, CardContent } from "@/components/ui/card";
import type { DashboardTarget } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

export function TargetSummaryCard({ target }: { target: DashboardTarget }) {
  return (
    <Card className="gap-0 py-4">
      <CardContent className="space-y-3 px-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-sm font-medium">{target.name}</div>
            <div className="text-xs text-muted-foreground">
              {COMPUTE_KIND_LABELS[target.kind]}
              {target.gpu_label && ` · ${target.gpu_label}`}
            </div>
          </div>
          <StatusBadge domain="health" status={target.health.status} />
        </div>
        <dl className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <dt className="text-muted-foreground">Active stages</dt>
            <dd className="tabular text-sm font-medium">{formatNumber(target.active_stages)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Quota left</dt>
            <dd className="tabular text-sm font-medium">
              {target.quota_left_hours === null ? "No quota" : `${formatNumber(target.quota_left_hours, 1)} GPU-h`}
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
