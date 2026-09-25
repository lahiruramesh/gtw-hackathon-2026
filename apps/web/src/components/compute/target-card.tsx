import { DateTime } from "@/components/common/date-time";
import { StatusBadge } from "@/components/common/status-badge";
import { HealthCheckButton } from "@/components/compute/health-check-button";
import { SecretDialog } from "@/components/compute/secret-dialog";
import { TargetFormDialog } from "@/components/compute/target-form-dialog";
import { COMPUTE_KIND_LABELS } from "@/components/compute/target-kind";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { ComputeTarget } from "@/lib/api/types";
import { formatCost, formatNumber, formatValue } from "@/lib/format";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="tabular text-sm font-medium">{value}</dd>
    </div>
  );
}

export function TargetCard({ target, canManage }: { target: ComputeTarget; canManage: boolean }) {
  const quota = target.weekly_quota_gpu_hours;
  const used = target.usage.gpu_hours_7d;
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          {target.name}
          {!target.enabled && <Badge variant="outline">Disabled</Badge>}
        </CardTitle>
        <CardDescription>
          {COMPUTE_KIND_LABELS[target.kind]}
          {target.gpu_label && ` · ${target.gpu_label}`}
          {target.description && ` · ${target.description}`}
        </CardDescription>
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <StatusBadge domain="health" status={target.health.status} />
          {target.health.checked_at && (
            <span className="text-xs text-muted-foreground">
              checked <DateTime value={target.health.checked_at} relative />
            </span>
          )}
        </div>
        {target.health.message && <p className="text-xs text-muted-foreground">{target.health.message}</p>}
      </CardHeader>
      <CardContent className="space-y-4">
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="GPU-h, 7 days" value={formatNumber(used, 1)} />
          <Stat label="Cost, 7 days" value={formatCost(target.usage.cost_7d)} />
          <Stat label="Active stages" value={`${target.usage.active_stages} / ${target.max_concurrent}`} />
          <Stat label="Cost per GPU-h" value={formatCost(target.cost_per_gpu_hour)} />
        </dl>
        {quota !== null && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Weekly quota</span>
              <span className="tabular">
                {formatNumber(target.usage.quota_left_hours, 1)} of {formatNumber(quota, 0)} GPU-h left
              </span>
            </div>
            <Progress value={quota > 0 ? Math.min(100, (used / quota) * 100) : 100} aria-label="Weekly quota used" />
          </div>
        )}
        <dl className="grid grid-cols-1 gap-x-4 gap-y-0.5 text-xs text-muted-foreground sm:grid-cols-2">
          <div>Throughput: {formatNumber(target.steps_per_second)} steps/s</div>
          <div>Overhead: {formatNumber(target.overhead_minutes)} min per run</div>
          <div>Approval above: {formatNumber(target.max_unapproved_gpu_hours, 1)} GPU-h</div>
          <div>Credentials: {target.has_secret ? "set" : "not set"}</div>
          {Object.entries(target.config).map(([key, value]) => (
            <div key={key} className="truncate">
              {key}: <span className="font-mono">{formatValue(value)}</span>
            </div>
          ))}
        </dl>
      </CardContent>
      {canManage && (
        <CardFooter className="mt-auto flex-wrap gap-1 border-t pt-4">
          <TargetFormDialog target={target} />
          <SecretDialog target={target} />
          <HealthCheckButton targetId={target.id} />
        </CardFooter>
      )}
    </Card>
  );
}
