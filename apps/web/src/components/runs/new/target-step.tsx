"use client";

import { useQueries } from "@tanstack/react-query";
import { CpuIcon, ShieldAlertIcon } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/common/empty-state";
import { StatusBadge } from "@/components/common/status-badge";
import { COMPUTE_KIND_LABELS } from "@/components/compute/target-kind";
import { EstimateSummary } from "@/components/runs/new/estimate-summary";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Skeleton } from "@/components/ui/skeleton";
import { clientFetch } from "@/lib/api/client";
import { describeError, toErrorInfo } from "@/lib/api/errors";
import type { ComputeTarget, Estimate, SkillDetail } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";
import { estimateRequest, type RunDraft } from "@/lib/run-draft";
import { cn } from "cn";

interface TargetStepProps {
  skill: SkillDetail;
  draft: RunDraft;
  targets: ComputeTarget[];
  onBack: () => void;
  onSelect: (targetId: string) => void;
  onNext: () => void;
}

export function estimateQuery(skill: SkillDetail, draft: RunDraft, targetId: string) {
  const body = estimateRequest(skill, draft, targetId);
  return {
    queryKey: ["estimate", body],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      clientFetch<Estimate>("/runs/estimate", { method: "POST", body, signal }),
    enabled: body !== null,
    staleTime: 60_000,
  };
}

export function TargetStep({ skill, draft, targets, onBack, onSelect, onNext }: TargetStepProps) {
  const enabled = targets.filter((target) => target.enabled);
  const disabled = targets.filter((target) => !target.enabled);
  const estimates = useQueries({ queries: enabled.map((target) => estimateQuery(skill, draft, target.id)) });
  const selected = estimates[enabled.findIndex((target) => target.id === draft.targetId)];
  const selectedBlocked = (selected?.data?.blockers.length ?? 0) > 0;

  if (enabled.length === 0) {
    return (
      <div className="space-y-4">
        <EmptyState
          icon={CpuIcon}
          title="No compute target is available"
          description="An admin needs to add or enable a compute target."
          action={
            <Button asChild variant="outline" size="sm">
              <Link href="/compute">Go to compute</Link>
            </Button>
          }
        />
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <RadioGroup
        value={draft.targetId ?? ""}
        onValueChange={onSelect}
        className="grid grid-cols-1 gap-3 lg:grid-cols-2"
        aria-label="Compute target"
      >
        {enabled.map((target, index) => {
          const estimate = estimates[index];
          const id = `target-${target.id}`;
          const blocked = (estimate?.data?.blockers.length ?? 0) > 0;
          return (
            <Label
              key={target.id}
              htmlFor={id}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-lg border p-4 font-normal transition-colors",
                draft.targetId === target.id && !blocked && "border-primary bg-muted/40",
                blocked && "cursor-not-allowed border-dashed",
              )}
            >
              <RadioGroupItem value={target.id} id={id} disabled={blocked} className="mt-0.5" />
              <div className="min-w-0 flex-1 space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">{target.name}</span>
                  <Badge variant="outline">{COMPUTE_KIND_LABELS[target.kind]}</Badge>
                  {target.gpu_label && <span className="text-xs text-muted-foreground">{target.gpu_label}</span>}
                  <StatusBadge domain="health" status={target.health.status} className="ml-auto" />
                </div>
                <div className="text-xs text-muted-foreground">
                  {target.usage.quota_left_hours === null
                    ? "No weekly quota"
                    : `${formatNumber(target.usage.quota_left_hours, 1)} GPU-h quota left this week`}
                  {` · ${formatNumber(target.usage.active_stages)} active stages`}
                </div>
                {estimate?.isPending ? (
                  <Skeleton className="h-9" />
                ) : estimate?.isError ? (
                  <p className="text-xs text-status-danger">{describeError(toErrorInfo(estimate.error))}</p>
                ) : estimate?.data ? (
                  <>
                    <EstimateSummary estimate={estimate.data} />
                    {estimate.data.needs_approval && (
                      <p className="flex gap-1.5 text-xs text-status-warning">
                        <ShieldAlertIcon className="mt-px size-3.5 shrink-0" aria-hidden />
                        Needs launch approval
                        {estimate.data.reasons.length > 0 && `: ${estimate.data.reasons.join("; ")}`}
                      </p>
                    )}
                  </>
                ) : null}
              </div>
            </Label>
          );
        })}
        {disabled.map((target) => (
          <div key={target.id} className="flex items-start gap-3 rounded-lg border border-dashed p-4 opacity-60">
            <RadioGroupItem value={target.id} disabled aria-label={`${target.name} (disabled)`} className="mt-0.5" />
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium">{target.name}</span>
                <Badge variant="outline">{COMPUTE_KIND_LABELS[target.kind]}</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Disabled by an admin{target.has_secret ? "" : " · no credentials set"}
              </p>
            </div>
          </div>
        ))}
      </RadioGroup>
      <div className="flex justify-between gap-2">
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
        <Button onClick={onNext} disabled={!draft.targetId || selectedBlocked}>
          Continue
        </Button>
      </div>
    </div>
  );
}
