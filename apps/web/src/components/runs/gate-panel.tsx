import { CheckIcon, ShieldCheckIcon, XIcon } from "lucide-react";

import { DateTime } from "@/components/common/date-time";
import { EmptyState } from "@/components/common/empty-state";
import { StatusBadge } from "@/components/common/status-badge";
import { GateLevelBadges } from "@/components/runs/gate-level-badges";
import { ReleaseReview } from "@/components/runs/release-review";
import { CriterionLabel, GateCriteriaTable, LevelHeaderRow } from "@/components/skills/gate-criteria-table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { GateCriterion, RunDetail } from "@/lib/api/types";
import { formatMetric, formatValue } from "@/lib/format";
import { groupByLevel } from "@/lib/gate-levels";

interface GatePanelProps {
  run: RunDetail;
  criteria: GateCriterion[];
  viewerId: string;
}

export function GatePanel({ run, criteria, viewerId }: GatePanelProps) {
  const gate = run.gate;
  if (!gate) {
    if (criteria.length === 0) {
      return <EmptyState icon={ShieldCheckIcon} title="No gates" description="This skill defines no gate criteria." />;
    }
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Not evaluated yet</CardTitle>
          <CardDescription>The gate runs automatically after the evaluation stage. It will check:</CardDescription>
        </CardHeader>
        <CardContent>
          <GateCriteriaTable criteria={criteria} />
        </CardContent>
      </Card>
    );
  }

  const canReview = run.permissions.can_review && gate.review_status === "pending";
  const levels = new Map(gate.levels.map((result) => [result.level, result]));
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
            <GateLevelBadges
              long
              simulation={levels.get("simulation")?.verdict ?? null}
              release={levels.get("release")?.verdict ?? null}
            />
            <StatusBadge domain="review" status={gate.review_status} />
          </CardTitle>
          <CardDescription>
            Evaluated <DateTime value={gate.evaluated_at} /> · {gate.criteria.filter((item) => item.passed).length} of{" "}
            {gate.criteria.length} criteria passed
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8">
                  <span className="sr-only">Result</span>
                </TableHead>
                <TableHead>Criterion</TableHead>
                <TableHead className="text-right">Actual</TableHead>
                <TableHead className="text-right">Threshold</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {groupByLevel(gate.criteria).map(({ level, items }) => {
                const result = levels.get(level);
                return [
                  <LevelHeaderRow key={level} level={level} colSpan={4}>
                    {result && (
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        {result.passed} of {result.total} passed
                        <StatusBadge domain="gate" status={result.verdict} />
                      </span>
                    )}
                  </LevelHeaderRow>,
                  ...items.map((criterion) => (
                    <TableRow key={criterion.metric}>
                      <TableCell className="align-top">
                        {criterion.passed ? (
                          <CheckIcon className="size-4 text-status-success" aria-label="Passed" />
                        ) : (
                          <XIcon className="size-4 text-status-danger" aria-label="Failed" />
                        )}
                      </TableCell>
                      <TableCell>
                        <CriterionLabel criterion={criterion} />
                      </TableCell>
                      <TableCell className="text-right align-top font-mono text-xs whitespace-nowrap">
                        {criterion.actual === null ? (
                          <span className="text-status-danger">not measured</span>
                        ) : (
                          formatMetric(criterion.actual)
                        )}
                      </TableCell>
                      <TableCell className="text-right align-top font-mono text-xs whitespace-nowrap">
                        {criterion.op} {formatValue(criterion.value)}
                      </TableCell>
                    </TableRow>
                  )),
                ];
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {(gate.reviewer || canReview) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Release review</CardTitle>
            <CardDescription>
              {gate.reviewer ? (
                <>
                  {gate.reviewer.name} decided <DateTime value={gate.reviewed_at} />
                </>
              ) : (
                "A safety reviewer approves or rejects this policy for hardware trials."
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {gate.comment && <blockquote className="border-l-2 pl-3 text-sm">{gate.comment}</blockquote>}
            {canReview && <ReleaseReview runId={run.id} runName={run.name} ownRun={run.created_by.id === viewerId} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
