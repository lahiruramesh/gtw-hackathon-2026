import { CheckCheckIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { ApprovalDecision } from "@/components/approvals/approval-decision";
import { ApiErrorState } from "@/components/common/api-error-state";
import { DateTime } from "@/components/common/date-time";
import { EmptyState } from "@/components/common/empty-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { StatusBadge } from "@/components/common/status-badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { Approvals, RunSummary } from "@/lib/api/types";
import { formatHeadline } from "@/lib/headline";
import { requireViewer, type Viewer } from "@/lib/session";
import { can, canAny } from "@/lib/viewer";

export const metadata: Metadata = { title: "Approvals" };

function ApprovalTable({ runs, kind, viewer }: { runs: RunSummary[]; kind: "launch" | "release"; viewer: Viewer }) {
  if (runs.length === 0) {
    return (
      <EmptyState
        icon={CheckCheckIcon}
        title="Nothing waiting"
        description={
          kind === "launch"
            ? "Operator runs over their GPU budget appear here."
            : "Runs that pass their release gate appear here."
        }
      />
    );
  }
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run</TableHead>
            <TableHead>Requested by</TableHead>
            <TableHead>{kind === "launch" ? "Target" : "Gate"}</TableHead>
            <TableHead>{kind === "launch" ? "Created" : "Headline"}</TableHead>
            <TableHead className="text-right">Decision</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
              <TableCell>
                <Link
                  href={`/runs/${run.id}${kind === "release" ? "?tab=gate" : ""}`}
                  className="font-medium hover:underline"
                >
                  {run.name}
                </Link>
                <div className="text-xs text-muted-foreground">
                  {run.skill_name}
                  {run.preset_id ? ` · ${run.preset_id}` : " · custom"}
                </div>
              </TableCell>
              <TableCell className="text-muted-foreground">{run.created_by.name}</TableCell>
              <TableCell>
                {kind === "launch" ? (
                  <span className="text-muted-foreground">{run.compute_target?.name ?? "—"}</span>
                ) : run.gate_verdict ? (
                  <StatusBadge domain="gate" status={run.gate_verdict} />
                ) : (
                  "—"
                )}
              </TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {kind === "launch" ? (
                  <DateTime value={run.created_at} />
                ) : (
                  run.headline.map((item) => `${item.label} ${formatHeadline(item)}`).join(" · ") || "—"
                )}
              </TableCell>
              <TableCell>
                <div className="flex justify-end">
                  <ApprovalDecision
                    kind={kind}
                    runId={run.id}
                    runName={run.name}
                    ownRun={run.created_by.id === viewer.id}
                  />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export default async function ApprovalsPage() {
  const viewer = await requireViewer();
  if (!canAny(viewer, ["release:review", "run:approve_launch"]))
    return <NoAccess what="approve launches or releases" />;

  const result = await toResult(apiFetch<Approvals>("/approvals"));
  return (
    <div className="space-y-6">
      <PageHeader
        title="Approvals"
        description="Launches waiting for approval and releases waiting for a safety review."
      />
      {!result.ok ? (
        <ApiErrorState error={result.error} subject="approvals" />
      ) : (
        <>
          {can(viewer, "run:approve_launch") && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Launches awaiting approval ({result.data.launches.length})</CardTitle>
                <CardDescription>
                  Runs start only after an ML engineer or admin approves them. You can&apos;t approve your own.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ApprovalTable runs={result.data.launches} kind="launch" viewer={viewer} />
              </CardContent>
            </Card>
          )}
          {can(viewer, "release:review") && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Releases awaiting review ({result.data.releases.length})</CardTitle>
                <CardDescription>
                  The release gate passed. Approve to clear the policy for hardware trials; rejecting needs a comment.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ApprovalTable runs={result.data.releases} kind="release" viewer={viewer} />
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
