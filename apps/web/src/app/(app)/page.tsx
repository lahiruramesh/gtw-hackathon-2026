import { ActivityIcon, CheckCheckIcon, ClockIcon, DollarSignIcon, GaugeIcon, PlusIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { MetricCard } from "@/components/common/metric-card";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { StatusBadge } from "@/components/common/status-badge";
import { TargetSummaryCard } from "@/components/compute/target-summary-card";
import { RunsTable } from "@/components/runs/runs-table";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { Dashboard } from "@/lib/api/types";
import { formatCost, formatNumber, formatPercent } from "@/lib/format";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Dashboard" };

export default async function DashboardPage() {
  const viewer = await requireViewer();
  if (!can(viewer, "run:read")) return <NoAccess what="view runs" />;

  const result = await toResult(apiFetch<Dashboard>("/dashboard"));
  const canLaunch = can(viewer, "run:create_preset");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        description="Training activity, compute usage and releases across all skills."
        actions={
          canLaunch && (
            <Button asChild size="sm">
              <Link href="/runs/new">
                <PlusIcon /> New run
              </Link>
            </Button>
          )
        }
      />
      {result.ok ? (
        <DashboardBody dashboard={result.data} canLaunch={canLaunch} />
      ) : (
        <ApiErrorState error={result.error} subject="the dashboard" />
      )}
    </div>
  );
}

function DashboardBody({ dashboard, canLaunch }: { dashboard: Dashboard; canLaunch: boolean }) {
  const awaiting = dashboard.awaiting_launch_approval + dashboard.awaiting_review;
  return (
    <>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <MetricCard label="Active runs" value={formatNumber(dashboard.active_runs)} icon={ActivityIcon} />
        <MetricCard label="GPU-hours, 7 days" value={formatNumber(dashboard.gpu_hours_7d, 1)} icon={ClockIcon} />
        <MetricCard label="Cost, 7 days" value={formatCost(dashboard.cost_7d)} icon={DollarSignIcon} />
        <MetricCard
          label="Awaiting decision"
          value={
            <Link href="/approvals" className="hover:underline">
              {formatNumber(awaiting)}
            </Link>
          }
          hint={`${dashboard.awaiting_launch_approval} launch · ${dashboard.awaiting_review} release`}
          icon={CheckCheckIcon}
        />
        <MetricCard
          label="Gate pass rate, 30 days"
          value={formatPercent(dashboard.gate_pass_rate_30d)}
          hint={dashboard.gate_pass_rate_30d === null ? "No gated runs yet" : undefined}
          icon={GaugeIcon}
        />
      </div>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold">Compute targets</h2>
        {dashboard.targets.length === 0 ? (
          <EmptyState
            title="No compute targets"
            description="An admin needs to add a compute target before runs can start."
            action={
              <Button asChild variant="outline" size="sm">
                <Link href="/compute">Go to compute</Link>
              </Button>
            }
          />
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {dashboard.targets.map((target) => (
              <TargetSummaryCard key={target.id} target={target} />
            ))}
          </div>
        )}
      </section>

      <Card className="gap-3">
        <CardHeader>
          <CardTitle className="text-sm">Recent runs</CardTitle>
          <CardAction>
            <Button asChild variant="ghost" size="sm">
              <Link href="/runs">All runs</Link>
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>
          {dashboard.recent_runs.length === 0 ? (
            <EmptyState
              title="No runs yet"
              description="Launch a smoke run to check the whole pipeline end to end."
              action={
                canLaunch && (
                  <Button asChild size="sm">
                    <Link href="/runs/new">New run</Link>
                  </Button>
                )
              }
            />
          ) : (
            <RunsTable runs={dashboard.recent_runs} compact />
          )}
        </CardContent>
      </Card>

      <Card className="gap-3">
        <CardHeader>
          <CardTitle className="text-sm">Skills</CardTitle>
          <CardAction>
            <Button asChild variant="ghost" size="sm">
              <Link href="/skills">All skills</Link>
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          {dashboard.skills.length === 0 ? (
            <EmptyState
              title="No skills registered"
              description="Skills are loaded from skills/*/skill.yaml in the repository."
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Skill</TableHead>
                  <TableHead className="text-right">Runs</TableHead>
                  <TableHead className="text-right">GPU-h total</TableHead>
                  <TableHead>Best run</TableHead>
                  <TableHead>Latest gate</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dashboard.skills.map((skill) => (
                  <TableRow key={skill.id}>
                    <TableCell>
                      <Link href={`/skills/${skill.id}`} className="font-medium hover:underline">
                        {skill.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-right">{formatNumber(skill.runs)}</TableCell>
                    <TableCell className="text-right">{formatNumber(skill.gpu_hours_total, 1)}</TableCell>
                    <TableCell>
                      {skill.best_run ? (
                        <Link href={`/runs/${skill.best_run.id}`} className="hover:underline">
                          {skill.best_run.name}
                        </Link>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {skill.latest_verdict ? (
                        <StatusBadge domain="gate" status={skill.latest_verdict} />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </>
  );
}
