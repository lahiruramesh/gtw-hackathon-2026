import { CheckIcon, GitCompareIcon, MinusIcon, XIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { StatusBadge } from "@/components/common/status-badge";
import { CompareMetricsChart } from "@/components/compare/compare-metrics-chart";
import { CompareRunPicker } from "@/components/compare/compare-run-picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toResult } from "@/lib/api/errors";
import { MAX_COMPARE } from "@/lib/compare";
import { apiFetch } from "@/lib/api/server";
import type { CompareResponse, RunPage } from "@/lib/api/types";
import { formatHeadline } from "@/lib/headline";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";
import { LevelHeaderRow } from "@/components/skills/gate-criteria-table";
import { groupByLevel } from "@/lib/gate-levels";

export const metadata: Metadata = { title: "Compare" };

function RunColumns({ runs }: { runs: CompareResponse["runs"] }) {
  return runs.map((run) => (
    <TableHead key={run.id} className="min-w-36 text-right">
      <Link href={`/runs/${run.id}`} className="hover:underline">
        {run.name}
      </Link>
    </TableHead>
  ));
}

function Comparison({ data }: { data: CompareResponse }) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Headline metrics</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <RunColumns runs={data.runs} />
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="text-muted-foreground">Status</TableCell>
                {data.runs.map((run) => (
                  <TableCell key={run.id} className="text-right">
                    <StatusBadge domain="run" status={run.status} />
                  </TableCell>
                ))}
              </TableRow>
              {data.headline_rows.map((row) => (
                <TableRow key={row.label}>
                  <TableCell>{row.label}</TableCell>
                  {row.values.map((value, index) => (
                    <TableCell key={data.runs[index]?.id ?? index} className="text-right">
                      {formatHeadline({ label: row.label, unit: row.unit, value })}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      {data.gate_rows.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Gates</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Criterion</TableHead>
                  <RunColumns runs={data.runs} />
                </TableRow>
              </TableHeader>
              <TableBody>
                {groupByLevel(data.gate_rows).map(({ level, items }) => [
                  <LevelHeaderRow key={level} level={level} colSpan={data.runs.length + 1} />,
                  ...items.map((row) => (
                    <TableRow key={row.label}>
                      <TableCell>{row.label}</TableCell>
                      {row.values.map((passed, index) => (
                        <TableCell key={data.runs[index]?.id ?? index} className="text-right">
                          {passed === null ? (
                            <span
                              className="inline-flex items-center gap-1 text-xs text-muted-foreground"
                              title="This run has no value for the criterion's metric"
                            >
                              <MinusIcon className="size-4" aria-hidden /> not measured
                            </span>
                          ) : passed ? (
                            <CheckIcon className="ml-auto size-4 text-status-success" aria-label="Passed" />
                          ) : (
                            <XIcon className="ml-auto size-4 text-status-danger" aria-label="Failed" />
                          )}
                        </TableCell>
                      ))}
                    </TableRow>
                  )),
                ])}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Training curves</CardTitle>
        </CardHeader>
        <CardContent>
          <CompareMetricsChart
            runs={data.runs.map(({ id, name, status }) => ({ id, name, status }))}
            metricKeys={data.metric_keys}
          />
        </CardContent>
      </Card>
    </div>
  );
}

export default async function ComparePage({ searchParams }: PageProps<"/compare">) {
  const viewer = await requireViewer();
  if (!can(viewer, "run:read")) return <NoAccess what="view runs" />;

  const { runs: runsParam } = await searchParams;
  const selected = (typeof runsParam === "string" ? runsParam.split(",") : []).filter(Boolean).slice(0, MAX_COMPARE);
  const [options, comparison] = await Promise.all([
    toResult(apiFetch<RunPage>("/runs", { query: { limit: 200 } })),
    selected.length >= 2
      ? toResult(apiFetch<CompareResponse>("/compare", { query: { run_ids: selected } }))
      : Promise.resolve(null),
  ]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Compare runs"
        description="Headline metrics, gate criteria and training curves side by side."
      />
      {options.ok ? (
        <CompareRunPicker
          options={options.data.items.map((run) => ({ id: run.id, name: run.name, skillName: run.skill_name }))}
          selected={selected}
        />
      ) : (
        <ApiErrorState error={options.error} subject="runs" />
      )}
      {comparison === null ? (
        <EmptyState
          icon={GitCompareIcon}
          title="Pick at least two runs"
          description="For example the step-length runs with and without domain randomisation, or stairs v9, v10 and v11."
        />
      ) : comparison.ok ? (
        <Comparison data={comparison.data} />
      ) : (
        <ApiErrorState error={comparison.error} subject="the comparison" />
      )}
    </div>
  );
}
