import { FlaskConicalIcon } from "lucide-react";

import { DateTime } from "@/components/common/date-time";
import { EmptyState } from "@/components/common/empty-state";
import { JsonView } from "@/components/common/json-view";
import { EvaluationHeatmap } from "@/components/runs/evaluation-heatmap";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Evaluation } from "@/lib/api/types";
import { describeEvaluation, formatEvalValue, type EvalTable } from "@/lib/evaluation";
import { formatCompact, humanizeKey } from "@/lib/format";

function EvalTableView({ table }: { table: EvalTable }) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">{humanizeKey(table.key)}</h4>
      <div className="overflow-x-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {table.columns.map((column) => (
                <TableHead key={column} className={column === "case" ? undefined : "text-right"}>
                  {humanizeKey(column)}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {table.rows.map((row) => (
              <TableRow key={row.id}>
                {table.columns.map((column) => (
                  <TableCell key={column} className={column === "case" ? "font-mono text-xs" : "text-right"}>
                    {formatEvalValue(column, row.cells[column])}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function EvaluationCard({ evaluation }: { evaluation: Evaluation }) {
  const view = describeEvaluation(evaluation.summary);
  const empty =
    view.scalars.length + view.tables.length + view.details.length === 0 &&
    !view.heatmap &&
    Object.keys(view.rest).length === 0;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <span className="font-mono">{evaluation.stage_key}</span>
          <Badge variant="outline">{evaluation.suite}</Badge>
          {evaluation.checkpoint?.step != null && (
            <Badge variant="secondary">checkpoint @ {formatCompact(evaluation.checkpoint.step)} steps</Badge>
          )}
        </CardTitle>
        <CardDescription>
          Evaluated <DateTime value={evaluation.created_at} />
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {empty && <p className="text-sm text-muted-foreground">The summary is empty.</p>}
        {view.scalars.length > 0 && (
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {view.scalars.map((scalar) => (
              <div key={scalar.key} className="rounded-md border p-2">
                <dt className="truncate text-xs text-muted-foreground" title={scalar.key}>
                  {humanizeKey(scalar.key)}
                </dt>
                <dd className="tabular text-sm font-semibold">{formatEvalValue(scalar.key, scalar.value)}</dd>
              </div>
            ))}
          </dl>
        )}
        {view.heatmap && <EvaluationHeatmap heatmap={view.heatmap} />}
        {view.tables.map((table) => (
          <EvalTableView key={table.key} table={table} />
        ))}
        {view.details.map((block) => (
          <div key={block.key} className="space-y-2">
            <h4 className="text-sm font-medium">{humanizeKey(block.key)}</h4>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
              {block.entries.map((entry) => (
                <div key={entry.key} className="flex justify-between gap-2 border-b py-1">
                  <dt className="text-muted-foreground">{humanizeKey(entry.key)}</dt>
                  <dd className="tabular">{formatEvalValue(entry.key, entry.value)}</dd>
                </div>
              ))}
            </dl>
          </div>
        ))}
        {Object.keys(view.rest).length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Other data</h4>
            <JsonView value={view.rest} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function EvaluationPanel({ evaluations }: { evaluations: Evaluation[] }) {
  if (evaluations.length === 0) {
    return (
      <EmptyState
        icon={FlaskConicalIcon}
        title="No evaluations yet"
        description="The evaluate stage runs after training. You can also evaluate any checkpoint from the Checkpoints tab."
      />
    );
  }
  const newestFirst = [...evaluations].sort((a, b) => b.created_at.localeCompare(a.created_at));
  return (
    <div className="space-y-4">
      {newestFirst.map((evaluation) => (
        <EvaluationCard key={evaluation.id} evaluation={evaluation} />
      ))}
    </div>
  );
}
