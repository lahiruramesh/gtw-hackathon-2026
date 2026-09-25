import Link from "next/link";

import { DateTime } from "@/components/common/date-time";
import { StatusBadge } from "@/components/common/status-badge";
import { RunStatusCell } from "@/components/runs/run-status-cell";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { RunSummary } from "@/lib/api/types";
import { formatCost, formatNumber } from "@/lib/format";
import { formatHeadline } from "@/lib/headline";

interface RunsTableProps {
  runs: RunSummary[];
  /** Headline labels to show as separate columns (all runs of one skill); otherwise one combined column. */
  headlineColumns?: string[];
  compact?: boolean;
}

export function RunsTable({ runs, headlineColumns, compact = false }: RunsTableProps) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run</TableHead>
            <TableHead>Status</TableHead>
            {!compact && <TableHead>Target</TableHead>}
            {!compact && <TableHead>Created by</TableHead>}
            <TableHead>Created</TableHead>
            <TableHead className="text-right">GPU-h</TableHead>
            {!compact && <TableHead className="text-right">Cost</TableHead>}
            <TableHead>Gate</TableHead>
            {headlineColumns ? (
              headlineColumns.map((label) => (
                <TableHead key={label} className="text-right">
                  {label}
                </TableHead>
              ))
            ) : (
              <TableHead>Headline</TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
              <TableCell className="max-w-[260px]">
                <Link href={`/runs/${run.id}`} className="block truncate font-medium hover:underline">
                  {run.name}
                </Link>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span className="truncate">{run.skill_name}</span>
                  {run.preset_id && <span className="truncate">· {run.preset_id}</span>}
                  {run.imported && (
                    <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                      Imported
                    </Badge>
                  )}
                </div>
              </TableCell>
              <TableCell>
                <RunStatusCell run={run} />
              </TableCell>
              {!compact && <TableCell className="text-muted-foreground">{run.compute_target?.name ?? "—"}</TableCell>}
              {!compact && <TableCell className="text-muted-foreground">{run.created_by.name}</TableCell>}
              <TableCell className="text-muted-foreground">
                <DateTime value={run.created_at} />
              </TableCell>
              <TableCell className="text-right">{formatNumber(run.gpu_hours, 1)}</TableCell>
              {!compact && <TableCell className="text-right">{formatCost(run.cost)}</TableCell>}
              <TableCell>
                {run.gate_verdict ? (
                  <StatusBadge domain="gate" status={run.gate_verdict} />
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </TableCell>
              {headlineColumns ? (
                headlineColumns.map((label) => {
                  const item = run.headline.find((headline) => headline.label === label);
                  return (
                    <TableCell key={label} className="text-right">
                      {item ? formatHeadline(item) : "—"}
                    </TableCell>
                  );
                })
              ) : (
                <TableCell className="text-xs whitespace-nowrap text-muted-foreground">
                  {run.headline.length === 0
                    ? "—"
                    : run.headline.map((item) => `${item.label} ${formatHeadline(item)}`).join(" · ")}
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
