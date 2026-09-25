import Link from "next/link";

import { DateTime } from "@/components/common/date-time";
import { StatusBadge } from "@/components/common/status-badge";
import { RunStatusCell } from "@/components/runs/run-status-cell";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { RunSummary } from "@/lib/api/types";
import { formatCost, formatNumber } from "@/lib/format";
import { formatHeadline } from "@/lib/headline";
import { cn } from "cn";

// Priority columns: on a phone the table keeps run, status and gate; the rest appear as width allows.
const SMALL_UP = "hidden sm:table-cell";
const MEDIUM_UP = "hidden md:table-cell";
const LARGE_ONLY = "hidden lg:table-cell";

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
            {!compact && <TableHead className={LARGE_ONLY}>Target</TableHead>}
            {!compact && <TableHead className={LARGE_ONLY}>Created by</TableHead>}
            <TableHead className={MEDIUM_UP}>Created</TableHead>
            <TableHead className={cn("text-right", SMALL_UP)}>GPU-h</TableHead>
            {!compact && <TableHead className={cn("text-right", LARGE_ONLY)}>Cost</TableHead>}
            <TableHead>Gate</TableHead>
            {headlineColumns ? (
              headlineColumns.map((label) => (
                <TableHead key={label} className={cn("text-right", MEDIUM_UP)}>
                  {label}
                </TableHead>
              ))
            ) : (
              <TableHead className={MEDIUM_UP}>Headline</TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
              <TableCell className="max-w-[45vw] sm:max-w-[260px]">
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
              {!compact && (
                <TableCell className={cn("text-muted-foreground", LARGE_ONLY)}>
                  {run.compute_target?.name ?? "—"}
                </TableCell>
              )}
              {!compact && (
                <TableCell className={cn("text-muted-foreground", LARGE_ONLY)}>{run.created_by.name}</TableCell>
              )}
              <TableCell className={cn("text-muted-foreground", MEDIUM_UP)}>
                <DateTime value={run.created_at} />
              </TableCell>
              <TableCell className={cn("text-right", SMALL_UP)}>{formatNumber(run.gpu_hours, 1)}</TableCell>
              {!compact && <TableCell className={cn("text-right", LARGE_ONLY)}>{formatCost(run.cost)}</TableCell>}
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
                    <TableCell key={label} className={cn("text-right", MEDIUM_UP)}>
                      {item ? formatHeadline(item) : "—"}
                    </TableCell>
                  );
                })
              ) : (
                <TableCell className={cn("text-xs whitespace-nowrap text-muted-foreground", MEDIUM_UP)}>
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
