import type { ReactNode } from "react";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { GateCriterion, GateLevel } from "@/lib/api/types";
import { formatValue } from "@/lib/format";
import { GATE_LEVELS, groupByLevel } from "@/lib/gate-levels";

/** Criterion label with its metric path underneath, so the table fits a narrow column. */
export function CriterionLabel({ criterion }: { criterion: Pick<GateCriterion, "label" | "metric"> }) {
  return (
    <div className="min-w-0 space-y-0.5 whitespace-normal">
      <div>{criterion.label}</div>
      <div className="font-mono text-xs break-all text-muted-foreground">{criterion.metric}</div>
    </div>
  );
}

/** A section row naming a gate level, with an optional verdict on the right. */
export function LevelHeaderRow({
  level,
  colSpan,
  children,
}: {
  level: GateLevel;
  colSpan: number;
  children?: ReactNode;
}) {
  return (
    <TableRow className="bg-muted/40 hover:bg-muted/40">
      <TableCell colSpan={colSpan} className="py-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-sm font-medium">{GATE_LEVELS[level].label}</div>
            <div className="text-xs whitespace-normal text-muted-foreground">{GATE_LEVELS[level].description}</div>
          </div>
          {children}
        </div>
      </TableCell>
    </TableRow>
  );
}

export function GateCriteriaTable({ criteria }: { criteria: GateCriterion[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Criterion</TableHead>
          <TableHead className="text-right">Threshold</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {groupByLevel(criteria).map(({ level, items }) => [
          <LevelHeaderRow key={level} level={level} colSpan={2} />,
          ...items.map((criterion) => (
            <TableRow key={criterion.metric}>
              <TableCell>
                <CriterionLabel criterion={criterion} />
              </TableCell>
              <TableCell className="text-right align-top font-mono text-xs whitespace-nowrap">
                {criterion.op} {formatValue(criterion.value)}
              </TableCell>
            </TableRow>
          )),
        ])}
      </TableBody>
    </Table>
  );
}
