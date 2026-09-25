import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { GateCriterion } from "@/lib/api/types";
import { formatValue } from "@/lib/format";

/** Criterion label with its metric path underneath, so the table fits a narrow column. */
export function CriterionLabel({ criterion }: { criterion: Pick<GateCriterion, "label" | "metric"> }) {
  return (
    <div className="min-w-0 space-y-0.5 whitespace-normal">
      <div>{criterion.label}</div>
      <div className="font-mono text-xs break-all text-muted-foreground">{criterion.metric}</div>
    </div>
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
        {criteria.map((criterion) => (
          <TableRow key={criterion.metric}>
            <TableCell>
              <CriterionLabel criterion={criterion} />
            </TableCell>
            <TableCell className="text-right align-top font-mono text-xs whitespace-nowrap">
              {criterion.op} {formatValue(criterion.value)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
