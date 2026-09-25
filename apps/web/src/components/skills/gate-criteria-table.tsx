import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { GateCriterion } from "@/lib/api/types";
import { formatValue } from "@/lib/format";

export function GateCriteriaTable({ criteria }: { criteria: GateCriterion[] }) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Criterion</TableHead>
            <TableHead>Metric</TableHead>
            <TableHead className="text-right">Threshold</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {criteria.map((criterion) => (
            <TableRow key={criterion.metric}>
              <TableCell>{criterion.label}</TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground">{criterion.metric}</TableCell>
              <TableCell className="text-right font-mono text-xs whitespace-nowrap">
                {criterion.op} {formatValue(criterion.value)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
