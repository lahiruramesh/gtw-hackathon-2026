import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { StairHeight } from "@/lib/evaluation";
import { formatNumber, formatPercent } from "@/lib/format";
import { cn } from "cn";

/** Stairs strict test per step height: what crossed, what fell, and how close to the stability limits. */
export function StairsHeightTable({ heights }: { heights: StairHeight[] }) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">By step height</h4>
      <div className="overflow-x-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Rise</TableHead>
              <TableHead className="text-right">Crossed</TableHead>
              <TableHead className="text-right">Falls</TableHead>
              <TableHead className="text-right">Stable</TableHead>
              <TableHead className="text-right">Max tilt</TableHead>
              <TableHead className="text-right">Min pelvis</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {heights.map((height) => (
              <TableRow key={height.riseCm} className={cn(height.certified && "bg-status-success/10")}>
                <TableCell className="tabular font-medium whitespace-nowrap">
                  {formatNumber(height.riseCm, 1)} cm
                  {height.certified && (
                    <Badge variant="outline" className="ml-2">
                      certified
                    </Badge>
                  )}
                </TableCell>
                <TableCell className="tabular text-right whitespace-nowrap">
                  {height.crossed}/{height.runs}{" "}
                  <span className="text-muted-foreground">({formatPercent(height.crossed / height.runs)})</span>
                </TableCell>
                <TableCell className={cn("tabular text-right", height.fell > 0 && "text-status-danger")}>
                  {height.fell}
                </TableCell>
                <TableCell className="text-right">
                  {height.stable === null ? "—" : height.stable ? "Yes" : "No"}
                </TableCell>
                <TableCell className="tabular text-right">
                  {height.maxTiltDeg === null ? "—" : `${formatNumber(height.maxTiltDeg, 1)}°`}
                </TableCell>
                <TableCell className="tabular text-right">
                  {height.minPelvisM === null ? "—" : `${formatNumber(height.minPelvisM, 2)} m`}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
