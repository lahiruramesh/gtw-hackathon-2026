import Link from "next/link";

import { DateTime } from "@/components/common/date-time";
import { StatusBadge } from "@/components/common/status-badge";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { RunDetail } from "@/lib/api/types";
import { formatCompact, formatCost, formatMinutes, formatNumber, formatValue, shortSha } from "@/lib/format";
import type { FieldSpec } from "@/lib/form-fields";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 border-b py-1.5 text-sm last:border-0">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="text-right">{children}</dd>
    </div>
  );
}

export function ConfigPanel({ run, fields }: { run: RunDetail; fields: FieldSpec[] }) {
  const defaults = new Map(fields.map((field) => [field.name, field.defaultValue]));
  const names = [...new Set([...defaults.keys(), ...Object.keys(run.params)])];
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Parameters</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Parameter</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead className="text-right">Default</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {names.map((name) => {
                const fallback = defaults.get(name);
                const value = name in run.params ? run.params[name] : fallback;
                const changed =
                  name in run.params && JSON.stringify(run.params[name]) !== JSON.stringify(fallback ?? null);
                return (
                  <TableRow key={name}>
                    <TableCell>
                      <span className="font-mono text-xs">{name}</span>
                      {changed && (
                        <Badge variant="secondary" className="ml-2">
                          Changed
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">{formatValue(value ?? null)}</TableCell>
                    <TableCell className="text-right font-mono text-xs text-muted-foreground">
                      {formatValue(fallback ?? null)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Run</CardTitle>
          </CardHeader>
          <CardContent>
            <dl>
              <Row label="Preset">{run.preset_id ?? "Custom"}</Row>
              <Row label="Commit">
                <span className="font-mono text-xs">{shortSha(run.git_sha)}</span>
              </Row>
              <Row label="Started">
                <DateTime value={run.started_at} />
              </Row>
              <Row label="Finished">
                <DateTime value={run.finished_at} />
              </Row>
              {run.launch_decided_by && (
                <Row label="Launch approved by">
                  {run.launch_decided_by.name} · <DateTime value={run.launch_decided_at} />
                </Row>
              )}
              {run.notes && <Row label="Notes">{run.notes}</Row>}
            </dl>
          </CardContent>
        </Card>
        {run.estimate && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Estimate at launch</CardTitle>
            </CardHeader>
            <CardContent>
              <dl>
                <Row label="Training">{formatMinutes(run.estimate.train_minutes)}</Row>
                <Row label="Total">{formatMinutes(run.estimate.total_minutes)}</Row>
                <Row label="GPU-hours">{formatNumber(run.estimate.gpu_hours, 2)}</Row>
                <Row label="Cost">{formatCost(run.estimate.cost)}</Row>
                <Row label="Needed approval">{run.estimate.needs_approval ? "Yes" : "No"}</Row>
              </dl>
            </CardContent>
          </Card>
        )}
        {(run.parent || run.children.length > 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Lineage</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {run.parent && (
                <div>
                  <div className="text-xs text-muted-foreground">Warm start from</div>
                  <Link href={`/runs/${run.parent.run.id}`} className="font-medium hover:underline">
                    {run.parent.run.name}
                  </Link>
                  {run.parent.checkpoint && (
                    <span className="text-muted-foreground">
                      {" "}
                      · {run.parent.checkpoint.name}
                      {run.parent.checkpoint.step !== null && ` @ ${formatCompact(run.parent.checkpoint.step)} steps`}
                    </span>
                  )}
                </div>
              )}
              {run.children.length > 0 && (
                <div>
                  <div className="text-xs text-muted-foreground">Warm-started runs</div>
                  <ul className="space-y-1">
                    {run.children.map((child) => (
                      <li key={child.id} className="flex items-center gap-2">
                        <Link href={`/runs/${child.id}`} className="hover:underline">
                          {child.name}
                        </Link>
                        <StatusBadge domain="run" status={child.status} />
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
