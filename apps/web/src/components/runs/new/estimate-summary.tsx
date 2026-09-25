import { BanIcon, TriangleAlertIcon } from "lucide-react";

import type { Estimate } from "@/lib/api/types";
import { formatCost, formatMinutes, formatNumber } from "@/lib/format";

export function EstimateSummary({ estimate }: { estimate: Estimate }) {
  return (
    <div className="space-y-2">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
        <div>
          <dt className="text-muted-foreground">Training</dt>
          <dd className="tabular font-medium">{formatMinutes(estimate.train_minutes)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Total</dt>
          <dd className="tabular font-medium">{formatMinutes(estimate.total_minutes)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">GPU-hours</dt>
          <dd className="tabular font-medium">{formatNumber(estimate.gpu_hours, 2)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Cost</dt>
          <dd className="tabular font-medium">{formatCost(estimate.cost)}</dd>
        </div>
      </dl>
      {estimate.blockers.length > 0 && (
        <ul className="space-y-1">
          {estimate.blockers.map((blocker) => (
            <li key={blocker} className="flex gap-1.5 text-xs text-status-danger">
              <BanIcon className="mt-px size-3.5 shrink-0" aria-hidden /> {blocker}
            </li>
          ))}
        </ul>
      )}
      {estimate.warnings.length > 0 && (
        <ul className="space-y-1">
          {estimate.warnings.map((warning) => (
            <li key={warning} className="flex gap-1.5 text-xs text-status-warning">
              <TriangleAlertIcon className="mt-px size-3.5 shrink-0" aria-hidden /> {warning}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
