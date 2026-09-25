import { cellKey, type EvalHeatmap } from "@/lib/evaluation";
import { formatNumber, formatPercent } from "@/lib/format";

/** Tracking error per (speed, requested step length) cell; darker = larger error. */
export function EvaluationHeatmap({ heatmap }: { heatmap: EvalHeatmap }) {
  const values = Object.values(heatmap.values);
  const max = Math.max(...values, 1e-9);
  return (
    <figure className="space-y-2">
      <figcaption className="text-sm font-medium">
        {heatmap.valueLabel} by {heatmap.yKey} × {heatmap.xKey}
      </figcaption>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="p-1.5 text-left font-medium text-muted-foreground">
                {heatmap.yKey} \ {heatmap.xKey}
              </th>
              {heatmap.xs.map((x) => (
                <th key={x} className="p-1.5 text-center font-medium text-muted-foreground">
                  {formatNumber(x, 2)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {heatmap.ys.map((y) => (
              <tr key={y}>
                <th scope="row" className="p-1.5 text-left font-medium text-muted-foreground">
                  {formatNumber(y, 2)}
                </th>
                {heatmap.xs.map((x) => {
                  const value = heatmap.values[cellKey(x, y)];
                  const fallRate = heatmap.fallRates[cellKey(x, y)];
                  const intensity = value === undefined ? 0 : Math.round((value / max) * 70) + 8;
                  return (
                    <td
                      key={x}
                      className="min-w-16 rounded-sm border border-transparent p-1.5 text-center"
                      style={{ background: `color-mix(in oklch, var(--chart-1) ${intensity}%, transparent)` }}
                      title={fallRate !== undefined ? `Falls: ${formatPercent(fallRate)}` : undefined}
                    >
                      <span className="tabular font-medium">{value === undefined ? "—" : formatNumber(value, 1)}</span>
                      {fallRate !== undefined && fallRate > 0 && (
                        <span className="block text-[10px] text-status-danger">{formatPercent(fallRate)} fell</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
