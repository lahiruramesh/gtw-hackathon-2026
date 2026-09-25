"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { CHART_COLORS, mergeSeries, type ChartSeries } from "@/lib/chart-data";
import { formatCompact, formatMetric } from "@/lib/format";

interface MetricLineChartProps {
  series: readonly ChartSeries[];
  className?: string;
}

export function MetricLineChart({ series, className }: MetricLineChartProps) {
  const config: ChartConfig = Object.fromEntries(
    series.map((item, index) => [item.id, { label: item.label, color: CHART_COLORS[index % CHART_COLORS.length] }]),
  );
  const data = mergeSeries(series);
  return (
    <ChartContainer config={config} className={className ?? "aspect-auto h-56 w-full"}>
      <LineChart data={data} margin={{ left: 4, right: 12, top: 8, bottom: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="step"
          type="number"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(value: number) => formatCompact(value)}
          tickLine={false}
          axisLine={false}
          minTickGap={24}
        />
        <YAxis
          width={52}
          tickFormatter={(value: number) => formatMetric(value)}
          tickLine={false}
          axisLine={false}
          domain={["auto", "auto"]}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelFormatter={(_, payload) => `Step ${formatCompact(Number(payload?.[0]?.payload?.step))}`}
              formatter={(value, name) => (
                <div className="flex w-full justify-between gap-4">
                  <span className="text-muted-foreground">{config[String(name)]?.label ?? name}</span>
                  <span className="tabular font-mono">{formatMetric(Number(value))}</span>
                </div>
              )}
            />
          }
        />
        {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
        {series.map((item) => (
          <Line
            key={item.id}
            dataKey={item.id}
            type="monotone"
            stroke={`var(--color-${item.id})`}
            strokeWidth={1.75}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartContainer>
  );
}
