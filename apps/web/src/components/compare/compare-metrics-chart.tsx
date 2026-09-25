"use client";

import { useQueries } from "@tanstack/react-query";
import { useState } from "react";

import { MetricLineChart } from "@/components/charts/metric-line-chart";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { clientFetch } from "@/lib/api/client";
import type { MetricsResponse, RunRef } from "@/lib/api/types";

interface CompareMetricsChartProps {
  runs: RunRef[];
  metricKeys: string[];
}

export function CompareMetricsChart({ runs, metricKeys }: CompareMetricsChartProps) {
  const [metricKey, setMetricKey] = useState(metricKeys[0] ?? "");
  const results = useQueries({
    queries: runs.map((run) => ({
      queryKey: ["run-metrics", run.id, metricKey],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        clientFetch<MetricsResponse>(`/runs/${encodeURIComponent(run.id)}/metrics`, {
          query: { keys: metricKey },
          signal,
        }),
      enabled: metricKey !== "",
    })),
  });

  if (metricKeys.length === 0) {
    return <p className="text-sm text-muted-foreground">These runs share no training metrics.</p>;
  }

  const loading = results.some((result) => result.isPending);
  const series = runs.map((run, index) => ({
    id: `run${index}`,
    label: run.name,
    points: results[index]?.data?.series.find((item) => item.key === metricKey)?.points ?? [],
  }));

  return (
    <div className="space-y-3">
      <Select value={metricKey} onValueChange={setMetricKey}>
        <SelectTrigger size="sm" className="w-full font-mono text-xs sm:w-80" aria-label="Metric">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {metricKeys.map((key) => (
            <SelectItem key={key} value={key} className="font-mono text-xs">
              {key}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {loading ? (
        <Skeleton className="h-72" />
      ) : (
        <MetricLineChart series={series} className="aspect-auto h-72 w-full" />
      )}
      {results.some((result) => result.isError) && (
        <p className="text-xs text-status-danger">Some runs&apos; metrics couldn&apos;t be loaded.</p>
      )}
    </div>
  );
}
