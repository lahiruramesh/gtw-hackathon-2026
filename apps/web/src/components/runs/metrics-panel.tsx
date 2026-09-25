"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { LineChartIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { MetricLineChart } from "@/components/charts/metric-line-chart";
import { MetricKeyPicker } from "@/components/runs/metric-key-picker";
import { useRunLive } from "@/components/runs/run-live-provider";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { clientFetch } from "@/lib/api/client";
import { toErrorInfo } from "@/lib/api/errors";
import type { MetricsResponse } from "@/lib/api/types";
import { formatMetric, humanizeKey } from "@/lib/format";
import { appendMetricPoint, defaultMetricKeys, orderSelection } from "@/lib/metrics";

interface MetricsPanelProps {
  primary: string;
  declaredKeys: string[];
}

export function MetricsPanel({ primary, declaredKeys }: MetricsPanelProps) {
  const { run, subscribe } = useRunLive();
  const queryClient = useQueryClient();
  const queryKey = ["run-metrics", run.id];
  const metrics = useQuery({
    queryKey,
    queryFn: ({ signal }) => clientFetch<MetricsResponse>(`/runs/${encodeURIComponent(run.id)}/metrics`, { signal }),
  });
  const [picked, setPicked] = useState<string[] | null>(null);

  useEffect(
    () =>
      subscribe((event) => {
        if (event.type !== "metric") return;
        queryClient.setQueryData<MetricsResponse>(["run-metrics", run.id], (current) =>
          current ? appendMetricPoint(current, event.data.step, event.data.values) : current,
        );
      }),
    [subscribe, queryClient, run.id],
  );

  if (metrics.isPending) {
    return (
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Skeleton className="h-72" />
        <Skeleton className="h-72" />
      </div>
    );
  }
  if (metrics.isError) return <ApiErrorState error={toErrorInfo(metrics.error)} subject="metrics" />;

  const { keys, series } = metrics.data;
  if (keys.length === 0) {
    return (
      <EmptyState
        icon={LineChartIcon}
        title="No metrics yet"
        description="Training metrics appear here after the first evaluation step of the train stage."
      />
    );
  }

  const selected = orderSelection(picked ?? defaultMetricKeys(keys, primary, declaredKeys), primary);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <MetricKeyPicker keys={keys} selected={selected} onChange={setPicked} />
        <span className="text-xs text-muted-foreground">x axis: environment steps</span>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {selected.map((key) => {
          const points = series.find((item) => item.key === key)?.points ?? [];
          const last = points.at(-1);
          return (
            <Card key={key} className="gap-2 py-4">
              <CardHeader className="px-4">
                <CardTitle className="flex items-center gap-2 text-sm">
                  {humanizeKey(key)}
                  {key === primary && <Badge variant="secondary">Primary</Badge>}
                  <span className="ml-auto font-mono text-xs font-normal text-muted-foreground">{key}</span>
                </CardTitle>
                <div className="tabular text-lg font-semibold">{formatMetric(last?.[1])}</div>
              </CardHeader>
              <CardContent className="px-2">
                {points.length === 0 ? (
                  <p className="px-2 text-sm text-muted-foreground">No points for this metric.</p>
                ) : (
                  <MetricLineChart series={[{ id: "value", label: humanizeKey(key), points }]} />
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
