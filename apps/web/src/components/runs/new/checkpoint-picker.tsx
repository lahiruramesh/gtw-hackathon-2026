"use client";

import { useQuery } from "@tanstack/react-query";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { clientFetch } from "@/lib/api/client";
import type { Artifact, RunRef } from "@/lib/api/types";
import { formatCompact } from "@/lib/format";

const NONE = "__none__";

interface CheckpointPickerProps {
  /** Runs the warm start may come from (the skill's runs). */
  runs: RunRef[];
  value: { runId: string; checkpointId: string } | null;
  onChange: (value: { runId: string; checkpointId: string } | null) => void;
  /** Keeps the chosen run while its checkpoints load, before a checkpoint is picked. */
  runId: string | null;
  onRunChange: (runId: string | null) => void;
}

function checkpointLabel(artifact: Artifact): string {
  const step = artifact.step === null ? "final" : `${formatCompact(artifact.step)} steps`;
  return `${artifact.name} (${step})`;
}

export function CheckpointPicker({ runs, value, onChange, runId, onRunChange }: CheckpointPickerProps) {
  const artifacts = useQuery({
    queryKey: ["run-artifacts", runId],
    queryFn: ({ signal }) => clientFetch<Artifact[]>(`/runs/${encodeURIComponent(runId!)}/artifacts`, { signal }),
    enabled: runId !== null,
  });
  const checkpoints = (artifacts.data ?? [])
    .filter((artifact) => artifact.kind === "checkpoint" || artifact.kind === "params")
    .sort((a, b) => (b.step ?? Number.MAX_SAFE_INTEGER) - (a.step ?? Number.MAX_SAFE_INTEGER));

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      <Select
        value={runId ?? NONE}
        onValueChange={(next) => {
          onRunChange(next === NONE ? null : next);
          onChange(null);
        }}
      >
        <SelectTrigger className="w-full" aria-label="Warm start from run">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={NONE}>No warm start (train from scratch)</SelectItem>
          {runs.map((run) => (
            <SelectItem key={run.id} value={run.id}>
              {run.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        value={value?.checkpointId ?? NONE}
        disabled={runId === null || artifacts.isPending}
        onValueChange={(next) => onChange(next === NONE || runId === null ? null : { runId, checkpointId: next })}
      >
        <SelectTrigger className="w-full" aria-label="Checkpoint">
          <SelectValue
            placeholder={
              runId === null
                ? "Choose a run first"
                : artifacts.isPending
                  ? "Loading checkpoints…"
                  : "Choose a checkpoint"
            }
          />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={NONE}>Choose a checkpoint</SelectItem>
          {checkpoints.map((artifact) => (
            <SelectItem key={artifact.id} value={artifact.id}>
              {checkpointLabel(artifact)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {artifacts.isError && (
        <p className="text-xs text-status-danger sm:col-span-2">Couldn&apos;t load this run&apos;s checkpoints.</p>
      )}
      {artifacts.isSuccess && checkpoints.length === 0 && (
        <p className="text-xs text-muted-foreground sm:col-span-2">This run has no checkpoints.</p>
      )}
    </div>
  );
}
