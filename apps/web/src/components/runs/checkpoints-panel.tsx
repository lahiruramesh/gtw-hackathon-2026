import { ArchiveIcon, GitBranchPlusIcon } from "lucide-react";
import Link from "next/link";

import { DateTime } from "@/components/common/date-time";
import { EmptyState } from "@/components/common/empty-state";
import { ArtifactDownloadButton } from "@/components/runs/artifact-download-button";
import { EvaluateCheckpointButton } from "@/components/runs/evaluate-checkpoint-button";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Artifact, Evaluation, RunDetail } from "@/lib/api/types";
import { formatBytes, formatCompact } from "@/lib/format";

interface CheckpointsPanelProps {
  run: RunDetail;
  artifacts: Artifact[];
  evaluations: Evaluation[];
  canWarmStart: boolean;
}

export function CheckpointsPanel({ run, artifacts, evaluations, canWarmStart }: CheckpointsPanelProps) {
  const checkpoints = artifacts
    .filter((artifact) => artifact.kind === "checkpoint" || artifact.kind === "params")
    .sort((a, b) => (b.step ?? Number.MAX_SAFE_INTEGER) - (a.step ?? Number.MAX_SAFE_INTEGER));
  const evaluated = new Set(
    evaluations.flatMap((evaluation) => (evaluation.checkpoint ? [evaluation.checkpoint.id] : [])),
  );

  if (checkpoints.length === 0) {
    return (
      <EmptyState
        icon={ArchiveIcon}
        title="No checkpoints yet"
        description="Checkpoints are collected when the train stage finishes."
      />
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Checkpoint</TableHead>
            <TableHead className="text-right">Step</TableHead>
            <TableHead className="text-right">Size</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {checkpoints.map((artifact) => (
            <TableRow key={artifact.id}>
              <TableCell>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs">{artifact.name}</span>
                  {artifact.kind === "params" && <Badge variant="secondary">Final</Badge>}
                  {evaluated.has(artifact.id) && <Badge variant="outline">Evaluated</Badge>}
                </div>
                {artifact.stage_key && <div className="text-xs text-muted-foreground">{artifact.stage_key}</div>}
              </TableCell>
              <TableCell className="text-right">
                {artifact.step === null ? "—" : formatCompact(artifact.step)}
              </TableCell>
              <TableCell className="text-right">{formatBytes(artifact.size_bytes)}</TableCell>
              <TableCell className="text-muted-foreground">
                <DateTime value={artifact.created_at} />
              </TableCell>
              <TableCell>
                <div className="flex justify-end gap-1">
                  {run.permissions.can_evaluate && (
                    <EvaluateCheckpointButton runId={run.id} checkpointId={artifact.id} />
                  )}
                  {canWarmStart && (
                    <Button asChild variant="ghost" size="sm">
                      <Link
                        href={`/runs/new?skill=${encodeURIComponent(run.skill_id)}&parent=${encodeURIComponent(run.id)}&checkpoint=${encodeURIComponent(artifact.id)}`}
                      >
                        <GitBranchPlusIcon /> Warm start
                      </Link>
                    </Button>
                  )}
                  <ArtifactDownloadButton artifactId={artifact.id} />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
