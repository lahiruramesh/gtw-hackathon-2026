import { FileIcon } from "lucide-react";

import { EmptyState } from "@/components/common/empty-state";
import { ArtifactDownloadButton } from "@/components/runs/artifact-download-button";
import { ArtifactPreview } from "@/components/runs/artifact-preview";
import { Badge } from "@/components/ui/badge";
import type { Artifact } from "@/lib/api/types";
import { isPreviewable } from "@/lib/artifacts";
import { formatBytes } from "@/lib/format";

function groupByStage(artifacts: Artifact[]): [string, Artifact[]][] {
  const groups = new Map<string, Artifact[]>();
  for (const artifact of artifacts) {
    const key = artifact.stage_key ?? "run";
    groups.set(key, [...(groups.get(key) ?? []), artifact]);
  }
  return [...groups.entries()];
}

export function ArtifactsPanel({ artifacts }: { artifacts: Artifact[] }) {
  if (artifacts.length === 0) {
    return (
      <EmptyState
        icon={FileIcon}
        title="No artifacts yet"
        description="Stage outputs are uploaded here when each stage finishes."
      />
    );
  }
  const media = artifacts.filter(isPreviewable);
  return (
    <div className="space-y-6">
      {media.length > 0 && (
        <section className="space-y-2">
          <h3 className="text-sm font-semibold">Videos and images</h3>
          <ul className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {media.map((artifact) => (
              <li key={artifact.id} className="space-y-1 rounded-md border p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs">{artifact.name}</span>
                  <span className="text-xs text-muted-foreground">{artifact.stage_key}</span>
                </div>
                <ArtifactPreview artifact={artifact} />
              </li>
            ))}
          </ul>
        </section>
      )}
      {groupByStage(artifacts).map(([stage, items]) => (
        <section key={stage} className="space-y-2">
          <h3 className="text-sm font-semibold">
            <span className="font-mono">{stage}</span>
            <span className="ml-2 font-normal text-muted-foreground">{items.length} files</span>
          </h3>
          <ul className="divide-y rounded-md border">
            {items.map((artifact) => (
              <li key={artifact.id} className="flex flex-wrap items-center gap-2 px-3 py-1.5">
                <span className="min-w-0 flex-1 truncate font-mono text-xs" title={artifact.name}>
                  {artifact.name}
                </span>
                <Badge variant="outline">{artifact.kind}</Badge>
                <span className="tabular w-16 text-right text-xs text-muted-foreground">
                  {formatBytes(artifact.size_bytes)}
                </span>
                <ArtifactDownloadButton artifactId={artifact.id} />
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
