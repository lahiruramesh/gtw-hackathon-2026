import type { Artifact } from "@/lib/api/types";

export function isPreviewable(artifact: Artifact): boolean {
  return artifact.kind === "video" || artifact.kind === "image";
}
