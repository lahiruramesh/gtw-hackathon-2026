"use client";

import { useQuery } from "@tanstack/react-query";

import { clientFetch } from "@/lib/api/client";
import type { ArtifactUrl } from "@/lib/api/types";

/** Presigned URLs live 15 minutes; refetch well before they expire. */
const URL_STALE_MS = 10 * 60 * 1000;

export function artifactUrlQuery(artifactId: string) {
  return {
    queryKey: ["artifact-url", artifactId],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      clientFetch<ArtifactUrl>(`/artifacts/${encodeURIComponent(artifactId)}/url`, { signal }),
    staleTime: URL_STALE_MS,
  };
}

export function useArtifactUrl(artifactId: string, enabled: boolean) {
  return useQuery({ ...artifactUrlQuery(artifactId), enabled });
}
