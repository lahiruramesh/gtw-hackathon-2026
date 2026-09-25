"use client";

import { EyeIcon, EyeOffIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useArtifactUrl } from "@/hooks/use-artifact-url";
import type { Artifact } from "@/lib/api/types";

/** Inline player/viewer, loaded on demand so a page of artifacts doesn't sign every URL up front. */
export function ArtifactPreview({ artifact }: { artifact: Artifact }) {
  const [open, setOpen] = useState(false);
  const url = useArtifactUrl(artifact.id, open);
  return (
    <div className="space-y-2">
      <Button variant="ghost" size="sm" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        {open ? <EyeOffIcon /> : <EyeIcon />}
        {open ? "Hide" : artifact.kind === "video" ? "Play" : "View"}
      </Button>
      {open &&
        (url.isPending ? (
          <Skeleton className="aspect-video w-full max-w-xl" />
        ) : url.isError ? (
          <p className="text-xs text-status-danger">Couldn&apos;t load the preview.</p>
        ) : artifact.kind === "video" ? (
          <video src={url.data.url} controls preload="metadata" className="w-full max-w-xl rounded-md border">
            <track kind="captions" />
          </video>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element -- presigned object-store URL, not optimisable
          <img src={url.data.url} alt={artifact.name} className="max-h-96 max-w-xl rounded-md border" />
        ))}
    </div>
  );
}
