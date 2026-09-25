"use client";

import { useQueryClient } from "@tanstack/react-query";
import { DownloadIcon } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { artifactUrlQuery } from "@/hooks/use-artifact-url";
import { describeError, toErrorInfo } from "@/lib/api/errors";

export function ArtifactDownloadButton({ artifactId, label = "Download" }: { artifactId: string; label?: string }) {
  const queryClient = useQueryClient();
  const [pending, setPending] = useState(false);

  async function download() {
    setPending(true);
    try {
      const { url } = await queryClient.fetchQuery(artifactUrlQuery(artifactId));
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (error) {
      toast.error("Couldn't get a download link", { description: describeError(toErrorInfo(error)) });
    } finally {
      setPending(false);
    }
  }

  return (
    <Button variant="ghost" size="sm" onClick={download} disabled={pending}>
      <DownloadIcon /> {label}
    </Button>
  );
}
