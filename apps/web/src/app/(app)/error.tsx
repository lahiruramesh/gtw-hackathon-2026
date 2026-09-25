"use client";

import { TriangleAlertIcon } from "lucide-react";

import { EmptyState } from "@/components/common/empty-state";
import { Button } from "@/components/ui/button";

export default function AppError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <EmptyState
      className="mt-10"
      icon={TriangleAlertIcon}
      title="Something went wrong"
      description={
        <>
          This page failed to render. Try again, and if it keeps happening share this reference with an admin:{" "}
          <code className="font-mono text-xs">{error.digest ?? error.message}</code>
        </>
      }
      action={
        <Button variant="outline" size="sm" onClick={reset}>
          Try again
        </Button>
      }
    />
  );
}
