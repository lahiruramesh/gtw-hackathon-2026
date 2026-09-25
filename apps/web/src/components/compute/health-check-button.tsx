"use client";

import { ActivityIcon } from "lucide-react";

import { checkTarget } from "@/app/(app)/compute/actions";
import { Button } from "@/components/ui/button";
import { useAction } from "@/hooks/use-action";

export function HealthCheckButton({ targetId }: { targetId: string }) {
  const { pending, run } = useAction();
  return (
    <Button
      variant="ghost"
      size="sm"
      disabled={pending}
      onClick={() =>
        run(() => checkTarget(targetId), {
          success: (target) =>
            `Health: ${target.health.status}${target.health.message ? ` · ${target.health.message}` : ""}`,
          error: "Health check failed",
        })
      }
    >
      <ActivityIcon className={pending ? "animate-pulse" : undefined} /> Check health
    </Button>
  );
}
