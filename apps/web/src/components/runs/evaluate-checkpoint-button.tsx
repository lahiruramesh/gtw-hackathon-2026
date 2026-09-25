"use client";

import { FlaskConicalIcon } from "lucide-react";

import { evaluateCheckpoint } from "@/app/(app)/runs/actions";
import { Button } from "@/components/ui/button";
import { useAction } from "@/hooks/use-action";

export function EvaluateCheckpointButton({ runId, checkpointId }: { runId: string; checkpointId: string }) {
  const { pending, run } = useAction();
  return (
    <Button
      variant="ghost"
      size="sm"
      disabled={pending}
      onClick={() =>
        run(() => evaluateCheckpoint(runId, checkpointId), {
          success: "Evaluation stage added",
          error: "Couldn't start the evaluation",
        })
      }
    >
      <FlaskConicalIcon /> Evaluate
    </Button>
  );
}
