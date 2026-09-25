"use client";

import { decideLaunch, reviewRelease } from "@/app/(app)/runs/actions";
import { DecisionButtons } from "@/components/common/decision-buttons";
import { useAction } from "@/hooks/use-action";

interface ApprovalDecisionProps {
  kind: "launch" | "release";
  runId: string;
  runName: string;
  ownRun: boolean;
}

export function ApprovalDecision({ kind, runId, runName, ownRun }: ApprovalDecisionProps) {
  const { pending, run } = useAction();
  const action = kind === "launch" ? decideLaunch : reviewRelease;
  return (
    <DecisionButtons
      subject={`${kind} of ${runName}`}
      pending={pending}
      disabledReason={ownRun ? `You can't decide on your own run` : undefined}
      onDecide={async (input) => {
        const result = await run(() => action(runId, input), {
          success: `${kind === "launch" ? "Launch" : "Release"} ${input.decision === "approve" ? "approved" : "rejected"}`,
          error: "Couldn't record the decision",
        });
        return result.ok;
      }}
    />
  );
}
