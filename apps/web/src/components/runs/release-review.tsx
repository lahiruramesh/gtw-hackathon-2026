"use client";

import { reviewRelease } from "@/app/(app)/runs/actions";
import { DecisionButtons } from "@/components/common/decision-buttons";
import { useAction } from "@/hooks/use-action";

export function ReleaseReview({ runId, runName, ownRun }: { runId: string; runName: string; ownRun: boolean }) {
  const { pending, run } = useAction();
  return (
    <DecisionButtons
      subject={`release of ${runName}`}
      approveLabel="Approve release"
      rejectLabel="Reject release"
      disabledReason={ownRun ? "You can't review your own run" : undefined}
      pending={pending}
      size="default"
      onDecide={async (input) => {
        const result = await run(() => reviewRelease(runId, input), {
          success: input.decision === "approve" ? "Release approved" : "Release rejected",
          error: "Couldn't record the review",
        });
        return result.ok;
      }}
    />
  );
}
