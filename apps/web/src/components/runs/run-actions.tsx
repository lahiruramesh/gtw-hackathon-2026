"use client";

import { RotateCcwIcon, SquareIcon } from "lucide-react";

import { cancelRun, decideLaunch, retryRun, reviewRelease } from "@/app/(app)/runs/actions";
import { DecisionButtons } from "@/components/common/decision-buttons";
import { useRunLive } from "@/components/runs/run-live-provider";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { useAction } from "@/hooks/use-action";
import { isRunActive } from "@/lib/status";

/** Header actions, shown only when the API says the viewer may perform them (RunDetail.permissions). */
export function RunActions({ viewerId }: { viewerId: string }) {
  const { run } = useRunLive();
  const { pending, run: execute } = useAction();
  const { permissions } = run;
  const ownRun = run.created_by.id === viewerId;

  return (
    <>
      {permissions.can_approve_launch && run.status === "pending_approval" && (
        <DecisionButtons
          subject={`launch of ${run.name}`}
          approveLabel="Approve launch"
          rejectLabel="Reject launch"
          disabledReason={ownRun ? "You can't approve your own run" : undefined}
          pending={pending}
          onDecide={async (input) => {
            const result = await execute(() => decideLaunch(run.id, input), {
              success: input.decision === "approve" ? "Launch approved" : "Launch rejected",
              error: "Couldn't record the decision",
            });
            return result.ok;
          }}
        />
      )}
      {permissions.can_review && run.status === "awaiting_review" && (
        <DecisionButtons
          subject={`release of ${run.name}`}
          approveLabel="Approve release"
          rejectLabel="Reject release"
          disabledReason={ownRun ? "You can't review your own run" : undefined}
          pending={pending}
          onDecide={async (input) => {
            const result = await execute(() => reviewRelease(run.id, input), {
              success: input.decision === "approve" ? "Release approved" : "Release rejected",
              error: "Couldn't record the review",
            });
            return result.ok;
          }}
        />
      )}
      {permissions.can_retry && run.status === "failed" && (
        <Button
          variant="outline"
          size="sm"
          disabled={pending}
          onClick={() =>
            execute(() => retryRun(run.id), { success: "Retrying the failed stage", error: "Couldn't retry" })
          }
        >
          <RotateCcwIcon /> Retry
        </Button>
      )}
      {permissions.can_cancel && isRunActive(run.status) && (
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="outline" size="sm" disabled={pending}>
              <SquareIcon /> Cancel run
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Cancel {run.name}?</AlertDialogTitle>
              <AlertDialogDescription>
                Active stages are stopped and their compute released. Collected artifacts are kept. This can&apos;t be
                undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Keep running</AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-white hover:bg-destructive/90"
                onClick={() =>
                  execute(() => cancelRun(run.id), { success: "Run cancelled", error: "Couldn't cancel the run" })
                }
              >
                Cancel run
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </>
  );
}
