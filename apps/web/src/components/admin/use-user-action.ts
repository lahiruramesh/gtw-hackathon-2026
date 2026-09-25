"use client";

import { toast } from "sonner";

import { useAction } from "@/hooks/use-action";
import type { UserActionOutcome } from "@/lib/admin-users";
import type { Result } from "@/lib/api/errors";

/** Runs a user-management action; warns when the change worked but its audit event was lost. */
export function useUserAction() {
  const { pending, run } = useAction();

  async function execute(action: () => Promise<Result<UserActionOutcome>>, success: string, error: string) {
    const result = await run(action, { success, error });
    if (result.ok && !result.data.audited) {
      toast.warning("Not recorded in the audit log", {
        description: "The change was made, but the API could not record it. Tell a platform admin.",
      });
    }
    return result.ok;
  }

  return { pending, execute };
}
