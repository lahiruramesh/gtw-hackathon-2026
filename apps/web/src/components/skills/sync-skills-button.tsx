"use client";

import { RefreshCwIcon } from "lucide-react";
import { toast } from "sonner";

import { syncSkills } from "@/app/(app)/skills/actions";
import { Button } from "@/components/ui/button";
import { useAction } from "@/hooks/use-action";

export function SyncSkillsButton() {
  const { pending, run } = useAction();

  async function onClick() {
    const result = await run(syncSkills, {
      success: (data) => `Synced ${data.synced.length} ${data.synced.length === 1 ? "skill" : "skills"}`,
      error: "Couldn't sync skills",
    });
    for (const problem of result.ok ? result.data.errors : []) {
      toast.warning(`Skipped ${problem.file}`, { description: problem.message });
    }
  }

  return (
    <Button variant="outline" size="sm" onClick={onClick} disabled={pending}>
      <RefreshCwIcon className={pending ? "animate-spin" : undefined} />
      Sync from repository
    </Button>
  );
}
