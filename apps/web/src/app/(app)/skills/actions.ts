"use server";

import type { Result } from "@/lib/api/errors";
import { mutate } from "@/lib/api/mutate";
import type { SkillSyncResult } from "@/lib/api/types";

export async function syncSkills(): Promise<Result<SkillSyncResult>> {
  return mutate<SkillSyncResult>("/skills/sync", { method: "POST" });
}
