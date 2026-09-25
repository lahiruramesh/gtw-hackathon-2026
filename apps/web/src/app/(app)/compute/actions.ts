"use server";

import { z } from "zod";

import { COMPUTE_KINDS } from "@/components/compute/target-kind";
import type { Result } from "@/lib/api/errors";
import { mutate, segment } from "@/lib/api/mutate";
import type { ComputeTarget, ComputeTargetInput } from "@/lib/api/types";

const id = z.string().min(1).max(200);

const targetSchema = z.object({
  name: z.string().trim().min(1).max(100),
  kind: z.enum(COMPUTE_KINDS),
  description: z.string().max(2000).nullable().optional(),
  enabled: z.boolean(),
  config: z.record(z.string(), z.unknown()),
  gpu_label: z.string().max(100).nullable().optional(),
  steps_per_second: z.number().positive(),
  overhead_minutes: z.number().min(0),
  cost_per_gpu_hour: z.number().min(0),
  weekly_quota_gpu_hours: z.number().min(0).nullable().optional(),
  max_unapproved_gpu_hours: z.number().min(0),
  max_concurrent: z.number().int().min(1),
});

function invalid(message: string): Result<never> {
  return { ok: false, error: { status: 422, code: "validation_error", message } };
}

export async function createTarget(input: ComputeTargetInput): Promise<Result<ComputeTarget>> {
  const parsed = targetSchema.safeParse(input);
  if (!parsed.success) return invalid(parsed.error.issues[0]?.message ?? "Invalid compute target");
  return mutate("/compute-targets", { method: "POST", body: parsed.data });
}

export async function updateTarget(
  targetId: string,
  input: Partial<ComputeTargetInput>,
): Promise<Result<ComputeTarget>> {
  const parsedId = id.safeParse(targetId);
  const parsed = targetSchema.omit({ kind: true }).partial().safeParse(input);
  if (!parsedId.success || !parsed.success) return invalid("Invalid compute target");
  return mutate(`/compute-targets/${segment(parsedId.data)}`, { method: "PATCH", body: parsed.data });
}

/** Write-only: the secret goes to the API and is never read back. */
export async function setTargetSecret(targetId: string, secret: Record<string, unknown>): Promise<Result<void>> {
  const parsedId = id.safeParse(targetId);
  const parsed = z.record(z.string(), z.unknown()).safeParse(secret);
  if (!parsedId.success || !parsed.success) return invalid("Invalid secret");
  return mutate(`/compute-targets/${segment(parsedId.data)}/secret`, { method: "PUT", body: { secret: parsed.data } });
}

export async function checkTarget(targetId: string): Promise<Result<ComputeTarget>> {
  const parsedId = id.safeParse(targetId);
  if (!parsedId.success) return invalid("Invalid compute target");
  return mutate(`/compute-targets/${segment(parsedId.data)}/check`, { method: "POST" });
}
