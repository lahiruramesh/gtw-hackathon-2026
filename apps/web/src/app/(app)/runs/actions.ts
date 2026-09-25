"use server";

import { z } from "zod";

import type { Result } from "@/lib/api/errors";
import { mutate, segment } from "@/lib/api/mutate";
import type { ApiRequestOptions } from "@/lib/api/server";
import type { CreateRunRequest, DecisionRequest, RunDetail } from "@/lib/api/types";
import { RUN_NAME_PATTERN } from "@/lib/run-draft";

const id = z.string().min(1).max(200);

const decisionSchema = z
  .object({ decision: z.enum(["approve", "reject"]), comment: z.string().trim().max(2000).optional() })
  .refine((value) => value.decision === "approve" || Boolean(value.comment), {
    message: "A comment is required when rejecting",
    path: ["comment"],
  });

function invalid(message: string): Result<never> {
  return { ok: false, error: { status: 422, code: "validation_error", message } };
}

async function runAction(runId: string, action: string, options: ApiRequestOptions): Promise<Result<RunDetail>> {
  const parsed = id.safeParse(runId);
  if (!parsed.success) return invalid("Invalid run id");
  return mutate(`/runs/${segment(parsed.data)}/${action}`, options);
}

async function decide(runId: string, action: string, input: DecisionRequest): Promise<Result<RunDetail>> {
  const parsed = decisionSchema.safeParse(input);
  if (!parsed.success) return invalid(parsed.error.issues[0]?.message ?? "Invalid decision");
  return runAction(runId, action, { method: "POST", body: parsed.data });
}

export async function cancelRun(runId: string): Promise<Result<RunDetail>> {
  return runAction(runId, "cancel", { method: "POST" });
}

export async function retryRun(runId: string): Promise<Result<RunDetail>> {
  return runAction(runId, "retry", { method: "POST" });
}

export async function decideLaunch(runId: string, input: DecisionRequest): Promise<Result<RunDetail>> {
  return decide(runId, "launch-decision", input);
}

export async function reviewRelease(runId: string, input: DecisionRequest): Promise<Result<RunDetail>> {
  return decide(runId, "review", input);
}

export async function evaluateCheckpoint(runId: string, checkpointId: string): Promise<Result<RunDetail>> {
  const checkpoint = id.safeParse(checkpointId);
  if (!checkpoint.success) return invalid("Invalid checkpoint id");
  return runAction(runId, "evaluate", { method: "POST", body: { checkpoint_id: checkpoint.data } });
}

const createRunSchema = z.object({
  skill_id: id,
  preset_id: id.nullable().optional(),
  params: z.record(z.string(), z.unknown()),
  compute_target_id: id,
  name: z.string().regex(RUN_NAME_PATTERN, "Use 3 to 63 lowercase letters, digits and dashes").optional(),
  notes: z.string().max(4000).optional(),
  parent_run_id: id.nullable().optional(),
  parent_checkpoint_id: id.nullable().optional(),
});

export async function createRun(request: CreateRunRequest): Promise<Result<RunDetail>> {
  const parsed = createRunSchema.safeParse(request);
  if (!parsed.success) return invalid(parsed.error.issues[0]?.message ?? "Invalid run request");
  return mutate("/runs", { method: "POST", body: parsed.data });
}
