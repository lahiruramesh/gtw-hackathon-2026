import type { GateVerdict, HealthStatus, ReviewStatus, RunStatus, Stage, StageStatus } from "@/lib/api/types";

export type StatusTone = "success" | "warning" | "danger" | "info" | "neutral";

export interface StatusStyle {
  label: string;
  tone: StatusTone;
  /** Work is in progress; rendered with a pulsing dot. */
  active?: boolean;
}

export const RUN_STATUS: Record<RunStatus, StatusStyle> = {
  pending_approval: { label: "Needs approval", tone: "warning" },
  queued: { label: "Queued", tone: "neutral", active: true },
  running: { label: "Running", tone: "info", active: true },
  awaiting_review: { label: "Awaiting review", tone: "warning" },
  approved: { label: "Approved", tone: "success" },
  rejected: { label: "Rejected", tone: "danger" },
  gate_failed: { label: "Gate failed", tone: "danger" },
  failed: { label: "Failed", tone: "danger" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

export const STAGE_STATUS: Record<StageStatus, StatusStyle> = {
  pending: { label: "Pending", tone: "neutral" },
  queued: { label: "Queued", tone: "neutral", active: true },
  provisioning: { label: "Provisioning", tone: "info", active: true },
  running: { label: "Running", tone: "info", active: true },
  collecting: { label: "Collecting", tone: "info", active: true },
  succeeded: { label: "Succeeded", tone: "success" },
  failed: { label: "Failed", tone: "danger" },
  cancelled: { label: "Cancelled", tone: "neutral" },
  skipped: { label: "Skipped", tone: "neutral" },
};

export const GATE_VERDICT: Record<GateVerdict, StatusStyle> = {
  pass: { label: "Passed", tone: "success" },
  fail: { label: "Failed", tone: "danger" },
};

export const REVIEW_STATUS: Record<ReviewStatus, StatusStyle> = {
  pending: { label: "Review pending", tone: "warning" },
  approved: { label: "Release approved", tone: "success" },
  rejected: { label: "Release rejected", tone: "danger" },
  not_required: { label: "No review needed", tone: "neutral" },
};

export const HEALTH_STATUS: Record<HealthStatus, StatusStyle> = {
  ok: { label: "Healthy", tone: "success" },
  degraded: { label: "Degraded", tone: "warning" },
  down: { label: "Down", tone: "danger" },
  unknown: { label: "Unchecked", tone: "neutral" },
};

export type StatusDomain = "run" | "stage" | "gate" | "review" | "health";

const DOMAINS: Record<StatusDomain, Record<string, StatusStyle>> = {
  run: RUN_STATUS,
  stage: STAGE_STATUS,
  gate: GATE_VERDICT,
  review: REVIEW_STATUS,
  health: HEALTH_STATUS,
};

export function statusStyle(domain: StatusDomain, status: string): StatusStyle {
  return DOMAINS[domain][status] ?? { label: status.replaceAll("_", " "), tone: "neutral" };
}

export const ACTIVE_RUN_STATUSES: readonly RunStatus[] = ["pending_approval", "queued", "running"];
export const ACTIVE_STAGE_STATUSES: readonly StageStatus[] = ["queued", "provisioning", "running", "collecting"];
export const TERMINAL_STAGE_STATUSES: readonly StageStatus[] = ["succeeded", "failed", "cancelled", "skipped"];

export function isRunActive(status: RunStatus): boolean {
  return ACTIVE_RUN_STATUSES.includes(status);
}

/**
 * The verdict a gate stage should show. Its execution status only says the check ran ("succeeded"
 * even when the run failed the gate), so once it has run the stage shows the gate's verdict instead.
 */
export function gateStageVerdict(
  stage: Pick<Stage, "kind" | "status">,
  verdict: GateVerdict | null,
): GateVerdict | null {
  return stage.kind === "gate" && stage.status === "succeeded" ? verdict : null;
}
