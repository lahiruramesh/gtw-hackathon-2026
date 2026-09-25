import type { ComputeTarget, CreateRunRequest, EstimateRequest, Preset, SkillDetail } from "@/lib/api/types";
import type { FieldValues } from "@/lib/form-fields";

export const RUN_NAME_PATTERN = /^[a-z0-9][a-z0-9-]{2,62}$/;

/** Everything the wizard collects before launch. */
export interface RunDraft {
  mode: "preset" | "custom";
  presetId: string | null;
  /** Full parameter values for a custom run (ignored for presets, which send the preset's own params). */
  customParams: FieldValues;
  parent: { runId: string; checkpointId: string } | null;
  targetId: string | null;
  /** The user picked the target; until then it follows the parameters (see defaultTargetId). */
  targetPicked: boolean;
  name: string;
  notes: string;
}

export function findPreset(skill: SkillDetail, presetId: string | null): Preset | undefined {
  return skill.presets.find((preset) => preset.id === presetId);
}

/** Whether the draft's parameters describe a smoke test (the tiny CPU run every skill offers). */
export function isSmokeDraft(skill: SkillDetail, draft: Pick<RunDraft, "mode" | "presetId" | "customParams">): boolean {
  const params = draft.mode === "preset" ? findPreset(skill, draft.presetId)?.params : draft.customParams;
  return params?.smoke === true;
}

/**
 * The target a launch starts on: the local CPU for a smoke test, a GPU target otherwise (a full
 * training run on the CPU would take days).
 */
export function defaultTargetId(targets: readonly ComputeTarget[], smoke: boolean): string | null {
  const enabled = targets.filter((target) => target.enabled);
  const fits = enabled.find((target) => (target.kind === "local_cpu") === smoke);
  return (fits ?? enabled[0])?.id ?? null;
}

/** The body for POST /runs/estimate; null until the draft names a target and a parameter source. */
export function estimateRequest(skill: SkillDetail, draft: RunDraft, targetId: string): EstimateRequest | null {
  if (draft.mode === "preset") {
    const preset = findPreset(skill, draft.presetId);
    if (!preset) return null;
    return { skill_id: skill.id, preset_id: preset.id, params: preset.params, compute_target_id: targetId };
  }
  return { skill_id: skill.id, preset_id: null, params: draft.customParams, compute_target_id: targetId };
}

export function createRunRequest(skill: SkillDetail, draft: RunDraft): CreateRunRequest | null {
  if (!draft.targetId) return null;
  const base = estimateRequest(skill, draft, draft.targetId);
  if (!base) return null;
  const name = draft.name.trim();
  const notes = draft.notes.trim();
  return {
    ...base,
    ...(name ? { name } : {}),
    ...(notes ? { notes } : {}),
    parent_run_id: draft.mode === "custom" ? (draft.parent?.runId ?? null) : null,
    parent_checkpoint_id: draft.mode === "custom" ? (draft.parent?.checkpointId ?? null) : null,
  };
}
