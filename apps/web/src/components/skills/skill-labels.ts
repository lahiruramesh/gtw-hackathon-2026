import type { Category, SkillStatus } from "@/lib/api/types";

export const CATEGORY_LABELS: Record<Category, string> = {
  locomotion: "Locomotion",
  manipulation: "Manipulation",
  workflow: "Workflow",
};

export const SKILL_STATUS_LABELS: Record<SkillStatus, string> = {
  draft: "Draft",
  active: "Active",
  deprecated: "Deprecated",
};
