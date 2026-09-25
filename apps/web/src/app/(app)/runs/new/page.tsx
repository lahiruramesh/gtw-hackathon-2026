import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { ApiErrorState } from "@/components/common/api-error-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { NewRunWizard } from "@/components/runs/new/new-run-wizard";
import { SkillStep } from "@/components/runs/new/skill-step";
import { WizardSteps } from "@/components/runs/new/wizard-steps";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { ComputeTarget, RunDetail, RunPage, RunRef, SkillDetail, SkillPage } from "@/lib/api/types";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "New run" };

function param(value: string | string[] | undefined): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

async function warmStartCandidates(skillId: string, parentId: string | null): Promise<RunRef[]> {
  const [runs, parent] = await Promise.all([
    toResult(apiFetch<RunPage>("/runs", { query: { skill_id: skillId, limit: 100 } })),
    parentId ? toResult(apiFetch<RunDetail>(`/runs/${encodeURIComponent(parentId)}`)) : Promise.resolve(null),
  ]);
  const candidates: RunRef[] = runs.ok ? runs.data.items.map(({ id, name, status }) => ({ id, name, status })) : [];
  if (parent?.ok && !candidates.some((run) => run.id === parent.data.id)) {
    candidates.unshift({ id: parent.data.id, name: parent.data.name, status: parent.data.status });
  }
  return candidates;
}

export default async function NewRunPage({ searchParams }: PageProps<"/runs/new">) {
  const viewer = await requireViewer();
  if (!can(viewer, "run:create_preset")) return <NoAccess what="launch runs" />;

  const search = await searchParams;
  const skillId = param(search.skill);
  const header = (
    <PageHeader
      title="New run"
      description="Train a skill on a compute target. Evaluation and the release gate run automatically."
    />
  );

  if (!skillId) {
    const skills = await toResult(apiFetch<SkillPage>("/skills", { query: { limit: 200 } }));
    return (
      <div className="space-y-5">
        {header}
        <WizardSteps current={1} />
        {skills.ok ? <SkillStep skills={skills.data.items} /> : <ApiErrorState error={skills.error} subject="skills" />}
      </div>
    );
  }

  const canCustomize = can(viewer, "run:create_custom");
  const parentId = param(search.parent);
  const checkpointId = param(search.checkpoint);
  const [skill, targets, warmStartRuns] = await Promise.all([
    toResult(apiFetch<SkillDetail>(`/skills/${encodeURIComponent(skillId)}`)),
    toResult(apiFetch<ComputeTarget[]>("/compute-targets")),
    canCustomize ? warmStartCandidates(skillId, parentId) : Promise.resolve([]),
  ]);
  if (!skill.ok) {
    if (skill.error.status === 404) notFound();
    return (
      <div className="space-y-5">
        {header}
        <ApiErrorState error={skill.error} subject="the skill" />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {header}
      {targets.ok ? (
        <NewRunWizard
          skill={skill.data}
          targets={targets.data}
          warmStartRuns={warmStartRuns}
          canCustomize={canCustomize}
          initialPresetId={param(search.preset)}
          initialParent={parentId && checkpointId ? { runId: parentId, checkpointId } : null}
        />
      ) : (
        <ApiErrorState error={targets.error} subject="compute targets" />
      )}
    </div>
  );
}
