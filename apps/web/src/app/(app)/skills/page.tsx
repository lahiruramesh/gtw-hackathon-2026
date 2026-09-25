import { BoxesIcon } from "lucide-react";
import type { Metadata } from "next";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { SkillCard } from "@/components/skills/skill-card";
import { SyncSkillsButton } from "@/components/skills/sync-skills-button";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { RunDetail, SkillPage } from "@/lib/api/types";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Skills" };

export default async function SkillsPage() {
  const viewer = await requireViewer();
  if (!can(viewer, "skill:read")) return <NoAccess what="view skills" />;

  const result = await toResult(apiFetch<SkillPage>("/skills", { query: { limit: 200 } }));
  const skills = result.ok ? result.data.items : [];
  // Headline numbers live on runs; a skill's card shows its best run's.
  const bestRuns = await Promise.all(
    skills.map((skill) =>
      skill.best_run ? toResult(apiFetch<RunDetail>(`/runs/${skill.best_run.id}`)) : Promise.resolve(null),
    ),
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Skills"
        description="Each skill is a repository manifest: parameters, presets, pipeline and release gate."
        actions={can(viewer, "skill:write") && <SyncSkillsButton />}
      />
      {!result.ok ? (
        <ApiErrorState error={result.error} subject="skills" />
      ) : skills.length === 0 ? (
        <EmptyState
          icon={BoxesIcon}
          title="No skills yet"
          description="Add a skills/<name>/skill.yaml manifest to the repository, then sync."
          action={can(viewer, "skill:write") && <SyncSkillsButton />}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {skills.map((skill, index) => {
            const best = bestRuns[index];
            return <SkillCard key={skill.id} skill={skill} headline={best?.ok ? best.data.headline : []} />;
          })}
        </div>
      )}
    </div>
  );
}
