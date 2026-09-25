import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { ApiErrorState } from "@/components/common/api-error-state";
import { NoAccess } from "@/components/common/no-access";
import { BreadcrumbLabel } from "@/components/layout/breadcrumbs";
import { ArtifactsPanel } from "@/components/runs/artifacts-panel";
import { CheckpointsPanel } from "@/components/runs/checkpoints-panel";
import { ConfigPanel } from "@/components/runs/config-panel";
import { EvaluationPanel } from "@/components/runs/evaluation-panel";
import { GatePanel } from "@/components/runs/gate-panel";
import { LogViewer } from "@/components/runs/log-viewer";
import { MetricsPanel } from "@/components/runs/metrics-panel";
import { PipelineStepper } from "@/components/runs/pipeline-stepper";
import { RunHeader } from "@/components/runs/run-header";
import { RunLiveProvider } from "@/components/runs/run-live-provider";
import { RunTabs } from "@/components/runs/run-tabs";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import { fieldsFromJsonSchema } from "@/lib/form-fields";
import type { Artifact, Evaluation, RunDetail, SkillDetail } from "@/lib/api/types";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Run" };

export default async function RunPage({ params }: PageProps<"/runs/[id]">) {
  const viewer = await requireViewer();
  if (!can(viewer, "run:read")) return <NoAccess what="view runs" />;

  const { id } = await params;
  const runPath = `/runs/${encodeURIComponent(id)}`;
  const [runResult, artifactsResult, evaluationsResult] = await Promise.all([
    toResult(apiFetch<RunDetail>(runPath)),
    toResult(apiFetch<Artifact[]>(`${runPath}/artifacts`)),
    toResult(apiFetch<Evaluation[]>(`${runPath}/evaluations`)),
  ]);
  if (!runResult.ok) {
    if (runResult.error.status === 404) notFound();
    return <ApiErrorState error={runResult.error} subject="this run" />;
  }
  const run = runResult.data;
  const skillResult = await toResult(apiFetch<SkillDetail>(`/skills/${encodeURIComponent(run.skill_id)}`));
  const skill = skillResult.ok ? skillResult.data : null;
  const artifacts = artifactsResult.ok ? artifactsResult.data : [];
  const evaluations = evaluationsResult.ok ? evaluationsResult.data : [];

  return (
    <RunLiveProvider initialRun={run}>
      <BreadcrumbLabel segment={id} label={run.name} />
      <div className="space-y-5">
        <RunHeader viewerId={viewer.id} />
        <PipelineStepper />
        <RunTabs
          panels={{
            logs: <LogViewer />,
            metrics: (
              <MetricsPanel
                primary={skill?.metrics.primary ?? "eval/episode_reward"}
                declaredKeys={skill?.metrics.keys ?? []}
              />
            ),
            checkpoints: artifactsResult.ok ? (
              <CheckpointsPanel
                run={run}
                artifacts={artifacts}
                evaluations={evaluations}
                canWarmStart={can(viewer, "run:create_custom")}
              />
            ) : (
              <ApiErrorState error={artifactsResult.error} subject="checkpoints" />
            ),
            evaluation: evaluationsResult.ok ? (
              <EvaluationPanel evaluations={evaluations} />
            ) : (
              <ApiErrorState error={evaluationsResult.error} subject="evaluations" />
            ),
            artifacts: artifactsResult.ok ? (
              <ArtifactsPanel artifacts={artifacts} />
            ) : (
              <ApiErrorState error={artifactsResult.error} subject="artifacts" />
            ),
            gate: <GatePanel run={run} criteria={skill?.gate ?? []} viewerId={viewer.id} />,
            config: <ConfigPanel run={run} fields={skill ? fieldsFromJsonSchema(skill.params_schema) : []} />,
          }}
        />
      </div>
    </RunLiveProvider>
  );
}
