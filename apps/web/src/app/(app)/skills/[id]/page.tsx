import { GitBranchIcon, PlayIcon, PlusIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { LinkTabs } from "@/components/common/link-tabs";
import { Markdown } from "@/components/common/markdown";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { BreadcrumbLabel } from "@/components/layout/breadcrumbs";
import { RunsTable } from "@/components/runs/runs-table";
import { GateCriteriaTable } from "@/components/skills/gate-criteria-table";
import { LineageTree } from "@/components/skills/lineage-tree";
import { ParamChips } from "@/components/skills/param-chips";
import { ParamsSchemaTable } from "@/components/skills/params-schema-table";
import { PipelineDefinition } from "@/components/skills/pipeline-definition";
import { CATEGORY_LABELS, SKILL_STATUS_LABELS } from "@/components/skills/skill-labels";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { RunPage, SkillDetail } from "@/lib/api/types";
import { shortSha } from "@/lib/format";
import { fieldsFromJsonSchema } from "@/lib/form-fields";
import { buildLineage } from "@/lib/lineage";
import { requireViewer, type Viewer } from "@/lib/session";
import { can } from "@/lib/viewer";

const TABS = ["overview", "presets", "runs", "lineage"] as const;
type Tab = (typeof TABS)[number];

export async function generateMetadata({ params }: PageProps<"/skills/[id]">): Promise<Metadata> {
  const { id } = await params;
  return { title: id };
}

export default async function SkillPage({ params, searchParams }: PageProps<"/skills/[id]">) {
  const viewer = await requireViewer();
  if (!can(viewer, "skill:read")) return <NoAccess what="view skills" />;

  const { id } = await params;
  const { tab: tabParam } = await searchParams;
  const tab: Tab = TABS.find((value) => value === tabParam) ?? "overview";

  const result = await toResult(apiFetch<SkillDetail>(`/skills/${encodeURIComponent(id)}`));
  if (!result.ok) {
    if (result.error.status === 404) notFound();
    return <ApiErrorState error={result.error} subject="this skill" />;
  }
  const skill = result.data;
  const canLaunch = can(viewer, "run:create_preset");

  return (
    <div className="space-y-5">
      <BreadcrumbLabel segment={id} label={skill.name} />
      <PageHeader
        title={skill.name}
        description={skill.summary}
        actions={
          canLaunch && (
            <Button asChild size="sm">
              <Link href={`/runs/new?skill=${encodeURIComponent(skill.id)}`}>
                <PlusIcon /> New run
              </Link>
            </Button>
          )
        }
      >
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          <Badge variant="secondary">{CATEGORY_LABELS[skill.category]}</Badge>
          <Badge variant="outline" className="font-mono">
            {skill.method}
          </Badge>
          <Badge variant="outline">{SKILL_STATUS_LABELS[skill.status]}</Badge>
          <span className="text-xs text-muted-foreground">
            {skill.robot} · manifest at {shortSha(skill.git_sha)}
          </span>
        </div>
      </PageHeader>

      <LinkTabs
        label="Skill sections"
        active={tab}
        tabs={TABS.map((value) => ({
          value,
          label: value.charAt(0).toUpperCase() + value.slice(1),
          href: `/skills/${encodeURIComponent(skill.id)}?tab=${value}`,
        }))}
      />

      {tab === "overview" && <Overview skill={skill} />}
      {tab === "presets" && <Presets skill={skill} canLaunch={canLaunch} />}
      {tab === "runs" && <SkillRuns skill={skill} viewer={viewer} />}
      {tab === "lineage" && <Lineage skill={skill} />}
    </div>
  );
}

function Overview({ skill }: { skill: SkillDetail }) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-sm">Pipeline</CardTitle>
          <CardDescription>Every run executes these stages in order.</CardDescription>
        </CardHeader>
        <CardContent>
          <PipelineDefinition stages={skill.pipeline} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Description</CardTitle>
        </CardHeader>
        <CardContent>
          {skill.description ? (
            <Markdown>{skill.description}</Markdown>
          ) : (
            <p className="text-sm text-muted-foreground">No description.</p>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Gates</CardTitle>
          <CardDescription>All criteria must pass before a safety reviewer can approve a release.</CardDescription>
        </CardHeader>
        <CardContent>
          {skill.gate.length === 0 ? (
            <p className="text-sm text-muted-foreground">This skill has no gate criteria.</p>
          ) : (
            <GateCriteriaTable criteria={skill.gate} />
          )}
        </CardContent>
      </Card>
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-sm">Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <ParamsSchemaTable fields={fieldsFromJsonSchema(skill.params_schema)} />
        </CardContent>
      </Card>
    </div>
  );
}

function Presets({ skill, canLaunch }: { skill: SkillDetail; canLaunch: boolean }) {
  if (skill.presets.length === 0) {
    return <EmptyState title="No presets" description="Presets are defined in the skill manifest." />;
  }
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      {skill.presets.map((preset) => (
        <Card key={preset.id} className="gap-4">
          <CardHeader>
            <CardTitle className="text-sm">{preset.name}</CardTitle>
            <CardDescription>
              <span className="font-mono text-xs">{preset.id}</span>
              {preset.description && ` · ${preset.description}`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ParamChips params={preset.params} />
          </CardContent>
          {canLaunch && (
            <CardFooter className="mt-auto">
              <Button asChild variant="outline" size="sm">
                <Link href={`/runs/new?skill=${encodeURIComponent(skill.id)}&preset=${encodeURIComponent(preset.id)}`}>
                  <PlayIcon /> Launch preset
                </Link>
              </Button>
            </CardFooter>
          )}
        </Card>
      ))}
    </div>
  );
}

async function fetchSkillRuns(skillId: string) {
  return toResult(apiFetch<RunPage>("/runs", { query: { skill_id: skillId, limit: 200 } }));
}

async function SkillRuns({ skill, viewer }: { skill: SkillDetail; viewer: Viewer }) {
  const result = await fetchSkillRuns(skill.id);
  if (!result.ok) return <ApiErrorState error={result.error} subject="runs" />;
  if (result.data.items.length === 0) {
    return (
      <EmptyState
        icon={PlayIcon}
        title="No runs for this skill"
        description="Start with the smoke preset to check the pipeline end to end."
        action={
          can(viewer, "run:create_preset") && (
            <Button asChild size="sm">
              <Link href={`/runs/new?skill=${encodeURIComponent(skill.id)}`}>New run</Link>
            </Button>
          )
        }
      />
    );
  }
  return (
    <Card className="py-2">
      <CardContent className="px-2">
        <RunsTable runs={result.data.items} headlineColumns={skill.headline.map((item) => item.label)} />
      </CardContent>
    </Card>
  );
}

async function Lineage({ skill }: { skill: SkillDetail }) {
  const result = await fetchSkillRuns(skill.id);
  if (!result.ok) return <ApiErrorState error={result.error} subject="runs" />;
  const runs = result.data.items;
  if (runs.length === 0) {
    return (
      <EmptyState
        icon={GitBranchIcon}
        title="No lineage yet"
        description="Warm-started runs appear here as a tree under the run they started from."
      />
    );
  }
  const roots = buildLineage(
    [...runs].reverse().map((run) => ({ id: run.id, parentId: run.parent?.run.id ?? null, value: run })),
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Warm-start lineage</CardTitle>
        <CardDescription>Oldest first. Each child started from a checkpoint of its parent.</CardDescription>
      </CardHeader>
      <CardContent>
        <LineageTree roots={roots} />
      </CardContent>
    </Card>
  );
}
