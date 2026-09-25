import { PlayIcon, PlusIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { RunFilters } from "@/components/runs/run-filters";
import { RunsTable } from "@/components/runs/runs-table";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { RunPage, SkillDetail, SkillPage } from "@/lib/api/types";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Runs" };

const PAGE_SIZE = 50;

function param(value: string | string[] | undefined): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

export default async function RunsPage({ searchParams }: PageProps<"/runs">) {
  const viewer = await requireViewer();
  if (!can(viewer, "run:read")) return <NoAccess what="view runs" />;

  const search = await searchParams;
  const skillId = param(search.skill);
  const query = {
    skill_id: skillId,
    status: param(search.status),
    created_by: search.mine === "1" ? "me" : undefined,
    q: param(search.q),
    cursor: param(search.cursor),
    limit: PAGE_SIZE,
  };

  const [runs, skills, skill] = await Promise.all([
    toResult(apiFetch<RunPage>("/runs", { query })),
    toResult(apiFetch<SkillPage>("/skills", { query: { limit: 200 } })),
    skillId ? toResult(apiFetch<SkillDetail>(`/skills/${encodeURIComponent(skillId)}`)) : Promise.resolve(null),
  ]);

  const nextParams = new URLSearchParams(
    Object.entries(search).flatMap(([key, value]) => (typeof value === "string" ? [[key, value]] : [])),
  );
  const firstPageParams = new URLSearchParams(nextParams);
  firstPageParams.delete("cursor");
  if (runs.ok && runs.data.next_cursor) nextParams.set("cursor", runs.data.next_cursor);
  const filtered = Boolean(skillId || query.status || query.created_by || query.q);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Runs"
        description="Every training run, imported history included."
        actions={
          can(viewer, "run:create_preset") && (
            <Button asChild size="sm">
              <Link href="/runs/new">
                <PlusIcon /> New run
              </Link>
            </Button>
          )
        }
      />
      <RunFilters skills={skills.ok ? skills.data.items.map(({ id, name }) => ({ id, name })) : []} />
      {!runs.ok ? (
        <ApiErrorState error={runs.error} subject="runs" />
      ) : runs.data.items.length === 0 ? (
        <EmptyState
          icon={PlayIcon}
          title={filtered ? "No runs match these filters" : "No runs yet"}
          description={
            filtered ? "Clear or change the filters to see more runs." : "Launch a smoke run to try the pipeline."
          }
          action={
            !filtered &&
            can(viewer, "run:create_preset") && (
              <Button asChild size="sm">
                <Link href="/runs/new">New run</Link>
              </Button>
            )
          }
        />
      ) : (
        <Card className="gap-2 py-2">
          <CardContent className="px-2">
            <RunsTable
              runs={runs.data.items}
              headlineColumns={skill?.ok ? skill.data.headline.map((item) => item.label) : undefined}
            />
          </CardContent>
          <nav className="flex items-center justify-end gap-2 px-4 pb-2" aria-label="Pagination">
            {query.cursor && (
              <Button asChild variant="outline" size="sm">
                <Link href={`/runs?${firstPageParams.toString()}`}>First page</Link>
              </Button>
            )}
            {runs.data.next_cursor && (
              <Button asChild variant="outline" size="sm">
                <Link href={`/runs?${nextParams.toString()}`}>Next page</Link>
              </Button>
            )}
          </nav>
        </Card>
      )}
    </div>
  );
}
