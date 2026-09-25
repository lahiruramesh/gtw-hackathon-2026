import Link from "next/link";

import { DateTime } from "@/components/common/date-time";
import { StatusBadge } from "@/components/common/status-badge";
import { CATEGORY_LABELS, SKILL_STATUS_LABELS } from "@/components/skills/skill-labels";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import type { HeadlineValue, SkillSummary } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";
import { formatHeadline } from "@/lib/headline";

interface SkillCardProps {
  skill: SkillSummary;
  /** Headline metrics of the skill's best run, when it has one. */
  headline: HeadlineValue[];
}

export function SkillCard({ skill, headline }: SkillCardProps) {
  return (
    <Card className="gap-4">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="secondary">{CATEGORY_LABELS[skill.category]}</Badge>
          <Badge variant="outline" className="font-mono">
            {skill.method}
          </Badge>
          {skill.status !== "active" && <Badge variant="outline">{SKILL_STATUS_LABELS[skill.status]}</Badge>}
        </div>
        <CardTitle className="pt-1">
          <Link href={`/skills/${skill.id}`} className="hover:underline">
            {skill.name}
          </Link>
        </CardTitle>
        <CardDescription>{skill.summary}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground">Best run</span>
          {skill.best_run ? (
            <span className="flex min-w-0 items-center gap-2">
              <Link href={`/runs/${skill.best_run.id}`} className="truncate font-medium hover:underline">
                {skill.best_run.name}
              </Link>
              <StatusBadge domain="run" status={skill.best_run.status} />
            </span>
          ) : (
            <span className="text-muted-foreground">None yet</span>
          )}
        </div>
        {headline.length > 0 && (
          <dl className="grid grid-cols-3 gap-2 rounded-md border p-2">
            {headline.map((item) => (
              <div key={item.label} className="min-w-0">
                <dt className="truncate text-xs text-muted-foreground">{item.label}</dt>
                <dd className="tabular text-sm font-medium">{formatHeadline(item)}</dd>
              </div>
            ))}
          </dl>
        )}
      </CardContent>
      <CardFooter className="mt-auto justify-between text-xs text-muted-foreground">
        <span>
          {formatNumber(skill.run_count)} {skill.run_count === 1 ? "run" : "runs"} · {skill.robot}
        </span>
        {skill.last_run_at && (
          <span>
            Last run <DateTime value={skill.last_run_at} relative />
          </span>
        )}
      </CardFooter>
    </Card>
  );
}
