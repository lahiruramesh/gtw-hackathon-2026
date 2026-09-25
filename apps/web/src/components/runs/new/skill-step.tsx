import { BoxesIcon } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/common/empty-state";
import { CATEGORY_LABELS } from "@/components/skills/skill-labels";
import { Badge } from "@/components/ui/badge";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { SkillSummary } from "@/lib/api/types";

export function SkillStep({ skills }: { skills: SkillSummary[] }) {
  const launchable = skills.filter((skill) => skill.status !== "deprecated");
  if (launchable.length === 0) {
    return (
      <EmptyState
        icon={BoxesIcon}
        title="No skills to train"
        description="Ask an ML engineer to register a skill manifest."
      />
    );
  }
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
      {launchable.map((skill) => (
        <Link
          key={skill.id}
          href={`/runs/new?skill=${encodeURIComponent(skill.id)}`}
          className="rounded-xl focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          <Card className="h-full transition-colors hover:border-foreground/30">
            <CardHeader>
              <div className="flex gap-1.5">
                <Badge variant="secondary">{CATEGORY_LABELS[skill.category]}</Badge>
                <Badge variant="outline" className="font-mono">
                  {skill.method}
                </Badge>
              </div>
              <CardTitle className="pt-1 text-base">{skill.name}</CardTitle>
              <CardDescription>{skill.summary}</CardDescription>
            </CardHeader>
          </Card>
        </Link>
      ))}
    </div>
  );
}
