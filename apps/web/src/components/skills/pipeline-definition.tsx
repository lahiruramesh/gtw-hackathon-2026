import { ChevronRightIcon, CpuIcon, FlaskConicalIcon, ShieldCheckIcon, type LucideIcon } from "lucide-react";
import { Fragment } from "react";

import type { PipelineStageDef, StageKind } from "@/lib/api/types";

const KIND_ICONS: Record<StageKind, LucideIcon> = {
  train: CpuIcon,
  evaluate: FlaskConicalIcon,
  gate: ShieldCheckIcon,
};

const RUNS_ON_LABELS = { target: "Runs on the chosen compute target", local: "Runs on the worker (CPU)" } as const;

export function PipelineDefinition({ stages }: { stages: PipelineStageDef[] }) {
  return (
    <ol className="flex flex-col gap-2 md:flex-row md:items-stretch">
      {stages.map((stage, index) => {
        const Icon = KIND_ICONS[stage.kind];
        return (
          <Fragment key={stage.id}>
            {index > 0 && (
              <ChevronRightIcon
                className="hidden size-4 shrink-0 self-center text-muted-foreground md:block"
                aria-hidden
              />
            )}
            <li className="flex flex-1 items-start gap-3 rounded-md border p-3">
              <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
              <div className="min-w-0 space-y-0.5">
                <div className="text-sm font-medium">{stage.title}</div>
                <div className="text-xs text-muted-foreground">
                  <span className="font-mono">{stage.id}</span>
                  {stage.runs_on && ` · ${RUNS_ON_LABELS[stage.runs_on]}`}
                </div>
              </div>
            </li>
          </Fragment>
        );
      })}
    </ol>
  );
}
