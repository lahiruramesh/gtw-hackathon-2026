"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { createRun } from "@/app/(app)/runs/actions";
import { ParamsStep } from "@/components/runs/new/params-step";
import { ReviewStep } from "@/components/runs/new/review-step";
import { TargetStep } from "@/components/runs/new/target-step";
import { WizardSteps } from "@/components/runs/new/wizard-steps";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAction } from "@/hooks/use-action";
import type { ComputeTarget, RunRef, SkillDetail } from "@/lib/api/types";
import { changedValues, fieldDefaults, fieldsFromJsonSchema } from "@/lib/form-fields";
import { createRunRequest, defaultTargetId, findPreset, isSmokeDraft, type RunDraft } from "@/lib/run-draft";

interface NewRunWizardProps {
  skill: SkillDetail;
  targets: ComputeTarget[];
  warmStartRuns: RunRef[];
  canCustomize: boolean;
  initialPresetId: string | null;
  /** Warm start requested from a run's Checkpoints tab. */
  initialParent: { runId: string; checkpointId: string } | null;
}

const STEP_COPY = {
  2: { title: "Parameters", description: "Launch an approved preset, or set parameters yourself." },
  3: { title: "Compute", description: "Pick where to train. Estimates use each target's measured throughput." },
  4: { title: "Review and launch", description: "Check the run before it starts." },
} as const;

export function NewRunWizard({
  skill,
  targets,
  warmStartRuns,
  canCustomize,
  initialPresetId,
  initialParent,
}: NewRunWizardProps) {
  const router = useRouter();
  const { pending, run } = useAction();
  const fields = useMemo(() => fieldsFromJsonSchema(skill.params_schema), [skill.params_schema]);
  const checkpointField = fields.find((field) => field.kind === "checkpoint");
  const warmStart = canCustomize && initialParent !== null && checkpointField !== undefined;

  const [step, setStep] = useState<2 | 3 | 4>(2);
  const [draft, setDraft] = useState<RunDraft>(() => {
    const params: Pick<RunDraft, "mode" | "presetId" | "customParams"> = {
      mode: warmStart ? "custom" : "preset",
      presetId: findPreset(skill, initialPresetId)?.id ?? skill.presets[0]?.id ?? null,
      customParams: {
        ...fieldDefaults(fields),
        ...(warmStart && checkpointField ? { [checkpointField.name]: initialParent.checkpointId } : {}),
      },
    };
    return {
      ...params,
      parent: warmStart ? initialParent : null,
      targetId: defaultTargetId(targets, isSmokeDraft(skill, params)),
      targetPicked: false,
      name: "",
      notes: "",
    };
  });
  const target = targets.find((item) => item.id === draft.targetId);

  async function launch() {
    const request = createRunRequest(skill, draft);
    if (!request) return;
    const result = await run(() => createRun(request), {
      success: (detail) =>
        detail.status === "pending_approval" ? `${detail.name} is waiting for approval` : `${detail.name} launched`,
      error: "Couldn't launch the run",
    });
    if (result.ok) router.push(`/runs/${result.data.id}`);
  }

  const copy = STEP_COPY[step];
  return (
    <div className="space-y-5">
      <WizardSteps current={step} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.title}</CardTitle>
          <CardDescription>
            {copy.description}{" "}
            <span className="whitespace-nowrap">
              Skill: <span className="font-medium text-foreground">{skill.name}</span>
              <span aria-hidden> · </span>
              <Button asChild variant="link" size="sm" className="h-auto p-0">
                <Link href="/runs/new" aria-label="Change skill">
                  Change
                </Link>
              </Button>
            </span>
          </CardDescription>
        </CardHeader>
        <CardContent>
          {step === 2 && (
            <ParamsStep
              skill={skill}
              draft={draft}
              canCustomize={canCustomize}
              warmStartRuns={warmStartRuns}
              onBack={() => router.push("/runs/new")}
              onNext={(patch) => {
                setDraft((current) => {
                  const next = { ...current, ...patch };
                  return next.targetPicked
                    ? next
                    : { ...next, targetId: defaultTargetId(targets, isSmokeDraft(skill, next)) };
                });
                setStep(3);
              }}
            />
          )}
          {step === 3 && (
            <TargetStep
              skill={skill}
              draft={draft}
              targets={targets}
              onBack={() => setStep(2)}
              onSelect={(targetId) => setDraft((current) => ({ ...current, targetId, targetPicked: true }))}
              onNext={() => setStep(4)}
            />
          )}
          {step === 4 && target && (
            <ReviewStep
              skill={skill}
              draft={draft}
              target={target}
              changedParams={changedValues(fields, draft.customParams)}
              warmStartFrom={warmStartRuns.find((item) => item.id === draft.parent?.runId)?.name}
              launching={pending}
              onChange={(patch) => setDraft((current) => ({ ...current, ...patch }))}
              onBack={() => setStep(3)}
              onLaunch={launch}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
