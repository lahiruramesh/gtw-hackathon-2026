"use client";

import { useQuery } from "@tanstack/react-query";
import { Loader2Icon, RocketIcon, ShieldAlertIcon } from "lucide-react";

import { EstimateSummary } from "@/components/runs/new/estimate-summary";
import { estimateQuery } from "@/components/runs/new/target-step";
import { ParamChips } from "@/components/skills/param-chips";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { ComputeTarget, SkillDetail } from "@/lib/api/types";
import type { FieldValues } from "@/lib/form-fields";
import { findPreset, RUN_NAME_PATTERN, type RunDraft } from "@/lib/run-draft";

interface ReviewStepProps {
  skill: SkillDetail;
  draft: RunDraft;
  target: ComputeTarget;
  changedParams: FieldValues;
  /** Name of the run the warm start comes from. */
  warmStartFrom?: string;
  launching: boolean;
  onChange: (patch: Pick<RunDraft, "name" | "notes">) => void;
  onBack: () => void;
  onLaunch: () => void;
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-1 border-b py-2 last:border-0 sm:grid-cols-[160px_1fr]">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="min-w-0 text-sm">{children}</dd>
    </div>
  );
}

export function ReviewStep({
  skill,
  draft,
  target,
  changedParams,
  warmStartFrom,
  launching,
  onChange,
  onBack,
  onLaunch,
}: ReviewStepProps) {
  const estimate = useQuery(estimateQuery(skill, draft, target.id));
  const nameInvalid = draft.name.trim() !== "" && !RUN_NAME_PATTERN.test(draft.name.trim());
  const preset = findPreset(skill, draft.presetId);

  return (
    <div className="space-y-5">
      <dl className="rounded-lg border px-4">
        <Item label="Skill">{skill.name}</Item>
        <Item label="Parameters">
          {draft.mode === "preset" && preset ? (
            <div className="space-y-1.5">
              <div>Preset: {preset.name}</div>
              <ParamChips params={preset.params} />
            </div>
          ) : (
            <div className="space-y-1.5">
              <div>Custom (changes from the skill defaults)</div>
              <ParamChips params={changedParams} />
            </div>
          )}
        </Item>
        {draft.mode === "custom" && draft.parent && (
          <Item label="Warm start">From a checkpoint of {warmStartFrom ?? "an earlier run"}</Item>
        )}
        <Item label="Compute">{target.name}</Item>
        <Item label="Estimate">
          {estimate.data ? (
            <EstimateSummary estimate={estimate.data} />
          ) : (
            <span className="text-muted-foreground">Calculating…</span>
          )}
        </Item>
      </dl>

      {estimate.data?.needs_approval && (
        <Alert>
          <ShieldAlertIcon />
          <AlertTitle>This run needs launch approval</AlertTitle>
          <AlertDescription>
            It waits in the approvals queue until an ML engineer or admin approves it.
            {estimate.data.reasons.length > 0 && ` ${estimate.data.reasons.join(" ")}`}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="run-name">
            Run name <span className="font-normal text-muted-foreground">(optional)</span>
          </Label>
          <Input
            id="run-name"
            value={draft.name}
            onChange={(event) => onChange({ name: event.target.value, notes: draft.notes })}
            placeholder={`${skill.id}-${draft.presetId ?? "custom"}-n`}
            aria-invalid={nameInvalid || undefined}
            aria-describedby="run-name-hint"
            autoComplete="off"
          />
          <p id="run-name-hint" className={nameInvalid ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>
            3 to 63 lowercase letters, digits and dashes. Leave empty for an automatic name.
          </p>
        </div>
        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="run-notes">
            Notes <span className="font-normal text-muted-foreground">(optional)</span>
          </Label>
          <Textarea
            id="run-notes"
            value={draft.notes}
            onChange={(event) => onChange({ name: draft.name, notes: event.target.value })}
            placeholder="What are you testing?"
            rows={3}
          />
        </div>
      </div>

      <div className="flex justify-between gap-2">
        <Button variant="outline" onClick={onBack} disabled={launching}>
          Back
        </Button>
        <Button onClick={onLaunch} disabled={launching || nameInvalid}>
          {launching ? <Loader2Icon className="animate-spin" /> : <RocketIcon />}
          {estimate.data?.needs_approval ? "Submit for approval" : "Launch run"}
        </Button>
      </div>
    </div>
  );
}
