"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { LockIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";

import { SchemaFields } from "@/components/forms/schema-fields";
import { CheckpointPicker } from "@/components/runs/new/checkpoint-picker";
import { ParamChips } from "@/components/skills/param-chips";
import { Button } from "@/components/ui/button";
import { Form } from "@/components/ui/form";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import type { RunRef, SkillDetail } from "@/lib/api/types";
import { fieldDefaults, fieldsFromJsonSchema, fieldsValidator, type FieldValues } from "@/lib/form-fields";
import { findPreset, type RunDraft } from "@/lib/run-draft";
import { cn } from "cn";

const CUSTOM = "__custom__";

interface ParamsStepProps {
  skill: SkillDetail;
  draft: RunDraft;
  canCustomize: boolean;
  warmStartRuns: RunRef[];
  onBack: () => void;
  onNext: (patch: Pick<RunDraft, "mode" | "presetId" | "customParams" | "parent">) => void;
}

function SourceOption({
  value,
  title,
  children,
  disabled,
  selected,
}: {
  value: string;
  title: string;
  children?: React.ReactNode;
  disabled?: boolean;
  selected: boolean;
}) {
  const id = `source-${value}`;
  return (
    <Label
      htmlFor={id}
      className={cn(
        "flex cursor-pointer items-start gap-3 rounded-lg border p-3 font-normal transition-colors",
        selected && "border-primary bg-muted/40",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      <RadioGroupItem value={value} id={id} disabled={disabled} className="mt-0.5" />
      <div className="min-w-0 space-y-1.5">
        <div className="text-sm font-medium">{title}</div>
        {children}
      </div>
    </Label>
  );
}

export function ParamsStep({ skill, draft, canCustomize, warmStartRuns, onBack, onNext }: ParamsStepProps) {
  const fields = useMemo(() => fieldsFromJsonSchema(skill.params_schema), [skill.params_schema]);
  const defaults = useMemo(() => fieldDefaults(fields), [fields]);
  const [source, setSource] = useState(
    draft.mode === "custom" ? CUSTOM : (draft.presetId ?? skill.presets[0]?.id ?? CUSTOM),
  );
  const [parent, setParent] = useState(draft.parent);
  const [parentRunId, setParentRunId] = useState(draft.parent?.runId ?? null);
  const form = useForm<FieldValues>({
    resolver: zodResolver(fieldsValidator(fields)),
    defaultValues: { ...defaults, ...draft.customParams },
  });

  function selectSource(value: string) {
    if (value === CUSTOM && source !== CUSTOM) {
      // Start the custom form from the preset the user was looking at.
      form.reset({ ...defaults, ...(findPreset(skill, source)?.params ?? {}) });
    }
    setSource(value);
  }

  function choosePresetAndContinue() {
    onNext({ mode: "preset", presetId: source, customParams: draft.customParams, parent: null });
  }

  const submitCustom = form.handleSubmit((values) =>
    onNext({ mode: "custom", presetId: null, customParams: values, parent }),
  );

  return (
    <div className="space-y-5">
      <RadioGroup
        value={source}
        onValueChange={selectSource}
        className="grid grid-cols-1 gap-3 md:grid-cols-2"
        aria-label="Parameter source"
      >
        {skill.presets.map((preset) => (
          <SourceOption key={preset.id} value={preset.id} title={preset.name} selected={source === preset.id}>
            {preset.description && <p className="text-xs text-muted-foreground">{preset.description}</p>}
            <ParamChips params={preset.params} />
          </SourceOption>
        ))}
        <SourceOption value={CUSTOM} title="Custom parameters" disabled={!canCustomize} selected={source === CUSTOM}>
          <p className="text-xs text-muted-foreground">
            {canCustomize ? (
              "Edit every parameter, or warm start from a checkpoint of an earlier run."
            ) : (
              <span className="inline-flex items-center gap-1">
                <LockIcon className="size-3" /> Needs the ML engineer or admin role. Operators launch approved presets.
              </span>
            )}
          </p>
        </SourceOption>
      </RadioGroup>

      {source === CUSTOM && canCustomize && (
        <Form {...form}>
          <form id="custom-params" onSubmit={submitCustom} className="rounded-lg border p-4" noValidate>
            <SchemaFields
              fields={fields}
              control={form.control}
              renderCheckpoint={(_, __, onChange) => (
                <CheckpointPicker
                  runs={warmStartRuns}
                  value={parent}
                  runId={parentRunId}
                  onRunChange={setParentRunId}
                  onChange={(next) => {
                    setParent(next);
                    onChange(next?.checkpointId ?? null);
                  }}
                />
              )}
            />
          </form>
        </Form>
      )}

      <div className="flex justify-between gap-2">
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
        {source === CUSTOM ? (
          <Button type="submit" form="custom-params" disabled={!canCustomize}>
            Continue
          </Button>
        ) : (
          <Button onClick={choosePresetAndContinue}>Continue</Button>
        )}
      </div>
    </div>
  );
}
