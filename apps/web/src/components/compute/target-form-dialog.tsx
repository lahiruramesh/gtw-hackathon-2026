"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { PencilIcon, PlusIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { createTarget, updateTarget } from "@/app/(app)/compute/actions";
import { ApiErrorState } from "@/components/common/api-error-state";
import { SchemaFields } from "@/components/forms/schema-fields";
import { COMMON_TARGET_FIELDS } from "@/components/compute/target-fields";
import { COMPUTE_KIND_LABELS, COMPUTE_KINDS } from "@/components/compute/target-kind";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Form } from "@/components/ui/form";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useAction } from "@/hooks/use-action";
import { useTargetSchema } from "@/hooks/use-target-schema";
import { toErrorInfo } from "@/lib/api/errors";
import type { ComputeKind, ComputeTarget, ComputeTargetInput } from "@/lib/api/types";
import {
  fieldDefaults,
  fieldsFromJsonSchema,
  fieldsValidator,
  type FieldSpec,
  type FieldValues,
} from "@/lib/form-fields";

function TargetForm({
  kind,
  target,
  configFields,
  onDone,
}: {
  kind: ComputeKind;
  target?: ComputeTarget;
  configFields: FieldSpec[];
  onDone: () => void;
}) {
  const { pending, run } = useAction();
  const nestedConfig = useMemo(
    () => configFields.map((field) => ({ ...field, name: `config.${field.name}` })),
    [configFields],
  );
  const schema = useMemo(
    () => fieldsValidator(COMMON_TARGET_FIELDS).extend({ config: fieldsValidator(configFields) }),
    [configFields],
  );
  const form = useForm<FieldValues>({
    resolver: zodResolver(schema),
    defaultValues: target
      ? {
          ...fieldDefaults(COMMON_TARGET_FIELDS),
          ...Object.fromEntries(
            COMMON_TARGET_FIELDS.map((field) => [field.name, target[field.name as keyof ComputeTarget] ?? null]),
          ),
          config: { ...fieldDefaults(configFields), ...target.config },
        }
      : { ...fieldDefaults(COMMON_TARGET_FIELDS), config: fieldDefaults(configFields) },
  });

  const onSubmit = form.handleSubmit(async (values) => {
    const input = { ...values, kind } as ComputeTargetInput;
    const result = await run(() => (target ? updateTarget(target.id, input) : createTarget(input)), {
      success: target ? "Compute target saved" : "Compute target created",
      error: target ? "Couldn't save the compute target" : "Couldn't create the compute target",
    });
    if (result.ok) onDone();
  });

  return (
    <Form {...form}>
      <form onSubmit={onSubmit} className="space-y-5" noValidate>
        <SchemaFields fields={COMMON_TARGET_FIELDS} control={form.control} />
        {nestedConfig.length > 0 && (
          <fieldset className="space-y-3 rounded-md border p-3">
            <legend className="px-1 text-sm font-medium">{COMPUTE_KIND_LABELS[kind]} settings</legend>
            <SchemaFields fields={nestedConfig} control={form.control} />
          </fieldset>
        )}
        <DialogFooter>
          <Button type="submit" disabled={pending}>
            {target ? "Save" : "Create target"}
          </Button>
        </DialogFooter>
      </form>
    </Form>
  );
}

export function TargetFormDialog({ target }: { target?: ComputeTarget }) {
  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState<ComputeKind>(target?.kind ?? "kaggle");
  const schema = useTargetSchema(kind, open);
  const configFields = useMemo(() => (schema.data ? fieldsFromJsonSchema(schema.data.config) : []), [schema.data]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {target ? (
          <Button variant="ghost" size="sm">
            <PencilIcon /> Edit
          </Button>
        ) : (
          <Button size="sm">
            <PlusIcon /> Add target
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="max-h-[90svh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{target ? `Edit ${target.name}` : "Add compute target"}</DialogTitle>
          <DialogDescription>
            Credentials are set separately and are never shown again. Estimates use the throughput and overhead below.
          </DialogDescription>
        </DialogHeader>
        {!target && (
          <div className="space-y-2">
            <Label htmlFor="target-kind">Kind</Label>
            <Select value={kind} onValueChange={(value) => setKind(z.enum(COMPUTE_KINDS).parse(value))}>
              <SelectTrigger id="target-kind" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {COMPUTE_KINDS.map((item) => (
                  <SelectItem key={item} value={item}>
                    {COMPUTE_KIND_LABELS[item]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {schema.isPending ? (
          <div className="space-y-3">
            <Skeleton className="h-9" />
            <Skeleton className="h-9" />
            <Skeleton className="h-9" />
          </div>
        ) : schema.isError ? (
          <ApiErrorState error={toErrorInfo(schema.error)} subject="the target settings" />
        ) : (
          <TargetForm
            key={kind}
            kind={kind}
            target={target}
            configFields={configFields}
            onDone={() => setOpen(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
