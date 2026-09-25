"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { KeyRoundIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";

import { setTargetSecret } from "@/app/(app)/compute/actions";
import { ApiErrorState } from "@/components/common/api-error-state";
import { SchemaFields } from "@/components/forms/schema-fields";
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
import { Skeleton } from "@/components/ui/skeleton";
import { useAction } from "@/hooks/use-action";
import { useTargetSchema } from "@/hooks/use-target-schema";
import { toErrorInfo } from "@/lib/api/errors";
import type { ComputeTarget } from "@/lib/api/types";
import {
  compactValues,
  fieldDefaults,
  fieldsFromJsonSchema,
  fieldsValidator,
  type FieldSpec,
  type FieldValues,
} from "@/lib/form-fields";

function SecretForm({ target, fields, onDone }: { target: ComputeTarget; fields: FieldSpec[]; onDone: () => void }) {
  const { pending, run } = useAction();
  const form = useForm<FieldValues>({
    resolver: zodResolver(fieldsValidator(fields)),
    defaultValues: fieldDefaults(fields),
  });
  const onSubmit = form.handleSubmit(async (values) => {
    const result = await run(() => setTargetSecret(target.id, compactValues(values)), {
      success: "Credentials saved",
      error: "Couldn't save the credentials",
    });
    if (result.ok) {
      form.reset(fieldDefaults(fields));
      onDone();
    }
  });
  return (
    <Form {...form}>
      <form onSubmit={onSubmit} className="space-y-4" noValidate autoComplete="off">
        <SchemaFields fields={fields} control={form.control} />
        <DialogFooter>
          <Button type="submit" disabled={pending}>
            Save credentials
          </Button>
        </DialogFooter>
      </form>
    </Form>
  );
}

/** Write-only credentials form: values are sent once and never displayed or read back. */
export function SecretDialog({ target }: { target: ComputeTarget }) {
  const [open, setOpen] = useState(false);
  const schema = useTargetSchema(target.kind, open);
  const fields = useMemo(
    () => (schema.data ? fieldsFromJsonSchema(schema.data.secret, { secret: true }) : []),
    [schema.data],
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm">
          <KeyRoundIcon /> {target.has_secret ? "Replace credentials" : "Set credentials"}
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90svh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Credentials for {target.name}</DialogTitle>
          <DialogDescription>
            Stored encrypted. They are never shown again; saving replaces what is stored.
          </DialogDescription>
        </DialogHeader>
        {schema.isPending ? (
          <Skeleton className="h-32" />
        ) : schema.isError ? (
          <ApiErrorState error={toErrorInfo(schema.error)} subject="the credential form" />
        ) : (
          <SecretForm target={target} fields={fields} onDone={() => setOpen(false)} />
        )}
      </DialogContent>
    </Dialog>
  );
}
