"use client";

import type { ReactNode } from "react";
import type { Control } from "react-hook-form";

import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { formatCompact } from "@/lib/format";
import type { FieldSpec, FieldValues } from "@/lib/form-fields";

const NULL_OPTION = "__none__";

interface SchemaFieldsProps {
  fields: readonly FieldSpec[];
  control: Control<FieldValues>;
  /** Renders `checkpoint` fields (the run wizard's warm-start picker). */
  renderCheckpoint?: (field: FieldSpec, value: unknown, onChange: (value: unknown) => void) => ReactNode;
}

function numberFromInput(raw: string): number | null {
  if (raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isNaN(value) ? null : value;
}

export function SchemaFields({ fields, control, renderCheckpoint }: SchemaFieldsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {fields.map((spec) => (
        <FormField
          key={spec.name}
          control={control}
          name={spec.name}
          render={({ field }) => {
            const hint =
              spec.kind === "integer" && typeof field.value === "number" && Math.abs(field.value) >= 10_000
                ? `= ${formatCompact(field.value)}`
                : undefined;
            if (spec.kind === "boolean") {
              return (
                <FormItem className="flex flex-row items-center justify-between gap-3 rounded-md border px-3 py-2 sm:col-span-2">
                  <div className="space-y-0.5">
                    <FormLabel>{spec.label}</FormLabel>
                    {spec.description && <FormDescription>{spec.description}</FormDescription>}
                  </div>
                  <FormControl>
                    <Switch checked={field.value === true} onCheckedChange={field.onChange} />
                  </FormControl>
                </FormItem>
              );
            }
            return (
              <FormItem
                className={spec.kind === "multiline" || spec.kind === "checkpoint" ? "sm:col-span-2" : undefined}
              >
                <FormLabel>
                  {spec.label}
                  {!spec.required && <span className="font-normal text-muted-foreground">(optional)</span>}
                </FormLabel>
                {spec.kind === "checkpoint" && renderCheckpoint ? (
                  renderCheckpoint(spec, field.value, field.onChange)
                ) : spec.kind === "enum" ? (
                  <Select
                    value={field.value === null || field.value === undefined ? NULL_OPTION : String(field.value)}
                    onValueChange={(raw) =>
                      field.onChange(
                        raw === NULL_OPTION
                          ? null
                          : (spec.options?.find((option) => String(option.value) === raw)?.value ?? raw),
                      )
                    }
                  >
                    <FormControl>
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {!spec.required && <SelectItem value={NULL_OPTION}>Not set</SelectItem>}
                      {spec.options?.map((option) => (
                        <SelectItem key={String(option.value)} value={String(option.value)}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : spec.kind === "multiline" ? (
                  <FormControl>
                    <Textarea
                      name={field.name}
                      ref={field.ref}
                      onBlur={field.onBlur}
                      value={typeof field.value === "string" ? field.value : ""}
                      onChange={(event) => field.onChange(event.target.value)}
                      rows={6}
                      spellCheck={false}
                      autoComplete="off"
                      className="font-mono text-xs"
                    />
                  </FormControl>
                ) : spec.kind === "integer" || spec.kind === "number" ? (
                  <FormControl>
                    <Input
                      name={field.name}
                      ref={field.ref}
                      onBlur={field.onBlur}
                      type="number"
                      inputMode={spec.kind === "integer" ? "numeric" : "decimal"}
                      step={spec.kind === "integer" ? 1 : "any"}
                      min={spec.min}
                      max={spec.max}
                      value={typeof field.value === "number" ? field.value : ""}
                      onChange={(event) => field.onChange(numberFromInput(event.target.value))}
                      className="tabular"
                    />
                  </FormControl>
                ) : (
                  <FormControl>
                    <Input
                      name={field.name}
                      ref={field.ref}
                      onBlur={field.onBlur}
                      type={spec.kind === "secret" ? "password" : "text"}
                      autoComplete={spec.kind === "secret" ? "new-password" : "off"}
                      value={typeof field.value === "string" ? field.value : ""}
                      onChange={(event) =>
                        field.onChange(event.target.value === "" && spec.nullable ? null : event.target.value)
                      }
                    />
                  </FormControl>
                )}
                {(spec.description || hint) && (
                  <FormDescription>{[spec.description, hint].filter(Boolean).join(" ")}</FormDescription>
                )}
                <FormMessage />
              </FormItem>
            );
          }}
        />
      ))}
    </div>
  );
}
