import { z } from "zod";

import type { JsonSchema } from "@/lib/api/types";

/**
 * One generated form field. Built from JSON Schema (a skill's `params_schema` or a backend's pydantic
 * config/secret schema), so the run wizard and compute target forms share validation and rendering.
 */
export type FieldKind = "integer" | "number" | "boolean" | "enum" | "string" | "multiline" | "secret" | "checkpoint";

export interface FieldOption {
  value: string | number;
  label: string;
}

export interface FieldSpec {
  name: string;
  label: string;
  description?: string;
  kind: FieldKind;
  nullable: boolean;
  required: boolean;
  defaultValue: unknown;
  options?: FieldOption[];
  min?: number;
  max?: number;
}

export type FieldValues = Record<string, unknown>;

function humanize(name: string): string {
  const words = name.replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Inlines a `$ref` to `$defs`, keeping the referencing property's own title, default and description. */
function resolveRef(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  const { $ref, ...overrides } = schema;
  const target = root.$defs?.[$ref.split("/").pop() ?? ""];
  return target ? { ...target, ...overrides } : schema;
}

/** Collapses pydantic's `anyOf: [X, {type: "null"}]` into X plus a nullable flag. */
function unwrapNullable(schema: JsonSchema, root: JsonSchema): { schema: JsonSchema; nullable: boolean } {
  const resolved = resolveRef(schema, root);
  const variants = resolved.anyOf;
  if (!variants) {
    const nullable = Array.isArray(resolved.type) ? resolved.type.includes("null") : resolved.type === "null";
    return { schema: resolved, nullable };
  }
  const nonNull = variants.filter((variant) => variant.type !== "null");
  const nullable = nonNull.length < variants.length;
  const inner = nonNull.length === 1 ? resolveRef(nonNull[0]!, root) : { type: "string" };
  return {
    schema: {
      ...inner,
      title: resolved.title ?? inner.title,
      description: resolved.description ?? inner.description,
      default: resolved.default,
    },
    nullable,
  };
}

const MULTILINE_NAME = /private_key|certificate|pem/i;

/**
 * Fields for a JSON Schema object. A non-nullable property with a default is required (it is
 * prefilled). With `secret`, text fields are write-only: masked, or multi-line for keys.
 */
export function fieldsFromJsonSchema(root: JsonSchema, options: { secret?: boolean } = {}): FieldSpec[] {
  const required = new Set(root.required ?? []);
  return Object.entries(root.properties ?? {}).map(([name, property]) => {
    const { schema, nullable } = unwrapNullable(property, root);
    const hasDefault = schema.default !== undefined && schema.default !== null;
    const base = {
      name,
      label: schema.title ?? humanize(name),
      description: schema.description,
      nullable,
      required: !nullable && (required.has(name) || (hasDefault && !options.secret)),
      defaultValue: options.secret ? null : (schema.default ?? null),
      min: schema.minimum,
      max: schema.maximum,
    };
    if (schema["x-kind"] === "checkpoint") return { ...base, kind: "checkpoint" };
    const choices = schema.enum ?? (schema.const !== undefined ? [schema.const] : undefined);
    if (choices) {
      return {
        ...base,
        kind: "enum",
        options: choices
          .filter((value) => value !== null)
          .map((value) => ({ value: value as string | number, label: String(value) })),
      };
    }
    const type = Array.isArray(schema.type) ? schema.type.find((item) => item !== "null") : schema.type;
    if (type === "integer" || type === "number" || type === "boolean") return { ...base, kind: type };
    if (options.secret) return { ...base, kind: MULTILINE_NAME.test(name) ? "multiline" : "secret" };
    return { ...base, kind: MULTILINE_NAME.test(name) ? "multiline" : "string" };
  });
}

export function fieldDefaults(fields: readonly FieldSpec[]): FieldValues {
  return Object.fromEntries(
    fields.map((field) => [field.name, field.defaultValue ?? (field.kind === "boolean" ? false : null)]),
  );
}

function fieldValidator(field: FieldSpec): z.ZodType {
  let validator: z.ZodType;
  switch (field.kind) {
    case "integer":
    case "number": {
      let number = z.number({ error: "Enter a number" });
      if (field.kind === "integer") number = number.int("Enter a whole number");
      if (field.min !== undefined) number = number.min(field.min, `Must be at least ${field.min}`);
      if (field.max !== undefined) number = number.max(field.max, `Must be at most ${field.max}`);
      validator = number;
      break;
    }
    case "boolean":
      validator = z.boolean();
      break;
    case "enum": {
      const values = (field.options ?? []).map((option) => option.value);
      validator = z.union(values.map((value) => z.literal(value)) as [z.ZodLiteral, ...z.ZodLiteral[]], {
        error: "Choose one of the options",
      });
      break;
    }
    default:
      validator = field.required ? z.string({ error: "Required" }).min(1, "Required") : z.string();
  }
  if (field.nullable || !field.required) return validator.nullable().optional();
  return validator;
}

export function fieldsValidator(fields: readonly FieldSpec[]) {
  return z.object(Object.fromEntries(fields.map((field) => [field.name, fieldValidator(field)])));
}

/** Values that differ from the field defaults (what a custom run changes). */
export function changedValues(fields: readonly FieldSpec[], values: FieldValues): FieldValues {
  return Object.fromEntries(
    fields
      .filter((field) => JSON.stringify(values[field.name] ?? null) !== JSON.stringify(field.defaultValue ?? null))
      .map((field) => [field.name, values[field.name]]),
  );
}

/** Drops empty optional values so write-only secrets and partial configs send only what was typed. */
export function compactValues(values: FieldValues): FieldValues {
  return Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== null && value !== undefined && value !== ""),
  );
}
