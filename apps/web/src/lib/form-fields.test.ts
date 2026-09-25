import { describe, expect, it } from "vitest";

import type { JsonSchema } from "@/lib/api/types";
import { changedValues, compactValues, fieldDefaults, fieldsFromJsonSchema, fieldsValidator } from "@/lib/form-fields";

/** Shape of `SkillDetail.params_schema` as the API sends it (g1-stairs). */
const params: JsonSchema = {
  type: "object",
  properties: {
    timesteps: { title: "Training steps", default: 200000000, type: "integer", minimum: 20000, maximum: 2000000000 },
    smoke: { title: "Smoke test", default: false, type: "boolean" },
    lr: { title: "Learning rate", default: null, type: ["number", "null"], minimum: 0.000001, maximum: 0.01 },
    scan_model: { default: "uniform", type: "string", enum: ["uniform", "camera"] },
    init_from: { default: null, format: "uuid", "x-kind": "checkpoint", type: ["string", "null"] },
  },
  additionalProperties: false,
};

describe("params schema fields", () => {
  const fields = fieldsFromJsonSchema(params);

  it("maps manifest params to fields", () => {
    expect(fields.map((field) => [field.name, field.kind, field.required, field.nullable])).toEqual([
      ["timesteps", "integer", true, false],
      ["smoke", "boolean", true, false],
      ["lr", "number", false, true],
      ["scan_model", "enum", true, false],
      ["init_from", "checkpoint", false, true],
    ]);
    expect(fieldDefaults(fields)).toEqual({
      timesteps: 200000000,
      smoke: false,
      lr: null,
      scan_model: "uniform",
      init_from: null,
    });
  });

  it("validates ranges, integers and enum values", () => {
    const validator = fieldsValidator(fields);
    const valid = { timesteps: 20000, smoke: true, lr: 1e-4, scan_model: "camera", init_from: null };
    expect(validator.safeParse(valid).success).toBe(true);
    expect(validator.safeParse({ ...valid, lr: null }).success).toBe(true);
    expect(validator.safeParse({ ...valid, timesteps: 100 }).success).toBe(false);
    expect(validator.safeParse({ ...valid, timesteps: 20000.5 }).success).toBe(false);
    expect(validator.safeParse({ ...valid, scan_model: "lidar" }).success).toBe(false);
    expect(validator.safeParse({ ...valid, timesteps: null }).success).toBe(false);
  });

  it("reports only changed values", () => {
    expect(changedValues(fields, { ...fieldDefaults(fields), lr: 1e-4 })).toEqual({ lr: 1e-4 });
  });
});

describe("fieldsFromJsonSchema", () => {
  const config: JsonSchema = {
    type: "object",
    properties: {
      region: { type: "string", title: "Region" },
      accelerator: { $ref: "#/$defs/Accelerator", default: "NvidiaTeslaT4" },
      aws_profile: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Aws Profile" },
      stop_when_idle: { type: "boolean", default: true },
    },
    required: ["region"],
    $defs: { Accelerator: { enum: ["NvidiaTeslaT4", "NvidiaTeslaP100"], type: "string" } },
  };

  it("resolves refs and optional values from pydantic schemas", () => {
    const fields = fieldsFromJsonSchema(config);
    expect(fields.map((field) => [field.name, field.kind, field.nullable, field.required])).toEqual([
      ["region", "string", false, true],
      ["accelerator", "enum", false, true],
      ["aws_profile", "string", true, false],
      ["stop_when_idle", "boolean", false, true],
    ]);
    expect(fields[1]?.defaultValue).toBe("NvidiaTeslaT4");
  });

  it("makes secret text write-only and never prefilled", () => {
    const secret: JsonSchema = {
      type: "object",
      properties: { ssh_private_key: { type: "string" }, aws_access_key_id: { type: "string", default: "x" } },
      required: ["ssh_private_key"],
    };
    const fields = fieldsFromJsonSchema(secret, { secret: true });
    expect(fields.map((field) => [field.kind, field.defaultValue])).toEqual([
      ["multiline", null],
      ["secret", null],
    ]);
    expect(compactValues({ ssh_private_key: "k", aws_access_key_id: "" })).toEqual({ ssh_private_key: "k" });
  });
});
