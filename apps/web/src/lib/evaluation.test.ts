import { describe, expect, it } from "vitest";

import { cellKey, describeEvaluation, formatEvalValue } from "@/lib/evaluation";

describe("describeEvaluation", () => {
  it("splits a step-length summary into tiles, heatmap and tables", () => {
    const view = describeEvaluation({
      grid_fall_rate: 0,
      grid_step_abs_err_cm: 10.68,
      grid: [
        { vx: 0.3, step_req: 0.15, seed: 0, fell: false, step_abs_err_m: 0.08 },
        { vx: 0.3, step_req: 0.15, seed: 1, fell: true, step_abs_err_m: 0.1 },
        { vx: 0.5, step_req: 0.25, seed: 0, fell: false, step_abs_err_m: 0.02 },
      ],
      stress: { nominal: { fall_rate: 0, step_abs_err_cm: 20.3 }, "push_0.5mps": { fall_rate: 0 } },
      demo: { fell: false, n_steps: 36 },
    });
    expect(view.scalars.map((scalar) => scalar.key)).toEqual(["grid_fall_rate", "grid_step_abs_err_cm"]);
    expect(view.heatmap?.xs).toEqual([0.15, 0.25]);
    expect(view.heatmap?.ys).toEqual([0.3, 0.5]);
    expect(view.heatmap?.values[cellKey(0.15, 0.3)]).toBeCloseTo(9);
    expect(view.heatmap?.fallRates[cellKey(0.15, 0.3)]).toBe(0.5);
    expect(view.tables[0]?.key).toBe("stress");
    expect(view.tables[0]?.columns).toEqual(["case", "fall_rate", "step_abs_err_cm"]);
    expect(view.tables[0]?.rows.map((row) => row.id)).toEqual(["nominal", "push_0.5mps"]);
    expect(view.details[0]?.key).toBe("demo");
    expect(view.rest).toEqual({});
  });

  it("gives a stairs summary its by-height view, ordered by rise with the certified height marked", () => {
    const view = describeEvaluation({
      n: 96,
      crossed: 85,
      certified_cm: 6.43,
      // JSONB returns keys in its own order: the view must not depend on it.
      by_height: [
        { fell: 4, runs: 12, stable: false, crossed: 8, rise_cm: 12, max_tilt_deg: 31, min_pelvis_m: 0.5 },
        { fell: 0, runs: 12, stable: true, crossed: 12, rise_cm: 6.43, max_tilt_deg: 14, min_pelvis_m: 0.66 },
        { fell: 0, runs: 12, stable: true, crossed: 12, rise_cm: 3, max_tilt_deg: 9, min_pelvis_m: 0.7 },
      ],
    });
    expect(view.heatmap).toBeNull();
    expect(view.tables).toEqual([]);
    expect(view.heights?.map((height) => height.riseCm)).toEqual([3, 6.43, 12]);
    expect(view.heights?.map((height) => height.certified)).toEqual([false, true, false]);
    expect(view.heights?.[2]).toMatchObject({ runs: 12, crossed: 8, fell: 4, stable: false, maxTiltDeg: 31 });
  });

  it("falls back to a generic table for other by-height shapes", () => {
    const view = describeEvaluation({ by_height: [{ height_cm: 3, n: 12, crossed: 12, fell: 0 }] });
    expect(view.heights).toBeNull();
    expect(view.tables[0]?.columns).toEqual(["height_cm", "n", "crossed", "fell"]);
  });

  it("keeps unrecognised structures for the JSON fallback", () => {
    expect(describeEvaluation({ nested: { deep: { value: [1, 2] } } }).rest).toEqual({
      nested: { deep: { value: [1, 2] } },
    });
  });
});

describe("formatEvalValue", () => {
  it("formats by key convention", () => {
    expect(formatEvalValue("fall_rate", 0.25)).toBe("25.0%");
    expect(formatEvalValue("crossed_rate", 1)).toBe("100%");
    expect(formatEvalValue("certified_cm", 8.1)).toBe("8.1 cm");
    expect(formatEvalValue("n", 96)).toBe("96");
    expect(formatEvalValue("fell", false)).toBe("No");
    expect(formatEvalValue("x", null)).toBe("—");
  });
});
