import { describe, expect, it } from "vitest";

import { DEFAULT_WEIGHTS, METHODS, REQUIREMENTS } from "@/content/methods";
import type { LogLine } from "@/lib/api/types";
import { formatBytes, formatCompact, formatDuration, humanizeKey } from "@/lib/format";
import { splitHighlight } from "@/lib/highlight";
import { buildLineage } from "@/lib/lineage";
import { LogStore, MAX_LOG_LINES } from "@/lib/log-store";
import { appendMetricPoint, defaultMetricKeys, orderSelection } from "@/lib/metrics";
import { rankMethods, weightedScore } from "@/lib/methods-ranking";
import { permissionsFor, roleLabel } from "@/lib/permissions";
import { safeNextPath } from "@/lib/safe-redirect";

function line(id: number): LogLine {
  return { id, stage_id: "s", ts: "2026-09-25T10:00:00Z", level: "info", text: `line ${id}` };
}

describe("LogStore", () => {
  it("appends in order, ignores replayed lines and caps the buffer", () => {
    const store = new LogStore();
    store.append([line(1), line(2)]);
    store.append([line(2), line(3)]);
    expect(store.lines.map((item) => item.id)).toEqual([1, 2, 3]);
    expect(store.lastId).toBe(3);
    store.append(Array.from({ length: MAX_LOG_LINES }, (_, index) => line(index + 4)));
    expect(store.lines).toHaveLength(MAX_LOG_LINES);
    expect(store.evictedCount).toBe(3);
  });
});

describe("metrics", () => {
  it("puts the primary metric first", () => {
    expect(defaultMetricKeys(["a", "b", "p"], "p", ["b", "a"])).toEqual(["p", "b", "a"]);
    expect(orderSelection(["a", "p"], "p")).toEqual(["p", "a"]);
  });

  it("appends live points without duplicating steps", () => {
    const start = { keys: ["r"], series: [{ key: "r", points: [[1, 0.5]] as [number, number][] }] };
    const next = appendMetricPoint(appendMetricPoint(start, 2, { r: 0.7, s: 1 }), 2, { r: 0.9 });
    expect(next.series.find((item) => item.key === "r")?.points).toEqual([
      [1, 0.5],
      [2, 0.7],
    ]);
    expect(next.keys).toEqual(["r", "s"]);
  });
});

describe("buildLineage", () => {
  it("nests warm starts and treats unknown parents as roots", () => {
    const roots = buildLineage([
      { id: "v9", parentId: "v7", value: 9 },
      { id: "v10", parentId: "v9", value: 10 },
      { id: "v11", parentId: "v10", value: 11 },
      { id: "flat", parentId: null, value: 0 },
    ]);
    expect(roots.map((node) => node.id)).toEqual(["v9", "flat"]);
    expect(roots[0]?.children[0]?.children[0]?.id).toBe("v11");
  });
});

describe("methods ranking", () => {
  it("has a complete score for every method, domain and requirement", () => {
    for (const method of METHODS) {
      for (const assessment of Object.values(method.assessments)) {
        expect(Object.keys(assessment.scores).sort()).toEqual(REQUIREMENTS.map((item) => item.id).sort());
      }
    }
  });

  it("weights scores and falls back to equal weights", () => {
    const zero = Object.fromEntries(REQUIREMENTS.map((item) => [item.id, 0])) as typeof DEFAULT_WEIGHTS;
    const manual = METHODS[0]!;
    const equal = Object.values(manual.assessments.locomotion.scores).reduce((a, b) => a + b, 0) / REQUIREMENTS.length;
    expect(weightedScore(manual, "locomotion", zero)).toBeCloseTo(equal);
    const onlyData = { ...zero, data: 1 };
    expect(weightedScore(manual, "locomotion", onlyData)).toBe(manual.assessments.locomotion.scores.data);
    const ranking = rankMethods(METHODS, "locomotion", DEFAULT_WEIGHTS);
    expect(ranking).toHaveLength(METHODS.length);
    expect(ranking[0]!.score).toBeGreaterThanOrEqual(ranking.at(-1)!.score);
  });
});

describe("permissions", () => {
  it("reads roles from shared/permissions.json", () => {
    expect(permissionsFor("operator").has("run:create_preset")).toBe(true);
    expect(permissionsFor("operator").has("run:create_custom")).toBe(false);
    expect(permissionsFor("admin").has("user:manage")).toBe(true);
    expect(permissionsFor("nobody").size).toBe(0);
    expect(roleLabel("safety_reviewer")).toBe("Safety reviewer");
  });
});

describe("helpers", () => {
  it("only allows same-origin paths after login", () => {
    expect(safeNextPath("/runs/1?tab=logs")).toBe("/runs/1?tab=logs");
    expect(safeNextPath("//evil.com")).toBe("/");
    expect(safeNextPath("/\\evil.com")).toBe("/");
    expect(safeNextPath("https://evil.com")).toBe("/");
    expect(safeNextPath(null)).toBe("/");
  });

  it("highlights case-insensitively and escapes regex characters", () => {
    expect(splitHighlight("Error: a.b ERROR", "error")).toEqual([
      { text: "Error", match: true },
      { text: ": a.b ", match: false },
      { text: "ERROR", match: true },
    ]);
    expect(splitHighlight("a.b", ".")).toEqual([
      { text: "a", match: false },
      { text: ".", match: true },
      { text: "b", match: false },
    ]);
  });

  it("formats numbers for tool UIs", () => {
    expect(formatCompact(202_000_000)).toBe("202M");
    expect(formatDuration(5700)).toBe("1 h 35 min");
    expect(formatDuration(42)).toBe("42 s");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(humanizeKey("eval/episode_step_len_err")).toBe("Step len err");
  });
});
