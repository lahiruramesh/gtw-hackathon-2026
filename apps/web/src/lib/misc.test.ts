import { describe, expect, it } from "vitest";

import { DEFAULT_WEIGHTS, METHODS, REQUIREMENTS } from "@/content/methods";
import type { ComputeTarget, LogLine } from "@/lib/api/types";
import { formatBytes, formatCompact, formatDuration, humanizeKey } from "@/lib/format";
import { splitHighlight } from "@/lib/highlight";
import { buildLineage } from "@/lib/lineage";
import { LogStore, MAX_LOG_LINES } from "@/lib/log-store";
import { appendMetricPoint, defaultMetricKeys, orderSelection } from "@/lib/metrics";
import { rankMethods, weightedScore } from "@/lib/methods-ranking";
import { permissionChanges, permissionsFor, roleLabel } from "@/lib/permissions";
import { defaultTargetId } from "@/lib/run-draft";
import { safeNextPath } from "@/lib/safe-redirect";
import { gateStageVerdict } from "@/lib/status";

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

  it("agrees with the written recommendation for locomotion and workflows under the default weights", () => {
    expect(rankMethods(METHODS, "locomotion", DEFAULT_WEIGHTS)[0]!.method.id).toBe("sim2real");
    expect(rankMethods(METHODS, "workflow", DEFAULT_WEIGHTS)[0]!.method.id).toBe("manual");
    const learned = rankMethods(METHODS, "manipulation", DEFAULT_WEIGHTS).map((item) => item.method.id);
    expect(learned.indexOf("teleop")).toBeLessThan(learned.indexOf("rl"));
    expect(learned.indexOf("imitation")).toBeLessThan(learned.indexOf("rl"));
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
    // URL parsing drops tabs and newlines: "/\t/evil.example" would become "//evil.example".
    for (const char of ["\t", "\n", "\r", "\u0000", "\u007f"]) {
      expect(safeNextPath(`/${char}/evil.example`)).toBe("/");
    }
    expect(safeNextPath(decodeURIComponent("/%09/evil.example"))).toBe("/");
    expect(safeNextPath("/runs/new?skill=g1-stairs#top")).toBe("/runs/new?skill=g1-stairs#top");
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

describe("gateStageVerdict", () => {
  it("shows the verdict of a gate stage that ran, never its execution status", () => {
    expect(gateStageVerdict({ kind: "gate", status: "succeeded" }, "fail")).toBe("fail");
    expect(gateStageVerdict({ kind: "gate", status: "succeeded" }, "pass")).toBe("pass");
    expect(gateStageVerdict({ kind: "gate", status: "pending" }, null)).toBeNull();
    expect(gateStageVerdict({ kind: "train", status: "succeeded" }, "fail")).toBeNull();
  });
});

describe("defaultTargetId", () => {
  const target = (id: string, kind: ComputeTarget["kind"], enabled = true) => ({ id, kind, enabled }) as ComputeTarget;

  it("starts smoke tests on the CPU and training on a GPU target", () => {
    const targets = [target("cpu", "local_cpu"), target("off", "aws_ec2", false), target("t4", "kaggle")];
    expect(defaultTargetId(targets, true)).toBe("cpu");
    expect(defaultTargetId(targets, false)).toBe("t4");
    expect(defaultTargetId([target("cpu", "local_cpu")], false)).toBe("cpu");
    expect(defaultTargetId([target("off", "kaggle", false)], false)).toBeNull();
  });
});

describe("permissionChanges", () => {
  it("lists what a role change grants and removes", () => {
    const promote = permissionChanges("operator", "ml_engineer");
    expect(promote.gained).toContain("run:create_custom");
    expect(promote.lost).toEqual([]);
    expect(permissionChanges("ml_engineer", "operator").lost).toEqual(promote.gained);
  });
});
