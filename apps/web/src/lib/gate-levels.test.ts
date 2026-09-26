import { describe, expect, it } from "vitest";

import { groupByLevel } from "@/lib/gate-levels";

describe("groupByLevel", () => {
  it("puts simulation criteria first and drops empty levels", () => {
    const criteria = [
      { metric: "err", level: "release" as const },
      { metric: "falls", level: "simulation" as const },
      { metric: "push", level: "simulation" as const },
    ];
    expect(groupByLevel(criteria).map(({ level, items }) => [level, items.map((item) => item.metric)])).toEqual([
      ["simulation", ["falls", "push"]],
      ["release", ["err"]],
    ]);
    expect(groupByLevel([{ level: "release" as const }]).map((group) => group.level)).toEqual(["release"]);
  });
});
