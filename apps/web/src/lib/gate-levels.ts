import type { GateLevel } from "@/lib/api/types";

/**
 * The two gates of the pipeline (docs/pipeline.md §6): a policy first has to work in simulation, then meet the
 * hardware release bar. The release gate includes the simulation criteria, so it can only pass after them.
 */
export const GATE_LEVELS: Record<GateLevel, { label: string; short: string; description: string }> = {
  simulation: {
    label: "Simulation gate",
    short: "Sim",
    description: "Works in simulation under nominal conditions.",
  },
  release: {
    label: "Hardware release gate",
    short: "Release",
    description: "The bar before gantry and floor trials. It includes the simulation criteria.",
  },
};

const ORDER: readonly GateLevel[] = ["simulation", "release"];

/** Items grouped by gate level, simulation first; levels without items are left out. */
export function groupByLevel<T extends { level: GateLevel }>(items: readonly T[]): { level: GateLevel; items: T[] }[] {
  return ORDER.map((level) => ({ level, items: items.filter((item) => item.level === level) })).filter(
    (group) => group.items.length > 0,
  );
}
