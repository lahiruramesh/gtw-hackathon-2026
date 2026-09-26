import { StatusBadge } from "@/components/common/status-badge";
import type { GateVerdict } from "@/lib/api/types";
import { GATE_LEVELS } from "@/lib/gate-levels";
import { cn } from "cn";

interface GateLevelBadgesProps {
  simulation: GateVerdict | null;
  release: GateVerdict | null;
  /** Full level names ("Simulation gate passed") instead of the table's short ones ("Sim passed"). */
  long?: boolean;
  className?: string;
}

/** Simulation and hardware release verdicts side by side; a skill without simulation criteria shows one. */
export function GateLevelBadges({ simulation, release, long = false, className }: GateLevelBadgesProps) {
  if (!simulation && !release) return <span className="text-muted-foreground">—</span>;
  const name = (level: keyof typeof GATE_LEVELS) => (long ? GATE_LEVELS[level].label : GATE_LEVELS[level].short);
  return (
    <span className={cn("inline-flex flex-wrap items-center gap-1", className)}>
      {simulation && <StatusBadge domain="gate" status={simulation} prefix={name("simulation")} />}
      {release && <StatusBadge domain="gate" status={release} prefix={name("release")} />}
    </span>
  );
}
