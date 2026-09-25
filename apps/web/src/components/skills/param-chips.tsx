import { formatValue } from "@/lib/format";

/** Compact key=value list of run or preset parameters. */
export function ParamChips({ params }: { params: Record<string, unknown> }) {
  const entries = Object.entries(params);
  if (entries.length === 0) return <span className="text-xs text-muted-foreground">Skill defaults</span>;
  return (
    <ul className="flex flex-wrap gap-1.5">
      {entries.map(([key, value]) => (
        <li key={key} className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
          {key}={formatValue(value)}
        </li>
      ))}
    </ul>
  );
}
