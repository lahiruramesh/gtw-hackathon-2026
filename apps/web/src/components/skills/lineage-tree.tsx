import Link from "next/link";

import { StatusBadge } from "@/components/common/status-badge";
import type { RunSummary } from "@/lib/api/types";
import { formatCompact } from "@/lib/format";
import { formatHeadline } from "@/lib/headline";
import type { LineageNode } from "@/lib/lineage";

function LineageItem({ node }: { node: LineageNode<RunSummary> }) {
  const run = node.value;
  const step = run.parent?.checkpoint?.step;
  return (
    <li className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border px-3 py-2">
        <Link href={`/runs/${run.id}`} className="font-medium hover:underline">
          {run.name}
        </Link>
        <StatusBadge domain="run" status={run.status} />
        {run.parent && (
          <span className="text-xs text-muted-foreground">
            warm start from {run.parent.run.name}
            {step !== undefined && step !== null && ` @ ${formatCompact(step)} steps`}
          </span>
        )}
        {run.headline.length > 0 && (
          <span className="ml-auto text-xs text-muted-foreground">
            {run.headline.map((item) => `${item.label} ${formatHeadline(item)}`).join(" · ")}
          </span>
        )}
      </div>
      {node.children.length > 0 && (
        <ul className="ml-4 space-y-2 border-l border-muted pl-4">
          {node.children.map((child) => (
            <LineageItem key={child.id} node={child} />
          ))}
        </ul>
      )}
    </li>
  );
}

export function LineageTree({ roots }: { roots: LineageNode<RunSummary>[] }) {
  return (
    <ul className="space-y-2">
      {roots.map((node) => (
        <LineageItem key={node.id} node={node} />
      ))}
    </ul>
  );
}
