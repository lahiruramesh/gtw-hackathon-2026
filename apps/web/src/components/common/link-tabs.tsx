import Link from "next/link";

import { cn } from "cn";

export interface LinkTab {
  value: string;
  label: string;
  href: string;
}

/** Server-rendered tabs whose state lives in the URL (?tab=). */
export function LinkTabs({ tabs, active, label }: { tabs: LinkTab[]; active: string; label: string }) {
  return (
    <nav
      aria-label={label}
      className="inline-flex h-9 max-w-full items-center overflow-x-auto rounded-lg bg-muted p-[3px] text-muted-foreground"
    >
      {tabs.map((tab) => {
        const current = tab.value === active;
        return (
          <Link
            key={tab.value}
            href={tab.href}
            scroll={false}
            aria-current={current ? "page" : undefined}
            className={cn(
              "inline-flex h-full items-center rounded-md px-3 text-sm font-medium whitespace-nowrap transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
              current ? "bg-background text-foreground shadow-sm" : "hover:text-foreground",
            )}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
