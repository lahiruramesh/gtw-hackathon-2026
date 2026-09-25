"use client";

import { useSearchParams } from "next/navigation";
import { useState, type ReactNode } from "react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const RUN_TABS = [
  { value: "logs", label: "Logs" },
  { value: "metrics", label: "Metrics" },
  { value: "checkpoints", label: "Checkpoints" },
  { value: "evaluation", label: "Evaluation" },
  { value: "artifacts", label: "Artifacts" },
  { value: "gate", label: "Gate" },
  { value: "config", label: "Config" },
] as const;

export type RunTab = (typeof RUN_TABS)[number]["value"];

function isRunTab(value: string | null): value is RunTab {
  return RUN_TABS.some((tab) => tab.value === value);
}

/** Client tabs so the live log stream survives tab switches; the active tab is mirrored in ?tab=. */
export function RunTabs({ panels }: { panels: Record<RunTab, ReactNode> }) {
  const searchParams = useSearchParams();
  const requested = searchParams.get("tab");
  const [tab, setTab] = useState<RunTab>(isRunTab(requested) ? requested : "logs");

  function select(value: string) {
    if (!isRunTab(value)) return;
    setTab(value);
    const next = new URLSearchParams(searchParams);
    next.set("tab", value);
    window.history.replaceState(null, "", `?${next.toString()}`);
  }

  return (
    <Tabs value={tab} onValueChange={select} className="gap-4">
      <div className="overflow-x-auto">
        <TabsList>
          {RUN_TABS.map((item) => (
            <TabsTrigger key={item.value} value={item.value}>
              {item.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>
      {RUN_TABS.map((item) => (
        <TabsContent key={item.value} value={item.value}>
          {panels[item.value]}
        </TabsContent>
      ))}
    </Tabs>
  );
}
