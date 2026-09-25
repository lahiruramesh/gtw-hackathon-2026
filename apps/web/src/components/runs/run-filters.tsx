"use client";

import { SearchIcon, XIcon } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { RunStatus } from "@/lib/api/types";
import { RUN_STATUS } from "@/lib/status";

const ALL = "all";

interface RunFiltersProps {
  skills: { id: string; name: string }[];
}

export function RunFilters({ skills }: RunFiltersProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [pending, startTransition] = useTransition();
  const [query, setQuery] = useState(searchParams.get("q") ?? "");

  function update(changes: Record<string, string | null>) {
    const next = new URLSearchParams(searchParams);
    for (const [key, value] of Object.entries(changes)) {
      if (value === null || value === "" || value === ALL) next.delete(key);
      else next.set(key, value);
    }
    next.delete("cursor");
    startTransition(() => router.replace(`${pathname}?${next.toString()}`, { scroll: false }));
  }

  const hasFilters = ["skill", "status", "mine", "q"].some((key) => searchParams.has(key));

  return (
    <div className="flex flex-wrap items-end gap-3" aria-busy={pending}>
      <form
        className="relative w-full sm:w-64"
        role="search"
        onSubmit={(event) => {
          event.preventDefault();
          update({ q: query.trim() });
        }}
      >
        <Label htmlFor="run-search" className="sr-only">
          Search runs
        </Label>
        <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          id="run-search"
          placeholder="Search by name"
          className="pl-8"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onBlur={() => query.trim() !== (searchParams.get("q") ?? "") && update({ q: query.trim() })}
        />
      </form>
      <Select value={searchParams.get("skill") ?? ALL} onValueChange={(value) => update({ skill: value })}>
        <SelectTrigger className="w-full sm:w-48" aria-label="Skill">
          <SelectValue placeholder="All skills" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All skills</SelectItem>
          {skills.map((skill) => (
            <SelectItem key={skill.id} value={skill.id}>
              {skill.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={searchParams.get("status") ?? ALL} onValueChange={(value) => update({ status: value })}>
        <SelectTrigger className="w-full sm:w-44" aria-label="Status">
          <SelectValue placeholder="Any status" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>Any status</SelectItem>
          {(Object.keys(RUN_STATUS) as RunStatus[]).map((status) => (
            <SelectItem key={status} value={status}>
              {RUN_STATUS[status].label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <div className="flex h-9 items-center gap-2">
        <Switch
          id="mine"
          checked={searchParams.get("mine") === "1"}
          onCheckedChange={(checked) => update({ mine: checked ? "1" : null })}
        />
        <Label htmlFor="mine">Only mine</Label>
      </div>
      {hasFilters && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            setQuery("");
            startTransition(() => router.replace(pathname, { scroll: false }));
          }}
        >
          <XIcon /> Clear
        </Button>
      )}
    </div>
  );
}
