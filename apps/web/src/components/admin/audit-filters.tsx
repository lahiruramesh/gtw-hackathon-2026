"use client";

import { XIcon } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const FILTERS = [
  { key: "action", label: "Action", placeholder: "e.g. run.create" },
  { key: "entity_type", label: "Entity type", placeholder: "e.g. run" },
  { key: "actor_id", label: "Actor id", placeholder: "User id" },
] as const;

export function AuditFilters() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [pending, startTransition] = useTransition();
  const [values, setValues] = useState(() =>
    Object.fromEntries(FILTERS.map(({ key }) => [key, searchParams.get(key) ?? ""])),
  );

  function apply(next: Record<string, string>) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(next)) if (value.trim()) params.set(key, value.trim());
    startTransition(() =>
      router.replace(params.size ? `${pathname}?${params.toString()}` : pathname, { scroll: false }),
    );
  }

  return (
    <form
      className="flex flex-wrap items-end gap-3"
      aria-busy={pending}
      onSubmit={(event) => {
        event.preventDefault();
        apply(values);
      }}
    >
      {FILTERS.map((filter) => (
        <div key={filter.key} className="w-full space-y-1.5 sm:w-48">
          <Label htmlFor={`audit-${filter.key}`} className="text-xs">
            {filter.label}
          </Label>
          <Input
            id={`audit-${filter.key}`}
            value={values[filter.key]}
            placeholder={filter.placeholder}
            onChange={(event) => setValues((current) => ({ ...current, [filter.key]: event.target.value }))}
          />
        </div>
      ))}
      <Button type="submit" variant="outline" disabled={pending}>
        Apply
      </Button>
      {searchParams.size > 0 && (
        <Button
          type="button"
          variant="ghost"
          onClick={() => {
            const cleared = Object.fromEntries(FILTERS.map(({ key }) => [key, ""]));
            setValues(cleared);
            apply(cleared);
          }}
        >
          <XIcon /> Clear
        </Button>
      )}
    </form>
  );
}
