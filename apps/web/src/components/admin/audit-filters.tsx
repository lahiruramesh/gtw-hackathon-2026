"use client";

import { XIcon } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const TEXT_FILTERS = [
  { key: "action", label: "Action", placeholder: "e.g. run.create" },
  { key: "entity_type", label: "Entity type", placeholder: "e.g. run" },
] as const;
const FILTER_KEYS = [...TEXT_FILTERS.map(({ key }) => key), "actor_id"] as const;
const ANY_ACTOR = "any";

export interface AuditActor {
  id: string;
  label: string;
}

export function AuditFilters({ actors }: { actors: AuditActor[] }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [pending, startTransition] = useTransition();
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(FILTER_KEYS.map((key) => [key, searchParams.get(key) ?? ""])),
  );
  // An actor from the URL that is not a current account (e.g. a deleted user) stays selectable by id.
  const options =
    values.actor_id && !actors.some((actor) => actor.id === values.actor_id)
      ? [...actors, { id: values.actor_id, label: values.actor_id }]
      : actors;

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
      {TEXT_FILTERS.map((filter) => (
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
      <div className="w-full space-y-1.5 sm:w-64">
        <Label htmlFor="audit-actor_id" className="text-xs">
          Actor
        </Label>
        <Select
          value={values.actor_id || ANY_ACTOR}
          onValueChange={(value) =>
            setValues((current) => ({ ...current, actor_id: value === ANY_ACTOR ? "" : value }))
          }
        >
          <SelectTrigger id="audit-actor_id" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ANY_ACTOR}>Anyone</SelectItem>
            {options.map((actor) => (
              <SelectItem key={actor.id} value={actor.id}>
                {actor.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <Button type="submit" variant="outline" disabled={pending}>
        Apply
      </Button>
      {searchParams.size > 0 && (
        <Button
          type="button"
          variant="ghost"
          onClick={() => {
            const cleared = Object.fromEntries(FILTER_KEYS.map((key) => [key, ""]));
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
