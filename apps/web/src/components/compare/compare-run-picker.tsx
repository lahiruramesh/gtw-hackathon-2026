"use client";

import { CheckIcon, PlusIcon, XIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTransition } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { MAX_COMPARE } from "@/lib/compare";
import { cn } from "cn";

interface Option {
  id: string;
  name: string;
  skillName: string;
}

export function CompareRunPicker({ options, selected }: { options: Option[]; selected: string[] }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  function update(ids: string[]) {
    startTransition(() =>
      router.replace(ids.length > 0 ? `/compare?runs=${ids.join(",")}` : "/compare", { scroll: false }),
    );
  }

  function toggle(id: string) {
    if (selected.includes(id)) update(selected.filter((item) => item !== id));
    else if (selected.length < MAX_COMPARE) update([...selected, id]);
  }

  const byId = new Map(options.map((option) => [option.id, option]));
  return (
    <div className="flex flex-wrap items-center gap-2" aria-busy={pending}>
      {selected.map((id) => (
        <Badge key={id} variant="secondary" className="h-7 gap-1 pr-1 text-sm font-normal">
          {byId.get(id)?.name ?? id}
          <Button
            variant="ghost"
            size="icon"
            className="size-5"
            onClick={() => toggle(id)}
            aria-label={`Remove ${byId.get(id)?.name ?? id}`}
          >
            <XIcon className="size-3" />
          </Button>
        </Badge>
      ))}
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" disabled={selected.length >= MAX_COMPARE}>
            <PlusIcon /> Add run
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80 p-0" align="start">
          <Command>
            <CommandInput placeholder="Search runs" />
            <CommandList>
              <CommandEmpty>No run matches.</CommandEmpty>
              <CommandGroup>
                {options.map((option) => (
                  <CommandItem
                    key={option.id}
                    value={`${option.name} ${option.skillName}`}
                    onSelect={() => toggle(option.id)}
                  >
                    <CheckIcon className={cn("size-4", selected.includes(option.id) ? "opacity-100" : "opacity-0")} />
                    <div className="min-w-0">
                      <div className="truncate">{option.name}</div>
                      <div className="truncate text-xs text-muted-foreground">{option.skillName}</div>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      <span className="text-xs text-muted-foreground">Pick 2 to {MAX_COMPARE} runs</span>
    </div>
  );
}
