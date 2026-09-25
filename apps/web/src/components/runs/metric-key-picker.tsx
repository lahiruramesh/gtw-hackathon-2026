"use client";

import { CheckIcon, ChevronsUpDownIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "cn";

interface MetricKeyPickerProps {
  keys: readonly string[];
  selected: readonly string[];
  onChange: (selected: string[]) => void;
}

export function MetricKeyPicker({ keys, selected, onChange }: MetricKeyPickerProps) {
  function toggle(key: string) {
    onChange(selected.includes(key) ? selected.filter((item) => item !== key) : [...selected, key]);
  }
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="justify-between" aria-label="Choose metrics">
          {selected.length} of {keys.length} metrics
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-0" align="start">
        <Command>
          <CommandInput placeholder="Filter metrics" />
          <CommandList>
            <CommandEmpty>No metric matches.</CommandEmpty>
            <CommandGroup>
              {keys.map((key) => (
                <CommandItem key={key} value={key} onSelect={() => toggle(key)}>
                  <CheckIcon className={cn("size-4", selected.includes(key) ? "opacity-100" : "opacity-0")} />
                  <span className="truncate font-mono text-xs">{key}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
