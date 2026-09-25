import { CheckIcon } from "lucide-react";

import { cn } from "cn";

export const WIZARD_STEPS = ["Skill", "Parameters", "Compute", "Review"] as const;

/** 1-based step indicator. */
export function WizardSteps({ current }: { current: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm" aria-label="Steps">
      {WIZARD_STEPS.map((label, index) => {
        const step = index + 1;
        const done = step < current;
        const active = step === current;
        return (
          <li key={label} className="flex items-center gap-2" aria-current={active ? "step" : undefined}>
            <span
              className={cn(
                "flex size-6 items-center justify-center rounded-full border text-xs font-medium",
                done && "border-primary bg-primary text-primary-foreground",
                active && "border-primary text-foreground",
                !done && !active && "text-muted-foreground",
              )}
            >
              {done ? <CheckIcon className="size-3.5" /> : step}
            </span>
            <span className={active ? "font-medium" : "text-muted-foreground"}>{label}</span>
          </li>
        );
      })}
    </ol>
  );
}
