import { cn } from "cn";

export function JsonView({ value, className }: { value: unknown; className?: string }) {
  return (
    <pre
      className={cn(
        "max-h-[480px] overflow-auto rounded-md border bg-muted/50 p-3 font-mono text-xs leading-relaxed",
        className,
      )}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
