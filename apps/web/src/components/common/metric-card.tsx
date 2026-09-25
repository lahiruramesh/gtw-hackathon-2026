import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: LucideIcon;
}

export function MetricCard({ label, value, hint, icon: Icon }: MetricCardProps) {
  return (
    <Card className="gap-0 py-4">
      <CardContent className="space-y-1 px-4">
        <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
          {label}
          {Icon && <Icon className="size-4" aria-hidden />}
        </div>
        <div className="tabular text-2xl font-semibold tracking-tight">{value}</div>
        {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
      </CardContent>
    </Card>
  );
}
