import { Skeleton } from "@/components/ui/skeleton";

/** Loading placeholder shaped like a typical page: header, metric row, table. */
export function PageSkeleton({ metrics = 0, rows = 6 }: { metrics?: number; rows?: number }) {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading">
      <div className="space-y-2">
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-4 w-72" />
      </div>
      {metrics > 0 && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          {Array.from({ length: metrics }, (_, index) => (
            <Skeleton key={index} className="h-[92px]" />
          ))}
        </div>
      )}
      <div className="space-y-2 rounded-lg border p-4">
        {Array.from({ length: rows }, (_, index) => (
          <Skeleton key={index} className="h-8 w-full" />
        ))}
      </div>
    </div>
  );
}
