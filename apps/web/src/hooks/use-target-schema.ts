"use client";

import { useQuery } from "@tanstack/react-query";

import { clientFetch } from "@/lib/api/client";
import type { ComputeKind, ComputeTargetSchema } from "@/lib/api/types";

export function useTargetSchema(kind: ComputeKind, enabled = true) {
  return useQuery({
    queryKey: ["compute-schema", kind],
    queryFn: ({ signal }) => clientFetch<ComputeTargetSchema>(`/compute-targets/schema/${kind}`, { signal }),
    enabled,
    staleTime: Infinity,
  });
}
