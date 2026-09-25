import type { ComputeKind } from "@/lib/api/types";

export const COMPUTE_KIND_LABELS: Record<ComputeKind, string> = {
  local_cpu: "Local CPU",
  kaggle: "Kaggle",
  aws_ec2: "AWS EC2",
};

export const COMPUTE_KINDS = Object.keys(COMPUTE_KIND_LABELS) as [ComputeKind, ...ComputeKind[]];
