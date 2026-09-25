import { CpuIcon } from "lucide-react";
import type { Metadata } from "next";

import { ApiErrorState } from "@/components/common/api-error-state";
import { EmptyState } from "@/components/common/empty-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { TargetCard } from "@/components/compute/target-card";
import { TargetFormDialog } from "@/components/compute/target-form-dialog";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { ComputeTarget } from "@/lib/api/types";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Compute" };

export default async function ComputePage() {
  const viewer = await requireViewer();
  if (!can(viewer, "compute:read")) return <NoAccess what="view compute targets" />;

  const canManage = can(viewer, "compute:write");
  const result = await toResult(apiFetch<ComputeTarget[]>("/compute-targets"));

  return (
    <div className="space-y-5">
      <PageHeader
        title="Compute"
        description="Where training runs: GPU targets, their health, usage and weekly quota."
        actions={canManage && <TargetFormDialog />}
      />
      {!result.ok ? (
        <ApiErrorState error={result.error} subject="compute targets" />
      ) : result.data.length === 0 ? (
        <EmptyState
          icon={CpuIcon}
          title="No compute targets yet"
          description={
            canManage
              ? "Add a local CPU target for smoke runs, then Kaggle or AWS for real training."
              : "An admin needs to add a compute target."
          }
          action={canManage && <TargetFormDialog />}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {result.data.map((target) => (
            <TargetCard key={target.id} target={target} canManage={canManage} />
          ))}
        </div>
      )}
    </div>
  );
}
