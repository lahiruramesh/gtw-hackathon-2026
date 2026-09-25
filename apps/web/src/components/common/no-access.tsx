import { LockIcon } from "lucide-react";

import { EmptyState } from "@/components/common/empty-state";

export function NoAccess({ what }: { what: string }) {
  return (
    <EmptyState
      icon={LockIcon}
      title="You don't have access"
      description={`Your role can't ${what}. Ask an admin if you need access.`}
    />
  );
}
