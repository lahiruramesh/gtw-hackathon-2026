import { SearchXIcon } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/common/empty-state";
import { Button } from "@/components/ui/button";

export default function AppNotFound() {
  return (
    <EmptyState
      className="mt-10"
      icon={SearchXIcon}
      title="Not found"
      description="This skill, run or page doesn't exist, or it was removed."
      action={
        <Button asChild variant="outline" size="sm">
          <Link href="/">Back to dashboard</Link>
        </Button>
      }
    />
  );
}
