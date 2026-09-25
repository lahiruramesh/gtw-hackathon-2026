import { CloudOffIcon, LockIcon, TriangleAlertIcon } from "lucide-react";

import { EmptyState } from "@/components/common/empty-state";
import { RetryButton } from "@/components/common/retry-button";
import { API_UNREACHABLE, type ApiErrorInfo } from "@/lib/api/errors";

interface ApiErrorStateProps {
  error: ApiErrorInfo;
  /** What failed to load, e.g. "runs". */
  subject?: string;
  className?: string;
}

export function ApiErrorState({ error, subject = "this page", className }: ApiErrorStateProps) {
  if (error.code === API_UNREACHABLE) {
    return (
      <EmptyState
        className={className}
        icon={CloudOffIcon}
        title="The API is not reachable"
        description={`Couldn't load ${subject}. The Skill Studio API may be starting up or down. Check that it runs on the configured API_INTERNAL_URL.`}
        action={<RetryButton />}
      />
    );
  }
  if (error.status === 403) {
    return (
      <EmptyState
        className={className}
        icon={LockIcon}
        title="You don't have access"
        description={error.message || `Your role can't view ${subject}. Ask an admin if you need access.`}
      />
    );
  }
  return (
    <EmptyState
      className={className}
      icon={TriangleAlertIcon}
      title={`Couldn't load ${subject}`}
      description={`${error.message} (${error.code})`}
      action={<RetryButton />}
    />
  );
}
