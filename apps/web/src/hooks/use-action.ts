"use client";

import { useTransition } from "react";
import { toast } from "sonner";

import { describeError, type Result } from "@/lib/api/errors";

interface ActionMessages<T> {
  success: string | ((data: T) => string);
  error: string;
}

/**
 * Runs a server action that returns a Result, with a pending flag and success/failure toasts.
 * Resolves to the same Result once the toast is shown.
 */
export function useAction() {
  const [pending, startTransition] = useTransition();

  function run<T>(action: () => Promise<Result<T>>, messages: ActionMessages<T>): Promise<Result<T>> {
    return new Promise((resolve) => {
      startTransition(async () => {
        const result = await action();
        if (result.ok) {
          toast.success(typeof messages.success === "function" ? messages.success(result.data) : messages.success);
        } else {
          toast.error(messages.error, { description: describeError(result.error) });
        }
        resolve(result);
      });
    });
  }

  return { pending, run };
}
