"use client";

import { useSyncExternalStore } from "react";

function subscribe(onTick: () => void): () => void {
  const timer = setInterval(onTick, 1000);
  return () => clearInterval(timer);
}

/** Current time in whole seconds (ms), ticking once a second; null during server rendering. */
export function useNow(): number | null {
  return useSyncExternalStore(
    subscribe,
    () => Math.floor(Date.now() / 1000) * 1000,
    () => null,
  );
}
