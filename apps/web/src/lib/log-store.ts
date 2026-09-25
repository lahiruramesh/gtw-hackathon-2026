import type { LogLine } from "@/lib/api/types";

export const MAX_LOG_LINES = 50_000;

/**
 * In-memory log buffer for one run: ordered by id, deduplicated, capped (oldest dropped).
 * Exposes a version number for useSyncExternalStore.
 */
export class LogStore {
  private buffer: LogLine[] = [];
  private listeners = new Set<() => void>();
  private versionCounter = 0;
  private evicted = 0;

  get lines(): readonly LogLine[] {
    return this.buffer;
  }

  get lastId(): number {
    return this.buffer.at(-1)?.id ?? 0;
  }

  /** Lines dropped from the front of the buffer to stay under MAX_LOG_LINES. */
  get evictedCount(): number {
    return this.evicted;
  }

  readonly version = (): number => this.versionCounter;

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  append(lines: readonly LogLine[]): void {
    const fresh = lines.filter((line) => line.id > this.lastId);
    if (fresh.length === 0) return;
    this.buffer = this.buffer.concat(fresh);
    const overflow = this.buffer.length - MAX_LOG_LINES;
    if (overflow > 0) {
      this.buffer = this.buffer.slice(overflow);
      this.evicted += overflow;
    }
    this.versionCounter += 1;
    for (const listener of this.listeners) listener();
  }
}
