import { proxyPath } from "@/lib/api/client";
import type { RunEvent } from "@/lib/api/types";

export type ConnectionState = "connecting" | "live" | "reconnecting" | "closed";

type Listener = (event: RunEvent) => void;

const EVENT_TYPES = ["log", "metric", "stage", "run"] as const;
const MAX_RETRY_MS = 30_000;

/**
 * One server-sent-event connection to `/runs/{id}/events` through the proxy. Reconnects with
 * backoff, resuming after the last log line it has seen (the browser's built-in retry would
 * replay from the original URL).
 */
export class RunEventStream {
  private source: EventSource | null = null;
  private listeners = new Set<Listener>();
  private retryMs = 1000;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private stopped = false;

  constructor(
    private readonly runId: string,
    private readonly afterLogId: () => number,
    private readonly onState: (state: ConnectionState) => void,
  ) {}

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  start(): void {
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    clearTimeout(this.retryTimer);
    this.source?.close();
    this.source = null;
    this.onState("closed");
  }

  private open(): void {
    this.onState(this.retryMs > 1000 ? "reconnecting" : "connecting");
    const url = proxyPath(`/runs/${encodeURIComponent(this.runId)}/events`, { after_log_id: this.afterLogId() });
    const source = new EventSource(url);
    this.source = source;

    source.onopen = () => {
      this.retryMs = 1000;
      this.onState("live");
    };
    source.onerror = () => {
      source.close();
      if (this.stopped) return;
      this.onState("reconnecting");
      this.retryTimer = setTimeout(() => this.open(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, MAX_RETRY_MS);
    };
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (message: MessageEvent<string>) => {
        let data: unknown;
        try {
          data = JSON.parse(message.data);
        } catch {
          return;
        }
        const event = { type, data } as RunEvent;
        for (const listener of this.listeners) listener(event);
      });
    }
  }
}
