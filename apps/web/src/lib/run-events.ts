import { proxyPath } from "@/lib/api/client";
import type { RunEvent } from "@/lib/api/types";

export type ConnectionState = "connecting" | "live" | "reconnecting" | "closed";

type Listener = (event: RunEvent) => void;

const EVENT_TYPES = ["log", "metric", "stage", "run"] as const;
const MAX_RETRY_MS = 30_000;
// EventSource hides the status of a failed connection, so after this many failures in a row we
// ask the proxy whether the session is still valid.
const SESSION_CHECK_AFTER_FAILURES = 3;

/**
 * One server-sent-event connection to `/runs/{id}/events` through the proxy. Reconnects with
 * backoff, resuming after the last log line it has seen (the browser's built-in retry would
 * replay from the original URL). Stops and calls `onSignedOut` once the session has expired.
 */
export class RunEventStream {
  private source: EventSource | null = null;
  private listeners = new Set<Listener>();
  private retryMs = 1000;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private failures = 0;
  private stopped = false;

  constructor(
    private readonly runId: string,
    private readonly afterLogId: () => number,
    private readonly onState: (state: ConnectionState) => void,
    private readonly onSignedOut: () => void,
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
      this.failures = 0;
      this.onState("live");
    };
    source.onerror = () => {
      source.close();
      if (this.stopped) return;
      this.failures += 1;
      if (this.failures % SESSION_CHECK_AFTER_FAILURES === 0) void this.checkSession();
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

  private async checkSession(): Promise<void> {
    let response: Response;
    try {
      response = await fetch(proxyPath("/me"), { cache: "no-store" });
    } catch {
      return; // offline: keep retrying
    }
    if (response.status === 401 && !this.stopped) {
      this.stop();
      this.onSignedOut();
    }
  }
}
