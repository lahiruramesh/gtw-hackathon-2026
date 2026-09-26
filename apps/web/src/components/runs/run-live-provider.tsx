"use client";

import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

import { clientFetch } from "@/lib/api/client";
import { toErrorInfo, type ApiErrorInfo } from "@/lib/api/errors";
import type { LogPage, RunDetail, RunEvent, RunSummary, Stage } from "@/lib/api/types";
import { LogStore } from "@/lib/log-store";
import { loginPath } from "@/lib/login-path";
import { RunEventStream, type ConnectionState } from "@/lib/run-events";
import { ACTIVE_STAGE_STATUSES, isRunActive, TERMINAL_STAGE_STATUSES } from "@/lib/status";

const LOG_PAGE_SIZE = 2000;
const REFRESH_THROTTLE_MS = 2000;
const SNAPSHOT_RETRY_MS = 2000;
const MAX_SNAPSHOT_RETRY_MS = 30_000;

type Listener = (event: RunEvent) => void;

interface RunLiveValue {
  /** Server snapshot with live stage and run updates applied. */
  run: RunDetail;
  connection: ConnectionState;
  logs: LogStore;
  logsLoading: boolean;
  logsError: ApiErrorInfo | null;
  subscribe: (listener: Listener) => () => void;
}

const RunLiveContext = createContext<RunLiveValue | null>(null);

export function useRunLive(): RunLiveValue {
  const value = useContext(RunLiveContext);
  if (!value) throw new Error("useRunLive must be used inside <RunLiveProvider>");
  return value;
}

interface LiveOverlay {
  /** The server-rendered snapshot the overlay was started from. */
  server: RunDetail;
  /** Latest full run: the server snapshot, or a newer one fetched after a live transition. */
  base: RunDetail;
  summary: RunSummary | null;
  stages: Record<string, Stage>;
}

function freshOverlay(server: RunDetail, base: RunDetail = server): LiveOverlay {
  return { server, base, summary: null, stages: {} };
}

const LIVE_SUMMARY_FIELDS = [
  "status",
  "current_stage",
  "gate_verdict",
  "simulation_verdict",
  "gpu_hours",
  "cost",
  "started_at",
  "finished_at",
  "headline",
] as const satisfies readonly (keyof RunSummary)[];

function applyOverlay(overlay: LiveOverlay): RunDetail {
  const { base, summary, stages } = overlay;
  const merged: RunDetail = { ...base };
  if (summary) {
    for (const field of LIVE_SUMMARY_FIELDS) Object.assign(merged, { [field]: summary[field] });
  }
  const known = new Set(base.stages.map((stage) => stage.id));
  merged.stages = [
    ...base.stages.map((stage) => stages[stage.id] ?? stage),
    ...Object.values(stages).filter((stage) => !known.has(stage.id)),
  ].sort((a, b) => a.position - b.position || a.attempt - b.attempt);
  return merged;
}

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason);
    });
  });
}

/** The run, fetched again until the API answers (the browser may be offline for a while). */
async function fetchRunUntilReachable(runId: string, signal: AbortSignal): Promise<RunDetail> {
  for (let delay = SNAPSHOT_RETRY_MS; ; delay = Math.min(delay * 2, MAX_SNAPSHOT_RETRY_MS)) {
    try {
      return await clientFetch<RunDetail>(`/runs/${encodeURIComponent(runId)}`, { signal });
    } catch (error) {
      if (signal.aborted) throw error;
    }
    await wait(delay, signal);
  }
}

async function loadLogHistory(runId: string, store: LogStore, signal: AbortSignal): Promise<void> {
  let afterId = store.lastId;
  for (;;) {
    const page = await clientFetch<LogPage>(`/runs/${encodeURIComponent(runId)}/logs`, {
      query: { after_id: afterId, limit: LOG_PAGE_SIZE },
      signal,
    });
    store.append(page.items);
    if (page.next_after_id === null || page.items.length === 0) return;
    afterId = page.next_after_id;
  }
}

export function RunLiveProvider({ initialRun, children }: { initialRun: RunDetail; children: ReactNode }) {
  const router = useRouter();
  const runId = initialRun.id;
  const [logs] = useState(() => new LogStore());
  const [overlay, setOverlay] = useState<LiveOverlay>(() => freshOverlay(initialRun));
  const [connection, setConnection] = useState<ConnectionState>("closed");
  const [logsLoading, setLogsLoading] = useState(true);
  const [logsError, setLogsError] = useState<ApiErrorInfo | null>(null);
  const listeners = useRef(new Set<Listener>());
  const lastRefresh = useRef(0);
  const refreshTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const refreshAbort = useRef<AbortController>(undefined);

  // A fresh server snapshot (after router.refresh) supersedes the live overlay.
  let current = overlay;
  if (overlay.server !== initialRun) {
    current = freshOverlay(initialRun);
    setOverlay(current);
  }
  const run = applyOverlay(current);
  const streaming = isRunActive(run.status) || run.stages.some((stage) => ACTIVE_STAGE_STATUSES.includes(stage.status));
  const status = useRef(run.status);
  useEffect(() => {
    status.current = run.status;
  }, [run.status]);

  const subscribe = useCallback((listener: Listener) => {
    listeners.current.add(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  // After a transition the run is fetched first, and only then are the server-rendered panels
  // (artifacts, evaluation, gate) refreshed: a router.refresh() that fails (offline) falls back
  // to a full page load, which would leave a blank page.
  const scheduleRefresh = useCallback(() => {
    clearTimeout(refreshTimer.current);
    const delay = Math.max(0, lastRefresh.current + REFRESH_THROTTLE_MS - Date.now());
    refreshTimer.current = setTimeout(() => {
      lastRefresh.current = Date.now();
      refreshAbort.current?.abort();
      const controller = new AbortController();
      refreshAbort.current = controller;
      fetchRunUntilReachable(runId, controller.signal)
        .then((fresh) => {
          setOverlay((previous) => freshOverlay(previous.server, fresh));
          router.refresh();
        })
        .catch(() => undefined); // aborted: a newer refresh or unmount took over
    }, delay);
  }, [router, runId]);

  useEffect(
    () => () => {
      clearTimeout(refreshTimer.current);
      refreshAbort.current?.abort();
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    loadLogHistory(runId, logs, controller.signal)
      .then(() => setLogsError(null))
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setLogsError(toErrorInfo(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLogsLoading(false);
      });
    return () => controller.abort();
  }, [runId, logs]);

  const liveEnabled = streaming && !logsLoading;
  useEffect(() => {
    if (!liveEnabled) return;
    const stream = new RunEventStream(
      runId,
      () => logs.lastId,
      setConnection,
      () => window.location.assign(loginPath(`${window.location.pathname}${window.location.search}`)),
    );
    const unsubscribe = stream.subscribe((event) => {
      if (event.type === "log") logs.append([event.data]);
      if (event.type === "stage") {
        setOverlay((previous) => ({ ...previous, stages: { ...previous.stages, [event.data.id]: event.data } }));
        if (TERMINAL_STAGE_STATUSES.includes(event.data.status)) scheduleRefresh();
      }
      if (event.type === "run") {
        setOverlay((previous) => ({ ...previous, summary: event.data }));
        if (event.data.status !== status.current) {
          status.current = event.data.status;
          scheduleRefresh();
        }
      }
      for (const listener of listeners.current) listener(event);
    });
    stream.start();
    return () => {
      unsubscribe();
      stream.stop();
    };
  }, [liveEnabled, runId, logs, scheduleRefresh]);

  return (
    <RunLiveContext.Provider value={{ run, connection, logs, logsLoading, logsError, subscribe }}>
      {children}
    </RunLiveContext.Provider>
  );
}
