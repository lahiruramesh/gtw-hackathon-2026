"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDownToLineIcon, DownloadIcon, SearchIcon } from "lucide-react";
import { useDeferredValue, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { ApiErrorState } from "@/components/common/api-error-state";
import { useRunLive } from "@/components/runs/run-live-provider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { proxyPath } from "@/lib/api/client";
import type { LogLevel, LogLine } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";
import { includesIgnoreCase, splitHighlight } from "@/lib/highlight";
import { MAX_LOG_LINES } from "@/lib/log-store";
import { cn } from "cn";

const ALL_STAGES = "all";
const LEVELS: LogLevel[] = ["info", "warn", "error"];
const LEVEL_CLASSES: Record<LogLevel, string> = {
  info: "text-foreground/90",
  warn: "text-status-warning",
  error: "text-status-danger",
};
const ROW_HEIGHT = 20;
const FOLLOW_THRESHOLD_PX = 48;

function timeOf(ts: string): string {
  return ts.length >= 19 ? ts.slice(11, 19) : ts;
}

function LogRow({ line, stageKey, query }: { line: LogLine; stageKey: string; query: string }) {
  return (
    <div className="flex gap-3 px-3 font-mono text-xs leading-5 hover:bg-muted/50">
      <span className="tabular shrink-0 text-muted-foreground select-none" title={line.ts}>
        {timeOf(line.ts)}
      </span>
      <span className="w-20 shrink-0 truncate text-muted-foreground select-none" title={stageKey}>
        {stageKey}
      </span>
      <span className={cn("min-w-0 flex-1 break-all whitespace-pre-wrap", LEVEL_CLASSES[line.level])}>
        {splitHighlight(line.text, query).map((part, index) =>
          part.match ? (
            <mark key={index} className="rounded-sm bg-status-warning/30 text-foreground">
              {part.text}
            </mark>
          ) : (
            part.text
          ),
        )}
      </span>
    </div>
  );
}

export function LogViewer() {
  const { run, logs, logsLoading, logsError } = useRunLive();
  useSyncExternalStore(logs.subscribe, logs.version, logs.version);
  const lines = logs.lines;

  const [stageId, setStageId] = useState(ALL_STAGES);
  const [levels, setLevels] = useState<LogLevel[]>(LEVELS);
  const [search, setSearch] = useState("");
  const [onlyMatches, setOnlyMatches] = useState(false);
  const [follow, setFollow] = useState(true);
  const query = useDeferredValue(search.trim());
  const scrollRef = useRef<HTMLDivElement>(null);

  const stageKeys = useMemo(() => new Map(run.stages.map((stage) => [stage.id, stage.key])), [run.stages]);
  const noiseDropped = run.stages.reduce((sum, stage) => sum + stage.noise_dropped, 0);

  const visible = useMemo(
    () =>
      lines.filter(
        (line) =>
          (stageId === ALL_STAGES || line.stage_id === stageId) &&
          levels.includes(line.level) &&
          (!onlyMatches || !query || includesIgnoreCase(line.text, query)),
      ),
    // `lines` is replaced on every append, so it is the change signal.
    [lines, stageId, levels, onlyMatches, query],
  );
  const matchCount = useMemo(
    () => (query ? visible.filter((line) => includesIgnoreCase(line.text, query)).length : 0),
    [visible, query],
  );

  // The React Compiler is off; the virtualizer's unmemoizable return value is fine here.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: visible.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 30,
  });

  useEffect(() => {
    if (follow && visible.length > 0) virtualizer.scrollToIndex(visible.length - 1, { align: "end" });
  }, [follow, visible.length, virtualizer]);

  function onScroll() {
    const element = scrollRef.current;
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < FOLLOW_THRESHOLD_PX;
    if (!atBottom && follow) setFollow(false);
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Select value={stageId} onValueChange={setStageId}>
          <SelectTrigger size="sm" className="w-40" aria-label="Stage">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL_STAGES}>All stages</SelectItem>
            {run.stages.map((stage) => (
              <SelectItem key={stage.id} value={stage.id}>
                {stage.title}
                {stage.attempt > 1 && ` (attempt ${stage.attempt})`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <ToggleGroup
          type="multiple"
          variant="outline"
          size="sm"
          value={levels}
          onValueChange={(value) => setLevels(value as LogLevel[])}
          aria-label="Log levels"
        >
          {LEVELS.map((level) => (
            <ToggleGroupItem key={level} value={level} className="px-2.5 capitalize">
              {level}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <div className="relative w-full sm:w-56">
          <Label htmlFor="log-search" className="sr-only">
            Search logs
          </Label>
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            id="log-search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search"
            className="h-8 pl-8"
          />
        </div>
        {query && (
          <div className="flex items-center gap-2">
            <Switch id="only-matches" checked={onlyMatches} onCheckedChange={setOnlyMatches} />
            <Label htmlFor="only-matches" className="text-xs font-normal">
              Only matches ({formatNumber(matchCount)})
            </Label>
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Switch id="follow" checked={follow} onCheckedChange={setFollow} />
          <Label htmlFor="follow" className="text-xs font-normal">
            Follow
          </Label>
          <Button asChild variant="outline" size="sm">
            <a href={proxyPath(`/runs/${encodeURIComponent(run.id)}/logs.txt`)} download={`${run.name}.log`}>
              <DownloadIcon /> Download
            </a>
          </Button>
        </div>
      </div>

      {logsError ? (
        <ApiErrorState error={logsError} subject="logs" />
      ) : (
        <div className="relative rounded-md border">
          <div
            ref={scrollRef}
            onScroll={onScroll}
            role="log"
            aria-label="Run logs"
            aria-busy={logsLoading}
            tabIndex={0}
            className="h-[min(60vh,560px)] overflow-auto bg-muted/30 py-1 focus-visible:outline-2"
          >
            {logsLoading && lines.length === 0 ? (
              <div className="space-y-2 p-3">
                {Array.from({ length: 8 }, (_, index) => (
                  <Skeleton key={index} className="h-4" style={{ width: `${60 + ((index * 13) % 35)}%` }} />
                ))}
              </div>
            ) : visible.length === 0 ? (
              <p className="p-4 text-sm text-muted-foreground">
                {lines.length === 0
                  ? "No log lines yet. They appear here as soon as a stage starts."
                  : "No lines match the filters."}
              </p>
            ) : (
              <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
                {virtualizer.getVirtualItems().map((item) => {
                  const line = visible[item.index]!;
                  return (
                    <div
                      key={line.id}
                      data-index={item.index}
                      ref={virtualizer.measureElement}
                      className="absolute top-0 left-0 w-full"
                      style={{ transform: `translateY(${item.start}px)` }}
                    >
                      <LogRow line={line} stageKey={stageKeys.get(line.stage_id) ?? ""} query={query} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          {!follow && visible.length > 0 && (
            <Button
              size="sm"
              variant="secondary"
              className="absolute right-4 bottom-3 shadow"
              onClick={() => setFollow(true)}
            >
              <ArrowDownToLineIcon /> Jump to latest
            </Button>
          )}
        </div>
      )}

      <p className="tabular text-xs text-muted-foreground">
        {formatNumber(lines.length)} lines
        {logs.evictedCount > 0 &&
          ` (older lines trimmed to the latest ${formatNumber(MAX_LOG_LINES)}; download for the full log)`}
        {noiseDropped > 0 && ` · ${formatNumber(noiseDropped)} noise lines dropped at the source`}
      </p>
    </div>
  );
}
