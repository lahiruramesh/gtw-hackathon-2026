import { describe, expect, it } from "vitest";

import { filterResponseHeaders, resolveUpstreamPath } from "@/lib/api/proxy-path";

describe("resolveUpstreamPath", () => {
  it("joins ordinary segments", () => {
    expect(resolveUpstreamPath(["runs", "3f2a", "events"], "/api/backend/runs/3f2a/events")).toBe("runs/3f2a/events");
    expect(resolveUpstreamPath(["runs", "abc", "logs.txt"], "/api/backend/runs/abc/logs.txt")).toBe(
      "runs/abc/logs.txt",
    );
  });

  it.each([
    [["..", "admin"], "/api/backend/../admin"],
    [["."], "/api/backend/."],
    [["runs", "a/b"], "/api/backend/runs/a%2Fb"],
    [["runs", "a\\b"], "/api/backend/runs/a%5Cb"],
    [["http:", "evil.com"], "/api/backend/http:/evil.com"],
    [["runs", "%2e%2e"], "/api/backend/runs/%252e%252e"],
    [[], "/api/backend"],
  ])("rejects %j", (segments, raw) => {
    expect(resolveUpstreamPath(segments, raw)).toBeNull();
  });

  it("rejects encoded separators in the raw path even when segments look clean", () => {
    expect(resolveUpstreamPath(["runs", "x"], "/api/backend/runs%2Fx")).toBeNull();
    expect(resolveUpstreamPath(["runs", "x"], "/api/backend/%2e%2e/x")).toBeNull();
  });

  it("limits the number of segments", () => {
    expect(resolveUpstreamPath(Array(9).fill("a"), "/api/backend/a")).toBeNull();
  });
});

describe("filterResponseHeaders", () => {
  it("drops hop-by-hop headers, cookies and stale encoding", () => {
    const upstream = new Headers({
      "content-type": "text/event-stream",
      connection: "keep-alive",
      "transfer-encoding": "chunked",
      "set-cookie": "a=b",
      "content-encoding": "gzip",
      "content-length": "10",
      "x-request-id": "r1",
      server: "uvicorn",
    });
    const filtered = filterResponseHeaders(upstream);
    expect([...filtered.keys()].sort()).toEqual(["content-type", "x-request-id"]);
  });
});
