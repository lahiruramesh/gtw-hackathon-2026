import { describe, expect, it } from "vitest";

import { filterResponseHeaders, isForeignRequest, readLimitedBody, resolveUpstreamPath } from "@/lib/api/proxy-path";

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

describe("isForeignRequest", () => {
  const app = "http://localhost:3100";
  const headers = (values: Record<string, string>) => new Headers(values);

  it("lets reads and same-origin writes through", () => {
    expect(
      isForeignRequest("GET", headers({ origin: "https://evil.example", "sec-fetch-site": "cross-site" }), app),
    ).toBe(false);
    expect(isForeignRequest("POST", headers({ origin: app, "sec-fetch-site": "same-origin" }), app)).toBe(false);
    expect(isForeignRequest("POST", headers({}), app)).toBe(false); // non-browser clients send neither
  });

  it.each([
    [{ origin: "https://evil.example" }],
    [{ origin: "https://studio.sibling.example", "sec-fetch-site": "same-site" }],
    [{ "sec-fetch-site": "cross-site" }],
  ])("rejects writes from elsewhere: %j", (values) => {
    expect(isForeignRequest("POST", headers(values), app)).toBe(true);
  });
});

describe("readLimitedBody", () => {
  const stream = (...chunks: string[]) => new Blob(chunks).stream();

  it("returns the body up to the limit", async () => {
    const body = await readLimitedBody(stream("ab", "cd"), 4);
    expect(new TextDecoder().decode(body ?? undefined)).toBe("abcd");
    expect(await readLimitedBody(null, 4)).toEqual(new Uint8Array());
  });

  it("refuses a larger body", async () => {
    expect(await readLimitedBody(stream("ab", "cde"), 4)).toBeNull();
  });
});
