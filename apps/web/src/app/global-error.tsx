"use client";

import "./globals.css";

export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <html lang="en">
      <body className="flex min-h-svh flex-col items-center justify-center gap-3 p-6 text-center font-sans">
        <h1 className="text-xl font-semibold">SKF Skill Studio is unavailable</h1>
        <p className="text-sm text-muted-foreground">
          Reference: <code className="font-mono text-xs">{error.digest ?? "unknown"}</code>
        </p>
        <button className="rounded-md border px-3 py-1.5 text-sm" onClick={reset}>
          Try again
        </button>
      </body>
    </html>
  );
}
