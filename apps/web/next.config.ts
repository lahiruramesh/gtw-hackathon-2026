import path from "node:path";

import type { NextConfig } from "next";

// The app imports shared/permissions.json from the repository root, so both bundling and
// standalone output tracing start there (server.js ends up at .next/standalone/apps/web/).
const repoRoot = path.join(__dirname, "../..");

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: repoRoot,
  turbopack: { root: repoRoot },
  poweredByHeader: false,
};

export default nextConfig;
