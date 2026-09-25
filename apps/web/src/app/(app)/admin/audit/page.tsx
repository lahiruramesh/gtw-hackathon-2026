import { ScrollTextIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { AuditFilters } from "@/components/admin/audit-filters";
import { ApiErrorState } from "@/components/common/api-error-state";
import { DateTime } from "@/components/common/date-time";
import { EmptyState } from "@/components/common/empty-state";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toResult } from "@/lib/api/errors";
import { apiFetch } from "@/lib/api/server";
import type { AuditEvent, AuditEventPage } from "@/lib/api/types";
import { roleLabel } from "@/lib/permissions";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Audit log" };

const ENTITY_LINKS: Record<string, (id: string) => string> = {
  run: (id) => `/runs/${id}`,
  skill: (id) => `/skills/${id}`,
};

function param(value: string | string[] | undefined): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function Entity({ event }: { event: AuditEvent }) {
  const link = event.entity_id ? ENTITY_LINKS[event.entity_type]?.(event.entity_id) : undefined;
  const label = event.entity_id ? `${event.entity_type} ${event.entity_id.slice(0, 8)}` : event.entity_type;
  return link ? (
    <Link href={link} className="font-mono text-xs hover:underline">
      {label}
    </Link>
  ) : (
    <span className="font-mono text-xs" title={event.entity_id ?? undefined}>
      {label}
    </span>
  );
}

export default async function AuditPage({ searchParams }: PageProps<"/admin/audit">) {
  const viewer = await requireViewer();
  if (!can(viewer, "audit:read")) return <NoAccess what="read the audit log" />;

  const search = await searchParams;
  const query = {
    action: param(search.action),
    entity_type: param(search.entity_type),
    actor_id: param(search.actor_id),
    cursor: param(search.cursor),
    limit: 100,
  };
  const result = await toResult(apiFetch<AuditEventPage>("/audit-events", { query }));
  const nextParams = new URLSearchParams(
    Object.entries({
      ...query,
      cursor: result.ok ? (result.data.next_cursor ?? undefined) : undefined,
      limit: undefined,
    }).flatMap(([key, value]) => (value ? [[key, String(value)]] : [])),
  );

  const firstPageParams = new URLSearchParams(nextParams);
  firstPageParams.delete("cursor");

  return (
    <div className="space-y-5">
      <PageHeader title="Audit log" description="Every launch, decision, review, compute change and user change." />
      <AuditFilters />
      {!result.ok ? (
        <ApiErrorState error={result.error} subject="the audit log" />
      ) : result.data.items.length === 0 ? (
        <EmptyState icon={ScrollTextIcon} title="No audit events" description="Nothing matches these filters yet." />
      ) : (
        <Card className="gap-2 py-2">
          <CardContent className="overflow-x-auto px-2">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Actor</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Entity</TableHead>
                  <TableHead>Detail</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.data.items.map((event) => (
                  <TableRow key={event.id}>
                    <TableCell className="text-muted-foreground">
                      <DateTime value={event.ts} />
                    </TableCell>
                    <TableCell>
                      <div>{event.actor.name}</div>
                      <div className="text-xs text-muted-foreground">{roleLabel(event.actor.role)}</div>
                    </TableCell>
                    <TableCell className="font-mono text-xs">{event.action}</TableCell>
                    <TableCell>
                      <Entity event={event} />
                    </TableCell>
                    <TableCell className="max-w-md">
                      {Object.keys(event.detail).length === 0 ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        <code
                          className="line-clamp-2 font-mono text-xs break-all text-muted-foreground"
                          title={JSON.stringify(event.detail, null, 2)}
                        >
                          {JSON.stringify(event.detail)}
                        </code>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
          {(query.cursor || result.data.next_cursor) && (
            <nav className="flex justify-end gap-2 px-4 pb-2" aria-label="Pagination">
              {query.cursor && (
                <Button asChild variant="outline" size="sm">
                  <Link href={`/admin/audit?${firstPageParams.toString()}`}>Newest events</Link>
                </Button>
              )}
              {result.data.next_cursor && (
                <Button asChild variant="outline" size="sm">
                  <Link href={`/admin/audit?${nextParams.toString()}`}>Older events</Link>
                </Button>
              )}
            </nav>
          )}
        </Card>
      )}
    </div>
  );
}
