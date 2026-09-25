import type { Metadata } from "next";

import { CreateUserDialog } from "@/components/admin/create-user-dialog";
import { UserRowActions } from "@/components/admin/user-row-actions";
import { ApiErrorState } from "@/components/common/api-error-state";
import { DateTime } from "@/components/common/date-time";
import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listManagedUsers } from "@/lib/admin-users";
import { roleLabel } from "@/lib/permissions";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Users" };

export default async function UsersPage() {
  const viewer = await requireViewer();
  if (!can(viewer, "user:manage")) return <NoAccess what="manage users" />;

  const result = await listManagedUsers();
  return (
    <div className="space-y-5">
      <PageHeader
        title="Users"
        description="Accounts and roles. There is no self sign-up: every account is created here."
        actions={<CreateUserDialog />}
      />
      {!result.ok ? (
        <ApiErrorState error={result.error} subject="users" />
      ) : (
        <Card className="py-2">
          <CardContent className="overflow-x-auto px-2">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="w-12">
                    <span className="sr-only">Actions</span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.data.map((user) => (
                  <TableRow key={user.id} id={`user-${user.id}`} className="target:bg-muted">
                    <TableCell>
                      <div className="font-medium">
                        {user.name}
                        {user.id === viewer.id && <span className="font-normal text-muted-foreground"> (you)</span>}
                      </div>
                      <div className="text-xs text-muted-foreground">{user.email}</div>
                    </TableCell>
                    <TableCell>{roleLabel(user.role)}</TableCell>
                    <TableCell>
                      {user.banned ? (
                        <Badge variant="outline" className="text-status-danger" title={user.banReason ?? undefined}>
                          Banned
                        </Badge>
                      ) : (
                        <span className="text-sm text-muted-foreground">Active</span>
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      <DateTime value={user.createdAt} />
                    </TableCell>
                    <TableCell>
                      <UserRowActions user={user} isSelf={user.id === viewer.id} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
