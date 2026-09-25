"use client";

import { BanIcon, KeyRoundIcon, LogOutIcon, MoreHorizontalIcon, ShieldIcon, UserCheckIcon } from "lucide-react";
import { useState } from "react";

import { banUser, revokeUserSessions, setUserPassword, setUserRole, unbanUser } from "@/app/(app)/admin/users/actions";
import { generatePassword } from "@/components/admin/password-generator";
import { useUserAction } from "@/components/admin/use-user-action";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ManagedUser } from "@/lib/admin-users";
import { MIN_PASSWORD_LENGTH } from "@/lib/password-policy";
import { permissionChanges, ROLES, roleLabel } from "@/lib/permissions";

type DialogKind = "ban" | "password" | "revoke" | "role" | null;

const DIALOG_COPY = {
  ban: { title: "Ban user", confirm: "Ban user" },
  password: { title: "Set a new password", confirm: "Set password" },
  revoke: { title: "Sign out everywhere", confirm: "Sign out" },
  role: { title: "Change role", confirm: "Change role" },
} as const;

function PermissionList({ label, permissions }: { label: string; permissions: string[] }) {
  if (permissions.length === 0) return null;
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <ul className="flex flex-wrap gap-1">
        {permissions.map((permission) => (
          <li key={permission} className="rounded border bg-muted px-1.5 font-mono text-xs">
            {permission}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function UserRowActions({ user, isSelf }: { user: ManagedUser; isSelf: boolean }) {
  const { pending, execute } = useUserAction();
  const [dialog, setDialog] = useState<DialogKind>(null);
  const [text, setText] = useState("");

  function openDialog(kind: DialogKind, value = "") {
    setText(kind === "password" ? generatePassword() : value);
    setDialog(kind);
  }

  async function confirm() {
    let ok = false;
    if (dialog === "ban")
      ok = await execute(() => banUser(user.id, text), `${user.email} is banned`, "Couldn't ban the user");
    if (dialog === "password")
      ok = await execute(
        () => setUserPassword(user.id, text),
        `Password set for ${user.email}`,
        "Couldn't set the password",
      );
    if (dialog === "revoke")
      ok = await execute(
        () => revokeUserSessions(user.id),
        `Signed ${user.email} out everywhere`,
        "Couldn't revoke sessions",
      );
    if (dialog === "role")
      ok = await execute(
        () => setUserRole(user.id, text),
        `${user.email} is now ${roleLabel(text)}`,
        "Couldn't change the role",
      );
    if (ok) setDialog(null);
  }

  const passwordTooShort = dialog === "password" && text.length < MIN_PASSWORD_LENGTH;
  const roleChange = dialog === "role" ? permissionChanges(user.role, text) : null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" aria-label={`Actions for ${user.email}`} disabled={pending}>
            <MoreHorizontalIcon />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuLabel className="truncate text-xs font-normal">{user.email}</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger disabled={isSelf}>
              <ShieldIcon /> Change role
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuRadioGroup
                value={user.role}
                onValueChange={(role) => role !== user.role && openDialog("role", role)}
              >
                {ROLES.map((role) => (
                  <DropdownMenuRadioItem key={role.id} value={role.id}>
                    {role.label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuItem onSelect={() => openDialog("password")}>
            <KeyRoundIcon /> Set password
          </DropdownMenuItem>
          <DropdownMenuItem disabled={isSelf} onSelect={() => openDialog("revoke")}>
            <LogOutIcon /> Sign out everywhere
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {user.banned ? (
            <DropdownMenuItem
              onSelect={() =>
                execute(() => unbanUser(user.id), `${user.email} can sign in again`, "Couldn't unban the user")
              }
            >
              <UserCheckIcon /> Unban
            </DropdownMenuItem>
          ) : (
            <DropdownMenuItem variant="destructive" disabled={isSelf} onSelect={() => openDialog("ban")}>
              <BanIcon /> Ban
            </DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={dialog !== null} onOpenChange={(open) => !open && setDialog(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{dialog && DIALOG_COPY[dialog].title}</DialogTitle>
            <DialogDescription>
              {dialog === "ban"
                ? `${user.email} is signed out and can't sign in until unbanned.`
                : dialog === "password"
                  ? `Share the new password with ${user.email} directly. It is not shown again.`
                  : dialog === "role"
                    ? `${user.email}: ${roleLabel(user.role)} → ${roleLabel(text)}.`
                    : `Ends every active session of ${user.email}. They can sign in again.`}
            </DialogDescription>
          </DialogHeader>
          {dialog === "ban" && (
            <div className="space-y-2">
              <Label htmlFor="ban-reason">Reason (optional)</Label>
              <Input id="ban-reason" value={text} onChange={(event) => setText(event.target.value)} />
            </div>
          )}
          {dialog === "password" && (
            <div className="space-y-2">
              <Label htmlFor="new-password">New password</Label>
              <Input
                id="new-password"
                value={text}
                onChange={(event) => setText(event.target.value)}
                autoComplete="new-password"
                className="font-mono"
                aria-invalid={passwordTooShort || undefined}
              />
              <p className="text-xs text-muted-foreground">At least {MIN_PASSWORD_LENGTH} characters.</p>
            </div>
          )}
          {roleChange && (
            <div className="space-y-3">
              <PermissionList label="Gains" permissions={roleChange.gained} />
              <PermissionList label="Loses" permissions={roleChange.lost} />
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              Cancel
            </Button>
            <Button
              variant={dialog === "ban" ? "destructive" : "default"}
              disabled={pending || passwordTooShort}
              onClick={confirm}
            >
              {dialog && DIALOG_COPY[dialog].confirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
