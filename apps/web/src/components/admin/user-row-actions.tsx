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
import { ROLES } from "@/lib/permissions";

type DialogKind = "ban" | "password" | "revoke" | null;

export function UserRowActions({ user, isSelf }: { user: ManagedUser; isSelf: boolean }) {
  const { pending, execute } = useUserAction();
  const [dialog, setDialog] = useState<DialogKind>(null);
  const [text, setText] = useState("");

  function openDialog(kind: DialogKind) {
    setText(kind === "password" ? generatePassword() : "");
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
    if (ok) setDialog(null);
  }

  const passwordTooShort = dialog === "password" && text.length < MIN_PASSWORD_LENGTH;

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
                onValueChange={(role) =>
                  execute(
                    () => setUserRole(user.id, role),
                    `Role updated for ${user.email}`,
                    "Couldn't change the role",
                  )
                }
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
            <DialogTitle>
              {dialog === "ban" ? "Ban user" : dialog === "password" ? "Set a new password" : "Sign out everywhere"}
            </DialogTitle>
            <DialogDescription>
              {dialog === "ban"
                ? `${user.email} is signed out and can't sign in until unbanned.`
                : dialog === "password"
                  ? `Share the new password with ${user.email} directly. It is not shown again.`
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
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              Cancel
            </Button>
            <Button
              variant={dialog === "ban" ? "destructive" : "default"}
              disabled={pending || passwordTooShort}
              onClick={confirm}
            >
              {dialog === "ban" ? "Ban user" : dialog === "password" ? "Set password" : "Sign out"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
