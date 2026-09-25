"use client";

import { CheckIcon, XIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Decision, DecisionRequest } from "@/lib/api/types";

interface DecisionButtonsProps {
  /** What is being decided, e.g. "launch of g1-stairs-v12". */
  subject: string;
  approveLabel?: string;
  rejectLabel?: string;
  /** When set, both buttons are disabled and this explains why. */
  disabledReason?: string;
  pending?: boolean;
  size?: "sm" | "default";
  /** Resolves true on success, which closes the dialog. */
  onDecide: (input: DecisionRequest) => Promise<boolean>;
}

function DecisionDialog({
  decision,
  subject,
  label,
  pending,
  disabled,
  size,
  onDecide,
}: {
  decision: Decision;
  subject: string;
  label: string;
  pending: boolean;
  disabled: boolean;
  size: "sm" | "default";
  onDecide: DecisionButtonsProps["onDecide"];
}) {
  const [open, setOpen] = useState(false);
  const [comment, setComment] = useState("");
  const rejecting = decision === "reject";
  const commentMissing = rejecting && comment.trim() === "";
  const fieldId = `decision-comment-${decision}`;

  async function submit() {
    const decided = await onDecide({ decision, comment: comment.trim() || undefined });
    if (decided) {
      setOpen(false);
      setComment("");
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant={rejecting ? "outline" : "default"} size={size} disabled={disabled}>
          {rejecting ? <XIcon /> : <CheckIcon />}
          {label}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{label}</DialogTitle>
          <DialogDescription>
            {rejecting ? "Reject" : "Approve"} the {subject}. The decision is recorded in the audit log.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor={fieldId}>Comment{rejecting ? "" : " (optional)"}</Label>
          <Textarea
            id={fieldId}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder={rejecting ? "Explain what needs to change" : "Anything the team should know"}
            rows={4}
            aria-invalid={commentMissing || undefined}
          />
          {commentMissing && <p className="text-xs text-muted-foreground">A comment is required when rejecting.</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button variant={rejecting ? "destructive" : "default"} disabled={pending || commentMissing} onClick={submit}>
            {label}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function DecisionButtons({
  subject,
  approveLabel = "Approve",
  rejectLabel = "Reject",
  disabledReason,
  pending = false,
  size = "sm",
  onDecide,
}: DecisionButtonsProps) {
  const buttons = (
    <div className="flex gap-2">
      <DecisionDialog
        decision="approve"
        subject={subject}
        label={approveLabel}
        pending={pending}
        disabled={Boolean(disabledReason) || pending}
        size={size}
        onDecide={onDecide}
      />
      <DecisionDialog
        decision="reject"
        subject={subject}
        label={rejectLabel}
        pending={pending}
        disabled={Boolean(disabledReason) || pending}
        size={size}
        onDecide={onDecide}
      />
    </div>
  );
  if (!disabledReason) return buttons;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className="inline-flex">
          {buttons}
        </span>
      </TooltipTrigger>
      <TooltipContent>{disabledReason}</TooltipContent>
    </Tooltip>
  );
}
