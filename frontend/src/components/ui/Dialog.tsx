/**
 * Confirmation for the things that cannot be undone.
 *
 * `DatasetsRoute` deleted a dataset -- its samples, its splits, every experiment that
 * referenced it -- on a single click of a red button, with nothing between the click and
 * the request. That was not a styling gap.
 */

import * as RadixDialog from "@radix-ui/react-dialog";
import type { ReactNode } from "react";

import { Button } from "./Button";
import { cn } from "./cn";

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
  loading = false,
  disabled = false,
  destructive = false,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  /** Say what will actually happen, naming the thing by the name the reader gave it. */
  description: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  loading?: boolean;
  disabled?: boolean;
  destructive?: boolean;
}) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-[1px]" />
        <RadixDialog.Content
          className={cn(
            "fixed top-1/2 left-1/2 z-50 w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2",
            "rounded-panel border border-line bg-overlay p-5 shadow-xl shadow-black/40",
            "focus:outline-none",
          )}
        >
          <RadixDialog.Title className="text-sm font-semibold tracking-tight text-fg">
            {title}
          </RadixDialog.Title>
          <RadixDialog.Description className="mt-2 text-sm leading-relaxed text-fg-muted">
            {description}
          </RadixDialog.Description>
          <div className="mt-5 flex justify-end gap-2">
            <RadixDialog.Close asChild>
              <Button variant="ghost">Cancel</Button>
            </RadixDialog.Close>
            <Button
              variant={destructive ? "danger" : "primary"}
              loading={loading}
              disabled={disabled}
              onClick={onConfirm}
            >
              {confirmLabel}
            </Button>
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}
