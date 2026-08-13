/**
 * Confirmation for the things that cannot be undone, and a shell for the things that need
 * a form.
 *
 * `DatasetsRoute` deleted a dataset -- its samples, its splits, every experiment that
 * referenced it -- on a single click of a red button, with nothing between the click and
 * the request. That was not a styling gap.
 *
 * `Dialog` is that same modal without the verdict: a title, a body the caller fills, and a
 * footer it chooses. `ConfirmDialog` is now one arrangement of it rather than a second
 * copy of the overlay, the centring and the width cap.
 */

import * as RadixDialog from "@radix-ui/react-dialog";
import type { ReactNode } from "react";

import { Button } from "./Button";
import { cn } from "./cn";

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  footer,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  /** Optional: what the reader needs before they can answer, not decoration. */
  description?: ReactNode;
  footer?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-[1px]" />
        <RadixDialog.Content
          className={cn(
            "fixed top-1/2 left-1/2 z-50 w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2",
            // A ceiling and a column, so a dialog that lists things keeps its footer on
            // screen and scrolls its body instead of growing past the viewport. Every
            // short dialog is far below the cap and is unaffected. This is a scroller
            // inside an overlay, not a second page scroller.
            "flex max-h-[calc(100vh-4rem)] flex-col",
            "rounded-panel border border-line bg-overlay p-5 shadow-xl shadow-black/40",
            "focus:outline-none",
          )}
        >
          <RadixDialog.Title className="text-sm font-semibold tracking-tight text-fg">
            {title}
          </RadixDialog.Title>
          {/* Radix warns when a dialog has no description; render an empty one rather than
              inventing prose for a form whose fields already carry their own labels. */}
          <RadixDialog.Description
            className={cn(
              "mt-2 text-sm leading-relaxed text-fg-muted",
              description === undefined && "sr-only",
            )}
          >
            {description ?? title}
          </RadixDialog.Description>
          <div className="min-h-0 overflow-y-auto">{children}</div>
          {footer && <div className="mt-5 flex shrink-0 justify-end gap-2">{footer}</div>}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}

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
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={description}
      footer={
        <>
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
        </>
      }
    />
  );
}

/** The dialog's own dismiss, for a footer the caller builds. */
export const DialogClose = RadixDialog.Close;
