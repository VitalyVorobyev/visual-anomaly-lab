import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn, focusRing } from "./cn";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-signal text-signal-fg hover:bg-signal-strong",
  secondary: "bg-surface text-fg ring-1 ring-inset ring-line-strong hover:bg-raised",
  ghost: "text-fg-muted hover:bg-raised hover:text-fg",
  // Soft rather than a solid red block. The verdict red is also the colour of a defect
  // badge, and a filled button of it reads as an alarm on a screen that is mostly grey.
  // Genuinely destructive actions are behind a confirmation dialog regardless.
  danger: "bg-defect/10 text-defect ring-1 ring-inset ring-defect/30 hover:bg-defect/20",
};

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: "h-7 gap-1.5 px-2.5 text-xs",
  md: "h-8 gap-2 px-3 text-sm",
};

export function Button({
  variant = "secondary",
  size = "md",
  type = "button",
  loading = false,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Swaps the icon for a spinner and blocks the click, without the label changing. */
  loading?: boolean;
  icon?: ReactNode;
}) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-control font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-45",
        SIZE_CLASSES[size],
        VARIANT_CLASSES[variant],
        focusRing,
        className,
      )}
    >
      {loading ? (
        <Loader2 className="size-3.5 shrink-0 animate-spin" aria-hidden />
      ) : (
        icon && (
          <span className="shrink-0 [&>svg]:size-3.5" aria-hidden>
            {icon}
          </span>
        )
      )}
      {children}
    </button>
  );
}
