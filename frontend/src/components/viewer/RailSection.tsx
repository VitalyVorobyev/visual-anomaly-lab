/**
 * One titled block of a viewer's side rail: the sample viewer's and the reference studio's.
 * Shared so two rails beside the same kind of stage read the same way.
 */

import type { ReactNode } from "react";

export function RailSection({
  title,
  hint,
  children,
}: {
  title: string | null;
  hint?: string | undefined;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2.5 border-b border-line p-4 last:border-b-0">
      {title !== null && (
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-xs font-semibold tracking-tight text-fg">{title}</h2>
          {hint && <span className="font-mono text-[11px] text-fg-subtle">{hint}</span>}
        </div>
      )}
      {children}
    </section>
  );
}
