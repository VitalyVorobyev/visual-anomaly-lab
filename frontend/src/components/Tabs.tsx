/**
 * The tab strip, extracted from the three places that had grown their own.
 *
 * `ChannelTabs`, the outcome-filter chips on the results screen and the workbench tabs all
 * drew the same control with the same active style and slightly different markup. One
 * component means the keyboard and ARIA behaviour is right in all three rather than in
 * whichever was written last.
 *
 * The count is set in mono beside the label rather than in parentheses inside it: it is a
 * quantity, and on the experiment form it is the number of options a reader has changed
 * behind that tab -- the one thing that makes hiding them acceptable.
 */

import { cn, focusRing } from "./ui/cn";

export interface TabItem<Id extends string> {
  id: Id;
  label: string;
  /** Shown after the label, for the chips that carry a row or override count. */
  count?: number;
  disabled?: boolean;
  /** Why this tab has nothing to show, for a reader wondering what they missed. */
  title?: string;
}

export function Tabs<Id extends string>({
  items,
  active,
  onSelect,
  label,
  className,
}: {
  items: TabItem<Id>[];
  active: Id;
  onSelect: (id: Id) => void;
  label: string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={label}
      className={cn(
        "inline-flex max-w-full flex-wrap items-center gap-0.5 rounded-control border border-line bg-raised p-0.5",
        className,
      )}
    >
      {items.map((item) => {
        const selected = item.id === active;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={selected}
            disabled={item.disabled}
            title={item.title}
            onClick={() => onSelect(item.id)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-[0.3rem] px-2.5 py-1 text-xs font-medium transition-colors",
              "disabled:cursor-not-allowed disabled:opacity-40",
              selected
                ? "bg-surface text-fg shadow-sm"
                : "text-fg-muted hover:text-fg",
              focusRing,
            )}
          >
            {item.label}
            {item.count !== undefined && item.count > 0 && (
              <span
                className={cn(
                  "rounded-full px-1.5 font-mono text-[10px] tabular-nums",
                  selected ? "bg-signal/15 text-signal" : "bg-line text-fg-muted",
                )}
              >
                {item.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
