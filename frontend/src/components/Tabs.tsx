/**
 * The tab strip, extracted from the three places that had grown their own.
 *
 * `ChannelTabs`, the outcome-filter chips on the results screen and the workbench tabs all
 * drew the same control with the same active style and slightly different markup. One
 * component means the keyboard and ARIA behaviour is right in all three rather than in
 * whichever was written last.
 */

export interface TabItem<Id extends string> {
  id: Id;
  label: string;
  /** Shown after the label, for the filter chips that carry a row count. */
  count?: number;
  disabled?: boolean;
}

export function Tabs<Id extends string>({
  items,
  active,
  onSelect,
  label,
}: {
  items: TabItem<Id>[];
  active: Id;
  onSelect: (id: Id) => void;
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex flex-wrap gap-1">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={item.id === active}
          disabled={item.disabled}
          onClick={() => onSelect(item.id)}
          className={`rounded px-2.5 py-1 text-xs transition-colors disabled:opacity-40 ${
            item.id === active
              ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
          }`}
        >
          {item.label}
          {item.count !== undefined && ` (${item.count})`}
        </button>
      ))}
    </div>
  );
}
