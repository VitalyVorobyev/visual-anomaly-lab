/**
 * Local navigation inside one dataset workspace.
 *
 * Rendered once, by `DatasetLayout`, and nowhere else -- five copies in five routes is what
 * made it move between tabs.
 *
 * An underline rather than the pill used by `components/Tabs.tsx`. These are real
 * navigations to real URLs, and the pill is what marks an in-page state switch; giving the
 * two mechanisms the same shape would say they were the same thing. Anchored to the band's
 * bottom border by `-mb-px`, so the active mark reads as the section owning the surface
 * below it. For the same reason there is no `role="tablist"`: ARIA tabs promise an
 * in-document panel switch and a roving tabindex, and `NavLink` implements neither.
 * `<nav aria-label>` plus the `aria-current="page"` NavLink already sets is the honest
 * description.
 *
 * **Grouped by the stage of the work, in the order it is done** (ADR-0040): the *data* — see
 * it, prepare its input — then the *truth* a run is measured against, then the *runs* —
 * splits and experiments. One row still, so the band keeps its height: the stage names are
 * quiet labels between the links, not a second tier, and they fold away below `md` where
 * the row has no room for them. Each stage is a `role="group"` named for itself.
 */

import { NavLink } from "react-router";

import { cn, focusRing } from "@vitavision/lab-ui";

const LINK = ({ isActive }: { isActive: boolean }) =>
  cn(
    // The border is always there and always 2px, so the label never shifts by the one pixel
    // that a border appearing on hover would cost.
    "border-b-2 px-3 py-2 text-xs font-medium transition-colors",
    isActive
      ? "border-signal text-fg"
      : "border-transparent text-fg-muted hover:border-line-strong hover:text-fg",
    focusRing,
  );

export function DatasetSectionNav({ datasetId }: { datasetId: number }) {
  const base = `/datasets/${datasetId}`;
  const stages = [
    {
      stage: "Data",
      items: [
        { label: "Browse", to: base, end: true },
        { label: "Prepare", to: `${base}/prepare`, end: false },
      ],
    },
    { stage: "Truth", items: [{ label: "Annotate", to: `${base}/annotate`, end: false }] },
    {
      stage: "Runs",
      items: [
        { label: "Splits", to: `${base}/splits`, end: false },
        { label: "Experiments", to: `${base}/experiments`, end: false },
      ],
    },
  ];

  return (
    <nav aria-label="Dataset workspace" className="-mb-px flex min-w-0 items-center gap-1">
      {stages.map((group, index) => (
        <div key={group.stage} role="group" aria-label={group.stage} className="flex items-center">
          {index > 0 && <span aria-hidden className="mx-1.5 h-4 w-px bg-line" />}
          <span
            aria-hidden
            className="hidden pr-1 pl-1.5 text-[10px] font-semibold tracking-wider text-fg-subtle uppercase md:inline"
          >
            {group.stage}
          </span>
          {group.items.map((item) => (
            <NavLink key={item.label} to={item.to} end={item.end} className={LINK}>
              {item.label}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}
