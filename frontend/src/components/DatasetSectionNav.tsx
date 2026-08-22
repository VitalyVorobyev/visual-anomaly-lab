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
 */

import { NavLink } from "react-router";

import { cn, focusRing } from "@vitavision/lab-ui";

export function DatasetSectionNav({ datasetId }: { datasetId: number }) {
  const items = [
    { label: "Browse", to: `/datasets/${datasetId}`, end: true },
    { label: "Annotate", to: `/datasets/${datasetId}/annotate`, end: false },
    { label: "Prepare", to: `/datasets/${datasetId}/prepare`, end: false },
    { label: "Splits", to: `/datasets/${datasetId}/splits`, end: false },
    { label: "Experiments", to: `/datasets/${datasetId}/experiments`, end: false },
  ];

  return (
    <nav
      aria-label="Dataset workspace"
      className="-mb-px flex min-w-0 items-center gap-1"
    >
      {items.map((item) => (
        <NavLink
          key={item.label}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            cn(
              // The border is always there and always 2px, so the label never shifts by the
              // one pixel that a border appearing on hover would cost.
              "border-b-2 px-3 py-2 text-xs font-medium transition-colors",
              isActive
                ? "border-signal text-fg"
                : "border-transparent text-fg-muted hover:border-line-strong hover:text-fg",
              focusRing,
            )
          }
        >
          {item.label}
        </NavLink>
      ))}
    </nav>
  );
}
