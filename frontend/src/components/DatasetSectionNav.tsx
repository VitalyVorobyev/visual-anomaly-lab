/** Local navigation inside one dataset workspace. */

import { NavLink } from "react-router";

import { cn, focusRing } from "./ui/cn";

export function DatasetSectionNav({ datasetId }: { datasetId: number }) {
  const items = [
    { label: "Browse", to: `/datasets/${datasetId}`, end: true },
    { label: "Annotate", to: `/datasets/${datasetId}/annotate`, end: false },
    { label: "Prepare", to: `/datasets/${datasetId}/prepare`, end: false },
    { label: "Splits", to: `/datasets/${datasetId}/splits`, end: false },
    { label: "Experiments", to: `/datasets/${datasetId}/experiments`, end: false },
  ];

  return (
    <nav aria-label="Dataset workspace" className="flex min-w-0 items-center gap-1">
      {items.map((item) => (
        <NavLink
          key={item.label}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            cn(
              "rounded-control px-2.5 py-1.5 text-xs font-medium transition-colors",
              isActive ? "bg-surface text-fg shadow-sm" : "text-fg-muted hover:text-fg",
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
