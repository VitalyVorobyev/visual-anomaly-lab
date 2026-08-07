import { NavLink, Outlet } from "react-router";

const NAV = [
  { to: "/", label: "Datasets", end: true },
  { to: "/import", label: "Import", end: false },
  { to: "/experiments", label: "Experiments", end: false },
  { to: "/health", label: "Health", end: false },
];

export function App() {
  return (
    <div className="min-h-screen bg-white text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <header className="border-b border-slate-200 dark:border-slate-700">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
          <h1 className="text-sm font-semibold tracking-tight">visual-anomaly-lab</h1>
          <nav className="flex gap-4">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `text-sm ${
                    isActive
                      ? "font-medium text-slate-900 dark:text-slate-100"
                      : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      {/* Wider than M1's reading-width shell: the browser grid and the side-by-side
          channel viewer are the screens this application exists for. */}
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
