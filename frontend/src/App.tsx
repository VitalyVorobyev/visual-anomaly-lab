import { NavLink, Outlet } from "react-router";

const NAV = [
  { to: "/", label: "Health", end: true },
  { to: "/echo", label: "Echo", end: false },
];

export function App() {
  return (
    <div className="min-h-screen bg-white text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <header className="border-b border-slate-200 dark:border-slate-700">
        <div className="mx-auto flex max-w-3xl items-center gap-6 px-6 py-4">
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

      <main className="mx-auto max-w-3xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
