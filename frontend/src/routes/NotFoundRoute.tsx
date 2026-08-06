import { Link, useLocation } from "react-router";

/**
 * A visible fallback for an unmatched route.
 *
 * Exists because the alternative is an empty document, which looks identical to a crashed
 * app and cost real debugging time once already.
 */
export function NotFoundRoute() {
  const { pathname } = useLocation();

  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
        No such screen
      </h2>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Nothing is routed at <code className="font-mono">{pathname}</code>.
      </p>
      <Link className="text-sm text-blue-600 underline dark:text-blue-400" to="/">
        Back to health
      </Link>
    </section>
  );
}
