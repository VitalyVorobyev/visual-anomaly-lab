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
    <section className="flex flex-col items-start gap-2">
      <h1 className="text-xl font-semibold tracking-tight text-fg">No such screen</h1>
      <p className="text-sm text-fg-muted">
        Nothing is routed at <code className="font-mono text-fg">{pathname}</code>.
      </p>
      <Link
        className="text-sm text-signal underline underline-offset-2"
        to="/"
      >
        Go to datasets
      </Link>
    </section>
  );
}
