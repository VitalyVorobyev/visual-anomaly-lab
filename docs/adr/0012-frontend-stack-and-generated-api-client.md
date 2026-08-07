# ADR-0012: Frontend stack and a generated API client

**Status:** Accepted (2026-08-06). Its **Styling** clause is superseded by **ADR-0021**; the rest —
routing, server state, and the generated client — stands.

## Context

ADR-0003 settled the boundary between the desktop shell and the backend, and ADR-0002
settled that the frontend is a `bun`-managed React + TypeScript + Vite application. Neither
says what the frontend is built *from*: how it routes, how it holds server state, how it is
styled, or how its TypeScript stays in agreement with the Python routes.

Those choices propagate through every screen the roadmap still has to build — the
virtualized thumbnail grid and grouped sample viewer of M2, the overlay and threshold
controls of M3, the comparison views of M5 — so making them implicitly, one screen at a
time, is how a codebase ends up with three ways to fetch data.

ADR-0003 also records an accepted cost that turns out to be avoidable:

> **Contract drift is a runtime failure.** The TypeScript client and the Python routes are
> checked against each other only by testing.

## Decision

**A deliberately small stack — `react-router`, TanStack Query, Tailwind — with the API
client generated from the backend's OpenAPI schema.**

- **Routing: `react-router`, in its `HashRouter` form.** Fragment routing is not a
  stylistic preference here. The bundle is served from three different places — Vite's dev
  server at `/`, the desktop WebView at `…/index.html`, and `tauri://localhost` once
  packaged — and a path-based router matches no route at the second, rendering an empty
  document that is indistinguishable from a crash. Routing off the fragment behaves
  identically everywhere and survives a reload on a nested route. A catch-all route makes
  an unmatched path visible rather than blank.

- **Server state: TanStack Query.** This application is almost entirely server state —
  datasets, samples, jobs, results — with polling and invalidation on top. Caching,
  refetch intervals and request deduplication are exactly what a job-progress and results
  UI needs, and hand-rolling them per screen is how the alternatives fail.

- **Styling: Tailwind v4.** No component library. Nothing in the milestones needs one yet,
  and the distinctive parts of this UI — the zoom/pan viewer, the anomaly-map overlay, the
  ranked lists — are custom regardless. Revisit if a dialog/slider/tab set is genuinely
  wanted; do not add it speculatively.

- **The API client is generated, not written.** `scripts/gen-api-types.sh` starts a
  throwaway backend on an ephemeral port, reads `/openapi.json`, and emits
  `frontend/src/api/generated.ts` via `openapi-typescript`; a thin `openapi-fetch` client
  is built on those types. The generated file is **committed**, so `tsc` and CI need no
  running backend, and a CI job regenerates it and fails on any diff. A route or field the
  backend does not serve becomes a type error rather than a runtime failure, which retires
  the drift cost ADR-0003 accepted.

- **The frontend imports no Tauri API.** The shell hands over the sidecar's base URL by
  injecting `window.__ANOMALY_LAB__` before the page loads. Resolution order is injected
  global → `VITE_API_BASE_URL` → `http://127.0.0.1:8000`. Nothing under `frontend/src/`
  imports `@tauri-apps/api`, which is what keeps the browser path first-class rather than
  a degraded mode.

- **TypeScript tracks the current release.** `openapi-typescript` builds its output through
  the TypeScript *JS compiler API* and still declares a `^5.x` peer range, so it crashes
  under the 7.0 native compiler. The generator therefore runs in an isolated throwaway
  project with its own TypeScript 5, rather than the whole frontend being held a major
  version back for one build-time tool.

## Consequences

Every screen from M2 onward has one obvious way to fetch, cache and invalidate server
state, and one way to route. The generated client means the backend's schema is the single
definition of the API's shape: renaming a field surfaces at typecheck time, in CI, with the
diff of `generated.ts` showing precisely what changed about the contract.

Negative consequences, accepted honestly:

- **Four dependencies that must be kept current.** React, react-router, TanStack Query and
  Tailwind all ship majors with migration work. This is the ordinary cost of not writing
  them ourselves.
- **Generated code in the tree.** `generated.ts` is committed and must be regenerated when
  the API changes. The CI check catches a stale file, but only after the fact, and merge
  conflicts in it are resolved by regenerating rather than by editing.
- **Hash URLs.** `/#/echo` rather than `/echo`. Invisible in the desktop app, mildly ugly
  in a browser, and the price of routing that does not depend on the serving path.
- **Tailwind in the markup.** Utility classes make components self-contained but verbose,
  and a design-token layer will eventually be wanted if the UI grows a real visual system.
- **One build tool pinned to an older TypeScript.** The isolation in the generation script
  is a workaround with a shelf life; it should be removed once `openapi-typescript`
  supports TypeScript 7.
- **No component library means building primitives by hand.** Sliders, dialogs and tabs
  will each cost real time in M3 and M5. That trade is deliberate now and reversible later.
