# ADR-0012: Frontend stack and a generated API client

**Status:** Accepted (2026-08-06)

## Context

ADR-0003 settles the boundary between shell and backend, and the frontend is a `bun`-managed
React + TypeScript + Vite application. That leaves how it routes, how it holds server state, and
how its TypeScript stays in agreement with the Python routes. Made implicitly, one screen at a
time, those choices are how a codebase ends up with three ways to fetch data.

The sidecar boundary also carries a cost that looks inherent: the TypeScript client and the Python
routes are two definitions of one contract, and drift between them surfaces only at runtime. How
the UI is styled is ADR-0021's.

## Decision

**`react-router` in `HashRouter` form, TanStack Query for server state, and an API client generated
from the backend's OpenAPI schema.**

- **Fragment routing.** The bundle is served from Vite's dev server, from the desktop WebView and,
  once packaged, from `tauri://localhost`; a path-based router matches no route at one of these and
  renders an empty document indistinguishable from a crash. Routing off the fragment behaves the
  same everywhere and survives a reload on a nested route. A catch-all route makes an unmatched path
  visible rather than blank.
- **Server state is TanStack Query.** The application is almost entirely server state — datasets,
  samples, jobs, results — with polling and invalidation on top, which is exactly what it provides.
- **The client is generated, not written.** A script starts a throwaway backend, reads its OpenAPI
  schema and emits TypeScript types; a thin `openapi-fetch` client is built on them. The generated
  file is **committed**, so type-checking needs no running backend, and CI regenerates it and fails
  on any diff. A route or field the backend does not serve is a type error, not a runtime failure.
- **The frontend imports no Tauri API.** The shell injects the sidecar's base URL before the page
  loads, with an environment variable and a default as fallbacks. The browser path stays
  first-class rather than a degraded mode.

## Alternatives considered

- **`BrowserRouter`.** Clean URLs, and a blank screen under one of the three origins the bundle is
  served from.
- **Hand-rolled fetch-and-cache per screen, or a global store.** Either re-implements polling,
  invalidation and deduplication that TanStack Query already has, and a store holds server state
  in a shape that goes stale silently.
- **A hand-written client checked by tests.** Tests catch the drift they were written for; a
  generated client makes every route and field a compile-time fact.

## Consequences

Every screen has one way to fetch, cache and invalidate, and one way to route. The backend's schema
is the single definition of the API: a renamed field surfaces at type-check time, in CI, with the
diff of the generated file showing exactly what changed.

- **Dependencies on the upgrade treadmill.** React, react-router and TanStack Query all ship majors
  with migration work.
- **Generated code in the tree.** It must be regenerated after every API change; CI catches a stale
  file only after the fact, and a merge conflict in it is resolved by regenerating.
- **Hash URLs.** `/#/datasets` rather than `/datasets`: invisible in the desktop app, mildly ugly in
  a browser.
- **The type generator can lag the TypeScript release.** When it does, it runs in its own isolated
  project rather than holding the whole frontend back, and that workaround goes when the tool
  catches up.
