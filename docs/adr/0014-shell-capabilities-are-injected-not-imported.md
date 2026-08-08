# ADR-0014: Shell capabilities are injected, not imported

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-06.

> **Read [`architecture/frontend.md`](../architecture/frontend.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

## Context

ADR-0012 established that nothing under `frontend/src/` imports a Tauri API, and that the
shell hands the sidecar's base URL over by injecting `window.__ANOMALY_LAB__` before the
page loads. That was framed as a fact about one value.

M2 produced the first capability that is not a value. The import flow needs a directory
chooser, and it needs one that returns an **absolute path the backend can open**. A browser
cannot provide that: a file input yields a `File` object and a bare filename, deliberately,
for the same reason the media endpoints refuse to serve files by client-supplied path. The
desktop shell can, through the OS picker.

The obvious implementation is to import `@tauri-apps/plugin-dialog` in the component that
needs it and guard the call. That would end the browser path's first-class status the day
it was written: the bundle would carry a Tauri dependency, `vite dev` would ship code that
cannot run, and every subsequent capability — reveal-in-Finder, save dialogs, a system
notification — would follow the same route until "runs in a browser" became "runs in a
browser with pieces missing".

## Decision

**Every capability the desktop shell offers is injected onto `window.__ANOMALY_LAB__` and
feature-detected by the UI. The frontend imports nothing from Tauri.**

- The shell's initialization script defines the capability as a function that calls into
  Rust, alongside the base URL it already injected.
- One module, `src/api/shell.ts`, declares the shape of that global and exposes a
  `has<Capability>()` / `<capability>()` pair. Nothing else touches the global.
- **A missing capability is a different affordance, not a broken one.** The directory
  picker's absence means a text field, which is a perfectly good way to enter a path and is
  what the browser path uses. The UI must never render a disabled control whose only
  explanation is "you are not in the desktop app".
- Capabilities are for things a browser genuinely cannot do. Anything expressible over HTTP
  belongs in the sidecar, which both hosts already reach identically.

## Consequences

The browser path stays exactly as capable as the API allows, which keeps it usable as the
primary development and debugging surface — the thing ADR-0003 was designed around and the
way most of this project is actually built. The Rust boundary stays one file, and a new
capability costs a `#[tauri::command]`, a line in the injection script, and a
feature-detected helper.

Negative consequences, accepted honestly:

- **The contract is a hand-written global, not a type-checked interface.** The Rust side
  and `shell.ts` must agree by convention; a renamed command fails at runtime, in the
  desktop build only, where the browser tests will never see it.
- **It is untyped at the boundary.** The injected functions are declared by hand in
  `shell.ts` and nothing verifies that the declaration matches what Rust returns — the
  opposite of the generated, checked HTTP contract of ADR-0012.
- **Feature detection multiplies UI paths.** Every capability adds a branch that exists in
  one host and not the other, and only one of them is covered by the browser-based tests.
- **It is not a plugin system.** Capabilities are enumerated in one script, so a third
  place that wants one has to modify the shell — acceptable for a single-window tool, and
  the reason this is a convention rather than an abstraction.
