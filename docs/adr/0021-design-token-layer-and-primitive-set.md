# ADR-0021: A design token layer, and primitives for the controls Tailwind does not have

**Status:** Accepted (2026-08-07)

## Context

The frontend is styled with Tailwind. Without a token layer, colour is a raw ramp step restated at
every call site and agreeing by luck, the type face is whatever the OS calls `system-ui`, and dark
mode cannot be overridden — on a tool whose whole job is judging images. Without primitives,
native `<select>` lists render in the OS palette, range inputs are unstyled, tables are copied,
destructive actions have no confirmation, and nothing has a visible focus ring. The schema-driven
form also needs a control for a closed set, or an enum option renders as a text box.

The same pressures apply to every lab application, not just this one, and a visual language
defined in one app and copied into the next stops agreeing within a release.

## Decision

**A semantic token layer and a primitive set, defined once in the shared `@vitavision/lab-ui`
package for every lab application, with Radix underneath only the controls that are hard to build
correctly.**

- **The direction is *instrument*: the chrome is grey so the data can be loud.** One accent,
  `signal`, means "you can act here". `normal`, `defect` and `warn` are reserved for verdicts and
  never decorate. Full saturation appears only in images and charts.
- **Greys are true neutral**, not a blue-cast ramp that shifts how a colormap's cool end reads.
- **Tokens are semantic names registered with `@theme inline`**, so a utility compiles to a custom
  property and both themes are one class on the root. Components name `surface`, `line`,
  `fg-muted`, `signal` — **never a raw ramp step**.
- **Light and dark both ship**, with a three-state choice (`light`, `dark`, `system`) applied before
  first paint.
- **IBM Plex Sans and Mono, self-hosted**, because the packaged shell has no network guarantee;
  numerals are tabular everywhere.
- **Radix sits under `Select`, `Dialog`, `Tooltip`, `Slider`, `Checkbox` and `Switch` only.**
  Everything else is built on native elements that already have the semantics.
- **Every interactive element has a visible focus outline** in the accent colour.
- **The application's own stylesheet only imports the package**, plus rules that are genuinely the
  application's own. A primitive that needs improving is improved upstream, never patched locally.

## Alternatives considered

- **Keep hand-rolling.** The remaining controls are the ones where accessibility is genuinely hard,
  and a wrong listbox keyboard model is worse than a dependency.
- **A full component library** (MUI, shadcn/ui). It brings an opinionated visual system this
  application does not want, and its distinctive views are custom regardless.
- **The token layer and primitives inside this repository.** Cheaper to change, and the next lab
  app copies them and drifts.

## Consequences

Palette, type and focus are defined once for every lab app, a screen cannot invent a fifth grey,
and dark mode is a real feature. The generated forms keep ADR-0006's and ADR-0007's promise for
every pydantic shape, so an enum option costs no frontend work.

- **More dependencies on the upgrade treadmill**, and a second repository to release when a
  primitive changes.
- **Two rendering models for controls.** A Radix select is a button with a portalled listbox, and a
  Radix checkbox has no `.checked`; tests must reach for ARIA roles.
- **The token layer pays off only if components stop naming raw colours.** A ramp step compiles and
  looks almost right; only a grep ratchet in the test suite holds the line.
- **Dropping the UA `<summary>` marker is a trap.** A raw `<details>` renders with no caret and
  reads as a dead panel.
- **Self-hosted fonts add bundle weight** — irrelevant from disk, a real cost on the browser path.
- **Nothing is tested visually.** A palette change that ruins contrast in one theme is caught by a
  person opening the app, or not at all.
