# ADR-0021: A design token layer, and primitives for the controls Tailwind does not have

**Status:** Accepted (2026-08-07)

## Context

The frontend is styled with Tailwind. With no token layer, colour was a raw ramp step restated at
hundreds of call sites and agreeing by luck, the type face was whatever the OS called `system-ui`,
and dark mode could not be overridden — on a tool whose whole job is judging images. With no
primitives, native `<select>` lists rendered in the OS palette, range inputs were unstyled, tables
were copy-pasted, destructive actions had no confirmation, and nothing had a visible focus ring.
One gap was not cosmetic: the schema-driven form had no control for a closed set, so an enum option
rendered as a text box or a JSON textarea.

The same pressures apply to every lab application, not just this one, and a visual language
defined in one app and copied into the next stops agreeing within a release.

The alternatives were: keep hand-rolling (the remaining controls are the ones where accessibility
is genuinely hard, and a wrong listbox keyboard model is worse than a dependency); a full component
library such as MUI or shadcn/ui (it brings an opinionated visual system this application does not
want, and its distinctive views are custom regardless); and unstyled primitives for the hard
controls only, kept either in this repository or in a shared package.

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
