# ADR-0021: A design token layer, and primitives for the controls Tailwind does not have

**Status:** Accepted (2026-08-07)

Supersedes the **Styling** clause of **ADR-0012** and the two consequences that follow from it.
Nothing else in that record changes: routing is still `HashRouter`, server state is still TanStack
Query, and the API client is still generated from the backend's OpenAPI schema.

## Context

ADR-0012 chose Tailwind v4 with no component library, and recorded the bill honestly:

> **Tailwind in the markup.** Utility classes make components self-contained but verbose, and a
> design-token layer will eventually be wanted if the UI grows a real visual system.

> **No component library means building primitives by hand.** Sliders, dialogs and tabs will each
> cost real time in M3 and M5. That trade is deliberate now and reversible later.

Both conditions arrived at the end of M4. `frontend/src/styles.css` was one line — `@import
"tailwindcss";` — with no `@theme` block, no custom properties, and no `font-family` declared
anywhere in the repository, so the application rendered in whatever the OS thought `system-ui` was.
Colour was a raw `slate-*` utility at more than 250 call sites, which is one palette restated 250
times and agreeing by luck. Dark mode was `prefers-color-scheme` with no override, on a tool whose
whole job is judging images.

The primitive gap had stopped being theoretical too. Eight native `<select>` elements rendered their
option lists in the OS palette in the middle of the app; three `<input type="range">` controls were
entirely unstyled; five screens hand-rolled a `<table>` over a copy-pasted `<thead>`; deleting a
dataset — and every split and experiment built on it — fired on one click with no confirmation; and
**no interactive element in the application had a visible focus ring**, because `inputClasses` set
`focus:outline-none` and replaced it with a border-colour change while `Button` had no focus style at
all.

One of those gaps was not cosmetic. `api/schemaForm.ts` had no control for a closed set, so
`Literal["small", "medium"]` rendered as a free text box and a `StrEnum` — which pydantic emits as a
`$ref` into `$defs` — fell through every branch to a **JSON textarea**, its options not merely
un-offered but invisible. On the experiment screen that was 15 fields, four of them enums, and zero
pickers.

The alternatives considered were: keep hand-rolling (rejected — the four remaining controls are the
ones where accessibility is genuinely hard, and getting a listbox's keyboard model wrong is worse
than a dependency); adopt a full component library such as MUI or shadcn/ui (rejected — it would
bring an opinionated visual system this application specifically does not want, and the distinctive
parts of this UI are still custom); and adopt unstyled primitives for the hard controls only.

## Decision

**A semantic token layer in `styles.css`, and Radix primitives for the four controls that are hard
to build correctly — nothing more.**

- **The direction is *instrument*: the chrome is grey so the data can be loud.** Anomaly maps,
  colormaps and grayscale sensor images are what this application exists to show, so chrome carries
  no saturation at all. Exactly one accent, `signal`, and it means "you can act here" — focus,
  selection, the active nav item, the primary button. `normal` / `defect` / `warn` are reserved for
  verdicts and are never decoration, because a colour that sometimes means nothing teaches a reader
  to ignore it. Charts and images are the only place full saturation appears.

- **Greys are true neutral, not Tailwind's blue-cast `slate`.** A blue-tinted grey beside a viridis
  or inferno map shifts how the map's cool end reads, on the one screen that matters most.

- **Tokens are semantic and registered with `@theme inline`**, so `bg-surface` compiles to
  `var(--surface)` rather than a baked hex, and both palettes are one class on `<html>`. Components
  name `surface`, `line`, `fg-muted`, `signal` — never a ramp step. Retuning the palette is one file.

- **Light and dark both ship, with a three-state choice** (`light` / `dark` / `system`) in
  `localStorage`, applied by an inline script in `index.html` before first paint. "System" is a real
  state and survives a reload as itself.

- **Typography is IBM Plex Sans and IBM Plex Mono, self-hosted via `@fontsource`.** Self-hosted
  because the desktop shell serves from `tauri://localhost` with no network guarantee. Mono is
  effectively the display face — this UI prints numbers everywhere — and `tabular-nums` is global so
  live counters and metric columns do not reflow as digits change.

- **Four Radix primitives, and only four: `Select`, `Dialog`, `Tooltip`, `Slider`** (plus `Checkbox`
  and `Switch`, which come from the same family and are cheap). These are exactly the controls
  ADR-0012 named as the test for revisiting. Everything else — tabs, segmented control, table, panel,
  disclosure, badge, stepper — stays hand-built on native elements, because native `<details>`,
  `<table>` and a radio group already have the semantics and the keyboard model.

- **`clsx` + `tailwind-merge` behind a `cn()` helper.** The old primitives *appended* the caller's
  `className` to their own, so any override won or lost on source order.

- **Every interactive element carries `focus-visible:outline-2 outline-signal`.** An `outline`
  rather than a `ring` because an outline follows the element's own border-radius and needs no
  offset *colour*, and these primitives sit on three different surfaces.

- **`enum` is read before `type`, and `$ref` is resolved through `$defs`.** A closed set of three or
  fewer renders as a segmented control, more as a select. Numeric bounds — `minimum`, `maximum`,
  `multipleOf` — reach the control, and a float gets `step="any"` so a learning rate of `0.0001` is
  not marked invalid by the browser's default step of 1.

- **The "empty means unset" contract is unchanged and remains load-bearing.** A control the operator
  never touched contributes nothing, so a default lives in Python alone. The new pickers honour it:
  a segmented control highlights the *effective* value and stores `""` when that value is the schema
  default, and a select carries an explicit `Default · <value>` entry. Verified on the wire, not
  assumed.

## Consequences

The palette, the type scale and the focus treatment are now defined once. A screen written after
this record cannot invent a fifth grey, and dark mode is a real feature rather than a set of
`dark:` variants nobody could override. The schema-driven form finally keeps ADR-0006 and ADR-0007's
promise for *every* pydantic shape rather than most of them — an adapter or method with a `StrEnum`
option now costs no frontend work, which was the claim all along.

Negative consequences, accepted honestly:

- **Eleven new dependencies**, six of them Radix packages. Each is small and independently
  versioned, but they are now on the upgrade treadmill alongside React, react-router, TanStack Query
  and Tailwind.
- **Two rendering models for controls.** A Radix `Select` is a `button[role="combobox"]` with a
  portalled listbox, not a `<select>`; a Radix `Checkbox` is a `button[role="checkbox"]` with no
  `.checked` property. Tests that reached for the DOM property had to move to the ARIA role, and the
  next person writing one will have to know that.
- **The token layer only pays off if components stop naming raw colours.** Nothing enforces it. A
  `text-slate-500` added in six months will work, look almost right, and quietly ignore the theme.
- **Removing the UA `<summary>` marker globally is a trap.** Any raw `<details>` outside the
  `Disclosure` primitive now renders with no caret and reads as a dead panel; this was found by
  looking at the rendered page, not by a test, and it will recur.
- **The fonts add roughly 200 KB to the bundle.** Irrelevant for a desktop app that loads from
  disk, and a real cost for the browser path the project deliberately keeps first-class.
- **Nothing here is tested visually.** The primitives have no snapshot or visual-regression cover, so
  a palette change that ruins contrast in one theme is caught by a human opening the app, or not at
  all.
