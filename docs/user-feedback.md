# User Feedback

This document contains the user feedback collected from the visual anomaly lab.

Status: reported, rejected, accepted, resolved.

## Reported

### 002

Make "New experiment" and "All experiments" two separate sub-tabs in the Experiment tab.

### 003

In All experiments, add sorting by AUROC and other columns. Add deleting experiments.

### 004

Make VisA datasets imported and available by default

### 005

make GKN datasets imported and available by default

## Rejected

## Accepted

## Resolved

### 001

Double vertical scroll. There should be only one vertical scroll bar.

Two causes, both fixed. The height chain rested on a single `h-screen` over ancestors with no
height and no `overflow`, so `100vh` could exceed the visible area and the document grew a
scrollbar beside the layout's; `html`, `body` and `#root` are now pinned to `height: 100%;
overflow: hidden` and the shell frame is `h-full`. And a `max-h-* overflow-y-auto` log console sat
inside pages that already scrolled; the tail is now clipped with a `Full log` disclosure. Measured
across five dataset tabs at five widths: `window.innerWidth - documentElement.clientWidth` is 0
everywhere, and each tab has exactly one live scroller.

### 006

The dataset tabs shifted position when switching between them, the "Back to the browser" link
duplicated the Browse tab beside it, and 130–190 px of the window was spent on chrome that differed
per tab. All five tabs now share one `DatasetLayout` band, measured at a constant 86 px with its
strip at a constant y on every tab and every width.

### 007

The dataset band stated too much: `samples 1100 · images 1100 · /Users/…/VisA…`, then the label
counts again on the row below. Two counting units for one dataset, and an absolute path as permanent
furniture. It is one run now — `1100 samples · 1000 normal · 100 defect` — counted in samples,
because `label_counts` and split membership are stored per sample (ADR-0005) and the badges beside
the number share its denominator. A channel count appears only when a sample is more than one image.
The path, the adapter and the import date moved behind an information mark. Band height is unchanged
at 86 px, measured at four widths across all five tabs.

### 008

The single-sample viewer did not look like the rest of the app: four stacked panels in a 72 rem
reading column, with the image third and boxed at a fixed 384 px, and no way to reach the annotation
editor from it. It is a canvas route now — a 48 px band carrying identity, paging and `Open in
editor`, the image filling what is left (778 px of a 900 px window, measured), and the label,
channel and file controls in a 288 px rail. Two latent bugs went with it: the back arrow stayed
enabled and did nothing when the sample had left the loaded page, and the keyboard guard missed
`<textarea>` and `contentEditable`, so typing `n` in a text field relabelled the sample behind it.

### 010

Grouping shipped with no visible way to start a group. The only path was to hover a card until its
action corner faded in, open the edit dialog and type a name into the Collection field that did not
exist yet, which also meant opening it once per dataset. There is now a `New collection` action in
the catalogue header and an edit action on each group heading, both opening one dialog that asks for
the name and the membership together — because a collection is a string on each dataset and exists
only while some dataset names it, so an empty one could not be stored and creating it separately
would be a lie. The same dialog renames a collection and re-files it; unticking every member is what
dissolves one.

### 011

The browse grid's selection box filled in one interaction late: the tile's ring and the selection
count moved on the click, and the tick appeared on the next state change. ⌘-click was unaffected.
The checkbox was a controlled `<input>` *inside* the tile's link, so it had to cancel the click to
stop the navigation, and cancelling a checkbox's click makes the browser restore the previous
`checked` value after React has already written the new one. It is a sibling of the link now, so
nothing needs cancelling; it also became the app's last bare `type="checkbox"` to move onto the
`Checkbox` primitive.

### 009

Registered reference datasets landed in the flat catalogue with nothing tying them to the pack they
came from, while the pack shelf above spent a third of the window on tiles that could not be clicked.
Datasets now carry a `collection` (migration 012) — a user override over the pack a dataset was
registered from — so VisA's twelve classes group under one heading with no backfill and no manual
filing, and your own datasets group by typing a name. The shelf shrank to a strip that renders
nothing once every pack is registered. Each card is a cover thumbnail, a name and an optional
sentence; the counts, the adapter and the absolute path are gone from it.
