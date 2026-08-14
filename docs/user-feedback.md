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

### 017

"I always see a warning like that: *This dataset cannot share one annotation per part. 11 image
drafts are open; complete or discard them first.*"

**Cause: the editor's read path was a write.** `useEditorDraft` was a TanStack `useQuery` whose
`queryFn` issued `POST .../annotations/draft`, and that route was an idempotent open — so rendering
the editor on an image persisted an `annotation_draft` row, and browsing eleven images left eleven
rows. Worse, `useCompleteDraft` invalidated the draft query while its observer was still mounted, so
every completion refetched, re-POSTed, and **recreated the draft it had just consumed** — one orphan
per unit of finished work. Nothing could remove a row but completing it: there was no `DELETE` route
at all. `count_open_image_drafts` counts rows, so the blocker was permanent and truthful about the
data while being nonsense about the work.

A draft is now created by the first save (ADR-0032's changelog): the `GET` is read-or-seed, the
`POST` is create-only behind `If-None-Match: *` — which also closes a lost-update hole an upsert
cannot — and `DELETE` gives the blocker's own advice somewhere to land, with `If-Match: *` as an
explicit force after a conflict. Migration 016 cleared the eleven. The count keeps no predicate,
because counting rows is now exactly counting work.

### 013

Let the annotation editor show every channel of a sample at once, and optionally share one
annotation across them, so a defect visible only under dark-field can be drawn while looking at it.
Registration was measured first, because the feature is only correct if the frames line up: the
three exposures are 1 ms apart on a moving conveyor, yet phase correlation puts the median offset at
**0 px in every capture group** — the rig triggers on part position. A shared annotation is therefore
exactly right rather than approximately right, and no geometric correction is needed.

A dataset now says whether truth is edited per image or per sample (ADR-0036), and under `sample`
scope one document is edited once and materialised as one ordinary revision per image — so
`resolve_ground_truth_masks`, pixel metrics, the `MetricSet` digest and the three interchange formats
are untouched and never learn that scope exists. The alternative, a sample-level annotation entity,
would have saved two PNGs per part and charged for it in every consumer of ground truth. Entering the
mode is refused, with **every** reason at once, when the dataset has imported source masks, has samples
whose channels differ in size, or has a draft open.

The editor gained a channel strip with three view modes: one channel, two side by side sharing a single
controlled view, and a blend that composites a second channel at adjustable alpha. The blend is not
decoration — it is the only way to *see* the registration the probe measured, since a few pixels of
drift are invisible side by side. Under sample scope switching channel is a display change and the
shapes stay on screen; under image scope it is real navigation and takes the dirty guard, and a
reference pane shows the bare photograph because the document is not truth for it.

### 015

Three follow-ups from opening the merged dataset, each a case of a screen not keeping up with a
dataset that finally has channels. **The browse grid ignored its own channel filter**: filtering by
channel selects whole samples that *have* one, and each tile went on previewing image one, so the same
bright-field thumbnails came back whichever illumination was chosen. Tiles now preview the filtered
channel and name it in place of the channel count, opening a sample lands on that channel, and a sample
missing it falls back to its first image rather than going blank. **The annotation queue could not be
narrowed**: it now filters to one label and to samples still missing ground truth, and each card says
whether that sample's images all have truth, some do, or none — resolved by the same SQL predicate the
filter uses, so the queue cannot disagree with what evaluation will read. **`unsorted/` did not belong**:
those groups are other end types with no normal cases, and a normal-only training population built from
them is not a baseline of anything.

### 016

`set1` and `set2` turned out to be different end types too, which made them two datasets rather than
one — but `dataset.root_path` is unique and is what a re-import resolves against (ADR-0013), so two
scans of one capture tree collided into a single dataset holding both. The scan request gained
`dataset_root`: the path recorded as the dataset's identity, constrained to the scan root or a directory
inside it, while the walk still starts wherever it started. Paired with the adapter's `exclude`, one
channel-first tree becomes one dataset per product variant. This is not a new mechanism — the reference
packs already give each of VisA's twelve classes the root `visa/<class>` while scanning from `visa/` —
only a previously private one made public and put on the import form.

### 014

"The dataset is three grayscale images stored as colour — should we preprocess them?" Measured rather
than assumed, and the obvious test gives the wrong answer: the planes are *not* identical (worst
per-pixel difference is a full 255) while one plane predicts another with a **median R² of 0.9933**.
It is a monochrome sensor with a white-balance cast, so `color=grayscale` is near-lossless — it drops
about 0.6 % of variance, most of it per-plane sensor noise — rather than free or wrong. Nothing was
preprocessed: source images are referenced in place and never copied (ADR-0001/0022), and "these are
monochrome" is one frozen experiment option.

The question exposed a real defect. `color=grayscale` **crashed both EfficientADs** —
`efficientad_nets.py` opens with `nn.Conv2d(3, 32, …)` and neither wrapper expanded a single plane —
while `dinomaly` and `glass` each carried their own copy of the expansion and PatchCore produced the
right answer only because `(B, 1, H, W) - (1, 3, 1, 1)` happens to broadcast into it. One shared
`expand_planes` now sits beside the ImageNet constants, on the method's side of the seam: putting it
in `load_array` would make `grayscale` and `rgb` produce identical arrays, so the experiment would
record a colour choice that changed nothing. The scan now reports plane redundancy as a number on
every import, so the next dataset does not need this investigation.

### 012

Three catalogue entries — one per illumination — were in fact one dataset whose every sample is
photographed three times. The cause was not a modelling decision: the images had been reorganised
from `set*/label/Channel/` into channel-first top-level folders so that each illumination could be
imported separately, and the original `set1/` and `set2/` directories were left behind holding
nothing but `.DS_Store`. That workaround cost exactly what ADR-0005 exists to prevent — three
independent splits over the same 189 physical parts, so one part's bright view could train while its
dark view was tested, and `eval/aggregate.py` had nothing to aggregate. `channel_folders` matches by
path *component* rather than position, so one scan of the parent directory regrouped all of it with
no file moved, and brought in 113 unlabeled samples from `unsorted/` that no dataset had covered.

What was missing was not the domain model but the run: an experiment could not say which channels it
reads, so "bright-field only" had no expression other than a separate dataset. `Experiment.channels`
(ADR-0035) makes that an experiment variable over one split, one region build and one set of labels.
`EvalConfig.channel_normalization` closes ADR-0011's own caveat, so `max` across channels stops
measuring which illumination the method finds noisiest. And `pixel_reference` was quietly wrong on a
merged dataset — one pooled per-pixel median over three illuminations has a MAD dominated by the gap
between them, which inflates the scale and flattens real defects — so it now fits one reference per
channel.

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
