# Diagnostics — what a method shows about itself

A score and a heatmap say whether a model works. They do not say *why*, and "why" is what separates a
research workbench from a batch script. Diagnostics are the channel for the second question.

The governing decision is **ADR-0018**: diagnostics are an optional, *declared* capability, a method
pushes them into a self-describing index, and **the UI renders by `kind` and never by method name.**
That is the property that makes every view written in M4 work unchanged for a method written in M6.
The read path, the numeric readout, the architecture tree and the on-demand entries were four
separate records (0019, 0023, 0024, 0027) under the old immutability rule; they are one page here.

## Authoring

A model that declares `produces_diagnostics` may call
`ctx.emit_diagnostic(key, title, kind, payload)` during `fit` or `predict`. `kind` is one of:

| `kind` | Payload | Stored as |
|---|---|---|
| `map` | 2-D float32 array | `.npy`, colormapped on read |
| `image` | `(H, W, 3)` float32 in `[0, 1]` | `.npy`, already a picture |
| `grid` | 3-D float32 array, addressed one frame at a time | `.npy` |
| `graph` | nodes and edges | inline JSON in the index |
| `table` | rows and columns | inline JSON in the index |

Passing `image_id` scopes a diagnostic to one image; omitting it scopes it to the run. Everything
lands under `artifacts/exp-<id>/diagnostics/`, described by `diagnostics.json` — no schema change, no
migration, and deleting the experiment directory deletes the diagnostics with it.

Two rules make this a capability rather than a conditional threaded through every plugin:

- **A disabled writer accepts every call and does nothing.** A plugin never asks whether diagnostics
  are wanted.
- **Per-image diagnostics are budgeted, and the truncation is recorded.** Three float32 maps per
  image over a few hundred images is hundreds of megabytes. The budget is enforced in the writer, in
  one place, and the index records how many images were dropped — a silent cap would read as "this is
  all there was". Images are chosen with `evenly_spaced`, never the first N.

**Scalar series do not come through here.** Per-branch losses and the learning rate are already
`metric` events in the job protocol ([jobs](jobs.md)); a second channel for the same data would be
the wrong kind of completeness.

## Reading a payload as a picture

`GET /api/experiments/{id}/diagnostics/payload?key=&image_id=&frame=` returns `image/png`.

The client names a `(key, image_id)` pair and the route resolves it **through the index**, never
through the `path` the index carries. Path traversal is therefore impossible by construction rather
than by sanitising a query parameter — the same property the media routes have
([security and privacy](security.md)).

- **`DiagnosticIndex.ranges` maps a key to the `(low, high)` span every emission of that key is drawn
  over**, accumulated by the writer, with the high end at the **99.9th percentile** so one hot pixel
  in one image cannot flatten the whole key to black. Without this, twelve images of
  `map_student_teacher` would be twelve independent normalizations presented as a comparison.
- **Only the colormapped kinds record a range.** An `image` payload is already a picture.
- **`grid` is addressed one frame at a time**, so the small-multiples layout is CSS and the renderer
  stays a single-array function.
- **`graph` and `table` are refused with 400.** Their payload is inline in the index; a second way to
  fetch the same bytes would be a second source of truth.
- **A missing entry is 404, an unreadable file is 410.** The artifact directory is deletable by
  design, so a referenced file that is gone is an expected state, not corruption.
- **Payloads revalidate rather than being immutable.** Re-running inference overwrites maps in place,
  so the ETag covers the file's size and mtime **and the display range**, and the response is
  `Cache-Control: no-cache`.

## Reading a payload as numbers

Reading *a number* is not drawing *a picture*, and the two were conflated once. The colormap, the
alpha rule and the display range stay server-side, in one language; the raw values are served
separately, for a readout only.

- **The format is not `.npy`.** A 24-byte header — ASCII `VAM1`, then `width`, `height`, `stride`,
  `channels` and one reserved word as little-endian `uint32` — then `channels` plane-major blocks of
  little-endian float32. No dtype string, no shape tuple, no pickle path, and therefore no numpy
  reader in TypeScript: the decoder is a `DataView` and a loop.
- **`channels` is in the header**, so nothing has to know the channel count in advance — which the
  channel-count rule ([domain model](domain-model.md)) forbids encoding in the UI anyway.
- **Bounded by an integer `stride` reported in the header.** Above roughly 4 MB the plane is
  decimated and the header says by how much, so a reader can tell an exact value from a sampled one.
  256×256 is always stride 1.
- **Three planes are served, by the routes that already own their addressing**: the stored anomaly
  map (`/api/images/{id}/anomaly-map/values`), any diagnostic array (`format=raw` on the payload
  route), and the **preprocessed source the model actually saw**
  (`/api/experiments/{id}/images/{image_id}/source-values`) — what `load_array` produced, at the
  experiment's own size and colour mode, not an 8-bit preview of it.

Because a diagnostic array is addressed the same way as an anomaly map, the per-branch panes inherit
the hover readout with no code written per method.

## The architecture tree

The `graph` kind carries a whole network rather than three boxes, and the thing that fills it is
**shared and method-agnostic** — `models/introspect.py`, not a recursion inside one plugin.

- **`build_tree(records, *, max_nodes)` is torch-free**, so the hierarchy, the bounding and the
  truncation reporting are tested in CI without the `dl` extra. **`collect(root, probe, *, prefix)`**
  imports torch inside the function, hooks every module, runs one `no_grad` pass, and removes every
  hook in a `finally` — a hook left on a module about to be trained would fire on every step.
- **A plugin contributes only what it alone knows**: which roots to walk, and how they are wired.
  Inter-branch edges live in the training loop's losses, not in any module's `forward`.
- **Two parameter counts.** `parameters_own` is `recurse=False`; `parameters` is the subtree. A
  container's recursive count double-counts against its children, so a single column would not sum
  to anything.
- **Bounded by node count (1500), not by depth**, with `truncated_nodes` reported. Deepest nodes go
  first, so the top of the tree always survives.
- **The view must not draw wiring it did not measure.** `named_modules()` enumerates *modules*;
  `F.relu`, `torch.cat` and an addition written into a `forward` are not modules. A node list from
  hooks is complete, an inferred edge list would not be, and the tab says so in a sentence.

A module that was never called is marked `executed: false` rather than blanked — a dash where a
shape should be says the same thing as a recording failure.

## Two producers: runs and on-demand requests

A run's per-image diagnostics are a **sample** — a budget spread across everything it scored — so for
most images the answer to "what did the branches do here?" was "that image was not one of the
sixty-four". A resident worker answers it for any image on demand ([jobs](jobs.md), **ADR-0026**),
which makes the index a file with two producers writing on different schedules.

Every merge rule is scoped by where an entry came from:

- **`DiagnosticOrigin ∈ {run, on_demand}` on every entry.** Additive, so `INDEX_VERSION` did not
  move; an index written before this reads as `run`, which is what it was.
- **Identity is `(key, image_id, origin)`.** Wholesale supersession applies only within `origin=run`
  and only when the writer is a run writer. Two runs still replace each other entirely; **neither
  origin can delete the other.**
- **A run replaces the scale of the keys it emitted; an on-demand emission may only widen one.**
  Widening keeps every already-drawn picture correct. Narrowing would reinterpret every other image
  against a span fitted from one.
- **`image_budget` and `truncated_images` are run-level facts**, carried forward unchanged by an
  on-demand flush. The UI counts run-origin entries for its budget note and says separately how many
  images were diagnosed on demand.
- **On-demand arrays land under `on-demand/image-{id}/`**, so "clear what I asked for" is a directory
  removal.
- **The index is written atomically** — staged sibling, then `Path.replace`. `load_index` returns an
  *empty* index on a `JSONDecodeError`, which is right for a file never written and misleading for one
  half-written.

Concurrency needs no lock here: the resident is evicted before any job spawns and holds a lock across
each request, so the two writers cannot overlap.

## Deleting them

`DELETE /api/experiments/{id}/diagnostics` reports `removed_entries`, `removed_files`,
`bytes_reclaimed` and `remaining_bytes` — the measured disk delta, not an estimate. Three scopes:

| Scope | Removes |
|---|---|
| `image` (default) | every per-image entry of **both** origins; keeps the model-scoped ones |
| `on_demand` | only what was asked for, leaving the run's own sample |
| `all` | the directory |

- **Model-scoped entries survive the default scope.** The architecture tree, the score-normalization
  table and the teacher views are kilobytes and are what the Architecture and Inspector tabs draw;
  losing them to a routine disk clear would read as those tabs breaking.
- **It deletes directories, not the paths the index names.** A run that crashed after writing arrays
  and before flushing left them unreferenced, and those are exactly the bytes somebody clearing disk
  space wants back.
- **A scale whose entries are all gone goes with them.**
- **`maps/` is out of scope at every scope, permanently.** Each anomaly map is referenced by an
  `ImageResult.map_path`; deleting one orphans a database row and silently breaks the overlay.
  Reclaiming that space means deleting the experiment.
- **Refused with 409 while any job is running.** An inference job's `flush()` merges with what is on
  disk, so a delete landing between the model returning and the flush would be quietly undone.

## What is weak here, and known to be

- **The contract is weakly typed by design.** `kind` says how to draw a payload and nothing about
  what it means. A model emitting a `map` under a misleading `title` produces a plausible, wrong
  picture, and nothing catches it.
- **Key agreement is by convention.** The overlay comparison expects `map_student_teacher` and
  `map_autoencoder`; a method naming them differently silently shows less.
- **The merge rules are genuinely subtle and invisible on screen when wrong** — an erased sample, a
  union that is nobody's sample, a silently re-fitted scale. They are specified by
  `tests/test_diagnostics_index.py` and `tests/test_diagnostics_prune.py`, two of which are
  regression pins for bugs that reached the running application.
- **On-demand entries are unbounded.** Each was individually asked for, so the only brake on disk is
  the delete button.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
