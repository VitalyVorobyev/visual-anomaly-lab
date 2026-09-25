# Diagnostics — what a method shows about itself

A score and a heatmap say whether a model works, not *why*. Diagnostics answer the second question.

Diagnostics are an optional, *declared* capability (ADR-0018): a method pushes them into a self-describing
index, and **the UI renders by `kind` and never by method name**, so every view works unchanged for a new
method.

## Authoring

A method that declares `produces_diagnostics` may call `ctx.emit_diagnostic(key, title, kind, payload)`
during `fit` or `predict`:

| `kind` | Payload | Stored as |
|---|---|---|
| `map` | 2-D float32 array | `.npy`, colormapped on read |
| `image` | `(H, W, 3)` float32 in `[0, 1]` | `.npy`, already a picture |
| `grid` | 3-D float32 array, addressed one frame at a time | `.npy` |
| `graph` | nodes and edges | inline JSON in the index |
| `table` | rows and columns | inline JSON in the index |

Passing `image_id` scopes a diagnostic to one image; omitting it scopes it to the run. Everything lands
under `artifacts/exp-<id>/diagnostics/`, described by `diagnostics.json` — no schema, no migration, and
deleting the experiment directory deletes the diagnostics.

- **A disabled writer accepts every call and does nothing**, so a plugin never asks whether diagnostics are
  wanted.
- **Per-image diagnostics are budgeted in the writer, and the truncation is recorded** in the index. Images
  are chosen with `evenly_spaced`, never the first N.
- **Scalar series do not come through here.** Losses and the learning rate are `metric` events in the job
  protocol ([jobs](jobs.md)).

## Payload as a picture

`GET /api/experiments/{id}/diagnostics/payload?key=&image_id=&frame=` returns `image/png`. The route
resolves `(key, image_id)` **through the index**, never through the `path` the index carries, so path
traversal is impossible by construction ([security](security.md)).

- **`DiagnosticIndex.ranges` maps a key to the `(low, high)` span every emission of that key is drawn
  over**, accumulated by the writer, with the high end at the **99.9th percentile** so one hot pixel cannot
  flatten the key. Without a shared span, twelve images of one key would be twelve independent
  normalizations presented as a comparison. Only colormapped kinds record a range.
- **`grid` is addressed one frame at a time**, so small multiples are CSS and the renderer stays a
  single-array function.
- **`graph` and `table` are refused with 400**: their payload is inline in the index.
- **A missing entry is 404, an unreadable file is 410** — the artifact directory is deletable by design.
- **Payloads revalidate.** Re-running inference overwrites in place, so the ETag covers size, mtime **and
  the display range**, with `Cache-Control: no-cache`.

**A diagnostic is served in the prepared frame.** A stored anomaly map is projected through the pinned
region transform whenever it is read ([methods](methods.md#anomaly-maps)); a diagnostic is not, because a per-branch
map means what it means on the grid the branch computed it on. The payload route renders at the array's
own size, and the pane is prepared-frame (or grid-frame, for a `grid`).

**What meets it is projected instead.** The ground-truth outline over a pane comes from
`GET /api/images/{id}/mask?frame=prepared&experiment_id=…`, which loads the run's pinned build, applies
`SpatialTransform.prepare_mask` and traces the contour **after** projection — a source-resolution boundary
shrunk to a small grid would drop out in places. The ETag carries frame and manifest digest; a profile
rebuilt under a finished run is a readable 409 from `load_prepared_build`. An identity extractor that keeps
the source aspect ratio differs from the source frame by a uniform scale only, so a frame error shows only
under a real crop or letterbox — test with one.

## Payload as numbers

Reading a number is not drawing a picture. The colormap, alpha rule and display range stay server-side; raw
values are served separately, for a readout only.

- **The format is not `.npy`.** A 24-byte header — ASCII `VAM1`, then `width`, `height`, `stride`,
  `channels` and one reserved word as little-endian `uint32` — then `channels` plane-major blocks of
  little-endian float32. The TypeScript decoder is a `DataView` and a loop.
- **`channels` is in the header**, so nothing encodes a channel count in advance.
- **Bounded by an integer `stride`** reported in the header: above roughly 4 MB the plane is decimated, so a
  reader can tell an exact value from a sampled one. 256×256 is always stride 1.
- **Three planes, served by the routes that own their addressing**: the stored anomaly map
  (`/api/images/{id}/anomaly-map/values`), any diagnostic array (`format=raw` on the payload route), and the
  **preprocessed source the model saw** (`/api/experiments/{id}/images/{image_id}/source-values`) — what
  `load_array` produced, projected into source coordinates (NaN outside the crop).

Because a diagnostic array is addressed like an anomaly map, every per-branch pane gets the hover readout
with no per-method code.

## Architecture tree

The `graph` kind carries a whole network, filled by the **shared, method-agnostic** `models/introspect.py`.

- **`build_tree(records, *, max_nodes)` is torch-free**, so hierarchy, bounding and truncation are tested
  without the `dl` extra. **`collect(root, probe, *, prefix)`** imports torch inside the function, hooks
  every module, runs one `no_grad` pass and removes every hook in a `finally` — a hook left on a module about
  to be trained would fire every step.
- **A plugin contributes only what it alone knows**: which roots to walk and how they are wired.
- **Two parameter counts**: `parameters_own` (`recurse=False`) and `parameters` (the subtree).
- **Bounded by node count (1500), not depth**, with `truncated_nodes` reported; deepest nodes go first.
- **No inferred wiring.** Hooks enumerate modules, not `F.relu` or `torch.cat` in a `forward`, so the node
  list is complete and an edge list would not be; the tab says so. A module never called is marked
  `executed: false`.

## Producers: runs and on-demand requests

A run's per-image diagnostics are a budgeted **sample**. The resident worker answers for any image on demand
([jobs](jobs.md), ADR-0026), so the index has two producers on different schedules, and every merge rule is
scoped by origin. A request hands the model what `infer` would: the image alone, or — for a
`channel_aware` method, which may fuse a sample's channels — the image with the rest of its sample, whose
siblings' entries are recorded too.

- **`DiagnosticOrigin ∈ {run, on_demand}` on every entry**; an entry without one reads as `run`.
- **Identity is `(key, image_id, origin)`.** Wholesale supersession applies only within `origin=run` and only
  for a run writer; **neither origin can delete the other**.
- **A run replaces the scale of the keys it emitted; an on-demand emission may only widen one**, which keeps
  every drawn picture correct.
- **`image_budget` and `truncated_images` are run-level facts**, carried forward by an on-demand flush. The UI
  counts run-origin entries for its budget note and reports on-demand images separately.
- **On-demand arrays land under `on-demand/image-{id}/`**, so clearing them is a directory removal.
- **The index is written atomically** — staged sibling, then `Path.replace` — because `load_index` returns an
  empty index on a `JSONDecodeError`.

No lock is needed: the resident is evicted before any job spawns and holds a lock across each request, so
the writers cannot overlap.

## Deletion

`DELETE /api/experiments/{id}/diagnostics` reports `removed_entries`, `removed_files`, `bytes_reclaimed` and
`remaining_bytes` — the measured disk delta.

| Scope | Removes |
|---|---|
| `image` (default) | every per-image entry of **both** origins; keeps the model-scoped ones |
| `on_demand` | only what was asked for, leaving the run's own sample |
| `all` | the directory |

- **Model-scoped entries survive the default scope** — the architecture tree, normalization table and teacher
  views are kilobytes and are what the Architecture and Inspector tabs draw.
- **It deletes directories, not the paths the index names**, so arrays left unreferenced by a crashed run
  are reclaimed too. A scale whose entries are all gone goes with them.
- **`maps/` is out of scope at every scope.** Each map is referenced by an `ImageResult.map_path`;
  reclaiming that space means deleting the experiment.
- **Refused with 409 while any job is running**, because an inference job's `flush()` merges with what is
  on disk and would undo the delete.

## Known weaknesses

- **The contract is weakly typed.** `kind` says how to draw a payload, not what it means; a `map` under a
  misleading `title` is a plausible, wrong picture.
- **Key agreement is by convention.** The overlay comparison expects `map_student_teacher` and
  `map_autoencoder`; a method naming them differently silently shows less.
- **The merge rules are subtle and invisible on screen when wrong.** `tests/test_diagnostics_index.py` and
  `tests/test_diagnostics_prune.py` specify them.
- **On-demand entries are unbounded**; the only brake on disk is the delete button.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
