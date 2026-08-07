# ADR-0019: Serving diagnostic payloads through the index, on a recorded scale

**Status:** Accepted (2026-08-07)

Extends **ADR-0018**. Nothing in that record is reversed: the capability flag, the authoring surface
and the self-describing index stay exactly as decided. This adds the read path they implied and
never specified.

## Context

M3 built the write half of the diagnostics contract and stopped there. A model emits
`ctx.emit_diagnostic(...)`, arrays land as float32 `.npy` under `artifacts/exp-<id>/diagnostics/`,
and `GET /api/experiments/{id}/diagnostics` returns the index verbatim. The `graph` and `table`
kinds carry their payload inline and are therefore readable; **`map`, `image` and `grid` are not.**
`DiagnosticEntry.path` is a string relative to the diagnostics directory that no route resolves, so
M4's views had an index of things they could name and not fetch.

Two questions had to be answered before anything could be drawn.

**How is a payload addressed?** The obvious answer is by its `path`, which the index already
carries. That means a client-supplied string reaching the filesystem, and the whole
path-traversal story in §11 rests on the opposite property: files are served by id, resolved
through the database, so there is no request that can name a file.

**On what scale is it drawn?** Anomaly maps have `range.json`, written by the inference job, because
rendering each map against its own extremes makes a clean part look exactly as alarming as a
defective one. Diagnostic arrays had no equivalent. `map_student_teacher` for twelve images would
have been twelve independent normalizations presented as a comparison.

## Decision

**A diagnostic payload is addressed through the index and rendered on a run-wide scale recorded when
it was written.**

- **`GET /api/experiments/{id}/diagnostics/payload?key=&image_id=&frame=`** returns `image/png`. The
  client names a `(key, image_id)` pair; the route resolves it against the index the model wrote and
  uses the `path` found there. **Path traversal is impossible by construction**, not by sanitising a
  query parameter, and `_SAFE_KEY` in the writer remains the single place a key is validated.
- **`DiagnosticIndex.ranges`** maps a key to the `(low, high)` span every emission of that key is
  drawn over, accumulated in the writer. The high end is the **99.9th percentile**, the same choice
  `InferContext.write_map` makes, so one hot pixel in one image cannot flatten the whole key to
  black. Ranges merge on `flush()` by the rule entries already use: this run's keys replace, the
  other run's are kept.
- **Only the colormapped kinds record a range.** An `image` payload is `(H, W, 3)` in `[0, 1]` and is
  already a picture; a range for it would be recorded, never read, and misleading to whoever found
  it.
- **`grid` is addressed one frame at a time.** The small-multiples layout is then CSS, each cell can
  carry its own label, and the renderer stays a single-array function.
- **`graph` and `table` are refused with 400.** Their payload is inline in the index; a second way to
  fetch the same bytes would be a second source of truth.
- **A missing entry is 404, an unreadable file is 410.** The artifact directory is deletable by
  design, so a referenced file that is gone is an expected state and not corruption — the same
  answer the anomaly-map route gives.
- **Payloads revalidate; they are not immutable.** Re-running inference overwrites an image's maps in
  place, so the ETag covers the file's size and mtime and the response is `Cache-Control: no-cache`.
  Claiming `immutable`, as the image tiers correctly do, would leave a browser showing the previous
  run's picture under the current run's caption.
- **`render_anomaly_map` gained `alpha_follows_score`.** Score-driven alpha is the *overlay*
  decision and is wrong outside an overlay: in a panel beside an opaque per-branch map it makes a
  clean image render blank and puts the two panes on visibly different scales.

**Ruled out:** serving by `entry.path` (a client-supplied path reaching the filesystem); a static
mount over the artifact directory (same, plus it exposes the checkpoints); returning raw `.npy` for
the client to decode (a numpy reader in TypeScript, and the colormap would then live in two
languages); and computing the range at read time (cheap per request, wrong across requests — it is
a property of the run, not of the array being fetched).

## Consequences

Every M4 view is one `<img src>` per diagnostic, and the browser does the request, the caching and
the decoding. `efficientad_custom` in M6 inherits the whole read path by emitting the same kinds.

Negative consequences, accepted honestly:

- **Two requests to draw one diagnostic**, the index then the payload — and a `grid` of sixty-four
  channels is sixty-five. Acceptable on a loopback interface to a local sidecar, and the reason the
  grid view caps its first paint and discloses the cap.
- **An index written before this change has no `ranges`**, and those runs fall back to per-array
  extremes — visibly worse, and better than refusing to draw them. The field is additive, so
  `INDEX_VERSION` did not move; a reader cannot distinguish "recorded no range" from "predates
  ranges" without looking at what else the entry carries.
- **The range is only as good as the run that recorded it.** A run cancelled after two images fixes
  the scale from those two, and re-running inference silently re-fixes it. Nothing warns.
- **The 400 on `graph` and `table` is a contract detail a plugin author will meet as an error**
  rather than as a signature. The weak typing ADR-0018 accepted makes that unavoidable here too.
- **`Capabilities` did not grow, but the index did.** ADR-0018's prediction about a growing
  weakly-typed catalogue now applies to `DiagnosticIndex` as well, and this is the first addition.
