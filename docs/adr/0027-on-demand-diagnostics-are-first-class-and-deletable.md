# ADR-0027: On-demand diagnostics are first-class in the index, and are deletable

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-08.

> **Read [`architecture/diagnostics.md`](../architecture/diagnostics.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

Extends **ADR-0018** (diagnostics as a declarative capability) and **ADR-0019** (payloads addressed
through the index). It says nothing about *how* an on-demand diagnostic is produced — that is
**ADR-0026** — only about what the index does with one once it exists, and how any of it is removed.

## Context

A run's per-image diagnostics are a **sample**. An inference job spreads a budget of 64 images across
however many it scored, so the great majority of images have none, and the answer to "show me what
the branches did on *this* image" was "that image was not one of the sixty-four". Making that
answerable means a second producer of index entries, writing into the same file, on a different
schedule, about one image at a time.

The index was not built for two producers, and the ways it breaks are all silent.

**Supersession was total.** Identity was `(key, image_id)`, and a key emitted per-image by a run
replaced *every* per-image entry under that key. That is right for two runs — the second run's sample
is the sample now — and catastrophic for a browsed image, which the next inference run would erase
without saying so. It is equally wrong in the other direction: one browsed image must not delete a
run's whole sample.

**Ranges are run-wide, and one array is not a run.** `ranges` exists so that every emission of a key
is colormapped on one scale; ADR-0019 already admits "the range is only as good as the run that
recorded it, and nothing warns". If a browsed image could re-fit that scale, every previously drawn
picture of that key would silently change meaning, and it would happen every time somebody opened a
hot image.

**Nothing could delete any of it.** Per-image diagnostics are the bulk of an experiment's disk — a
few float32 maps per image — and the only way to reclaim them was to delete the experiment. Adding a
second producer with no cap at all makes that worse, not better.

## Decision

**An entry records where it came from, every merge rule is scoped by that, and there is a delete.**

- **`DiagnosticOrigin ∈ {run, on_demand}` on every entry.** Additive, so `INDEX_VERSION` does not
  move — the same argument ADR-0019 made for `ranges` — and an index written before this reads as
  `run`, which is what it was. A **literal default is correct here**, unlike on a request body: this
  is a response model, so "required in TypeScript" means the client reads it without a null check.
- **Identity is `(key, image_id, origin)`, and wholesale supersession applies only within
  `origin=run`, and only when the writer is a run writer.** Two runs still replace each other
  entirely, which is the bug ADR-0019's era left fixed. Neither origin can delete the other.
- **A run replaces the scale of the keys it emitted; an on-demand emission may only widen one.**
  Widening keeps every already-drawn picture correct — which is only true because the payload ETag
  now covers the range (**ADR-0023**). Narrowing would reinterpret every other image against a span
  fitted from one.
- **`image_budget` and `truncated_images` are run-level facts and are carried forward unchanged by an
  on-demand flush.** Rewriting them would describe a request that never had a budget. The UI counts
  run-origin entries for its budget note and says separately how many images were diagnosed on
  demand; without that split the note starts lying the moment anybody asks a question.
- **On-demand arrays land under `on-demand/image-{id}/`,** so "clear what I asked for" is a directory
  removal rather than a walk that tells two producers' files apart by reading the index.
- **The index is written atomically** — staged sibling, then `Path.replace`. `load_index` swallows a
  `JSONDecodeError` and returns an *empty* index, which is the right answer for a file that was never
  written and the most misleading one available for a file that was half written. Shared by `flush`
  and `prune`, so the guarantee has one implementation.
- **`prune(root, scope)` with three scopes**, behind `DELETE /api/experiments/{id}/diagnostics`,
  reporting `removed_entries`, `removed_files`, `bytes_reclaimed` and `remaining_bytes` — the
  measured disk delta, not an estimate:
  - `image` (the default) drops every per-image entry of **both** origins and **keeps the
    model-scoped ones**. The architecture tree, the score-normalization table and the teacher views
    are kilobytes, and they are what the Architecture and Inspector tabs draw; losing them to a
    routine disk clear would read as those tabs breaking.
  - `on_demand` drops only what was asked for, leaving the run's own sample.
  - `all` removes the directory.
- **It deletes directories, not the paths the index names.** A run that crashed after writing arrays
  and before flushing left them unreferenced; walking `entry.path` would preserve those for ever, and
  they are exactly the bytes somebody clearing disk space is trying to recover. A test pins this.
- **A scale whose entries are all gone goes with them**, or it would silently re-colour the next
  emission of that key against a span fitted from data that no longer exists.
- **`maps/` is out of scope at every scope, permanently.** It is a sibling directory, and each
  anomaly map is referenced by an `ImageResult.map_path`: deleting one orphans a database row and
  silently breaks the overlay, `has_map` and the map scale. Reclaiming that space means deleting the
  experiment. The tooltip on the button says so.
- **The delete is refused with 409 while any job is running.** An inference job's `flush()` merges
  with what is on disk, so a delete landing between the model returning and the flush would be
  quietly undone — the button would appear to work and sometimes do nothing.

**Concurrency is a non-issue by construction, not by locking**, and this is worth saying where the
code would otherwise invite a lock: ADR-0026's resident is evicted before any job spawns and holds a
lock across each request, so the two writers of this file cannot overlap.

**Ruled out:** a second index file for on-demand entries (two files to merge at read time, and every
consumer would have to know about both); giving on-demand entries their own key namespace (the UI
switches on `kind` and the *same* diagnostic is being asked for — a different key would make it
render as a different thing); a per-experiment cap on on-demand entries (the user asked for each one
individually, and a silent cap on explicit requests is the opposite of the budget note's purpose);
and deleting by walking `entry.path` (leaves orphans behind exactly when disk matters).

## Consequences

A browsed image's diagnostics survive the next inference run, and the disk they cost can be
reclaimed without deleting the experiment.

Negative consequences, accepted honestly:

- **The merge rules are now genuinely subtle**, and every one of them is invisible on screen when
  wrong: an erased sample, a union that is nobody's sample, a silently re-fitted scale. They are
  specified by `tests/test_diagnostics_index.py` and `tests/test_diagnostics_prune.py` rather than by
  prose, and two of those tests are regression pins for bugs that reached the running application.
- **On-demand entries are unbounded.** Nothing caps how many images a user may diagnose, so the only
  brake on disk is the delete button. That is deliberate — each entry was individually asked for —
  but it means the artifact listing's byte count is the thing to watch.
- **A widened range is still a changed range.** Every previously rendered PNG of that key is
  re-fetched because the ETag covers the range, which is correct and is also a cache miss on every
  image of that key. Browsing one hot image therefore costs a re-render of the panes already on
  screen.
- **`prune` measures bytes by walking the tree twice.** On an experiment with thousands of per-image
  arrays that is real I/O in a request handler. It is bounded by the directory being deleted and is
  the price of reporting a number that the file manager beside it will agree with.
