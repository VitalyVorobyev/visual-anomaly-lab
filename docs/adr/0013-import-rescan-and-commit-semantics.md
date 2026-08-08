# ADR-0013: Import re-scan and commit semantics

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-06.

> **Read [`architecture/import.md`](../architecture/import.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

## Context

ADR-0006 settled that import is two-phase — a named adapter proposes a manifest, a human
reviews it, a commit writes rows — and that images are referenced in place with a `sha256`
recorded per file. It did not say what happens the *second* time a directory is imported,
which is the case that actually occurs: a scan is re-run after a folder is corrected, after
files are added, or simply because someone wants to check that nothing has drifted.

Implementing M2 also measured three things ADR-0006's Context asserted without measuring.
Two of them were wrong, and both were load-bearing:

- **"Files with machine-generated timestamped filenames do not group by stem at all."**
  They do. The timestamped stems match across every channel directory of both affected
  capture groups (39/39 and 42/42). Nothing in the reference data fails to group.
- **"Hashing 3.2 GB costs time — minutes, not seconds."** It takes about two seconds.
  Reading and hashing every byte of the corpus is not the expensive part of import.
  Rendering thumbnails is: roughly 160 ms per image, minutes for a dataset.
- One claim was understated rather than wrong. The tree contains a capture group whose
  channel name is **fused into its own directory name** — two sibling directories named
  `"<Channel> <Group>"` rather than a `<Group>/<Channel>/` pair. An adapter that matches
  whole path components reads these as two unrelated single-image samples. That group is
  also the only one with two illuminations rather than three, and the only 8-bit data.

Together those change what needs designing. The scan is cheap and the review is the point;
the commit is trivial; and the risk in a re-import is not cost but **silent divergence** —
a second parallel dataset, duplicated samples, or a hand-made correction quietly reverted.

## Decision

**Scan is a job. Commit is one synchronous, idempotent transaction. Neither ever destroys
anything.**

- **The scan runs as a `Job`** (ADR-0009) because it walks and hashes an arbitrarily large
  tree and needs progress and cancellation. Its output is a manifest file, not rows.
  *`docs/roadmap.md` and `docs/backlog.md` said the commit was the job; `system-design.md`
  §7's own sequence diagram said the scan was. §7 is correct and the other two are
  corrected.*
- **Commit is a plain endpoint**, one transaction, no job. It measures at ~10 ms for the
  reference corpus because the expensive work already happened during the scan.
- **Identity is by natural key, so re-import updates in place.** A dataset is resolved by
  `root_path` (unique), a sample by `(dataset_id, group_key, external_id)`, an image by
  `(sample_id, path)`. Committing the same manifest twice leaves the same rows.
- **`group_key` retains the label component.** The same numeric stem exists under both a
  defect and a no-defect directory, so a group key that dropped the label would collide two
  different physical parts onto one identity.
- **A `manual` label is never overwritten by an imported one.** `Sample.label_source`
  exists for exactly this: a correction made in the UI survives every future import of the
  tree that originally mislabelled it.
- **A recorded file the manifest no longer mentions is reported, not deleted.** An absent
  file is far more often an unmounted disk than a deletion, and silently dropping catalog
  rows would turn a mount problem into data loss. `verify` reports the same class of drift
  and likewise never repairs it.
- **Import scope is an adapter option, not a review-time gesture.** The `exclude` globs
  travel in the manifest, so narrowing an import to part of a tree is reproducible; a
  one-off deletion of proposed rows in the review UI would be undone by the next scan.
- **Adapters match path components by vocabulary, not by position**, and prefix matching
  applies to *tokens* rather than whole components. Normalization strips separators, so
  `"Brightfield Bl7"` normalizes to a string that begins with `bright`; matching it whole
  would swallow the group name and merge that group into its parent.

## Consequences

Re-importing is safe and boring, which is what makes the reference-in-place model usable
over months: a scan can be re-run at any time to see what changed, and running it twice by
accident costs nothing. Because scope, vocabulary and channel mapping all live in the
manifest, an import is reproducible from the record rather than from someone's memory of
which checkboxes they clicked.

Negative consequences, accepted honestly:

- **A regular dataset gets a review step with nothing in it.** ADR-0006 predicted this;
  the measurements confirm it. With the unlabelled tree excluded, the working corpus
  produces zero warnings, so the review is ceremony exactly where it is least likely to be
  read. The mitigation — the panel states plainly that it found nothing, and demands an
  acknowledgement only when it did — reduces the noise but does not remove the risk.
- **Reported-not-deleted accumulates.** A dataset whose source tree really did shrink keeps
  rows pointing at files that no longer exist, and nothing prunes them. That is a deliberate
  trade against data loss, and it will eventually want a reconciliation tool.
- **`root_path` uniqueness forbids two datasets over one tree.** Importing the same
  directory twice under different options — say, two different channel vocabularies — is
  not expressible without moving or symlinking the source.
- **Token-level channel matching can still be confidently wrong.** It resolves one real
  case that whole-component matching gets silently wrong, and introduces its own: a group
  folder containing a channel word as a token now loses that token from its group key. The
  channel mapping in the manifest is the escape hatch, and it only helps if someone reads it.
- **The corrected cost model shifts where the pain is.** Import is fast and pre-warming
  thumbnails is slow, so the job machinery earns its place through the media layer rather
  than through hashing — the opposite of what ADR-0006 assumed when it called for progress
  reporting on the scan.
