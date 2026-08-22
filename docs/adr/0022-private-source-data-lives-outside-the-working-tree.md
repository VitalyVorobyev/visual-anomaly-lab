# ADR-0022: Private source data lives outside the repository working tree

**Status:** Accepted (2026-08-08)

Supersedes **ADR-0001** (*private data never leaves the machine*, since removed — see ADR-0030).
Its four mechanisms are kept; the arrangement they were protecting is
replaced.

## Context

ADR-0001 opened with the private dataset already sitting at `privatedata/` — inside the git working
directory, in a repository with a configured remote — and spent its whole Decision section building
guards around that fact. Layered ignore rules, reference-in-place, a safety script, a ban on
`git add -A`. Each is sound. None of them was ever asked to justify why 3.2 GB of images nobody is
allowed to publish were living inside the tree in the first place.

That is the load-bearing choice, and it was never a decision — it was a starting condition the
record inherited and then defended. Read back, ADR-0001 is a page of mitigation for a risk that one
line of its own Context created.

Two things make the arrangement worse than it looks.

**Every protection is conditional.** The ignore rules hold unless a file lands somewhere unexpected
with an extension nobody predicted. The safety script holds unless someone forgets to run it. The
`git add -A` ban holds unless someone types it. ADR-0001 said this itself — "the guard is advisory
unless invoked", "discipline is still required" — and then relied on it anyway. Three mechanisms
that must all hold are not defence in depth against a mistake that trips all three at once, which is
exactly what a careless `git add -A && git push` is.

**Nothing required it.** Checked across the whole repository: **no code names that path.** Not the
backend, not the frontend, not the tests. The application imports datasets by absolute path chosen
in the import screen, and the one test suite that touches the private tree finds it through
`ANOMALY_LAB_SHOWCASE_ROOT`, an environment variable that can already point anywhere. The directory
was inside the working tree for convenience of `ls`, and for nothing else.

## Decision

**Private source data lives outside the repository working tree, and the repository has no path
that reaches it.**

- **No `privatedata/` directory, and no symlink standing in for one.** A symlink would restore the
  problem: `git add -A` follows it into the tree it points at.
- **The showcase tree is reached the way every other dataset already is** — an absolute path typed
  into the import screen, and `ANOMALY_LAB_SHOWCASE_ROOT` for the tests that assert its composition.
  Both mechanisms already existed; neither needed the directory to be where it was.
- **The guards from ADR-0001 stay, and stop being the only thing that stands between a typo and a
  leak.** `scripts/check-repo-safety.sh`, the layered ignore rules, synthetic-PNG-only fixtures, and
  explicit staging are all unchanged. They now catch a second, unlikely mistake rather than a first,
  plausible one.
- **Reference in place, never copy** carries over from ADR-0001 unaltered. It was always the right
  half of that record, it shapes the import layer, and ADR-0004 and ADR-0006 both depend on it.

The point is the change in kind. Before, a leak was prevented by three mechanisms that each had to
work. After, a leak requires someone to first move the data into the tree — because **git cannot
stage what is not under the working directory.** The failure mode does not become less likely; it
stops existing.

## Consequences

The commonest catastrophic mistake in this repository is now structurally unavailable rather than
guarded against, and the guards become what they should have been from the start: a check on the
unforeseen, not the primary control.

Negative consequences, accepted honestly:

- **The data is one more thing to locate.** `ls` at the repository root no longer shows where the
  images are, and a new machine needs the path supplied rather than inferred. That is the trade, and
  it is the correct direction: making private data slightly harder to find is not a cost worth
  optimising away.
- **The move is manual and unverifiable from here.** Nothing in the repository can confirm the
  directory is gone, because the absence of a path is not something the repository can observe. The
  safety script still fails if anything named `privatedata` is ever staged, which is the part it can
  check.
- **The superseded record read oddly for as long as it stayed**, describing an arrangement that no
  longer existed. It has since been removed under ADR-0030's rule that a record which no longer
  settles anything is history rather than a decision; what it argued is summarised above.
- **Documentation referencing `privatedata/` had to be swept** — `system-design.md` mentioned it in
  seven places, including a diagram. A stale reference to a directory that must not exist is worse
  than none, because it reads as instruction.
- **This does not make the repository reproducible.** A fresh clone still cannot reproduce a
  showcase result without the operator supplying the data, exactly as before. The public reference
  datasets are the answer to that, and they were already the answer.
