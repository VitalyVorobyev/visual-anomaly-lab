# ADR-0022: Private source data lives outside the repository working tree

**Status:** Accepted (2026-08-08)

## Context

The private showcase dataset may never be published, and the repository has a configured remote.
The dataset once sat inside the working tree, protected by guards: layered ignore rules,
reference-in-place, a safety script run before every commit, and a ban on `git add -A`. Each is
sound, and every one is conditional. The ignore rules hold unless a file lands somewhere unexpected
with an extension nobody predicted; the script holds unless someone forgets to run it; the ban holds
unless someone types the command. Three mechanisms that must all hold are not defence in depth
against one careless `git add -A && git push`, which trips all three at once.

Nothing required the data to be inside the tree. No code names the path: datasets are imported by
an absolute path chosen in the import screen, and the one test suite that touches the private tree
finds it through an environment variable that can point anywhere.

The alternatives were to keep the data in the tree behind the guards, or to keep it outside behind a
symlink for convenience.

## Decision

**Private source data lives outside the repository working tree, and the repository has no path
that reaches it.**

- **No in-tree data directory, and no symlink standing in for one.** A symlink restores the
  problem: `git add -A` follows it.
- **The showcase data is reached like every other dataset** — by absolute path in the import
  screen, and by `ANOMALY_LAB_SHOWCASE_ROOT` for the tests that assert its composition.
- **Source images are referenced in place, never copied** into the data directory. The import layer
  and the persistence model depend on this (see ADR-0004, ADR-0006).
- **The guards stay**: the safety script, the ignore rules, synthetic-PNG-only fixtures and explicit
  staging. They now catch a second, unlikely mistake rather than a first, plausible one.

The change is one of kind. A leak now requires someone to move the data into the tree first,
because **git cannot stage what is not under the working directory.**

## Consequences

The commonest catastrophic mistake in the repository is structurally unavailable rather than guarded
against, and the guards become a check on the unforeseen instead of the primary control.

- **The data is one more thing to locate.** `ls` at the root no longer shows it, and a new machine
  needs the path supplied. Making private data slightly harder to find is not a cost worth
  optimising away.
- **The arrangement is unverifiable from inside.** The repository cannot observe the absence of a
  path elsewhere; the safety script can only refuse a staged file that looks like the data.
- **A stale reference to an in-tree data directory reads as an instruction**, and has to be swept
  wherever documentation mentions one.
- **The repository is not reproducible on the showcase data.** A clone cannot reproduce a showcase
  result without the operator supplying the data; the public reference datasets (see ADR-0015) are
  the answer to reproducibility.
