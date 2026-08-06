# ADR-0001: Private data never leaves the machine

**Status:** Accepted (2026-08-06)

## Context

The reference dataset (`privatedata/<dataset>/`) consists of ~3.2 GB of proprietary industrial
images of a manufactured circular part: BMP files, roughly 3.9 MB each, several hundred of them.
These images are not ours to publish. The repository has a configured GitHub remote, so a single
careless `git add -A` followed by a push would leak the entire dataset irreversibly — Git history
rewrites do not un-publish what a remote has already accepted.

The risk is amplified by the working style: a solo developer working with an AI coding assistant
that routinely creates, moves, and stages files. Human review of every staged path is not a
reliable control on its own. We need mechanical guards that fail loudly and early.

## Decision

Private data stays on the local filesystem and is never tracked, never copied into tracked paths,
and never pushed:

1. **Ignore rules, layered.** `.gitignore` excludes `privatedata/` (the directory) *and* the
   global patterns `*.bmp` / `*.BMP` (the file type). The redundancy is deliberate defense in
   depth: a stray image copied outside `privatedata/` is still ignored. All future test fixtures
   will be **synthetic PNGs**, so the global BMP ban costs us nothing.
2. **Reference in place, never copy.** The application records absolute or dataset-root-relative
   paths to source images and reads them where they lie (see ADR-0004, ADR-0006). Import never
   duplicates pixel data into the repository or into `data/`; only derived, small artifacts
   (thumbnails, anomaly maps) are written, and those live under the gitignored `data/` tree.
3. **A safety script.** `scripts/check-repo-safety.sh` inspects tracked and staged files and exits
   non-zero if any path matches `privatedata` or `\.bmp$`, or if any file exceeds 5 MB. The size
   check catches unforeseen leak shapes — a checkpoint, an accidental archive, a converted PNG of a
   real part — that the pattern checks would miss.
4. **Explicit first commit.** The initial commit stages an enumerated list of paths. `git add -A`
   and `git add .` are forbidden in this repository, in the first commit and afterwards.

## Consequences

The dataset is protected by three independent mechanisms (ignore rules, path/size gate, explicit
staging), any one of which would catch the common failure mode. The safety script is runnable in
CI or a pre-push hook later without modification.

Negative consequences, accepted honestly:

- **Friction on legitimate binaries.** The 5 MB gate and the BMP ban will eventually block
  something we actually want to commit (a design asset, a large sample fixture). Overriding
  requires editing the script or the ignore rules — a deliberate, reviewable act, but still an
  interruption.
- **Fixtures are synthetic, so they are not the real distribution.** Tests will exercise geometry
  and plumbing on generated PNGs, not on real parts. Bugs that only manifest on genuine sensor noise,
  specular highlights, or 8-bit grayscale captures will not be caught by the test suite (see
  ADR-0010 for the failure modes this leaves unguarded).
- **The guard is advisory unless invoked.** Nothing forces the script to run before a push until a
  hook or CI job exists; discipline is still required in the interim.
- **The repository is not self-contained.** A fresh clone cannot reproduce any experiment without
  the operator separately supplying the dataset. That is the intended trade, but it means results
  in this repository are not independently verifiable by anyone else.
