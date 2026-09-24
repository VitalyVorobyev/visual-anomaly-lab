# ADR-0030: Decisions are amendable, and the handbook holds current truth

**Status:** Accepted (2026-08-08)

## Context

These records were once immutable, as in Nygard's original proposal: a decision that no longer held
was superseded by a new record rather than edited. Used as the system's documentation, that rule
produced a pile. Understanding diagnostics took five records, four of which existed only because
the first could not be amended. Records also drifted into changelogs and measurement reports,
restating what git and `measurements.md` already hold.

The alternatives were to keep immutability, or to prune and renumber. Renumbering would invalidate
hundreds of `ADR-NNNN` citations in code and docs, and a number is an address.

## Decision

- **The handbook (`docs/architecture/`) says what the code does now.** Pages carry no status or
  date and are edited in the same change as the code. When a page and a record disagree, the page
  is right about *what* and the record about *why*.
- **A record captures a choice that had a live alternative.** The bar has two parts, and both must
  be yes: would a competent engineer plausibly have chosen otherwise, and would changing it now
  cost more than a refactor? A contract detail, a helper, a read path or a new option on an
  existing seam is handbook material.
- **A record is edited in place when its decision is refined.** It has Context, Decision and
  Consequences, and nothing else. It has no changelog, no narrative of what a change did, and no
  measured numbers; those live in git history and `measurements.md`.
- **A reversal gets a new number** and supersedes the old record explicitly, restating whatever it
  keeps from it.
- **A record whose truth has moved into the handbook is removed**, and every citation of it is
  repointed in the same change. Numbers are never reused.

## Consequences

- There is one place to learn the system, and the records stay short enough to re-read.
- **An edited record can be quietly rewritten to look prescient.** What was believed at the time,
  including what turned out to be wrong, now lives only in git history. This is taken knowingly.
- Removal is a judgement call. A removed argument that comes back has to be recovered from `git log`.
- The bar is a judgement wearing a rule's clothes. It is clear at the extremes and arguable in the
  middle.
