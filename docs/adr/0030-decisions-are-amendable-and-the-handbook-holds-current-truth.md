# ADR-0030: Decisions are amendable, and the handbook holds current truth

**Status:** Accepted (2026-08-08)

Supersedes the **immutability rule** and the **Conventions** section of `docs/adr/README.md`. It
reverses no individual decision: every record 0001–0029 still says what it said, and the ones that
still describe live choices are still binding.

## Context

Twenty-nine records exist. They were written over six weeks by one author, under a rule taken from
Nygard's original ADR proposal: *records are immutable once accepted; a decision that no longer
holds is superseded by a new record rather than edited in place.*

That rule assumes a decision log — read chronologically, by people who were not in the room, to
learn why a system is shaped as it is. It does not assume the records will also be used as the
system's documentation. Here they were, and the two uses pull in opposite directions.

The result is visible in the pile. Understanding how diagnostics work means reading **0018** (the
capability), **0019** (the read path it implied), **0023** (a second read path, for numbers rather
than pictures), **0024** (the helper that fills the `graph` kind) and **0027** (what happens when a
second producer writes to the index) — five records, four of which exist because the first could
not be amended. Understanding jobs means **0009**, **0020**, **0025** and **0026** plus
`system-design.md` §6. Not one of those records is wrong. Each was the correct move under the rule,
and the pile is what the rule produces when the thing being recorded is a growing implementation
rather than a sequence of forks in the road.

Two further symptoms, both already recorded inside the records themselves:

- **`system-design.md` and the ADRs disagreed, and the ADR won by being newer.** ADR-0013 had to
  spend a paragraph correcting the roadmap and the backlog about which half of import is a job. A
  document that must be corrected by a later document is not a reference.
- **A record's Consequences section ages into a to-do list.** ADR-0029 predicted its own measurement
  protocol would eventually be wrong; it was wrong within a day. Under immutability the only
  response is a thirtieth record.

The alternative considered and rejected was renumbering: prune to a dozen records and start the
numbering again. **658 citations of the form `ADR-NNNN` exist** across `backend/src`,
`backend/tests`, `frontend/src` and `docs/`. A number is an address, and invalidating every address
to tidy a directory is a bad trade.

## Decision

**`docs/architecture/` is the handbook and holds current truth. An ADR records a choice that had a
live alternative. Accepted records may be amended; only a reversal gets a new number.**

- **The handbook is what you read to learn the system.** One page per bounded area — domain model,
  import, methods, jobs, diagnostics, evaluation, media, frontend, security — replacing the single
  929-line `system-design.md`. Pages carry no status and no date, are edited freely, and describe
  the system as it is *now*. When a page and a record disagree, **the page is right about what the
  code does and the record is right about why it was chosen**.

- **The bar for a new ADR, stated so it can be applied rather than felt:** *would a competent
  engineer plausibly have chosen otherwise, and would changing it now cost more than a refactor?*
  Both must be yes. A contract detail, a helper, a read path for something already decided, and a
  new option on an existing seam are all handbook material.

- **An accepted record may be edited when the decision it describes is refined**, with a dated entry
  in a `## Changelog` section at the foot naming what changed and why. The Decision section then
  reads as one coherent thing rather than as an original plus four extensions filed elsewhere.

- **A reversal still gets a new number and supersedes explicitly.** Reversal is a different
  decision, and the superseded reasoning is what makes the new one legible — ADR-0001 → ADR-0022 is
  the pattern and it stays. Editing a record to say the opposite of what it said would destroy the
  only thing the directory is for.

- **Numbers are permanent, including for folded records.** A record whose current truth has moved
  into the handbook keeps its file, its number and its full text, and gains a status of **Folded**
  and a line at the top naming the page to read instead. Every one of the 658 citations still
  resolves.

- **The existing twenty-nine are classified once, here, by that bar** — seventeen decisions, eleven
  folded, one already superseded. The classification is in `docs/adr/README.md` and is itself
  amendable: a folded record that turns out to still be settling an argument can be un-folded.

## Consequences

There is one place to learn how the system works, and it is not a chronological reconstruction from
thirty documents. A refinement to an existing decision costs a paragraph and a changelog line
instead of a new record, which removes the pressure that produced four of the five diagnostics
records. The `ADR-NNNN` citations in the code keep working unchanged.

Negative consequences, accepted honestly:

- **An amendable record can be quietly rewritten to look prescient.** The whole value of a decision
  log is that it preserves what was believed *at the time*, including what turned out to be wrong —
  ADR-0013's list of three claims ADR-0006 asserted without measuring, two of them false, is the
  most useful paragraph in the directory. The changelog is the only defence and nothing enforces it.
  This is a real loss, taken knowingly.
- **"Folded" is one reader's judgement made in one sitting.** Some of the eleven will turn out to
  have been settling an argument that comes back. The mitigation is that folding costs nothing to
  undo, and that the full text stays.
- **Two places to write now, and the failure mode is writing in neither.** Previously every decision
  had one obvious home. The handbook has no template and no status field, so nothing prompts for
  the honest-negatives section that makes these records worth re-reading.
- **A handbook page describing something that no longer exists looks exactly like one that is
  current.** `system-design.md` already had this failure and it is why ADR-0013 exists; splitting it
  into ten files makes each page smaller and the total surface no smaller.
- **The bar is a judgement call wearing a rule's clothes.** "Would a competent engineer plausibly
  have chosen otherwise" is answerable in the clear cases and arguable in exactly the cases where a
  rule would help most.
