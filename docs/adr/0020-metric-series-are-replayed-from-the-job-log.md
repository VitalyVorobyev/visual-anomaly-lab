# ADR-0020: Metric series are replayed from the job log, not buffered

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-07.

> **Read [`architecture/jobs.md`](../architecture/jobs.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

Extends **ADR-0009** and **ADR-0018**. Neither is reversed: the event protocol is unchanged, there is
still exactly one channel for scalar series, and the server still holds no replay buffer.

## Context

ADR-0018 decided that per-epoch losses and the learning rate stay `metric` events in the job
protocol rather than becoming a second diagnostics channel, and called inventing one "the wrong kind
of completeness". That was right, and it left something unfinished.

`metric` events are **streamed and tee'd, and stored in no column**. The queue writes every worker
line to the job's log file and fans it out to live WebSocket subscribers; `_observe` handles
`progress`, `done` and `error`, and a `MetricEvent` falls through with no branch at all. So the
history exists — in the log file, as JSON lines — and nothing serves it in a usable form.

M4's second exit criterion is that **training is watchable live and the charts survive a page reload
mid-run**. The existing snapshot-then-subscribe rule (§6) covers the console because `JobDetail`
carries `log_tail`, but that is `LOG_TAIL_LINES = 200` raw lines of a stream that also carries
progress frames and library chatter. A 20 000-step EfficientAD run logs five metrics every twentieth
step; two hundred lines is its final seconds. A chart rebuilt from the tail would show a stub of the
end of the run and look like a chart, which is worse than showing nothing.

The tempting fixes were a `job_metric` table (a migration, for data already durable on disk) and a
server-side ring buffer of recent events (state in the API process, lost on restart, and a direct
contradiction of the "no replay buffer" rule the reconnection design depends on).

## Decision

**`GET /api/jobs/{id}/metrics` parses the job's own log file and returns the named series it finds.**

- The log is **already the tee'd source of truth** for the event stream (ADR-0009). Reading it needs
  no table, no migration, and no second event channel. A read over a persisted file is not a replay
  buffer: the server still holds nothing, and a subscriber that was not connected still missed
  those frames — it simply has somewhere to go and look them up.
- **Lines that do not parse, or parse to something other than a metric, are skipped.** The log
  deliberately carries library output and native crash messages verbatim, and tqdm's `\r` frames land
  there too. A malformed metric event — no name, no value, a value that is not a number — is dropped
  rather than charted.
- **Series are capped at 1 000 points using `evenly_spaced`, and `total` and `dropped` are
  returned.** Never the first N: a truncated curve must still show the whole run's shape rather than
  its first seconds. At the current logging cadence the cap almost never bites, and when it does the
  UI says so.
- **The client follows the same snapshot-then-subscribe rule as the console, with the same freeze.**
  `useJob` takes the metric snapshot when the socket opens and appends live frames to it. The
  snapshot cannot be re-read live for the same reason `log_tail` cannot: every event invalidates it,
  and by the time it comes back it already contains the points the socket just delivered.
- **A series that has only ever appeared on the socket is still charted**, so a curve moves in the
  first seconds of a run rather than after the first refetch.

**Ruled out:** a `job_metric` table (a schema migration for data that is already durable, and one
row per point for a series nothing queries relationally); a server-side ring buffer (process state,
lost on restart, contradicting §6); raising `LOG_TAIL_LINES` (the tail is a console, and making it
large enough to carry a training run would ship megabytes of progress frames to draw a line); and a
second `diagnostic` event channel, which ADR-0018 already refused.

## Consequences

The training charts survive a reload, a crash, and reopening the app tomorrow — the log file outlives
all three. `pixel_reference` and `efficientad_anomalib` both light them up with no plugin change,
and any future method that calls `ctx.metric` does too.

Negative consequences, accepted honestly:

- **Parsing a log is a contract by convention.** Nothing enforces that a line claiming
  `{"ev":"metric"}` came from the protocol rather than from a library that happens to print JSON.
  The name-and-value type checks are the whole defence.
- **The read is linear in the log's size, on every request.** A long run's log is megabytes and this
  re-reads all of it to serve a chart that mostly has not changed. Fine for one local user watching
  one run; it is the first thing to look at if the comparison view in M5 ever asks for several runs
  at once.
- **`_log_tail` already read the whole file**, and its docstring claiming a job log is kilobytes was
  written before training runs existed. Corrected, not fixed — both reads still slurp.
- **A cancelled or crashed run's series are whatever reached the log**, which is right, and means a
  chart can end mid-curve with no marker saying why. The job's own terminal status is the only place
  that is stated.
- **The cap is a second place `evenly_spaced` decides what a user sees**, after the diagnostics image
  budget. Both report what they dropped; neither lets the user raise it.
