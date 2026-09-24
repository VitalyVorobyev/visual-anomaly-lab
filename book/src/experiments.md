# Experiments and jobs

An experiment is an immutable scientific question. It freezes dataset, split, model type, method options,
shared preprocessing, and region-profile revision. Jobs execute work against that question; they do not
change its definition.

## Creating an experiment

Start from a dataset workspace so its history and available actions stay together. Give the run a name that
states the hypothesis, not merely the architecture—for example `patchcore identity 50k bank` or
`efficientad top64 localized-r3`.

The method form is generated from the plugin's JSON Schema. Leaving a field empty means **unset**: the Python
default remains the authority. This prevents the frontend from freezing a stale copy of a backend default.
Numeric ranges, enums, literals, booleans, and descriptions all originate in the schema.

Experiment history is filterable by method, status, split, and text. Saved search/filter state lets a
researcher return to “all PatchCore runs” or one protocol without reconstructing the query.

## Job lifecycle

Long operations run in worker subprocesses behind one FIFO slot. The API process records the job, streams
JSON-line events, persists the log, and fans progress out over WebSocket. Cancellation is cooperative and
method loops must check it at bounded intervals. A crashed process becomes a failed job with its log; it must
not leave an experiment looking successfully trained.

Training and inference are separate jobs. Training writes the fitted model only after a successful pass.
Inference reads it and writes image scores and raw maps. Export is another ordinary job using the same queue
and cancellation mechanism.

## Resume

Methods declare whether continuation has meaning. `efficientad_custom`, Dinomaly, and GLASS can preserve
optimizer, schedule, step counter, and random streams. PatchCore cannot “continue training”: its fitted bank
is complete or rebuilt under a new experiment. A resumed budget is additional work and its learning-rate
schedule is stated on screen.

## Resource planning

Bound resource use before execution:

- reference images and calibration pixels for statistical methods;
- steps, batch, diagnostics, and asset downloads for deep training;
- candidate vectors and final bank size for memory-bank methods;
- source-map persistence and per-image diagnostics for inference.

When data is dropped by a cap, the log says how many items were used and how they were sampled. Silent first-N
truncation is not acceptable evidence.

## Deletion

Deleting an experiment removes its jobs, results, checkpoints, maps, diagnostics, and exports from the local
workspace after confirmation. Source datasets and annotations are separate. Finished runs are otherwise
immutable; changing a configuration creates another experiment.
