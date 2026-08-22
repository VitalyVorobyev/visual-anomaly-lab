# ADR-0003: Tauri-to-Python boundary is a FastAPI sidecar

**Status:** Accepted (2026-08-06)

## Context

The brief requires a clear boundary between the Tauri application and the Python inference service,
and requires that long-running training and inference operations be asynchronous from the UI's
perspective, exposing progress, logs, completion, and failure states.

The anomaly-detection work is unavoidably Python (PyTorch, anomalib, OpenCV, NumPy). The UI is
unavoidably TypeScript. Something has to carry structured requests one way and a stream of progress
events the other, for jobs that may run for minutes and must be cancellable mid-flight.

Three shapes were considered:

- **Python CLI invoked per operation** — Tauri spawns `python ... --train ...` for each action.
- **Embedded Python** (PyTauri, PyO3, or a bundled interpreter) — one process, direct calls.
- **Local HTTP service (sidecar)** — Python runs as a server the frontend talks to over loopback.

## Decision

**The boundary is a local FastAPI server, spawned by Tauri as a sidecar.** (User decision.)

- On startup the Tauri shell launches the Python process bound to `127.0.0.1` on an ephemeral port,
  receives the chosen port back from the child (port handoff, rather than a hard-coded port that
  could collide), and passes it to the web view. On exit — including unexpected exit — the shell
  terminates the child.
- React talks **REST over HTTP** for commands and queries (datasets, imports, experiments, results)
  and **WebSocket** for job progress and log streaming (see ADR-0009).
- The backend is **fully usable without the desktop shell**: `uv run` starts the same server, and
  the API is exercisable from a plain browser, `curl`, or pytest. The shell is a packaging and
  window-management layer, not a dependency of the domain logic.

Rejected alternatives:

- **CLI per operation.** Fine for one-shot inference, awkward for the actual requirement: streaming
  progress means parsing a child's stdout anyway, cancellation means process signalling anyway, and
  every query would pay interpreter and model import startup cost. We would end up reimplementing a
  worse server.
- **PyTauri / embedded Python.** Young tooling with a small community; packaging PyTorch (and its
  MPS wheels) inside a Rust binary is a known source of pain; and, decisively, it collapses the
  layer separation the brief asks for — Python and Rust would share a process, making the backend
  untestable and unrunnable on its own.

## Consequences

The backend is developed and tested as an ordinary web service, with the fastest possible edit-run
loop and no desktop build in the way. The API contract is inspectable (OpenAPI docs come free), and
a browser-based or headless mode is available at zero extra cost. Should the tool ever need to run
on a remote workstation, only the host binding changes.

Negative consequences, accepted honestly:

- **Two processes to supervise.** Startup ordering, readiness probing, and orphan cleanup are now
  our problem. A crashed or hung backend leaves a window showing an unresponsive UI; a crashed shell
  must not leave a stray Python server holding a port. This lifecycle code is fiddly and easy to get
  subtly wrong on macOS.
- **Serialization overhead and no shared memory.** Images and anomaly maps cross the boundary as
  file paths or encoded bytes rather than in-process arrays. Large-payload endpoints need care.
- **A local port is a local attack surface.** Binding to `127.0.0.1` limits exposure to processes on
  the same machine, but any of them can call the API. With no authentication (explicitly out of
  scope per the brief), we rely on the loopback binding alone.
- **Contract drift is a runtime failure.** The TypeScript client and the Python routes are checked
  against each other only by testing (see ADR-0002).
- **Debugging spans two runtimes.** A failure may live in the Rust shell, the HTTP layer, or the
  Python worker, and stack traces do not cross the boundary.

## Changelog

### 2026-08-22 — The shell resolves `uv` itself, and its setup hook cannot fail

The "fiddly and easy to get subtly wrong on macOS" cost above came due. An installed build aborted
with `SIGABRT` before drawing anything: `Command::new("uv")` searched only the inherited `PATH`, and
an app started by launchd from Finder gets `/usr/bin:/bin:/usr/sbin:/sbin` — no `~/.local/bin`, so
the spawn failed with `ENOENT`. That error then returned from Tauri's `setup` hook, which runs inside
`did_finish_launching`; a panic may not unwind out of an Objective-C callback, so Tauri's `panic!` on
a setup error became `abort()`.

The boundary is unchanged — the shell still spawns the backend from the checkout and reads the port
back. Two obligations are now explicit: the shell **resolves `uv` to an absolute path** across `PATH`
and the known install directories, and **nothing in its setup hook returns an error** — a backend
that will not start is reported through the window it builds regardless. See
[the handbook](../architecture/README.md) for what it does, and
[frontend](../architecture/frontend.md) for what the page shows.
