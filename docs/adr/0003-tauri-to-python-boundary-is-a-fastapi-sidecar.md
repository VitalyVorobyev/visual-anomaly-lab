# ADR-0003: Tauri-to-Python boundary is a FastAPI sidecar

**Status:** Accepted (2026-08-06)

## Context

The anomaly-detection work is unavoidably Python (PyTorch, NumPy, OpenCV). The UI is unavoidably
TypeScript, wrapped in a Tauri desktop shell. Something has to carry structured requests one way
and a stream of progress events the other, for jobs that run for minutes and must be cancellable
mid-flight, with progress, logs, completion and failure visible to the UI.

Three shapes were considered:

- **A Python CLI invoked per operation** — the shell spawns a process for each action.
- **Embedded Python** (PyTauri, PyO3, or a bundled interpreter) — one process, direct calls.
- **A local HTTP service** — Python runs as a server the frontend talks to over loopback.

## Decision

**The boundary is a local FastAPI server, spawned by the Tauri shell as a sidecar.**

- The shell launches the backend bound to `127.0.0.1` on an ephemeral port that **the child
  chooses and reports back**, rather than a fixed port that could collide, and hands that base URL
  to the web view. When the shell exits, for any reason, it terminates the child.
- The frontend talks **REST** for commands and queries and **WebSocket** for job progress and logs
  (see ADR-0009).
- **The backend is fully usable without the shell.** `uv run` starts the same server, and the API
  is exercisable from a browser, `curl` or pytest. The shell is a packaging and window layer, not a
  dependency of the domain logic.

Rejected:

- **CLI per operation.** Streaming progress would mean parsing a child's stdout anyway,
  cancellation would mean process signalling anyway, and every query would pay interpreter and
  import start-up. It ends as a worse server.
- **Embedded Python.** Young tooling; packaging PyTorch and its MPS wheels inside a Rust binary is
  a known source of pain; and, decisively, it collapses the layer separation — the backend would
  become untestable and unrunnable on its own.

## Consequences

The backend is developed and tested as an ordinary web service with no desktop build in the way.
The contract is inspectable (OpenAPI comes free), a browser mode costs nothing, and running on a
remote workstation would change only the host binding.

- **Two processes to supervise.** Start-up ordering, readiness, and orphan cleanup are ours. A hung
  backend leaves an unresponsive window; a crashed shell must not leave a stray server holding a
  port. This lifecycle code is fiddly on macOS — an app launched from Finder does not inherit the
  user's `PATH`, and a setup error inside the shell aborts rather than unwinds — and the handbook
  records the obligations that follow.
- **No shared memory.** Images and anomaly maps cross the boundary as paths or encoded bytes, so
  large-payload endpoints need care.
- **A local port is a local attack surface.** Loopback limits exposure to processes on the same
  machine, but any of them can call the API; there is no authentication.
- **Debugging spans two runtimes.** A failure may live in the shell, the HTTP layer or the Python
  worker, and stack traces do not cross the boundary.
