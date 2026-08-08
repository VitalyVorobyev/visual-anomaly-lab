"""Resident worker entrypoint: `python -m anomaly_lab.jobs.inspector <experiment_id>`.

One process per *experiment*, not per request (ADR-0026) — that is the whole point. It
loads the checkpoint once and then answers request frames from stdin until it is evicted,
which is what turns "diagnose this image" from a job you wait a minute for into a button.

It is deliberately dumb. It holds one experiment, chosen on the command line, and it never
switches: when the experiment or the checkpoint changes, the manager kills it and starts
another. That keeps every question about staleness in one place, on the manager's side of
the pipe, rather than distributed across a protocol.

Two rules it inherits verbatim from `jobs/worker.py`, and one it adds:

  * **stdout carries events and nothing else.** Logging goes to stderr, here, once.
  * **SIGTERM is a request, not a kill.** The read loop exits and the process unwinds.
  * **a failed request is an `error` event, not an exit.** A malformed image id or a model
    that raised must not cost the loaded checkpoint — the next request should still be
    warm. Only a failure to *start* is fatal.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import traceback
from types import FrameType
from typing import Any

from anomaly_lab.config import Settings, get_settings
from anomaly_lab.experiments.context import ExperimentJobError, LoadedExperiment
from anomaly_lab.experiments.diagnose import DiagnoseError, diagnose_image, prepare
from anomaly_lab.jobs.protocol import REQUEST_ID, DoneEvent, ErrorEvent, LogEvent, emit

EXIT_OK = 0
EXIT_FAILED = 1


def _configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def install_stop_handler(flag: threading.Event) -> None:
    def _handle(_signum: int, _frame: FrameType | None) -> None:
        flag.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def parse_request(line: str) -> dict[str, Any] | None:
    """Read one request frame, or `None` if this line is not one.

    The same tolerance `parse_line` applies to worker output, for the same reason: a blank
    line or a stray fragment on the request channel is not worth killing a loaded model
    over.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def serve(experiment_id: int, settings: Settings) -> int:
    """Load the experiment, then answer requests until stdin closes or we are asked to stop."""
    stop = threading.Event()
    install_stop_handler(stop)

    try:
        loaded = prepare(settings, experiment_id)
    except (DiagnoseError, ExperimentJobError) as exc:
        emit(ErrorEvent(error_type=type(exc).__name__, message=str(exc)))
        return EXIT_FAILED
    # Broad on purpose, exactly as in `worker.py`: the manager must learn why from an
    # event, not by reconstructing a traceback out of stderr.
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return EXIT_FAILED

    emit(LogEvent(level="info", message=f"resident ready for experiment {experiment_id}"))

    # `readline`, not `for line in sys.stdin`: iterating a text stream reads ahead in
    # chunks, which for a request channel means a frame can sit in a buffer until the next
    # one arrives — a request that appears to hang until you make another.
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            break
        request = parse_request(line)
        if request is not None:
            _answer(loaded, settings, request)

    return EXIT_OK


def _answer(loaded: LoadedExperiment, settings: Settings, request: dict[str, Any]) -> None:
    """Serve one request. Never raises: a bad request costs an answer, not the process."""
    rid = request.get(REQUEST_ID)
    try:
        image_id = int(request["image_id"])
    except (KeyError, TypeError, ValueError):
        emit(ErrorEvent(error_type="ValueError", message=f"malformed request: {request!r}"))
        return

    try:
        keys = diagnose_image(loaded, settings, image_id)
    except DiagnoseError as exc:
        emit(ErrorEvent(error_type=type(exc).__name__, message=str(exc)))
        return
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return

    emit(DoneEvent(result={REQUEST_ID: rid, "image_id": image_id, "keys": keys}))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m anomaly_lab.jobs.inspector <experiment_id>\n")
        return EXIT_FAILED

    try:
        experiment_id = int(args[0])
    except ValueError:
        sys.stderr.write(f"not an experiment id: {args[0]!r}\n")
        return EXIT_FAILED

    _configure_logging()
    return serve(experiment_id, get_settings())


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
