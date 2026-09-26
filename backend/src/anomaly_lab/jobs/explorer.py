"""Resident Explore entrypoint: `python -m anomaly_lab.jobs.explorer <backbone>` (ADR-0026).

Loads one frozen encoder, then answers "what does it see here" frames from stdin until it is
evicted (`explore/session.py`). The backbone is the resident's whole identity and arrives on
the command line, never on the request channel. Same rules as the other residents: stdout
carries events only, SIGTERM is a request to stop, and a failed request is an error event
rather than an exit.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from typing import Any

from anomaly_lab.config import Settings, get_settings
from anomaly_lab.explore.session import ExploreError, ExploreSession
from anomaly_lab.jobs.inspector import install_stop_handler, parse_request
from anomaly_lab.jobs.protocol import REQUEST_ID, DoneEvent, ErrorEvent, LogEvent, emit
from anomaly_lab.models.dino_backbone import DinoBackbone

EXIT_OK = 0
EXIT_FAILED = 1


def serve(backbone: DinoBackbone, settings: Settings) -> int:
    stop = threading.Event()
    install_stop_handler(stop)
    try:
        session = ExploreSession(settings, backbone)
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return EXIT_FAILED

    emit(LogEvent(message=f"resident ready: {backbone.value} on {session.device}"))
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            break
        request = parse_request(line)
        if request is not None:
            _answer(session, request)
    return EXIT_OK


def _answer(session: ExploreSession, request: dict[str, Any]) -> None:
    rid = request.get(REQUEST_ID)
    try:
        result = session.answer(request)
    except ExploreError as exc:
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
    emit(DoneEvent(result={REQUEST_ID: rid, **result}))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m anomaly_lab.jobs.explorer <backbone>\n")
        return EXIT_FAILED
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return serve(DinoBackbone(args[0]), get_settings())


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
