"""Resident preview entrypoint: `python -m anomaly_lab.jobs.previewer '<spec json>'` (ADR-0026).

Fits one few-shot method on the reference studio's references once, then answers
"segment this image" frames from stdin until it is evicted (`experiments/preview.py`). The
spec is the whole identity and arrives on the command line, never on the request channel,
so a request cannot change what the resident was fitted on. Same rules as the inspector:
stdout carries events only, SIGTERM is a request to stop, and a failed request is an error
event rather than an exit.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from typing import Any

from anomaly_lab.config import Settings, get_settings
from anomaly_lab.experiments.preview import PreviewError, PreviewSession, PreviewSpec
from anomaly_lab.jobs.inspector import install_stop_handler, parse_request
from anomaly_lab.jobs.protocol import REQUEST_ID, DoneEvent, ErrorEvent, LogEvent, emit

EXIT_OK = 0
EXIT_FAILED = 1


def serve(spec: PreviewSpec, settings: Settings) -> int:
    stop = threading.Event()
    install_stop_handler(stop)
    try:
        session = PreviewSession(settings, spec)
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return EXIT_FAILED

    emit(LogEvent(message=f"resident ready: {spec.method} on {spec.class_key} ({session.device})"))
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            break
        request = parse_request(line)
        if request is not None:
            _answer(session, settings, request)
    return EXIT_OK


def _answer(session: PreviewSession, settings: Settings, request: dict[str, Any]) -> None:
    rid = request.get(REQUEST_ID)
    try:
        result = session.answer(int(request["image_id"]), settings)
    except PreviewError as exc:
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
        sys.stderr.write("usage: python -m anomaly_lab.jobs.previewer '<spec json>'\n")
        return EXIT_FAILED
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return serve(PreviewSpec.model_validate_json(args[0]), get_settings())


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
