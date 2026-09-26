"""Resident SAM 3 entrypoint: `python -m anomaly_lab.jobs.text_segmenter <asset_key>`.

Loads the catalogued SAM 3 checkpoint once, then answers "what does this phrase name here"
frames from stdin until it is evicted (`explore/text.py`, ADR-0026). The asset key arrives
on the command line, never on the request channel. Same rules as the other residents:
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
from anomaly_lab.explore.text import TextExploreError, TextSession
from anomaly_lab.jobs.inspector import install_stop_handler, parse_request
from anomaly_lab.jobs.protocol import REQUEST_ID, DoneEvent, ErrorEvent, LogEvent, emit

EXIT_OK = 0
EXIT_FAILED = 1


def serve(asset_key: str, settings: Settings) -> int:
    stop = threading.Event()
    install_stop_handler(stop)
    try:
        session = TextSession(settings, asset_key)
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return EXIT_FAILED

    emit(LogEvent(message=f"resident ready: {asset_key} on {session.device}"))
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            break
        request = parse_request(line)
        if request is not None:
            _answer(session, request)
    return EXIT_OK


def _answer(session: TextSession, request: dict[str, Any]) -> None:
    rid = request.get(REQUEST_ID)
    try:
        result = session.answer(request)
    except TextExploreError as exc:
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
        sys.stderr.write("usage: python -m anomaly_lab.jobs.text_segmenter <asset_key>\n")
        return EXIT_FAILED
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return serve(args[0], get_settings())


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
