"""Resident MobileSAM entrypoint: one asset load, many prompt requests."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import traceback
from types import FrameType
from typing import Any

from anomaly_lab.config import Settings, get_settings
from anomaly_lab.jobs.inspector import parse_request
from anomaly_lab.jobs.protocol import REQUEST_ID, DoneEvent, ErrorEvent, LogEvent, emit
from anomaly_lab.model_assets.mobile_sam import MobileSamError, MobileSamSession

EXIT_OK = 0
EXIT_FAILED = 1


def _stop_handler(flag: threading.Event) -> None:
    def handle(_signum: int, _frame: FrameType | None) -> None:
        flag.set()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def serve(asset_key: str, settings: Settings) -> int:
    stop = threading.Event()
    _stop_handler(stop)
    try:
        session = MobileSamSession(settings, asset_key)
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return EXIT_FAILED

    emit(LogEvent(message=f"resident ready for {asset_key} on {session.device}"))
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            break
        request = parse_request(line)
        if request is not None:
            _answer(session, request)
    return EXIT_OK


def _answer(session: MobileSamSession, request: dict[str, Any]) -> None:
    rid = request.get(REQUEST_ID)
    try:
        result = session.segment(request)
    except MobileSamError as exc:
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
        sys.stderr.write("usage: python -m anomaly_lab.jobs.segmenter <asset_key>\n")
        return EXIT_FAILED
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return serve(args[0], get_settings())


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
