"""Sidecar entrypoint — `python -m anomaly_lab.serve`.

Started by the Tauri shell (ADR-0003). Three things distinguish it from plain `uvicorn`:

1. **It binds the listening socket itself.** With `ANOMALY_LAB_PORT=0` the OS assigns a
   free port, and because the socket is never released between choosing and serving,
   there is no bind → close → re-bind race. This is why the *child* announces the port
   rather than the shell allocating one (ADR-0003).
2. **stdout is a structured channel.** The port is announced as one JSON line using the
   ADR-0009 event envelope, so uvicorn's logging is redirected to stderr. ADR-0009 names
   "a stray print corrupts the stream" as an accepted risk; keeping the two streams
   separate from the start is the cheap mitigation.
3. **It self-terminates when orphaned.** macOS has no `PDEATHSIG` equivalent, so a
   force-quit or crash of the shell runs no cleanup handler at all. This watchdog is what
   actually satisfies "closing the app leaves no orphaned Python process".
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import sys
import threading
import time
from typing import Any

import uvicorn

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings, get_settings
from anomaly_lab.db.migrate import apply_migrations

logger = logging.getLogger("anomaly_lab.serve")

WATCHDOG_POLL_SECONDS = 1.0
WATCHDOG_GRACE_SECONDS = 3.0


def _stderr_log_config() -> dict[str, Any]:
    """uvicorn's default log config writes to stdout; move it to stderr."""
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    for handler in config["handlers"].values():
        if handler.get("stream") == "ext://sys.stdout":
            handler["stream"] = "ext://sys.stderr"
    return config


def _parent_is_alive(parent_pid: int) -> bool:
    """Signal 0 checks for the existence of a process without touching it.

    Deliberately *not* `os.getppid() == parent_pid`: the shell launches this process via
    `uv run`, which may sit in between as an intermediate process, so the immediate parent
    is not necessarily the shell. Probing the recorded pid works either way — and it also
    covers the case where an intermediate process fails to forward SIGTERM.
    """
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The pid exists but belongs to another user. Assume alive rather than exit.
        return True
    return True


def _start_orphan_watchdog(server: uvicorn.Server, parent_pid: int) -> None:
    def watch() -> None:
        while _parent_is_alive(parent_pid):
            time.sleep(WATCHDOG_POLL_SECONDS)

        # The parent's end of the pipe is closed, so any write here may raise.
        with contextlib.suppress(OSError, ValueError):
            logger.warning("Parent process %d is gone; shutting down.", parent_pid)

        server.should_exit = True
        time.sleep(WATCHDOG_GRACE_SECONDS)
        # Hard exit: an orphaned sidecar holding a port is worse than a skipped teardown.
        os._exit(1)

    threading.Thread(target=watch, name="orphan-watchdog", daemon=True).start()


def main() -> int:
    settings: Settings = get_settings()
    settings.ensure_directories()

    # Migrate before announcing readiness. A shell waiting on the ready line should see
    # this process exit with a message rather than hang until its timeout.
    apply_migrations(settings.db_path)

    config = uvicorn.Config(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=_stderr_log_config(),
        access_log=False,
    )
    server = uvicorn.Server(config)

    sock = config.bind_socket()
    port = int(sock.getsockname()[1])

    if settings.parent_pid is not None:
        _start_orphan_watchdog(server, settings.parent_pid)

    sys.stdout.write(json.dumps({"ev": "ready", "port": port, "pid": os.getpid()}) + "\n")
    sys.stdout.flush()

    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    sys.exit(main())
