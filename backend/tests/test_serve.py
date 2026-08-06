"""The sidecar entrypoint's two non-obvious guarantees.

The end-to-end behaviour (ephemeral port, ready line, teardown) is exercised by running
the app; these tests pin the pieces that are easy to regress silently.
"""

from __future__ import annotations

import os
import subprocess
import sys

from anomaly_lab.serve import _parent_is_alive, _stderr_log_config


def test_stdout_is_left_free_for_structured_events() -> None:
    """uvicorn logs to stdout by default, which would corrupt the ready line."""
    config = _stderr_log_config()
    streams = {handler.get("stream") for handler in config["handlers"].values()}

    assert "ext://sys.stdout" not in streams
    assert "ext://sys.stderr" in streams


def test_a_live_parent_is_detected() -> None:
    assert _parent_is_alive(os.getpid())


def test_a_dead_parent_is_detected() -> None:
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait()  # reap it, so the pid is released rather than left a zombie
    assert not _parent_is_alive(child.pid)


def test_liveness_does_not_assume_a_direct_parent() -> None:
    """`uv run` sits between the shell and this process, so the recorded parent pid is
    not `os.getppid()`. Probing the pid directly is what makes the watchdog correct."""
    assert _parent_is_alive(1)  # launchd: exists, owned by root
    assert os.getppid() != 1
