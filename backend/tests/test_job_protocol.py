"""The worker-to-parent event protocol.

The parser's tolerance is the point of these tests. ADR-0009 accepts that stdout is a
fragile channel — any library may print to it — so every "not an event" case below has to
be a `None`, never an exception.
"""

from __future__ import annotations

import pytest

from anomaly_lab.jobs.protocol import (
    DoneEvent,
    ErrorEvent,
    JobEvent,
    LogEvent,
    MetricEvent,
    ProgressEvent,
    encode,
    parse_line,
)


def test_every_event_survives_a_round_trip() -> None:
    events: list[JobEvent] = [
        ProgressEvent(fraction=0.42, message="42 of 100"),
        LogEvent(level="warning", message="device=cpu"),
        MetricEvent(name="train_loss", value=0.0137, step=800),
        DoneEvent(result={"images": 189}),
        ErrorEvent(error_type="RuntimeError", message="boom", traceback="Traceback..."),
    ]

    for event in events:
        assert parse_line(encode(event)) == event


def test_the_error_event_uses_the_documented_wire_name() -> None:
    """ADR-0009 writes `type`; the Python attribute avoids shadowing the builtin."""
    payload = encode(ErrorEvent(error_type="ValueError", message="bad"))

    assert '"type":"ValueError"' in payload
    assert "error_type" not in payload

    parsed = parse_line(payload)
    assert isinstance(parsed, ErrorEvent)
    assert parsed.error_type == "ValueError"


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "Downloading weights: 100%|##########| 45.2M/45.2M",
        "libc++abi: terminating with uncaught exception",
        "not json at all {",
        '{"ev": "unheard-of"}',
        '{"ev": "progress"}',  # missing the required fraction
        '{"ev": "progress", "fraction": 4.2}',  # out of range
        "[1, 2, 3]",
        "null",
    ],
)
def test_anything_that_is_not_an_event_is_log_material(line: str) -> None:
    assert parse_line(line) is None


def test_a_progress_fraction_outside_the_unit_interval_is_rejected() -> None:
    """The `job.progress` column is CHECK-constrained; the parser refuses first."""
    assert parse_line('{"ev":"progress","fraction":-0.1}') is None
    assert parse_line('{"ev":"progress","fraction":1.5}') is None
    assert parse_line('{"ev":"progress","fraction":1.0}') is not None


def test_trailing_whitespace_and_newlines_are_tolerated() -> None:
    line = encode(LogEvent(message="hello")) + "\r\n"
    assert parse_line(line) == LogEvent(message="hello")
