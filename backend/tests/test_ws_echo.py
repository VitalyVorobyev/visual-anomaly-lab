"""The WebSocket echo endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_echo_round_trips_a_frame(client: TestClient) -> None:
    with client.websocket_connect("/ws/echo") as socket:
        socket.send_text("hello")
        assert socket.receive_json() == {"ev": "echo", "message": "hello"}


def test_echo_stays_open_across_frames(client: TestClient) -> None:
    with client.websocket_connect("/ws/echo") as socket:
        for value in ("one", "two", "three"):
            socket.send_text(value)
            assert socket.receive_json()["message"] == value


def test_echo_uses_the_job_event_envelope(client: TestClient) -> None:
    """Frames carry `ev`, so the M3 job socket is the same client code path (ADR-0009)."""
    with client.websocket_connect("/ws/echo") as socket:
        socket.send_text("x")
        assert "ev" in socket.receive_json()
