"""WebSocket endpoints.

Mounted without the `/api` prefix, matching `/ws/jobs/{id}` in §2.

The echo endpoint exists to prove the WebSocket path end-to-end in both the desktop
shell and a plain browser. Its frames deliberately use the ADR-0009 event envelope
(`{"ev": ...}`) so that the M3 job socket is the same client code path, not a new one.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["ws"])


@router.websocket("/ws/echo")
async def echo(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_json({"ev": "echo", "message": message})
    except WebSocketDisconnect:
        return
