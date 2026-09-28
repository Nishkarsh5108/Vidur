"""WS /ws/live — pushes {"type": "event" | "issue_update" | "vehicle_update", "data": {...}} to dashboards."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["live"])


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        for websocket in list(self.active):
            try:
                await websocket.send_json(message)
            except Exception:
                self.disconnect(websocket)   # client went away; forget it


manager = ConnectionManager()


@router.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    await websocket.send_json({"type": "connected", "data": {"clients": len(manager.active)}})
    try:
        while True:
            await websocket.receive_text()   # client messages are ignored; this just waits for disconnect
    except WebSocketDisconnect:
        manager.disconnect(websocket)
