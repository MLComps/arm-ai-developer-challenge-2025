"""
Global WebSocket endpoint for all updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ...services.websocket_manager import manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_global(websocket: WebSocket):
    """
    Global WebSocket endpoint for all job updates.

    Connect here to receive events for all jobs.
    """
    await manager.connect(websocket)

    try:
        await websocket.send_json({
            "event_type": "connected",
            "message": "Connected to global updates stream"
        })

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@router.get("/ws/stats")
async def websocket_stats():
    """
    Get WebSocket connection statistics.
    """
    return {
        "total_connections": manager.get_connection_count()
    }
