"""
WebSocket Manager for real-time updates
"""

from fastapi import WebSocket
from typing import Dict, List, Set, Any
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        # Map of job_id -> set of connected websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Global connections (receive all updates)
        self.global_connections: Set[WebSocket] = set()
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str = None):
        """
        Accept and register a WebSocket connection.

        Args:
            websocket: The WebSocket connection
            job_id: Optional job ID to subscribe to specific job updates
        """
        await websocket.accept()

        async with self._lock:
            if job_id:
                if job_id not in self.active_connections:
                    self.active_connections[job_id] = set()
                self.active_connections[job_id].add(websocket)
                logger.info(f"WebSocket connected for job: {job_id}")
            else:
                self.global_connections.add(websocket)
                logger.info("WebSocket connected (global)")

    async def disconnect(self, websocket: WebSocket, job_id: str = None):
        """
        Remove a WebSocket connection.

        Args:
            websocket: The WebSocket connection
            job_id: Optional job ID the connection was subscribed to
        """
        async with self._lock:
            if job_id and job_id in self.active_connections:
                self.active_connections[job_id].discard(websocket)
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]
                logger.info(f"WebSocket disconnected from job: {job_id}")
            else:
                self.global_connections.discard(websocket)
                logger.info("WebSocket disconnected (global)")

    async def send_to_job(self, job_id: str, message: Dict[str, Any]):
        """
        Send message to all connections subscribed to a job.

        Args:
            job_id: The job ID
            message: Message to send
        """
        message['job_id'] = job_id
        message['timestamp'] = datetime.now().isoformat()

        json_message = json.dumps(message, default=str)

        # Send to job-specific connections
        if job_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[job_id]:
                try:
                    await websocket.send_text(json_message)
                except Exception as e:
                    logger.warning(f"Failed to send to websocket: {e}")
                    disconnected.add(websocket)

            # Clean up disconnected sockets
            for ws in disconnected:
                await self.disconnect(ws, job_id)

        # Also send to global connections
        disconnected = set()
        for websocket in self.global_connections:
            try:
                await websocket.send_text(json_message)
            except Exception as e:
                logger.warning(f"Failed to send to global websocket: {e}")
                disconnected.add(websocket)

        for ws in disconnected:
            await self.disconnect(ws)

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connections.

        Args:
            message: Message to broadcast
        """
        message['timestamp'] = datetime.now().isoformat()
        json_message = json.dumps(message, default=str)

        # Send to all job-specific connections
        for job_id, connections in list(self.active_connections.items()):
            disconnected = set()
            for websocket in connections:
                try:
                    await websocket.send_text(json_message)
                except Exception:
                    disconnected.add(websocket)

            for ws in disconnected:
                await self.disconnect(ws, job_id)

        # Send to global connections
        disconnected = set()
        for websocket in self.global_connections:
            try:
                await websocket.send_text(json_message)
            except Exception:
                disconnected.add(websocket)

        for ws in disconnected:
            await self.disconnect(ws)

    def get_connection_count(self, job_id: str = None) -> int:
        """Get number of active connections"""
        if job_id:
            return len(self.active_connections.get(job_id, set()))
        return len(self.global_connections) + sum(
            len(conns) for conns in self.active_connections.values()
        )


# Global connection manager instance
manager = ConnectionManager()
