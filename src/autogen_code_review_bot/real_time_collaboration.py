#!/usr/bin/env python3
"""
Real-time Collaboration System for AutoGen Code Review Bot.

WebSocket-based real-time agent conversations, live analysis updates,
and collaborative code review sessions.
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import websockets
from redis.asyncio import Redis
from websockets.server import WebSocketServerProtocol

from .agents import ConversationManager
from .logging_utils import get_request_logger as get_logger
from .metrics import get_metrics_registry
from .models import PRAnalysisResult

logger = get_logger(__name__)
metrics = get_metrics_registry()


class MessageType(Enum):
    """WebSocket message types."""
    # Client -> Server
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    START_ANALYSIS = "start_analysis"
    AGENT_MESSAGE = "agent_message"
    USER_COMMENT = "user_comment"

    # Server -> Client
    SESSION_JOINED = "session_joined"
    SESSION_LEFT = "session_left"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_UPDATE = "analysis_update"
    ANALYSIS_COMPLETED = "analysis_completed"
    AGENT_RESPONSE = "agent_response"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ERROR = "error"


@dataclass
class CollaborationSession:
    """Real-time collaboration session."""
    session_id: str
    repository: str
    pr_number: Optional[int]
    created_at: datetime
    participants: Set[str]
    status: str = "active"
    analysis_result: Optional[PRAnalysisResult] = None
    conversation_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

    def add_participant(self, user_id: str):
        """Add participant to session."""
        self.participants.add(user_id)

    def remove_participant(self, user_id: str):
        """Remove participant from session."""
        self.participants.discard(user_id)

    def add_message(self, message: Dict[str, Any]):
        """Add message to conversation history."""
        self.conversation_history.append({
            **message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'repository': self.repository,
            'pr_number': self.pr_number,
            'created_at': self.created_at.isoformat(),
            'participants': list(self.participants),
            'status': self.status,
            'conversation_count': len(self.conversation_history)
        }


@dataclass
class WebSocketConnection:
    """WebSocket connection with user context."""
    websocket: WebSocketServerProtocol
    user_id: str
    session_id: Optional[str] = None
    connected_at: datetime = None

    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = datetime.now(timezone.utc)


class RealTimeCollaborationManager:
    """Manages real-time collaboration sessions and WebSocket connections."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.sessions: Dict[str, CollaborationSession] = {}
        self.connections: Dict[str, WebSocketConnection] = {}
        self.session_connections: Dict[str, Set[str]] = {}
        self.conversation_managers: Dict[str, ConversationManager] = {}
        self.logger = get_logger(__name__ + ".RealTimeCollaborationManager")

    async def create_session(self, repository: str, pr_number: Optional[int] = None,
                           creator_id: str = None) -> str:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())

        session = CollaborationSession(
            session_id=session_id,
            repository=repository,
            pr_number=pr_number,
            created_at=datetime.now(timezone.utc),
            participants=set()
        )

        if creator_id:
            session.add_participant(creator_id)

        self.sessions[session_id] = session
        self.session_connections[session_id] = set()

        # Initialize conversation manager for this session
        self.conversation_managers[session_id] = ConversationManager()

        # Store session in Redis for persistence
        await self.redis.setex(
            f"session:{session_id}",
            3600 * 24,  # 24 hours
            json.dumps(session.to_dict())
        )

        self.logger.info("Collaboration session created", extra={
            'session_id': session_id,
            'repository': repository,
            'pr_number': pr_number,
            'creator_id': creator_id
        })

        return session_id

    async def join_session(self, session_id: str, user_id: str,
                          websocket: WebSocketServerProtocol) -> bool:
        """Add user to collaboration session."""
        if session_id not in self.sessions:
            # Try to load from Redis
            session_data = await self.redis.get(f"session:{session_id}")
            if not session_data:
                return False

            # Recreate session from Redis data
            data = json.loads(session_data)
            session = CollaborationSession(
                session_id=data['session_id'],
                repository=data['repository'],
                pr_number=data.get('pr_number'),
                created_at=datetime.fromisoformat(data['created_at']),
                participants=set(data['participants'])
            )
            self.sessions[session_id] = session
            self.session_connections[session_id] = set()
            self.conversation_managers[session_id] = ConversationManager()

        session = self.sessions[session_id]
        session.add_participant(user_id)

        # Store WebSocket connection
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id
        )
        self.session_connections[session_id].add(connection_id)

        # Notify other participants
        await self.broadcast_to_session(session_id, {
            'type': MessageType.USER_JOINED.value,
            'user_id': user_id,
            'participants_count': len(session.participants)
        }, exclude_connection=connection_id)

        # Send session info to joining user
        await websocket.send(json.dumps({
            'type': MessageType.SESSION_JOINED.value,
            'session': session.to_dict(),
            'connection_id': connection_id
        }))

        self.logger.info("User joined collaboration session", extra={
            'session_id': session_id,
            'user_id': user_id,
            'total_participants': len(session.participants)
        })

        return True

    async def leave_session(self, connection_id: str) -> bool:
        """Remove user from collaboration session."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        session_id = connection.session_id
        user_id = connection.user_id

        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.remove_participant(user_id)

            # Remove connection
            self.session_connections[session_id].discard(connection_id)

            # Notify other participants
            await self.broadcast_to_session(session_id, {
                'type': MessageType.USER_LEFT.value,
                'user_id': user_id,
                'participants_count': len(session.participants)
            })

            # Clean up empty session
            if not session.participants:
                await self.cleanup_session(session_id)

        del self.connections[connection_id]

        self.logger.info("User left collaboration session", extra={
            'session_id': session_id,
            'user_id': user_id
        })

        return True

    async def handle_agent_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle agent message from client."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.session_id:
            return

        session = self.sessions.get(connection.session_id)
        if not session:
            return

        # Get conversation manager for this session
        conv_manager = self.conversation_managers.get(connection.session_id)
        if not conv_manager:
            return

        try:
            # Process message through agent system
            agent_type = message.get('agent_type', 'coder')
            user_message = message.get('message', '')

            # Add to conversation history
            session.add_message({
                'type': 'user_message',
                'user_id': connection.user_id,
                'agent_type': agent_type,
                'message': user_message
            })

            # Get agent response (simulate for now)
            agent_response = await self.get_agent_response(
                agent_type, user_message, session.analysis_result
            )

            # Add agent response to history
            session.add_message({
                'type': 'agent_response',
                'agent_type': agent_type,
                'message': agent_response
            })

            # Broadcast agent response to all session participants
            await self.broadcast_to_session(connection.session_id, {
                'type': MessageType.AGENT_RESPONSE.value,
                'agent_type': agent_type,
                'message': agent_response,
                'user_id': connection.user_id
            })

            self.logger.info("Agent message processed", extra={
                'session_id': connection.session_id,
                'user_id': connection.user_id,
                'agent_type': agent_type
            })

        except Exception as e:
            await connection.websocket.send(json.dumps({
                'type': MessageType.ERROR.value,
                'message': f"Agent message processing failed: {str(e)}"
            }))

    async def start_live_analysis(self, session_id: str, repo_path: str,
                                 config: Optional[Dict[str, Any]] = None):
        """Start live code analysis with real-time updates."""
        session = self.sessions.get(session_id)
        if not session:
            return

        try:
            # Notify session participants that analysis is starting
            await self.broadcast_to_session(session_id, {
                'type': MessageType.ANALYSIS_STARTED.value,
                'repository': session.repository,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            # Import analysis function

            # Run analysis with progress callbacks
            result = await self.run_analysis_with_updates(
                repo_path, session_id, config
            )

            # Store result in session
            session.analysis_result = result
            session.status = "analysis_completed"

            # Broadcast completion
            await self.broadcast_to_session(session_id, {
                'type': MessageType.ANALYSIS_COMPLETED.value,
                'result': {
                    'security': asdict(result.security),
                    'style': asdict(result.style),
                    'performance': asdict(result.performance),
                    'metadata': result.metadata
                }
            })

            self.logger.info("Live analysis completed", extra={
                'session_id': session_id,
                'repository': session.repository
            })

        except Exception as e:
            await self.broadcast_to_session(session_id, {
                'type': MessageType.ERROR.value,
                'message': f"Analysis failed: {str(e)}"
            })

    async def run_analysis_with_updates(self, repo_path: str, session_id: str,
                                       config: Optional[Dict[str, Any]] = None) -> PRAnalysisResult:
        """Run analysis with live progress updates."""
        # Security analysis
        await self.broadcast_to_session(session_id, {
            'type': MessageType.ANALYSIS_UPDATE.value,
            'stage': 'security',
            'status': 'running',
            'message': 'Running security analysis...'
        })

        # Style analysis
        await self.broadcast_to_session(session_id, {
            'type': MessageType.ANALYSIS_UPDATE.value,
            'stage': 'style',
            'status': 'running',
            'message': 'Running style analysis...'
        })

        # Performance analysis
        await self.broadcast_to_session(session_id, {
            'type': MessageType.ANALYSIS_UPDATE.value,
            'stage': 'performance',
            'status': 'running',
            'message': 'Running performance analysis...'
        })

        # Import and run actual analysis
        from .pr_analysis import analyze_pr
        result = analyze_pr(repo_path, config.get('linter_config') if config else None)

        return result

    async def get_agent_response(self, agent_type: str, message: str,
                               analysis_result: Optional[PRAnalysisResult] = None) -> str:
        """Get response from AI agent (simulated for now)."""
        # Simulate agent thinking time
        await asyncio.sleep(1)

        responses = {
            'coder': [
                "I've analyzed the code and found a few potential bugs in the error handling logic.",
                "The implementation looks solid, but consider adding more edge case handling.",
                "This function could be optimized by caching the results of expensive operations."
            ],
            'reviewer': [
                "The code follows most best practices, but documentation could be improved.",
                "Security-wise, make sure to validate all user inputs before processing.",
                "Consider extracting this logic into a separate service for better maintainability."
            ]
        }

        import random
        return random.choice(responses.get(agent_type, ["I need more context to provide a meaningful response."]))

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any],
                                  exclude_connection: Optional[str] = None):
        """Broadcast message to all connections in a session."""
        if session_id not in self.session_connections:
            return

        message_json = json.dumps(message)

        # Send to all connections in the session
        for connection_id in self.session_connections[session_id].copy():
            if connection_id == exclude_connection:
                continue

            connection = self.connections.get(connection_id)
            if not connection:
                continue

            try:
                await connection.websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                # Remove disconnected connection
                self.session_connections[session_id].discard(connection_id)
                if connection_id in self.connections:
                    del self.connections[connection_id]

    async def cleanup_session(self, session_id: str):
        """Clean up empty session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

        if session_id in self.session_connections:
            del self.session_connections[session_id]

        if session_id in self.conversation_managers:
            del self.conversation_managers[session_id]

        # Remove from Redis
        await self.redis.delete(f"session:{session_id}")

        self.logger.info("Collaboration session cleaned up", extra={
            'session_id': session_id
        })


class WebSocketHandler:
    """WebSocket connection handler."""

    def __init__(self, collaboration_manager: RealTimeCollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.logger = get_logger(__name__ + ".WebSocketHandler")

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        connection_id = None

        try:
            self.logger.info("WebSocket connection opened", extra={'path': path})

            async for message_raw in websocket:
                try:
                    message = json.loads(message_raw)
                    message_type = message.get('type')

                    if message_type == MessageType.JOIN_SESSION.value:
                        session_id = message.get('session_id')
                        user_id = message.get('user_id')

                        if session_id and user_id:
                            success = await self.collaboration_manager.join_session(
                                session_id, user_id, websocket
                            )

                            if success:
                                # Find the connection ID for this websocket
                                for cid, conn in self.collaboration_manager.connections.items():
                                    if conn.websocket == websocket:
                                        connection_id = cid
                                        break
                            else:
                                await websocket.send(json.dumps({
                                    'type': MessageType.ERROR.value,
                                    'message': 'Failed to join session'
                                }))

                    elif message_type == MessageType.AGENT_MESSAGE.value:
                        if connection_id:
                            await self.collaboration_manager.handle_agent_message(
                                connection_id, message
                            )

                    elif message_type == MessageType.START_ANALYSIS.value:
                        session_id = message.get('session_id')
                        repo_path = message.get('repo_path')
                        config = message.get('config', {})

                        if session_id and repo_path:
                            await self.collaboration_manager.start_live_analysis(
                                session_id, repo_path, config
                            )

                    elif message_type == MessageType.USER_COMMENT.value:
                        # Handle user comment
                        if connection_id:
                            connection = self.collaboration_manager.connections[connection_id]
                            session_id = connection.session_id

                            await self.collaboration_manager.broadcast_to_session(
                                session_id, {
                                    'type': 'user_comment',
                                    'user_id': connection.user_id,
                                    'comment': message.get('comment', ''),
                                    'timestamp': datetime.now(timezone.utc).isoformat()
                                }, exclude_connection=connection_id
                            )

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': MessageType.ERROR.value,
                        'message': 'Invalid JSON message'
                    }))
                except Exception as e:
                    self.logger.error(f"Message handling error: {e}")
                    await websocket.send(json.dumps({
                        'type': MessageType.ERROR.value,
                        'message': f'Message processing failed: {str(e)}'
                    }))

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")

        finally:
            # Clean up connection
            if connection_id:
                await self.collaboration_manager.leave_session(connection_id)


async def start_collaboration_server(host: str = "0.0.0.0", port: int = 8765,
                                   redis_url: str = "redis://localhost:6379"):
    """Start the real-time collaboration WebSocket server."""
    collaboration_manager = RealTimeCollaborationManager(redis_url)
    handler = WebSocketHandler(collaboration_manager)

    logger.info("Starting real-time collaboration server", extra={
        'host': host,
        'port': port,
        'redis_url': redis_url
    })

    async with websockets.serve(handler.handle_connection, host, port):
        logger.info("Real-time collaboration server started")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    # Start the collaboration server
    asyncio.run(start_collaboration_server())
