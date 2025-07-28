"""
Real-time Collaboration and Multi-user Synchronization Module
Implements WebSocket-based real-time collaboration features for the LLM Risk Visualizer
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
import time
import websockets
import redis
from queue import Queue
import streamlit as st

@dataclass
class CollaborationEvent:
    """Represents a collaboration event"""
    event_id: str
    event_type: str  # 'filter_change', 'annotation', 'chat_message', 'cursor_move', etc.
    user_id: str
    username: str
    timestamp: datetime
    data: Dict[str, Any]
    room_id: str = "default"

@dataclass
class ActiveUser:
    """Represents an active user in the collaboration session"""
    user_id: str
    username: str
    role: str
    joined_at: datetime
    last_seen: datetime
    current_page: str = ""
    cursor_position: Dict[str, float] = None
    active_filters: Dict[str, Any] = None

class CollaborationManager:
    """Manages real-time collaboration sessions"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = self._init_redis(redis_host, redis_port)
        self.active_users: Dict[str, ActiveUser] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.rooms: Dict[str, Set[str]] = {}  # room_id -> set of user_ids
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        
        # Event queue for processing
        self.event_queue = Queue()
        self.processing_thread = None
        self.is_running = False
        
        # WebSocket connections
        self.websocket_connections: Dict[str, Any] = {}
        
    def _init_redis(self, host: str, port: int) -> redis.Redis:
        """Initialize Redis connection for real-time synchronization"""
        try:
            client = redis.Redis(host=host, port=port, decode_responses=True)
            client.ping()  # Test connection
            return client
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None
    
    def start_collaboration_service(self):
        """Start the collaboration service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        # Start WebSocket server
        if self.redis_client:
            threading.Thread(target=self._start_websocket_server, daemon=True).start()
    
    def stop_collaboration_service(self):
        """Stop the collaboration service"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
    
    def join_collaboration_room(self, user_id: str, username: str, role: str, room_id: str = "default") -> bool:
        """Add user to collaboration room"""
        try:
            # Create active user
            active_user = ActiveUser(
                user_id=user_id,
                username=username,
                role=role,
                joined_at=datetime.now(),
                last_seen=datetime.now()
            )
            
            self.active_users[user_id] = active_user
            
            # Add to room
            if room_id not in self.rooms:
                self.rooms[room_id] = set()
            self.rooms[room_id].add(user_id)
            
            # Store in Redis for persistence
            if self.redis_client:
                user_data = {
                    'username': username,
                    'role': role,
                    'joined_at': active_user.joined_at.isoformat(),
                    'room_id': room_id
                }
                self.redis_client.hset(f"active_users:{room_id}", user_id, json.dumps(user_data))
                self.redis_client.expire(f"active_users:{room_id}", 3600)  # 1 hour expiry
            
            # Broadcast user joined event
            self._broadcast_event(CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type="user_joined",
                user_id=user_id,
                username=username,
                timestamp=datetime.now(),
                data={"role": role},
                room_id=room_id
            ))
            
            return True
            
        except Exception as e:
            print(f"Error joining collaboration room: {e}")
            return False
    
    def leave_collaboration_room(self, user_id: str, room_id: str = "default"):
        """Remove user from collaboration room"""
        try:
            if user_id in self.active_users:
                username = self.active_users[user_id].username
                del self.active_users[user_id]
                
                # Remove from room
                if room_id in self.rooms:
                    self.rooms[room_id].discard(user_id)
                    
                    # Clean up empty rooms
                    if not self.rooms[room_id]:
                        del self.rooms[room_id]
                
                # Remove from Redis
                if self.redis_client:
                    self.redis_client.hdel(f"active_users:{room_id}", user_id)
                
                # Broadcast user left event
                self._broadcast_event(CollaborationEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="user_left",
                    user_id=user_id,
                    username=username,
                    timestamp=datetime.now(),
                    data={},
                    room_id=room_id
                ))
                
        except Exception as e:
            print(f"Error leaving collaboration room: {e}")
    
    def update_user_activity(self, user_id: str, activity_data: Dict[str, Any], room_id: str = "default"):
        """Update user activity (page, filters, cursor position, etc.)"""
        if user_id not in self.active_users:
            return
        
        user = self.active_users[user_id]
        user.last_seen = datetime.now()
        
        # Update specific activity data
        if 'current_page' in activity_data:
            user.current_page = activity_data['current_page']
        
        if 'cursor_position' in activity_data:
            user.cursor_position = activity_data['cursor_position']
        
        if 'active_filters' in activity_data:
            user.active_filters = activity_data['active_filters']
        
        # Broadcast activity update
        self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type="user_activity",
            user_id=user_id,
            username=user.username,
            timestamp=datetime.now(),
            data=activity_data,
            room_id=room_id
        ))
    
    def send_chat_message(self, user_id: str, message: str, room_id: str = "default") -> bool:
        """Send chat message to room"""
        if user_id not in self.active_users:
            return False
        
        user = self.active_users[user_id]
        
        # Create chat event
        chat_event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type="chat_message",
            user_id=user_id,
            username=user.username,
            timestamp=datetime.now(),
            data={
                "message": message,
                "message_id": str(uuid.uuid4())
            },
            room_id=room_id
        )
        
        # Store chat message in Redis for history
        if self.redis_client:
            chat_data = asdict(chat_event)
            self.redis_client.lpush(f"chat_history:{room_id}", json.dumps(chat_data, default=str))
            self.redis_client.expire(f"chat_history:{room_id}", 86400)  # 24 hours
            self.redis_client.ltrim(f"chat_history:{room_id}", 0, 99)  # Keep last 100 messages
        
        # Broadcast chat message
        self._broadcast_event(chat_event)
        return True
    
    def add_annotation(self, user_id: str, annotation_data: Dict[str, Any], room_id: str = "default") -> str:
        """Add annotation to dashboard element"""
        if user_id not in self.active_users:
            return None
        
        user = self.active_users[user_id]
        annotation_id = str(uuid.uuid4())
        
        annotation_event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type="annotation_added",
            user_id=user_id,
            username=user.username,
            timestamp=datetime.now(),
            data={
                "annotation_id": annotation_id,
                "content": annotation_data.get("content", ""),
                "position": annotation_data.get("position", {}),
                "element_id": annotation_data.get("element_id", ""),
                "annotation_type": annotation_data.get("type", "comment")
            },
            room_id=room_id
        )
        
        # Store annotation in Redis
        if self.redis_client:
            annotation_data_full = asdict(annotation_event)
            self.redis_client.hset(f"annotations:{room_id}", annotation_id, json.dumps(annotation_data_full, default=str))
            self.redis_client.expire(f"annotations:{room_id}", 86400)  # 24 hours
        
        # Broadcast annotation
        self._broadcast_event(annotation_event)
        return annotation_id
    
    def sync_filter_changes(self, user_id: str, filter_changes: Dict[str, Any], room_id: str = "default"):
        """Synchronize filter changes across users"""
        if user_id not in self.active_users:
            return
        
        user = self.active_users[user_id]
        
        filter_event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type="filter_changed",
            user_id=user_id,
            username=user.username,
            timestamp=datetime.now(),
            data={
                "filters": filter_changes,
                "sync_id": str(uuid.uuid4())
            },
            room_id=room_id
        )
        
        # Store current filters state
        if self.redis_client:
            self.redis_client.hset(f"room_state:{room_id}", "current_filters", json.dumps(filter_changes))
            self.redis_client.expire(f"room_state:{room_id}", 3600)
        
        # Broadcast filter changes
        self._broadcast_event(filter_event)
    
    def get_active_users(self, room_id: str = "default") -> List[Dict[str, Any]]:
        """Get list of active users in room"""
        if room_id not in self.rooms:
            return []
        
        active_users_list = []
        for user_id in self.rooms[room_id]:
            if user_id in self.active_users:
                user = self.active_users[user_id]
                # Check if user is still active (within last 5 minutes)
                if (datetime.now() - user.last_seen).total_seconds() < 300:
                    active_users_list.append({
                        'user_id': user.user_id,
                        'username': user.username,
                        'role': user.role,
                        'current_page': user.current_page,
                        'last_seen': user.last_seen.isoformat(),
                        'joined_at': user.joined_at.isoformat()
                    })
                else:
                    # Remove inactive users
                    self._remove_inactive_user(user_id, room_id)
        
        return active_users_list
    
    def get_chat_history(self, room_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for room"""
        if not self.redis_client:
            return []
        
        try:
            messages = self.redis_client.lrange(f"chat_history:{room_id}", 0, limit - 1)
            chat_history = []
            
            for message in messages:
                try:
                    chat_data = json.loads(message)
                    chat_history.append(chat_data)
                except json.JSONDecodeError:
                    continue
            
            return list(reversed(chat_history))  # Reverse to get chronological order
            
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return []
    
    def get_annotations(self, room_id: str = "default") -> List[Dict[str, Any]]:
        """Get all annotations for room"""
        if not self.redis_client:
            return []
        
        try:
            annotations_data = self.redis_client.hgetall(f"annotations:{room_id}")
            annotations = []
            
            for annotation_id, annotation_json in annotations_data.items():
                try:
                    annotation_data = json.loads(annotation_json)
                    annotations.append(annotation_data)
                except json.JSONDecodeError:
                    continue
            
            return annotations
            
        except Exception as e:
            print(f"Error getting annotations: {e}")
            return []
    
    def _broadcast_event(self, event: CollaborationEvent):
        """Broadcast event to all users in room"""
        self.event_queue.put(event)
        
        # Also store in Redis for real-time sync
        if self.redis_client:
            event_data = asdict(event)
            self.redis_client.publish(f"collaboration:{event.room_id}", json.dumps(event_data, default=str))
    
    def _process_events(self):
        """Process collaboration events in background thread"""
        while self.is_running:
            try:
                if not self.event_queue.empty():
                    event = self.event_queue.get(timeout=1)
                    self._handle_event(event)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing collaboration event: {e}")
                time.sleep(1)
    
    def _handle_event(self, event: CollaborationEvent):
        """Handle specific collaboration event"""
        # Call registered event handlers
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")
        
        # Update user activity timestamp
        if event.user_id in self.active_users:
            self.active_users[event.user_id].last_seen = datetime.now()
    
    def _remove_inactive_user(self, user_id: str, room_id: str):
        """Remove inactive user from collaboration"""
        if user_id in self.active_users:
            username = self.active_users[user_id].username
            del self.active_users[user_id]
            
            if room_id in self.rooms:
                self.rooms[room_id].discard(user_id)
            
            if self.redis_client:
                self.redis_client.hdel(f"active_users:{room_id}", user_id)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time communication"""
        try:
            async def websocket_handler(websocket, path):
                # Handle WebSocket connections
                user_id = None
                room_id = "default"
                
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        
                        if data.get("type") == "join":
                            user_id = data.get("user_id")
                            room_id = data.get("room_id", "default")
                            self.websocket_connections[user_id] = websocket
                            
                            # Send current state to new user
                            await self._send_current_state(websocket, room_id)
                        
                        elif data.get("type") == "event":
                            # Handle real-time events
                            event_data = data.get("event")
                            if user_id and event_data:
                                event = CollaborationEvent(**event_data)
                                self._broadcast_event(event)
                
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    if user_id:
                        if user_id in self.websocket_connections:
                            del self.websocket_connections[user_id]
                        self.leave_collaboration_room(user_id, room_id)
            
            # Start server on port 8765
            start_server = websockets.serve(websocket_handler, "localhost", 8765)
            await start_server
            
        except Exception as e:
            print(f"WebSocket server error: {e}")
    
    async def _send_current_state(self, websocket, room_id: str):
        """Send current collaboration state to newly connected user"""
        try:
            state = {
                "type": "current_state",
                "active_users": self.get_active_users(room_id),
                "chat_history": self.get_chat_history(room_id, 20),
                "annotations": self.get_annotations(room_id)
            }
            
            await websocket.send(json.dumps(state))
            
        except Exception as e:
            print(f"Error sending current state: {e}")

class StreamlitCollaborationIntegration:
    """Integration class for Streamlit collaboration features"""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.collab_manager = collaboration_manager
        self.current_user_id = None
        self.current_room_id = "default"
    
    def initialize_collaboration(self, user_info: Dict[str, Any], room_id: str = "default"):
        """Initialize collaboration for current user session"""
        self.current_user_id = user_info.get('id', str(uuid.uuid4()))
        self.current_room_id = room_id
        
        # Join collaboration room
        self.collab_manager.join_collaboration_room(
            user_id=self.current_user_id,
            username=user_info.get('username', 'Anonymous'),
            role=user_info.get('role', 'viewer'),
            room_id=room_id
        )
        
        # Store in session state
        st.session_state.collaboration_active = True
        st.session_state.collaboration_user_id = self.current_user_id
        st.session_state.collaboration_room_id = room_id
    
    def render_collaboration_sidebar(self):
        """Render collaboration features in sidebar"""
        if not st.session_state.get('collaboration_active', False):
            return
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("👥 Collaboration")
        
        # Active users
        active_users = self.collab_manager.get_active_users(self.current_room_id)
        
        if active_users:
            st.sidebar.write(f"**Active Users ({len(active_users)}):**")
            for user in active_users:
                status_icon = "🟢" if user['role'] == 'admin' else "🔵" if user['role'] == 'analyst' else "⚪"
                current_page = user.get('current_page', 'Unknown')
                st.sidebar.write(f"{status_icon} {user['username']} ({current_page})")
        else:
            st.sidebar.write("No other users online")
        
        # Chat interface
        with st.sidebar.expander("💬 Chat", expanded=False):
            self._render_chat_interface()
        
        # Annotations
        with st.sidebar.expander("📝 Annotations", expanded=False):
            self._render_annotations_interface()
    
    def _render_chat_interface(self):
        """Render chat interface"""
        # Chat history
        chat_history = self.collab_manager.get_chat_history(self.current_room_id, 10)
        
        if chat_history:
            st.write("**Recent Messages:**")
            for message in chat_history[-5:]:  # Show last 5 messages
                timestamp = datetime.fromisoformat(message['timestamp']).strftime("%H:%M")
                st.write(f"**{message['username']}** ({timestamp}): {message['data']['message']}")
        
        # Send message
        with st.form("chat_form", clear_on_submit=True):
            message_input = st.text_input("Type a message...")
            send_button = st.form_submit_button("Send")
            
            if send_button and message_input:
                self.collab_manager.send_chat_message(
                    user_id=self.current_user_id,
                    message=message_input,
                    room_id=self.current_room_id
                )
                st.rerun()
    
    def _render_annotations_interface(self):
        """Render annotations interface"""
        annotations = self.collab_manager.get_annotations(self.current_room_id)
        
        if annotations:
            st.write("**Active Annotations:**")
            for annotation in annotations[-3:]:  # Show last 3 annotations
                st.write(f"**{annotation['username']}**: {annotation['data']['content']}")
        
        # Add annotation
        with st.form("annotation_form", clear_on_submit=True):
            annotation_content = st.text_area("Add annotation...")
            element_id = st.text_input("Element ID (optional)")
            add_annotation_button = st.form_submit_button("Add Annotation")
            
            if add_annotation_button and annotation_content:
                annotation_data = {
                    "content": annotation_content,
                    "element_id": element_id,
                    "position": {},
                    "type": "comment"
                }
                
                self.collab_manager.add_annotation(
                    user_id=self.current_user_id,
                    annotation_data=annotation_data,
                    room_id=self.current_room_id
                )
                st.rerun()
    
    def sync_filter_state(self, filters: Dict[str, Any]):
        """Synchronize filter state with other users"""
        if st.session_state.get('collaboration_active', False):
            self.collab_manager.sync_filter_changes(
                user_id=self.current_user_id,
                filter_changes=filters,
                room_id=self.current_room_id
            )
    
    def update_page_activity(self, page_name: str):
        """Update current page activity"""
        if st.session_state.get('collaboration_active', False):
            self.collab_manager.update_user_activity(
                user_id=self.current_user_id,
                activity_data={"current_page": page_name},
                room_id=self.current_room_id
            )
    
    def cleanup_collaboration(self):
        """Clean up collaboration when session ends"""
        if st.session_state.get('collaboration_active', False):
            self.collab_manager.leave_collaboration_room(
                user_id=self.current_user_id,
                room_id=self.current_room_id
            )

# JavaScript code for client-side WebSocket integration
WEBSOCKET_CLIENT_JS = """
<script>
let ws = null;
let userId = null;
let roomId = 'default';

function initCollaboration(userIdParam, roomIdParam) {
    userId = userIdParam;
    roomId = roomIdParam;
    
    try {
        ws = new WebSocket('ws://localhost:8765');
        
        ws.onopen = function(event) {
            console.log('WebSocket connected');
            // Join collaboration room
            ws.send(JSON.stringify({
                type: 'join',
                user_id: userId,
                room_id: roomId
            }));
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleCollaborationEvent(data);
        };
        
        ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => initCollaboration(userId, roomId), 5000);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
    }
}

function handleCollaborationEvent(data) {
    if (data.type === 'current_state') {
        // Handle initial state
        updateActiveUsers(data.active_users);
        updateChatHistory(data.chat_history);
        displayAnnotations(data.annotations);
    } else if (data.type === 'event') {
        // Handle real-time events
        const event = data.event;
        
        switch (event.event_type) {
            case 'user_joined':
            case 'user_left':
                refreshActiveUsers();
                break;
            case 'chat_message':
                addChatMessage(event);
                break;
            case 'filter_changed':
                syncFilters(event.data.filters);
                break;
            case 'annotation_added':
                addAnnotation(event.data);
                break;
        }
    }
}

function sendCollaborationEvent(eventType, data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'event',
            event: {
                event_type: eventType,
                user_id: userId,
                data: data,
                room_id: roomId,
                timestamp: new Date().toISOString()
            }
        }));
    }
}

function updateActiveUsers(users) {
    // Update active users display
    const usersList = document.getElementById('active-users-list');
    if (usersList) {
        usersList.innerHTML = users.map(user => 
            `<div class="active-user">
                <span class="user-status">🟢</span>
                <span class="username">${user.username}</span>
                <span class="user-page">(${user.current_page})</span>
            </div>`
        ).join('');
    }
}

function addChatMessage(event) {
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message';
        messageDiv.innerHTML = `
            <div class="message-header">
                <strong>${event.username}</strong>
                <span class="timestamp">${new Date(event.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${event.data.message}</div>
        `;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

function syncFilters(filters) {
    // Sync filter changes from other users
    console.log('Syncing filters:', filters);
    
    // Update Streamlit components if possible
    // This would require custom Streamlit components or server-side handling
}

function addAnnotation(annotationData) {
    // Add visual annotation to the interface
    const elementId = annotationData.element_id;
    if (elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            const annotation = document.createElement('div');
            annotation.className = 'collaboration-annotation';
            annotation.innerHTML = `
                <div class="annotation-popup">
                    <div class="annotation-author">${annotationData.username}</div>
                    <div class="annotation-content">${annotationData.content}</div>
                </div>
            `;
            element.appendChild(annotation);
        }
    }
}

// Initialize collaboration when page loads
document.addEventListener('DOMContentLoaded', function() {
    // This would be called from Streamlit with actual user/room data
    // initCollaboration('user123', 'default');
});
</script>

<style>
.active-user {
    display: flex;
    align-items: center;
    padding: 2px 0;
    font-size: 0.9em;
}

.user-status {
    margin-right: 5px;
}

.username {
    font-weight: bold;
    margin-right: 5px;
}

.user-page {
    color: #666;
    font-size: 0.8em;
}

.chat-message {
    margin-bottom: 10px;
    padding: 5px;
    border-radius: 5px;
    background-color: #f0f0f0;
}

.message-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}

.timestamp {
    color: #666;
    font-size: 0.8em;
}

.message-content {
    font-size: 0.9em;
}

.collaboration-annotation {
    position: relative;
    display: inline-block;
}

.annotation-popup {
    position: absolute;
    background: #fff;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    z-index: 1000;
    min-width: 200px;
}

.annotation-author {
    font-weight: bold;
    color: #333;
    margin-bottom: 5px;
}

.annotation-content {
    color: #666;
}
</style>
"""

# Integration function for main application
def initialize_collaboration_features():
    """Initialize collaboration features for the main application"""
    if 'collaboration_manager' not in st.session_state:
        st.session_state.collaboration_manager = CollaborationManager()
        st.session_state.collaboration_manager.start_collaboration_service()
    
    if 'collaboration_integration' not in st.session_state:
        st.session_state.collaboration_integration = StreamlitCollaborationIntegration(
            st.session_state.collaboration_manager
        )
    
    return st.session_state.collaboration_integration

def render_collaboration_features(user_info: Dict[str, Any], room_id: str = "default"):
    """Render collaboration features in the main application"""
    collab_integration = initialize_collaboration_features()
    
    # Initialize collaboration for current user
    if not st.session_state.get('collaboration_active', False):
        collab_integration.initialize_collaboration(user_info, room_id)
    
    # Render collaboration sidebar
    collab_integration.render_collaboration_sidebar()
    
    # Add WebSocket client code
    st.markdown(WEBSOCKET_CLIENT_JS, unsafe_allow_html=True)
    
    return collab_integration

if __name__ == "__main__":
    # Example usage and testing
    import time
    
    # Initialize collaboration manager
    collab_manager = CollaborationManager()
    collab_manager.start_collaboration_service()
    
    # Simulate users joining
    user1_id = "user1"
    user2_id = "user2"
    
    collab_manager.join_collaboration_room(user1_id, "Alice", "admin")
    collab_manager.join_collaboration_room(user2_id, "Bob", "analyst")
    
    # Simulate chat messages
    collab_manager.send_chat_message(user1_id, "Hello everyone!")
    collab_manager.send_chat_message(user2_id, "Hi Alice! Ready to analyze some risks?")
    
    # Simulate filter sync
    collab_manager.sync_filter_changes(user1_id, {
        "models": ["GPT-4", "Claude"],
        "date_range": ["2025-01-01", "2025-01-31"]
    })
    
    # Check active users
    active_users = collab_manager.get_active_users()
    print(f"Active users: {len(active_users)}")
    for user in active_users:
        print(f"  - {user['username']} ({user['role']})")
    
    # Check chat history
    chat_history = collab_manager.get_chat_history()
    print(f"Chat messages: {len(chat_history)}")
    for message in chat_history:
        print(f"  - {message['username']}: {message['data']['message']}")
    
    # Keep running for a bit
    time.sleep(2)
    
    # Clean up
    collab_manager.leave_collaboration_room(user1_id)
    collab_manager.leave_collaboration_room(user2_id)
    collab_manager.stop_collaboration_service()
    
    print("Collaboration test completed")