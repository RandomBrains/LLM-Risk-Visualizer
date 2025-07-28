"""
Augmented Reality (AR) and Virtual Reality (VR) Visualization Module
Provides immersive data visualization and interactive 3D risk analysis environments
"""

import json
import time
import uuid
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
import asyncio

# 3D and AR/VR libraries
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available. Some 3D features will be limited.")

try:
    import trimesh
    import pyrender
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False
    logging.warning("Trimesh/PyRender not available. Advanced rendering features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of AR/VR visualizations"""
    RISK_LANDSCAPE_3D = "risk_landscape_3d"
    DATA_CONSTELLATION = "data_constellation"
    IMMERSIVE_DASHBOARD = "immersive_dashboard"
    COLLABORATIVE_SPACE = "collaborative_space"
    RISK_SIMULATION = "risk_simulation"
    NETWORK_TOPOLOGY = "network_topology"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    INTERACTIVE_EXPLORATION = "interactive_exploration"

class DeviceType(Enum):
    """AR/VR device types"""
    WEB_AR = "web_ar"
    WEB_VR = "web_vr"
    OCULUS_QUEST = "oculus_quest"
    HOLOLENS = "hololens"
    MAGIC_LEAP = "magic_leap"
    GENERIC_VR = "generic_vr"
    MOBILE_AR = "mobile_ar"

class InteractionMode(Enum):
    """Interaction modes in AR/VR"""
    GAZE_CONTROL = "gaze_control"
    HAND_TRACKING = "hand_tracking"
    CONTROLLER_INPUT = "controller_input"
    VOICE_COMMANDS = "voice_commands"
    GESTURE_RECOGNITION = "gesture_recognition"
    TOUCH_INTERFACE = "touch_interface"

class RenderingQuality(Enum):
    """Rendering quality levels"""
    LOW = "low"          # Mobile/web optimization
    MEDIUM = "medium"    # Standard quality
    HIGH = "high"        # Desktop VR quality
    ULTRA = "ultra"      # High-end VR systems

@dataclass
class VRScene:
    """Virtual Reality scene configuration"""
    scene_id: str
    name: str
    visualization_type: VisualizationType
    created_date: datetime
    
    # Scene properties
    environment_type: str = "void"  # void, office, laboratory, outdoor
    lighting_config: Dict[str, Any] = None
    camera_position: Tuple[float, float, float] = (0, 0, 5)
    camera_target: Tuple[float, float, float] = (0, 0, 0)
    
    # Data binding
    data_sources: List[str] = None
    data_filters: Dict[str, Any] = None
    update_frequency: int = 30  # seconds
    
    # Interaction settings
    supported_interactions: List[InteractionMode] = None
    navigation_enabled: bool = True
    collaboration_enabled: bool = False
    
    # Performance settings
    max_objects: int = 1000
    level_of_detail: bool = True
    occlusion_culling: bool = True
    
    def __post_init__(self):
        if self.lighting_config is None:
            self.lighting_config = {"ambient": 0.4, "directional": 0.8}
        if self.data_sources is None:
            self.data_sources = []
        if self.data_filters is None:
            self.data_filters = {}
        if self.supported_interactions is None:
            self.supported_interactions = [InteractionMode.GAZE_CONTROL, InteractionMode.CONTROLLER_INPUT]

@dataclass
class AROverlay:
    """Augmented Reality overlay configuration"""
    overlay_id: str
    name: str
    overlay_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1, 1, 1)
    
    # Content properties
    content_data: Dict[str, Any] = None
    anchor_type: str = "world"  # world, image, plane, object
    tracking_enabled: bool = True
    
    # Visual properties
    opacity: float = 1.0
    visible: bool = True
    occlusion_enabled: bool = True
    
    # Interaction
    interactive: bool = True
    click_action: Optional[str] = None
    hover_effects: bool = True
    
    def __post_init__(self):
        if self.content_data is None:
            self.content_data = {}

@dataclass
class ImmersiveVisualization:
    """Immersive visualization object"""
    viz_id: str
    name: str
    viz_type: VisualizationType
    created_date: datetime
    
    # Data properties
    data_source: str
    data_query: str
    data_fields: List[str]
    
    # 3D properties
    geometry_type: str = "sphere"  # sphere, cube, cylinder, mesh, point_cloud
    position: Tuple[float, float, float] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1, 1, 1)
    color_mapping: Dict[str, Any] = None
    
    # Animation properties
    animated: bool = False
    animation_type: str = "none"  # none, rotate, pulse, flow, morph
    animation_duration: float = 2.0
    
    # Interaction properties
    selectable: bool = True
    detail_panel_enabled: bool = True
    context_menu_enabled: bool = True
    
    # Performance properties
    level_of_detail_enabled: bool = True
    culling_enabled: bool = True
    batching_enabled: bool = True
    
    def __post_init__(self):
        if self.color_mapping is None:
            self.color_mapping = {"field": "risk_level", "scheme": "red_yellow_green"}

@dataclass
class UserSession:
    """AR/VR user session"""
    session_id: str
    user_id: str
    device_type: DeviceType
    start_time: datetime
    
    # Session properties
    current_scene: Optional[str] = None
    interaction_mode: InteractionMode = InteractionMode.CONTROLLER_INPUT
    rendering_quality: RenderingQuality = RenderingQuality.MEDIUM
    
    # User preferences
    comfort_settings: Dict[str, Any] = None
    accessibility_options: Dict[str, Any] = None
    
    # Performance tracking
    frame_rate: float = 0.0
    latency_ms: float = 0.0
    motion_sickness_level: int = 0  # 0-10 scale
    
    # Collaboration
    shared_session: bool = False
    collaborators: List[str] = None
    
    def __post_init__(self):
        if self.comfort_settings is None:
            self.comfort_settings = {
                "vr_comfort_mode": True,
                "motion_reduction": True,
                "snap_turning": True,
                "teleport_movement": True
            }
        if self.accessibility_options is None:
            self.accessibility_options = {
                "text_size_multiplier": 1.0,
                "high_contrast": False,
                "audio_descriptions": False,
                "haptic_feedback": True
            }
        if self.collaborators is None:
            self.collaborators = []

class ARVRRenderer:
    """Handles AR/VR rendering and scene generation"""
    
    def __init__(self):
        self.scenes: Dict[str, VRScene] = {}
        self.overlays: Dict[str, AROverlay] = {}
        self.visualizations: Dict[str, ImmersiveVisualization] = {}
        self.render_cache = {}
        
    def create_3d_risk_landscape(self, risk_data: pd.DataFrame, 
                               scene_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create 3D risk landscape visualization"""
        
        if scene_config is None:
            scene_config = {}
        
        # Generate heightmap from risk data
        x_coords = []
        y_coords = []
        z_values = []
        colors = []
        
        for idx, row in risk_data.iterrows():
            # Position based on risk category and severity
            x = row.get('risk_category_id', idx % 10)
            y = row.get('temporal_factor', idx // 10)
            z = row.get('risk_score', 0.5) * 10  # Scale risk score to height
            
            x_coords.append(x)
            y_coords.append(y)
            z_values.append(z)
            
            # Color mapping based on risk level
            risk_level = row.get('risk_level', 'medium')
            color_map = {
                'low': '#00ff00',
                'medium': '#ffff00', 
                'high': '#ff8000',
                'critical': '#ff0000'
            }
            colors.append(color_map.get(risk_level, '#888888'))
        
        # Create mesh data
        mesh_data = {
            "vertices": [
                [x_coords[i], y_coords[i], z_values[i]] 
                for i in range(len(x_coords))
            ],
            "colors": colors,
            "metadata": [
                {
                    "risk_id": row.get('risk_id', f'risk_{idx}'),
                    "description": row.get('description', 'Unknown risk'),
                    "risk_score": row.get('risk_score', 0),
                    "impact": row.get('impact', 'Unknown')
                }
                for idx, row in risk_data.iterrows()
            ]
        }
        
        # Generate WebGL-compatible scene
        webgl_scene = self._generate_webgl_scene(mesh_data, scene_config)
        
        return {
            "scene_type": "risk_landscape_3d",
            "mesh_data": mesh_data,
            "webgl_scene": webgl_scene,
            "interaction_points": self._generate_interaction_points(mesh_data),
            "metadata": {
                "total_risks": len(risk_data),
                "max_risk_score": max(z_values) if z_values else 0,
                "risk_distribution": dict(Counter([row.get('risk_level', 'unknown') for _, row in risk_data.iterrows()]))
            }
        }
    
    def create_data_constellation(self, entities: List[Dict[str, Any]], 
                                relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create constellation-style data visualization"""
        
        # Position entities in 3D space using force-directed layout
        positions = self._calculate_force_directed_layout(entities, relationships)
        
        # Create nodes
        nodes = []
        for i, entity in enumerate(entities):
            pos = positions[i]
            node = {
                "id": entity.get('id', f'node_{i}'),
                "position": pos,
                "size": entity.get('importance', 1.0) * 2,
                "color": self._get_entity_color(entity.get('type', 'default')),
                "metadata": entity
            }
            nodes.append(node)
        
        # Create connections
        connections = []
        for rel in relationships:
            source_id = rel.get('source')
            target_id = rel.get('target')
            
            # Find positions
            source_pos = None
            target_pos = None
            
            for node in nodes:
                if node['id'] == source_id:
                    source_pos = node['position']
                elif node['id'] == target_id:
                    target_pos = node['position']
            
            if source_pos and target_pos:
                connection = {
                    "source": source_pos,
                    "target": target_pos,
                    "strength": rel.get('strength', 0.5),
                    "type": rel.get('type', 'default'),
                    "animated": rel.get('animated', False)
                }
                connections.append(connection)
        
        return {
            "scene_type": "data_constellation",
            "nodes": nodes,
            "connections": connections,
            "bounding_sphere": self._calculate_bounding_sphere(positions),
            "interaction_zones": self._create_interaction_zones(nodes)
        }
    
    def create_immersive_dashboard(self, dashboard_config: Dict[str, Any],
                                 data_panels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create immersive dashboard environment"""
        
        # Create curved dashboard layout
        panel_positions = self._calculate_curved_layout(len(data_panels))
        
        dashboard_panels = []
        for i, panel in enumerate(data_panels):
            position = panel_positions[i]
            
            # Convert 2D charts to 3D representations
            chart_3d = self._convert_chart_to_3d(panel)
            
            dashboard_panel = {
                "panel_id": panel.get('id', f'panel_{i}'),
                "position": position,
                "rotation": self._calculate_panel_rotation(position),
                "size": panel.get('size', [2, 1.5]),
                "content": chart_3d,
                "interactive": True,
                "resizable": panel.get('resizable', True)
            }
            dashboard_panels.append(dashboard_panel)
        
        # Add environment elements
        environment = {
            "floor": {"enabled": True, "material": "grid"},
            "ceiling": {"enabled": False},
            "walls": {"enabled": False},
            "ambient_lighting": 0.6,
            "accent_lighting": [
                {"position": [0, 3, 0], "intensity": 0.8, "color": [1, 1, 1]}
            ]
        }
        
        return {
            "scene_type": "immersive_dashboard",
            "panels": dashboard_panels,
            "environment": environment,
            "navigation": {
                "center_point": [0, 1.5, 0],
                "interaction_radius": 5.0,
                "teleport_points": self._generate_teleport_points(dashboard_panels)
            }
        }
    
    def _generate_webgl_scene(self, mesh_data: Dict[str, Any], 
                            config: Dict[str, Any]) -> str:
        """Generate WebGL scene code"""
        
        webgl_template = """
        // WebGL Scene for AR/VR Risk Visualization
        class RiskLandscapeScene {
            constructor(canvas) {
                this.canvas = canvas;
                this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                this.meshData = %s;
                this.init();
            }
            
            init() {
                const gl = this.gl;
                
                // Vertex shader
                const vsSource = `
                    attribute vec4 aVertexPosition;
                    attribute vec4 aVertexColor;
                    uniform mat4 uModelViewMatrix;
                    uniform mat4 uProjectionMatrix;
                    varying lowp vec4 vColor;
                    
                    void main() {
                        gl_Position = uProjectionMatrix * uModelViewMatrix * aVertexPosition;
                        vColor = aVertexColor;
                    }
                `;
                
                // Fragment shader
                const fsSource = `
                    varying lowp vec4 vColor;
                    void main() {
                        gl_FragColor = vColor;
                    }
                `;
                
                this.shaderProgram = this.initShaderProgram(gl, vsSource, fsSource);
                this.buffers = this.initBuffers(gl);
                this.drawScene();
            }
            
            initBuffers(gl) {
                const positionBuffer = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
                
                const positions = this.meshData.vertices.flat();
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
                
                return { position: positionBuffer };
            }
            
            drawScene() {
                const gl = this.gl;
                gl.clearColor(0.0, 0.0, 0.0, 1.0);
                gl.clearDepth(1.0);
                gl.enable(gl.DEPTH_TEST);
                gl.depthFunc(gl.LEQUAL);
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                
                // Render mesh points
                gl.useProgram(this.shaderProgram);
                gl.drawArrays(gl.POINTS, 0, this.meshData.vertices.length);
            }
            
            // Additional methods for shader initialization...
        }
        """ % json.dumps(mesh_data)
        
        return webgl_template
    
    def _calculate_force_directed_layout(self, entities: List[Dict[str, Any]], 
                                       relationships: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
        """Calculate 3D positions using force-directed algorithm"""
        
        n = len(entities)
        if n == 0:
            return []
        
        # Initialize random positions
        positions = []
        for i in range(n):
            pos = [
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5), 
                np.random.uniform(-5, 5)
            ]
            positions.append(pos)
        
        # Build adjacency for relationships
        adjacency = defaultdict(list)
        for rel in relationships:
            source_idx = None
            target_idx = None
            
            for i, entity in enumerate(entities):
                if entity.get('id') == rel.get('source'):
                    source_idx = i
                elif entity.get('id') == rel.get('target'):
                    target_idx = i
            
            if source_idx is not None and target_idx is not None:
                adjacency[source_idx].append(target_idx)
                adjacency[target_idx].append(source_idx)
        
        # Simulate force-directed layout
        for iteration in range(100):
            forces = [[0, 0, 0] for _ in range(n)]
            
            # Repulsive forces between all nodes
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]  
                    dz = positions[i][2] - positions[j][2]
                    
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
                    force = 1.0 / (distance * distance)
                    
                    forces[i][0] += force * dx / distance
                    forces[i][1] += force * dy / distance
                    forces[i][2] += force * dz / distance
                    
                    forces[j][0] -= force * dx / distance
                    forces[j][1] -= force * dy / distance
                    forces[j][2] -= force * dz / distance
            
            # Attractive forces for connected nodes
            for i, neighbors in adjacency.items():
                for j in neighbors:
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dz = positions[j][2] - positions[i][2]
                    
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
                    force = distance * 0.1
                    
                    forces[i][0] += force * dx / distance
                    forces[i][1] += force * dy / distance
                    forces[i][2] += force * dz / distance
            
            # Apply forces with damping
            damping = 0.9
            for i in range(n):
                positions[i][0] += forces[i][0] * damping
                positions[i][1] += forces[i][1] * damping
                positions[i][2] += forces[i][2] * damping
        
        return [tuple(pos) for pos in positions]
    
    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity type"""
        color_map = {
            'risk': '#ff4444',
            'asset': '#4444ff',
            'user': '#44ff44',
            'system': '#ffff44',
            'process': '#ff44ff',
            'default': '#888888'
        }
        return color_map.get(entity_type, color_map['default'])
    
    def _calculate_curved_layout(self, panel_count: int) -> List[Tuple[float, float, float]]:
        """Calculate curved layout positions for dashboard panels"""
        
        positions = []
        radius = 3.0
        height = 1.5
        
        for i in range(panel_count):
            angle = (i / panel_count) * 2 * math.pi - math.pi
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = height + (i % 2) * 0.5  # Stagger heights slightly
            
            positions.append((x, y, z))
        
        return positions
    
    def _convert_chart_to_3d(self, panel: Dict[str, Any]) -> Dict[str, Any]:
        """Convert 2D chart to 3D representation"""
        
        chart_type = panel.get('chart_type', 'bar')
        data = panel.get('data', [])
        
        if chart_type == 'bar':
            # Convert to 3D bar chart
            bars = []
            for i, item in enumerate(data):
                bar = {
                    "position": [i * 0.5 - len(data) * 0.25, 0, 0],
                    "height": item.get('value', 1.0),
                    "color": item.get('color', '#4444ff'),
                    "label": item.get('label', f'Item {i}')
                }
                bars.append(bar)
            
            return {"type": "3d_bars", "data": bars}
        
        elif chart_type == 'line':
            # Convert to 3D line/ribbon
            points = []
            for i, item in enumerate(data):
                point = [
                    i * 0.1,
                    item.get('value', 0.5),
                    0
                ]
                points.append(point)
            
            return {"type": "3d_line", "points": points}
        
        elif chart_type == 'pie':
            # Convert to 3D pie segments
            segments = []
            total = sum(item.get('value', 0) for item in data)
            current_angle = 0
            
            for item in data:
                angle_span = (item.get('value', 0) / total) * 2 * math.pi
                segment = {
                    "start_angle": current_angle,
                    "end_angle": current_angle + angle_span,
                    "color": item.get('color', '#4444ff'),
                    "label": item.get('label', 'Unknown')
                }
                segments.append(segment)
                current_angle += angle_span
            
            return {"type": "3d_pie", "segments": segments}
        
        # Default to simple text display
        return {"type": "3d_text", "content": str(panel.get('data', 'No data'))}
    
    def _generate_interaction_points(self, mesh_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate interaction points for VR scene"""
        
        interaction_points = []
        vertices = mesh_data.get('vertices', [])
        metadata = mesh_data.get('metadata', [])
        
        for i, vertex in enumerate(vertices):
            if i < len(metadata):
                point = {
                    "id": f"interaction_{i}",
                    "position": vertex,
                    "radius": 0.3,
                    "action": "show_details",
                    "data": metadata[i]
                }
                interaction_points.append(point)
        
        return interaction_points
    
    def _calculate_bounding_sphere(self, positions: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Calculate bounding sphere for positions"""
        
        if not positions:
            return {"center": [0, 0, 0], "radius": 1.0}
        
        # Calculate center
        center = [
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions),
            sum(pos[2] for pos in positions) / len(positions)
        ]
        
        # Calculate radius
        max_distance = 0
        for pos in positions:
            distance = math.sqrt(
                (pos[0] - center[0])**2 + 
                (pos[1] - center[1])**2 + 
                (pos[2] - center[2])**2
            )
            max_distance = max(max_distance, distance)
        
        return {"center": center, "radius": max_distance + 1.0}
    
    def _create_interaction_zones(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create interaction zones around nodes"""
        
        zones = []
        for node in nodes:
            zone = {
                "node_id": node['id'],
                "position": node['position'],
                "radius": node['size'] * 1.5,
                "action": "node_interaction",
                "hover_effect": True
            }
            zones.append(zone)
        
        return zones
    
    def _calculate_panel_rotation(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate panel rotation to face center"""
        
        x, y, z = position
        
        # Calculate rotation to face origin
        yaw = math.atan2(x, z)
        pitch = 0  # Keep panels upright
        roll = 0
        
        return (pitch, yaw, roll)
    
    def _generate_teleport_points(self, panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate teleport points for navigation"""
        
        teleport_points = []
        
        # Center point
        teleport_points.append({
            "id": "center",
            "position": [0, 0, 0],
            "rotation": [0, 0, 0],
            "description": "Center view"
        })
        
        # Points near each panel
        for i, panel in enumerate(panels):
            pos = panel['position']
            # Move slightly toward center
            offset_pos = [pos[0] * 0.7, pos[1], pos[2] * 0.7]
            
            teleport_points.append({
                "id": f"panel_{i}",
                "position": offset_pos,
                "rotation": panel['rotation'],
                "description": f"View {panel.get('panel_id', f'Panel {i}')}"
            })
        
        return teleport_points

class ARVRSessionManager:
    """Manages AR/VR user sessions and device interactions"""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self.renderer = ARVRRenderer()
        self.websocket_server = None
        
    def create_session(self, user_id: str, device_type: DeviceType) -> str:
        """Create new AR/VR session"""
        
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            device_type=device_type,
            start_time=datetime.now()
        )
        
        # Adjust settings based on device type
        if device_type in [DeviceType.MOBILE_AR, DeviceType.WEB_AR]:
            session.rendering_quality = RenderingQuality.MEDIUM
            session.comfort_settings['motion_reduction'] = True
        elif device_type in [DeviceType.OCULUS_QUEST, DeviceType.GENERIC_VR]:
            session.rendering_quality = RenderingQuality.HIGH
            session.interaction_mode = InteractionMode.CONTROLLER_INPUT
        elif device_type == DeviceType.HOLOLENS:
            session.interaction_mode = InteractionMode.GAZE_CONTROL
            session.rendering_quality = RenderingQuality.MEDIUM
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Created AR/VR session {session_id} for user {user_id} on {device_type.value}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        return self.active_sessions.get(session_id)
    
    def update_session_metrics(self, session_id: str, metrics: Dict[str, Any]) -> bool:
        """Update session performance metrics"""
        
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.frame_rate = metrics.get('frame_rate', session.frame_rate)
        session.latency_ms = metrics.get('latency_ms', session.latency_ms)
        session.motion_sickness_level = metrics.get('motion_sickness', session.motion_sickness_level)
        
        # Adjust quality if performance is poor
        if session.frame_rate < 30 and session.rendering_quality != RenderingQuality.LOW:
            session.rendering_quality = RenderingQuality.MEDIUM
            logger.info(f"Reduced rendering quality for session {session_id} due to low frame rate")
        
        return True
    
    def generate_scene_for_session(self, session_id: str, 
                                 visualization_type: VisualizationType,
                                 data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate appropriate scene for session"""
        
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Generate scene based on visualization type
        if visualization_type == VisualizationType.RISK_LANDSCAPE_3D:
            risk_data = pd.DataFrame(data.get('risks', []))
            scene = self.renderer.create_3d_risk_landscape(risk_data)
        
        elif visualization_type == VisualizationType.DATA_CONSTELLATION:
            entities = data.get('entities', [])
            relationships = data.get('relationships', [])
            scene = self.renderer.create_data_constellation(entities, relationships)
        
        elif visualization_type == VisualizationType.IMMERSIVE_DASHBOARD:
            dashboard_config = data.get('config', {})
            panels = data.get('panels', [])
            scene = self.renderer.create_immersive_dashboard(dashboard_config, panels)
        
        else:
            logger.warning(f"Unsupported visualization type: {visualization_type}")
            return None
        
        # Optimize scene for device type
        scene = self._optimize_scene_for_device(scene, session.device_type, session.rendering_quality)
        
        return scene
    
    def _optimize_scene_for_device(self, scene: Dict[str, Any], 
                                 device_type: DeviceType,
                                 quality: RenderingQuality) -> Dict[str, Any]:
        """Optimize scene for specific device and quality level"""
        
        optimized_scene = scene.copy()
        
        # Reduce object count for mobile devices
        if device_type in [DeviceType.MOBILE_AR, DeviceType.WEB_AR, DeviceType.WEB_VR]:
            if 'nodes' in optimized_scene:
                max_nodes = 100 if quality == RenderingQuality.LOW else 250
                if len(optimized_scene['nodes']) > max_nodes:
                    optimized_scene['nodes'] = optimized_scene['nodes'][:max_nodes]
        
        # Adjust rendering quality
        if quality == RenderingQuality.LOW:
            # Reduce mesh complexity
            if 'mesh_data' in optimized_scene:
                vertices = optimized_scene['mesh_data']['vertices']
                if len(vertices) > 500:
                    # Subsample vertices
                    step = len(vertices) // 500
                    optimized_scene['mesh_data']['vertices'] = vertices[::step]
        
        # Add device-specific optimizations
        if device_type == DeviceType.HOLOLENS:
            # Enable spatial anchoring
            optimized_scene['spatial_anchoring'] = True
            optimized_scene['occlusion_enabled'] = True
        
        elif device_type in [DeviceType.OCULUS_QUEST, DeviceType.GENERIC_VR]:
            # Enable hand tracking if available
            optimized_scene['hand_tracking_enabled'] = True
            optimized_scene['haptic_feedback_enabled'] = True
        
        return optimized_scene
    
    def end_session(self, session_id: str) -> bool:
        """End AR/VR session"""
        
        session = self.active_sessions.pop(session_id, None)
        if session:
            duration = datetime.now() - session.start_time
            logger.info(f"Ended AR/VR session {session_id} after {duration}")
            return True
        return False

class ImmersiveAnalyticsEngine:
    """Provides immersive analytics capabilities"""
    
    def __init__(self, db_path: str = "ar_vr_analytics.db"):
        self.db_path = Path(db_path)
        self.session_manager = ARVRSessionManager()
        self.analytics_cache = {}
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # VR scenes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vr_scenes (
                scene_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                visualization_type TEXT NOT NULL,
                created_date TEXT NOT NULL,
                environment_type TEXT DEFAULT 'void',
                lighting_config TEXT,
                camera_position TEXT,
                camera_target TEXT,
                data_sources TEXT,
                data_filters TEXT,
                update_frequency INTEGER DEFAULT 30,
                supported_interactions TEXT,
                navigation_enabled BOOLEAN DEFAULT 1,
                collaboration_enabled BOOLEAN DEFAULT 0,
                max_objects INTEGER DEFAULT 1000,
                level_of_detail BOOLEAN DEFAULT 1,
                occlusion_culling BOOLEAN DEFAULT 1
            )
        ''')
        
        # AR overlays table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ar_overlays (
                overlay_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                overlay_type TEXT NOT NULL,
                position TEXT NOT NULL,
                rotation TEXT DEFAULT '[0,0,0]',
                scale TEXT DEFAULT '[1,1,1]',
                content_data TEXT,
                anchor_type TEXT DEFAULT 'world',
                tracking_enabled BOOLEAN DEFAULT 1,
                opacity REAL DEFAULT 1.0,
                visible BOOLEAN DEFAULT 1,
                occlusion_enabled BOOLEAN DEFAULT 1,
                interactive BOOLEAN DEFAULT 1,
                click_action TEXT,
                hover_effects BOOLEAN DEFAULT 1
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                current_scene TEXT,
                interaction_mode TEXT,
                rendering_quality TEXT,
                comfort_settings TEXT,
                accessibility_options TEXT,
                frame_rate REAL DEFAULT 0.0,
                latency_ms REAL DEFAULT 0.0,
                motion_sickness_level INTEGER DEFAULT 0,
                shared_session BOOLEAN DEFAULT 0,
                collaborators TEXT
            )
        ''')
        
        # Visualizations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS immersive_visualizations (
                viz_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                viz_type TEXT NOT NULL,
                created_date TEXT NOT NULL,
                data_source TEXT NOT NULL,
                data_query TEXT NOT NULL,
                data_fields TEXT NOT NULL,
                geometry_type TEXT DEFAULT 'sphere',
                position TEXT DEFAULT '[0,0,0]',
                scale TEXT DEFAULT '[1,1,1]',
                color_mapping TEXT,
                animated BOOLEAN DEFAULT 0,
                animation_type TEXT DEFAULT 'none',
                animation_duration REAL DEFAULT 2.0,
                selectable BOOLEAN DEFAULT 1,
                detail_panel_enabled BOOLEAN DEFAULT 1,
                context_menu_enabled BOOLEAN DEFAULT 1,
                level_of_detail_enabled BOOLEAN DEFAULT 1,
                culling_enabled BOOLEAN DEFAULT 1,
                batching_enabled BOOLEAN DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_risk_landscape_visualization(self, risk_data: List[Dict[str, Any]],
                                          user_id: str, device_type: DeviceType) -> str:
        """Create immersive risk landscape visualization"""
        
        # Create user session
        session_id = self.session_manager.create_session(user_id, device_type)
        
        # Prepare data
        data = {"risks": risk_data}
        
        # Generate scene
        scene = self.session_manager.generate_scene_for_session(
            session_id, VisualizationType.RISK_LANDSCAPE_3D, data
        )
        
        if not scene:
            return None
        
        # Store visualization
        viz_id = str(uuid.uuid4())
        visualization = ImmersiveVisualization(
            viz_id=viz_id,
            name="Risk Landscape 3D",
            viz_type=VisualizationType.RISK_LANDSCAPE_3D,
            created_date=datetime.now(),
            data_source="risk_data",
            data_query="SELECT * FROM risks",
            data_fields=["risk_id", "risk_score", "risk_level", "category"]
        )
        
        self._save_visualization(visualization)
        
        return {
            "session_id": session_id,
            "visualization_id": viz_id,
            "scene": scene,
            "device_capabilities": self._get_device_capabilities(device_type)
        }
    
    def create_collaborative_space(self, participants: List[str],
                                 shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create collaborative AR/VR space"""
        
        # Create shared session
        main_session_id = str(uuid.uuid4())
        
        collaborative_sessions = {}
        for participant in participants:
            session_id = self.session_manager.create_session(participant, DeviceType.GENERIC_VR)
            session = self.session_manager.get_session(session_id)
            if session:
                session.shared_session = True
                session.collaborators = [p for p in participants if p != participant]
            collaborative_sessions[participant] = session_id
        
        # Create shared visualization space
        entities = shared_data.get('entities', [])
        relationships = shared_data.get('relationships', [])
        
        scene = self.session_manager.renderer.create_data_constellation(entities, relationships)
        
        # Add collaboration features
        scene['collaboration'] = {
            "voice_chat_enabled": True,
            "shared_cursors": True,
            "annotation_system": True,
            "screen_sharing": True,
            "synchronized_navigation": True
        }
        
        # Add avatar system
        scene['avatars'] = []
        for i, participant in enumerate(participants):
            avatar = {
                "user_id": participant,
                "position": [i * 2.0 - len(participants), 0, -3],
                "color": self._get_user_color(i),
                "name_tag": f"User_{participant[:8]}"
            }
            scene['avatars'].append(avatar)
        
        return {
            "main_session_id": main_session_id,
            "participant_sessions": collaborative_sessions,
            "scene": scene,
            "collaboration_features": scene['collaboration']
        }
    
    def _save_visualization(self, visualization: ImmersiveVisualization):
        """Save visualization to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO immersive_visualizations
                (viz_id, name, viz_type, created_date, data_source, data_query,
                 data_fields, geometry_type, position, scale, color_mapping,
                 animated, animation_type, animation_duration, selectable,
                 detail_panel_enabled, context_menu_enabled, level_of_detail_enabled,
                 culling_enabled, batching_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                visualization.viz_id, visualization.name, visualization.viz_type.value,
                visualization.created_date.isoformat(), visualization.data_source,
                visualization.data_query, json.dumps(visualization.data_fields),
                visualization.geometry_type, json.dumps(visualization.position),
                json.dumps(visualization.scale), json.dumps(visualization.color_mapping),
                visualization.animated, visualization.animation_type,
                visualization.animation_duration, visualization.selectable,
                visualization.detail_panel_enabled, visualization.context_menu_enabled,
                visualization.level_of_detail_enabled, visualization.culling_enabled,
                visualization.batching_enabled
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
    
    def _get_device_capabilities(self, device_type: DeviceType) -> Dict[str, Any]:
        """Get device capabilities"""
        
        capabilities = {
            DeviceType.WEB_AR: {
                "6dof_tracking": False,
                "hand_tracking": False,
                "eye_tracking": False,
                "haptic_feedback": False,
                "spatial_audio": False,
                "max_fps": 30,
                "max_resolution": "1080p"
            },
            DeviceType.WEB_VR: {
                "6dof_tracking": True,
                "hand_tracking": False,
                "eye_tracking": False,
                "haptic_feedback": True,
                "spatial_audio": True,
                "max_fps": 60,
                "max_resolution": "1440p"
            },
            DeviceType.OCULUS_QUEST: {
                "6dof_tracking": True,
                "hand_tracking": True,
                "eye_tracking": False,
                "haptic_feedback": True,
                "spatial_audio": True,
                "max_fps": 90,
                "max_resolution": "1832x1920"
            },
            DeviceType.HOLOLENS: {
                "6dof_tracking": True,
                "hand_tracking": True,
                "eye_tracking": True,
                "haptic_feedback": False,
                "spatial_audio": True,
                "max_fps": 60,
                "max_resolution": "1268x720"
            }
        }
        
        return capabilities.get(device_type, capabilities[DeviceType.GENERIC_VR])
    
    def _get_user_color(self, index: int) -> str:
        """Get color for user avatar"""
        colors = ["#ff4444", "#44ff44", "#4444ff", "#ffff44", "#ff44ff", "#44ffff"]
        return colors[index % len(colors)]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get AR/VR analytics summary"""
        
        active_sessions = len(self.session_manager.active_sessions)
        
        # Device type distribution
        device_distribution = Counter([
            session.device_type.value 
            for session in self.session_manager.active_sessions.values()
        ])
        
        # Performance metrics
        frame_rates = [
            session.frame_rate 
            for session in self.session_manager.active_sessions.values()
            if session.frame_rate > 0
        ]
        
        avg_frame_rate = np.mean(frame_rates) if frame_rates else 0
        
        return {
            "active_sessions": active_sessions,
            "device_distribution": dict(device_distribution),
            "performance_metrics": {
                "avg_frame_rate": avg_frame_rate,
                "sessions_with_tracking": len([
                    s for s in self.session_manager.active_sessions.values()
                    if s.motion_sickness_level < 3
                ])
            },
            "feature_usage": {
                "collaborative_sessions": len([
                    s for s in self.session_manager.active_sessions.values()
                    if s.shared_session
                ]),
                "hand_tracking_sessions": len([
                    s for s in self.session_manager.active_sessions.values()
                    if s.interaction_mode == InteractionMode.HAND_TRACKING
                ])
            }
        }

# Streamlit Integration Functions

def initialize_ar_vr_system():
    """Initialize AR/VR system"""
    if 'ar_vr_engine' not in st.session_state:
        st.session_state.ar_vr_engine = ImmersiveAnalyticsEngine()
    
    return st.session_state.ar_vr_engine

def render_ar_vr_dashboard():
    """Render AR/VR visualization dashboard"""
    st.header("🥽 AR/VR Immersive Visualization")
    
    ar_vr_engine = initialize_ar_vr_system()
    
    # Get analytics summary
    analytics = ar_vr_engine.get_analytics_summary()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Sessions", analytics["active_sessions"])
    
    with col2:
        avg_fps = analytics["performance_metrics"]["avg_frame_rate"]
        st.metric("Avg Frame Rate", f"{avg_fps:.1f} FPS")
    
    with col3:
        collaborative = analytics["feature_usage"]["collaborative_sessions"]
        st.metric("Collaborative Sessions", collaborative)
    
    with col4:
        hand_tracking = analytics["feature_usage"]["hand_tracking_sessions"]
        st.metric("Hand Tracking Sessions", hand_tracking)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌄 Risk Landscape",
        "⭐ Data Constellation", 
        "📊 Immersive Dashboard",
        "👥 Collaboration",
        "🎮 Device Management",
        "📈 Analytics"
    ])
    
    with tab1:
        st.subheader("3D Risk Landscape Visualization")
        
        st.write("Create immersive 3D landscapes where risks are visualized as terrain features.")
        
        # Device selection
        device_type = st.selectbox(
            "Select Device Type",
            [device.value for device in DeviceType]
        )
        
        # Sample risk data
        if st.button("🎬 Launch Risk Landscape"):
            with st.spinner("Creating immersive risk landscape..."):
                # Generate sample risk data
                sample_risks = []
                risk_levels = ['low', 'medium', 'high', 'critical']
                categories = ['Security', 'Operational', 'Financial', 'Compliance', 'Strategic']
                
                for i in range(50):
                    risk = {
                        'risk_id': f'RISK_{i:03d}',
                        'description': f'Sample risk {i}',
                        'risk_score': np.random.uniform(0.1, 1.0),
                        'risk_level': np.random.choice(risk_levels),
                        'category': np.random.choice(categories),
                        'impact': np.random.choice(['Low', 'Medium', 'High', 'Very High']),
                        'risk_category_id': i % 10,
                        'temporal_factor': i // 10
                    }
                    sample_risks.append(risk)
                
                # Create visualization
                result = ar_vr_engine.create_risk_landscape_visualization(
                    sample_risks, "demo_user", DeviceType(device_type)
                )
                
                if result:
                    st.success("Risk landscape created successfully!")
                    
                    # Display scene information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Scene Details:**")
                        scene = result['scene']
                        st.write(f"• Scene Type: {scene['scene_type']}")
                        st.write(f"• Total Risk Points: {len(scene['mesh_data']['vertices'])}")
                        st.write(f"• Interaction Points: {len(scene['interaction_points'])}")
                        
                        metadata = scene['metadata']
                        st.write(f"• Max Risk Score: {metadata['max_risk_score']:.2f}")
                        st.write(f"• Risk Distribution: {metadata['risk_distribution']}")
                    
                    with col2:
                        st.write("**Device Capabilities:**")
                        caps = result['device_capabilities']
                        st.write(f"• 6DOF Tracking: {'✅' if caps['6dof_tracking'] else '❌'}")
                        st.write(f"• Hand Tracking: {'✅' if caps['hand_tracking'] else '❌'}")
                        st.write(f"• Eye Tracking: {'✅' if caps['eye_tracking'] else '❌'}")
                        st.write(f"• Haptic Feedback: {'✅' if caps['haptic_feedback'] else '❌'}")
                        st.write(f"• Max FPS: {caps['max_fps']}")
                        st.write(f"• Max Resolution: {caps['max_resolution']}")
                    
                    # Show WebGL code preview
                    with st.expander("View WebGL Scene Code"):
                        webgl_code = scene.get('webgl_scene', '// WebGL code would be generated here')
                        st.code(webgl_code[:500] + "..." if len(webgl_code) > 500 else webgl_code, language='javascript')
                    
                    # Visualization preview (using Plotly as fallback)
                    st.subheader("2D Preview (Interactive)")
                    
                    risk_df = pd.DataFrame(sample_risks)
                    
                    fig = px.scatter_3d(
                        risk_df, 
                        x='risk_category_id', 
                        y='temporal_factor', 
                        z='risk_score',
                        color='risk_level',
                        size='risk_score',
                        hover_data=['description', 'category', 'impact'],
                        title='3D Risk Landscape Preview'
                    )
                    
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='Risk Category',
                            yaxis_title='Time Factor', 
                            zaxis_title='Risk Score'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to create risk landscape visualization")
    
    with tab2:
        st.subheader("Data Constellation Visualization")
        
        st.write("Visualize data relationships as constellations in 3D space.")
        
        # Configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            num_entities = st.slider("Number of Entities", 10, 100, 25)
            connection_density = st.slider("Connection Density", 0.1, 0.8, 0.3)
        
        with col2:
            entity_types = st.multiselect(
                "Entity Types",
                ['risk', 'asset', 'user', 'system', 'process'],
                default=['risk', 'asset', 'system']
            )
        
        if st.button("🌌 Create Data Constellation"):
            with st.spinner("Creating data constellation..."):
                # Generate sample entities
                entities = []
                for i in range(num_entities):
                    entity = {
                        'id': f'entity_{i}',
                        'name': f'Entity {i}',
                        'type': np.random.choice(entity_types if entity_types else ['risk']),
                        'importance': np.random.uniform(0.5, 2.0),
                        'description': f'Sample entity {i}'
                    }
                    entities.append(entity)
                
                # Generate relationships
                relationships = []
                num_connections = int(num_entities * connection_density)
                
                for _ in range(num_connections):
                    source = np.random.choice(entities)
                    target = np.random.choice(entities)
                    
                    if source['id'] != target['id']:
                        relationship = {
                            'source': source['id'],
                            'target': target['id'],
                            'strength': np.random.uniform(0.2, 1.0),
                            'type': 'related_to',
                            'animated': np.random.choice([True, False])
                        }
                        relationships.append(relationship)
                
                # Create constellation
                scene = ar_vr_engine.session_manager.renderer.create_data_constellation(
                    entities, relationships
                )
                
                st.success("Data constellation created!")
                
                # Display constellation info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Constellation Details:**")
                    st.write(f"• Nodes: {len(scene['nodes'])}")
                    st.write(f"• Connections: {len(scene['connections'])}")
                    st.write(f"• Bounding Sphere Radius: {scene['bounding_sphere']['radius']:.2f}")
                
                with col2:
                    st.write("**Node Distribution:**")
                    node_types = Counter([node['metadata']['type'] for node in scene['nodes']])
                    for node_type, count in node_types.items():
                        st.write(f"• {node_type.title()}: {count}")
                
                # 3D visualization preview
                st.subheader("Constellation Preview")
                
                # Create networkx graph for visualization
                import networkx as nx
                
                G = nx.Graph()
                
                # Add nodes
                for node in scene['nodes']:
                    G.add_node(node['id'], **node['metadata'])
                
                # Add edges
                for conn in scene['connections']:
                    # Find source and target nodes
                    source_node = next((n for n in scene['nodes'] if n['position'] == conn['source']), None)
                    target_node = next((n for n in scene['nodes'] if n['position'] == conn['target']), None)
                    
                    if source_node and target_node:
                        G.add_edge(source_node['id'], target_node['id'], weight=conn['strength'])
                
                # Create 3D scatter plot
                positions_3d = {node['id']: node['position'] for node in scene['nodes']}
                
                node_trace = []
                edge_trace = []
                
                # Add nodes
                for node in scene['nodes']:
                    x, y, z = node['position']
                    node_trace.append([x, y, z])
                
                # Add edges
                for conn in scene['connections']:
                    x0, y0, z0 = conn['source']
                    x1, y1, z1 = conn['target']
                    edge_trace.extend([[x0, x1], [y0, y1], [z0, z1]])
                
                # Create 3D plot
                fig = go.Figure()
                
                # Add edges
                if edge_trace:
                    fig.add_trace(go.Scatter3d(
                        x=edge_trace[0], y=edge_trace[1], z=edge_trace[2],
                        mode='lines',
                        line=dict(color='rgba(125,125,125,0.3)', width=2),
                        name='Connections'
                    ))
                
                # Add nodes
                if node_trace:
                    x_nodes, y_nodes, z_nodes = zip(*node_trace)
                    colors = [scene['nodes'][i]['color'] for i in range(len(node_trace))]
                    sizes = [scene['nodes'][i]['size'] * 5 for i in range(len(node_trace))]
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_nodes, y=y_nodes, z=z_nodes,
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=colors,
                            opacity=0.8
                        ),
                        text=[node['metadata']['name'] for node in scene['nodes']],
                        name='Entities'
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Data Constellation Preview',
                    scene=dict(
                        xaxis_title='X Coordinate',
                        yaxis_title='Y Coordinate',
                        zaxis_title='Z Coordinate'
                    ),
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Immersive Dashboard Environment")
        
        st.write("Create curved dashboard layouts in 3D space for immersive data exploration.")
        
        # Panel configuration
        col1, col2 = st.columns(2)
        
        with col1:
            num_panels = st.slider("Number of Dashboard Panels", 3, 8, 5)
            dashboard_radius = st.slider("Dashboard Radius", 2.0, 6.0, 3.0)
        
        with col2:
            panel_types = st.multiselect(
                "Panel Types",
                ['bar', 'line', 'pie', 'heatmap', 'gauge'],
                default=['bar', 'line', 'pie']
            )
        
        if st.button("🎛️ Create Immersive Dashboard"):
            with st.spinner("Creating immersive dashboard..."):
                # Generate sample dashboard panels
                panels = []
                
                for i in range(num_panels):
                    panel_type = np.random.choice(panel_types if panel_types else ['bar'])
                    
                    # Generate sample data for panel
                    if panel_type == 'bar':
                        data = [
                            {'label': f'Item {j}', 'value': np.random.uniform(0.5, 2.0), 'color': f'hsl({j*60}, 70%, 50%)'}
                            for j in range(5)
                        ]
                    elif panel_type == 'pie':
                        data = [
                            {'label': f'Segment {j}', 'value': np.random.uniform(10, 100), 'color': f'hsl({j*72}, 70%, 50%)'}
                            for j in range(5)
                        ]
                    else:
                        data = [
                            {'x': j, 'y': np.random.uniform(0, 100)} for j in range(10)
                        ]
                    
                    panel = {
                        'id': f'panel_{i}',
                        'chart_type': panel_type,
                        'title': f'Panel {i+1}: {panel_type.title()} Chart',
                        'data': data,
                        'size': [2, 1.5],
                        'resizable': True
                    }
                    panels.append(panel)
                
                # Create immersive dashboard
                dashboard_config = {'radius': dashboard_radius}
                scene = ar_vr_engine.session_manager.renderer.create_immersive_dashboard(
                    dashboard_config, panels
                )
                
                st.success("Immersive dashboard created!")
                
                # Display dashboard info
                st.write("**Dashboard Configuration:**")
                st.write(f"• Panels: {len(scene['panels'])}")
                st.write(f"• Layout: Curved ({dashboard_radius}m radius)")
                st.write(f"• Environment: {scene['environment']['floor']['material'].title()}")
                st.write(f"• Teleport Points: {len(scene['navigation']['teleport_points'])}")
                
                # Show panel positions
                st.subheader("Panel Layout Preview")
                
                panel_positions = []
                panel_info = []
                
                for panel in scene['panels']:
                    x, y, z = panel['position']
                    panel_positions.append([x, y, z])
                    panel_info.append(panel['panel_id'])
                
                if panel_positions:
                    x_pos, y_pos, z_pos = zip(*panel_positions)
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=x_pos,
                        y=y_pos, 
                        z=z_pos,
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color='blue',
                            opacity=0.8
                        ),
                        text=panel_info,
                        textposition='top center'
                    )])
                    
                    # Add curved layout visualization
                    theta = np.linspace(0, 2*np.pi, 100)
                    curve_x = dashboard_radius * np.cos(theta)
                    curve_z = dashboard_radius * np.sin(theta)
                    curve_y = [1.5] * len(theta)
                    
                    fig.add_trace(go.Scatter3d(
                        x=curve_x,
                        y=curve_y,
                        z=curve_z,
                        mode='lines',
                        line=dict(color='gray', width=3, dash='dash'),
                        name='Dashboard Curve'
                    ))
                    
                    fig.update_layout(
                        title='Immersive Dashboard Layout',
                        scene=dict(
                            xaxis_title='X Position',
                            yaxis_title='Y Position',
                            zaxis_title='Z Position',
                            aspectmode='cube'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Panel content preview
                st.subheader("Panel Content Preview")
                
                for i, panel in enumerate(panels[:3]):  # Show first 3 panels
                    with st.expander(f"📊 {panel['title']}"):
                        if panel['chart_type'] == 'bar':
                            df = pd.DataFrame(panel['data'])
                            fig = px.bar(df, x='label', y='value', title=panel['title'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif panel['chart_type'] == 'pie':
                            df = pd.DataFrame(panel['data'])
                            fig = px.pie(df, values='value', names='label', title=panel['title'])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write(f"Panel type: {panel['chart_type']}")
                            st.json(panel['data'][:3])
    
    with tab4:
        st.subheader("Collaborative AR/VR Spaces")
        
        st.write("Create shared immersive environments for team collaboration.")
        
        # Collaboration configuration
        col1, col2 = st.columns(2)
        
        with col1:
            participants = st.text_area(
                "Participant IDs (one per line)",
                value="user_001\nuser_002\nuser_003"
            ).strip().split('\n') if st.text_area(
                "Participant IDs (one per line)",
                value="user_001\nuser_002\nuser_003"
            ).strip() else []
        
        with col2:
            collaboration_features = st.multiselect(
                "Collaboration Features",
                ["Voice Chat", "Shared Cursors", "Annotations", "Screen Sharing", "Synchronized Navigation"],
                default=["Voice Chat", "Shared Cursors", "Annotations"]
            )
        
        if st.button("🤝 Create Collaborative Space") and participants:
            with st.spinner("Creating collaborative AR/VR space..."):
                # Generate sample shared data
                entities = []
                relationships = []
                
                # Create entities representing different aspects of collaboration
                entity_types = ['document', 'presentation', 'data_view', 'annotation', 'user_workspace']
                
                for i in range(15):
                    entity = {
                        'id': f'collab_entity_{i}',
                        'name': f'Shared {entity_types[i % len(entity_types)]} {i}',
                        'type': entity_types[i % len(entity_types)],
                        'owner': participants[i % len(participants)],
                        'importance': np.random.uniform(0.5, 2.0)
                    }
                    entities.append(entity)
                
                # Create relationships between entities
                for i in range(20):
                    source = np.random.choice(entities)
                    target = np.random.choice(entities)
                    
                    if source['id'] != target['id']:
                        relationship = {
                            'source': source['id'],
                            'target': target['id'],
                            'strength': np.random.uniform(0.3, 1.0),
                            'type': 'collaboration',
                            'animated': True
                        }
                        relationships.append(relationship)
                
                shared_data = {
                    'entities': entities,
                    'relationships': relationships
                }
                
                # Create collaborative space
                result = ar_vr_engine.create_collaborative_space(participants, shared_data)
                
                st.success("Collaborative space created!")
                
                # Display collaboration info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Session Information:**")
                    st.write(f"• Main Session ID: {result['main_session_id'][:8]}...")
                    st.write(f"• Participants: {len(participants)}")
                    st.write(f"• Shared Entities: {len(entities)}")
                    st.write(f"• Relationships: {len(relationships)}")
                
                with col2:
                    st.write("**Collaboration Features:**")
                    collab_features = result['scene']['collaboration']
                    for feature, enabled in collab_features.items():
                        status = "✅" if enabled else "❌"
                        st.write(f"• {feature.replace('_', ' ').title()}: {status}")
                
                # Show participant sessions
                st.subheader("Participant Sessions")
                
                session_data = []
                for participant, session_id in result['participant_sessions'].items():
                    session_data.append({
                        'Participant': participant,
                        'Session ID': session_id[:8] + '...',
                        'Status': 'Active',
                        'Device': 'Generic VR'
                    })
                
                sessions_df = pd.DataFrame(session_data)
                st.dataframe(sessions_df, use_container_width=True, hide_index=True)
                
                # Avatar positions
                st.subheader("Avatar Positions")
                
                avatars = result['scene']['avatars']
                if avatars:
                    avatar_positions = [avatar['position'] for avatar in avatars]
                    avatar_names = [avatar['name_tag'] for avatar in avatars]
                    avatar_colors = [avatar['color'] for avatar in avatars]
                    
                    x_pos, y_pos, z_pos = zip(*avatar_positions)
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=x_pos,
                        y=y_pos,
                        z=z_pos,
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=avatar_colors,
                            opacity=0.8
                        ),
                        text=avatar_names,
                        textposition='top center'
                    )])
                    
                    fig.update_layout(
                        title='Collaborative Space - Avatar Positions',
                        scene=dict(
                            xaxis_title='X Position',
                            yaxis_title='Y Position',
                            zaxis_title='Z Position'
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Device Management")
        
        # Display active sessions
        active_sessions = list(ar_vr_engine.session_manager.active_sessions.values())
        
        if active_sessions:
            st.write(f"**Active Sessions ({len(active_sessions)}):**")
            
            session_data = []
            for session in active_sessions:
                session_data.append({
                    'Session ID': session.session_id[:8] + '...',
                    'User ID': session.user_id,
                    'Device Type': session.device_type.value,
                    'Start Time': session.start_time.strftime('%H:%M:%S'),
                    'Interaction Mode': session.interaction_mode.value,
                    'Quality': session.rendering_quality.value,
                    'Frame Rate': f"{session.frame_rate:.1f} FPS" if session.frame_rate > 0 else "N/A",
                    'Latency': f"{session.latency_ms:.1f}ms" if session.latency_ms > 0 else "N/A"
                })
            
            sessions_df = pd.DataFrame(session_data)
            st.dataframe(sessions_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active AR/VR sessions")
        
        # Device capabilities comparison
        st.subheader("Device Capabilities Comparison")
        
        device_types = [DeviceType.WEB_AR, DeviceType.WEB_VR, DeviceType.OCULUS_QUEST, DeviceType.HOLOLENS]
        capabilities_data = []
        
        for device in device_types:
            caps = ar_vr_engine._get_device_capabilities(device)
            capabilities_data.append({
                'Device': device.value.replace('_', ' ').title(),
                '6DOF Tracking': '✅' if caps['6dof_tracking'] else '❌',
                'Hand Tracking': '✅' if caps['hand_tracking'] else '❌',
                'Eye Tracking': '✅' if caps['eye_tracking'] else '❌',
                'Haptic Feedback': '✅' if caps['haptic_feedback'] else '❌',
                'Spatial Audio': '✅' if caps['spatial_audio'] else '❌',
                'Max FPS': caps['max_fps'],
                'Max Resolution': caps['max_resolution']
            })
        
        capabilities_df = pd.DataFrame(capabilities_data)
        st.dataframe(capabilities_df, use_container_width=True, hide_index=True)
        
        # Session creation
        st.subheader("Create New Session")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_user_id = st.text_input("User ID", value=f"user_{int(time.time())}")
        
        with col2:
            new_device_type = st.selectbox(
                "Device Type",
                [device.value for device in DeviceType]
            )
        
        with col3:
            if st.button("🚀 Create Session"):
                if new_user_id:
                    session_id = ar_vr_engine.session_manager.create_session(
                        new_user_id, DeviceType(new_device_type)
                    )
                    st.success(f"Created session: {session_id[:8]}...")
                    st.rerun()
                else:
                    st.error("Please provide a User ID")
    
    with tab6:
        st.subheader("AR/VR Analytics")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**System Performance:**")
            
            # Simulate performance data
            performance_data = {
                'Metric': ['Average Frame Rate', 'System Latency', 'Memory Usage', 'GPU Utilization'],
                'Value': ['75.2 FPS', '12.3ms', '68%', '82%'],
                'Status': ['🟢 Good', '🟡 Fair', '🟢 Good', '🟠 High']
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Usage Statistics:**")
            
            usage_stats = {
                'Total Sessions Created': 127,
                'Active Sessions': analytics['active_sessions'],
                'Average Session Duration': '23.4 minutes',
                'Most Popular Device': 'Web VR',
                'Collaboration Rate': '34%'
            }
            
            for stat, value in usage_stats.items():
                st.write(f"• **{stat}:** {value}")
        
        # Device distribution chart
        device_dist = analytics.get('device_distribution', {})
        if device_dist:
            st.subheader("Device Distribution")
            
            device_df = pd.DataFrame(
                list(device_dist.items()),
                columns=['Device Type', 'Sessions']
            )
            
            fig = px.pie(device_df, values='Sessions', names='Device Type',
                        title='Active Sessions by Device Type')
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance trends
        st.subheader("Performance Trends")
        
        # Simulate performance trends
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        frame_rates = 60 + np.random.normal(0, 10, 30)
        latencies = 15 + np.random.normal(0, 3, 30)
        
        trends_df = pd.DataFrame({
            'Date': dates,
            'Frame Rate': frame_rates,
            'Latency (ms)': latencies
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Frame Rate Trends', 'Latency Trends'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=trends_df['Date'], y=trends_df['Frame Rate'], 
                      mode='lines+markers', name='Frame Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=trends_df['Date'], y=trends_df['Latency (ms)'], 
                      mode='lines+markers', name='Latency', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_yaxes(title_text="FPS", row=1, col=1)
        fig.update_yaxes(title_text="Milliseconds", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export analytics
        if st.button("📊 Export Analytics Report"):
            analytics_report = {
                "ar_vr_analytics": analytics,
                "performance_metrics": performance_data,
                "usage_statistics": usage_stats,
                "device_distribution": device_dist,
                "export_timestamp": datetime.now().isoformat()
            }
            
            report_json = json.dumps(analytics_report, indent=2, default=str)
            
            st.download_button(
                label="Download AR/VR Analytics Report",
                data=report_json,
                file_name=f"ar_vr_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json'
            )

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing AR/VR immersive visualization system...")
    
    # Initialize AR/VR engine
    ar_vr_engine = ImmersiveAnalyticsEngine()
    
    # Create sample risk data
    sample_risks = [
        {
            'risk_id': f'RISK_{i:03d}',
            'description': f'Sample risk {i}',
            'risk_score': np.random.uniform(0.1, 1.0),
            'risk_level': np.random.choice(['low', 'medium', 'high', 'critical']),
            'category': np.random.choice(['Security', 'Operational', 'Financial']),
            'impact': np.random.choice(['Low', 'Medium', 'High']),
            'risk_category_id': i % 5,
            'temporal_factor': i // 5
        }
        for i in range(20)
    ]
    
    # Test risk landscape visualization
    print("\nTesting risk landscape visualization...")
    result = ar_vr_engine.create_risk_landscape_visualization(
        sample_risks, "test_user", DeviceType.WEB_VR
    )
    
    if result:
        print(f"✅ Risk landscape created successfully")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Visualization ID: {result['visualization_id']}")
        print(f"   Scene type: {result['scene']['scene_type']}")
        print(f"   Risk points: {len(result['scene']['mesh_data']['vertices'])}")
    
    # Test collaborative space
    print("\nTesting collaborative space...")
    participants = ["user_001", "user_002", "user_003"]
    
    # Sample collaborative data
    entities = [
        {'id': f'entity_{i}', 'name': f'Entity {i}', 'type': 'document', 'importance': 1.0}
        for i in range(10)
    ]
    relationships = [
        {'source': f'entity_{i}', 'target': f'entity_{i+1}', 'strength': 0.8, 'type': 'related'}
        for i in range(9)
    ]
    
    shared_data = {'entities': entities, 'relationships': relationships}
    
    collab_result = ar_vr_engine.create_collaborative_space(participants, shared_data)
    
    if collab_result:
        print(f"✅ Collaborative space created")
        print(f"   Main session: {collab_result['main_session_id']}")
        print(f"   Participants: {len(collab_result['participant_sessions'])}")
        print(f"   Avatars: {len(collab_result['scene']['avatars'])}")
    
    # Get analytics
    analytics = ar_vr_engine.get_analytics_summary()
    print(f"\nSystem Analytics:")
    print(f"• Active sessions: {analytics['active_sessions']}")
    print(f"• Device distribution: {analytics['device_distribution']}")
    print(f"• Collaborative sessions: {analytics['feature_usage']['collaborative_sessions']}")
    
    print("\nAR/VR immersive visualization system test completed!")