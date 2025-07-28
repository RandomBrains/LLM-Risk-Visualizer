"""
Edge Computing and IoT Device Integration Module
Provides edge computing capabilities, IoT device management, and distributed risk analysis
"""

import json
import time
import asyncio
import threading
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import socket
import ssl
import logging
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import paho.mqtt.client as mqtt
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from collections import deque, defaultdict
import pickle
import base64
import zlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Types of IoT devices"""
    SENSOR = "sensor"
    CAMERA = "camera"
    GATEWAY = "gateway"
    ACTUATOR = "actuator"
    EDGE_SERVER = "edge_server"
    SMART_METER = "smart_meter"
    ENVIRONMENTAL = "environmental"
    SECURITY = "security"
    INDUSTRIAL = "industrial"
    WEARABLE = "wearable"

class DeviceStatus(Enum):
    """Device status enumeration"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UPDATING = "updating"
    UNKNOWN = "unknown"

class DataPriority(Enum):
    """Data processing priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

class EdgeLocation(Enum):
    """Edge computing location types"""
    ON_DEVICE = "on_device"
    EDGE_GATEWAY = "edge_gateway"
    EDGE_SERVER = "edge_server"
    REGIONAL_HUB = "regional_hub"
    CLOUD = "cloud"

@dataclass
class IoTDevice:
    """IoT device definition"""
    device_id: str
    name: str
    device_type: DeviceType
    manufacturer: str
    model: str
    firmware_version: str
    
    # Network configuration
    ip_address: str
    port: int
    protocol: str  # MQTT, HTTP, CoAP, etc.
    
    # Location and deployment
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    deployment_date: datetime = None
    
    # Device capabilities
    capabilities: List[str] = None
    sensors: List[str] = None
    max_data_rate: float = 0.0  # MB/s
    storage_capacity: float = 0.0  # GB
    compute_power: float = 0.0  # GFLOPS
    
    # Status and health
    status: DeviceStatus = DeviceStatus.UNKNOWN
    battery_level: Optional[float] = None
    temperature: Optional[float] = None
    last_seen: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # Data and metrics
    data_generated_mb: float = 0.0
    data_transmitted_mb: float = 0.0
    error_count: int = 0
    
    # Security
    security_key: Optional[str] = None
    certificate_path: Optional[str] = None
    encrypted: bool = False
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.sensors is None:
            self.sensors = []
        if self.deployment_date is None:
            self.deployment_date = datetime.now()

@dataclass
class EdgeNode:
    """Edge computing node definition"""
    node_id: str
    name: str
    location: EdgeLocation
    description: str
    
    # Hardware specifications
    cpu_cores: int
    ram_gb: float
    storage_gb: float
    gpu_available: bool = False
    
    # Network configuration
    ip_address: str
    port: int
    bandwidth_mbps: float
    
    # Geographic location
    latitude: float
    longitude: float
    region: str
    
    # Computational capabilities
    max_concurrent_jobs: int = 10
    supported_frameworks: List[str] = None
    ai_accelerators: List[str] = None
    
    # Status and metrics
    status: DeviceStatus = DeviceStatus.UNKNOWN
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    storage_usage: float = 0.0
    temperature: float = 0.0
    
    # Connected devices
    connected_devices: List[str] = None
    active_jobs: int = 0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    jobs_completed: int = 0
    jobs_failed: int = 0
    last_heartbeat: Optional[datetime] = None
    
    def __post_init__(self):
        if self.supported_frameworks is None:
            self.supported_frameworks = ["tensorflow", "pytorch", "onnx"]
        if self.ai_accelerators is None:
            self.ai_accelerators = []
        if self.connected_devices is None:
            self.connected_devices = []

@dataclass
class EdgeJob:
    """Edge computing job definition"""
    job_id: str
    name: str
    job_type: str  # risk_analysis, data_processing, inference, etc.
    priority: DataPriority
    
    # Job configuration
    input_data: Dict[str, Any]
    model_path: Optional[str] = None
    parameters: Dict[str, Any] = None
    
    # Resource requirements
    cpu_cores_required: int = 1
    memory_mb_required: float = 512
    gpu_required: bool = False
    max_execution_time_seconds: int = 300
    
    # Scheduling
    target_node_id: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Status tracking
    status: str = "pending"  # pending, running, completed, failed, cancelled
    created_time: datetime = None
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # Results
    output_data: Dict[str, Any] = None
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    
    # Origin information
    source_device_id: Optional[str] = None
    user_id: str = "system"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.output_data is None:
            self.output_data = {}
        if self.created_time is None:
            self.created_time = datetime.now()

@dataclass
class IoTDataMessage:
    """IoT data message structure"""
    message_id: str
    device_id: str
    timestamp: datetime
    data_type: str
    payload: Dict[str, Any]
    
    # Message metadata
    priority: DataPriority = DataPriority.NORMAL
    size_bytes: int = 0
    format: str = "json"
    compressed: bool = False
    encrypted: bool = False
    
    # Quality metrics
    signal_strength: Optional[float] = None
    accuracy: Optional[float] = None
    confidence: Optional[float] = None
    
    # Processing tracking
    processed: bool = False
    edge_node_id: Optional[str] = None
    processing_time_ms: float = 0.0
    
    # Risk assessment
    risk_score: Optional[float] = None
    risk_factors: List[str] = None
    anomaly_detected: bool = False
    
    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = []
        if not self.message_id:
            self.message_id = str(uuid.uuid4())

class MQTTManager:
    """MQTT client manager for IoT communication"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = None
        self.connected = False
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = deque(maxlen=10000)
        
    def connect(self, username: str = None, password: str = None) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client = mqtt.Client()
            
            if username and password:
                self.client.username_pw_set(username, password)
            
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = time.time() + 10
            while not self.connected and time.time() < timeout:
                time.sleep(0.1)
            
            return self.connected
            
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.connected = False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # Add to message queue
            self.message_queue.append({
                'topic': topic,
                'payload': payload,
                'timestamp': datetime.now()
            })
            
            # Call registered handlers
            for pattern, handler in self.message_handlers.items():
                if pattern in topic:
                    try:
                        handler(topic, payload)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
                        
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def subscribe(self, topic: str, handler: Callable = None):
        """Subscribe to MQTT topic"""
        if self.client and self.connected:
            self.client.subscribe(topic)
            if handler:
                self.message_handlers[topic] = handler
            logger.info(f"Subscribed to topic: {topic}")
    
    def publish(self, topic: str, payload: str, qos: int = 0):
        """Publish message to MQTT topic"""
        if self.client and self.connected:
            self.client.publish(topic, payload, qos)
    
    def get_recent_messages(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent messages from queue"""
        return list(self.message_queue)[-count:]

class EdgeComputingEngine:
    """Edge computing orchestration engine"""
    
    def __init__(self):
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.job_queue: deque = deque()
        self.running_jobs: Dict[str, EdgeJob] = {}
        self.completed_jobs: Dict[str, EdgeJob] = {}
        self.job_executor = ThreadPoolExecutor(max_workers=20)
        self.scheduler_running = False
        self.scheduler_thread = None
        
    def register_edge_node(self, node: EdgeNode):
        """Register an edge computing node"""
        self.edge_nodes[node.node_id] = node
        logger.info(f"Edge node {node.name} registered")
    
    def submit_job(self, job: EdgeJob) -> str:
        """Submit job for edge processing"""
        job.job_id = str(uuid.uuid4())
        job.created_time = datetime.now()
        
        # Add to job queue
        self.job_queue.append(job)
        logger.info(f"Job {job.name} submitted with ID {job.job_id}")
        
        return job.job_id
    
    def start_scheduler(self):
        """Start job scheduler"""
        if not self.scheduler_running:
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            logger.info("Edge job scheduler started")
    
    def stop_scheduler(self):
        """Stop job scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Edge job scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_running:
            try:
                if self.job_queue:
                    job = self.job_queue.popleft()
                    
                    # Find suitable node
                    target_node = self._find_best_node(job)
                    
                    if target_node:
                        # Schedule job
                        job.target_node_id = target_node.node_id
                        job.scheduled_time = datetime.now()
                        job.status = "scheduled"
                        
                        # Submit for execution
                        future = self.job_executor.submit(self._execute_job, job, target_node)
                        self.running_jobs[job.job_id] = job
                        
                        # Update node status
                        target_node.active_jobs += 1
                        
                        logger.info(f"Job {job.job_id} scheduled on node {target_node.name}")
                    else:
                        # No suitable node, requeue
                        self.job_queue.append(job)
                        time.sleep(1)
                
                time.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1)
    
    def _find_best_node(self, job: EdgeJob) -> Optional[EdgeNode]:
        """Find best edge node for job execution"""
        suitable_nodes = []
        
        for node in self.edge_nodes.values():
            if (node.status == DeviceStatus.ONLINE and
                node.active_jobs < node.max_concurrent_jobs and
                node.cpu_usage < 80.0 and
                node.memory_usage < 80.0):
                
                # Check resource requirements
                if (job.cpu_cores_required <= node.cpu_cores and
                    job.memory_mb_required <= (node.ram_gb * 1024 * (1 - node.memory_usage / 100))):
                    
                    # Check GPU requirement
                    if not job.gpu_required or node.gpu_available:
                        suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Sort by priority criteria
        def node_score(node):
            # Prefer nodes with lower utilization and better performance
            utilization_score = (100 - node.cpu_usage - node.memory_usage) / 100
            performance_score = 1 / (node.avg_response_time_ms + 1)
            return utilization_score * 0.7 + performance_score * 0.3
        
        return max(suitable_nodes, key=node_score)
    
    def _execute_job(self, job: EdgeJob, node: EdgeNode):
        """Execute job on edge node"""
        try:
            job.start_time = datetime.now()
            job.status = "running"
            
            # Simulate job execution based on job type
            if job.job_type == "risk_analysis":
                result = self._execute_risk_analysis(job, node)
            elif job.job_type == "data_processing":
                result = self._execute_data_processing(job, node)
            elif job.job_type == "inference":
                result = self._execute_inference(job, node)
            else:
                result = self._execute_generic_job(job, node)
            
            # Mark as completed
            job.completion_time = datetime.now()
            job.execution_time_seconds = (job.completion_time - job.start_time).total_seconds()
            job.status = "completed"
            job.output_data = result
            
            # Update metrics
            node.jobs_completed += 1
            node.avg_response_time_ms = (
                (node.avg_response_time_ms * (node.jobs_completed - 1) + 
                 job.execution_time_seconds * 1000) / node.jobs_completed
            )
            
            logger.info(f"Job {job.job_id} completed in {job.execution_time_seconds:.2f}s")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completion_time = datetime.now()
            node.jobs_failed += 1
            logger.error(f"Job {job.job_id} failed: {e}")
        
        finally:
            # Clean up
            node.active_jobs = max(0, node.active_jobs - 1)
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            self.completed_jobs[job.job_id] = job
    
    def _execute_risk_analysis(self, job: EdgeJob, node: EdgeNode) -> Dict[str, Any]:
        """Execute risk analysis job"""
        time.sleep(np.random.uniform(1, 5))  # Simulate processing time
        
        # Mock risk analysis
        input_data = job.input_data
        risk_score = np.random.uniform(0.1, 0.9)
        
        return {
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "factors": ["anomalous_pattern", "unusual_values"] if risk_score > 0.5 else ["normal_pattern"],
            "confidence": np.random.uniform(0.7, 0.95),
            "processed_records": len(input_data.get("data", [])),
            "node_id": node.node_id
        }
    
    def _execute_data_processing(self, job: EdgeJob, node: EdgeNode) -> Dict[str, Any]:
        """Execute data processing job"""
        time.sleep(np.random.uniform(0.5, 3))  # Simulate processing time
        
        input_data = job.input_data
        processed_count = len(input_data.get("data", []))
        
        return {
            "processed_records": processed_count,
            "filtered_records": int(processed_count * 0.8),
            "anomalies_detected": int(processed_count * 0.05),
            "processing_time_ms": np.random.uniform(100, 1000),
            "node_id": node.node_id
        }
    
    def _execute_inference(self, job: EdgeJob, node: EdgeNode) -> Dict[str, Any]:
        """Execute ML inference job"""
        time.sleep(np.random.uniform(0.2, 2))  # Simulate inference time
        
        input_data = job.input_data
        predictions = []
        
        for i in range(len(input_data.get("samples", []))):
            predictions.append({
                "prediction": np.random.uniform(0, 1),
                "confidence": np.random.uniform(0.8, 0.99)
            })
        
        return {
            "predictions": predictions,
            "model_version": "v1.0",
            "inference_time_ms": np.random.uniform(50, 500),
            "node_id": node.node_id
        }
    
    def _execute_generic_job(self, job: EdgeJob, node: EdgeNode) -> Dict[str, Any]:
        """Execute generic job"""
        time.sleep(np.random.uniform(1, 3))  # Simulate processing time
        
        return {
            "status": "completed",
            "message": f"Generic job processed on {node.name}",
            "node_id": node.node_id
        }
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get status of all edge nodes"""
        return {
            "total_nodes": len(self.edge_nodes),
            "online_nodes": len([n for n in self.edge_nodes.values() if n.status == DeviceStatus.ONLINE]),
            "total_capacity": sum(n.max_concurrent_jobs for n in self.edge_nodes.values()),
            "active_jobs": sum(n.active_jobs for n in self.edge_nodes.values()),
            "nodes": {node_id: asdict(node) for node_id, node in self.edge_nodes.items()}
        }
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get job execution statistics"""
        total_jobs = len(self.completed_jobs)
        successful_jobs = len([j for j in self.completed_jobs.values() if j.status == "completed"])
        failed_jobs = len([j for j in self.completed_jobs.values() if j.status == "failed"])
        
        avg_execution_time = 0
        if successful_jobs > 0:
            avg_execution_time = np.mean([
                j.execution_time_seconds for j in self.completed_jobs.values() 
                if j.status == "completed"
            ])
        
        return {
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": successful_jobs / max(1, total_jobs),
            "avg_execution_time_seconds": avg_execution_time,
            "pending_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs)
        }

class IoTDeviceManager:
    """IoT device management system"""
    
    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.mqtt_manager = MQTTManager()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.data_buffer: deque = deque(maxlen=50000)
        
    def register_device(self, device: IoTDevice):
        """Register IoT device"""
        self.devices[device.device_id] = device
        logger.info(f"IoT device {device.name} registered")
        
        # Subscribe to device topics if MQTT is connected
        if self.mqtt_manager.connected:
            self._subscribe_device_topics(device)
    
    def connect_mqtt(self, broker_host: str = "localhost", broker_port: int = 1883) -> bool:
        """Connect to MQTT broker"""
        success = self.mqtt_manager.connect(broker_host, broker_port)
        
        if success:
            # Subscribe to all device topics
            for device in self.devices.values():
                self._subscribe_device_topics(device)
        
        return success
    
    def _subscribe_device_topics(self, device: IoTDevice):
        """Subscribe to device MQTT topics"""
        base_topic = f"devices/{device.device_id}"
        
        # Subscribe to data topic
        self.mqtt_manager.subscribe(
            f"{base_topic}/data",
            lambda topic, payload: self._handle_device_data(device.device_id, payload)
        )
        
        # Subscribe to status topic
        self.mqtt_manager.subscribe(
            f"{base_topic}/status",
            lambda topic, payload: self._handle_device_status(device.device_id, payload)
        )
    
    def _handle_device_data(self, device_id: str, payload: str):
        """Handle incoming device data"""
        try:
            data = json.loads(payload)
            
            message = IoTDataMessage(
                message_id=str(uuid.uuid4()),
                device_id=device_id,
                timestamp=datetime.now(),
                data_type=data.get("type", "sensor"),
                payload=data,
                size_bytes=len(payload)
            )
            
            # Add to buffer
            self.data_buffer.append(message)
            
            # Update device metrics
            if device_id in self.devices:
                device = self.devices[device_id]
                device.last_seen = datetime.now()
                device.data_generated_mb += len(payload) / (1024 * 1024)
                device.status = DeviceStatus.ONLINE
            
        except Exception as e:
            logger.error(f"Error handling device data: {e}")
    
    def _handle_device_status(self, device_id: str, payload: str):
        """Handle device status updates"""
        try:
            status_data = json.loads(payload)
            
            if device_id in self.devices:
                device = self.devices[device_id]
                device.last_seen = datetime.now()
                device.battery_level = status_data.get("battery_level")
                device.temperature = status_data.get("temperature")
                device.status = DeviceStatus(status_data.get("status", "online"))
                
        except Exception as e:
            logger.error(f"Error handling device status: {e}")
    
    def start_monitoring(self):
        """Start device monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Device monitoring started")
    
    def stop_monitoring(self):
        """Stop device monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Device monitoring stopped")
    
    def _monitoring_loop(self):
        """Device monitoring loop"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for device in self.devices.values():
                    # Check if device is offline
                    if device.last_seen:
                        time_since_seen = (current_time - device.last_seen).total_seconds()
                        if time_since_seen > 300:  # 5 minutes
                            device.status = DeviceStatus.OFFLINE
                    
                    # Update uptime for online devices
                    if device.status == DeviceStatus.ONLINE:
                        device.uptime_seconds += 30
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)
    
    def get_device_data(self, device_id: str, hours: int = 1) -> List[IoTDataMessage]:
        """Get recent data from specific device"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            msg for msg in self.data_buffer
            if msg.device_id == device_id and msg.timestamp >= cutoff_time
        ]
    
    def get_all_recent_data(self, minutes: int = 60) -> List[IoTDataMessage]:
        """Get all recent data from all devices"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            msg for msg in self.data_buffer
            if msg.timestamp >= cutoff_time
        ]
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get device statistics"""
        total_devices = len(self.devices)
        online_devices = len([d for d in self.devices.values() if d.status == DeviceStatus.ONLINE])
        offline_devices = len([d for d in self.devices.values() if d.status == DeviceStatus.OFFLINE])
        
        total_data_generated = sum(d.data_generated_mb for d in self.devices.values())
        avg_battery_level = np.mean([
            d.battery_level for d in self.devices.values() 
            if d.battery_level is not None
        ]) if any(d.battery_level is not None for d in self.devices.values()) else 0
        
        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "device_availability": online_devices / max(1, total_devices),
            "total_data_generated_mb": total_data_generated,
            "avg_battery_level": avg_battery_level,
            "data_messages_buffered": len(self.data_buffer)
        }

class EdgeRiskAnalyzer:
    """Edge-based risk analysis engine"""
    
    def __init__(self, edge_engine: EdgeComputingEngine):
        self.edge_engine = edge_engine
        self.risk_models: Dict[str, Any] = {}
        self.analysis_results: deque = deque(maxlen=10000)
        
    def load_risk_model(self, model_name: str, model_data: Dict[str, Any]):
        """Load risk analysis model"""
        self.risk_models[model_name] = {
            "model_data": model_data,
            "loaded_time": datetime.now(),
            "version": model_data.get("version", "1.0")
        }
        logger.info(f"Risk model {model_name} loaded")
    
    def analyze_iot_data(self, messages: List[IoTDataMessage], 
                        model_name: str = "default") -> Dict[str, Any]:
        """Analyze IoT data for risks using edge computing"""
        
        # Create edge job for risk analysis
        job = EdgeJob(
            job_id="",
            name=f"IoT Risk Analysis - {model_name}",
            job_type="risk_analysis",
            priority=DataPriority.HIGH,
            input_data={
                "messages": [asdict(msg) for msg in messages],
                "model_name": model_name
            },
            parameters={
                "sensitivity_threshold": 0.7,
                "anomaly_detection": True
            },
            cpu_cores_required=2,
            memory_mb_required=1024,
            max_execution_time_seconds=60
        )
        
        # Submit job
        job_id = self.edge_engine.submit_job(job)
        
        # Wait for completion (in real implementation, this would be async)
        timeout = time.time() + 30
        while time.time() < timeout:
            if job_id in self.edge_engine.completed_jobs:
                completed_job = self.edge_engine.completed_jobs[job_id]
                
                if completed_job.status == "completed":
                    # Store results
                    analysis_result = {
                        "job_id": job_id,
                        "timestamp": completed_job.completion_time,
                        "messages_analyzed": len(messages),
                        "results": completed_job.output_data,
                        "execution_time": completed_job.execution_time_seconds,
                        "node_id": completed_job.target_node_id
                    }
                    
                    self.analysis_results.append(analysis_result)
                    return analysis_result
                else:
                    return {
                        "error": completed_job.error_message,
                        "status": "failed"
                    }
            
            time.sleep(0.5)
        
        return {"error": "Analysis timeout", "status": "timeout"}
    
    def detect_anomalies(self, device_data: List[IoTDataMessage]) -> List[Dict[str, Any]]:
        """Detect anomalies in device data"""
        anomalies = []
        
        for message in device_data:
            # Simple anomaly detection rules
            payload = message.payload
            
            # Check for unusual values
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        # Basic threshold checks
                        if key == "temperature" and (value < -40 or value > 100):
                            anomalies.append({
                                "device_id": message.device_id,
                                "timestamp": message.timestamp,
                                "anomaly_type": "temperature_out_of_range",
                                "value": value,
                                "severity": "high"
                            })
                        elif key == "humidity" and (value < 0 or value > 100):
                            anomalies.append({
                                "device_id": message.device_id,
                                "timestamp": message.timestamp,
                                "anomaly_type": "humidity_out_of_range",
                                "value": value,
                                "severity": "medium"
                            })
        
        return anomalies
    
    def get_risk_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get risk trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_results = [
            result for result in self.analysis_results
            if result["timestamp"] >= cutoff_time
        ]
        
        if not recent_results:
            return {"message": "No recent analysis results"}
        
        # Calculate trends
        risk_scores = []
        timestamps = []
        
        for result in recent_results:
            if "results" in result and "risk_score" in result["results"]:
                risk_scores.append(result["results"]["risk_score"])
                timestamps.append(result["timestamp"])
        
        if risk_scores:
            return {
                "avg_risk_score": np.mean(risk_scores),
                "max_risk_score": np.max(risk_scores),
                "min_risk_score": np.min(risk_scores),
                "risk_trend": "increasing" if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else "stable",
                "total_analyses": len(recent_results),
                "time_range_hours": hours
            }
        
        return {"message": "No risk scores found in results"}

class EdgeIoTSystem:
    """Integrated Edge Computing and IoT System"""
    
    def __init__(self, db_path: str = "edge_iot.db"):
        self.db_path = Path(db_path)
        self.device_manager = IoTDeviceManager()
        self.edge_engine = EdgeComputingEngine()
        self.risk_analyzer = EdgeRiskAnalyzer(self.edge_engine)
        
        # Initialize database
        self.init_database()
        
        # Load existing data
        self.load_devices()
        self.load_edge_nodes()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # IoT devices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iot_devices (
                device_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                device_type TEXT NOT NULL,
                manufacturer TEXT,
                model TEXT,
                firmware_version TEXT,
                ip_address TEXT,
                port INTEGER,
                protocol TEXT,
                location TEXT,
                latitude REAL,
                longitude REAL,
                deployment_date TEXT,
                capabilities TEXT,
                sensors TEXT,
                max_data_rate REAL,
                storage_capacity REAL,
                compute_power REAL,
                status TEXT,
                battery_level REAL,
                temperature REAL,
                last_seen TEXT,
                uptime_seconds REAL,
                data_generated_mb REAL,
                data_transmitted_mb REAL,
                error_count INTEGER,
                security_key TEXT,
                certificate_path TEXT,
                encrypted BOOLEAN
            )
        ''')
        
        # Edge nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edge_nodes (
                node_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                description TEXT,
                cpu_cores INTEGER,
                ram_gb REAL,
                storage_gb REAL,
                gpu_available BOOLEAN,
                ip_address TEXT,
                port INTEGER,
                bandwidth_mbps REAL,
                latitude REAL,
                longitude REAL,
                region TEXT,
                max_concurrent_jobs INTEGER,
                supported_frameworks TEXT,
                ai_accelerators TEXT,
                status TEXT,
                cpu_usage REAL,
                memory_usage REAL,
                storage_usage REAL,
                temperature REAL,
                connected_devices TEXT,
                active_jobs INTEGER,
                avg_response_time_ms REAL,
                jobs_completed INTEGER,
                jobs_failed INTEGER,
                last_heartbeat TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_devices(self):
        """Load IoT devices from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM iot_devices')
            rows = cursor.fetchall()
            
            for row in rows:
                device = self._row_to_device(row)
                if device:
                    self.device_manager.register_device(device)
            
            conn.close()
            logger.info(f"Loaded {len(rows)} IoT devices from database")
            
        except Exception as e:
            logger.error(f"Error loading devices: {e}")
    
    def _row_to_device(self, row) -> Optional[IoTDevice]:
        """Convert database row to IoTDevice object"""
        try:
            return IoTDevice(
                device_id=row[0],
                name=row[1],
                device_type=DeviceType(row[2]),
                manufacturer=row[3] or "",
                model=row[4] or "",
                firmware_version=row[5] or "",
                ip_address=row[6] or "",
                port=row[7] or 0,
                protocol=row[8] or "MQTT",
                location=row[9] or "",
                latitude=row[10],
                longitude=row[11],
                deployment_date=datetime.fromisoformat(row[12]) if row[12] else datetime.now(),
                capabilities=json.loads(row[13]) if row[13] else [],
                sensors=json.loads(row[14]) if row[14] else [],
                max_data_rate=row[15] or 0.0,
                storage_capacity=row[16] or 0.0,
                compute_power=row[17] or 0.0,
                status=DeviceStatus(row[18]) if row[18] else DeviceStatus.UNKNOWN,
                battery_level=row[19],
                temperature=row[20],
                last_seen=datetime.fromisoformat(row[21]) if row[21] else None,
                uptime_seconds=row[22] or 0.0,
                data_generated_mb=row[23] or 0.0,
                data_transmitted_mb=row[24] or 0.0,
                error_count=row[25] or 0,
                security_key=row[26],
                certificate_path=row[27],
                encrypted=bool(row[28]) if row[28] is not None else False
            )
        except Exception as e:
            logger.error(f"Error converting row to device: {e}")
            return None
    
    def load_edge_nodes(self):
        """Load edge nodes from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM edge_nodes')
            rows = cursor.fetchall()
            
            for row in rows:
                node = self._row_to_edge_node(row)
                if node:
                    self.edge_engine.register_edge_node(node)
            
            conn.close()
            logger.info(f"Loaded {len(rows)} edge nodes from database")
            
        except Exception as e:
            logger.error(f"Error loading edge nodes: {e}")
    
    def _row_to_edge_node(self, row) -> Optional[EdgeNode]:
        """Convert database row to EdgeNode object"""
        try:
            return EdgeNode(
                node_id=row[0],
                name=row[1],
                location=EdgeLocation(row[2]),
                description=row[3] or "",
                cpu_cores=row[4] or 1,
                ram_gb=row[5] or 1.0,
                storage_gb=row[6] or 10.0,
                gpu_available=bool(row[7]) if row[7] is not None else False,
                ip_address=row[8] or "",
                port=row[9] or 8080,
                bandwidth_mbps=row[10] or 100.0,
                latitude=row[11] or 0.0,
                longitude=row[12] or 0.0,
                region=row[13] or "",
                max_concurrent_jobs=row[14] or 10,
                supported_frameworks=json.loads(row[15]) if row[15] else [],
                ai_accelerators=json.loads(row[16]) if row[16] else [],
                status=DeviceStatus(row[17]) if row[17] else DeviceStatus.UNKNOWN,
                cpu_usage=row[18] or 0.0,
                memory_usage=row[19] or 0.0,
                storage_usage=row[20] or 0.0,
                temperature=row[21] or 0.0,
                connected_devices=json.loads(row[22]) if row[22] else [],
                active_jobs=row[23] or 0,
                avg_response_time_ms=row[24] or 0.0,
                jobs_completed=row[25] or 0,
                jobs_failed=row[26] or 0,
                last_heartbeat=datetime.fromisoformat(row[27]) if row[27] else None
            )
        except Exception as e:
            logger.error(f"Error converting row to edge node: {e}")
            return None
    
    def start_system(self):
        """Start the edge IoT system"""
        # Start edge scheduler
        self.edge_engine.start_scheduler()
        
        # Start device monitoring
        self.device_manager.start_monitoring()
        
        logger.info("Edge IoT system started")
    
    def stop_system(self):
        """Stop the edge IoT system"""
        # Stop edge scheduler
        self.edge_engine.stop_scheduler()
        
        # Stop device monitoring
        self.device_manager.stop_monitoring()
        
        # Disconnect MQTT
        self.device_manager.mqtt_manager.disconnect()
        
        logger.info("Edge IoT system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        device_stats = self.device_manager.get_device_statistics()
        edge_stats = self.edge_engine.get_node_status()
        job_stats = self.edge_engine.get_job_statistics()
        
        return {
            "timestamp": datetime.now(),
            "devices": device_stats,
            "edge_nodes": edge_stats,
            "jobs": job_stats,
            "system_health": {
                "device_availability": device_stats["device_availability"],
                "edge_utilization": edge_stats["active_jobs"] / max(1, edge_stats["total_capacity"]),
                "job_success_rate": job_stats["success_rate"]
            }
        }

# Streamlit Integration Functions

def initialize_edge_iot_system():
    """Initialize edge IoT system"""
    if 'edge_iot_system' not in st.session_state:
        st.session_state.edge_iot_system = EdgeIoTSystem()
        st.session_state.edge_iot_system.start_system()
    
    return st.session_state.edge_iot_system

def render_edge_iot_dashboard():
    """Render edge computing and IoT dashboard"""
    st.header("🌐 Edge Computing & IoT Integration")
    
    edge_iot = initialize_edge_iot_system()
    
    # Get system status
    system_status = edge_iot.get_system_status()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("IoT Devices", system_status["devices"]["total_devices"])
        st.metric("Online Devices", system_status["devices"]["online_devices"])
    
    with col2:
        st.metric("Edge Nodes", system_status["edge_nodes"]["total_nodes"])
        st.metric("Active Jobs", system_status["edge_nodes"]["active_jobs"])
    
    with col3:
        availability = system_status["devices"]["device_availability"]
        st.metric("Device Availability", f"{availability:.1%}")
        
        success_rate = system_status["jobs"]["success_rate"]
        st.metric("Job Success Rate", f"{success_rate:.1%}")
    
    with col4:
        data_generated = system_status["devices"]["total_data_generated_mb"]
        st.metric("Data Generated", f"{data_generated:.1f} MB")
        
        avg_battery = system_status["devices"]["avg_battery_level"]
        st.metric("Avg Battery", f"{avg_battery:.1f}%")
    
    # System health indicators
    device_health = system_status["system_health"]["device_availability"]
    edge_health = system_status["system_health"]["edge_utilization"]
    job_health = system_status["system_health"]["job_success_rate"]
    
    overall_health = (device_health + (1 - edge_health) + job_health) / 3
    
    if overall_health >= 0.8:
        st.success("🟢 System Operating Optimally")
    elif overall_health >= 0.6:
        st.warning("🟡 System Performance Degraded")
    else:
        st.error("🔴 System Issues Detected")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📱 IoT Devices",
        "🖥️ Edge Nodes",
        "⚡ Job Processing",
        "📊 Real-time Data",
        "🔍 Risk Analysis",
        "⚙️ System Management"
    ])
    
    with tab1:
        st.subheader("IoT Device Management")
        
        # Device actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Create Sample Devices"):
                with st.spinner("Creating sample IoT devices..."):
                    # Create sample devices
                    sample_devices = [
                        IoTDevice(
                            device_id="temp_sensor_001",
                            name="Temperature Sensor 001",
                            device_type=DeviceType.SENSOR,
                            manufacturer="SensorTech",
                            model="TempPro-X1",
                            firmware_version="v2.1.3",
                            ip_address="192.168.1.100",
                            port=1883,
                            protocol="MQTT",
                            location="Building A - Floor 1",
                            latitude=40.7128,
                            longitude=-74.0060,
                            capabilities=["temperature", "humidity"],
                            sensors=["DS18B20", "DHT22"],
                            max_data_rate=0.1,  # 0.1 MB/s
                            storage_capacity=0.5,  # 0.5 GB
                            compute_power=0.1,  # 0.1 GFLOPS
                            status=DeviceStatus.ONLINE,
                            battery_level=85.0,
                            temperature=22.5
                        ),
                        IoTDevice(
                            device_id="camera_001",
                            name="Security Camera 001",
                            device_type=DeviceType.CAMERA,
                            manufacturer="VisionCorp",
                            model="SecureCam-Pro",
                            firmware_version="v1.5.2",
                            ip_address="192.168.1.101",
                            port=8080,
                            protocol="HTTP",
                            location="Building A - Entrance",
                            latitude=40.7129,
                            longitude=-74.0061,
                            capabilities=["video_recording", "motion_detection", "night_vision"],
                            sensors=["CMOS", "PIR"],
                            max_data_rate=5.0,  # 5 MB/s
                            storage_capacity=32.0,  # 32 GB
                            compute_power=2.5,  # 2.5 GFLOPS
                            status=DeviceStatus.ONLINE,
                            temperature=35.2
                        ),
                        IoTDevice(
                            device_id="gateway_001",
                            name="Edge Gateway 001",
                            device_type=DeviceType.GATEWAY,
                            manufacturer="EdgeTech",
                            model="Gateway-X200",
                            firmware_version="v3.0.1",
                            ip_address="192.168.1.10",
                            port=8883,
                            protocol="MQTT",
                            location="Building A - Network Room",
                            latitude=40.7130,
                            longitude=-74.0059,
                            capabilities=["data_aggregation", "edge_processing", "protocol_translation"],
                            sensors=["system_monitor"],
                            max_data_rate=100.0,  # 100 MB/s
                            storage_capacity=1000.0,  # 1 TB
                            compute_power=50.0,  # 50 GFLOPS
                            status=DeviceStatus.ONLINE,
                            temperature=45.8
                        )
                    ]
                    
                    for device in sample_devices:
                        edge_iot.device_manager.register_device(device)
                    
                    st.success(f"Created {len(sample_devices)} sample devices!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Start MQTT Connection"):
                success = edge_iot.device_manager.connect_mqtt()
                if success:
                    st.success("MQTT connection established!")
                else:
                    st.error("Failed to connect to MQTT broker")
        
        with col3:
            if st.button("📡 Start Device Monitoring"):
                edge_iot.device_manager.start_monitoring()
                st.success("Device monitoring started!")
        
        # Device list
        devices = list(edge_iot.device_manager.devices.values())
        
        if devices:
            st.subheader("Registered Devices")
            
            device_data = []
            for device in devices:
                status_icons = {
                    DeviceStatus.ONLINE: "🟢",
                    DeviceStatus.OFFLINE: "🔴",
                    DeviceStatus.MAINTENANCE: "🟡",
                    DeviceStatus.ERROR: "❌",
                    DeviceStatus.UPDATING: "🔄",
                    DeviceStatus.UNKNOWN: "⚪"
                }
                
                device_data.append({
                    'Device ID': device.device_id[:15] + '...' if len(device.device_id) > 15 else device.device_id,
                    'Name': device.name,
                    'Type': device.device_type.value.replace('_', ' ').title(),
                    'Status': f"{status_icons.get(device.status, '⚪')} {device.status.value}",
                    'Location': device.location[:20] + '...' if len(device.location) > 20 else device.location,
                    'Battery': f"{device.battery_level:.1f}%" if device.battery_level else "N/A",
                    'Data (MB)': f"{device.data_generated_mb:.2f}"
                })
            
            device_df = pd.DataFrame(device_data)
            st.dataframe(device_df, use_container_width=True)
            
            # Device details
            selected_device_id = st.selectbox("Select Device for Details", [d.device_id for d in devices])
            
            if selected_device_id:
                selected_device = edge_iot.device_manager.devices[selected_device_id]
                
                with st.expander(f"Device Details: {selected_device.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Manufacturer:** {selected_device.manufacturer}")
                        st.write(f"**Model:** {selected_device.model}")
                        st.write(f"**Firmware:** {selected_device.firmware_version}")
                        st.write(f"**IP Address:** {selected_device.ip_address}:{selected_device.port}")
                        st.write(f"**Protocol:** {selected_device.protocol}")
                        st.write(f"**Capabilities:** {', '.join(selected_device.capabilities)}")
                    
                    with col2:
                        st.write(f"**Status:** {selected_device.status.value}")
                        if selected_device.battery_level:
                            st.write(f"**Battery:** {selected_device.battery_level:.1f}%")
                        if selected_device.temperature:
                            st.write(f"**Temperature:** {selected_device.temperature:.1f}°C")
                        st.write(f"**Uptime:** {selected_device.uptime_seconds / 3600:.1f} hours")
                        st.write(f"**Data Generated:** {selected_device.data_generated_mb:.2f} MB")
                        if selected_device.last_seen:
                            st.write(f"**Last Seen:** {selected_device.last_seen.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No IoT devices registered. Click 'Create Sample Devices' to get started.")
    
    with tab2:
        st.subheader("Edge Node Management")
        
        # Edge node actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏭 Create Sample Edge Nodes"):
                with st.spinner("Creating sample edge nodes..."):
                    # Create sample edge nodes
                    sample_nodes = [
                        EdgeNode(
                            node_id="edge_server_001",
                            name="Edge Server 001",
                            location=EdgeLocation.EDGE_SERVER,
                            description="Primary edge server for building A",
                            cpu_cores=8,
                            ram_gb=32.0,
                            storage_gb=1000.0,
                            gpu_available=True,
                            ip_address="192.168.1.50",
                            port=8080,
                            bandwidth_mbps=1000.0,
                            latitude=40.7128,
                            longitude=-74.0060,
                            region="North America - East",
                            max_concurrent_jobs=20,
                            supported_frameworks=["tensorflow", "pytorch", "onnx", "scikit-learn"],
                            ai_accelerators=["nvidia_t4", "intel_ncs"],
                            status=DeviceStatus.ONLINE,
                            cpu_usage=25.3,
                            memory_usage=45.2,
                            storage_usage=15.7,
                            temperature=42.5
                        ),
                        EdgeNode(
                            node_id="gateway_edge_001",
                            name="Gateway Edge 001",
                            location=EdgeLocation.EDGE_GATEWAY,
                            description="Edge gateway for IoT device aggregation",
                            cpu_cores=4,
                            ram_gb=8.0,
                            storage_gb=256.0,
                            gpu_available=False,
                            ip_address="192.168.1.20",
                            port=8080,
                            bandwidth_mbps=100.0,
                            latitude=40.7130,
                            longitude=-74.0062,
                            region="North America - East",
                            max_concurrent_jobs=10,
                            supported_frameworks=["tensorflow_lite", "onnx"],
                            ai_accelerators=[],
                            status=DeviceStatus.ONLINE,
                            cpu_usage=55.8,
                            memory_usage=72.1,
                            storage_usage=28.4,
                            temperature=38.2
                        ),
                        EdgeNode(
                            node_id="regional_hub_001",
                            name="Regional Hub 001",
                            location=EdgeLocation.REGIONAL_HUB,
                            description="Regional processing hub",
                            cpu_cores=16,
                            ram_gb=128.0,
                            storage_gb=5000.0,
                            gpu_available=True,
                            ip_address="10.0.1.100",
                            port=8080,
                            bandwidth_mbps=10000.0,
                            latitude=40.7500,
                            longitude=-74.0000,
                            region="North America - East",
                            max_concurrent_jobs=50,
                            supported_frameworks=["tensorflow", "pytorch", "onnx", "scikit-learn", "xgboost"],
                            ai_accelerators=["nvidia_v100", "nvidia_a100"],
                            status=DeviceStatus.ONLINE,
                            cpu_usage=15.2,
                            memory_usage=30.5,
                            storage_usage=8.9,
                            temperature=35.7
                        )
                    ]
                    
                    for node in sample_nodes:
                        edge_iot.edge_engine.register_edge_node(node)
                    
                    st.success(f"Created {len(sample_nodes)} edge nodes!")
                    st.rerun()
        
        with col2:
            if st.button("▶️ Start Job Scheduler"):
                edge_iot.edge_engine.start_scheduler()
                st.success("Edge job scheduler started!")
        
        with col3:
            if st.button("📊 Update Node Status"):
                # Simulate status updates
                for node in edge_iot.edge_engine.edge_nodes.values():
                    node.cpu_usage = np.random.uniform(10, 80)
                    node.memory_usage = np.random.uniform(20, 90)
                    node.temperature = np.random.uniform(30, 60)
                    node.last_heartbeat = datetime.now()
                
                st.success("Node status updated!")
                st.rerun()
        
        # Edge nodes list
        edge_nodes = list(edge_iot.edge_engine.edge_nodes.values())
        
        if edge_nodes:
            st.subheader("Edge Nodes")
            
            node_data = []
            for node in edge_nodes:
                status_icons = {
                    DeviceStatus.ONLINE: "🟢",
                    DeviceStatus.OFFLINE: "🔴",
                    DeviceStatus.MAINTENANCE: "🟡",
                    DeviceStatus.ERROR: "❌",
                    DeviceStatus.UNKNOWN: "⚪"
                }
                
                node_data.append({
                    'Node ID': node.node_id[:15] + '...' if len(node.node_id) > 15 else node.node_id,
                    'Name': node.name,
                    'Location': node.location.value.replace('_', ' ').title(),
                    'Status': f"{status_icons.get(node.status, '⚪')} {node.status.value}",
                    'CPU Usage': f"{node.cpu_usage:.1f}%",
                    'Memory Usage': f"{node.memory_usage:.1f}%",
                    'Active Jobs': f"{node.active_jobs}/{node.max_concurrent_jobs}",
                    'Jobs Completed': node.jobs_completed
                })
            
            node_df = pd.DataFrame(node_data)
            st.dataframe(node_df, use_container_width=True)
            
            # Node performance visualization
            st.subheader("Node Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU usage chart
                cpu_data = [(node.name, node.cpu_usage) for node in edge_nodes]
                cpu_df = pd.DataFrame(cpu_data, columns=['Node', 'CPU Usage (%)'])
                
                fig_cpu = px.bar(cpu_df, x='Node', y='CPU Usage (%)', 
                               title='CPU Usage by Node')
                fig_cpu.update_xaxis(tickangle=45)
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col2:
                # Memory usage chart
                memory_data = [(node.name, node.memory_usage) for node in edge_nodes]
                memory_df = pd.DataFrame(memory_data, columns=['Node', 'Memory Usage (%)'])
                
                fig_memory = px.bar(memory_df, x='Node', y='Memory Usage (%)', 
                                  title='Memory Usage by Node')
                fig_memory.update_xaxis(tickangle=45)
                st.plotly_chart(fig_memory, use_container_width=True)
        else:
            st.info("No edge nodes registered. Click 'Create Sample Edge Nodes' to get started.")
    
    with tab3:
        st.subheader("Job Processing")
        
        # Job submission
        with st.expander("Submit New Job"):
            col1, col2 = st.columns(2)
            
            with col1:
                job_name = st.text_input("Job Name")
                job_type = st.selectbox("Job Type", ["risk_analysis", "data_processing", "inference", "generic"])
                priority = st.selectbox("Priority", [p.value for p in DataPriority])
            
            with col2:
                cpu_cores = st.number_input("CPU Cores Required", min_value=1, max_value=16, value=2)
                memory_mb = st.number_input("Memory (MB)", min_value=256, max_value=8192, value=1024)
                max_time = st.number_input("Max Execution Time (s)", min_value=30, max_value=600, value=120)
            
            if st.button("Submit Job"):
                if job_name:
                    job = EdgeJob(
                        job_id="",
                        name=job_name,
                        job_type=job_type,
                        priority=DataPriority(priority),
                        input_data={"sample_data": list(range(100))},
                        cpu_cores_required=cpu_cores,
                        memory_mb_required=memory_mb,
                        max_execution_time_seconds=max_time,
                        user_id="streamlit_user"
                    )
                    
                    job_id = edge_iot.edge_engine.submit_job(job)
                    st.success(f"Job submitted with ID: {job_id}")
                    st.rerun()
        
        # Job statistics
        job_stats = edge_iot.edge_engine.get_job_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pending Jobs", job_stats["pending_jobs"])
        
        with col2:
            st.metric("Running Jobs", job_stats["running_jobs"])
        
        with col3:
            st.metric("Completed Jobs", job_stats["successful_jobs"])
        
        with col4:
            success_rate = job_stats["success_rate"]
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        # Job execution time chart
        if job_stats["successful_jobs"] > 0:
            st.subheader("Job Performance")
            
            # Simulate job execution times for visualization
            completed_jobs = list(edge_iot.edge_engine.completed_jobs.values())
            
            if completed_jobs:
                job_performance_data = []
                for job in completed_jobs[-20:]:  # Last 20 jobs
                    job_performance_data.append({
                        'Job Name': job.name[:20] + '...' if len(job.name) > 20 else job.name,
                        'Job Type': job.job_type,
                        'Execution Time (s)': job.execution_time_seconds,
                        'Status': job.status
                    })
                
                if job_performance_data:
                    job_df = pd.DataFrame(job_performance_data)
                    
                    fig_jobs = px.bar(job_df, x='Job Name', y='Execution Time (s)', 
                                    color='Job Type', title='Recent Job Execution Times')
                    fig_jobs.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_jobs, use_container_width=True)
    
    with tab4:
        st.subheader("Real-time IoT Data")
        
        # Data refresh controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_range = st.selectbox("Time Range", ["Last 5 minutes", "Last 15 minutes", "Last 1 hour"])
            minutes = {"Last 5 minutes": 5, "Last 15 minutes": 15, "Last 1 hour": 60}[time_range]
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col3:
            if st.button("📡 Simulate Data"):
                # Simulate incoming IoT data
                devices = list(edge_iot.device_manager.devices.values())
                
                if devices:
                    for _ in range(10):  # Generate 10 sample messages
                        device = np.random.choice(devices)
                        
                        # Generate sample data based on device type
                        if device.device_type == DeviceType.SENSOR:
                            payload = {
                                "temperature": np.random.normal(22, 5),
                                "humidity": np.random.normal(50, 10),
                                "pressure": np.random.normal(1013, 20)
                            }
                        elif device.device_type == DeviceType.CAMERA:
                            payload = {
                                "motion_detected": np.random.choice([True, False]),
                                "confidence": np.random.uniform(0.7, 0.99),
                                "frame_count": np.random.randint(1, 100)
                            }
                        else:
                            payload = {
                                "value": np.random.uniform(0, 100),
                                "status": "normal"
                            }
                        
                        message = IoTDataMessage(
                            message_id=str(uuid.uuid4()),
                            device_id=device.device_id,
                            timestamp=datetime.now(),
                            data_type="sensor_data",
                            payload=payload,
                            size_bytes=len(json.dumps(payload))
                        )
                        
                        edge_iot.device_manager.data_buffer.append(message)
                    
                    st.success("Generated 10 sample data messages!")
                    st.rerun()
        
        # Recent data display
        recent_data = edge_iot.device_manager.get_all_recent_data(minutes)
        
        if recent_data:
            st.write(f"**Recent Data Messages ({len(recent_data)} messages):**")
            
            # Data summary by device
            device_message_counts = defaultdict(int)
            total_data_size = 0
            
            for message in recent_data:
                device_message_counts[message.device_id] += 1
                total_data_size += message.size_bytes
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Message count by device
                if device_message_counts:
                    device_data = list(device_message_counts.items())
                    device_msg_df = pd.DataFrame(device_data, columns=['Device ID', 'Message Count'])
                    
                    fig_devices = px.pie(device_msg_df, values='Message Count', names='Device ID',
                                       title='Messages by Device')
                    st.plotly_chart(fig_devices, use_container_width=True)
            
            with col2:
                # Data timeline
                timeline_data = []
                for message in recent_data[-50:]:  # Last 50 messages
                    timeline_data.append({
                        'Timestamp': message.timestamp,
                        'Device ID': message.device_id[:10] + '...' if len(message.device_id) > 10 else message.device_id,
                        'Size (bytes)': message.size_bytes
                    })
                
                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    fig_timeline = px.scatter(timeline_df, x='Timestamp', y='Size (bytes)', 
                                            color='Device ID', title='Data Message Timeline')
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Recent messages table
            st.write("**Recent Messages:**")
            
            message_data = []
            for message in recent_data[-20:]:  # Last 20 messages
                message_data.append({
                    'Timestamp': message.timestamp.strftime('%H:%M:%S'),
                    'Device ID': message.device_id[:12] + '...' if len(message.device_id) > 12 else message.device_id,
                    'Type': message.data_type,
                    'Size (bytes)': message.size_bytes,
                    'Priority': message.priority.value
                })
            
            if message_data:
                message_df = pd.DataFrame(message_data)
                st.dataframe(message_df, use_container_width=True)
        else:
            st.info("No recent data messages. Click 'Simulate Data' to generate sample messages.")
    
    with tab5:
        st.subheader("Edge Risk Analysis")
        
        # Risk analysis controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_time_range = st.selectbox("Analysis Time Range", 
                                             ["Last 10 minutes", "Last 30 minutes", "Last 1 hour"],
                                             key="risk_time_range")
            analysis_minutes = {"Last 10 minutes": 10, "Last 30 minutes": 30, "Last 1 hour": 60}[analysis_time_range]
        
        with col2:
            risk_model = st.selectbox("Risk Model", ["default", "anomaly_detection", "pattern_analysis"])
        
        with col3:
            if st.button("🔍 Run Risk Analysis"):
                with st.spinner("Running edge risk analysis..."):
                    # Get recent data for analysis
                    analysis_data = edge_iot.device_manager.get_all_recent_data(analysis_minutes)
                    
                    if analysis_data:
                        # Run risk analysis
                        analysis_result = edge_iot.risk_analyzer.analyze_iot_data(
                            analysis_data, risk_model
                        )
                        
                        if "error" not in analysis_result:
                            st.success("Risk analysis completed!")
                            
                            # Display results
                            st.write("**Analysis Results:**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if "results" in analysis_result and "risk_score" in analysis_result["results"]:
                                    risk_score = analysis_result["results"]["risk_score"]
                                    st.metric("Risk Score", f"{risk_score:.3f}")
                                    
                                    risk_level = analysis_result["results"].get("risk_level", "unknown")
                                    st.metric("Risk Level", risk_level.title())
                            
                            with col2:
                                messages_analyzed = analysis_result["messages_analyzed"]
                                st.metric("Messages Analyzed", messages_analyzed)
                                
                                execution_time = analysis_result["execution_time"]
                                st.metric("Analysis Time", f"{execution_time:.2f}s")
                            
                            with col3:
                                if "results" in analysis_result and "confidence" in analysis_result["results"]:
                                    confidence = analysis_result["results"]["confidence"]
                                    st.metric("Confidence", f"{confidence:.1%}")
                                
                                node_id = analysis_result.get("node_id", "unknown")
                                st.metric("Processed on", node_id[:10] + '...' if len(node_id) > 10 else node_id)
                            
                            # Risk factors
                            if "results" in analysis_result and "factors" in analysis_result["results"]:
                                factors = analysis_result["results"]["factors"]
                                st.write("**Risk Factors:**")
                                for factor in factors:
                                    st.write(f"• {factor.replace('_', ' ').title()}")
                        else:
                            st.error(f"Analysis failed: {analysis_result['error']}")
                    else:
                        st.warning("No data available for analysis")
        
        # Anomaly detection
        st.subheader("Anomaly Detection")
        
        if st.button("🚨 Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                recent_data = edge_iot.device_manager.get_all_recent_data(30)  # Last 30 minutes
                
                if recent_data:
                    anomalies = edge_iot.risk_analyzer.detect_anomalies(recent_data)
                    
                    if anomalies:
                        st.write(f"**Found {len(anomalies)} anomalies:**")
                        
                        anomaly_data = []
                        for anomaly in anomalies:
                            severity_icons = {
                                "critical": "🔴",
                                "high": "🟠",
                                "medium": "🟡",
                                "low": "🟢"
                            }
                            
                            anomaly_data.append({
                                'Timestamp': anomaly['timestamp'].strftime('%H:%M:%S'),
                                'Device ID': anomaly['device_id'][:12] + '...' if len(anomaly['device_id']) > 12 else anomaly['device_id'],
                                'Type': anomaly['anomaly_type'].replace('_', ' ').title(),
                                'Value': anomaly['value'],
                                'Severity': f"{severity_icons.get(anomaly['severity'], '⚪')} {anomaly['severity'].title()}"
                            })
                        
                        anomaly_df = pd.DataFrame(anomaly_data)
                        st.dataframe(anomaly_df, use_container_width=True)
                    else:
                        st.success("No anomalies detected in recent data")
                else:
                    st.info("No data available for anomaly detection")
        
        # Risk trends
        risk_trends = edge_iot.risk_analyzer.get_risk_trends(24)  # Last 24 hours
        
        if "avg_risk_score" in risk_trends:
            st.subheader("Risk Trends (24 hours)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_risk = risk_trends["avg_risk_score"]
                st.metric("Average Risk", f"{avg_risk:.3f}")
            
            with col2:
                max_risk = risk_trends["max_risk_score"]
                st.metric("Maximum Risk", f"{max_risk:.3f}")
            
            with col3:
                trend = risk_trends["risk_trend"]
                trend_icon = "📈" if trend == "increasing" else "📊"
                st.metric("Trend", f"{trend_icon} {trend.title()}")
            
            with col4:
                total_analyses = risk_trends["total_analyses"]
                st.metric("Analyses Run", total_analyses)
    
    with tab6:
        st.subheader("System Management")
        
        # System controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart System"):
                edge_iot.stop_system()
                time.sleep(1)
                edge_iot.start_system()
                st.success("System restarted!")
        
        with col2:
            if st.button("🧹 Clear Data Buffer"):
                edge_iot.device_manager.data_buffer.clear()
                st.success("Data buffer cleared!")
        
        with col3:
            if st.button("📊 Generate System Report"):
                system_status = edge_iot.get_system_status()
                
                report_json = json.dumps(system_status, indent=2, default=str)
                
                st.download_button(
                    label="Download System Report",
                    data=report_json,
                    file_name=f"edge_iot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime='application/json'
                )
        
        # System configuration
        st.subheader("Configuration")
        
        with st.expander("MQTT Configuration"):
            mqtt_host = st.text_input("MQTT Broker Host", value="localhost")
            mqtt_port = st.number_input("MQTT Broker Port", min_value=1, max_value=65535, value=1883)
            
            if st.button("Update MQTT Config"):
                edge_iot.device_manager.mqtt_manager.broker_host = mqtt_host
                edge_iot.device_manager.mqtt_manager.broker_port = mqtt_port
                st.success("MQTT configuration updated!")
        
        with st.expander("Edge Computing Configuration"):
            max_workers = st.number_input("Max Worker Threads", min_value=1, max_value=50, value=20)
            job_timeout = st.number_input("Default Job Timeout (s)", min_value=30, max_value=3600, value=300)
            
            if st.button("Update Edge Config"):
                # Update configuration (in real implementation)
                st.success("Edge computing configuration updated!")
        
        # System health monitoring
        st.subheader("System Health")
        
        health_data = []
        
        # Device health
        device_stats = system_status["devices"]
        health_data.append({
            'Component': 'IoT Devices',
            'Status': '🟢 Healthy' if device_stats["device_availability"] > 0.8 else '🟡 Warning',
            'Metric': f"{device_stats['online_devices']}/{device_stats['total_devices']} online",
            'Health Score': device_stats["device_availability"]
        })
        
        # Edge nodes health
        edge_stats = system_status["edge_nodes"]
        edge_utilization = edge_stats["active_jobs"] / max(1, edge_stats["total_capacity"])
        health_data.append({
            'Component': 'Edge Nodes',
            'Status': '🟢 Healthy' if edge_utilization < 0.8 else '🟡 Warning',
            'Metric': f"{edge_stats['active_jobs']}/{edge_stats['total_capacity']} jobs",
            'Health Score': 1 - edge_utilization
        })
        
        # Job processing health
        job_stats = system_status["jobs"]
        health_data.append({
            'Component': 'Job Processing',
            'Status': '🟢 Healthy' if job_stats["success_rate"] > 0.9 else '🟡 Warning',
            'Metric': f"{job_stats['success_rate']:.1%} success rate",
            'Health Score': job_stats["success_rate"]
        })
        
        health_df = pd.DataFrame(health_data)
        st.dataframe(health_df, use_container_width=True)

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing edge computing and IoT integration...")
    
    # Initialize system
    edge_iot = EdgeIoTSystem()
    edge_iot.start_system()
    
    # Create sample device
    sample_device = IoTDevice(
        device_id="test_sensor_001",
        name="Test Temperature Sensor",
        device_type=DeviceType.SENSOR,
        manufacturer="TestCorp",
        model="TempSensor-X1",
        firmware_version="v1.0.0",
        ip_address="192.168.1.200",
        port=1883,
        protocol="MQTT",
        location="Test Lab",
        capabilities=["temperature", "humidity"],
        sensors=["DS18B20"],
        status=DeviceStatus.ONLINE,
        battery_level=90.0
    )
    
    edge_iot.device_manager.register_device(sample_device)
    print(f"Registered device: {sample_device.name}")
    
    # Create sample edge node
    sample_node = EdgeNode(
        node_id="test_edge_001",
        name="Test Edge Server",
        location=EdgeLocation.EDGE_SERVER,
        description="Test edge server",
        cpu_cores=4,
        ram_gb=16.0,
        storage_gb=500.0,
        ip_address="192.168.1.100",
        port=8080,
        bandwidth_mbps=1000.0,
        latitude=40.7128,
        longitude=-74.0060,
        region="Test Region",
        status=DeviceStatus.ONLINE
    )
    
    edge_iot.edge_engine.register_edge_node(sample_node)
    print(f"Registered edge node: {sample_node.name}")
    
    # Submit test job
    test_job = EdgeJob(
        job_id="",
        name="Test Risk Analysis Job",
        job_type="risk_analysis",
        priority=DataPriority.HIGH,
        input_data={"test_data": [1, 2, 3, 4, 5]},
        cpu_cores_required=1,
        memory_mb_required=512
    )
    
    job_id = edge_iot.edge_engine.submit_job(test_job)
    print(f"Submitted job with ID: {job_id}")
    
    # Wait for job completion
    time.sleep(3)
    
    # Get system status
    status = edge_iot.get_system_status()
    print(f"System status - Devices: {status['devices']['total_devices']}, Nodes: {status['edge_nodes']['total_nodes']}")
    
    # Stop system
    edge_iot.stop_system()
    
    print("Edge computing and IoT integration test completed!")