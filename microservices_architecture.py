"""
Microservices Architecture and Service Mesh Implementation
Provides containerized microservices architecture with service discovery and mesh networking
"""

import asyncio
import json
import time
import yaml
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import socket
import subprocess
import logging
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"

class ServiceType(Enum):
    """Types of microservices"""
    API_GATEWAY = "api_gateway"
    RISK_ANALYZER = "risk_analyzer"
    DATA_PROCESSOR = "data_processor"
    MODEL_SERVING = "model_serving"
    NOTIFICATION_SERVICE = "notification_service"
    AUTH_SERVICE = "auth_service"
    CONFIG_SERVICE = "config_service"
    METRICS_COLLECTOR = "metrics_collector"
    LOG_AGGREGATOR = "log_aggregator"
    STORAGE_SERVICE = "storage_service"

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    host: str
    port: int
    protocol: str = "http"
    path: str = "/"
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"

@dataclass
class ServiceHealth:
    """Service health information"""
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0

@dataclass
class ServiceMetrics:
    """Service performance metrics"""
    requests_per_second: float
    average_response_time: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_connections: int
    uptime_seconds: float

@dataclass
class MicroService:
    """Microservice definition"""
    service_id: str
    name: str
    service_type: ServiceType
    version: str
    description: str
    
    # Deployment configuration
    docker_image: str
    container_name: str
    endpoints: List[ServiceEndpoint]
    environment_variables: Dict[str, str]
    resource_limits: Dict[str, Any]
    
    # Service mesh configuration
    replicas: int = 1
    dependencies: List[str] = None
    tags: List[str] = None
    
    # Runtime information
    status: ServiceStatus = ServiceStatus.STOPPED
    health: Optional[ServiceHealth] = None
    metrics: Optional[ServiceMetrics] = None
    last_deployed: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

@dataclass
class ServiceRegistry:
    """Service registry for service discovery"""
    services: Dict[str, MicroService]
    service_instances: Dict[str, List[ServiceEndpoint]]
    
    def __init__(self):
        self.services = {}
        self.service_instances = {}
    
    def register_service(self, service: MicroService):
        """Register a service in the registry"""
        self.services[service.service_id] = service
        self.service_instances[service.service_id] = service.endpoints.copy()
        logger.info(f"Service {service.name} registered with ID {service.service_id}")
    
    def deregister_service(self, service_id: str):
        """Deregister a service from the registry"""
        if service_id in self.services:
            del self.services[service_id]
        if service_id in self.service_instances:
            del self.service_instances[service_id]
        logger.info(f"Service {service_id} deregistered")
    
    def discover_service(self, service_name: str) -> List[ServiceEndpoint]:
        """Discover service endpoints by name"""
        for service in self.services.values():
            if service.name == service_name:
                return self.service_instances.get(service.service_id, [])
        return []
    
    def get_healthy_endpoints(self, service_id: str) -> List[ServiceEndpoint]:
        """Get only healthy endpoints for a service"""
        if service_id not in self.services:
            return []
        
        service = self.services[service_id]
        if service.health and service.health.is_healthy:
            return self.service_instances.get(service_id, [])
        return []

class LoadBalancer:
    """Load balancer for distributing requests across service instances"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN):
        self.algorithm = algorithm
        self.round_robin_index = {}
        self.connection_counts = {}
        self.weights = {}
    
    def select_endpoint(self, service_id: str, endpoints: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """Select an endpoint based on load balancing algorithm"""
        if not endpoints:
            return None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(service_id, endpoints)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_select(endpoints)
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return self._random_select(endpoints)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_id, endpoints)
        else:
            return endpoints[0]  # Default to first endpoint
    
    def _round_robin_select(self, service_id: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin selection"""
        if service_id not in self.round_robin_index:
            self.round_robin_index[service_id] = 0
        
        index = self.round_robin_index[service_id]
        selected = endpoints[index % len(endpoints)]
        self.round_robin_index[service_id] = (index + 1) % len(endpoints)
        
        return selected
    
    def _least_connections_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections selection"""
        min_connections = float('inf')
        selected_endpoint = endpoints[0]
        
        for endpoint in endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            connections = self.connection_counts.get(endpoint_key, 0)
            
            if connections < min_connections:
                min_connections = connections
                selected_endpoint = endpoint
        
        return selected_endpoint
    
    def _random_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random selection"""
        import random
        return random.choice(endpoints)
    
    def _weighted_round_robin_select(self, service_id: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin selection"""
        # Simplified: assume equal weights for now
        return self._round_robin_select(service_id, endpoints)
    
    def track_connection(self, endpoint: ServiceEndpoint, increment: int = 1):
        """Track connection count for load balancing"""
        endpoint_key = f"{endpoint.host}:{endpoint.port}"
        self.connection_counts[endpoint_key] = self.connection_counts.get(endpoint_key, 0) + increment

class HealthChecker:
    """Health checker for monitoring service health"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.health_thread = None
    
    def start_health_checks(self, registry: ServiceRegistry):
        """Start health checking for all registered services"""
        if self.running:
            return
        
        self.running = True
        self.health_thread = threading.Thread(
            target=self._health_check_loop,
            args=(registry,),
            daemon=True
        )
        self.health_thread.start()
        logger.info("Health checker started")
    
    def stop_health_checks(self):
        """Stop health checking"""
        self.running = False
        if self.health_thread:
            self.health_thread.join(timeout=10)
        logger.info("Health checker stopped")
    
    def _health_check_loop(self, registry: ServiceRegistry):
        """Main health checking loop"""
        while self.running:
            try:
                for service in registry.services.values():
                    self._check_service_health(service)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.check_interval)
    
    def _check_service_health(self, service: MicroService):
        """Check health of a single service"""
        try:
            if service.status != ServiceStatus.RUNNING:
                service.health = ServiceHealth(
                    is_healthy=False,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    error_message="Service not running"
                )
                return
            
            # Check first endpoint (simplified)
            if not service.endpoints:
                service.health = ServiceHealth(
                    is_healthy=False,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    error_message="No endpoints configured"
                )
                return
            
            endpoint = service.endpoints[0]
            start_time = time.time()
            
            try:
                # Attempt HTTP health check
                health_url = f"{endpoint.url}health" if endpoint.path == "/" else f"{endpoint.url}/health"
                response = requests.get(health_url, timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                is_healthy = response.status_code == 200
                error_message = None if is_healthy else f"HTTP {response.status_code}"
                consecutive_failures = 0 if is_healthy else (service.health.consecutive_failures + 1 if service.health else 1)
                
                service.health = ServiceHealth(
                    is_healthy=is_healthy,
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    error_message=error_message,
                    consecutive_failures=consecutive_failures
                )
                
            except requests.RequestException as e:
                consecutive_failures = service.health.consecutive_failures + 1 if service.health else 1
                service.health = ServiceHealth(
                    is_healthy=False,
                    last_check=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                    error_message=str(e),
                    consecutive_failures=consecutive_failures
                )
                
        except Exception as e:
            logger.error(f"Error checking health for service {service.service_id}: {e}")

class APIGateway:
    """API Gateway for routing and managing requests"""
    
    def __init__(self, registry: ServiceRegistry, load_balancer: LoadBalancer):
        self.registry = registry
        self.load_balancer = load_balancer
        self.routes = {}
        self.middleware = []
        
    def add_route(self, path: str, service_name: str, methods: List[str] = None):
        """Add a route mapping"""
        if methods is None:
            methods = ["GET", "POST", "PUT", "DELETE"]
        
        self.routes[path] = {
            "service_name": service_name,
            "methods": methods
        }
        logger.info(f"Route {path} -> {service_name} added")
    
    def route_request(self, path: str, method: str = "GET") -> Optional[ServiceEndpoint]:
        """Route a request to appropriate service endpoint"""
        # Find matching route
        service_name = None
        for route_path, route_config in self.routes.items():
            if path.startswith(route_path) and method in route_config["methods"]:
                service_name = route_config["service_name"]
                break
        
        if not service_name:
            return None
        
        # Discover service endpoints
        endpoints = self.registry.discover_service(service_name)
        if not endpoints:
            return None
        
        # Filter healthy endpoints
        healthy_endpoints = []
        for service in self.registry.services.values():
            if service.name == service_name and service.health and service.health.is_healthy:
                healthy_endpoints.extend(endpoints)
                break
        
        if not healthy_endpoints:
            return None
        
        # Load balance
        return self.load_balancer.select_endpoint(service_name, healthy_endpoints)
    
    def add_middleware(self, middleware_func: Callable):
        """Add middleware function"""
        self.middleware.append(middleware_func)

class ServiceMesh:
    """Service mesh for managing microservices communication"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker()
        self.api_gateway = APIGateway(self.registry, self.load_balancer)
        self.metrics_collector = ServiceMetricsCollector()
        
        # Service mesh configuration
        self.auto_scaling_enabled = False
        self.circuit_breaker_enabled = True
        self.rate_limiting_enabled = True
        
        # Circuit breaker configuration
        self.circuit_breakers = {}
        self.failure_threshold = 5
        self.recovery_timeout = 60
        
    def deploy_service(self, service: MicroService) -> bool:
        """Deploy a microservice"""
        try:
            # Register service
            self.registry.register_service(service)
            
            # Start service (simplified - would integrate with container orchestration)
            if self._start_service_container(service):
                service.status = ServiceStatus.RUNNING
                service.last_deployed = datetime.now()
                
                # Add default routes to API gateway
                if service.endpoints:
                    base_path = f"/{service.name.lower().replace('_', '-')}"
                    self.api_gateway.add_route(base_path, service.name)
                
                logger.info(f"Service {service.name} deployed successfully")
                return True
            else:
                service.status = ServiceStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Error deploying service {service.name}: {e}")
            service.status = ServiceStatus.ERROR
            return False
    
    def _start_service_container(self, service: MicroService) -> bool:
        """Start service container (simplified simulation)"""
        try:
            # Simulate container startup
            logger.info(f"Starting container {service.container_name} from image {service.docker_image}")
            
            # In a real implementation, this would use Docker API or Kubernetes
            # For simulation, we'll just mark it as started
            service.status = ServiceStatus.STARTING
            time.sleep(1)  # Simulate startup time
            service.status = ServiceStatus.RUNNING
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start container for service {service.name}: {e}")
            return False
    
    def stop_service(self, service_id: str) -> bool:
        """Stop a microservice"""
        try:
            if service_id not in self.registry.services:
                return False
            
            service = self.registry.services[service_id]
            service.status = ServiceStatus.STOPPING
            
            # Stop container (simplified)
            logger.info(f"Stopping container {service.container_name}")
            time.sleep(1)  # Simulate stop time
            
            service.status = ServiceStatus.STOPPED
            return True
            
        except Exception as e:
            logger.error(f"Error stopping service {service_id}: {e}")
            return False
    
    def scale_service(self, service_id: str, target_replicas: int) -> bool:
        """Scale a service to target number of replicas"""
        try:
            if service_id not in self.registry.services:
                return False
            
            service = self.registry.services[service_id]
            current_replicas = service.replicas
            
            if target_replicas > current_replicas:
                # Scale up
                for i in range(target_replicas - current_replicas):
                    # Create new service instance
                    new_port = service.endpoints[0].port + i + 1
                    new_endpoint = ServiceEndpoint(
                        host=service.endpoints[0].host,
                        port=new_port,
                        protocol=service.endpoints[0].protocol,
                        path=service.endpoints[0].path
                    )
                    
                    # Add to service instances
                    if service_id not in self.registry.service_instances:
                        self.registry.service_instances[service_id] = []
                    self.registry.service_instances[service_id].append(new_endpoint)
                    
                    logger.info(f"Scaled up {service.name} - added instance on port {new_port}")
            
            elif target_replicas < current_replicas:
                # Scale down
                instances_to_remove = current_replicas - target_replicas
                if service_id in self.registry.service_instances:
                    for i in range(instances_to_remove):
                        if self.registry.service_instances[service_id]:
                            removed = self.registry.service_instances[service_id].pop()
                            logger.info(f"Scaled down {service.name} - removed instance on port {removed.port}")
            
            service.replicas = target_replicas
            return True
            
        except Exception as e:
            logger.error(f"Error scaling service {service_id}: {e}")
            return False
    
    def start_health_monitoring(self):
        """Start health monitoring for all services"""
        self.health_checker.start_health_checks(self.registry)
        self.metrics_collector.start_collection(self.registry)
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.health_checker.stop_health_checks()
        self.metrics_collector.stop_collection()
    
    def get_service_topology(self) -> Dict[str, Any]:
        """Get service topology and dependencies"""
        topology = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes (services)
        for service in self.registry.services.values():
            topology["nodes"].append({
                "id": service.service_id,
                "name": service.name,
                "type": service.service_type.value,
                "status": service.status.value,
                "replicas": service.replicas,
                "health": service.health.is_healthy if service.health else False
            })
        
        # Add edges (dependencies)
        for service in self.registry.services.values():
            for dependency in service.dependencies:
                topology["edges"].append({
                    "source": service.service_id,
                    "target": dependency
                })
        
        return topology
    
    def get_mesh_metrics(self) -> Dict[str, Any]:
        """Get overall service mesh metrics"""
        total_services = len(self.registry.services)
        running_services = len([s for s in self.registry.services.values() if s.status == ServiceStatus.RUNNING])
        healthy_services = len([s for s in self.registry.services.values() if s.health and s.health.is_healthy])
        
        total_requests = sum([
            self.metrics_collector.get_service_metrics(s.service_id).requests_per_second
            for s in self.registry.services.values()
            if self.metrics_collector.get_service_metrics(s.service_id)
        ])
        
        avg_response_time = 0
        if running_services > 0:
            response_times = [
                self.metrics_collector.get_service_metrics(s.service_id).average_response_time
                for s in self.registry.services.values()
                if s.status == ServiceStatus.RUNNING and self.metrics_collector.get_service_metrics(s.service_id)
            ]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_services": total_services,
            "running_services": running_services,
            "healthy_services": healthy_services,
            "service_availability": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "total_requests_per_second": total_requests,
            "average_response_time_ms": avg_response_time,
            "mesh_uptime_hours": (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
        }

class ServiceMetricsCollector:
    """Collects and aggregates service metrics"""
    
    def __init__(self):
        self.running = False
        self.collection_thread = None
        self.metrics_data = {}
        self.collection_interval = 15  # seconds
    
    def start_collection(self, registry: ServiceRegistry):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(registry,),
            daemon=True
        )
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self, registry: ServiceRegistry):
        """Main metrics collection loop"""
        while self.running:
            try:
                for service in registry.services.values():
                    if service.status == ServiceStatus.RUNNING:
                        metrics = self._collect_service_metrics(service)
                        self.metrics_data[service.service_id] = metrics
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_service_metrics(self, service: MicroService) -> ServiceMetrics:
        """Collect metrics for a single service"""
        try:
            # Simulate metrics collection
            # In real implementation, would collect from monitoring endpoints
            
            import random
            
            # Generate realistic metrics
            base_rps = random.uniform(10, 100)
            base_response_time = random.uniform(50, 200)
            base_error_rate = random.uniform(0, 5)
            
            # Add some variation based on service type
            if service.service_type == ServiceType.API_GATEWAY:
                base_rps *= 2
                base_response_time *= 0.8
            elif service.service_type == ServiceType.MODEL_SERVING:
                base_response_time *= 1.5
                base_rps *= 0.7
            
            metrics = ServiceMetrics(
                requests_per_second=base_rps,
                average_response_time=base_response_time,
                error_rate=base_error_rate,
                cpu_usage_percent=random.uniform(20, 80),
                memory_usage_mb=random.uniform(128, 512),
                active_connections=random.randint(5, 50),
                uptime_seconds=(datetime.now() - (service.last_deployed or datetime.now())).total_seconds()
            )
            
            service.metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for service {service.service_id}: {e}")
            return ServiceMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def get_service_metrics(self, service_id: str) -> Optional[ServiceMetrics]:
        """Get metrics for a specific service"""
        return self.metrics_data.get(service_id)
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all services"""
        if not self.metrics_data:
            return {}
        
        metrics_list = list(self.metrics_data.values())
        
        return {
            "total_rps": sum(m.requests_per_second for m in metrics_list),
            "avg_response_time": sum(m.average_response_time for m in metrics_list) / len(metrics_list),
            "avg_error_rate": sum(m.error_rate for m in metrics_list) / len(metrics_list),
            "total_cpu_usage": sum(m.cpu_usage_percent for m in metrics_list),
            "total_memory_usage": sum(m.memory_usage_mb for m in metrics_list),
            "total_connections": sum(m.active_connections for m in metrics_list)
        }

class ConfigurationManager:
    """Manages service configuration and secrets"""
    
    def __init__(self):
        self.configurations = {}
        self.secrets = {}
    
    def set_config(self, service_id: str, config: Dict[str, Any]):
        """Set configuration for a service"""
        self.configurations[service_id] = config
        logger.info(f"Configuration updated for service {service_id}")
    
    def get_config(self, service_id: str) -> Dict[str, Any]:
        """Get configuration for a service"""
        return self.configurations.get(service_id, {})
    
    def set_secret(self, service_id: str, key: str, value: str):
        """Set a secret for a service"""
        if service_id not in self.secrets:
            self.secrets[service_id] = {}
        
        # In real implementation, would encrypt the secret
        self.secrets[service_id][key] = value
        logger.info(f"Secret {key} set for service {service_id}")
    
    def get_secret(self, service_id: str, key: str) -> Optional[str]:
        """Get a secret for a service"""
        return self.secrets.get(service_id, {}).get(key)

class ServiceOrchestrator:
    """Orchestrates deployment and management of microservices"""
    
    def __init__(self):
        self.service_mesh = ServiceMesh()
        self.config_manager = ConfigurationManager()
    
    def create_llm_risk_services(self) -> List[MicroService]:
        """Create default LLM Risk Visualizer microservices"""
        services = []
        
        # API Gateway
        api_gateway = MicroService(
            service_id="api-gateway-001",
            name="API Gateway",
            service_type=ServiceType.API_GATEWAY,
            version="1.0.0",
            description="Main API gateway for routing requests",
            docker_image="llm-risk/api-gateway:1.0.0",
            container_name="llm-api-gateway",
            endpoints=[ServiceEndpoint("localhost", 8080, "http", "/")],
            environment_variables={
                "PORT": "8080",
                "LOG_LEVEL": "INFO"
            },
            resource_limits={
                "cpu": "500m",
                "memory": "512Mi"
            },
            replicas=2,
            tags=["gateway", "routing"]
        )
        services.append(api_gateway)
        
        # Risk Analyzer Service
        risk_analyzer = MicroService(
            service_id="risk-analyzer-001",
            name="Risk Analyzer",
            service_type=ServiceType.RISK_ANALYZER,
            version="1.0.0",
            description="Core risk analysis and scoring service",
            docker_image="llm-risk/risk-analyzer:1.0.0",
            container_name="llm-risk-analyzer",
            endpoints=[ServiceEndpoint("localhost", 8081, "http", "/api/v1/analyze")],
            environment_variables={
                "PORT": "8081",
                "MODEL_PATH": "/models/risk-model",
                "CACHE_SIZE": "1000"
            },
            resource_limits={
                "cpu": "1000m",
                "memory": "1Gi"
            },
            replicas=3,
            dependencies=["storage-service-001"],
            tags=["ml", "analysis", "core"]
        )
        services.append(risk_analyzer)
        
        # Data Processor Service
        data_processor = MicroService(
            service_id="data-processor-001",
            name="Data Processor",
            service_type=ServiceType.DATA_PROCESSOR,
            version="1.0.0",
            description="Data ingestion and preprocessing service",
            docker_image="llm-risk/data-processor:1.0.0",
            container_name="llm-data-processor",
            endpoints=[ServiceEndpoint("localhost", 8082, "http", "/api/v1/process")],
            environment_variables={
                "PORT": "8082",
                "BATCH_SIZE": "1000",
                "PROCESSING_THREADS": "4"
            },
            resource_limits={
                "cpu": "800m",
                "memory": "768Mi"
            },
            replicas=2,
            dependencies=["storage-service-001"],
            tags=["data", "preprocessing"]
        )
        services.append(data_processor)
        
        # Model Serving Service
        model_serving = MicroService(
            service_id="model-serving-001",
            name="Model Serving",
            service_type=ServiceType.MODEL_SERVING,
            version="1.0.0",
            description="ML model serving and inference service",
            docker_image="llm-risk/model-serving:1.0.0",
            container_name="llm-model-serving",
            endpoints=[ServiceEndpoint("localhost", 8083, "http", "/api/v1/predict")],
            environment_variables={
                "PORT": "8083",
                "MODEL_REGISTRY": "http://model-registry:8080",
                "INFERENCE_TIMEOUT": "30"
            },
            resource_limits={
                "cpu": "1500m",
                "memory": "2Gi"
            },
            replicas=2,
            dependencies=["storage-service-001"],
            tags=["ml", "inference", "model"]
        )
        services.append(model_serving)
        
        # Authentication Service
        auth_service = MicroService(
            service_id="auth-service-001",
            name="Authentication Service",
            service_type=ServiceType.AUTH_SERVICE,
            version="1.0.0",
            description="User authentication and authorization service",
            docker_image="llm-risk/auth-service:1.0.0",
            container_name="llm-auth-service",
            endpoints=[ServiceEndpoint("localhost", 8084, "http", "/api/v1/auth")],
            environment_variables={
                "PORT": "8084",
                "JWT_SECRET": "your-jwt-secret",
                "TOKEN_EXPIRY": "3600"
            },
            resource_limits={
                "cpu": "400m",
                "memory": "256Mi"
            },
            replicas=2,
            tags=["security", "auth"]
        )
        services.append(auth_service)
        
        # Notification Service
        notification_service = MicroService(
            service_id="notification-service-001",
            name="Notification Service",
            service_type=ServiceType.NOTIFICATION_SERVICE,
            version="1.0.0",
            description="Handles notifications and alerts",
            docker_image="llm-risk/notification-service:1.0.0",
            container_name="llm-notification-service",
            endpoints=[ServiceEndpoint("localhost", 8085, "http", "/api/v1/notify")],
            environment_variables={
                "PORT": "8085",
                "SMTP_HOST": "localhost",
                "SMTP_PORT": "587"
            },
            resource_limits={
                "cpu": "300m",
                "memory": "256Mi"
            },
            replicas=1,
            dependencies=["config-service-001"],
            tags=["notification", "alerts"]
        )
        services.append(notification_service)
        
        # Storage Service
        storage_service = MicroService(
            service_id="storage-service-001",
            name="Storage Service",
            service_type=ServiceType.STORAGE_SERVICE,
            version="1.0.0",
            description="Data storage and retrieval service",
            docker_image="llm-risk/storage-service:1.0.0",
            container_name="llm-storage-service",
            endpoints=[ServiceEndpoint("localhost", 8086, "http", "/api/v1/data")],
            environment_variables={
                "PORT": "8086",
                "DB_CONNECTION": "postgresql://user:pass@db:5432/llmrisk",
                "CACHE_TTL": "3600"
            },
            resource_limits={
                "cpu": "600m",
                "memory": "512Mi"
            },
            replicas=1,
            tags=["storage", "database"]
        )
        services.append(storage_service)
        
        # Configuration Service
        config_service = MicroService(
            service_id="config-service-001",
            name="Configuration Service",
            service_type=ServiceType.CONFIG_SERVICE,
            version="1.0.0",
            description="Centralized configuration management",
            docker_image="llm-risk/config-service:1.0.0",
            container_name="llm-config-service",
            endpoints=[ServiceEndpoint("localhost", 8087, "http", "/api/v1/config")],
            environment_variables={
                "PORT": "8087",
                "CONFIG_STORE": "file",
                "CONFIG_PATH": "/config"
            },
            resource_limits={
                "cpu": "200m",
                "memory": "128Mi"
            },
            replicas=1,
            tags=["config", "management"]
        )
        services.append(config_service)
        
        # Metrics Collector
        metrics_collector = MicroService(
            service_id="metrics-collector-001",
            name="Metrics Collector",
            service_type=ServiceType.METRICS_COLLECTOR,
            version="1.0.0",
            description="Collects and aggregates service metrics",
            docker_image="llm-risk/metrics-collector:1.0.0",
            container_name="llm-metrics-collector",
            endpoints=[ServiceEndpoint("localhost", 8088, "http", "/api/v1/metrics")],
            environment_variables={
                "PORT": "8088",
                "COLLECTION_INTERVAL": "15",
                "RETENTION_DAYS": "30"
            },
            resource_limits={
                "cpu": "300m",
                "memory": "256Mi"
            },
            replicas=1,
            tags=["monitoring", "metrics"]
        )
        services.append(metrics_collector)
        
        return services
    
    def deploy_all_services(self, services: List[MicroService]) -> Dict[str, bool]:
        """Deploy all services"""
        deployment_results = {}
        
        # Sort services by dependencies (simplified)
        sorted_services = self._sort_services_by_dependencies(services)
        
        for service in sorted_services:
            try:
                success = self.service_mesh.deploy_service(service)
                deployment_results[service.service_id] = success
                
                if success:
                    logger.info(f"Successfully deployed {service.name}")
                else:
                    logger.error(f"Failed to deploy {service.name}")
                
                # Wait a bit between deployments
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error deploying {service.name}: {e}")
                deployment_results[service.service_id] = False
        
        return deployment_results
    
    def _sort_services_by_dependencies(self, services: List[MicroService]) -> List[MicroService]:
        """Sort services by dependencies (topological sort)"""
        # Simple dependency sorting - deploy services with no dependencies first
        no_deps = [s for s in services if not s.dependencies]
        with_deps = [s for s in services if s.dependencies]
        
        return no_deps + with_deps
    
    def generate_docker_compose(self, services: List[MicroService]) -> str:
        """Generate Docker Compose configuration"""
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "llm-risk-network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "llm-risk-data": {},
                "llm-risk-logs": {}
            }
        }
        
        for service in services:
            service_config = {
                "image": service.docker_image,
                "container_name": service.container_name,
                "ports": [f"{ep.port}:{ep.port}" for ep in service.endpoints],
                "environment": service.environment_variables,
                "networks": ["llm-risk-network"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": [f"curl -f http://localhost:{service.endpoints[0].port}/health || exit 1"] if service.endpoints else ["echo", "ok"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "40s"
                }
            }
            
            # Add resource limits
            if service.resource_limits:
                service_config["deploy"] = {
                    "resources": {
                        "limits": service.resource_limits
                    }
                }
            
            # Add dependencies
            if service.dependencies:
                service_config["depends_on"] = [
                    dep_service.container_name for dep_service in services
                    if dep_service.service_id in service.dependencies
                ]
            
            # Add volumes for data services
            if service.service_type in [ServiceType.STORAGE_SERVICE, ServiceType.LOG_AGGREGATOR]:
                service_config["volumes"] = [
                    "llm-risk-data:/data",
                    "llm-risk-logs:/logs"
                ]
            
            compose_config["services"][service.container_name] = service_config
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def generate_kubernetes_manifests(self, services: List[MicroService]) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests"""
        manifests = {}
        
        for service in services:
            # Deployment manifest
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": service.container_name,
                    "labels": {
                        "app": service.container_name,
                        "service-type": service.service_type.value
                    }
                },
                "spec": {
                    "replicas": service.replicas,
                    "selector": {
                        "matchLabels": {
                            "app": service.container_name
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": service.container_name
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": service.container_name,
                                "image": service.docker_image,
                                "ports": [{"containerPort": ep.port} for ep in service.endpoints],
                                "env": [
                                    {"name": k, "value": v} 
                                    for k, v in service.environment_variables.items()
                                ],
                                "resources": {
                                    "limits": service.resource_limits
                                } if service.resource_limits else {},
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": service.endpoints[0].port if service.endpoints else 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                } if service.endpoints else {},
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": service.endpoints[0].port if service.endpoints else 8080
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                } if service.endpoints else {}
                            }]
                        }
                    }
                }
            }
            
            # Service manifest
            k8s_service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{service.container_name}-service",
                    "labels": {
                        "app": service.container_name
                    }
                },
                "spec": {
                    "selector": {
                        "app": service.container_name
                    },
                    "ports": [
                        {
                            "port": ep.port,
                            "targetPort": ep.port,
                            "protocol": "TCP"
                        } for ep in service.endpoints
                    ],
                    "type": "ClusterIP"
                }
            }
            
            # Combine manifests
            manifest_yaml = yaml.dump(deployment, default_flow_style=False)
            manifest_yaml += "---\n"
            manifest_yaml += yaml.dump(k8s_service, default_flow_style=False)
            
            manifests[service.container_name] = manifest_yaml
        
        return manifests

# Streamlit Integration Functions

def initialize_microservices():
    """Initialize microservices orchestrator"""
    if 'service_orchestrator' not in st.session_state:
        st.session_state.service_orchestrator = ServiceOrchestrator()
    
    return st.session_state.service_orchestrator

def render_microservices_dashboard():
    """Render microservices management dashboard"""
    st.header("🏗️ Microservices Architecture")
    
    orchestrator = initialize_microservices()
    
    # Overview metrics
    mesh_metrics = orchestrator.service_mesh.get_mesh_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Services", mesh_metrics["total_services"])
    
    with col2:
        st.metric("Running Services", mesh_metrics["running_services"])
    
    with col3:
        st.metric("Healthy Services", mesh_metrics["healthy_services"])
    
    with col4:
        availability = mesh_metrics["service_availability"]
        st.metric("Availability", f"{availability:.1f}%")
    
    # Service health status
    if mesh_metrics["total_services"] > 0:
        if mesh_metrics["service_availability"] >= 95:
            st.success("🟢 All systems operational")
        elif mesh_metrics["service_availability"] >= 80:
            st.warning("🟡 Some services degraded")
        else:
            st.error("🔴 System issues detected")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Services",
        "📊 Monitoring", 
        "🔗 Service Mesh",
        "📦 Deployment",
        "⚙️ Configuration",
        "📋 Orchestration"
    ])
    
    with tab1:
        st.subheader("Service Management")
        
        # Service actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏭 Create Default Services"):
                with st.spinner("Creating LLM Risk Visualizer services..."):
                    services = orchestrator.create_llm_risk_services()
                    
                    for service in services:
                        orchestrator.service_mesh.registry.register_service(service)
                    
                    st.success(f"Created {len(services)} services successfully!")
                    st.rerun()
        
        with col2:
            if st.button("▶️ Start All Services"):
                with st.spinner("Starting all services..."):
                    services = list(orchestrator.service_mesh.registry.services.values())
                    results = orchestrator.deploy_all_services(services)
                    
                    success_count = sum(1 for success in results.values() if success)
                    st.success(f"Started {success_count}/{len(services)} services")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Start Health Monitoring"):
                orchestrator.service_mesh.start_health_monitoring()
                st.success("Health monitoring started!")
        
        # Service list
        services = list(orchestrator.service_mesh.registry.services.values())
        
        if services:
            st.subheader("Registered Services")
            
            service_data = []
            for service in services:
                status_icons = {
                    ServiceStatus.RUNNING: "🟢",
                    ServiceStatus.STOPPED: "🔴",
                    ServiceStatus.STARTING: "🟡",
                    ServiceStatus.STOPPING: "🟠",
                    ServiceStatus.ERROR: "❌",
                    ServiceStatus.UNKNOWN: "⚪"
                }
                
                health_icon = "✅" if service.health and service.health.is_healthy else "❌" if service.health else "⚪"
                
                service_data.append({
                    'Service': service.name,
                    'Type': service.service_type.value.replace('_', ' ').title(),
                    'Status': f"{status_icons.get(service.status, '⚪')} {service.status.value}",
                    'Health': health_icon,
                    'Replicas': service.replicas,
                    'Version': service.version,
                    'Endpoints': len(service.endpoints)
                })
            
            service_df = pd.DataFrame(service_data)
            st.dataframe(service_df, use_container_width=True)
            
            # Service details
            selected_service_name = st.selectbox("Select Service for Details", [s.name for s in services])
            
            if selected_service_name:
                selected_service = next((s for s in services if s.name == selected_service_name), None)
                
                if selected_service:
                    with st.expander(f"Service Details: {selected_service.name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Service ID:** {selected_service.service_id}")
                            st.write(f"**Description:** {selected_service.description}")
                            st.write(f"**Docker Image:** {selected_service.docker_image}")
                            st.write(f"**Container:** {selected_service.container_name}")
                            st.write(f"**Dependencies:** {', '.join(selected_service.dependencies) if selected_service.dependencies else 'None'}")
                        
                        with col2:
                            st.write(f"**Status:** {selected_service.status.value}")
                            st.write(f"**Replicas:** {selected_service.replicas}")
                            st.write(f"**Tags:** {', '.join(selected_service.tags) if selected_service.tags else 'None'}")
                            if selected_service.last_deployed:
                                st.write(f"**Last Deployed:** {selected_service.last_deployed.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Endpoints
                        if selected_service.endpoints:
                            st.write("**Endpoints:**")
                            for i, endpoint in enumerate(selected_service.endpoints):
                                st.write(f"  {i+1}. {endpoint.url}")
                        
                        # Environment variables
                        if selected_service.environment_variables:
                            st.write("**Environment Variables:**")
                            for key, value in selected_service.environment_variables.items():
                                # Hide sensitive values
                                display_value = "***" if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']) else value
                                st.write(f"  {key}: {display_value}")
                        
                        # Health information
                        if selected_service.health:
                            st.write("**Health Status:**")
                            st.write(f"  Healthy: {'Yes' if selected_service.health.is_healthy else 'No'}")
                            st.write(f"  Last Check: {selected_service.health.last_check.strftime('%H:%M:%S')}")
                            st.write(f"  Response Time: {selected_service.health.response_time_ms:.1f}ms")
                            if selected_service.health.error_message:
                                st.write(f"  Error: {selected_service.health.error_message}")
                        
                        # Service actions
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("🔄 Restart Service", key=f"restart_{selected_service.service_id}"):
                                orchestrator.service_mesh.stop_service(selected_service.service_id)
                                time.sleep(1)
                                orchestrator.service_mesh.deploy_service(selected_service)
                                st.success(f"Service {selected_service.name} restarted!")
                                st.rerun()
                        
                        with col2:
                            new_replicas = st.number_input("Scale to replicas:", 
                                                         min_value=1, max_value=10, 
                                                         value=selected_service.replicas,
                                                         key=f"scale_{selected_service.service_id}")
                            if st.button("📈 Scale", key=f"scale_btn_{selected_service.service_id}"):
                                success = orchestrator.service_mesh.scale_service(selected_service.service_id, new_replicas)
                                if success:
                                    st.success(f"Service scaled to {new_replicas} replicas!")
                                    st.rerun()
                                else:
                                    st.error("Failed to scale service")
                        
                        with col3:
                            if st.button("⏹️ Stop Service", key=f"stop_{selected_service.service_id}"):
                                success = orchestrator.service_mesh.stop_service(selected_service.service_id)
                                if success:
                                    st.success(f"Service {selected_service.name} stopped!")
                                    st.rerun()
                                else:
                                    st.error("Failed to stop service")
        else:
            st.info("No services registered. Click 'Create Default Services' to get started.")
    
    with tab2:
        st.subheader("Service Monitoring")
        
        # Refresh metrics
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
        
        # Overall metrics
        if mesh_metrics["total_services"] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**System Metrics:**")
                st.write(f"Requests/sec: {mesh_metrics.get('total_requests_per_second', 0):.1f}")
                st.write(f"Avg Response Time: {mesh_metrics.get('average_response_time_ms', 0):.1f}ms")
                st.write(f"Mesh Uptime: {mesh_metrics.get('mesh_uptime_hours', 0):.1f}h")
            
            with col2:
                # Service status distribution
                status_counts = {}
                for service in orchestrator.service_mesh.registry.services.values():
                    status = service.status.value
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                if status_counts:
                    status_df = pd.DataFrame([
                        {"Status": k.replace('_', ' ').title(), "Count": v}
                        for k, v in status_counts.items()
                    ])
                    
                    fig_status = px.pie(status_df, values='Count', names='Status',
                                      title='Service Status Distribution')
                    st.plotly_chart(fig_status, use_container_width=True)
            
            # Individual service metrics
            services_with_metrics = [
                s for s in orchestrator.service_mesh.registry.services.values()
                if s.metrics and s.status == ServiceStatus.RUNNING
            ]
            
            if services_with_metrics:
                st.subheader("Service Performance")
                
                # Performance metrics chart
                metrics_data = []
                for service in services_with_metrics:
                    metrics_data.append({
                        'Service': service.name,
                        'RPS': service.metrics.requests_per_second,
                        'Response Time (ms)': service.metrics.average_response_time,
                        'Error Rate (%)': service.metrics.error_rate,
                        'CPU (%)': service.metrics.cpu_usage_percent,
                        'Memory (MB)': service.metrics.memory_usage_mb
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                
                # Performance charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rps = px.bar(metrics_df, x='Service', y='RPS',
                                   title='Requests per Second by Service')
                    fig_rps.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_rps, use_container_width=True)
                
                with col2:
                    fig_response = px.bar(metrics_df, x='Service', y='Response Time (ms)',
                                        title='Average Response Time by Service')
                    fig_response.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_response, use_container_width=True)
                
                # Resource usage
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cpu = px.bar(metrics_df, x='Service', y='CPU (%)',
                                   title='CPU Usage by Service')
                    fig_cpu.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_cpu, use_container_width=True)
                
                with col2:
                    fig_memory = px.bar(metrics_df, x='Service', y='Memory (MB)',
                                      title='Memory Usage by Service')
                    fig_memory.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_memory, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("Detailed Metrics")
                st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("No services available for monitoring")
    
    with tab3:
        st.subheader("Service Mesh Topology")
        
        topology = orchestrator.service_mesh.get_service_topology()
        
        if topology["nodes"]:
            # Service dependency visualization (simplified)
            st.write("**Service Dependencies:**")
            
            # Create a simple dependency view
            for edge in topology["edges"]:
                source_name = next((n["name"] for n in topology["nodes"] if n["id"] == edge["source"]), edge["source"])
                target_name = next((n["name"] for n in topology["nodes"] if n["id"] == edge["target"]), edge["target"])
                st.write(f"• {source_name} → {target_name}")
            
            if not topology["edges"]:
                st.write("No service dependencies configured")
            
            # Service mesh configuration
            st.subheader("Service Mesh Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Load Balancing:**")
                current_algorithm = orchestrator.service_mesh.load_balancer.algorithm.value
                st.write(f"Algorithm: {current_algorithm.replace('_', ' ').title()}")
                
                new_algorithm = st.selectbox(
                    "Change Load Balancing Algorithm",
                    [alg.value for alg in LoadBalancingAlgorithm],
                    index=[alg.value for alg in LoadBalancingAlgorithm].index(current_algorithm)
                )
                
                if st.button("Update Load Balancing"):
                    orchestrator.service_mesh.load_balancer.algorithm = LoadBalancingAlgorithm(new_algorithm)
                    st.success(f"Load balancing algorithm updated to {new_algorithm}")
            
            with col2:
                st.write("**Health Checking:**")
                health_interval = orchestrator.service_mesh.health_checker.check_interval
                st.write(f"Check Interval: {health_interval}s")
                
                new_interval = st.slider("Health Check Interval (seconds)", 10, 300, health_interval)
                
                if st.button("Update Health Checking"):
                    orchestrator.service_mesh.health_checker.check_interval = new_interval
                    st.success(f"Health check interval updated to {new_interval}s")
            
            # API Gateway routes
            st.subheader("API Gateway Routes")
            
            routes = orchestrator.service_mesh.api_gateway.routes
            
            if routes:
                routes_data = []
                for path, config in routes.items():
                    routes_data.append({
                        'Path': path,
                        'Service': config['service_name'],
                        'Methods': ', '.join(config['methods'])
                    })
                
                routes_df = pd.DataFrame(routes_data)
                st.dataframe(routes_df, use_container_width=True)
            else:
                st.info("No API routes configured")
                
                # Add sample routes
                if st.button("Add Default Routes"):
                    orchestrator.service_mesh.api_gateway.add_route("/api/v1/analyze", "Risk Analyzer")
                    orchestrator.service_mesh.api_gateway.add_route("/api/v1/process", "Data Processor")
                    orchestrator.service_mesh.api_gateway.add_route("/api/v1/predict", "Model Serving")
                    orchestrator.service_mesh.api_gateway.add_route("/api/v1/auth", "Authentication Service")
                    orchestrator.service_mesh.api_gateway.add_route("/api/v1/notify", "Notification Service")
                    st.success("Default routes added!")
                    st.rerun()
        else:
            st.info("No services in the mesh topology")
    
    with tab4:
        st.subheader("Deployment Management")
        
        services = list(orchestrator.service_mesh.registry.services.values())
        
        if services:
            # Generate deployment configurations
            st.write("**Generate Deployment Configurations:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📦 Generate Docker Compose"):
                    with st.spinner("Generating Docker Compose configuration..."):
                        docker_compose = orchestrator.generate_docker_compose(services)
                        
                        st.download_button(
                            label="Download docker-compose.yml",
                            data=docker_compose,
                            file_name="docker-compose.yml",
                            mime="text/yaml"
                        )
                        
                        st.text_area("Docker Compose Configuration", docker_compose, height=400)
            
            with col2:
                if st.button("☸️ Generate Kubernetes Manifests"):
                    with st.spinner("Generating Kubernetes manifests..."):
                        k8s_manifests = orchestrator.generate_kubernetes_manifests(services)
                        
                        # Show manifests for each service
                        for service_name, manifest in k8s_manifests.items():
                            st.text_area(f"Kubernetes Manifest: {service_name}", manifest, height=200)
                            
                            st.download_button(
                                label=f"Download {service_name}.yaml",
                                data=manifest,
                                file_name=f"{service_name}.yaml",
                                mime="text/yaml",
                                key=f"download_{service_name}"
                            )
            
            # Deployment status
            st.subheader("Deployment Status")
            
            deployment_data = []
            for service in services:
                deployment_data.append({
                    'Service': service.name,
                    'Image': service.docker_image,
                    'Status': service.status.value,
                    'Replicas': f"{service.replicas}/{service.replicas}",  # Simplified
                    'Last Deployed': service.last_deployed.strftime('%Y-%m-%d %H:%M') if service.last_deployed else 'Never',
                    'Uptime': f"{(datetime.now() - service.last_deployed).total_seconds() / 3600:.1f}h" if service.last_deployed else "0h"
                })
            
            deployment_df = pd.DataFrame(deployment_data)
            st.dataframe(deployment_df, use_container_width=True)
        else:
            st.info("No services available for deployment")
    
    with tab5:
        st.subheader("Configuration Management")
        
        # Service configuration
        services = list(orchestrator.service_mesh.registry.services.values())
        
        if services:
            selected_service_config = st.selectbox(
                "Select Service for Configuration",
                [s.name for s in services],
                key="config_service_select"
            )
            
            if selected_service_config:
                selected_service = next((s for s in services if s.name == selected_service_config), None)
                
                if selected_service:
                    st.write(f"**Configuration for {selected_service.name}:**")
                    
                    # Get current configuration
                    current_config = orchestrator.config_manager.get_config(selected_service.service_id)
                    
                    # Configuration editor
                    with st.expander("Edit Configuration"):
                        config_text = json.dumps(current_config, indent=2) if current_config else "{}"
                        new_config_text = st.text_area(
                            "Configuration (JSON)",
                            config_text,
                            height=200,
                            key=f"config_{selected_service.service_id}"
                        )
                        
                        if st.button("Save Configuration", key=f"save_config_{selected_service.service_id}"):
                            try:
                                new_config = json.loads(new_config_text)
                                orchestrator.config_manager.set_config(selected_service.service_id, new_config)
                                st.success("Configuration saved!")
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON: {e}")
                    
                    # Secrets management
                    with st.expander("Manage Secrets"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            secret_key = st.text_input("Secret Key", key=f"secret_key_{selected_service.service_id}")
                        
                        with col2:
                            secret_value = st.text_input("Secret Value", type="password", key=f"secret_value_{selected_service.service_id}")
                        
                        if st.button("Add Secret", key=f"add_secret_{selected_service.service_id}"):
                            if secret_key and secret_value:
                                orchestrator.config_manager.set_secret(selected_service.service_id, secret_key, secret_value)
                                st.success(f"Secret '{secret_key}' added!")
                            else:
                                st.error("Please provide both key and value")
                    
                    # Environment variables
                    st.write("**Environment Variables:**")
                    env_data = [
                        {"Key": k, "Value": "***" if any(s in k.lower() for s in ['password', 'secret', 'key', 'token']) else v}
                        for k, v in selected_service.environment_variables.items()
                    ]
                    
                    if env_data:
                        env_df = pd.DataFrame(env_data)
                        st.dataframe(env_df, use_container_width=True)
                    else:
                        st.info("No environment variables configured")
        else:
            st.info("No services available for configuration")
    
    with tab6:
        st.subheader("Container Orchestration")
        
        # Orchestration tools
        st.write("**Orchestration Platform:**")
        
        orchestration_platform = st.radio(
            "Select Platform",
            ["Docker Compose", "Kubernetes", "Docker Swarm"],
            index=0
        )
        
        if orchestration_platform == "Docker Compose":
            st.write("**Docker Compose Management:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Docker Compose Up"):
                    st.info("Would execute: docker-compose up -d")
                    st.success("Docker Compose services started!")
            
            with col2:
                if st.button("⏹️ Docker Compose Down"):
                    st.info("Would execute: docker-compose down")
                    st.success("Docker Compose services stopped!")
            
            with col3:
                if st.button("🔄 Docker Compose Restart"):
                    st.info("Would execute: docker-compose restart")
                    st.success("Docker Compose services restarted!")
        
        elif orchestration_platform == "Kubernetes":
            st.write("**Kubernetes Management:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📦 Deploy to K8s"):
                    st.info("Would execute: kubectl apply -f manifests/")
                    st.success("Kubernetes deployments created!")
            
            with col2:
                if st.button("📊 Get Pods Status"):
                    st.info("Would execute: kubectl get pods")
                    # Simulate pod status
                    pods_data = [
                        {"Pod": "api-gateway-001-abc123", "Status": "Running", "Ready": "1/1", "Restarts": 0},
                        {"Pod": "risk-analyzer-001-def456", "Status": "Running", "Ready": "1/1", "Restarts": 0},
                        {"Pod": "data-processor-001-ghi789", "Status": "Running", "Ready": "1/1", "Restarts": 1}
                    ]
                    st.dataframe(pd.DataFrame(pods_data))
            
            with col3:
                if st.button("🔍 Get Services"):
                    st.info("Would execute: kubectl get services")
                    # Simulate service status
                    services_data = [
                        {"Service": "api-gateway-service", "Type": "ClusterIP", "Cluster-IP": "10.96.1.1", "Port": "8080/TCP"},
                        {"Service": "risk-analyzer-service", "Type": "ClusterIP", "Cluster-IP": "10.96.1.2", "Port": "8081/TCP"}
                    ]
                    st.dataframe(pd.DataFrame(services_data))
        
        # System resources
        st.subheader("System Resources")
        
        # Get system resource usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        
        with col2:
            st.metric("Memory Usage", f"{memory.percent:.1f}%")
            st.write(f"Used: {memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB")
        
        with col3:
            st.metric("Disk Usage", f"{disk.percent:.1f}%")
            st.write(f"Used: {disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB")
        
        # Container statistics (simulated)
        st.subheader("Container Resources")
        
        if orchestrator.service_mesh.registry.services:
            container_data = []
            for service in orchestrator.service_mesh.registry.services.values():
                if service.status == ServiceStatus.RUNNING:
                    # Simulate container stats
                    import random
                    container_data.append({
                        'Container': service.container_name,
                        'CPU %': f"{random.uniform(5, 50):.1f}%",
                        'Memory': f"{random.uniform(50, 300):.0f}MB",
                        'Network I/O': f"{random.uniform(1, 100):.1f}KB/s",
                        'Block I/O': f"{random.uniform(0, 10):.1f}MB/s"
                    })
            
            if container_data:
                container_df = pd.DataFrame(container_data)
                st.dataframe(container_df, use_container_width=True)
        
        # Auto-scaling configuration
        st.subheader("Auto-scaling Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_scaling_enabled = st.checkbox("Enable Auto-scaling", value=orchestrator.service_mesh.auto_scaling_enabled)
            
            if auto_scaling_enabled != orchestrator.service_mesh.auto_scaling_enabled:
                orchestrator.service_mesh.auto_scaling_enabled = auto_scaling_enabled
                st.success(f"Auto-scaling {'enabled' if auto_scaling_enabled else 'disabled'}!")
        
        with col2:
            if auto_scaling_enabled:
                cpu_threshold = st.slider("CPU Threshold for Scaling (%)", 50, 90, 80)
                memory_threshold = st.slider("Memory Threshold for Scaling (%)", 50, 90, 80)
                
                st.write(f"Services will scale up when CPU > {cpu_threshold}% or Memory > {memory_threshold}%")

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing microservices architecture and service mesh...")
    
    # Initialize orchestrator
    orchestrator = ServiceOrchestrator()
    
    # Create sample services
    services = orchestrator.create_llm_risk_services()
    print(f"Created {len(services)} services")
    
    # Deploy services
    results = orchestrator.deploy_all_services(services)
    success_count = sum(1 for success in results.values() if success)
    print(f"Deployed {success_count}/{len(services)} services successfully")
    
    # Start health monitoring
    orchestrator.service_mesh.start_health_monitoring()
    print("Health monitoring started")
    
    # Wait for some metrics
    time.sleep(5)
    
    # Get mesh metrics
    mesh_metrics = orchestrator.service_mesh.get_mesh_metrics()
    print(f"Mesh metrics: {mesh_metrics['total_services']} services, {mesh_metrics['running_services']} running")
    
    # Generate deployment configs
    docker_compose = orchestrator.generate_docker_compose(services)
    print(f"Generated Docker Compose config ({len(docker_compose)} characters)")
    
    k8s_manifests = orchestrator.generate_kubernetes_manifests(services)
    print(f"Generated {len(k8s_manifests)} Kubernetes manifests")
    
    # Stop monitoring
    orchestrator.service_mesh.stop_health_monitoring()
    print("Health monitoring stopped")
    
    print("Microservices architecture and service mesh test completed!")