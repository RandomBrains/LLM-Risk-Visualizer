"""
High Availability and Load Balancing Architecture Module
Implements enterprise-grade HA architecture with automatic failover and load distribution
"""

import asyncio
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import hashlib
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import redis
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"

@dataclass
class ServiceNode:
    """Represents a service node in the cluster"""
    node_id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    response_time: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def utilization(self) -> float:
        return self.current_connections / self.max_connections if self.max_connections > 0 else 0.0

@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    interval: int = 30  # seconds
    timeout: int = 5    # seconds
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    path: str = "/health"
    expected_codes: List[int] = None
    
    def __post_init__(self):
        if self.expected_codes is None:
            self.expected_codes = [200]

class LoadBalancer:
    """Advanced load balancer with multiple strategies and health checking"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.nodes: Dict[str, ServiceNode] = {}
        self.healthy_nodes: List[str] = []
        self.current_index = 0
        self.session_affinity: Dict[str, str] = {}  # session_id -> node_id
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        # Health checking
        self.health_check_config = HealthCheckConfig()
        self.health_check_thread = None
        self.health_checking_active = False
        
        # Circuit breaker pattern
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    def add_node(self, node: ServiceNode) -> bool:
        """Add a service node to the load balancer"""
        try:
            self.nodes[node.node_id] = node
            
            # Initialize circuit breaker
            self.circuit_breakers[node.node_id] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': None,
                'timeout': 60  # seconds
            }
            
            # Add to healthy nodes if it passes initial health check
            if self._initial_health_check(node):
                self.healthy_nodes.append(node.node_id)
                logger.info(f"Node {node.node_id} added and marked as healthy")
            else:
                logger.warning(f"Node {node.node_id} added but marked as unhealthy")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add node {node.node_id}: {e}")
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a service node from the load balancer"""
        try:
            if node_id in self.nodes:
                del self.nodes[node_id]
                
                if node_id in self.healthy_nodes:
                    self.healthy_nodes.remove(node_id)
                
                if node_id in self.circuit_breakers:
                    del self.circuit_breakers[node_id]
                
                # Remove session affinities for this node
                sessions_to_remove = [
                    session_id for session_id, mapped_node_id in self.session_affinity.items()
                    if mapped_node_id == node_id
                ]
                for session_id in sessions_to_remove:
                    del self.session_affinity[session_id]
                
                logger.info(f"Node {node_id} removed from load balancer")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove node {node_id}: {e}")
            return False
    
    def get_next_node(self, session_id: Optional[str] = None, 
                     client_ip: Optional[str] = None) -> Optional[ServiceNode]:
        """Get the next available node based on load balancing strategy"""
        
        if not self.healthy_nodes:
            logger.warning("No healthy nodes available")
            return None
        
        # Check session affinity first
        if session_id and session_id in self.session_affinity:
            node_id = self.session_affinity[session_id]
            if node_id in self.healthy_nodes and self._is_circuit_closed(node_id):
                return self.nodes[node_id]
        
        # Select node based on strategy
        selected_node_id = None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_node_id = self._round_robin_selection()
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_node_id = self._weighted_round_robin_selection()
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_node_id = self._least_connections_selection()
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_node_id = self._least_response_time_selection()
        
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            selected_node_id = self._ip_hash_selection(client_ip)
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            selected_node_id = self._random_selection()
        
        if selected_node_id and self._is_circuit_closed(selected_node_id):
            # Create session affinity if session_id provided
            if session_id:
                self.session_affinity[session_id] = selected_node_id
            
            return self.nodes[selected_node_id]
        
        return None
    
    def _round_robin_selection(self) -> Optional[str]:
        """Round robin node selection"""
        if not self.healthy_nodes:
            return None
        
        node_id = self.healthy_nodes[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.healthy_nodes)
        return node_id
    
    def _weighted_round_robin_selection(self) -> Optional[str]:
        """Weighted round robin selection"""
        if not self.healthy_nodes:
            return None
        
        # Calculate total weight
        total_weight = sum(self.nodes[node_id].weight for node_id in self.healthy_nodes)
        
        if total_weight == 0:
            return self._round_robin_selection()
        
        # Generate random number and select based on weight
        rand_weight = random.randint(1, total_weight)
        current_weight = 0
        
        for node_id in self.healthy_nodes:
            current_weight += self.nodes[node_id].weight
            if rand_weight <= current_weight:
                return node_id
        
        return self.healthy_nodes[0]  # Fallback
    
    def _least_connections_selection(self) -> Optional[str]:
        """Select node with least connections"""
        if not self.healthy_nodes:
            return None
        
        return min(
            self.healthy_nodes,
            key=lambda node_id: self.nodes[node_id].current_connections
        )
    
    def _least_response_time_selection(self) -> Optional[str]:
        """Select node with least response time"""
        if not self.healthy_nodes:
            return None
        
        return min(
            self.healthy_nodes,
            key=lambda node_id: self.nodes[node_id].response_time
        )
    
    def _ip_hash_selection(self, client_ip: Optional[str]) -> Optional[str]:
        """Select node based on client IP hash"""
        if not self.healthy_nodes or not client_ip:
            return self._round_robin_selection()
        
        # Create hash of client IP
        ip_hash = hashlib.md5(client_ip.encode()).hexdigest()
        hash_int = int(ip_hash, 16)
        
        # Select node based on hash
        index = hash_int % len(self.healthy_nodes)
        return self.healthy_nodes[index]
    
    def _random_selection(self) -> Optional[str]:
        """Random node selection"""
        if not self.healthy_nodes:
            return None
        
        return random.choice(self.healthy_nodes)
    
    def _is_circuit_closed(self, node_id: str) -> bool:
        """Check if circuit breaker is closed (node is available)"""
        if node_id not in self.circuit_breakers:
            return True
        
        circuit = self.circuit_breakers[node_id]
        
        if circuit['state'] == 'closed':
            return True
        elif circuit['state'] == 'open':
            # Check if timeout has passed
            if circuit['last_failure_time']:
                time_since_failure = time.time() - circuit['last_failure_time']
                if time_since_failure >= circuit['timeout']:
                    # Move to half-open state
                    circuit['state'] = 'half_open'
                    return True
            return False
        elif circuit['state'] == 'half_open':
            return True
        
        return False
    
    def record_request_success(self, node_id: str, response_time: float):
        """Record successful request to update node metrics"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.response_time = (node.response_time + response_time) / 2  # Moving average
            node.failure_count = 0
            
            # Reset circuit breaker
            if node_id in self.circuit_breakers:
                circuit = self.circuit_breakers[node_id]
                circuit['state'] = 'closed'
                circuit['failure_count'] = 0
            
            # Update stats
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self._update_avg_response_time(response_time)
    
    def record_request_failure(self, node_id: str):
        """Record failed request to update node metrics"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.failure_count += 1
            
            # Update circuit breaker
            if node_id in self.circuit_breakers:
                circuit = self.circuit_breakers[node_id]
                circuit['failure_count'] += 1
                
                # Open circuit if failure threshold reached
                if circuit['failure_count'] >= 5:  # Configurable threshold
                    circuit['state'] = 'open'
                    circuit['last_failure_time'] = time.time()
                    logger.warning(f"Circuit breaker opened for node {node_id}")
            
            # Update stats
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        total_successful = self.stats['successful_requests']
        if total_successful > 0:
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (total_successful - 1) + response_time) 
                / total_successful
            )
    
    def start_health_checking(self):
        """Start background health checking"""
        if self.health_checking_active:
            return
        
        self.health_checking_active = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        logger.info("Health checking started")
    
    def stop_health_checking(self):
        """Stop background health checking"""
        self.health_checking_active = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        logger.info("Health checking stopped")
    
    def _health_check_loop(self):
        """Background health checking loop"""
        while self.health_checking_active:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_config.interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(5)
    
    def _perform_health_checks(self):
        """Perform health checks on all nodes"""
        for node_id, node in self.nodes.items():
            try:
                is_healthy = self._check_node_health(node)
                previous_status = node.status
                
                if is_healthy:
                    if node.status == NodeStatus.UNHEALTHY:
                        node.failure_count = 0
                    
                    node.status = NodeStatus.HEALTHY
                    
                    if node_id not in self.healthy_nodes:
                        self.healthy_nodes.append(node_id)
                        logger.info(f"Node {node_id} marked as healthy")
                
                else:
                    node.failure_count += 1
                    
                    if node.failure_count >= self.health_check_config.unhealthy_threshold:
                        node.status = NodeStatus.UNHEALTHY
                        
                        if node_id in self.healthy_nodes:
                            self.healthy_nodes.remove(node_id)
                            logger.warning(f"Node {node_id} marked as unhealthy")
                    else:
                        node.status = NodeStatus.DEGRADED
                
                node.last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Health check failed for node {node_id}: {e}")
                node.status = NodeStatus.UNHEALTHY
                if node_id in self.healthy_nodes:
                    self.healthy_nodes.remove(node_id)
    
    def _initial_health_check(self, node: ServiceNode) -> bool:
        """Perform initial health check on a node"""
        return self._check_node_health(node)
    
    def _check_node_health(self, node: ServiceNode) -> bool:
        """Check health of a single node"""
        try:
            health_url = f"{node.endpoint}{self.health_check_config.path}"
            
            start_time = time.time()
            response = requests.get(
                health_url,
                timeout=self.health_check_config.timeout
            )
            response_time = time.time() - start_time
            
            # Update response time
            node.response_time = (node.response_time + response_time) / 2
            
            return response.status_code in self.health_check_config.expected_codes
            
        except Exception as e:
            logger.debug(f"Health check failed for {node.node_id}: {e}")
            return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        total_nodes = len(self.nodes)
        healthy_nodes = len(self.healthy_nodes)
        
        return {
            'total_nodes': total_nodes,
            'healthy_nodes': healthy_nodes,
            'unhealthy_nodes': total_nodes - healthy_nodes,
            'cluster_health': 'healthy' if healthy_nodes > 0 else 'critical',
            'load_balancing_strategy': self.strategy.value,
            'stats': self.stats.copy(),
            'nodes': {
                node_id: {
                    'status': node.status.value,
                    'current_connections': node.current_connections,
                    'utilization': f"{node.utilization:.1%}",
                    'response_time': f"{node.response_time:.3f}s",
                    'failure_count': node.failure_count,
                    'circuit_breaker': self.circuit_breakers.get(node_id, {}).get('state', 'unknown')
                }
                for node_id, node in self.nodes.items()
            }
        }

class HighAvailabilityManager:
    """Manages high availability across multiple services"""
    
    def __init__(self):
        self.services: Dict[str, LoadBalancer] = {}
        self.backup_services: Dict[str, List[str]] = {}  # service -> backup service IDs
        self.failover_history: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Redis for distributed coordination
        self.redis_client = self._init_redis()
        
        # Service discovery
        self.service_registry: Dict[str, Dict[str, Any]] = {}
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis for distributed coordination"""
        try:
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    
    def register_service(self, service_name: str, load_balancer: LoadBalancer) -> bool:
        """Register a service with its load balancer"""
        try:
            self.services[service_name] = load_balancer
            
            # Register in service discovery
            self.service_registry[service_name] = {
                'registered_at': datetime.now().isoformat(),
                'node_count': len(load_balancer.nodes),
                'strategy': load_balancer.strategy.value,
                'status': 'active'
            }
            
            # Register in Redis if available
            if self.redis_client:
                service_info = {
                    'nodes': [
                        {
                            'node_id': node.node_id,
                            'endpoint': node.endpoint,
                            'status': node.status.value
                        }
                        for node in load_balancer.nodes.values()
                    ],
                    'strategy': load_balancer.strategy.value,
                    'last_updated': datetime.now().isoformat()
                }
                
                self.redis_client.hset(
                    'service_registry',
                    service_name,
                    json.dumps(service_info)
                )
                self.redis_client.expire('service_registry', 3600)  # 1 hour TTL
            
            logger.info(f"Service {service_name} registered with {len(load_balancer.nodes)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
            return False
    
    def configure_backup_service(self, primary_service: str, backup_services: List[str]) -> bool:
        """Configure backup services for failover"""
        try:
            self.backup_services[primary_service] = backup_services
            logger.info(f"Configured backup services for {primary_service}: {backup_services}")
            return True
        except Exception as e:
            logger.error(f"Failed to configure backup services: {e}")
            return False
    
    def get_service_node(self, service_name: str, session_id: Optional[str] = None,
                        client_ip: Optional[str] = None) -> Optional[ServiceNode]:
        """Get available node for service with automatic failover"""
        
        # Try primary service first
        if service_name in self.services:
            load_balancer = self.services[service_name]
            node = load_balancer.get_next_node(session_id, client_ip)
            
            if node:
                return node
            
            # If no healthy nodes in primary service, try failover
            logger.warning(f"No healthy nodes in primary service {service_name}, attempting failover")
            return self._attempt_failover(service_name, session_id, client_ip)
        
        return None
    
    def _attempt_failover(self, primary_service: str, session_id: Optional[str] = None,
                         client_ip: Optional[str] = None) -> Optional[ServiceNode]:
        """Attempt failover to backup services"""
        
        if primary_service not in self.backup_services:
            logger.error(f"No backup services configured for {primary_service}")
            return None
        
        backup_services = self.backup_services[primary_service]
        
        for backup_service in backup_services:
            if backup_service in self.services:
                load_balancer = self.services[backup_service]
                node = load_balancer.get_next_node(session_id, client_ip)
                
                if node:
                    # Record failover event
                    failover_event = {
                        'timestamp': datetime.now().isoformat(),
                        'primary_service': primary_service,
                        'backup_service': backup_service,
                        'node_id': node.node_id,
                        'session_id': session_id,
                        'client_ip': client_ip
                    }
                    
                    self.failover_history.append(failover_event)
                    
                    # Keep only last 100 failover events
                    if len(self.failover_history) > 100:
                        self.failover_history = self.failover_history[-100:]
                    
                    logger.info(f"Failover successful: {primary_service} -> {backup_service} (node: {node.node_id})")
                    return node
        
        logger.error(f"All failover attempts failed for service {primary_service}")
        return None
    
    def start_monitoring(self):
        """Start HA monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start health checking for all services
        for service_name, load_balancer in self.services.items():
            load_balancer.start_health_checking()
        
        logger.info("HA monitoring started")
    
    def stop_monitoring(self):
        """Stop HA monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Stop health checking for all services
        for service_name, load_balancer in self.services.items():
            load_balancer.stop_health_checking()
        
        logger.info("HA monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_service_health()
                self._update_service_registry()
                self._cleanup_old_sessions()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _check_service_health(self):
        """Check health of all services"""
        for service_name, load_balancer in self.services.items():
            cluster_status = load_balancer.get_cluster_status()
            
            # Update service registry status
            if service_name in self.service_registry:
                self.service_registry[service_name]['last_health_check'] = datetime.now().isoformat()
                self.service_registry[service_name]['healthy_nodes'] = cluster_status['healthy_nodes']
                
                if cluster_status['healthy_nodes'] == 0:
                    self.service_registry[service_name]['status'] = 'critical'
                    logger.critical(f"Service {service_name} has no healthy nodes!")
                elif cluster_status['healthy_nodes'] < cluster_status['total_nodes']:
                    self.service_registry[service_name]['status'] = 'degraded'
                else:
                    self.service_registry[service_name]['status'] = 'healthy'
    
    def _update_service_registry(self):
        """Update service registry in Redis"""
        if not self.redis_client:
            return
        
        try:
            for service_name, service_info in self.service_registry.items():
                if service_name in self.services:
                    load_balancer = self.services[service_name]
                    
                    updated_info = {
                        'nodes': [
                            {
                                'node_id': node.node_id,
                                'endpoint': node.endpoint,
                                'status': node.status.value,
                                'current_connections': node.current_connections,
                                'response_time': node.response_time
                            }
                            for node in load_balancer.nodes.values()
                        ],
                        'strategy': load_balancer.strategy.value,
                        'stats': load_balancer.stats,
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    self.redis_client.hset(
                        'service_registry',
                        service_name,
                        json.dumps(updated_info)
                    )
        
        except Exception as e:
            logger.error(f"Failed to update service registry: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old session affinities"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # 24 hour session timeout
        
        for service_name, load_balancer in self.services.items():
            # This would need session timestamp tracking in a real implementation
            # For now, we'll just limit the session affinity size
            if len(load_balancer.session_affinity) > 1000:
                # Remove oldest 20% of sessions
                sessions_to_remove = list(load_balancer.session_affinity.keys())[:200]
                for session_id in sessions_to_remove:
                    del load_balancer.session_affinity[session_id]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_services = len(self.services)
        healthy_services = 0
        total_nodes = 0
        healthy_nodes = 0
        
        service_details = {}
        
        for service_name, load_balancer in self.services.items():
            cluster_status = load_balancer.get_cluster_status()
            
            total_nodes += cluster_status['total_nodes']
            healthy_nodes += cluster_status['healthy_nodes']
            
            if cluster_status['healthy_nodes'] > 0:
                healthy_services += 1
            
            service_details[service_name] = cluster_status
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'healthy' if healthy_services == total_services else 'degraded',
            'services': {
                'total': total_services,
                'healthy': healthy_services,
                'degraded': total_services - healthy_services
            },
            'nodes': {
                'total': total_nodes,
                'healthy': healthy_nodes,
                'unhealthy': total_nodes - healthy_nodes
            },
            'failover_events_24h': len([
                event for event in self.failover_history
                if datetime.fromisoformat(event['timestamp']) > 
                   datetime.now() - timedelta(hours=24)
            ]),
            'service_details': service_details
        }
    
    def simulate_node_failure(self, service_name: str, node_id: str) -> bool:
        """Simulate node failure for testing"""
        if service_name in self.services:
            load_balancer = self.services[service_name]
            if node_id in load_balancer.nodes:
                node = load_balancer.nodes[node_id]
                node.status = NodeStatus.UNHEALTHY
                node.failure_count = 999  # Force unhealthy status
                
                if node_id in load_balancer.healthy_nodes:
                    load_balancer.healthy_nodes.remove(node_id)
                
                logger.info(f"Simulated failure for node {node_id} in service {service_name}")
                return True
        
        return False
    
    def recover_node(self, service_name: str, node_id: str) -> bool:
        """Recover a failed node"""
        if service_name in self.services:
            load_balancer = self.services[service_name]
            if node_id in load_balancer.nodes:
                node = load_balancer.nodes[node_id]
                node.status = NodeStatus.HEALTHY
                node.failure_count = 0
                
                if node_id not in load_balancer.healthy_nodes:
                    load_balancer.healthy_nodes.append(node_id)
                
                # Reset circuit breaker
                if node_id in load_balancer.circuit_breakers:
                    circuit = load_balancer.circuit_breakers[node_id]
                    circuit['state'] = 'closed'
                    circuit['failure_count'] = 0
                
                logger.info(f"Recovered node {node_id} in service {service_name}")
                return True
        
        return False

# Docker Compose configuration for HA deployment
DOCKER_COMPOSE_HA = """
version: '3.8'

services:
  # Load Balancer (Nginx)
  nginx-lb:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - app1
      - app2
      - app3
    restart: unless-stopped
    networks:
      - llm-risk-network

  # Application instances
  app1:
    build: .
    environment:
      - NODE_ID=app1
      - PORT=8501
      - REDIS_HOST=redis
      - DB_HOST=postgres-primary
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres-primary
      - redis
    restart: unless-stopped
    networks:
      - llm-risk-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  app2:
    build: .
    environment:
      - NODE_ID=app2
      - PORT=8501
      - REDIS_HOST=redis
      - DB_HOST=postgres-primary
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres-primary
      - redis
    restart: unless-stopped
    networks:
      - llm-risk-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  app3:
    build: .
    environment:
      - NODE_ID=app3
      - PORT=8501
      - REDIS_HOST=redis
      - DB_HOST=postgres-primary
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres-primary
      - redis
    restart: unless-stopped
    networks:
      - llm-risk-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Database cluster
  postgres-primary:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: llm_risk_db
      POSTGRES_USER: llm_user
      POSTGRES_PASSWORD: secure_password_123
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replicator_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
      - ./init-replication.sql:/docker-entrypoint-initdb.d/init-replication.sql
    restart: unless-stopped
    networks:
      - llm-risk-network

  postgres-replica:
    image: postgres:15-alpine
    environment:
      POSTGRES_MASTER_SERVICE: postgres-primary
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replicator_password
      POSTGRES_MASTER_PORT_NUMBER: 5432
    depends_on:
      - postgres-primary
    restart: unless-stopped
    networks:
      - llm-risk-network

  # Redis cluster
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --replica-read-only no
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - llm-risk-network

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    networks:
      - llm-risk-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - llm-risk-network

volumes:
  postgres_primary_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  llm-risk-network:
    driver: bridge
"""

# Nginx configuration for load balancing
NGINX_CONFIG = """
events {
    worker_connections 1024;
}

http {
    upstream llm_risk_app {
        least_conn;
        server app1:8501 max_fails=3 fail_timeout=30s;
        server app2:8501 max_fails=3 fail_timeout=30s;
        server app3:8501 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        
        location /health {
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }
        
        location / {
            proxy_pass http://llm_risk_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Health check
            proxy_connect_timeout 5s;
            proxy_send_timeout 5s;
            proxy_read_timeout 30s;
        }
    }
}
"""

# Streamlit integration functions
def initialize_ha_system():
    """Initialize HA system for Streamlit app"""
    if 'ha_manager' not in st.session_state:
        st.session_state.ha_manager = HighAvailabilityManager()
    
    return st.session_state.ha_manager

def render_ha_dashboard():
    """Render HA monitoring dashboard"""
    st.header("🔄 High Availability & Load Balancing")
    
    ha_manager = initialize_ha_system()
    
    # System overview
    system_status = ha_manager.get_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health_color = "🟢" if system_status['system_health'] == 'healthy' else "🟡"
        st.metric("System Health", f"{health_color} {system_status['system_health'].title()}")
    
    with col2:
        st.metric("Services", f"{system_status['services']['healthy']}/{system_status['services']['total']}")
    
    with col3:
        st.metric("Healthy Nodes", f"{system_status['nodes']['healthy']}/{system_status['nodes']['total']}")
    
    with col4:
        st.metric("Failovers (24h)", system_status['failover_events_24h'])
    
    # Tabs for different HA aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Service Status", "⚖️ Load Balancing", "🔧 Configuration", "📈 Monitoring"])
    
    with tab1:
        st.subheader("Service Status")
        
        if system_status['service_details']:
            for service_name, service_status in system_status['service_details'].items():
                with st.expander(f"Service: {service_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Cluster Health:** {service_status['cluster_health']}")
                        st.write(f"**Strategy:** {service_status['load_balancing_strategy']}")
                        st.write(f"**Total Requests:** {service_status['stats']['total_requests']:,}")
                    
                    with col2:
                        success_rate = 0
                        if service_status['stats']['total_requests'] > 0:
                            success_rate = (service_status['stats']['successful_requests'] / 
                                          service_status['stats']['total_requests'] * 100)
                        
                        st.write(f"**Success Rate:** {success_rate:.1f}%")
                        st.write(f"**Avg Response Time:** {service_status['stats']['avg_response_time']:.3f}s")
                    
                    # Node details
                    st.write("**Nodes:**")
                    for node_id, node_info in service_status['nodes'].items():
                        status_icon = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴", "offline": "⚫"}
                        icon = status_icon.get(node_info['status'], "❓")
                        
                        st.write(f"{icon} {node_id} - {node_info['status']} - "
                               f"{node_info['utilization']} utilized - "
                               f"{node_info['response_time']} response time")
        else:
            st.info("No services registered. Configure services in the Configuration tab.")
    
    with tab2:
        st.subheader("Load Balancing Configuration")
        
        # Service selection
        if ha_manager.services:
            service_names = list(ha_manager.services.keys())
            selected_service = st.selectbox("Select Service", service_names)
            
            if selected_service:
                load_balancer = ha_manager.services[selected_service]
                
                # Strategy selection
                new_strategy = st.selectbox(
                    "Load Balancing Strategy",
                    [strategy.value for strategy in LoadBalancingStrategy],
                    index=[strategy.value for strategy in LoadBalancingStrategy].index(load_balancer.strategy.value)
                )
                
                if st.button("Update Strategy"):
                    load_balancer.strategy = LoadBalancingStrategy(new_strategy)
                    st.success(f"Strategy updated to {new_strategy}")
                    st.rerun()
        else:
            st.info("No services available for load balancing configuration.")
    
    with tab3:
        st.subheader("High Availability Configuration")
        
        # Quick service setup
        with st.form("service_setup"):
            st.write("**Add New Service:**")
            
            service_name = st.text_input("Service Name")
            strategy = st.selectbox("Load Balancing Strategy", [s.value for s in LoadBalancingStrategy])
            
            st.write("**Nodes:**")
            node_count = st.number_input("Number of Nodes", min_value=1, max_value=10, value=2)
            
            nodes_config = []
            for i in range(node_count):
                col1, col2, col3 = st.columns(3)
                with col1:
                    host = st.text_input(f"Node {i+1} Host", value="localhost", key=f"host_{i}")
                with col2:
                    port = st.number_input(f"Node {i+1} Port", value=8501+i, key=f"port_{i}")
                with col3:
                    weight = st.number_input(f"Node {i+1} Weight", value=1, key=f"weight_{i}")
                
                nodes_config.append({'host': host, 'port': port, 'weight': weight})
            
            submitted = st.form_submit_button("Create Service")
            
            if submitted and service_name:
                # Create load balancer
                load_balancer = LoadBalancer(LoadBalancingStrategy(strategy))
                
                # Add nodes
                for i, node_config in enumerate(nodes_config):
                    node = ServiceNode(
                        node_id=f"{service_name}_node_{i+1}",
                        host=node_config['host'],
                        port=node_config['port'],
                        weight=node_config['weight']
                    )
                    load_balancer.add_node(node)
                
                # Register service
                ha_manager.register_service(service_name, load_balancer)
                
                st.success(f"Service '{service_name}' created with {len(nodes_config)} nodes!")
                st.rerun()
    
    with tab4:
        st.subheader("Real-time Monitoring")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not ha_manager.monitoring_active:
                if st.button("▶️ Start Monitoring"):
                    ha_manager.start_monitoring()
                    st.success("Monitoring started!")
                    st.rerun()
            else:
                if st.button("⏹️ Stop Monitoring"):
                    ha_manager.stop_monitoring()
                    st.success("Monitoring stopped!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        # Testing controls
        st.write("**Testing Controls:**")
        
        if ha_manager.services:
            test_service = st.selectbox("Service to Test", list(ha_manager.services.keys()))
            
            if test_service:
                load_balancer = ha_manager.services[test_service]
                node_ids = list(load_balancer.nodes.keys())
                
                if node_ids:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fail_node = st.selectbox("Node to Fail", node_ids)
                        if st.button("🔥 Simulate Failure"):
                            ha_manager.simulate_node_failure(test_service, fail_node)
                            st.warning(f"Simulated failure for {fail_node}")
                            st.rerun()
                    
                    with col2:
                        recover_node = st.selectbox("Node to Recover", node_ids, key="recover")
                        if st.button("🔧 Recover Node"):
                            ha_manager.recover_node(test_service, recover_node)
                            st.success(f"Recovered {recover_node}")
                            st.rerun()

if __name__ == "__main__":
    # Example usage and testing
    
    # Create HA manager
    ha_manager = HighAvailabilityManager()
    
    # Create load balancer for main service
    main_lb = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Add nodes
    nodes = [
        ServiceNode("node1", "localhost", 8501, weight=2),
        ServiceNode("node2", "localhost", 8502, weight=1),
        ServiceNode("node3", "localhost", 8503, weight=1)
    ]
    
    for node in nodes:
        main_lb.add_node(node)
    
    # Create backup service
    backup_lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
    backup_nodes = [
        ServiceNode("backup1", "localhost", 8504),
        ServiceNode("backup2", "localhost", 8505)
    ]
    
    for node in backup_nodes:
        backup_lb.add_node(node)
    
    # Register services
    ha_manager.register_service("llm_risk_main", main_lb)
    ha_manager.register_service("llm_risk_backup", backup_lb)
    
    # Configure failover
    ha_manager.configure_backup_service("llm_risk_main", ["llm_risk_backup"])
    
    # Start monitoring
    ha_manager.start_monitoring()
    
    # Test load balancing
    print("Testing load balancing...")
    for i in range(10):
        node = ha_manager.get_service_node("llm_risk_main", session_id=f"session_{i}")
        if node:
            print(f"Request {i}: Routed to {node.node_id}")
            # Simulate successful request
            main_lb.record_request_success(node.node_id, random.uniform(0.1, 0.5))
        else:
            print(f"Request {i}: No available nodes")
    
    # Test failover
    print("\nTesting failover...")
    ha_manager.simulate_node_failure("llm_risk_main", "node1")
    ha_manager.simulate_node_failure("llm_risk_main", "node2")
    ha_manager.simulate_node_failure("llm_risk_main", "node3")
    
    # Try to get node (should failover to backup)
    node = ha_manager.get_service_node("llm_risk_main", session_id="failover_test")
    if node:
        print(f"Failover successful: Routed to {node.node_id}")
    else:
        print("Failover failed: No available nodes")
    
    # Get system status
    status = ha_manager.get_system_status()
    print(f"\nSystem Status: {status['system_health']}")
    print(f"Healthy Services: {status['services']['healthy']}/{status['services']['total']}")
    print(f"Failover Events: {status['failover_events_24h']}")
    
    # Stop monitoring
    time.sleep(2)
    ha_manager.stop_monitoring()
    
    print("HA system test completed")

    # Write configuration files
    with open("docker-compose-ha.yml", "w") as f:
        f.write(DOCKER_COMPOSE_HA)
    
    with open("nginx.conf", "w") as f:
        f.write(NGINX_CONFIG)
    
    print("HA configuration files written")