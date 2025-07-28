"""
Third-Party Integrations and API Connectors Module
Provides extensible framework for integrating with external APIs and services
"""

import asyncio
import aiohttp
import requests
import json
import time
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import streamlit as st
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthenticationType(Enum):
    """Types of authentication methods"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"
    HMAC_SIGNATURE = "hmac_signature"
    JWT = "jwt"

class IntegrationType(Enum):
    """Types of integrations"""
    DATA_SOURCE = "data_source"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    AI_SERVICE = "ai_service"
    STORAGE = "storage"
    MESSAGING = "messaging"
    AUTHENTICATION = "authentication"

@dataclass
class APICredentials:
    """API credentials configuration"""
    auth_type: AuthenticationType
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    custom_headers: Dict[str, str] = None
    token_expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.custom_headers is None:
            self.custom_headers = {}

@dataclass
class IntegrationConfig:
    """Configuration for a third-party integration"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    base_url: str
    credentials: APICredentials
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    retry_delay: int = 1  # seconds
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    status_code: int
    data: Any
    error_message: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = None
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.headers is None:
            self.headers = {}

class BaseConnector(ABC):
    """Abstract base class for API connectors"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.last_request_time = 0
        
        # Setup authentication
        self._setup_authentication()
    
    @abstractmethod
    def test_connection(self) -> APIResponse:
        """Test the connection to the API"""
        pass
    
    @abstractmethod
    def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Fetch data from the API"""
        pass
    
    def _setup_authentication(self):
        """Setup authentication for the session"""
        creds = self.config.credentials
        
        if creds.auth_type == AuthenticationType.API_KEY:
            self.session.params['api_key'] = creds.api_key
        
        elif creds.auth_type == AuthenticationType.BEARER_TOKEN:
            self.session.headers['Authorization'] = f'Bearer {creds.access_token}'
        
        elif creds.auth_type == AuthenticationType.BASIC_AUTH:
            from requests.auth import HTTPBasicAuth
            self.session.auth = HTTPBasicAuth(creds.username, creds.password)
        
        elif creds.auth_type == AuthenticationType.CUSTOM_HEADER:
            self.session.headers.update(creds.custom_headers)
        
        elif creds.auth_type == AuthenticationType.JWT:
            self.session.headers['Authorization'] = f'JWT {creds.access_token}'
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make authenticated API request with rate limiting and retries"""
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Add timeout
        kwargs.setdefault('timeout', self.config.timeout)
        
        start_time = time.time()
        
        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.credentials.auth_type == AuthenticationType.HMAC_SIGNATURE:
                    # Add HMAC signature
                    self._add_hmac_signature(method, endpoint, kwargs)
                
                response = self.session.request(method, url, **kwargs)
                response_time = time.time() - start_time
                
                # Check if token needs refresh
                if response.status_code == 401 and self.config.credentials.auth_type == AuthenticationType.OAUTH2:
                    if self._refresh_token():
                        continue  # Retry with new token
                
                return APIResponse(
                    success=response.status_code < 400,
                    status_code=response.status_code,
                    data=response.json() if response.content else None,
                    error_message=None if response.status_code < 400 else response.text,
                    response_time=response_time,
                    headers=dict(response.headers)
                )
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.retry_attempts - 1:
                    return APIResponse(
                        success=False,
                        status_code=0,
                        data=None,
                        error_message=str(e),
                        response_time=time.time() - start_time
                    )
                
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
    
    def _add_hmac_signature(self, method: str, endpoint: str, request_kwargs: Dict[str, Any]):
        """Add HMAC signature to request"""
        timestamp = str(int(time.time()))
        
        # Create string to sign
        string_to_sign = f"{method.upper()}\n{endpoint}\n{timestamp}"
        
        if 'json' in request_kwargs:
            string_to_sign += f"\n{json.dumps(request_kwargs['json'], sort_keys=True)}"
        
        # Generate signature
        signature = hmac.new(
            self.config.credentials.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature headers
        request_kwargs.setdefault('headers', {})
        request_kwargs['headers'].update({
            'X-API-Key': self.config.credentials.api_key,
            'X-Timestamp': timestamp,
            'X-Signature': signature
        })
    
    def _refresh_token(self) -> bool:
        """Refresh OAuth2 token"""
        # Implementation would depend on OAuth2 provider
        # This is a placeholder
        return False

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock() if asyncio.iscoroutinefunction(self.__init__) else None
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.requests_per_minute:
            # Calculate how long to wait
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request)
            if wait_time > 0:
                time.sleep(wait_time)
        
        self.requests.append(now)

# Specific Connector Implementations

class OpenAIConnector(BaseConnector):
    """OpenAI API connector for LLM services"""
    
    def test_connection(self) -> APIResponse:
        """Test OpenAI API connection"""
        return self._make_request('GET', 'models')
    
    def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Fetch data from OpenAI API"""
        return self._make_request('GET', endpoint, params=params)
    
    def generate_completion(self, prompt: str, model: str = "gpt-3.5-turbo", 
                          max_tokens: int = 100) -> APIResponse:
        """Generate text completion"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        return self._make_request('POST', 'chat/completions', json=data)
    
    def analyze_risk_content(self, content: str) -> APIResponse:
        """Analyze content for potential risks using OpenAI"""
        prompt = f"""
        Analyze the following content for potential risks in LLM applications:
        
        Content: {content}
        
        Please identify:
        1. Potential bias or discrimination
        2. Harmful or inappropriate content
        3. Privacy concerns
        4. Misinformation risks
        5. Overall risk level (low/medium/high)
        
        Provide a structured analysis in JSON format.
        """
        
        return self.generate_completion(prompt, max_tokens=500)

class HuggingFaceConnector(BaseConnector):
    """Hugging Face API connector"""
    
    def test_connection(self) -> APIResponse:
        """Test Hugging Face API connection"""
        return self._make_request('GET', 'api/whoami-v2')
    
    def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Fetch data from Hugging Face API"""
        return self._make_request('GET', endpoint, params=params)
    
    def get_model_info(self, model_id: str) -> APIResponse:
        """Get information about a specific model"""
        return self._make_request('GET', f'api/models/{model_id}')
    
    def inference(self, model_id: str, inputs: Union[str, Dict], 
                 parameters: Dict[str, Any] = None) -> APIResponse:
        """Run inference on a model"""
        data = {
            "inputs": inputs,
            "parameters": parameters or {}
        }
        return self._make_request('POST', f'models/{model_id}', json=data)

class SlackConnector(BaseConnector):
    """Slack API connector for notifications"""
    
    def test_connection(self) -> APIResponse:
        """Test Slack API connection"""
        return self._make_request('GET', 'auth.test')
    
    def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Fetch data from Slack API"""
        return self._make_request('GET', endpoint, params=params)
    
    def send_message(self, channel: str, text: str, blocks: List[Dict] = None) -> APIResponse:
        """Send message to Slack channel"""
        data = {
            "channel": channel,
            "text": text
        }
        if blocks:
            data["blocks"] = blocks
        
        return self._make_request('POST', 'chat.postMessage', json=data)
    
    def send_risk_alert(self, channel: str, risk_data: Dict[str, Any]) -> APIResponse:
        """Send formatted risk alert to Slack"""
        severity_colors = {
            "low": "#36a64f",
            "medium": "#ff9500", 
            "high": "#ff4444",
            "critical": "#ff0000"
        }
        
        color = severity_colors.get(risk_data.get('severity', 'medium'), "#808080")
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🚨 Risk Alert: {risk_data.get('title', 'Risk Detected')}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{risk_data.get('severity', 'Unknown').title()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Model:*\n{risk_data.get('model', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Risk Rate:*\n{risk_data.get('risk_rate', 0):.3f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Detected:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        ]
        
        if risk_data.get('description'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{risk_data['description']}"
                }
            })
        
        return self.send_message(channel, "Risk Alert", blocks)

class DatadogConnector(BaseConnector):
    """Datadog API connector for monitoring"""
    
    def test_connection(self) -> APIResponse:
        """Test Datadog API connection"""
        return self._make_request('GET', 'validate')
    
    def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Fetch data from Datadog API"""
        return self._make_request('GET', endpoint, params=params)
    
    def send_metric(self, metric_name: str, value: float, tags: List[str] = None) -> APIResponse:
        """Send metric to Datadog"""
        data = {
            "series": [
                {
                    "metric": metric_name,
                    "points": [[int(time.time()), value]],
                    "tags": tags or []
                }
            ]
        }
        return self._make_request('POST', 'series', json=data)
    
    def create_dashboard(self, title: str, widgets: List[Dict]) -> APIResponse:
        """Create Datadog dashboard"""
        data = {
            "title": title,
            "widgets": widgets,
            "layout_type": "ordered"
        }
        return self._make_request('POST', 'dashboard', json=data)

class ElasticsearchConnector(BaseConnector):
    """Elasticsearch connector for data storage and search"""
    
    def test_connection(self) -> APIResponse:
        """Test Elasticsearch connection"""
        return self._make_request('GET', '')
    
    def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Fetch data from Elasticsearch"""
        return self._make_request('GET', endpoint, params=params)
    
    def index_document(self, index: str, doc_id: str, document: Dict[str, Any]) -> APIResponse:
        """Index a document in Elasticsearch"""
        return self._make_request('PUT', f'{index}/_doc/{doc_id}', json=document)
    
    def search(self, index: str, query: Dict[str, Any]) -> APIResponse:
        """Search documents in Elasticsearch"""
        return self._make_request('POST', f'{index}/_search', json=query)
    
    def index_risk_data(self, risk_data: Dict[str, Any]) -> APIResponse:
        """Index risk assessment data"""
        doc_id = hashlib.md5(
            f"{risk_data.get('model', '')}{risk_data.get('timestamp', '')}"
            .encode()
        ).hexdigest()
        
        return self.index_document('llm-risks', doc_id, risk_data)

class IntegrationManager:
    """Manages all third-party integrations"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        self.connection_pool = ThreadPoolExecutor(max_workers=10)
        
        # Initialize database for storing integration data
        self.init_database()
        
        # Load predefined integrations
        self._load_predefined_integrations()
    
    def init_database(self):
        """Initialize database for integration management"""
        conn = sqlite3.connect("integrations.db")
        cursor = conn.cursor()
        
        # Integrations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                base_url TEXT NOT NULL,
                auth_type TEXT NOT NULL,
                credentials TEXT NOT NULL,
                config TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0
            )
        ''')
        
        # API calls log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                integration_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                response_time REAL,
                success BOOLEAN,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (integration_id) REFERENCES integrations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_predefined_integrations(self):
        """Load predefined integration templates"""
        self.integration_templates = {
            "openai": {
                "name": "OpenAI API",
                "type": IntegrationType.AI_SERVICE,
                "base_url": "https://api.openai.com/v1",
                "auth_type": AuthenticationType.BEARER_TOKEN,
                "connector_class": OpenAIConnector
            },
            "huggingface": {
                "name": "Hugging Face",
                "type": IntegrationType.AI_SERVICE,
                "base_url": "https://huggingface.co",
                "auth_type": AuthenticationType.BEARER_TOKEN,
                "connector_class": HuggingFaceConnector
            },
            "slack": {
                "name": "Slack",
                "type": IntegrationType.NOTIFICATION,
                "base_url": "https://slack.com/api",
                "auth_type": AuthenticationType.BEARER_TOKEN,
                "connector_class": SlackConnector
            },
            "datadog": {
                "name": "Datadog",
                "type": IntegrationType.MONITORING,
                "base_url": "https://api.datadoghq.com/api/v1",
                "auth_type": AuthenticationType.API_KEY,
                "connector_class": DatadogConnector
            },
            "elasticsearch": {
                "name": "Elasticsearch",
                "type": IntegrationType.STORAGE,
                "base_url": "http://localhost:9200",
                "auth_type": AuthenticationType.BASIC_AUTH,
                "connector_class": ElasticsearchConnector
            }
        }
    
    def register_integration(self, config: IntegrationConfig) -> bool:
        """Register a new integration"""
        try:
            # Store configuration
            self.configs[config.integration_id] = config
            
            # Create connector based on integration type
            if config.integration_id in self.integration_templates:
                template = self.integration_templates[config.integration_id]
                connector_class = template["connector_class"]
                self.connectors[config.integration_id] = connector_class(config)
            else:
                # Use generic connector
                self.connectors[config.integration_id] = BaseConnector(config)
            
            # Persist to database
            self._save_integration_to_db(config)
            
            logger.info(f"Integration {config.integration_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register integration {config.integration_id}: {e}")
            return False
    
    def _save_integration_to_db(self, config: IntegrationConfig):
        """Save integration configuration to database"""
        conn = sqlite3.connect("integrations.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO integrations 
            (id, name, type, base_url, auth_type, credentials, config, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            config.integration_id,
            config.name,
            config.integration_type.value,
            config.base_url,
            config.credentials.auth_type.value,
            json.dumps(asdict(config.credentials)),
            json.dumps(asdict(config)),
            config.enabled
        ))
        
        conn.commit()
        conn.close()
    
    def test_integration(self, integration_id: str) -> APIResponse:
        """Test a specific integration"""
        if integration_id not in self.connectors:
            return APIResponse(
                success=False,
                status_code=0,
                data=None,
                error_message=f"Integration {integration_id} not found"
            )
        
        try:
            connector = self.connectors[integration_id]
            response = connector.test_connection()
            
            # Log the API call
            self._log_api_call(integration_id, "test", "GET", response)
            
            return response
            
        except Exception as e:
            error_response = APIResponse(
                success=False,
                status_code=0,
                data=None,
                error_message=str(e)
            )
            self._log_api_call(integration_id, "test", "GET", error_response)
            return error_response
    
    def call_integration(self, integration_id: str, endpoint: str, 
                        method: str = "GET", **kwargs) -> APIResponse:
        """Make a call to a specific integration"""
        if integration_id not in self.connectors:
            return APIResponse(
                success=False,
                status_code=0,
                data=None,
                error_message=f"Integration {integration_id} not found"
            )
        
        try:
            connector = self.connectors[integration_id]
            response = connector._make_request(method, endpoint, **kwargs)
            
            # Log the API call
            self._log_api_call(integration_id, endpoint, method, response)
            
            return response
            
        except Exception as e:
            error_response = APIResponse(
                success=False,
                status_code=0,
                data=None,
                error_message=str(e)
            )
            self._log_api_call(integration_id, endpoint, method, error_response)
            return error_response
    
    def _log_api_call(self, integration_id: str, endpoint: str, method: str, response: APIResponse):
        """Log API call for monitoring and analytics"""
        try:
            conn = sqlite3.connect("integrations.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_calls 
                (integration_id, endpoint, method, status_code, response_time, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                integration_id,
                endpoint,
                method,
                response.status_code,
                response.response_time,
                response.success,
                response.error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log API call: {e}")
    
    def get_integration_stats(self, integration_id: str) -> Dict[str, Any]:
        """Get statistics for a specific integration"""
        try:
            conn = sqlite3.connect("integrations.db")
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_calls,
                    COUNT(CASE WHEN success = 1 THEN 1 END) as successful_calls,
                    COUNT(CASE WHEN success = 0 THEN 1 END) as failed_calls,
                    AVG(response_time) as avg_response_time,
                    MAX(timestamp) as last_call
                FROM api_calls 
                WHERE integration_id = ?
            ''', (integration_id,))
            
            stats = cursor.fetchone()
            
            if stats:
                total_calls, successful_calls, failed_calls, avg_response_time, last_call = stats
                success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
                
                return {
                    "total_calls": total_calls,
                    "successful_calls": successful_calls,
                    "failed_calls": failed_calls,
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time or 0,
                    "last_call": last_call
                }
            
            conn.close()
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get integration stats: {e}")
            return {}
    
    def get_all_integrations(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered integrations with their status"""
        integrations = {}
        
        for integration_id, config in self.configs.items():
            stats = self.get_integration_stats(integration_id)
            
            integrations[integration_id] = {
                "config": asdict(config),
                "stats": stats,
                "status": "active" if config.enabled else "disabled"
            }
        
        return integrations
    
    def enable_integration(self, integration_id: str) -> bool:
        """Enable an integration"""
        if integration_id in self.configs:
            self.configs[integration_id].enabled = True
            self._save_integration_to_db(self.configs[integration_id])
            return True
        return False
    
    def disable_integration(self, integration_id: str) -> bool:
        """Disable an integration"""
        if integration_id in self.configs:
            self.configs[integration_id].enabled = False
            self._save_integration_to_db(self.configs[integration_id])
            return True
        return False
    
    def remove_integration(self, integration_id: str) -> bool:
        """Remove an integration"""
        try:
            if integration_id in self.connectors:
                del self.connectors[integration_id]
            
            if integration_id in self.configs:
                del self.configs[integration_id]
            
            # Remove from database
            conn = sqlite3.connect("integrations.db")
            cursor = conn.cursor()
            cursor.execute('DELETE FROM integrations WHERE id = ?', (integration_id,))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove integration {integration_id}: {e}")
            return False
    
    async def test_all_integrations(self) -> Dict[str, APIResponse]:
        """Test all registered integrations asynchronously"""
        results = {}
        
        async def test_single_integration(integration_id):
            return integration_id, self.test_integration(integration_id)
        
        tasks = [test_single_integration(iid) for iid in self.connectors.keys()]
        
        for task in asyncio.as_completed(tasks):
            integration_id, response = await task
            results[integration_id] = response
        
        return results

# Webhook and Event Handling

class WebhookHandler:
    """Handles incoming webhooks from external services"""
    
    def __init__(self, integration_manager: IntegrationManager):
        self.integration_manager = integration_manager
        self.webhook_handlers: Dict[str, Callable] = {}
        
    def register_webhook_handler(self, integration_id: str, handler: Callable):
        """Register a webhook handler for an integration"""
        self.webhook_handlers[integration_id] = handler
    
    def handle_webhook(self, integration_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook payload"""
        if integration_id in self.webhook_handlers:
            try:
                return self.webhook_handlers[integration_id](payload)
            except Exception as e:
                logger.error(f"Webhook handler error for {integration_id}: {e}")
                return {"error": str(e)}
        
        return {"error": f"No handler registered for {integration_id}"}

# Streamlit Integration Functions

def initialize_integrations():
    """Initialize integration manager for Streamlit app"""
    if 'integration_manager' not in st.session_state:
        st.session_state.integration_manager = IntegrationManager()
    
    return st.session_state.integration_manager

def render_integrations_dashboard():
    """Render integrations management dashboard"""
    st.header("🔌 Third-Party Integrations")
    
    integration_manager = initialize_integrations()
    
    # Overview metrics
    all_integrations = integration_manager.get_all_integrations()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Integrations", len(all_integrations))
    
    with col2:
        active_count = sum(1 for i in all_integrations.values() if i['status'] == 'active')
        st.metric("Active Integrations", active_count)
    
    with col3:
        total_calls = sum(i['stats'].get('total_calls', 0) for i in all_integrations.values())
        st.metric("Total API Calls", f"{total_calls:,}")
    
    with col4:
        if all_integrations:
            avg_success_rate = sum(i['stats'].get('success_rate', 0) for i in all_integrations.values()) / len(all_integrations)
            st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
        else:
            st.metric("Avg Success Rate", "0.0%")
    
    # Tabs for different aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "➕ Add Integration", "⚙️ Manage", "📈 Analytics"])
    
    with tab1:
        st.subheader("Integration Status")
        
        if all_integrations:
            for integration_id, integration_data in all_integrations.items():
                with st.expander(f"{integration_data['config']['name']} ({integration_id})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {integration_data['config']['integration_type']}")
                        st.write(f"**Status:** {integration_data['status']}")
                        st.write(f"**Base URL:** {integration_data['config']['base_url']}")
                    
                    with col2:
                        stats = integration_data['stats']
                        st.write(f"**Total Calls:** {stats.get('total_calls', 0):,}")
                        st.write(f"**Success Rate:** {stats.get('success_rate', 0):.1f}%")
                        st.write(f"**Avg Response Time:** {stats.get('avg_response_time', 0):.3f}s")
                    
                    # Test connection button
                    if st.button(f"Test Connection", key=f"test_{integration_id}"):
                        with st.spinner("Testing connection..."):
                            response = integration_manager.test_integration(integration_id)
                            
                            if response.success:
                                st.success(f"Connection successful! Status: {response.status_code}")
                            else:
                                st.error(f"Connection failed: {response.error_message}")
        else:
            st.info("No integrations configured. Add your first integration in the 'Add Integration' tab.")
    
    with tab2:
        st.subheader("Add New Integration")
        
        # Integration template selector
        template_options = ["Custom"] + list(integration_manager.integration_templates.keys())
        selected_template = st.selectbox("Integration Template", template_options)
        
        with st.form("add_integration"):
            if selected_template != "Custom":
                template = integration_manager.integration_templates[selected_template]
                integration_name = st.text_input("Integration Name", value=template["name"])
                base_url = st.text_input("Base URL", value=template["base_url"])
                integration_type = st.selectbox(
                    "Integration Type", 
                    [t.value for t in IntegrationType],
                    index=[t.value for t in IntegrationType].index(template["type"].value)
                )
                auth_type = st.selectbox(
                    "Authentication Type",
                    [a.value for a in AuthenticationType],
                    index=[a.value for a in AuthenticationType].index(template["auth_type"].value)
                )
            else:
                integration_name = st.text_input("Integration Name")
                base_url = st.text_input("Base URL")
                integration_type = st.selectbox("Integration Type", [t.value for t in IntegrationType])
                auth_type = st.selectbox("Authentication Type", [a.value for a in AuthenticationType])
            
            # Authentication credentials
            st.write("**Authentication Credentials:**")
            
            auth_type_enum = AuthenticationType(auth_type)
            
            api_key = ""
            secret_key = ""
            access_token = ""
            username = ""
            password = ""
            
            if auth_type_enum in [AuthenticationType.API_KEY, AuthenticationType.HMAC_SIGNATURE]:
                api_key = st.text_input("API Key", type="password")
                if auth_type_enum == AuthenticationType.HMAC_SIGNATURE:
                    secret_key = st.text_input("Secret Key", type="password")
            
            elif auth_type_enum in [AuthenticationType.BEARER_TOKEN, AuthenticationType.JWT]:
                access_token = st.text_input("Access Token", type="password")
            
            elif auth_type_enum == AuthenticationType.BASIC_AUTH:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                rate_limit = st.number_input("Rate Limit (requests/min)", min_value=1, max_value=10000, value=60)
                timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=300, value=30)
                retry_attempts = st.number_input("Retry Attempts", min_value=0, max_value=10, value=3)
            
            submitted = st.form_submit_button("Add Integration")
            
            if submitted and integration_name and base_url:
                # Create integration configuration
                integration_id = integration_name.lower().replace(" ", "_")
                
                credentials = APICredentials(
                    auth_type=auth_type_enum,
                    api_key=api_key or None,
                    secret_key=secret_key or None,
                    access_token=access_token or None,
                    username=username or None,
                    password=password or None
                )
                
                config = IntegrationConfig(
                    integration_id=integration_id,
                    name=integration_name,
                    integration_type=IntegrationType(integration_type),
                    base_url=base_url,
                    credentials=credentials,
                    rate_limit=rate_limit,
                    timeout=timeout,
                    retry_attempts=retry_attempts
                )
                
                # Register integration
                if integration_manager.register_integration(config):
                    st.success(f"Integration '{integration_name}' added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add integration")
    
    with tab3:
        st.subheader("Manage Integrations")
        
        if all_integrations:
            for integration_id, integration_data in all_integrations.items():
                st.write(f"**{integration_data['config']['name']}**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_status = integration_data['status'] == 'active'
                    if st.checkbox("Enabled", value=current_status, key=f"enable_{integration_id}"):
                        integration_manager.enable_integration(integration_id)
                    else:
                        integration_manager.disable_integration(integration_id)
                
                with col2:
                    if st.button("Test", key=f"manage_test_{integration_id}"):
                        response = integration_manager.test_integration(integration_id)
                        if response.success:
                            st.success("✅ Connected")
                        else:
                            st.error("❌ Failed")
                
                with col3:
                    if st.button("View Stats", key=f"stats_{integration_id}"):
                        stats = integration_manager.get_integration_stats(integration_id)
                        st.json(stats)
                
                with col4:
                    if st.button("Remove", key=f"remove_{integration_id}"):
                        if integration_manager.remove_integration(integration_id):
                            st.success(f"Integration {integration_id} removed")
                            st.rerun()
                
                st.divider()
        else:
            st.info("No integrations to manage")
    
    with tab4:
        st.subheader("Integration Analytics")
        
        if all_integrations:
            # API calls over time (mock data for demo)
            st.write("**API Calls Over Time**")
            
            # Create sample data
            dates = pd.date_range(start='2025-01-01', end=datetime.now(), freq='D')
            api_calls_data = []
            
            for integration_id in all_integrations.keys():
                for date in dates[-30:]:  # Last 30 days
                    api_calls_data.append({
                        'Date': date,
                        'Integration': integration_id,
                        'Calls': np.random.randint(0, 50),
                        'Success_Rate': np.random.uniform(0.85, 1.0)
                    })
            
            df = pd.DataFrame(api_calls_data)
            
            # Line chart for API calls
            import plotly.express as px
            
            fig_calls = px.line(df, x='Date', y='Calls', color='Integration', 
                              title='API Calls Over Time')
            st.plotly_chart(fig_calls, use_container_width=True)
            
            # Success rate chart
            fig_success = px.line(df, x='Date', y='Success_Rate', color='Integration',
                                title='Success Rate Over Time')
            st.plotly_chart(fig_success, use_container_width=True)
            
            # Integration usage distribution
            usage_data = []
            for integration_id, integration_data in all_integrations.items():
                usage_data.append({
                    'Integration': integration_id,
                    'Total_Calls': integration_data['stats'].get('total_calls', 0),
                    'Success_Rate': integration_data['stats'].get('success_rate', 0)
                })
            
            usage_df = pd.DataFrame(usage_data)
            
            if not usage_df.empty:
                fig_usage = px.bar(usage_df, x='Integration', y='Total_Calls',
                                 title='Total API Calls by Integration')
                st.plotly_chart(fig_usage, use_container_width=True)
        else:
            st.info("No analytics data available. Add and use some integrations first!")

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize integration manager
    integration_manager = IntegrationManager()
    
    # Test OpenAI integration (example)
    openai_credentials = APICredentials(
        auth_type=AuthenticationType.BEARER_TOKEN,
        access_token="sk-test-key"  # Replace with actual key for testing
    )
    
    openai_config = IntegrationConfig(
        integration_id="openai_test",
        name="OpenAI Test",
        integration_type=IntegrationType.AI_SERVICE,
        base_url="https://api.openai.com/v1",
        credentials=openai_credentials
    )
    
    # Register integration
    success = integration_manager.register_integration(openai_config)
    print(f"OpenAI integration registered: {success}")
    
    # Test connection (will fail without valid API key)
    response = integration_manager.test_integration("openai_test")
    print(f"OpenAI connection test: {response.success}, Status: {response.status_code}")
    
    # Get integration stats
    stats = integration_manager.get_integration_stats("openai_test")
    print(f"Integration stats: {stats}")
    
    # Test Slack integration
    slack_credentials = APICredentials(
        auth_type=AuthenticationType.BEARER_TOKEN,
        access_token="xoxb-test-token"  # Replace with actual token
    )
    
    slack_config = IntegrationConfig(
        integration_id="slack_notifications",
        name="Slack Notifications",
        integration_type=IntegrationType.NOTIFICATION,
        base_url="https://slack.com/api",
        credentials=slack_credentials
    )
    
    integration_manager.register_integration(slack_config)
    
    # Get all integrations
    all_integrations = integration_manager.get_all_integrations()
    print(f"Total integrations: {len(all_integrations)}")
    
    print("Third-party integrations module test completed!")