"""
API Integration module for LLM Risk Visualizer
Supports real-time data fetching from various LLM monitoring APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import asyncio
import aiohttp
from config import MODELS, LANGUAGES, RISK_CATEGORIES

class APIConnector:
    """Base class for API connections to LLM monitoring services"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
    
    def authenticate(self) -> bool:
        """Authenticate with the API service"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/auth/verify", headers=headers)
            return response.status_code == 200
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def fetch_risk_data(self, 
                       start_date: datetime, 
                       end_date: datetime,
                       models: Optional[List[str]] = None,
                       languages: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch risk data from API"""
        raise NotImplementedError("Subclasses must implement fetch_risk_data")

class OpenAIConnector(APIConnector):
    """Connector for OpenAI monitoring API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.openai.com/v1")
    
    def fetch_risk_data(self, 
                       start_date: datetime, 
                       end_date: datetime,
                       models: Optional[List[str]] = None,
                       languages: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch risk data from OpenAI monitoring API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "models": models or ["GPT-4", "GPT-3.5"],
                "languages": languages or ["English"],
                "metrics": ["safety", "accuracy", "bias"]
            }
            
            # Note: This is a hypothetical endpoint for demonstration
            response = requests.get(
                f"{self.base_url}/monitoring/risks",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_openai_response(data)
            else:
                print(f"API request failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching OpenAI data: {e}")
            return pd.DataFrame()
    
    def _parse_openai_response(self, data: Dict) -> pd.DataFrame:
        """Parse OpenAI API response into standardized format"""
        records = []
        
        for item in data.get('risk_metrics', []):
            record = {
                'Date': datetime.fromisoformat(item['timestamp']),
                'Model': item['model'],
                'Language': item.get('language', 'English'),
                'Risk_Category': self._map_risk_category(item['category']),
                'Risk_Rate': item['risk_score'],
                'Sample_Size': item.get('sample_size', 100),
                'Confidence': item.get('confidence', 0.95)
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _map_risk_category(self, category: str) -> str:
        """Map API risk categories to internal categories"""
        mapping = {
            'safety': 'Toxicity',
            'accuracy': 'Factual Error',
            'bias': 'Bias',
            'hallucination': 'Hallucination',
            'refusal': 'Refusal',
            'privacy': 'Privacy Leakage'
        }
        return mapping.get(category.lower(), 'Other')

class AnthropicConnector(APIConnector):
    """Connector for Anthropic Claude monitoring API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.anthropic.com/v1")
    
    def fetch_risk_data(self, 
                       start_date: datetime, 
                       end_date: datetime,
                       models: Optional[List[str]] = None,
                       languages: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch risk data from Anthropic monitoring API"""
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "models": models or ["Claude"],
                "languages": languages or ["English"],
                "risk_types": RISK_CATEGORIES
            }
            
            # Note: This is a hypothetical endpoint for demonstration
            response = requests.post(
                f"{self.base_url}/monitoring/risks",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_anthropic_response(data)
            else:
                print(f"Anthropic API request failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Anthropic data: {e}")
            return pd.DataFrame()
    
    def _parse_anthropic_response(self, data: Dict) -> pd.DataFrame:
        """Parse Anthropic API response into standardized format"""
        records = []
        
        for metric in data.get('risk_data', []):
            record = {
                'Date': datetime.fromisoformat(metric['date']),
                'Model': 'Claude',
                'Language': metric['language'],
                'Risk_Category': metric['risk_type'],
                'Risk_Rate': metric['risk_score'],
                'Sample_Size': metric.get('samples', 100),
                'Confidence': metric.get('confidence_interval', 0.95)
            }
            records.append(record)
        
        return pd.DataFrame(records)

class GoogleConnector(APIConnector):
    """Connector for Google Gemini monitoring API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://generativelanguage.googleapis.com/v1")
    
    def fetch_risk_data(self, 
                       start_date: datetime, 
                       end_date: datetime,
                       models: Optional[List[str]] = None,
                       languages: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch risk data from Google Gemini monitoring API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "models": models or ["Gemini"],
                "locales": languages or ["en"],
                "safety_categories": ["harassment", "hate_speech", "dangerous_content"]
            }
            
            # Note: This is a hypothetical endpoint for demonstration
            response = requests.get(
                f"{self.base_url}/safety/metrics",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_google_response(data)
            else:
                print(f"Google API request failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Google data: {e}")
            return pd.DataFrame()
    
    def _parse_google_response(self, data: Dict) -> pd.DataFrame:
        """Parse Google API response into standardized format"""
        records = []
        
        for entry in data.get('safety_metrics', []):
            # Map Google safety categories to our risk categories
            risk_mapping = {
                'harassment': 'Toxicity',
                'hate_speech': 'Bias',
                'dangerous_content': 'Toxicity',
                'medical': 'Factual Error',
                'derogatory': 'Bias'
            }
            
            record = {
                'Date': datetime.fromisoformat(entry['timestamp']),
                'Model': 'Gemini',
                'Language': self._map_locale(entry.get('locale', 'en')),
                'Risk_Category': risk_mapping.get(entry['category'], 'Other'),
                'Risk_Rate': entry['probability'],
                'Sample_Size': entry.get('total_samples', 100),
                'Confidence': entry.get('confidence', 0.95)
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _map_locale(self, locale: str) -> str:
        """Map locale codes to language names"""
        locale_mapping = {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ar': 'Arabic',
            'ru': 'Russian',
            'ja': 'Japanese',
            'pt': 'Portuguese',
            'hi': 'Hindi'
        }
        return locale_mapping.get(locale, 'English')

class APIManager:
    """Manage multiple API connections and aggregate data"""
    
    def __init__(self):
        self.connectors = {}
    
    def add_connector(self, name: str, connector: APIConnector):
        """Add an API connector"""
        self.connectors[name] = connector
    
    def remove_connector(self, name: str):
        """Remove an API connector"""
        if name in self.connectors:
            del self.connectors[name]
    
    async def fetch_all_data(self, 
                            start_date: datetime, 
                            end_date: datetime,
                            models: Optional[List[str]] = None,
                            languages: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch data from all connected APIs asynchronously"""
        all_data = []
        
        tasks = []
        for name, connector in self.connectors.items():
            task = asyncio.create_task(
                self._fetch_connector_data(connector, start_date, end_date, models, languages)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, pd.DataFrame) and not result.empty:
                all_data.append(result)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data.drop_duplicates()
        else:
            return pd.DataFrame()
    
    async def _fetch_connector_data(self, 
                                  connector: APIConnector,
                                  start_date: datetime, 
                                  end_date: datetime,
                                  models: Optional[List[str]],
                                  languages: Optional[List[str]]) -> pd.DataFrame:
        """Fetch data from a single connector asynchronously"""
        try:
            # Convert synchronous call to async
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, 
                connector.fetch_risk_data, 
                start_date, 
                end_date, 
                models, 
                languages
            )
            return data
        except Exception as e:
            print(f"Error fetching data from connector: {e}")
            return pd.DataFrame()
    
    def test_connections(self) -> Dict[str, bool]:
        """Test all API connections"""
        results = {}
        for name, connector in self.connectors.items():
            try:
                results[name] = connector.authenticate()
            except Exception as e:
                print(f"Connection test failed for {name}: {e}")
                results[name] = False
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of models available from all connected APIs"""
        available_models = set()
        for connector in self.connectors.values():
            if hasattr(connector, 'get_available_models'):
                available_models.update(connector.get_available_models())
        return list(available_models)

# Factory function for easy connector creation
def create_connector(provider: str, api_key: str) -> Optional[APIConnector]:
    """Create API connector based on provider"""
    providers = {
        'openai': OpenAIConnector,
        'anthropic': AnthropicConnector,
        'google': GoogleConnector
    }
    
    if provider.lower() in providers:
        return providers[provider.lower()](api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Example usage and configuration
def setup_api_manager(config: Dict[str, str]) -> APIManager:
    """Setup API manager with provided configuration"""
    manager = APIManager()
    
    for provider, api_key in config.items():
        if api_key:  # Only add connectors with valid API keys
            try:
                connector = create_connector(provider, api_key)
                manager.add_connector(provider, connector)
                print(f"Added {provider} connector")
            except Exception as e:
                print(f"Failed to add {provider} connector: {e}")
    
    return manager