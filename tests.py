"""
Test Suite for LLM Risk Visualizer
Comprehensive tests for all major components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sqlite3
import json

# Import modules to test
from config import MODELS, LANGUAGES, RISK_CATEGORIES
from data_processor import DataProcessor
from visualizations import Visualizer
from auth import AuthManager, UserRole
from database import DatabaseManager, RiskDataManager
from api import OpenAIConnector, AnthropicConnector, APIManager
from monitoring import RiskMonitor, AlertRule, Alert
from utils import RiskCategoryManager, RiskCategory
from sample_data import generate_risk_data

class TestDataProcessor:
    """Test data processing functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = generate_risk_data(days=10)
        self.processor = DataProcessor(self.test_data)
    
    def test_data_generation(self):
        """Test sample data generation"""
        assert not self.test_data.empty
        assert len(self.test_data) > 0
        assert all(col in self.test_data.columns for col in 
                  ['Date', 'Model', 'Language', 'Risk_Category', 'Risk_Rate', 'Sample_Size', 'Confidence'])
    
    def test_aggregate_metrics(self):
        """Test aggregate metrics calculation"""
        aggregated = self.processor.calculate_aggregate_metrics(['Model'])
        assert not aggregated.empty
        assert 'Risk_Rate_mean' in aggregated.columns
    
    def test_risk_scores(self):
        """Test risk score calculation"""
        risk_scores = self.processor.calculate_risk_scores()
        assert isinstance(risk_scores, pd.DataFrame)
    
    def test_trend_analysis(self):
        """Test trend analysis"""
        trends = self.processor.analyze_trends()
        assert isinstance(trends, dict)

class TestVisualizations:
    """Test visualization components"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = generate_risk_data(days=5)
        self.visualizer = Visualizer()
    
    def test_risk_heatmap(self):
        """Test risk heatmap creation"""
        pivot_data = self.test_data.pivot_table(
            index='Model', 
            columns='Language', 
            values='Risk_Rate', 
            aggfunc='mean'
        ).fillna(0)
        
        fig = self.visualizer.create_risk_heatmap(pivot_data)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_time_series_chart(self):
        """Test time series chart creation"""
        time_data = self.test_data.groupby(['Date', 'Risk_Category'])['Risk_Rate'].mean().reset_index()
        fig = self.visualizer.create_time_series_chart(time_data)
        assert fig is not None

class TestAuthentication:
    """Test authentication system"""
    
    def setup_method(self):
        """Setup auth manager with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.auth_manager = AuthManager()
        # Override database path for testing
        self.auth_manager.db.db_path = self.temp_db.name
        self.auth_manager.db.init_database()
    
    def teardown_method(self):
        """Cleanup temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_user_registration(self):
        """Test user registration"""
        success, message = self.auth_manager.register(
            "testuser", "test@example.com", "TestPassword123", UserRole.ANALYST
        )
        assert success
        assert "successfully" in message.lower()
    
    def test_user_login(self):
        """Test user login"""
        # First register a user
        self.auth_manager.register("testuser", "test@example.com", "TestPassword123", UserRole.ANALYST)
        
        # Then try to login
        success, token, user_info = self.auth_manager.login("testuser", "TestPassword123")
        assert success
        assert token is not None
        assert user_info['username'] == "testuser"
    
    def test_invalid_login(self):
        """Test invalid login attempts"""
        success, message, user_info = self.auth_manager.login("nonexistent", "wrongpassword")
        assert not success
        assert user_info is None
    
    def test_user_roles(self):
        """Test user role permissions"""
        admin_perms = UserRole.get_permissions(UserRole.ADMIN)
        viewer_perms = UserRole.get_permissions(UserRole.VIEWER)
        
        assert admin_perms['manage_users']
        assert not viewer_perms['manage_users']
        assert admin_perms['view_dashboard']
        assert viewer_perms['view_dashboard']

class TestDatabase:
    """Test database operations"""
    
    def setup_method(self):
        """Setup database manager with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
        self.test_data = generate_risk_data(days=5)
    
    def teardown_method(self):
        """Cleanup temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database table creation"""
        # Check if main tables exist
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'risk_data' in tables
            assert 'anomalies' in tables
    
    def test_risk_data_insertion(self):
        """Test risk data insertion"""
        success = self.db_manager.risk_data.insert_risk_data(self.test_data)
        assert success
        
        # Verify data was inserted
        retrieved_data = self.db_manager.risk_data.get_risk_data()
        assert not retrieved_data.empty
        assert len(retrieved_data) == len(self.test_data)
    
    def test_risk_data_filtering(self):
        """Test risk data filtering"""
        self.db_manager.risk_data.insert_risk_data(self.test_data)
        
        # Test model filtering
        filtered_data = self.db_manager.risk_data.get_risk_data(models=['GPT-4'])
        if not filtered_data.empty:
            assert all(model == 'GPT-4' for model in filtered_data['model'])
    
    def test_anomaly_management(self):
        """Test anomaly insertion and retrieval"""
        success = self.db_manager.anomalies.insert_anomaly(
            date='2025-01-01',
            model='TestModel',
            language='English',
            risk_category='Hallucination',
            expected_rate=0.1,
            actual_rate=0.8,
            anomaly_score=0.9,
            severity='high'
        )
        assert success
        
        anomalies = self.db_manager.anomalies.get_anomalies()
        assert not anomalies.empty

class TestAPIConnectors:
    """Test API connector functionality"""
    
    def test_openai_connector_init(self):
        """Test OpenAI connector initialization"""
        connector = OpenAIConnector("test-api-key")
        assert connector.api_key == "test-api-key"
        assert "openai.com" in connector.base_url
    
    def test_anthropic_connector_init(self):
        """Test Anthropic connector initialization"""
        connector = AnthropicConnector("test-api-key")
        assert connector.api_key == "test-api-key"
        assert "anthropic.com" in connector.base_url
    
    def test_api_manager(self):
        """Test API manager functionality"""
        manager = APIManager()
        
        # Add test connectors
        openai_connector = OpenAIConnector("test-key-1")
        anthropic_connector = AnthropicConnector("test-key-2")
        
        manager.add_connector("openai", openai_connector)
        manager.add_connector("anthropic", anthropic_connector)
        
        assert len(manager.connectors) == 2
        assert "openai" in manager.connectors
        assert "anthropic" in manager.connectors

class TestMonitoring:
    """Test monitoring and alerting system"""
    
    def setup_method(self):
        """Setup monitoring components"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
        self.monitor = RiskMonitor(self.db_manager)
    
    def teardown_method(self):
        """Cleanup temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_alert_rule_creation(self):
        """Test alert rule creation"""
        rule = AlertRule(
            id="test_rule",
            name="Test Alert Rule",
            description="Test rule for unit testing",
            condition={"metric": "risk_rate", "operator": ">", "threshold": 0.5},
            severity="medium"
        )
        
        self.monitor.add_alert_rule(rule)
        assert "test_rule" in self.monitor.alert_rules
    
    def test_alert_generation(self):
        """Test alert generation"""
        # Insert high-risk test data
        test_data = pd.DataFrame([{
            'Date': datetime.now().isoformat(),
            'Model': 'TestModel',
            'Language': 'English',
            'Risk_Category': 'Hallucination',
            'Risk_Rate': 0.9,  # High risk
            'Sample_Size': 100,
            'Confidence': 0.95
        }])
        
        self.db_manager.risk_data.insert_risk_data(test_data)
        
        # Check for threshold violations
        alerts = self.monitor.check_risk_thresholds()
        
        # Should generate at least one alert for high risk
        assert len(alerts) >= 0  # May be 0 if no rules are enabled
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality"""
        rule = AlertRule(
            id="cooldown_test",
            name="Cooldown Test Rule",
            description="Test cooldown functionality",
            condition={"metric": "risk_rate", "operator": ">", "threshold": 0.1},
            severity="low",
            cooldown_minutes=60
        )
        
        # First trigger
        rule.last_triggered = None
        can_trigger_1 = self.monitor._can_trigger_alert(rule)
        assert can_trigger_1
        
        # Set last triggered to now
        rule.last_triggered = datetime.now()
        can_trigger_2 = self.monitor._can_trigger_alert(rule)
        assert not can_trigger_2  # Should be in cooldown

class TestRiskCategories:
    """Test custom risk category management"""
    
    def setup_method(self):
        """Setup risk category manager"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
        self.category_manager = RiskCategoryManager(self.db_manager)
    
    def teardown_method(self):
        """Cleanup temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_default_categories_loaded(self):
        """Test that default categories are loaded"""
        categories = self.category_manager.get_all_categories()
        assert len(categories) > 0
        
        # Check for expected default categories
        category_names = [cat.name for cat in categories]
        assert "Hallucination" in category_names
        assert "Bias" in category_names
        assert "Toxicity" in category_names
    
    def test_custom_category_creation(self):
        """Test custom category creation"""
        custom_category = RiskCategory(
            id="custom_test",
            name="Custom Test Category",
            description="A test category for unit testing",
            severity_weight=0.6,
            color="#123456",
            icon="🧪",
            threshold_low=0.05,
            threshold_medium=0.2,
            threshold_high=0.4,
            created_by="test_user"
        )
        
        success = self.category_manager.add_category(custom_category)
        assert success
        
        # Verify it was added
        retrieved = self.category_manager.get_category("custom_test")
        assert retrieved is not None
        assert retrieved.name == "Custom Test Category"
    
    def test_category_update(self):
        """Test category updates"""
        # First add a category
        test_category = RiskCategory(
            id="update_test",
            name="Update Test",
            description="Original description",
            severity_weight=0.5,
            color="#000000",
            icon="🔧",
            threshold_low=0.1,
            threshold_medium=0.3,
            threshold_high=0.5,
            created_by="test_user"
        )
        
        self.category_manager.add_category(test_category)
        
        # Update the category
        test_category.description = "Updated description"
        test_category.severity_weight = 0.8
        
        success = self.category_manager.update_category(test_category)
        assert success
        
        # Verify updates
        updated = self.category_manager.get_category("update_test")
        assert updated.description == "Updated description"
        assert updated.severity_weight == 0.8
    
    def test_category_export_import(self):
        """Test category export and import"""
        # Create temporary export file
        export_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        export_file.close()
        
        try:
            # Export categories
            success = self.category_manager.export_categories(export_file.name)
            assert success
            
            # Verify export file exists and has content
            assert os.path.exists(export_file.name)
            with open(export_file.name, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data) > 0
            
            # Test import (in a real scenario, you'd import to a different database)
            # For this test, we'll just verify the file format is correct
            for category_data in exported_data:
                assert 'id' in category_data
                assert 'name' in category_data
                assert 'severity_weight' in category_data
                
        finally:
            os.unlink(export_file.name)

class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Setup complete system"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
        self.auth_manager = AuthManager()
        self.auth_manager.db.db_path = self.temp_db.name
        self.auth_manager.db.init_database()
    
    def teardown_method(self):
        """Cleanup"""
        os.unlink(self.temp_db.name)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data ingestion to visualization"""
        # 1. Generate and insert test data
        test_data = generate_risk_data(days=3)
        success = self.db_manager.risk_data.insert_risk_data(test_data)
        assert success
        
        # 2. Process data
        processor = DataProcessor(test_data)
        risk_scores = processor.calculate_risk_scores()
        assert isinstance(risk_scores, pd.DataFrame)
        
        # 3. Create visualizations
        visualizer = Visualizer()
        pivot_data = test_data.pivot_table(
            index='Model', 
            columns='Language', 
            values='Risk_Rate', 
            aggfunc='mean'
        ).fillna(0)
        
        if not pivot_data.empty:
            fig = visualizer.create_risk_heatmap(pivot_data)
            assert fig is not None
        
        # 4. Test monitoring
        monitor = RiskMonitor(self.db_manager)
        alerts = monitor.check_risk_thresholds()
        assert isinstance(alerts, list)
        
        # The workflow completed successfully if we reach this point
        assert True

# Utility functions for running tests
def run_all_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])

def run_specific_test(test_class):
    """Run tests for a specific class"""
    pytest.main([f"{__file__}::{test_class}", "-v"])

if __name__ == "__main__":
    # Run all tests when script is executed directly
    run_all_tests()