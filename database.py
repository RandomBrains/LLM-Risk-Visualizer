"""
Database management module for LLM Risk Visualizer
Handles data persistence, caching, and database operations
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
import redis
import os
from contextlib import contextmanager
from config import MODELS, LANGUAGES, RISK_CATEGORIES

class DatabaseConnection:
    """Manage database connections and transactions"""
    
    def __init__(self, db_path: str = "risk_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Risk data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                model TEXT NOT NULL,
                language TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                risk_rate REAL NOT NULL,
                sample_size INTEGER DEFAULT 100,
                confidence REAL DEFAULT 0.95,
                data_source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_risk_data_date 
            ON risk_data(date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_risk_data_model 
            ON risk_data(model)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_risk_data_composite 
            ON risk_data(date, model, language, risk_category)
        ''')
        
        # Anomalies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                model TEXT NOT NULL,
                language TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                expected_rate REAL NOT NULL,
                actual_rate REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                severity TEXT DEFAULT 'medium',
                description TEXT,
                acknowledged BOOLEAN DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # API configurations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT UNIQUE NOT NULL,
                api_key TEXT NOT NULL,
                base_url TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_sync TIMESTAMP,
                sync_frequency INTEGER DEFAULT 3600,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_name TEXT NOT NULL,
                report_type TEXT NOT NULL,
                parameters TEXT,
                generated_by TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        # Data export logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS export_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                export_type TEXT NOT NULL,
                filters TEXT,
                file_name TEXT,
                exported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                row_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        try:
            with self.get_connection() as conn:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
                return df
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()
    
    def execute_update(self, query: str, params: tuple = None) -> bool:
        """Execute update/insert/delete query"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return True
        except Exception as e:
            print(f"Update execution error: {e}")
            return False

class RiskDataManager:
    """Manage risk data operations"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def insert_risk_data(self, data: pd.DataFrame) -> bool:
        """Insert risk data into database"""
        try:
            with self.db.get_connection() as conn:
                data.to_sql('risk_data', conn, if_exists='append', index=False)
                return True
        except Exception as e:
            print(f"Error inserting risk data: {e}")
            return False
    
    def get_risk_data(self, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      models: Optional[List[str]] = None,
                      languages: Optional[List[str]] = None,
                      risk_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieve filtered risk data"""
        
        query = "SELECT * FROM risk_data WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())
        
        if models:
            placeholders = ','.join(['?' for _ in models])
            query += f" AND model IN ({placeholders})"
            params.extend(models)
        
        if languages:
            placeholders = ','.join(['?' for _ in languages])
            query += f" AND language IN ({placeholders})"
            params.extend(languages)
        
        if risk_categories:
            placeholders = ','.join(['?' for _ in risk_categories])
            query += f" AND risk_category IN ({placeholders})"
            params.extend(risk_categories)
        
        query += " ORDER BY date DESC, model, language, risk_category"
        
        return self.db.execute_query(query, tuple(params))
    
    def get_latest_data_by_model(self) -> pd.DataFrame:
        """Get latest risk data for each model"""
        query = '''
            SELECT model, language, risk_category, risk_rate, date
            FROM risk_data r1
            WHERE date = (
                SELECT MAX(date)
                FROM risk_data r2
                WHERE r1.model = r2.model 
                AND r1.language = r2.language
                AND r1.risk_category = r2.risk_category
            )
            ORDER BY model, language, risk_category
        '''
        return self.db.execute_query(query)
    
    def calculate_trend_metrics(self, days: int = 30) -> pd.DataFrame:
        """Calculate trend metrics for the specified period"""
        query = '''
            SELECT 
                model,
                language,
                risk_category,
                AVG(risk_rate) as avg_risk_rate,
                MIN(risk_rate) as min_risk_rate,
                MAX(risk_rate) as max_risk_rate,
                COUNT(*) as data_points,
                (SELECT risk_rate FROM risk_data r2 
                 WHERE r2.model = r1.model AND r2.language = r1.language 
                 AND r2.risk_category = r1.risk_category 
                 ORDER BY date DESC LIMIT 1) as latest_rate,
                (SELECT risk_rate FROM risk_data r3 
                 WHERE r3.model = r1.model AND r3.language = r1.language 
                 AND r3.risk_category = r1.risk_category 
                 ORDER BY date ASC LIMIT 1) as earliest_rate
            FROM risk_data r1
            WHERE date >= date('now', '-{} days')
            GROUP BY model, language, risk_category
            ORDER BY model, language, risk_category
        '''.format(days)
        
        return self.db.execute_query(query)
    
    def update_risk_data(self, record_id: int, updates: Dict) -> bool:
        """Update specific risk data record"""
        if not updates:
            return False
        
        set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
        query = f"UPDATE risk_data SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        
        params = list(updates.values()) + [record_id]
        return self.db.execute_update(query, tuple(params))
    
    def delete_risk_data(self, record_id: int) -> bool:
        """Delete specific risk data record"""
        query = "DELETE FROM risk_data WHERE id = ?"
        return self.db.execute_update(query, (record_id,))
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the data"""
        queries = {
            'total_records': "SELECT COUNT(*) as count FROM risk_data",
            'date_range': "SELECT MIN(date) as min_date, MAX(date) as max_date FROM risk_data",
            'unique_models': "SELECT COUNT(DISTINCT model) as count FROM risk_data",
            'unique_languages': "SELECT COUNT(DISTINCT language) as count FROM risk_data",
            'avg_risk_by_category': '''
                SELECT risk_category, AVG(risk_rate) as avg_rate 
                FROM risk_data 
                GROUP BY risk_category 
                ORDER BY avg_rate DESC
            '''
        }
        
        summary = {}
        for key, query in queries.items():
            result = self.db.execute_query(query)
            if not result.empty:
                summary[key] = result.to_dict('records')
        
        return summary

class AnomalyManager:
    """Manage anomaly detection and tracking"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def insert_anomaly(self, 
                      date: str,
                      model: str,
                      language: str,
                      risk_category: str,
                      expected_rate: float,
                      actual_rate: float,
                      anomaly_score: float,
                      severity: str = 'medium',
                      description: str = '') -> bool:
        """Insert detected anomaly"""
        query = '''
            INSERT INTO anomalies 
            (date, model, language, risk_category, expected_rate, actual_rate, 
             anomaly_score, severity, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (date, model, language, risk_category, expected_rate, 
                 actual_rate, anomaly_score, severity, description)
        return self.db.execute_update(query, params)
    
    def get_anomalies(self,
                     acknowledged: Optional[bool] = None,
                     severity: Optional[str] = None,
                     days: Optional[int] = None) -> pd.DataFrame:
        """Retrieve anomalies with filters"""
        query = "SELECT * FROM anomalies WHERE 1=1"
        params = []
        
        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(int(acknowledged))
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if days:
            query += " AND date >= date('now', '-{} days')".format(days)
        
        query += " ORDER BY created_at DESC"
        
        return self.db.execute_query(query, tuple(params))
    
    def acknowledge_anomaly(self, anomaly_id: int, acknowledged_by: str) -> bool:
        """Mark anomaly as acknowledged"""
        query = '''
            UPDATE anomalies 
            SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = CURRENT_TIMESTAMP
            WHERE id = ?
        '''
        return self.db.execute_update(query, (acknowledged_by, anomaly_id))
    
    def get_anomaly_summary(self) -> Dict:
        """Get anomaly summary statistics"""
        queries = {
            'total_anomalies': "SELECT COUNT(*) as count FROM anomalies",
            'unacknowledged': "SELECT COUNT(*) as count FROM anomalies WHERE acknowledged = 0",
            'by_severity': '''
                SELECT severity, COUNT(*) as count 
                FROM anomalies 
                GROUP BY severity 
                ORDER BY count DESC
            ''',
            'recent_anomalies': '''
                SELECT COUNT(*) as count 
                FROM anomalies 
                WHERE date >= date('now', '-7 days')
            '''
        }
        
        summary = {}
        for key, query in queries.items():
            result = self.db.execute_query(query)
            if not result.empty:
                if key in ['by_severity']:
                    summary[key] = result.to_dict('records')
                else:
                    summary[key] = result.iloc[0].to_dict()
        
        return summary

class CacheManager:
    """Manage Redis caching for performance optimization"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None
            self.enabled = False
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data"""
        if not self.enabled:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data.encode('latin1'))
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, data: pd.DataFrame, expire_seconds: int = 3600):
        """Cache data with expiration"""
        if not self.enabled:
            return
        
        try:
            serialized_data = pickle.dumps(data).decode('latin1')
            self.redis_client.setex(key, expire_seconds, serialized_data)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete cached data"""
        if not self.enabled:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    def clear_all(self):
        """Clear all cached data"""
        if not self.enabled:
            return
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    def generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, list):
                v = '_'.join(map(str, v))
            key_parts.append(f"{k}:{v}")
        return ':'.join(key_parts)

class DataExporter:
    """Handle data export operations"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def export_to_csv(self, 
                     data: pd.DataFrame, 
                     filename: str,
                     user_id: Optional[int] = None) -> Tuple[bool, str]:
        """Export data to CSV file"""
        try:
            full_path = os.path.join('exports', filename)
            os.makedirs('exports', exist_ok=True)
            
            data.to_csv(full_path, index=False)
            
            # Log export
            if user_id:
                self._log_export(user_id, 'csv', '', filename, len(data))
            
            return True, full_path
        except Exception as e:
            return False, f"Export failed: {e}"
    
    def export_to_json(self, 
                      data: pd.DataFrame, 
                      filename: str,
                      user_id: Optional[int] = None) -> Tuple[bool, str]:
        """Export data to JSON file"""
        try:
            full_path = os.path.join('exports', filename)
            os.makedirs('exports', exist_ok=True)
            
            data.to_json(full_path, orient='records', date_format='iso')
            
            # Log export
            if user_id:
                self._log_export(user_id, 'json', '', filename, len(data))
            
            return True, full_path
        except Exception as e:
            return False, f"Export failed: {e}"
    
    def _log_export(self, user_id: int, export_type: str, filters: str, filename: str, row_count: int):
        """Log export operation"""
        query = '''
            INSERT INTO export_logs (user_id, export_type, filters, file_name, row_count)
            VALUES (?, ?, ?, ?, ?)
        '''
        self.db.execute_update(query, (user_id, export_type, filters, filename, row_count))

class DatabaseManager:
    """Main database manager coordinating all database operations"""
    
    def __init__(self, 
                 db_path: str = "risk_data.db",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        self.connection = DatabaseConnection(db_path)
        self.risk_data = RiskDataManager(self.connection)
        self.anomalies = AnomalyManager(self.connection)
        self.cache = CacheManager(redis_host, redis_port)
        self.exporter = DataExporter(self.connection)
    
    def get_cached_risk_data(self, **filters) -> pd.DataFrame:
        """Get risk data with caching"""
        cache_key = self.cache.generate_cache_key('risk_data', **filters)
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch from database
        data = self.risk_data.get_risk_data(**filters)
        
        # Cache for 1 hour
        if not data.empty:
            self.cache.set(cache_key, data, 3600)
        
        return data
    
    def invalidate_cache(self, pattern: str = 'risk_data*'):
        """Invalidate cache entries matching pattern"""
        # For simplicity, clear all cache
        # In production, implement pattern-based deletion
        self.cache.clear_all()
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all database components"""
        return {
            'database': self._test_database_connection(),
            'cache': self.cache.enabled,
            'exports_dir': os.path.exists('exports')
        }
    
    def _test_database_connection(self) -> bool:
        """Test database connection"""
        try:
            result = self.connection.execute_query("SELECT 1")
            return not result.empty
        except:
            return False