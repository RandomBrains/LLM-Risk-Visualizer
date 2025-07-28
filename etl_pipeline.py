"""
Advanced Data Pipeline and ETL Automation Module
Implements comprehensive data ingestion, transformation, and loading capabilities
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import redis
from pathlib import Path
import yaml

from database import DatabaseManager
from api import APIManager
from monitoring import MonitoringService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Represents a data source configuration"""
    source_id: str
    source_type: str  # 'api', 'file', 'database', 'stream'
    name: str
    connection_config: Dict[str, Any]
    schedule_config: Dict[str, Any]
    transformation_rules: List[Dict[str, Any]]
    data_quality_rules: List[Dict[str, Any]]
    is_active: bool = True
    last_sync: Optional[datetime] = None
    next_sync: Optional[datetime] = None

@dataclass
class PipelineJob:
    """Represents a data pipeline job"""
    job_id: str
    job_name: str
    source_ids: List[str]
    transformations: List[Dict[str, Any]]
    destination_config: Dict[str, Any]
    schedule: str  # Cron-like schedule
    dependencies: List[str]  # Other job IDs this depends on
    retry_config: Dict[str, Any]
    notification_config: Dict[str, Any]
    is_active: bool = True

@dataclass
class JobExecution:
    """Represents a job execution instance"""
    execution_id: str
    job_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    records_processed: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class DataTransformer(ABC):
    """Abstract base class for data transformers"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data according to configuration"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate transformation configuration"""
        pass

class StandardDataTransformer(DataTransformer):
    """Standard data transformation operations"""
    
    def transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply standard transformations"""
        transformed_data = data.copy()
        
        # Apply transformations based on config
        for operation in config.get('operations', []):
            operation_type = operation['type']
            
            if operation_type == 'rename_columns':
                column_mapping = operation['mapping']
                transformed_data = transformed_data.rename(columns=column_mapping)
            
            elif operation_type == 'filter_rows':
                filter_condition = operation['condition']
                transformed_data = self._apply_filter(transformed_data, filter_condition)
            
            elif operation_type == 'add_calculated_column':
                column_name = operation['column_name']
                expression = operation['expression']
                transformed_data[column_name] = self._evaluate_expression(transformed_data, expression)
            
            elif operation_type == 'aggregate':
                group_by = operation['group_by']
                aggregations = operation['aggregations']
                transformed_data = self._apply_aggregation(transformed_data, group_by, aggregations)
            
            elif operation_type == 'normalize_values':
                columns = operation['columns']
                method = operation.get('method', 'min_max')
                transformed_data = self._normalize_columns(transformed_data, columns, method)
            
            elif operation_type == 'handle_missing_values':
                strategy = operation['strategy']
                columns = operation.get('columns', [])
                transformed_data = self._handle_missing_values(transformed_data, strategy, columns)
            
            elif operation_type == 'data_type_conversion':
                conversions = operation['conversions']
                transformed_data = self._convert_data_types(transformed_data, conversions)
            
            elif operation_type == 'remove_duplicates':
                subset = operation.get('subset', None)
                transformed_data = transformed_data.drop_duplicates(subset=subset, keep='first')
            
            elif operation_type == 'pivot_table':
                pivot_config = operation['config']
                transformed_data = self._create_pivot_table(transformed_data, pivot_config)
        
        return transformed_data
    
    def _apply_filter(self, data: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Apply filter condition to data"""
        try:
            # Simple condition evaluation (enhance for production)
            return data.query(condition)
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return data
    
    def _evaluate_expression(self, data: pd.DataFrame, expression: str) -> pd.Series:
        """Evaluate mathematical expression for new column"""
        try:
            # Simple expression evaluation (enhance for production)
            return data.eval(expression)
        except Exception as e:
            logger.error(f"Expression evaluation failed: {e}")
            return pd.Series([None] * len(data))
    
    def _apply_aggregation(self, data: pd.DataFrame, group_by: List[str], aggregations: Dict[str, str]) -> pd.DataFrame:
        """Apply aggregation operations"""
        try:
            grouped = data.groupby(group_by)
            return grouped.agg(aggregations).reset_index()
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return data
    
    def _normalize_columns(self, data: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
        """Normalize specified columns"""
        normalized_data = data.copy()
        
        for column in columns:
            if column in data.columns:
                if method == 'min_max':
                    min_val = data[column].min()
                    max_val = data[column].max()
                    if max_val != min_val:
                        normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
                
                elif method == 'z_score':
                    mean_val = data[column].mean()
                    std_val = data[column].std()
                    if std_val != 0:
                        normalized_data[column] = (data[column] - mean_val) / std_val
        
        return normalized_data
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str, columns: List[str]) -> pd.DataFrame:
        """Handle missing values in data"""
        processed_data = data.copy()
        target_columns = columns if columns else data.columns
        
        if strategy == 'drop':
            processed_data = processed_data.dropna(subset=target_columns)
        elif strategy == 'fill_mean':
            for col in target_columns:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    processed_data[col] = processed_data[col].fillna(data[col].mean())
        elif strategy == 'fill_median':
            for col in target_columns:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    processed_data[col] = processed_data[col].fillna(data[col].median())
        elif strategy == 'fill_mode':
            for col in target_columns:
                if col in data.columns:
                    mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else data[col].iloc[0]
                    processed_data[col] = processed_data[col].fillna(mode_value)
        elif strategy == 'forward_fill':
            processed_data[target_columns] = processed_data[target_columns].fillna(method='ffill')
        elif strategy == 'backward_fill':
            processed_data[target_columns] = processed_data[target_columns].fillna(method='bfill')
        
        return processed_data
    
    def _convert_data_types(self, data: pd.DataFrame, conversions: Dict[str, str]) -> pd.DataFrame:
        """Convert data types for specified columns"""
        converted_data = data.copy()
        
        for column, target_type in conversions.items():
            if column in data.columns:
                try:
                    if target_type == 'datetime':
                        converted_data[column] = pd.to_datetime(data[column])
                    elif target_type == 'category':
                        converted_data[column] = data[column].astype('category')
                    else:
                        converted_data[column] = data[column].astype(target_type)
                except Exception as e:
                    logger.error(f"Type conversion failed for column {column}: {e}")
        
        return converted_data
    
    def _create_pivot_table(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create pivot table from data"""
        try:
            return pd.pivot_table(
                data,
                values=config.get('values'),
                index=config.get('index'),
                columns=config.get('columns'),
                aggfunc=config.get('aggfunc', 'mean'),
                fill_value=config.get('fill_value', 0)
            ).reset_index()
        except Exception as e:
            logger.error(f"Pivot table creation failed: {e}")
            return data
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate transformation configuration"""
        if 'operations' not in config:
            return False, "Missing 'operations' in configuration"
        
        for operation in config['operations']:
            if 'type' not in operation:
                return False, "Missing 'type' in operation"
        
        return True, "Configuration is valid"

class DataQualityValidator:
    """Validates data quality based on defined rules"""
    
    def __init__(self):
        self.validation_rules = {
            'not_null': self._validate_not_null,
            'unique': self._validate_unique,
            'range': self._validate_range,
            'format': self._validate_format,
            'custom': self._validate_custom
        }
    
    def validate_data(self, data: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data against quality rules"""
        validation_results = {
            'overall_status': 'passed',
            'total_records': len(data),
            'rule_results': [],
            'failed_records': 0,
            'warnings': []
        }
        
        for rule in rules:
            rule_type = rule['type']
            rule_name = rule.get('name', f"{rule_type}_rule")
            
            if rule_type in self.validation_rules:
                result = self.validation_rules[rule_type](data, rule)
                result['rule_name'] = rule_name
                validation_results['rule_results'].append(result)
                
                if not result['passed']:
                    validation_results['overall_status'] = 'failed'
                    validation_results['failed_records'] += result.get('failed_count', 0)
            else:
                validation_results['warnings'].append(f"Unknown rule type: {rule_type}")
        
        return validation_results
    
    def _validate_not_null(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that specified columns are not null"""
        columns = rule['columns']
        null_counts = data[columns].isnull().sum()
        
        failed_columns = null_counts[null_counts > 0]
        
        return {
            'type': 'not_null',
            'passed': len(failed_columns) == 0,
            'failed_count': null_counts.sum(),
            'details': failed_columns.to_dict() if not failed_columns.empty else {}
        }
    
    def _validate_unique(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that specified columns have unique values"""
        columns = rule['columns']
        
        if isinstance(columns, list):
            # Check combination uniqueness
            duplicate_count = data.duplicated(subset=columns).sum()
        else:
            # Check single column uniqueness
            duplicate_count = data.duplicated(subset=[columns]).sum()
        
        return {
            'type': 'unique',
            'passed': duplicate_count == 0,
            'failed_count': duplicate_count,
            'details': {'duplicate_records': duplicate_count}
        }
    
    def _validate_range(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that numeric values are within specified range"""
        column = rule['column']
        min_value = rule.get('min')
        max_value = rule.get('max')
        
        out_of_range_mask = pd.Series([False] * len(data))
        
        if min_value is not None:
            out_of_range_mask |= data[column] < min_value
        
        if max_value is not None:
            out_of_range_mask |= data[column] > max_value
        
        failed_count = out_of_range_mask.sum()
        
        return {
            'type': 'range',
            'passed': failed_count == 0,
            'failed_count': failed_count,
            'details': {
                'min_value': min_value,
                'max_value': max_value,
                'out_of_range_records': failed_count
            }
        }
    
    def _validate_format(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that string values match specified format/pattern"""
        column = rule['column']
        pattern = rule['pattern']
        
        import re
        pattern_matches = data[column].astype(str).str.match(pattern, na=False)
        failed_count = (~pattern_matches).sum()
        
        return {
            'type': 'format',
            'passed': failed_count == 0,
            'failed_count': failed_count,
            'details': {
                'pattern': pattern,
                'invalid_format_records': failed_count
            }
        }
    
    def _validate_custom(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using custom function"""
        function_name = rule['function']
        parameters = rule.get('parameters', {})
        
        # This would be extended to support custom validation functions
        # For now, return a placeholder result
        return {
            'type': 'custom',
            'passed': True,
            'failed_count': 0,
            'details': {'function': function_name, 'parameters': parameters}
        }

class ETLPipelineEngine:
    """Main ETL pipeline execution engine"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.data_sources: Dict[str, DataSource] = {}
        self.pipeline_jobs: Dict[str, PipelineJob] = {}
        self.transformers: Dict[str, DataTransformer] = {
            'standard': StandardDataTransformer()
        }
        self.quality_validator = DataQualityValidator()
        self.job_executions: Dict[str, JobExecution] = {}
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Initialize pipeline database
        self.init_pipeline_database()
    
    def init_pipeline_database(self):
        """Initialize pipeline management database"""
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        
        # Data sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                name TEXT NOT NULL,
                connection_config TEXT,
                schedule_config TEXT,
                transformation_rules TEXT,
                data_quality_rules TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_sync TIMESTAMP,
                next_sync TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pipeline jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_jobs (
                job_id TEXT PRIMARY KEY,
                job_name TEXT NOT NULL,
                source_ids TEXT,
                transformations TEXT,
                destination_config TEXT,
                schedule TEXT,
                dependencies TEXT,
                retry_config TEXT,
                notification_config TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Job executions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_executions (
                execution_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                records_processed INTEGER DEFAULT 0,
                error_message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_data_source(self, data_source: DataSource) -> bool:
        """Register a new data source"""
        try:
            self.data_sources[data_source.source_id] = data_source
            
            # Persist to database
            conn = sqlite3.connect("pipeline.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO data_sources 
                (source_id, source_type, name, connection_config, schedule_config, 
                 transformation_rules, data_quality_rules, is_active, last_sync, next_sync)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_source.source_id,
                data_source.source_type,
                data_source.name,
                json.dumps(data_source.connection_config),
                json.dumps(data_source.schedule_config),
                json.dumps(data_source.transformation_rules),
                json.dumps(data_source.data_quality_rules),
                data_source.is_active,
                data_source.last_sync.isoformat() if data_source.last_sync else None,
                data_source.next_sync.isoformat() if data_source.next_sync else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Data source {data_source.source_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register data source: {e}")
            return False
    
    def register_pipeline_job(self, job: PipelineJob) -> bool:
        """Register a new pipeline job"""
        try:
            self.pipeline_jobs[job.job_id] = job
            
            # Persist to database
            conn = sqlite3.connect("pipeline.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pipeline_jobs 
                (job_id, job_name, source_ids, transformations, destination_config,
                 schedule, dependencies, retry_config, notification_config, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.job_id,
                job.job_name,
                json.dumps(job.source_ids),
                json.dumps(job.transformations),
                json.dumps(job.destination_config),
                job.schedule,
                json.dumps(job.dependencies),
                json.dumps(job.retry_config),
                json.dumps(job.notification_config),
                job.is_active
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Pipeline job {job.job_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register pipeline job: {e}")
            return False
    
    def execute_job(self, job_id: str) -> JobExecution:
        """Execute a pipeline job"""
        if job_id not in self.pipeline_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.pipeline_jobs[job_id]
        execution_id = f"{job_id}_{int(time.time())}"
        
        execution = JobExecution(
            execution_id=execution_id,
            job_id=job_id,
            start_time=datetime.now(),
            status="running"
        )
        
        self.job_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting execution of job {job_id}")
            
            # Check dependencies
            if not self._check_dependencies(job.dependencies):
                execution.status = "failed"
                execution.error_message = "Job dependencies not met"
                execution.end_time = datetime.now()
                return execution
            
            # Collect data from sources
            all_data = []
            for source_id in job.source_ids:
                if source_id in self.data_sources:
                    source_data = self._fetch_data_from_source(self.data_sources[source_id])
                    if source_data is not None and not source_data.empty:
                        all_data.append(source_data)
            
            if not all_data:
                execution.status = "failed"
                execution.error_message = "No data retrieved from sources"
                execution.end_time = datetime.now()
                return execution
            
            # Combine data from multiple sources
            combined_data = pd.concat(all_data, ignore_index=True)
            initial_record_count = len(combined_data)
            
            # Apply transformations
            for transformation in job.transformations:
                transformer_type = transformation.get('transformer', 'standard')
                if transformer_type in self.transformers:
                    transformer = self.transformers[transformer_type]
                    combined_data = transformer.transform(combined_data, transformation)
            
            # Validate data quality
            quality_results = self._run_quality_checks(combined_data, job_id)
            
            if quality_results['overall_status'] == 'failed':
                execution.status = "failed"
                execution.error_message = f"Data quality validation failed: {quality_results}"
                execution.end_time = datetime.now()
                return execution
            
            # Load data to destination
            load_result = self._load_data_to_destination(combined_data, job.destination_config)
            
            if load_result:
                execution.status = "completed"
                execution.records_processed = len(combined_data)
                execution.metadata = {
                    'initial_records': initial_record_count,
                    'final_records': len(combined_data),
                    'quality_results': quality_results,
                    'transformations_applied': len(job.transformations)
                }
            else:
                execution.status = "failed"
                execution.error_message = "Failed to load data to destination"
            
            execution.end_time = datetime.now()
            
            # Send notifications if configured
            self._send_job_notifications(job, execution)
            
            logger.info(f"Job {job_id} execution completed with status: {execution.status}")
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            logger.error(f"Job {job_id} execution failed: {e}")
        
        # Persist execution record
        self._persist_job_execution(execution)
        
        return execution
    
    def _fetch_data_from_source(self, data_source: DataSource) -> Optional[pd.DataFrame]:
        """Fetch data from a specific data source"""
        try:
            if data_source.source_type == 'api':
                return self._fetch_from_api(data_source)
            elif data_source.source_type == 'file':
                return self._fetch_from_file(data_source)
            elif data_source.source_type == 'database':
                return self._fetch_from_database(data_source)
            else:
                logger.warning(f"Unsupported source type: {data_source.source_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch data from source {data_source.source_id}: {e}")
            return None
    
    def _fetch_from_api(self, data_source: DataSource) -> Optional[pd.DataFrame]:
        """Fetch data from API source"""
        config = data_source.connection_config
        
        try:
            # This would integrate with the existing API module
            # For now, return mock data
            
            # Example implementation:
            # api_manager = APIManager()
            # return api_manager.fetch_data(config)
            
            # Mock data for demonstration
            mock_data = pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Model': ['GPT-4'] * 10,
                'Language': ['English'] * 10,
                'Risk_Category': ['Hallucination'] * 10,
                'Risk_Rate': np.random.uniform(0.1, 0.9, 10),
                'Sample_Size': np.random.randint(50, 200, 10),
                'Confidence': np.random.uniform(0.8, 0.99, 10)
            })
            
            return mock_data
            
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return None
    
    def _fetch_from_file(self, data_source: DataSource) -> Optional[pd.DataFrame]:
        """Fetch data from file source"""
        config = data_source.connection_config
        file_path = config.get('file_path')
        file_type = config.get('file_type', 'csv')
        
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path)
            elif file_type == 'json':
                return pd.read_json(file_path)
            elif file_type == 'excel':
                return pd.read_excel(file_path)
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return None
        
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return None
    
    def _fetch_from_database(self, data_source: DataSource) -> Optional[pd.DataFrame]:
        """Fetch data from database source"""
        config = data_source.connection_config
        
        try:
            # Use existing database manager
            query = config.get('query')
            if query:
                return self.db_manager.connection.execute_query(query)
            else:
                # Fetch from risk_data table
                return self.db_manager.risk_data.get_risk_data()
                
        except Exception as e:
            logger.error(f"Database fetch failed: {e}")
            return None
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if job dependencies are satisfied"""
        for dep_job_id in dependencies:
            # Check if dependency job has completed successfully recently
            recent_executions = self._get_recent_executions(dep_job_id, hours=24)
            
            if not recent_executions or not any(exec.status == "completed" for exec in recent_executions):
                logger.warning(f"Dependency {dep_job_id} not satisfied")
                return False
        
        return True
    
    def _run_quality_checks(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Run data quality checks"""
        # Get quality rules for the job (simplified)
        default_rules = [
            {
                'type': 'not_null',
                'name': 'required_fields_check',
                'columns': ['Date', 'Model', 'Risk_Rate']
            },
            {
                'type': 'range',
                'name': 'risk_rate_range_check',
                'column': 'Risk_Rate',
                'min': 0.0,
                'max': 1.0
            }
        ]
        
        return self.quality_validator.validate_data(data, default_rules)
    
    def _load_data_to_destination(self, data: pd.DataFrame, destination_config: Dict[str, Any]) -> bool:
        """Load data to specified destination"""
        destination_type = destination_config.get('type', 'database')
        
        try:
            if destination_type == 'database':
                table_name = destination_config.get('table', 'risk_data')
                return self.db_manager.risk_data.insert_risk_data(data)
            
            elif destination_type == 'file':
                file_path = destination_config.get('file_path')
                file_format = destination_config.get('format', 'csv')
                
                if file_format == 'csv':
                    data.to_csv(file_path, index=False)
                elif file_format == 'json':
                    data.to_json(file_path, orient='records', date_format='iso')
                elif file_format == 'parquet':
                    data.to_parquet(file_path, index=False)
                
                return True
            
            else:
                logger.error(f"Unsupported destination type: {destination_type}")
                return False
                
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False
    
    def _send_job_notifications(self, job: PipelineJob, execution: JobExecution):
        """Send job execution notifications"""
        notification_config = job.notification_config
        
        if not notification_config.get('enabled', False):
            return
        
        # Email notification
        if notification_config.get('email'):
            self._send_email_notification(job, execution, notification_config['email'])
        
        # Webhook notification
        if notification_config.get('webhook'):
            self._send_webhook_notification(job, execution, notification_config['webhook'])
    
    def _send_email_notification(self, job: PipelineJob, execution: JobExecution, email_config: Dict[str, Any]):
        """Send email notification about job execution"""
        # This would integrate with the monitoring/notification system
        logger.info(f"Email notification sent for job {job.job_id} execution {execution.execution_id}")
    
    def _send_webhook_notification(self, job: PipelineJob, execution: JobExecution, webhook_config: Dict[str, Any]):
        """Send webhook notification about job execution"""
        try:
            webhook_url = webhook_config.get('url')
            payload = {
                'job_id': job.job_id,
                'job_name': job.job_name,
                'execution_id': execution.execution_id,
                'status': execution.status,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'records_processed': execution.records_processed,
                'error_message': execution.error_message
            }
            
            import requests
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    def _persist_job_execution(self, execution: JobExecution):
        """Persist job execution record to database"""
        try:
            conn = sqlite3.connect("pipeline.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO job_executions 
                (execution_id, job_id, start_time, end_time, status, 
                 records_processed, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.job_id,
                execution.start_time.isoformat(),
                execution.end_time.isoformat() if execution.end_time else None,
                execution.status,
                execution.records_processed,
                execution.error_message,
                json.dumps(execution.metadata) if execution.metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist job execution: {e}")
    
    def _get_recent_executions(self, job_id: str, hours: int = 24) -> List[JobExecution]:
        """Get recent executions for a job"""
        try:
            conn = sqlite3.connect("pipeline.db")
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cursor.execute('''
                SELECT execution_id, job_id, start_time, end_time, status, 
                       records_processed, error_message, metadata
                FROM job_executions
                WHERE job_id = ? AND start_time >= ?
                ORDER BY start_time DESC
            ''', (job_id, cutoff_time.isoformat()))
            
            executions = []
            for row in cursor.fetchall():
                execution = JobExecution(
                    execution_id=row[0],
                    job_id=row[1],
                    start_time=datetime.fromisoformat(row[2]),
                    end_time=datetime.fromisoformat(row[3]) if row[3] else None,
                    status=row[4],
                    records_processed=row[5],
                    error_message=row[6],
                    metadata=json.loads(row[7]) if row[7] else None
                )
                executions.append(execution)
            
            conn.close()
            return executions
            
        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []
    
    def start_scheduler(self):
        """Start the pipeline job scheduler"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Pipeline scheduler started")
    
    def stop_scheduler(self):
        """Stop the pipeline job scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Pipeline scheduler stopped")
    
    def _run_scheduler(self):
        """Run the job scheduler"""
        while self.scheduler_running:
            try:
                # Check for jobs that need to be executed
                for job_id, job in self.pipeline_jobs.items():
                    if job.is_active and self._should_execute_job(job):
                        try:
                            self.execute_job(job_id)
                        except Exception as e:
                            logger.error(f"Scheduled job execution failed: {e}")
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _should_execute_job(self, job: PipelineJob) -> bool:
        """Check if a job should be executed based on its schedule"""
        # Simple implementation - would be enhanced with proper cron parsing
        # For now, just check if it's been more than an hour since last execution
        
        recent_executions = self._get_recent_executions(job.job_id, hours=1)
        return len(recent_executions) == 0
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current status and statistics for a job"""
        if job_id not in self.pipeline_jobs:
            return {'error': 'Job not found'}
        
        job = self.pipeline_jobs[job_id]
        recent_executions = self._get_recent_executions(job_id, hours=24)
        
        return {
            'job_id': job_id,
            'job_name': job.job_name,
            'is_active': job.is_active,
            'last_execution': recent_executions[0] if recent_executions else None,
            'executions_today': len(recent_executions),
            'success_rate': self._calculate_success_rate(job_id),
            'avg_execution_time': self._calculate_avg_execution_time(job_id),
            'data_sources': len(job.source_ids)
        }
    
    def _calculate_success_rate(self, job_id: str, days: int = 7) -> float:
        """Calculate job success rate over specified period"""
        executions = self._get_recent_executions(job_id, hours=days * 24)
        
        if not executions:
            return 0.0
        
        successful = sum(1 for exec in executions if exec.status == "completed")
        return successful / len(executions)
    
    def _calculate_avg_execution_time(self, job_id: str, days: int = 7) -> Optional[float]:
        """Calculate average execution time in seconds"""
        executions = self._get_recent_executions(job_id, hours=days * 24)
        
        completed_executions = [
            exec for exec in executions 
            if exec.status == "completed" and exec.end_time
        ]
        
        if not completed_executions:
            return None
        
        total_time = sum(
            (exec.end_time - exec.start_time).total_seconds() 
            for exec in completed_executions
        )
        
        return total_time / len(completed_executions)

# Streamlit integration functions
def initialize_etl_pipeline():
    """Initialize ETL pipeline for Streamlit app"""
    if 'etl_engine' not in st.session_state:
        from database import DatabaseManager
        db_manager = DatabaseManager()
        st.session_state.etl_engine = ETLPipelineEngine(db_manager)
    
    return st.session_state.etl_engine

def render_etl_dashboard():
    """Render ETL pipeline management dashboard"""
    st.header("🔄 Data Pipeline & ETL Management")
    
    etl_engine = initialize_etl_pipeline()
    
    # Pipeline overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Sources", len(etl_engine.data_sources))
    
    with col2:
        st.metric("Pipeline Jobs", len(etl_engine.pipeline_jobs))
    
    with col3:
        active_jobs = sum(1 for job in etl_engine.pipeline_jobs.values() if job.is_active)
        st.metric("Active Jobs", active_jobs)
    
    with col4:
        scheduler_status = "Running" if etl_engine.scheduler_running else "Stopped"
        st.metric("Scheduler", scheduler_status)
    
    # Tabs for different ETL aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Jobs Status", "🔧 Configuration", "📈 Monitoring", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Pipeline Jobs Status")
        
        if etl_engine.pipeline_jobs:
            for job_id, job in etl_engine.pipeline_jobs.items():
                with st.expander(f"{job.job_name} ({job_id})"):
                    status = etl_engine.get_job_status(job_id)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Status:** {'🟢 Active' if job.is_active else '🔴 Inactive'}")
                        st.write(f"**Success Rate:** {status['success_rate']:.1%}")
                    
                    with col2:
                        st.write(f"**Executions Today:** {status['executions_today']}")
                        avg_time = status['avg_execution_time']
                        if avg_time:
                            st.write(f"**Avg Duration:** {avg_time:.1f}s")
                        else:
                            st.write("**Avg Duration:** N/A")
                    
                    with col3:
                        st.write(f"**Data Sources:** {status['data_sources']}")
                        
                        if st.button(f"Run Now", key=f"run_{job_id}"):
                            with st.spinner("Executing job..."):
                                execution = etl_engine.execute_job(job_id)
                                if execution.status == "completed":
                                    st.success(f"Job completed successfully! Processed {execution.records_processed} records.")
                                else:
                                    st.error(f"Job failed: {execution.error_message}")
        else:
            st.info("No pipeline jobs configured. Add jobs in the Configuration tab.")
    
    with tab2:
        st.subheader("Pipeline Configuration")
        
        # Quick job setup
        with st.form("quick_job_setup"):
            st.write("**Quick Job Setup:**")
            
            job_name = st.text_input("Job Name")
            source_type = st.selectbox("Data Source Type", ["database", "api", "file"])
            schedule = st.selectbox("Schedule", ["hourly", "daily", "weekly"])
            
            submitted = st.form_submit_button("Create Job")
            
            if submitted and job_name:
                # Create a simple job configuration
                job_id = f"job_{int(time.time())}"
                
                data_source = DataSource(
                    source_id=f"source_{job_id}",
                    source_type=source_type,
                    name=f"{job_name} Source",
                    connection_config={},
                    schedule_config={"frequency": schedule},
                    transformation_rules=[],
                    data_quality_rules=[]
                )
                
                pipeline_job = PipelineJob(
                    job_id=job_id,
                    job_name=job_name,
                    source_ids=[data_source.source_id],
                    transformations=[{
                        "transformer": "standard",
                        "operations": []
                    }],
                    destination_config={"type": "database", "table": "risk_data"},
                    schedule=schedule,
                    dependencies=[],
                    retry_config={"max_retries": 3},
                    notification_config={"enabled": False}
                )
                
                # Register the source and job
                etl_engine.register_data_source(data_source)
                etl_engine.register_pipeline_job(pipeline_job)
                
                st.success(f"Job '{job_name}' created successfully!")
                st.rerun()
    
    with tab3:
        st.subheader("Pipeline Monitoring")
        
        # Recent executions
        st.write("**Recent Job Executions:**")
        
        all_executions = []
        for job_id in etl_engine.pipeline_jobs.keys():
            recent = etl_engine._get_recent_executions(job_id, hours=24)
            all_executions.extend(recent)
        
        if all_executions:
            # Sort by start time
            all_executions.sort(key=lambda x: x.start_time, reverse=True)
            
            # Display in table format
            execution_data = []
            for exec in all_executions[:10]:  # Show last 10
                duration = ""
                if exec.end_time:
                    duration = f"{(exec.end_time - exec.start_time).total_seconds():.1f}s"
                
                execution_data.append({
                    'Job ID': exec.job_id,
                    'Status': exec.status,
                    'Start Time': exec.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Duration': duration,
                    'Records': exec.records_processed,
                    'Error': exec.error_message or ""
                })
            
            st.dataframe(pd.DataFrame(execution_data), use_container_width=True)
        else:
            st.info("No recent job executions found.")
    
    with tab4:
        st.subheader("Pipeline Settings")
        
        # Scheduler controls
        col1, col2 = st.columns(2)
        
        with col1:
            if not etl_engine.scheduler_running:
                if st.button("▶️ Start Scheduler"):
                    etl_engine.start_scheduler()
                    st.success("Scheduler started!")
                    st.rerun()
            else:
                if st.button("⏹️ Stop Scheduler"):
                    etl_engine.stop_scheduler()
                    st.success("Scheduler stopped!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Restart Scheduler"):
                etl_engine.stop_scheduler()
                time.sleep(1)
                etl_engine.start_scheduler()
                st.success("Scheduler restarted!")
                st.rerun()
        
        # Global settings
        st.write("**Global Pipeline Settings:**")
        
        max_concurrent_jobs = st.number_input("Max Concurrent Jobs", min_value=1, max_value=10, value=3)
        default_retry_attempts = st.number_input("Default Retry Attempts", min_value=0, max_value=10, value=3)
        execution_timeout = st.number_input("Execution Timeout (minutes)", min_value=1, max_value=120, value=30)
        
        if st.button("Save Settings"):
            # Save settings (would be persisted in a real implementation)
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    # Example usage and testing
    from database import DatabaseManager
    
    # Initialize components
    db_manager = DatabaseManager()
    etl_engine = ETLPipelineEngine(db_manager)
    
    # Create sample data source
    api_source = DataSource(
        source_id="llm_api_source",
        source_type="api",
        name="LLM API Data Source",
        connection_config={
            "endpoint": "https://api.example.com/llm-metrics",
            "api_key": "demo_key"
        },
        schedule_config={"frequency": "hourly"},
        transformation_rules=[
            {
                "transformer": "standard",
                "operations": [
                    {"type": "rename_columns", "mapping": {"risk": "Risk_Rate"}},
                    {"type": "filter_rows", "condition": "Risk_Rate > 0"}
                ]
            }
        ],
        data_quality_rules=[
            {"type": "not_null", "columns": ["Date", "Model", "Risk_Rate"]},
            {"type": "range", "column": "Risk_Rate", "min": 0.0, "max": 1.0}
        ]
    )
    
    # Create sample pipeline job
    pipeline_job = PipelineJob(
        job_id="daily_risk_ingestion",
        job_name="Daily Risk Data Ingestion",
        source_ids=["llm_api_source"],
        transformations=[
            {
                "transformer": "standard",
                "operations": [
                    {"type": "handle_missing_values", "strategy": "fill_mean"},
                    {"type": "remove_duplicates", "subset": ["Date", "Model", "Language", "Risk_Category"]}
                ]
            }
        ],
        destination_config={"type": "database", "table": "risk_data"},
        schedule="0 2 * * *",  # Daily at 2 AM
        dependencies=[],
        retry_config={"max_retries": 3, "retry_delay": 300},
        notification_config={
            "enabled": True,
            "email": {"recipients": ["admin@example.com"]},
            "webhook": {"url": "https://hooks.example.com/pipeline-status"}
        }
    )
    
    # Register components
    etl_engine.register_data_source(api_source)
    etl_engine.register_pipeline_job(pipeline_job)
    
    # Test job execution
    print("Testing job execution...")
    execution = etl_engine.execute_job("daily_risk_ingestion")
    print(f"Job execution completed with status: {execution.status}")
    print(f"Records processed: {execution.records_processed}")
    
    # Test scheduler
    print("Starting scheduler...")
    etl_engine.start_scheduler()
    
    # Let it run for a short time
    time.sleep(5)
    
    print("Stopping scheduler...")
    etl_engine.stop_scheduler()
    
    print("ETL pipeline module test completed")