"""
Advanced Data Governance and Lineage Tracking Module
Implements comprehensive data governance, lineage tracking, and compliance management
"""

import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
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
import networkx as nx
from collections import defaultdict, deque
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAssetType(Enum):
    """Types of data assets"""
    RAW_DATA = "raw_data"
    PROCESSED_DATA = "processed_data"
    MODEL = "model"
    FEATURE = "feature"
    DATASET = "dataset"
    PIPELINE = "pipeline"
    REPORT = "report"
    API_ENDPOINT = "api_endpoint"

class DataSensitivityLevel(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class DataQualityStatus(Enum):
    """Data quality status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"

class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"

class LineageEventType(Enum):
    """Types of lineage events"""
    CREATED = "created"
    UPDATED = "updated"
    TRANSFORMED = "transformed"
    MERGED = "merged"
    SPLIT = "split"
    DELETED = "deleted"
    ACCESSED = "accessed"
    EXPORTED = "exported"

@dataclass
class DataAsset:
    """Data asset with governance metadata"""
    asset_id: str
    name: str
    asset_type: DataAssetType
    description: str
    owner: str
    steward: str
    created_date: datetime
    last_modified: datetime
    
    # Data properties
    schema_definition: Dict[str, Any]
    data_location: str
    size_bytes: int
    row_count: int
    
    # Governance properties
    sensitivity_level: DataSensitivityLevel
    retention_period_days: int
    data_classification: List[str]
    business_terms: List[str]
    
    # Quality properties
    quality_status: DataQualityStatus
    quality_score: float
    quality_rules: List[str]
    
    # Compliance properties
    compliance_status: ComplianceStatus
    compliance_frameworks: List[str]
    privacy_flags: List[str]
    
    # Lineage properties
    parent_assets: List[str]
    child_assets: List[str]
    transformation_logic: str
    
    # Usage properties
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    active_users: List[str] = None
    
    def __post_init__(self):
        if self.active_users is None:
            self.active_users = []

@dataclass
class LineageEvent:
    """Data lineage event"""
    event_id: str
    timestamp: datetime
    event_type: LineageEventType
    source_asset_id: str
    target_asset_id: Optional[str]
    user_id: str
    operation: str
    metadata: Dict[str, Any]
    transformation_code: Optional[str] = None
    impact_score: float = 0.0

@dataclass
class DataQualityRule:
    """Data quality rule definition"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # completeness, accuracy, consistency, validity, etc.
    target_assets: List[str]
    rule_definition: Dict[str, Any]
    threshold: float
    severity: str  # critical, high, medium, low
    active: bool = True
    created_by: str = "system"
    created_date: datetime = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()

@dataclass 
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    framework: str  # GDPR, HIPAA, SOX, etc.
    description: str
    rule_definition: Dict[str, Any]
    applicable_asset_types: List[DataAssetType]
    severity: str
    automated_check: bool = False
    check_frequency_hours: int = 24
    last_checked: Optional[datetime] = None

class DataLineageGraph:
    """Data lineage graph manager"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.asset_metadata: Dict[str, DataAsset] = {}
        
    def add_asset(self, asset: DataAsset):
        """Add data asset to lineage graph"""
        self.graph.add_node(asset.asset_id, **asdict(asset))
        self.asset_metadata[asset.asset_id] = asset
        
        # Add parent relationships
        for parent_id in asset.parent_assets:
            if parent_id in self.graph:
                self.graph.add_edge(parent_id, asset.asset_id)
    
    def add_lineage_relationship(self, source_id: str, target_id: str, 
                               relationship_type: str = "derives_from",
                               metadata: Dict[str, Any] = None):
        """Add lineage relationship between assets"""
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(
                source_id, target_id,
                relationship_type=relationship_type,
                metadata=metadata or {},
                created_at=datetime.now()
            )
            
            # Update asset parent/child relationships
            if target_id in self.asset_metadata:
                if source_id not in self.asset_metadata[target_id].parent_assets:
                    self.asset_metadata[target_id].parent_assets.append(source_id)
            
            if source_id in self.asset_metadata:
                if target_id not in self.asset_metadata[source_id].child_assets:
                    self.asset_metadata[source_id].child_assets.append(target_id)
    
    def get_upstream_lineage(self, asset_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get upstream lineage for an asset"""
        if asset_id not in self.graph:
            return {}
        
        upstream = {}
        visited = set()
        queue = deque([(asset_id, 0)])
        
        while queue and len(visited) < 1000:  # Prevent infinite loops
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            
            # Get predecessors (upstream dependencies)
            predecessors = list(self.graph.predecessors(current_id))
            
            if predecessors:
                upstream[current_id] = {
                    'depth': depth,
                    'parents': predecessors,
                    'asset_info': self.asset_metadata.get(current_id, {})
                }
                
                # Add parents to queue
                for parent_id in predecessors:
                    if parent_id not in visited:
                        queue.append((parent_id, depth + 1))
        
        return upstream
    
    def get_downstream_lineage(self, asset_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get downstream lineage for an asset"""
        if asset_id not in self.graph:
            return {}
        
        downstream = {}
        visited = set()
        queue = deque([(asset_id, 0)])
        
        while queue and len(visited) < 1000:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            
            # Get successors (downstream dependencies)  
            successors = list(self.graph.successors(current_id))
            
            if successors:
                downstream[current_id] = {
                    'depth': depth,
                    'children': successors,
                    'asset_info': self.asset_metadata.get(current_id, {})
                }
                
                # Add children to queue
                for child_id in successors:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))
        
        return downstream
    
    def get_impact_analysis(self, asset_id: str, change_type: str = "modification") -> Dict[str, Any]:
        """Analyze impact of changes to an asset"""
        downstream = self.get_downstream_lineage(asset_id)
        
        impact_levels = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        impact_summary = {
            "total_affected_assets": 0,
            "affected_by_type": defaultdict(int),
            "affected_by_sensitivity": defaultdict(int),
            "critical_paths": []
        }
        
        for affected_id, lineage_info in downstream.items():
            if affected_id == asset_id:
                continue
                
            asset_info = lineage_info.get('asset_info', {})
            if not asset_info:
                continue
            
            # Determine impact level based on asset properties
            depth = lineage_info['depth']
            sensitivity = getattr(asset_info, 'sensitivity_level', DataSensitivityLevel.INTERNAL)
            asset_type = getattr(asset_info, 'asset_type', DataAssetType.RAW_DATA)
            
            # Calculate impact score
            impact_score = 1.0 / (depth + 1)  # Closer assets have higher impact
            
            if sensitivity in [DataSensitivityLevel.RESTRICTED, DataSensitivityLevel.TOP_SECRET]:
                impact_score *= 2.0
            elif sensitivity == DataSensitivityLevel.CONFIDENTIAL:
                impact_score *= 1.5
            
            if asset_type in [DataAssetType.MODEL, DataAssetType.REPORT]:
                impact_score *= 1.3
            
            # Categorize impact
            if impact_score > 1.5:
                impact_levels["critical"].append(affected_id)
            elif impact_score > 1.0:
                impact_levels["high"].append(affected_id)
            elif impact_score > 0.5:
                impact_levels["medium"].append(affected_id)
            else:
                impact_levels["low"].append(affected_id)
            
            # Update summary
            impact_summary["total_affected_assets"] += 1
            impact_summary["affected_by_type"][asset_type.value] += 1
            impact_summary["affected_by_sensitivity"][sensitivity.value] += 1
        
        return {
            "source_asset": asset_id,
            "change_type": change_type,
            "impact_levels": impact_levels,
            "summary": impact_summary,
            "timestamp": datetime.now()
        }
    
    def find_data_flow_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """Find all paths between two assets"""
        try:
            if source_id not in self.graph or target_id not in self.graph:
                return []
            
            # Find all simple paths (no cycles)
            paths = list(nx.all_simple_paths(
                self.graph, 
                source_id, 
                target_id, 
                cutoff=10  # Maximum path length
            ))
            
            return paths
            
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return []

class DataQualityEngine:
    """Data quality assessment and monitoring engine"""
    
    def __init__(self):
        self.quality_rules: Dict[str, DataQualityRule] = {}
        self.quality_results: Dict[str, Dict[str, Any]] = {}
        
    def add_quality_rule(self, rule: DataQualityRule):
        """Add a data quality rule"""
        self.quality_rules[rule.rule_id] = rule
        
    def assess_data_quality(self, asset: DataAsset, data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Assess data quality for an asset"""
        quality_results = {
            "asset_id": asset.asset_id,
            "assessment_timestamp": datetime.now(),
            "overall_score": 0.0,
            "rule_results": {},
            "recommendations": []
        }
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.quality_rules.values()
            if asset.asset_id in rule.target_assets or not rule.target_assets
        ]
        
        if not applicable_rules:
            quality_results["overall_score"] = asset.quality_score
            return quality_results
        
        total_score = 0.0
        rule_count = 0
        
        for rule in applicable_rules:
            try:
                rule_result = self._evaluate_quality_rule(rule, asset, data_sample)
                quality_results["rule_results"][rule.rule_id] = rule_result
                
                if rule_result["score"] is not None:
                    total_score += rule_result["score"]
                    rule_count += 1
                    
                    # Add recommendations for failed rules
                    if rule_result["score"] < rule.threshold:
                        quality_results["recommendations"].append({
                            "rule": rule.name,
                            "issue": rule_result.get("issue", "Quality threshold not met"),
                            "recommendation": self._generate_quality_recommendation(rule, rule_result)
                        })
                        
            except Exception as e:
                logger.error(f"Error evaluating quality rule {rule.rule_id}: {e}")
                quality_results["rule_results"][rule.rule_id] = {
                    "score": None,
                    "error": str(e)
                }
        
        # Calculate overall score
        if rule_count > 0:
            quality_results["overall_score"] = total_score / rule_count
        else:
            quality_results["overall_score"] = asset.quality_score
        
        # Update asset quality status
        if quality_results["overall_score"] >= 0.9:
            asset.quality_status = DataQualityStatus.EXCELLENT
        elif quality_results["overall_score"] >= 0.8:
            asset.quality_status = DataQualityStatus.GOOD
        elif quality_results["overall_score"] >= 0.6:
            asset.quality_status = DataQualityStatus.FAIR
        else:
            asset.quality_status = DataQualityStatus.POOR
        
        asset.quality_score = quality_results["overall_score"]
        
        # Store results
        self.quality_results[asset.asset_id] = quality_results
        
        return quality_results
    
    def _evaluate_quality_rule(self, rule: DataQualityRule, asset: DataAsset, 
                             data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Evaluate a single quality rule"""
        result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,  
            "score": None,
            "details": {},
            "timestamp": datetime.now()
        }
        
        try:
            if rule.rule_type == "completeness":
                result = self._check_completeness(rule, asset, data_sample)
            elif rule.rule_type == "accuracy":
                result = self._check_accuracy(rule, asset, data_sample)
            elif rule.rule_type == "consistency":
                result = self._check_consistency(rule, asset, data_sample)
            elif rule.rule_type == "validity":
                result = self._check_validity(rule, asset, data_sample)
            elif rule.rule_type == "uniqueness":
                result = self._check_uniqueness(rule, asset, data_sample)
            elif rule.rule_type == "timeliness":
                result = self._check_timeliness(rule, asset, data_sample)
            else:
                result["score"] = 0.5  # Default neutral score
                result["details"]["message"] = f"Unknown rule type: {rule.rule_type}"
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error evaluating quality rule {rule.rule_id}: {e}")
        
        return result
    
    def _check_completeness(self, rule: DataQualityRule, asset: DataAsset, 
                          data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Check data completeness"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "score": 0.5, "details": {}}
        
        if data_sample is not None and not data_sample.empty:
            # Calculate null percentage
            null_percentage = data_sample.isnull().sum().sum() / (len(data_sample) * len(data_sample.columns))
            completeness_score = 1.0 - null_percentage
            
            result["score"] = float(completeness_score)
            result["details"] = {
                "total_cells": len(data_sample) * len(data_sample.columns),
                "null_cells": int(data_sample.isnull().sum().sum()),
                "null_percentage": float(null_percentage),
                "completeness_score": float(completeness_score)
            }
            
            if completeness_score < rule.threshold:
                result["issue"] = f"Completeness score {completeness_score:.2f} below threshold {rule.threshold}"
        else:
            # Use metadata-based assessment
            if asset.row_count > 0:
                result["score"] = 0.8  # Assume good completeness if we have data
            else:
                result["score"] = 0.0
                result["issue"] = "No data available"
        
        return result
    
    def _check_accuracy(self, rule: DataQualityRule, asset: DataAsset, 
                       data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Check data accuracy"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "score": 0.5, "details": {}}
        
        # Accuracy is difficult to assess without reference data
        # Use heuristic based on data patterns
        if data_sample is not None and not data_sample.empty:
            accuracy_indicators = []
            
            # Check for obvious data issues
            for column in data_sample.select_dtypes(include=[np.number]).columns:
                # Check for outliers
                Q1 = data_sample[column].quantile(0.25)
                Q3 = data_sample[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = len(data_sample[
                    (data_sample[column] < (Q1 - 1.5 * IQR)) | 
                    (data_sample[column] > (Q3 + 1.5 * IQR))
                ])
                
                outlier_percentage = outlier_count / len(data_sample) if len(data_sample) > 0 else 0
                accuracy_indicators.append(1.0 - min(outlier_percentage * 2, 1.0))
            
            if accuracy_indicators:
                result["score"] = float(np.mean(accuracy_indicators))
                result["details"]["outlier_analysis"] = {
                    "columns_analyzed": len(accuracy_indicators),
                    "average_accuracy": float(np.mean(accuracy_indicators))
                }
            else:
                result["score"] = 0.7  # Default for non-numeric data
        
        return result
    
    def _check_consistency(self, rule: DataQualityRule, asset: DataAsset, 
                         data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Check data consistency"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "score": 0.5, "details": {}}
        
        if data_sample is not None and not data_sample.empty:
            consistency_scores = []
            
            # Check format consistency for string columns
            for column in data_sample.select_dtypes(include=['object']).columns:
                if len(data_sample[column].dropna()) > 0:
                    # Check length consistency
                    lengths = data_sample[column].dropna().str.len()
                    length_std = lengths.std()
                    length_mean = lengths.mean()
                    
                    if length_mean > 0:
                        cv = length_std / length_mean  # Coefficient of variation
                        consistency_score = max(0, 1.0 - cv)
                        consistency_scores.append(consistency_score)
            
            if consistency_scores:
                result["score"] = float(np.mean(consistency_scores))
                result["details"]["format_consistency"] = {
                    "columns_analyzed": len(consistency_scores),
                    "average_consistency": float(np.mean(consistency_scores))
                }
            else:
                result["score"] = 0.7
        
        return result
    
    def _check_validity(self, rule: DataQualityRule, asset: DataAsset, 
                       data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Check data validity"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "score": 0.5, "details": {}}
        
        if data_sample is not None and not data_sample.empty:
            validity_scores = []
            
            # Check data type validity
            for column in data_sample.columns:
                try:
                    # Try to maintain data type consistency
                    if data_sample[column].dtype == 'object':
                        # For string columns, check if they contain expected patterns
                        non_null_values = data_sample[column].dropna()
                        if len(non_null_values) > 0:
                            # Simple validity check - no completely empty strings
                            valid_count = len(non_null_values[non_null_values.str.strip() != ''])
                            validity_score = valid_count / len(non_null_values) if len(non_null_values) > 0 else 0
                            validity_scores.append(validity_score)
                    else:
                        # For numeric columns, check for inf/nan
                        valid_count = len(data_sample[column].dropna())
                        total_count = len(data_sample[column])
                        validity_score = valid_count / total_count if total_count > 0 else 0
                        validity_scores.append(validity_score)
                        
                except Exception:
                    validity_scores.append(0.5)  # Default for problematic columns
            
            if validity_scores:
                result["score"] = float(np.mean(validity_scores))
                result["details"]["validity_analysis"] = {
                    "columns_analyzed": len(validity_scores),
                    "average_validity": float(np.mean(validity_scores))
                }
        
        return result
    
    def _check_uniqueness(self, rule: DataQualityRule, asset: DataAsset, 
                         data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Check data uniqueness"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "score": 0.5, "details": {}}
        
        if data_sample is not None and not data_sample.empty:
            duplicate_count = len(data_sample) - len(data_sample.drop_duplicates())
            uniqueness_score = 1.0 - (duplicate_count / len(data_sample))
            
            result["score"] = float(uniqueness_score)
            result["details"] = {
                "total_rows": len(data_sample),
                "duplicate_rows": duplicate_count,
                "uniqueness_score": float(uniqueness_score)
            }
            
            if uniqueness_score < rule.threshold:
                result["issue"] = f"Uniqueness score {uniqueness_score:.2f} below threshold {rule.threshold}"
        
        return result
    
    def _check_timeliness(self, rule: DataQualityRule, asset: DataAsset, 
                         data_sample: pd.DataFrame = None) -> Dict[str, Any]:
        """Check data timeliness"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "score": 0.5, "details": {}}
        
        # Check how recent the data is
        now = datetime.now()
        
        if asset.last_modified:
            hours_since_update = (now - asset.last_modified).total_seconds() / 3600
            
            # Define timeliness based on expected update frequency
            expected_frequency_hours = rule.rule_definition.get("expected_frequency_hours", 24)
            
            if hours_since_update <= expected_frequency_hours:
                timeliness_score = 1.0
            elif hours_since_update <= expected_frequency_hours * 2:
                timeliness_score = 0.7
            elif hours_since_update <= expected_frequency_hours * 7:
                timeliness_score = 0.4
            else:
                timeliness_score = 0.1
            
            result["score"] = timeliness_score
            result["details"] = {
                "last_modified": asset.last_modified.isoformat(),
                "hours_since_update": hours_since_update,
                "expected_frequency_hours": expected_frequency_hours,
                "timeliness_score": timeliness_score
            }
            
            if timeliness_score < rule.threshold:
                result["issue"] = f"Data is {hours_since_update:.1f} hours old, exceeding expected frequency"
        
        return result
    
    def _generate_quality_recommendation(self, rule: DataQualityRule, 
                                       rule_result: Dict[str, Any]) -> str:
        """Generate recommendation based on quality rule failure"""
        recommendations = {
            "completeness": "Review data collection processes and implement null value handling",
            "accuracy": "Validate data sources and implement accuracy checks",
            "consistency": "Standardize data formats and implement validation rules",
            "validity": "Review data validation rules and fix invalid entries",
            "uniqueness": "Implement deduplication processes and unique constraints",
            "timeliness": "Review data refresh schedules and implement timely updates"
        }
        
        return recommendations.get(rule.rule_type, "Review data quality processes")

class ComplianceEngine:
    """Compliance assessment and monitoring engine"""
    
    def __init__(self):
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.compliance_results: Dict[str, Dict[str, Any]] = {}
        
    def add_compliance_rule(self, rule: ComplianceRule):
        """Add a compliance rule"""
        self.compliance_rules[rule.rule_id] = rule
        
    def assess_compliance(self, asset: DataAsset) -> Dict[str, Any]:
        """Assess compliance for an asset"""
        compliance_results = {
            "asset_id": asset.asset_id,
            "assessment_timestamp": datetime.now(),
            "overall_status": ComplianceStatus.NOT_ASSESSED,
            "framework_results": {},
            "violations": [],
            "recommendations": []
        }
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.compliance_rules.values()
            if asset.asset_type in rule.applicable_asset_types or not rule.applicable_asset_types
        ]
        
        if not applicable_rules:
            compliance_results["overall_status"] = asset.compliance_status
            return compliance_results
        
        framework_statuses = defaultdict(list)
        
        for rule in applicable_rules:
            try:
                rule_result = self._evaluate_compliance_rule(rule, asset)
                framework_statuses[rule.framework].append(rule_result["compliant"])
                
                compliance_results["framework_results"][rule.rule_id] = rule_result
                
                if not rule_result["compliant"]:
                    compliance_results["violations"].append({
                        "rule": rule.name,
                        "framework": rule.framework,
                        "severity": rule.severity,
                        "details": rule_result.get("details", ""),
                        "recommendation": self._generate_compliance_recommendation(rule, rule_result)
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating compliance rule {rule.rule_id}: {e}")
                compliance_results["framework_results"][rule.rule_id] = {
                    "compliant": False,
                    "error": str(e)
                }
                framework_statuses[rule.framework].append(False)
        
        # Determine overall compliance status
        all_compliant = all(
            all(statuses) for statuses in framework_statuses.values()
        )
        any_compliant = any(
            any(statuses) for statuses in framework_statuses.values()
        )
        
        if all_compliant:
            compliance_results["overall_status"] = ComplianceStatus.COMPLIANT
        elif any_compliant:
            compliance_results["overall_status"] = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            compliance_results["overall_status"] = ComplianceStatus.NON_COMPLIANT
        
        # Update asset compliance status
        asset.compliance_status = compliance_results["overall_status"]
        
        # Store results
        self.compliance_results[asset.asset_id] = compliance_results
        
        return compliance_results
    
    def _evaluate_compliance_rule(self, rule: ComplianceRule, asset: DataAsset) -> Dict[str, Any]:
        """Evaluate a single compliance rule"""
        result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "framework": rule.framework,
            "compliant": False,
            "details": {},
            "timestamp": datetime.now()
        }
        
        try:
            if rule.framework == "GDPR":
                result = self._check_gdpr_compliance(rule, asset)
            elif rule.framework == "HIPAA":
                result = self._check_hipaa_compliance(rule, asset)
            elif rule.framework == "SOX":
                result = self._check_sox_compliance(rule, asset)
            elif rule.framework == "PCI_DSS":
                result = self._check_pci_compliance(rule, asset)
            else:
                result = self._check_generic_compliance(rule, asset)
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error evaluating compliance rule {rule.rule_id}: {e}")
        
        return result
    
    def _check_gdpr_compliance(self, rule: ComplianceRule, asset: DataAsset) -> Dict[str, Any]:
        """Check GDPR compliance"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "framework": "GDPR", "compliant": True, "details": {}}
        
        checks = []
        
        # Check data retention
        if asset.retention_period_days > 0:
            max_retention = rule.rule_definition.get("max_retention_days", 2555)  # ~7 years default
            if asset.retention_period_days <= max_retention:
                checks.append(("retention_policy", True, f"Retention period {asset.retention_period_days} days is compliant"))
            else:
                checks.append(("retention_policy", False, f"Retention period {asset.retention_period_days} days exceeds maximum {max_retention} days"))
        else:
            checks.append(("retention_policy", False, "No retention policy defined"))
        
        # Check for personal data flags
        personal_data_indicators = ["pii", "personal", "gdpr", "sensitive"]
        has_personal_data = any(
            indicator in " ".join(asset.privacy_flags + asset.data_classification + [asset.description]).lower()
            for indicator in personal_data_indicators
        )
        
        if has_personal_data:
            # Check for appropriate sensitivity level
            if asset.sensitivity_level in [DataSensitivityLevel.CONFIDENTIAL, DataSensitivityLevel.RESTRICTED, DataSensitivityLevel.TOP_SECRET]:
                checks.append(("sensitivity_classification", True, "Personal data has appropriate sensitivity classification"))
            else:
                checks.append(("sensitivity_classification", False, "Personal data lacks appropriate sensitivity classification"))
            
            # Check for data steward
            if asset.steward and asset.steward != "unknown":
                checks.append(("data_steward", True, f"Data steward assigned: {asset.steward}"))
            else:
                checks.append(("data_steward", False, "No data steward assigned for personal data"))
        else:
            checks.append(("personal_data_check", True, "No personal data indicators found"))
        
        # Overall compliance
        result["compliant"] = all(check[1] for check in checks)
        result["details"] = {
            "checks_performed": len(checks),
            "checks_passed": sum(1 for check in checks if check[1]),
            "check_results": [{"check": check[0], "passed": check[1], "message": check[2]} for check in checks]
        }
        
        return result
    
    def _check_hipaa_compliance(self, rule: ComplianceRule, asset: DataAsset) -> Dict[str, Any]:
        """Check HIPAA compliance"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "framework": "HIPAA", "compliant": True, "details": {}}
        
        checks = []
        
        # Check for healthcare data indicators
        healthcare_indicators = ["phi", "health", "medical", "hipaa", "patient"]
        has_healthcare_data = any(
            indicator in " ".join(asset.privacy_flags + asset.data_classification + [asset.description]).lower()
            for indicator in healthcare_indicators
        )
        
        if has_healthcare_data:
            # Check minimum security level
            if asset.sensitivity_level in [DataSensitivityLevel.RESTRICTED, DataSensitivityLevel.TOP_SECRET]:
                checks.append(("security_level", True, "Healthcare data has appropriate security level"))
            else:
                checks.append(("security_level", False, "Healthcare data requires higher security level"))
            
            # Check access controls
            if len(asset.active_users) <= rule.rule_definition.get("max_users", 10):
                checks.append(("access_control", True, f"Limited access: {len(asset.active_users)} users"))
            else:
                checks.append(("access_control", False, f"Too many users with access: {len(asset.active_users)}"))
            
            # Check audit trail
            if "audit" in asset.privacy_flags or "logging" in asset.privacy_flags:
                checks.append(("audit_trail", True, "Audit trail enabled"))
            else:
                checks.append(("audit_trail", False, "Audit trail not enabled for healthcare data"))
        else:
            checks.append(("healthcare_data_check", True, "No healthcare data indicators found"))
        
        result["compliant"] = all(check[1] for check in checks)
        result["details"] = {
            "checks_performed": len(checks),
            "checks_passed": sum(1 for check in checks if check[1]),
            "check_results": [{"check": check[0], "passed": check[1], "message": check[2]} for check in checks]
        }
        
        return result
    
    def _check_sox_compliance(self, rule: ComplianceRule, asset: DataAsset) -> Dict[str, Any]:
        """Check SOX compliance"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "framework": "SOX", "compliant": True, "details": {}}
        
        checks = []
        
        # Check for financial data indicators
        financial_indicators = ["financial", "sox", "revenue", "accounting", "audit"]
        has_financial_data = any(
            indicator in " ".join(asset.data_classification + [asset.description]).lower()
            for indicator in financial_indicators
        )
        
        if has_financial_data:
            # Check data retention (SOX requires 7 years for certain documents)
            min_retention = rule.rule_definition.get("min_retention_days", 2555)  # 7 years
            if asset.retention_period_days >= min_retention:
                checks.append(("retention_requirement", True, f"Retention period {asset.retention_period_days} days meets SOX requirements"))
            else:
                checks.append(("retention_requirement", False, f"Retention period {asset.retention_period_days} days below SOX requirement of {min_retention} days"))
            
            # Check for proper ownership and stewardship
            if asset.owner and asset.steward:
                checks.append(("ownership", True, f"Owner: {asset.owner}, Steward: {asset.steward}"))
            else:
                checks.append(("ownership", False, "Financial data must have clearly defined owner and steward"))
            
            # Check for change tracking
            if asset.last_modified and (datetime.now() - asset.last_modified).days < 30:
                checks.append(("change_tracking", True, "Recent modifications tracked"))
            else:
                checks.append(("change_tracking", False, "Change tracking may be insufficient"))
        else:
            checks.append(("financial_data_check", True, "No financial data indicators found"))
        
        result["compliant"] = all(check[1] for check in checks)
        result["details"] = {
            "checks_performed": len(checks),
            "checks_passed": sum(1 for check in checks if check[1]),
            "check_results": [{"check": check[0], "passed": check[1], "message": check[2]} for check in checks]
        }
        
        return result
    
    def _check_pci_compliance(self, rule: ComplianceRule, asset: DataAsset) -> Dict[str, Any]:
        """Check PCI DSS compliance"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "framework": "PCI_DSS", "compliant": True, "details": {}}
        
        checks = []
        
        # Check for payment card data indicators
        pci_indicators = ["pci", "card", "payment", "credit", "cardholder"]
        has_pci_data = any(
            indicator in " ".join(asset.privacy_flags + asset.data_classification + [asset.description]).lower()
            for indicator in pci_indicators
        )
        
        if has_pci_data:
            # Check security level
            if asset.sensitivity_level in [DataSensitivityLevel.RESTRICTED, DataSensitivityLevel.TOP_SECRET]:
                checks.append(("security_level", True, "Payment data has appropriate security level"))
            else:
                checks.append(("security_level", False, "Payment data requires highest security level"))
            
            # Check access restrictions
            max_users = rule.rule_definition.get("max_users", 5)
            if len(asset.active_users) <= max_users:
                checks.append(("access_restriction", True, f"Access limited to {len(asset.active_users)} users"))
            else:
                checks.append(("access_restriction", False, f"Too many users with access to payment data: {len(asset.active_users)}"))
            
            # Check for encryption requirements
            if "encrypted" in asset.privacy_flags or "encryption" in asset.privacy_flags:
                checks.append(("encryption", True, "Payment data encryption indicated"))
            else:
                checks.append(("encryption", False, "Payment data encryption not indicated"))
        else:
            checks.append(("pci_data_check", True, "No payment card data indicators found"))
        
        result["compliant"] = all(check[1] for check in checks)
        result["details"] = {
            "checks_performed": len(checks),
            "checks_passed": sum(1 for check in checks if check[1]),
            "check_results": [{"check": check[0], "passed": check[1], "message": check[2]} for check in checks]
        }
        
        return result
    
    def _check_generic_compliance(self, rule: ComplianceRule, asset: DataAsset) -> Dict[str, Any]:
        """Check generic compliance rule"""
        result = {"rule_id": rule.rule_id, "rule_name": rule.name, "framework": rule.framework, "compliant": True, "details": {}}
        
        # Basic compliance checks
        checks = []
        
        # Check if asset has required metadata
        required_fields = rule.rule_definition.get("required_fields", [])
        for field in required_fields:
            field_value = getattr(asset, field, None)
            if field_value:
                checks.append((f"required_field_{field}", True, f"Field {field} is populated"))
            else:
                checks.append((f"required_field_{field}", False, f"Required field {field} is missing"))
        
        # Check sensitivity level requirements
        min_sensitivity = rule.rule_definition.get("min_sensitivity_level")
        if min_sensitivity:
            sensitivity_levels = {
                DataSensitivityLevel.PUBLIC: 1,
                DataSensitivityLevel.INTERNAL: 2,
                DataSensitivityLevel.CONFIDENTIAL: 3,
                DataSensitivityLevel.RESTRICTED: 4,
                DataSensitivityLevel.TOP_SECRET: 5
            }
            
            current_level = sensitivity_levels.get(asset.sensitivity_level, 1)
            required_level = sensitivity_levels.get(DataSensitivityLevel(min_sensitivity), 1)
            
            if current_level >= required_level:
                checks.append(("sensitivity_level", True, f"Sensitivity level {asset.sensitivity_level.value} meets requirements"))
            else:
                checks.append(("sensitivity_level", False, f"Sensitivity level {asset.sensitivity_level.value} below required {min_sensitivity}"))
        
        result["compliant"] = all(check[1] for check in checks) if checks else True
        result["details"] = {
            "checks_performed": len(checks),
            "checks_passed": sum(1 for check in checks if check[1]),
            "check_results": [{"check": check[0], "passed": check[1], "message": check[2]} for check in checks]
        }
        
        return result
    
    def _generate_compliance_recommendation(self, rule: ComplianceRule, 
                                         rule_result: Dict[str, Any]) -> str:
        """Generate recommendation based on compliance rule failure"""
        framework = rule.framework.lower()
        
        recommendations = {
            "gdpr": "Review GDPR requirements for personal data handling, retention policies, and consent management",
            "hipaa": "Implement HIPAA safeguards for healthcare data including access controls and audit trails",
            "sox": "Ensure SOX compliance with proper financial data retention, ownership, and change controls",
            "pci_dss": "Implement PCI DSS requirements for payment card data security and access restrictions"
        }
        
        return recommendations.get(framework, f"Review {rule.framework} compliance requirements and implement necessary controls")

class DataGovernanceManager:
    """Main data governance management system"""
    
    def __init__(self, db_path: str = "data_governance.db"):
        self.db_path = Path(db_path)
        self.lineage_graph = DataLineageGraph()
        self.quality_engine = DataQualityEngine()
        self.compliance_engine = ComplianceEngine()
        
        # Initialize database
        self.init_database()
        
        # Load existing data
        self.load_assets()
        self.load_quality_rules()
        self.load_compliance_rules()
        
    def init_database(self):
        """Initialize SQLite database for governance data"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Data assets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_assets (
                asset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                description TEXT,
                owner TEXT,
                steward TEXT,
                created_date TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                schema_definition TEXT,
                data_location TEXT,
                size_bytes INTEGER,
                row_count INTEGER,
                sensitivity_level TEXT,
                retention_period_days INTEGER,
                data_classification TEXT,
                business_terms TEXT,
                quality_status TEXT,
                quality_score REAL,
                quality_rules TEXT,
                compliance_status TEXT,
                compliance_frameworks TEXT,
                privacy_flags TEXT,
                parent_assets TEXT,
                child_assets TEXT,
                transformation_logic TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                active_users TEXT
            )
        ''')
        
        # Lineage events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lineage_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source_asset_id TEXT NOT NULL,
                target_asset_id TEXT,
                user_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                metadata TEXT,
                transformation_code TEXT,
                impact_score REAL DEFAULT 0.0
            )
        ''')
        
        # Quality rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                rule_type TEXT NOT NULL,
                target_assets TEXT,
                rule_definition TEXT,
                threshold REAL,
                severity TEXT,
                active BOOLEAN DEFAULT 1,
                created_by TEXT,
                created_date TEXT
            )
        ''')
        
        # Compliance rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                framework TEXT NOT NULL,
                description TEXT,
                rule_definition TEXT,
                applicable_asset_types TEXT,
                severity TEXT,
                automated_check BOOLEAN DEFAULT 0,
                check_frequency_hours INTEGER DEFAULT 24,
                last_checked TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_data_asset(self, asset: DataAsset) -> bool:
        """Add a data asset to governance system"""
        try:
            # Add to lineage graph
            self.lineage_graph.add_asset(asset)
            
            # Save to database
            self._save_asset_to_db(asset)
            
            # Log lineage event
            self.log_lineage_event(
                LineageEventType.CREATED,
                asset.asset_id,
                None,
                asset.owner,
                "asset_created",
                {"asset_type": asset.asset_type.value, "size_bytes": asset.size_bytes}
            )
            
            logger.info(f"Data asset {asset.asset_id} added to governance system")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data asset {asset.asset_id}: {e}")
            return False
    
    def _save_asset_to_db(self, asset: DataAsset):
        """Save data asset to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_assets
            (asset_id, name, asset_type, description, owner, steward, created_date, last_modified,
             schema_definition, data_location, size_bytes, row_count, sensitivity_level, 
             retention_period_days, data_classification, business_terms, quality_status, 
             quality_score, quality_rules, compliance_status, compliance_frameworks, 
             privacy_flags, parent_assets, child_assets, transformation_logic, 
             access_count, last_accessed, active_users)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            asset.asset_id, asset.name, asset.asset_type.value, asset.description,
            asset.owner, asset.steward, asset.created_date.isoformat(), asset.last_modified.isoformat(),
            json.dumps(asset.schema_definition), asset.data_location, asset.size_bytes, asset.row_count,
            asset.sensitivity_level.value, asset.retention_period_days, json.dumps(asset.data_classification),
            json.dumps(asset.business_terms), asset.quality_status.value, asset.quality_score,
            json.dumps(asset.quality_rules), asset.compliance_status.value, json.dumps(asset.compliance_frameworks),
            json.dumps(asset.privacy_flags), json.dumps(asset.parent_assets), json.dumps(asset.child_assets),
            asset.transformation_logic, asset.access_count, 
            asset.last_accessed.isoformat() if asset.last_accessed else None,
            json.dumps(asset.active_users)
        ))
        
        conn.commit()
        conn.close()
    
    def load_assets(self):
        """Load data assets from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM data_assets')
            rows = cursor.fetchall()
            
            for row in rows:
                asset = self._row_to_asset(row)
                if asset:
                    self.lineage_graph.add_asset(asset)
            
            conn.close()
            logger.info(f"Loaded {len(rows)} data assets from database")
            
        except Exception as e:
            logger.error(f"Error loading assets: {e}")
    
    def _row_to_asset(self, row) -> Optional[DataAsset]:
        """Convert database row to DataAsset object"""
        try:
            return DataAsset(
                asset_id=row[0],
                name=row[1],
                asset_type=DataAssetType(row[2]),
                description=row[3] or "",
                owner=row[4] or "",
                steward=row[5] or "",
                created_date=datetime.fromisoformat(row[6]),
                last_modified=datetime.fromisoformat(row[7]),
                schema_definition=json.loads(row[8]) if row[8] else {},
                data_location=row[9] or "",
                size_bytes=row[10] or 0,
                row_count=row[11] or 0,
                sensitivity_level=DataSensitivityLevel(row[12]),
                retention_period_days=row[13] or 0,
                data_classification=json.loads(row[14]) if row[14] else [],
                business_terms=json.loads(row[15]) if row[15] else [],
                quality_status=DataQualityStatus(row[16]),
                quality_score=row[17] or 0.0,
                quality_rules=json.loads(row[18]) if row[18] else [],
                compliance_status=ComplianceStatus(row[19]),
                compliance_frameworks=json.loads(row[20]) if row[20] else [],
                privacy_flags=json.loads(row[21]) if row[21] else [],
                parent_assets=json.loads(row[22]) if row[22] else [],
                child_assets=json.loads(row[23]) if row[23] else [],
                transformation_logic=row[24] or "",
                access_count=row[25] or 0,
                last_accessed=datetime.fromisoformat(row[26]) if row[26] else None,
                active_users=json.loads(row[27]) if row[27] else []
            )
        except Exception as e:
            logger.error(f"Error converting row to asset: {e}")
            return None
    
    def load_quality_rules(self):
        """Load quality rules from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM quality_rules')
            rows = cursor.fetchall()
            
            for row in rows:
                rule = DataQualityRule(
                    rule_id=row[0],
                    name=row[1],
                    description=row[2] or "",
                    rule_type=row[3],
                    target_assets=json.loads(row[4]) if row[4] else [],
                    rule_definition=json.loads(row[5]) if row[5] else {},
                    threshold=row[6] or 0.8,
                    severity=row[7] or "medium",
                    active=bool(row[8]),
                    created_by=row[9] or "system",
                    created_date=datetime.fromisoformat(row[10]) if row[10] else datetime.now()
                )
                self.quality_engine.add_quality_rule(rule)
            
            conn.close()
            logger.info(f"Loaded {len(rows)} quality rules from database")
            
        except Exception as e:
            logger.error(f"Error loading quality rules: {e}")
    
    def load_compliance_rules(self):
        """Load compliance rules from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM compliance_rules')
            rows = cursor.fetchall()
            
            for row in rows:
                rule = ComplianceRule(
                    rule_id=row[0],
                    name=row[1],
                    framework=row[2],
                    description=row[3] or "",
                    rule_definition=json.loads(row[4]) if row[4] else {},
                    applicable_asset_types=[DataAssetType(t) for t in json.loads(row[5])] if row[5] else [],
                    severity=row[6] or "medium",
                    automated_check=bool(row[7]),
                    check_frequency_hours=row[8] or 24,
                    last_checked=datetime.fromisoformat(row[9]) if row[9] else None
                )
                self.compliance_engine.add_compliance_rule(rule)
            
            conn.close()
            logger.info(f"Loaded {len(rows)} compliance rules from database")
            
        except Exception as e:
            logger.error(f"Error loading compliance rules: {e}")
    
    def log_lineage_event(self, event_type: LineageEventType, source_asset_id: str,
                         target_asset_id: Optional[str], user_id: str, operation: str,
                         metadata: Dict[str, Any] = None, transformation_code: str = None):
        """Log a lineage event"""
        try:
            event = LineageEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type=event_type,
                source_asset_id=source_asset_id,
                target_asset_id=target_asset_id,
                user_id=user_id,
                operation=operation,
                metadata=metadata or {},
                transformation_code=transformation_code
            )
            
            # Save to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO lineage_events
                (event_id, timestamp, event_type, source_asset_id, target_asset_id, 
                 user_id, operation, metadata, transformation_code, impact_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.timestamp.isoformat(), event.event_type.value,
                event.source_asset_id, event.target_asset_id, event.user_id,
                event.operation, json.dumps(event.metadata), event.transformation_code,
                event.impact_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging lineage event: {e}")
    
    def get_governance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for governance dashboard"""
        dashboard_data = {
            "asset_summary": {},
            "quality_summary": {},
            "compliance_summary": {},
            "lineage_summary": {},
            "recent_events": []
        }
        
        try:
            # Asset summary
            assets = list(self.lineage_graph.asset_metadata.values())
            dashboard_data["asset_summary"] = {
                "total_assets": len(assets),
                "by_type": {},
                "by_sensitivity": {},
                "by_owner": {}
            }
            
            for asset in assets:
                # By type
                asset_type = asset.asset_type.value
                dashboard_data["asset_summary"]["by_type"][asset_type] = \
                    dashboard_data["asset_summary"]["by_type"].get(asset_type, 0) + 1
                
                # By sensitivity
                sensitivity = asset.sensitivity_level.value
                dashboard_data["asset_summary"]["by_sensitivity"][sensitivity] = \
                    dashboard_data["asset_summary"]["by_sensitivity"].get(sensitivity, 0) + 1
                
                # By owner
                owner = asset.owner or "Unknown"
                dashboard_data["asset_summary"]["by_owner"][owner] = \
                    dashboard_data["asset_summary"]["by_owner"].get(owner, 0) + 1
            
            # Quality summary
            quality_statuses = [asset.quality_status.value for asset in assets]
            dashboard_data["quality_summary"] = {
                "average_score": np.mean([asset.quality_score for asset in assets]) if assets else 0,
                "by_status": {status: quality_statuses.count(status) for status in set(quality_statuses)},
                "total_rules": len(self.quality_engine.quality_rules)
            }
            
            # Compliance summary
            compliance_statuses = [asset.compliance_status.value for asset in assets]
            dashboard_data["compliance_summary"] = {
                "by_status": {status: compliance_statuses.count(status) for status in set(compliance_statuses)},
                "total_rules": len(self.compliance_engine.compliance_rules),
                "frameworks": list(set(rule.framework for rule in self.compliance_engine.compliance_rules.values()))
            }
            
            # Lineage summary
            dashboard_data["lineage_summary"] = {
                "total_nodes": len(self.lineage_graph.graph.nodes),
                "total_edges": len(self.lineage_graph.graph.edges),
                "orphaned_assets": len([n for n in self.lineage_graph.graph.nodes if self.lineage_graph.graph.degree(n) == 0])
            }
            
            # Recent events (last 10)
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute('''
                SELECT event_id, timestamp, event_type, source_asset_id, target_asset_id, 
                       user_id, operation, metadata
                FROM lineage_events 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            events = cursor.fetchall()
            conn.close()
            
            dashboard_data["recent_events"] = [
                {
                    "event_id": event[0],
                    "timestamp": event[1],
                    "event_type": event[2],
                    "source_asset_id": event[3],
                    "target_asset_id": event[4],
                    "user_id": event[5],
                    "operation": event[6],
                    "metadata": json.loads(event[7]) if event[7] else {}
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
        
        return dashboard_data

# Streamlit Integration Functions

def initialize_data_governance():
    """Initialize data governance system"""
    if 'governance_manager' not in st.session_state:
        st.session_state.governance_manager = DataGovernanceManager()
    
    return st.session_state.governance_manager

def render_data_governance_dashboard():
    """Render data governance dashboard"""
    st.header("🏛️ Data Governance & Lineage")
    
    governance_manager = initialize_data_governance()
    
    # Get dashboard data
    dashboard_data = governance_manager.get_governance_dashboard_data()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assets", dashboard_data["asset_summary"]["total_assets"])
    
    with col2:
        avg_quality = dashboard_data["quality_summary"]["average_score"]
        st.metric("Avg Quality Score", f"{avg_quality:.3f}")
    
    with col3:
        compliant_count = dashboard_data["compliance_summary"]["by_status"].get("compliant", 0)
        st.metric("Compliant Assets", compliant_count)
    
    with col4:
        total_lineage = dashboard_data["lineage_summary"]["total_edges"]
        st.metric("Lineage Connections", total_lineage)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🗂️ Asset Catalog", 
        "🔗 Lineage Viewer",
        "✅ Data Quality",
        "⚖️ Compliance",
        "⚙️ Administration"
    ])
    
    with tab1:
        st.subheader("Governance Overview")
        
        # Asset distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            if dashboard_data["asset_summary"]["by_type"]:
                type_data = list(dashboard_data["asset_summary"]["by_type"].items())
                type_df = pd.DataFrame(type_data, columns=['Asset Type', 'Count'])
                
                fig_types = px.pie(type_df, values='Count', names='Asset Type',
                                 title='Assets by Type')
                st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            if dashboard_data["asset_summary"]["by_sensitivity"]:
                sens_data = list(dashboard_data["asset_summary"]["by_sensitivity"].items())
                sens_df = pd.DataFrame(sens_data, columns=['Sensitivity Level', 'Count'])
                
                color_map = {
                    'public': '#2E8B57',
                    'internal': '#4682B4', 
                    'confidential': '#DAA520',
                    'restricted': '#CD853F',
                    'top_secret': '#DC143C'
                }
                
                fig_sens = px.bar(sens_df, x='Sensitivity Level', y='Count',
                                title='Assets by Sensitivity Level',
                                color='Sensitivity Level',
                                color_discrete_map=color_map)
                st.plotly_chart(fig_sens, use_container_width=True)
        
        # Quality and compliance status
        col1, col2 = st.columns(2)
        
        with col1:
            if dashboard_data["quality_summary"]["by_status"]:
                quality_data = list(dashboard_data["quality_summary"]["by_status"].items())
                quality_df = pd.DataFrame(quality_data, columns=['Quality Status', 'Count'])
                
                fig_quality = px.bar(quality_df, x='Quality Status', y='Count',
                                   title='Data Quality Status Distribution')
                st.plotly_chart(fig_quality, use_container_width=True)
        
        with col2:
            if dashboard_data["compliance_summary"]["by_status"]:
                comp_data = list(dashboard_data["compliance_summary"]["by_status"].items())
                comp_df = pd.DataFrame(comp_data, columns=['Compliance Status', 'Count'])
                
                fig_comp = px.bar(comp_df, x='Compliance Status', y='Count',
                                title='Compliance Status Distribution')
                st.plotly_chart(fig_comp, use_container_width=True)
        
        # Recent events
        st.subheader("Recent Lineage Events")
        
        if dashboard_data["recent_events"]:
            events_data = []
            for event in dashboard_data["recent_events"][:10]:
                events_data.append({
                    'Timestamp': event['timestamp'][:19],  # Remove microseconds
                    'Event Type': event['event_type'].replace('_', ' ').title(),
                    'Source Asset': event['source_asset_id'][:20] + '...' if len(event['source_asset_id']) > 20 else event['source_asset_id'],
                    'User': event['user_id'],
                    'Operation': event['operation']
                })
            
            events_df = pd.DataFrame(events_data)
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No recent lineage events")
    
    with tab2:
        st.subheader("Asset Catalog")
        
        # Asset search and filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("Search Assets")
        
        with col2:
            asset_type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + [t.value for t in DataAssetType]
            )
        
        with col3:
            sensitivity_filter = st.selectbox(
                "Filter by Sensitivity",
                ["All"] + [s.value for s in DataSensitivityLevel]
            )
        
        # Display assets
        assets = list(governance_manager.lineage_graph.asset_metadata.values())
        
        # Apply filters
        filtered_assets = assets
        
        if search_term:
            filtered_assets = [
                asset for asset in filtered_assets
                if search_term.lower() in asset.name.lower() or 
                   search_term.lower() in asset.description.lower()
            ]
        
        if asset_type_filter != "All":
            filtered_assets = [
                asset for asset in filtered_assets
                if asset.asset_type.value == asset_type_filter
            ]
        
        if sensitivity_filter != "All":
            filtered_assets = [
                asset for asset in filtered_assets
                if asset.sensitivity_level.value == sensitivity_filter
            ]
        
        st.write(f"Found {len(filtered_assets)} assets")
        
        # Asset list
        for asset in filtered_assets[:20]:  # Show first 20
            with st.expander(f"{asset.name} ({asset.asset_type.value})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**ID:** {asset.asset_id}")
                    st.write(f"**Description:** {asset.description}")
                    st.write(f"**Owner:** {asset.owner}")
                    st.write(f"**Steward:** {asset.steward}")
                    st.write(f"**Size:** {asset.size_bytes:,} bytes")
                
                with col2:
                    st.write(f"**Sensitivity:** {asset.sensitivity_level.value}")
                    st.write(f"**Quality Score:** {asset.quality_score:.3f}")
                    st.write(f"**Compliance Status:** {asset.compliance_status.value}")
                    st.write(f"**Last Modified:** {asset.last_modified.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Access Count:** {asset.access_count}")
                
                # Show lineage connections
                if asset.parent_assets or asset.child_assets:
                    st.write("**Lineage Connections:**")
                    if asset.parent_assets:
                        st.write(f"Parents: {', '.join(asset.parent_assets[:3])}...")
                    if asset.child_assets:
                        st.write(f"Children: {', '.join(asset.child_assets[:3])}...")
        
        if len(filtered_assets) > 20:
            st.info(f"Showing 20 of {len(filtered_assets)} assets")
    
    with tab3:
        st.subheader("Lineage Viewer")
        
        # Asset selection for lineage analysis
        asset_ids = list(governance_manager.lineage_graph.asset_metadata.keys())
        
        if asset_ids:
            selected_asset = st.selectbox("Select Asset for Lineage Analysis", asset_ids)
            
            if selected_asset:
                col1, col2 = st.columns(2)
                
                with col1:
                    lineage_direction = st.radio(
                        "Lineage Direction",
                        ["Upstream", "Downstream", "Both"]
                    )
                
                with col2:
                    max_depth = st.slider("Maximum Depth", 1, 10, 5)
                
                if st.button("Generate Lineage View"):
                    with st.spinner("Analyzing lineage..."):
                        if lineage_direction in ["Upstream", "Both"]:
                            upstream = governance_manager.lineage_graph.get_upstream_lineage(
                                selected_asset, max_depth
                            )
                            
                            if upstream:
                                st.subheader("Upstream Dependencies")
                                
                                upstream_data = []
                                for asset_id, info in upstream.items():
                                    upstream_data.append({
                                        'Asset ID': asset_id,
                                        'Depth': info['depth'],
                                        'Parents': len(info.get('parents', [])),
                                        'Type': info.get('asset_info', {}).get('asset_type', {}).get('value', 'Unknown') if isinstance(info.get('asset_info', {}), dict) else 'Unknown'
                                    })
                                
                                if upstream_data:
                                    upstream_df = pd.DataFrame(upstream_data)
                                    st.dataframe(upstream_df, use_container_width=True)
                        
                        if lineage_direction in ["Downstream", "Both"]:
                            downstream = governance_manager.lineage_graph.get_downstream_lineage(
                                selected_asset, max_depth
                            )
                            
                            if downstream:
                                st.subheader("Downstream Dependencies")
                                
                                downstream_data = []
                                for asset_id, info in downstream.items():
                                    downstream_data.append({
                                        'Asset ID': asset_id,
                                        'Depth': info['depth'],
                                        'Children': len(info.get('children', [])),
                                        'Type': info.get('asset_info', {}).get('asset_type', {}).get('value', 'Unknown') if isinstance(info.get('asset_info', {}), dict) else 'Unknown'
                                    })
                                
                                if downstream_data:
                                    downstream_df = pd.DataFrame(downstream_data)
                                    st.dataframe(downstream_df, use_container_width=True)
                
                # Impact analysis
                st.subheader("Impact Analysis")
                
                if st.button("Analyze Impact"):
                    with st.spinner("Analyzing impact..."):
                        impact = governance_manager.lineage_graph.get_impact_analysis(selected_asset)
                        
                        if impact:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Impact Summary:**")
                                st.write(f"Total Affected Assets: {impact['summary']['total_affected_assets']}")
                                
                                st.write("**By Impact Level:**")
                                for level, assets in impact['impact_levels'].items():
                                    if assets:
                                        st.write(f"• {level.title()}: {len(assets)} assets")
                            
                            with col2:
                                st.write("**Affected Asset Types:**")
                                for asset_type, count in impact['summary']['affected_by_type'].items():
                                    st.write(f"• {asset_type}: {count}")
        else:
            st.info("No assets available for lineage analysis")
    
    with tab4:
        st.subheader("Data Quality Management")
        
        # Quality rules management
        st.write("**Quality Rules:**")
        
        quality_rules = governance_manager.quality_engine.quality_rules
        
        if quality_rules:
            rules_data = []
            for rule in quality_rules.values():
                rules_data.append({
                    'Rule ID': rule.rule_id,
                    'Name': rule.name,
                    'Type': rule.rule_type,
                    'Threshold': rule.threshold,
                    'Severity': rule.severity,
                    'Active': rule.active
                })
            
            rules_df = pd.DataFrame(rules_data)
            st.dataframe(rules_df, use_container_width=True)
        else:
            st.info("No quality rules configured")
        
        # Add new quality rule
        with st.expander("Add New Quality Rule"):
            col1, col2 = st.columns(2)
            
            with col1:
                rule_name = st.text_input("Rule Name")
                rule_type = st.selectbox(
                    "Rule Type",
                    ["completeness", "accuracy", "consistency", "validity", "uniqueness", "timeliness"]
                )
                threshold = st.slider("Threshold", 0.0, 1.0, 0.8)
            
            with col2:
                rule_description = st.text_area("Description")
                severity = st.selectbox("Severity", ["low", "medium", "high", "critical"])
                target_assets = st.multiselect("Target Assets", asset_ids if asset_ids else [])
            
            if st.button("Add Quality Rule"):
                if rule_name and rule_type:
                    rule = DataQualityRule(
                        rule_id=f"rule_{int(time.time())}",
                        name=rule_name,
                        description=rule_description,
                        rule_type=rule_type,
                        target_assets=target_assets,
                        rule_definition={},
                        threshold=threshold,
                        severity=severity,
                        created_by="user"
                    )
                    
                    governance_manager.quality_engine.add_quality_rule(rule)
                    st.success(f"Quality rule '{rule_name}' added successfully!")
                    st.rerun()
        
        # Quality assessment
        st.subheader("Quality Assessment")
        
        if asset_ids:
            selected_asset_quality = st.selectbox("Select Asset for Quality Assessment", 
                                                 asset_ids, key="quality_asset")
            
            if st.button("Run Quality Assessment"):
                asset = governance_manager.lineage_graph.asset_metadata[selected_asset_quality]
                
                with st.spinner("Assessing data quality..."):
                    quality_results = governance_manager.quality_engine.assess_data_quality(asset)
                    
                    st.write(f"**Overall Quality Score:** {quality_results['overall_score']:.3f}")
                    
                    if quality_results['rule_results']:
                        st.write("**Rule Results:**")
                        
                        for rule_id, result in quality_results['rule_results'].items():
                            if result['score'] is not None:
                                status = "✅" if result['score'] >= 0.8 else "⚠️" if result['score'] >= 0.6 else "❌"
                                st.write(f"{status} {result['rule_name']}: {result['score']:.3f}")
                    
                    if quality_results['recommendations']:
                        st.write("**Recommendations:**")
                        for rec in quality_results['recommendations']:
                            st.write(f"• {rec['recommendation']}")
    
    with tab5:
        st.subheader("Compliance Management")
        
        # Compliance rules management
        st.write("**Compliance Rules:**")
        
        compliance_rules = governance_manager.compliance_engine.compliance_rules
        
        if compliance_rules:
            comp_rules_data = []
            for rule in compliance_rules.values():
                comp_rules_data.append({
                    'Rule ID': rule.rule_id,
                    'Name': rule.name,
                    'Framework': rule.framework,
                    'Severity': rule.severity,
                    'Automated': rule.automated_check
                })
            
            comp_rules_df = pd.DataFrame(comp_rules_data)
            st.dataframe(comp_rules_df, use_container_width=True)
        else:
            st.info("No compliance rules configured")
        
        # Add new compliance rule
        with st.expander("Add New Compliance Rule"):
            col1, col2 = st.columns(2)
            
            with col1:
                comp_rule_name = st.text_input("Rule Name", key="comp_rule_name")
                framework = st.selectbox("Framework", ["GDPR", "HIPAA", "SOX", "PCI_DSS", "Other"])
                comp_severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], key="comp_severity")
            
            with col2:
                comp_rule_description = st.text_area("Description", key="comp_rule_desc")
                automated_check = st.checkbox("Automated Check")
                applicable_types = st.multiselect(
                    "Applicable Asset Types",
                    [t.value for t in DataAssetType]
                )
            
            if st.button("Add Compliance Rule"):
                if comp_rule_name and framework:
                    rule = ComplianceRule(
                        rule_id=f"comp_rule_{int(time.time())}",
                        name=comp_rule_name,
                        framework=framework,
                        description=comp_rule_description,
                        rule_definition={},
                        applicable_asset_types=[DataAssetType(t) for t in applicable_types],
                        severity=comp_severity,
                        automated_check=automated_check
                    )
                    
                    governance_manager.compliance_engine.add_compliance_rule(rule)
                    st.success(f"Compliance rule '{comp_rule_name}' added successfully!")
                    st.rerun()
        
        # Compliance assessment
        st.subheader("Compliance Assessment")
        
        if asset_ids:
            selected_asset_compliance = st.selectbox("Select Asset for Compliance Assessment", 
                                                    asset_ids, key="compliance_asset")
            
            if st.button("Run Compliance Assessment"):
                asset = governance_manager.lineage_graph.asset_metadata[selected_asset_compliance]
                
                with st.spinner("Assessing compliance..."):
                    compliance_results = governance_manager.compliance_engine.assess_compliance(asset)
                    
                    st.write(f"**Overall Compliance Status:** {compliance_results['overall_status'].value}")
                    
                    if compliance_results['framework_results']:
                        st.write("**Framework Results:**")
                        
                        for rule_id, result in compliance_results['framework_results'].items():
                            status = "✅" if result['compliant'] else "❌"
                            st.write(f"{status} {result['rule_name']} ({result['framework']})")
                    
                    if compliance_results['violations']:
                        st.write("**Violations:**")
                        for violation in compliance_results['violations']:
                            st.write(f"⚠️ **{violation['rule']}** ({violation['framework']})")
                            st.write(f"   Severity: {violation['severity']}")
                            st.write(f"   Recommendation: {violation['recommendation']}")
    
    with tab6:
        st.subheader("Administration")
        
        # Sample data generation
        with st.expander("Generate Sample Data"):
            if st.button("🚀 Create Sample Assets"):
                with st.spinner("Creating sample governance data..."):
                    # Create sample assets
                    sample_assets = [
                        DataAsset(
                            asset_id="customer_data_raw",
                            name="Customer Data Raw",
                            asset_type=DataAssetType.RAW_DATA,
                            description="Raw customer data from CRM system",
                            owner="data_team",
                            steward="john.doe",
                            created_date=datetime.now() - timedelta(days=30),
                            last_modified=datetime.now() - timedelta(days=1),
                            schema_definition={"customer_id": "string", "name": "string", "email": "string"},
                            data_location="/data/raw/customers.csv",
                            size_bytes=1024000,
                            row_count=10000,
                            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                            retention_period_days=2555,
                            data_classification=["pii", "customer_data"],
                            business_terms=["customer", "personal_data"],
                            quality_status=DataQualityStatus.GOOD,
                            quality_score=0.85,
                            quality_rules=["completeness", "accuracy"],
                            compliance_status=ComplianceStatus.COMPLIANT,
                            compliance_frameworks=["GDPR"],
                            privacy_flags=["pii", "gdpr"],
                            parent_assets=[],
                            child_assets=["customer_data_processed"],
                            transformation_logic="",
                            access_count=150,
                            last_accessed=datetime.now() - timedelta(hours=2),
                            active_users=["alice", "bob", "charlie"]
                        ),
                        DataAsset(
                            asset_id="customer_data_processed",
                            name="Customer Data Processed",
                            asset_type=DataAssetType.PROCESSED_DATA,
                            description="Processed and cleaned customer data",
                            owner="data_team",
                            steward="jane.smith",
                            created_date=datetime.now() - timedelta(days=25),
                            last_modified=datetime.now() - timedelta(hours=6),
                            schema_definition={"customer_id": "string", "name_cleaned": "string", "email_validated": "string"},
                            data_location="/data/processed/customers_clean.parquet",
                            size_bytes=512000,
                            row_count=9800,
                            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                            retention_period_days=2555,
                            data_classification=["pii", "processed_data"],
                            business_terms=["customer", "analytics_ready"],
                            quality_status=DataQualityStatus.EXCELLENT,
                            quality_score=0.95,
                            quality_rules=["completeness", "accuracy", "consistency"],
                            compliance_status=ComplianceStatus.COMPLIANT,
                            compliance_frameworks=["GDPR"],
                            privacy_flags=["pii", "gdpr"],
                            parent_assets=["customer_data_raw"],
                            child_assets=["customer_analytics_model"],
                            transformation_logic="Clean names, validate emails, remove duplicates",
                            access_count=75,
                            last_accessed=datetime.now() - timedelta(minutes=30),
                            active_users=["alice", "dave"]
                        ),
                        DataAsset(
                            asset_id="customer_analytics_model",
                            name="Customer Analytics Model",
                            asset_type=DataAssetType.MODEL,
                            description="ML model for customer behavior analysis",
                            owner="ml_team",
                            steward="ml.engineer",
                            created_date=datetime.now() - timedelta(days=15),
                            last_modified=datetime.now() - timedelta(days=3),
                            schema_definition={"model_version": "1.2", "accuracy": "0.87"},
                            data_location="/models/customer_analytics/v1.2/",
                            size_bytes=204800,
                            row_count=0,
                            sensitivity_level=DataSensitivityLevel.INTERNAL,
                            retention_period_days=1825,
                            data_classification=["model", "analytics"],
                            business_terms=["machine_learning", "customer_insights"],
                            quality_status=DataQualityStatus.GOOD,
                            quality_score=0.87,
                            quality_rules=["accuracy", "performance"],
                            compliance_status=ComplianceStatus.COMPLIANT,
                            compliance_frameworks=["Internal"],
                            privacy_flags=[],
                            parent_assets=["customer_data_processed"],
                            child_assets=["customer_insights_report"],
                            transformation_logic="Random Forest classifier with feature engineering",
                            access_count=25,
                            last_accessed=datetime.now() - timedelta(hours=8),
                            active_users=["eve", "frank"]
                        )
                    ]
                    
                    # Add assets to governance system
                    for asset in sample_assets:
                        governance_manager.add_data_asset(asset)
                    
                    # Add sample quality rules
                    quality_rules = [
                        DataQualityRule(
                            rule_id="completeness_rule",
                            name="Data Completeness Check",
                            description="Ensure data has minimal null values",
                            rule_type="completeness",
                            target_assets=[],
                            rule_definition={"max_null_percentage": 0.05},
                            threshold=0.95,
                            severity="high"
                        ),
                        DataQualityRule(
                            rule_id="uniqueness_rule",
                            name="Data Uniqueness Check",
                            description="Ensure minimal duplicate records",
                            rule_type="uniqueness",
                            target_assets=[],
                            rule_definition={"max_duplicate_percentage": 0.02},
                            threshold=0.98,
                            severity="medium"
                        )
                    ]
                    
                    for rule in quality_rules:
                        governance_manager.quality_engine.add_quality_rule(rule)
                    
                    # Add sample compliance rules
                    compliance_rules = [
                        ComplianceRule(
                            rule_id="gdpr_retention_rule",
                            name="GDPR Data Retention",
                            framework="GDPR",
                            description="Ensure personal data retention complies with GDPR",
                            rule_definition={"max_retention_days": 2555},
                            applicable_asset_types=[DataAssetType.RAW_DATA, DataAssetType.PROCESSED_DATA],
                            severity="critical",
                            automated_check=True
                        ),
                        ComplianceRule(
                            rule_id="pii_security_rule",
                            name="PII Security Requirements",
                            framework="Internal",
                            description="Ensure PII data has appropriate security classification",
                            rule_definition={"min_sensitivity_level": "confidential"},
                            applicable_asset_types=[DataAssetType.RAW_DATA, DataAssetType.PROCESSED_DATA],
                            severity="high",
                            automated_check=True
                        )
                    ]
                    
                    for rule in compliance_rules:
                        governance_manager.compliance_engine.add_compliance_rule(rule)
                    
                    st.success("Sample governance data created successfully!")
                    st.rerun()
        
        # Database management
        st.subheader("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Governance Data"):
                dashboard_data = governance_manager.get_governance_dashboard_data()
                export_json = json.dumps(dashboard_data, indent=2, default=str)
                
                st.download_button(
                    label="Download JSON",
                    data=export_json,
                    file_name=f"governance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime='application/json'
                )
        
        with col2:
            if st.button("🔄 Refresh All Data"):
                # Reload from database
                governance_manager.lineage_graph.asset_metadata.clear()
                governance_manager.lineage_graph.graph.clear()
                governance_manager.quality_engine.quality_rules.clear()
                governance_manager.compliance_engine.compliance_rules.clear()
                
                governance_manager.load_assets()
                governance_manager.load_quality_rules()
                governance_manager.load_compliance_rules()
                
                st.success("All data refreshed from database!")
                st.rerun()

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing data governance and lineage tracking...")
    
    # Initialize governance manager
    governance_manager = DataGovernanceManager()
    
    # Create sample assets
    sample_asset = DataAsset(
        asset_id="test_dataset",
        name="Test Dataset",
        asset_type=DataAssetType.DATASET,
        description="Test dataset for governance system",
        owner="test_user",
        steward="data_steward",
        created_date=datetime.now(),
        last_modified=datetime.now(),
        schema_definition={"col1": "string", "col2": "integer"},
        data_location="/data/test.csv",
        size_bytes=1024,
        row_count=100,
        sensitivity_level=DataSensitivityLevel.INTERNAL,
        retention_period_days=365,
        data_classification=["test", "sample"],
        business_terms=["testing"],
        quality_status=DataQualityStatus.GOOD,
        quality_score=0.8,
        quality_rules=["completeness"],
        compliance_status=ComplianceStatus.COMPLIANT,
        compliance_frameworks=["Internal"],
        privacy_flags=[],
        parent_assets=[],
        child_assets=[],
        transformation_logic=""
    )
    
    # Add asset
    success = governance_manager.add_data_asset(sample_asset)
    print(f"Asset added: {success}")
    
    # Create quality rule
    quality_rule = DataQualityRule(
        rule_id="test_quality_rule",
        name="Test Completeness Rule",
        description="Test rule for completeness",
        rule_type="completeness",
        target_assets=["test_dataset"],
        rule_definition={},
        threshold=0.9,
        severity="medium"
    )
    
    governance_manager.quality_engine.add_quality_rule(quality_rule)
    
    # Run quality assessment
    quality_results = governance_manager.quality_engine.assess_data_quality(sample_asset)
    print(f"Quality assessment: {quality_results['overall_score']}")
    
    # Create compliance rule
    compliance_rule = ComplianceRule(
        rule_id="test_compliance_rule",
        name="Test Compliance Rule",
        framework="Internal",
        description="Test compliance rule",
        rule_definition={},
        applicable_asset_types=[DataAssetType.DATASET],
        severity="medium"
    )
    
    governance_manager.compliance_engine.add_compliance_rule(compliance_rule)
    
    # Run compliance assessment
    compliance_results = governance_manager.compliance_engine.assess_compliance(sample_asset)
    print(f"Compliance assessment: {compliance_results['overall_status']}")
    
    # Get dashboard data
    dashboard_data = governance_manager.get_governance_dashboard_data()
    print(f"Dashboard data generated with {dashboard_data['asset_summary']['total_assets']} assets")
    
    print("Data governance and lineage tracking test completed!")