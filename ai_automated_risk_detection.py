"""
AI-Driven Automated Risk Detection and Early Warning System
Implements advanced AI algorithms for proactive risk identification and automated response
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskSeverity(Enum):
    """Risk severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AlertType(Enum):
    """Types of automated alerts"""
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    PATTERN_DEVIATION = "pattern_deviation"
    TREND_REVERSAL = "trend_reversal"
    CORRELATION_BREAK = "correlation_break"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"

@dataclass
class RiskAlert:
    """Risk alert with detailed information"""
    alert_id: str
    alert_type: AlertType
    severity: RiskSeverity
    title: str
    description: str
    affected_models: List[str]
    risk_score: float
    confidence: float
    timestamp: datetime
    evidence: Dict[str, Any]
    recommended_actions: List[str]
    auto_resolvable: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class RiskPattern:
    """Identified risk pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: float
    severity_impact: float
    affected_entities: List[str]
    first_observed: datetime
    last_observed: datetime
    pattern_features: Dict[str, float]

class AutomatedAnomalyDetector:
    """Advanced anomaly detection using multiple algorithms"""
    
    def __init__(self, contamination: float = 0.1, sensitivity: float = 0.8):
        self.contamination = contamination
        self.sensitivity = sensitivity
        
        # Initialize multiple detection algorithms
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_features=1.0
        )
        
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        
        # Model state
        self.is_trained = False
        self.baseline_stats = {}
        self.feature_importance = {}
        
    def train(self, data: pd.DataFrame, features: List[str]):
        """Train anomaly detection models"""
        try:
            if data.empty or not features:
                logger.warning("No data or features provided for training")
                return False
            
            # Prepare features
            feature_data = data[features].dropna()
            if feature_data.empty:
                logger.warning("No valid feature data after dropping NaN values")
                return False
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Apply PCA for dimensionality reduction
            if scaled_features.shape[1] > 3:
                reduced_features = self.pca.fit_transform(scaled_features)
            else:
                reduced_features = scaled_features
            
            # Train isolation forest
            self.isolation_forest.fit(reduced_features)
            
            # Calculate baseline statistics
            self.baseline_stats = {
                'mean': np.mean(reduced_features, axis=0),
                'std': np.std(reduced_features, axis=0),
                'percentiles': {
                    'p25': np.percentile(reduced_features, 25, axis=0),
                    'p75': np.percentile(reduced_features, 75, axis=0),
                    'p95': np.percentile(reduced_features, 95, axis=0),
                    'p99': np.percentile(reduced_features, 99, axis=0)
                }
            }
            
            # Calculate feature importance (based on variance)
            feature_variance = np.var(scaled_features, axis=0)
            total_variance = np.sum(feature_variance)
            
            self.feature_importance = {
                features[i]: float(feature_variance[i] / total_variance)
                for i in range(len(features))
            }
            
            self.is_trained = True
            logger.info(f"Anomaly detector trained successfully with {len(feature_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Detect anomalies in new data"""
        if not self.is_trained:
            logger.warning("Anomaly detector not trained")
            return {'anomalies': [], 'anomaly_scores': [], 'summary': {}}
        
        try:
            # Prepare features
            feature_data = data[features].dropna()
            if feature_data.empty:
                return {'anomalies': [], 'anomaly_scores': [], 'summary': {}}
            
            # Scale features
            scaled_features = self.scaler.transform(feature_data)
            
            # Apply PCA
            if hasattr(self.pca, 'components_'):
                reduced_features = self.pca.transform(scaled_features)
            else:
                reduced_features = scaled_features
            
            # Detect anomalies using isolation forest
            anomaly_labels = self.isolation_forest.predict(reduced_features)
            anomaly_scores = self.isolation_forest.score_samples(reduced_features)
            
            # Convert to anomaly indicators (1 = normal, -1 = anomaly)
            anomalies = anomaly_labels == -1
            
            # Calculate detailed anomaly information
            anomaly_details = []
            for idx, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    original_idx = feature_data.index[idx]
                    anomaly_details.append({
                        'index': int(original_idx),
                        'anomaly_score': float(anomaly_scores[idx]),
                        'severity': self._calculate_severity(anomaly_scores[idx]),
                        'features': {
                            feature: float(feature_data.iloc[idx][feature])
                            for feature in features
                        },
                        'deviations': self._calculate_deviations(
                            scaled_features[idx], features
                        )
                    })
            
            # Summary statistics
            summary = {
                'total_samples': len(feature_data),
                'anomaly_count': int(np.sum(anomalies)),
                'anomaly_rate': float(np.mean(anomalies)),
                'avg_anomaly_score': float(np.mean(anomaly_scores[anomalies])) if np.any(anomalies) else 0.0,
                'severity_distribution': self._get_severity_distribution(anomaly_scores[anomalies])
            }
            
            return {
                'anomalies': anomaly_details,
                'anomaly_scores': anomaly_scores.tolist(),
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'anomalies': [], 'anomaly_scores': [], 'summary': {}}
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate severity based on anomaly score"""
        # Isolation forest scores are typically between -1 and 1
        # More negative scores indicate stronger anomalies
        if anomaly_score < -0.6:
            return "critical"
        elif anomaly_score < -0.4:
            return "high"
        elif anomaly_score < -0.2:
            return "medium"
        else:
            return "low"
    
    def _calculate_deviations(self, sample_features: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature deviations from baseline"""
        deviations = {}
        
        for i, feature_name in enumerate(feature_names):
            if i < len(sample_features) and i < len(self.baseline_stats['mean']):
                mean_val = self.baseline_stats['mean'][i]
                std_val = self.baseline_stats['std'][i]
                
                if std_val > 0:
                    z_score = abs((sample_features[i] - mean_val) / std_val)
                    deviations[feature_name] = float(z_score)
                else:
                    deviations[feature_name] = 0.0
        
        return deviations
    
    def _get_severity_distribution(self, anomaly_scores: np.ndarray) -> Dict[str, int]:
        """Get distribution of anomaly severities"""
        if len(anomaly_scores) == 0:
            return {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        severities = [self._calculate_severity(score) for score in anomaly_scores]
        return {
            "critical": severities.count("critical"),
            "high": severities.count("high"),
            "medium": severities.count("medium"),
            "low": severities.count("low")
        }

class PatternRecognitionEngine:
    """Advanced pattern recognition for risk analysis"""
    
    def __init__(self):
        self.known_patterns = {}
        self.pattern_history = []
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.is_trained = False
    
    def identify_patterns(self, data: pd.DataFrame, features: List[str]) -> List[RiskPattern]:
        """Identify recurring risk patterns"""
        try:
            if data.empty or not features:
                return []
            
            # Prepare data
            feature_data = data[features].dropna()
            if len(feature_data) < 10:  # Need minimum data for pattern recognition
                return []
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Perform clustering to identify patterns
            clusters = self.clustering_model.fit_predict(scaled_features)
            
            patterns = []
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Noise cluster in DBSCAN
                    continue
                
                cluster_mask = clusters == cluster_id
                cluster_data = feature_data[cluster_mask]
                cluster_features = scaled_features[cluster_mask]
                
                # Calculate pattern characteristics
                pattern = self._analyze_cluster_pattern(
                    cluster_id, cluster_data, cluster_features, features, data
                )
                
                if pattern:
                    patterns.append(pattern)
            
            # Update pattern history
            self.pattern_history.extend(patterns)
            
            # Keep only recent patterns (last 1000)
            if len(self.pattern_history) > 1000:
                self.pattern_history = self.pattern_history[-1000:]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return []
    
    def _analyze_cluster_pattern(self, cluster_id: int, cluster_data: pd.DataFrame, 
                               cluster_features: np.ndarray, feature_names: List[str],
                               original_data: pd.DataFrame) -> Optional[RiskPattern]:
        """Analyze individual cluster to extract pattern"""
        try:
            if len(cluster_data) < 3:  # Need minimum samples for pattern
                return None
            
            # Calculate pattern features
            pattern_features = {}
            for i, feature_name in enumerate(feature_names):
                if i < cluster_features.shape[1]:
                    pattern_features[feature_name] = {
                        'mean': float(np.mean(cluster_features[:, i])),
                        'std': float(np.std(cluster_features[:, i])),
                        'range': float(np.ptp(cluster_features[:, i]))
                    }
            
            # Determine pattern type based on feature characteristics
            pattern_type = self._classify_pattern_type(pattern_features, cluster_data)
            
            # Calculate severity impact
            severity_impact = self._calculate_pattern_severity(cluster_data, original_data)
            
            # Get time information
            if 'Date' in cluster_data.columns:
                dates = pd.to_datetime(cluster_data['Date'])
                first_observed = dates.min()
                last_observed = dates.max()
            else:
                first_observed = datetime.now() - timedelta(days=30)
                last_observed = datetime.now()
            
            # Create pattern
            pattern = RiskPattern(
                pattern_id=f"pattern_{cluster_id}_{int(time.time())}",
                pattern_type=pattern_type,
                description=self._generate_pattern_description(pattern_type, pattern_features),
                frequency=len(cluster_data) / len(original_data),
                severity_impact=severity_impact,
                affected_entities=self._get_affected_entities(cluster_data),
                first_observed=first_observed,
                last_observed=last_observed,
                pattern_features=pattern_features
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing cluster pattern: {e}")
            return None
    
    def _classify_pattern_type(self, pattern_features: Dict[str, Dict[str, float]], 
                              cluster_data: pd.DataFrame) -> str:
        """Classify the type of pattern based on features"""
        
        # High risk rate pattern
        if 'Risk_Rate' in pattern_features:
            risk_mean = pattern_features['Risk_Rate']['mean']
            if risk_mean > 0.7:
                return "high_risk_cluster"
            elif risk_mean < 0.3:
                return "low_risk_cluster"
        
        # High variability pattern
        high_variability_features = [
            name for name, stats in pattern_features.items()
            if stats['std'] > 1.5  # High standard deviation in scaled features
        ]
        
        if len(high_variability_features) > len(pattern_features) / 2:
            return "high_variability_pattern"
        
        # Model-specific pattern
        if 'Model' in cluster_data.columns:
            unique_models = cluster_data['Model'].nunique()
            if unique_models == 1:
                return "model_specific_pattern"
        
        # Language-specific pattern
        if 'Language' in cluster_data.columns:
            unique_languages = cluster_data['Language'].nunique()
            if unique_languages == 1:
                return "language_specific_pattern"
        
        # Default pattern type
        return "general_risk_pattern"
    
    def _calculate_pattern_severity(self, cluster_data: pd.DataFrame, 
                                  original_data: pd.DataFrame) -> float:
        """Calculate the severity impact of a pattern"""
        try:
            # Base severity on risk rate if available
            if 'Risk_Rate' in cluster_data.columns:
                cluster_risk = cluster_data['Risk_Rate'].mean()
                overall_risk = original_data['Risk_Rate'].mean()
                return float(cluster_risk / overall_risk) if overall_risk > 0 else 1.0
            
            # Alternative: base on sample size and confidence
            if 'Sample_Size' in cluster_data.columns and 'Confidence' in cluster_data.columns:
                weighted_impact = (
                    cluster_data['Sample_Size'] * (1 - cluster_data['Confidence'])
                ).mean()
                return min(weighted_impact / 1000, 2.0)  # Normalize
            
            return 1.0  # Default severity
            
        except Exception:
            return 1.0
    
    def _get_affected_entities(self, cluster_data: pd.DataFrame) -> List[str]:
        """Get list of entities affected by this pattern"""
        entities = []
        
        for col in ['Model', 'Language', 'Risk_Category']:
            if col in cluster_data.columns:
                unique_values = cluster_data[col].unique()
                entities.extend([f"{col}: {val}" for val in unique_values])
        
        return entities[:10]  # Limit to top 10
    
    def _generate_pattern_description(self, pattern_type: str, 
                                    pattern_features: Dict[str, Dict[str, float]]) -> str:
        """Generate human-readable pattern description"""
        descriptions = {
            "high_risk_cluster": "Cluster of consistently high-risk instances",
            "low_risk_cluster": "Cluster of low-risk instances with stable behavior",
            "high_variability_pattern": "Pattern showing high variability across risk metrics",
            "model_specific_pattern": "Risk pattern specific to a particular model",
            "language_specific_pattern": "Risk pattern associated with specific language processing",
            "general_risk_pattern": "General risk pattern identified in the data"
        }
        
        base_description = descriptions.get(pattern_type, "Unclassified risk pattern")
        
        # Add feature details
        if 'Risk_Rate' in pattern_features:
            risk_mean = pattern_features['Risk_Rate']['mean']
            base_description += f" (avg risk: {risk_mean:.3f})"
        
        return base_description

class IntelligentAlertManager:
    """Manages intelligent alerting with priority ranking and auto-resolution"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.alert_rules = self._initialize_alert_rules()
        self.auto_resolution_handlers = {}
        self.notification_channels = []
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default alert rules"""
        return {
            'high_risk_rate': {
                'condition': lambda data: data.get('Risk_Rate', 0) > 0.8,
                'severity': RiskSeverity.HIGH,
                'message': 'Risk rate exceeded critical threshold',
                'auto_resolvable': False
            },
            'anomaly_burst': {
                'condition': lambda data: data.get('anomaly_count', 0) > 10,
                'severity': RiskSeverity.CRITICAL,
                'message': 'Anomaly burst detected - multiple unusual patterns',
                'auto_resolvable': True
            },
            'model_degradation': {
                'condition': lambda data: data.get('confidence_drop', 0) > 0.3,
                'severity': RiskSeverity.HIGH,
                'message': 'Model performance degradation detected',
                'auto_resolvable': False
            },
            'pattern_deviation': {
                'condition': lambda data: data.get('pattern_deviation_score', 0) > 0.7,
                'severity': RiskSeverity.MEDIUM,
                'message': 'Significant deviation from established patterns',
                'auto_resolvable': True
            }
        }
    
    def evaluate_alerts(self, risk_data: pd.DataFrame, anomaly_results: Dict[str, Any],
                       patterns: List[RiskPattern]) -> List[RiskAlert]:
        """Evaluate data and generate intelligent alerts"""
        new_alerts = []
        
        try:
            # Prepare evaluation context
            context = self._prepare_evaluation_context(risk_data, anomaly_results, patterns)
            
            # Evaluate each alert rule
            for rule_name, rule_config in self.alert_rules.items():
                if rule_config['condition'](context):
                    alert = self._create_alert(rule_name, rule_config, context)
                    if alert:
                        new_alerts.append(alert)
            
            # Check for pattern-based alerts
            pattern_alerts = self._evaluate_pattern_alerts(patterns, context)
            new_alerts.extend(pattern_alerts)
            
            # Prioritize and deduplicate alerts
            prioritized_alerts = self._prioritize_alerts(new_alerts)
            
            # Update active alerts
            for alert in prioritized_alerts:
                self.active_alerts[alert.alert_id] = alert
            
            # Attempt auto-resolution
            self._attempt_auto_resolution(prioritized_alerts)
            
            return prioritized_alerts
            
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")
            return []
    
    def _prepare_evaluation_context(self, risk_data: pd.DataFrame, 
                                  anomaly_results: Dict[str, Any],
                                  patterns: List[RiskPattern]) -> Dict[str, Any]:
        """Prepare context for alert evaluation"""
        context = {}
        
        # Risk data statistics
        if not risk_data.empty:
            context.update({
                'Risk_Rate': risk_data['Risk_Rate'].mean() if 'Risk_Rate' in risk_data.columns else 0,
                'max_risk_rate': risk_data['Risk_Rate'].max() if 'Risk_Rate' in risk_data.columns else 0,
                'risk_variance': risk_data['Risk_Rate'].var() if 'Risk_Rate' in risk_data.columns else 0,
                'sample_count': len(risk_data),
                'unique_models': risk_data['Model'].nunique() if 'Model' in risk_data.columns else 0,
                'avg_confidence': risk_data['Confidence'].mean() if 'Confidence' in risk_data.columns else 0
            })
        
        # Anomaly results
        if anomaly_results and 'summary' in anomaly_results:
            summary = anomaly_results['summary']
            context.update({
                'anomaly_count': summary.get('anomaly_count', 0),
                'anomaly_rate': summary.get('anomaly_rate', 0),
                'avg_anomaly_score': summary.get('avg_anomaly_score', 0)
            })
        
        # Pattern information
        context.update({
            'pattern_count': len(patterns),
            'high_severity_patterns': len([p for p in patterns if p.severity_impact > 1.5]),
            'pattern_deviation_score': self._calculate_pattern_deviation_score(patterns)
        })
        
        # Temporal context
        if 'Date' in risk_data.columns:
            recent_data = risk_data[risk_data['Date'] >= (datetime.now() - timedelta(hours=24))]
            context['recent_sample_count'] = len(recent_data)
            
            if len(recent_data) > 0 and len(risk_data) > len(recent_data):
                historical_data = risk_data[risk_data['Date'] < (datetime.now() - timedelta(hours=24))]
                if 'Confidence' in risk_data.columns:
                    recent_confidence = recent_data['Confidence'].mean()
                    historical_confidence = historical_data['Confidence'].mean()
                    context['confidence_drop'] = historical_confidence - recent_confidence
        
        return context
    
    def _calculate_pattern_deviation_score(self, patterns: List[RiskPattern]) -> float:
        """Calculate how much current patterns deviate from historical norms"""
        if not patterns:
            return 0.0
        
        # Simple heuristic: more patterns with high severity = higher deviation
        high_severity_count = len([p for p in patterns if p.severity_impact > 1.2])
        total_patterns = len(patterns)
        
        return high_severity_count / total_patterns if total_patterns > 0 else 0.0
    
    def _create_alert(self, rule_name: str, rule_config: Dict[str, Any], 
                     context: Dict[str, Any]) -> Optional[RiskAlert]:
        """Create an alert based on rule and context"""
        try:
            alert_id = f"{rule_name}_{int(time.time())}_{hash(str(context)) % 10000}"
            
            # Determine affected models
            affected_models = []
            if 'unique_models' in context and context['unique_models'] > 0:
                affected_models = ["Multiple Models"]  # Simplification
            
            # Calculate risk score based on context
            risk_score = self._calculate_contextual_risk_score(rule_name, context)
            
            # Generate evidence
            evidence = self._generate_alert_evidence(rule_name, context)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(rule_name, context)
            
            alert = RiskAlert(
                alert_id=alert_id,
                alert_type=AlertType.THRESHOLD_EXCEEDED,  # Default type
                severity=rule_config['severity'],
                title=f"{rule_name.replace('_', ' ').title()} Alert",
                description=rule_config['message'],
                affected_models=affected_models,
                risk_score=risk_score,
                confidence=0.8,  # Default confidence
                timestamp=datetime.now(),
                evidence=evidence,
                recommended_actions=recommendations,
                auto_resolvable=rule_config.get('auto_resolvable', False)
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert for rule {rule_name}: {e}")
            return None
    
    def _calculate_contextual_risk_score(self, rule_name: str, context: Dict[str, Any]) -> float:
        """Calculate risk score based on rule and context"""
        base_scores = {
            'high_risk_rate': 0.9,
            'anomaly_burst': 0.95,
            'model_degradation': 0.8,
            'pattern_deviation': 0.7
        }
        
        base_score = base_scores.get(rule_name, 0.5)
        
        # Adjust based on context
        if context.get('anomaly_rate', 0) > 0.2:
            base_score = min(base_score + 0.1, 1.0)
        
        if context.get('risk_variance', 0) > 0.1:
            base_score = min(base_score + 0.05, 1.0)
        
        return base_score
    
    def _generate_alert_evidence(self, rule_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evidence for the alert"""
        evidence = {
            'rule_triggered': rule_name,
            'evaluation_time': datetime.now().isoformat(),
            'context_snapshot': {k: v for k, v in context.items() if isinstance(v, (int, float, str))}
        }
        
        # Add rule-specific evidence
        if rule_name == 'high_risk_rate':
            evidence['risk_metrics'] = {
                'current_risk_rate': context.get('Risk_Rate', 0),
                'max_risk_rate': context.get('max_risk_rate', 0),
                'threshold': 0.8
            }
        elif rule_name == 'anomaly_burst':
            evidence['anomaly_metrics'] = {
                'anomaly_count': context.get('anomaly_count', 0),
                'anomaly_rate': context.get('anomaly_rate', 0),
                'threshold': 10
            }
        
        return evidence
    
    def _generate_recommendations(self, rule_name: str, context: Dict[str, Any]) -> List[str]:
        """Generate contextual recommendations"""
        recommendations = {
            'high_risk_rate': [
                "Review model parameters and retrain with recent data",
                "Implement additional safety filters",
                "Conduct manual review of high-risk instances",
                "Consider model rollback if degradation is severe"
            ],
            'anomaly_burst': [
                "Investigate data pipeline for potential issues",
                "Check for input data corruption or drift",
                "Verify model serving infrastructure",
                "Implement temporary rate limiting"
            ],
            'model_degradation': [
                "Compare current model performance with baseline",
                "Check for training data distribution shift",
                "Consider model retraining with fresh data",
                "Implement A/B testing with previous model version"
            ],
            'pattern_deviation': [
                "Analyze new patterns for potential risks",
                "Update pattern recognition thresholds",
                "Review recent changes in data sources",
                "Consider pattern-specific mitigation strategies"
            ]
        }
        
        return recommendations.get(rule_name, ["Investigate the issue and take appropriate action"])
    
    def _evaluate_pattern_alerts(self, patterns: List[RiskPattern], 
                                context: Dict[str, Any]) -> List[RiskAlert]:
        """Evaluate patterns for additional alerts"""
        pattern_alerts = []
        
        for pattern in patterns:
            if pattern.severity_impact > 1.5:  # High impact pattern
                alert = RiskAlert(
                    alert_id=f"pattern_{pattern.pattern_id}_{int(time.time())}",
                    alert_type=AlertType.PATTERN_DEVIATION,
                    severity=RiskSeverity.MEDIUM if pattern.severity_impact < 2.0 else RiskSeverity.HIGH,
                    title=f"High Impact Pattern Detected",
                    description=f"Pattern '{pattern.pattern_type}' shows significant risk impact",
                    affected_models=pattern.affected_entities,
                    risk_score=min(pattern.severity_impact / 2.0, 1.0),
                    confidence=0.75,
                    timestamp=datetime.now(),
                    evidence={
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'severity_impact': pattern.severity_impact,
                        'frequency': pattern.frequency,
                        'affected_entities': pattern.affected_entities
                    },
                    recommended_actions=[
                        f"Investigate {pattern.pattern_type} pattern",
                        "Review affected entities for common factors",
                        "Consider pattern-specific risk mitigation"
                    ],
                    auto_resolvable=False
                )
                pattern_alerts.append(alert)
        
        return pattern_alerts
    
    def _prioritize_alerts(self, alerts: List[RiskAlert]) -> List[RiskAlert]:
        """Prioritize alerts based on severity, confidence, and impact"""
        def priority_score(alert: RiskAlert) -> float:
            severity_weights = {
                RiskSeverity.LOW: 1.0,
                RiskSeverity.MEDIUM: 2.0,
                RiskSeverity.HIGH: 3.0,
                RiskSeverity.CRITICAL: 4.0
            }
            
            return (
                severity_weights[alert.severity] * 0.4 +
                alert.risk_score * 0.3 +
                alert.confidence * 0.2 +
                len(alert.affected_models) * 0.1
            )
        
        return sorted(alerts, key=priority_score, reverse=True)
    
    def _attempt_auto_resolution(self, alerts: List[RiskAlert]):
        """Attempt to auto-resolve alerts where possible"""
        for alert in alerts:
            if alert.auto_resolvable and not alert.resolved:
                success = self._auto_resolve_alert(alert)
                if success:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    logger.info(f"Auto-resolved alert: {alert.alert_id}")
    
    def _auto_resolve_alert(self, alert: RiskAlert) -> bool:
        """Attempt to automatically resolve an alert"""
        try:
            # Implement auto-resolution logic based on alert type
            if alert.alert_type == AlertType.ANOMALY_DETECTED:
                # Example: Clear anomaly flags or adjust thresholds
                return True
            elif alert.alert_type == AlertType.PATTERN_DEVIATION:
                # Example: Update pattern baselines
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error auto-resolving alert {alert.alert_id}: {e}")
            return False

class AIRiskAssessmentEngine:
    """Main AI-driven risk assessment engine"""
    
    def __init__(self):
        self.anomaly_detector = AutomatedAnomalyDetector()
        self.pattern_engine = PatternRecognitionEngine()
        self.alert_manager = IntelligentAlertManager()
        
        self.assessment_history = []
        self.continuous_monitoring = False
        self.monitoring_thread = None
        
    def initialize(self, historical_data: pd.DataFrame):
        """Initialize the AI engine with historical data"""
        try:
            if historical_data.empty:
                logger.warning("No historical data provided for initialization")
                return False
            
            # Define feature columns for training
            feature_columns = []
            for col in ['Risk_Rate', 'Confidence', 'Sample_Size']:
                if col in historical_data.columns:
                    feature_columns.append(col)
            
            if not feature_columns:
                logger.warning("No suitable features found for training")
                return False
            
            # Train anomaly detector
            anomaly_success = self.anomaly_detector.train(historical_data, feature_columns)
            
            # Initialize pattern recognition
            initial_patterns = self.pattern_engine.identify_patterns(historical_data, feature_columns)
            logger.info(f"Identified {len(initial_patterns)} initial patterns")
            
            logger.info(f"AI Risk Assessment Engine initialized with {len(historical_data)} samples")
            return anomaly_success
            
        except Exception as e:
            logger.error(f"Error initializing AI engine: {e}")
            return False
    
    def assess_current_risk(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive AI-driven risk assessment"""
        try:
            assessment_start = time.time()
            
            # Feature columns
            feature_columns = []
            for col in ['Risk_Rate', 'Confidence', 'Sample_Size']:
                if col in current_data.columns:
                    feature_columns.append(col)
            
            if not feature_columns:
                return {'error': 'No suitable features found for assessment'}
            
            # 1. Anomaly Detection
            anomaly_results = self.anomaly_detector.detect_anomalies(current_data, feature_columns)
            
            # 2. Pattern Recognition
            patterns = self.pattern_engine.identify_patterns(current_data, feature_columns)
            
            # 3. Alert Generation
            alerts = self.alert_manager.evaluate_alerts(current_data, anomaly_results, patterns)
            
            # 4. Risk Score Calculation
            overall_risk_score = self._calculate_overall_risk_score(
                current_data, anomaly_results, patterns, alerts
            )
            
            # 5. Generate Insights
            insights = self._generate_ai_insights(current_data, anomaly_results, patterns, alerts)
            
            # Compile assessment results
            assessment = {
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - assessment_start,
                'overall_risk_score': overall_risk_score,
                'anomaly_results': anomaly_results,
                'patterns': [asdict(pattern) for pattern in patterns],
                'alerts': [asdict(alert) for alert in alerts],
                'insights': insights,
                'recommendations': self._generate_strategic_recommendations(alerts, patterns)
            }
            
            # Store in history
            self.assessment_history.append(assessment)
            if len(self.assessment_history) > 100:
                self.assessment_history = self.assessment_history[-100:]
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in AI risk assessment: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_risk_score(self, data: pd.DataFrame, anomaly_results: Dict[str, Any],
                                    patterns: List[RiskPattern], alerts: List[RiskAlert]) -> float:
        """Calculate comprehensive risk score"""
        try:
            risk_components = []
            
            # Base risk from data
            if 'Risk_Rate' in data.columns and not data.empty:
                base_risk = data['Risk_Rate'].mean()
                risk_components.append(('base_risk', base_risk, 0.3))
            
            # Anomaly contribution
            if anomaly_results and 'summary' in anomaly_results:
                anomaly_score = min(anomaly_results['summary'].get('anomaly_rate', 0) * 2, 1.0)
                risk_components.append(('anomaly_risk', anomaly_score, 0.25))
            
            # Pattern contribution
            pattern_risk = 0
            if patterns:
                avg_pattern_severity = np.mean([p.severity_impact for p in patterns])
                pattern_risk = min(avg_pattern_severity / 2.0, 1.0)
            risk_components.append(('pattern_risk', pattern_risk, 0.2))
            
            # Alert contribution
            alert_risk = 0
            if alerts:
                severity_weights = {
                    RiskSeverity.LOW: 0.25,
                    RiskSeverity.MEDIUM: 0.5,
                    RiskSeverity.HIGH: 0.75,
                    RiskSeverity.CRITICAL: 1.0
                }
                alert_risk = np.mean([severity_weights[alert.severity] for alert in alerts])
            risk_components.append(('alert_risk', alert_risk, 0.25))
            
            # Calculate weighted average
            total_score = sum(score * weight for _, score, weight in risk_components)
            total_weight = sum(weight for _, _, weight in risk_components)
            
            overall_risk = total_score / total_weight if total_weight > 0 else 0.0
            
            return min(max(overall_risk, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 0.5  # Default moderate risk
    
    def _generate_ai_insights(self, data: pd.DataFrame, anomaly_results: Dict[str, Any],
                            patterns: List[RiskPattern], alerts: List[RiskAlert]) -> List[str]:
        """Generate AI-driven insights"""
        insights = []
        
        try:
            # Data volume insight
            if not data.empty:
                insights.append(f"Analyzed {len(data)} risk assessments from recent data")
            
            # Anomaly insights
            if anomaly_results and 'summary' in anomaly_results:
                summary = anomaly_results['summary']
                anomaly_rate = summary.get('anomaly_rate', 0)
                
                if anomaly_rate > 0.1:
                    insights.append(f"High anomaly rate detected: {anomaly_rate:.1%} of samples show unusual patterns")
                elif anomaly_rate > 0.05:
                    insights.append(f"Moderate anomaly activity: {anomaly_rate:.1%} anomaly rate requires monitoring")
                else:
                    insights.append("Low anomaly rate indicates stable risk patterns")
            
            # Pattern insights
            if patterns:
                high_impact_patterns = [p for p in patterns if p.severity_impact > 1.2]
                if high_impact_patterns:
                    insights.append(f"Identified {len(high_impact_patterns)} high-impact risk patterns requiring attention")
                
                frequent_patterns = [p for p in patterns if p.frequency > 0.1]
                if frequent_patterns:
                    insights.append(f"Found {len(frequent_patterns)} frequently occurring patterns affecting system behavior")
            
            # Alert insights
            if alerts:
                critical_alerts = [a for a in alerts if a.severity == RiskSeverity.CRITICAL]
                if critical_alerts:
                    insights.append(f"⚠️ {len(critical_alerts)} critical alerts require immediate attention")
                
                auto_resolvable = [a for a in alerts if a.auto_resolvable and not a.resolved]
                if auto_resolvable:
                    insights.append(f"🔧 {len(auto_resolvable)} alerts can be automatically resolved")
            
            # Model performance insights
            if 'Model' in data.columns and 'Risk_Rate' in data.columns:
                model_risks = data.groupby('Model')['Risk_Rate'].mean().sort_values(ascending=False)
                if len(model_risks) > 1:
                    highest_risk_model = model_risks.index[0]
                    lowest_risk_model = model_risks.index[-1]
                    insights.append(f"📊 Model performance varies: '{highest_risk_model}' shows highest risk, '{lowest_risk_model}' shows lowest")
            
            # Confidence insights
            if 'Confidence' in data.columns:
                avg_confidence = data['Confidence'].mean()
                if avg_confidence < 0.7:
                    insights.append(f"⚠️ Low average confidence ({avg_confidence:.2f}) suggests model uncertainty")
                elif avg_confidence > 0.9:
                    insights.append(f"✅ High average confidence ({avg_confidence:.2f}) indicates reliable predictions")
            
            # Temporal insights
            if 'Date' in data.columns:
                data_with_dates = data.copy()
                data_with_dates['Date'] = pd.to_datetime(data_with_dates['Date'])
                recent_data = data_with_dates[data_with_dates['Date'] >= (datetime.now() - timedelta(hours=24))]
                
                if len(recent_data) > 0:
                    recent_risk = recent_data['Risk_Rate'].mean() if 'Risk_Rate' in recent_data.columns else 0
                    overall_risk = data['Risk_Rate'].mean() if 'Risk_Rate' in data.columns else 0
                    
                    if recent_risk > overall_risk * 1.1:
                        insights.append("📈 Risk levels have increased in the last 24 hours")
                    elif recent_risk < overall_risk * 0.9:
                        insights.append("📉 Risk levels have decreased in the last 24 hours")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            insights.append("Error generating insights - system monitoring continues")
        
        return insights[:10]  # Limit to top 10 insights
    
    def _generate_strategic_recommendations(self, alerts: List[RiskAlert], 
                                          patterns: List[RiskPattern]) -> List[str]:
        """Generate strategic recommendations based on AI analysis"""
        recommendations = []
        
        try:
            # Alert-based recommendations
            if alerts:
                critical_count = len([a for a in alerts if a.severity == RiskSeverity.CRITICAL])
                high_count = len([a for a in alerts if a.severity == RiskSeverity.HIGH])
                
                if critical_count > 0:
                    recommendations.append("🚨 Implement emergency response procedures for critical alerts")
                    recommendations.append("🔍 Conduct immediate investigation of critical risk factors")
                
                if high_count > 2:
                    recommendations.append("⚡ Establish dedicated response team for high-priority issues")
                
                auto_resolvable_count = len([a for a in alerts if a.auto_resolvable])
                if auto_resolvable_count > 0:
                    recommendations.append(f"🤖 Enable automated resolution for {auto_resolvable_count} eligible alerts")
            
            # Pattern-based recommendations
            if patterns:
                high_impact_patterns = [p for p in patterns if p.severity_impact > 1.5]
                if high_impact_patterns:
                    recommendations.append("📊 Develop targeted mitigation strategies for high-impact patterns")
                
                model_specific_patterns = [p for p in patterns if 'model_specific' in p.pattern_type]
                if model_specific_patterns:
                    recommendations.append("🔧 Review and retrain models showing specific risk patterns")
                
                frequent_patterns = [p for p in patterns if p.frequency > 0.15]
                if frequent_patterns:
                    recommendations.append("🔄 Implement proactive monitoring for frequently occurring patterns")
            
            # General strategic recommendations
            recommendations.extend([
                "📈 Establish continuous monitoring dashboard for real-time risk tracking",
                "🎯 Implement predictive maintenance based on risk pattern analysis",
                "📚 Create knowledge base of resolved issues for faster future response",
                "🔬 Conduct regular model performance reviews and updates",
                "🛡️ Enhance data quality validation to prevent anomalous inputs"
            ])
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")
            recommendations.append("Continue monitoring and manual review of system performance")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def start_continuous_monitoring(self, data_source_callback: Callable[[], pd.DataFrame], 
                                  interval_minutes: int = 30):
        """Start continuous AI monitoring"""
        if self.continuous_monitoring:
            logger.warning("Continuous monitoring already active")
            return
        
        self.continuous_monitoring = True
        
        def monitoring_loop():
            while self.continuous_monitoring:
                try:
                    # Get fresh data
                    current_data = data_source_callback()
                    
                    if not current_data.empty:
                        # Perform assessment
                        assessment = self.assess_current_risk(current_data)
                        
                        # Log significant findings
                        if 'alerts' in assessment and assessment['alerts']:
                            alert_count = len(assessment['alerts'])
                            logger.info(f"Continuous monitoring generated {alert_count} alerts")
                        
                        # Check for critical alerts
                        if 'alerts' in assessment:
                            critical_alerts = [
                                alert for alert in assessment['alerts'] 
                                if alert.get('severity') == RiskSeverity.CRITICAL.value
                            ]
                            
                            if critical_alerts:
                                logger.critical(f"CRITICAL: {len(critical_alerts)} critical alerts detected!")
                    
                    # Wait for next interval
                    time.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    logger.error(f"Error in continuous monitoring: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Started continuous AI monitoring with {interval_minutes}-minute intervals")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.continuous_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Stopped continuous AI monitoring")

# Streamlit Integration Functions

def initialize_ai_system():
    """Initialize AI risk assessment system"""
    if 'ai_risk_engine' not in st.session_state:
        st.session_state.ai_risk_engine = AIRiskAssessmentEngine()
    
    return st.session_state.ai_risk_engine

def render_ai_risk_dashboard():
    """Render AI-driven risk assessment dashboard"""
    st.header("🤖 AI-Driven Risk Assessment")
    
    ai_engine = initialize_ai_system()
    
    # Check if engine is initialized
    if not ai_engine.anomaly_detector.is_trained:
        st.warning("⚠️ AI system not initialized. Please provide historical data for training.")
        
        with st.expander("Initialize AI System"):
            # Sample data generation for demo
            if st.button("Generate Sample Data for Demo"):
                # Generate sample historical data
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    'Date': pd.date_range(start='2024-01-01', end='2024-12-31', freq='D'),
                    'Model': np.random.choice(['GPT-4', 'Claude-3', 'Gemini-Pro'], size=365),
                    'Language': np.random.choice(['English', 'Spanish', 'French'], size=365),
                    'Risk_Category': np.random.choice(['Bias', 'Toxicity', 'Misinformation'], size=365),
                    'Risk_Rate': np.random.beta(2, 5, size=365),  # Skewed towards lower values
                    'Confidence': np.random.beta(8, 2, size=365),  # Skewed towards higher values
                    'Sample_Size': np.random.randint(50, 500, size=365)
                })
                
                # Add some anomalies
                anomaly_indices = np.random.choice(365, size=20, replace=False)
                sample_data.loc[anomaly_indices, 'Risk_Rate'] = np.random.uniform(0.8, 1.0, size=20)
                sample_data.loc[anomaly_indices, 'Confidence'] = np.random.uniform(0.3, 0.6, size=20)
                
                # Initialize AI system
                success = ai_engine.initialize(sample_data)
                
                if success:
                    st.success("✅ AI system initialized successfully with sample data!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize AI system")
        
        return
    
    # AI Assessment Interface
    st.subheader("🔍 Current Risk Assessment")
    
    # Generate current data for assessment (in real app, this would come from live data)
    if st.button("🚀 Run AI Risk Assessment"):
        with st.spinner("Running AI analysis..."):
            # Generate current data sample
            np.random.seed(int(time.time()))  # Different seed for current data
            current_data = pd.DataFrame({
                'Date': [datetime.now() - timedelta(hours=i) for i in range(24)],
                'Model': np.random.choice(['GPT-4', 'Claude-3', 'Gemini-Pro'], size=24),
                'Language': np.random.choice(['English', 'Spanish', 'French'], size=24),
                'Risk_Category': np.random.choice(['Bias', 'Toxicity', 'Misinformation'], size=24),
                'Risk_Rate': np.random.beta(2, 5, size=24),
                'Confidence': np.random.beta(8, 2, size=24),
                'Sample_Size': np.random.randint(50, 500, size=24)
            })
            
            # Add some anomalies for demonstration
            if np.random.random() > 0.7:
                anomaly_indices = np.random.choice(24, size=3, replace=False)
                current_data.loc[anomaly_indices, 'Risk_Rate'] = np.random.uniform(0.8, 1.0, size=3)
            
            # Run AI assessment
            assessment = ai_engine.assess_current_risk(current_data)
            
            if 'error' in assessment:
                st.error(f"Assessment failed: {assessment['error']}")
                return
            
            # Store assessment in session state
            st.session_state.current_assessment = assessment
    
    # Display Assessment Results
    if 'current_assessment' in st.session_state:
        assessment = st.session_state.current_assessment
        
        # Overall Risk Score
        risk_score = assessment.get('overall_risk_score', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_color = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.4 else "🟢"
            st.metric("Overall Risk Score", f"{risk_score:.3f}", delta=f"{risk_color}")
        
        with col2:
            anomaly_count = assessment.get('anomaly_results', {}).get('summary', {}).get('anomaly_count', 0)
            st.metric("Anomalies Detected", str(anomaly_count))
        
        with col3:
            pattern_count = len(assessment.get('patterns', []))
            st.metric("Risk Patterns", str(pattern_count))
        
        with col4:
            alert_count = len(assessment.get('alerts', []))
            st.metric("Active Alerts", str(alert_count))
        
        # Tabs for detailed results
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Alerts", 
            "🔍 Anomalies", 
            "📊 Patterns", 
            "💡 AI Insights", 
            "🎯 Recommendations"
        ])
        
        with tab1:
            st.subheader("Active Alerts")
            
            alerts = assessment.get('alerts', [])
            
            if alerts:
                for i, alert in enumerate(alerts):
                    severity_colors = {
                        1: "🟢",  # LOW
                        2: "🟡",  # MEDIUM
                        3: "🟠",  # HIGH
                        4: "🔴"   # CRITICAL
                    }
                    
                    severity_icon = severity_colors.get(alert.get('severity', 1), "⚪")
                    
                    with st.expander(f"{severity_icon} {alert.get('title', 'Alert')} (Risk: {alert.get('risk_score', 0):.3f})"):
                        st.write(f"**Description:** {alert.get('description', 'No description')}")
                        st.write(f"**Confidence:** {alert.get('confidence', 0):.2f}")
                        st.write(f"**Timestamp:** {alert.get('timestamp', 'Unknown')}")
                        
                        if alert.get('affected_models'):
                            st.write(f"**Affected Models:** {', '.join(alert.get('affected_models', []))}")
                        
                        if alert.get('recommended_actions'):
                            st.write("**Recommended Actions:**")
                            for action in alert.get('recommended_actions', []):
                                st.write(f"• {action}")
                        
                        if alert.get('evidence'):
                            with st.expander("Show Evidence"):
                                st.json(alert['evidence'])
            else:
                st.success("✅ No active alerts - system operating normally")
        
        with tab2:
            st.subheader("Anomaly Detection Results")
            
            anomaly_results = assessment.get('anomaly_results', {})
            
            if anomaly_results and 'summary' in anomaly_results:
                summary = anomaly_results['summary']
                
                # Anomaly summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Anomaly Rate", f"{summary.get('anomaly_rate', 0):.1%}")
                
                with col2:
                    st.metric("Anomaly Count", str(summary.get('anomaly_count', 0)))
                
                with col3:
                    avg_score = summary.get('avg_anomaly_score', 0)
                    st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
                
                # Severity distribution
                if 'severity_distribution' in summary:
                    st.write("**Anomaly Severity Distribution:**")
                    severity_dist = summary['severity_distribution']
                    
                    severity_df = pd.DataFrame([
                        {'Severity': k.title(), 'Count': v} 
                        for k, v in severity_dist.items()
                    ])
                    
                    if not severity_df.empty and severity_df['Count'].sum() > 0:
                        fig = px.bar(severity_df, x='Severity', y='Count', 
                                   title='Anomaly Severity Distribution',
                                   color='Severity',
                                   color_discrete_map={
                                       'Critical': '#ff0000',
                                       'High': '#ff4444',
                                       'Medium': '#ff9500',
                                       'Low': '#36a64f'
                                   })
                        st.plotly_chart(fig, use_container_width=True)
                
                # Individual anomalies
                anomalies = anomaly_results.get('anomalies', [])
                if anomalies:
                    st.write(f"**Individual Anomalies ({len(anomalies)}):**")
                    
                    for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
                        with st.expander(f"Anomaly {i+1} - {anomaly.get('severity', 'unknown').title()} Severity"):
                            st.write(f"**Anomaly Score:** {anomaly.get('anomaly_score', 0):.3f}")
                            st.write(f"**Data Index:** {anomaly.get('index', 'Unknown')}")
                            
                            if anomaly.get('features'):
                                st.write("**Feature Values:**")
                                for feature, value in anomaly.get('features', {}).items():
                                    st.write(f"• {feature}: {value:.3f}")
                            
                            if anomaly.get('deviations'):
                                st.write("**Feature Deviations (Z-scores):**")
                                for feature, deviation in anomaly.get('deviations', {}).items():
                                    st.write(f"• {feature}: {deviation:.3f}")
                    
                    if len(anomalies) > 5:
                        st.info(f"Showing 5 of {len(anomalies)} anomalies")
            else:
                st.info("No anomaly detection results available")
        
        with tab3:
            st.subheader("Risk Patterns")
            
            patterns = assessment.get('patterns', [])
            
            if patterns:
                st.write(f"**Identified {len(patterns)} risk patterns:**")
                
                for i, pattern in enumerate(patterns):
                    severity_icon = "🔴" if pattern.get('severity_impact', 0) > 1.5 else "🟡" if pattern.get('severity_impact', 0) > 1.0 else "🟢"
                    
                    with st.expander(f"{severity_icon} {pattern.get('pattern_type', 'Unknown')} (Impact: {pattern.get('severity_impact', 0):.2f})"):
                        st.write(f"**Description:** {pattern.get('description', 'No description')}")
                        st.write(f"**Frequency:** {pattern.get('frequency', 0):.1%}")
                        st.write(f"**Severity Impact:** {pattern.get('severity_impact', 0):.3f}")
                        
                        if pattern.get('affected_entities'):
                            st.write("**Affected Entities:**")
                            for entity in pattern.get('affected_entities', []):
                                st.write(f"• {entity}")
                        
                        if pattern.get('pattern_features'):
                            with st.expander("Show Pattern Features"):
                                st.json(pattern['pattern_features'])
            else:
                st.info("No significant patterns detected in current data")
        
        with tab4:
            st.subheader("AI-Generated Insights")
            
            insights = assessment.get('insights', [])
            
            if insights:
                for insight in insights:
                    st.write(f"💡 {insight}")
            else:
                st.info("No specific insights generated for current assessment")
        
        with tab5:
            st.subheader("Strategic Recommendations")
            
            recommendations = assessment.get('recommendations', [])
            
            if recommendations:
                for i, recommendation in enumerate(recommendations, 1):
                    st.write(f"{i}. {recommendation}")
            else:
                st.info("No specific recommendations generated")
        
        # Assessment metadata
        with st.expander("Assessment Metadata"):
            st.write(f"**Assessment Time:** {assessment.get('timestamp', 'Unknown')}")
            st.write(f"**Processing Time:** {assessment.get('processing_time', 0):.3f} seconds")
    
    # Continuous Monitoring Controls
    st.subheader("🔄 Continuous Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not ai_engine.continuous_monitoring:
            if st.button("▶️ Start Continuous Monitoring"):
                # Define a simple data source callback for demo
                def demo_data_source():
                    return pd.DataFrame({
                        'Date': [datetime.now()],
                        'Model': ['GPT-4'],
                        'Language': ['English'],
                        'Risk_Category': ['Bias'],
                        'Risk_Rate': [np.random.beta(2, 5)],
                        'Confidence': [np.random.beta(8, 2)],
                        'Sample_Size': [np.random.randint(50, 500)]
                    })
                
                ai_engine.start_continuous_monitoring(demo_data_source, interval_minutes=1)
                st.success("✅ Continuous monitoring started!")
                st.rerun()
        else:
            if st.button("⏹️ Stop Continuous Monitoring"):
                ai_engine.stop_continuous_monitoring()
                st.success("⏹️ Continuous monitoring stopped!")
                st.rerun()
    
    with col2:
        monitoring_status = "🟢 Active" if ai_engine.continuous_monitoring else "🔴 Inactive"
        st.write(f"**Status:** {monitoring_status}")
    
    with col3:
        if st.button("📊 View Assessment History"):
            if ai_engine.assessment_history:
                st.write(f"**Assessment History ({len(ai_engine.assessment_history)} records):**")
                
                history_df = pd.DataFrame([
                    {
                        'Timestamp': assessment['timestamp'],
                        'Risk Score': assessment.get('overall_risk_score', 0),
                        'Anomalies': len(assessment.get('anomaly_results', {}).get('anomalies', [])),
                        'Patterns': len(assessment.get('patterns', [])),
                        'Alerts': len(assessment.get('alerts', []))
                    }
                    for assessment in ai_engine.assessment_history[-10:]  # Last 10
                ])
                
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No assessment history available")

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize AI system
    ai_engine = AIRiskAssessmentEngine()
    
    # Generate sample historical data
    np.random.seed(42)
    historical_data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
        'Model': np.random.choice(['GPT-4', 'Claude-3', 'Gemini-Pro'], size=1000),
        'Language': np.random.choice(['English', 'Spanish', 'French', 'German'], size=1000),
        'Risk_Category': np.random.choice(['Bias', 'Toxicity', 'Misinformation', 'Privacy'], size=1000),
        'Risk_Rate': np.random.beta(2, 5, size=1000),
        'Confidence': np.random.beta(8, 2, size=1000),
        'Sample_Size': np.random.randint(50, 500, size=1000)
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(1000, size=50, replace=False)
    historical_data.loc[anomaly_indices, 'Risk_Rate'] = np.random.uniform(0.8, 1.0, size=50)
    
    print("Testing AI Risk Assessment Engine...")
    
    # Initialize system
    init_success = ai_engine.initialize(historical_data)
    print(f"✅ System initialized: {init_success}")
    
    # Generate current data
    current_data = pd.DataFrame({
        'Date': [datetime.now() - timedelta(hours=i) for i in range(24)],
        'Model': np.random.choice(['GPT-4', 'Claude-3', 'Gemini-Pro'], size=24),
        'Language': np.random.choice(['English', 'Spanish', 'French'], size=24),
        'Risk_Category': np.random.choice(['Bias', 'Toxicity', 'Misinformation'], size=24),
        'Risk_Rate': np.random.beta(2, 5, size=24),
        'Confidence': np.random.beta(8, 2, size=24),
        'Sample_Size': np.random.randint(50, 500, size=24)
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(24, size=3, replace=False)
    current_data.loc[anomaly_indices, 'Risk_Rate'] = np.random.uniform(0.9, 1.0, size=3)
    
    # Run assessment
    assessment = ai_engine.assess_current_risk(current_data)
    
    if 'error' not in assessment:
        print(f"✅ Assessment completed:")
        print(f"  - Overall Risk Score: {assessment.get('overall_risk_score', 0):.3f}")
        print(f"  - Anomalies Detected: {len(assessment.get('anomaly_results', {}).get('anomalies', []))}")
        print(f"  - Patterns Identified: {len(assessment.get('patterns', []))}")
        print(f"  - Alerts Generated: {len(assessment.get('alerts', []))}")
        print(f"  - Processing Time: {assessment.get('processing_time', 0):.3f} seconds")
        
        # Display insights
        insights = assessment.get('insights', [])
        if insights:
            print(f"\n💡 AI Insights:")
            for insight in insights[:3]:
                print(f"  - {insight}")
        
        # Display top recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n🎯 Top Recommendations:")
            for rec in recommendations[:3]:
                print(f"  - {rec}")
    else:
        print(f"❌ Assessment failed: {assessment['error']}")
    
    print("\nAI-driven automated risk detection system test completed!")