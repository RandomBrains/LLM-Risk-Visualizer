"""
Adaptive User Interface and Personalized Recommendations Module
Provides intelligent UI adaptation and personalized recommendations based on user behavior
"""

import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
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
from collections import defaultdict, Counter, deque
import hashlib
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles in the system"""
    ANALYST = "analyst"
    MANAGER = "manager"
    ADMINISTRATOR = "administrator"
    VIEWER = "viewer"
    AUDITOR = "auditor"
    EXECUTIVE = "executive"

class InteractionType(Enum):
    """Types of user interactions"""
    PAGE_VIEW = "page_view"
    CHART_INTERACTION = "chart_interaction"
    FILTER_APPLIED = "filter_applied"
    EXPORT_DATA = "export_data"
    SEARCH_QUERY = "search_query"
    CONFIGURATION_CHANGE = "configuration_change"
    REPORT_GENERATED = "report_generated"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"

class UIComponent(Enum):
    """UI components that can be adapted"""
    DASHBOARD_LAYOUT = "dashboard_layout"
    CHART_PREFERENCES = "chart_preferences"
    FILTER_PRESETS = "filter_presets"
    NAVIGATION_MENU = "navigation_menu"
    NOTIFICATION_SETTINGS = "notification_settings"
    DATA_DISPLAY = "data_display"
    TOOL_RECOMMENDATIONS = "tool_recommendations"

class PersonalizationLevel(Enum):
    """Levels of personalization"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class UserProfile:
    """User profile with behavior patterns and preferences"""
    user_id: str
    username: str
    email: str
    role: UserRole
    created_date: datetime
    last_login: datetime
    
    # Behavior patterns
    total_sessions: int = 0
    avg_session_duration_minutes: float = 0.0
    most_used_features: List[str] = None
    preferred_chart_types: List[str] = None
    common_filters: Dict[str, Any] = None
    
    # Preferences
    personalization_level: PersonalizationLevel = PersonalizationLevel.MODERATE
    ui_theme: str = "default"
    language: str = "en"
    timezone: str = "UTC"
    
    # Performance metrics
    task_completion_rate: float = 0.0
    efficiency_score: float = 0.0
    feature_adoption_rate: float = 0.0
    
    # Recommendation history
    recommendations_accepted: int = 0
    recommendations_rejected: int = 0
    
    def __post_init__(self):
        if self.most_used_features is None:
            self.most_used_features = []
        if self.preferred_chart_types is None:
            self.preferred_chart_types = []
        if self.common_filters is None:
            self.common_filters = {}

@dataclass
class UserInteraction:
    """Individual user interaction event"""
    interaction_id: str
    user_id: str
    timestamp: datetime
    interaction_type: InteractionType
    component: str
    details: Dict[str, Any]
    
    # Context information
    session_id: str
    page_path: str
    duration_seconds: float = 0.0
    
    # Interaction outcomes
    successful: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.details:
            self.details = {}

@dataclass
class UIAdaptation:
    """UI adaptation configuration"""
    adaptation_id: str
    user_id: str
    component: UIComponent
    adaptation_data: Dict[str, Any]
    created_date: datetime
    
    # Adaptation metadata
    confidence_score: float = 0.0
    a_b_test_group: Optional[str] = None
    performance_impact: Optional[float] = None
    user_feedback: Optional[str] = None
    
    # Status
    active: bool = True
    effectiveness_score: float = 0.0

@dataclass 
class Recommendation:
    """Personalized recommendation"""
    rec_id: str
    user_id: str
    rec_type: str
    title: str
    description: str
    action_data: Dict[str, Any]
    
    # Relevance scoring
    relevance_score: float
    confidence_score: float
    priority: str = "medium"  # low, medium, high, critical
    
    # Lifecycle management
    created_date: datetime
    expires_date: Optional[datetime] = None
    shown_count: int = 0
    clicked_count: int = 0
    
    # Status
    status: str = "pending"  # pending, shown, accepted, rejected, expired
    user_feedback: Optional[str] = None

class BehaviorAnalyzer:
    """Analyzes user behavior patterns"""
    
    def __init__(self):
        self.interaction_buffer = deque(maxlen=10000)
        self.user_clusters = {}
        self.behavior_models = {}
        
    def record_interaction(self, interaction: UserInteraction):
        """Record user interaction"""
        self.interaction_buffer.append(interaction)
        logger.debug(f"Recorded interaction: {interaction.interaction_type.value} for user {interaction.user_id}")
        
    def analyze_user_behavior(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze behavior patterns for a specific user"""
        
        # Filter interactions for the user and time period
        cutoff_date = datetime.now() - timedelta(days=days)
        user_interactions = [
            interaction for interaction in self.interaction_buffer
            if interaction.user_id == user_id and interaction.timestamp >= cutoff_date
        ]
        
        if not user_interactions:
            return {"message": "No interactions found for analysis"}
        
        # Analyze interaction patterns
        interaction_counts = Counter([i.interaction_type.value for i in user_interactions])
        component_usage = Counter([i.component for i in user_interactions])
        
        # Calculate session patterns
        sessions = defaultdict(list)
        for interaction in user_interactions:
            sessions[interaction.session_id].append(interaction)
        
        session_durations = []
        for session_interactions in sessions.values():
            if len(session_interactions) > 1:
                start_time = min(i.timestamp for i in session_interactions)
                end_time = max(i.timestamp for i in session_interactions)
                duration = (end_time - start_time).total_seconds() / 60  # minutes
                session_durations.append(duration)
        
        # Identify usage patterns
        peak_hours = [i.timestamp.hour for i in user_interactions]
        peak_hour_distribution = Counter(peak_hours)
        
        # Calculate efficiency metrics
        successful_interactions = len([i for i in user_interactions if i.successful])
        efficiency_rate = successful_interactions / len(user_interactions) if user_interactions else 0
        
        # Feature adoption analysis
        unique_components = set([i.component for i in user_interactions])
        total_available_components = 20  # Assumed total components
        adoption_rate = len(unique_components) / total_available_components
        
        return {
            "analysis_period_days": days,
            "total_interactions": len(user_interactions),
            "unique_sessions": len(sessions),
            "avg_session_duration_minutes": np.mean(session_durations) if session_durations else 0,
            "interaction_types": dict(interaction_counts),
            "component_usage": dict(component_usage),
            "peak_hours": dict(peak_hour_distribution.most_common(3)),
            "efficiency_rate": efficiency_rate,
            "feature_adoption_rate": adoption_rate,
            "most_used_component": component_usage.most_common(1)[0][0] if component_usage else None
        }
    
    def cluster_users(self, users: List[UserProfile]) -> Dict[str, List[str]]:
        """Cluster users based on behavior patterns"""
        
        if len(users) < 3:
            return {"default": [user.user_id for user in users]}
        
        # Create feature matrix
        features = []
        user_ids = []
        
        for user in users:
            # Extract behavioral features
            feature_vector = [
                user.avg_session_duration_minutes,
                user.total_sessions,
                user.task_completion_rate,
                user.efficiency_score,
                user.feature_adoption_rate,
                len(user.most_used_features),
                len(user.preferred_chart_types)
            ]
            
            features.append(feature_vector)
            user_ids.append(user.user_id)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Determine optimal number of clusters (max 5)
        n_clusters = min(5, max(2, len(users) // 3))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Group users by cluster
        clusters = defaultdict(list)
        for user_id, cluster_label in zip(user_ids, cluster_labels):
            clusters[f"cluster_{cluster_label}"].append(user_id)
        
        # Store cluster information
        self.user_clusters = dict(clusters)
        
        return self.user_clusters
    
    def predict_next_action(self, user_id: str) -> Dict[str, Any]:
        """Predict user's next likely action"""
        
        # Get recent interactions
        recent_interactions = [
            i for i in list(self.interaction_buffer)[-100:]
            if i.user_id == user_id
        ]
        
        if len(recent_interactions) < 3:
            return {"prediction": "unknown", "confidence": 0.0}
        
        # Analyze recent interaction sequence
        recent_components = [i.component for i in recent_interactions[-5:]]
        recent_types = [i.interaction_type.value for i in recent_interactions[-5:]]
        
        # Simple pattern-based prediction
        component_transitions = defaultdict(Counter)
        type_transitions = defaultdict(Counter)
        
        # Build transition matrices
        for i in range(len(recent_interactions) - 1):
            current_comp = recent_interactions[i].component
            next_comp = recent_interactions[i + 1].component
            component_transitions[current_comp][next_comp] += 1
            
            current_type = recent_interactions[i].interaction_type.value
            next_type = recent_interactions[i + 1].interaction_type.value
            type_transitions[current_type][next_type] += 1
        
        # Predict next component
        if recent_components:
            last_component = recent_components[-1]
            if last_component in component_transitions:
                next_component_counter = component_transitions[last_component]
                if next_component_counter:
                    predicted_component = next_component_counter.most_common(1)[0][0]
                    confidence = next_component_counter[predicted_component] / sum(next_component_counter.values())
                else:
                    predicted_component = "dashboard"
                    confidence = 0.3
            else:
                predicted_component = "dashboard"
                confidence = 0.3
        else:
            predicted_component = "dashboard"
            confidence = 0.3
        
        # Predict next interaction type
        if recent_types:
            last_type = recent_types[-1]
            if last_type in type_transitions:
                next_type_counter = type_transitions[last_type]
                if next_type_counter:
                    predicted_type = next_type_counter.most_common(1)[0][0]
                else:
                    predicted_type = "page_view"
            else:
                predicted_type = "page_view"
        else:
            predicted_type = "page_view"
        
        return {
            "predicted_component": predicted_component,
            "predicted_interaction_type": predicted_type,
            "confidence": confidence
        }

class UIAdaptationEngine:
    """Adapts UI based on user behavior"""
    
    def __init__(self):
        self.adaptations: Dict[str, List[UIAdaptation]] = defaultdict(list)
        self.a_b_tests: Dict[str, Dict[str, Any]] = {}
        
    def generate_adaptations(self, user_profile: UserProfile, 
                           behavior_analysis: Dict[str, Any]) -> List[UIAdaptation]:
        """Generate UI adaptations based on user profile and behavior"""
        
        adaptations = []
        
        # Dashboard layout adaptation
        if behavior_analysis.get("most_used_component"):
            most_used = behavior_analysis["most_used_component"]
            
            dashboard_adaptation = UIAdaptation(
                adaptation_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                component=UIComponent.DASHBOARD_LAYOUT,
                adaptation_data={
                    "prioritize_component": most_used,
                    "layout_type": "component_focused",
                    "quick_access_items": behavior_analysis.get("component_usage", {})
                },
                created_date=datetime.now(),
                confidence_score=0.8
            )
            adaptations.append(dashboard_adaptation)
        
        # Chart preferences adaptation
        if user_profile.preferred_chart_types:
            chart_adaptation = UIAdaptation(
                adaptation_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                component=UIComponent.CHART_PREFERENCES,
                adaptation_data={
                    "default_chart_types": user_profile.preferred_chart_types[:3],
                    "chart_suggestions": user_profile.preferred_chart_types,
                    "auto_select_chart": True
                },
                created_date=datetime.now(),
                confidence_score=0.9
            )
            adaptations.append(chart_adaptation)
        
        # Navigation menu adaptation based on role
        nav_items = self._get_role_based_navigation(user_profile.role)
        nav_adaptation = UIAdaptation(
            adaptation_id=str(uuid.uuid4()),
            user_id=user_profile.user_id,
            component=UIComponent.NAVIGATION_MENU,
            adaptation_data={
                "menu_items": nav_items,
                "menu_order": nav_items,
                "quick_actions": nav_items[:3]
            },
            created_date=datetime.now(),
            confidence_score=0.7
        )
        adaptations.append(nav_adaptation)
        
        # Filter presets based on common filters
        if user_profile.common_filters:
            filter_adaptation = UIAdaptation(
                adaptation_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                component=UIComponent.FILTER_PRESETS,
                adaptation_data={
                    "preset_filters": user_profile.common_filters,
                    "auto_apply_defaults": True,
                    "suggested_combinations": self._generate_filter_combinations(user_profile.common_filters)
                },
                created_date=datetime.now(),
                confidence_score=0.6
            )
            adaptations.append(filter_adaptation)
        
        # Notification settings based on role and behavior
        notification_frequency = self._determine_notification_frequency(
            user_profile.role, behavior_analysis
        )
        
        notification_adaptation = UIAdaptation(
            adaptation_id=str(uuid.uuid4()),
            user_id=user_profile.user_id,
            component=UIComponent.NOTIFICATION_SETTINGS,
            adaptation_data={
                "frequency": notification_frequency,
                "priority_threshold": self._get_priority_threshold(user_profile.role),
                "channels": self._get_preferred_channels(user_profile.role)
            },
            created_date=datetime.now(),
            confidence_score=0.8
        )
        adaptations.append(notification_adaptation)
        
        # Store adaptations
        self.adaptations[user_profile.user_id].extend(adaptations)
        
        return adaptations
    
    def _get_role_based_navigation(self, role: UserRole) -> List[str]:
        """Get navigation items based on user role"""
        
        role_navigation = {
            UserRole.ANALYST: [
                "Risk Analysis", "Data Exploration", "Visualizations", 
                "Statistical Analysis", "Reports", "Export"
            ],
            UserRole.MANAGER: [
                "Executive Dashboard", "Risk Summary", "Team Performance", 
                "Compliance Status", "Reports", "Notifications"
            ],
            UserRole.ADMINISTRATOR: [
                "System Management", "User Management", "Configuration", 
                "Security", "Audit Logs", "Performance"
            ],
            UserRole.VIEWER: [
                "Dashboard", "Reports", "Visualizations", "Export"
            ],
            UserRole.AUDITOR: [
                "Audit Trail", "Compliance Reports", "Risk Assessment", 
                "Documentation", "Export"
            ],
            UserRole.EXECUTIVE: [
                "Executive Summary", "KPI Dashboard", "Strategic Insights", 
                "Risk Overview", "Reports"
            ]
        }
        
        return role_navigation.get(role, ["Dashboard", "Reports", "Settings"])
    
    def _generate_filter_combinations(self, common_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggested filter combinations"""
        
        combinations = []
        
        # Create combinations of the most common filters
        filter_items = list(common_filters.items())
        
        if len(filter_items) >= 2:
            # Two-filter combinations
            for i in range(len(filter_items)):
                for j in range(i + 1, len(filter_items)):
                    combination = {
                        filter_items[i][0]: filter_items[i][1],
                        filter_items[j][0]: filter_items[j][1]
                    }
                    combinations.append(combination)
        
        return combinations[:5]  # Limit to 5 combinations
    
    def _determine_notification_frequency(self, role: UserRole, 
                                        behavior_analysis: Dict[str, Any]) -> str:
        """Determine optimal notification frequency"""
        
        # Base frequency on role
        role_frequencies = {
            UserRole.EXECUTIVE: "daily",
            UserRole.MANAGER: "hourly", 
            UserRole.ANALYST: "real_time",
            UserRole.AUDITOR: "daily",
            UserRole.ADMINISTRATOR: "real_time",
            UserRole.VIEWER: "weekly"
        }
        
        base_frequency = role_frequencies.get(role, "daily")
        
        # Adjust based on behavior
        avg_session_duration = behavior_analysis.get("avg_session_duration_minutes", 30)
        
        if avg_session_duration > 60:  # Long sessions suggest engagement
            if base_frequency == "weekly":
                return "daily"
            elif base_frequency == "daily":
                return "hourly"
        elif avg_session_duration < 15:  # Short sessions suggest less engagement
            if base_frequency == "real_time":
                return "hourly"
            elif base_frequency == "hourly":
                return "daily"
        
        return base_frequency
    
    def _get_priority_threshold(self, role: UserRole) -> str:
        """Get priority threshold for notifications"""
        
        priority_thresholds = {
            UserRole.EXECUTIVE: "high",
            UserRole.MANAGER: "medium",
            UserRole.ANALYST: "low",
            UserRole.AUDITOR: "medium",
            UserRole.ADMINISTRATOR: "medium",
            UserRole.VIEWER: "high"
        }
        
        return priority_thresholds.get(role, "medium")
    
    def _get_preferred_channels(self, role: UserRole) -> List[str]:
        """Get preferred notification channels"""
        
        role_channels = {
            UserRole.EXECUTIVE: ["email", "dashboard"],
            UserRole.MANAGER: ["email", "dashboard", "mobile"],
            UserRole.ANALYST: ["dashboard", "email"],
            UserRole.AUDITOR: ["email", "dashboard"],
            UserRole.ADMINISTRATOR: ["dashboard", "email", "sms"],
            UserRole.VIEWER: ["dashboard"]
        }
        
        return role_channels.get(role, ["dashboard", "email"])
    
    def apply_adaptation(self, user_id: str, component: UIComponent) -> Dict[str, Any]:
        """Apply UI adaptation for user and component"""
        
        user_adaptations = self.adaptations.get(user_id, [])
        
        # Find active adaptation for the component
        for adaptation in user_adaptations:
            if adaptation.component == component and adaptation.active:
                return adaptation.adaptation_data
        
        # Return default configuration if no adaptation found
        return self._get_default_configuration(component)
    
    def _get_default_configuration(self, component: UIComponent) -> Dict[str, Any]:
        """Get default configuration for UI component"""
        
        defaults = {
            UIComponent.DASHBOARD_LAYOUT: {
                "layout_type": "standard",
                "columns": 2,
                "show_sidebar": True
            },
            UIComponent.CHART_PREFERENCES: {
                "default_chart_types": ["bar", "line", "pie"],
                "auto_select_chart": False
            },
            UIComponent.NAVIGATION_MENU: {
                "menu_items": ["Dashboard", "Analysis", "Reports", "Settings"],
                "show_icons": True
            },
            UIComponent.FILTER_PRESETS: {
                "preset_filters": {},
                "auto_apply_defaults": False
            },
            UIComponent.NOTIFICATION_SETTINGS: {
                "frequency": "daily",
                "priority_threshold": "medium",
                "channels": ["dashboard", "email"]
            }
        }
        
        return defaults.get(component, {})

class RecommendationEngine:
    """Generates personalized recommendations"""
    
    def __init__(self):
        self.recommendation_history: Dict[str, List[Recommendation]] = defaultdict(list)
        self.content_similarity_matrix = {}
        self.collaborative_filters = {}
        
    def generate_recommendations(self, user_profile: UserProfile, 
                               behavior_analysis: Dict[str, Any],
                               context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        # Feature discovery recommendations
        feature_recs = self._generate_feature_recommendations(user_profile, behavior_analysis)
        recommendations.extend(feature_recs)
        
        # Optimization recommendations
        optimization_recs = self._generate_optimization_recommendations(user_profile, behavior_analysis)
        recommendations.extend(optimization_recs)
        
        # Content recommendations
        content_recs = self._generate_content_recommendations(user_profile, behavior_analysis)
        recommendations.extend(content_recs)
        
        # Workflow recommendations
        workflow_recs = self._generate_workflow_recommendations(user_profile, behavior_analysis)
        recommendations.extend(workflow_recs)
        
        # Learning recommendations
        learning_recs = self._generate_learning_recommendations(user_profile, behavior_analysis)
        recommendations.extend(learning_recs)
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Store recommendations
        self.recommendation_history[user_profile.user_id].extend(recommendations)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _generate_feature_recommendations(self, user_profile: UserProfile, 
                                        behavior_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Generate feature discovery recommendations"""
        
        recommendations = []
        
        # Identify unused features
        adoption_rate = behavior_analysis.get("feature_adoption_rate", 0)
        
        if adoption_rate < 0.5:  # Low feature adoption
            unused_features = [
                "Advanced Visualizations", "Custom Filters", "Automated Reports",
                "Data Export", "Collaboration Tools", "API Integration"
            ]
            
            for feature in unused_features[:3]:
                rec = Recommendation(
                    rec_id=str(uuid.uuid4()),
                    user_id=user_profile.user_id,
                    rec_type="feature_discovery",
                    title=f"Explore {feature}",
                    description=f"Try {feature} to enhance your workflow efficiency. This feature can help you accomplish tasks more quickly.",
                    action_data={"feature": feature, "action": "explore"},
                    relevance_score=0.7,
                    confidence_score=0.8,
                    priority="medium",
                    created_date=datetime.now()
                )
                recommendations.append(rec)
        
        # Role-specific feature recommendations
        role_features = {
            UserRole.ANALYST: ["Statistical Analysis", "Data Mining", "Predictive Models"],
            UserRole.MANAGER: ["Executive Dashboard", "Team Analytics", "Performance Metrics"],
            UserRole.AUDITOR: ["Compliance Reports", "Audit Trail", "Risk Documentation"]
        }
        
        if user_profile.role in role_features:
            for feature in role_features[user_profile.role][:2]:
                rec = Recommendation(
                    rec_id=str(uuid.uuid4()),
                    user_id=user_profile.user_id,
                    rec_type="role_based_feature",
                    title=f"Try {feature} for {user_profile.role.value.title()}s",
                    description=f"{feature} is designed specifically for users in your role and can significantly improve your productivity.",
                    action_data={"feature": feature, "role": user_profile.role.value},
                    relevance_score=0.9,
                    confidence_score=0.9,
                    priority="high",
                    created_date=datetime.now()
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_optimization_recommendations(self, user_profile: UserProfile, 
                                            behavior_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Generate workflow optimization recommendations"""
        
        recommendations = []
        
        # Efficiency improvement recommendations
        efficiency_score = user_profile.efficiency_score
        
        if efficiency_score < 0.7:
            # Suggest shortcuts and automation
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="efficiency_improvement",
                title="Set Up Filter Presets",
                description="Create filter presets for your most common searches to save time and reduce repetitive tasks.",
                action_data={"action": "setup_presets", "component": "filters"},
                relevance_score=0.8,
                confidence_score=0.7,
                priority="medium",
                created_date=datetime.now()
            )
            recommendations.append(rec)
            
            # Suggest keyboard shortcuts
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="efficiency_improvement",
                title="Learn Keyboard Shortcuts",
                description="Master keyboard shortcuts to navigate faster and increase your productivity by up to 40%.",
                action_data={"action": "learn_shortcuts", "resource": "help_guide"},
                relevance_score=0.6,
                confidence_score=0.8,
                priority="low",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        # Session duration optimization
        avg_session = behavior_analysis.get("avg_session_duration_minutes", 30)
        
        if avg_session > 90:  # Very long sessions
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="session_optimization",
                title="Break Up Long Sessions",
                description="Consider taking breaks during long analysis sessions to maintain focus and productivity.",
                action_data={"action": "session_management", "suggestion": "break_reminders"},
                relevance_score=0.5,
                confidence_score=0.6,
                priority="low",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        elif avg_session < 10:  # Very short sessions
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="session_optimization",
                title="Explore More Features",
                description="You seem to have quick sessions. Consider exploring additional features that might be valuable for your work.",
                action_data={"action": "feature_exploration", "suggestion": "guided_tour"},
                relevance_score=0.7,
                confidence_score=0.6,
                priority="medium",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_content_recommendations(self, user_profile: UserProfile, 
                                        behavior_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Generate content and data recommendations"""
        
        recommendations = []
        
        # Data source recommendations
        most_used_component = behavior_analysis.get("most_used_component")
        
        if most_used_component == "data_exploration":
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="data_source",
                title="Connect Additional Data Sources",
                description="Expand your analysis by connecting additional data sources to get more comprehensive insights.",
                action_data={"action": "connect_data", "component": "integrations"},
                relevance_score=0.7,
                confidence_score=0.7,
                priority="medium",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        # Report template recommendations
        if user_profile.role in [UserRole.MANAGER, UserRole.EXECUTIVE]:
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="report_template",
                title="Use Executive Report Templates",
                description="Save time with pre-built executive report templates tailored for leadership decision-making.",
                action_data={"action": "use_template", "template_type": "executive"},
                relevance_score=0.8,
                confidence_score=0.9,
                priority="high",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_workflow_recommendations(self, user_profile: UserProfile, 
                                         behavior_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Generate workflow improvement recommendations"""
        
        recommendations = []
        
        # Automation recommendations
        interaction_types = behavior_analysis.get("interaction_types", {})
        
        if interaction_types.get("export_data", 0) > 5:  # Frequent exports
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="automation",
                title="Schedule Automated Reports",
                description="Set up automated reports to receive regular data exports without manual intervention.",
                action_data={"action": "setup_automation", "type": "scheduled_reports"},
                relevance_score=0.9,
                confidence_score=0.8,
                priority="high",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        if interaction_types.get("filter_applied", 0) > 10:  # Frequent filtering
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="workflow_optimization",
                title="Create Custom Views",
                description="Save your frequently used filter combinations as custom views for faster access.",
                action_data={"action": "create_views", "component": "filters"},
                relevance_score=0.8,
                confidence_score=0.7,
                priority="medium",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_learning_recommendations(self, user_profile: UserProfile, 
                                         behavior_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Generate learning and skill development recommendations"""
        
        recommendations = []
        
        # Skill development based on role
        role_skills = {
            UserRole.ANALYST: ["Advanced Analytics", "Machine Learning", "Statistical Modeling"],
            UserRole.MANAGER: ["Data Interpretation", "Dashboard Design", "KPI Management"],
            UserRole.ADMINISTRATOR: ["System Optimization", "Security Best Practices", "User Management"]
        }
        
        if user_profile.role in role_skills:
            for skill in role_skills[user_profile.role][:1]:
                rec = Recommendation(
                    rec_id=str(uuid.uuid4()),
                    user_id=user_profile.user_id,
                    rec_type="skill_development",
                    title=f"Learn {skill}",
                    description=f"Enhance your expertise in {skill} to become more effective in your role.",
                    action_data={"action": "learn_skill", "skill": skill, "resource": "training_module"},
                    relevance_score=0.6,
                    confidence_score=0.7,
                    priority="low",
                    created_date=datetime.now()
                )
                recommendations.append(rec)
        
        # Feature adoption learning
        adoption_rate = behavior_analysis.get("feature_adoption_rate", 0)
        
        if adoption_rate < 0.3:
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                rec_type="feature_learning",
                title="Take the Guided Platform Tour",
                description="Discover all available features with our interactive guided tour to maximize your platform usage.",
                action_data={"action": "guided_tour", "component": "onboarding"},
                relevance_score=0.8,
                confidence_score=0.9,
                priority="medium",
                created_date=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations
    
    def get_recommendation_feedback(self, rec_id: str, user_id: str, 
                                  feedback: str) -> bool:
        """Record user feedback on recommendation"""
        
        user_recommendations = self.recommendation_history.get(user_id, [])
        
        for rec in user_recommendations:
            if rec.rec_id == rec_id:
                rec.user_feedback = feedback
                
                if feedback in ["accepted", "helpful"]:
                    rec.status = "accepted"
                elif feedback in ["rejected", "not_helpful"]:
                    rec.status = "rejected"
                
                logger.info(f"Recorded feedback '{feedback}' for recommendation {rec_id}")
                return True
        
        return False
    
    def get_recommendation_performance(self, user_id: str) -> Dict[str, Any]:
        """Get recommendation performance metrics"""
        
        user_recommendations = self.recommendation_history.get(user_id, [])
        
        if not user_recommendations:
            return {"message": "No recommendations found"}
        
        total_recs = len(user_recommendations)
        shown_recs = len([r for r in user_recommendations if r.shown_count > 0])
        clicked_recs = len([r for r in user_recommendations if r.clicked_count > 0])
        accepted_recs = len([r for r in user_recommendations if r.status == "accepted"])
        
        return {
            "total_recommendations": total_recs,
            "show_rate": shown_recs / total_recs if total_recs > 0 else 0,
            "click_rate": clicked_recs / shown_recs if shown_recs > 0 else 0,
            "acceptance_rate": accepted_recs / shown_recs if shown_recs > 0 else 0,
            "most_common_type": Counter([r.rec_type for r in user_recommendations]).most_common(1)[0][0] if user_recommendations else None
        }

class PersonalizationManager:
    """Main personalization system manager"""
    
    def __init__(self, db_path: str = "personalization.db"):
        self.db_path = Path(db_path)
        self.behavior_analyzer = BehaviorAnalyzer()
        self.ui_adaptation_engine = UIAdaptationEngine()
        self.recommendation_engine = RecommendationEngine()
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Initialize database
        self.init_database()
        
        # Load existing data
        self.load_user_profiles()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT,
                role TEXT NOT NULL,
                created_date TEXT NOT NULL,
                last_login TEXT NOT NULL,
                total_sessions INTEGER DEFAULT 0,
                avg_session_duration_minutes REAL DEFAULT 0.0,
                most_used_features TEXT,
                preferred_chart_types TEXT,
                common_filters TEXT,
                personalization_level TEXT DEFAULT 'moderate',
                ui_theme TEXT DEFAULT 'default',
                language TEXT DEFAULT 'en',
                timezone TEXT DEFAULT 'UTC',
                task_completion_rate REAL DEFAULT 0.0,
                efficiency_score REAL DEFAULT 0.0,
                feature_adoption_rate REAL DEFAULT 0.0,
                recommendations_accepted INTEGER DEFAULT 0,
                recommendations_rejected INTEGER DEFAULT 0
            )
        ''')
        
        # User interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                interaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                component TEXT NOT NULL,
                details TEXT,
                session_id TEXT,
                page_path TEXT,
                duration_seconds REAL DEFAULT 0.0,
                successful BOOLEAN DEFAULT 1,
                error_message TEXT
            )
        ''')
        
        # UI adaptations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ui_adaptations (
                adaptation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                component TEXT NOT NULL,
                adaptation_data TEXT NOT NULL,
                created_date TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                a_b_test_group TEXT,
                performance_impact REAL,
                user_feedback TEXT,
                active BOOLEAN DEFAULT 1,
                effectiveness_score REAL DEFAULT 0.0
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                rec_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                rec_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                action_data TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                confidence_score REAL NOT NULL,
                priority TEXT DEFAULT 'medium',
                created_date TEXT NOT NULL,
                expires_date TEXT,
                shown_count INTEGER DEFAULT 0,
                clicked_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                user_feedback TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_user_profiles(self):
        """Load user profiles from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM user_profiles')
            rows = cursor.fetchall()
            
            for row in rows:
                profile = self._row_to_user_profile(row)
                if profile:
                    self.user_profiles[profile.user_id] = profile
            
            conn.close()
            logger.info(f"Loaded {len(rows)} user profiles from database")
            
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
    
    def _row_to_user_profile(self, row) -> Optional[UserProfile]:
        """Convert database row to UserProfile object"""
        try:
            return UserProfile(
                user_id=row[0],
                username=row[1],
                email=row[2] or "",
                role=UserRole(row[3]),
                created_date=datetime.fromisoformat(row[4]),
                last_login=datetime.fromisoformat(row[5]),
                total_sessions=row[6] or 0,
                avg_session_duration_minutes=row[7] or 0.0,
                most_used_features=json.loads(row[8]) if row[8] else [],
                preferred_chart_types=json.loads(row[9]) if row[9] else [],
                common_filters=json.loads(row[10]) if row[10] else {},
                personalization_level=PersonalizationLevel(row[11]),
                ui_theme=row[12] or "default",
                language=row[13] or "en",
                timezone=row[14] or "UTC",
                task_completion_rate=row[15] or 0.0,
                efficiency_score=row[16] or 0.0,
                feature_adoption_rate=row[17] or 0.0,
                recommendations_accepted=row[18] or 0,
                recommendations_rejected=row[19] or 0
            )
        except Exception as e:
            logger.error(f"Error converting row to user profile: {e}")
            return None
    
    def get_or_create_user_profile(self, user_id: str, username: str, 
                                  email: str = "", role: UserRole = UserRole.VIEWER) -> UserProfile:
        """Get existing user profile or create new one"""
        
        if user_id in self.user_profiles:
            # Update last login
            self.user_profiles[user_id].last_login = datetime.now()
            return self.user_profiles[user_id]
        
        # Create new profile
        profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_date=datetime.now(),
            last_login=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        self._save_user_profile(profile)
        
        logger.info(f"Created new user profile: {user_id}")
        return profile
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles
                (user_id, username, email, role, created_date, last_login,
                 total_sessions, avg_session_duration_minutes, most_used_features,
                 preferred_chart_types, common_filters, personalization_level,
                 ui_theme, language, timezone, task_completion_rate,
                 efficiency_score, feature_adoption_rate, recommendations_accepted,
                 recommendations_rejected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.user_id, profile.username, profile.email, profile.role.value,
                profile.created_date.isoformat(), profile.last_login.isoformat(),
                profile.total_sessions, profile.avg_session_duration_minutes,
                json.dumps(profile.most_used_features), json.dumps(profile.preferred_chart_types),
                json.dumps(profile.common_filters), profile.personalization_level.value,
                profile.ui_theme, profile.language, profile.timezone,
                profile.task_completion_rate, profile.efficiency_score,
                profile.feature_adoption_rate, profile.recommendations_accepted,
                profile.recommendations_rejected
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
    
    def record_interaction(self, user_id: str, interaction_type: InteractionType,
                          component: str, details: Dict[str, Any] = None,
                          session_id: str = None, page_path: str = "",
                          duration_seconds: float = 0.0) -> bool:
        """Record user interaction"""
        
        try:
            interaction = UserInteraction(
                interaction_id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=datetime.now(),
                interaction_type=interaction_type,
                component=component,
                details=details or {},
                session_id=session_id or str(uuid.uuid4()),
                page_path=page_path,
                duration_seconds=duration_seconds
            )
            
            # Record in behavior analyzer
            self.behavior_analyzer.record_interaction(interaction)
            
            # Save to database
            self._save_interaction(interaction)
            
            # Update user profile metrics
            self.update_user_metrics(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            return False
    
    def _save_interaction(self, interaction: UserInteraction):
        """Save interaction to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_interactions
                (interaction_id, user_id, timestamp, interaction_type, component,
                 details, session_id, page_path, duration_seconds, successful, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction.interaction_id, interaction.user_id,
                interaction.timestamp.isoformat(), interaction.interaction_type.value,
                interaction.component, json.dumps(interaction.details),
                interaction.session_id, interaction.page_path,
                interaction.duration_seconds, interaction.successful,
                interaction.error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
    
    def update_user_metrics(self, user_id: str):
        """Update user profile metrics based on recent interactions"""
        
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Get behavior analysis
        behavior_analysis = self.behavior_analyzer.analyze_user_behavior(user_id, days=30)
        
        # Update profile metrics
        profile.avg_session_duration_minutes = behavior_analysis.get("avg_session_duration_minutes", 0)
        profile.efficiency_score = behavior_analysis.get("efficiency_rate", 0)
        profile.feature_adoption_rate = behavior_analysis.get("feature_adoption_rate", 0)
        
        # Update component usage
        component_usage = behavior_analysis.get("component_usage", {})
        if component_usage:
            profile.most_used_features = list(Counter(component_usage).most_common(10))
        
        # Save updated profile
        self._save_user_profile(profile)
    
    def get_personalized_experience(self, user_id: str) -> Dict[str, Any]:
        """Get complete personalized experience for user"""
        
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        profile = self.user_profiles[user_id]
        
        # Get behavior analysis
        behavior_analysis = self.behavior_analyzer.analyze_user_behavior(user_id)
        
        # Generate UI adaptations
        adaptations = self.ui_adaptation_engine.generate_adaptations(profile, behavior_analysis)
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            profile, behavior_analysis
        )
        
        # Predict next action
        next_action = self.behavior_analyzer.predict_next_action(user_id)
        
        return {
            "user_profile": asdict(profile),
            "behavior_analysis": behavior_analysis,
            "ui_adaptations": [asdict(a) for a in adaptations],
            "recommendations": [asdict(r) for r in recommendations],
            "next_action_prediction": next_action,
            "personalization_summary": {
                "total_adaptations": len(adaptations),
                "total_recommendations": len(recommendations),
                "personalization_level": profile.personalization_level.value,
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def apply_ui_adaptation(self, user_id: str, component: UIComponent) -> Dict[str, Any]:
        """Apply UI adaptation for specific component"""
        return self.ui_adaptation_engine.apply_adaptation(user_id, component)
    
    def get_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get current recommendations for user"""
        
        user_recommendations = self.recommendation_engine.recommendation_history.get(user_id, [])
        
        # Filter active recommendations
        active_recs = [
            r for r in user_recommendations 
            if r.status == "pending" and (not r.expires_date or r.expires_date > datetime.now())
        ]
        
        # Sort by relevance and return top recommendations
        active_recs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return [asdict(rec) for rec in active_recs[:limit]]
    
    def record_recommendation_interaction(self, user_id: str, rec_id: str, 
                                        action: str) -> bool:
        """Record interaction with recommendation"""
        
        user_recommendations = self.recommendation_engine.recommendation_history.get(user_id, [])
        
        for rec in user_recommendations:
            if rec.rec_id == rec_id:
                if action == "shown":
                    rec.shown_count += 1
                elif action == "clicked":
                    rec.clicked_count += 1
                elif action in ["accepted", "rejected"]:
                    rec.status = action
                    
                    # Update user profile statistics
                    if user_id in self.user_profiles:
                        profile = self.user_profiles[user_id]
                        if action == "accepted":
                            profile.recommendations_accepted += 1
                        else:
                            profile.recommendations_rejected += 1
                        self._save_user_profile(profile)
                
                return True
        
        return False
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide personalization analytics"""
        
        total_users = len(self.user_profiles)
        active_users = len([p for p in self.user_profiles.values() 
                           if (datetime.now() - p.last_login).days <= 7])
        
        # Role distribution
        role_distribution = Counter([p.role.value for p in self.user_profiles.values()])
        
        # Personalization level distribution
        personalization_distribution = Counter([
            p.personalization_level.value for p in self.user_profiles.values()
        ])
        
        # Average metrics
        avg_efficiency = np.mean([p.efficiency_score for p in self.user_profiles.values()]) if self.user_profiles else 0
        avg_adoption = np.mean([p.feature_adoption_rate for p in self.user_profiles.values()]) if self.user_profiles else 0
        
        # Recommendation performance
        total_recommendations = sum(
            len(self.recommendation_engine.recommendation_history.get(uid, []))
            for uid in self.user_profiles.keys()
        )
        
        total_accepted = sum(p.recommendations_accepted for p in self.user_profiles.values())
        acceptance_rate = total_accepted / max(1, total_recommendations)
        
        return {
            "user_statistics": {
                "total_users": total_users,
                "active_users": active_users,
                "role_distribution": dict(role_distribution),
                "personalization_distribution": dict(personalization_distribution)
            },
            "performance_metrics": {
                "avg_efficiency_score": avg_efficiency,
                "avg_feature_adoption": avg_adoption,
                "recommendation_acceptance_rate": acceptance_rate
            },
            "system_health": {
                "active_user_ratio": active_users / max(1, total_users),
                "high_efficiency_users": len([p for p in self.user_profiles.values() if p.efficiency_score > 0.8]),
                "low_adoption_users": len([p for p in self.user_profiles.values() if p.feature_adoption_rate < 0.3])
            }
        }

# Streamlit Integration Functions

def initialize_personalization_system():
    """Initialize personalization system"""
    if 'personalization_manager' not in st.session_state:
        st.session_state.personalization_manager = PersonalizationManager()
        
        # Create or get current user profile
        user_id = "streamlit_user"
        username = "Streamlit User"
        st.session_state.current_user_profile = st.session_state.personalization_manager.get_or_create_user_profile(
            user_id, username, role=UserRole.ANALYST
        )
    
    return st.session_state.personalization_manager

def render_personalization_dashboard():
    """Render adaptive UI and personalization dashboard"""
    st.header("🎯 Adaptive UI & Personalized Recommendations")
    
    personalization_manager = initialize_personalization_system()
    current_user = st.session_state.current_user_profile
    
    # Record page view interaction
    personalization_manager.record_interaction(
        current_user.user_id,
        InteractionType.PAGE_VIEW,
        "personalization_dashboard",
        {"page": "personalization"}
    )
    
    # Get personalized experience
    personalized_experience = personalization_manager.get_personalized_experience(current_user.user_id)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        efficiency = current_user.efficiency_score
        st.metric("Efficiency Score", f"{efficiency:.2f}")
    
    with col2:
        adoption = current_user.feature_adoption_rate
        st.metric("Feature Adoption", f"{adoption:.1%}")
    
    with col3:
        total_recs = len(personalized_experience.get("recommendations", []))
        st.metric("Active Recommendations", total_recs)
    
    with col4:
        adaptations = len(personalized_experience.get("ui_adaptations", []))
        st.metric("UI Adaptations", adaptations)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👤 User Profile",
        "🤖 Recommendations", 
        "🎨 UI Adaptations",
        "📊 Behavior Analysis",
        "🔮 Predictions",
        "📈 Analytics"
    ])
    
    with tab1:
        st.subheader("User Profile & Preferences")
        
        # User information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"• **User ID:** {current_user.user_id}")
            st.write(f"• **Username:** {current_user.username}")
            st.write(f"• **Role:** {current_user.role.value.title()}")
            st.write(f"• **Member Since:** {current_user.created_date.strftime('%Y-%m-%d')}")
            st.write(f"• **Last Login:** {current_user.last_login.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.write("**Performance Metrics:**")
            st.write(f"• **Total Sessions:** {current_user.total_sessions}")
            st.write(f"• **Avg Session Duration:** {current_user.avg_session_duration_minutes:.1f} min")
            st.write(f"• **Task Completion Rate:** {current_user.task_completion_rate:.1%}")
            st.write(f"• **Efficiency Score:** {current_user.efficiency_score:.2f}")
            st.write(f"• **Feature Adoption Rate:** {current_user.feature_adoption_rate:.1%}")
        
        # Preferences settings
        st.subheader("Personalization Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Personalization level
            current_level = current_user.personalization_level
            new_level = st.selectbox(
                "Personalization Level",
                [level.value for level in PersonalizationLevel],
                index=[level.value for level in PersonalizationLevel].index(current_level.value)
            )
            
            if new_level != current_level.value:
                current_user.personalization_level = PersonalizationLevel(new_level)
                personalization_manager._save_user_profile(current_user)
                st.success("Personalization level updated!")
        
        with col2:
            # UI Theme
            new_theme = st.selectbox(
                "UI Theme",
                ["default", "dark", "light", "blue", "green"],
                index=["default", "dark", "light", "blue", "green"].index(current_user.ui_theme)
            )
            
            if new_theme != current_user.ui_theme:
                current_user.ui_theme = new_theme
                personalization_manager._save_user_profile(current_user)
                st.success("UI theme updated!")
        
        with col3:
            # Language
            new_language = st.selectbox(
                "Language",
                ["en", "es", "fr", "de", "zh", "ja"],
                index=["en", "es", "fr", "de", "zh", "ja"].index(current_user.language)
            )
            
            if new_language != current_user.language:
                current_user.language = new_language
                personalization_manager._save_user_profile(current_user)
                st.success("Language updated!")
        
        # Most used features
        if current_user.most_used_features:
            st.subheader("Your Most Used Features")
            
            features_data = []
            for feature, count in current_user.most_used_features[:10]:
                features_data.append({"Feature": feature, "Usage Count": count})
            
            if features_data:
                features_df = pd.DataFrame(features_data)
                
                fig_features = px.bar(features_df, x='Usage Count', y='Feature',
                                    orientation='h', title='Feature Usage Frequency')
                st.plotly_chart(fig_features, use_container_width=True)
        
        # Preferred chart types
        if current_user.preferred_chart_types:
            st.subheader("Your Preferred Chart Types")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Preferences:**")
                for chart_type in current_user.preferred_chart_types[:5]:
                    st.write(f"• {chart_type.replace('_', ' ').title()}")
            
            with col2:
                # Chart type preference selector
                all_chart_types = ["bar", "line", "pie", "scatter", "histogram", "heatmap", "box", "violin"]
                
                new_preferences = st.multiselect(
                    "Update Chart Preferences",
                    all_chart_types,
                    default=current_user.preferred_chart_types
                )
                
                if st.button("Update Chart Preferences"):
                    current_user.preferred_chart_types = new_preferences
                    personalization_manager._save_user_profile(current_user)
                    st.success("Chart preferences updated!")
    
    with tab2:
        st.subheader("Personalized Recommendations")
        
        # Get current recommendations
        recommendations = personalization_manager.get_recommendations(current_user.user_id, limit=10)
        
        if recommendations:
            st.write(f"**You have {len(recommendations)} personalized recommendations:**")
            
            for i, rec in enumerate(recommendations):
                with st.container():
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        # Recommendation title and description
                        priority_icons = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                        priority_icon = priority_icons.get(rec["priority"], "⚪")
                        
                        st.write(f"**{priority_icon} {rec['title']}**")
                        st.write(rec["description"])
                        
                        # Show recommendation type and confidence
                        st.caption(f"Type: {rec['rec_type'].replace('_', ' ').title()} | "
                                 f"Relevance: {rec['relevance_score']:.2f} | "
                                 f"Confidence: {rec['confidence_score']:.2f}")
                    
                    with col2:
                        # Accept recommendation
                        if st.button("✅ Accept", key=f"accept_{rec['rec_id']}"):
                            personalization_manager.record_recommendation_interaction(
                                current_user.user_id, rec["rec_id"], "accepted"
                            )
                            
                            # Record interaction
                            personalization_manager.record_interaction(
                                current_user.user_id,
                                InteractionType.CONFIGURATION_CHANGE,
                                "recommendations",
                                {"action": "accept", "rec_id": rec["rec_id"]}
                            )
                            
                            st.success("Recommendation accepted!")
                            st.rerun()
                    
                    with col3:
                        # Reject recommendation
                        if st.button("❌ Not Helpful", key=f"reject_{rec['rec_id']}"):
                            personalization_manager.record_recommendation_interaction(
                                current_user.user_id, rec["rec_id"], "rejected"
                            )
                            
                            # Record interaction
                            personalization_manager.record_interaction(
                                current_user.user_id,
                                InteractionType.CONFIGURATION_CHANGE,
                                "recommendations",
                                {"action": "reject", "rec_id": rec["rec_id"]}
                            )
                            
                            st.info("Thanks for the feedback!")
                            st.rerun()
                    
                    # Show action data if available
                    if rec.get("action_data"):
                        with st.expander("View Details"):
                            st.json(rec["action_data"])
        else:
            st.info("No current recommendations. Keep using the system to get personalized suggestions!")
            
            # Generate new recommendations button
            if st.button("🔄 Generate New Recommendations"):
                with st.spinner("Generating personalized recommendations..."):
                    # Force regeneration of recommendations
                    behavior_analysis = personalization_manager.behavior_analyzer.analyze_user_behavior(current_user.user_id)
                    new_recs = personalization_manager.recommendation_engine.generate_recommendations(
                        current_user, behavior_analysis
                    )
                    
                    if new_recs:
                        st.success(f"Generated {len(new_recs)} new recommendations!")
                        st.rerun()
                    else:
                        st.info("No new recommendations at this time.")
        
        # Recommendation performance
        st.subheader("Recommendation Performance")
        
        rec_performance = personalization_manager.recommendation_engine.get_recommendation_performance(current_user.user_id)
        
        if "message" not in rec_performance:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                show_rate = rec_performance.get("show_rate", 0)
                st.metric("Show Rate", f"{show_rate:.1%}")
            
            with col2:
                click_rate = rec_performance.get("click_rate", 0)
                st.metric("Click Rate", f"{click_rate:.1%}")
            
            with col3:
                acceptance_rate = rec_performance.get("acceptance_rate", 0)
                st.metric("Acceptance Rate", f"{acceptance_rate:.1%}")
            
            with col4:
                total_recs = rec_performance.get("total_recommendations", 0)
                st.metric("Total Recommendations", total_recs)
    
    with tab3:
        st.subheader("UI Adaptations")
        
        # Show current UI adaptations
        ui_adaptations = personalized_experience.get("ui_adaptations", [])
        
        if ui_adaptations:
            st.write(f"**Active UI adaptations ({len(ui_adaptations)}):**")
            
            for adaptation in ui_adaptations:
                with st.expander(f"🎨 {adaptation['component'].replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Component:** {adaptation['component']}")
                        st.write(f"**Created:** {adaptation['created_date'][:19]}")
                        st.write(f"**Confidence:** {adaptation['confidence_score']:.2f}")
                        st.write(f"**Status:** {'Active' if adaptation['active'] else 'Inactive'}")
                    
                    with col2:
                        st.write("**Adaptation Data:**")
                        st.json(adaptation["adaptation_data"])
                    
                    # Test adaptation button
                    if st.button(f"Apply {adaptation['component'].replace('_', ' ').title()}", 
                               key=f"apply_{adaptation['adaptation_id']}"):
                        st.success(f"Applied {adaptation['component']} adaptation!")
                        
                        # Record interaction
                        personalization_manager.record_interaction(
                            current_user.user_id,
                            InteractionType.CONFIGURATION_CHANGE,
                            "ui_adaptation",
                            {"adaptation_id": adaptation["adaptation_id"]}
                        )
        else:
            st.info("No UI adaptations available yet. Use the system more to unlock personalized UI improvements!")
        
        # Manual UI preferences
        st.subheader("Manual UI Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dashboard layout preference
            st.write("**Dashboard Layout:**")
            layout_options = ["Standard", "Compact", "Detailed", "Focus Mode"]
            selected_layout = st.selectbox("Choose Layout", layout_options)
            
            if st.button("Apply Layout"):
                # Record interaction
                personalization_manager.record_interaction(
                    current_user.user_id,
                    InteractionType.CONFIGURATION_CHANGE,
                    "dashboard_layout",
                    {"layout": selected_layout}
                )
                st.success(f"Applied {selected_layout} layout!")
        
        with col2:
            # Component visibility
            st.write("**Component Visibility:**")
            
            components = ["Sidebar", "Toolbar", "Status Bar", "Help Panel"]
            visible_components = st.multiselect(
                "Show Components",
                components,
                default=components[:3]
            )
            
            if st.button("Update Visibility"):
                # Record interaction
                personalization_manager.record_interaction(
                    current_user.user_id,
                    InteractionType.CONFIGURATION_CHANGE,
                    "component_visibility",
                    {"visible_components": visible_components}
                )
                st.success("Component visibility updated!")
    
    with tab4:
        st.subheader("Behavior Analysis")
        
        # Get behavior analysis
        behavior_analysis = personalized_experience.get("behavior_analysis", {})
        
        if "message" in behavior_analysis:
            st.info(behavior_analysis["message"])
        else:
            # Display behavior metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session Statistics:**")
                st.write(f"• Total Interactions: {behavior_analysis.get('total_interactions', 0)}")
                st.write(f"• Unique Sessions: {behavior_analysis.get('unique_sessions', 0)}")
                st.write(f"• Avg Session Duration: {behavior_analysis.get('avg_session_duration_minutes', 0):.1f} min")
                st.write(f"• Efficiency Rate: {behavior_analysis.get('efficiency_rate', 0):.1%}")
                st.write(f"• Feature Adoption: {behavior_analysis.get('feature_adoption_rate', 0):.1%}")
            
            with col2:
                most_used = behavior_analysis.get("most_used_component")
                if most_used:
                    st.write(f"**Most Used Component:** {most_used}")
                
                peak_hours = behavior_analysis.get("peak_hours", {})
                if peak_hours:
                    st.write("**Peak Usage Hours:**")
                    for hour, count in list(peak_hours.items())[:3]:
                        st.write(f"• {hour}:00 - {count} interactions")
            
            # Interaction types distribution
            interaction_types = behavior_analysis.get("interaction_types", {})
            if interaction_types:
                st.subheader("Interaction Pattern Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Interaction types pie chart
                    types_df = pd.DataFrame(
                        list(interaction_types.items()),
                        columns=['Interaction Type', 'Count']
                    )
                    
                    fig_types = px.pie(types_df, values='Count', names='Interaction Type',
                                     title='Interaction Types Distribution')
                    st.plotly_chart(fig_types, use_container_width=True)
                
                with col2:
                    # Component usage bar chart
                    component_usage = behavior_analysis.get("component_usage", {})
                    if component_usage:
                        comp_df = pd.DataFrame(
                            list(component_usage.items()),
                            columns=['Component', 'Usage Count']
                        )
                        
                        fig_comp = px.bar(comp_df.head(10), x='Component', y='Usage Count',
                                        title='Top 10 Component Usage')
                        fig_comp.update_xaxis(tickangle=45)
                        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Generate behavior insights
        if st.button("🔍 Generate Behavior Insights"):
            with st.spinner("Analyzing behavior patterns..."):
                insights = []
                
                # Generate insights based on behavior data
                if behavior_analysis.get("efficiency_rate", 0) < 0.7:
                    insights.append("💡 Your efficiency could be improved. Consider using keyboard shortcuts and filter presets.")
                
                if behavior_analysis.get("feature_adoption_rate", 0) < 0.3:
                    insights.append("🚀 You're only using a small fraction of available features. Explore more tools to boost productivity.")
                
                avg_session = behavior_analysis.get("avg_session_duration_minutes", 0)
                if avg_session > 60:
                    insights.append("⏰ Your sessions are quite long. Consider taking breaks to maintain focus.")
                elif avg_session < 10:
                    insights.append("⚡ Your sessions are very short. You might benefit from exploring features more deeply.")
                
                if insights:
                    st.success("**Behavioral Insights Generated:**")
                    for insight in insights:
                        st.write(insight)
                else:
                    st.success("✅ Your usage patterns look great! Keep up the good work.")
    
    with tab5:
        st.subheader("Predictive Insights")
        
        # Next action prediction
        next_action = personalized_experience.get("next_action_prediction", {})
        
        if next_action:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Next Action Prediction:**")
                predicted_component = next_action.get("predicted_component", "unknown")
                predicted_type = next_action.get("predicted_interaction_type", "unknown")
                confidence = next_action.get("confidence", 0)
                
                st.write(f"• **Component:** {predicted_component.replace('_', ' ').title()}")
                st.write(f"• **Interaction Type:** {predicted_type.replace('_', ' ').title()}")
                st.write(f"• **Confidence:** {confidence:.1%}")
                
                # Confidence indicator
                if confidence > 0.7:
                    st.success("🎯 High confidence prediction")
                elif confidence > 0.4:
                    st.warning("🤔 Medium confidence prediction")
                else:
                    st.info("💭 Low confidence prediction")
            
            with col2:
                # Suggested quick actions
                st.write("**Suggested Quick Actions:**")
                
                quick_actions = [
                    "View Dashboard",
                    "Generate Report",
                    "Apply Filters",
                    "Export Data",
                    "Check Notifications"
                ]
                
                for action in quick_actions[:3]:
                    if st.button(f"🚀 {action}", key=f"quick_{action.lower().replace(' ', '_')}"):
                        # Record interaction
                        personalization_manager.record_interaction(
                            current_user.user_id,
                            InteractionType.PAGE_VIEW,
                            action.lower().replace(' ', '_'),
                            {"source": "quick_action"}
                        )
                        st.success(f"Executed: {action}")
        
        # Usage pattern predictions
        st.subheader("Usage Pattern Predictions")
        
        # Simulate pattern predictions
        patterns = [
            {
                "pattern": "Peak Usage Time",
                "prediction": "2:00 PM - 4:00 PM",
                "description": "Based on your history, you're most active during afternoon hours."
            },
            {
                "pattern": "Preferred Feature Set",
                "prediction": "Analytics & Visualization",
                "description": "You tend to focus on data analysis and visualization features."
            },
            {
                "pattern": "Session Length",
                "prediction": "45-60 minutes",
                "description": "Your typical session duration suggests focused, productive work periods."
            }
        ]
        
        for pattern in patterns:
            with st.container():
                st.write(f"**{pattern['pattern']}:**")
                st.write(f"Prediction: {pattern['prediction']}")
                st.caption(pattern['description'])
                st.markdown("---")
    
    with tab6:
        st.subheader("System Analytics")
        
        # Get system-wide analytics
        system_analytics = personalization_manager.get_system_analytics()
        
        # User statistics
        user_stats = system_analytics.get("user_statistics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", user_stats.get("total_users", 0))
        
        with col2:
            st.metric("Active Users", user_stats.get("active_users", 0))
        
        with col3:
            perf_metrics = system_analytics.get("performance_metrics", {})
            avg_efficiency = perf_metrics.get("avg_efficiency_score", 0)
            st.metric("Avg Efficiency", f"{avg_efficiency:.2f}")
        
        with col4:
            acceptance_rate = perf_metrics.get("recommendation_acceptance_rate", 0)
            st.metric("Rec. Acceptance", f"{acceptance_rate:.1%}")
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Role distribution
            role_dist = user_stats.get("role_distribution", {})
            if role_dist:
                role_df = pd.DataFrame(
                    list(role_dist.items()),
                    columns=['Role', 'Count']
                )
                
                fig_roles = px.pie(role_df, values='Count', names='Role',
                                 title='User Role Distribution')
                st.plotly_chart(fig_roles, use_container_width=True)
        
        with col2:
            # Personalization level distribution
            pers_dist = user_stats.get("personalization_distribution", {})
            if pers_dist:
                pers_df = pd.DataFrame(
                    list(pers_dist.items()),
                    columns=['Level', 'Count']
                )
                
                fig_pers = px.bar(pers_df, x='Level', y='Count',
                                title='Personalization Level Distribution')
                st.plotly_chart(fig_pers, use_container_width=True)
        
        # System health
        system_health = system_analytics.get("system_health", {})
        
        st.subheader("System Health Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            active_ratio = system_health.get("active_user_ratio", 0)
            st.metric("Active User Ratio", f"{active_ratio:.1%}")
        
        with col2:
            high_efficiency = system_health.get("high_efficiency_users", 0)
            st.metric("High Efficiency Users", high_efficiency)
        
        with col3:
            low_adoption = system_health.get("low_adoption_users", 0)
            st.metric("Low Adoption Users", low_adoption)
        
        # Export analytics
        if st.button("📊 Export Analytics Data"):
            analytics_json = json.dumps(system_analytics, indent=2, default=str)
            
            st.download_button(
                label="Download Analytics Report",
                data=analytics_json,
                file_name=f"personalization_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json'
            )

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing adaptive UI and personalized recommendations...")
    
    # Initialize personalization manager
    personalization_manager = PersonalizationManager()
    
    # Create test user profile
    user_profile = personalization_manager.get_or_create_user_profile(
        "test_user_001",
        "Test User",
        "test@example.com",
        UserRole.ANALYST
    )
    
    print(f"Created user profile: {user_profile.username}")
    
    # Record some test interactions
    interactions = [
        (InteractionType.PAGE_VIEW, "dashboard", {"page": "main_dashboard"}),
        (InteractionType.CHART_INTERACTION, "risk_analysis", {"chart_type": "bar"}),
        (InteractionType.FILTER_APPLIED, "data_explorer", {"filter": "risk_level", "value": "high"}),
        (InteractionType.EXPORT_DATA, "reports", {"format": "pdf"}),
        (InteractionType.SEARCH_QUERY, "search", {"query": "compliance risk"})
    ]
    
    for interaction_type, component, details in interactions:
        success = personalization_manager.record_interaction(
            user_profile.user_id,
            interaction_type,
            component,
            details,
            session_id="test_session_001"
        )
        print(f"Recorded interaction: {interaction_type.value} - {success}")
    
    # Get personalized experience
    personalized_experience = personalization_manager.get_personalized_experience(user_profile.user_id)
    
    print(f"Generated personalized experience:")
    print(f"- UI Adaptations: {len(personalized_experience['ui_adaptations'])}")
    print(f"- Recommendations: {len(personalized_experience['recommendations'])}")
    print(f"- Predicted next action: {personalized_experience['next_action_prediction']['predicted_component']}")
    
    # Get system analytics
    analytics = personalization_manager.get_system_analytics()
    print(f"\nSystem Analytics:")
    print(f"- Total users: {analytics['user_statistics']['total_users']}")
    print(f"- Active users: {analytics['user_statistics']['active_users']}")
    print(f"- Avg efficiency: {analytics['performance_metrics']['avg_efficiency_score']:.2f}")
    
    print("Adaptive UI and personalized recommendations test completed!")