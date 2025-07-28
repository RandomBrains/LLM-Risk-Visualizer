"""
LLM Risk Visualizer - Enhanced Main Application
A comprehensive dashboard for analyzing and visualizing LLM risks with advanced features
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import asyncio
import json
import os
from typing import Optional, Dict, List

# Import custom modules
from config import MODELS, LANGUAGES, RISK_CATEGORIES, COLOR_SCHEME, RISK_THRESHOLDS
from sample_data import (
    generate_risk_data, 
    generate_cluster_data, 
    generate_comparison_matrix,
    generate_incident_log
)
from data_processor import DataProcessor
from visualizations import Visualizer
from auth import (
    AuthManager, init_auth_state, login_form, logout, 
    is_authenticated, has_permission, get_current_user, require_permission
)
from database import DatabaseManager
from api import APIManager, setup_api_manager
from monitoring import MonitoringService, MONITORING_CONFIG

# Page configuration
st.set_page_config(
    page_title="LLM Risk Visualizer Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
init_auth_state()

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        margin: 1rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .user-info {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize global objects
@st.cache_resource
def init_database():
    """Initialize database manager"""
    return DatabaseManager()

@st.cache_resource
def init_monitoring():
    """Initialize monitoring service"""
    db_manager = init_database()
    monitoring_service = MonitoringService(MONITORING_CONFIG, db_manager)
    return monitoring_service

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(use_api=False, api_config=None):
    """Load data from database or API"""
    db_manager = init_database()
    
    if use_api and api_config:
        try:
            # Load from API
            api_manager = setup_api_manager(api_config)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # This would be async in a real implementation
            api_data = asyncio.run(api_manager.fetch_all_data(start_date, end_date))
            
            if not api_data.empty:
                # Store in database
                db_manager.risk_data.insert_risk_data(api_data)
                risk_data = api_data
            else:
                # Fallback to database
                risk_data = db_manager.risk_data.get_risk_data()
        except Exception as e:
            st.error(f"API fetch failed: {e}")
            risk_data = db_manager.risk_data.get_risk_data()
    else:
        # Load from database or generate sample data
        risk_data = db_manager.risk_data.get_risk_data()
        
        if risk_data.empty:
            # Generate sample data for demo
            risk_data = generate_risk_data(days=90)
            db_manager.risk_data.insert_risk_data(risk_data)
    
    # Generate additional demo data
    cluster_data = generate_cluster_data()
    comparison_data = generate_comparison_matrix()
    incident_data = generate_incident_log()
    
    return risk_data, cluster_data, comparison_data, incident_data

def show_user_info():
    """Display current user information"""
    if is_authenticated():
        user = get_current_user()
        st.markdown(f"""
        <div class="user-info">
            👤 <strong>{user['username']}</strong> ({user['role']}) | 
            <span style="color: #666;">Last activity: {datetime.now().strftime('%H:%M')}</span>
        </div>
        """, unsafe_allow_html=True)

def admin_panel():
    """Admin panel for user management and system configuration"""
    if not has_permission('manage_users'):
        st.error("Access denied. Admin privileges required.")
        return
    
    st.header("🔧 Administration Panel")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "🔌 API Config", "🚨 Monitoring", "📊 System Health"])
    
    with tab1:
        st.subheader("User Management")
        
        # Add new user
        with st.expander("➕ Add New User"):
            with st.form("add_user_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["viewer", "analyst", "admin"])
                
                if st.form_submit_button("Create User"):
                    auth_manager = AuthManager()
                    success, message = auth_manager.register(new_username, new_email, new_password, new_role)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        # Display existing users
        auth_manager = AuthManager()
        users = auth_manager.db.get_all_users()
        
        if users:
            users_df = pd.DataFrame(users)
            st.dataframe(users_df, use_container_width=True)
    
    with tab2:
        st.subheader("API Configuration")
        
        # API connection settings
        with st.form("api_config_form"):
            st.write("Configure LLM API connections:")
            
            openai_key = st.text_input("OpenAI API Key", type="password")
            anthropic_key = st.text_input("Anthropic API Key", type="password")
            google_key = st.text_input("Google API Key", type="password")
            
            if st.form_submit_button("Save API Configuration"):
                # In a real app, store securely
                st.session_state.api_config = {
                    'openai': openai_key if openai_key else None,
                    'anthropic': anthropic_key if anthropic_key else None,
                    'google': google_key if google_key else None
                }
                st.success("API configuration saved!")
        
        # Test API connections
        if st.button("🔍 Test API Connections"):
            if 'api_config' in st.session_state:
                try:
                    api_manager = setup_api_manager(st.session_state.api_config)
                    results = api_manager.test_connections()
                    
                    for provider, status in results.items():
                        if status:
                            st.success(f"✅ {provider.title()}: Connected")
                        else:
                            st.error(f"❌ {provider.title()}: Failed")
                except Exception as e:
                    st.error(f"Connection test failed: {e}")
            else:
                st.warning("No API configuration found")
    
    with tab3:
        st.subheader("Monitoring & Alerts")
        
        monitoring_service = init_monitoring()
        status = monitoring_service.get_monitoring_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monitoring Status", "🟢 Running" if status['running'] else "🔴 Stopped")
        
        with col2:
            st.metric("Active Rules", status['active_rules'])
        
        with col3:
            st.metric("Active Alerts", status['active_alerts'])
        
        # Manual monitoring trigger
        if st.button("🚨 Run Manual Check"):
            result = monitoring_service.trigger_manual_check()
            if result['success']:
                st.success(result['message'])
            else:
                st.error(result['message'])
        
        # Display recent alerts
        st.subheader("Recent Alerts")
        alerts = monitoring_service.alert_manager.get_active_alerts()
        if not alerts.empty:
            st.dataframe(alerts, use_container_width=True)
        else:
            st.info("No active alerts")
    
    with tab4:
        st.subheader("System Health")
        
        db_manager = init_database()
        health = db_manager.health_check()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "🟢 Healthy" if health['database'] else "🔴 Failed"
            st.metric("Database", status)
        
        with col2:
            status = "🟢 Connected" if health['cache'] else "🟡 Disconnected"
            st.metric("Cache", status)
        
        with col3:
            status = "🟢 Ready" if health['exports_dir'] else "🔴 Missing"
            st.metric("Export Directory", status)
        
        # System statistics
        st.subheader("Data Summary")
        summary = db_manager.risk_data.get_data_summary()
        
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = summary.get('total_records', [{'count': 0}])[0]['count']
                st.metric("Total Records", f"{total:,}")
            
            with col2:
                models = summary.get('unique_models', [{'count': 0}])[0]['count']
                st.metric("Unique Models", models)
            
            with col3:
                languages = summary.get('unique_languages', [{'count': 0}])[0]['count']
                st.metric("Languages", languages)
            
            with col4:
                if 'date_range' in summary and summary['date_range']:
                    date_info = summary['date_range'][0]
                    if date_info['min_date'] and date_info['max_date']:
                        days = (datetime.fromisoformat(date_info['max_date']) - 
                               datetime.fromisoformat(date_info['min_date'])).days
                        st.metric("Data Range (days)", days)
                    else:
                        st.metric("Data Range (days)", "N/A")

def dashboard_main():
    """Main dashboard interface"""
    
    # Load data
    use_api = st.sidebar.checkbox("Use Live API Data", value=False)
    api_config = st.session_state.get('api_config', {}) if use_api else None
    
    with st.spinner("Loading data..."):
        risk_data, cluster_data, comparison_data, incident_data = load_data(use_api, api_config)
    
    processor = DataProcessor(risk_data)
    visualizer = Visualizer()
    
    # Sidebar filters
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Date range filter
        st.subheader("📅 Date Range")
        if not risk_data.empty:
            min_date = pd.to_datetime(risk_data['Date']).min().date()
            max_date = pd.to_datetime(risk_data['Date']).max().date()
            
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = (datetime.now().date() - timedelta(days=30), datetime.now().date())
        
        # Model filter
        st.subheader("🤖 Models")
        available_models = risk_data['Model'].unique().tolist() if not risk_data.empty else MODELS
        selected_models = st.multiselect(
            "Select models to analyze",
            options=available_models,
            default=available_models[:4] if len(available_models) >= 4 else available_models
        )
        
        # Language filter
        st.subheader("🌍 Languages")
        available_languages = risk_data['Language'].unique().tolist() if not risk_data.empty else LANGUAGES
        selected_languages = st.multiselect(
            "Select languages",
            options=available_languages,
            default=available_languages[:5] if len(available_languages) >= 5 else available_languages
        )
        
        # Risk category filter
        st.subheader("⚠️ Risk Categories")
        available_categories = risk_data['Risk_Category'].unique().tolist() if not risk_data.empty else RISK_CATEGORIES
        selected_categories = st.multiselect(
            "Select risk categories",
            options=available_categories,
            default=available_categories
        )
        
        # Real-time refresh
        st.subheader("🔄 Refresh")
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
        if st.button("Refresh Now") or auto_refresh:
            st.cache_data.clear()
            st.rerun()
    
    # Filter data
    if not risk_data.empty:
        filtered_data = risk_data[
            (pd.to_datetime(risk_data['Date']).dt.date >= date_range[0]) &
            (pd.to_datetime(risk_data['Date']).dt.date <= date_range[1]) &
            (risk_data['Model'].isin(selected_models)) &
            (risk_data['Language'].isin(selected_languages)) &
            (risk_data['Risk_Category'].isin(selected_categories))
        ]
    else:
        filtered_data = risk_data
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Dashboard", 
        "📈 Trend Analysis", 
        "🔍 Deep Analysis", 
        "🚨 Anomaly Detection", 
        "📋 Reports & Export",
        "👤 Profile"
    ])
    
    with tab1:
        st.header("📊 Executive Dashboard")
        
        if filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_risk = filtered_data['Risk_Rate'].mean()
            st.metric("Average Risk Score", f"{avg_risk:.3f}")
        
        with col2:
            high_risk_count = len(filtered_data[filtered_data['Risk_Rate'] > RISK_THRESHOLDS['high']])
            st.metric("High Risk Items", high_risk_count)
        
        with col3:
            total_samples = filtered_data['Sample_Size'].sum()
            st.metric("Total Samples", f"{total_samples:,}")
        
        with col4:
            data_freshness = (datetime.now() - pd.to_datetime(filtered_data['Date']).max()).days
            st.metric("Data Age (days)", data_freshness)
        
        # Risk heatmap
        st.subheader("Risk Score Heatmap")
        if len(selected_models) > 0 and len(selected_languages) > 0:
            heatmap_data = filtered_data.groupby(['Model', 'Language'])['Risk_Rate'].mean().unstack(fill_value=0)
            if not heatmap_data.empty:
                fig_heatmap = visualizer.create_risk_heatmap(heatmap_data)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution by Category")
            risk_by_category = filtered_data.groupby('Risk_Category')['Risk_Rate'].mean().sort_values(ascending=False)
            fig_bar = px.bar(
                x=risk_by_category.values,
                y=risk_by_category.index,
                orientation='h',
                title="Average Risk by Category",
                color=risk_by_category.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Model Performance Comparison")
            model_performance = filtered_data.groupby('Model')['Risk_Rate'].mean().sort_values()
            fig_model = px.bar(
                x=model_performance.index,
                y=model_performance.values,
                title="Average Risk by Model",
                color=model_performance.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_model, use_container_width=True)
    
    with tab2:
        st.header("📈 Trend Analysis")
        
        if filtered_data.empty:
            st.warning("No data available for trend analysis.")
            return
        
        # Time series analysis
        st.subheader("Risk Trends Over Time")
        daily_trends = filtered_data.groupby(['Date', 'Risk_Category'])['Risk_Rate'].mean().reset_index()
        
        if not daily_trends.empty:
            fig_trends = px.line(
                daily_trends,
                x='Date',
                y='Risk_Rate',
                color='Risk_Category',
                title="Risk Rate Trends by Category"
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        
        # Trend metrics
        st.subheader("Trend Summary")
        db_manager = init_database()
        trend_metrics = db_manager.risk_data.calculate_trend_metrics(30)
        
        if not trend_metrics.empty:
            # Calculate trend direction
            trend_metrics['trend'] = trend_metrics.apply(
                lambda row: 'Improving' if row['latest_rate'] < row['earliest_rate'] 
                else 'Worsening' if row['latest_rate'] > row['earliest_rate'] 
                else 'Stable', axis=1
            )
            
            # Display trends
            improving = len(trend_metrics[trend_metrics['trend'] == 'Improving'])
            worsening = len(trend_metrics[trend_metrics['trend'] == 'Worsening'])
            stable = len(trend_metrics[trend_metrics['trend'] == 'Stable'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Improving Trends", improving, delta=improving-worsening)
            with col2:
                st.metric("Worsening Trends", worsening, delta=-(worsening-improving))
            with col3:
                st.metric("Stable Trends", stable)
    
    with tab3:
        st.header("🔍 Deep Analysis")
        
        # Advanced analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Correlation Analysis", "Anomaly Detection", "Model Comparison", "Language Impact"]
        )
        
        if analysis_type == "Correlation Analysis":
            st.subheader("Risk Category Correlations")
            if not filtered_data.empty:
                corr_data = filtered_data.pivot_table(
                    index=['Date', 'Model', 'Language'],
                    columns='Risk_Category',
                    values='Risk_Rate'
                ).corr()
                
                if not corr_data.empty:
                    fig_corr = px.imshow(
                        corr_data,
                        title="Risk Category Correlation Matrix",
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        elif analysis_type == "Model Comparison":
            st.subheader("Detailed Model Comparison")
            if not filtered_data.empty and len(selected_models) > 1:
                comparison_stats = filtered_data.groupby(['Model', 'Risk_Category']).agg({
                    'Risk_Rate': ['mean', 'std', 'min', 'max'],
                    'Sample_Size': 'sum'
                }).round(4)
                
                st.dataframe(comparison_stats, use_container_width=True)
    
    with tab4:
        st.header("🚨 Anomaly Detection")
        
        # Display detected anomalies
        db_manager = init_database()
        recent_anomalies = db_manager.anomalies.get_anomalies(days=7)
        
        if not recent_anomalies.empty:
            st.subheader("Recent Anomalies (Last 7 days)")
            
            for _, anomaly in recent_anomalies.iterrows():
                severity_class = f"alert-{anomaly['severity']}"
                
                st.markdown(f"""
                <div class="{severity_class}">
                    <strong>{anomaly['model']} - {anomaly['risk_category']}</strong><br>
                    Expected: {anomaly['expected_rate']:.3f} | Actual: {anomaly['actual_rate']:.3f}<br>
                    Anomaly Score: {anomaly['anomaly_score']:.3f} | Date: {anomaly['date']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No anomalies detected in the last 7 days.")
        
        # Anomaly statistics
        anomaly_summary = db_manager.anomalies.get_anomaly_summary()
        
        if anomaly_summary:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total = anomaly_summary.get('total_anomalies', {}).get('count', 0)
                st.metric("Total Anomalies", total)
            
            with col2:
                unack = anomaly_summary.get('unacknowledged', {}).get('count', 0)
                st.metric("Unacknowledged", unack)
            
            with col3:
                recent = anomaly_summary.get('recent_anomalies', {}).get('count', 0)
                st.metric("Recent (7 days)", recent)
    
    with tab5:
        st.header("📋 Reports & Export")
        
        if not has_permission('export_data'):
            st.error("You don't have permission to export data.")
            return
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Filtered Data")
            
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
            
            if st.button("📥 Export Data"):
                try:
                    db_manager = init_database()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if export_format == "CSV":
                        filename = f"risk_data_{timestamp}.csv"
                        success, path = db_manager.exporter.export_to_csv(
                            filtered_data, filename, get_current_user()['id'] if is_authenticated() else None
                        )
                    elif export_format == "JSON":
                        filename = f"risk_data_{timestamp}.json"
                        success, path = db_manager.exporter.export_to_json(
                            filtered_data, filename, get_current_user()['id'] if is_authenticated() else None
                        )
                    
                    if success:
                        st.success(f"Data exported successfully to {path}")
                        
                        # Provide download link
                        with open(path, 'rb') as f:
                            st.download_button(
                                label="Download Export File",
                                data=f.read(),
                                file_name=filename,
                                mime="text/csv" if export_format == "CSV" else "application/json"
                            )
                    else:
                        st.error(f"Export failed: {path}")
                        
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        with col2:
            st.subheader("Quick Stats")
            
            if not filtered_data.empty:
                stats_data = {
                    "Total Records": len(filtered_data),
                    "Date Range": f"{filtered_data['Date'].min()} to {filtered_data['Date'].max()}",
                    "Models": len(filtered_data['Model'].unique()),
                    "Languages": len(filtered_data['Language'].unique()),
                    "Risk Categories": len(filtered_data['Risk_Category'].unique()),
                    "Avg Risk Score": f"{filtered_data['Risk_Rate'].mean():.3f}",
                    "Max Risk Score": f"{filtered_data['Risk_Rate'].max():.3f}",
                    "Min Risk Score": f"{filtered_data['Risk_Rate'].min():.3f}"
                }
                
                for key, value in stats_data.items():
                    st.write(f"**{key}:** {value}")
    
    with tab6:
        st.header("👤 User Profile")
        
        if is_authenticated():
            user = get_current_user()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Account Information")
                st.write(f"**Username:** {user['username']}")
                st.write(f"**Role:** {user['role'].title()}")
                
                # Display permissions
                st.subheader("Permissions")
                permissions = user.get('permissions', {})
                for perm, granted in permissions.items():
                    icon = "✅" if granted else "❌"
                    st.write(f"{icon} {perm.replace('_', ' ').title()}")
            
            with col2:
                st.subheader("Actions")
                
                if st.button("🚪 Logout"):
                    logout()
        else:
            st.info("Please log in to view profile information.")

def main():
    """Main application entry point"""
    
    # Check authentication
    if not is_authenticated():
        login_form()
        return
    
    # Show user info
    show_user_info()
    
    # Navigation
    user = get_current_user()
    
    if user['role'] == 'admin':
        page = st.sidebar.selectbox("Navigation", ["Dashboard", "Admin Panel"])
        
        if page == "Admin Panel":
            admin_panel()
        else:
            dashboard_main()
    else:
        dashboard_main()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>LLM Risk Visualizer Pro v3.0 | Enhanced with Authentication, API Integration & Monitoring | © 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()