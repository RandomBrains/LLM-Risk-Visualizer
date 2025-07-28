"""
Advanced Visualization and Interactive Charts Module
Provides enhanced data visualization capabilities with interactive features
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import json

from config import COLOR_SCHEME, CHART_SETTINGS, RISK_THRESHOLDS

class AdvancedVisualizer:
    """Enhanced visualization class with advanced chart types and interactivity"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.risk_colors = {
            'low': '#2E8B57',      # Sea Green
            'medium': '#FF8C00',   # Dark Orange
            'high': '#DC143C',     # Crimson
            'critical': '#8B0000'  # Dark Red
        }
        
    def create_3d_risk_landscape(self, data: pd.DataFrame, title: str = "3D Risk Landscape") -> go.Figure:
        """Create 3D surface plot showing risk landscape across models and languages"""
        
        # Prepare data for 3D surface
        pivot_data = data.pivot_table(
            index='Model', 
            columns='Language', 
            values='Risk_Rate', 
            aggfunc='mean'
        ).fillna(0)
        
        if pivot_data.empty:
            return self._create_empty_chart("No data available for 3D landscape")
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                z=pivot_data.values,
                x=list(pivot_data.columns),
                y=list(pivot_data.index),
                colorscale='RdYlGn_r',
                colorbar=dict(title="Risk Score"),
                hovertemplate='<b>Model:</b> %{y}<br>' +
                             '<b>Language:</b> %{x}<br>' +
                             '<b>Risk Score:</b> %{z:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Language",
                yaxis_title="Model",
                zaxis_title="Risk Score",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
            ),
            height=600,
            width=800
        )
        
        return fig
    
    def create_network_diagram(self, data: pd.DataFrame, title: str = "Risk Relationship Network") -> go.Figure:
        """Create network diagram showing relationships between models, languages, and risk categories"""
        
        if data.empty:
            return self._create_empty_chart("No data available for network diagram")
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges based on correlations
        risk_pivot = data.pivot_table(
            index=['Model', 'Language'], 
            columns='Risk_Category', 
            values='Risk_Rate', 
            aggfunc='mean'
        ).fillna(0)
        
        # Calculate correlations between risk categories
        if len(risk_pivot.columns) > 1:
            corr_matrix = risk_pivot.corr()
            
            # Add nodes
            for risk_category in corr_matrix.columns:
                G.add_node(risk_category, node_type='risk_category')
            
            # Add edges for strong correlations
            for i, cat1 in enumerate(corr_matrix.columns):
                for j, cat2 in enumerate(corr_matrix.columns):
                    if i < j and abs(corr_matrix.iloc[i, j]) > 0.5:  # Strong correlation threshold
                        G.add_edge(cat1, cat2, weight=abs(corr_matrix.iloc[i, j]))
        
        # Get layout positions
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        if not pos:
            return self._create_empty_chart("Insufficient data for network analysis")
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]].get('weight', 1))
        
        # Create network plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=30,
                color=self.color_palette[:len(node_text)],
                line=dict(width=2, color='rgb(50,50,50)')
            )
        ))
        
        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Network shows correlations between risk categories",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="grey", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            width=700
        )
        
        return fig
    
    def create_sankey_diagram(self, data: pd.DataFrame, title: str = "Risk Flow Diagram") -> go.Figure:
        """Create Sankey diagram showing flow between models, languages, and risk levels"""
        
        if data.empty:
            return self._create_empty_chart("No data available for Sankey diagram")
        
        # Categorize risk levels
        data = data.copy()
        data['Risk_Level'] = pd.cut(
            data['Risk_Rate'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Create flow data
        flows = data.groupby(['Model', 'Language', 'Risk_Level']).size().reset_index(name='Count')
        
        # Create unique labels and indices
        all_labels = []
        label_to_index = {}
        
        # Add models
        models = flows['Model'].unique()
        for model in models:
            label_to_index[model] = len(all_labels)
            all_labels.append(model)
        
        # Add languages
        languages = flows['Language'].unique()
        for lang in languages:
            label_to_index[f"Lang: {lang}"] = len(all_labels)
            all_labels.append(f"Lang: {lang}")
        
        # Add risk levels
        risk_levels = flows['Risk_Level'].dropna().unique()
        for level in risk_levels:
            label_to_index[level] = len(all_labels)
            all_labels.append(level)
        
        # Create source, target, and value arrays
        source = []
        target = []
        value = []
        
        # Model to Language flows
        model_lang_flows = flows.groupby(['Model', 'Language'])['Count'].sum().reset_index()
        for _, row in model_lang_flows.iterrows():
            source.append(label_to_index[row['Model']])
            target.append(label_to_index[f"Lang: {row['Language']}"])
            value.append(row['Count'])
        
        # Language to Risk Level flows
        lang_risk_flows = flows.groupby(['Language', 'Risk_Level'])['Count'].sum().reset_index()
        for _, row in lang_risk_flows.iterrows():
            if pd.notna(row['Risk_Level']):
                source.append(label_to_index[f"Lang: {row['Language']}"])
                target.append(label_to_index[row['Risk_Level']])
                value.append(row['Count'])
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=[
                    self.color_palette[i % len(self.color_palette)]
                    for i in range(len(all_labels))
                ]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color='rgba(255,0,255,0.4)'
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=10,
            height=500,
            width=800
        )
        
        return fig
    
    def create_parallel_coordinates(self, data: pd.DataFrame, title: str = "Parallel Coordinates Plot") -> go.Figure:
        """Create parallel coordinates plot for multi-dimensional analysis"""
        
        if data.empty:
            return self._create_empty_chart("No data available for parallel coordinates")
        
        # Prepare data for parallel coordinates
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        if numeric_data.empty:
            return self._create_empty_chart("No numeric data available")
        
        # Add categorical dimensions as numeric
        if 'Model' in data.columns:
            model_mapping = {model: i for i, model in enumerate(data['Model'].unique())}
            numeric_data['Model_Code'] = data['Model'].map(model_mapping)
        
        if 'Language' in data.columns:
            lang_mapping = {lang: i for i, lang in enumerate(data['Language'].unique())}
            numeric_data['Language_Code'] = data['Language'].map(lang_mapping)
        
        if 'Risk_Category' in data.columns:
            risk_mapping = {risk: i for i, risk in enumerate(data['Risk_Category'].unique())}
            numeric_data['Risk_Category_Code'] = data['Risk_Category'].map(risk_mapping)
        
        # Create dimensions for parallel coordinates
        dimensions = []
        
        for col in numeric_data.columns:
            if col in ['Risk_Rate', 'Sample_Size', 'Confidence']:
                dimensions.append(dict(
                    label=col,
                    values=numeric_data[col],
                    range=[numeric_data[col].min(), numeric_data[col].max()]
                ))
            elif col.endswith('_Code'):
                # Categorical dimension
                original_col = col.replace('_Code', '')
                if original_col in data.columns:
                    unique_values = data[original_col].unique()
                    dimensions.append(dict(
                        label=original_col,
                        values=numeric_data[col],
                        range=[0, len(unique_values) - 1],
                        tickvals=list(range(len(unique_values))),
                        ticktext=unique_values
                    ))
        
        # Create parallel coordinates plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=numeric_data.get('Risk_Rate', np.random.rand(len(numeric_data))),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Rate")
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            width=900
        )
        
        return fig
    
    def create_treemap(self, data: pd.DataFrame, title: str = "Risk Treemap") -> go.Figure:
        """Create treemap visualization for hierarchical risk data"""
        
        if data.empty:
            return self._create_empty_chart("No data available for treemap")
        
        # Aggregate data for treemap
        treemap_data = data.groupby(['Model', 'Language', 'Risk_Category']).agg({
            'Risk_Rate': 'mean',
            'Sample_Size': 'sum'
        }).reset_index()
        
        # Create hierarchical labels
        treemap_data['ids'] = treemap_data.apply(
            lambda row: f"{row['Model']} - {row['Language']} - {row['Risk_Category']}", axis=1
        )
        treemap_data['parents'] = treemap_data.apply(
            lambda row: f"{row['Model']} - {row['Language']}", axis=1
        )
        treemap_data['labels'] = treemap_data['Risk_Category']
        
        # Add parent nodes
        parent_data = treemap_data.groupby(['Model', 'Language']).agg({
            'Risk_Rate': 'mean',
            'Sample_Size': 'sum'
        }).reset_index()
        
        parent_data['ids'] = parent_data.apply(
            lambda row: f"{row['Model']} - {row['Language']}", axis=1
        )
        parent_data['parents'] = parent_data['Model']
        parent_data['labels'] = parent_data['Language']
        
        # Add root nodes
        root_data = treemap_data.groupby('Model').agg({
            'Risk_Rate': 'mean',
            'Sample_Size': 'sum'
        }).reset_index()
        
        root_data['ids'] = root_data['Model']
        root_data['parents'] = ""
        root_data['labels'] = root_data['Model']
        
        # Combine all data
        all_treemap_data = pd.concat([
            root_data[['ids', 'parents', 'labels', 'Risk_Rate', 'Sample_Size']],
            parent_data[['ids', 'parents', 'labels', 'Risk_Rate', 'Sample_Size']],
            treemap_data[['ids', 'parents', 'labels', 'Risk_Rate', 'Sample_Size']]
        ], ignore_index=True)
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            ids=all_treemap_data['ids'],
            labels=all_treemap_data['labels'],
            parents=all_treemap_data['parents'],
            values=all_treemap_data['Sample_Size'],
            branchvalues="total",
            marker=dict(
                colorscale='RdYlGn_r',
                cmid=0.5,
                colorbar=dict(title="Risk Rate")
            ),
            hovertemplate='<b>%{label}</b><br>Risk Rate: %{color:.3f}<br>Sample Size: %{value}<extra></extra>',
            maxdepth=3
        ))
        
        # Update color based on risk rate
        fig.data[0].marker.color = all_treemap_data['Risk_Rate']
        
        fig.update_layout(
            title=title,
            height=600,
            width=800
        )
        
        return fig
    
    def create_waterfall_chart(self, data: pd.DataFrame, title: str = "Risk Change Waterfall") -> go.Figure:
        """Create waterfall chart showing risk changes over time"""
        
        if data.empty or 'Date' not in data.columns:
            return self._create_empty_chart("No temporal data available for waterfall chart")
        
        # Calculate daily risk changes
        daily_risk = data.groupby('Date')['Risk_Rate'].mean().sort_index()
        
        if len(daily_risk) < 2:
            return self._create_empty_chart("Insufficient temporal data")
        
        # Calculate changes
        changes = daily_risk.diff().fillna(0)
        
        # Prepare data for waterfall
        x_labels = []
        y_values = []
        colors = []
        
        # Starting value
        x_labels.append(f"Start ({daily_risk.index[0]})")
        y_values.append(daily_risk.iloc[0])
        colors.append('blue')
        
        # Changes
        cumulative = daily_risk.iloc[0]
        for i in range(1, len(changes)):
            change = changes.iloc[i]
            x_labels.append(f"{daily_risk.index[i]}")
            y_values.append(change)
            colors.append('green' if change >= 0 else 'red')
            cumulative += change
        
        # Final value
        x_labels.append(f"End ({daily_risk.index[-1]})")
        y_values.append(cumulative)
        colors.append('blue')
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Risk Changes",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(changes) - 1) + ["total"],
            x=x_labels,
            textposition="outside",
            text=[f"{val:.3f}" for val in y_values],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=500,
            width=800,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_radar_chart_comparison(self, data: pd.DataFrame, models: List[str], 
                                    title: str = "Multi-Model Radar Comparison") -> go.Figure:
        """Create advanced radar chart comparing multiple models"""
        
        if data.empty:
            return self._create_empty_chart("No data available for radar chart")
        
        # Filter data for selected models
        model_data = data[data['Model'].isin(models)]
        
        if model_data.empty:
            return self._create_empty_chart("No data available for selected models")
        
        # Calculate average risk by category for each model
        radar_data = model_data.groupby(['Model', 'Risk_Category'])['Risk_Rate'].mean().unstack(fill_value=0)
        
        # Create radar chart
        fig = go.Figure()
        
        categories = list(radar_data.columns)
        
        for i, model in enumerate(radar_data.index):
            values = radar_data.loc[model].tolist()
            values += values[:1]  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model,
                line=dict(color=self.color_palette[i % len(self.color_palette)]),
                fillcolor=f"rgba{px.colors.hex_to_rgb(self.color_palette[i % len(self.color_palette)]) + (0.1,)}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=title,
            height=500,
            width=600
        )
        
        return fig
    
    def create_bubble_chart(self, data: pd.DataFrame, title: str = "Risk Bubble Chart") -> go.Figure:
        """Create bubble chart with risk rate, confidence, and sample size"""
        
        if data.empty:
            return self._create_empty_chart("No data available for bubble chart")
        
        # Aggregate data
        bubble_data = data.groupby(['Model', 'Language']).agg({
            'Risk_Rate': 'mean',
            'Confidence': 'mean',
            'Sample_Size': 'sum'
        }).reset_index()
        
        # Create bubble chart
        fig = px.scatter(
            bubble_data,
            x='Risk_Rate',
            y='Confidence',
            size='Sample_Size',
            color='Model',
            hover_name='Language',
            hover_data={'Sample_Size': ':,'},
            title=title,
            labels={
                'Risk_Rate': 'Average Risk Rate',
                'Confidence': 'Average Confidence',
                'Sample_Size': 'Total Samples'
            }
        )
        
        fig.update_traces(marker=dict(sizemode='diameter', sizeref=2.*max(bubble_data['Sample_Size'])/(40.**2)))
        
        fig.update_layout(
            height=500,
            width=700,
            showlegend=True
        )
        
        return fig
    
    def create_animated_timeline(self, data: pd.DataFrame, title: str = "Risk Evolution Timeline") -> go.Figure:
        """Create animated timeline showing risk evolution"""
        
        if data.empty or 'Date' not in data.columns:
            return self._create_empty_chart("No temporal data available for timeline")
        
        # Prepare data for animation
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Create animated scatter plot
        fig = px.scatter(
            data,
            x='Risk_Rate',
            y='Confidence',
            animation_frame=data['Date'].dt.strftime('%Y-%m-%d'),
            animation_group='Model',
            size='Sample_Size',
            color='Risk_Category',
            hover_name='Model',
            size_max=55,
            range_x=[0, 1],
            range_y=[0, 1],
            title=title
        )
        
        fig.update_layout(
            height=600,
            width=800,
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = "Risk Correlation Matrix") -> go.Figure:
        """Create enhanced correlation heatmap with clustering"""
        
        if data.empty:
            return self._create_empty_chart("No data available for correlation analysis")
        
        # Create pivot table for correlation analysis
        risk_pivot = data.pivot_table(
            index=['Model', 'Language'],
            columns='Risk_Category',
            values='Risk_Rate',
            aggfunc='mean'
        ).fillna(0)
        
        if risk_pivot.empty or len(risk_pivot.columns) < 2:
            return self._create_empty_chart("Insufficient data for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = risk_pivot.corr()
        
        # Create enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hoverongaps=False
        ))
        
        # Add annotations for strong correlations
        annotations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                    annotations.append(dict(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.columns[i],
                        text="●",
                        showarrow=False,
                        font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.8 else "black", size=20)
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Risk Categories",
            yaxis_title="Risk Categories",
            height=500,
            width=600,
            annotations=annotations
        )
        
        return fig
    
    def create_distribution_violin(self, data: pd.DataFrame, title: str = "Risk Distribution Analysis") -> go.Figure:
        """Create violin plot showing risk rate distributions"""
        
        if data.empty:
            return self._create_empty_chart("No data available for distribution analysis")
        
        # Create violin plot
        fig = go.Figure()
        
        models = data['Model'].unique()
        
        for i, model in enumerate(models):
            model_data = data[data['Model'] == model]['Risk_Rate']
            
            fig.add_trace(go.Violin(
                y=model_data,
                name=model,
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.color_palette[i % len(self.color_palette)],
                opacity=0.6,
                x0=model
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Risk Rate",
            xaxis_title="Model",
            height=500,
            width=700,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            width=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

# Interactive dashboard components
class InteractiveDashboard:
    """Interactive dashboard with advanced filtering and exploration"""
    
    def __init__(self):
        self.visualizer = AdvancedVisualizer()
    
    def render_advanced_visualizations(self, data: pd.DataFrame):
        """Render advanced visualization dashboard"""
        
        st.header("🎨 Advanced Visualizations")
        
        if data.empty:
            st.warning("No data available for visualization.")
            return
        
        # Visualization selector
        viz_options = {
            "3D Risk Landscape": "3d_landscape",
            "Network Diagram": "network",
            "Sankey Flow": "sankey",
            "Parallel Coordinates": "parallel",
            "Treemap": "treemap",
            "Waterfall Chart": "waterfall",
            "Multi-Model Radar": "radar",
            "Bubble Chart": "bubble",
            "Animated Timeline": "timeline",
            "Correlation Heatmap": "correlation",
            "Distribution Violin": "violin"
        }
        
        selected_viz = st.selectbox("Select Visualization Type", list(viz_options.keys()))
        viz_type = viz_options[selected_viz]
        
        # Create visualization based on selection
        if viz_type == "3d_landscape":
            fig = self.visualizer.create_3d_risk_landscape(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "network":
            fig = self.visualizer.create_network_diagram(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "sankey":
            fig = self.visualizer.create_sankey_diagram(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "parallel":
            fig = self.visualizer.create_parallel_coordinates(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "treemap":
            fig = self.visualizer.create_treemap(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "waterfall":
            fig = self.visualizer.create_waterfall_chart(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "radar":
            models = st.multiselect(
                "Select Models for Comparison",
                data['Model'].unique(),
                default=data['Model'].unique()[:3]
            )
            if models:
                fig = self.visualizer.create_radar_chart_comparison(data, models)
                st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "bubble":
            fig = self.visualizer.create_bubble_chart(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "timeline":
            fig = self.visualizer.create_animated_timeline(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "correlation":
            fig = self.visualizer.create_correlation_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "violin":
            fig = self.visualizer.create_distribution_violin(data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Add interactivity options
        self._add_interactivity_controls(data, viz_type)
    
    def _add_interactivity_controls(self, data: pd.DataFrame, viz_type: str):
        """Add interactive controls for visualizations"""
        
        st.subheader("🎛️ Interactive Controls")
        
        with st.expander("Customization Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Color scheme selector
                color_schemes = ['Set3', 'Pastel1', 'Dark2', 'Viridis', 'Plasma']
                selected_scheme = st.selectbox("Color Scheme", color_schemes)
            
            with col2:
                # Chart size options
                chart_sizes = {'Small': (400, 300), 'Medium': (600, 500), 'Large': (800, 600)}
                selected_size = st.selectbox("Chart Size", list(chart_sizes.keys()))
            
            with col3:
                # Export options
                export_format = st.selectbox("Export Format", ["PNG", "HTML", "SVG", "PDF"])
                
                if st.button("📥 Export Chart"):
                    st.info(f"Chart would be exported as {export_format} format")
        
        # Data insights
        self._show_data_insights(data, viz_type)
    
    def _show_data_insights(self, data: pd.DataFrame, viz_type: str):
        """Show automatic insights based on visualization type"""
        
        st.subheader("🔍 Automatic Insights")
        
        insights = []
        
        # Generate insights based on data
        if not data.empty:
            avg_risk = data['Risk_Rate'].mean()
            max_risk = data['Risk_Rate'].max()
            min_risk = data['Risk_Rate'].min()
            
            insights.append(f"📊 Average risk rate: {avg_risk:.3f}")
            insights.append(f"⬆️ Maximum risk rate: {max_risk:.3f}")
            insights.append(f"⬇️ Minimum risk rate: {min_risk:.3f}")
            
            # Model-specific insights
            if 'Model' in data.columns:
                riskiest_model = data.groupby('Model')['Risk_Rate'].mean().idxmax()
                safest_model = data.groupby('Model')['Risk_Rate'].mean().idxmin()
                
                insights.append(f"🚨 Highest risk model: {riskiest_model}")
                insights.append(f"✅ Lowest risk model: {safest_model}")
            
            # Language-specific insights
            if 'Language' in data.columns and len(data['Language'].unique()) > 1:
                riskiest_lang = data.groupby('Language')['Risk_Rate'].mean().idxmax()
                insights.append(f"🌍 Highest risk language: {riskiest_lang}")
            
            # Temporal insights
            if 'Date' in data.columns:
                data_copy = data.copy()
                data_copy['Date'] = pd.to_datetime(data_copy['Date'])
                daily_risk = data_copy.groupby('Date')['Risk_Rate'].mean()
                
                if len(daily_risk) > 1:
                    trend = "increasing" if daily_risk.is_monotonic_increasing else "decreasing" if daily_risk.is_monotonic_decreasing else "mixed"
                    insights.append(f"📈 Risk trend: {trend}")
        
        # Display insights
        for insight in insights:
            st.write(insight)

# Integration function
def integrate_advanced_visualizations():
    """Integration function for main application"""
    return InteractiveDashboard()

if __name__ == "__main__":
    # Example usage and testing
    from sample_data import generate_risk_data
    
    # Generate sample data
    sample_data = generate_risk_data(days=30)
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer()
    
    # Test different visualizations
    print("Testing advanced visualizations...")
    
    # 3D Landscape
    fig_3d = visualizer.create_3d_risk_landscape(sample_data)
    print("✅ 3D landscape created")
    
    # Network diagram
    fig_network = visualizer.create_network_diagram(sample_data)
    print("✅ Network diagram created")
    
    # Sankey diagram
    fig_sankey = visualizer.create_sankey_diagram(sample_data)
    print("✅ Sankey diagram created")
    
    # Parallel coordinates
    fig_parallel = visualizer.create_parallel_coordinates(sample_data)
    print("✅ Parallel coordinates created")
    
    # Treemap
    fig_treemap = visualizer.create_treemap(sample_data)
    print("✅ Treemap created")
    
    # Waterfall chart
    fig_waterfall = visualizer.create_waterfall_chart(sample_data)
    print("✅ Waterfall chart created")
    
    # Radar comparison
    models = sample_data['Model'].unique()[:3]
    fig_radar = visualizer.create_radar_chart_comparison(sample_data, models)
    print("✅ Radar chart created")
    
    # Bubble chart
    fig_bubble = visualizer.create_bubble_chart(sample_data)
    print("✅ Bubble chart created")
    
    # Animated timeline
    fig_timeline = visualizer.create_animated_timeline(sample_data)
    print("✅ Animated timeline created")
    
    # Correlation heatmap
    fig_corr = visualizer.create_correlation_heatmap(sample_data)
    print("✅ Correlation heatmap created")
    
    # Distribution violin
    fig_violin = visualizer.create_distribution_violin(sample_data)
    print("✅ Distribution violin created")
    
    print("Advanced visualization module test completed successfully!")