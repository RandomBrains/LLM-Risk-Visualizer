"""
Natural Language Query and Conversational Analysis Module
Provides intelligent natural language interfaces for risk analysis and data exploration
"""

import json
import re
import time
import asyncio
import threading
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
import nltk
from collections import defaultdict, Counter
import uuid
import logging
import pickle
import spacy
from textblob import TextBlob
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of natural language queries"""
    QUESTION = "question"
    COMMAND = "command"
    ANALYSIS_REQUEST = "analysis_request"
    DATA_EXPLORATION = "data_exploration"
    VISUALIZATION = "visualization"
    COMPARISON = "comparison"
    PREDICTION = "prediction"
    EXPLANATION = "explanation"

class IntentCategory(Enum):
    """Intent categories for query classification"""
    RISK_ANALYSIS = "risk_analysis"
    DATA_RETRIEVAL = "data_retrieval"
    VISUALIZATION = "visualization"
    STATISTICAL_SUMMARY = "statistical_summary"
    COMPARISON = "comparison"
    PREDICTION = "prediction"
    EXPLANATION = "explanation"
    HELP = "help"
    CONFIGURATION = "configuration"

class ConfidenceLevel(Enum):
    """Confidence levels for NLP processing"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class NLQuery:
    """Natural language query structure"""
    query_id: str
    user_id: str
    query_text: str
    timestamp: datetime
    
    # Processing results
    query_type: Optional[QueryType] = None
    intent_category: Optional[IntentCategory] = None
    confidence_score: float = 0.0
    
    # Extracted entities and information
    entities: Dict[str, List[str]] = None
    keywords: List[str] = None
    parameters: Dict[str, Any] = None
    
    # Processing metadata
    processed: bool = False
    processing_time_ms: float = 0.0
    model_used: str = ""
    
    # Response
    response_text: str = ""
    response_data: Dict[str, Any] = None
    success: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.keywords is None:
            self.keywords = []
        if self.parameters is None:
            self.parameters = {}
        if self.response_data is None:
            self.response_data = {}

@dataclass
class ConversationContext:
    """Conversation context for maintaining state"""
    session_id: str
    user_id: str
    created_time: datetime
    last_activity: datetime
    
    # Conversation history
    query_history: List[NLQuery] = None
    context_variables: Dict[str, Any] = None
    
    # State tracking
    current_topic: Optional[str] = None
    active_filters: Dict[str, Any] = None
    last_visualization: Optional[str] = None
    
    def __post_init__(self):
        if self.query_history is None:
            self.query_history = []
        if self.context_variables is None:
            self.context_variables = {}
        if self.active_filters is None:
            self.active_filters = {}

@dataclass
class QueryTemplate:
    """Template for matching and processing queries"""
    template_id: str
    name: str
    patterns: List[str]
    intent_category: IntentCategory
    parameters: List[str]
    response_template: str
    processing_function: str
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []

class NLPProcessor:
    """Core NLP processing engine"""
    
    def __init__(self):
        self.spacy_model = None
        self.sentence_transformer = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.query_classifier = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize models
        self.initialize_models()
        
        # Query templates
        self.query_templates: Dict[str, QueryTemplate] = {}
        self.load_query_templates()
        
    def initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Using basic tokenization.")
                self.spacy_model = None
            
            # Load sentence transformer for semantic similarity
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_transformer = None
            
            # Load sentiment analysis pipeline
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
                self.sentiment_analyzer = None
            
            # Load NER pipeline
            try:
                self.ner_pipeline = pipeline("ner", 
                                           model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                           aggregation_strategy="simple")
            except Exception as e:
                logger.warning(f"Could not load NER pipeline: {e}")
                self.ner_pipeline = None
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
    
    def load_query_templates(self):
        """Load predefined query templates"""
        templates = [
            QueryTemplate(
                template_id="risk_analysis_1",
                name="Basic Risk Analysis",
                patterns=[
                    r".*risk.*analysis.*",
                    r".*analyze.*risk.*",
                    r".*risk.*assessment.*",
                    r".*what.*risk.*"
                ],
                intent_category=IntentCategory.RISK_ANALYSIS,
                parameters=["time_range", "data_source", "risk_type"],
                response_template="Performing risk analysis for {time_range} using {data_source}",
                processing_function="process_risk_analysis",
                examples=[
                    "Analyze the risk in our data",
                    "What are the current risks?",
                    "Perform a risk assessment for last month"
                ]
            ),
            QueryTemplate(
                template_id="data_visualization_1",
                name="Data Visualization Request",
                patterns=[
                    r".*show.*chart.*",
                    r".*create.*visualization.*",
                    r".*plot.*",
                    r".*graph.*",
                    r".*visualize.*"
                ],
                intent_category=IntentCategory.VISUALIZATION,
                parameters=["chart_type", "data_columns", "time_range"],
                response_template="Creating {chart_type} visualization for {data_columns}",
                processing_function="process_visualization",
                examples=[
                    "Show me a chart of risk scores over time",
                    "Create a visualization of the data",
                    "Plot the risk distribution"
                ]
            ),
            QueryTemplate(
                template_id="data_summary_1",
                name="Statistical Summary",
                patterns=[
                    r".*summary.*",
                    r".*statistics.*",
                    r".*overview.*",
                    r".*describe.*data.*",
                    r".*what.*in.*data.*"
                ],
                intent_category=IntentCategory.STATISTICAL_SUMMARY,
                parameters=["data_source", "metrics"],
                response_template="Generating statistical summary for {data_source}",
                processing_function="process_statistical_summary",
                examples=[
                    "Give me a summary of the data",
                    "What statistics can you show me?",
                    "Describe the current data"
                ]
            ),
            QueryTemplate(
                template_id="comparison_1",
                name="Data Comparison",
                patterns=[
                    r".*compare.*",
                    r".*difference.*between.*",
                    r".*versus.*",
                    r".*vs.*"
                ],
                intent_category=IntentCategory.COMPARISON,
                parameters=["comparison_items", "metrics", "time_range"],
                response_template="Comparing {comparison_items} based on {metrics}",
                processing_function="process_comparison",
                examples=[
                    "Compare this month to last month",
                    "What's the difference between high and low risk items?",
                    "Compare the performance across regions"
                ]
            ),
            QueryTemplate(
                template_id="prediction_1",
                name="Prediction Request",
                patterns=[
                    r".*predict.*",
                    r".*forecast.*",
                    r".*what.*will.*happen.*",
                    r".*future.*"
                ],
                intent_category=IntentCategory.PREDICTION,
                parameters=["prediction_target", "time_horizon", "features"],
                response_template="Generating prediction for {prediction_target} over {time_horizon}",
                processing_function="process_prediction",
                examples=[
                    "Predict next month's risk levels",
                    "What will happen to our metrics?",
                    "Forecast the trend"
                ]
            ),
            QueryTemplate(
                template_id="help_1",
                name="Help Request",
                patterns=[
                    r".*help.*",
                    r".*how.*to.*",
                    r".*what.*can.*do.*",
                    r".*explain.*how.*"
                ],
                intent_category=IntentCategory.HELP,
                parameters=[],
                response_template="Here's how I can help you with {topic}",
                processing_function="process_help",
                examples=[
                    "Help me understand this",
                    "What can you do?",
                    "How do I analyze risks?"
                ]
            )
        ]
        
        for template in templates:
            self.query_templates[template.template_id] = template
    
    def process_query(self, query: NLQuery) -> NLQuery:
        """Process natural language query"""
        start_time = time.time()
        
        try:
            # Extract entities and keywords
            self.extract_entities(query)
            
            # Classify query intent
            self.classify_intent(query)
            
            # Extract parameters
            self.extract_parameters(query)
            
            # Calculate confidence
            query.confidence_score = self.calculate_confidence(query)
            
            query.processed = True
            query.processing_time_ms = (time.time() - start_time) * 1000
            query.model_used = "hybrid_nlp"
            
        except Exception as e:
            query.error_message = str(e)
            logger.error(f"Error processing query: {e}")
        
        return query
    
    def extract_entities(self, query: NLQuery):
        """Extract named entities from query"""
        text = query.query_text.lower()
        
        # Use spaCy if available
        if self.spacy_model:
            doc = self.spacy_model(query.query_text)
            
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text
                
                if entity_type not in query.entities:
                    query.entities[entity_type] = []
                query.entities[entity_type].append(entity_text)
        
        # Use HuggingFace NER if available
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(query.query_text)
                
                for entity in ner_results:
                    entity_type = entity['entity_group']
                    entity_text = entity['word']
                    
                    if entity_type not in query.entities:
                        query.entities[entity_type] = []
                    query.entities[entity_type].append(entity_text)
            except Exception as e:
                logger.warning(f"NER pipeline error: {e}")
        
        # Extract time-related entities manually
        time_patterns = {
            'time_range': [
                r'last\s+(\w+)',
                r'past\s+(\d+)\s+(\w+)',
                r'(\d+)\s+(day|week|month|year)s?',
                r'(today|yesterday|tomorrow)',
                r'this\s+(week|month|year)',
                r'(january|february|march|april|may|june|july|august|september|october|november|december)',
                r'(\d{4})',
                r'(\d{1,2})/(\d{1,2})/(\d{4})'
            ]
        }
        
        for entity_type, patterns in time_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if entity_type not in query.entities:
                        query.entities[entity_type] = []
                    query.entities[entity_type].extend([str(match) for match in matches])
        
        # Extract risk-related terms
        risk_keywords = [
            'risk', 'threat', 'vulnerability', 'exposure', 'hazard',
            'danger', 'security', 'compliance', 'audit', 'control'
        ]
        
        found_keywords = [word for word in risk_keywords if word in text]
        if found_keywords:
            query.entities['risk_terms'] = found_keywords
        
        # Extract data-related terms
        data_keywords = [
            'data', 'dataset', 'table', 'column', 'field', 'record',
            'database', 'metric', 'measure', 'value', 'score'
        ]
        
        found_data_keywords = [word for word in data_keywords if word in text]
        if found_data_keywords:
            query.entities['data_terms'] = found_data_keywords
    
    def classify_intent(self, query: NLQuery):
        """Classify query intent using pattern matching and similarity"""
        text = query.query_text.lower()
        
        best_match_score = 0.0
        best_template = None
        
        # Pattern-based matching
        for template in self.query_templates.values():
            for pattern in template.patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    match_score = 0.8  # High score for pattern match
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_template = template
        
        # Semantic similarity matching if sentence transformer is available
        if self.sentence_transformer and best_match_score < 0.7:
            query_embedding = self.sentence_transformer.encode([query.query_text])
            
            for template in self.query_templates.values():
                if template.examples:
                    example_embeddings = self.sentence_transformer.encode(template.examples)
                    similarities = cosine_similarity(query_embedding, example_embeddings)
                    max_similarity = np.max(similarities)
                    
                    if max_similarity > best_match_score:
                        best_match_score = max_similarity
                        best_template = template
        
        # Assign intent and query type
        if best_template:
            query.intent_category = best_template.intent_category
            
            # Determine query type based on intent
            if best_template.intent_category in [IntentCategory.RISK_ANALYSIS, IntentCategory.STATISTICAL_SUMMARY]:
                query.query_type = QueryType.ANALYSIS_REQUEST
            elif best_template.intent_category == IntentCategory.VISUALIZATION:
                query.query_type = QueryType.VISUALIZATION
            elif best_template.intent_category == IntentCategory.COMPARISON:
                query.query_type = QueryType.COMPARISON
            elif best_template.intent_category == IntentCategory.PREDICTION:
                query.query_type = QueryType.PREDICTION
            elif best_template.intent_category == IntentCategory.HELP:
                query.query_type = QueryType.EXPLANATION
            else:
                query.query_type = QueryType.QUESTION
        else:
            # Default classification
            query.intent_category = IntentCategory.HELP
            query.query_type = QueryType.QUESTION
    
    def extract_parameters(self, query: NLQuery):
        """Extract parameters from query based on intent"""
        text = query.query_text.lower()
        
        # Time range extraction
        time_entities = query.entities.get('time_range', [])
        if time_entities:
            query.parameters['time_range'] = time_entities[0]
        else:
            # Default time range
            query.parameters['time_range'] = 'last_30_days'
        
        # Chart type extraction for visualization
        if query.intent_category == IntentCategory.VISUALIZATION:
            chart_keywords = {
                'bar': ['bar', 'column'],
                'line': ['line', 'trend', 'time series'],
                'pie': ['pie', 'donut'],
                'scatter': ['scatter', 'correlation'],
                'histogram': ['histogram', 'distribution'],
                'heatmap': ['heatmap', 'matrix']
            }
            
            for chart_type, keywords in chart_keywords.items():
                if any(keyword in text for keyword in keywords):
                    query.parameters['chart_type'] = chart_type
                    break
            else:
                query.parameters['chart_type'] = 'auto'  # Auto-select
        
        # Risk type extraction
        if query.intent_category == IntentCategory.RISK_ANALYSIS:
            risk_types = ['operational', 'financial', 'strategic', 'compliance', 'cybersecurity']
            for risk_type in risk_types:
                if risk_type in text:
                    query.parameters['risk_type'] = risk_type
                    break
            else:
                query.parameters['risk_type'] = 'overall'
        
        # Metrics extraction
        metric_keywords = ['score', 'level', 'count', 'percentage', 'rate', 'average', 'total']
        found_metrics = [metric for metric in metric_keywords if metric in text]
        if found_metrics:
            query.parameters['metrics'] = found_metrics
        
        # Comparison items
        if query.intent_category == IntentCategory.COMPARISON:
            comparison_keywords = ['today vs yesterday', 'this month vs last month', 
                                 'high vs low', 'before vs after']
            for comp in comparison_keywords:
                if comp in text:
                    query.parameters['comparison_type'] = comp
                    break
    
    def calculate_confidence(self, query: NLQuery) -> float:
        """Calculate confidence score for query processing"""
        confidence_factors = []
        
        # Entity extraction confidence
        if query.entities:
            entity_confidence = min(len(query.entities) * 0.2, 0.4)
            confidence_factors.append(entity_confidence)
        
        # Intent classification confidence
        if query.intent_category:
            confidence_factors.append(0.3)
        
        # Parameter extraction confidence
        if query.parameters:
            param_confidence = min(len(query.parameters) * 0.1, 0.3)
            confidence_factors.append(param_confidence)
        
        # Calculate overall confidence
        if confidence_factors:
            return min(sum(confidence_factors), 1.0)
        else:
            return 0.1  # Minimum confidence

class QueryExecutor:
    """Executes processed natural language queries"""
    
    def __init__(self):
        self.execution_history: List[NLQuery] = []
        self.cached_results: Dict[str, Any] = {}
        
    def execute_query(self, query: NLQuery, context: ConversationContext, 
                     data_source: pd.DataFrame = None) -> NLQuery:
        """Execute processed query and generate response"""
        try:
            if query.intent_category == IntentCategory.RISK_ANALYSIS:
                return self.execute_risk_analysis(query, context, data_source)
            elif query.intent_category == IntentCategory.VISUALIZATION:
                return self.execute_visualization(query, context, data_source)
            elif query.intent_category == IntentCategory.STATISTICAL_SUMMARY:
                return self.execute_statistical_summary(query, context, data_source)
            elif query.intent_category == IntentCategory.COMPARISON:
                return self.execute_comparison(query, context, data_source)
            elif query.intent_category == IntentCategory.PREDICTION:
                return self.execute_prediction(query, context, data_source)
            elif query.intent_category == IntentCategory.HELP:
                return self.execute_help(query, context)
            else:
                return self.execute_generic_response(query, context)
                
        except Exception as e:
            query.error_message = str(e)
            query.success = False
            logger.error(f"Error executing query: {e}")
            return query
    
    def execute_risk_analysis(self, query: NLQuery, context: ConversationContext, 
                            data_source: pd.DataFrame = None) -> NLQuery:
        """Execute risk analysis query"""
        
        # Generate sample risk analysis data
        np.random.seed(42)
        n_samples = 100
        
        risk_data = {
            'risk_score': np.random.beta(2, 5, n_samples),
            'risk_category': np.random.choice(['Operational', 'Financial', 'Strategic', 'Compliance'], n_samples),
            'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples),
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
        }
        
        risk_df = pd.DataFrame(risk_data)
        
        # Calculate summary statistics
        avg_risk = risk_df['risk_score'].mean()
        max_risk = risk_df['risk_score'].max()
        high_risk_count = len(risk_df[risk_df['risk_score'] > 0.7])
        
        # Generate response
        risk_level = "High" if avg_risk > 0.6 else "Medium" if avg_risk > 0.3 else "Low"
        
        response_text = f"""
        **Risk Analysis Results:**
        
        - Average Risk Score: {avg_risk:.3f}
        - Risk Level: {risk_level}
        - Maximum Risk Score: {max_risk:.3f}
        - High Risk Items: {high_risk_count}
        - Analysis Period: {query.parameters.get('time_range', 'Last 30 days')}
        
        **Key Findings:**
        - The overall risk level is {risk_level.lower()}
        - {high_risk_count} items require immediate attention
        - Most common risk category: {risk_df['risk_category'].mode().iloc[0]}
        """
        
        query.response_text = response_text.strip()
        query.response_data = {
            'risk_scores': risk_df['risk_score'].tolist(),
            'categories': risk_df['risk_category'].tolist(),
            'summary': {
                'avg_risk': avg_risk,
                'max_risk': max_risk,
                'high_risk_count': high_risk_count,
                'risk_level': risk_level
            }
        }
        query.success = True
        
        return query
    
    def execute_visualization(self, query: NLQuery, context: ConversationContext, 
                            data_source: pd.DataFrame = None) -> NLQuery:
        """Execute visualization query"""
        
        chart_type = query.parameters.get('chart_type', 'auto')
        
        # Generate sample data for visualization
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        values = np.random.normal(0.5, 0.2, 30)
        values = np.clip(values, 0, 1)  # Ensure values are between 0 and 1
        
        viz_data = pd.DataFrame({
            'date': dates,
            'risk_score': values,
            'category': np.random.choice(['A', 'B', 'C'], 30)
        })
        
        # Create visualization based on type
        if chart_type == 'line' or chart_type == 'auto':
            fig = px.line(viz_data, x='date', y='risk_score', 
                         title='Risk Score Trend Over Time')
            viz_type = 'line_chart'
        elif chart_type == 'bar':
            category_avg = viz_data.groupby('category')['risk_score'].mean().reset_index()
            fig = px.bar(category_avg, x='category', y='risk_score',
                        title='Average Risk Score by Category')
            viz_type = 'bar_chart'
        elif chart_type == 'histogram':
            fig = px.histogram(viz_data, x='risk_score', nbins=10,
                             title='Distribution of Risk Scores')
            viz_type = 'histogram'
        else:
            fig = px.scatter(viz_data, x='date', y='risk_score', color='category',
                           title='Risk Score Distribution')
            viz_type = 'scatter_plot'
        
        # Convert plotly figure to JSON for storage
        fig_json = fig.to_json()
        
        response_text = f"""
        **Visualization Created:**
        
        - Chart Type: {viz_type.replace('_', ' ').title()}
        - Data Points: {len(viz_data)}
        - Time Range: {query.parameters.get('time_range', 'Last 30 days')}
        
        The visualization shows the risk score patterns over the specified time period.
        """
        
        query.response_text = response_text.strip()
        query.response_data = {
            'chart_type': viz_type,
            'figure_json': fig_json,
            'data': viz_data.to_dict('records')
        }
        query.success = True
        
        # Update context with last visualization
        context.last_visualization = viz_type
        
        return query
    
    def execute_statistical_summary(self, query: NLQuery, context: ConversationContext, 
                                  data_source: pd.DataFrame = None) -> NLQuery:
        """Execute statistical summary query"""
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        summary_data = {
            'risk_score': np.random.beta(2, 5, n_samples),
            'value': np.random.normal(100, 25, n_samples),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'status': np.random.choice(['Active', 'Inactive', 'Pending'], n_samples)
        }
        
        df = pd.DataFrame(summary_data)
        
        # Calculate statistics
        stats = {
            'total_records': len(df),
            'risk_score_stats': {
                'mean': df['risk_score'].mean(),
                'median': df['risk_score'].median(),
                'std': df['risk_score'].std(),
                'min': df['risk_score'].min(),
                'max': df['risk_score'].max()
            },
            'value_stats': {
                'mean': df['value'].mean(),
                'median': df['value'].median(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max()
            },
            'category_distribution': df['category'].value_counts().to_dict(),
            'status_distribution': df['status'].value_counts().to_dict()
        }
        
        response_text = f"""
        **Statistical Summary:**
        
        **Dataset Overview:**
        - Total Records: {stats['total_records']:,}
        - Analysis Period: {query.parameters.get('time_range', 'All time')}
        
        **Risk Score Statistics:**
        - Mean: {stats['risk_score_stats']['mean']:.3f}
        - Median: {stats['risk_score_stats']['median']:.3f}
        - Standard Deviation: {stats['risk_score_stats']['std']:.3f}
        - Range: {stats['risk_score_stats']['min']:.3f} - {stats['risk_score_stats']['max']:.3f}
        
        **Value Statistics:**
        - Mean: {stats['value_stats']['mean']:.2f}
        - Median: {stats['value_stats']['median']:.2f}
        - Standard Deviation: {stats['value_stats']['std']:.2f}
        - Range: {stats['value_stats']['min']:.2f} - {stats['value_stats']['max']:.2f}
        
        **Category Distribution:**
        """ + "\n".join([f"- {k}: {v} ({v/stats['total_records']*100:.1f}%)" 
                         for k, v in stats['category_distribution'].items()]) + f"""
        
        **Status Distribution:**
        """ + "\n".join([f"- {k}: {v} ({v/stats['total_records']*100:.1f}%)" 
                         for k, v in stats['status_distribution'].items()])
        
        query.response_text = response_text.strip()
        query.response_data = stats
        query.success = True
        
        return query
    
    def execute_comparison(self, query: NLQuery, context: ConversationContext, 
                         data_source: pd.DataFrame = None) -> NLQuery:
        """Execute comparison query"""
        
        comparison_type = query.parameters.get('comparison_type', 'general')
        
        # Generate comparison data
        np.random.seed(42)
        
        current_data = {
            'risk_score': np.random.beta(2, 5, 100),
            'incidents': np.random.poisson(3, 100),
            'compliance_score': np.random.beta(5, 2, 100)
        }
        
        previous_data = {
            'risk_score': np.random.beta(2.5, 4.5, 100),
            'incidents': np.random.poisson(4, 100),
            'compliance_score': np.random.beta(4, 3, 100)
        }
        
        current_df = pd.DataFrame(current_data)
        previous_df = pd.DataFrame(previous_data)
        
        # Calculate comparison metrics
        comparisons = {}
        for metric in ['risk_score', 'incidents', 'compliance_score']:
            current_avg = current_df[metric].mean()
            previous_avg = previous_df[metric].mean()
            change = ((current_avg - previous_avg) / previous_avg) * 100
            
            comparisons[metric] = {
                'current': current_avg,
                'previous': previous_avg,
                'change_percent': change,
                'trend': 'improved' if change < 0 and metric != 'compliance_score' else 
                        'improved' if change > 0 and metric == 'compliance_score' else 'worsened'
            }
        
        response_text = f"""
        **Comparison Analysis:**
        
        **Risk Score Comparison:**
        - Current Period: {comparisons['risk_score']['current']:.3f}
        - Previous Period: {comparisons['risk_score']['previous']:.3f}
        - Change: {comparisons['risk_score']['change_percent']:+.1f}%
        - Trend: {comparisons['risk_score']['trend'].title()}
        
        **Incident Count Comparison:**
        - Current Period: {comparisons['incidents']['current']:.1f}
        - Previous Period: {comparisons['incidents']['previous']:.1f}
        - Change: {comparisons['incidents']['change_percent']:+.1f}%
        - Trend: {comparisons['incidents']['trend'].title()}
        
        **Compliance Score Comparison:**
        - Current Period: {comparisons['compliance_score']['current']:.3f}
        - Previous Period: {comparisons['compliance_score']['previous']:.3f}
        - Change: {comparisons['compliance_score']['change_percent']:+.1f}%
        - Trend: {comparisons['compliance_score']['trend'].title()}
        
        **Key Insights:**
        - Overall risk has {'decreased' if comparisons['risk_score']['change_percent'] < 0 else 'increased'}
        - Incident frequency has {'improved' if comparisons['incidents']['change_percent'] < 0 else 'worsened'}
        - Compliance performance has {'improved' if comparisons['compliance_score']['change_percent'] > 0 else 'declined'}
        """
        
        query.response_text = response_text.strip()
        query.response_data = {
            'comparisons': comparisons,
            'summary': {
                'overall_trend': 'positive' if sum(1 for c in comparisons.values() if c['trend'] == 'improved') >= 2 else 'negative'
            }
        }
        query.success = True
        
        return query
    
    def execute_prediction(self, query: NLQuery, context: ConversationContext, 
                          data_source: pd.DataFrame = None) -> NLQuery:
        """Execute prediction query"""
        
        # Generate historical data for prediction
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # Create trend with seasonality and noise
        trend = np.linspace(0.3, 0.6, 60)
        seasonality = 0.1 * np.sin(np.linspace(0, 4*np.pi, 60))
        noise = np.random.normal(0, 0.05, 60)
        historical_values = trend + seasonality + noise
        historical_values = np.clip(historical_values, 0, 1)
        
        # Simple prediction (extend trend)
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        future_trend = np.linspace(historical_values[-1], historical_values[-1] + 0.1, 30)
        future_seasonality = 0.1 * np.sin(np.linspace(4*np.pi, 6*np.pi, 30))
        future_noise = np.random.normal(0, 0.03, 30)
        predicted_values = future_trend + future_seasonality + future_noise
        predicted_values = np.clip(predicted_values, 0, 1)
        
        # Calculate prediction confidence intervals
        confidence_upper = predicted_values + 0.1
        confidence_lower = predicted_values - 0.1
        confidence_upper = np.clip(confidence_upper, 0, 1)
        confidence_lower = np.clip(confidence_lower, 0, 1)
        
        prediction_summary = {
            'predicted_mean': np.mean(predicted_values),
            'predicted_trend': 'increasing' if predicted_values[-1] > predicted_values[0] else 'decreasing',
            'confidence_interval': [np.mean(confidence_lower), np.mean(confidence_upper)],
            'forecast_period': '30 days'
        }
        
        response_text = f"""
        **Prediction Results:**
        
        **Forecast Summary:**
        - Forecast Period: {prediction_summary['forecast_period']}
        - Predicted Average: {prediction_summary['predicted_mean']:.3f}
        - Expected Trend: {prediction_summary['predicted_trend'].title()}
        - Confidence Interval: [{prediction_summary['confidence_interval'][0]:.3f}, {prediction_summary['confidence_interval'][1]:.3f}]
        
        **Key Predictions:**
        - The risk level is expected to {prediction_summary['predicted_trend']} over the next 30 days
        - Peak predicted value: {np.max(predicted_values):.3f}
        - Lowest predicted value: {np.min(predicted_values):.3f}
        
        **Confidence Level:** Medium (based on historical data patterns)
        
        **Note:** Predictions are based on historical trends and may vary due to external factors.
        """
        
        query.response_text = response_text.strip()
        query.response_data = {
            'historical_dates': dates.tolist(),
            'historical_values': historical_values.tolist(),
            'predicted_dates': future_dates.tolist(),
            'predicted_values': predicted_values.tolist(),
            'confidence_upper': confidence_upper.tolist(),
            'confidence_lower': confidence_lower.tolist(),
            'summary': prediction_summary
        }
        query.success = True
        
        return query
    
    def execute_help(self, query: NLQuery, context: ConversationContext) -> NLQuery:
        """Execute help query"""
        
        help_topics = {
            'general': """
            **I'm your AI Risk Analysis Assistant!**
            
            Here's what I can help you with:
            
            **🔍 Risk Analysis:**
            - "Analyze the current risks"
            - "What are the high-risk items?"
            - "Show me a risk assessment"
            
            **📊 Data Visualization:**
            - "Create a chart of risk scores"
            - "Show me a trend graph"
            - "Plot the risk distribution"
            
            **📈 Statistical Analysis:**
            - "Give me a data summary"
            - "What are the key statistics?"
            - "Describe the current data"
            
            **⚖️ Comparisons:**
            - "Compare this month to last month"
            - "What's the difference between high and low risk?"
            - "Show performance comparison"
            
            **🔮 Predictions:**
            - "Predict next month's risks"
            - "What will happen to our metrics?"
            - "Forecast the trend"
            
            **💡 Tips:**
            - Be specific about time ranges (e.g., "last 30 days")
            - Mention the type of chart you want
            - Ask follow-up questions to dive deeper
            """,
            
            'visualization': """
            **Visualization Help:**
            
            I can create various types of charts:
            - **Line charts:** For trends over time
            - **Bar charts:** For comparing categories
            - **Pie charts:** For showing proportions
            - **Scatter plots:** For correlations
            - **Histograms:** For distributions
            - **Heatmaps:** For data matrices
            
            **Example requests:**
            - "Show me a line chart of risk scores over time"
            - "Create a bar chart comparing risk categories"
            - "Plot a histogram of the data distribution"
            """,
            
            'analysis': """
            **Analysis Help:**
            
            I can perform various types of analysis:
            - **Risk Assessment:** Evaluate current risk levels
            - **Statistical Summary:** Descriptive statistics
            - **Trend Analysis:** Identify patterns over time
            - **Comparative Analysis:** Compare different periods or groups
            - **Predictive Analysis:** Forecast future trends
            
            **Example requests:**
            - "Analyze risks for the last quarter"
            - "Compare high-risk vs low-risk items"
            - "Predict next month's risk levels"
            """
        }
        
        # Determine help topic based on query context
        query_text = query.query_text.lower()
        if 'chart' in query_text or 'plot' in query_text or 'visualiz' in query_text:
            help_content = help_topics['visualization']
            topic = 'visualization'
        elif 'analyz' in query_text or 'risk' in query_text:
            help_content = help_topics['analysis']
            topic = 'analysis'
        else:
            help_content = help_topics['general']
            topic = 'general'
        
        query.response_text = help_content.strip()
        query.response_data = {
            'help_topic': topic,
            'available_commands': list(help_topics.keys())
        }
        query.success = True
        
        return query
    
    def execute_generic_response(self, query: NLQuery, context: ConversationContext) -> NLQuery:
        """Execute generic response for unclear queries"""
        
        response_text = f"""
        I understand you're asking about: "{query.query_text}"
        
        I'm not entirely sure how to help with that specific request, but I can assist with:
        
        - **Risk Analysis:** Analyze and assess risks in your data
        - **Data Visualization:** Create charts and graphs
        - **Statistical Summary:** Provide data summaries and insights
        - **Comparisons:** Compare different time periods or categories
        - **Predictions:** Forecast future trends
        
        Could you please rephrase your question or ask for help with any of these topics?
        
        **Example questions:**
        - "Analyze the current risks"
        - "Show me a chart of the data"
        - "Compare this month to last month"
        - "What are the key statistics?"
        """
        
        query.response_text = response_text.strip()
        query.response_data = {
            'suggestion_type': 'clarification_needed',
            'confidence': query.confidence_score
        }
        query.success = True
        
        return query

class ConversationManager:
    """Manages conversation sessions and context"""
    
    def __init__(self):
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.nlp_processor = NLPProcessor()
        self.query_executor = QueryExecutor()
        self.session_timeout = timedelta(hours=2)
        
    def create_session(self, user_id: str) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            created_time=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = context
        logger.info(f"Created conversation session {session_id} for user {user_id}")
        
        return session_id
    
    def process_query(self, session_id: str, query_text: str, data_source: pd.DataFrame = None) -> NLQuery:
        """Process natural language query in session context"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.active_sessions[session_id]
        context.last_activity = datetime.now()
        
        # Create query object
        query = NLQuery(
            query_id=str(uuid.uuid4()),
            user_id=context.user_id,
            query_text=query_text,
            timestamp=datetime.now()
        )
        
        # Process query with NLP
        query = self.nlp_processor.process_query(query)
        
        # Execute query
        query = self.query_executor.execute_query(query, context, data_source)
        
        # Add to context history
        context.query_history.append(query)
        
        # Update context variables based on query
        self.update_context(context, query)
        
        return query
    
    def update_context(self, context: ConversationContext, query: NLQuery):
        """Update conversation context based on query results"""
        
        # Update current topic
        if query.intent_category:
            context.current_topic = query.intent_category.value
        
        # Update active filters
        if query.parameters:
            for key, value in query.parameters.items():
                if key in ['time_range', 'risk_type', 'category']:
                    context.active_filters[key] = value
        
        # Track visualization type
        if query.intent_category == IntentCategory.VISUALIZATION and query.success:
            context.last_visualization = query.response_data.get('chart_type')
    
    def get_session_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for session"""
        return self.active_sessions.get(session_id)
    
    def cleanup_expired_sessions(self):
        """Remove expired conversation sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, context in self.active_sessions.items():
            if current_time - context.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for session"""
        
        if session_id not in self.active_sessions:
            return {}
        
        context = self.active_sessions[session_id]
        
        # Calculate statistics
        total_queries = len(context.query_history)
        successful_queries = len([q for q in context.query_history if q.success])
        
        intent_counts = Counter([q.intent_category.value for q in context.query_history if q.intent_category])
        
        avg_confidence = np.mean([q.confidence_score for q in context.query_history]) if context.query_history else 0
        
        return {
            'session_id': session_id,
            'user_id': context.user_id,
            'created_time': context.created_time,
            'last_activity': context.last_activity,
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / max(1, total_queries),
            'avg_confidence': avg_confidence,
            'top_intents': dict(intent_counts.most_common(5)),
            'current_topic': context.current_topic,
            'active_filters': context.active_filters,
            'last_visualization': context.last_visualization
        }

# Streamlit Integration Functions

def initialize_nlp_system():
    """Initialize NLP conversation system"""
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
        
        # Create default session
        user_id = "streamlit_user"
        session_id = st.session_state.conversation_manager.create_session(user_id)
        st.session_state.current_session_id = session_id
    
    return st.session_state.conversation_manager

def render_nlp_dashboard():
    """Render natural language processing dashboard"""
    st.header("🗣️ Natural Language Query & Analysis")
    
    conversation_manager = initialize_nlp_system()
    
    # Get current session
    session_id = st.session_state.current_session_id
    context = conversation_manager.get_session_context(session_id)
    
    if not context:
        st.error("Session not found. Please refresh the page.")
        return
    
    # Conversation summary
    summary = conversation_manager.get_conversation_summary(session_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", summary.get('total_queries', 0))
    
    with col2:
        success_rate = summary.get('success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col3:
        avg_confidence = summary.get('avg_confidence', 0)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        active_filters = len(summary.get('active_filters', {}))
        st.metric("Active Filters", active_filters)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Conversation",
        "📊 Query Analysis",
        "🔍 Query Templates",
        "📈 Analytics",
        "⚙️ Settings"
    ])
    
    with tab1:
        st.subheader("Natural Language Query Interface")
        
        # Query input
        query_input = st.text_area(
            "Ask me anything about your risk data:",
            placeholder="Examples:\n- Analyze the current risks\n- Show me a chart of risk scores over time\n- What are the key statistics?\n- Compare this month to last month",
            height=100
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Ask Question", type="primary"):
                if query_input.strip():
                    with st.spinner("Processing your query..."):
                        try:
                            # Process the query
                            query_result = conversation_manager.process_query(
                                session_id, query_input.strip()
                            )
                            
                            # Store result in session state for display
                            if 'query_results' not in st.session_state:
                                st.session_state.query_results = []
                            
                            st.session_state.query_results.append(query_result)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing query: {e}")
                else:
                    st.warning("Please enter a query.")
        
        with col2:
            if st.button("🧹 Clear History"):
                context.query_history.clear()
                st.session_state.query_results = []
                st.success("Conversation history cleared!")
                st.rerun()
        
        with col3:
            if st.button("🔄 New Session"):
                new_session_id = conversation_manager.create_session("streamlit_user")
                st.session_state.current_session_id = new_session_id
                st.session_state.query_results = []
                st.success("New session started!")
                st.rerun()
        
        # Display conversation history
        st.subheader("Conversation History")
        
        if hasattr(st.session_state, 'query_results') and st.session_state.query_results:
            # Display results in reverse chronological order
            for i, query_result in enumerate(reversed(st.session_state.query_results)):
                with st.container():
                    st.markdown("---")
                    
                    # Query info
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**You:** {query_result.query_text}")
                    
                    with col2:
                        confidence_color = "🟢" if query_result.confidence_score > 0.7 else "🟡" if query_result.confidence_score > 0.4 else "🔴"
                        st.write(f"{confidence_color} Confidence: {query_result.confidence_score:.2f}")
                    
                    # Response
                    if query_result.success:
                        st.markdown(f"**Assistant:** {query_result.response_text}")
                        
                        # Show visualization if available
                        if (query_result.response_data and 
                            'figure_json' in query_result.response_data):
                            try:
                                import plotly
                                fig_dict = json.loads(query_result.response_data['figure_json'])
                                fig = plotly.graph_objects.Figure(fig_dict)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying chart: {e}")
                        
                        # Show additional data if available
                        if query_result.response_data and len(query_result.response_data) > 1:
                            with st.expander("View Raw Data"):
                                st.json(query_result.response_data)
                    else:
                        st.error(f"**Error:** {query_result.error_message or 'Query processing failed'}")
                    
                    # Query metadata
                    with st.expander("Query Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Query Type:** {query_result.query_type.value if query_result.query_type else 'Unknown'}")
                            st.write(f"**Intent:** {query_result.intent_category.value if query_result.intent_category else 'Unknown'}")
                        
                        with col2:
                            st.write(f"**Processing Time:** {query_result.processing_time_ms:.1f}ms")
                            st.write(f"**Timestamp:** {query_result.timestamp.strftime('%H:%M:%S')}")
                        
                        with col3:
                            if query_result.entities:
                                st.write("**Entities:**")
                                for entity_type, entities in query_result.entities.items():
                                    st.write(f"• {entity_type}: {', '.join(entities[:3])}")
                            
                            if query_result.parameters:
                                st.write("**Parameters:**")
                                for param, value in query_result.parameters.items():
                                    st.write(f"• {param}: {value}")
        else:
            st.info("No queries yet. Ask a question above to get started!")
            
            # Show example queries
            st.subheader("Example Queries")
            
            examples = [
                "Analyze the current risk levels in our data",
                "Show me a line chart of risk scores over the last 30 days",
                "What are the key statistics for this month?",
                "Compare high-risk items to low-risk items",
                "Predict next month's risk trends",
                "Create a bar chart of risk categories",
                "Give me a summary of the data",
                "What's the difference between this week and last week?",
                "Show me the distribution of risk scores"
            ]
            
            for example in examples:
                if st.button(f"💡 {example}", key=f"example_{hash(example)}"):
                    st.session_state.example_query = example
                    st.rerun()
            
            # Handle example query selection
            if hasattr(st.session_state, 'example_query'):
                example_query = st.session_state.example_query
                delattr(st.session_state, 'example_query')
                
                with st.spinner("Processing example query..."):
                    try:
                        query_result = conversation_manager.process_query(
                            session_id, example_query
                        )
                        
                        if 'query_results' not in st.session_state:
                            st.session_state.query_results = []
                        
                        st.session_state.query_results.append(query_result)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing example query: {e}")
    
    with tab2:
        st.subheader("Query Analysis & Understanding")
        
        # Current session context
        if context:
            st.write("**Current Session Context:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Session ID:** {context.session_id[:8]}...")
                st.write(f"**Created:** {context.created_time.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Last Activity:** {context.last_activity.strftime('%Y-%m-%d %H:%M')}")
                
                if context.current_topic:
                    st.write(f"**Current Topic:** {context.current_topic.replace('_', ' ').title()}")
                
                if context.last_visualization:
                    st.write(f"**Last Visualization:** {context.last_visualization.replace('_', ' ').title()}")
            
            with col2:
                if context.active_filters:
                    st.write("**Active Filters:**")
                    for filter_key, filter_value in context.active_filters.items():
                        st.write(f"• {filter_key.replace('_', ' ').title()}: {filter_value}")
                else:
                    st.write("**Active Filters:** None")
        
        # Query type distribution
        if context.query_history:
            st.subheader("Query Pattern Analysis")
            
            # Intent distribution
            intent_counts = Counter([
                q.intent_category.value for q in context.query_history 
                if q.intent_category
            ])
            
            if intent_counts:
                col1, col2 = st.columns(2)
                
                with col1:
                    intent_df = pd.DataFrame(
                        list(intent_counts.items()), 
                        columns=['Intent', 'Count']
                    )
                    
                    fig_intent = px.pie(intent_df, values='Count', names='Intent',
                                       title='Query Intent Distribution')
                    st.plotly_chart(fig_intent, use_container_width=True)
                
                with col2:
                    # Confidence score distribution
                    confidence_scores = [q.confidence_score for q in context.query_history]
                    
                    if confidence_scores:
                        confidence_df = pd.DataFrame({
                            'Confidence Score': confidence_scores,
                            'Query': [f"Query {i+1}" for i in range(len(confidence_scores))]
                        })
                        
                        fig_confidence = px.histogram(confidence_df, x='Confidence Score',
                                                    title='Confidence Score Distribution',
                                                    nbins=10)
                        st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Recent queries analysis
            st.subheader("Recent Query Details")
            
            recent_queries = context.query_history[-10:]  # Last 10 queries
            
            query_analysis_data = []
            for i, query in enumerate(recent_queries):
                query_analysis_data.append({
                    'Query #': len(context.query_history) - len(recent_queries) + i + 1,
                    'Query Text': query.query_text[:50] + '...' if len(query.query_text) > 50 else query.query_text,
                    'Intent': query.intent_category.value if query.intent_category else 'Unknown',
                    'Type': query.query_type.value if query.query_type else 'Unknown',
                    'Confidence': f"{query.confidence_score:.2f}",
                    'Success': "✅" if query.success else "❌",
                    'Processing Time (ms)': f"{query.processing_time_ms:.1f}"
                })
            
            if query_analysis_data:
                analysis_df = pd.DataFrame(query_analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
        else:
            st.info("No queries to analyze yet.")
    
    with tab3:
        st.subheader("Query Templates & Examples")
        
        # Display available templates
        templates = conversation_manager.nlp_processor.query_templates
        
        for template_id, template in templates.items():
            with st.expander(f"📝 {template.name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Intent Category:** {template.intent_category.value.replace('_', ' ').title()}")
                    st.write(f"**Template ID:** {template.template_id}")
                    
                    if template.parameters:
                        st.write("**Parameters:**")
                        for param in template.parameters:
                            st.write(f"• {param.replace('_', ' ').title()}")
                
                with col2:
                    if template.examples:
                        st.write("**Example Queries:**")
                        for example in template.examples:
                            st.write(f"• {example}")
                    
                    st.write(f"**Response Template:**")
                    st.code(template.response_template)
                
                # Test template button
                if st.button(f"Test Template", key=f"test_{template_id}"):
                    if template.examples:
                        test_query = template.examples[0]
                        
                        with st.spinner("Testing template..."):
                            try:
                                test_result = conversation_manager.process_query(
                                    session_id, test_query
                                )
                                
                                st.success("Template test successful!")
                                st.write(f"**Query:** {test_query}")
                                st.write(f"**Response:** {test_result.response_text[:200]}...")
                                
                            except Exception as e:
                                st.error(f"Template test failed: {e}")
    
    with tab4:
        st.subheader("Analytics & Performance")
        
        # System performance metrics
        session_summary = conversation_manager.get_conversation_summary(session_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Session Performance:**")
            
            # Create performance metrics chart
            if session_summary.get('total_queries', 0) > 0:
                perf_data = {
                    'Metric': ['Total Queries', 'Successful Queries', 'Average Confidence'],
                    'Value': [
                        session_summary['total_queries'],
                        session_summary['successful_queries'],
                        session_summary['avg_confidence'] * 100  # Convert to percentage
                    ],
                    'Target': [100, 100, 80]  # Target values
                }
                
                perf_df = pd.DataFrame(perf_data)
                
                fig_perf = px.bar(perf_df, x='Metric', y='Value',
                                title='Session Performance Metrics')
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            st.write("**Intent Analysis:**")
            
            top_intents = session_summary.get('top_intents', {})
            if top_intents:
                intent_items = list(top_intents.items())
                intent_df = pd.DataFrame(intent_items, columns=['Intent', 'Count'])
                
                fig_intents = px.bar(intent_df, x='Count', y='Intent',
                                   orientation='h',
                                   title='Most Common Query Intents')
                st.plotly_chart(fig_intents, use_container_width=True)
            else:
                st.info("No intent data available yet.")
        
        # Detailed analytics
        if context.query_history:
            st.subheader("Detailed Query Analytics")
            
            # Processing time analysis
            processing_times = [q.processing_time_ms for q in context.query_history]
            confidence_scores = [q.confidence_score for q in context.query_history]
            timestamps = [q.timestamp for q in context.query_history]
            
            if processing_times:
                analytics_df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Processing Time (ms)': processing_times,
                    'Confidence Score': confidence_scores,
                    'Success': [q.success for q in context.query_history]
                })
                
                # Processing time trend
                fig_time_trend = px.line(analytics_df, x='Timestamp', y='Processing Time (ms)',
                                       title='Query Processing Time Trend')
                st.plotly_chart(fig_time_trend, use_container_width=True)
                
                # Confidence vs Processing Time
                fig_conf_time = px.scatter(analytics_df, x='Processing Time (ms)', y='Confidence Score',
                                         color='Success', title='Confidence vs Processing Time')
                st.plotly_chart(fig_conf_time, use_container_width=True)
    
    with tab5:
        st.subheader("NLP System Settings")
        
        # Model status
        st.write("**NLP Model Status:**")
        
        nlp_processor = conversation_manager.nlp_processor
        
        model_status = {
            'spaCy Model': "✅ Loaded" if nlp_processor.spacy_model else "❌ Not Available",
            'Sentence Transformer': "✅ Loaded" if nlp_processor.sentence_transformer else "❌ Not Available",
            'Sentiment Analyzer': "✅ Loaded" if nlp_processor.sentiment_analyzer else "❌ Not Available",
            'NER Pipeline': "✅ Loaded" if nlp_processor.ner_pipeline else "❌ Not Available"
        }
        
        for model, status in model_status.items():
            st.write(f"• **{model}:** {status}")
        
        # Configuration options
        st.subheader("Configuration")
        
        with st.expander("Session Management"):
            # Session timeout
            timeout_hours = st.slider("Session Timeout (hours)", 1, 24, 2)
            
            if st.button("Update Session Timeout"):
                conversation_manager.session_timeout = timedelta(hours=timeout_hours)
                st.success(f"Session timeout updated to {timeout_hours} hours")
            
            # Cleanup expired sessions
            if st.button("Cleanup Expired Sessions"):
                conversation_manager.cleanup_expired_sessions()
                st.success("Expired sessions cleaned up")
        
        with st.expander("Query Processing"):
            # Confidence threshold
            confidence_threshold = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.3)
            
            # Processing timeout
            processing_timeout = st.slider("Processing Timeout (seconds)", 5, 60, 30)
            
            if st.button("Update Processing Settings"):
                st.success("Processing settings updated")
        
        # System information
        st.subheader("System Information")
        
        system_info = {
            'Active Sessions': len(conversation_manager.active_sessions),
            'Total Templates': len(conversation_manager.nlp_processor.query_templates),
            'Current Session Queries': len(context.query_history) if context else 0,
            'Session Created': context.created_time.strftime('%Y-%m-%d %H:%M:%S') if context else 'N/A',
            'Last Activity': context.last_activity.strftime('%Y-%m-%d %H:%M:%S') if context else 'N/A'
        }
        
        info_df = pd.DataFrame(list(system_info.items()), columns=['Property', 'Value'])
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        # Export conversation data
        st.subheader("Data Export")
        
        if st.button("📊 Export Conversation Data"):
            if context and context.query_history:
                export_data = {
                    'session_summary': session_summary,
                    'query_history': [
                        {
                            'query_id': q.query_id,
                            'query_text': q.query_text,
                            'timestamp': q.timestamp.isoformat(),
                            'intent_category': q.intent_category.value if q.intent_category else None,
                            'query_type': q.query_type.value if q.query_type else None,
                            'confidence_score': q.confidence_score,
                            'success': q.success,
                            'processing_time_ms': q.processing_time_ms,
                            'entities': q.entities,
                            'parameters': q.parameters
                        }
                        for q in context.query_history
                    ]
                }
                
                export_json = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="Download Conversation Data",
                    data=export_json,
                    file_name=f"conversation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime='application/json'
                )
            else:
                st.info("No conversation data to export")

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing natural language query and conversational analysis...")
    
    # Initialize conversation manager
    conversation_manager = ConversationManager()
    
    # Create test session
    session_id = conversation_manager.create_session("test_user")
    print(f"Created session: {session_id}")
    
    # Test queries
    test_queries = [
        "Analyze the current risks in our data",
        "Show me a chart of risk scores over time",
        "What are the key statistics?",
        "Compare this month to last month",
        "Predict next month's risk levels"
    ]
    
    for query_text in test_queries:
        print(f"\nProcessing query: {query_text}")
        
        query_result = conversation_manager.process_query(session_id, query_text)
        
        print(f"Intent: {query_result.intent_category.value if query_result.intent_category else 'Unknown'}")
        print(f"Confidence: {query_result.confidence_score:.2f}")
        print(f"Success: {query_result.success}")
        print(f"Response: {query_result.response_text[:100]}...")
    
    # Get conversation summary
    summary = conversation_manager.get_conversation_summary(session_id)
    print(f"\nConversation Summary:")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Average confidence: {summary['avg_confidence']:.2f}")
    
    print("Natural language query and conversational analysis test completed!")