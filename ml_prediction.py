"""
Advanced Machine Learning Risk Prediction Module
Implements intelligent risk forecasting and pattern analysis using ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from database import DatabaseManager

class RiskPredictor:
    """Advanced risk prediction using machine learning models"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.model_performance = {}
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        # Ensure Date column is datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Create time-based features
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['month'] = data['Date'].dt.month
        data['quarter'] = data['Date'].dt.quarter
        data['days_since_start'] = (data['Date'] - data['Date'].min()).dt.days
        
        # Create lag features (previous values)
        for lag in [1, 3, 7, 14]:
            data[f'risk_rate_lag_{lag}'] = data.groupby(['Model', 'Language', 'Risk_Category'])['Risk_Rate'].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14, 30]:
            data[f'risk_rate_mean_{window}d'] = data.groupby(['Model', 'Language', 'Risk_Category'])['Risk_Rate'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            data[f'risk_rate_std_{window}d'] = data.groupby(['Model', 'Language', 'Risk_Category'])['Risk_Rate'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
        
        # Encode categorical variables
        categorical_cols = ['Model', 'Language', 'Risk_Category']
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.encoders[col].fit_transform(data[col].astype(str))
            else:
                # Handle unseen categories
                data[f'{col}_temp'] = data[col].astype(str)
                mask = data[f'{col}_temp'].isin(self.encoders[col].classes_)
                data[f'{col}_encoded'] = 0  # Default value for unseen categories
                data.loc[mask, f'{col}_encoded'] = self.encoders[col].transform(data.loc[mask, f'{col}_temp'])
                data.drop(f'{col}_temp', axis=1, inplace=True)
        
        # Create interaction features
        data['model_language_interaction'] = data['Model_encoded'] * data['Language_encoded']
        data['model_risk_interaction'] = data['Model_encoded'] * data['Risk_Category_encoded']
        
        # Fill missing values (from lag features)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        
        return data
    
    def train_risk_prediction_model(self, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train risk prediction model"""
        # Get historical data
        data = self.db.risk_data.get_risk_data()
        
        if data.empty:
            return {'success': False, 'error': 'No training data available'}
        
        # Prepare features
        data = self.prepare_features(data)
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['Date', 'Risk_Rate', 'Model', 'Language', 'Risk_Category', 'Sample_Size', 'Confidence']
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare training data
        X = data[self.feature_columns].copy()
        y = data['Risk_Rate'].copy()
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            return {'success': False, 'error': 'Insufficient training data'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        if model_type not in self.scalers:
            self.scalers[model_type] = StandardScaler()
        
        X_train_scaled = self.scalers[model_type].fit_transform(X_train)
        X_test_scaled = self.scalers[model_type].transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
        
        elif model_type == 'linear_regression':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        
        else:
            return {'success': False, 'error': f'Unsupported model type: {model_type}'}
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        performance = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(self.feature_columns)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        performance['cv_mae'] = -cv_scores.mean()
        performance['cv_mae_std'] = cv_scores.std()
        
        # Store model and performance
        self.models[model_type] = model
        self.model_performance[model_type] = performance
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            performance['feature_importance'] = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            'success': True,
            'model_type': model_type,
            'performance': performance,
            'training_date': datetime.now().isoformat()
        }
    
    def predict_future_risks(self, days_ahead: int = 7, model_type: str = 'random_forest') -> pd.DataFrame:
        """Predict future risk rates"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        # Get recent data for prediction
        recent_data = self.db.risk_data.get_risk_data()
        
        if recent_data.empty:
            return pd.DataFrame()
        
        # Prepare features
        recent_data = self.prepare_features(recent_data)
        
        # Get the most recent data point for each model-language-category combination
        latest_data = recent_data.groupby(['Model', 'Language', 'Risk_Category']).last().reset_index()
        
        predictions = []
        
        for _, row in latest_data.iterrows():
            # Create future dates
            last_date = pd.to_datetime(row['Date'])
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
            
            for future_date in future_dates:
                # Create prediction features
                pred_row = row.copy()
                pred_row['Date'] = future_date
                pred_row['day_of_week'] = future_date.dayofweek
                pred_row['month'] = future_date.month
                pred_row['quarter'] = future_date.quarter
                pred_row['days_since_start'] = (future_date - pd.to_datetime(recent_data['Date']).min()).days
                
                # Use current risk rate as lag features (simplified approach)
                current_risk = row['Risk_Rate']
                for lag in [1, 3, 7, 14]:
                    pred_row[f'risk_rate_lag_{lag}'] = current_risk
                
                # Use recent rolling statistics
                for window in [3, 7, 14, 30]:
                    pred_row[f'risk_rate_mean_{window}d'] = current_risk
                    pred_row[f'risk_rate_std_{window}d'] = 0.01  # Small default std
                
                # Prepare feature vector
                try:
                    X_pred = pred_row[self.feature_columns].values.reshape(1, -1)
                    X_pred = np.nan_to_num(X_pred, nan=0.0)  # Replace NaN with 0
                    X_pred_scaled = self.scalers[model_type].transform(X_pred)
                    
                    # Make prediction
                    predicted_risk = self.models[model_type].predict(X_pred_scaled)[0]
                    
                    # Ensure prediction is within valid bounds
                    predicted_risk = max(0.0, min(1.0, predicted_risk))
                    
                    predictions.append({
                        'Date': future_date,
                        'Model': row['Model'],
                        'Language': row['Language'],
                        'Risk_Category': row['Risk_Category'],
                        'Predicted_Risk_Rate': predicted_risk,
                        'Prediction_Confidence': self._calculate_prediction_confidence(predicted_risk, current_risk),
                        'Model_Type': model_type
                    })
                    
                except Exception as e:
                    print(f"Prediction error for {row['Model']}-{row['Language']}-{row['Risk_Category']}: {e}")
                    continue
        
        return pd.DataFrame(predictions)
    
    def _calculate_prediction_confidence(self, predicted_risk: float, current_risk: float) -> float:
        """Calculate prediction confidence based on various factors"""
        # Simple heuristic: confidence decreases with the difference from current risk
        diff = abs(predicted_risk - current_risk)
        base_confidence = 0.8
        confidence = base_confidence * np.exp(-diff * 2)  # Exponential decay
        return max(0.1, min(0.95, confidence))
    
    def detect_anomalies(self, contamination: float = 0.1) -> pd.DataFrame:
        """Detect anomalies in risk data using Isolation Forest"""
        # Get recent data
        data = self.db.risk_data.get_risk_data()
        
        if data.empty:
            return pd.DataFrame()
        
        # Prepare features
        data = self.prepare_features(data)
        
        # Select features for anomaly detection
        anomaly_features = [col for col in self.feature_columns if col in data.columns]
        
        if not anomaly_features:
            return pd.DataFrame()
        
        X = data[anomaly_features].copy()
        X = X.fillna(X.mean())  # Fill missing values
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        
        # Add results to dataframe
        data['is_anomaly'] = anomaly_labels == -1
        data['anomaly_score'] = anomaly_scores
        
        # Return only anomalies
        anomalies = data[data['is_anomaly']].copy()
        
        # Add severity based on anomaly score
        if not anomalies.empty:
            # Normalize anomaly scores to [0, 1] for severity
            min_score = anomalies['anomaly_score'].min()
            max_score = anomalies['anomaly_score'].max()
            
            if max_score != min_score:
                anomalies['severity'] = 1 - (anomalies['anomaly_score'] - min_score) / (max_score - min_score)
            else:
                anomalies['severity'] = 0.5
            
            # Categorize severity
            anomalies['severity_level'] = pd.cut(
                anomalies['severity'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['low', 'medium', 'high']
            )
        
        return anomalies[['Date', 'Model', 'Language', 'Risk_Category', 'Risk_Rate', 
                        'anomaly_score', 'severity', 'severity_level']]
    
    def analyze_risk_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze risk trends and patterns"""
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = self.db.risk_data.get_risk_data(start_date=start_date, end_date=end_date)
        
        if data.empty:
            return {'error': 'No data available for trend analysis'}
        
        data['Date'] = pd.to_datetime(data['Date'])
        
        trends = {}
        
        # Overall trend
        overall_trend = data.groupby('Date')['Risk_Rate'].mean().reset_index()
        overall_trend = overall_trend.sort_values('Date')
        
        if len(overall_trend) > 1:
            # Calculate trend slope
            X = np.arange(len(overall_trend)).reshape(-1, 1)
            y = overall_trend['Risk_Rate'].values
            
            lr = LinearRegression()
            lr.fit(X, y)
            
            trends['overall'] = {
                'slope': lr.coef_[0],
                'direction': 'increasing' if lr.coef_[0] > 0.001 else 'decreasing' if lr.coef_[0] < -0.001 else 'stable',
                'start_risk': overall_trend['Risk_Rate'].iloc[0],
                'end_risk': overall_trend['Risk_Rate'].iloc[-1],
                'change': overall_trend['Risk_Rate'].iloc[-1] - overall_trend['Risk_Rate'].iloc[0],
                'volatility': overall_trend['Risk_Rate'].std()
            }
        
        # Model-specific trends
        trends['by_model'] = {}
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model].groupby('Date')['Risk_Rate'].mean().reset_index()
            model_data = model_data.sort_values('Date')
            
            if len(model_data) > 1:
                X = np.arange(len(model_data)).reshape(-1, 1)
                y = model_data['Risk_Rate'].values
                
                lr = LinearRegression()
                lr.fit(X, y)
                
                trends['by_model'][model] = {
                    'slope': lr.coef_[0],
                    'direction': 'increasing' if lr.coef_[0] > 0.001 else 'decreasing' if lr.coef_[0] < -0.001 else 'stable',
                    'current_risk': model_data['Risk_Rate'].iloc[-1],
                    'change': model_data['Risk_Rate'].iloc[-1] - model_data['Risk_Rate'].iloc[0]
                }
        
        # Risk category trends
        trends['by_category'] = {}
        for category in data['Risk_Category'].unique():
            cat_data = data[data['Risk_Category'] == category].groupby('Date')['Risk_Rate'].mean().reset_index()
            cat_data = cat_data.sort_values('Date')
            
            if len(cat_data) > 1:
                X = np.arange(len(cat_data)).reshape(-1, 1)
                y = cat_data['Risk_Rate'].values
                
                lr = LinearRegression()
                lr.fit(X, y)
                
                trends['by_category'][category] = {
                    'slope': lr.coef_[0],
                    'direction': 'increasing' if lr.coef_[0] > 0.001 else 'decreasing' if lr.coef_[0] < -0.001 else 'stable',
                    'current_risk': cat_data['Risk_Rate'].iloc[-1],
                    'change': cat_data['Risk_Rate'].iloc[-1] - cat_data['Risk_Rate'].iloc[0]
                }
        
        return trends
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models to file"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'performance': self.model_performance,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.encoders = model_data.get('encoders', {})
            self.feature_columns = model_data.get('feature_columns', [])
            self.model_performance = model_data.get('performance', {})
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all trained models"""
        return self.model_performance
    
    def retrain_models_if_needed(self, min_new_data_points: int = 100) -> Dict[str, Any]:
        """Automatically retrain models if enough new data is available"""
        # Check if we have enough new data since last training
        recent_data = self.db.risk_data.get_risk_data()
        
        if len(recent_data) < min_new_data_points:
            return {'retrained': False, 'reason': 'Insufficient new data'}
        
        # Retrain all model types
        results = {}
        for model_type in ['random_forest', 'linear_regression']:
            result = self.train_risk_prediction_model(model_type)
            results[model_type] = result
        
        return {
            'retrained': True,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

class RiskInsightGenerator:
    """Generate intelligent insights from risk data and predictions"""
    
    def __init__(self, risk_predictor: RiskPredictor):
        self.predictor = risk_predictor
    
    def generate_risk_summary(self) -> Dict[str, Any]:
        """Generate comprehensive risk summary with insights"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'insights': [],
            'recommendations': [],
            'alerts': []
        }
        
        # Analyze trends
        trends = self.predictor.analyze_risk_trends(days=30)
        
        if 'overall' in trends:
            overall = trends['overall']
            
            if overall['direction'] == 'increasing':
                summary['insights'].append({
                    'type': 'trend',
                    'severity': 'high' if overall['slope'] > 0.01 else 'medium',
                    'message': f"Overall risk trend is increasing by {overall['slope']:.4f} per day",
                    'details': {
                        'change': overall['change'],
                        'volatility': overall['volatility']
                    }
                })
                
                summary['recommendations'].append({
                    'category': 'monitoring',
                    'priority': 'high',
                    'action': 'Increase monitoring frequency and review model configurations',
                    'rationale': 'Rising risk trend detected'
                })
            
            elif overall['direction'] == 'decreasing':
                summary['insights'].append({
                    'type': 'trend',
                    'severity': 'low',
                    'message': f"Overall risk trend is improving by {abs(overall['slope']):.4f} per day",
                    'details': {
                        'change': overall['change'],
                        'volatility': overall['volatility']
                    }
                })
        
        # Analyze model-specific trends
        if 'by_model' in trends:
            for model, model_trend in trends['by_model'].items():
                if model_trend['direction'] == 'increasing' and model_trend['slope'] > 0.01:
                    summary['alerts'].append({
                        'type': 'model_risk',
                        'severity': 'high',
                        'message': f"High risk increase detected for {model}",
                        'model': model,
                        'current_risk': model_trend['current_risk'],
                        'change': model_trend['change']
                    })
                    
                    summary['recommendations'].append({
                        'category': 'model_specific',
                        'priority': 'high',
                        'action': f'Review and potentially retrain {model} model',
                        'rationale': f'Risk increasing at rate of {model_trend["slope"]:.4f} per day'
                    })
        
        # Check for anomalies
        try:
            anomalies = self.predictor.detect_anomalies(contamination=0.05)
            
            if not anomalies.empty:
                high_severity_anomalies = anomalies[anomalies['severity_level'] == 'high']
                
                if not high_severity_anomalies.empty:
                    summary['alerts'].append({
                        'type': 'anomaly',
                        'severity': 'critical',
                        'message': f"Detected {len(high_severity_anomalies)} high-severity anomalies",
                        'count': len(high_severity_anomalies),
                        'models_affected': high_severity_anomalies['Model'].unique().tolist()
                    })
                    
                    summary['recommendations'].append({
                        'category': 'anomaly_response',
                        'priority': 'critical',
                        'action': 'Immediate investigation of anomalous risk patterns required',
                        'rationale': 'High-severity anomalies detected in risk data'
                    })
        except Exception as e:
            print(f"Anomaly detection failed: {e}")
        
        # Generate predictions and insights
        try:
            predictions = self.predictor.predict_future_risks(days_ahead=7)
            
            if not predictions.empty:
                # Check for predicted high risks
                high_risk_predictions = predictions[predictions['Predicted_Risk_Rate'] > 0.7]
                
                if not high_risk_predictions.empty:
                    summary['alerts'].append({
                        'type': 'prediction',
                        'severity': 'high',
                        'message': f"Predicted high risk events in next 7 days: {len(high_risk_predictions)}",
                        'count': len(high_risk_predictions),
                        'models_affected': high_risk_predictions['Model'].unique().tolist()
                    })
                    
                    summary['recommendations'].append({
                        'category': 'prevention',
                        'priority': 'high',
                        'action': 'Prepare risk mitigation strategies for predicted high-risk scenarios',
                        'rationale': 'ML models predict elevated risk levels in coming days'
                    })
        except Exception as e:
            print(f"Prediction generation failed: {e}")
        
        return summary
    
    def generate_model_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific recommendations for each model"""
        recommendations = []
        
        # Get recent performance data
        recent_data = self.predictor.db.risk_data.get_risk_data()
        
        if recent_data.empty:
            return recommendations
        
        # Analyze each model
        for model in recent_data['Model'].unique():
            model_data = recent_data[recent_data['Model'] == model]
            avg_risk = model_data['Risk_Rate'].mean()
            risk_std = model_data['Risk_Rate'].std()
            
            recommendation = {
                'model': model,
                'current_avg_risk': avg_risk,
                'risk_volatility': risk_std,
                'recommendations': []
            }
            
            # Risk level recommendations
            if avg_risk > 0.7:
                recommendation['recommendations'].append({
                    'type': 'high_risk',
                    'priority': 'critical',
                    'action': f'Consider replacing or retraining {model} - average risk {avg_risk:.3f} exceeds safe threshold',
                    'expected_impact': 'Significant risk reduction'
                })
            elif avg_risk > 0.5:
                recommendation['recommendations'].append({
                    'type': 'medium_risk',
                    'priority': 'high',
                    'action': f'Monitor {model} closely and consider optimization - average risk {avg_risk:.3f}',
                    'expected_impact': 'Moderate risk reduction'
                })
            
            # Volatility recommendations
            if risk_std > 0.2:
                recommendation['recommendations'].append({
                    'type': 'high_volatility',
                    'priority': 'medium',
                    'action': f'Investigate {model} consistency - high volatility {risk_std:.3f} detected',
                    'expected_impact': 'Improved predictability'
                })
            
            recommendations.append(recommendation)
        
        return recommendations

# Integration functions for main application
def integrate_ml_prediction():
    """Integration function to add ML prediction capabilities to main app"""
    from database import DatabaseManager
    
    db_manager = DatabaseManager()
    predictor = RiskPredictor(db_manager)
    insight_generator = RiskInsightGenerator(predictor)
    
    return predictor, insight_generator

if __name__ == "__main__":
    # Example usage and testing
    from database import DatabaseManager
    from sample_data import generate_risk_data
    
    # Initialize components
    db_manager = DatabaseManager()
    
    # Generate and insert sample data for testing
    sample_data = generate_risk_data(days=60)
    db_manager.risk_data.insert_risk_data(sample_data)
    
    # Initialize predictor
    predictor = RiskPredictor(db_manager)
    
    # Train models
    print("Training Random Forest model...")
    rf_result = predictor.train_risk_prediction_model('random_forest')
    print(f"Random Forest training result: {rf_result}")
    
    print("\nTraining Linear Regression model...")
    lr_result = predictor.train_risk_prediction_model('linear_regression')
    print(f"Linear Regression training result: {lr_result}")
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict_future_risks(days_ahead=7)
    print(f"Generated {len(predictions)} predictions")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    anomalies = predictor.detect_anomalies()
    print(f"Detected {len(anomalies)} anomalies")
    
    # Generate insights
    print("\nGenerating insights...")
    insight_generator = RiskInsightGenerator(predictor)
    summary = insight_generator.generate_risk_summary()
    print(f"Generated {len(summary['insights'])} insights and {len(summary['recommendations'])} recommendations")