"""
Federated Learning Framework for Privacy-Preserving Risk Assessment
Implements federated learning to train models without sharing raw data
"""

import asyncio
import json
import time
import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedModelType(Enum):
    """Types of federated models"""
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"

class ClientStatus(Enum):
    """Federated learning client status"""
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    ERROR = "error"
    DISCONNECTED = "disconnected"

class AggregationStrategy(Enum):
    """Model aggregation strategies"""
    FEDERATED_AVERAGING = "federated_averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"

@dataclass
class ModelUpdate:
    """Model update from federated client"""
    client_id: str
    round_number: int
    model_weights: Dict[str, np.ndarray]
    data_size: int
    training_loss: float
    validation_accuracy: float
    training_time: float
    timestamp: datetime
    privacy_budget: float = 1.0
    noise_scale: float = 0.0

@dataclass
class FederatedClient:
    """Federated learning client information"""
    client_id: str
    client_name: str
    data_size: int
    capabilities: List[str]
    last_seen: datetime
    status: ClientStatus
    total_rounds: int = 0
    avg_training_time: float = 0.0
    contribution_score: float = 1.0
    privacy_level: str = "standard"
    location: str = "unknown"

@dataclass
class TrainingRound:
    """Information about a federated training round"""
    round_number: int
    start_time: datetime
    end_time: Optional[datetime]
    participating_clients: List[str]
    global_model_accuracy: float
    convergence_metric: float
    aggregation_strategy: AggregationStrategy
    privacy_budget_used: float
    model_version: str

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for federated learning"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class PrivacyEngine:
    """Handles privacy-preserving mechanisms"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta
        self.noise_multiplier = self._calculate_noise_multiplier()
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier for differential privacy"""
        # Simplified calculation - in practice, would use more sophisticated methods
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_gaussian_noise(self, gradients: Dict[str, np.ndarray], 
                          sensitivity: float = 1.0) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to gradients for differential privacy"""
        noisy_gradients = {}
        
        for layer_name, gradient in gradients.items():
            noise = np.random.normal(
                0, 
                self.noise_multiplier * sensitivity, 
                gradient.shape
            )
            noisy_gradients[layer_name] = gradient + noise
        
        return noisy_gradients
    
    def clip_gradients(self, gradients: Dict[str, np.ndarray], 
                      max_norm: float = 1.0) -> Dict[str, np.ndarray]:
        """Clip gradients to bound sensitivity"""
        clipped_gradients = {}
        
        # Calculate global norm
        total_norm = 0
        for gradient in gradients.values():
            total_norm += np.sum(gradient ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            for layer_name, gradient in gradients.items():
                clipped_gradients[layer_name] = gradient * clip_factor
        else:
            clipped_gradients = gradients.copy()
        
        return clipped_gradients
    
    def secure_aggregation_mask(self, model_weights: Dict[str, np.ndarray], 
                               client_id: str) -> Dict[str, np.ndarray]:
        """Create secure aggregation mask"""
        masked_weights = {}
        
        # Generate deterministic mask based on client ID
        np.random.seed(hash(client_id) % (2**32))
        
        for layer_name, weights in model_weights.items():
            mask = np.random.normal(0, 0.1, weights.shape)
            masked_weights[layer_name] = weights + mask
        
        return masked_weights

class FederatedLearningClient:
    """Simulated federated learning client"""
    
    def __init__(self, client_id: str, client_name: str, 
                 data: pd.DataFrame, privacy_level: str = "standard"):
        self.client_id = client_id
        self.client_name = client_name
        self.data = data
        self.privacy_level = privacy_level
        self.model = None
        self.scaler = StandardScaler()
        self.privacy_engine = PrivacyEngine(
            epsilon=2.0 if privacy_level == "high" else 5.0,
            delta=1e-5
        )
        
        # Client statistics
        self.total_rounds = 0
        self.avg_training_time = 0.0
        self.last_accuracy = 0.0
        
    def prepare_data(self, feature_columns: List[str], target_column: str):
        """Prepare client data for training"""
        try:
            # Extract features and target
            X = self.data[feature_columns].fillna(0)
            y = self.data[target_column].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y.values
            
        except Exception as e:
            logger.error(f"Data preparation error for client {self.client_id}: {e}")
            return None, None
    
    def train_local_model(self, global_model_weights: Dict[str, np.ndarray],
                         model_type: FederatedModelType,
                         feature_columns: List[str], target_column: str,
                         epochs: int = 5) -> Optional[ModelUpdate]:
        """Train local model on client data"""
        try:
            start_time = time.time()
            
            # Prepare data
            X, y = self.prepare_data(feature_columns, target_column)
            if X is None or len(X) == 0:
                return None
            
            # Initialize model with global weights
            if model_type == FederatedModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(max_iter=100)
                model.fit(X, y)
                
                # Extract weights
                model_weights = {
                    'coef_': model.coef_,
                    'intercept_': model.intercept_
                }
                
                # Calculate metrics
                y_pred = model.predict(X)
                training_loss = -model.score(X, y)
                validation_accuracy = accuracy_score(y, y_pred)
                
            elif model_type == FederatedModelType.LINEAR_REGRESSION:
                model = LinearRegression()
                model.fit(X, y)
                
                model_weights = {
                    'coef_': model.coef_.reshape(-1, 1) if model.coef_.ndim == 1 else model.coef_,
                    'intercept_': np.array([model.intercept_])
                }
                
                y_pred = model.predict(X)
                training_loss = mean_squared_error(y, y_pred)
                validation_accuracy = model.score(X, y)
                
            elif model_type == FederatedModelType.NEURAL_NETWORK:
                # PyTorch neural network training
                model_weights, training_loss, validation_accuracy = self._train_neural_network(
                    X, y, global_model_weights, epochs
                )
                
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
            
            # Apply privacy mechanisms
            if self.privacy_level == "high":
                model_weights = self.privacy_engine.clip_gradients(model_weights)
                model_weights = self.privacy_engine.add_gaussian_noise(model_weights)
            
            training_time = time.time() - start_time
            
            # Update client statistics
            self.total_rounds += 1
            self.avg_training_time = (
                (self.avg_training_time * (self.total_rounds - 1) + training_time) / 
                self.total_rounds
            )
            self.last_accuracy = validation_accuracy
            
            # Create model update
            update = ModelUpdate(
                client_id=self.client_id,
                round_number=self.total_rounds,
                model_weights=model_weights,
                data_size=len(X),
                training_loss=training_loss,
                validation_accuracy=validation_accuracy,
                training_time=training_time,
                timestamp=datetime.now(),
                privacy_budget=self.privacy_engine.epsilon,
                noise_scale=self.privacy_engine.noise_multiplier
            )
            
            logger.info(f"Client {self.client_id} completed training round {self.total_rounds}")
            return update
            
        except Exception as e:
            logger.error(f"Training error for client {self.client_id}: {e}")
            return None
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray, 
                            global_weights: Dict[str, np.ndarray],
                            epochs: int) -> Tuple[Dict[str, np.ndarray], float, float]:
        """Train neural network locally"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        model = SimpleNeuralNetwork(X.shape[1])
        
        # Load global weights if available
        if global_weights and 'fc1.weight' in global_weights:
            state_dict = {}
            for key, value in global_weights.items():
                state_dict[key] = torch.FloatTensor(value)
            model.load_state_dict(state_dict, strict=False)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        # Extract model weights
        model_weights = {}
        for name, param in model.named_parameters():
            model_weights[name] = param.data.numpy()
        
        # Calculate validation accuracy
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)
            val_loss = criterion(predictions, y_tensor).item()
            validation_accuracy = 1.0 / (1.0 + val_loss)  # Simplified accuracy metric
        
        avg_training_loss = total_loss / max(num_batches, 1)
        
        return model_weights, avg_training_loss, validation_accuracy

class FederatedLearningServer:
    """Federated learning central server"""
    
    def __init__(self, model_type: FederatedModelType, 
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING):
        self.model_type = model_type
        self.aggregation_strategy = aggregation_strategy
        self.global_model_weights: Dict[str, np.ndarray] = {}
        self.clients: Dict[str, FederatedClient] = {}
        self.training_rounds: List[TrainingRound] = []
        self.current_round = 0
        
        # Training configuration
        self.min_clients_per_round = 2
        self.max_rounds = 50
        self.convergence_threshold = 0.01
        self.client_selection_fraction = 0.8
        
        # Privacy configuration
        self.privacy_budget_per_round = 0.1
        self.total_privacy_budget = 5.0
        self.used_privacy_budget = 0.0
        
        # Performance tracking
        self.global_accuracy_history = []
        self.convergence_history = []
        
    def register_client(self, client_info: FederatedClient):
        """Register a new federated learning client"""
        self.clients[client_info.client_id] = client_info
        logger.info(f"Client {client_info.client_id} registered")
    
    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for training round"""
        available_clients = [
            client_id for client_id, client in self.clients.items()
            if client.status in [ClientStatus.IDLE]
        ]
        
        if len(available_clients) < self.min_clients_per_round:
            logger.warning(f"Not enough available clients: {len(available_clients)}")
            return []
        
        # Select fraction of available clients
        num_selected = max(
            self.min_clients_per_round,
            int(len(available_clients) * self.client_selection_fraction)
        )
        
        # Weighted selection based on contribution scores
        client_weights = [
            self.clients[client_id].contribution_score 
            for client_id in available_clients
        ]
        
        # Normalize weights
        total_weight = sum(client_weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in client_weights]
        else:
            probabilities = [1.0 / len(available_clients)] * len(available_clients)
        
        # Select clients
        selected_indices = np.random.choice(
            len(available_clients),
            size=min(num_selected, len(available_clients)),
            replace=False,
            p=probabilities
        )
        
        selected_clients = [available_clients[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_clients)} clients for round {round_number}")
        return selected_clients
    
    def aggregate_model_updates(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate model updates from clients"""
        if not updates:
            return self.global_model_weights
        
        try:
            if self.aggregation_strategy == AggregationStrategy.FEDERATED_AVERAGING:
                return self._federated_averaging(updates)
            elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGING:
                return self._weighted_averaging(updates)
            elif self.aggregation_strategy == AggregationStrategy.SECURE_AGGREGATION:
                return self._secure_aggregation(updates)
            elif self.aggregation_strategy == AggregationStrategy.DIFFERENTIAL_PRIVACY:
                return self._differential_privacy_aggregation(updates)
            else:
                logger.error(f"Unknown aggregation strategy: {self.aggregation_strategy}")
                return self.global_model_weights
                
        except Exception as e:
            logger.error(f"Model aggregation error: {e}")
            return self.global_model_weights
    
    def _federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Federated averaging aggregation"""
        if not updates:
            return {}
        
        # Get all parameter names
        all_params = set()
        for update in updates:
            all_params.update(update.model_weights.keys())
        
        aggregated_weights = {}
        total_samples = sum(update.data_size for update in updates)
        
        if total_samples == 0:
            return self.global_model_weights
        
        # Aggregate each parameter
        for param_name in all_params:
            weighted_sum = None
            
            for update in updates:
                if param_name in update.model_weights:
                    weight = update.data_size / total_samples
                    param_contrib = update.model_weights[param_name] * weight
                    
                    if weighted_sum is None:
                        weighted_sum = param_contrib
                    else:
                        weighted_sum += param_contrib
            
            if weighted_sum is not None:
                aggregated_weights[param_name] = weighted_sum
        
        return aggregated_weights
    
    def _weighted_averaging(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Weighted averaging based on client performance"""
        if not updates:
            return {}
        
        # Calculate weights based on validation accuracy and data size
        weights = []
        for update in updates:
            accuracy_weight = max(0.1, update.validation_accuracy)  # Minimum weight
            size_weight = np.log(max(1, update.data_size))
            combined_weight = accuracy_weight * size_weight
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return self._federated_averaging(updates)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Aggregate parameters
        all_params = set()
        for update in updates:
            all_params.update(update.model_weights.keys())
        
        aggregated_weights = {}
        
        for param_name in all_params:
            weighted_sum = None
            
            for i, update in enumerate(updates):
                if param_name in update.model_weights:
                    param_contrib = update.model_weights[param_name] * normalized_weights[i]
                    
                    if weighted_sum is None:
                        weighted_sum = param_contrib
                    else:
                        weighted_sum += param_contrib
            
            if weighted_sum is not None:
                aggregated_weights[param_name] = weighted_sum
        
        return aggregated_weights
    
    def _secure_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Secure aggregation with noise addition"""
        # Start with federated averaging
        aggregated_weights = self._federated_averaging(updates)
        
        # Add security noise
        privacy_engine = PrivacyEngine(epsilon=self.privacy_budget_per_round)
        
        for param_name in aggregated_weights:
            noise = np.random.normal(
                0, 
                privacy_engine.noise_multiplier * 0.1,
                aggregated_weights[param_name].shape
            )
            aggregated_weights[param_name] += noise
        
        return aggregated_weights
    
    def _differential_privacy_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Differential privacy aggregation"""
        # Clip and add noise to individual updates first
        privacy_engine = PrivacyEngine(epsilon=self.privacy_budget_per_round)
        
        noisy_updates = []
        for update in updates:
            clipped_weights = privacy_engine.clip_gradients(
                update.model_weights, max_norm=1.0
            )
            noisy_weights = privacy_engine.add_gaussian_noise(
                clipped_weights, sensitivity=1.0
            )
            
            # Create new update with noisy weights
            noisy_update = ModelUpdate(
                client_id=update.client_id,
                round_number=update.round_number,
                model_weights=noisy_weights,
                data_size=update.data_size,
                training_loss=update.training_loss,
                validation_accuracy=update.validation_accuracy,
                training_time=update.training_time,
                timestamp=update.timestamp,
                privacy_budget=update.privacy_budget,
                noise_scale=privacy_engine.noise_multiplier
            )
            noisy_updates.append(noisy_update)
        
        # Then aggregate
        return self._federated_averaging(noisy_updates)
    
    def evaluate_global_model(self, test_data: pd.DataFrame, 
                            feature_columns: List[str], target_column: str) -> Dict[str, float]:
        """Evaluate global model performance"""
        try:
            if not self.global_model_weights:
                return {"error": "No global model available"}
            
            # Prepare test data
            X = test_data[feature_columns].fillna(0)
            y = test_data[target_column].fillna(0)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Evaluate based on model type
            if self.model_type == FederatedModelType.LOGISTIC_REGRESSION:
                # Simulate logistic regression prediction
                if 'coef_' in self.global_model_weights and 'intercept_' in self.global_model_weights:
                    linear_combo = np.dot(X_scaled, self.global_model_weights['coef_'].T) + self.global_model_weights['intercept_']
                    predictions = 1 / (1 + np.exp(-linear_combo.flatten()))
                    pred_classes = (predictions > 0.5).astype(int)
                    
                    accuracy = accuracy_score(y, pred_classes)
                    f1 = f1_score(y, pred_classes, average='weighted')
                    
                    return {
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "predictions_mean": float(np.mean(predictions))
                    }
            
            elif self.model_type == FederatedModelType.LINEAR_REGRESSION:
                if 'coef_' in self.global_model_weights and 'intercept_' in self.global_model_weights:
                    predictions = np.dot(X_scaled, self.global_model_weights['coef_'].flatten()) + self.global_model_weights['intercept_'][0]
                    
                    mse = mean_squared_error(y, predictions)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y - predictions))
                    
                    return {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "predictions_mean": float(np.mean(predictions))
                    }
            
            elif self.model_type == FederatedModelType.NEURAL_NETWORK:
                # PyTorch evaluation
                model = SimpleNeuralNetwork(X_scaled.shape[1])
                
                # Load weights
                state_dict = {}
                for key, value in self.global_model_weights.items():
                    state_dict[key] = torch.FloatTensor(value)
                model.load_state_dict(state_dict, strict=False)
                
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    predictions = model(X_tensor).numpy().flatten()
                    
                    mse = mean_squared_error(y, predictions)
                    rmse = np.sqrt(mse)
                    
                    return {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "predictions_mean": float(np.mean(predictions))
                    }
            
            return {"error": "Unsupported model type for evaluation"}
            
        except Exception as e:
            logger.error(f"Global model evaluation error: {e}")
            return {"error": str(e)}
    
    def check_convergence(self, current_accuracy: float) -> bool:
        """Check if the model has converged"""
        if len(self.global_accuracy_history) < 3:
            return False
        
        # Check if improvement is below threshold for last few rounds
        recent_accuracies = self.global_accuracy_history[-3:]
        improvements = [
            abs(recent_accuracies[i] - recent_accuracies[i-1]) 
            for i in range(1, len(recent_accuracies))
        ]
        
        avg_improvement = np.mean(improvements)
        return avg_improvement < self.convergence_threshold
    
    def update_client_contributions(self, updates: List[ModelUpdate]):
        """Update client contribution scores based on performance"""
        for update in updates:
            if update.client_id in self.clients:
                client = self.clients[update.client_id]
                
                # Calculate contribution score based on multiple factors
                accuracy_factor = max(0.1, update.validation_accuracy)
                data_factor = np.log(max(1, update.data_size)) / 10
                time_factor = max(0.1, 1.0 / max(1, update.training_time))
                
                contribution_score = (accuracy_factor + data_factor + time_factor) / 3
                
                # Exponential moving average
                alpha = 0.3
                client.contribution_score = (
                    alpha * contribution_score + 
                    (1 - alpha) * client.contribution_score
                )
                
                client.last_seen = datetime.now()
                client.total_rounds += 1
                client.avg_training_time = (
                    (client.avg_training_time * (client.total_rounds - 1) + update.training_time) /
                    client.total_rounds
                )

class FederatedLearningCoordinator:
    """Coordinates federated learning training"""
    
    def __init__(self):
        self.server = None
        self.clients: Dict[str, FederatedLearningClient] = {}
        self.training_active = False
        self.training_thread = None
        
    def initialize_federation(self, model_type: FederatedModelType,
                            aggregation_strategy: AggregationStrategy,
                            client_data: Dict[str, pd.DataFrame]):
        """Initialize federated learning setup"""
        try:
            # Create server
            self.server = FederatedLearningServer(model_type, aggregation_strategy)
            
            # Create and register clients
            for client_id, data in client_data.items():
                # Determine privacy level based on data sensitivity
                privacy_level = "high" if len(data) < 100 else "standard"
                
                client = FederatedLearningClient(
                    client_id=client_id,
                    client_name=f"Client_{client_id}",
                    data=data,
                    privacy_level=privacy_level
                )
                
                self.clients[client_id] = client
                
                # Register with server
                client_info = FederatedClient(
                    client_id=client_id,
                    client_name=client.client_name,
                    data_size=len(data),
                    capabilities=["training", "evaluation"],
                    last_seen=datetime.now(),
                    status=ClientStatus.IDLE,
                    privacy_level=privacy_level
                )
                
                self.server.register_client(client_info)
            
            logger.info(f"Federated learning initialized with {len(self.clients)} clients")
            return True
            
        except Exception as e:
            logger.error(f"Federation initialization error: {e}")
            return False
    
    def start_training(self, feature_columns: List[str], target_column: str,
                      max_rounds: int = 10, test_data: pd.DataFrame = None):
        """Start federated training process"""
        if not self.server or not self.clients:
            logger.error("Federation not initialized")
            return False
        
        self.training_active = True
        
        def training_loop():
            for round_num in range(1, max_rounds + 1):
                if not self.training_active:
                    break
                
                try:
                    logger.info(f"Starting federated learning round {round_num}")
                    
                    # Select clients for this round
                    selected_client_ids = self.server.select_clients(round_num)
                    
                    if len(selected_client_ids) < self.server.min_clients_per_round:
                        logger.warning(f"Not enough clients for round {round_num}")
                        continue
                    
                    # Start training round
                    round_start = datetime.now()
                    updates = []
                    
                    # Train on selected clients
                    for client_id in selected_client_ids:
                        if client_id in self.clients:
                            client = self.clients[client_id]
                            
                            # Update client status
                            self.server.clients[client_id].status = ClientStatus.TRAINING
                            
                            # Train local model
                            update = client.train_local_model(
                                self.server.global_model_weights,
                                self.server.model_type,
                                feature_columns,
                                target_column,
                                epochs=5
                            )
                            
                            if update:
                                updates.append(update)
                                self.server.clients[client_id].status = ClientStatus.IDLE
                            else:
                                self.server.clients[client_id].status = ClientStatus.ERROR
                    
                    if not updates:
                        logger.warning(f"No successful updates in round {round_num}")
                        continue
                    
                    # Aggregate model updates
                    logger.info(f"Aggregating {len(updates)} model updates")
                    self.server.global_model_weights = self.server.aggregate_model_updates(updates)
                    
                    # Evaluate global model
                    global_metrics = {}
                    if test_data is not None and not test_data.empty:
                        global_metrics = self.server.evaluate_global_model(
                            test_data, feature_columns, target_column
                        )
                    
                    # Update client contributions
                    self.server.update_client_contributions(updates)
                    
                    # Record training round
                    round_end = datetime.now()
                    
                    current_accuracy = global_metrics.get('accuracy', 0.0)
                    if 'mse' in global_metrics:
                        # For regression, use inverse of RMSE as accuracy metric
                        current_accuracy = 1.0 / (1.0 + global_metrics.get('rmse', 1.0))
                    
                    training_round = TrainingRound(
                        round_number=round_num,
                        start_time=round_start,
                        end_time=round_end,
                        participating_clients=selected_client_ids,
                        global_model_accuracy=current_accuracy,
                        convergence_metric=abs(current_accuracy - self.server.global_accuracy_history[-1]) if self.server.global_accuracy_history else 1.0,
                        aggregation_strategy=self.server.aggregation_strategy,
                        privacy_budget_used=self.server.privacy_budget_per_round,
                        model_version=f"v{round_num}"
                    )
                    
                    self.server.training_rounds.append(training_round)
                    self.server.global_accuracy_history.append(current_accuracy)
                    self.server.current_round = round_num
                    
                    # Update privacy budget
                    self.server.used_privacy_budget += self.server.privacy_budget_per_round
                    
                    logger.info(f"Round {round_num} completed. Global accuracy: {current_accuracy:.4f}")
                    
                    # Check convergence
                    if self.server.check_convergence(current_accuracy):
                        logger.info(f"Model converged after {round_num} rounds")
                        break
                    
                    # Check privacy budget
                    if self.server.used_privacy_budget >= self.server.total_privacy_budget:
                        logger.info("Privacy budget exhausted, stopping training")
                        break
                    
                    # Wait between rounds
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in training round {round_num}: {e}")
                    continue
            
            self.training_active = False
            logger.info("Federated learning training completed")
        
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        
        return True
    
    def stop_training(self):
        """Stop federated training"""
        self.training_active = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
        
        logger.info("Federated learning training stopped")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        if not self.server:
            return {"error": "Federation not initialized"}
        
        status = {
            "training_active": self.training_active,
            "current_round": self.server.current_round,
            "total_clients": len(self.server.clients),
            "active_clients": len([
                c for c in self.server.clients.values() 
                if c.status == ClientStatus.IDLE
            ]),
            "global_accuracy": self.server.global_accuracy_history[-1] if self.server.global_accuracy_history else 0.0,
            "privacy_budget_used": self.server.used_privacy_budget,
            "privacy_budget_remaining": self.server.total_privacy_budget - self.server.used_privacy_budget,
            "aggregation_strategy": self.server.aggregation_strategy.value,
            "model_type": self.server.model_type.value,
            "convergence_status": "converged" if len(self.server.global_accuracy_history) > 0 and self.server.check_convergence(self.server.global_accuracy_history[-1]) else "training"
        }
        
        return status

# Streamlit Integration Functions

def initialize_federated_learning():
    """Initialize federated learning system"""
    if 'fl_coordinator' not in st.session_state:
        st.session_state.fl_coordinator = FederatedLearningCoordinator()
    
    return st.session_state.fl_coordinator

def render_federated_learning_dashboard():
    """Render federated learning dashboard"""
    st.header("🤝 Federated Learning")
    
    coordinator = initialize_federated_learning()
    
    # Check if federation is initialized
    if not coordinator.server:
        st.warning("⚠️ Federated learning not initialized. Please set up the federation first.")
        
        with st.expander("Initialize Federated Learning"):
            # Configuration options
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Type",
                    [t.value for t in FederatedModelType]
                )
            
            with col2:
                aggregation_strategy = st.selectbox(
                    "Aggregation Strategy",
                    [s.value for s in AggregationStrategy]
                )
            
            # Generate sample federated data
            if st.button("🚀 Initialize with Sample Data"):
                with st.spinner("Creating federated dataset..."):
                    # Generate sample data for multiple clients
                    np.random.seed(42)
                    
                    client_data = {}
                    base_features = ['risk_score', 'confidence', 'sample_size', 'model_age']
                    
                    for i in range(5):  # 5 simulated clients
                        client_id = f"client_{i+1}"
                        
                        # Generate data with some variation per client
                        n_samples = np.random.randint(50, 200)
                        
                        data = pd.DataFrame({
                            'risk_score': np.random.beta(2, 5, n_samples),
                            'confidence': np.random.beta(8, 2, n_samples),
                            'sample_size': np.random.randint(10, 1000, n_samples),
                            'model_age': np.random.exponential(2, n_samples),
                            'target': np.random.binomial(1, 0.3, n_samples)  # Binary target
                        })
                        
                        # Add client-specific bias
                        data['risk_score'] += np.random.normal(0, 0.1, n_samples)
                        data['confidence'] += np.random.normal(0, 0.05, n_samples)
                        
                        # Clip values to valid ranges
                        data['risk_score'] = np.clip(data['risk_score'], 0, 1)
                        data['confidence'] = np.clip(data['confidence'], 0, 1)
                        
                        client_data[client_id] = data
                    
                    # Initialize federation
                    success = coordinator.initialize_federation(
                        FederatedModelType(model_type),
                        AggregationStrategy(aggregation_strategy),
                        client_data
                    )
                    
                    if success:
                        st.success("✅ Federated learning initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to initialize federated learning")
        
        return
    
    # Training status overview
    status = coordinator.get_training_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", status['total_clients'])
    
    with col2:
        st.metric("Active Clients", status['active_clients'])
    
    with col3:
        st.metric("Current Round", status['current_round'])
    
    with col4:
        accuracy = status['global_accuracy']
        st.metric("Global Accuracy", f"{accuracy:.4f}")
    
    # Training status indicator
    if status['training_active']:
        st.success("🟢 Training Active")
    else:
        st.info("⚪ Training Idle")
    
    # Privacy budget indicator
    privacy_remaining = status['privacy_budget_remaining']
    privacy_used = status['privacy_budget_used']
    privacy_total = privacy_used + privacy_remaining
    
    if privacy_total > 0:
        privacy_percentage = (privacy_used / privacy_total) * 100
        st.progress(privacy_percentage / 100)
        st.write(f"Privacy Budget Used: {privacy_used:.2f} / {privacy_total:.2f} ({privacy_percentage:.1f}%)")
    
    # Tabs for different aspects
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Training Control",
        "👥 Client Management", 
        "📊 Performance Analytics",
        "🔒 Privacy & Security",
        "⚙️ Configuration"
    ])
    
    with tab1:
        st.subheader("Training Control")
        
        # Training configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            feature_columns = st.multiselect(
                "Feature Columns",
                ['risk_score', 'confidence', 'sample_size', 'model_age'],
                default=['risk_score', 'confidence', 'sample_size']
            )
        
        with col2:
            target_column = st.selectbox(
                "Target Column",
                ['target'],
                index=0
            )
        
        with col3:
            max_rounds = st.slider("Max Training Rounds", 5, 50, 10)
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not status['training_active']:
                if st.button("▶️ Start Training"):
                    if feature_columns and target_column:
                        success = coordinator.start_training(
                            feature_columns, target_column, max_rounds
                        )
                        if success:
                            st.success("Training started!")
                            st.rerun()
                        else:
                            st.error("Failed to start training")
                    else:
                        st.error("Please select features and target")
            else:
                if st.button("⏹️ Stop Training"):
                    coordinator.stop_training()
                    st.success("Training stopped!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset Federation"):
                coordinator.server = None
                coordinator.clients.clear()
                st.success("Federation reset!")
                st.rerun()
        
        with col3:
            if st.button("📊 Refresh Status"):
                st.rerun()
        
        # Training progress
        if coordinator.server and coordinator.server.global_accuracy_history:
            st.subheader("Training Progress")
            
            # Accuracy over rounds
            accuracy_data = []
            for i, acc in enumerate(coordinator.server.global_accuracy_history):
                accuracy_data.append({
                    'Round': i + 1,
                    'Global Accuracy': acc
                })
            
            if accuracy_data:
                accuracy_df = pd.DataFrame(accuracy_data)
                
                fig_accuracy = px.line(
                    accuracy_df, 
                    x='Round', 
                    y='Global Accuracy',
                    title='Global Model Accuracy Over Training Rounds'
                )
                st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Recent training rounds
        if coordinator.server and coordinator.server.training_rounds:
            st.subheader("Recent Training Rounds")
            
            recent_rounds = coordinator.server.training_rounds[-5:]  # Last 5 rounds
            
            round_data = []
            for round_info in recent_rounds:
                round_data.append({
                    'Round': round_info.round_number,
                    'Participants': len(round_info.participating_clients),
                    'Accuracy': f"{round_info.global_model_accuracy:.4f}",
                    'Duration': f"{(round_info.end_time - round_info.start_time).total_seconds():.1f}s" if round_info.end_time else "N/A",
                    'Privacy Budget': f"{round_info.privacy_budget_used:.3f}"
                })
            
            round_df = pd.DataFrame(round_data)
            st.dataframe(round_df, use_container_width=True)
    
    with tab2:
        st.subheader("Client Management")
        
        if coordinator.server and coordinator.server.clients:
            # Client overview
            client_data = []
            for client_id, client_info in coordinator.server.clients.items():
                status_icons = {
                    ClientStatus.IDLE: "🟢",
                    ClientStatus.TRAINING: "🟡",
                    ClientStatus.UPLOADING: "🔵",
                    ClientStatus.ERROR: "🔴",
                    ClientStatus.DISCONNECTED: "⚫"
                }
                
                client_data.append({
                    'Client ID': client_id,
                    'Status': f"{status_icons.get(client_info.status, '⚪')} {client_info.status.value}",
                    'Data Size': client_info.data_size,
                    'Total Rounds': client_info.total_rounds,
                    'Avg Training Time': f"{client_info.avg_training_time:.2f}s",
                    'Contribution Score': f"{client_info.contribution_score:.3f}",
                    'Privacy Level': client_info.privacy_level,
                    'Last Seen': client_info.last_seen.strftime('%H:%M:%S')
                })
            
            client_df = pd.DataFrame(client_data)
            st.dataframe(client_df, use_container_width=True)
            
            # Client contribution chart
            if len(client_data) > 0:
                fig_contrib = px.bar(
                    client_df,
                    x='Client ID',
                    y='Contribution Score',
                    title='Client Contribution Scores',
                    color='Contribution Score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            st.info("No clients registered")
    
    with tab3:
        st.subheader("Performance Analytics")
        
        if coordinator.server and coordinator.server.training_rounds:
            # Performance metrics over time
            perf_data = []
            for round_info in coordinator.server.training_rounds:
                perf_data.append({
                    'Round': round_info.round_number,
                    'Accuracy': round_info.global_model_accuracy,
                    'Convergence Metric': round_info.convergence_metric,
                    'Participants': len(round_info.participating_clients),
                    'Privacy Budget Used': round_info.privacy_budget_used
                })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                
                # Accuracy and convergence
                fig_perf = go.Figure()
                
                fig_perf.add_trace(go.Scatter(
                    x=perf_df['Round'],
                    y=perf_df['Accuracy'],
                    mode='lines+markers',
                    name='Global Accuracy',
                    line=dict(color='blue')
                ))
                
                fig_perf.add_trace(go.Scatter(
                    x=perf_df['Round'],
                    y=perf_df['Convergence Metric'],
                    mode='lines+markers',
                    name='Convergence Metric',
                    yaxis='y2',
                    line=dict(color='red')
                ))
                
                fig_perf.update_layout(
                    title='Training Performance Metrics',
                    xaxis_title='Training Round',
                    yaxis=dict(title='Global Accuracy', side='left'),
                    yaxis2=dict(title='Convergence Metric', side='right', overlaying='y'),
                    legend=dict(x=0.7, y=0.9)
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Participation analysis
                fig_participation = px.bar(
                    perf_df,
                    x='Round',
                    y='Participants',
                    title='Client Participation per Round'
                )
                st.plotly_chart(fig_participation, use_container_width=True)
        else:
            st.info("No performance data available. Start training to see analytics.")
    
    with tab4:
        st.subheader("Privacy & Security")
        
        # Privacy budget tracking
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Privacy Budget Status:**")
            st.write(f"Used: {status['privacy_budget_used']:.3f}")
            st.write(f"Remaining: {status['privacy_budget_remaining']:.3f}")
            st.write(f"Total: {status['privacy_budget_used'] + status['privacy_budget_remaining']:.3f}")
        
        with col2:
            st.write("**Privacy Configuration:**")
            st.write(f"Aggregation Strategy: {status['aggregation_strategy']}")
            
            # Privacy level distribution
            if coordinator.server:
                privacy_levels = {}
                for client in coordinator.server.clients.values():
                    level = client.privacy_level
                    privacy_levels[level] = privacy_levels.get(level, 0) + 1
                
                for level, count in privacy_levels.items():
                    st.write(f"{level.title()} Privacy: {count} clients")
        
        # Privacy budget usage over time
        if coordinator.server and coordinator.server.training_rounds:
            privacy_data = []
            cumulative_budget = 0
            
            for round_info in coordinator.server.training_rounds:
                cumulative_budget += round_info.privacy_budget_used
                privacy_data.append({
                    'Round': round_info.round_number,
                    'Cumulative Budget Used': cumulative_budget,
                    'Round Budget Used': round_info.privacy_budget_used
                })
            
            if privacy_data:
                privacy_df = pd.DataFrame(privacy_data)
                
                fig_privacy = px.line(
                    privacy_df,
                    x='Round',
                    y='Cumulative Budget Used',
                    title='Privacy Budget Usage Over Time'
                )
                
                # Add total budget line
                total_budget = coordinator.server.total_privacy_budget
                fig_privacy.add_hline(
                    y=total_budget,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Total Budget Limit"
                )
                
                st.plotly_chart(fig_privacy, use_container_width=True)
        
        # Security recommendations
        st.subheader("Security Recommendations")
        
        recommendations = []
        
        if status['privacy_budget_remaining'] < 1.0:
            recommendations.append("⚠️ Privacy budget running low - consider reducing noise or stopping training")
        
        if status['active_clients'] < 3:
            recommendations.append("⚠️ Low client participation may reduce privacy guarantees")
        
        if status['aggregation_strategy'] == 'federated_averaging':
            recommendations.append("💡 Consider using secure aggregation for enhanced privacy")
        
        if not recommendations:
            recommendations.append("✅ No immediate security concerns detected")
        
        for rec in recommendations:
            st.write(rec)
    
    with tab5:
        st.subheader("Configuration")
        
        if coordinator.server:
            # Display current configuration
            st.write("**Current Configuration:**")
            st.write(f"Model Type: {coordinator.server.model_type.value}")
            st.write(f"Aggregation Strategy: {coordinator.server.aggregation_strategy.value}")
            st.write(f"Min Clients per Round: {coordinator.server.min_clients_per_round}")
            st.write(f"Max Rounds: {coordinator.server.max_rounds}")
            st.write(f"Client Selection Fraction: {coordinator.server.client_selection_fraction}")
            st.write(f"Convergence Threshold: {coordinator.server.convergence_threshold}")
            
            # Configuration updates
            st.subheader("Update Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_min_clients = st.slider(
                    "Min Clients per Round",
                    1, 10, coordinator.server.min_clients_per_round
                )
                
                new_selection_fraction = st.slider(
                    "Client Selection Fraction",
                    0.1, 1.0, coordinator.server.client_selection_fraction
                )
            
            with col2:
                new_convergence_threshold = st.slider(
                    "Convergence Threshold",
                    0.001, 0.1, coordinator.server.convergence_threshold
                )
                
                new_privacy_budget_per_round = st.slider(
                    "Privacy Budget per Round",
                    0.01, 0.5, coordinator.server.privacy_budget_per_round
                )
            
            if st.button("Update Configuration"):
                coordinator.server.min_clients_per_round = new_min_clients
                coordinator.server.client_selection_fraction = new_selection_fraction
                coordinator.server.convergence_threshold = new_convergence_threshold
                coordinator.server.privacy_budget_per_round = new_privacy_budget_per_round
                
                st.success("Configuration updated!")
        
        # Export/Import settings
        st.subheader("Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Configuration"):
                if coordinator.server:
                    config = {
                        "model_type": coordinator.server.model_type.value,
                        "aggregation_strategy": coordinator.server.aggregation_strategy.value,
                        "min_clients_per_round": coordinator.server.min_clients_per_round,
                        "max_rounds": coordinator.server.max_rounds,
                        "client_selection_fraction": coordinator.server.client_selection_fraction,
                        "convergence_threshold": coordinator.server.convergence_threshold,
                        "privacy_budget_per_round": coordinator.server.privacy_budget_per_round,
                        "total_privacy_budget": coordinator.server.total_privacy_budget
                    }
                    
                    config_json = json.dumps(config, indent=2)
                    
                    st.download_button(
                        label="Download Configuration",
                        data=config_json,
                        file_name=f"fl_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json'
                    )
        
        with col2:
            if st.button("📊 Export Training Results"):
                if coordinator.server and coordinator.server.training_rounds:
                    results_data = []
                    
                    for round_info in coordinator.server.training_rounds:
                        results_data.append({
                            'Round': round_info.round_number,
                            'Start Time': round_info.start_time.isoformat(),
                            'End Time': round_info.end_time.isoformat() if round_info.end_time else None,
                            'Participants': ','.join(round_info.participating_clients),
                            'Global Accuracy': round_info.global_model_accuracy,
                            'Convergence Metric': round_info.convergence_metric,
                            'Aggregation Strategy': round_info.aggregation_strategy.value,
                            'Privacy Budget Used': round_info.privacy_budget_used,
                            'Model Version': round_info.model_version
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    csv = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Training Results",
                        data=csv,
                        file_name=f"fl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing federated learning framework...")
    
    # Generate sample federated data
    np.random.seed(42)
    client_data = {}
    
    for i in range(3):
        client_id = f"client_{i+1}"
        n_samples = np.random.randint(100, 300)
        
        data = pd.DataFrame({
            'risk_score': np.random.beta(2, 5, n_samples),
            'confidence': np.random.beta(8, 2, n_samples), 
            'sample_size': np.random.randint(10, 1000, n_samples),
            'model_age': np.random.exponential(2, n_samples),
            'target': np.random.binomial(1, 0.3, n_samples)
        })
        
        client_data[client_id] = data
    
    # Initialize coordinator
    coordinator = FederatedLearningCoordinator()
    
    # Initialize federation
    success = coordinator.initialize_federation(
        FederatedModelType.LOGISTIC_REGRESSION,
        AggregationStrategy.FEDERATED_AVERAGING,
        client_data
    )
    
    print(f"Federation initialized: {success}")
    
    if success:
        # Start training
        feature_columns = ['risk_score', 'confidence', 'sample_size']
        target_column = 'target'
        
        print("Starting federated training...")
        training_success = coordinator.start_training(
            feature_columns, target_column, max_rounds=5
        )
        
        if training_success:
            # Wait a bit for training to progress
            time.sleep(10)
            
            # Get status
            status = coordinator.get_training_status()
            print(f"Training status: {status}")
            
            # Stop training
            coordinator.stop_training()
            print("Training stopped")
    
    print("Federated learning framework test completed!")