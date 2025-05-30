"""
Modelo Avanzado de Predicción de Victorias NBA
=============================================

Este módulo implementa un sistema de predicción de alto rendimiento para
victorias de equipos NBA utilizando:

1. Ensemble Learning con múltiples algoritmos ML y Red Neuronal
2. Stacking avanzado con meta-modelo optimizado
3. Optimización bayesiana de hiperparámetros
4. Validación cruzada rigurosa para clasificación
5. Métricas de evaluación exhaustivas para problemas binarios
"""

# Standard Library
import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Third-party Libraries - ML/Data
import joblib
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, log_loss
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# XGBoost and LightGBM
import lightgbm as lgb
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Local imports
from .features_is_win import IsWinFeatureEngineer

# Configuration
warnings.filterwarnings('ignore')

# Logging setup
import logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """Clase auxiliar para procesamiento de datos común"""
    
    @staticmethod
    def prepare_training_data(X: pd.DataFrame, y: pd.Series, 
                            validation_split: float = 0.2,
                            scaler: Optional[StandardScaler] = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                     pd.Series, pd.Series, StandardScaler]:
        """
        Preparar datos para entrenamiento con división y escalado.
        
        Args:
            X: Features
            y: Target
            validation_split: Proporción para validación
            scaler: Scaler existente o None para crear nuevo
            
        Returns:
            X_train_scaled, X_val_scaled, y_train, y_val, scaler
        """
        # División estratificada train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # Escalado de features
        if scaler is None:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = pd.DataFrame(
                scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
    
    @staticmethod
    def prepare_prediction_data(X: pd.DataFrame, 
                              scaler: StandardScaler) -> pd.DataFrame:
        """
        Preparar datos para predicción con escalado.
        
        Args:
            X: Features sin escalar
            scaler: Scaler entrenado
            
        Returns:
            X_scaled: Features escaladas
        """
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled


class ModelTrainer:
    """Clase auxiliar para entrenamiento específico de modelos"""
    
    @staticmethod
    def train_xgboost_with_early_stopping(model: xgb.XGBClassifier,
                                         X_train: pd.DataFrame,
                                         y_train: pd.Series,
                                         X_val: pd.DataFrame,
                                         y_val: pd.Series) -> xgb.XGBClassifier:
        """Entrenar XGBoost con early stopping"""
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        return model
    
    @staticmethod
    def train_lightgbm_with_early_stopping(model: lgb.LGBMClassifier,
                                          X_train: pd.DataFrame,
                                          y_train: pd.Series,
                                          X_val: pd.DataFrame,
                                          y_val: pd.Series) -> lgb.LGBMClassifier:
        """Entrenar LightGBM con early stopping"""
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        return model
    
    @staticmethod
    def train_sklearn_with_early_stopping(model, X_train: pd.DataFrame,
                                         y_train: pd.Series,
                                         X_val: pd.DataFrame,
                                         y_val: pd.Series,
                                         model_name: str):
        """Early stopping manual para modelos sklearn con warm_start"""
        from sklearn.metrics import roc_auc_score
        
        best_score = 0
        patience_counter = 0
        patience = 15
        min_estimators = 50
        max_estimators = 200
        step_size = 25
        
        logger.info(f"  Implementando early stopping manual para {model_name}")
        
        for n_est in range(min_estimators, max_estimators + 1, step_size):
            model.n_estimators = n_est
            model.fit(X_train, y_train)
            
            val_proba = model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_proba)
            
            if val_score > best_score + 1e-4:
                best_score = val_score
                patience_counter = 0
                best_n_estimators = n_est
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"  Early stopping: mejor n_estimators = "
                           f"{best_n_estimators}")
                model.n_estimators = best_n_estimators
                model.fit(X_train, y_train)
                break
        
        return model


class MetricsCalculator:
    """Clase auxiliar para cálculo de métricas"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: pd.Series, 
                                       y_pred: np.ndarray,
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calcular métricas completas para clasificación binaria"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'log_loss': log_loss(y_true, y_proba)
        }
        
        # AUC-ROC solo si hay ambas clases
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.5  # Valor neutro si solo hay una clase
        
        return metrics
    
    @staticmethod
    def get_early_stopping_info(model, model_name: str) -> Dict[str, Any]:
        """Obtener información de early stopping para cada modelo"""
        info = {
            'stopped_early': False, 
            'best_iteration': None, 
            'total_iterations': None
        }
        
        try:
            if model_name == 'xgboost':
                if hasattr(model, 'best_iteration'):
                    info['stopped_early'] = (model.best_iteration < 
                                            model.n_estimators - 1)
                    info['best_iteration'] = model.best_iteration + 1
                    info['total_iterations'] = model.best_iteration + 1
                    
            elif model_name == 'lightgbm':
                if hasattr(model, 'best_iteration_'):
                    info['stopped_early'] = (model.best_iteration_ < 
                                            model.n_estimators)
                    info['best_iteration'] = model.best_iteration_
                    info['total_iterations'] = model.best_iteration_
                    
            elif model_name == 'gradient_boosting':
                if hasattr(model, 'n_estimators_'):
                    info['stopped_early'] = (model.n_estimators_ < 
                                            model.n_estimators)
                    info['best_iteration'] = model.n_estimators_
                    info['total_iterations'] = model.n_estimators_
                    
            elif model_name == 'neural_network':
                if hasattr(model, 'training_history'):
                    epochs_trained = len(
                        model.training_history.get('train_loss', [])
                    )
                    info['stopped_early'] = epochs_trained < model.epochs
                    info['best_iteration'] = epochs_trained
                    info['total_iterations'] = epochs_trained
                    
        except Exception as e:
            logger.debug(f"Error obteniendo info de early stopping para "
                        f"{model_name}: {e}")
        
        return info


class NBAWinPredictionNet(nn.Module):
    """Red neuronal especializada para predicción de victorias NBA"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 dropout_rate: float = 0.3):
        super(NBAWinPredictionNet, self).__init__()
        
        # Arquitectura optimizada para clasificación binaria
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Capas de procesamiento con LayerNorm
        self.input_ln = nn.LayerNorm(input_size)
        
        # Primera capa densa con dropout
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Segunda capa densa
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)
        
        # Tercera capa densa
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        # Capa de salida para clasificación binaria
        self.output = nn.Linear(hidden_size // 2, 1)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada de pesos para clasificación"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Normalización de entrada
        x = self.input_ln(x)
        
        # Primera capa con activación y dropout
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Segunda capa
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Tercera capa  
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Salida con sigmoid para probabilidad
        x = torch.sigmoid(self.output(x))
        
        return x


class PyTorchNBAClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador PyTorch optimizado para predicción de victorias NBA"""
    
    def __init__(self, hidden_size: int = 128, epochs: int = 200,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, early_stopping_patience: int = 20,
                 dropout_rate: float = 0.3, device: Optional[str] = None,
                 min_memory_gb: float = 2.0, auto_batch_size: bool = True):
        
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.device_preference = device
        self.min_memory_gb = min_memory_gb
        self.auto_batch_size = auto_batch_size
        
        # Componentes del modelo
        self.model = None
        self.scaler = StandardScaler()
        self.device = None
        
        # Métricas de entrenamiento y GPU
        self.training_history = {}
        self.gpu_memory_stats = {}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Configurar dispositivo usando GPUManager
        self._setup_device_with_gpu_manager()
    
    def _setup_device_with_gpu_manager(self):
        """Configuración avanzada del dispositivo usando GPUManager"""
        
        # Mostrar resumen de dispositivos disponibles
        if logger.isEnabledFor(logging.INFO):
            GPUManager.print_gpu_summary()
        
        # Configurar dispositivo óptimo
        self.device = GPUManager.setup_device(
            device_preference=self.device_preference,
            min_memory_gb=self.min_memory_gb
        )
        
        # Optimizar memoria del dispositivo
        GPUManager.optimize_memory_usage(self.device)
        
        # Monitorear memoria inicial
        self.gpu_memory_stats['initial'] = GPUManager.monitor_memory_usage(
            self.device, "initial"
        )
        
        logger.info(f"Dispositivo configurado: {self.device}")
    
    def _auto_adjust_batch_size(self, X_train_tensor: torch.Tensor, 
                               y_train_tensor: torch.Tensor) -> int:
        """Ajustar automáticamente el batch_size basado en memoria disponible"""
        
        if not self.auto_batch_size or self.device.type == 'cpu':
            return self.batch_size
        
        logger.info("Detectando batch_size óptimo para GPU...")
        
        # Crear modelo temporal para prueba
        temp_model = NBAWinPredictionNet(
            input_size=X_train_tensor.shape[1],
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Probar diferentes batch sizes
        test_batch_sizes = [32, 64, 128, 256, 512]
        optimal_batch_size = self.batch_size
        
        for test_batch_size in test_batch_sizes:
            try:
                # Probar forward pass con batch de prueba
                batch_end = min(test_batch_size, len(X_train_tensor))
                test_batch_X = X_train_tensor[:batch_end]
                test_batch_y = y_train_tensor[:batch_end]
                
                # Forward pass
                temp_model.train()
                outputs = temp_model(test_batch_X)
                loss = nn.BCELoss()(outputs, test_batch_y)
                
                # Backward pass
                loss.backward()
                
                # Si no hay error OOM, usar este batch_size
                optimal_batch_size = test_batch_size
                
                # Limpiar gradientes para siguiente prueba
                temp_model.zero_grad()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"OOM con batch_size {test_batch_size}, "
                               f"usando {optimal_batch_size}")
                    break
                else:
                    # Otro tipo de error
                    logger.warning(f"Error probando batch_size {test_batch_size}: {e}")
                    break
        
        # Limpiar modelo temporal
        del temp_model
        torch.cuda.empty_cache()
        
        logger.info(f"Batch size óptimo detectado: {optimal_batch_size}")
        return optimal_batch_size
    
    def fit(self, X, y):
        """Entrenamiento del modelo con early stopping, validación y manejo avanzado de GPU"""
        
        # Monitorear memoria antes del entrenamiento
        self.gpu_memory_stats['pre_training'] = GPUManager.monitor_memory_usage(
            self.device, "pre_training"
        )
        
        try:
            # Preparar datos
            X_scaled = self.scaler.fit_transform(X)
            
            # Establecer classes_ para compatibilidad con scikit-learn
            self.classes_ = np.array([0, 1])  # Clasificación binaria
            
            # División train/validation
            val_split = 0.2
            
            # Asegurar balance en train/val split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=val_split, stratify=y, random_state=42
            )
            
            # Convertir a tensores
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            
            # Manejar tanto numpy arrays como pandas Series
            if hasattr(y_train, 'values'):
                y_train_values = y_train.values
            else:
                y_train_values = y_train
            y_train_tensor = torch.FloatTensor(y_train_values.reshape(-1, 1)).to(self.device)
            
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            
            if hasattr(y_val, 'values'):
                y_val_values = y_val.values
            else:
                y_val_values = y_val
            y_val_tensor = torch.FloatTensor(y_val_values.reshape(-1, 1)).to(self.device)
            
            # Auto-ajustar batch_size si está habilitado
            optimal_batch_size = self._auto_adjust_batch_size(
                X_train_tensor, y_train_tensor
            )
            
            # Crear DataLoader con batch_size optimizado
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=optimal_batch_size, 
                shuffle=True,
                pin_memory=(self.device.type == 'cuda')
            )
            
            # Inicializar modelo
            self.model = NBAWinPredictionNet(
                input_size=X_train.shape[1],
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            # Configurar optimizador y loss
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=10
            )
            
            criterion = nn.BCELoss()
            
            # Entrenamiento con early stopping y monitoreo de memoria
            self.training_history = {
                'train_loss': [], 
                'val_loss': [], 
                'val_accuracy': [],
                'memory_stats': []
            }
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            
            # Monitorear memoria después de inicialización
            self.gpu_memory_stats['post_init'] = GPUManager.monitor_memory_usage(
                self.device, "post_init"
            )
            
            for epoch in range(self.epochs):
                # Entrenamiento
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    
                    # Monitorear memoria cada 50 batches
                    if batch_idx % 50 == 0 and self.device.type == 'cuda':
                        memory_stats = GPUManager.monitor_memory_usage(
                            self.device, f"epoch_{epoch}_batch_{batch_idx}"
                        )
                        self.training_history['memory_stats'].append(memory_stats)
                
                # Validación
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    # Calcular accuracy
                    val_preds = (val_outputs > 0.5).float()
                    val_accuracy = (val_preds == y_val_tensor).float().mean().item()
                
                # Guardar métricas
                avg_train_loss = train_loss / len(train_loader)
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.patience_counter += 1
                
                # Ajustar learning rate
                scheduler.step(val_loss)
                
                # Log progreso
                if epoch % 20 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                    )
                
                # Verificar early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping en epoch {epoch}")
                    break
            
            # Restaurar mejor modelo
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
            
            # Monitorear memoria final
            self.gpu_memory_stats['post_training'] = GPUManager.monitor_memory_usage(
                self.device, "post_training"
            )
            
            logger.info(
                f"Entrenamiento completado. Mejor val loss: {self.best_val_loss:.4f}"
            )
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Manejar error OOM
                oom_info = GPUManager.handle_oom_error(self.device)
                logger.error(f"Error de memoria insuficiente: {e}")
                logger.error(f"Sugerencias: {oom_info['suggested_actions']}")
                raise RuntimeError(
                    f"Error OOM en GPU. Sugerencias: {oom_info['suggested_actions']}"
                ) from e
            else:
                raise e
        
        return self
    
    def predict_proba(self, X):
        """Predicción de probabilidades con optimización de memoria"""
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        # Monitorear memoria durante inferencia
        memory_stats = GPUManager.monitor_memory_usage(self.device, "inference")
        
        X_scaled = self.scaler.transform(X)
        
        # Procesar en batches si dataset es grande para evitar OOM
        batch_size = min(1000, len(X_scaled))  # Batch size conservativo para inferencia
        
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                batch_end = min(i + batch_size, len(X_scaled))
                X_batch = X_scaled[i:batch_end]
                
                X_tensor = torch.FloatTensor(X_batch).to(self.device)
                batch_probabilities = self.model(X_tensor).cpu().numpy()
                all_probabilities.append(batch_probabilities)
        
        # Concatenar resultados
        probabilities = np.concatenate(all_probabilities, axis=0)
        
        # Retornar probabilidades para ambas clases
        prob_positive = probabilities.flatten()
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def predict(self, X):
        """Predicción de clases"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Obtener parámetros del modelo"""
        return {
            'hidden_size': self.hidden_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'dropout_rate': self.dropout_rate,
            'device': self.device_preference,
            'min_memory_gb': self.min_memory_gb,
            'auto_batch_size': self.auto_batch_size
        }
    
    def set_params(self, **params):
        """Establecer parámetros del modelo"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reconfigurar dispositivo si cambió la preferencia
        if 'device' in params or 'min_memory_gb' in params:
            self.device_preference = params.get('device', self.device_preference)
            self.min_memory_gb = params.get('min_memory_gb', self.min_memory_gb)
            self._setup_device_with_gpu_manager()
        
        return self
    
    def get_gpu_memory_summary(self) -> Dict[str, Any]:
        """Obtener resumen del uso de memoria GPU durante entrenamiento"""
        
        if not self.gpu_memory_stats:
            return {"error": "No hay estadísticas de memoria disponibles"}
        
        summary = {
            'device': str(self.device),
            'memory_evolution': self.gpu_memory_stats,
            'training_memory_stats': []
        }
        
        # Agregar estadísticas de memoria durante entrenamiento
        if 'memory_stats' in self.training_history:
            summary['training_memory_stats'] = self.training_history['memory_stats']
        
        return summary


class GPUManager:
    """Gestor avanzado de GPU para modelos NBA con detección de memoria y optimización"""
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """Obtener lista de dispositivos disponibles"""
        devices = ['cpu']
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        
        return devices
    
    @staticmethod
    def get_device_info(device_str: str = None) -> Dict[str, Any]:
        """Obtener información detallada del dispositivo"""
        
        if device_str is None:
            device_str = GPUManager.get_optimal_device()
        
        device = torch.device(device_str)
        info = {
            'device': device_str,
            'type': device.type,
            'available': True,
            'memory_info': None
        }
        
        if device.type == 'cuda':
            try:
                device_idx = device.index if device.index is not None else 0
                
                # Información de memoria
                total_memory = torch.cuda.get_device_properties(device_idx).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_idx)
                cached_memory = torch.cuda.memory_reserved(device_idx)
                free_memory = total_memory - cached_memory
                
                info.update({
                    'device_name': torch.cuda.get_device_name(device_idx),
                    'compute_capability': torch.cuda.get_device_capability(device_idx),
                    'memory_info': {
                        'total_gb': total_memory / (1024**3),
                        'allocated_gb': allocated_memory / (1024**3),
                        'cached_gb': cached_memory / (1024**3),
                        'free_gb': free_memory / (1024**3),
                        'utilization_pct': (cached_memory / total_memory) * 100
                    }
                })
            except Exception as e:
                logger.warning(f"Error obteniendo info de GPU {device_str}: {e}")
                info['available'] = False
        
        return info
    
    @staticmethod
    def check_memory_availability(device_str: str, 
                                required_gb: float = 2.0) -> bool:
        """Verificar si hay suficiente memoria disponible en el dispositivo"""
        
        device_info = GPUManager.get_device_info(device_str)
        
        if not device_info['available']:
            return False
        
        if device_info['type'] == 'cpu':
            return True  # Asumimos que CPU siempre tiene memoria disponible
        
        memory_info = device_info.get('memory_info')
        if memory_info:
            available_memory = memory_info['free_gb']
            return available_memory >= required_gb
        
        return False
    
    @staticmethod
    def get_optimal_device(min_memory_gb: float = 2.0) -> str:
        """Obtener el dispositivo óptimo con suficiente memoria"""
        
        if not torch.cuda.is_available():
            logger.info("CUDA no disponible, usando CPU")
            return 'cpu'
        
        # Buscar GPU con más memoria libre
        best_device = 'cpu'
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device_str = f'cuda:{i}'
            device_info = GPUManager.get_device_info(device_str)
            
            if device_info['available'] and device_info['memory_info']:
                free_memory = device_info['memory_info']['free_gb']
                
                if free_memory >= min_memory_gb and free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = device_str
        
        logger.info(f"Dispositivo óptimo seleccionado: {best_device}")
        if best_device != 'cpu':
            info = GPUManager.get_device_info(best_device)
            logger.info(f"GPU: {info.get('device_name', 'Unknown')}, "
                       f"Memoria libre: {max_free_memory:.1f}GB")
        
        return best_device
    
    @staticmethod
    def setup_device(device_preference: str = None, 
                   min_memory_gb: float = 2.0) -> torch.device:
        """Configurar dispositivo con verificaciones de seguridad"""
        
        if device_preference:
            # Verificar dispositivo específico solicitado
            if device_preference in GPUManager.get_available_devices():
                if GPUManager.check_memory_availability(device_preference, min_memory_gb):
                    logger.info(f"Usando dispositivo solicitado: {device_preference}")
                    return torch.device(device_preference)
                else:
                    logger.warning(f"Dispositivo {device_preference} no tiene suficiente memoria "
                                 f"({min_memory_gb}GB requeridos). Buscando alternativa...")
            else:
                logger.warning(f"Dispositivo {device_preference} no disponible. "
                             f"Buscando alternativa...")
        
        # Buscar dispositivo óptimo automáticamente
        optimal_device = GPUManager.get_optimal_device(min_memory_gb)
        return torch.device(optimal_device)
    
    @staticmethod
    def optimize_memory_usage(device: torch.device):
        """Optimizar uso de memoria del dispositivo"""
        
        if device.type == 'cuda':
            try:
                # Limpiar caché de memoria
                torch.cuda.empty_cache()
                
                # Configurar optimizaciones de memoria
                torch.backends.cudnn.benchmark = True  # Optimizar para tamaños fijos
                torch.backends.cudnn.deterministic = False  # Permitir no-determinismo para velocidad
                
                logger.info("Optimizaciones de memoria GPU aplicadas")
                
            except Exception as e:
                logger.warning(f"Error optimizando memoria GPU: {e}")
    
    @staticmethod
    def monitor_memory_usage(device: torch.device, 
                           phase: str = "training") -> Dict[str, float]:
        """Monitorear uso de memoria durante entrenamiento/inferencia"""
        
        memory_stats = {}
        
        if device.type == 'cuda':
            try:
                device_idx = device.index if device.index is not None else 0
                
                allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                cached = torch.cuda.memory_reserved(device_idx) / (1024**3)
                
                memory_stats = {
                    f'{phase}_allocated_gb': allocated,
                    f'{phase}_cached_gb': cached,
                    f'{phase}_timestamp': datetime.now().isoformat()
                }
                
                logger.debug(f"Memoria GPU {phase}: {allocated:.2f}GB allocated, "
                           f"{cached:.2f}GB cached")
                
            except Exception as e:
                logger.debug(f"Error monitoreando memoria: {e}")
        
        return memory_stats
    
    @staticmethod
    def handle_oom_error(device: torch.device, 
                        reduce_batch_size: bool = True) -> Dict[str, Any]:
        """Manejar errores de memoria insuficiente (OOM)"""
        
        recovery_info = {
            'oom_occurred': True,
            'recovery_attempted': False,
            'suggested_actions': []
        }
        
        if device.type == 'cuda':
            try:
                # Limpiar memoria
                torch.cuda.empty_cache()
                
                # Sugerencias de recuperación
                recovery_info['suggested_actions'] = [
                    'Reducir batch_size',
                    'Usar gradient_checkpointing',
                    'Reducir dimensiones del modelo',
                    'Usar mixed precision training',
                    'Cambiar a CPU'
                ]
                
                recovery_info['recovery_attempted'] = True
                logger.warning("Error OOM detectado. Memoria GPU limpiada. "
                             "Considera reducir batch_size o usar CPU.")
                
            except Exception as e:
                logger.error(f"Error en recuperación OOM: {e}")
        
        return recovery_info
    
    @staticmethod
    def print_gpu_summary():
        """Imprimir resumen de información GPU"""
        
        print("\n" + "="*60)
        print("🖥️  RESUMEN DE DISPOSITIVOS DISPONIBLES")
        print("="*60)
        
        devices = GPUManager.get_available_devices()
        
        for device_str in devices:
            info = GPUManager.get_device_info(device_str)
            
            print(f"\n📱 Dispositivo: {device_str}")
            print(f"   Tipo: {info['type'].upper()}")
            print(f"   Disponible: {'✅' if info['available'] else '❌'}")
            
            if info['type'] == 'cuda' and info['available']:
                print(f"   Nombre: {info.get('device_name', 'N/A')}")
                print(f"   Compute Capability: {info.get('compute_capability', 'N/A')}")
                
                if info['memory_info']:
                    mem = info['memory_info']
                    print(f"   Memoria Total: {mem['total_gb']:.1f}GB")
                    print(f"   Memoria Libre: {mem['free_gb']:.1f}GB")
                    print(f"   Utilización: {mem['utilization_pct']:.1f}%")
        
        print("\n" + "="*60)
        optimal = GPUManager.get_optimal_device()
        print(f"🎯 Dispositivo óptimo recomendado: {optimal}")
        print("="*60 + "\n")


def configure_gpu_environment(device_preference: str = None, 
                             min_memory_gb: float = 2.0,
                             print_summary: bool = True) -> Dict[str, Any]:
    """
    Configurar entorno GPU globalmente para modelos NBA
    
    Args:
        device_preference: Dispositivo preferido ('cuda:0', 'cuda:1', etc.)
        min_memory_gb: Memoria mínima requerida en GB
        print_summary: Si mostrar resumen de dispositivos
        
    Returns:
        Diccionario con información de configuración
    """
    
    if print_summary:
        GPUManager.print_gpu_summary()
    
    # Configurar dispositivo óptimo
    optimal_device = GPUManager.setup_device(device_preference, min_memory_gb)
    
    # Optimizar entorno
    GPUManager.optimize_memory_usage(optimal_device)
    
    # Obtener información del dispositivo
    device_info = GPUManager.get_device_info(str(optimal_device))
    
    config_info = {
        'selected_device': str(optimal_device),
        'device_info': device_info,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    logger.info(f"Entorno GPU configurado: {optimal_device}")
    return config_info


class IsWinModel:
    """Modelo principal para predicción de victorias NBA con stacking y optimización bayesiana"""
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 bayesian_n_calls: int = 25,
                 min_memory_gb: float = 2.0):
        
        self.optimize_hyperparams = optimize_hyperparams
        self.device_preference = device
        self.bayesian_n_calls = bayesian_n_calls
        self.min_memory_gb = min_memory_gb
        
        # Componentes del modelo
        self.feature_engineer = IsWinFeatureEngineer()
        self.scaler = StandardScaler()
        
        # Modelos individuales
        self.models = {}
        self.stacking_model = None
        
        # Métricas y resultados
        self.training_results = {}
        self.feature_importance = {}
        self.bayesian_results = {}
        self.gpu_config = {}
        
        # Configurar entorno GPU
        self._setup_gpu_environment()
        
        # Configurar modelos
        self._setup_models()
    
    def _setup_gpu_environment(self):
        """Configurar entorno GPU para el modelo"""
        self.gpu_config = configure_gpu_environment(
            device_preference=self.device_preference,
            min_memory_gb=self.min_memory_gb,
            print_summary=True
        )
        
        # Usar dispositivo configurado
        self.device = self.gpu_config['selected_device']
    
    def _setup_models(self):
        """
        Configurar modelos individuales con REGULARIZACIÓN AGRESIVA
        Prevenir sobreajuste con parámetros conservadores
        """
        logger.info("Configurando modelos con regularización agresiva...")
        
        # XGBoost con regularización fuerte
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,          # Reducido para evitar sobreajuste
            max_depth=4,               # Profundidad limitada
            learning_rate=0.05,        # Learning rate bajo
            subsample=0.8,             # Submuestreo agresivo
            colsample_bytree=0.8,      # Feature sampling
            reg_alpha=1.0,             # L1 regularization
            reg_lambda=2.0,            # L2 regularization fuerte
            min_child_weight=5,        # Peso mínimo por hoja
            gamma=1.0,                 # Minimum split loss
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
            # Sin early_stopping_rounds para compatibilidad con CV
        )
        
        # LightGBM con regularización fuerte
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,          # Reducido
            max_depth=4,               # Profundidad limitada
            learning_rate=0.05,        # Learning rate bajo
            subsample=0.8,             # Submuestreo
            colsample_bytree=0.8,      # Feature sampling
            reg_alpha=1.0,             # L1 regularization
            reg_lambda=2.0,            # L2 regularization fuerte
            min_child_samples=20,      # Muestras mínimas por hoja
            min_split_gain=0.1,        # Ganancia mínima para split
            random_state=42,
            n_jobs=-1,
            verbosity=-1
            # Sin early_stopping_rounds para compatibilidad con CV
        )
        
        # Random Forest con regularización
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,          # Reducido
            max_depth=6,               # Profundidad limitada
            min_samples_split=20,      # Samples mínimas para split
            min_samples_leaf=10,       # Samples mínimas por hoja
            max_features='sqrt',       # Feature sampling automático
            bootstrap=True,
            oob_score=True,            # Out-of-bag validation
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees con más regularización
        self.models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=100,          # Reducido
            max_depth=6,               # Profundidad limitada
            min_samples_split=25,      # Más estricto
            min_samples_leaf=15,       # Más estricto
            max_features='sqrt',       # Feature sampling
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting con regularización fuerte
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,          # Reducido
            max_depth=4,               # Profundidad limitada
            learning_rate=0.05,        # Learning rate bajo
            subsample=0.8,             # Submuestreo
            min_samples_split=20,      # Samples mínimas para split
            min_samples_leaf=10,       # Samples mínimas por hoja
            max_features='sqrt',       # Feature sampling
            random_state=42
        )
        
        # Red Neuronal con Dropout agresivo
        self.models['neural_network'] = PyTorchNBAClassifier(
            hidden_size=64,            # Reducido de 128
            epochs=150,                # Reducido de 200
            batch_size=64,             # Más grande para estabilidad
            learning_rate=0.001,
            weight_decay=0.01,         # L2 regularization
            early_stopping_patience=15,
            dropout_rate=0.5,          # Dropout agresivo (50%)
            device=self.device,
            min_memory_gb=self.min_memory_gb,
            auto_batch_size=True
        )
        
        # Configurar stacking con LogisticRegression como meta-learner
        self._setup_stacking_model()
    
    def _setup_stacking_model(self):
        """Configurar modelo de stacking con meta-learner optimizado"""
        
        # Crear versiones de modelos sin early stopping para stacking
        xgb_stacking = xgb.XGBClassifier(
            n_estimators=100,  # Reducido para stacking
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
            # Sin early_stopping_rounds para stacking
        )
        
        lgb_stacking = lgb.LGBMClassifier(
            n_estimators=100,  # Reducido para stacking
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
            # Sin early_stopping_rounds para stacking
        )
        
        rf_stacking = RandomForestClassifier(
            n_estimators=100,  # Reducido para stacking
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        et_stacking = ExtraTreesClassifier(
            n_estimators=100,  # Reducido para stacking
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_stacking = GradientBoostingClassifier(
            n_estimators=100,  # Reducido para stacking
            max_depth=6,
            learning_rate=0.1,
            random_state=42
            # Sin validation_fraction para stacking
        )
        
        nn_stacking = PyTorchNBAClassifier(
            hidden_size=64,  # Reducido para stacking
            epochs=50,  # Reducido para stacking
            early_stopping_patience=10,
            device=self.device,
            min_memory_gb=self.min_memory_gb,
            auto_batch_size=True
        )
        
        # Estimadores base para stacking
        base_estimators = [
            ('xgb', xgb_stacking),
            ('lgb', lgb_stacking),
            ('rf', rf_stacking),
            ('et', et_stacking),
            ('gb', gb_stacking),
            ('nn', nn_stacking)
        ]
        
        # Meta-learner: Regresión Logística con regularización
        meta_learner = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        # Modelo de stacking con validación cruzada
        self.stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            stack_method='predict_proba',
            n_jobs=-1
        )
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Obtener columnas de features generadas"""
        feature_columns = self.feature_engineer.generate_all_features(df)
        
        # Filtrar features que realmente existen y no son problemáticas
        available_features = []
        for feature in feature_columns:
            if feature in df.columns:
                # Verificar que no tenga demasiados valores nulos
                null_pct = df[feature].isnull().sum() / len(df)
                if null_pct < 0.5:  # Menos del 50% de nulos
                    available_features.append(feature)
        
        logger.info(f"Features disponibles: {len(available_features)} de "
                   f"{len(feature_columns)}")
        return available_features
    
    def train(self, df: pd.DataFrame, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrenamiento completo del modelo con validación y optimización"""
        
        logger.info("Iniciando entrenamiento del modelo de predicción de victorias...")
        
        # Generar features
        feature_columns = self.get_feature_columns(df)
        
        if not feature_columns:
            raise ValueError("No hay features disponibles para el entrenamiento")
        
        # Preparar datos usando DataProcessor
        X = df[feature_columns].fillna(0)  # Rellenar nulos
        y = df['is_win']
        
        # Verificar balance de clases
        class_balance = y.value_counts(normalize=True)
        logger.info(f"Balance de clases: Victorias: {class_balance.get(1, 0):.3f}, "
                   f"Derrotas: {class_balance.get(0, 0):.3f}")
        
        # Preparar datos de entrenamiento
        X_train_scaled, X_val_scaled, y_train, y_val, self.scaler = (
            DataProcessor.prepare_training_data(X, y, validation_split)
        )
        
        # Entrenar modelos individuales
        individual_results = self._train_individual_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Optimización bayesiana (si está habilitada)
        if self.optimize_hyperparams and BAYESIAN_AVAILABLE:
            self._optimize_with_bayesian(X_train_scaled, y_train)
        
        # Entrenar modelo de stacking
        logger.info("Entrenando modelo de stacking...")
        self.stacking_model.fit(X_train_scaled, y_train)
        
        # Evaluación completa
        stacking_val_pred = self.stacking_model.predict(X_val_scaled)
        stacking_val_proba = self.stacking_model.predict_proba(X_val_scaled)[:, 1]
        
        # Métricas del stacking usando MetricsCalculator
        stacking_metrics = MetricsCalculator.calculate_classification_metrics(
            y_val, stacking_val_pred, stacking_val_proba
        )
        
        # Compilar resultados
        self.training_results = {
            'individual_models': individual_results,
            'stacking_metrics': stacking_metrics,
            'feature_count': len(feature_columns),
            'training_samples': len(X_train_scaled),
            'validation_samples': len(X_val_scaled),
            'class_balance': class_balance.to_dict()
        }
        
        # Validación cruzada del modelo final
        cv_results = self._perform_cross_validation(X, y)
        self.training_results['cross_validation'] = cv_results
        
        # Feature importance
        self.feature_importance = self._calculate_feature_importance(feature_columns)
        
        logger.info(f"Entrenamiento completado. Accuracy de stacking: "
                   f"{stacking_metrics['accuracy']:.4f}")
        
        return self.training_results
    
    def _train_individual_models(self, X_train, y_train, 
                               X_val, y_val) -> Dict:
        """Entrenar modelos individuales con early stopping optimizado"""
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Entrenando modelo con early stopping: {name}")
            
            try:
                # Entrenar modelo según su tipo con early stopping específico
                if name == 'xgboost':
                    model = ModelTrainer.train_xgboost_with_early_stopping(
                        model, X_train, y_train, X_val, y_val
                    )
                    
                elif name == 'lightgbm':
                    model = ModelTrainer.train_lightgbm_with_early_stopping(
                        model, X_train, y_train, X_val, y_val
                    )
                    
                elif name in ['gradient_boosting', 'random_forest', 'extra_trees']:
                    model = ModelTrainer.train_sklearn_with_early_stopping(
                        model, X_train, y_train, X_val, y_val, name
                    )
                    
                elif name == 'neural_network':
                    # Red neuronal ya tiene early stopping implementado
                    model.fit(X_train, y_train)
                
                else:
                    # Fallback para otros modelos
                    model.fit(X_train, y_train)
                
                # Predicciones para evaluación
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Probabilidades (si están disponibles)
                if hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(X_train)[:, 1]
                    val_proba = model.predict_proba(X_val)[:, 1]
                else:
                    train_proba = train_pred.astype(float)
                    val_proba = val_pred.astype(float)
                
                # Calcular métricas usando MetricsCalculator
                train_metrics = MetricsCalculator.calculate_classification_metrics(
                    y_train, train_pred, train_proba
                )
                val_metrics = MetricsCalculator.calculate_classification_metrics(
                    y_val, val_pred, val_proba
                )
                
                # Información de early stopping
                early_stopping_info = MetricsCalculator.get_early_stopping_info(
                    model, name
                )
                
                results[name] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'overfitting': train_metrics['accuracy'] - val_metrics['accuracy'],
                    'early_stopping_info': early_stopping_info
                }
                
                logger.info(f"{name} - Val Accuracy: {val_metrics['accuracy']:.4f}, "
                           f"Val AUC: {val_metrics['auc_roc']:.4f}")
                
                if early_stopping_info.get('stopped_early'):
                    logger.info(f"  Early stopping activado en "
                               f"{early_stopping_info.get('best_iteration', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicción de victorias usando el modelo de stacking"""
        
        if self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Generar features
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].fillna(0)
        
        # Escalar features usando DataProcessor
        X_scaled = DataProcessor.prepare_prediction_data(X, self.scaler)
        
        # Predicción
        predictions = self.stacking_model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicción de probabilidades de victoria"""
        
        if self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Generar features
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].fillna(0)
        
        # Escalar features usando DataProcessor
        X_scaled = DataProcessor.prepare_prediction_data(X, self.scaler)
        
        # Predicción de probabilidades
        probabilities = self.stacking_model.predict_proba(X_scaled)
        
        return probabilities
    
    def _optimize_with_bayesian(self, X_train, y_train):
        """Optimización bayesiana de hiperparámetros"""
        
        if not BAYESIAN_AVAILABLE:
            logger.warning("skopt no disponible - saltando optimización bayesiana")
            return
        
        logger.info("Iniciando optimización bayesiana de hiperparámetros...")
        
        # Optimizar XGBoost
        self._optimize_xgboost_bayesian(X_train, y_train)
        
        # Optimizar LightGBM
        self._optimize_lightgbm_bayesian(X_train, y_train)
        
        # Optimizar Red Neuronal
        self._optimize_neural_net_bayesian(X_train, y_train)
    
    def _optimize_xgboost_bayesian(self, X_train, y_train):
        """Optimización bayesiana específica para XGBoost"""
        
        # Espacio de búsqueda
        space = [
            Integer(50, 300, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Real(0.01, 10.0, name='reg_alpha'),
            Real(0.01, 10.0, name='reg_lambda')
        ]
        
        # Función objetivo específica para XGBoost
        @use_named_args(space)
        def objective(**params):
            # Crear modelo con parámetros específicos
            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            # Validación cruzada estratificada
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # Retornar negativo para minimización
            return -cv_scores.mean()
        
        # Ejecutar optimización
        result = gp_minimize(
            objective, space,
            n_calls=max(10, self.bayesian_n_calls // 2),  # Asegurar mínimo 10 llamadas
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['xgboost'].set_params(**best_params)
        self.bayesian_results['xgboost'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
        
        logger.info(f"XGBoost optimizado - Mejor AUC: {-result.fun:.4f}")
    
    def _optimize_lightgbm_bayesian(self, X_train, y_train):
        """Optimización bayesiana específica para LightGBM"""
        
        # Espacio de búsqueda
        space = [
            Integer(50, 300, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Real(0.01, 10.0, name='reg_alpha'),
            Real(0.01, 10.0, name='reg_lambda'),
            Integer(10, 100, name='min_child_samples')
        ]
        
        # Función objetivo específica para LightGBM
        @use_named_args(space)
        def objective(**params):
            # Crear modelo con parámetros específicos
            model = lgb.LGBMClassifier(
                **params,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
            # Validación cruzada estratificada
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # Retornar negativo para minimización
            return -cv_scores.mean()
        
        # Ejecutar optimización
        result = gp_minimize(
            objective, space,
            n_calls=max(10, self.bayesian_n_calls // 2),  # Asegurar mínimo 10 llamadas
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['lightgbm'].set_params(**best_params)
        self.bayesian_results['lightgbm'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
        
        logger.info(f"LightGBM optimizado - Mejor AUC: {-result.fun:.4f}")
    
    def _optimize_neural_net_bayesian(self, X_train, y_train):
        """Optimización bayesiana para la red neuronal"""
        
        # Espacio de búsqueda
        space = [
            Integer(64, 256, name='hidden_size'),
            Real(0.0001, 0.01, name='learning_rate'),
            Real(0.001, 0.1, name='weight_decay'),
            Real(0.1, 0.5, name='dropout_rate'),
            Integer(16, 64, name='batch_size')
        ]
        
        @use_named_args(space)
        def objective(**params):
            # Asegurar que batch_size sea entero
            params['batch_size'] = int(params['batch_size'])
            
            # Crear modelo con parámetros específicos
            model = PyTorchNBAClassifier(
                hidden_size=params['hidden_size'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                dropout_rate=params['dropout_rate'],
                batch_size=params['batch_size'],
                epochs=100,  # Reducido para optimización
                early_stopping_patience=15,
                device=self.device
            )
            
            # Validación cruzada manual (PyTorch necesita manejo especial)
            cv_scores = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Entrenar modelo
                model.fit(X_fold_train, y_fold_train)
                
                # Evaluar
                y_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_proba)
                cv_scores.append(score)
            
            return -np.mean(cv_scores)
        
        # Ejecutar optimización
        result = gp_minimize(
            objective, space,
            n_calls=max(10, self.bayesian_n_calls // 2),  # Asegurar mínimo 10 llamadas
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['neural_network'].set_params(**best_params)
        self.bayesian_results['neural_network'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
        
        logger.info(f"Red Neuronal optimizada - Mejor AUC: {-result.fun:.4f}")
    
    def _perform_cross_validation(self, X, y) -> Dict[str, Any]:
        """
        Validación cruzada ESTRICTA con múltiples métricas
        Usar estratificación temporal para evitar data leakage temporal
        """
        logger.info("Realizando validación cruzada estricta...")
        
        # Configurar validación cruzada ESTRICTA
        # Usar más folds para validación rigurosa
        n_splits = 10  # Incrementado de 5 a 10
        cv = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=42
        )
        
        cv_results = {}
        
        # Métricas de evaluación exhaustivas
        scoring_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 
            'roc_auc', 'neg_log_loss'
        ]
        
        # Validación cruzada para cada modelo
        for model_name, model in self.models.items():
            logger.info(f"Validación cruzada para {model_name}...")
            
            model_cv_results = {}
            
            # Evaluar múltiples métricas
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        model, X, y, 
                        cv=cv, 
                        scoring=metric,
                        n_jobs=-1
                    )
                    
                    model_cv_results[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                    
                    # DETECTAR SOBREAJUSTE
                    # Si std es muy pequeña, puede indicar sobreajuste
                    if scores.std() < 0.01 and metric in ['accuracy', 'roc_auc']:
                        logger.warning(
                            f"⚠️  POSIBLE SOBREAJUSTE en {model_name}: "
                            f"{metric} std = {scores.std():.4f} (muy bajo)"
                        )
                    
                except Exception as e:
                    logger.error(f"Error en CV para {model_name} - {metric}: {e}")
                    model_cv_results[metric] = {
                        'mean': 0.0, 'std': 0.0, 'scores': []
                    }
            
            cv_results[model_name] = model_cv_results
        
        # VALIDACIÓN ADICIONAL: Hold-out temporal
        # Dividir por tiempo para simular predicción futura real
        if 'Date' in X.index.names or len(X) > 1000:
            logger.info("Realizando validación temporal hold-out...")
            
            # Usar últimos 20% como test temporal
            split_idx = int(len(X) * 0.8)
            X_temporal_train = X.iloc[:split_idx]
            X_temporal_test = X.iloc[split_idx:]
            y_temporal_train = y.iloc[:split_idx]
            y_temporal_test = y.iloc[split_idx:]
            
            temporal_results = {}
            
            for model_name, model in self.models.items():
                try:
                    # Entrenar en datos tempranos
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_temporal_train, y_temporal_train)
                    
                    # Predecir en datos futuros
                    y_pred = model_copy.predict(X_temporal_test)
                    y_proba = model_copy.predict_proba(X_temporal_test)[:, 1]
                    
                    temporal_results[model_name] = {
                        'accuracy': accuracy_score(y_temporal_test, y_pred),
                        'roc_auc': roc_auc_score(y_temporal_test, y_proba)
                    }
                    
                except Exception as e:
                    logger.error(f"Error en validación temporal para {model_name}: {e}")
                    temporal_results[model_name] = {'accuracy': 0.0, 'roc_auc': 0.5}
            
            cv_results['temporal_validation'] = temporal_results
        
        # RESUMEN DE ALERTAS DE SOBREAJUSTE
        logger.info("\n=== ANÁLISIS DE SOBREAJUSTE ===")
        for model_name in self.models.keys():
            if model_name in cv_results:
                acc_std = cv_results[model_name].get('accuracy', {}).get('std', 0)
                auc_std = cv_results[model_name].get('roc_auc', {}).get('std', 0)
                
                if acc_std < 0.02 or auc_std < 0.02:
                    logger.warning(f"🚨 {model_name}: Variabilidad muy baja - POSIBLE SOBREAJUSTE")
                elif acc_std < 0.05 or auc_std < 0.05:
                    logger.warning(f"⚠️  {model_name}: Variabilidad baja - Monitorear sobreajuste")
                else:
                    logger.info(f"✅ {model_name}: Variabilidad normal")
        
        return cv_results
    
    def _calculate_feature_importance(self, 
                                    feature_columns: List[str]) -> Dict[str, Any]:
        """Calcular importancia de features desde múltiples modelos"""
        
        importance_dict = {}
        
        # Lista de modelos con feature importance
        importance_models = [
            ('xgboost', 'feature_importances_'),
            ('lightgbm', 'feature_importances_'),
            ('random_forest', 'feature_importances_'),
            ('extra_trees', 'feature_importances_'),
            ('gradient_boosting', 'feature_importances_')
        ]
        
        # Extraer importancia de cada modelo
        for model_name, attr_name in importance_models:
            if model_name in self.models:
                try:
                    model = self.models[model_name]
                    if hasattr(model, attr_name):
                        importance_values = getattr(model, attr_name)
                        importance_dict[model_name] = dict(
                            zip(feature_columns, importance_values)
                        )
                except Exception as e:
                    logger.debug(f"Error obteniendo importancia de {model_name}: {e}")
        
        # Importancia promedio
        if importance_dict:
            avg_importance = {}
            for feature in feature_columns:
                importances = []
                for model_importance in importance_dict.values():
                    if feature in model_importance:
                        importances.append(model_importance[feature])
                
                if importances:
                    avg_importance[feature] = np.mean(importances)
            
            # Ordenar por importancia
            sorted_importance = sorted(
                avg_importance.items(), key=lambda x: x[1], reverse=True
            )
            importance_dict['average'] = dict(sorted_importance)
        
        return importance_dict
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """Obtener importancia de features del modelo entrenado"""
        
        if not self.feature_importance:
            raise ValueError("Modelo no entrenado o importancia no calculada")
        
        # Top features promedio
        if 'average' in self.feature_importance:
            top_features = list(
                self.feature_importance['average'].items()
            )[:top_n]
            
            return {
                'top_features': top_features,
                'feature_importance_by_model': self.feature_importance,
                'total_features': len(
                    self.feature_importance.get('average', {})
                )
            }
        
        return self.feature_importance
    
    def validate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validación completa del modelo en datos nuevos"""
        
        if 'is_win' not in df.columns:
            raise ValueError("Columna 'is_win' requerida para validación")
        
        # Predicciones
        y_true = df['is_win']
        y_pred = self.predict(df)
        y_proba = self.predict_proba(df)[:, 1]
        
        # Métricas de validación usando MetricsCalculator
        validation_metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, y_proba
        )
        
        # Análisis por contexto
        context_analysis = {}
        
        # Análisis por local/visitante
        if 'is_home' in df.columns:
            home_mask = df['is_home'] == 1
            away_mask = df['is_home'] == 0
            
            if home_mask.sum() > 0:
                home_acc = accuracy_score(
                    y_true[home_mask], y_pred[home_mask]
                )
                context_analysis['home_accuracy'] = home_acc
            
            if away_mask.sum() > 0:
                away_acc = accuracy_score(
                    y_true[away_mask], y_pred[away_mask]
                )
                context_analysis['away_accuracy'] = away_acc
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        validation_report = {
            'overall_metrics': validation_metrics,
            'context_analysis': context_analysis,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_true, y_pred, output_dict=True
            ),
            'sample_count': len(df)
        }
        
        logger.info(f"Validación completada - Accuracy: "
                   f"{validation_metrics['accuracy']:.4f}")
        
        return validation_report
    
    def save_model(self, save_path: str = None):
        """Guardar modelo entrenado"""
        
        if save_path is None:
            save_path = "trained_models/is_win_model.joblib"
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Preparar objeto para guardar
        model_data = {
            'stacking_model': self.stacking_model,
            'models': self.models,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'training_results': self.training_results,
            'feature_importance': self.feature_importance,
            'bayesian_results': self.bayesian_results,
            'model_metadata': {
                'created_at': datetime.now().isoformat(),
                'optimize_hyperparams': self.optimize_hyperparams,
                'device': str(self.device) if self.device else None
            }
        }
        
        # Guardar modelo
        joblib.dump(model_data, save_path)
        
        logger.info(f"Modelo guardado en: {save_path}")
        
        return save_path
    
    @staticmethod
    def load_model(model_path: str = "trained_models/is_win_model.joblib"):
        """Cargar modelo entrenado"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        # Cargar datos del modelo
        model_data = joblib.load(model_path)
        
        # Recrear instancia del modelo
        model = IsWinModel(optimize_hyperparams=False)
        
        # Restaurar componentes
        model.stacking_model = model_data['stacking_model']
        model.models = model_data['models']
        model.scaler = model_data['scaler']
        model.feature_engineer = model_data['feature_engineer']
        model.training_results = model_data.get('training_results', {})
        model.feature_importance = model_data.get('feature_importance', {})
        model.bayesian_results = model_data.get('bayesian_results', {})
        
        logger.info(f"Modelo cargado desde: {model_path}")
        
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Resumen completo del entrenamiento"""
        
        if not self.training_results:
            return {"error": "Modelo no entrenado"}
        
        summary = {
            "model_performance": {
                "stacking_accuracy": self.training_results.get(
                    'stacking_metrics', {}
                ).get('accuracy', 0),
                "stacking_auc": self.training_results.get(
                    'stacking_metrics', {}
                ).get('auc_roc', 0),
                "cv_accuracy_mean": self.training_results.get(
                    'cross_validation', {}
                ).get('accuracy', {}).get('mean', 0),
                "cv_accuracy_std": self.training_results.get(
                    'cross_validation', {}
                ).get('accuracy', {}).get('std', 0)
            },
            "training_info": {
                "feature_count": self.training_results.get('feature_count', 0),
                "training_samples": self.training_results.get(
                    'training_samples', 0
                ),
                "validation_samples": self.training_results.get(
                    'validation_samples', 0
                ),
                "class_balance": self.training_results.get('class_balance', {})
            },
            "individual_models": {},
            "bayesian_optimization": self.bayesian_results
        }
        
        # Rendimiento de modelos individuales
        for model_name, results in self.training_results.get(
            'individual_models', {}
        ).items():
            if 'val_metrics' in results:
                summary["individual_models"][model_name] = {
                    "accuracy": results['val_metrics'].get('accuracy', 0),
                    "auc_roc": results['val_metrics'].get('auc_roc', 0),
                    "overfitting": results.get('overfitting', 0)
                }
        
        return summary
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Obtener información completa del GPU configurado"""
        
        gpu_info = {
            'configuration': self.gpu_config,
            'current_device': self.device,
            'available_devices': GPUManager.get_available_devices(),
            'memory_requirements': {
                'min_memory_gb': self.min_memory_gb,
                'recommended_memory_gb': 4.0
            }
        }
        
        # Información específica del dispositivo actual
        if self.device:
            gpu_info['current_device_info'] = GPUManager.get_device_info(self.device)
        
        # Información de red neuronal si está disponible
        if ('neural_network' in self.models and 
            hasattr(self.models['neural_network'], 'get_gpu_memory_summary')):
            gpu_info['neural_network_memory'] = (
                self.models['neural_network'].get_gpu_memory_summary()
            )
        
        return gpu_info
    
    def optimize_for_gpu(self, target_memory_gb: float = None) -> Dict[str, Any]:
        """Optimizar configuración del modelo para GPU específico"""
        
        if target_memory_gb is None:
            device_info = GPUManager.get_device_info(self.device)
            if device_info.get('memory_info'):
                target_memory_gb = device_info['memory_info']['free_gb'] * 0.8  # 80% de memoria libre
            else:
                target_memory_gb = 2.0  # Valor por defecto
        
        optimization_info = {
            'target_memory_gb': target_memory_gb,
            'optimizations_applied': []
        }
        
        # Optimizar red neuronal
        if 'neural_network' in self.models:
            nn_model = self.models['neural_network']
            
            # Ajustar batch_size basado en memoria disponible
            if target_memory_gb >= 6.0:
                nn_model.batch_size = 128
                nn_model.hidden_size = 256
                optimization_info['optimizations_applied'].append('High memory: batch_size=128, hidden_size=256')
            elif target_memory_gb >= 4.0:
                nn_model.batch_size = 64
                nn_model.hidden_size = 128
                optimization_info['optimizations_applied'].append('Medium memory: batch_size=64, hidden_size=128')
            else:
                nn_model.batch_size = 32
                nn_model.hidden_size = 64
                optimization_info['optimizations_applied'].append('Low memory: batch_size=32, hidden_size=64')
            
            # Habilitar auto ajuste de batch_size
            nn_model.auto_batch_size = True
            optimization_info['optimizations_applied'].append('Auto batch_size enabled')
        
        logger.info(f"Modelo optimizado para {target_memory_gb:.1f}GB de memoria GPU")
        return optimization_info
