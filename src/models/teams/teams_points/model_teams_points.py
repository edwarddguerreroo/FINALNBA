"""
Modelo Avanzado de Predicción de Puntos de Equipo NBA
====================================================

Este módulo implementa un sistema de predicción de alto rendimiento para
puntos de equipo NBA utilizando:

1. Ensemble Learning con múltiples algoritmos ML y Red Neuronal
2. Stacking avanzado con meta-modelo optimizado
3. Optimización automática de hiperparámetros
4. Validación cruzada rigurosa
5. Métricas de evaluación exhaustivas
6. Feature engineering especializado
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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
    StackingRegressor, VotingRegressor
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

# XGBoost and LightGBM
import lightgbm as lgb
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Bayesian Optimization
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Local imports
from .features_teams_points import TeamPointsFeatureEngineer

# Configuration
warnings.filterwarnings('ignore')

# Logging setup
import logging
logger = logging.getLogger(__name__)


class GPUUtils:
    """Utilidades para gestión inteligente de GPU"""
    
    @staticmethod
    def get_optimal_device(device: Optional[str] = None,
                          memory_threshold_gb: float = 2.0) -> torch.device:
        """
        Selecciona el dispositivo óptimo considerando memoria disponible.
        
        Args:
            device: Dispositivo específico ('cuda:0', 'cuda:1', 'cpu', None)
            memory_threshold_gb: Memoria mínima requerida en GB
            
        Returns:
            torch.device optimizado
        """
        # Si se especifica CPU explícitamente
        if device == 'cpu':
            logger.info("Usando CPU por especificación explícita")
            return torch.device('cpu')
        
        # Si no hay CUDA disponible
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        # Si se especifica un dispositivo CUDA específico
        if device and device.startswith('cuda'):
            try:
                target_device = torch.device(device)
                if GPUUtils._check_gpu_memory(target_device, memory_threshold_gb):
                    logger.info(f"Usando dispositivo especificado: {device}")
                    return target_device
                else:
                    logger.warning(
                        f"Memoria insuficiente en {device}, "
                        f"buscando alternativa..."
                    )
            except Exception as e:
                logger.warning(f"Error con dispositivo {device}: {e}")
        
        # Buscar el mejor dispositivo GPU disponible
        best_device = GPUUtils._find_best_gpu(memory_threshold_gb)
        if best_device:
            return best_device
        
        # Fallback a CPU
        logger.info("Usando CPU como fallback")
        return torch.device('cpu')
    
    @staticmethod
    def _check_gpu_memory(device: torch.device,
                         memory_threshold_gb: float) -> bool:
        """Verifica si hay memoria suficiente en la GPU"""
        try:
            if device.type != 'cuda':
                return True
            
            torch.cuda.set_device(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            
            logger.info(
                f"GPU {device}: {free_gb:.1f}GB libre de {total_gb:.1f}GB total"
            )
            
            return free_gb >= memory_threshold_gb
            
        except Exception as e:
            logger.warning(f"Error verificando memoria GPU: {e}")
            return False
    
    @staticmethod
    def _find_best_gpu(memory_threshold_gb: float) -> Optional[torch.device]:
        """Encuentra la mejor GPU disponible"""
        if not torch.cuda.is_available():
            return None
        
        best_device = None
        best_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            try:
                torch.cuda.set_device(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                free_memory = total_memory - allocated_memory
                
                free_gb = free_memory / (1024**3)
                
                if free_gb >= memory_threshold_gb and free_gb > best_free_memory:
                    best_free_memory = free_gb
                    best_device = device
                    
            except Exception as e:
                logger.warning(f"Error evaluando GPU {i}: {e}")
                continue
        
        if best_device:
            logger.info(
                f"Mejor GPU seleccionada: {best_device} "
                f"({best_free_memory:.1f}GB libre)"
            )
        
        return best_device


class DataPreprocessor:
    """Clase unificada para procesamiento de datos, eliminando duplicación"""
    
    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()
        self._is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str],
                    target_column: str, validation_split: float = 0.2) -> Dict:
        """
        Prepara datos unificadamente para entrenamiento y predicción.
        
        Args:
            df: DataFrame con datos
            feature_columns: Lista de columnas de características
            target_column: Columna objetivo
            validation_split: Fracción para validación
            
        Returns:
            Diccionario con datos preparados
        """
        # Validar entrada
        self._validate_input_data(df, feature_columns, target_column)
        
        # Preparar características y target
        X = df[feature_columns].fillna(0)
        y = df[target_column] if target_column in df.columns else None
        
        # División temporal si hay target
        if y is not None:
            split_idx = int(len(df) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Escalar datos de entrenamiento
            X_train_scaled = self._fit_transform_features(X_train, feature_columns)
            X_val_scaled = self._transform_features(X_val, feature_columns)
            
            return {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'split_info': {
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'split_idx': split_idx
                }
            }
        else:
            # Solo predicción
            X_scaled = self._transform_features(X, feature_columns)
            return {
                'X': X_scaled,
                'original_X': X
            }
    
    def _validate_input_data(self, df: pd.DataFrame, feature_columns: List[str],
                           target_column: str) -> None:
        """Valida datos de entrada"""
        if df.empty:
            raise ValueError("DataFrame está vacío")
        
        if not feature_columns:
            raise ValueError("Lista de características está vacía")
        
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Características faltantes: {list(missing_features)[:10]}"
            )
        
        if target_column in df.columns and df[target_column].isna().all():
            raise ValueError(f"Columna objetivo '{target_column}' está vacía")
    
    def _fit_transform_features(self, X: pd.DataFrame,
                              feature_columns: List[str]) -> pd.DataFrame:
        """Ajusta scaler y transforma características"""
        X_scaled_array = self.scaler.fit_transform(X)
        self._is_fitted = True
        return pd.DataFrame(
            X_scaled_array, columns=feature_columns, index=X.index
        )
    
    def _transform_features(self, X: pd.DataFrame,
                          feature_columns: List[str]) -> pd.DataFrame:
        """Transforma características usando scaler ajustado"""
        if not self._is_fitted:
            raise ValueError("Scaler debe ser ajustado antes de transformar")
        
        X_scaled_array = self.scaler.transform(X)
        return pd.DataFrame(
            X_scaled_array, columns=feature_columns, index=X.index
        )


class MetricsCalculator:
    """Calculadora unificada de métricas, eliminando duplicación"""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas básicas de regresión"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def calculate_accuracy_by_tolerance(y_true: np.ndarray, y_pred: np.ndarray,
                                      tolerances: List[float]) -> Dict[str, float]:
        """Calcula precisión por diferentes tolerancias"""
        results = {}
        for tolerance in tolerances:
            accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
            results[f'accuracy_{tolerance}pt'] = accuracy
        return results
    
    @staticmethod
    def analyze_prediction_stability(predictions_dict: Dict[str, np.ndarray],
                                   y_true: np.ndarray) -> Dict[str, Any]:
        """Analiza estabilidad de predicciones entre modelos"""
        metrics_by_model = {}
        
        for model_name, pred in predictions_dict.items():
            metrics_by_model[model_name] = MetricsCalculator.calculate_basic_metrics(
                y_true, pred
            )
        
        # Encontrar mejor modelo
        best_model = min(
            metrics_by_model.keys(),
            key=lambda k: metrics_by_model[k]['mae']
        )
        
        # Calcular estadísticas de estabilidad
        mae_values = [m['mae'] for m in metrics_by_model.values()]
        r2_values = [m['r2'] for m in metrics_by_model.values()]
        
        return {
            'metrics_by_model': metrics_by_model,
            'best_model': best_model,
            'stability_stats': {
                'mae_std': np.std(mae_values),
                'mae_range': np.max(mae_values) - np.min(mae_values),
                'r2_std': np.std(r2_values),
                'r2_range': np.max(r2_values) - np.min(r2_values)
            }
        }


class NBATeamPointsNet(nn.Module):
    """
    Red Neuronal Avanzada para Predicción de Puntos de Equipo NBA
    
    Arquitectura optimizada sin muchas capas pero con regularización agresiva:
    - Input Layer
    - 2 Hidden Layers con Layer Normalization y Dropout
    - Output Layer
    - Skip connections para mejor flujo de gradientes
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(NBATeamPointsNet, self).__init__()
        
        # Arquitectura compacta pero efectiva
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.hidden1 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.4)
        
        self.hidden2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.ln3 = nn.LayerNorm(hidden_size // 4)
        self.dropout3 = nn.Dropout(0.2)
        
        # Skip connection layer
        self.skip_layer = nn.Linear(input_size, hidden_size // 4)
        
        # Output layer
        self.output = nn.Linear(hidden_size // 4, 1)
        
        # Inicialización de pesos optimizada para regresión
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada de pesos para predicción NBA"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization para mejor convergencia
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass con skip connections y LayerNorm robusto"""
        # Guardar input para skip connection
        skip = self.skip_layer(x)
        
        # Capas principales con LayerNorm
        x = F.relu(self.ln1(self.input_layer(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.hidden1(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.hidden2(x)))
        x = self.dropout3(x)
        
        # Skip connection para mejor flujo de gradientes
        x = x + skip
        
        # Output final
        x = self.output(x)
        
        return x


class PyTorchNBARegressor(RegressorMixin):
    """
    Wrapper de PyTorch para integración con scikit-learn y stacking
    
    Implementa una red neuronal optimizada para predicción de puntos NBA
    con regularización agresiva, entrenamiento robusto y soporte GPU mejorado.
    """
    
    def __init__(self, hidden_size: int = 128, epochs: int = 150,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, early_stopping_patience: int = 15,
                 device: Optional[str] = None, memory_threshold_gb: float = 2.0):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device_preference = device
        self.memory_threshold_gb = memory_threshold_gb
        
        self.model = None
        self.data_preprocessor = DataPreprocessor()
        self.device = None
        
        # Para early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Configurar dispositivo óptimo
        self._setup_device()
    
    def _setup_device(self):
        """Configura el dispositivo óptimo para entrenamiento"""
        self.device = GPUUtils.get_optimal_device(
            device=self.device_preference,
            memory_threshold_gb=self.memory_threshold_gb
        )
    
    @property
    def _estimator_type(self):
        """Identifica este estimador como regresor para sklearn"""
        return "regressor"
    
    def _check_n_features(self, X, reset):
        """Método para compatibilidad con sklearn"""
        pass
    
    def score(self, X, y, sample_weight=None):
        """Calcula R² score para compatibilidad con sklearn"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def fit(self, X, y):
        """
        Entrenamiento con regularización agresiva y early stopping avanzado
        """
        # Preparar datos usando el preprocessor unificado
        X, y = self._prepare_input_data(X, y)
        
        # Usar DataPreprocessor para división y escalado interno
        preprocessor = DataPreprocessor()
        
        # División train/val para early stopping
        val_size = 0.15
        val_split = int(len(X) * (1 - val_size))
        
        if val_split < 2:
            val_split = max(2, len(X) - 1)
        
        X_train = X[:val_split]
        X_val = X[val_split:]
        y_train = y[:val_split]
        y_val = y[val_split:]
        
        # Ajustar batch_size si es necesario
        effective_batch_size = min(self.batch_size, len(X_train))
        if effective_batch_size < 2:
            effective_batch_size = len(X_train)
        
        # Convertir a tensores de PyTorch
        tensors = self._convert_to_tensors(X_train, y_train, X_val, y_val)
        
        # Crear modelo
        input_size = X.shape[1]
        self.model = NBATeamPointsNet(input_size, self.hidden_size).to(self.device)
        
        # Configurar entrenamiento
        optimizer, scheduler, criterion = self._setup_training_components()
        
        # Crear dataloader para entrenamiento
        train_dataset = TensorDataset(tensors['X_train'], tensors['y_train'])
        train_dataloader = DataLoader(
            train_dataset, batch_size=effective_batch_size, shuffle=True
        )
        
        # Entrenar con early stopping
        training_stats = self._train_with_early_stopping(
            train_dataloader, tensors, optimizer, scheduler, criterion
        )
        
        # Restaurar mejor modelo
        self._restore_best_model()
        
        # Log resultados del entrenamiento
        self._log_training_results(training_stats)
        
        return self
    
    def _prepare_input_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos de entrada para entrenamiento"""
        # Convertir a numpy arrays si es necesario
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        return X, y
    
    def _convert_to_tensors(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convierte arrays numpy a tensores PyTorch en el dispositivo correcto"""
        return {
            'X_train': torch.FloatTensor(X_train).to(self.device),
            'y_train': torch.FloatTensor(y_train).view(-1, 1).to(self.device),
            'X_val': torch.FloatTensor(X_val).to(self.device),
            'y_val': torch.FloatTensor(y_val).view(-1, 1).to(self.device)
        }
    
    def _setup_training_components(self):
        """Configura componentes para entrenamiento"""
        # Optimizador con regularización L2 agresiva
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler para learning rate dinámico
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8
        )
        
        # Función de pérdida robusta
        criterion = nn.SmoothL1Loss()
        
        return optimizer, scheduler, criterion
    
    def _train_with_early_stopping(self, train_dataloader, tensors,
                                  optimizer, scheduler, criterion) -> Dict:
        """Entrenamiento principal con early stopping avanzado"""
        # Inicializar variables de control
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs_completed': 0
        }
        
        logger.info(
            f"Iniciando entrenamiento red neuronal - "
            f"Train: {len(tensors['X_train'])}, Val: {len(tensors['X_val'])}"
        )
        
        for epoch in range(self.epochs):
            # Fase de entrenamiento
            train_loss = self._train_epoch(
                train_dataloader, optimizer, criterion
            )
            
            # Fase de validación
            val_loss = self._validate_epoch(tensors, criterion)
            
            # Actualizar scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar estadísticas
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['learning_rates'].append(current_lr)
            training_stats['epochs_completed'] = epoch + 1
            
            # Early stopping logic
            if self._check_early_stopping(val_loss, epoch):
                break
            
            # Log progreso
            self._log_epoch_progress(epoch, train_loss, val_loss, current_lr)
        
        return training_stats
    
    def _train_epoch(self, train_dataloader, optimizer, criterion) -> float:
        """Entrena una época"""
        self.model.train()
        epoch_train_loss = 0.0
        train_batch_count = 0
        
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batch_count += 1
        
        return epoch_train_loss / train_batch_count
    
    def _validate_epoch(self, tensors, criterion) -> float:
        """Valida una época"""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(tensors['X_val'])
            val_loss = criterion(val_outputs, tensors['y_val']).item()
        return val_loss
    
    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Verifica condiciones de early stopping"""
        improvement_threshold = 0.01
        
        # Verificar mejora
        if val_loss < (self.best_loss - improvement_threshold):
            self.best_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.patience_counter += 1
        
        # Condiciones de parada
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Early stopping por patience "
                f"({self.early_stopping_patience} épocas sin mejora)"
            )
            return True
        
        if val_loss < 0.5:
            logger.info(f"Early stopping por convergencia excelente")
            return True
        
        return False
    
    def _restore_best_model(self):
        """Restaura el mejor estado del modelo"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"Modelo restaurado al mejor estado "
                f"(val_loss={self.best_loss:.4f})"
            )
    
    def _log_epoch_progress(self, epoch: int, train_loss: float,
                          val_loss: float, current_lr: float):
        """Log del progreso por época"""
        if epoch % 10 == 0 or epoch < 10:
            status = "MEJOR" if self.patience_counter == 0 else f"{self.patience_counter}/{self.early_stopping_patience}"
            logger.info(
                f"Época {epoch+1:3d}: Train={train_loss:.4f}, "
                f"Val={val_loss:.4f} [{status}] LR={current_lr:.6f}"
            )
    
    def _log_training_results(self, training_stats: Dict):
        """Log de resultados finales del entrenamiento"""
        epochs_completed = training_stats['epochs_completed']
        val_losses = training_stats['val_losses']
        
        if not val_losses:
            return
        
        initial_val_loss = val_losses[0]
        final_val_loss = val_losses[-1]
        min_val_loss = min(val_losses)
        
        total_improvement = (
            (initial_val_loss - min_val_loss) / initial_val_loss 
            if initial_val_loss > 0 else 0
        )
    
    def predict(self, X):
        """Predicción usando la red neuronal entrenada"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Preparar datos
        X = self._prepare_prediction_data(X)
        
        self.model.eval()
        with torch.no_grad():
            # Manejar predicción por batches para evitar problemas de memoria
            predictions = self._predict_in_batches(X)
        
        # Aplicar límites realistas para puntos NBA
        predictions = np.clip(predictions, 70, 140)
        
        return predictions
    
    def _prepare_prediction_data(self, X) -> torch.Tensor:
        """Prepara datos para predicción"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        return torch.FloatTensor(X).to(self.device)
    
    def _predict_in_batches(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Predicción por batches para gestión eficiente de memoria"""
        if len(X_tensor) > 1000:
            predictions = []
            batch_size = 500
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                batch_pred = self.model(batch).cpu().numpy().flatten()
                predictions.extend(batch_pred)
            return np.array(predictions)
        else:
            return self.model(X_tensor).cpu().numpy().flatten()
    
    def get_params(self, deep=True):
        """Parámetros para compatibilidad con scikit-learn"""
        return {
            'hidden_size': self.hidden_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device_preference,
            'memory_threshold_gb': self.memory_threshold_gb
        }
    
    def set_params(self, **params):
        """Configurar parámetros para compatibilidad con scikit-learn"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reconfigurar dispositivo si cambió
        if 'device' in params or 'memory_threshold_gb' in params:
            self._setup_device()
        
        return self

class BaseNBATeamModel:
    """Clase base para modelos NBA de equipos con funcionalidades comunes."""
    
    def __init__(self, target_column: str, model_type: str = 'regression'):
        self.target_column = target_column
        self.model_type = model_type
        self.feature_columns = []
        self.trained_models = {}
        self.data_preprocessor = DataPreprocessor()
        self.is_trained = False


class TeamPointsModel(BaseNBATeamModel):
    """
    Modelo especializado para predicción de puntos de equipo por partido.
    
    Implementa un sistema ensemble con optimización automática de hiperparámetros
    y características específicamente diseñadas para maximizar la precisión
    en la predicción de puntos de equipo.
    """
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 optimization_method: str = 'random',
                 bayesian_n_calls: int = 20,
                 bayesian_acquisition: str = 'EI'):
        """
        Inicializa el modelo de puntos de equipo.
        
        Args:
            optimize_hyperparams: Si optimizar hiperparámetros automáticamente
            device: Dispositivo para PyTorch ('cuda:0', 'cuda:1', 'cpu', None)
            optimization_method: Método de optimización ('random', 'bayesian')
            bayesian_n_calls: Número de evaluaciones para optimización bayesiana
            bayesian_acquisition: Función de adquisición ('EI', 'PI', 'LCB')
        """
        super().__init__(
            target_column='PTS',
            model_type='regression'
        )
        
        self.feature_engineer = TeamPointsFeatureEngineer()
        self.optimize_hyperparams = optimize_hyperparams
        self.device_preference = device
        self.optimization_method = optimization_method.lower()
        self.bayesian_n_calls = bayesian_n_calls
        self.bayesian_acquisition = bayesian_acquisition
        self.best_model_name = None
        self.ensemble_weights = {}
        
        # Validar método de optimización
        if self.optimization_method not in ['random', 'bayesian']:
            raise ValueError(
                "optimization_method debe ser 'random' o 'bayesian'"
            )
        
        # Verificar disponibilidad de optimización bayesiana
        if self.optimization_method == 'bayesian' and not BAYESIAN_AVAILABLE:
            logger.warning(
                "scikit-optimize no disponible. Usando búsqueda aleatoria."
            )
            self.optimization_method = 'random'
        
        # Inicializar optimizador bayesiano si es necesario
        self.bayesian_optimizer = None
        if self.optimization_method == 'bayesian':
            self.bayesian_optimizer = BayesianHyperparameterOptimizer(
                n_calls=self.bayesian_n_calls,
                acquisition_func=self.bayesian_acquisition,
                random_state=42
            )
        
        # Stacking components
        self.stacking_model = None
        self.base_models = {}
        self.meta_model = None
        
        # Métricas de evaluación
        self.evaluation_metrics = {}
        
        # Configurar modelos optimizados para puntos de equipo
        self._setup_optimized_models()
        self._setup_stacking_model()
        
        # Log información de configuración
        logger.info(f"TeamPointsModel inicializado:")
        logger.info(f"   • Optimización: {self.optimization_method.upper()}")
        if self.optimization_method == 'bayesian':
            logger.info(f"   • Evaluaciones bayesianas: {self.bayesian_n_calls}")
            logger.info(f"   • Función adquisición: {self.bayesian_acquisition}")
        logger.info(f"   • Dispositivo preferido: {self.device_preference or 'auto'}")
    
    def _setup_optimized_models(self):
        """Configura modelos base optimizados para predicción de puntos de equipo."""
        
        # Modelos principales con REGULARIZACIÓN AGRESIVA
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_weight=8,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                max_delta_step=1,
                gamma=0.1
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_samples=35,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                min_split_gain=0.1,
                feature_fraction=0.8
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features=0.6,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                min_weight_fraction_leaf=0.01,
                max_leaf_nodes=500
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.75,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                alpha=0.9,
                max_features=0.6
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features=0.6,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                min_weight_fraction_leaf=0.01,
                max_leaf_nodes=500
            ),
            
            # RED NEURONAL con soporte GPU mejorado
            'pytorch_neural_net': PyTorchNBARegressor(
                hidden_size=128,
                epochs=100,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=0.01,
                early_stopping_patience=15,
                device=self.device_preference
            )
        }
        
        logger.info(
            "Modelos base configurados con REGULARIZACIÓN AGRESIVA "
            "para mayor estabilidad"
        )
    
    def _setup_stacking_model(self):
        """Configura el modelo de stacking robusto con REGULARIZACIÓN MÁXIMA."""
        
        # Modelos base para stacking con REGULARIZACIÓN EXTREMA
        base_models_stacking = [
            ('xgb_regularized', xgb.XGBRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.4,
                reg_lambda=0.4, min_child_weight=10, gamma=0.2,
                random_state=42, n_jobs=-1
            )),
            ('lgb_regularized', lgb.LGBMRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.4,
                reg_lambda=0.4, min_child_samples=40, min_split_gain=0.2,
                random_state=42, n_jobs=-1, verbosity=-1
            )),
            ('rf_regularized', RandomForestRegressor(
                n_estimators=100, max_depth=6, min_samples_split=20,
                min_samples_leaf=10, max_features=0.5, max_leaf_nodes=300,
                random_state=42, n_jobs=-1
            )),
            ('gb_regularized', GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.7, min_samples_split=20, alpha=0.9,
                random_state=42, max_features=0.5
            )),
            ('et_regularized', ExtraTreesRegressor(
                n_estimators=100, max_depth=6, min_samples_split=20,
                min_samples_leaf=10, max_features=0.5, max_leaf_nodes=300,
                random_state=42, n_jobs=-1
            )),
            # Neural Network MEJORADA con GPU inteligente
            ('pytorch_nn', PyTorchNBARegressor(
                hidden_size=96,
                epochs=100,
                batch_size=64,
                learning_rate=0.002,
                weight_decay=0.02,
                early_stopping_patience=12,
                device=self.device_preference
            ))
        ]
        
        # Meta-modelo con REGULARIZACIÓN MÁXIMA
        meta_model = Ridge(
            alpha=10.0,
            random_state=42,
            max_iter=2000,
            solver='auto'
        )
        
        # Stacking con validación cruzada más robusta
        self.stacking_model = StackingRegressor(
            estimators=base_models_stacking,
            final_estimator=meta_model,
            cv=7,
            n_jobs=-1,
            passthrough=False
        )
        
        # Guardar modelos base para análisis posterior
        self.base_models = dict(base_models_stacking)
        self.meta_model = meta_model
        
        logger.info(
            "Modelo de stacking configurado con REGULARIZACIÓN MÁXIMA "
            "+ Red Neuronal mejorada"
        )
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las columnas de características específicas para puntos de equipo.
        
        Args:
            df: DataFrame con datos de equipos
            
        Returns:
            Lista de nombres de características
        """
        # Generar todas las características usando el feature engineer
        features = self.feature_engineer.generate_all_features(df)
        
        # Filtrar características que realmente existen en el DataFrame
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(
                f"Características faltantes para equipo: {missing}"
            )
            logger.info("Características faltantes más comunes:")
            for feat in list(missing)[:10]:
                logger.info(f"  - {feat}")
        
        logger.info(
            f"Características disponibles para puntos de equipo: "
            f"{len(available_features)}"
        )
        return available_features
    
    def train(self, df: pd.DataFrame,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el modelo con validación temporal y optimización de hiperparámetros.
        
        Args:
            df: DataFrame con datos de entrenamiento
            validation_split: Fracción de datos para validación
            
        Returns:
            Métricas de entrenamiento y validación
        """
        logger.info("Iniciando entrenamiento del modelo de puntos de equipo...")
        
        # Generar características
        logger.info("Generando características avanzadas...")
        self.feature_columns = self.get_feature_columns(df)
        
        if len(self.feature_columns) == 0:
            raise ValueError("No se encontraron características válidas")
        
        # Preparar datos usando DataPreprocessor unificado
        data_prepared = self.data_preprocessor.prepare_data(
            df, self.feature_columns, self.target_column, validation_split
        )
        
        X_train = data_prepared['X_train']
        X_val = data_prepared['X_val']
        y_train = data_prepared['y_train']
        y_val = data_prepared['y_val']
        
        logger.info(
            f"División temporal: {data_prepared['split_info']['train_size']} "
            f"entrenamiento, {data_prepared['split_info']['val_size']} validación"
        )
        
        # Entrenar modelos individuales
        logger.info("Entrenando modelos individuales...")
        model_predictions = self._train_individual_models(
            X_train, y_train, X_val, y_val
        )
        
        # Entrenar ensemble models
        ensemble_predictions = self._train_ensemble_models(
            X_train, y_train, X_val, y_val
        )
        
        # Combinar todas las predicciones
        all_predictions_val = {**model_predictions['val'], **ensemble_predictions['val']}
        all_predictions_train = {**model_predictions['train'], **ensemble_predictions['train']}
        
        # Validación cruzada para modelo stacking
        logger.info("Ejecutando validación cruzada...")
        cv_scores = self._perform_cross_validation(X_train, y_train)
        
        # Seleccionar mejor modelo usando MetricsCalculator
        logger.info("Seleccionando mejor modelo...")
        analysis_results = MetricsCalculator.analyze_prediction_stability(
            all_predictions_val, y_val
        )
        self.best_model_name = analysis_results['best_model']
        
        # Análisis de rendimiento
        best_pred_train = all_predictions_train[self.best_model_name]
        best_pred_val = all_predictions_val[self.best_model_name]
        
        metrics = self._analyze_model_performance_cv(
            y_train, best_pred_train, y_val, best_pred_val,
            ensemble_predictions['train']['stacking'], ensemble_predictions['val']['stacking'],
            ensemble_predictions['train']['voting'], ensemble_predictions['val']['voting'],
            cv_scores
        )
        
        self.is_trained = True
        logger.info("Entrenamiento completado exitosamente")
        
        # Guardar el modelo final de producción
        self.save_production_model()
        
        return metrics
    
    def _train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Entrena modelos individuales y retorna predicciones"""
        model_predictions_train = {}
        model_predictions_val = {}
        
        for name, model in self.models.items():
            logger.info(f"Entrenando {name}...")
            
            # Optimización de hiperparámetros si está habilitada
            if (self.optimize_hyperparams and 
                name in ['xgboost', 'lightgbm', 'pytorch_neural_net']):
                model = self._optimize_model_hyperparams(
                    model, X_train, y_train, name
                )
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            
            model_predictions_train[name] = pred_train
            model_predictions_val[name] = pred_val
            
            # Guardar modelo entrenado
            self.trained_models[name] = model
            
            # Métricas individuales usando MetricsCalculator
            metrics = MetricsCalculator.calculate_basic_metrics(y_val, pred_val)
            logger.info(f"{name} - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.4f}")
        
        return {
            'train': model_predictions_train,
            'val': model_predictions_val
        }
    
    def _train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Entrena modelos ensemble y retorna predicciones"""
        ensemble_predictions_train = {}
        ensemble_predictions_val = {}
        
        # Entrenar ensemble voting
        logger.info("Entrenando ensemble voting con Red Neuronal...")
        voting_models = [(name, model) for name, model in self.trained_models.items()]
        
        if not voting_models:
            logger.warning("No hay modelos válidos para voting")
            voting_models = [list(self.trained_models.items())[0]]
        
        voting_regressor = VotingRegressor(voting_models)
        logger.info(
            f"VotingRegressor creado con {len(voting_models)} modelos: "
            f"{[name for name, _ in voting_models]}"
        )
        voting_regressor.fit(X_train, y_train)
        self.trained_models['voting'] = voting_regressor
        
        voting_pred_train = voting_regressor.predict(X_train)
        voting_pred_val = voting_regressor.predict(X_val)
        ensemble_predictions_train['voting'] = voting_pred_train
        ensemble_predictions_val['voting'] = voting_pred_val
        
        # Entrenar stacking
        logger.info("Entrenando stacking avanzado...")
        self.stacking_model.fit(X_train, y_train)
        self.trained_models['stacking'] = self.stacking_model
        
        stacking_pred_train = self.stacking_model.predict(X_train)
        stacking_pred_val = self.stacking_model.predict(X_val)
        ensemble_predictions_train['stacking'] = stacking_pred_train
        ensemble_predictions_val['stacking'] = stacking_pred_val
        
        return {
            'train': ensemble_predictions_train,
            'val': ensemble_predictions_val
        }
    
    def _optimize_model_hyperparams(self, model, X_train, y_train, model_name):
        """
        Optimiza hiperparámetros usando búsqueda aleatoria o bayesiana.
        
        Args:
            model: Modelo a optimizar
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento  
            model_name: Nombre del modelo
            
        Returns:
            Modelo optimizado
        """
        logger.info(
            f"Optimizando hiperparámetros para {model_name} "
            f"usando {self.optimization_method.upper()}..."
        )
        
        # Usar optimización bayesiana si está disponible y seleccionada
        if self.optimization_method == 'bayesian':
            return self._optimize_with_bayesian(model, X_train, y_train, model_name)
        else:
            return self._optimize_with_random_search(model, X_train, y_train, model_name)
    
    def _optimize_with_bayesian(self, model, X_train, y_train, model_name):
        """Optimiza usando búsqueda bayesiana"""
        if model_name == 'xgboost':
            return self.bayesian_optimizer.optimize_xgboost(model, X_train, y_train)
        elif model_name == 'lightgbm':
            return self.bayesian_optimizer.optimize_lightgbm(model, X_train, y_train)
        elif model_name == 'pytorch_neural_net':
            return self.bayesian_optimizer.optimize_pytorch_neural_net(model, X_train, y_train)
        else:
            logger.warning(f"Optimización bayesiana no implementada para {model_name}, usando búsqueda aleatoria")
            return self._optimize_with_random_search(model, X_train, y_train, model_name)
    
    def _optimize_with_random_search(self, model, X_train, y_train, model_name):
        """Optimiza usando búsqueda aleatoria (método original)"""
        # Parámetros específicos por modelo
        param_distributions = self._get_param_distributions(model_name)
        
        if not param_distributions:
            return model
        
        # Búsqueda aleatoria con validación cruzada temporal
        random_search = RandomizedSearchCV(
            model, param_distributions,
            n_iter=12 if model_name == 'pytorch_neural_net' else 15,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1 if model_name != 'pytorch_neural_net' else 1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(
            f"Mejores parámetros REGULARIZADOS para {model_name}: "
            f"{random_search.best_params_}"
        )
        return random_search.best_estimator_
    
    def _get_param_distributions(self, model_name: str) -> Dict:
        """Obtiene distribuciones de parámetros para optimización"""
        param_distributions = {
            'xgboost': {
                'n_estimators': [200, 300, 400],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.02, 0.03, 0.05],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.2, 0.3, 0.5],
                'reg_lambda': [0.2, 0.3, 0.5],
                'min_child_weight': [8, 10, 15],
                'gamma': [0.1, 0.2, 0.3]
            },
            'lightgbm': {
                'n_estimators': [200, 300, 400],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.02, 0.03, 0.05],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.2, 0.3, 0.5],
                'reg_lambda': [0.2, 0.3, 0.5],
                'min_child_samples': [30, 40, 50],
                'min_split_gain': [0.1, 0.2, 0.3],
                'feature_fraction': [0.7, 0.8, 0.9]
            },
            'pytorch_neural_net': {
                'hidden_size': [64, 96, 128],
                'learning_rate': [0.0005, 0.001, 0.002],
                'weight_decay': [0.01, 0.02, 0.03],
                'batch_size': [32, 64],
                'epochs': [120, 150, 180],
                'early_stopping_patience': [10, 15, 20]
            }
        }
        
        return param_distributions.get(model_name, {})
    
    def _perform_cross_validation(self, X, y):
        """Ejecuta validación cruzada temporal ROBUSTA para el modelo stacking."""
        # Validación cruzada temporal MÁS ROBUSTA
        tscv = TimeSeriesSplit(n_splits=7)
        
        # MAE scores con más evaluaciones
        mae_scores = cross_val_score(
            self.stacking_model, X, y, cv=tscv,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        mae_scores = -mae_scores
        
        # R² scores
        r2_scores = cross_val_score(
            self.stacking_model, X, y, cv=tscv,
            scoring='r2', n_jobs=-1
        )
        
        # Accuracy scores (tolerancia ±3 puntos)
        def accuracy_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            y_pred_clipped = np.clip(y_pred, 70, 160)
            return MetricsCalculator.calculate_accuracy_by_tolerance(
                y, y_pred_clipped, [3.0]
            )['accuracy_3.0pt']
        
        accuracy_scores = cross_val_score(
            self.stacking_model, X, y, cv=tscv,
            scoring=accuracy_scorer, n_jobs=-1
        )
        
        cv_results = {
            'mae_scores': mae_scores,
            'r2_scores': r2_scores,
            'accuracy_scores': accuracy_scores,
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'cv_stability': np.std(mae_scores) / np.mean(mae_scores),
            'mae_min': np.min(mae_scores),
            'mae_max': np.max(mae_scores),
            'mae_range': np.max(mae_scores) - np.min(mae_scores)
        }
        
        logger.info(
            f"Validación cruzada ROBUSTA (7-fold) - "
            f"MAE: {cv_results['mean_mae']:.3f}±{cv_results['std_mae']:.3f}"
        )
        logger.info(
            f"Estabilidad CV (std/mean): {cv_results['cv_stability']:.3f}"
        )
        logger.info(
            f"Rango MAE: [{cv_results['mae_min']:.3f}, "
            f"{cv_results['mae_max']:.3f}] (±{cv_results['mae_range']:.3f})"
        )
        logger.info(
            f"Validación cruzada - R²: "
            f"{cv_results['mean_r2']:.4f}±{cv_results['std_r2']:.4f}"
        )
        logger.info(
            f"Validación cruzada - Precisión ±3pts: "
            f"{cv_results['mean_accuracy']:.1f}%±{cv_results['std_accuracy']:.1f}%"
        )
        
        return cv_results
    
    def _analyze_model_performance_cv(self, y_train, pred_train, y_val, pred_val,
                                     stacking_train, stacking_val,
                                     voting_train, voting_val, cv_scores):
        """Análisis completo del rendimiento del modelo con validación cruzada."""
        
        # Métricas usando MetricsCalculator unificado
        train_metrics = MetricsCalculator.calculate_basic_metrics(y_train, pred_train)
        val_metrics = MetricsCalculator.calculate_basic_metrics(y_val, pred_val)
        stacking_metrics = MetricsCalculator.calculate_basic_metrics(y_val, stacking_val)
        voting_metrics = MetricsCalculator.calculate_basic_metrics(y_val, voting_val)
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("ANÁLISIS DE RENDIMIENTO - MODELO PUNTOS DE EQUIPO")
        print("="*80)
        
        print(f"\nMEJOR MODELO: {self.best_model_name.upper()}")
        print(f"{'Métrica':<15} {'Entrenamiento':<15} {'Validación':<15} {'Diferencia':<15}")
        print("-" * 60)
        print(f"{'MAE':<15} {train_metrics['mae']:<15.3f} {val_metrics['mae']:<15.3f} "
              f"{abs(train_metrics['mae'] - val_metrics['mae']):<15.3f}")
        print(f"{'RMSE':<15} {train_metrics['rmse']:<15.3f} {val_metrics['rmse']:<15.3f} "
              f"{abs(train_metrics['rmse'] - val_metrics['rmse']):<15.3f}")
        print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {val_metrics['r2']:<15.4f} "
              f"{abs(train_metrics['r2'] - val_metrics['r2']):<15.4f}")
        
        # Análisis de overfitting y estabilidad
        self._analyze_model_stability(train_metrics, val_metrics, cv_scores)
        
        # Análisis de precisión por tolerancia
        print(f"\nPRECISIÓN POR TOLERANCIA (Validación Final):")
        tolerances = [1, 2, 3, 5, 7, 10]
        accuracy_results = MetricsCalculator.calculate_accuracy_by_tolerance(
            y_val, pred_val, tolerances
        )
        for tolerance in tolerances:
            acc = accuracy_results[f'accuracy_{tolerance}pt']
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        # Rendimiento de ensembles
        print(f"\nCOMPARACIÓN DE ENSEMBLES (Validación Final):")
        print(f"{'Modelo':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 50)
        print(f"{'Mejor Individual':<20} {val_metrics['mae']:<10.3f} "
              f"{val_metrics['rmse']:<10.3f} {val_metrics['r2']:<10.4f}")
        print(f"{'Voting':<20} {voting_metrics['mae']:<10.3f} "
              f"{voting_metrics['rmse']:<10.3f} {voting_metrics['r2']:<10.4f}")
        print(f"{'Stacking':<20} {stacking_metrics['mae']:<10.3f} "
              f"{stacking_metrics['rmse']:<10.3f} {stacking_metrics['r2']:<10.4f}")
        
        # Validación cruzada detallada
        self._print_cross_validation_results(cv_scores)
        
        # Guardar métricas
        self.evaluation_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'stacking': stacking_metrics,
            'voting': voting_metrics,
            'cross_validation': cv_scores,
            'best_model': self.best_model_name
        }
        
        return self.evaluation_metrics
    
    def _analyze_model_stability(self, train_metrics: Dict, val_metrics: Dict,
                               cv_scores: Dict):
        """Analiza estabilidad del modelo y overfitting"""
        mae_diff = abs(train_metrics['mae'] - val_metrics['mae'])
        r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
        cv_stability = cv_scores.get('cv_stability', 0)
        mae_range = cv_scores.get('mae_range', 0)
        
        print(f"\nANÁLISIS DE ROBUSTEZ:")
        print(f"Estabilidad CV (std/mean): {cv_stability:.3f}")
        print(f"Rango MAE en CV: ±{mae_range:.3f}")
        print(f"Diferencia Entrenamiento-Validación MAE: {mae_diff:.3f}")
        
        # Clasificación de estabilidad
        if cv_stability < 0.15 and mae_range < 1.0:
            print("Modelo MUY ESTABLE - Excelente robustez")
        elif cv_stability < 0.25 and mae_range < 2.0:
            print("Modelo ESTABLE - Buena robustez")
        elif cv_stability < 0.35:
            print("Modelo MODERADAMENTE ESTABLE - Aceptable con cuidado")
        else:
            print("Modelo INESTABLE - Requiere más regularización")
        
        # Evaluación de overfitting
        if mae_diff < 1.0 and r2_diff < 0.03:
            print("Sin overfitting - Excelente generalización")
        elif mae_diff < 2.0 and r2_diff < 0.08:
            print("Overfitting mínimo - Buena generalización")
        elif mae_diff < 3.0 and r2_diff < 0.15:
            print("Ligero overfitting - Monitorear en producción")
        else:
            print("Overfitting significativo - Aumentar regularización")
        
        # Recomendaciones específicas
        self._print_improvement_recommendations(cv_stability, mae_range)
    
    def _print_improvement_recommendations(self, cv_stability: float,
                                         mae_range: float):
        """Imprime recomendaciones para mejorar el modelo"""
        if cv_stability > 0.3:
            print("\n🔧 RECOMENDACIONES PARA MEJORAR ESTABILIDAD:")
            print("- Aumentar regularización (alpha, lambda)")
            print("- Reducir complejidad del modelo (max_depth, n_estimators)")
            print("- Incrementar min_samples_split y min_samples_leaf")
            print("- Considerar más datos de entrenamiento")
        
        if mae_range > 2.0:
            print("\n🔧 RECOMENDACIONES PARA REDUCIR VARIABILIDAD:")
            print("- Usar ensemble con más modelos base")
            print("- Incrementar CV folds en stacking")
            print("- Aplicar feature selection más agresiva")
            print("- Normalizar características de entrada")
    
    def _print_cross_validation_results(self, cv_scores: Dict):
        """Imprime resultados detallados de validación cruzada"""
        print(f"\nVALIDACIÓN CRUZADA (7-FOLD TEMPORAL):")
        print(f"MAE: {cv_scores['mean_mae']:.3f} ± {cv_scores['std_mae']:.3f}")
        print(f"R²: {cv_scores['mean_r2']:.4f} ± {cv_scores['std_r2']:.4f}")
        print(f"Precisión ±3pts: {cv_scores['mean_accuracy']:.1f}% ± "
              f"{cv_scores['std_accuracy']:.1f}%")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones usando el mejor modelo entrenado.
        
        Args:
            df: DataFrame con datos para predicción
            
        Returns:
            Array con predicciones de puntos
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Generar características
        _ = self.feature_engineer.generate_all_features(df)
        
        # Verificar que las características existen en el DataFrame
        available_features = [f for f in self.feature_columns if f in df.columns]
        if len(available_features) != len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            logger.warning(f"Características faltantes para predicción: {list(missing_features)[:5]}")
        
        # Preparar datos para predicción (sin target)
        X = df[available_features].fillna(0)
        
        # Escalar usando el scaler entrenado
        X_scaled_array = self.data_preprocessor.scaler.transform(X)
        X_scaled = pd.DataFrame(
            X_scaled_array, columns=available_features, index=X.index
        )
        
        # Usar el mejor modelo
        best_model = self.trained_models[self.best_model_name]
        predictions = best_model.predict(X_scaled)
        
        # Aplicar límites realistas para puntos de equipo NBA
        predictions = np.clip(predictions, 70, 160)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Obtiene importancia de características del mejor modelo.
        
        Args:
            top_n: Número de características más importantes a retornar
            
        Returns:
            Diccionario con importancia de características
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        result = {}
        best_model = self.trained_models[self.best_model_name]
        
        # Obtener importancia según el tipo de modelo
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_)
        else:
            logger.warning(
                f"No se puede obtener importancia para {self.best_model_name}"
            )
            return result
        
        # Crear DataFrame con importancias
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Top características
        top_features = feature_importance_df.head(top_n)
        
        result = {
            'top_features': top_features.to_dict('records'),
            'feature_groups': self._analyze_feature_groups(feature_importance_df),
            'model_used': self.best_model_name
        }
        
        # Mostrar resultados
        print(f"\nTOP {top_n} CARACTERÍSTICAS MÁS IMPORTANTES:")
        print(f"{'Característica':<40} {'Importancia':<15}")
        print("-" * 55)
        for _, row in top_features.iterrows():
            print(f"{row['feature']:<40} {row['importance']:<15.6f}")
        
        return result
    
    def _analyze_feature_groups(self, feature_importance_df: pd.DataFrame) -> Dict[str, float]:
        """Analiza importancia por grupos de características."""
        groups = self.feature_engineer.get_feature_importance_groups()
        group_importance = {}
        
        for group_name, group_features in groups.items():
            group_features_in_model = [
                f for f in group_features if f in self.feature_columns
            ]
            if group_features_in_model:
                group_total = feature_importance_df[
                    feature_importance_df['feature'].isin(group_features_in_model)
                ]['importance'].sum()
                group_importance[group_name] = group_total
        
        return group_importance
    
    def validate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida el modelo en un conjunto de datos independiente.
        
        Args:
            df: DataFrame con datos de validación
            
        Returns:
            Métricas de validación
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Verificar que el target existe en los datos
        if self.target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{self.target_column}' no encontrada en datos de validación")
        
        # Realizar predicciones
        predictions = self.predict(df)
        y_true = df[self.target_column]
        
        # Asegurar que tenemos el mismo número de predicciones y targets
        min_length = min(len(predictions), len(y_true))
        predictions = predictions[:min_length]
        y_true = y_true.iloc[:min_length]
        
        # Calcular métricas usando MetricsCalculator
        basic_metrics = MetricsCalculator.calculate_basic_metrics(y_true, predictions)
        accuracy_metrics = MetricsCalculator.calculate_accuracy_by_tolerance(
            y_true, predictions, [1, 2, 3, 5]
        )
        
        # Combinar métricas
        metrics = {**basic_metrics, **accuracy_metrics}
        
        logger.info("Validación completada:")
        logger.info(f"MAE: {metrics['mae']:.3f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"Precisión ±3pts: {metrics['accuracy_3pt']:.1f}%")
        
        return metrics
    
    def save_production_model(self, save_path: str = None):
        """
        Guarda el modelo de producción final en la carpeta trained_models.
        
        Args:
            save_path: Ruta personalizada para guardar. Si None, usa ruta por defecto.
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")
        
        # Configurar ruta de guardado
        if save_path is None:
            os.makedirs("trained_models", exist_ok=True)
            save_path = "trained_models/teams_points.joblib"
        
        # Preparar objeto del modelo para producción
        production_model = {
            'model': self.trained_models[self.best_model_name],
            'scaler': self.data_preprocessor.scaler,
            'feature_columns': self.feature_columns,
            'feature_engineer': self.feature_engineer,
            'best_model_name': self.best_model_name,
            'evaluation_metrics': self.evaluation_metrics,
            'target_column': self.target_column,
            'model_metadata': self._create_model_metadata()
        }
        
        try:
            # Guardar con joblib para compatibilidad sklearn
            joblib.dump(production_model, save_path)
            self._log_model_saved(save_path, production_model)
            
        except Exception as e:
            logger.error(f"Error al guardar modelo de produccion: {e}")
            raise
    
    def _create_model_metadata(self) -> Dict[str, Any]:
        """Crea metadatos del modelo para producción"""
        val_metrics = self.evaluation_metrics.get('validation', {})
        
        return {
            'training_date': datetime.now().isoformat(),
            'model_type': 'NBA Team Points Predictor',
            'version': '1.0',
            'accuracy_3pts': val_metrics.get('accuracy_3pt', None),
            'mae': val_metrics.get('mae', None),
            'r2': val_metrics.get('r2', None),
            'total_features': len(self.feature_columns),
            'best_model': self.best_model_name,
            'device_used': getattr(self, 'device_preference', 'auto')
        }
    
    def _log_model_saved(self, save_path: str, production_model: Dict):
        """Log información del modelo guardado"""
        metadata = production_model['model_metadata']
        
        logger.info("[OK] MODELO DE PRODUCCION GUARDADO EXITOSAMENTE:")
        logger.info(f"   • Ruta: {save_path}")
        logger.info(f"   • Mejor modelo: {self.best_model_name}")
        logger.info(f"   • Features: {len(self.feature_columns)}")
        logger.info(f"   • MAE: {metadata['mae']:.3f}")
        logger.info(f"   • R²: {metadata['r2']:.4f}")
        
        accuracy_3pts = metadata['accuracy_3pts']
        if accuracy_3pts is not None:
            logger.info(f"   • Precision ±3pts: {accuracy_3pts:.1f}%")
        else:
            logger.info(f"   • Precision ±3pts: No disponible")
        
        logger.info(f"   • Fecha: {metadata['training_date']}")
        logger.info(f"   • Dispositivo: {metadata['device_used']}")
    
    @staticmethod
    def load_production_model(model_path: str = "trained_models/teams_points.joblib"):
        """
        Carga un modelo de producción guardado.
        
        Args:
            model_path: Ruta del modelo guardado
            
        Returns:
            Diccionario con el modelo y metadatos
        """
        try:
            production_model = joblib.load(model_path)
            
            metadata = production_model['model_metadata']
            
            logger.info("[OK] MODELO DE PRODUCCION CARGADO:")
            logger.info(f"   • Ruta: {model_path}")
            logger.info(f"   • Modelo: {production_model['best_model_name']}")
            logger.info(f"   • Features: {len(production_model['feature_columns'])}")
            logger.info(f"   • Fecha entrenamiento: {metadata['training_date']}")
            logger.info(f"   • MAE: {metadata['mae']:.3f}")
            logger.info(f"   • R²: {metadata['r2']:.4f}")
            logger.info(f"   • Dispositivo: {metadata.get('device_used', 'N/A')}")
            
            return production_model
            
        except Exception as e:
            logger.error(f"Error al cargar modelo de produccion: {e}")
            raise
    
    def get_bayesian_optimization_results(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene resultados de la optimización bayesiana si fue utilizada.
        
        Returns:
            Diccionario con resultados de optimización bayesiana o None
        """
        if self.optimization_method != 'bayesian' or not self.bayesian_optimizer:
            logger.info("No se utilizó optimización bayesiana en este modelo")
            return None
        
        if not self.bayesian_optimizer.optimization_results:
            logger.info("No hay resultados de optimización bayesiana disponibles")
            return None
        
        return self.bayesian_optimizer.get_optimization_summary()
    
    def plot_bayesian_convergence(self, model_name: str = None):
        """
        Grafica la convergencia de la optimización bayesiana.
        
        Args:
            model_name: Nombre del modelo específico. Si None, grafica todos.
        """
        if self.optimization_method != 'bayesian' or not self.bayesian_optimizer:
            logger.warning("No se utilizó optimización bayesiana en este modelo")
            return
        
        if not self.bayesian_optimizer.optimization_results:
            logger.warning("No hay resultados de optimización bayesiana para graficar")
            return
        
        if model_name:
            self.bayesian_optimizer.plot_convergence(model_name)
        else:
            # Graficar todos los modelos optimizados
            for model in self.bayesian_optimizer.optimization_results.keys():
                self.bayesian_optimizer.plot_convergence(model)
    
    def compare_optimization_methods(self, df: pd.DataFrame, 
                                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Compara diferentes métodos de optimización de hiperparámetros.
        
        Args:
            df: DataFrame con datos para comparación
            validation_split: Fracción de datos para validación
            
        Returns:
            Diccionario con comparación de métodos
        """
        if not BAYESIAN_AVAILABLE:
            logger.warning("scikit-optimize no disponible para comparación")
            return {}
        
        logger.info("Iniciando comparación de métodos de optimización...")
        
        # Preparar datos
        self.feature_columns = self.get_feature_columns(df)
        data_prepared = self.data_preprocessor.prepare_data(
            df, self.feature_columns, self.target_column, validation_split
        )
        
        X_train = data_prepared['X_train']
        y_train = data_prepared['y_train']
        X_val = data_prepared['X_val']
        y_val = data_prepared['y_val']
        
        results = {}
        
        # Probar con búsqueda aleatoria
        logger.info("Evaluando búsqueda aleatoria...")
        random_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        random_optimized = self._optimize_with_random_search(
            random_model, X_train, y_train, 'xgboost'
        )
        random_optimized.fit(X_train, y_train)
        random_pred = random_optimized.predict(X_val)
        random_metrics = MetricsCalculator.calculate_basic_metrics(y_val, random_pred)
        results['random_search'] = random_metrics
        
        # Probar con optimización bayesiana
        logger.info("Evaluando optimización bayesiana...")
        bayesian_optimizer = BayesianHyperparameterOptimizer(
            n_calls=self.bayesian_n_calls, random_state=42
        )
        bayesian_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        bayesian_optimized = bayesian_optimizer.optimize_xgboost(
            bayesian_model, X_train, y_train
        )
        bayesian_optimized.fit(X_train, y_train)
        bayesian_pred = bayesian_optimized.predict(X_val)
        bayesian_metrics = MetricsCalculator.calculate_basic_metrics(y_val, bayesian_pred)
        results['bayesian_optimization'] = bayesian_metrics
        
        # Análisis comparativo
        mae_improvement = ((random_metrics['mae'] - bayesian_metrics['mae']) / 
                          random_metrics['mae'] * 100)
        r2_improvement = ((bayesian_metrics['r2'] - random_metrics['r2']) / 
                         abs(random_metrics['r2']) * 100)
        
        results['comparison'] = {
            'mae_improvement_pct': mae_improvement,
            'r2_improvement_pct': r2_improvement,
            'bayesian_better': bayesian_metrics['mae'] < random_metrics['mae'],
            'bayesian_convergence': bayesian_optimizer.optimization_results.get('xgboost', {})
        }
        
        # Log resultados
        logger.info("COMPARACIÓN DE MÉTODOS DE OPTIMIZACIÓN:")
        logger.info(f"Búsqueda Aleatoria - MAE: {random_metrics['mae']:.4f}, R²: {random_metrics['r2']:.4f}")
        logger.info(f"Optimización Bayesiana - MAE: {bayesian_metrics['mae']:.4f}, R²: {bayesian_metrics['r2']:.4f}")
        logger.info(f"Mejora en MAE: {mae_improvement:+.2f}%")
        logger.info(f"Mejora en R²: {r2_improvement:+.2f}%")
        
        if bayesian_metrics['mae'] < random_metrics['mae']:
            logger.info("✅ Optimización bayesiana obtuvo MEJOR rendimiento")
        else:
            logger.info("⚠️ Búsqueda aleatoria obtuvo mejor rendimiento")
        
        return results

class BayesianHyperparameterOptimizer:
    """
    Optimizador bayesiano de hiperparámetros para modelos NBA
    
    Implementa búsqueda bayesiana usando scikit-optimize para encontrar
    hiperparámetros óptimos de manera más eficiente que búsqueda aleatoria.
    """
    
    def __init__(self, n_calls: int = 20, random_state: int = 42,
                 acquisition_func: str = 'EI', n_jobs: int = 1):
        """
        Inicializa el optimizador bayesiano.
        
        Args:
            n_calls: Número de evaluaciones de la función objetivo
            random_state: Semilla para reproducibilidad
            acquisition_func: Función de adquisición ('EI', 'PI', 'LCB')
            n_jobs: Número de trabajos paralelos (1 para PyTorch)
        """
        self.n_calls = n_calls
        self.random_state = random_state
        self.acquisition_func = acquisition_func
        self.n_jobs = n_jobs
        self.optimization_results = {}
        
        if not BAYESIAN_AVAILABLE:
            raise ImportError(
                "scikit-optimize no está instalado. "
                "Instálalo con: pip install scikit-optimize"
            )
    
    def optimize_xgboost(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                        cv_folds: int = 5) -> Dict:
        """Optimiza hiperparámetros de XGBoost usando búsqueda bayesiana"""
        
        # Definir espacio de búsqueda para XGBoost
        space = [
            Integer(150, 500, name='n_estimators'),
            Integer(3, 6, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(0.1, 0.6, name='reg_alpha'),
            Real(0.1, 0.6, name='reg_lambda'),
            Integer(5, 20, name='min_child_weight'),
            Real(0.05, 0.4, name='gamma')
        ]
        
        @use_named_args(space)
        def objective(**params):
            # Crear modelo con parámetros específicos
            temp_model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **params
            )
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(
                temp_model, X_train, y_train,
                cv=tscv, scoring='neg_mean_absolute_error',
                n_jobs=self.n_jobs
            )
            
            # Retornar MAE (minimizar)
            return -np.mean(scores)
        
        # Ejecutar optimización bayesiana
        logger.info(f"Iniciando optimización bayesiana XGBoost ({self.n_calls} evaluaciones)...")
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acquisition_func=self.acquisition_func.lower(),
            n_jobs=self.n_jobs
        )
        
        # Extraer mejores parámetros
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1],
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4],
            'reg_alpha': result.x[5],
            'reg_lambda': result.x[6],
            'min_child_weight': result.x[7],
            'gamma': result.x[8],
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Crear modelo optimizado
        optimized_model = xgb.XGBRegressor(**best_params)
        
        self.optimization_results['xgboost'] = {
            'best_params': best_params,
            'best_score': result.fun,
            'n_evaluations': len(result.func_vals),
            'convergence': result.func_vals
        }
        
        logger.info(f"XGBoost optimizado - Mejor MAE: {result.fun:.4f}")
        logger.info(f"Mejores parámetros: {best_params}")
        
        return optimized_model
    
    def optimize_lightgbm(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                         cv_folds: int = 5) -> Dict:
        """Optimiza hiperparámetros de LightGBM usando búsqueda bayesiana"""
        
        # Definir espacio de búsqueda para LightGBM
        space = [
            Integer(150, 500, name='n_estimators'),
            Integer(4, 8, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(0.1, 0.6, name='reg_alpha'),
            Real(0.1, 0.6, name='reg_lambda'),
            Integer(20, 60, name='min_child_samples'),
            Real(0.05, 0.4, name='min_split_gain'),
            Real(0.6, 0.95, name='feature_fraction')
        ]
        
        @use_named_args(space)
        def objective(**params):
            # Crear modelo con parámetros específicos
            temp_model = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
                **params
            )
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(
                temp_model, X_train, y_train,
                cv=tscv, scoring='neg_mean_absolute_error',
                n_jobs=self.n_jobs
            )
            
            # Retornar MAE (minimizar)
            return -np.mean(scores)
        
        # Ejecutar optimización bayesiana
        logger.info(f"Iniciando optimización bayesiana LightGBM ({self.n_calls} evaluaciones)...")
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acquisition_func=self.acquisition_func.lower(),
            n_jobs=self.n_jobs
        )
        
        # Extraer mejores parámetros
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1],
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4],
            'reg_alpha': result.x[5],
            'reg_lambda': result.x[6],
            'min_child_samples': result.x[7],
            'min_split_gain': result.x[8],
            'feature_fraction': result.x[9],
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        # Crear modelo optimizado
        optimized_model = lgb.LGBMRegressor(**best_params)
        
        self.optimization_results['lightgbm'] = {
            'best_params': best_params,
            'best_score': result.fun,
            'n_evaluations': len(result.func_vals),
            'convergence': result.func_vals
        }
        
        logger.info(f"LightGBM optimizado - Mejor MAE: {result.fun:.4f}")
        logger.info(f"Mejores parámetros: {best_params}")
        
        return optimized_model
    
    def optimize_pytorch_neural_net(self, model, X_train: pd.DataFrame, 
                                   y_train: pd.Series, cv_folds: int = 3) -> Dict:
        """Optimiza hiperparámetros de PyTorch Neural Net usando búsqueda bayesiana"""
        
        # Espacio de búsqueda más conservador para redes neuronales
        space = [
            Integer(32, 256, name='hidden_size'),
            Real(0.0001, 0.01, name='learning_rate'),
            Real(0.005, 0.05, name='weight_decay'),
            Categorical([16, 32, 64, 128], name='batch_size'),
            Integer(80, 200, name='epochs'),
            Integer(8, 25, name='early_stopping_patience')
        ]
        
        @use_named_args(space)
        def objective(**params):
            # Crear modelo con parámetros específicos
            temp_model = PyTorchNBARegressor(
                hidden_size=params['hidden_size'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                early_stopping_patience=params['early_stopping_patience'],
                device=getattr(model, 'device_preference', None)
            )
            
            # Validación cruzada temporal (menos folds para NN por tiempo)
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(
                temp_model, X_train, y_train,
                cv=tscv, scoring='neg_mean_absolute_error',
                n_jobs=1  # PyTorch en serial
            )
            
            # Retornar MAE (minimizar)
            return -np.mean(scores)
        
        # Ejecutar optimización bayesiana
        logger.info(f"Iniciando optimización bayesiana PyTorch NN ({self.n_calls} evaluaciones)...")
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acquisition_func=self.acquisition_func.lower(),
            n_jobs=1  # PyTorch en serial
        )
        
        # Extraer mejores parámetros
        best_params = {
            'hidden_size': result.x[0],
            'learning_rate': result.x[1],
            'weight_decay': result.x[2],
            'batch_size': result.x[3],
            'epochs': result.x[4],
            'early_stopping_patience': result.x[5],
            'device': getattr(model, 'device_preference', None)
        }
        
        # Crear modelo optimizado
        optimized_model = PyTorchNBARegressor(**best_params)
        
        self.optimization_results['pytorch_neural_net'] = {
            'best_params': best_params,
            'best_score': result.fun,
            'n_evaluations': len(result.func_vals),
            'convergence': result.func_vals
        }
        
        logger.info(f"PyTorch NN optimizado - Mejor MAE: {result.fun:.4f}")
        logger.info(f"Mejores parámetros: {best_params}")
        
        return optimized_model
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de todas las optimizaciones realizadas"""
        summary = {}
        
        for model_name, results in self.optimization_results.items():
            summary[model_name] = {
                'best_mae': results['best_score'],
                'total_evaluations': results['n_evaluations'],
                'improvement_over_iterations': len(results['convergence']),
                'final_convergence': results['convergence'][-5:] if len(results['convergence']) >= 5 else results['convergence']
            }
        
        return summary
    
    def plot_convergence(self, model_name: str):
        """Grafica la convergencia de la optimización bayesiana"""
        if model_name not in self.optimization_results:
            logger.warning(f"No hay resultados para {model_name}")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            convergence = self.optimization_results[model_name]['convergence']
            
            plt.figure(figsize=(10, 6))
            plt.plot(convergence, 'b-', label='MAE por evaluación')
            plt.plot(np.minimum.accumulate(convergence), 'r-', label='Mejor MAE hasta el momento')
            plt.xlabel('Número de evaluaciones')
            plt.ylabel('MAE')
            plt.title(f'Convergencia Optimización Bayesiana - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib no disponible para graficar convergencia")
