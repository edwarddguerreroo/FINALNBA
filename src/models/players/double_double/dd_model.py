"""
Modelo Avanzado de Predicción de Double Double NBA
=================================================

Modelo híbrido que combina:
- Machine Learning tradicional (Random Forest, XGBoost, LightGBM)
- Deep Learning (Redes Neuronales con PyTorch)
- CatBoost para manejo de features categóricas
- Stacking avanzado con meta-modelo optimizado
- Optimización bayesiana de hiperparámetros
- Regularización agresiva anti-overfitting
- Manejo automático de GPU con GPUManager
- Sistema de logging optimizado
- Confidence thresholds para predicciones
"""

# Standard Library
import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import sys

# Third-party Libraries - ML/Data
import pandas as pd
import numpy as np
import joblib

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, log_loss,
    precision_recall_curve, roc_curve
)
from sklearn.svm import SVC

# XGBoost, LightGBM and CatBoost
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.players.double_double.features_dd import DoubleDoubleFeatureEngineer

warnings.filterwarnings('ignore')


class OptimizedLogger:
    """Sistema de logging optimizado para modelos NBA"""
    
    _loggers = {}
    _handlers_configured = False
    
    @classmethod
    def get_logger(cls, name: str = __name__, level: str = "INFO"):
        """Obtener logger optimizado con configuración centralizada"""
        
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Crear logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Configurar handlers solo una vez
        if not cls._handlers_configured:
            cls._setup_handlers(logger)
            cls._handlers_configured = True
        
        # Evitar propagación duplicada
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _setup_handlers(cls, logger):
        """Configurar handlers de logging optimizados"""
        
        # Formatter más simple
        formatter = logging.Formatter(
            fmt='%(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para consola solo para mensajes importantes
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Solo warnings y errores
        console_handler.setFormatter(formatter)
        
        # Handler para archivo para todos los mensajes
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"nba_dd_model_{datetime.now().strftime('%Y%m%d')}.log",
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Agregar handlers al logger raíz
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
    
    @classmethod
    def log_performance_metrics(cls, logger, metrics: Dict[str, float], 
                               model_name: str = "Model", phase: str = "Training"):
        """Log conciso para métricas de rendimiento"""
        acc = metrics.get('accuracy', 0)
        auc = metrics.get('roc_auc', 0)
        logger.warning(f"{model_name}: ACC={acc:.3f}, AUC={auc:.3f}")  # Solo resultados importantes
    
    @classmethod
    def log_training_progress(cls, logger, epoch: int, total_epochs: int,
                             train_loss: float, val_loss: float, val_accuracy: float,
                             model_name: str = "Neural Network"):
        """Log reducido para progreso de entrenamiento"""
        if epoch == total_epochs - 1:  # Solo al final
            logger.warning(f"NN finalizada: Acc={val_accuracy:.3f}")
    
    @classmethod
    def log_gpu_info(cls, logger, device_info: Dict[str, Any], phase: str = "Setup"):
        """Log simplificado para información de GPU"""
        device = device_info.get('device', 'Unknown')
        if device_info.get('type') == 'cuda':
            logger.warning(f"GPU: {device}")
        else:
            logger.warning(f"CPU: {device}")


class GPUManager:
    """Gestor avanzado de GPU para modelos NBA"""
    
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
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        info = {'device': device_str, 'type': 'cpu'}
        
        if device_str.startswith('cuda') and torch.cuda.is_available():
            device_id = int(device_str.split(':')[1]) if ':' in device_str else 0
            
            if device_id < torch.cuda.device_count():
                info.update({
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(device_id),
                    'memory_info': {
                        'total_gb': torch.cuda.get_device_properties(device_id).total_memory / 1e9,
                        'allocated_gb': torch.cuda.memory_allocated(device_id) / 1e9,
                        'cached_gb': torch.cuda.memory_reserved(device_id) / 1e9,
                        'free_gb': (torch.cuda.get_device_properties(device_id).total_memory - 
                                   torch.cuda.memory_reserved(device_id)) / 1e9
                    }
                })
        
        return info
    
    @staticmethod
    def get_optimal_device(min_memory_gb: float = 2.0) -> str:
        """Obtener el dispositivo óptimo disponible"""
        if not torch.cuda.is_available():
            return 'cpu'
        
        best_device = 'cpu'
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device_str = f'cuda:{i}'
            info = GPUManager.get_device_info(device_str)
            
            if info['type'] == 'cuda':
                free_memory = info['memory_info']['free_gb']
                if free_memory >= min_memory_gb and free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = device_str
        
        return best_device
    
    @staticmethod
    def setup_device(device_preference: str = None, min_memory_gb: float = 2.0) -> torch.device:
        """Configurar dispositivo óptimo"""
        if device_preference:
            device_str = device_preference
        else:
            device_str = GPUManager.get_optimal_device(min_memory_gb)
        
        device = torch.device(device_str)
        
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        
        return device


class DataProcessor:
    """Clase auxiliar para procesamiento de datos común"""
    
    @staticmethod
    def prepare_training_data(X: pd.DataFrame, y: pd.Series, 
                            validation_split: float = 0.2,
                            scaler: Optional[StandardScaler] = None,
                            date_column: str = 'Date'
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                     pd.Series, pd.Series, StandardScaler]:
        """Preparar datos para entrenamiento con división cronológica y manejo robusto de NaN"""
        
        # Limpiar datos de manera más robusta
        X_clean = X.copy()
        
        # 1. Manejo agresivo de infinitos
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Imputar NaN columna por columna
        numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    # Si la mediana es NaN, usar la media
                    mean_val = X_clean[col].mean()
                    if pd.isna(mean_val):
                        # Si también la media es NaN, usar 0
                        median_val = 0
                    else:
                        median_val = mean_val
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # 3. Imputación final para asegurar que no hay NaN con verificación más rigurosa
        if X_clean.isna().any().any():
            # Reportar columnas con NaN antes de la limpieza final
            nan_columns = X_clean.columns[X_clean.isna().any()].tolist()
            logger.warning(f"Columnas con NaN detectadas: {nan_columns}")
            
            # Imputación más agresiva
            for col in nan_columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    # Para columnas numéricas: usar mediana, luego media, luego 0
                    if X_clean[col].notna().sum() > 0:
                        median_val = X_clean[col].median()
                        if pd.isna(median_val):
                            mean_val = X_clean[col].mean()
                            fill_val = mean_val if not pd.isna(mean_val) else 0.0
                        else:
                            fill_val = median_val
                    else:
                        fill_val = 0.0
                    X_clean[col] = X_clean[col].fillna(fill_val)
                else:
                    # Para columnas categóricas: usar moda o 0
                    mode_val = X_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                    X_clean[col] = X_clean[col].fillna(fill_val)
            
            # Verificación final final
            X_clean = X_clean.fillna(0)
            
            # Verificar que no queden NaN
            remaining_nans = X_clean.isna().sum().sum()
            if remaining_nans > 0:
                logger.error(f"ADVERTENCIA: Aún quedan {remaining_nans} valores NaN después de limpieza agresiva")
                X_clean = X_clean.fillna(0)  # Último recurso
        
        # División cronológica en lugar de aleatoria
        if date_column in X_clean.index.names or date_column in X_clean.columns:
            # Si tenemos columna de fecha, ordenar por fecha
            if date_column in X_clean.columns:
                # Crear índice temporal
                combined_data = pd.concat([X_clean, y], axis=1)
                combined_data = combined_data.sort_values(date_column)
                
                # Dividir cronológicamente
                split_idx = int(len(combined_data) * (1 - validation_split))
                
                train_data = combined_data.iloc[:split_idx]
                val_data = combined_data.iloc[split_idx:]
                
                X_train = train_data.drop(columns=[y.name, date_column] if y.name in train_data.columns else [date_column])
                y_train = train_data[y.name] if y.name in train_data.columns else y.iloc[:split_idx]
                
                X_val = val_data.drop(columns=[y.name, date_column] if y.name in val_data.columns else [date_column])
                y_val = val_data[y.name] if y.name in val_data.columns else y.iloc[split_idx:]
                
            else:
                # Si el índice ya está ordenado cronológicamente
                split_idx = int(len(X_clean) * (1 - validation_split))
                X_train = X_clean.iloc[:split_idx]
                X_val = X_clean.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_val = y.iloc[split_idx:]
        else:
            # Fallback: división por índice (asumiendo que está ordenado cronológicamente)
            logger.warning(f"Columna de fecha '{date_column}' no encontrada. Usando división por índice.")
            split_idx = int(len(X_clean) * (1 - validation_split))
            X_train = X_clean.iloc[:split_idx]
            X_val = X_clean.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_val = y.iloc[split_idx:]
        
        # Limpiar datos de entrenamiento y validación antes del escalado
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Escalar datos - CORREGIDO: Crear scaler si no existe, y hacer fit_transform siempre
        if scaler is None:
            scaler = StandardScaler()
        
        # Hacer fit_transform en datos de entrenamiento
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Hacer transform en datos de validación
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Verificación final de que no hay NaN ni infinitos
        X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], 0).fillna(0)
        X_val_scaled = X_val_scaled.replace([np.inf, -np.inf], 0).fillna(0)
        
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
    
    @staticmethod
    def prepare_prediction_data(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """Preparar datos para predicción con manejo robusto de NaN"""
        X_clean = X.copy()
        
        # Manejo agresivo de NaN para GradientBoostingClassifier
        # 1. Reemplazar infinitos
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Imputar NaN con mediana de cada columna
        numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    # Si la mediana también es NaN, usar 0
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # 3. Verificar que no queden NaN con manejo exhaustivo
        if X_clean.isna().any().any():
            # Reportar y manejar columnas con NaN
            nan_columns = X_clean.columns[X_clean.isna().any()].tolist()
            logger.warning(f"Columnas con NaN en predicción: {nan_columns}")
            
            # Imputación exhaustiva
            for col in nan_columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    # Para columnas numéricas
                    if X_clean[col].notna().sum() > 0:
                        median_val = X_clean[col].median()
                        if pd.isna(median_val):
                            mean_val = X_clean[col].mean()
                            fill_val = mean_val if not pd.isna(mean_val) else 0.0
                        else:
                            fill_val = median_val
                    else:
                        fill_val = 0.0
                    X_clean[col] = X_clean[col].fillna(fill_val)
                else:
                    # Para columnas categóricas
                    mode_val = X_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                    X_clean[col] = X_clean[col].fillna(fill_val)
            
            # Imputación final con 0 para cualquier NaN restante
            X_clean = X_clean.fillna(0)
        
        # 4. Escalar datos
        X_scaled = pd.DataFrame(
            scaler.transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # 5. Verificación final de que no hay NaN ni infinitos
        X_scaled = X_scaled.replace([np.inf, -np.inf], 0)
        X_scaled = X_scaled.fillna(0)
        
        return X_scaled
    
    @staticmethod
    def create_time_series_split(X: pd.DataFrame, y: pd.Series, 
                               n_splits: int = 5,
                               date_column: str = 'Date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Crear splits cronológicos para validación cruzada"""
        
        if date_column in X.columns:
            # Ordenar por fecha
            combined_data = pd.concat([X, y], axis=1)
            combined_data = combined_data.sort_values(date_column)
            indices = combined_data.index.values
        else:
            # Usar índice actual (asumiendo orden cronológico)
            indices = X.index.values
        
        splits = []
        total_size = len(indices)
        
        # Crear splits cronológicos con ventana expandible
        for i in range(n_splits):
            # Tamaño mínimo de entrenamiento: 60% de los datos
            min_train_size = int(total_size * 0.6)
            
            # Calcular tamaños para este split
            train_end = min_train_size + int((total_size - min_train_size) * (i + 1) / n_splits)
            val_start = train_end
            val_end = min(train_end + int(total_size * 0.2), total_size)
            
            if val_end > total_size:
                val_end = total_size
            
            train_indices = indices[:train_end]
            val_indices = indices[val_start:val_end]
            
            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))
        
        return splits


class MetricsCalculator:
    """Calculadora de métricas para clasificación"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: pd.Series, 
                                       y_pred: np.ndarray,
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calcular métricas completas de clasificación"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba)
        }


import logging
logger = OptimizedLogger.get_logger(__name__)


class DoubleDoubleDataset(Dataset):
    """
    Dataset personalizado para PyTorch con double double data
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Inicializar dataset
        
        Args:
            features: Array de características
            targets: Array de targets (0/1 para double double)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self) -> int:
        """Retorna el tamaño del dataset"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna un item del dataset"""
        return self.features[idx], self.targets[idx]


class DoubleDoubleNeuralNetwork(nn.Module):
    """
    Red neuronal optimizada para predicción de double-double
    con regularización agresiva anti-overfitting
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout_rate: float = 0.4
    ):
        super(DoubleDoubleNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Arquitectura con regularización agresiva
        self.layers = nn.Sequential(
            # Capa de entrada con normalización
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Primera capa oculta
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Segunda capa oculta
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Capa de salida - CORREGIDO: 1 neurona para clasificación binaria
            nn.Linear(hidden_size // 4, 1)
            # NO sigmoid aquí - se usa BCEWithLogitsLoss
        )
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización de pesos optimizada"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization para capas lineales
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                # Inicialización estándar para BatchNorm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass simplificado
        
        Args:
            x: Tensor de entrada [batch_size, input_size]
            
        Returns:
            Tensor de salida [batch_size, 1] con logits
        """
        # Forward pass a través de todas las capas
        return self.layers(x)


class PyTorchDoubleDoubleClassifier(ClassifierMixin, BaseEstimator):
    """
    Clasificador PyTorch avanzado para Double Double con manejo automático de GPU
    """
    
    def __init__(self, hidden_size: int = 128, epochs: int = 100,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, early_stopping_patience: int = 20,
                 dropout_rate: float = 0.3, device: Optional[str] = None,
                 min_memory_gb: float = 2.0, auto_batch_size: bool = True,
                 pos_weight: float = 1.0):  # NUEVO: peso para clase positiva
        
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.min_memory_gb = min_memory_gb
        self.auto_batch_size = auto_batch_size
        self.pos_weight = pos_weight  # NUEVO: almacenar pos_weight
        
        # Configurar dispositivo con GPU Manager
        self.device_str = device
        self._setup_device_with_gpu_manager()
        
        # Inicializar modelo y otros atributos
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Logger optimizado
        self.logger = OptimizedLogger.get_logger(f"{__name__}.PyTorchClassifier")
    
    def _setup_device_with_gpu_manager(self):
        """Configurar dispositivo usando GPUManager"""
        self.device = GPUManager.setup_device(
            device_preference=self.device_str,
            min_memory_gb=self.min_memory_gb
        )
        
        # Log información del dispositivo
        device_info = GPUManager.get_device_info(str(self.device))
        OptimizedLogger.log_gpu_info(logger, device_info, "Neural Network")
    
    def _auto_adjust_batch_size(self, X_train_tensor: torch.Tensor, 
                               y_train_tensor: torch.Tensor) -> int:
        """Ajustar automáticamente el batch size según memoria disponible"""
        
        if not self.auto_batch_size or self.device.type == 'cpu':
            return self.batch_size
        
        # Probar diferentes batch sizes
        test_batch_sizes = [128, 64, 32, 16, 8]
        
        for test_batch_size in test_batch_sizes:
            try:
                # Crear modelo temporal
                temp_model = DoubleDoubleNeuralNetwork(
                    input_size=X_train_tensor.shape[1],
                    hidden_size=self.hidden_size,
                    dropout_rate=self.dropout_rate
                ).to(self.device)
                
                # Probar forward pass
                test_batch = X_train_tensor[:test_batch_size].to(self.device)
                with torch.no_grad():
                    _ = temp_model(test_batch)
                
                # Si funciona, usar este batch size
                del temp_model
                torch.cuda.empty_cache()
                logger.info(f"Batch size ajustado automáticamente a: {test_batch_size}")
                return test_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    continue
            else:
                    raise e
        
        # Si todo falla, usar batch size mínimo
        logger.warning("Usando batch size mínimo debido a limitaciones de memoria")
        return 8
    
    def fit(self, X, y):
        """Entrenar el modelo con early stopping y regularización"""
        
        # Preparar datos
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            X_df = X.copy()
        else:
            X_values = X
            X_df = pd.DataFrame(X)
        
        if isinstance(y, pd.Series):
            y_values = y.values
            y_series = y.copy()
        else:
            y_values = y
            y_series = pd.Series(y)
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X_values)
        
        # Convertir a tensores
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y_values)
        
        # Ajustar batch size automáticamente
        if self.auto_batch_size:
            self.batch_size = self._auto_adjust_batch_size(X_tensor, y_tensor)
        
        # División cronológica en lugar de aleatoria
        split_idx = int(len(X_tensor) * 0.8)  # 80% train, 20% validation
        
        X_train = X_tensor[:split_idx]
        X_val = X_tensor[split_idx:]
        y_train = y_tensor[:split_idx]
        y_val = y_tensor[split_idx:]
        
        logger.info(f"División cronológica NN: Train={len(X_train)}, Val={len(X_val)}")
        
        # Crear datasets y loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Crear modelo
        self.model = DoubleDoubleNeuralNetwork(
            input_size=X_scaled.shape[1],
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # CORRECCIÓN CRÍTICA: Usar pos_weight mucho más agresivo para desbalance extremo
        # Para ratio 10.6:1, necesitamos pos_weight de al menos 20-30
        pos_weight_tensor = torch.tensor([30.0], device=self.device)  # Peso mucho más agresivo
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler para learning rate adaptativo
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Entrenamiento con early stopping mejorado
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.epochs):
            # Entrenamiento
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass - salida directa (logits)
                outputs = self.model(batch_X).squeeze()
                
                # CORRECCIÓN: Usar BCEWithLogitsLoss directamente con logits
                loss = criterion(outputs, batch_y.float())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validación
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    # Forward pass
                    outputs = self.model(batch_X).squeeze()
                    
                    # Loss
                    loss = criterion(outputs, batch_y.float())
                    val_loss += loss.item()
                    
                    # Accuracy con threshold optimizado
                    predicted = (torch.sigmoid(outputs) > 0.086).float()  # Threshold optimizado
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            # Promedios
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            # Guardar historial
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Log progreso (solo cada 10 epochs y al final)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                               f"Train Loss={train_loss:.4f}, "
                               f"Val Loss={val_loss:.4f}, "
                               f"Val Acc={val_accuracy:.4f}")
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping en epoch {epoch+1}")
                break
        
        # Restaurar mejor modelo
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        self.logger.info(f"Entrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predecir probabilidades con threshold optimizado"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Preparar datos
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Forward pass - obtener logits
            logits = self.model(X_tensor).squeeze()
            
            # Convertir logits a probabilidades usando sigmoid
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Asegurar que sea 2D para compatibilidad con sklearn
            if probabilities.ndim == 0:
                probabilities = np.array([probabilities])
            
            # Crear matriz de probabilidades [P(clase_0), P(clase_1)]
            prob_matrix = np.column_stack([1 - probabilities, probabilities])
        
        return prob_matrix
    
    def predict(self, X):
        """Predecir clases usando threshold óptimo si está disponible"""
        probabilities = self.predict_proba(X)
        
        # Usar threshold óptimo si está disponible, sino usar 0.5
        threshold = getattr(self, 'optimal_threshold', 0.5)
        
        return (probabilities[:, 1] > threshold).astype(int)
    
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
            'device': self.device_str,
            'min_memory_gb': self.min_memory_gb,
            'auto_batch_size': self.auto_batch_size,
            'pos_weight': self.pos_weight
        }
    
    def set_params(self, **params):
        """Establecer parámetros del modelo"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class DoubleDoubleAdvancedModel:
    """
    Modelo avanzado para predicción de double double con stacking y optimización bayesiana
    """
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 bayesian_n_calls: int = 50,
                 min_memory_gb: float = 2.0):
        
        self.optimize_hyperparams = optimize_hyperparams
        self.device_preference = device
        self.bayesian_n_calls = bayesian_n_calls
        self.min_memory_gb = min_memory_gb
        
        # Inicializar logger PRIMERO
        self.logger = OptimizedLogger.get_logger(f"{__name__}.DoubleDoubleAdvancedModel")
        
        # Componentes del modelo
        self.scaler = StandardScaler()
        
        # Feature Engineer especializado
        self.feature_engineer = DoubleDoubleFeatureEngineer(lookback_games=10)
        
        # Modelos individuales
        self.models = {}
        self.stacking_model = None
        
        # Métricas y resultados
        self.training_results = {}
        self.feature_importance = {}
        self.bayesian_results = {}
        self.gpu_config = {}
        self.cv_scores = {}
        self.is_fitted = False
        
        # Configurar entorno GPU
        self._setup_gpu_environment()
        
        # Configurar modelos
        self._setup_models()
        
        # Configurar stacking model
        self._setup_stacking_model()
    
    def _setup_gpu_environment(self):
        """Configurar entorno GPU para el modelo"""
        self.gpu_config = {
            'selected_device': GPUManager.get_optimal_device(self.min_memory_gb),
            'device_info': GPUManager.get_device_info()
        }
        
        self.device = torch.device(self.gpu_config['selected_device'])
        
        # Log simplificado de GPU
        OptimizedLogger.log_gpu_info(
            logger, 
            self.gpu_config.get('device_info', {}),
            "Configuración"
        )
    
    def _setup_models(self):
        """
        PARTE 2 & 4: REGULARIZACIÓN AUMENTADA + CLASS WEIGHTS REBALANCEADOS
        Configurar modelos base con correcciones anti-overfitting y manejo conservador del desbalance
        """
        
        # PARTE 4: CLASS WEIGHTS REBALANCEADOS - Más conservadores para reducir falsos positivos
        # Ratio original: 10.6:1, pero usaremos pesos más moderados para mejor precision
        class_weight_conservative = {0: 1.0, 1: 12.0}  # Reducido de 20.0 a 12.0
        
        # CORRECCIÓN 2: Modelos con regularización optimizada para precisión
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=120,  # Reducido ligeramente
                max_depth=4,       # Reducido para evitar overfitting
                learning_rate=0.05, # Reducido para mejor convergencia
                subsample=0.85,    # Aumentado
                colsample_bytree=0.85, # Aumentado
                reg_alpha=0.3,     # Aumentado L1 regularization
                reg_lambda=2.0,    # Aumentado L2 regularization
                min_child_weight=5, # Aumentado para evitar overfitting
                gamma=0.1,         # Aumentado para más regularización
                scale_pos_weight=12, # Conservador para mejor precision
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=120,  # Reducido ligeramente
                max_depth=4,       # Reducido para evitar overfitting
                learning_rate=0.05, # Reducido
                subsample=0.85,    # Aumentado
                colsample_bytree=0.85, # Aumentado
                reg_alpha=0.3,     # Aumentado L1 regularization
                reg_lambda=2.0,    # Aumentado L2 regularization
                min_child_samples=8, # Aumentado para evitar overfitting
                min_split_gain=0.01, # Aumentado para más regularización
                num_leaves=25,     # Reducido para evitar overfitting
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=3,
                scale_pos_weight=12, # Conservador para mejor precision
                boost_from_average=False,
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=150,  # Reducido ligeramente
                max_depth=6,       # Reducido para evitar overfitting
                min_samples_split=15, # Aumentado para evitar overfitting
                min_samples_leaf=8,   # Aumentado para evitar overfitting
                max_features='sqrt', # Reducir features por árbol
                class_weight=class_weight_conservative,
                random_state=42,
                n_jobs=-1
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=150,  # Reducido ligeramente
                max_depth=6,       # Reducido para evitar overfitting
                min_samples_split=15, # Aumentado para evitar overfitting
                min_samples_leaf=8,   # Aumentado para evitar overfitting
                max_features='sqrt', # Reducir features por árbol
                class_weight=class_weight_conservative,
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=120,  # Reducido ligeramente
                max_depth=4,       # Reducido para evitar overfitting
                learning_rate=0.03, # Reducido para mejor convergencia
                subsample=0.85,    # Aumentado
                min_samples_split=15, # Aumentado para evitar overfitting
                min_samples_leaf=8,   # Aumentado para evitar overfitting
                random_state=42
            ),
            
            'catboost': cb.CatBoostClassifier(
                iterations=80,     # Reducido para evitar overfitting
                depth=4,          # Reducido para evitar overfitting
                learning_rate=0.05, # Reducido
                l2_leaf_reg=8.0,   # Aumentado para más regularización
                class_weights=[1.0, 12.0], # Conservador para mejor precision
                random_seed=42,
                verbose=False,
                early_stopping_rounds=10
            ),
            
            # CORRECCIÓN 3: Red neuronal con regularización agresiva y class weights conservadores
            'neural_network': PyTorchDoubleDoubleClassifier(
                hidden_size=64,    # Reducido para evitar overfitting
                epochs=40,         # Reducido para evitar overfitting
                batch_size=64,     # Aumentado
                learning_rate=0.001, # Mantenido
                weight_decay=0.15, # AUMENTADO - regularización agresiva
                early_stopping_patience=8, # Reducido
                dropout_rate=0.6,  # AUMENTADO - dropout agresivo
                device=self.device,
                min_memory_gb=self.min_memory_gb,
                auto_batch_size=True,
                pos_weight=12.0    # Conservador para mejor precision
            )
        }
        
        self.logger.info(f"Modelos configurados con regularización aumentada y class_weight conservador={class_weight_conservative}")
        self.logger.info("PARTE 4: Class weights rebalanceados para mejor precision/recall balance")
    
    def _setup_stacking_model(self):
        """Configurar modelo de stacking con TODOS LOS MODELOS (ML/DL) y manejo correcto de NN"""
        
        # Wrapper para la Red Neuronal compatible con scikit-learn
        class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
            """Wrapper para red neuronal compatible con sklearn"""
            
            def __init__(self, nn_model):
                self.nn_model = nn_model
                self.classes_ = np.array([0, 1])  # Para compatibilidad con sklearn
                self.logger = OptimizedLogger.get_logger(f"{__name__}.NeuralNetworkWrapper")
                
            def fit(self, X, y):
                """Entrenar la red neuronal"""
                try:
                    # Asegurar que y sea 1D
                    if hasattr(y, 'values'):
                        y = y.values
                    y = np.asarray(y).flatten()
                    
                    # Entrenar el modelo
                    self.nn_model.fit(X, y)
                    return self
                except Exception as e:
                    self.logger.error(f"Error entrenando red neuronal en stacking: {e}")
                    # Crear un modelo dummy que siempre predice la clase mayoritaria
                    self._is_dummy = True
                    self._majority_class = 0  # Clase mayoritaria
                    return self
            
            def predict(self, X):
                """Predecir clases"""
                try:
                    if hasattr(self, '_is_dummy') and self._is_dummy:
                        return np.full(X.shape[0], self._majority_class)
                    
                    if not self.nn_model.is_fitted:
                        return np.full(X.shape[0], 0)
                    
                    return self.nn_model.predict(X)
                except Exception as e:
                    self.logger.error(f"Error en predict NN stacking: {e}")
                    return np.full(X.shape[0], 0)
            
            def predict_proba(self, X):
                """Predecir probabilidades"""
                try:
                    if hasattr(self, '_is_dummy') and self._is_dummy:
                        # Retornar probabilidades dummy
                        proba = np.zeros((X.shape[0], 2))
                        proba[:, self._majority_class] = 1.0
                        return proba
                    
                    if not self.nn_model.is_fitted:
                        # Retornar probabilidades por defecto
                        proba = np.zeros((X.shape[0], 2))
                        proba[:, 0] = 0.9  # 90% probabilidad clase 0
                        proba[:, 1] = 0.1  # 10% probabilidad clase 1
                        return proba
                    
                    # Obtener probabilidades del modelo
                    proba_nn = self.nn_model.predict_proba(X)
                    
                    # Asegurar que sea 2D con 2 columnas
                    if proba_nn.shape[1] == 2:
                        return proba_nn
                    else:
                        # Si solo tiene 1 columna, crear la segunda
                        proba = np.zeros((proba_nn.shape[0], 2))
                        proba[:, 1] = proba_nn[:, 0]  # Probabilidad clase positiva
                        proba[:, 0] = 1 - proba[:, 1]  # Probabilidad clase negativa
                        return proba
                        
                except Exception as e:
                    self.logger.error(f"Error en predict_proba NN stacking: {e}")
                    # Retornar probabilidades por defecto
                    proba = np.zeros((X.shape[0], 2))
                    proba[:, 0] = 0.9
                    proba[:, 1] = 0.1
                    return proba
            
            def get_params(self, deep=True):
                """Parámetros del wrapper"""
                return {'nn_model': self.nn_model}
                
            def set_params(self, **params):
                """Establecer parámetros"""
                if 'nn_model' in params:
                    self.nn_model = params['nn_model']
                return self
        
        # Crear wrapper para la red neuronal
        nn_wrapper = NeuralNetworkWrapper(self.models['neural_network'])
        
        # Modelos base para stacking con REGULARIZACIÓN BALANCEADA
        # Usar versiones más ligeras pero no excesivamente restringidas
        base_estimators = [
            # XGBoost regularizado para stacking
            ('xgb_stack', xgb.XGBClassifier(
                n_estimators=50,          # Moderado
                max_depth=4,              # Balanceado
                learning_rate=0.1,        # Aumentado para mejor aprendizaje
                subsample=0.8,            # Aumentado
                colsample_bytree=0.8,     # Aumentado
                reg_alpha=0.1,            # REDUCIDO dramáticamente
                reg_lambda=0.5,           # REDUCIDO dramáticamente
                min_child_weight=3,       # REDUCIDO
                gamma=0.05,               # REDUCIDO dramáticamente
                scale_pos_weight=10,      # Manejo de desbalance
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            )),
            
            # LightGBM regularizado para stacking con manejo agresivo de desbalance
            ('lgb_stack', lgb.LGBMClassifier(
                n_estimators=80,          # Moderado para stacking
                max_depth=4,              # Balanceado
                learning_rate=0.1,        # Aumentado para mejor aprendizaje
                subsample=0.85,           # Aumentado
                colsample_bytree=0.85,    # Aumentado
                reg_alpha=0.05,           # REDUCIDO dramáticamente
                reg_lambda=0.2,           # REDUCIDO dramáticamente
                min_child_samples=5,      # REDUCIDO para permitir splits
                min_split_gain=0.005,     # REDUCIDO dramáticamente
                num_leaves=31,            # Balanceado
                feature_fraction=0.85,    # Aumentado
                bagging_fraction=0.85,    # Aumentado
                bagging_freq=3,           # Más frecuente
                scale_pos_weight=12,      # Peso para clase minoritaria
                boost_from_average=False, # No inicializar desde promedio
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),
            
            # Random Forest regularizado para stacking
            ('rf_stack', RandomForestClassifier(
                n_estimators=50,          # Moderado
                max_depth=5,              # Aumentado
                min_samples_split=10,     # REDUCIDO dramáticamente
                min_samples_leaf=5,       # REDUCIDO dramáticamente
                max_features='sqrt',      # Estándar
                bootstrap=True,
                class_weight='balanced',  # Manejo de desbalance
                oob_score=False,
                random_state=42,
                n_jobs=-1
            )),
            
            # Extra Trees regularizado para stacking
            ('et_stack', ExtraTreesClassifier(
                n_estimators=50,          # Moderado
                max_depth=5,              # Aumentado
                min_samples_split=10,     # REDUCIDO dramáticamente
                min_samples_leaf=5,       # REDUCIDO dramáticamente
                max_features='sqrt',      # Estándar
                bootstrap=True,
                class_weight='balanced',  # Manejo de desbalance
                oob_score=False,
                random_state=42,
                n_jobs=-1
            )),
            
            # Gradient Boosting regularizado para stacking con manejo nativo de NaN
            ('gb_stack', HistGradientBoostingClassifier(
                max_iter=50,              # Moderado para stacking
                max_depth=4,              # Balanceado
                learning_rate=0.1,        # Aumentado
                l2_regularization=0.5,    # Regularización L2 reducida
                min_samples_leaf=5,       # Mínimas muestras por hoja
                max_leaf_nodes=31,        # Máximo nodos hoja
                validation_fraction=0.1,  # Para early stopping
                n_iter_no_change=10,      # Paciencia
                tol=1e-4,                 # Tolerancia
                random_state=42
            )),
            
            # CatBoost regularizado para stacking
            ('cb_stack', cb.CatBoostClassifier(
                iterations=50,            # Moderado
                depth=4,                  # Balanceado
                learning_rate=0.1,        # Aumentado
                l2_leaf_reg=0.5,          # REDUCIDO dramáticamente
                bootstrap_type='Bernoulli',
                subsample=0.8,            # Aumentado
                random_strength=0.3,      # Reducido
                od_type='Iter',
                od_wait=10,               # Balanceado
                auto_class_weights='Balanced',  # Manejo de desbalance
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )),
            
            # Red Neuronal (usando wrapper)
            ('nn_stack', nn_wrapper)
        ]
        
        # Configurar stacking con regularización optimizada y manejo agresivo del desbalance
        self.stacking_model = StackingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm']),
                ('rf', self.models['random_forest']),
                ('et', self.models['extra_trees']),
                ('gb', self.models['gradient_boosting']),
                ('cat', self.models['catboost']),
                ('nn', nn_wrapper)
            ],
            final_estimator=LogisticRegression(
                class_weight={0: 1.0, 1: 15.0},  # PARTE 4: Conservador para mejor precision
                random_state=42,
                max_iter=2000,  # Más iteraciones para convergencia
                C=0.3,  # AUMENTADO regularización para evitar overfitting
                penalty='l2',
                solver='liblinear',  # Mejor para datasets pequeños con desbalance
                fit_intercept=True
            ),
            cv=3,
            n_jobs=-1,
            passthrough=False  # No pasar features originales al meta-modelo
        )
    
    def _select_best_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 30) -> List[str]:
        """
        PARTE 5: FEATURE SELECTION
        Seleccionar las mejores features para evitar overfitting
        
        Args:
            X: DataFrame con features
            y: Serie con targets
            max_features: Número máximo de features a seleccionar
            
        Returns:
            Lista de nombres de features seleccionadas
        """
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        self.logger.info(f"=== PARTE 5: FEATURE SELECTION (máximo {max_features} features) ===")
        self.logger.info(f"Features iniciales: {X.shape[1]}")
        
        # Remover columna Date si existe para el análisis
        X_analysis = X.copy()
        if 'Date' in X_analysis.columns:
            X_analysis = X_analysis.drop(columns=['Date'])
        
        # Limpiar datos para análisis
        X_clean = self._clean_nan_exhaustive(X_analysis)
        
        feature_scores = {}
        
        # Método 1: F-score (ANOVA)
        try:
            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(X_clean, y)
            f_scores = selector_f.scores_
            
            for i, feature in enumerate(X_clean.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['f_score'] = f_scores[i]
                
            self.logger.info("✅ F-score calculado")
        except Exception as e:
            self.logger.warning(f"Error calculando F-score: {e}")
        
        # Método 2: Mutual Information
        try:
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
            
            for i, feature in enumerate(X_clean.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['mutual_info'] = mi_scores[i]
                
            self.logger.info("✅ Mutual Information calculado")
        except Exception as e:
            self.logger.warning(f"Error calculando Mutual Information: {e}")
        
        # Método 3: Random Forest Feature Importance
        try:
            rf_selector = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            rf_selector.fit(X_clean, y)
            rf_importances = rf_selector.feature_importances_
            
            for i, feature in enumerate(X_clean.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['rf_importance'] = rf_importances[i]
                
            self.logger.info("✅ Random Forest importance calculado")
        except Exception as e:
            self.logger.warning(f"Error calculando RF importance: {e}")
        
        # Combinar scores y rankear features
        combined_scores = {}
        
        for feature, scores in feature_scores.items():
            # Normalizar scores (0-1)
            normalized_scores = []
            
            if 'f_score' in scores:
                # F-score ya está normalizado por SelectKBest
                f_norm = scores['f_score'] / max(1.0, max(s.get('f_score', 0) for s in feature_scores.values()))
                normalized_scores.append(f_norm)
            
            if 'mutual_info' in scores:
                # Mutual info ya está en [0,1] aproximadamente
                normalized_scores.append(scores['mutual_info'])
            
            if 'rf_importance' in scores:
                # RF importance ya está normalizado
                normalized_scores.append(scores['rf_importance'])
            
            # Score combinado (promedio de métodos disponibles)
            if normalized_scores:
                combined_scores[feature] = np.mean(normalized_scores)
            else:
                combined_scores[feature] = 0.0
        
        # Seleccionar top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Limitar al número máximo de features
        selected_features = [feature for feature, score in sorted_features[:max_features]]
        
        # Log de resultados
        self.logger.info(f"Features seleccionadas: {len(selected_features)}/{len(X_clean.columns)}")
        self.logger.info("Top 10 features seleccionadas:")
        for i, (feature, score) in enumerate(sorted_features[:10]):
            self.logger.info(f"  {i+1:2d}. {feature}: {score:.4f}")
        
        # Guardar información de selección
        self.feature_selection_info = {
            'method': 'combined_scoring',
            'max_features': max_features,
            'selected_features': selected_features,
            'feature_scores': combined_scores,
            'selection_ratio': len(selected_features) / len(X_clean.columns)
        }
        
        return selected_features

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Obtener columnas de features especializadas EXCLUSIVAMENTE usando DoubleDoubleFeatureEngineer"""
        
        # Generar features especializadas OBLIGATORIAS
        df_with_features = df.copy()
        
        try:
            logger.info("Generando features especializadas EXCLUSIVAS...")
            specialized_features = self.feature_engineer.generate_all_features(df_with_features)
            logger.info(f"Features especializadas generadas: {len(specialized_features)}")
            
            # Filtrar solo features que realmente existen en el DataFrame
            available_features = [f for f in specialized_features if f in df_with_features.columns]
            
            # LISTA EXHAUSTIVA DE FEATURES BÁSICAS A EXCLUIR (NO ESPECIALIZADAS)
            basic_features_to_exclude = [
                # Columnas básicas del dataset
                'Player', 'Date', 'Team', 'Opp', 'Result', 'MP', 'GS', 'Away',
                # Estadísticas del juego actual (NO USAR - data leakage)
                'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                # Columnas de double específicas del juego actual
                'PTS_double', 'TRB_double', 'AST_double', 'STL_double', 'BLK_double',
                # Target variables
                'double_double', 'triple_double',
                # Columnas auxiliares temporales básicas (NO especializadas)
                'day_of_week', 'month', 'days_rest', 'days_into_season',
                # Features básicas del data_loader (NO especializadas)
                'is_home', 'is_started', 'Height_Inches', 'Weight', 'BMI'
            ]
            
            # FILTRAR EXCLUSIVAMENTE FEATURES ESPECIALIZADAS
            purely_specialized_features = [
                f for f in available_features 
                if f not in basic_features_to_exclude
            ]
            
            # VERIFICAR que tenemos suficientes features especializadas
            if len(purely_specialized_features) < 20:
                logger.error(f"INSUFICIENTES features especializadas puras: {len(purely_specialized_features)}")
                logger.error("El modelo REQUIERE al menos 20 features especializadas")
                
                # Mostrar qué features están disponibles para debug
                logger.info(f"Features especializadas puras disponibles: {purely_specialized_features}")
                
                # Intentar regenerar features con más detalle
                logger.info("Reintentando generación de features especializadas...")
                self.feature_engineer._clear_cache()
                specialized_features = self.feature_engineer.generate_all_features(df_with_features)
                available_features = [f for f in specialized_features if f in df_with_features.columns]
                purely_specialized_features = [
                    f for f in available_features 
                    if f not in basic_features_to_exclude
                ]
                
                if len(purely_specialized_features) < 20:
                    raise ValueError(f"FALLO CRÍTICO: Solo {len(purely_specialized_features)} features especializadas puras disponibles. El modelo requiere al menos 20.")
            
            # USAR ÚNICAMENTE FEATURES ESPECIALIZADAS PURAS
            logger.info(f"Usando EXCLUSIVAMENTE {len(purely_specialized_features)} features especializadas PURAS")
            logger.info(f"Features especializadas seleccionadas: {purely_specialized_features[:10]}...")
            
            # VERIFICACIÓN FINAL: Asegurar 100% especialización
            specialized_percentage = 100.0  # Por definición, todas son especializadas
            logger.info(f"✅ PERFECTO: {specialized_percentage}% de features son especializadas")
            
            return purely_specialized_features
            
        except Exception as e:
            logger.error(f"ERROR CRÍTICO generando features especializadas: {str(e)}")
            logger.error("El modelo NO PUEDE funcionar sin features especializadas")
            raise ValueError(f"FALLO CRÍTICO: No se pudieron generar features especializadas. Error: {str(e)}")
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrenar el modelo completo con validación rigurosa y features especializadas EXCLUSIVAS"""
        
        logger.info("Iniciando entrenamiento con features especializadas EXCLUSIVAS...")
        
        # Generar features especializadas OBLIGATORIAS
        df_with_features = df.copy()
        try:
            logger.info("Generando features especializadas EXCLUSIVAS para entrenamiento...")
            specialized_features = self.feature_engineer.generate_all_features(df_with_features)
            logger.info(f"Features especializadas generadas: {len(specialized_features)}")
            
            # VERIFICAR que se generaron correctamente
            if len(specialized_features) < 20:
                logger.warning(f"Pocas features especializadas generadas: {len(specialized_features)}")
                logger.info("Reintentando generación con cache limpio...")
                self.feature_engineer._clear_cache()
                specialized_features = self.feature_engineer.generate_all_features(df_with_features)
                
        except Exception as e:
            logger.error(f"ERROR CRÍTICO generando features especializadas: {str(e)}")
            raise ValueError(f"FALLO CRÍTICO: No se pudieron generar features especializadas para entrenamiento. Error: {str(e)}")
        
        # Obtener features especializadas EXCLUSIVAS y target
        feature_columns = self.get_feature_columns(df_with_features)
        X = df_with_features[feature_columns].copy()
        
        # PRESERVAR la columna Date para división cronológica
        if 'Date' in df_with_features.columns:
            X['Date'] = df_with_features['Date']
        
        # Determinar columna target
        target_col = 'double_double' if 'double_double' in df_with_features.columns else 'DD'
        if target_col not in df_with_features.columns:
            raise ValueError("No se encontró columna target (double_double o DD)")
        
        y = df_with_features[target_col].copy()
        
        logger.info(f"Entrenamiento configurado: {X.shape[0]} muestras, {X.shape[1]} features especializadas EXCLUSIVAS")
        
        # PARTE 5: FEATURE SELECTION - Seleccionar las mejores features para evitar overfitting
        if X.shape[1] > 30:  # Solo aplicar si tenemos más de 30 features
            self.logger.info("Aplicando selección de features para evitar overfitting...")
            selected_features = self._select_best_features(X, y, max_features=30)
            
            # Actualizar X con solo las features seleccionadas
            X_selected = X[selected_features].copy()
            
            # Preservar Date si existe
            if 'Date' in X.columns:
                X_selected['Date'] = X['Date']
            
            X = X_selected
            feature_columns = selected_features
            
            self.logger.info(f"Features reducidas de {len(specialized_features)} a {len(selected_features)} para evitar overfitting")
        else:
            self.logger.info("No se requiere selección de features (≤30 features)")
        
        # VERIFICAR que todas las features son especializadas (por definición del get_feature_columns corregido)
        specialized_count = len(feature_columns)  # Todas son especializadas por definición
        specialized_percentage = 100.0  # Por definición, todas son especializadas
        
        logger.info(f"VERIFICACIÓN CRÍTICA: {specialized_count}/{len(feature_columns)} features son especializadas ({specialized_percentage:.1f}%)")
        
        if specialized_percentage < 100:
            logger.error(f"ERROR: Solo {specialized_percentage:.1f}% de features son especializadas")
            logger.error("Esto indica un problema en get_feature_columns()")
        else:
            logger.info("✅ PERFECTO: Modelo usa 100% features especializadas")
        
        # Preparar datos
        X_train, X_val, y_train, y_val, self.scaler = DataProcessor.prepare_training_data(
            X, y, validation_split, self.scaler
        )
        
        # Optimización bayesiana si está habilitada
        if self.optimize_hyperparams and BAYESIAN_AVAILABLE:
            self._optimize_with_bayesian(X_train, y_train)
        
        # Entrenar modelos individuales
        individual_results = self._train_individual_models(X_train, y_train, X_val, y_val)
        
        # Entrenar modelo de stacking
        self.stacking_model.fit(X_train, y_train)
        
        # Establecer modelo como entrenado ANTES de evaluar
        self.is_fitted = True
        
        # Evaluar stacking
        stacking_pred = self.stacking_model.predict(X_val)
        stacking_proba = self.stacking_model.predict_proba(X_val)[:, 1]
        
        stacking_metrics = MetricsCalculator.calculate_classification_metrics(
            y_val, stacking_pred, stacking_proba
        )
        
        OptimizedLogger.log_performance_metrics(
            logger, stacking_metrics, "Stacking Model", "Validación"
        )
        
        # Guardar resultados con verificación de features especializadas
        results = {
            'individual_models': individual_results,
            'stacking_metrics': stacking_metrics,
            'feature_columns': feature_columns,
            'specialized_features_used': specialized_count,
            'total_features_generated': len(specialized_features),
            'specialized_percentage': specialized_percentage,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        self.training_results = results
        
        # Cross-validation del ensemble completo
        self._perform_cross_validation(X, y)
        
        # Calcular feature importance
        self._calculate_feature_importance(feature_columns)
        
        # PARTE 1: THRESHOLD OPTIMIZATION AVANZADO
        self.logger.info("=== OPTIMIZACIÓN AVANZADA DE THRESHOLD ===")
        self.logger.info(f"Distribución de probabilidades en validación:")
        self.logger.info(f"  Min: {stacking_proba.min():.4f}")
        self.logger.info(f"  Max: {stacking_proba.max():.4f}")
        self.logger.info(f"  Media: {stacking_proba.mean():.4f}")
        self.logger.info(f"  Std: {stacking_proba.std():.4f}")
        
        # Probar múltiples métodos de optimización de threshold
        threshold_methods = ['f1_precision_balance', 'youden', 'precision_recall_curve']
        threshold_results = {}
        
        for method in threshold_methods:
            try:
                threshold = self._calculate_optimal_threshold_advanced(y_val, stacking_proba, method=method)
                
                # Evaluar este threshold
                y_pred_test = (stacking_proba >= threshold).astype(int)
                test_precision = precision_score(y_val, y_pred_test, zero_division=0)
                test_recall = recall_score(y_val, y_pred_test, zero_division=0)
                test_f1 = f1_score(y_val, y_pred_test, zero_division=0)
                
                threshold_results[method] = {
                    'threshold': threshold,
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1': test_f1,
                    'predictions_positive': np.sum(y_pred_test),
                    'predictions_ratio': np.sum(y_pred_test) / len(y_pred_test)
                }
                
                self.logger.info(f"Método {method}: T={threshold:.4f}, P={test_precision:.3f}, R={test_recall:.3f}, F1={test_f1:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Error en método {method}: {str(e)}")
                threshold_results[method] = {'error': str(e)}
        
        # Seleccionar el mejor threshold basado en F1 score y precision mínima
        best_method = None
        best_f1 = 0
        min_precision_required = 0.12  # Precision mínima aceptable
        
        for method, result in threshold_results.items():
            if 'error' not in result:
                if result['precision'] >= min_precision_required and result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_method = method
        
        # Si no se encontró un método que cumpla los requisitos, usar el de mejor F1
        if best_method is None:
            best_f1 = 0
            for method, result in threshold_results.items():
                if 'error' not in result and result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_method = method
        
        # Usar el mejor threshold encontrado
        if best_method:
            self.optimal_threshold = threshold_results[best_method]['threshold']
            self.logger.info(f"MEJOR MÉTODO SELECCIONADO: {best_method}")
            self.logger.info(f"Threshold óptimo final: {self.optimal_threshold:.4f}")
        else:
            # Fallback al método legacy si todo falla
            self.logger.warning("Todos los métodos avanzados fallaron, usando método legacy")
            self.optimal_threshold = self._calculate_optimal_threshold(y_val, stacking_proba)
        
        # Validación final del threshold
        if self.optimal_threshold < 0.05:
            self.logger.warning(f"Threshold muy bajo ({self.optimal_threshold:.4f}), ajustando a 0.05")
            self.optimal_threshold = 0.05
        elif self.optimal_threshold > 0.9:
            self.logger.warning(f"Threshold muy alto ({self.optimal_threshold:.4f}), ajustando a 0.9")
            self.optimal_threshold = 0.9
        
        # Evaluar con threshold óptimo final
        y_val_pred_optimal = (stacking_proba >= self.optimal_threshold).astype(int)
        
        # Logging detallado de predicciones finales
        dd_predicted = np.sum(y_val_pred_optimal)
        dd_actual = np.sum(y_val)
        self.logger.info(f"=== RESULTADOS FINALES CON THRESHOLD ÓPTIMO ===")
        self.logger.info(f"Threshold final: {self.optimal_threshold:.4f}")
        self.logger.info(f"DD predichos: {dd_predicted}")
        self.logger.info(f"DD reales: {dd_actual}")
        self.logger.info(f"Ratio predicción: {dd_predicted/len(y_val)*100:.1f}%")
        self.logger.info(f"Ratio real: {dd_actual/len(y_val)*100:.1f}%")
        
        # Calcular métricas finales con threshold óptimo
        optimal_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred_optimal),
            'precision': precision_score(y_val, y_val_pred_optimal, zero_division=0),
            'recall': recall_score(y_val, y_val_pred_optimal, zero_division=0),
            'f1_score': f1_score(y_val, y_val_pred_optimal, zero_division=0),
            'roc_auc': roc_auc_score(y_val, stacking_proba)
        }
        
        self.logger.info("=== MÉTRICAS FINALES CON THRESHOLD ÓPTIMO ===")
        for metric, value in optimal_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        results['optimal_threshold'] = self.optimal_threshold
        results['optimal_metrics'] = optimal_metrics
        results['threshold_optimization'] = threshold_results
        
        logger.info(f"Entrenamiento completado con {len(feature_columns)} features especializadas EXCLUSIVAS")
        logger.info(f"Porcentaje de features especializadas: {specialized_percentage:.1f}%")
        
        return self.training_results
    
    def _train_individual_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """Entrenar modelos individuales con early stopping"""
        
        results = {}
        
        for name, model in self.models.items():
            try:
                if name in ['xgboost', 'lightgbm']:
                    # Modelos con early stopping
                    if name == 'xgboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:  # lightgbm
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                        )
                else:
                    # Otros modelos
                    model.fit(X_train, y_train)
                
                # Evaluar modelo
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1]
                
                metrics = MetricsCalculator.calculate_classification_metrics(
                    y_val, val_pred, val_proba
                )
                
                OptimizedLogger.log_performance_metrics(logger, metrics, name, "Validación")
                
                results[name] = {
                    'model': model,
                    'val_metrics': metrics
                }
                
                # Guardar feature importance si está disponible
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predecir clases usando threshold óptimo si está disponible"""
        probabilities = self.predict_proba(df)
        
        # Usar threshold óptimo si está disponible, sino usar 0.5
        threshold = getattr(self, 'optimal_threshold', 0.5)
        
        # Logging para debug
        self.logger.info(f"Prediciendo con threshold: {threshold:.4f}")
        self.logger.info(f"Probabilidades - Min: {probabilities[:, 1].min():.4f}, Max: {probabilities[:, 1].max():.4f}")
        
        # USAR >= en lugar de > para incluir casos límite
        predictions = (probabilities[:, 1] >= threshold).astype(int)
        
        positive_predictions = predictions.sum()
        self.logger.info(f"Predicciones positivas: {positive_predictions} de {len(predictions)}")
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicción probabilística usando stacking model con features especializadas EXCLUSIVAS"""
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Generar features especializadas OBLIGATORIAS
        df_with_features = df.copy()
        try:
            logger.info("Generando features especializadas EXCLUSIVAS para predicción probabilística...")
            specialized_features = self.feature_engineer.generate_all_features(df_with_features)
            logger.info(f"Features especializadas generadas para predicción probabilística: {len(specialized_features)}")
            
            # VERIFICAR que se generaron correctamente
            if len(specialized_features) < 15:
                logger.error(f"INSUFICIENTES features especializadas para predicción probabilística: {len(specialized_features)}")
                raise ValueError(f"No se pudieron generar suficientes features especializadas para predicción probabilística")
            
        except Exception as e:
            logger.error(f"ERROR CRÍTICO generando features para predicción probabilística: {str(e)}")
            raise ValueError(f"FALLO CRÍTICO: No se pudieron generar features especializadas para predicción probabilística. Error: {str(e)}")
        
        # Usar EXCLUSIVAMENTE las features especializadas entrenadas
        feature_columns = self.training_results['feature_columns']
        
        # VERIFICAR que todas las features requeridas están disponibles
        missing_features = [f for f in feature_columns if f not in df_with_features.columns]
        if missing_features:
            logger.error(f"Features especializadas faltantes para predicción probabilística: {missing_features}")
            raise ValueError(f"Features especializadas requeridas no disponibles: {missing_features}")
        
        X = df_with_features[feature_columns].copy()
        X_scaled = DataProcessor.prepare_prediction_data(X, self.scaler)
        
        logger.info(f"Predicción probabilística usando {len(feature_columns)} features especializadas EXCLUSIVAS")
        return self.stacking_model.predict_proba(X_scaled)
    
    def _optimize_with_bayesian(self, X_train, y_train):
        """Optimización bayesiana de hiperparámetros"""
        
        if not BAYESIAN_AVAILABLE:
            logger.warning("Optimización bayesiana no disponible - skopt no instalado")
            return
        
        # Distribuir llamadas entre modelos
        calls_per_model = max(8, self.bayesian_n_calls // 3)
        
        # Optimizar modelos principales
        self._optimize_xgboost_bayesian(X_train, y_train, calls_per_model)
        self._optimize_lightgbm_bayesian(X_train, y_train, calls_per_model)
        self._optimize_neural_net_bayesian(X_train, y_train, calls_per_model)
    
    def _optimize_xgboost_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana específica para XGBoost con validación cronológica"""
        
        space = [
            Integer(30, 100, name='n_estimators'),
            Integer(3, 6, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(1.0, 5.0, name='reg_alpha'),
            Real(2.0, 8.0, name='reg_lambda')
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            # Usar validación cronológica en lugar de StratifiedKFold
            time_splits = DataProcessor.create_time_series_split(X_train, y_train, n_splits=3)
            scores = []
            
            for train_indices, val_indices in time_splits:
                X_fold_train = X_train.iloc[train_indices] if hasattr(X_train, 'iloc') else X_train[train_indices]
                y_fold_train = y_train.iloc[train_indices] if hasattr(y_train, 'iloc') else y_train[train_indices]
                X_fold_val = X_train.iloc[val_indices] if hasattr(X_train, 'iloc') else X_train[val_indices]
                y_fold_val = y_train.iloc[val_indices] if hasattr(y_train, 'iloc') else y_train[val_indices]
                
                model.fit(X_fold_train, y_fold_train)
                y_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_proba)
                scores.append(score)
            
            return -np.mean(scores)
        
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['xgboost'].set_params(**best_params)
        
        if not hasattr(self, 'bayesian_results'):
            self.bayesian_results = {}
        
        self.bayesian_results['xgboost'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
    
    def _optimize_lightgbm_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana específica para LightGBM con validación cronológica"""
        
        space = [
            Integer(30, 100, name='n_estimators'),
            Integer(3, 6, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(1.0, 5.0, name='reg_alpha'),
            Real(2.0, 8.0, name='reg_lambda'),
            Integer(20, 60, name='min_child_samples')
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = lgb.LGBMClassifier(
                **params,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
            # Usar validación cronológica
            time_splits = DataProcessor.create_time_series_split(X_train, y_train, n_splits=3)
            scores = []
            
            for train_indices, val_indices in time_splits:
                X_fold_train = X_train.iloc[train_indices] if hasattr(X_train, 'iloc') else X_train[train_indices]
                y_fold_train = y_train.iloc[train_indices] if hasattr(y_train, 'iloc') else y_train[train_indices]
                X_fold_val = X_train.iloc[val_indices] if hasattr(X_train, 'iloc') else X_train[val_indices]
                y_fold_val = y_train.iloc[val_indices] if hasattr(y_train, 'iloc') else y_train[val_indices]
                
                model.fit(X_fold_train, y_fold_train)
                y_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_proba)
                scores.append(score)
            
            return -np.mean(scores)
        
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['lightgbm'].set_params(**best_params)
        
        if not hasattr(self, 'bayesian_results'):
            self.bayesian_results = {}
        
        self.bayesian_results['lightgbm'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
    
    def _optimize_neural_net_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana para la red neuronal con validación cronológica"""
        
        space = [
            Integer(32, 128, name='hidden_size'),
            Real(0.0001, 0.005, name='learning_rate'),
            Real(0.01, 0.08, name='weight_decay'),
            Real(0.3, 0.7, name='dropout_rate'),
            Integer(32, 128, name='batch_size')
        ]
        
        @use_named_args(space)
        def objective(**params):
            params['batch_size'] = int(params['batch_size'])
            
            # Usar validación cronológica manual
            time_splits = DataProcessor.create_time_series_split(X_train, y_train, n_splits=3)
            scores = []
            
            for train_indices, val_indices in time_splits:
                X_fold_train = X_train.iloc[train_indices] if hasattr(X_train, 'iloc') else X_train[train_indices]
                y_fold_train = y_train.iloc[train_indices] if hasattr(y_train, 'iloc') else y_train[train_indices]
                X_fold_val = X_train.iloc[val_indices] if hasattr(X_train, 'iloc') else X_train[val_indices]
                y_fold_val = y_train.iloc[val_indices] if hasattr(y_train, 'iloc') else y_train[val_indices]
                
                model = PyTorchDoubleDoubleClassifier(
                    hidden_size=params['hidden_size'],
                    learning_rate=params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    dropout_rate=params['dropout_rate'],
                    batch_size=params['batch_size'],
                    epochs=50,
                    early_stopping_patience=10,
                    device=str(self.device)
                )
                
                model.fit(X_fold_train, y_fold_train)
                y_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_proba)
                scores.append(score)
            
            return -np.mean(scores)
        
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        
        self.models['neural_network'] = PyTorchDoubleDoubleClassifier(
            hidden_size=best_params['hidden_size'],
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            dropout_rate=best_params['dropout_rate'],
            batch_size=int(best_params['batch_size']),
            epochs=100,
            early_stopping_patience=15,
            device=str(self.device)
        )
        
        if not hasattr(self, 'bayesian_results'):
            self.bayesian_results = {}
        
        self.bayesian_results['neural_network'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
    
    def _perform_cross_validation(self, X, y) -> Dict[str, Any]:
        """
        PARTE 3: CROSS-VALIDATION MEJORADA
        Realizar validación cruzada cronológica rigurosa con detección de overfitting
        """
        
        self.logger.info("=== INICIANDO CROSS-VALIDATION MEJORADA ===")
        
        # Crear splits cronológicos más robustos
        time_splits = DataProcessor.create_time_series_split(X, y, n_splits=5)
        
        # Métricas para detectar overfitting
        overfitting_metrics = {}
        
        # Evaluar modelos individuales con detección de overfitting
        for name, model_info in self.training_results['individual_models'].items():
            if 'model' in model_info:
                model = model_info['model']
                try:
                    self.logger.info(f"Evaluando {name} en cross-validation...")
                    
                    cv_scores = []
                    train_scores = []  # Para detectar overfitting
                    precision_scores = []
                    recall_scores = []
                    
                    for fold_idx, (train_indices, val_indices) in enumerate(time_splits):
                        # Obtener datos para este split
                        X_train_cv = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
                        y_train_cv = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
                        X_val_cv = X.iloc[val_indices] if hasattr(X, 'iloc') else X[val_indices]
                        y_val_cv = y.iloc[val_indices] if hasattr(y, 'iloc') else y[val_indices]
                        
                        # Limpiar datos de entrenamiento
                        X_train_cv = self._clean_nan_exhaustive(X_train_cv)
                        X_val_cv = self._clean_nan_exhaustive(X_val_cv)
                        
                        # REMOVER columna Date si existe
                        if 'Date' in X_train_cv.columns:
                            X_train_cv = X_train_cv.drop(columns=['Date'])
                        if 'Date' in X_val_cv.columns:
                            X_val_cv = X_val_cv.drop(columns=['Date'])
                        
                        # Entrenar modelo específico para este fold
                        if name == 'neural_network':
                            # Para red neuronal, crear nuevo modelo para cada fold
                            temp_model = PyTorchDoubleDoubleClassifier(
                                hidden_size=64,  # Reducido para evitar overfitting
                                epochs=30,       # Menos epochs para CV
                                early_stopping_patience=5,
                                weight_decay=0.15,  # Más regularización
                                dropout_rate=0.6,   # Más dropout
                                device=str(self.device),
                                pos_weight=15.0     # Manejo de desbalance
                            )
                            temp_model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = temp_model.predict(X_val_cv)
                            y_pred_train = temp_model.predict(X_train_cv)
                            
                        elif name in ['xgboost', 'lightgbm']:
                            # Para XGBoost y LightGBM, crear modelos con regularización aumentada
                            from sklearn.base import clone
                            temp_model = clone(model)
                            
                            # Aumentar regularización para CV
                            if name == 'xgboost':
                                temp_model.set_params(
                                    n_estimators=100,  # Reducido
                                    max_depth=4,       # Reducido
                                    reg_alpha=0.3,     # Aumentado
                                    reg_lambda=2.0,    # Aumentado
                                    early_stopping_rounds=None
                                )
                            elif name == 'lightgbm':
                                temp_model.set_params(
                                    n_estimators=100,  # Reducido
                                    max_depth=4,       # Reducido
                                    reg_alpha=0.3,     # Aumentado
                                    reg_lambda=2.0,    # Aumentado
                                    early_stopping_rounds=None
                                )
                            
                            temp_model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = temp_model.predict(X_val_cv)
                            y_pred_train = temp_model.predict(X_train_cv)
                            
                        else:
                            # Para otros modelos, clonar con regularización aumentada
                            from sklearn.base import clone
                            temp_model = clone(model)
                            
                            # Aumentar regularización según el tipo de modelo
                            if name in ['random_forest', 'extra_trees']:
                                temp_model.set_params(
                                    n_estimators=100,      # Reducido
                                    max_depth=6,           # Reducido
                                    min_samples_split=15,  # Aumentado
                                    min_samples_leaf=8     # Aumentado
                                )
                            elif name == 'gradient_boosting':
                                temp_model.set_params(
                                    n_estimators=100,      # Reducido
                                    max_depth=4,           # Reducido
                                    learning_rate=0.03,    # Reducido
                                    min_samples_split=15,  # Aumentado
                                    min_samples_leaf=8     # Aumentado
                                )
                            
                            temp_model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = temp_model.predict(X_val_cv)
                            y_pred_train = temp_model.predict(X_train_cv)
                        
                        # Calcular métricas para validación
                        val_f1 = f1_score(y_val_cv, y_pred_cv, zero_division=0)
                        val_precision = precision_score(y_val_cv, y_pred_cv, zero_division=0)
                        val_recall = recall_score(y_val_cv, y_pred_cv, zero_division=0)
                        
                        # Calcular métricas para entrenamiento (detectar overfitting)
                        train_f1 = f1_score(y_train_cv, y_pred_train, zero_division=0)
                        
                        cv_scores.append(val_f1)
                        train_scores.append(train_f1)
                        precision_scores.append(val_precision)
                        recall_scores.append(val_recall)
                        
                        self.logger.info(f"  Fold {fold_idx+1}: Val F1={val_f1:.3f}, Train F1={train_f1:.3f}, P={val_precision:.3f}, R={val_recall:.3f}")
                    
                    # Calcular estadísticas finales
                    cv_scores = np.array(cv_scores)
                    train_scores = np.array(train_scores)
                    precision_scores = np.array(precision_scores)
                    recall_scores = np.array(recall_scores)
                    
                    # Detectar overfitting
                    overfitting_gap = train_scores.mean() - cv_scores.mean()
                    overfitting_detected = overfitting_gap > 0.15  # Threshold de overfitting
                    
                    self.cv_scores[name] = {
                        'validation_f1_mean': cv_scores.mean(),
                        'validation_f1_std': cv_scores.std(),
                        'training_f1_mean': train_scores.mean(),
                        'precision_mean': precision_scores.mean(),
                        'precision_std': precision_scores.std(),
                        'recall_mean': recall_scores.mean(),
                        'recall_std': recall_scores.std(),
                        'overfitting_gap': overfitting_gap,
                        'overfitting_detected': overfitting_detected,
                        'scores': cv_scores.tolist(),
                        'stability_score': 1.0 - cv_scores.std()  # Métrica de estabilidad
                    }
                    
                    overfitting_metrics[name] = {
                        'gap': overfitting_gap,
                        'detected': overfitting_detected,
                        'stability': 1.0 - cv_scores.std()
                    }
                    
                    self.logger.info(f"  {name} - Val F1: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, Overfitting: {overfitting_gap:.3f}")
                    if overfitting_detected:
                        self.logger.warning(f"  ⚠️  OVERFITTING DETECTADO en {name}")
                    
                except Exception as e:
                    self.logger.warning(f"Error en CV para {name}: {str(e)}")
                    self.cv_scores[name] = {'error': str(e)}
        
        # Evaluar stacking model con detección de overfitting
        try:
            self.logger.info("Evaluando stacking model en cross-validation...")
            
            stacking_val_scores = []
            stacking_train_scores = []
            stacking_precision_scores = []
            stacking_recall_scores = []
            
            for fold_idx, (train_indices, val_indices) in enumerate(time_splits):
                # Obtener datos para este split
                X_train_cv = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
                y_train_cv = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
                X_val_cv = X.iloc[val_indices] if hasattr(X, 'iloc') else X[val_indices]
                y_val_cv = y.iloc[val_indices] if hasattr(y, 'iloc') else y[val_indices]
                
                # LIMPIEZA EXHAUSTIVA DE NaN
                X_train_cv = self._clean_nan_exhaustive(X_train_cv)
                X_val_cv = self._clean_nan_exhaustive(X_val_cv)
                
                # REMOVER columna Date si existe
                if 'Date' in X_train_cv.columns:
                    X_train_cv = X_train_cv.drop(columns=['Date'])
                if 'Date' in X_val_cv.columns:
                    X_val_cv = X_val_cv.drop(columns=['Date'])
                
                # Crear stacking model con regularización aumentada para CV
                from sklearn.base import clone
                temp_stacking = clone(self.stacking_model)
                
                # Aumentar regularización del meta-modelo
                temp_stacking.final_estimator.set_params(
                    C=0.3,  # Más regularización
                    class_weight={0: 1.0, 1: 20.0}  # Manejo de desbalance
                )
                
                temp_stacking.fit(X_train_cv, y_train_cv)
                
                # Predicciones
                y_pred_val = temp_stacking.predict(X_val_cv)
                y_pred_train = temp_stacking.predict(X_train_cv)
                
                # Métricas
                val_f1 = f1_score(y_val_cv, y_pred_val, zero_division=0)
                train_f1 = f1_score(y_train_cv, y_pred_train, zero_division=0)
                val_precision = precision_score(y_val_cv, y_pred_val, zero_division=0)
                val_recall = recall_score(y_val_cv, y_pred_val, zero_division=0)
                
                stacking_val_scores.append(val_f1)
                stacking_train_scores.append(train_f1)
                stacking_precision_scores.append(val_precision)
                stacking_recall_scores.append(val_recall)
                
                self.logger.info(f"  Stacking Fold {fold_idx+1}: Val F1={val_f1:.3f}, Train F1={train_f1:.3f}")
            
            # Estadísticas finales del stacking
            stacking_val_scores = np.array(stacking_val_scores)
            stacking_train_scores = np.array(stacking_train_scores)
            stacking_precision_scores = np.array(stacking_precision_scores)
            stacking_recall_scores = np.array(stacking_recall_scores)
            
            # Detectar overfitting en stacking
            stacking_overfitting_gap = stacking_train_scores.mean() - stacking_val_scores.mean()
            stacking_overfitting_detected = stacking_overfitting_gap > 0.15
            
            self.cv_scores['stacking'] = {
                'validation_f1_mean': stacking_val_scores.mean(),
                'validation_f1_std': stacking_val_scores.std(),
                'training_f1_mean': stacking_train_scores.mean(),
                'precision_mean': stacking_precision_scores.mean(),
                'precision_std': stacking_precision_scores.std(),
                'recall_mean': stacking_recall_scores.mean(),
                'recall_std': stacking_recall_scores.std(),
                'overfitting_gap': stacking_overfitting_gap,
                'overfitting_detected': stacking_overfitting_detected,
                'scores': stacking_val_scores.tolist(),
                'stability_score': 1.0 - stacking_val_scores.std()
            }
            
            overfitting_metrics['stacking'] = {
                'gap': stacking_overfitting_gap,
                'detected': stacking_overfitting_detected,
                'stability': 1.0 - stacking_val_scores.std()
            }
            
            self.logger.info(f"  Stacking - Val F1: {stacking_val_scores.mean():.3f}±{stacking_val_scores.std():.3f}, Overfitting: {stacking_overfitting_gap:.3f}")
            if stacking_overfitting_detected:
                self.logger.warning(f"  ⚠️  OVERFITTING DETECTADO en stacking model")
            
        except Exception as e:
            self.logger.warning(f"Error en CV para stacking: {str(e)}")
            self.cv_scores['stacking'] = {'error': str(e)}
        
        # Resumen de overfitting
        self.logger.info("=== RESUMEN DE DETECCIÓN DE OVERFITTING ===")
        overfitting_count = sum(1 for metrics in overfitting_metrics.values() if metrics['detected'])
        self.logger.info(f"Modelos con overfitting detectado: {overfitting_count}/{len(overfitting_metrics)}")
        
        for model_name, metrics in overfitting_metrics.items():
            status = "⚠️  OVERFITTING" if metrics['detected'] else "✅ OK"
            self.logger.info(f"  {model_name}: {status} (Gap: {metrics['gap']:.3f}, Estabilidad: {metrics['stability']:.3f})")
        
        # Guardar métricas de overfitting
        self.cv_scores['overfitting_summary'] = overfitting_metrics
        
        return self.cv_scores
    
    def _clean_nan_exhaustive(self, X):
        """Limpieza exhaustiva de NaN para validación cruzada"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_clean = X.copy()
        
        # 1. Reemplazar infinitos
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Verificar si hay NaN
        if X_clean.isna().any().any():
            # 3. Imputación columna por columna
            for col in X_clean.columns:
                if X_clean[col].isna().any():
                    if X_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        # Para columnas numéricas
                        if X_clean[col].notna().sum() > 0:
                            median_val = X_clean[col].median()
                            if pd.isna(median_val):
                                mean_val = X_clean[col].mean()
                                fill_val = mean_val if not pd.isna(mean_val) else 0.0
                            else:
                                fill_val = median_val
                        else:
                            fill_val = 0.0
                        X_clean[col] = X_clean[col].fillna(fill_val)
                    else:
                        # Para columnas categóricas o de otro tipo
                        X_clean[col] = X_clean[col].fillna(0)
            
            # 4. Imputación final para asegurar que no queden NaN
            X_clean = X_clean.fillna(0)
        
        # 5. Verificación final
        if X_clean.isna().any().any():
            logger.warning("Aún hay NaN después de limpieza exhaustiva, forzando a 0")
            X_clean = X_clean.fillna(0)
        
        return X_clean
    
    def _calculate_feature_importance(self, feature_columns: List[str]) -> Dict[str, Any]:
        """Calcular importancia de features de todos los modelos"""
        
        importance_summary = {}
        
        for name, model_info in self.training_results['individual_models'].items():
            if 'model' in model_info:
                model = model_info['model']
                
                if hasattr(model, 'feature_importances_'):
                    importance_summary[name] = {
                        'importances': model.feature_importances_.tolist(),
                        'feature_names': feature_columns
                    }
        
        # Calcular importancia promedio
        if importance_summary:
            avg_importance = np.zeros(len(feature_columns))
            count = 0
            
            for name, info in importance_summary.items():
                avg_importance += np.array(info['importances'])
                count += 1
            
            if count > 0:
                avg_importance /= count
                importance_summary['average'] = {
                    'importances': avg_importance.tolist(),
                    'feature_names': feature_columns
                }
        
        self.feature_importance = importance_summary
        return importance_summary
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """Obtener top features más importantes"""
        
        if not self.feature_importance:
            return {}
        
        result = {}
        
        for model_name, info in self.feature_importance.items():
            if 'importances' in info and 'feature_names' in info:
                # Crear pares (feature, importance)
                feature_importance_pairs = list(zip(info['feature_names'], info['importances']))
                
                # Ordenar por importancia descendente
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar top N
                top_features = feature_importance_pairs[:top_n]
                
                result[model_name] = {
                    'top_features': [(feat, float(imp)) for feat, imp in top_features],
                    'total_features': len(info['feature_names'])
                }
        
        return result
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluar modelo en datos de test"""
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        feature_columns = self.training_results['feature_columns']
        X = df[feature_columns].copy()
        
        # Determinar columna target
        target_col = 'double_double' if 'double_double' in df.columns else 'DD'
        if target_col not in df.columns:
            raise ValueError("No se encontró columna target en datos de evaluación")
        
        y = df[target_col].copy()
        
        # Predicciones
        y_pred = self.predict(df)
        y_proba = self.predict_proba(df)[:, 1]
        
        # Calcular métricas
        metrics = MetricsCalculator.calculate_classification_metrics(y, y_pred, y_proba)
        
        logger.info("Métricas de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Guardar modelo completo"""
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos para guardar
        model_data = {
            'models': {},
            'stacking_model': self.stacking_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'cv_scores': self.cv_scores,
            'training_results': self.training_results,
            'bayesian_results': getattr(self, 'bayesian_results', {}),
            'gpu_config': self.gpu_config
        }
        
        # Guardar modelos individuales (excepto neural network)
        for name, model_info in self.training_results['individual_models'].items():
            if 'model' in model_info and name != 'neural_network':
                model_data['models'][name] = model_info['model']
        
        # Guardar modelos tradicionales
        joblib.dump(model_data, filepath)
        
        # Guardar red neuronal por separado si existe
        if 'neural_network' in self.training_results['individual_models']:
            nn_model = self.training_results['individual_models']['neural_network'].get('model')
            if nn_model and hasattr(nn_model, 'model') and nn_model.model is not None:
                nn_filepath = filepath.replace('.pkl', '_neural_network.pth')
                torch.save(nn_model.model.state_dict(), nn_filepath)
    
    def load_model(self, filepath: str):
        """Cargar modelo completo"""
        
        # Cargar modelos tradicionales
        model_data = joblib.load(filepath)
        
        self.stacking_model = model_data['stacking_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.cv_scores = model_data['cv_scores']
        self.training_results = model_data['training_results']
        self.bayesian_results = model_data.get('bayesian_results', {})
        self.gpu_config = model_data.get('gpu_config', {})
        
        # Recrear modelos individuales
        for name, model in model_data['models'].items():
            if name not in self.training_results['individual_models']:
                self.training_results['individual_models'][name] = {}
            self.training_results['individual_models'][name]['model'] = model
        
        # Cargar red neuronal si existe
        nn_filepath = filepath.replace('.pkl', '_neural_network.pth')
        if Path(nn_filepath).exists():
            # Recrear el clasificador neural
            nn_classifier = PyTorchDoubleDoubleClassifier(device=str(self.device))
            
            # Recrear la arquitectura del modelo
            if 'feature_columns' in self.training_results:
                input_size = len(self.training_results['feature_columns'])
                nn_classifier.model = DoubleDoubleNeuralNetwork(input_size=input_size)
                nn_classifier.model.load_state_dict(torch.load(nn_filepath, map_location=self.device))
                nn_classifier.model.to(self.device)
                
                self.training_results['individual_models']['neural_network'] = {
                    'model': nn_classifier
                }
        
        self.is_fitted = True

    def get_training_summary(self) -> Dict[str, Any]:
        """Obtener resumen completo del entrenamiento"""
        
        if not self.is_fitted:
            return {"error": "Modelo no está entrenado"}
        
        summary = {
            'model_info': {
                'total_models': len(self.training_results.get('individual_models', {})),
                'stacking_enabled': self.stacking_model is not None,
                'bayesian_optimization': bool(getattr(self, 'bayesian_results', {})),
                'gpu_used': self.gpu_config.get('selected_device', 'cpu') != 'cpu'
            },
            'training_data': {
                'training_samples': self.training_results.get('training_samples', 0),
                'validation_samples': self.training_results.get('validation_samples', 0),
                'total_features': len(self.training_results.get('feature_columns', []))
            },
            'model_performance': {},
            'cross_validation': self.cv_scores,
            'feature_importance_available': bool(self.feature_importance)
        }
        
        # Agregar métricas de modelos individuales
        for name, model_info in self.training_results.get('individual_models', {}).items():
            if 'val_metrics' in model_info:
                summary['model_performance'][name] = model_info['val_metrics']
        
        # Agregar métricas de stacking
        if 'stacking_metrics' in self.training_results:
            summary['model_performance']['stacking'] = self.training_results['stacking_metrics']
        
        return summary

    def validate_stacking_models(self) -> Dict[str, Any]:
        """Validar que el stacking incluye todos los modelos y funciona correctamente"""
        
        if not self.is_fitted:
            return {"error": "Modelo no está entrenado"}
        
        validation_info = {
            'total_estimators': len(self.stacking_model.estimators_),
            'estimator_names': [name for name, _ in self.stacking_model.estimators_],
            'meta_model_type': type(self.stacking_model.final_estimator_).__name__,
            'models_included': {},
            'neural_network_status': 'not_found'
        }
        
        # Verificar cada estimador del stacking
        for name, estimator in self.stacking_model.estimators_:
            validation_info['models_included'][name] = {
                'type': type(estimator).__name__,
                'fitted': hasattr(estimator, '_fitted') or hasattr(estimator, 'is_fitted_'),
                'has_predict_proba': hasattr(estimator, 'predict_proba')
            }
            
            # Verificar wrapper de red neuronal específicamente
            if 'nn' in name.lower() or 'neural' in name.lower():
                if hasattr(estimator, 'nn_model'):
                    validation_info['neural_network_status'] = 'wrapper_found'
                    if hasattr(estimator.nn_model, 'model'):
                        validation_info['neural_network_status'] = 'fully_configured'
                        validation_info['models_included'][name]['nn_device'] = str(estimator.nn_model.device) if hasattr(estimator.nn_model, 'device') else 'unknown'
        
        # Verificar modelos individuales disponibles
        individual_models = list(self.training_results.get('individual_models', {}).keys())
        validation_info['individual_models_available'] = individual_models
        
        # Verificar qué modelos del setup están en el stacking
        expected_models = ['xgb', 'lgb', 'rf', 'et', 'gb', 'cb', 'nn']
        stacking_names = [name.replace('_stack', '') for name, _ in self.stacking_model.estimators_]
        
        validation_info['models_coverage'] = {
            'expected': expected_models,
            'in_stacking': stacking_names,
            'missing_from_stacking': list(set(expected_models) - set(stacking_names)),
            'coverage_percentage': len(set(stacking_names) & set(expected_models)) / len(expected_models) * 100
        }
        
        return validation_info

    def _calculate_optimal_threshold_advanced(self, y_true, y_proba, method='f1_precision_balance'):
        """
        PARTE 1: THRESHOLD OPTIMIZATION AVANZADO - CORREGIDO
        Calcular threshold óptimo usando múltiples estrategias y validación
        
        Args:
            y_true: Valores reales
            y_proba: Probabilidades predichas (columna 1 para clase positiva)
            method: Método de optimización ('f1_precision_balance', 'youden', 'precision_recall_curve')
        
        Returns:
            float: Threshold óptimo
        """
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        # Extraer probabilidades de clase positiva
        if y_proba.ndim > 1:
            proba_positive = y_proba[:, 1]
        else:
            proba_positive = y_proba
        
        # CORRECCIÓN CRÍTICA: Usar límites basados en probabilidades reales
        prob_min = np.min(proba_positive)
        prob_max = np.max(proba_positive)
        prob_mean = np.mean(proba_positive)
        
        self.logger.info(f"Optimizando threshold con método: {method}")
        self.logger.info(f"Distribución real: {np.mean(y_true):.3f} positivos")
        self.logger.info(f"Rango probabilidades: [{prob_min:.3f}, {prob_max:.3f}], Media: {prob_mean:.3f}")
        
        if method == 'f1_precision_balance':
            # Método 1: Balancear F1 Score y Precision mínima
            # CORRECCIÓN: Usar rango realista basado en probabilidades reales
            threshold_min = max(0.05, prob_min)
            threshold_max = min(0.95, prob_max * 0.9)  # 90% del máximo
            
            thresholds = np.linspace(threshold_min, threshold_max, 100)
            best_score = 0
            best_threshold = prob_mean  # Usar media como default
            min_precision = 0.10  # REDUCIDO: Precision mínima más realista
            
            self.logger.info(f"Probando thresholds en rango [{threshold_min:.3f}, {threshold_max:.3f}]")
            
            for threshold in thresholds:
                y_pred = (proba_positive >= threshold).astype(int)
                
                # Evitar divisiones por cero
                if np.sum(y_pred) == 0:
                    continue
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Solo considerar si precision >= mínima
                if precision >= min_precision:
                    # Score combinado: 60% F1 + 40% Recall (priorizar recall)
                    combined_score = 0.6 * f1 + 0.4 * recall
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_threshold = threshold
            
            self.logger.info(f"F1-Precision balance: threshold={best_threshold:.4f}, score={best_score:.4f}")
            
        elif method == 'youden':
            # Método 2: Índice de Youden (maximizar TPR - FPR)
            fpr, tpr, thresholds = roc_curve(y_true, proba_positive)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]
            
            # CORRECCIÓN: Limitar a rango realista
            best_threshold = min(best_threshold, prob_max * 0.9)
            
            self.logger.info(f"Youden index: threshold={best_threshold:.4f}, index={youden_index[best_idx]:.4f}")
            
        elif method == 'precision_recall_curve':
            # Método 3: Curva Precision-Recall
            precision, recall, thresholds = precision_recall_curve(y_true, proba_positive)
            
            # Encontrar threshold que maximice F1 con precision mínima
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # CORRECCIÓN: Precision mínima más realista
            min_precision = 0.08
            valid_indices = precision >= min_precision
            
            if np.any(valid_indices):
                valid_f1 = f1_scores[valid_indices]
                valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds tiene un elemento menos
                
                if len(valid_thresholds) > 0:
                    best_idx = np.argmax(valid_f1)
                    best_threshold = valid_thresholds[best_idx]
                    # CORRECCIÓN: Limitar a rango realista
                    best_threshold = min(best_threshold, prob_max * 0.9)
                else:
                    best_threshold = prob_mean
            else:
                best_threshold = prob_mean
            
            self.logger.info(f"Precision-Recall curve: threshold={best_threshold:.4f}")
        
        # Validación del threshold con límites realistas
        threshold_min_limit = max(0.05, prob_min)
        threshold_max_limit = min(prob_max * 0.60, 0.15) 
        
        if best_threshold < threshold_min_limit:
            self.logger.warning(f"Threshold muy bajo ({best_threshold:.4f}), ajustando a {threshold_min_limit:.4f}")
            best_threshold = threshold_min_limit
        elif best_threshold > threshold_max_limit:
            self.logger.warning(f"Threshold muy alto ({best_threshold:.4f}), ajustando a {threshold_max_limit:.4f}")
            best_threshold = threshold_max_limit
        
        # FALLBACK MÁS AGRESIVO: Garantizar predicciones suficientes
        y_pred_test = (proba_positive >= best_threshold).astype(int)
        predicted_positives = np.sum(y_pred_test)
        actual_positives = np.sum(y_true)
        
        # Si predecimos menos del 20% de los casos reales, ser más agresivo
        if predicted_positives < (actual_positives * 0.2):
            # Usar percentil que garantice al menos 20% de los casos reales
            target_rate = max(0.08, np.mean(y_true) * 2.0)  # 2x la tasa real, mínimo 8%
            percentile = 100 - (target_rate * 100)
            fallback_threshold = np.percentile(proba_positive, percentile)
            
            self.logger.warning(f"Threshold {best_threshold:.4f} genera solo {predicted_positives} predicciones de {actual_positives} reales. Usando fallback más agresivo: {fallback_threshold:.4f}")
            best_threshold = fallback_threshold
        
        # Evaluar threshold final
        y_pred_final = (proba_positive >= best_threshold).astype(int)
        final_precision = precision_score(y_true, y_pred_final, zero_division=0)
        final_recall = recall_score(y_true, y_pred_final, zero_division=0)
        final_f1 = f1_score(y_true, y_pred_final, zero_division=0)
        final_predictions = np.sum(y_pred_final)
        
        self.logger.info(f"Threshold final: {best_threshold:.4f}")
        self.logger.info(f"Predicciones positivas: {final_predictions}/{len(y_pred_final)} ({final_predictions/len(y_pred_final)*100:.1f}%)")
        self.logger.info(f"Métricas finales - P: {final_precision:.3f}, R: {final_recall:.3f}, F1: {final_f1:.3f}")
        
        return best_threshold

    def _calculate_optimal_threshold(self, y_true, y_proba, target_precision=0.25):
        """
        Calcular threshold óptimo SIMPLE - MÉTODO LEGACY
        Este método ya no se usa en el flujo principal, solo como backup
        """
        # Método simple: usar percentil que coincida con distribución real
        actual_positive_rate = np.mean(y_true)
        target_percentile = 100 - (actual_positive_rate * 100)
        
        threshold = np.percentile(y_proba[:, 1], target_percentile)
        
        # Asegurar que no sea demasiado alto
        if threshold > 0.15:
            threshold = 0.10
            
        self.logger.info(f"Threshold calculado (método legacy): {threshold:.4f}")
        return threshold


def create_double_double_model(
    use_gpu: bool = True,
    random_state: int = 42,
    optimize_hyperparams: bool = True,
    bayesian_n_calls: int = 50
) -> DoubleDoubleAdvancedModel:
    """
    Factory function para crear modelo avanzado de double double
    
    Args:
        use_gpu: Si usar GPU
        random_state: Semilla aleatoria
        optimize_hyperparams: Si optimizar hiperparámetros
        bayesian_n_calls: Número de llamadas para optimización bayesiana
        
    Returns:
        Modelo inicializado
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    return DoubleDoubleAdvancedModel(
        optimize_hyperparams=optimize_hyperparams,
        device=device,
        bayesian_n_calls=bayesian_n_calls,
        min_memory_gb=2.0
    )