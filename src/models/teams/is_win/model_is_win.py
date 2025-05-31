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
import json

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

# Logging setup optimizado
import logging
import sys
from pathlib import Path

class OptimizedLogger:
    """Sistema de logging optimizado para modelos NBA"""
    
    _loggers = {}
    _handlers_configured = False
    
    @classmethod
    def get_logger(cls, name: str = __name__, level: str = "INFO") -> logging.Logger:
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
    def _setup_handlers(cls, logger: logging.Logger):
        """Configurar handlers de logging optimizados"""
        
        # Formatter más simple
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para consola con filtro de nivel
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Handler para archivo solo para errores importantes
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"nba_model_{datetime.now().strftime('%Y%m%d')}.log",
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.WARNING)  # Solo warnings y errores
        file_handler.setFormatter(formatter)
        
        # Agregar handlers al logger raíz
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
    
    @classmethod
    def log_performance_metrics(cls, logger: logging.Logger, 
                               metrics: Dict[str, float], 
                               model_name: str = "Model",
                               phase: str = "Training"):
        """Log conciso para métricas de rendimiento"""
        
        # Solo mostrar métricas principales
        acc = metrics.get('accuracy', 0)
        auc = metrics.get('auc_roc', 0)
        
        logger.info(f"{model_name}: ACC={acc:.3f}, AUC={auc:.3f}")
    
    @classmethod
    def log_training_progress(cls, logger: logging.Logger,
                             epoch: int, total_epochs: int,
                             train_loss: float, val_loss: float,
                             val_accuracy: float,
                             model_name: str = "Neural Network"):
        """Log reducido para progreso de entrenamiento"""
        
        # Solo cada 50 epochs para reducir verbosidad
        if epoch % 50 == 0 or epoch == total_epochs - 1:
            logger.info(f"Epoch {epoch}/{total_epochs}: Loss={val_loss:.3f}, Acc={val_accuracy:.3f}")
    
    @classmethod
    def log_gpu_info(cls, logger: logging.Logger, 
                     device_info: Dict[str, Any],
                     phase: str = "Setup"):
        """Log simplificado para información de GPU"""
        
        device = device_info.get('device', 'Unknown')
        
        if device_info.get('type') == 'cuda':
            memory_info = device_info.get('memory_info', {})
            free_gb = memory_info.get('free_gb', 0)
            logger.info(f"GPU: {device} ({free_gb:.1f}GB libre)")
        else:
            logger.info(f"CPU: {device}")
    
    @classmethod
    def log_model_summary(cls, logger: logging.Logger,
                         model_results: Dict[str, Any]):
        """Log resumido para resumen de modelos"""
        
        logger.info("Resumen de modelos:")
        
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'val_metrics' in results:
                metrics = results['val_metrics']
                acc = metrics.get('accuracy', 0)
                auc = metrics.get('auc_roc', 0)
                
                logger.info(f"  {model_name}: ACC={acc:.3f}, AUC={auc:.3f}")

# Configurar logger global optimizado
logger = OptimizedLogger.get_logger(__name__)


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
        
        # Log usando sistema optimizado
        device_info = GPUManager.get_device_info(str(self.device))
        OptimizedLogger.log_gpu_info(
            OptimizedLogger.get_logger(f"{__name__}.PyTorchNBAClassifier"),
            device_info,
            "Configuración"
        )
    
    def _auto_adjust_batch_size(self, X_train_tensor: torch.Tensor, 
                               y_train_tensor: torch.Tensor) -> int:
        """Detectar automáticamente el batch_size óptimo para la GPU"""
        
        nn_logger = OptimizedLogger.get_logger(f"{__name__}.PyTorchNBAClassifier")
        
        if self.device.type == 'cpu':
            optimal_size = min(self.batch_size, 64)
            nn_logger.debug(f"💻 CPU detectada | Batch size: {optimal_size}")
            return optimal_size
        
        nn_logger.debug("🔍 Detectando batch_size óptimo para GPU...")
        
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
                    # OOM detectado, usar el batch size anterior
                    nn_logger.debug(f"🚫 OOM en batch_size {test_batch_size}")
                    break
                else:
                    # Otro tipo de error
                    nn_logger.warning(f"⚠️  Error probando batch_size {test_batch_size}: {e}")
                    break
        
        # Limpiar modelo temporal
        del temp_model
        torch.cuda.empty_cache()
        
        nn_logger.info(f"🎯 Batch size óptimo detectado: {optimal_batch_size}")
        return optimal_batch_size
    
    def fit(self, X, y):
        """Entrenamiento del modelo con early stopping, validación y manejo avanzado de GPU"""
        
        nn_logger = OptimizedLogger.get_logger(f"{__name__}.PyTorchNBAClassifier")
        
        # Monitorear memoria antes del entrenamiento
        self.gpu_memory_stats['pre_training'] = GPUManager.monitor_memory_usage(
            self.device, "pre_training"
        )
        
        try:
            nn_logger.info("🚀 Iniciando entrenamiento de red neuronal...")
            
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
            
            nn_logger.info(f"📊 Datos preparados | Train: {len(X_train)} | Val: {len(X_val)}")
            
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
            
            nn_logger.info(f"🧠 Red neuronal inicializada | Input: {X_train.shape[1]} | "
                          f"Hidden: {self.hidden_size} | Dropout: {self.dropout_rate}")
            
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
            
            nn_logger.info(f"🔄 Iniciando entrenamiento | Epochs: {self.epochs} | "
                          f"Batch size: {optimal_batch_size} | LR: {self.learning_rate}")
            
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
                
                # Log progreso usando sistema optimizado cada 25 epochs
                if epoch % 25 == 0 or epoch == self.epochs - 1:
                    OptimizedLogger.log_training_progress(
                        nn_logger, epoch, self.epochs,
                        avg_train_loss, val_loss, val_accuracy,
                        "Red Neuronal"
                    )
                
                # Verificar early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    nn_logger.info(f"⏹️  Early stopping activado en epoch {epoch}")
                    break
            
            # Cargar mejor modelo
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                nn_logger.info("✅ Mejor modelo cargado")
            
            # Monitorear memoria final
            self.gpu_memory_stats['post_training'] = GPUManager.monitor_memory_usage(
                self.device, "post_training"
            )
            
            # Log resumen final
            final_metrics = {
                'final_val_loss': self.best_val_loss,
                'final_val_accuracy': max(self.training_history['val_accuracy']),
                'epochs_trained': len(self.training_history['train_loss'])
            }
            
            OptimizedLogger.log_performance_metrics(
                nn_logger, final_metrics, "Red Neuronal", "Final"
            )
            
            return self
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Manejar error de memoria insuficiente
                oom_info = GPUManager.handle_oom_error(self.device)
                nn_logger.error(f"❌ Error de memoria insuficiente: {e}")
                raise RuntimeError(f"GPU sin memoria suficiente. {oom_info['suggested_actions']}")
            else:
                nn_logger.error(f"❌ Error durante entrenamiento: {e}")
                raise e
    
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
        """Seleccionar el dispositivo óptimo basado en memoria disponible"""
        
        devices = GPUManager.get_available_devices()
        best_device = "cpu"
        best_memory = 0
        
        for device_str in devices:
            if device_str == "cpu":
                continue
                
            info = GPUManager.get_device_info(device_str)
            if (info['available'] and 
                GPUManager.check_memory_availability(device_str, min_memory_gb)):
                
                memory_gb = info.get('memory', {}).get('free_gb', 0)
                if memory_gb > best_memory:
                    best_memory = memory_gb
                    best_device = device_str
        
        if best_device == "cpu":
            logger.info("CUDA no disponible, usando CPU")
        
        return best_device
    
    @staticmethod
    def setup_device(device_preference: str = None, 
                   min_memory_gb: float = 2.0) -> torch.device:
        """Configurar dispositivo con validación de memoria"""
        
        if device_preference:
            # Verificar dispositivo solicitado
            if GPUManager.check_memory_availability(device_preference, min_memory_gb):
                return torch.device(device_preference)
            else:
                logger.warning(f"Dispositivo {device_preference} no disponible o sin memoria suficiente. "
                             f"Seleccionando automáticamente...")
        
        # Seleccionar dispositivo óptimo
        optimal_device = GPUManager.get_optimal_device(min_memory_gb)
        device = torch.device(optimal_device)
        
        # Optimizar memoria si es GPU
        if device.type == 'cuda':
            GPUManager.optimize_memory_usage(device)
        
        return device
    
    @staticmethod
    def optimize_memory_usage(device: torch.device):
        """Optimizar uso de memoria GPU"""
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Configuraciones adicionales de optimización
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
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
        """Manejar errores de memoria insuficiente"""
        
        recovery_info = {
            'memory_cleared': False,
            'suggested_actions': [],
            'current_memory': {}
        }
        
        if device.type == 'cuda':
            try:
                # Limpiar memoria GPU
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                recovery_info['memory_cleared'] = True
                recovery_info['current_memory'] = GPUManager.monitor_memory_usage(
                    device, "post_oom_cleanup"
                )
                
                # Sugerencias de recuperación
                recovery_info['suggested_actions'] = [
                    "Reducir batch_size",
                    "Reducir hidden_size del modelo",
                    "Usar gradient checkpointing",
                    "Cambiar a CPU si persiste el problema"
                ]
                
            except Exception as e:
                logger.error(f"Error en recuperación OOM: {e}")
                recovery_info['suggested_actions'] = ["Reiniciar proceso y usar CPU"]
        
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
    
    gpu_logger = OptimizedLogger.get_logger(f"{__name__}.GPUConfig")
    
    if print_summary:
        gpu_logger.info("🖥️  Configurando entorno GPU para modelos NBA...")
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
    
    # Log información de configuración usando sistema optimizado
    OptimizedLogger.log_gpu_info(gpu_logger, device_info, "Final")
    
    gpu_logger.info(f"✅ Entorno GPU configurado | PyTorch: {torch.__version__} | "
                   f"CUDA: {config_info['cuda_available']} | GPUs: {config_info['gpu_count']}")
    
    return config_info


class IsWinModel:
    """Modelo principal para predicción de victorias NBA con stacking y optimización bayesiana"""
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 bayesian_n_calls: int = 50,
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
            print_summary=False
        )
        
        # Usar dispositivo configurado
        self.device = self.gpu_config['selected_device']
        
        # Log simplificado de GPU
        OptimizedLogger.log_gpu_info(
            logger, 
            self.gpu_config.get('device_info', {}),
            "Configuración"
        )
    
    def _setup_models(self):
        """Configurar modelos individuales con REGULARIZACIÓN EXTREMA ANTI-OVERFITTING"""
        
        # XGBoost con regularización ULTRA AGRESIVA
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=50,        # Reducido drásticamente de 100
            max_depth=3,            # Reducido de 4 para árboles más simples
            learning_rate=0.02,     # Reducido de 0.05 para aprendizaje más lento
            subsample=0.6,          # Reducido de 0.8 para más regularización
            colsample_bytree=0.6,   # Reducido de 0.8
            reg_alpha=3.0,          # Aumentado de 1.0 (regularización L1)
            reg_lambda=5.0,         # Aumentado de 2.0 (regularización L2)
            min_child_weight=10,    # Aumentado de 5
            gamma=2.0,              # Aumentado de 1.0
            max_delta_step=1,       # Nuevo: limitar cambios extremos
            random_state=42, 
            n_jobs=-1, 
            eval_metric='logloss'
        )
        
        # LightGBM con regularización ULTRA AGRESIVA
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=50,        # Reducido drásticamente de 100
            max_depth=3,            # Reducido de 4
            learning_rate=0.02,     # Reducido de 0.05
            subsample=0.6,          # Reducido de 0.8
            colsample_bytree=0.6,   # Reducido de 0.8
            reg_alpha=3.0,          # Aumentado de 1.0
            reg_lambda=5.0,         # Aumentado de 2.0
            min_child_samples=50,   # Aumentado drásticamente de 20
            min_split_gain=0.5,     # Aumentado de 0.1
            num_leaves=15,          # Nuevo: limitar complejidad del árbol
            feature_fraction=0.6,   # Nuevo: usar solo 60% de features
            bagging_fraction=0.6,   # Nuevo: usar solo 60% de datos
            bagging_freq=5,         # Nuevo: bagging cada 5 iteraciones
            random_state=42, 
            n_jobs=-1, 
            verbosity=-1
        )
        
        # Random Forest con regularización EXTREMA
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=50,        # Reducido drásticamente de 100
            max_depth=4,            # Reducido de 6
            min_samples_split=50,   # Aumentado drásticamente de 20
            min_samples_leaf=25,    # Aumentado drásticamente de 10
            max_features=0.4,       # Reducido de 'sqrt' para usar menos features
            bootstrap=True,         # Asegurar que bootstrap esté habilitado
            oob_score=True, 
            random_state=42, 
            n_jobs=-1
        )
        
        # Extra Trees con regularización EXTREMA
        self.models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=50,        # Reducido drásticamente de 100
            max_depth=4,            # Reducido de 6
            min_samples_split=60,   # Aumentado drásticamente de 25
            min_samples_leaf=30,    # Aumentado drásticamente de 15
            max_features=0.3,       # Reducido drásticamente para usar menos features
            bootstrap=True,         # Asegurar que bootstrap esté habilitado
            oob_score=True, 
            random_state=42, 
            n_jobs=-1
        )
        
        # Gradient Boosting con regularización EXTREMA
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=50,        # Reducido drásticamente de 100
            max_depth=3,            # Reducido de 4
            learning_rate=0.02,     # Reducido drásticamente de 0.05
            subsample=0.5,          # Reducido drásticamente de 0.8
            min_samples_split=50,   # Aumentado drásticamente de 20
            min_samples_leaf=25,    # Aumentado drásticamente de 10
            max_features=0.3,       # Reducido drásticamente de 'sqrt'
            validation_fraction=0.2, # Nuevo: usar 20% para validación interna
            n_iter_no_change=5,     # Nuevo: early stopping agresivo
            tol=1e-3,               # Nuevo: tolerancia para early stopping
            random_state=42
        )
        
        # Red Neuronal con DROPOUT EXTREMO y arquitectura mínima
        self.models['neural_network'] = PyTorchNBAClassifier(
            hidden_size=32,         # Reducido drásticamente de 64
            epochs=50,              # Reducido drásticamente de 100
            batch_size=128,         # Aumentado para más estabilidad
            learning_rate=0.0005,   # Reducido de 0.001
            weight_decay=0.05,      # Aumentado drásticamente de 0.01
            early_stopping_patience=8,  # Reducido de 15
            dropout_rate=0.7,       # Aumentado drásticamente de 0.5
            device=self.device, 
            min_memory_gb=self.min_memory_gb, 
            auto_batch_size=True
        )
        
        # Configurar stacking
        self._setup_stacking_model()
    
    def _setup_stacking_model(self):
        """Configurar modelo de stacking con REGULARIZACIÓN EXTREMA"""
        
        logger.debug("🔗 Configurando modelo de stacking con regularización extrema...")
        
        # Crear versiones ULTRA REGULARIZADAS para stacking
        xgb_stacking = xgb.XGBClassifier(
            n_estimators=30,        # Reducido drásticamente
            max_depth=3,            # Muy limitado
            learning_rate=0.05,     # Lento
            subsample=0.6,          # Muy reducido
            colsample_bytree=0.6,   # Muy reducido
            reg_alpha=2.0,          # Alta regularización L1
            reg_lambda=3.0,         # Alta regularización L2
            min_child_weight=10,    # Alto
            gamma=1.0,              # Alto
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        lgb_stacking = lgb.LGBMClassifier(
            n_estimators=30,        # Reducido drásticamente
            max_depth=3,            # Muy limitado
            learning_rate=0.05,     # Lento
            subsample=0.6,          # Muy reducido
            colsample_bytree=0.6,   # Muy reducido
            reg_alpha=2.0,          # Alta regularización L1
            reg_lambda=3.0,         # Alta regularización L2
            min_child_samples=40,   # Muy alto
            min_split_gain=0.3,     # Alto
            num_leaves=10,          # Muy limitado
            feature_fraction=0.5,   # Muy reducido
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        rf_stacking = RandomForestClassifier(
            n_estimators=30,        # Reducido drásticamente
            max_depth=4,            # Muy limitado
            min_samples_split=40,   # Muy alto
            min_samples_leaf=20,    # Muy alto
            max_features=0.3,       # Muy reducido
            bootstrap=True,         # Asegurar que bootstrap esté habilitado
            random_state=42,
            n_jobs=-1
        )
        
        et_stacking = ExtraTreesClassifier(
            n_estimators=30,        # Reducido drásticamente
            max_depth=4,            # Muy limitado
            min_samples_split=50,   # Muy alto
            min_samples_leaf=25,    # Muy alto
            max_features=0.2,       # Extremadamente reducido
            bootstrap=True,         # Asegurar que bootstrap esté habilitado
            random_state=42,
            n_jobs=-1
        )
        
        gb_stacking = GradientBoostingClassifier(
            n_estimators=30,        # Reducido drásticamente
            max_depth=3,            # Muy limitado
            learning_rate=0.05,     # Lento
            subsample=0.5,          # Muy reducido
            min_samples_split=40,   # Muy alto
            min_samples_leaf=20,    # Muy alto
            max_features=0.3,       # Muy reducido
            validation_fraction=0.2,
            n_iter_no_change=3,     # Early stopping muy agresivo
            tol=1e-3,
            random_state=42
        )
        
        nn_stacking = PyTorchNBAClassifier(
            hidden_size=16,         # Extremadamente pequeño
            epochs=30,              # Muy reducido
            early_stopping_patience=5,  # Muy agresivo
            dropout_rate=0.8,       # Extremadamente alto
            weight_decay=0.1,       # Muy alto
            learning_rate=0.0001,   # Muy lento
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
        
        # Meta-learner: Regresión Logística con REGULARIZACIÓN EXTREMA
        meta_learner = LogisticRegression(
            C=0.1,                  # Reducido drásticamente de 1.0 (más regularización)
            penalty='l2', 
            solver='liblinear', 
            random_state=42, 
            max_iter=1000
        )
        
        # Modelo de stacking con validación cruzada MÁS AGRESIVA
        self.stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # Reducido de 5 a 3
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        logger.debug("✅ Modelo de stacking configurado con regularización extrema")
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Obtener columnas de features generadas CON CACHE"""
        
        # Si ya tenemos features generadas y el DataFrame es similar, reutilizar
        if (hasattr(self.feature_engineer, '_last_data_hash') and 
            self.feature_engineer._last_data_hash is not None and 
            self.feature_engineer.feature_columns):
            
            # Verificar si es el mismo DataFrame
            current_hash = self.feature_engineer._get_data_hash(df)
            if current_hash == self.feature_engineer._last_data_hash:
                logger.debug("Reutilizando features desde cache")
                return self.feature_engineer.feature_columns
        
        # Solo generar features si es necesario
        feature_columns = self.feature_engineer.generate_all_features(df)
        
        # Filtrar features que realmente existen y no son problemáticas
        available_features = []
        for feature in feature_columns:
            if feature in df.columns:
                # Verificar que no tenga demasiados valores nulos
                null_pct = df[feature].isnull().sum() / len(df)
                if null_pct < 0.5:  # Menos del 50% de nulos
                    available_features.append(feature)
        
        # VERIFICACIÓN ADICIONAL: Asegurar que features críticas estén presentes
        critical_features = ['home_win_rate_10g', 'away_win_rate_10g']
        for critical_feature in critical_features:
            if critical_feature not in available_features and critical_feature in df.columns:
                available_features.append(critical_feature)
                logger.debug(f"Agregada feature crítica: {critical_feature}")
            elif critical_feature not in df.columns:
                logger.warning(f"Feature crítica faltante: {critical_feature}")
        
        # Solo log si hay diferencias significativas
        if len(available_features) != len(feature_columns):
            logger.debug(f"Features filtradas: {len(available_features)}/{len(feature_columns)} disponibles")
        
        return available_features
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrenamiento completo del modelo con validación y optimización"""
        
        logger.info("Iniciando entrenamiento...")
        
        # Generar features
        feature_columns = self.get_feature_columns(df)
        
        if not feature_columns:
            raise ValueError("No hay features disponibles para el entrenamiento")
        
        # Preparar datos
        X = df[feature_columns].fillna(0)
        y = df['is_win']
        
        # Verificar balance de clases
        class_balance = y.value_counts(normalize=True)
        logger.info(f"Balance: Victorias={class_balance.get(1, 0):.3f}, Derrotas={class_balance.get(0, 0):.3f}")
        
        # Preparar datos de entrenamiento
        X_train_scaled, X_val_scaled, y_train, y_val, self.scaler = (
            DataProcessor.prepare_training_data(X, y, validation_split)
        )
        
        # Entrenar modelos individuales
        individual_results = self._train_individual_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Log resumen de modelos individuales
        OptimizedLogger.log_model_summary(logger, individual_results)
        
        # Optimización bayesiana (si está habilitada)
        if self.optimize_hyperparams and BAYESIAN_AVAILABLE:
            logger.info("Optimización bayesiana...")
            self._optimize_with_bayesian(X_train_scaled, y_train)
        
        # Entrenar modelo de stacking
        logger.info("Entrenando stacking...")
        self.stacking_model.fit(X_train_scaled, y_train)
        
        # Evaluación completa
        stacking_val_pred = self.stacking_model.predict(X_val_scaled)
        stacking_val_proba = self.stacking_model.predict_proba(X_val_scaled)[:, 1]
        
        # Métricas del stacking
        stacking_metrics = MetricsCalculator.calculate_classification_metrics(
            y_val, stacking_val_pred, stacking_val_proba
        )
        
        # Log métricas de stacking
        OptimizedLogger.log_performance_metrics(
            logger, stacking_metrics, "Stacking", "Validación"
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
        
        logger.info(f"Entrenamiento completado. Accuracy: {stacking_metrics['accuracy']:.3f}")
        
        return self.training_results
    
    def _train_individual_models(self, X_train, y_train, 
                               X_val, y_val) -> Dict:
        """Entrenar modelos individuales con early stopping optimizado"""
        
        results = {}
        total_models = len(self.models)
        
        logger.info(f"🎯 Entrenando {total_models} modelos individuales...")
        
        for idx, (name, model) in enumerate(self.models.items(), 1):
            logger.info(f"🔄 [{idx}/{total_models}] Entrenando: {name}")
            
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
                
                # Calcular overfitting
                overfitting = train_metrics['accuracy'] - val_metrics['accuracy']
                
                results[name] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'overfitting': overfitting,
                    'early_stopping_info': early_stopping_info
                }
                
                # Log métricas usando sistema optimizado (solo métricas principales)
                acc = val_metrics['accuracy']
                auc = val_metrics['auc_roc']
                logger.info(f"✅ {name}: ACC={acc:.3f}, AUC={auc:.3f}")
                
                # Log información de early stopping si aplica (solo si es relevante)
                if early_stopping_info.get('stopped_early'):
                    best_iter = early_stopping_info.get('best_iteration', 'N/A')
                    logger.debug(f"⏹️  {name} | Early stopping en iteración {best_iter}")
                
                # Advertencia de overfitting (solo si es significativo)
                if overfitting > 0.1:
                    logger.warning(f"🚨 {name} | Overfitting detectado: {overfitting:+.4f}")
                elif overfitting > 0.05:
                    logger.warning(f"⚠️  {name} | Overfitting moderado: {overfitting:+.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {name}: {e}")
                results[name] = {'error': str(e)}
        
        logger.info(f"✅ Entrenamiento individual completado ({len(results)} modelos)")
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Realizar predicciones con el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Validar consistencia de features
        self._validate_feature_consistency(df, "predicción")
        
        # Obtener features
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].fillna(0)
        
        # Verificar que tenemos las mismas features que en entrenamiento
        if hasattr(self, 'feature_columns_') and self.feature_columns_:
            # Asegurar que tenemos exactamente las mismas features
            missing_features = set(self.feature_columns_) - set(feature_columns)
            if missing_features:
                logger.warning(f"Agregando features faltantes con valores por defecto: {list(missing_features)}")
                for feature in missing_features:
                    X[feature] = 0.5  # Valor neutro
            
            # Reordenar columnas para que coincidan con el entrenamiento
            X = X.reindex(columns=self.feature_columns_, fill_value=0.5)
        
        # Escalar features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Predicción con modelo stacking
        if self.stacking_model is not None:
            predictions = self.stacking_model.predict(X_scaled)
        else:
            # Fallback: usar modelo con mejor rendimiento
            best_model_name = max(self.cv_results.items(), key=lambda x: x[1]['accuracy_mean'])[0]
            best_model = self.models[best_model_name]
            predictions = best_model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Obtener probabilidades de predicción con el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Validar consistencia de features
        self._validate_feature_consistency(df, "predicción de probabilidades")
        
        # Obtener features
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].fillna(0)
        
        # Verificar que tenemos las mismas features que en entrenamiento
        if hasattr(self, 'feature_columns_') and self.feature_columns_:
            # Asegurar que tenemos exactamente las mismas features
            missing_features = set(self.feature_columns_) - set(feature_columns)
            if missing_features:
                logger.warning(f"Agregando features faltantes con valores por defecto: {list(missing_features)}")
                for feature in missing_features:
                    X[feature] = 0.5  # Valor neutro
            
            # Reordenar columnas para que coincidan con el entrenamiento
            X = X.reindex(columns=self.feature_columns_, fill_value=0.5)
        
        # Escalar features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Predicción de probabilidades con modelo stacking
        if self.stacking_model is not None:
            probabilities = self.stacking_model.predict_proba(X_scaled)
        else:
            # Fallback: usar modelo con mejor rendimiento
            best_model_name = max(self.cv_results.items(), key=lambda x: x[1]['accuracy_mean'])[0]
            best_model = self.models[best_model_name]
            probabilities = best_model.predict_proba(X_scaled)
        
        return probabilities
    
    def _optimize_with_bayesian(self, X_train, y_train):
        """Optimización bayesiana de hiperparámetros MEJORADA"""
        
        if not BAYESIAN_AVAILABLE:
            logger.warning("⚠️  Optimización bayesiana no disponible - skopt no instalado")
            return
        
        logger.info("🔧 Iniciando optimización bayesiana de hiperparámetros...")
        
        # Distribuir llamadas entre modelos de manera eficiente
        calls_per_model = max(8, self.bayesian_n_calls // 3)  # Mínimo 8 llamadas por modelo
        
        # Optimizar solo los modelos más importantes
        logger.info(f"🎯 Optimizando 3 modelos principales con {calls_per_model} llamadas cada uno...")
        
        # 1. Optimizar XGBoost (modelo principal)
        logger.info("🚀 Optimizando XGBoost...")
        self._optimize_xgboost_bayesian(X_train, y_train, calls_per_model)
        
        # 2. Optimizar LightGBM (modelo complementario)
        logger.info("🚀 Optimizando LightGBM...")
        self._optimize_lightgbm_bayesian(X_train, y_train, calls_per_model)
        
        # 3. Optimizar Red Neuronal (modelo diverso)
        logger.info("🚀 Optimizando Red Neuronal...")
        self._optimize_neural_net_bayesian(X_train, y_train, calls_per_model)
        
        logger.info("✅ Optimización bayesiana completada")
        
        # Log resumen de resultados
        if hasattr(self, 'bayesian_results') and self.bayesian_results:
            logger.info("📊 Resultados de optimización bayesiana:")
            for model_name, results in self.bayesian_results.items():
                if 'best_score' in results:
                    score = results['best_score']
                    logger.info(f"  • {model_name}: Mejor AUC = {score:.4f}")
    
    def _optimize_xgboost_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana específica para XGBoost MEJORADA"""
        
        # Espacio de búsqueda REDUCIDO pero efectivo
        space = [
            Integer(30, 100, name='n_estimators'),
            Integer(3, 6, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(1.0, 5.0, name='reg_alpha'),
            Real(2.0, 8.0, name='reg_lambda')
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
        logger.debug(f"Ejecutando {n_calls} llamadas de optimización para XGBoost...")
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,  # Asegurar mínimo 10 llamadas
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
        
        logger.info(f"✅ XGBoost optimizado | Mejor AUC: {-result.fun:.4f}")
    
    def _optimize_lightgbm_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana específica para LightGBM MEJORADA"""
        
        # Espacio de búsqueda REDUCIDO pero efectivo
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
            
            # Validación cruzada estratificada RÁPIDA
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # Retornar negativo para minimización
            return -cv_scores.mean()
        
        # Ejecutar optimización
        logger.debug(f"Ejecutando {n_calls} llamadas de optimización para LightGBM...")
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
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
        
        logger.info(f"✅ LightGBM optimizado | Mejor AUC: {-result.fun:.4f}")
    
    def _optimize_neural_net_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana para la red neuronal MEJORADA"""
        
        # Espacio de búsqueda REDUCIDO pero efectivo
        space = [
            Integer(32, 128, name='hidden_size'),
            Real(0.0001, 0.005, name='learning_rate'),
            Real(0.01, 0.08, name='weight_decay'),
            Real(0.3, 0.7, name='dropout_rate'),
            Integer(32, 128, name='batch_size')
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
                epochs=50,  # Reducido para optimización más rápida
                early_stopping_patience=10,
                device=self.device
            )
            
            # Validación cruzada manual RÁPIDA (solo 3 folds)
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
        logger.debug(f"Ejecutando {n_calls} llamadas de optimización para Red Neuronal...")
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        
        # Crear nuevo modelo con mejores parámetros
        self.models['neural_network'] = PyTorchNBAClassifier(
            hidden_size=best_params['hidden_size'],
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            dropout_rate=best_params['dropout_rate'],
            batch_size=int(best_params['batch_size']),
            epochs=100,  # Usar más epochs para el modelo final
            early_stopping_patience=15,
            device=self.device
        )
        
        self.bayesian_results['neural_network'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
        
        logger.info(f"✅ Red Neuronal optimizada | Mejor AUC: {-result.fun:.4f}")
    
    def _perform_cross_validation(self, X, y) -> Dict[str, Any]:
        """
        Validación cruzada CORREGIDA con 5 folds
        Generar métricas CV válidas para el modelo
        """
        logger.info("🔄 Realizando validación cruzada con 5 folds...")
        
        # Configurar validación cruzada con 5 folds
        n_splits = 5
        cv = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=42
        )
        
        cv_results = {}
        
        # Métricas de evaluación
        scoring_metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        
        logger.info(f"📊 Configuración CV | Folds: {n_splits} | Métricas: {len(scoring_metrics)}")
        
        # Validación cruzada para cada modelo
        total_models = len(self.models)
        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            logger.debug(f"🔄 [{idx}/{total_models}] Validación cruzada: {model_name}")
            
            model_cv_results = {}
            
            # Evaluar métricas
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        model, X, y, 
                        cv=cv, 
                        scoring=metric,
                        n_jobs=-1
                    )
                    
                    model_cv_results[metric] = {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'scores': scores.tolist()
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error en CV para {model_name} - {metric}: {e}")
                    model_cv_results[metric] = {
                        'mean': 0.0, 'std': 0.0, 'scores': []
                    }
            
            # Log métricas principales del modelo (solo accuracy y AUC)
            if 'accuracy' in model_cv_results and 'roc_auc' in model_cv_results:
                acc_mean = model_cv_results['accuracy']['mean']
                acc_std = model_cv_results['accuracy']['std']
                auc_mean = model_cv_results['roc_auc']['mean']
                auc_std = model_cv_results['roc_auc']['std']
                
                logger.info(f"📈 {model_name} CV | ACC: {acc_mean:.4f}±{acc_std:.4f} | "
                           f"AUC: {auc_mean:.4f}±{auc_std:.4f}")
            
            cv_results[model_name] = model_cv_results
        
        # VALIDACIÓN CRUZADA DEL STACKING MODEL
        logger.info("🔄 Validación cruzada del modelo Stacking...")
        try:
            stacking_cv_results = {}
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        self.stacking_model, X, y,
                        cv=cv,
                        scoring=metric,
                        n_jobs=-1
                    )
                    
                    stacking_cv_results[metric] = {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'scores': scores.tolist()
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error en CV Stacking - {metric}: {e}")
                    stacking_cv_results[metric] = {
                        'mean': 0.0, 'std': 0.0, 'scores': []
                    }
            
            cv_results['stacking'] = stacking_cv_results
            
            # Log métricas del stacking
            if 'accuracy' in stacking_cv_results and 'roc_auc' in stacking_cv_results:
                acc_mean = stacking_cv_results['accuracy']['mean']
                acc_std = stacking_cv_results['accuracy']['std']
                auc_mean = stacking_cv_results['roc_auc']['mean']
                auc_std = stacking_cv_results['roc_auc']['std']
                
                logger.info(f"📈 Stacking CV | ACC: {acc_mean:.4f}±{acc_std:.4f} | "
                           f"AUC: {auc_mean:.4f}±{auc_std:.4f}")
                
        except Exception as e:
            logger.error(f"❌ Error en validación cruzada del Stacking: {e}")
            cv_results['stacking'] = {'error': str(e)}
        
        # RESUMEN DE ALERTAS DE SOBREAJUSTE (simplificado)
        logger.info("🔍 ANÁLISIS DE SOBREAJUSTE")
        logger.info("=" * 50)
        
        overfitting_models = []
        normal_models = []
        
        for model_name in self.models.keys():
            if model_name in cv_results:
                acc_std = cv_results[model_name].get('accuracy', {}).get('std', 0)
                auc_std = cv_results[model_name].get('roc_auc', {}).get('std', 0)
                
                if acc_std < 0.02 or auc_std < 0.02:
                    overfitting_models.append(model_name)
                else:
                    normal_models.append(model_name)
        
        # Log consolidado
        if overfitting_models:
            logger.warning(f"🚨 Modelos con posible sobreajuste: {', '.join(overfitting_models)}")
        if normal_models:
            logger.info(f"✅ Modelos con variabilidad normal: {', '.join(normal_models)}")
        
        logger.info("=" * 50)
        logger.info("✅ Validación cruzada completada")
        
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
        
        logger.info("🔍 Iniciando validación del modelo...")
        
        # Predicciones
        y_true = df['is_win']
        y_pred = self.predict(df)
        y_proba = self.predict_proba(df)[:, 1]
        
        # Métricas de validación usando MetricsCalculator
        validation_metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, y_proba
        )
        
        # Log métricas principales
        OptimizedLogger.log_performance_metrics(
            logger, validation_metrics, "Modelo", "Validación"
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
                logger.info(f"🏠 Accuracy en casa: {home_acc:.4f}")
            
            if away_mask.sum() > 0:
                away_acc = accuracy_score(
                    y_true[away_mask], y_pred[away_mask]
                )
                context_analysis['away_accuracy'] = away_acc
                logger.info(f"✈️  Accuracy visitante: {away_acc:.4f}")
        
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
        
        logger.info(f"✅ Validación completada | Muestras: {len(df)} | "
                   f"Accuracy: {validation_metrics['accuracy']:.4f}")
        
        return validation_report
    
    def save_model(self, save_path: str = None):
        """Guardar modelo entrenado"""
        
        if save_path is None:
            save_path = "trained_models/is_win_model.joblib"
        
        logger.info(f"💾 Guardando modelo en: {save_path}")
        
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
        
        # Log información del modelo guardado
        model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"✅ Modelo guardado | Tamaño: {model_size_mb:.1f}MB | "
                   f"Modelos: {len(self.models)} | Features: {len(self.feature_importance.get('average', {}))}")
        
        return save_path
    
    @staticmethod
    def load_model(model_path: str = "trained_models/is_win_model.joblib"):
        """Cargar modelo entrenado"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        logger.info(f"📂 Cargando modelo desde: {model_path}")
        
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
        
        # Log información del modelo cargado
        metadata = model_data.get('model_metadata', {})
        created_at = metadata.get('created_at', 'Desconocido')
        device = metadata.get('device', 'Desconocido')
        
        logger.info(f"✅ Modelo cargado | Creado: {created_at} | "
                   f"Dispositivo: {device} | Modelos: {len(model.models)}")
        
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Resumen completo del entrenamiento"""
        
        if not self.training_results:
            logger.warning("⚠️  Modelo no entrenado - no hay resumen disponible")
            return {"error": "Modelo no entrenado"}
        
        logger.info("📊 Generando resumen de entrenamiento...")
        
        # Obtener métricas CV del stacking si están disponibles
        cv_data = self.training_results.get('cross_validation', {})
        stacking_cv = cv_data.get('stacking', {})
        
        summary = {
            "model_performance": {
                "stacking_accuracy": self.training_results.get(
                    'stacking_metrics', {}
                ).get('accuracy', 0),
                "stacking_auc": self.training_results.get(
                    'stacking_metrics', {}
                ).get('auc_roc', 0),
                "cv_accuracy_mean": stacking_cv.get('accuracy', {}).get('mean', 0),
                "cv_accuracy_std": stacking_cv.get('accuracy', {}).get('std', 0),
                "cv_auc_mean": stacking_cv.get('roc_auc', {}).get('mean', 0),
                "cv_auc_std": stacking_cv.get('roc_auc', {}).get('std', 0)
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
        
        # Log resumen de rendimiento
        stacking_acc = summary["model_performance"]["stacking_accuracy"]
        stacking_auc = summary["model_performance"]["stacking_auc"]
        cv_acc_mean = summary["model_performance"]["cv_accuracy_mean"]
        cv_acc_std = summary["model_performance"]["cv_accuracy_std"]
        feature_count = summary["training_info"]["feature_count"]
        
        logger.info(f"📈 Resumen generado | Stacking ACC: {stacking_acc:.4f} | "
                   f"AUC: {stacking_auc:.4f} | CV ACC: {cv_acc_mean:.4f}±{cv_acc_std:.4f} | "
                   f"Features: {feature_count}")
        
        return summary
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Obtener información completa del GPU configurado"""
        
        logger.debug("🖥️  Recopilando información de GPU...")
        
        gpu_info = {
            'configuration': self.gpu_config,
            'current_device': str(self.device) if self.device else None,
            'available_devices': GPUManager.get_available_devices(),
            'memory_requirements': {
                'min_memory_gb': self.min_memory_gb,
                'recommended_memory_gb': 4.0
            }
        }
        
        # Información específica del dispositivo actual
        if self.device:
            gpu_info['current_device_info'] = GPUManager.get_device_info(str(self.device))
        
        # Información de red neuronal si está disponible
        if ('neural_network' in self.models and 
            hasattr(self.models['neural_network'], 'get_gpu_memory_summary')):
            gpu_info['neural_network_memory'] = (
                self.models['neural_network'].get_gpu_memory_summary()
            )
        
        logger.debug(f"✅ Información GPU recopilada | Dispositivo: {self.device}")
        
        return gpu_info
    
    def _validate_feature_consistency(self, df: pd.DataFrame, context: str = "evaluación") -> bool:
        """
        Validar que las features del DataFrame sean consistentes con las del entrenamiento
        
        Args:
            df: DataFrame a validar
            context: Contexto de la validación (entrenamiento/evaluación)
            
        Returns:
            True si las features son consistentes, False si hay problemas
        """
        if not hasattr(self, 'feature_columns_') or not self.feature_columns_:
            logger.warning(f"No hay features de referencia para validar en {context}")
            return True  # No podemos validar, asumir que está bien
        
        # Generar features para este DataFrame
        current_features = self.get_feature_columns(df)
        
        # Comparar con features de entrenamiento
        missing_features = set(self.feature_columns_) - set(current_features)
        extra_features = set(current_features) - set(self.feature_columns_)
        
        if missing_features:
            logger.error(f"Features faltantes en {context}: {list(missing_features)}")
            
            # Intentar generar features faltantes con valores por defecto
            for feature in missing_features:
                if feature not in df.columns:
                    logger.warning(f"Creando feature faltante '{feature}' con valor por defecto")
                    df[feature] = 0.5  # Valor neutro para features de win rate
        
        if extra_features:
            logger.debug(f"Features adicionales en {context}: {list(extra_features)}")
        
        # Verificar que las features críticas estén presentes
        critical_features = ['home_win_rate_10g', 'away_win_rate_10g']
        missing_critical = [f for f in critical_features if f not in current_features and f in self.feature_columns_]
        
        if missing_critical:
            logger.error(f"Features críticas faltantes en {context}: {missing_critical}")
            return False
        
        return len(missing_features) == 0
    
    def setup_confidence_thresholds(self, test_data: pd.DataFrame, 
                                   strategies: List[str] = None) -> Dict[str, Any]:
        """
        Configurar umbrales de confianza basados en datos de prueba
        
        Args:
            test_data: Datos de prueba para optimizar umbrales
            strategies: Lista de estrategias a configurar
            
        Returns:
            Resultados de configuración de umbrales
        """
        if not self.is_trained:
            raise ValueError("El modelo debe estar entrenado antes de configurar umbrales")
        
        logger.info("🎯 Configurando umbrales de confianza para decisiones...")
        
        # Inicializar optimizador de umbrales
        self.threshold_optimizer = NBAConfidenceThresholdOptimizer()
        
        # Obtener predicciones
        y_true = test_data['is_win']
        y_proba = self.predict_proba(test_data)[:, 1]
        
        # Analizar rendimiento por confianza
        confidence_analysis = self.threshold_optimizer.analyze_confidence_performance(
            y_true.values, y_proba
        )
        
        # Configurar estrategias
        if strategies is None:
            strategies = ["balanced", "conservative", "aggressive", "high_confidence"]
        
        threshold_results = {}
        for strategy in strategies:
            logger.info(f"🔧 Configurando estrategia: {strategy}")
            
            # Optimizar umbrales
            thresholds = self.threshold_optimizer.optimize_decision_thresholds(
                y_true.values, y_proba, strategy
            )
            
            # Crear reglas de decisión
            rules = self.threshold_optimizer.create_decision_rules(strategy)
            
            # Evaluar rendimiento
            evaluation = self.threshold_optimizer.evaluate_threshold_performance(
                y_true.values, y_proba, strategy
            )
            
            threshold_results[strategy] = {
                'thresholds': thresholds,
                'rules': rules,
                'evaluation': evaluation
            }
        
        # Comparar estrategias
        strategy_comparison = self.threshold_optimizer.get_strategy_comparison(
            y_true.values, y_proba
        )
        
        # Guardar resultados
        self.threshold_results = {
            'confidence_analysis': confidence_analysis,
            'strategy_results': threshold_results,
            'strategy_comparison': strategy_comparison,
            'test_samples': len(test_data),
            'configured_at': datetime.now().isoformat()
        }
        
        logger.info("✅ Umbrales de confianza configurados exitosamente")
        
        return self.threshold_results
    
    def predict_with_confidence(self, df: pd.DataFrame, 
                              strategy: str = "balanced") -> List[Dict[str, Any]]:
        """
        Realizar predicciones con análisis de confianza y recomendaciones
        
        Args:
            df: DataFrame con datos para predicción
            strategy: Estrategia de umbrales a usar
            
        Returns:
            Lista de predicciones con análisis de confianza
        """
        if not hasattr(self, 'threshold_optimizer'):
            raise ValueError("Umbrales no configurados. Ejecuta setup_confidence_thresholds() primero.")
        
        logger.info(f"🔮 Realizando predicciones con análisis de confianza ({strategy})...")
        
        # Obtener predicciones básicas
        y_proba = self.predict_proba(df)[:, 1]
        
        # Generar decisiones con confianza
        decisions = self.threshold_optimizer.batch_decisions(y_proba, strategy)
        
        # Enriquecer con información del juego
        enriched_predictions = []
        for i, decision in enumerate(decisions):
            game_info = {
                'game_index': i,
                'team': df.iloc[i].get('Team', 'Unknown'),
                'opponent': df.iloc[i].get('Opp', 'Unknown'),
                'date': df.iloc[i].get('Date', 'Unknown'),
                'is_home': df.iloc[i].get('is_home', None)
            }
            
            # Combinar información del juego con decisión
            enriched_prediction = {**game_info, **decision}
            enriched_predictions.append(enriched_prediction)
        
        logger.info(f"✅ {len(enriched_predictions)} predicciones generadas con análisis de confianza")
        
        return enriched_predictions
    
    def get_confidence_summary(self, predictions: List[Dict[str, Any]] = None,
                             strategy: str = "balanced") -> Dict[str, Any]:
        """
        Obtener resumen de confianza de las predicciones
        
        Args:
            predictions: Lista de predicciones (opcional)
            strategy: Estrategia usada
            
        Returns:
            Resumen de confianza
        """
        if predictions is None and not hasattr(self, 'threshold_results'):
            raise ValueError("No hay predicciones o umbrales configurados")
        
        if predictions is None:
            # Usar resultados de configuración
            summary = {
                'threshold_configuration': self.threshold_results,
                'best_strategies': self.threshold_results['strategy_comparison']['best_by_metric']
            }
        else:
            # Analizar predicciones proporcionadas
            confidence_counts = {}
            win_predictions = 0
            high_confidence_predictions = 0
            
            for pred in predictions:
                level = pred['confidence_level']
                confidence_counts[level] = confidence_counts.get(level, 0) + 1
                
                if pred['predicted_win']:
                    win_predictions += 1
                
                if level in ['high_confidence', 'very_high_confidence']:
                    high_confidence_predictions += 1
            
            summary = {
                'total_predictions': len(predictions),
                'win_predictions': win_predictions,
                'loss_predictions': len(predictions) - win_predictions,
                'win_rate_predicted': win_predictions / len(predictions),
                'confidence_distribution': confidence_counts,
                'high_confidence_rate': high_confidence_predictions / len(predictions),
                'strategy_used': strategy,
                'recommendations': self._generate_confidence_recommendations(confidence_counts, len(predictions))
            }
        
        return summary
    
    def _generate_confidence_recommendations(self, confidence_counts: Dict[str, int], 
                                           total: int) -> List[str]:
        """Generar recomendaciones basadas en distribución de confianza"""
        recommendations = []
        
        no_decision_pct = confidence_counts.get('no_decision', 0) / total * 100
        high_confidence_pct = confidence_counts.get('high_confidence', 0) / total * 100
        
        if no_decision_pct > 30:
            recommendations.append(
                f"⚠️ {no_decision_pct:.1f}% de predicciones tienen confianza insuficiente. "
                "Considera usar estrategia más agresiva o recopilar más datos."
            )
        
        if high_confidence_pct > 20:
            recommendations.append(
                f"🎯 {high_confidence_pct:.1f}% de predicciones tienen alta confianza. "
                "Estas son las más confiables para tomar decisiones."
            )
        
        if high_confidence_pct < 5:
            recommendations.append(
                "📊 Pocas predicciones de alta confianza. "
                "El modelo puede beneficiarse de más datos o features adicionales."
            )
        
        return recommendations
    
    def predict_single_game(self, team: str, opponent: str, is_home: bool,
                          strategy: str = "balanced") -> Dict[str, Any]:
        """
        Predecir resultado de un juego específico con análisis de confianza
        
        Args:
            team: Equipo que juega
            opponent: Equipo oponente
            is_home: Si el equipo juega en casa
            strategy: Estrategia de umbrales
            
        Returns:
            Predicción detallada con confianza
        """
        if not hasattr(self, 'threshold_optimizer'):
            raise ValueError("Umbrales no configurados. Ejecuta setup_confidence_thresholds() primero.")
        
        logger.info(f"🏀 Prediciendo: {team} vs {opponent} ({'Casa' if is_home else 'Visitante'})")
        
        # Crear DataFrame para predicción (necesitaríamos datos históricos reales)
        # Por ahora, esto es un ejemplo de la estructura
        game_data = pd.DataFrame({
            'Team': [team],
            'Opp': [opponent],
            'is_home': [1 if is_home else 0],
            'Date': [datetime.now()]
        })
        
        # Nota: En implementación real, necesitaríamos cargar datos históricos
        # y generar todas las features necesarias
        
        try:
            # Obtener predicción con confianza
            predictions = self.predict_with_confidence(game_data, strategy)
            prediction = predictions[0]
            
            # Enriquecer con contexto del juego
            prediction['game_context'] = {
                'matchup': f"{team} vs {opponent}",
                'venue': "Casa" if is_home else "Visitante",
                'prediction_time': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Predicción: {prediction['recommendation']}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return {
                'error': str(e),
                'team': team,
                'opponent': opponent,
                'is_home': is_home
            }
    
    def export_threshold_configuration(self, filepath: str = None) -> str:
        """
        Exportar configuración de umbrales a archivo JSON
        
        Args:
            filepath: Ruta del archivo (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        if not hasattr(self, 'threshold_results'):
            raise ValueError("No hay configuración de umbrales para exportar")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"json/nba_confidence_thresholds_{timestamp}.json"
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convertir a JSON serializable
        export_data = safe_json_serialize(self.threshold_results)
        
        # Guardar archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Configuración de umbrales exportada a: {filepath}")
        
        return filepath
    
    @property
    def is_trained(self) -> bool:
        """Verificar si el modelo está entrenado"""
        return (self.stacking_model is not None and 
                hasattr(self, 'training_results') and 
                self.training_results)


class NBAConfidenceThresholdOptimizer:
    """
    Optimizador de umbrales de confianza para decisiones de victorias NBA
    Basado en análisis de rendimiento por rangos de confianza
    """
    
    def __init__(self):
        self.thresholds = {}
        self.performance_by_confidence = {}
        self.calibration_data = {}
        self.decision_rules = {}
        
    def analyze_confidence_performance(self, y_true: np.ndarray, 
                                     y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analizar rendimiento del modelo por rangos de confianza
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            
        Returns:
            Análisis detallado por rangos de confianza
        """
        logger.info("🎯 Analizando rendimiento por rangos de confianza...")
        
        # Definir rangos de confianza más granulares
        confidence_ranges = [
            (0.0, 0.3, "Muy Baja"),
            (0.3, 0.4, "Baja"),
            (0.4, 0.6, "Media"),
            (0.6, 0.7, "Alta"),
            (0.7, 0.8, "Muy Alta"),
            (0.8, 1.0, "Extrema")
        ]
        
        analysis = {}
        
        for low, high, label in confidence_ranges:
            # Filtrar predicciones en este rango
            mask = (y_proba >= low) & (y_proba < high)
            
            if mask.sum() > 0:
                range_y_true = y_true[mask]
                range_y_proba = y_proba[mask]
                range_y_pred = (range_y_proba > 0.5).astype(int)
                
                # Calcular métricas para este rango
                accuracy = accuracy_score(range_y_true, range_y_pred)
                precision = precision_score(range_y_true, range_y_pred, zero_division=0)
                recall = recall_score(range_y_true, range_y_pred, zero_division=0)
                
                # Calcular calibración (qué tan bien alineadas están las probabilidades)
                avg_predicted_prob = range_y_proba.mean()
                actual_win_rate = range_y_true.mean()
                calibration_error = abs(avg_predicted_prob - actual_win_rate)
                
                analysis[f"{low:.1f}-{high:.1f}"] = {
                    'label': label,
                    'range': (low, high),
                    'samples': int(mask.sum()),
                    'percentage': float(mask.sum() / len(y_true) * 100),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'avg_predicted_prob': float(avg_predicted_prob),
                    'actual_win_rate': float(actual_win_rate),
                    'calibration_error': float(calibration_error),
                    'confidence_level': label
                }
                
                logger.info(f"📊 {label} ({low:.1f}-{high:.1f}): "
                           f"ACC={accuracy:.3f}, Muestras={mask.sum()}, "
                           f"Calibración={calibration_error:.3f}")
        
        self.performance_by_confidence = analysis
        return analysis
    
    def optimize_decision_thresholds(self, y_true: np.ndarray, 
                                   y_proba: np.ndarray,
                                   strategy: str = "balanced") -> Dict[str, float]:
        """
        Optimizar umbrales de decisión basados en estrategia
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            strategy: Estrategia de optimización
                     - "balanced": Balance entre precision y recall
                     - "conservative": Minimizar falsos positivos
                     - "aggressive": Maximizar detección de victorias
                     - "high_confidence": Solo decisiones de alta confianza
                     
        Returns:
            Umbrales optimizados por estrategia
        """
        logger.info(f"🔧 Optimizando umbrales para estrategia: {strategy}")
        
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        # Calcular curvas para optimización
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        
        thresholds = {}
        
        if strategy == "balanced":
            # Maximizar F1-Score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores[:-1])  # Excluir último elemento
            thresholds['decision'] = float(pr_thresholds[best_idx])
            thresholds['min_confidence'] = 0.3
            thresholds['high_confidence'] = 0.7
            
        elif strategy == "conservative":
            # Minimizar falsos positivos (alta precision)
            target_precision = 0.8
            valid_indices = precision >= target_precision
            if valid_indices.any():
                best_idx = np.where(valid_indices)[0][-1]  # Último índice válido
                thresholds['decision'] = float(pr_thresholds[min(best_idx, len(pr_thresholds)-1)])
            else:
                thresholds['decision'] = 0.7  # Fallback conservativo
            thresholds['min_confidence'] = 0.6
            thresholds['high_confidence'] = 0.8
            
        elif strategy == "aggressive":
            # Maximizar detección de victorias (alta recall)
            target_recall = 0.8
            valid_indices = recall >= target_recall
            if valid_indices.any():
                best_idx = np.where(valid_indices)[0][0]  # Primer índice válido
                thresholds['decision'] = float(pr_thresholds[best_idx])
            else:
                thresholds['decision'] = 0.3  # Fallback agresivo
            thresholds['min_confidence'] = 0.2
            thresholds['high_confidence'] = 0.6
            
        elif strategy == "high_confidence":
            # Solo decisiones de muy alta confianza
            thresholds['decision'] = 0.5  # Umbral estándar
            thresholds['min_confidence'] = 0.7  # Solo alta confianza
            thresholds['high_confidence'] = 0.85  # Muy alta confianza
            
        # Umbrales adicionales para diferentes niveles de decisión
        thresholds['very_low_confidence'] = 0.1
        thresholds['low_confidence'] = 0.3
        thresholds['medium_confidence'] = 0.6
        thresholds['very_high_confidence'] = 0.9
        
        self.thresholds[strategy] = thresholds
        
        logger.info(f"✅ Umbrales optimizados para {strategy}:")
        for key, value in thresholds.items():
            logger.info(f"  • {key}: {value:.3f}")
        
        return thresholds
    
    def create_decision_rules(self, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Crear reglas de decisión basadas en umbrales optimizados
        
        Args:
            strategy: Estrategia de umbrales a usar
            
        Returns:
            Reglas de decisión estructuradas
        """
        if strategy not in self.thresholds:
            raise ValueError(f"Estrategia {strategy} no encontrada. "
                           f"Ejecuta optimize_decision_thresholds() primero.")
        
        thresholds = self.thresholds[strategy]
        
        rules = {
            'strategy': strategy,
            'thresholds': thresholds,
            'decision_levels': {
                'no_decision': {
                    'condition': f"confianza < {thresholds['min_confidence']:.3f}",
                    'action': "ABSTENERSE - Confianza insuficiente",
                    'confidence_range': (0.0, thresholds['min_confidence']),
                    'recommendation': "No hacer predicción"
                },
                'low_confidence': {
                    'condition': f"{thresholds['min_confidence']:.3f} <= confianza < {thresholds.get('medium_confidence', 0.6):.3f}",
                    'action': "PRECAUCIÓN - Baja confianza",
                    'confidence_range': (thresholds['min_confidence'], thresholds.get('medium_confidence', 0.6)),
                    'recommendation': "Considerar factores adicionales"
                },
                'medium_confidence': {
                    'condition': f"{thresholds.get('medium_confidence', 0.6):.3f} <= confianza < {thresholds['high_confidence']:.3f}",
                    'action': "DECIDIR - Confianza moderada",
                    'confidence_range': (thresholds.get('medium_confidence', 0.6), thresholds['high_confidence']),
                    'recommendation': "Predicción confiable"
                },
                'high_confidence': {
                    'condition': f"confianza >= {thresholds['high_confidence']:.3f}",
                    'action': "DECIDIR CON CONFIANZA - Alta certeza",
                    'confidence_range': (thresholds['high_confidence'], 1.0),
                    'recommendation': "Predicción muy confiable"
                }
            }
        }
        
        self.decision_rules[strategy] = rules
        
        logger.info(f"📋 Reglas de decisión creadas para estrategia: {strategy}")
        for level, rule in rules['decision_levels'].items():
            logger.info(f"  • {level}: {rule['action']}")
        
        return rules
    
    def make_decision(self, probability: float, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Tomar decisión basada en probabilidad y estrategia
        
        Args:
            probability: Probabilidad de victoria predicha
            strategy: Estrategia de umbrales a usar
            
        Returns:
            Decisión estructurada con recomendaciones
        """
        if strategy not in self.decision_rules:
            self.create_decision_rules(strategy)
        
        rules = self.decision_rules[strategy]
        thresholds = rules['thresholds']
        
        # Determinar nivel de confianza
        confidence_level = "no_decision"
        if probability >= thresholds['high_confidence']:
            confidence_level = "high_confidence"
        elif probability >= thresholds.get('medium_confidence', 0.6):
            confidence_level = "medium_confidence"
        elif probability >= thresholds['min_confidence']:
            confidence_level = "low_confidence"
        
        # Determinar predicción
        predicted_win = probability > thresholds['decision']
        
        # Calcular confianza absoluta (distancia del 0.5)
        absolute_confidence = abs(probability - 0.5) * 2
        
        decision = {
            'probability': float(probability),
            'predicted_win': bool(predicted_win),
            'confidence_level': confidence_level,
            'absolute_confidence': float(absolute_confidence),
            'decision_rule': rules['decision_levels'][confidence_level],
            'strategy_used': strategy,
            'thresholds_used': thresholds,
            'recommendation': self._get_detailed_recommendation(
                probability, predicted_win, confidence_level, absolute_confidence
            )
        }
        
        return decision
    
    def _get_detailed_recommendation(self, probability: float, predicted_win: bool, 
                                   confidence_level: str, absolute_confidence: float) -> str:
        """Generar recomendación detallada basada en la decisión"""
        
        win_loss = "VICTORIA" if predicted_win else "DERROTA"
        prob_pct = probability * 100
        conf_pct = absolute_confidence * 100
        
        if confidence_level == "no_decision":
            return f"🚫 NO DECIDIR - Probabilidad {prob_pct:.1f}% demasiado incierta"
        
        elif confidence_level == "low_confidence":
            return f"⚠️ {win_loss} con {prob_pct:.1f}% - BAJA CONFIANZA ({conf_pct:.1f}%) - Considerar factores adicionales"
        
        elif confidence_level == "medium_confidence":
            return f"✅ {win_loss} con {prob_pct:.1f}% - CONFIANZA MODERADA ({conf_pct:.1f}%) - Predicción confiable"
        
        elif confidence_level == "high_confidence":
            return f"🎯 {win_loss} con {prob_pct:.1f}% - ALTA CONFIANZA ({conf_pct:.1f}%) - Predicción muy confiable"
        
        return f"{win_loss} con {prob_pct:.1f}% de probabilidad"
    
    def batch_decisions(self, probabilities: np.ndarray, 
                       strategy: str = "balanced") -> List[Dict[str, Any]]:
        """
        Tomar decisiones en lote para múltiples predicciones
        
        Args:
            probabilities: Array de probabilidades
            strategy: Estrategia de umbrales
            
        Returns:
            Lista de decisiones
        """
        logger.info(f"🔄 Procesando {len(probabilities)} decisiones con estrategia: {strategy}")
        
        decisions = []
        for prob in probabilities:
            decision = self.make_decision(float(prob), strategy)
            decisions.append(decision)
        
        # Estadísticas del lote
        confidence_counts = {}
        for decision in decisions:
            level = decision['confidence_level']
            confidence_counts[level] = confidence_counts.get(level, 0) + 1
        
        logger.info("📊 Distribución de decisiones:")
        for level, count in confidence_counts.items():
            pct = count / len(decisions) * 100
            logger.info(f"  • {level}: {count} ({pct:.1f}%)")
        
        return decisions
    
    def evaluate_threshold_performance(self, y_true: np.ndarray, 
                                     y_proba: np.ndarray,
                                     strategy: str = "balanced") -> Dict[str, Any]:
        """
        Evaluar rendimiento de los umbrales optimizados
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            strategy: Estrategia a evaluar
            
        Returns:
            Métricas de evaluación
        """
        if strategy not in self.thresholds:
            self.optimize_decision_thresholds(y_true, y_proba, strategy)
        
        thresholds = self.thresholds[strategy]
        decisions = self.batch_decisions(y_proba, strategy)
        
        # Filtrar solo decisiones donde se hace predicción
        decisive_mask = np.array([
            d['confidence_level'] != 'no_decision' for d in decisions
        ])
        
        if decisive_mask.sum() == 0:
            return {'error': 'No hay decisiones suficientemente confiables'}
        
        # Evaluar solo decisiones tomadas
        decisive_true = y_true[decisive_mask]
        decisive_pred = np.array([
            d['predicted_win'] for d in decisions if d['confidence_level'] != 'no_decision'
        ])
        decisive_proba = y_proba[decisive_mask]
        
        # Métricas de rendimiento
        evaluation = {
            'strategy': strategy,
            'total_samples': len(y_true),
            'decisive_samples': int(decisive_mask.sum()),
            'decision_rate': float(decisive_mask.sum() / len(y_true)),
            'accuracy': float(accuracy_score(decisive_true, decisive_pred)),
            'precision': float(precision_score(decisive_true, decisive_pred, zero_division=0)),
            'recall': float(recall_score(decisive_true, decisive_pred, zero_division=0)),
            'f1_score': float(f1_score(decisive_true, decisive_pred, zero_division=0)),
            'thresholds_used': thresholds,
            'confidence_distribution': {}
        }
        
        # Distribución por nivel de confianza
        for decision in decisions:
            level = decision['confidence_level']
            if level not in evaluation['confidence_distribution']:
                evaluation['confidence_distribution'][level] = 0
            evaluation['confidence_distribution'][level] += 1
        
        logger.info(f"📈 Evaluación de umbrales ({strategy}):")
        logger.info(f"  • Tasa de decisión: {evaluation['decision_rate']:.3f}")
        logger.info(f"  • Accuracy: {evaluation['accuracy']:.3f}")
        logger.info(f"  • Precision: {evaluation['precision']:.3f}")
        logger.info(f"  • Recall: {evaluation['recall']:.3f}")
        
        return evaluation
    
    def get_strategy_comparison(self, y_true: np.ndarray, 
                              y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Comparar todas las estrategias de umbrales
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            
        Returns:
            Comparación de estrategias
        """
        logger.info("🔍 Comparando estrategias de umbrales...")
        
        strategies = ["balanced", "conservative", "aggressive", "high_confidence"]
        comparison = {}
        
        for strategy in strategies:
            try:
                evaluation = self.evaluate_threshold_performance(y_true, y_proba, strategy)
                comparison[strategy] = evaluation
            except Exception as e:
                logger.error(f"Error evaluando estrategia {strategy}: {e}")
                comparison[strategy] = {'error': str(e)}
        
        # Encontrar mejor estrategia por métrica
        best_strategies = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'decision_rate']
        
        for metric in metrics:
            best_score = 0
            best_strategy = None
            
            for strategy, results in comparison.items():
                if 'error' not in results and metric in results:
                    if results[metric] > best_score:
                        best_score = results[metric]
                        best_strategy = strategy
            
            best_strategies[metric] = {
                'strategy': best_strategy,
                'score': best_score
            }
        
        comparison['best_by_metric'] = best_strategies
        
        logger.info("🏆 Mejores estrategias por métrica:")
        for metric, best in best_strategies.items():
            if best['strategy']:
                logger.info(f"  • {metric}: {best['strategy']} ({best['score']:.3f})")
        
        return comparison