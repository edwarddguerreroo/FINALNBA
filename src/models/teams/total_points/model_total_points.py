import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
from typing import Dict, List, Tuple
import warnings
import logging
from .features_total_points import TotalPointsFeatureEngine
import joblib
import sys
from tqdm import tqdm
import os
from datetime import datetime
import copy

warnings.filterwarnings('ignore')
# Silenciar warnings específicos de sklearn para LightGBM
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStoppingWithCheckpoint:
    """
    EARLY STOPPING AVANZADO CON CHECKPOINT AUTOMÁTICO
    Implementa múltiples criterios de parada y guarda automáticamente el mejor modelo
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True, mode: str = 'min',
                 monitor_overfitting: bool = True, overfitting_threshold: float = 0.1):
        """
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Mejora mínima para considerar como progreso
            restore_best_weights: Si restaurar los mejores pesos al final
            mode: 'min' para minimizar, 'max' para maximizar
            monitor_overfitting: Si monitorear overfitting (train_loss << val_loss)
            overfitting_threshold: Umbral para detectar overfitting
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.monitor_overfitting = monitor_overfitting
        self.overfitting_threshold = overfitting_threshold
        
        # Estado interno
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.stop_reason = ""
        
        # Histórico para análisis de tendencias
        self.history = {'val_loss': [], 'train_loss': []}
        
    def __call__(self, val_loss: float, train_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Evalúa si debe parar el entrenamiento y guarda checkpoint si hay mejora
        
        Args:
            val_loss: Pérdida de validación actual
            train_loss: Pérdida de entrenamiento actual
            model: Modelo para guardar checkpoint
            epoch: Época actual
            
        Returns:
            True si debe parar el entrenamiento
        """
        # Actualizar histórico
        self.history['val_loss'].append(val_loss)
        self.history['train_loss'].append(train_loss)
        
        # Evaluar si hay mejora
        current_value = val_loss
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            # CHECKPOINT: Guardar mejor modelo
            self.best_value = current_value
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self.wait = 0
            logger.debug(f"Época {epoch}: Nuevo mejor modelo guardado (val_loss: {val_loss:.6f})")
        else:
            self.wait += 1
        
        # CRITERIO 1: Paciencia estándar
        if self.wait >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Paciencia agotada ({self.patience} épocas sin mejora)"
            
        # CRITERIO 2: Detección de overfitting severo
        if self.monitor_overfitting and len(self.history['val_loss']) >= 5:
            recent_train_losses = self.history['train_loss'][-5:]
            recent_val_losses = self.history['val_loss'][-5:]
            
            avg_train_loss = np.mean(recent_train_losses)
            avg_val_loss = np.mean(recent_val_losses)
            
            # Si train_loss es significativamente menor que val_loss
            overfitting_ratio = (avg_val_loss - avg_train_loss) / (avg_train_loss + 1e-8)
            
            if overfitting_ratio > self.overfitting_threshold and epoch > 10:
                self.should_stop = True
                self.stopped_epoch = epoch
                self.stop_reason = f"Overfitting detectado (ratio: {overfitting_ratio:.3f})"
        
        # CRITERIO 3: Pérdida de validación aumentando consistentemente
        if len(self.history['val_loss']) >= 10:
            recent_losses = self.history['val_loss'][-10:]
            if len(recent_losses) >= 8:
                # Verificar si las últimas 8 épocas muestran tendencia creciente
                increasing_trend = sum([recent_losses[i] > recent_losses[i-1] 
                                      for i in range(1, 8)]) >= 6
                if increasing_trend and epoch > 15:
                    self.should_stop = True
                    self.stopped_epoch = epoch
                    self.stop_reason = "Tendencia creciente en pérdida de validación"
        
        return self.should_stop
    
    def restore_best_model(self, model: nn.Module) -> None:
        """Restaura el mejor modelo guardado"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Mejor modelo restaurado (época {self.best_epoch}, val_loss: {self.best_value:.6f})")
            logger.info(f"Razón de parada: {self.stop_reason}")
        else:
            logger.warning("No se pudo restaurar el mejor modelo")
    
    def get_summary(self) -> Dict:
        """Retorna resumen del early stopping"""
        return {
            'stopped': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'stop_reason': self.stop_reason,
            'total_wait': self.wait,
            'epochs_trained': len(self.history['val_loss'])
        }

class OptimizationProgressCallback:
    """Callback para mostrar progreso de optimización con barra"""
    
    def __init__(self, n_trials: int, description: str = "Optimizando"):
        self.n_trials = n_trials
        self.description = description
        self.pbar = None
        self.best_value = float('inf')
        
    def __call__(self, study, trial):
        if self.pbar is None:
                    self.pbar = tqdm(total=self.n_trials, desc=self.description, 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] MAE: {postfix}',
                       ncols=100)
        
        # Actualizar mejor valor
        if trial.value < self.best_value:
            self.best_value = trial.value
            
        self.pbar.set_postfix_str(f"{self.best_value:.3f}")
        self.pbar.update(1)
        
        if trial.number + 1 >= self.n_trials:
            self.pbar.close()

class AdvancedNeuralNetwork(nn.Module):
    """
    Red neuronal ultra-optimizada para predicción de puntos totales NBA
    Arquitectura simple con regularización extrema anti-overfitting
    """
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], dropout_rate: float = 0.7):
        super(AdvancedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Capa de entrada con normalización fuerte
        layers.extend([
            nn.Linear(prev_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.LeakyReLU(0.1),  # LeakyReLU para mejor gradiente
            nn.Dropout(dropout_rate)
        ])
        prev_size = hidden_sizes[0]
        
        # Capas ocultas más simples
        for i, hidden_size in enumerate(hidden_sizes[1:], 1):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate * (0.7 ** i))  # Dropout decreciente más agresivo
            ])
            prev_size = hidden_size
        
        # Capa de salida simplificada
        layers.extend([
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_size, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(16, 1)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Inicialización más conservadora
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.5)  # Gain más bajo
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Normalización de entrada más suave
        x = torch.nn.functional.normalize(x, p=2, dim=1) * 0.5  # Escalar más bajo
        output = self.network(x)
        # Aplicar límites más estrictos NBA
        return torch.clamp(output, 200, 250)  # Rango más estrecho

class DataValidator:
    """
    VALIDADOR DE DATOS ROBUSTO
    Asegura que no pasen NaNs, Infs o datos problemáticos a los modelos
    """
    
    @staticmethod
    def clean_and_validate_data(X: np.ndarray, y: np.ndarray, 
                               model_name: str = "unknown") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LIMPIEZA Y VALIDACIÓN ROBUSTA DE DATOS
        
        Args:
            X: Features de entrada
            y: Target
            model_name: Nombre del modelo para logging
            
        Returns:
            X_clean, y_clean, valid_mask
        """
        try:
            # Convertir a arrays NumPy si no lo son
            if not isinstance(X, np.ndarray):
                X = np.asarray(X, dtype=np.float32)
            if not isinstance(y, np.ndarray):
                y = np.asarray(y, dtype=np.float32)
            
            original_size = len(X)
            
            # PASO 1: Detectar valores problemáticos
            # NaNs en features
            nan_mask_X = np.isnan(X).any(axis=1)
            # Infinitos en features
            inf_mask_X = np.isinf(X).any(axis=1)
            # NaNs en target
            nan_mask_y = np.isnan(y)
            # Infinitos en target
            inf_mask_y = np.isinf(y)
            # Target fuera de rango NBA (muy importante para nuestro dominio)
            range_mask_y = (y < 150) | (y > 350)
            
            # Crear máscara de datos válidos
            valid_mask = ~(nan_mask_X | inf_mask_X | nan_mask_y | inf_mask_y | range_mask_y)
            
            # PASO 2: Reportar problemas encontrados
            n_nan_X = np.sum(nan_mask_X)
            n_inf_X = np.sum(inf_mask_X)
            n_nan_y = np.sum(nan_mask_y)
            n_inf_y = np.sum(inf_mask_y)
            n_range_y = np.sum(range_mask_y)
            n_valid = np.sum(valid_mask)
            
            if n_nan_X > 0:
                logger.warning(f"{model_name}: {n_nan_X} filas con NaN en features eliminadas")
            if n_inf_X > 0:
                logger.warning(f"{model_name}: {n_inf_X} filas con Inf en features eliminadas")
            if n_nan_y > 0:
                logger.warning(f"{model_name}: {n_nan_y} filas con NaN en target eliminadas")
            if n_inf_y > 0:
                logger.warning(f"{model_name}: {n_inf_y} filas con Inf en target eliminadas")
            if n_range_y > 0:
                logger.warning(f"{model_name}: {n_range_y} filas con target fuera de rango NBA eliminadas")
            
            # PASO 3: Validar que quedan suficientes datos
            if n_valid < 10:
                raise ValueError(f"{model_name}: Muy pocos datos válidos ({n_valid}) después de limpieza")
            
            retention_rate = n_valid / original_size
            if retention_rate < 0.5:
                logger.warning(f"{model_name}: Baja retención de datos ({retention_rate:.1%})")
            
            # PASO 4: Aplicar máscara y limpiar datos restantes
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            # PASO 5: Limpieza adicional de outliers extremos en features
            X_clean = DataValidator._clean_feature_outliers(X_clean, model_name)
            
            # PASO 6: Validación final
            assert not np.any(np.isnan(X_clean)), f"{model_name}: NaNs detectados después de limpieza en X"
            assert not np.any(np.isinf(X_clean)), f"{model_name}: Infs detectados después de limpieza en X"
            assert not np.any(np.isnan(y_clean)), f"{model_name}: NaNs detectados después de limpieza en y"
            assert not np.any(np.isinf(y_clean)), f"{model_name}: Infs detectados después de limpieza en y"
            
            logger.debug(f"{model_name}: Datos validados - {len(X_clean)} muestras limpias de {original_size} originales")
            
            return X_clean, y_clean, valid_mask
            
        except Exception as e:
            logger.error(f"Error en validación de datos para {model_name}: {e}")
            raise
    
    @staticmethod
    def _clean_feature_outliers(X: np.ndarray, model_name: str, 
                               method: str = 'iqr', factor: float = 3.0) -> np.ndarray:
        """
        LIMPIEZA DE OUTLIERS EN FEATURES
        
        Args:
            X: Features array
            model_name: Nombre del modelo
            method: 'iqr' o 'zscore'
            factor: Factor multiplicativo para el umbral
        """
        try:
            X_clean = X.copy()
            n_features = X.shape[1]
            
            if method == 'iqr':
                # Método IQR (Interquartile Range)
                for col in range(n_features):
                    Q1 = np.percentile(X_clean[:, col], 25)
                    Q3 = np.percentile(X_clean[:, col], 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    # Recortar valores extremos
                    X_clean[:, col] = np.clip(X_clean[:, col], lower_bound, upper_bound)
            
            elif method == 'zscore':
                # Método Z-score
                for col in range(n_features):
                    mean = np.mean(X_clean[:, col])
                    std = np.std(X_clean[:, col])
                    z_scores = np.abs((X_clean[:, col] - mean) / (std + 1e-8))
                    
                    # Recortar valores con z-score > factor
                    outlier_mask = z_scores > factor
                    if np.any(outlier_mask):
                        # Reemplazar outliers con valores en el límite
                        limit_value = mean + np.sign(X_clean[:, col] - mean) * factor * std
                        X_clean[outlier_mask, col] = limit_value[outlier_mask]
            
            return X_clean
            
        except Exception as e:
            logger.warning(f"Error limpiando outliers para {model_name}: {e}")
            return X  # Retornar datos originales si falla la limpieza
    
    @staticmethod
    def validate_prediction_input(X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        VALIDACIÓN ESPECÍFICA PARA PREDICCIONES
        
        Args:
            X: Array de features para predicción
            feature_names: Nombres de features (opcional)
            
        Returns:
            X_validated: Array validado y limpio
        """
        try:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X, dtype=np.float32)
            
            # Verificar dimensiones
            if X.ndim != 2:
                raise ValueError(f"X debe ser 2D, recibido: {X.ndim}D")
            
            if X.shape[0] == 0:
                raise ValueError("X no puede estar vacío")
            
            # Detectar y corregir NaNs/Infs
            nan_mask = np.isnan(X)
            inf_mask = np.isinf(X)
            
            if np.any(nan_mask):
                n_nans = np.sum(nan_mask)
                logger.warning(f"Predicción: {n_nans} valores NaN detectados y corregidos")
                X = np.where(nan_mask, 0.0, X)  # Reemplazar NaNs con 0
            
            if np.any(inf_mask):
                n_infs = np.sum(inf_mask)
                logger.warning(f"Predicción: {n_infs} valores Inf detectados y corregidos")
                # Reemplazar Infs con valores límite
                X = np.where(inf_mask & (X > 0), 1000.0, X)  # +Inf -> 1000
                X = np.where(inf_mask & (X < 0), -1000.0, X)  # -Inf -> -1000
            
            # Verificar rango de features
            if np.any(np.abs(X) > 1e6):
                logger.warning("Predicción: Features con valores muy grandes detectadas")
                X = np.clip(X, -1e6, 1e6)
            
            return X.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error validando input de predicción: {e}")
            raise
    
    @staticmethod
    def safe_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE ENGINEERING SEGURO CONTRA NaNs
        Aplica transformaciones robustas que no generen NaNs
        
        Args:
            df: DataFrame con features
            
        Returns:
            df_safe: DataFrame con features seguras
        """
        try:
            df_safe = df.copy()
            
            # Reemplazar NaNs antes de cualquier operación
            numeric_cols = df_safe.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df_safe[col].isna().any():
                    # Estrategia de reemplazo específica por tipo de feature
                    if 'pct' in col.lower() or '%' in col:
                        fill_value = 0.45  # Promedio NBA para porcentajes
                    elif 'pts' in col.lower() or 'score' in col.lower():
                        fill_value = 110   # Promedio NBA para puntos
                    elif 'pace' in col.lower():
                        fill_value = 100   # Pace promedio NBA
                    else:
                        fill_value = df_safe[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                    
                    df_safe[col] = df_safe[col].fillna(fill_value)
            
            # Verificar que no queden NaNs
            remaining_nans = df_safe.isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"Feature engineering: {remaining_nans} NaNs restantes después de limpieza")
                df_safe = df_safe.fillna(0.0)  # Último recurso
            
            return df_safe
            
        except Exception as e:
            logger.error(f"Error en feature engineering seguro: {e}")
            return df.fillna(0.0)  # Fallback ultra-conservador

class NBATotalPointsPredictor:
    """
    Predictor de élite mundial para total de puntos NBA
    Arquitectura ensemble con optimización bayesiana y deep learning
    Objetivo: >97% precisión
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engine = TotalPointsFeatureEngine()
        self.scalers = {
            'standard': StandardScaler(),
            'robust': StandardScaler()
        }
        
        # Modelos base del ensemble
        self.base_models = {}
        self.neural_network = None
        self.meta_model = None
        self.is_trained = False
        
        # Configuración de dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Métricas de rendimiento
        self.performance_metrics = {}
        
        # CONFIGURACIÓN CENTRALIZADA DEL ENSEMBLE
        self.ensemble_config = {
            'model_weights': {
                'extra_trees_primary': 0.35,
                'gradient_boost_primary': 0.30,
                'random_forest_primary': 0.25,
                'ridge_ultra_conservative': 0.10
            },
            'prediction_limits': {
                'min_value': 200,
                'max_value': 250,
                'extreme_penalty_threshold': 8.0
            },
            'confidence_thresholds': {
                'min_confidence': 70,
                'max_confidence': 95,
                'consistency_weight': 0.7,
                'performance_weight': 0.3
            }
        }
    
    # ==================== FUNCIONES UTILITARIAS CENTRALIZADAS ====================
    
    def _prepare_scaled_features(self, X_data: np.ndarray, scaler_type: str = 'standard', 
                                fit_scaler: bool = False) -> np.ndarray:
        """
        FUNCIÓN UTILITARIA OPTIMIZADA: Preparación y escalado de features centralizada
        Optimizada para trabajar exclusivamente con NumPy arrays
        
        Args:
            X_data: Datos a escalar (NumPy array)
            scaler_type: 'standard' o 'robust'
            fit_scaler: Si ajustar el scaler (True para entrenamiento, False para predicción)
        """
        try:
            if scaler_type not in self.scalers:
                raise ValueError(f"Scaler tipo '{scaler_type}' no disponible")
            
            scaler = self.scalers[scaler_type]
            
            # OPTIMIZACIÓN: Asegurar que trabajamos con arrays NumPy contiguos
            if not isinstance(X_data, np.ndarray):
                X_data = np.asarray(X_data, dtype=np.float32)
            elif not X_data.flags['C_CONTIGUOUS']:
                X_data = np.ascontiguousarray(X_data, dtype=np.float32)
            
            if fit_scaler:
                X_scaled = scaler.fit_transform(X_data)
                logger.debug(f"Scaler {scaler_type} ajustado y aplicado")
            else:
                X_scaled = scaler.transform(X_data)
                logger.debug(f"Scaler {scaler_type} aplicado")
            
            # OPTIMIZACIÓN: Validar resultados usando operaciones vectorizadas de NumPy
            nan_mask = np.isnan(X_scaled)
            inf_mask = np.isinf(X_scaled)
            
            if np.any(nan_mask) or np.any(inf_mask):
                logger.warning("NaN/Inf detectado después del escalado, aplicando corrección vectorizada")
                # Corrección vectorizada más eficiente
                X_scaled = np.where(nan_mask, 0.0, X_scaled)
                X_scaled = np.where(inf_mask & (X_scaled > 0), 1.0, X_scaled)
                X_scaled = np.where(inf_mask & (X_scaled < 0), -1.0, X_scaled)
            
            return X_scaled.astype(np.float32)  # Asegurar tipo consistente
            
        except Exception as e:
            logger.error(f"Error en escalado de features: {e}")
            return X_data.astype(np.float32)  # Retornar datos originales como fallback
    
    def _get_model_appropriate_data(self, model_name: str, X_train: np.ndarray, X_val: np.ndarray,
                                   X_train_scaled: np.ndarray, X_val_scaled: np.ndarray,
                                   X_train_robust: np.ndarray, X_val_robust: np.ndarray,
                                   feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        FUNCIÓN UTILITARIA OPTIMIZADA: Selecciona datos apropiados para cada tipo de modelo
        Usa NumPy arrays para mejor rendimiento en secciones críticas
        
        Args:
            model_name: Nombre del modelo
            X_train, X_val: Datos originales
            X_train_scaled, X_val_scaled: Datos escalados estándar
            X_train_robust, X_val_robust: Datos escalados robustos
            feature_names: Nombres de features (para modelos que los requieren)
        """
        try:
            # Modelos que requieren escalado estándar
            if any(keyword in model_name.lower() for keyword in ['ridge', 'elastic_net', 'neural']):
                return X_train_scaled, X_val_scaled
            
            # Modelos que prefieren escalado robusto
            elif any(keyword in model_name.lower() for keyword in ['lightgbm', 'catboost', 'xgboost']):
                # OPTIMIZACIÓN: Solo crear DataFrame si es absolutamente necesario
                # Muchos modelos de boosting funcionan bien con arrays NumPy
                if feature_names is not None and len(feature_names) > 0:
                    # Verificar si el modelo realmente necesita nombres de features
                    if 'lightgbm' in model_name.lower():
                        # LightGBM puede usar arrays NumPy directamente
                        return X_train_robust, X_val_robust
                    else:
                        # Solo XGBoost y CatBoost realmente necesitan DataFrames en algunos casos
                        X_tr = pd.DataFrame(X_train_robust, columns=feature_names)
                        X_v = pd.DataFrame(X_val_robust, columns=feature_names)
                        return X_tr, X_v
                else:
                    return X_train_robust, X_val_robust
            
            # Modelos que usan datos originales
            else:
                return X_train, X_val
                
        except Exception as e:
            logger.warning(f"Error seleccionando datos para {model_name}: {e}")
            return X_train, X_val  # Fallback a datos originales
    
    def _apply_prediction_limits_and_validation(self, predictions: np.ndarray, 
                                               model_name: str = "unknown") -> np.ndarray:
        """
        FUNCIÓN UTILITARIA: Aplica límites y validación a predicciones
        
        Args:
            predictions: Array de predicciones
            model_name: Nombre del modelo (para logging)
        """
        try:
            config = self.ensemble_config['prediction_limits']
            
            # Aplicar límites básicos
            clipped_predictions = np.clip(predictions, config['min_value'], config['max_value'])
            
            # Contar predicciones que fueron limitadas
            clipped_count = np.sum((predictions < config['min_value']) | (predictions > config['max_value']))
            if clipped_count > 0:
                logger.debug(f"Modelo {model_name}: {clipped_count} predicciones limitadas")
            
            # Validar que no hay NaN/Inf
            if np.any(np.isnan(clipped_predictions)) or np.any(np.isinf(clipped_predictions)):
                logger.warning(f"Modelo {model_name}: NaN/Inf detectado, aplicando corrección")
                clipped_predictions = np.nan_to_num(clipped_predictions, 
                                                   nan=220.0, posinf=config['max_value'], neginf=config['min_value'])
            
            return clipped_predictions
            
        except Exception as e:
            logger.error(f"Error aplicando límites a predicciones de {model_name}: {e}")
            # Fallback conservador
            return np.full_like(predictions, 220.0)
    
    def _calculate_ensemble_prediction(self, base_predictions: Dict[str, np.ndarray], 
                                     method: str = 'weighted') -> np.ndarray:
        """
        FUNCIÓN UTILITARIA CENTRALIZADA OPTIMIZADA: Calcula predicción del ensemble
        Optimizada para usar operaciones vectorizadas de NumPy
        
        Args:
            base_predictions: Diccionario con predicciones de cada modelo
            method: 'weighted', 'best_models_only', 'robust_median'
        """
        try:
            config = self.ensemble_config
            model_weights = config['model_weights']
            
            # OPTIMIZACIÓN: Convertir a arrays NumPy una sola vez
            pred_values = []
            pred_weights = []
            pred_names = []
            
            for model_name, predictions in base_predictions.items():
                if model_name in model_weights:
                    # Asegurar que predictions es un array NumPy
                    if not isinstance(predictions, np.ndarray):
                        predictions = np.array([predictions])
                    
                    pred_values.append(predictions)
                    pred_weights.append(model_weights[model_name])
                    pred_names.append(model_name)
            
            if not pred_values:
                logger.warning("No hay predicciones válidas para el ensemble")
                return np.array([220.0])
            
            # OPTIMIZACIÓN: Operaciones vectorizadas con NumPy
            pred_array = np.vstack(pred_values)  # Shape: (n_models, n_predictions)
            weight_array = np.array(pred_weights).reshape(-1, 1)  # Shape: (n_models, 1)
            
            if method == 'weighted':
                # OPTIMIZACIÓN: Validar predicciones usando operaciones vectorizadas
                pred_means = np.mean(pred_array, axis=1)
                min_val, max_val = config['prediction_limits']['min_value'], config['prediction_limits']['max_value']
                
                # Máscara para predicciones válidas
                valid_mask = (pred_means >= min_val) & (pred_means <= max_val)
                
                if np.any(valid_mask):
                    # Usar solo predicciones válidas
                    valid_preds = pred_array[valid_mask]
                    valid_weights = weight_array[valid_mask]
                    
                    # Normalizar pesos
                    valid_weights = valid_weights / np.sum(valid_weights)
                    
                    # Promedio ponderado vectorizado
                    ensemble_pred = np.sum(valid_preds * valid_weights, axis=0)
                else:
                    # Fallback: promedio simple con pesos reducidos
                    reduced_weights = weight_array * 0.3
                    reduced_weights = reduced_weights / np.sum(reduced_weights)
                    ensemble_pred = np.sum(pred_array * reduced_weights, axis=0)
                    logger.debug("Usando pesos reducidos por predicciones extremas")
            
            elif method == 'best_models_only':
                # Solo usar los mejores modelos
                best_models = ['extra_trees_primary', 'gradient_boost_primary', 'random_forest_primary']
                best_indices = [i for i, name in enumerate(pred_names) if name in best_models]
                
                if best_indices:
                    best_preds = pred_array[best_indices]
                    ensemble_pred = np.mean(best_preds, axis=0)
                else:
                    # Fallback a método weighted
                    return self._calculate_ensemble_prediction(base_predictions, 'weighted')
            
            elif method == 'robust_median':
                # Mediana robusta para reducir impacto de outliers
                ensemble_pred = np.median(pred_array, axis=0)
            
            else:
                raise ValueError(f"Método de ensemble desconocido: {method}")
            
            
            # Aplicar límites finales
            return self._apply_prediction_limits_and_validation(ensemble_pred, f"ensemble_{method}")
            
        except Exception as e:
            logger.error(f"Error calculando ensemble con método {method}: {e}")
            # Fallback ultra-conservador
            n_predictions = len(next(iter(base_predictions.values()))) if base_predictions else 1
            return np.full(n_predictions, 220.0, dtype=np.float32)
    
    def _calculate_prediction_confidence_unified(self, base_predictions: Dict[str, np.ndarray],
                                               additional_predictions: List[float] = None) -> float:
        """
        FUNCIÓN UTILITARIA CENTRALIZADA: Calcula confianza de predicción
        
        Args:
            base_predictions: Predicciones de modelos base
            additional_predictions: Predicciones adicionales (matemática, neural, etc.)
        """
        try:
            config = self.ensemble_config['confidence_thresholds']
            
            # Recopilar todas las predicciones válidas
            all_predictions = []
            
            # Agregar predicciones de modelos base
            for name, preds in base_predictions.items():
                if len(preds) > 0:
                    pred_mean = np.mean(preds)
                    if 180 <= pred_mean <= 280:  # Rango amplio para validación
                        all_predictions.append(pred_mean)
            
            # Agregar predicciones adicionales
            if additional_predictions:
                for pred in additional_predictions:
                    if 180 <= pred <= 280:
                        all_predictions.append(pred)
            
            if len(all_predictions) < 2:
                return config['min_confidence']
            
            # Calcular métricas de consistencia
            pred_std = np.std(all_predictions)
            pred_mean = np.mean(all_predictions)
            
            # Confianza basada en consistencia (menor std = mayor confianza)
            consistency_score = max(0, 100 - (pred_std * 8))
            
            # Confianza basada en rango típico NBA
            range_score = 100
            if not (210 <= pred_mean <= 240):
                range_score = max(70, 100 - abs(pred_mean - 225) * 2)
            
            # Combinar scores
            final_confidence = (
                consistency_score * config['consistency_weight'] + 
                range_score * config['performance_weight']
            )
            
            # Aplicar límites configurados
            return np.clip(final_confidence, config['min_confidence'], config['max_confidence'])
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return config['min_confidence']
    
    def _initialize_base_models(self, conservative_level: str = 'moderate') -> Dict:
        """
        MODELOS BASE UNIFICADOS - OPTIMIZADO PARA PRODUCCIÓN
        Solo entrena los modelos que efectivamente se usan en el ensemble final
        
        Args:
            conservative_level: 'moderate', 'ultra_conservative', 'aggressive'
        """
        
        # Solo entrenar modelos que tienen peso > 0 en la configuración del ensemble
        active_models = {name: weight for name, weight in self.ensemble_config['model_weights'].items() if weight > 0}
        
        logger.info(f"Entrenando solo {len(active_models)} modelos activos en lugar de todos los modelos")
        logger.info(f"Modelos activos: {list(active_models.keys())}")
        
        # Configuraciones por nivel
        configs = {
            'moderate': {
                'rf_n_estimators': 500, 'rf_max_depth': 12, 'rf_min_samples_split': 8,
                'et_n_estimators': 400, 'et_max_depth': 10, 'et_min_samples_split': 10,
                'gb_n_estimators': 300, 'gb_learning_rate': 0.05, 'gb_max_depth': 6,
                'ridge_alpha': 50.0, 'elastic_alpha': 10.0,
                'xgb_n_estimators': 200, 'xgb_max_depth': 5, 'xgb_learning_rate': 0.03,
                'lgb_n_estimators': 150, 'lgb_max_depth': 4, 'lgb_learning_rate': 0.02
            },
            'ultra_conservative': {
                'rf_n_estimators': 200, 'rf_max_depth': 8, 'rf_min_samples_split': 20,
                'et_n_estimators': 150, 'et_max_depth': 6, 'et_min_samples_split': 25,
                'gb_n_estimators': 100, 'gb_learning_rate': 0.01, 'gb_max_depth': 4,
                'ridge_alpha': 100.0, 'elastic_alpha': 20.0,
                'xgb_n_estimators': 80, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.01,
                'lgb_n_estimators': 60, 'lgb_max_depth': 3, 'lgb_learning_rate': 0.005
            },
            'aggressive': {
                'rf_n_estimators': 800, 'rf_max_depth': 15, 'rf_min_samples_split': 5,
                'et_n_estimators': 600, 'et_max_depth': 12, 'et_min_samples_split': 5,
                'gb_n_estimators': 500, 'gb_learning_rate': 0.1, 'gb_max_depth': 8,
                'ridge_alpha': 10.0, 'elastic_alpha': 5.0,
                'xgb_n_estimators': 400, 'xgb_max_depth': 8, 'xgb_learning_rate': 0.05,
                'lgb_n_estimators': 300, 'lgb_max_depth': 6, 'lgb_learning_rate': 0.05
            }
        }
        
        config = configs.get(conservative_level, configs['moderate'])
        
        # Diccionario completo de modelos disponibles
        all_models = {
            # RANDOM FOREST
            'random_forest_primary': lambda: RandomForestRegressor(
                n_estimators=config['rf_n_estimators'],
                max_depth=config['rf_max_depth'],
                min_samples_split=config['rf_min_samples_split'],
                min_samples_leaf=config['rf_min_samples_split'] // 2,
                max_features=0.7 if conservative_level == 'moderate' else 0.5,
                bootstrap=True,
                oob_score=True,
                min_impurity_decrease=0.01 if conservative_level != 'aggressive' else 0.001,
                n_jobs=-1,
                random_state=self.random_state
            ),
            
            # EXTRA TREES
            'extra_trees_primary': lambda: ExtraTreesRegressor(
                n_estimators=config['et_n_estimators'],
                max_depth=config['et_max_depth'],
                min_samples_split=config['et_min_samples_split'],
                min_samples_leaf=config['et_min_samples_split'] // 2,
                max_features=0.8 if conservative_level == 'moderate' else 0.4,
                bootstrap=False,
                min_impurity_decrease=0.02 if conservative_level == 'ultra_conservative' else 0.01,
                n_jobs=-1,
                random_state=self.random_state
            ),
            
            # GRADIENT BOOSTING
            'gradient_boost_primary': lambda: GradientBoostingRegressor(
                n_estimators=config['gb_n_estimators'],
                learning_rate=config['gb_learning_rate'],
                max_depth=config['gb_max_depth'],
                min_samples_split=config['gb_max_depth'] * 5,
                min_samples_leaf=config['gb_max_depth'] * 2,
                subsample=0.8 if conservative_level == 'moderate' else 0.6,
                max_features=0.6 if conservative_level == 'moderate' else 0.3,
                validation_fraction=0.2 if conservative_level == 'moderate' else 0.3,
                n_iter_no_change=20 if conservative_level == 'moderate' else 10,
                tol=1e-4 if conservative_level == 'moderate' else 1e-3,
                random_state=self.random_state
            ),
            
            # RIDGE REGRESSION
            'ridge_ultra_conservative': lambda: Ridge(
                alpha=config['ridge_alpha'],
                fit_intercept=True,
                copy_X=True,
                max_iter=2000 if conservative_level == 'moderate' else 3000,
                tol=1e-4 if conservative_level == 'moderate' else 1e-5,
                solver='auto',
                random_state=self.random_state
            ),
            
            # ELASTIC NET
            'elastic_net_ultra_conservative': lambda: ElasticNet(
                alpha=config['elastic_alpha'],
                l1_ratio=0.5 if conservative_level == 'moderate' else 0.7,
                fit_intercept=True,
                precompute=False,
                max_iter=2000 if conservative_level == 'moderate' else 3000,
                copy_X=True,
                tol=1e-4 if conservative_level == 'moderate' else 1e-5,
                warm_start=False,
                positive=False,
                random_state=self.random_state,
                selection='cyclic'
            ),
            
            # XGBOOST
            'xgboost_primary': lambda: XGBRegressor(
                n_estimators=config['xgb_n_estimators'],
                max_depth=config['xgb_max_depth'],
                learning_rate=config['xgb_learning_rate'],
                subsample=0.8 if conservative_level == 'moderate' else 0.6,
                colsample_bytree=0.7 if conservative_level == 'moderate' else 0.5,
                colsample_bylevel=0.8 if conservative_level == 'moderate' else 0.6,
                reg_alpha=5.0 if conservative_level == 'moderate' else 20.0,
                reg_lambda=10.0 if conservative_level == 'moderate' else 30.0,
                min_child_weight=10 if conservative_level == 'moderate' else 20,
                gamma=1.0 if conservative_level == 'moderate' else 5.0,
                objective='reg:squarederror',
                eval_metric='mae',
                verbosity=0,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # LIGHTGBM
            'lightgbm_primary': lambda: LGBMRegressor(
                n_estimators=config['lgb_n_estimators'],
                max_depth=config['lgb_max_depth'],
                learning_rate=config['lgb_learning_rate'],
                num_leaves=15 if conservative_level == 'moderate' else 8,
                min_child_samples=25 if conservative_level == 'moderate' else 50,
                min_child_weight=0.01,
                subsample=0.8 if conservative_level == 'moderate' else 0.6,
                colsample_bytree=0.7 if conservative_level == 'moderate' else 0.5,
                reg_alpha=8.0 if conservative_level == 'moderate' else 25.0,
                reg_lambda=12.0 if conservative_level == 'moderate' else 35.0,
                min_split_gain=0.5 if conservative_level == 'moderate' else 1.0,
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                importance_type='gain',
                verbosity=-1,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Solo instanciar y retornar los modelos que se usan en el ensemble
        active_models_dict = {}
        for model_name in active_models.keys():
            if model_name in all_models:
                active_models_dict[model_name] = all_models[model_name]()
                logger.info(f"Modelo {model_name} inicializado (peso: {active_models[model_name]:.2f})")
            else:
                logger.warning(f"Modelo {model_name} no encontrado en definiciones")
        
        logger.info(f"Optimizacion: {len(active_models_dict)} modelos vs {len(all_models)} totales")
        return active_models_dict
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimización bayesiana con regularización agresiva para evitar overfitting"""
        
        def objective(trial):
            # Seleccionar modelo a optimizar
            model_type = trial.suggest_categorical('model_type', ['xgboost', 'lightgbm', 'catboost', 'random_forest'])
            
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Reducido
                    'max_depth': trial.suggest_int('max_depth', 3, 6),           # Más conservador
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),  # Más lento
                    'subsample': trial.suggest_float('subsample', 0.5, 0.8),     # Más agresivo
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
                    'reg_alpha': trial.suggest_float('reg_alpha', 5.0, 50.0),    # Más regularización
                    'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 50.0),  # Más regularización
                    'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),  # Más conservador
                    'gamma': trial.suggest_float('gamma', 1.0, 10.0),            # Más restrictivo
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Reducido
                    'max_depth': trial.suggest_int('max_depth', 3, 6),           # Más conservador
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                    'reg_alpha': trial.suggest_float('reg_alpha', 5.0, 50.0),    # Más regularización
                    'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 50.0),  # Más regularización
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),  # Más conservador
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.5, 5.0),     # Más restrictivo
                    'num_leaves': trial.suggest_int('num_leaves', 10, 50),       # Reducido
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),     # Reducido
                    'depth': trial.suggest_int('depth', 3, 6),                  # Más conservador
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5.0, 50.0),  # Más regularización
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # Más conservador
                    'random_strength': trial.suggest_float('random_strength', 1.0, 5.0),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),
                    'random_state': self.random_state,
                    'verbose': False
                }
                model = CatBoostRegressor(**params)
                
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),   # Reducido
                    'max_depth': trial.suggest_int('max_depth', 3, 10),          # Más conservador
                    'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),  # Más conservador
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),     # Más conservador
                    'max_features': trial.suggest_categorical('max_features', [0.3, 0.5, 0.7]),  # Limitado
                    'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.01, 0.1),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=3)  # Reducido para velocidad
            
            # Asegurar que X_train tenga nombres de features para LightGBM
            if hasattr(X_train, 'columns'):
                X_train_cv = X_train
            else:
                X_train_cv = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
            
            scores = cross_val_score(model, X_train_cv, y_train, cv=tscv, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            
            return -scores.mean()
        
        # Optimización bayesiana con menos trials pero más enfocada
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Silenciar logs de Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        n_trials = 50  # Reducido de 100 para enfocarse en calidad
        logger.info("Iniciando optimización bayesiana con regularización agresiva...")
        
        # Callback para barra de progreso
        progress_callback = OptimizationProgressCallback(n_trials, "Optimizando con regularización")
        
        study.optimize(objective, n_trials=n_trials, timeout=1800, callbacks=[progress_callback])  # Timeout reducido
        
        print()  # Nueva línea después de la barra
        logger.info(f"Optimización completada - Mejor MAE: {study.best_value:.4f}")
        logger.info(f"Mejor modelo: {study.best_params.get('model_type', 'N/A')}")
        
        return study.best_params
    
    def _create_neural_network(self, input_size: int, hidden_sizes: List[int] = None, dropout_rate: float = 0.7) -> AdvancedNeuralNetwork:
        """Crea y configura la red neuronal con arquitectura ultra-conservadora"""
        if hidden_sizes is None:
            hidden_sizes = [32, 16]  # Arquitectura muy conservadora
        model = AdvancedNeuralNetwork(input_size, hidden_sizes, dropout_rate).to(self.device)
        return model
    
    def _optimize_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Optimiza la arquitectura de la red neuronal con regularización agresiva"""
        
        def nn_objective(trial):
            # Arquitectura de la red más conservadora
            n_layers = trial.suggest_int('n_layers', 2, 3)  # Reducido de 5
            hidden_sizes = []
            
            for i in range(n_layers):
                # Tamaños más pequeños para evitar overfitting
                size = trial.suggest_int(f'layer_{i}_size', 32, 256, step=32)  # Reducido de 1024
                hidden_sizes.append(size)
            
            # Hiperparámetros de entrenamiento más conservadores
            dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)  # Más dropout
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)  # Más lento
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Batches más pequeños
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)  # Regularización L2
            
            # Crear y entrenar modelo
            model = self._create_neural_network(X_train.shape[1], hidden_sizes, dropout_rate)
            
            # Entrenamiento rápido para optimización
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Optimizador con weight decay más agresivo
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            
            # Early stopping para optimización
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 5  # Paciencia reducida para optimización
            
            model.train()
            for epoch in range(30):  # Menos épocas para optimización
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Evaluación cada 5 épocas
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        break
                    
                    model.train()
            
            return best_val_loss
        
        # Optimización de la red neuronal con menos trials
        nn_study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
        )
        
        n_trials_nn = 15  # Reducido de 20
        logger.info("Optimizando arquitectura de red neuronal con regularización...")
        
        # Callback para barra de progreso de red neuronal
        nn_progress_callback = OptimizationProgressCallback(n_trials_nn, "Optimizando red neuronal")
        
        nn_study.optimize(nn_objective, n_trials=n_trials_nn, timeout=300, callbacks=[nn_progress_callback])  # Timeout reducido
        
        best_params = nn_study.best_params
        
        # Construir configuración óptima
        n_layers = best_params['n_layers']
        hidden_sizes = [best_params[f'layer_{i}_size'] for i in range(n_layers)]
        
        config = {
            'hidden_sizes': hidden_sizes,
            'dropout_rate': best_params['dropout_rate'],
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size'],
            'weight_decay': best_params.get('weight_decay', 1e-3)
        }
        
        logger.info(f"Mejor configuración de red neuronal: {config}")
        return config
    
    def _train_neural_network_unified(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray,
                                     model: AdvancedNeuralNetwork = None,
                                     regularization_level: str = 'moderate',
                                     hidden_sizes: List[int] = None, 
                                     dropout_rate: float = None,
                                     learning_rate: float = None, 
                                     batch_size: int = None,
                                     weight_decay: float = None) -> AdvancedNeuralNetwork:
        """
        ENTRENAMIENTO UNIFICADO DE REDES NEURONALES - OPTIMIZADO CON VALIDACIÓN ROBUSTA
        
        Args:
            regularization_level: 'light', 'moderate', 'aggressive', 'ultra_aggressive'
            model: Modelo preexistente o None para crear uno nuevo
            Otros parámetros: Si son None, se usan valores por defecto según regularization_level
        """
        
        # VALIDACIÓN ROBUSTA DE DATOS ANTES DEL ENTRENAMIENTO
        logger.info("Validando datos para entrenamiento de red neuronal...")
        try:
            X_train_clean, y_train_clean, train_mask = DataValidator.clean_and_validate_data(
                X_train, y_train, "neural_network_train"
            )
            X_val_clean, y_val_clean, val_mask = DataValidator.clean_and_validate_data(
                X_val, y_val, "neural_network_val"
            )
            
            logger.info(f"Datos limpiados - Entrenamiento: {len(X_train_clean)}/{len(X_train)}, "
                       f"Validación: {len(X_val_clean)}/{len(X_val)}")
            
        except Exception as e:
            logger.error(f"Error en validación de datos: {e}")
            raise ValueError(f"Datos no válidos para entrenamiento de red neuronal: {e}")
        
        # Configuraciones por nivel de regularización
        configs = {
            'light': {
                'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 64,
                'weight_decay': 1e-4, 'patience': 20, 'max_epochs': 500,
                'grad_clip': 1.0, 'l1_reg': 0.0, 'l2_reg': 0.0,
                'early_stopping_params': {'patience': 20, 'min_delta': 1e-5, 'overfitting_threshold': 0.15}
            },
            'moderate': {
                'dropout_rate': 0.5, 'learning_rate': 0.001, 'batch_size': 32,
                'weight_decay': 1e-3, 'patience': 15, 'max_epochs': 300,
                'grad_clip': 1.0, 'l1_reg': 0.001, 'l2_reg': 0.001,
                'early_stopping_params': {'patience': 15, 'min_delta': 1e-5, 'overfitting_threshold': 0.12}
            },
            'aggressive': {
                'dropout_rate': 0.7, 'learning_rate': 0.0005, 'batch_size': 16,
                'weight_decay': 1e-2, 'patience': 10, 'max_epochs': 200,
                'grad_clip': 0.5, 'l1_reg': 0.01, 'l2_reg': 0.01,
                'early_stopping_params': {'patience': 10, 'min_delta': 1e-4, 'overfitting_threshold': 0.10}
            },
            'ultra_aggressive': {
                'dropout_rate': 0.8, 'learning_rate': 0.00005, 'batch_size': 8,
                'weight_decay': 0.3, 'patience': 5, 'max_epochs': 50,
                'grad_clip': 0.1, 'l1_reg': 0.001, 'l2_reg': 0.001,
                'early_stopping_params': {'patience': 5, 'min_delta': 1e-4, 'overfitting_threshold': 0.08}
            }
        }
        
        config = configs.get(regularization_level, configs['moderate'])
        
        # Usar parámetros proporcionados o valores por defecto
        dropout_rate = dropout_rate if dropout_rate is not None else config['dropout_rate']
        learning_rate = learning_rate if learning_rate is not None else config['learning_rate']
        batch_size = batch_size if batch_size is not None else config['batch_size']
        weight_decay = weight_decay if weight_decay is not None else config['weight_decay']
        
        # Crear modelo si no se proporciona
        if model is None:
            if hidden_sizes is None:
                hidden_sizes = [32, 16] if regularization_level in ['aggressive', 'ultra_aggressive'] else [64, 32]
            model = self._create_neural_network(X_train_clean.shape[1], hidden_sizes, dropout_rate)
        
        # OPTIMIZACIÓN: Convertir a tensores con tipos optimizados y pin_memory
        # Usar float32 en lugar de float64 para mejor rendimiento en GPU
        X_train_tensor = torch.FloatTensor(X_train_clean.astype(np.float32)).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_clean.astype(np.float32).reshape(-1, 1)).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_clean.astype(np.float32)).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_clean.astype(np.float32).reshape(-1, 1)).to(self.device)
        
        # VALIDACIÓN FINAL DE TENSORES
        assert not torch.any(torch.isnan(X_train_tensor)), "NaNs detectados en X_train_tensor"
        assert not torch.any(torch.isnan(y_train_tensor)), "NaNs detectados en y_train_tensor"
        assert not torch.any(torch.isnan(X_val_tensor)), "NaNs detectados en X_val_tensor"
        assert not torch.any(torch.isnan(y_val_tensor)), "NaNs detectados en y_val_tensor"
        
        # OPTIMIZACIÓN: Crear datasets con pin_memory para transferencia GPU más rápida
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        # pin_memory=True solo si usamos GPU y no estamos en el mismo dispositivo
        use_pin_memory = self.device.type == 'cuda' and not X_train_tensor.is_cuda
        num_workers = 2 if self.device.type == 'cuda' else 0  # Workers para GPU
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=use_pin_memory,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,  # Mantener workers vivos
            drop_last=True if len(train_dataset) > batch_size else False  # Evitar batches pequeños
        )
        
        # Optimizador y scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Scheduler más o menos agresivo según el nivel
        patience_scheduler = config['patience'] // 2 if regularization_level == 'ultra_aggressive' else config['patience']
        factor = 0.3 if regularization_level in ['aggressive', 'ultra_aggressive'] else 0.5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience_scheduler, factor=factor, min_lr=1e-7
        )
        
        # EARLY STOPPING AVANZADO CON CHECKPOINT
        early_stopping = EarlyStoppingWithCheckpoint(
            patience=config['early_stopping_params']['patience'],
            min_delta=config['early_stopping_params']['min_delta'],
            restore_best_weights=True,
            mode='min',
            monitor_overfitting=True,
            overfitting_threshold=config['early_stopping_params']['overfitting_threshold']
        )
        
        # Función de pérdida
        criterion = nn.MSELoss()
        
        # OPTIMIZACIÓN: Pre-calcular regularización si es necesaria
        use_l1_l2_reg = config['l1_reg'] > 0 or config['l2_reg'] > 0
        l1_reg_weight = config['l1_reg']
        l2_reg_weight = config['l2_reg']
        max_epochs = config['max_epochs']
        
        model.train()
        
        # Barra de progreso
        desc = f"Entrenando NN ({regularization_level})"
        with tqdm(total=max_epochs, desc=desc, 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Val Loss: {postfix}') as pbar:
            
            for epoch in range(max_epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                # OPTIMIZACIÓN: Usar context manager para autocast si hay GPU
                if self.device.type == 'cuda':
                    scaler = torch.cuda.amp.GradScaler()  # Para mixed precision
                    
                for batch_X, batch_y in train_loader:
                    # VALIDACIÓN DE BATCH: Asegurar que no hay NaNs en el batch
                    if torch.any(torch.isnan(batch_X)) or torch.any(torch.isnan(batch_y)):
                        logger.warning(f"NaNs detectados en batch de época {epoch}, saltando...")
                        continue
                    
                    optimizer.zero_grad()
                    
                    # OPTIMIZACIÓN: Mixed precision training en GPU
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_X)
                            
                            # Validar outputs antes de calcular pérdida
                            if torch.any(torch.isnan(outputs)):
                                logger.warning(f"NaNs en outputs de época {epoch}, saltando batch...")
                                continue
                            
                            loss = criterion(outputs, batch_y)
                            
                            # Regularización adicional según el nivel
                            if use_l1_l2_reg:
                                l1_reg = torch.tensor(0., device=self.device, dtype=torch.float32)
                                l2_reg = torch.tensor(0., device=self.device, dtype=torch.float32)
                                for param in model.parameters():
                                    l1_reg += torch.norm(param, 1)
                                    l2_reg += torch.norm(param, 2)
                                
                                loss += l1_reg_weight * l1_reg + l2_reg_weight * l2_reg
                        
                        # Validar pérdida antes de backward pass
                        if torch.isnan(loss):
                            logger.warning(f"NaN loss en época {epoch}, saltando batch...")
                            continue
                        
                        # Backward pass con scaling
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # CPU training (sin mixed precision)
                        outputs = model(batch_X)
                        
                        # Validar outputs
                        if torch.any(torch.isnan(outputs)):
                            logger.warning(f"NaNs en outputs de época {epoch}, saltando batch...")
                            continue
                        
                        loss = criterion(outputs, batch_y)
                        
                        # Regularización adicional según el nivel
                        if use_l1_l2_reg:
                            l1_reg = torch.tensor(0., device=self.device)
                            l2_reg = torch.tensor(0., device=self.device)
                            for param in model.parameters():
                                l1_reg += torch.norm(param, 1)
                                l2_reg += torch.norm(param, 2)
                            
                            loss += l1_reg_weight * l1_reg + l2_reg_weight * l2_reg
                        
                        # Validar pérdida
                        if torch.isnan(loss):
                            logger.warning(f"NaN loss en época {epoch}, saltando batch...")
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
                        optimizer.step()
                
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Validación optimizada
                model.eval()
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            val_outputs = model(X_val_tensor)
                            val_loss = criterion(val_outputs, y_val_tensor).item()
                    else:
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Validar val_loss
                if np.isnan(val_loss) or np.isinf(val_loss):
                    logger.error(f"Val loss inválida en época {epoch}: {val_loss}")
                    break
                
                scheduler.step(val_loss)
                
                # Calcular train_loss promedio
                avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
                
                # EARLY STOPPING AVANZADO
                should_stop = early_stopping(val_loss, avg_train_loss, model, epoch)
                
                # Actualizar barra de progreso
                pbar.set_postfix_str(f"{val_loss:.4f}")
                pbar.update(1)
                
                if should_stop:
                    pbar.set_description(f"Entrenando NN ({regularization_level}) - {early_stopping.stop_reason}")
                    break
                
                model.train()
        
        # RESTAURAR EL MEJOR MODELO AUTOMÁTICAMENTE
        early_stopping.restore_best_model(model)
        
        # Obtener resumen del entrenamiento
        summary = early_stopping.get_summary()
        logger.info(f"Entrenamiento completado: {summary}")
        
        # OPTIMIZACIÓN: Limpiar cache de GPU si se usó
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return model
    
    def train(self, teams_data: pd.DataFrame, target_col: str = 'total_points') -> Dict:
        """
        Entrena el modelo ensemble con VALIDACIÓN CRUZADA TEMPORAL MEJORADA
        Enfoque en estabilidad CV y reducción del gap train/validation
        
        Args:
            teams_data: DataFrame con datos de equipos
            target_col: Columna objetivo (se creará si no existe)
            
        Returns:
            Métricas de rendimiento
        """
        logger.info("Iniciando entrenamiento con VALIDACIÓN CRUZADA TEMPORAL MEJORADA...")
        
        # VALIDACIÓN ROBUSTA DE DATOS DE ENTRADA
        logger.info("Aplicando validación robusta a datos de entrada...")
        teams_data_safe = DataValidator.safe_feature_engineering(teams_data)
        
        # Crear features INDEPENDIENTES
        logger.info("Generando features independientes (sin data leakage)...")
        df_features = self.feature_engine.create_features(teams_data_safe)
        
        # APLICAR FILTRO FINAL DE CORRELACIÓN MÁS AGRESIVO (85% en lugar de 95%)
        logger.info("Aplicando filtro final de correlación >85% para estabilidad...")
        df_features = self.feature_engine.apply_final_correlation_filter(df_features, correlation_threshold=0.85)
        
        # Usar total_score si existe, sino crear target_col
        if 'total_score' in df_features.columns:
            target_col = 'total_score'
            logger.info("Usando 'total_score' como columna objetivo")
        elif target_col not in df_features.columns:
            df_features[target_col] = df_features['PTS'] + df_features['PTS_Opp']
            logger.info(f"Creando '{target_col}' como PTS + PTS_Opp")
        
        # Preparar datos para entrenamiento
        feature_cols = self.feature_engine.feature_columns
        X = df_features[feature_cols] 
        y = df_features[target_col].values
        
        # VALIDACIÓN ROBUSTA DE DATOS PREPARADOS
        logger.info("Validando datos preparados para entrenamiento...")
        X_clean, y_clean, valid_mask = DataValidator.clean_and_validate_data(
            X.values, y, "ensemble_training"
        )
        
        logger.info(f"Datos de entrenamiento validados: {len(X_clean)} muestras válidas de {len(X)} originales")
        logger.info(f"Features independientes: {len(feature_cols)} features")
        
        # VALIDACIÓN CRUZADA TEMPORAL MEJORADA (7 folds para mayor robustez)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=7)  # Aumentado de 5 a 7 para mayor estabilidad
        
        # Ordenar por fecha para mantener cronología (solo datos válidos)
        df_features_valid = df_features[valid_mask]
        if 'Date' in df_features_valid.columns:
            df_sorted = df_features_valid.sort_values(by='Date').reset_index(drop=True)
        else:
            df_sorted = df_features_valid.reset_index(drop=True)
        
        # Recalcular X e y con orden cronológico
        X_final = df_sorted[feature_cols].values  # Convertir a NumPy array directamente
        y_final = df_sorted[target_col].values
        
        # VALIDACIÓN FINAL ANTES DE CV
        X_final, y_final, final_mask = DataValidator.clean_and_validate_data(
            X_final, y_final, "final_cv_data"
        )
        
        if 'Date' in df_sorted.columns:
            dates = df_sorted['Date'].values[final_mask]
            logger.info(f"Rango temporal: {dates[0]} a {dates[-1]}")
        else:
            logger.info(f"Datos preparados para CV: {len(X_final)} muestras")
        
        # Inicializar modelos base con REGULARIZACIÓN MÁS AGRESIVA
        self.base_models = self._initialize_base_models(conservative_level='ultra_conservative')
        
        # ENTRENAMIENTO CON VALIDACIÓN CRUZADA TEMPORAL MEJORADA
        cv_scores = []
        fold_predictions = []
        
        logger.info("Iniciando validación cruzada temporal (7 folds)...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_final)):
            logger.info(f"\n--- FOLD {fold + 1}/7 ---")
            
            X_train_fold, X_val_fold = X_final[train_idx], X_final[val_idx]
            y_train_fold, y_val_fold = y_final[train_idx], y_final[val_idx]
            
            logger.info(f"Entrenamiento: {dates[train_idx[0]]} a {dates[train_idx[-1]]}")
            logger.info(f"Validación: {dates[val_idx[0]]} a {dates[val_idx[-1]]}")
            
            # Escalado de features con validación más estricta
            scaler_standard = StandardScaler()
            scaler_robust = StandardScaler()
            
            X_train_scaled = scaler_standard.fit_transform(X_train_fold)
            X_val_scaled = scaler_standard.transform(X_val_fold)
            
            X_train_robust = scaler_robust.fit_transform(X_train_fold)
            X_val_robust = scaler_robust.transform(X_val_fold)
            
            # Entrenar modelos base para este fold con REGULARIZACIÓN EXTREMA
            fold_models = self._initialize_base_models(conservative_level='ultra_conservative')
            base_predictions_val = np.zeros((len(X_val_fold), len(fold_models)))
            
            for i, (name, model) in enumerate(fold_models.items()):
                # Seleccionar datos escalados apropiados
                if name in ['elastic_net_ultra_conservative', 'ridge_ultra_conservative']:
                    X_tr, X_v = X_train_scaled, X_val_scaled
                elif name in ['xgboost_primary', 'lightgbm_primary', 'catboost_primary']:
                    X_tr = pd.DataFrame(X_train_robust, columns=feature_cols)
                    X_v = pd.DataFrame(X_val_robust, columns=feature_cols)
                else:
                    X_tr, X_v = X_train_fold, X_val_fold  # Ya son numpy arrays
                
                # VALIDACIÓN ANTES DE ENTRENAR CADA MODELO
                if hasattr(X_tr, 'values'):
                    X_tr_array = X_tr.values
                else:
                    X_tr_array = X_tr
                
                if np.any(np.isnan(X_tr_array)) or np.any(np.isnan(y_train_fold)):
                    logger.error(f"NaNs detectados antes de entrenar {name} en fold {fold+1}")
                    base_predictions_val[:, i] = 220.0  # Fallback
                    continue
                
                # Entrenar modelo con early stopping más agresivo
                try:
                    if 'xgboost' in name:
                        # Corregir para versiones nuevas de XGBoost
                        model.fit(X_tr, y_train_fold, 
                                eval_set=[(X_v, y_val_fold)], 
                                verbose=False)
                    elif 'lightgbm' in name:
                        model.fit(X_tr, y_train_fold, 
                                eval_set=[(X_v, y_val_fold)],
                                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
                    elif 'catboost' in name:
                        model.fit(X_tr, y_train_fold, 
                                eval_set=(X_v, y_val_fold), 
                                early_stopping_rounds=10,
                                verbose=False)
                    else:
                        model.fit(X_tr, y_train_fold)
                    
                    # Predicciones con validación estricta
                    pred_val = model.predict(X_v)
                    pred_val = np.clip(pred_val, 190, 240)  # Límites más estrictos
                    base_predictions_val[:, i] = pred_val
                    
                except Exception as e:
                    logger.warning(f"Error en modelo {name} fold {fold+1}: {e}")
                    base_predictions_val[:, i] = 220.0  # Fallback conservador
            
            # Ensemble simple para este fold (promedio ponderado conservador)
            # Usar solo los 3 mejores modelos para reducir varianza
            model_weights_cv = {
                'random_forest_primary': 0.4,
                'extra_trees_primary': 0.35,
                'ridge_ultra_conservative': 0.25
            }
            
            ensemble_pred_val = np.zeros(len(X_val_fold))
            total_weight = 0
            
            for i, (name, model) in enumerate(fold_models.items()):
                if name in model_weights_cv:
                    weight = model_weights_cv[name]
                    pred = base_predictions_val[:, i]
                    # Validar predicciones antes de combinar
                    if np.all((pred >= 180) & (pred <= 280)):
                        ensemble_pred_val += pred * weight
                        total_weight += weight
            
            if total_weight > 0:
                ensemble_pred_val /= total_weight
            else:
                ensemble_pred_val = np.mean(base_predictions_val, axis=1)
            
            # Aplicar límites finales
            ensemble_pred_val = np.clip(ensemble_pred_val, 195, 265)
            
            # Calcular métricas del fold con tolerancia más estricta
            fold_mae = mean_absolute_error(y_val_fold, ensemble_pred_val)
            fold_acc = self._calculate_accuracy(y_val_fold, ensemble_pred_val, tolerance=2.5)  # Más estricto
            fold_r2 = r2_score(y_val_fold, ensemble_pred_val)
            
            cv_scores.append({
                'fold': fold + 1,
                'mae': fold_mae,
                'accuracy': fold_acc,
                'r2': fold_r2,
                'n_train': len(X_train_fold),
                'n_val': len(X_val_fold)
            })
            
            fold_predictions.extend(list(zip(y_val_fold, ensemble_pred_val)))
            
            logger.info(f"Fold {fold + 1} - MAE: {fold_mae:.3f}, Acc: {fold_acc:.1f}%, R²: {fold_r2:.3f}")
        
        # ENTRENAMIENTO FINAL CON TODOS LOS DATOS Y REGULARIZACIÓN EXTREMA
        logger.info("\nEntrenando modelo final con regularización extrema...")
        
        # División final 85-15 para validación más estricta
        split_idx = int(0.85 * len(X_final))  # Más datos para entrenamiento
        X_train_final, X_val_final = X_final[:split_idx], X_final[split_idx:]
        y_train_final, y_val_final = y_final[:split_idx], y_final[split_idx:]
        
        # Escalado final
        self.scalers['standard'].fit(X_train_final)
        self.scalers['robust'].fit(X_train_final)
        
        X_train_scaled_final = self.scalers['standard'].transform(X_train_final)
        X_val_scaled_final = self.scalers['standard'].transform(X_val_final)
        
        X_train_robust_final = self.scalers['robust'].transform(X_train_final)
        X_val_robust_final = self.scalers['robust'].transform(X_val_final)
        
        # Entrenar modelos base finales con REGULARIZACIÓN EXTREMA
        base_predictions_train_final = np.zeros((len(X_train_final), len(self.base_models)))
        base_predictions_val_final = np.zeros((len(X_val_final), len(self.base_models)))
        
        logger.info("Entrenando modelos base con regularización extrema...")
        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"Entrenando {name}...")
            
            # Seleccionar datos apropiados para cada modelo
            if 'lightgbm' in name or 'catboost' in name or 'xgboost' in name:
                X_tr = pd.DataFrame(X_train_robust_final, columns=feature_cols)
                X_v = pd.DataFrame(X_val_robust_final, columns=feature_cols)
            elif 'ridge' in name or 'elastic_net' in name:
                X_tr, X_v = X_train_scaled_final, X_val_scaled_final
            else:
                X_tr, X_v = X_train_final, X_val_final  # Ya son numpy arrays
            
            # Entrenar modelo con configuración ultra-conservadora
            try:
                if 'lightgbm' in name:
                    model.fit(X_tr, y_train_final, 
                             eval_set=[(X_v, y_val_final)],
                             callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
                elif 'catboost' in name:
                    model.fit(X_tr, y_train_final, 
                             eval_set=(X_v, y_val_final), 
                             early_stopping_rounds=15,
                             verbose=False)
                elif 'xgboost' in name:
                    # Corregir para versiones nuevas de XGBoost
                    model.fit(X_tr, y_train_final, 
                             eval_set=[(X_v, y_val_final)], 
                             verbose=False)
                else:
                    model.fit(X_tr, y_train_final)
                
                # Predicciones con validación estricta
                pred_train = model.predict(X_tr)
                pred_val = model.predict(X_v)
                
                # Aplicar límites NBA estrictos
                pred_train = np.clip(pred_train, 185, 250)
                pred_val = np.clip(pred_val, 185, 250)
                
                base_predictions_train_final[:, i] = pred_train
                base_predictions_val_final[:, i] = pred_val
                
                # Métricas individuales del modelo
                train_mae = mean_absolute_error(y_train_final, pred_train)
                val_mae = mean_absolute_error(y_val_final, pred_val)
                val_acc = self._calculate_accuracy(y_val_final, pred_val, tolerance=2.5)
                
                logger.info(f"{name} - Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}, Val Acc: {val_acc:.1f}%")
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {e}")
                # Fallback con predicciones conservadoras
                base_predictions_train_final[:, i] = 220.0
                base_predictions_val_final[:, i] = 220.0
        
        # ENSEMBLE FINAL ULTRA-CONSERVADOR
        # Usar solo los mejores modelos con pesos conservadores
        final_model_weights = {
            'random_forest_primary': 0.35,
            'extra_trees_primary': 0.30,
            'ridge_ultra_conservative': 0.20,
            'gradient_boost_primary': 0.15
        }
        
        ensemble_pred_train = np.zeros(len(y_train_final))
        ensemble_pred_val = np.zeros(len(y_val_final))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if name in final_model_weights:
                weight = final_model_weights[name]
                pred_train = base_predictions_train_final[:, i]
                pred_val = base_predictions_val_final[:, i]
                
                # Validar predicciones antes de combinar
                if np.all((pred_train >= 180) & (pred_train <= 280)) and np.all((pred_val >= 180) & (pred_val <= 280)):
                    ensemble_pred_train += pred_train * weight
                    ensemble_pred_val += pred_val * weight
                    logger.info(f"Modelo {name}: peso = {weight:.3f}")
        
        # Aplicar límites finales conservadores
        ensemble_pred_train = np.clip(ensemble_pred_train, 200, 260)
        ensemble_pred_val = np.clip(ensemble_pred_val, 200, 260)
        
        # EVALUACIÓN ULTRA-ESTRICTA DE MODELOS
        model_performance = {}
        predictions = {}
        
        # Evaluar cada modelo con métricas estrictas
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                pred_val = base_predictions_val_final[:, i]
                
                # Calcular métricas ultra-estrictas
                val_mae = mean_absolute_error(y_val_final, pred_val)
                val_r2 = r2_score(y_val_final, pred_val)
                val_accuracy_strict = self._calculate_accuracy(y_val_final, pred_val, tolerance=2.0)  # ±2 puntos
                val_accuracy_normal = self._calculate_accuracy(y_val_final, pred_val, tolerance=3.0)  # ±3 puntos
                
                # Penalizar modelos con predicciones extremas
                extreme_penalty = np.mean((pred_val < 190) | (pred_val > 270)) * 20  # Penalización por extremos
                adjusted_accuracy = val_accuracy_normal - extreme_penalty
                
                # Guardar métricas y predicciones
                model_performance[name] = {
                    'mae': val_mae,
                    'r2': val_r2,
                    'accuracy': adjusted_accuracy,
                    'accuracy_strict': val_accuracy_strict,
                    'extreme_penalty': extreme_penalty
                }
                predictions[name] = pred_val
                
                logger.info(f"{name} - Val MAE: {val_mae:.3f}, Val R²: {val_r2:.3f}, Val Acc: {adjusted_accuracy:.1f}%")
                
            except Exception as e:
                logger.warning(f"Error evaluando modelo {name}: {e}")
                continue
        
        # CÁLCULO DE PESOS ULTRA-OPTIMIZADO PARA 97%
        model_weights = {}
        total_weight = 0
        
        # Filtrar solo modelos con rendimiento aceptable (>60% precisión ajustada)
        valid_models = {name: metrics for name, metrics in model_performance.items() 
                       if metrics['accuracy'] > 60.0 and metrics['mae'] < 8.0}
        
        if not valid_models:
            logger.warning("No hay modelos con rendimiento aceptable, usando todos con pesos iguales")
            valid_models = model_performance
        
        for name, metrics in valid_models.items():
            # Peso basado en precisión ajustada y MAE inverso
            accuracy_weight = (metrics['accuracy'] / 100) ** 2  # Cuadrático para enfatizar diferencias
            mae_weight = 1 / (metrics['mae'] + 1e-6)  # Inverso del MAE
            
            # Combinación de pesos con énfasis en precisión
            combined_weight = (accuracy_weight * 0.7) + (mae_weight * 0.3)
            
            model_weights[name] = combined_weight
            total_weight += combined_weight
        
        # Normalizar pesos para que sumen 1
        if total_weight > 0:
            for name in model_weights:
                model_weights[name] /= total_weight
        else:
            # Fallback: pesos iguales para modelos válidos
            num_models = len(valid_models) if valid_models else len(model_performance)
            for name in (valid_models if valid_models else model_performance):
                model_weights[name] = 1.0 / num_models
        
        # ENSEMBLE ULTRA-OPTIMIZADO CON VALIDACIÓN ESTRICTA
        ensemble_pred = np.zeros(len(y_val_final))
        
        for name, model in self.base_models.items():
            if name in predictions and name in model_weights:
                weight = model_weights[name]
                if weight > 0.01:  # Solo usar modelos con peso significativo
                    pred = predictions[name]
                    # Aplicar límites estrictos antes de combinar
                    pred_clipped = np.clip(pred, 195, 265)
                    ensemble_pred += pred_clipped * weight
                    logger.info(f"Modelo {name}: peso = {weight:.3f}")
        
        # Validación final del ensemble
        if np.sum(ensemble_pred) == 0 or np.any(np.isnan(ensemble_pred)):
            logger.warning("Ensemble inválido, usando promedio de mejores modelos")
            best_models = sorted(valid_models.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
            if best_models:
                best_predictions = [predictions[name] for name, _ in best_models if name in predictions]
                if best_predictions:
                    ensemble_pred = np.mean(best_predictions, axis=0)
                else:
                    ensemble_pred = np.full(len(y_val_final), 225.0)
            else:
                ensemble_pred = np.full(len(y_val_final), 225.0)
        
        # Aplicar límites finales ultra-estrictos
        ensemble_pred = np.clip(ensemble_pred, 200, 260)
        
        # Calcular métricas finales del entrenamiento
        # Calcular predicciones del ensemble para entrenamiento
        ensemble_pred_train = np.zeros(len(y_train_final))
        
        for name, model in self.base_models.items():
            if name in predictions and name in model_weights:
                weight = model_weights[name]
                if weight > 0:
                    # Seleccionar datos apropiados para cada modelo
                    if 'lightgbm' in name or 'catboost' in name or 'gradient_boost' in name or 'xgboost' in name:
                        X_tr = pd.DataFrame(X_train_robust_final, columns=feature_cols)
                        pred_train = model.predict(X_tr)
                    elif 'ridge' in name or 'elastic_net' in name:
                        pred_train = model.predict(X_train_scaled_final)
                    else:
                        pred_train = model.predict(X_train_final)
                    
                    ensemble_pred_train += pred_train * weight
        
        # Si no hay modelos con peso > 0, usar promedio simple
        if np.sum(ensemble_pred_train) == 0:
            # Calcular predicciones de entrenamiento para todos los modelos
            train_predictions = []
            for name, model in self.base_models.items():
                try:
                    if 'lightgbm' in name or 'catboost' in name or 'gradient_boost' in name or 'xgboost' in name:
                        X_tr = pd.DataFrame(X_train_robust_final, columns=feature_cols)
                        pred_train = model.predict(X_tr)
                    elif 'ridge' in name or 'elastic_net' in name:
                        pred_train = model.predict(X_train_scaled_final)
                    else:
                        pred_train = model.predict(X_train_final)
                    train_predictions.append(pred_train)
                except:
                    continue
            
            if train_predictions:
                ensemble_pred_train = np.mean(train_predictions, axis=0)
            else:
                ensemble_pred_train = np.full(len(y_train_final), 225.0)
        
        # Aplicar límites realistas NBA
        ensemble_pred_train = np.clip(ensemble_pred_train, 180, 280)
        
        final_pred_train = ensemble_pred_train
        final_pred_val = ensemble_pred
        
        # Definir métricas de rendimiento
        self.performance_metrics = {
            'cross_validation': {
                'mean_mae': np.mean([score['mae'] for score in cv_scores]),
                'std_mae': np.std([score['mae'] for score in cv_scores]),
                'mean_accuracy': np.mean([score['accuracy'] for score in cv_scores]),
                'std_accuracy': np.std([score['accuracy'] for score in cv_scores]),
                'mean_r2': np.mean([score['r2'] for score in cv_scores]),
                'std_r2': np.std([score['r2'] for score in cv_scores]),
                'fold_scores': cv_scores
            },
            'train': {
                'mae': mean_absolute_error(y_train_final, final_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train_final, final_pred_train)),
                'r2': r2_score(y_train_final, final_pred_train),
                'accuracy': self._calculate_accuracy(y_train_final, final_pred_train)
            },
            'validation': {
                'mae': mean_absolute_error(y_val_final, final_pred_val),
                'rmse': np.sqrt(mean_squared_error(y_val_final, final_pred_val)),
                'r2': r2_score(y_val_final, final_pred_val),
                'accuracy': self._calculate_accuracy(y_val_final, final_pred_val)
            }
        }
        
        self.is_trained = True
        
        # Análisis mejorado de resultados
        self._analyze_model_performance_unified(y_train_final, final_pred_train, y_val_final, ensemble_pred, 
                                          base_predictions_train_final, base_predictions_val_final, 
                                              final_pred_train, ensemble_pred, cv_scores,
                                              detailed=True, stability_focus=True)
        
        # GUARDADO ÚNICO DEL MODELO
        try:
            # Crear carpeta trained_models si no existe
            models_dir = "trained_models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Nombre fijo que siempre se sobrescribe
            filepath = os.path.join(models_dir, "total_points_teams.joblib")
            
            # Guardar modelo
            self.save_model(filepath)
            logger.info(f"✅ Modelo guardado en: {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error guardando modelo: {e}")
        
        return self.performance_metrics
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 3.0) -> float:
        """Calcula precisión con tolerancia de puntos (±3 puntos para mayor exigencia)"""
        return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
    
    def _analyze_model_performance_cv(self, y_train, pred_train, y_val, pred_val, 
                                     base_pred_train, base_pred_val, ensemble_train, ensemble_val, cv_scores):
        """Análisis completo del rendimiento del modelo OPTIMIZADO CON VALIDACIÓN CRUZADA"""
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETO DEL MODELO OPTIMIZADO CON VALIDACIÓN CRUZADA")
        print("="*80)
        
        # Métricas de validación cruzada
        cv_metrics = self.performance_metrics['cross_validation']
        train_metrics = self.performance_metrics['train']
        val_metrics = self.performance_metrics['validation']
        
        print(f"\n📊 VALIDACIÓN CRUZADA TEMPORAL (7 FOLDS):")
        print(f"{'Métrica':<15} {'Media':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        print("-" * 75)
        
        fold_maes = [score['mae'] for score in cv_scores]
        fold_accs = [score['accuracy'] for score in cv_scores]
        
        print(f"{'MAE':<15} {cv_metrics['mean_mae']:<15.3f} {cv_metrics['std_mae']:<15.3f} {min(fold_maes):<15.3f} {max(fold_maes):<15.3f}")
        print(f"{'Precisión (%)':<15} {cv_metrics['mean_accuracy']:<15.2f} {cv_metrics['std_accuracy']:<15.2f} {min(fold_accs):<15.2f} {max(fold_accs):<15.2f}")
        
        print(f"\n📈 DETALLES POR FOLD:")
        for i, score in enumerate(cv_scores):
            print(f"Fold {score['fold']}: MAE={score['mae']:.3f}, Acc={score['accuracy']:.1f}%, Train={score['n_train']}, Val={score['n_val']}")
        
        print(f"\n MÉTRICAS FINALES (Hold-out):")
        print(f"{'Métrica':<15} {'Entrenamiento':<15} {'Validación':<15} {'Diferencia':<15}")
        print("-" * 60)
        print(f"{'Precisión (%)':<15} {train_metrics['accuracy']:<15.2f} {val_metrics['accuracy']:<15.2f} {abs(train_metrics['accuracy'] - val_metrics['accuracy']):<15.2f}")
        print(f"{'MAE':<15} {train_metrics['mae']:<15.3f} {val_metrics['mae']:<15.3f} {abs(train_metrics['mae'] - val_metrics['mae']):<15.3f}")
        print(f"{'RMSE':<15} {train_metrics['rmse']:<15.3f} {val_metrics['rmse']:<15.3f} {abs(train_metrics['rmse'] - val_metrics['rmse']):<15.3f}")
        print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {val_metrics['r2']:<15.4f} {abs(train_metrics['r2'] - val_metrics['r2']):<15.4f}")
        
        # Análisis de overfitting mejorado
        mae_diff = abs(train_metrics['mae'] - val_metrics['mae'])
        r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
        cv_stability = cv_metrics['std_mae'] / cv_metrics['mean_mae'] if cv_metrics['mean_mae'] > 0 else 0
        acc_cv = cv_metrics['std_accuracy'] / cv_metrics['mean_accuracy'] if cv_metrics['mean_accuracy'] > 0 else 0
        r2_cv = cv_metrics['std_r2'] / cv_metrics['mean_r2'] if cv_metrics['mean_r2'] > 0 else 0
        
        print(f"\n🔍 ANÁLISIS DE ROBUSTEZ:")
        print(f"Estabilidad CV (std/mean): {cv_stability:.3f}")
        if cv_stability < 0.1:
            print("✅ Modelo muy estable en validación cruzada")
        elif cv_stability < 0.2:
            print("⚠️  Modelo moderadamente estable")
        else:
            print("❌ Modelo inestable - Alta variabilidad entre folds")
        
        if mae_diff < 1.0 and r2_diff < 0.05:
            print("✅ Sin overfitting significativo")
        elif mae_diff < 2.0 and r2_diff < 0.1:
            print("⚠️  Ligero overfitting - Aceptable")
        else:
            print("❌ Overfitting detectado")
        
        # Análisis de precisión por tolerancia
        print(f"\n🎯 PRECISIÓN POR TOLERANCIA (Validación Final):")
        for tolerance in [1, 2, 3, 5, 7, 10]:
            acc = self._calculate_accuracy(y_val, pred_val, tolerance)
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        # Rendimiento de modelos individuales OPTIMIZADOS
        print(f"\n🤖 RENDIMIENTO DE MODELOS INDIVIDUALES OPTIMIZADOS (Validación Final):")
        model_names = list(self.base_models.keys())
        
        for i, name in enumerate(model_names):
            mae_individual = mean_absolute_error(y_val, base_pred_val[:, i])
            r2_individual = r2_score(y_val, base_pred_val[:, i])
            acc_individual = self._calculate_accuracy(y_val, base_pred_val[:, i])
            print(f"{name:<25}: MAE={mae_individual:.3f}, R²={r2_individual:.4f}, Acc={acc_individual:.1f}%")
        
        # Mostrar ensemble final
        ensemble_mae = mean_absolute_error(y_val, pred_val)
        ensemble_r2 = r2_score(y_val, pred_val)
        ensemble_acc = self._calculate_accuracy(y_val, pred_val)
        print(f"{'ENSEMBLE_OPTIMIZADO':<25}: MAE={ensemble_mae:.3f}, R²={ensemble_r2:.4f}, Acc={ensemble_acc:.1f}%")
        
        # Análisis de residuos
        residuals = y_val - pred_val
        print(f"\n📈 ANÁLISIS DE RESIDUOS:")
        print(f"Media de residuos: {np.mean(residuals):.3f}")
        print(f"Std de residuos: {np.std(residuals):.3f}")
        print(f"Sesgo (skewness): {self._calculate_skewness(residuals):.3f}")
        print(f"Curtosis: {self._calculate_kurtosis(residuals):.3f}")
        
        # Percentiles de error
        abs_errors = np.abs(residuals)
        print(f"\n📊 DISTRIBUCIÓN DE ERRORES ABSOLUTOS:")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            error_p = np.percentile(abs_errors, p)
            print(f"P{p}: {error_p:.2f} puntos")
        
        # Importancia de features
        self._display_feature_importance()
        
        # Evaluación del objetivo con validación cruzada
        print(f"\n🎯 EVALUACIÓN DEL OBJETIVO:")
        cv_acc = cv_metrics['mean_accuracy']
        final_acc = val_metrics['accuracy']
        
        print(f"Precisión CV: {cv_acc:.2f}% ± {cv_metrics['std_accuracy']:.2f}%")
        print(f"Precisión Final: {final_acc:.2f}%")
        
        
        if cv_acc >= 97.0 and final_acc >= 97.0:
            print(f"✅ OBJETIVO ALCANZADO: Ambas métricas >= 97%")
            print("🏆 Modelo listo para producción")
        elif cv_acc >= 95.0 and final_acc >= 95.0:
            print(f"🟡 CERCA DEL OBJETIVO: Ambas métricas >= 95%")
            print("📈 Modelo prometedor, necesita ajustes menores")
        elif cv_acc >= 50.0 and final_acc >= 50.0:
            print(f"🟡 PROGRESO SIGNIFICATIVO: Ambas métricas >= 50%")
            gap_cv = 97.0 - cv_acc
            gap_final = 97.0 - final_acc
            print(f"📈 Gap CV: {gap_cv:.2f}%, Gap Final: {gap_final:.2f}%")
            print("💡 Continuar optimización de features y modelos")
        else:
            print(f"❌ OBJETIVO NO ALCANZADO")
            gap_cv = 97.0 - cv_acc
            gap_final = 97.0 - final_acc
            print(f"📈 Gap CV: {gap_cv:.2f}%, Gap Final: {gap_final:.2f}%")
            
            # Recomendaciones específicas mejoradas
            print(f"\n💡 RECOMENDACIONES ESPECÍFICAS:")
            if cv_stability > 0.2:
                print("• Mejorar estabilidad del modelo (más datos o regularización)")
            if mae_diff > 2.0:
                print("• Reducir overfitting con regularización adicional")
            if cv_acc < 30.0:
                print("• Revisar feature engineering - Features actuales insuficientes")
            if len(y_val) < 500:
                print("• Aumentar datos de validación")
            print("• Considerar features de dominio específico adicionales")
            print("• Explorar modelos más sofisticados (Transformers, GNNs)")
            print("• Analizar outliers y casos extremos")
        
        print("="*80)
    
    def _calculate_skewness(self, data):
        """Calcula el sesgo de los datos"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calcula la curtosis de los datos"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _display_feature_importance(self):
        """Muestra importancia de features de forma avanzada"""
        print(f"\nIMPORTANCIA DE FEATURES:")
        
        try:
            feature_importance = self.get_feature_importance()
            
            # Top features por modelo
            for model_name in ['XGBoost', 'RandomForest']:
                model_features = feature_importance[feature_importance['model'] == model_name]
                if not model_features.empty:
                    top_features = model_features.nlargest(10, 'importance')
                    print(f"\n{model_name} - Top 10 Features:")
                    for i, (_, row) in enumerate(top_features.iterrows(), 1):
                        print(f"{i:2d}. {row['feature']:<25}: {row['importance']:.4f}")
            
            # Features más importantes en promedio
            avg_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            print(f"\nTOP 15 FEATURES PROMEDIO:")
            for i, (feature, importance) in enumerate(avg_importance.head(15).items(), 1):
                print(f"{i:2d}. {feature:<25}: {importance:.4f}")
                
        except Exception as e:
            print(f"Error calculando importancia: {e}")

    def _create_ultra_hybrid_predictor(self, df_features: pd.DataFrame, team1: str, team2: str, 
                                     base_predictions: Dict, neural_pred: float, 
                                     is_team1_home: bool = True) -> Tuple[float, float, Dict]:
        """
        PREDICTOR HÍBRIDO ULTRA-OPTIMIZADO PARA 97% PRECISIÓN
        Combina predictor directo matemático + ensemble ML + validación de consistencia
        """
        try:
            # 1. PREDICTOR DIRECTO MATEMÁTICO (85% peso)
            direct_pred = self._create_mathematical_predictor_unified(
                df_features, team1, team2, None, None, is_team1_home, 'correlation_based'
            )
            
            # 2. ENSEMBLE ML OPTIMIZADO (15% peso)
            # Filtrar solo los mejores modelos (>50% precisión histórica)
            best_models = {
                'extra_trees_primary': base_predictions.get('extra_trees_primary', direct_pred),
                'gradient_boost_primary': base_predictions.get('gradient_boost_primary', direct_pred),
                'ridge_ultra_conservative': base_predictions.get('ridge_ultra_conservative', direct_pred)
            }
            
            # Pesos dinámicos basados en performance histórica
            model_weights = {
                'extra_trees_primary': 0.4,      # Mejor modelo estable (56.6%)
                'gradient_boost_primary': 0.4,   # Segundo mejor (73.8% pero overfitting)
                'ridge_ultra_conservative': 0.2  # Estabilidad (98.3% pero overfitting)
            }
            
            # Ensemble ML ponderado
            ml_ensemble = 0
            total_weight = 0
            for model, pred in best_models.items():
                if 180 <= pred <= 280:  # Validar rango
                    weight = model_weights.get(model, 0.1)
                    ml_ensemble += pred * weight
                    total_weight += weight
            
            if total_weight > 0:
                ml_ensemble /= total_weight
            else:
                ml_ensemble = direct_pred
            
            # 3. COMBINACIÓN HÍBRIDA FINAL
            # 85% predictor directo + 15% ensemble ML
            hybrid_prediction = (direct_pred * 0.85) + (ml_ensemble * 0.15)
            
            # 4. VALIDACIÓN DE CONSISTENCIA
            all_predictions = [direct_pred, ml_ensemble, neural_pred] + list(best_models.values())
            valid_preds = [p for p in all_predictions if 180 <= p <= 280]
            
            if len(valid_preds) >= 3:
                pred_std = np.std(valid_preds)
                pred_mean = np.mean(valid_preds)
                
                # Calcular confianza basada en consistencia
                consistency_score = max(0, 100 - (pred_std * 10))  # Penalizar alta varianza
                
                # Ajustar predicción si hay alta inconsistencia
                if pred_std > 8:  # Alta varianza
                    # Usar mediana en lugar de media para robustez
                    robust_prediction = np.median(valid_preds)
                    hybrid_prediction = (hybrid_prediction * 0.7) + (robust_prediction * 0.3)
                    consistency_score *= 0.8  # Penalizar confianza
                
            else:
                consistency_score = 50.0  # Confianza baja por datos insuficientes
            
            # 5. AJUSTES FINALES NBA
            # Ajuste por momentum reciente (si disponible)
            momentum_adjustment = 0
            try:
                team1_data = df_features[df_features['TEAM'] == team1].tail(5)
                team2_data = df_features[df_features['TEAM'] == team2].tail(5)
                
                if not team1_data.empty and not team2_data.empty and 'PTS' in df_features.columns:
                    team1_recent_avg = team1_data['PTS'].mean()
                    team2_recent_avg = team2_data['PTS'].mean()
                    recent_total = team1_recent_avg + team2_recent_avg
                    
                    # Ajuste suave hacia momentum reciente
                    momentum_adjustment = (recent_total - hybrid_prediction) * 0.1
                    
            except Exception as e:
                logger.debug(f"No se pudo calcular momentum: {e}")
            
            hybrid_prediction += momentum_adjustment
            
            # LÍMITES FINALES ESTRICTOS
            final_prediction = np.clip(hybrid_prediction, 195, 265)
            final_confidence = np.clip(consistency_score, 60, 95)  # Confianza conservadora
            
            # Detalles para debugging
            prediction_details = {
                'direct_mathematical': direct_pred,
                'ml_ensemble': ml_ensemble,
                'neural_network': neural_pred,
                'hybrid_final': final_prediction,
                'momentum_adjustment': momentum_adjustment,
                'consistency_std': pred_std if 'pred_std' in locals() else 0,
                'method_weights': {'direct': 0.85, 'ml_ensemble': 0.15}
            }
            
            logger.info(f"Híbrido ultra: {final_prediction:.1f} (conf: {final_confidence:.1f}%, direct: {direct_pred:.1f}, ml: {ml_ensemble:.1f})")
            
            return float(final_prediction), float(final_confidence), prediction_details
            
        except Exception as e:
            logger.error(f"Error en predictor híbrido: {e}")
            # Fallback al predictor directo
            direct_fallback = self._create_mathematical_predictor_unified(
                df_features, team1, team2, None, None, is_team1_home, 'fallback'
            )
            return float(direct_fallback), 70.0, {'error': str(e), 'fallback': direct_fallback}
    
    def predict(self, team1: str, team2: str, teams_data: pd.DataFrame, 
                is_team1_home: bool = True) -> Dict:
        """
        PREDICCIÓN ULTRA-OPTIMIZADA PARA 97% PRECISIÓN - REFACTORIZADA
        Método híbrido usando funciones utilitarias centralizadas
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Validar y convertir inputs
        team1_str = str(team1) if team1 is not None else "UNKNOWN"
        team2_str = str(team2) if team2 is not None else "UNKNOWN"
        is_team1_home = bool(is_team1_home)
        
        try:
            # 1. PREPARAR FEATURES USANDO PIPELINE CENTRALIZADO
            df_features = self._prepare_features_for_prediction(teams_data, team1_str, team2_str)
            if df_features is None:
                return self._create_emergency_fallback_prediction(team1_str, team2_str)
            
            # 2. CREAR VECTOR DE FEATURES OPTIMIZADO
            feature_vector = self._create_feature_vector_for_teams(df_features, team1_str, team2_str, is_team1_home)
            if feature_vector is None:
                return self._create_emergency_fallback_prediction(team1_str, team2_str)
            
            # 3. OBTENER PREDICCIONES DE MODELOS BASE USANDO FUNCIONES UTILITARIAS
            base_predictions = self._get_base_model_predictions(feature_vector)
            
            # 4. CALCULAR ENSEMBLE USANDO FUNCIÓN CENTRALIZADA
            ensemble_prediction = self._calculate_ensemble_prediction(base_predictions, method='weighted')
            ensemble_pred_value = float(np.mean(ensemble_prediction))
            
            # 5. PREDICTOR MATEMÁTICO DIRECTO
            mathematical_pred = self._create_mathematical_predictor_unified(
                df_features, team1_str, team2_str, None, None, is_team1_home, 'comprehensive'
            )
            
            # 6. COMBINACIÓN HÍBRIDA FINAL USANDO CONFIGURACIÓN CENTRALIZADA
            final_prediction = self._combine_predictions_hybrid(
                ensemble_pred_value, mathematical_pred, method='optimized_70_30'
            )
            
            # 7. CALCULAR CONFIANZA USANDO FUNCIÓN CENTRALIZADA
            additional_preds = [mathematical_pred, ensemble_pred_value]
            confidence = self._calculate_prediction_confidence_unified(base_predictions, additional_preds)
            
            # 8. PREPARAR RESPUESTA ESTRUCTURADA
            result = self._build_prediction_response(
                final_prediction, confidence, base_predictions, 
                ensemble_pred_value, mathematical_pred, team1_str, team2_str
            )
            
            logger.info(f"Predicción {team1_str} vs {team2_str}: {final_prediction:.1f} puntos (confianza: {confidence:.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error crítico en predicción: {e}")
            return self._create_emergency_fallback_prediction(team1_str, team2_str)
    
    def _prepare_features_for_prediction(self, teams_data: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
        """FUNCIÓN UTILITARIA: Preparar features para predicción"""
        try:
            # Crear features para predicción
            df_features = self.feature_engine.create_features(teams_data)
            
            # Aplicar el mismo filtro de correlación que en entrenamiento
            df_features = self.feature_engine.apply_final_correlation_filter(df_features, correlation_threshold=0.85)
            
            # Validar que hay datos para ambos equipos
            if 'Team' not in df_features.columns:
                logger.warning("Columna 'Team' no encontrada")
                return None
            
            team1_data = df_features[df_features['Team'] == team1]
            team2_data = df_features[df_features['Team'] == team2]
            
            if team1_data.empty or team2_data.empty:
                logger.warning(f"Datos insuficientes para {team1} vs {team2}")
                return None
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            return None
    
    def _create_feature_vector_for_teams(self, df_features: pd.DataFrame, team1: str, team2: str, 
                                       is_team1_home: bool) -> np.ndarray:
        """FUNCIÓN UTILITARIA: Crear vector de features para dos equipos"""
        try:
            feature_cols = self.feature_engine.feature_columns
            if not feature_cols:
                logger.error("No hay features seleccionadas del entrenamiento")
                return None
            
            # Verificar features disponibles
            missing_features = [col for col in feature_cols if col not in df_features.columns]
            if missing_features:
                logger.warning(f"Features faltantes: {len(missing_features)}")
                available_features = [col for col in feature_cols if col in df_features.columns]
                if len(available_features) < 10:
                    logger.error("Muy pocas features disponibles")
                    return None
                feature_cols = available_features
            
            # Obtener datos recientes de ambos equipos
            team1_data = df_features[df_features['Team'] == team1]
            team2_data = df_features[df_features['Team'] == team2]
            
            team1_recent = team1_data.tail(3).mean(numeric_only=True)
            team2_recent = team2_data.tail(3).mean(numeric_only=True)
            
            # Crear vector de features combinado
            feature_vector = []
            for col in feature_cols:
                try:
                    if col in team1_recent.index and col in team2_recent.index:
                        val1 = float(team1_recent[col]) if not pd.isna(team1_recent[col]) else 0.0
                        val2 = float(team2_recent[col]) if not pd.isna(team2_recent[col]) else 0.0
                        
                        # Lógica de combinación optimizada
                        combined_val = self._combine_team_feature_values(col, val1, val2, is_team1_home)
                        feature_vector.append(float(combined_val))
                    else:
                        # Valor por defecto basado en estadísticas NBA
                        default_val = self._get_default_feature_value(col)
                        feature_vector.append(default_val)
                        
                except Exception as e:
                    logger.debug(f"Error procesando feature {col}: {e}")
                    feature_vector.append(0.0)
            
            # Validar vector
            if len(feature_vector) != len(feature_cols):
                logger.error(f"Mismatch en features: esperado {len(feature_cols)}, obtenido {len(feature_vector)}")
                return None
            
            # Convertir y validar usando DataValidator
            X_pred = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
            
            # VALIDACIÓN ROBUSTA DEL VECTOR DE PREDICCIÓN
            X_pred_validated = DataValidator.validate_prediction_input(X_pred, feature_cols)
            
            return X_pred_validated
            
        except Exception as e:
            logger.error(f"Error creando vector de features: {e}")
            return None
    
    def _combine_team_feature_values(self, feature_name: str, val1: float, val2: float, 
                                   is_team1_home: bool) -> float:
        """FUNCIÓN UTILITARIA: Combinar valores de features de dos equipos"""
        # Features críticas que se suman
        if any(keyword in feature_name.lower() for keyword in ['pts', 'score', 'total', 'projection', 'volume', 'shots']):
            return val1 + val2
        
        # Porcentajes que se promedian
        elif '%' in feature_name or 'pct' in feature_name.lower() or 'efficiency' in feature_name.lower():
            return (val1 + val2) / 2
        
        # Features de ventaja local
        elif 'home' in feature_name.lower():
            return 1.0 if is_team1_home else 0.0
        
        # Otras features se promedian
        else:
            return (val1 + val2) / 2
    
    def _get_default_feature_value(self, feature_name: str) -> float:
        """FUNCIÓN UTILITARIA: Obtener valor por defecto para features faltantes"""
        if 'pts' in feature_name.lower() or 'score' in feature_name.lower():
            return 110.0  # Promedio NBA por equipo
        elif '%' in feature_name or 'pct' in feature_name.lower():
            return 0.45   # Promedio NBA shooting
        elif 'home' in feature_name.lower():
            return 0.5    # Neutral
        else:
            return 0.0
    
    def _get_base_model_predictions(self, feature_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """FUNCIÓN UTILITARIA: Obtener predicciones de todos los modelos base"""
        try:
            # Escalar features usando función utilitaria
            X_pred_scaled = self._prepare_scaled_features(feature_vector, 'standard', fit_scaler=False)
            
            base_predictions = {}
            feature_names = self.feature_engine.feature_columns
            
            for model_name, model in self.base_models.items():
                try:
                    # Seleccionar datos apropiados usando función utilitaria
                    X_for_model, _ = self._get_model_appropriate_data(
                        model_name, feature_vector, feature_vector,
                        X_pred_scaled, X_pred_scaled,
                        X_pred_scaled, X_pred_scaled,  # Usar scaled para robust también
                        feature_names
                    )
                    
                    # Hacer predicción
                    pred = float(model.predict(X_for_model)[0])
                    
                    # Aplicar límites usando función utilitaria
                    pred_array = np.array([pred])
                    pred_limited = self._apply_prediction_limits_and_validation(pred_array, model_name)
                    
                    base_predictions[model_name] = pred_limited
                    
                except Exception as e:
                    logger.warning(f"Error en modelo {model_name}: {e}")
                    # Fallback conservador
                    fallback_pred = np.array([220.0])
                    base_predictions[model_name] = self._apply_prediction_limits_and_validation(fallback_pred, model_name)
            
            return base_predictions
            
        except Exception as e:
            logger.error(f"Error obteniendo predicciones base: {e}")
            # Fallback con predicciones conservadoras
            fallback_predictions = {}
            for model_name in self.base_models.keys():
                fallback_predictions[model_name] = np.array([220.0])
            return fallback_predictions
    
    def _combine_predictions_hybrid(self, ensemble_pred: float, mathematical_pred: float, 
                                  method: str = 'optimized_70_30') -> float:
        """FUNCIÓN UTILITARIA: Combinar predicciones usando diferentes métodos"""
        try:
            if method == 'optimized_70_30':
                # 70% matemático + 30% ensemble ML (configuración optimizada)
                combined = (mathematical_pred * 0.70) + (ensemble_pred * 0.30)
            
            elif method == 'balanced_50_50':
                combined = (mathematical_pred * 0.50) + (ensemble_pred * 0.50)
            
            elif method == 'conservative_80_20':
                # Más peso al predictor matemático
                combined = (mathematical_pred * 0.80) + (ensemble_pred * 0.20)
            
            else:
                logger.warning(f"Método de combinación desconocido: {method}, usando optimized_70_30")
                combined = (mathematical_pred * 0.70) + (ensemble_pred * 0.30)
            
            # Aplicar límites usando configuración centralizada
            config = self.ensemble_config['prediction_limits']
            final_prediction = np.clip(combined, config['min_value'], config['max_value'])
            
            return float(final_prediction)
            
        except Exception as e:
            logger.error(f"Error combinando predicciones: {e}")
            return 220.0  # Fallback conservador
    
    def _build_prediction_response(self, final_prediction: float, confidence: float,
                                 base_predictions: Dict[str, np.ndarray], ensemble_pred: float,
                                 mathematical_pred: float, team1: str, team2: str) -> Dict:
        """FUNCIÓN UTILITARIA: Construir respuesta estructurada de predicción"""
        try:
            # Convertir predicciones base a formato simple
            individual_preds = {k: float(v[0]) if len(v) > 0 else 220.0 
                              for k, v in base_predictions.items()}
            
            # Calcular métricas de consistencia
            all_preds = list(individual_preds.values()) + [mathematical_pred, ensemble_pred]
            pred_std = np.std(all_preds)
            pred_range = max(all_preds) - min(all_preds)
            
            # Obtener configuración de pesos
            model_weights = self.ensemble_config['model_weights']
            
            result = {
                'total_points': final_prediction,
                'total_points_prediction': final_prediction,
                'confidence': confidence,
                'method': 'ultra_optimized_hybrid_refactored',
                'individual_predictions': individual_preds,
                'neural_network_prediction': ensemble_pred,
                'prediction_details': {
                    'mathematical_prediction': mathematical_pred,
                    'ensemble_prediction': ensemble_pred,
                    'model_weights': model_weights,
                    'teams': f"{team1} vs {team2}",
                    'confidence_factors': {
                        'model_consistency': pred_std,
                        'prediction_range': pred_range,
                        'mathematical_weight': 0.70,
                        'ml_weight': 0.30
                    },
                    'prediction_limits': self.ensemble_config['prediction_limits']
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error construyendo respuesta: {e}")
            return {
                'total_points': final_prediction,
                'total_points_prediction': final_prediction,
                'confidence': confidence,
                'method': 'fallback_response',
                'error': str(e)
            }

    def _create_mathematical_predictor_unified(self, df_features: pd.DataFrame = None, 
                                             team1: str = None, team2: str = None,
                                             team1_stats: pd.Series = None, team2_stats: pd.Series = None,
                                             is_team1_home: bool = True,
                                             prediction_method: str = 'comprehensive') -> float:
        """
        PREDICTOR MATEMÁTICO UNIFICADO
        
        Args:
            prediction_method: 'comprehensive', 'correlation_based', 'stats_based', 'fallback'
            df_features: DataFrame completo con features (para correlation_based)
            team1/team2: Nombres de equipos (para correlation_based)
            team1_stats/team2_stats: Series con estadísticas (para stats_based)
            is_team1_home: Ventaja de local
        """
        
        try:
            if prediction_method == 'comprehensive':
                # MÉTODO COMPREHENSIVO: Combina todos los enfoques disponibles
                predictions = []
                weights = []
                
                # 1. Predicción basada en correlaciones (si hay datos)
                if df_features is not None and team1 and team2:
                    corr_pred = self._correlation_based_prediction(df_features, team1, team2, is_team1_home)
                    if corr_pred > 0:
                        predictions.append(corr_pred)
                        weights.append(0.5)  # 50% peso
                
                # 2. Predicción basada en estadísticas (si hay datos)
                if team1_stats is not None and team2_stats is not None:
                    stats_pred = self._stats_based_prediction(team1_stats, team2_stats, is_team1_home)
                    if stats_pred > 0:
                        predictions.append(stats_pred)
                        weights.append(0.3)  # 30% peso
                
                # 3. Predicción de fallback
                fallback_pred = self._fallback_prediction(df_features)
                predictions.append(fallback_pred)
                weights.append(0.2)  # 20% peso
                
                # Combinar predicciones
                if len(predictions) > 1:
                    # Normalizar pesos
                    total_weight = sum(weights)
                    weights = [w / total_weight for w in weights]
                    final_pred = sum(pred * weight for pred, weight in zip(predictions, weights))
                else:
                    final_pred = predictions[0] if predictions else 220.0
                
                return np.clip(final_pred, 195, 265)
            
            elif prediction_method == 'correlation_based':
                return self._correlation_based_prediction(df_features, team1, team2, is_team1_home)
            
            elif prediction_method == 'stats_based':
                return self._stats_based_prediction(team1_stats, team2_stats, is_team1_home)
            
            elif prediction_method == 'fallback':
                return self._fallback_prediction(df_features)
            
            else:
                logger.warning(f"Método desconocido: {prediction_method}, usando fallback")
                return self._fallback_prediction(df_features)
            
        except Exception as e:
            logger.error(f"Error en predictor matemático unificado: {e}")
            return 220.0
    
    def _correlation_based_prediction(self, df_features: pd.DataFrame, team1: str, team2: str, is_team1_home: bool) -> float:
        """Predicción basada en correlaciones directas con features"""
        try:
            # Filtrar datos de equipos
            team1_data = df_features[df_features['Team'] == team1].copy()
            team2_data = df_features[df_features['Team'] == team2].copy()
            
            if team1_data.empty or team2_data.empty:
                return 0.0  # Indicar que no hay datos
            
            # MÉTODOS ORDENADOS POR CORRELACIÓN CON TARGET
            methods = []
            
            # Método 1: Ensemble Projection (correlación alta)
            if 'ensemble_projection_v1' in df_features.columns:
                team1_proj = team1_data['ensemble_projection_v1'].iloc[-1] if not team1_data.empty else 110
                team2_proj = team2_data['ensemble_projection_v1'].iloc[-1] if not team2_data.empty else 110
                methods.append(('ensemble_projection', team1_proj + team2_proj, 0.4))
            
            # Método 2: Direct Scoring Projection
            if 'direct_scoring_projection' in df_features.columns:
                team1_direct = team1_data['direct_scoring_projection'].iloc[-1] if not team1_data.empty else 110
                team2_direct = team2_data['direct_scoring_projection'].iloc[-1] if not team2_data.empty else 110
                methods.append(('direct_scoring', team1_direct + team2_direct, 0.3))
            
            # Método 3: Weighted Shot Volume
            if 'weighted_shot_volume' in df_features.columns:
                team1_shots = team1_data['weighted_shot_volume'].iloc[-1] if not team1_data.empty else 110
                team2_shots = team2_data['weighted_shot_volume'].iloc[-1] if not team2_data.empty else 110
                methods.append(('shot_volume', team1_shots + team2_shots, 0.2))
            
            # Método 4: FG% + FGA básico
            if 'FG%' in df_features.columns and 'FGA' in df_features.columns:
                team1_fg_pct = team1_data['FG%'].iloc[-1] if not team1_data.empty else 0.45
                team1_fga = team1_data['FGA'].iloc[-1] if not team1_data.empty else 85
                team2_fg_pct = team2_data['FG%'].iloc[-1] if not team2_data.empty else 0.45
                team2_fga = team2_data['FGA'].iloc[-1] if not team2_data.empty else 85
                
                team1_pts = team1_fg_pct * team1_fga * 2.2  # Factor NBA promedio
                team2_pts = team2_fg_pct * team2_fga * 2.2
                methods.append(('fg_basic', team1_pts + team2_pts, 0.1))
            
            # Combinar métodos disponibles
            if not methods:
                return 0.0
            
            # Filtrar predicciones válidas y combinar
            valid_methods = [(name, pred, weight) for name, pred, weight in methods if 180 <= pred <= 280]
            
            if not valid_methods:
                # Ajustar predicciones fuera de rango
                adjusted_methods = [(name, np.clip(pred, 200, 260), weight * 0.5) for name, pred, weight in methods]
                valid_methods = adjusted_methods
            
            # Normalizar pesos y combinar
            total_weight = sum(weight for _, _, weight in valid_methods)
            if total_weight > 0:
                prediction = sum(pred * (weight / total_weight) for _, pred, weight in valid_methods)
            else:
                prediction = np.mean([pred for _, pred, _ in valid_methods])
            
            # Ajustes contextuales
            home_advantage = 2.5 if is_team1_home else -2.5
            prediction += home_advantage
            
            # Ajuste por pace si disponible
            if 'PACE' in df_features.columns:
                team1_pace = team1_data['PACE'].iloc[-1] if not team1_data.empty else 100
                team2_pace = team2_data['PACE'].iloc[-1] if not team2_data.empty else 100
                avg_pace = (team1_pace + team2_pace) / 2
                pace_adjustment = (avg_pace - 100) * 0.8
                prediction += pace_adjustment
            
            return np.clip(prediction, 190, 270)
            
        except Exception as e:
            logger.debug(f"Error en predicción por correlación: {e}")
            return 0.0
    
    def _stats_based_prediction(self, team1_stats: pd.Series, team2_stats: pd.Series, is_team1_home: bool) -> float:
        """Predicción basada en estadísticas de equipos"""
        try:
            # 1. Proyección directa de scoring
            if 'direct_scoring_projection' in team1_stats.index:
                scoring_proj = team1_stats['direct_scoring_projection'] + team2_stats['direct_scoring_projection']
            else:
                # Fallback: usar PTS si está disponible
                pts1 = team1_stats.get('PTS', 110)
                pts2 = team2_stats.get('PTS', 110)
                scoring_proj = pts1 + pts2
            
            # 2. Factor de eficiencia
            efficiency_factor = 1.0
            if 'FG%' in team1_stats.index:
                avg_fg_pct = (team1_stats['FG%'] + team2_stats['FG%']) / 2
                efficiency_factor = 0.8 + (avg_fg_pct * 0.4)  # Rango 0.8-1.2
            
            # 3. Factor de volumen
            volume_factor = 1.0
            if 'total_expected_shots' in team1_stats.index:
                total_shots = team1_stats['total_expected_shots'] + team2_stats['total_expected_shots']
                volume_factor = 0.9 + (total_shots / 200) * 0.2
            
            # 4. Ventaja de local
            home_advantage = 3.2 if is_team1_home else -3.2
            
            # 5. Combinación final
            prediction = (scoring_proj * efficiency_factor * volume_factor) + home_advantage
            
            return np.clip(prediction, 200, 250)
            
        except Exception as e:
            logger.debug(f"Error en predicción por estadísticas: {e}")
            return 0.0
    
    def _fallback_prediction(self, df_features: pd.DataFrame = None) -> float:
        """Predicción de fallback robusta"""
        try:
            if df_features is not None:
                # Usar features más correlacionadas disponibles
                if 'ensemble_projection_v1' in df_features.columns:
                    return df_features['ensemble_projection_v1'].tail(10).mean()
                elif 'PTS' in df_features.columns and 'PTS_Opp' in df_features.columns:
                    recent_total = (df_features['PTS'] + df_features['PTS_Opp']).tail(10).mean()
                    return np.clip(recent_total, 200, 240)
            
            # Promedio NBA histórico como último recurso
            return 215.0
            
        except Exception:
            return 215.0

    def _calculate_prediction_confidence(self, base_predictions: Dict, 
                                       mathematical_pred: float, 
                                       ensemble_pred: float) -> float:
        """Calcula confianza basada en consistencia de predicciones"""
        try:
            all_predictions = list(base_predictions.values()) + [mathematical_pred, ensemble_pred]
            valid_predictions = [p for p in all_predictions if 180 <= p <= 280]
            
            if len(valid_predictions) < 3:
                return 70.0
            
            # Calcular consistencia
            pred_std = np.std(valid_predictions)
            pred_mean = np.mean(valid_predictions)
            
            # Confianza basada en consistencia (menor std = mayor confianza)
            base_confidence = max(70, 95 - (pred_std * 5))
            
            # Bonificación por predicciones en rango típico NBA (210-240)
            if 210 <= pred_mean <= 240:
                base_confidence += 5
            
            # Penalización por alta varianza
            if pred_std > 10:
                base_confidence -= 10
            
            return np.clip(base_confidence, 70, 95)
            
        except:
            return 75.0

    def _apply_season_context_adjustment(self, prediction: float) -> float:
        """Aplica ajustes contextuales de temporada"""
        try:
            # Ajustes típicos NBA por contexto
            # En temporada regular: sin ajuste
            # Playoffs: +2-3 puntos (mayor intensidad)
            # Back-to-back: -3-5 puntos (fatiga)
            
            # Por ahora, ajuste conservador
            return 0.0  # Sin ajuste hasta tener datos contextuales
            
        except:
            return 0.0

    def _create_emergency_fallback_prediction(self, team1: str, team2: str) -> Dict:
        """Predicción de emergencia cuando fallan todos los métodos"""
        fallback_pred = 220.0  # Promedio histórico NBA
        
        # Asegurar que team1 y team2 sean strings
        team1_str = str(team1) if team1 is not None else "UNKNOWN"
        team2_str = str(team2) if team2 is not None else "UNKNOWN"
        
        return {
            'total_points': float(fallback_pred),
            'total_points_prediction': float(fallback_pred),
            'confidence': 60.0,
            'method': 'emergency_fallback',
            'individual_predictions': {'emergency': float(fallback_pred)},
            'neural_network_prediction': float(fallback_pred),
            'prediction_details': {
                'error': 'Fallback de emergencia activado',
                'teams': f"{team1_str} vs {team2_str}",
                'method': 'historical_average'
            }
        }
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")
        
        model_data = {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'scalers': self.scalers,
            'feature_engine': self.feature_engine,
            'performance_metrics': self.performance_metrics,
            'neural_network_state': self.neural_network.state_dict() if self.neural_network else None
        }
        
        try:
            joblib.dump(model_data, filepath, compress=3)
            logger.info(f"Modelo guardado en: {filepath}")
        except Exception as e:
            logger.warning(f"Error guardando modelo: {e}")
            # Intentar guardado sin compresión
            try:
                joblib.dump(model_data, filepath)
                logger.info(f"Modelo guardado sin compresión en: {filepath}")
            except Exception as e2:
                logger.error(f"Error crítico guardando modelo: {e2}")
                raise
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        model_data = joblib.load(filepath)
        
        self.base_models = model_data['base_models']
        self.meta_model = model_data['meta_model']
        self.scalers = model_data['scalers']
        self.feature_engine = model_data['feature_engine']
        self.performance_metrics = model_data['performance_metrics']
        
        # Cargar red neuronal si existe
        if model_data['neural_network_state']:
            input_size = len(self.feature_engine.feature_columns)
            self.neural_network = self._create_neural_network(input_size)
            self.neural_network.load_state_dict(model_data['neural_network_state'])
            self.neural_network.eval()
        
        self.is_trained = True
        logger.info(f"Modelo cargado desde: {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importancia de features de los modelos"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        importance_data = []
        
        try:
            feature_names = self.feature_engine.feature_columns
        
            # XGBoost Primary
            if 'xgboost_primary' in self.base_models:
                try:
                    xgb_importance = self.base_models['xgboost_primary'].feature_importances_
                    # Asegurar que el número de features coincida
                    min_features = min(len(xgb_importance), len(feature_names))
                    for i in range(min_features):
                        importance_data.append({
                            'feature': feature_names[i],
                            'importance': xgb_importance[i],
                            'model': 'XGBoost'
                    })
                except Exception as e:
                    print(f"Error obteniendo importancia de XGBoost: {e}")
            
            # Random Forest Primary
            if 'random_forest_primary' in self.base_models:
                try:
                    rf_importance = self.base_models['random_forest_primary'].feature_importances_
                    # Asegurar que el número de features coincida
                    min_features = min(len(rf_importance), len(feature_names))
                    for i in range(min_features):
                        importance_data.append({
                            'feature': feature_names[i],
                            'importance': rf_importance[i],
                            'model': 'RandomForest'
                        })
                except Exception as e:
                    print(f"Error obteniendo importancia de Random Forest: {e}")
            
            # Extra Trees Primary
            if 'extra_trees_primary' in self.base_models:
                try:
                    et_importance = self.base_models['extra_trees_primary'].feature_importances_
                    # Asegurar que el número de features coincida
                    min_features = min(len(et_importance), len(feature_names))
                    for i in range(min_features):
                        importance_data.append({
                            'feature': feature_names[i],
                            'importance': et_importance[i],
                            'model': 'ExtraTrees'
                        })
                except Exception as e:
                    print(f"Error obteniendo importancia de Extra Trees: {e}")
            
            if not importance_data:
                print("No se pudo obtener importancia de ningún modelo")
                return pd.DataFrame()
        
            return pd.DataFrame(importance_data)
            
        except Exception as e:
            print(f"Error general en get_feature_importance: {e}")
            return pd.DataFrame()

    def _initialize_ultra_conservative_models(self) -> Dict:
        """
        MODELOS BASE ULTRA-CONSERVADORES PARA MÁXIMA ESTABILIDAD CV
        Configuración extremadamente conservadora para reducir overfitting
        """
        return {
            # RANDOM FOREST ULTRA-CONSERVADOR - Máxima estabilidad
            'random_forest_primary': RandomForestRegressor(
                n_estimators=200,           # Reducido para evitar overfitting
                max_depth=8,                # Profundidad muy limitada
                min_samples_split=20,       # Splits muy conservadores
                min_samples_leaf=10,        # Hojas grandes para generalización
                max_features=0.5,           # Solo 50% de features
                bootstrap=True,
                oob_score=True,
                min_impurity_decrease=0.01, # Requiere mejora mínima para split
                n_jobs=-1,
                random_state=self.random_state
            ),
            
            # EXTRA TREES ULTRA-CONSERVADOR - Diversidad controlada
            'extra_trees_primary': ExtraTreesRegressor(
                n_estimators=150,           # Reducido significativamente
                max_depth=6,                # Muy limitado
                min_samples_split=25,       # Muy conservador
                min_samples_leaf=12,        # Hojas grandes
                max_features=0.4,           # Features muy limitadas
                bootstrap=False,
                min_impurity_decrease=0.02, # Más restrictivo
                n_jobs=-1,
                random_state=self.random_state
            ),
            
            # GRADIENT BOOSTING EXTREMADAMENTE CONSERVADOR
            'gradient_boost_primary': GradientBoostingRegressor(
                n_estimators=100,           # Muy reducido
                learning_rate=0.01,         # Learning rate extremadamente bajo
                max_depth=4,                # Profundidad mínima
                min_samples_split=30,       # Muy conservador
                min_samples_leaf=15,        # Hojas muy grandes
                subsample=0.6,              # Subsampling agresivo
                max_features=0.3,           # Features muy limitadas
                validation_fraction=0.3,    # Validación interna grande
                n_iter_no_change=10,        # Early stopping muy agresivo
                tol=1e-3,                   # Tolerancia más estricta
                random_state=self.random_state
            ),
            
            # RIDGE REGRESSION EXTREMADAMENTE CONSERVADOR
            'ridge_ultra_conservative': Ridge(
                alpha=100.0,                # Regularización extrema
                fit_intercept=True,
                copy_X=True,
                max_iter=3000,
                tol=1e-5,                   # Tolerancia muy estricta
                solver='auto',
                random_state=self.random_state
            ),
            
            # ELASTIC NET EXTREMADAMENTE CONSERVADOR
            'elastic_net_ultra_conservative': ElasticNet(
                alpha=20.0,                 # Regularización muy fuerte
                l1_ratio=0.7,               # Más L1 para selección de features
                fit_intercept=True,
                precompute=False,
                max_iter=3000,
                copy_X=True,
                tol=1e-5,
                warm_start=False,
                positive=False,
                random_state=self.random_state,
                selection='cyclic'
            ),
            
            # XGBOOST EXTREMADAMENTE CONSERVADOR
            'xgboost_primary': XGBRegressor(
                n_estimators=80,            # Muy limitado
                max_depth=3,                # Profundidad mínima
                learning_rate=0.01,         # Learning rate extremadamente bajo
                subsample=0.6,              # Subsampling agresivo
                colsample_bytree=0.5,       # Features muy limitadas
                colsample_bylevel=0.6,
                reg_alpha=20.0,             # Regularización L1 extrema
                reg_lambda=30.0,            # Regularización L2 extrema
                min_child_weight=20,        # Peso mínimo muy alto
                gamma=5.0,                  # Complejidad mínima muy alta
                objective='reg:squarederror',
                eval_metric='mae',
                early_stopping_rounds=8,    # Early stopping muy agresivo
                verbosity=0,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # LIGHTGBM EXTREMADAMENTE CONSERVADOR
            'lightgbm_primary': LGBMRegressor(
                n_estimators=60,            # Muy limitado
                max_depth=3,                # Profundidad mínima
                learning_rate=0.005,        # Learning rate extremadamente bajo
                num_leaves=8,               # Hojas muy limitadas
                min_child_samples=50,       # Muestras mínimas muy altas
                min_child_weight=0.1,
                subsample=0.6,              # Subsampling agresivo
                colsample_bytree=0.5,       # Features muy limitadas
                reg_alpha=25.0,             # Regularización L1 extrema
                reg_lambda=35.0,            # Regularización L2 extrema
                min_split_gain=1.0,         # Ganancia mínima alta
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                importance_type='gain',
                verbosity=-1,
                random_state=self.random_state,
                n_jobs=-1
            )
        }

    def _analyze_model_performance_unified(self, y_train, pred_train, y_val, pred_val, 
                                          base_pred_train, base_pred_val, ensemble_train, ensemble_val, cv_scores,
                                          detailed: bool = True, stability_focus: bool = True):
        """
        ANÁLISIS UNIFICADO DEL RENDIMIENTO DEL MODELO
        
        Args:
            detailed: Si incluir análisis detallado de features y residuos
            stability_focus: Si enfocarse en estabilidad CV y consistencia
        """
        
        print("\n" + "="*80)
        title = "ANÁLISIS COMPLETO DEL MODELO" if detailed else "ANÁLISIS BÁSICO DEL MODELO"
        if stability_focus:
            title += " - ENFOQUE EN ESTABILIDAD CV"
        print(title)
        print("="*80)
        
        # Métricas de validación cruzada
        cv_metrics = self.performance_metrics['cross_validation']
        train_metrics = self.performance_metrics['train']
        val_metrics = self.performance_metrics['validation']
        
        print(f"\nVALIDACION CRUZADA TEMPORAL (7 FOLDS):")
        print(f"{'Métrica':<15} {'Media':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        print("-" * 75)
        
        fold_maes = [score['mae'] for score in cv_scores]
        fold_accs = [score['accuracy'] for score in cv_scores]
        fold_r2s = [score['r2'] for score in cv_scores]
        
        # Coeficiente de variación para medir estabilidad
        mae_cv = cv_metrics['std_mae'] / cv_metrics['mean_mae'] if cv_metrics['mean_mae'] > 0 else 0
        acc_cv = cv_metrics['std_accuracy'] / cv_metrics['mean_accuracy'] if cv_metrics['mean_accuracy'] > 0 else 0
        r2_cv = cv_metrics['std_r2'] / cv_metrics['mean_r2'] if cv_metrics['mean_r2'] > 0 else 0
        
        print(f"{'MAE':<15} {cv_metrics['mean_mae']:<15.3f} {cv_metrics['std_mae']:<15.3f} {mae_cv:<15.3f} {min(fold_maes):<15.3f} {max(fold_maes):<15.3f}")
        print(f"{'Precisión (%)':<15} {cv_metrics['mean_accuracy']:<15.2f} {cv_metrics['std_accuracy']:<15.2f} {acc_cv:<15.3f} {min(fold_accs):<15.2f} {max(fold_accs):<15.2f}")
        print(f"{'R²':<15} {cv_metrics['mean_r2']:<15.3f} {cv_metrics['std_r2']:<15.3f} {r2_cv:<15.3f} {min(fold_r2s):<15.3f} {max(fold_r2s):<15.3f}")
        
        if stability_focus:
            # Análisis de estabilidad CV
            print(f"\nANALISIS DE ESTABILIDAD CV:")
            if mae_cv < 0.1 and acc_cv < 0.1:
                print("EXCELENTE ESTABILIDAD - Modelo muy consistente entre folds")
            elif mae_cv < 0.2 and acc_cv < 0.2:
                print("BUENA ESTABILIDAD - Modelo moderadamente consistente")
            else:
                print("BAJA ESTABILIDAD - Modelo inconsistente entre folds")
        
            print(f"Coeficiente de variación MAE: {mae_cv:.3f} ({'Excelente' if mae_cv < 0.1 else 'Bueno' if mae_cv < 0.2 else 'Problemático'})")
            print(f"Coeficiente de variación Precisión: {acc_cv:.3f} ({'Excelente' if acc_cv < 0.1 else 'Bueno' if acc_cv < 0.2 else 'Problemático'})")
        
            print(f"DETALLES POR FOLD:")
            for i, score in enumerate(cv_scores):
                print(f"Fold {score['fold']}: MAE={score['mae']:.3f}, Acc={score['accuracy']:.1f}%, R²={score['r2']:.3f}, Train={score['n_train']}, Val={score['n_val']}")
        
        print(f"MÉTRICAS FINALES (Hold-out):")
        print(f"{'Métrica':<15} {'Entrenamiento':<15} {'Validación':<15} {'Gap':<15} {'Gap %':<15}")
        print("-" * 75)
        
        # Calcular gaps
        acc_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
        mae_gap = abs(train_metrics['mae'] - val_metrics['mae'])
        r2_gap = abs(train_metrics['r2'] - val_metrics['r2'])
        
        acc_gap_pct = (acc_gap / val_metrics['accuracy']) * 100 if val_metrics['accuracy'] > 0 else 0
        mae_gap_pct = (mae_gap / val_metrics['mae']) * 100 if val_metrics['mae'] > 0 else 0
        r2_gap_pct = (r2_gap / val_metrics['r2']) * 100 if val_metrics['r2'] > 0 else 0
        
        print(f"{'Precisión (%)':<15} {train_metrics['accuracy']:<15.2f} {val_metrics['accuracy']:<15.2f} {acc_gap:<15.2f} {acc_gap_pct:<15.1f}")
        print(f"{'MAE':<15} {train_metrics['mae']:<15.3f} {val_metrics['mae']:<15.3f} {mae_gap:<15.3f} {mae_gap_pct:<15.1f}")
        print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {val_metrics['r2']:<15.4f} {r2_gap:<15.4f} {r2_gap_pct:<15.1f}")
        
        # Análisis de overfitting
        print(f"\nANALISIS DE OVERFITTING:")
        if acc_gap < 2.0 and mae_gap < 0.5 and r2_gap < 0.05:
            print("SIN OVERFITTING - Excelente generalizacion")
        elif acc_gap < 5.0 and mae_gap < 1.0 and r2_gap < 0.1:
            print("OVERFITTING LEVE - Generalizacion aceptable")
        else:
            print("OVERFITTING SIGNIFICATIVO - Generalizacion problematica")
        
        if stability_focus:
            # Comparación CV vs Hold-out
            cv_val_gap = abs(cv_metrics['mean_accuracy'] - val_metrics['accuracy'])
            print(f"\nCONSISTENCIA CV vs HOLD-OUT:")
            print(f"Gap CV-Holdout (Precisión): {cv_val_gap:.2f}%")
            if cv_val_gap < 3.0:
                print("EXCELENTE CONSISTENCIA - CV predice bien el rendimiento final")
            elif cv_val_gap < 7.0:
                print("BUENA CONSISTENCIA - CV es un buen indicador")
            else:
                print("BAJA CONSISTENCIA - CV no predice bien el rendimiento final")
        
        # Análisis de precisión por tolerancia
        print(f"\nPRECISION POR TOLERANCIA (Validacion Final):")
        for tolerance in [1, 2, 3, 5, 7, 10]:
            acc = self._calculate_accuracy(y_val, pred_val, tolerance)
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        # Rendimiento de modelos individuales OPTIMIZADOS
        print(f"\nRENDIMIENTO DE MODELOS INDIVIDUALES (Validacion Final):")
        model_names = list(self.base_models.keys())
        
        for i, name in enumerate(model_names):
            if i < base_pred_val.shape[1]:
                mae_individual = mean_absolute_error(y_val, base_pred_val[:, i])
                r2_individual = r2_score(y_val, base_pred_val[:, i])
                acc_individual = self._calculate_accuracy(y_val, base_pred_val[:, i], tolerance=2.5)
                print(f"{name:<30}: MAE={mae_individual:.3f}, R²={r2_individual:.4f}, Acc={acc_individual:.1f}%")
        
        # Mostrar ensemble final
        ensemble_mae = mean_absolute_error(y_val, pred_val)
        ensemble_r2 = r2_score(y_val, pred_val)
        ensemble_acc = self._calculate_accuracy(y_val, pred_val, tolerance=2.5)
        ensemble_name = "ENSEMBLE_ULTRA_CONSERVADOR" if stability_focus else "ENSEMBLE_OPTIMIZADO"
        print(f"{ensemble_name:<30}: MAE={ensemble_mae:.3f}, R²={ensemble_r2:.4f}, Acc={ensemble_acc:.1f}%")
        
        if detailed:
            # Análisis de residuos
            residuals = y_val - pred_val
            print(f"\nANALISIS DE RESIDUOS:")
            print(f"Media de residuos: {np.mean(residuals):.3f}")
            print(f"Std de residuos: {np.std(residuals):.3f}")
            print(f"Sesgo (skewness): {self._calculate_skewness(residuals):.3f}")
            print(f"Curtosis: {self._calculate_kurtosis(residuals):.3f}")
        
            # Percentiles de error
            abs_errors = np.abs(residuals)
            print(f"\nDISTRIBUCION DE ERRORES ABSOLUTOS:")
            percentiles = [50, 75, 90, 95, 99]
            for p in percentiles:
                error_p = np.percentile(abs_errors, p)
                print(f"P{p}: {error_p:.2f} puntos")
        
            # Importancia de features
            self._display_feature_importance()
        
        # Evaluación del objetivo
        print(f"\nEVALUACION DEL OBJETIVO:")
        cv_acc = cv_metrics['mean_accuracy']
        final_acc = val_metrics['accuracy']
        
        print(f"Precisión CV: {cv_acc:.2f}% ± {cv_metrics['std_accuracy']:.2f}%")
        print(f"Precisión Final: {final_acc:.2f}%")
        
        if stability_focus:
            print(f"Estabilidad CV (MAE): {mae_cv:.3f}")
            print(f"Gap CV-Final: {cv_val_gap:.2f}%")
        
            # Criterios de éxito mejorados
            stability_good = mae_cv < 0.15 and acc_cv < 0.15
            consistency_good = cv_val_gap < 5.0
            performance_good = cv_acc >= 85.0 and final_acc >= 85.0
        
            print(f"\nCRITERIOS DE EXITO:")
            print(f"Estabilidad CV: {'PASS' if stability_good else 'FAIL'} (CV < 0.15)")
            print(f"Consistencia CV-Final: {'PASS' if consistency_good else 'FAIL'} (Gap < 5%)")
            print(f"Rendimiento: {'PASS' if performance_good else 'FAIL'} (Acc >= 85%)")
        
            if stability_good and consistency_good and performance_good:
                print(f"MODELO EXITOSO - Cumple todos los criterios de estabilidad")
            elif stability_good and consistency_good:
                print(f"MODELO PROMETEDOR - Estable pero necesita mejorar rendimiento")
            elif performance_good:
                print(f"MODELO POTENTE - Buen rendimiento pero inestable")
            else:
                print(f"MODELO NECESITA MEJORAS - No cumple criterios criticos")
        
        # Recomendaciones específicas
        print(f"\nRECOMENDACIONES ESPECIFICAS:")
        if stability_focus and not stability_good:
            print("• Aumentar regularización para mejorar estabilidad CV")
            print("• Reducir complejidad del modelo (menos features/parámetros)")
        if stability_focus and not consistency_good:
            print("• Revisar estrategia de validación cruzada")
            print("• Verificar data leakage temporal")
        if not performance_good:
            print("• Mejorar feature engineering")
            print("• Considerar modelos más sofisticados")
        if mae_cv > 0.2:
            print("• Modelo muy inestable - Revisar features problemáticas")
        if stability_focus and cv_val_gap > 10.0:
            print("• Gran inconsistencia CV-Final - Revisar división temporal")
        
        print("="*80)
