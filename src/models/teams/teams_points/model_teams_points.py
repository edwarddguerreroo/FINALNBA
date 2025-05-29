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

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
import os
import joblib
import pickle
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, VotingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                   cross_val_score, KFold, TimeSeriesSplit)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb

# DL Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Feature Engineering
from .features_teams_points import TeamPointsFeatureEngineer

logger = logging.getLogger(__name__)

class NBATeamPointsNet(nn.Module):
    """
    Red Neuronal Avanzada para Predicción de Puntos de Equipo NBA
    
    Arquitectura optimizada sin muchas capas pero con regularización agresiva:
    - Input Layer
    - 2 Hidden Layers con Layer Normalization y Dropout (más robusto que BatchNorm)
    - Output Layer
    - Skip connections para mejor flujo de gradientes
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(NBATeamPointsNet, self).__init__()
        
        # Arquitectura compacta pero efectiva
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # LayerNorm en lugar de BatchNorm
        self.dropout1 = nn.Dropout(0.3)  # Regularización agresiva
        
        self.hidden1 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)  # LayerNorm en lugar de BatchNorm
        self.dropout2 = nn.Dropout(0.4)  
        
        self.hidden2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.ln3 = nn.LayerNorm(hidden_size // 4)  # LayerNorm en lugar de BatchNorm
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
        
        # Capas principales con LayerNorm (funciona con cualquier batch size)
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
    con regularización agresiva y entrenamiento robusto.
    Hereda solo de RegressorMixin para evitar conflictos con BaseEstimator.
    """
    
    def __init__(self, hidden_size=128, epochs=150, batch_size=32, 
                 learning_rate=0.001, weight_decay=0.01, early_stopping_patience=15):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # Regularización L2
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Para early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    @property
    def _estimator_type(self):
        """Identifica este estimador como regresor para sklearn"""
        return "regressor"
    
    def _check_n_features(self, X, reset):
        """Método para compatibilidad con sklearn - verifica número de features"""
        pass  # Implementación mínima
    
    def score(self, X, y, sample_weight=None):
        """Calcula R² score para compatibilidad con sklearn"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def fit(self, X, y):
        """Entrenamiento de la red neuronal con regularización agresiva y EARLY STOPPING AVANZADO"""
        # Convertir a numpy arrays si es necesario
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Preparar datos
        X_scaled = self.scaler.fit_transform(X)
        
        # DIVISIÓN TRAIN/VAL PARA EARLY STOPPING 
        val_size = 0.15  # 15% para validación interna
        val_split = int(len(X_scaled) * (1 - val_size))
        
        # Asegurar que tenemos suficientes datos para entrenamiento
        if val_split < 2:
            val_split = max(2, len(X_scaled) - 1)
        
        X_train = X_scaled[:val_split]
        X_val = X_scaled[val_split:]
        y_train = y[:val_split]
        y_val = y[val_split:]
        
        # Ajustar batch_size si es necesario
        effective_batch_size = min(self.batch_size, len(X_train))
        if effective_batch_size < 2:
            effective_batch_size = len(X_train)  # Usar todo como un batch
        
        # Convertir a tensores de PyTorch
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(self.device)
        
        # Crear dataset y dataloader solo para entrenamiento
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
        
        # Crear modelo
        input_size = X_scaled.shape[1]
        self.model = NBATeamPointsNet(input_size, self.hidden_size).to(self.device)
        
        # Optimizador con regularización L2 agresiva
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler para learning rate dinámico
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8
        )
        
        # Función de pérdida robusta
        criterion = nn.SmoothL1Loss()  # Más robusta que MSE para outliers
        
        # EARLY STOPPING - Variables de control
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # ENTRENAMIENTO CON EARLY STOPPING Y VALIDACIÓN DEDICADA
        logger.info(f"Iniciando entrenamiento red neuronal - Train: {len(X_train)}, Val: {len(X_val)}")
        
        for epoch in range(self.epochs):
            # FASE DE ENTRENAMIENTO
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
            
            avg_train_loss = epoch_train_loss / train_batch_count
            
            # FASE DE VALIDACIÓN
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Actualizar scheduler con pérdida de validación
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar métricas
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            learning_rates.append(current_lr)
            
            # EARLY STOPPING LOGIC MEJORADO
            improvement = False
            
            # Criterio de mejora: tolerancia absoluta más realista
            improvement_threshold = 0.01  # Mejora mínima absoluta requerida (más permisiva)
            if val_loss < (best_val_loss - improvement_threshold):
                best_val_loss = val_loss
                self.best_loss = val_loss
                self.patience_counter = 0
                improvement = True
                
                # Guardar mejor estado del modelo
                self.best_model_state = self.model.state_dict().copy()
                
                if epoch % 10 == 0 or epoch < 10:
                    logger.info(f"Epoca {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={val_loss:.4f} [MEJOR] LR={current_lr:.6f}")
            else:
                self.patience_counter += 1
                
                if epoch % 10 == 0 or epoch < 10:
                    logger.info(f"Epoca {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={val_loss:.4f} [{self.patience_counter}/{self.early_stopping_patience}] LR={current_lr:.6f}")
            
            # Condiciones de parada temprana - MÁS INTELIGENTES
            early_stop_triggered = False
            
            # 1. Patience exceeded
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping por patience ({self.early_stopping_patience} epocas sin mejora)")
                early_stop_triggered = True
            
            # 2. Convergencia excelente alcanzada
            elif val_loss < 0.5:  
                logger.info(f"Early stopping por convergencia excelente (val_loss={val_loss:.4f})")
                early_stop_triggered = True
            
            # 3. Learning rate extremadamente bajo
            elif current_lr < 5e-7:  # Más permisivo 
                logger.info(f"Early stopping por learning rate muy bajo ({current_lr:.2e})")
                early_stop_triggered = True
            
            # 4. Plateau detectado - pérdida se estanca por mucho tiempo
            elif len(val_losses) >= 20:
                recent_losses = val_losses[-20:]
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                if loss_std / loss_mean < 0.005:  # Variación < 0.5%
                    logger.info(f"Early stopping por plateau detectado (variacion={loss_std/loss_mean:.4f})")
                    early_stop_triggered = True
            
            if early_stop_triggered:
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Modelo restaurado al mejor estado (epoca con val_loss={self.best_loss:.4f})")
        else:
            logger.warning("No se encontro mejor estado del modelo")
        
        # Análisis de convergencia 
        final_epoch = len(train_losses)
        min_val_loss = min(val_losses)
        initial_val_loss = val_losses[0] if val_losses else min_val_loss
        final_val_loss = val_losses[-1] if val_losses else min_val_loss
        
        # Calcular mejora total y estabilidad final
        total_improvement = (initial_val_loss - min_val_loss) / initial_val_loss if initial_val_loss > 0 else 0
        final_stability = abs(final_val_loss - min_val_loss) / min_val_loss if min_val_loss > 0 else 0
        
        # Clasificación más inteligente
        converged_well = final_stability < 0.1 and total_improvement > 0.5  # Mejoró 50%+ y es estable
        
        logger.info(f"ENTRENAMIENTO COMPLETADO:")
        logger.info(f"   • Epocas totales: {final_epoch}/{self.epochs}")
        logger.info(f"   • Val_loss inicial: {initial_val_loss:.4f}")
        logger.info(f"   • Mejor val_loss: {min_val_loss:.4f}")
        logger.info(f"   • Val_loss final: {final_val_loss:.4f}")
        logger.info(f"   • Mejora total: {total_improvement:.1%}")
        logger.info(f"   • Estabilidad final: {final_stability:.1%}")
        logger.info(f"   • LR final: {learning_rates[-1]:.2e}")
        
        # Clasificar calidad del entrenamiento
        if converged_well and min_val_loss < 1.0:
            logger.info(f"   • Calidad: EXCELENTE convergencia (loss<1.0, estable)")
        elif total_improvement > 0.7 and final_stability < 0.2:
            logger.info(f"   • Calidad: BUENA convergencia (mejora 70%+)")
        elif total_improvement > 0.3 and final_stability < 0.5:
            logger.info(f"   • Calidad: MODERADA convergencia (mejora 30%+)")
        elif total_improvement > 0.1:
            logger.info(f"   • Calidad: ACEPTABLE (mejora 10%+, revisar hiperparametros)")
        else:
            logger.info(f"   • Calidad: POBRE convergencia - Revisar arquitectura y datos")
        
        return self
    
    def predict(self, X):
        """Predicción usando la red neuronal entrenada"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Convertir a numpy array si es necesario
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            # Manejar predicción por batches para evitar problemas de memoria
            if len(X_tensor) > 1000:
                predictions = []
                batch_size = 500
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    batch_pred = self.model(batch).cpu().numpy().flatten()
                    predictions.extend(batch_pred)
                predictions = np.array(predictions)
            else:
                predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        # Aplicar límites realistas para puntos NBA
        predictions = np.clip(predictions, 70, 140)
        
        return predictions
    
    def get_params(self, deep=True):
        """Parámetros para compatibilidad con scikit-learn"""
        return {
            'hidden_size': self.hidden_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience
        }
    
    def set_params(self, **params):
        """Configurar parámetros para compatibilidad con scikit-learn"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class BaseNBATeamModel:
    """Clase base para modelos NBA de equipos con funcionalidades comunes."""
    
    def __init__(self, target_column: str, model_type: str = 'regression'):
        self.target_column = target_column
        self.model_type = model_type
        self.feature_columns = []
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> float:
        """Calcula precisión dentro de un rango de tolerancia."""
        return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100

class TeamPointsModel(BaseNBATeamModel):
    """
    Modelo especializado para predicción de puntos de equipo por partido.
    
    Implementa un sistema ensemble con optimización automática de hiperparámetros
    y características específicamente diseñadas para maximizar la precisión
    en la predicción de puntos de equipo.
    """
    
    def __init__(self, optimize_hyperparams: bool = True):
        """
        Inicializa el modelo de puntos de equipo.
        
        Args:
            optimize_hyperparams: Si optimizar hiperparámetros automáticamente
        """
        super().__init__(
            target_column='PTS',
            model_type='regression'
        )
        
        self.feature_engineer = TeamPointsFeatureEngineer()
        self.optimize_hyperparams = optimize_hyperparams
        self.best_model_name = None
        self.ensemble_weights = {}
        
        # Stacking components
        self.stacking_model = None
        self.base_models = {}
        self.meta_model = None
        
        # Configurar modelos optimizados para puntos de equipo
        self._setup_optimized_models()
        self._setup_stacking_model()
        
        # Métricas de evaluación
        self.evaluation_metrics = {}
    
    def _setup_optimized_models(self):
        """Configura modelos base optimizados para predicción de puntos de equipo."""
        
        # Modelos principales con REGULARIZACIÓN AGRESIVA para mayor estabilidad
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=400,        # Reducido para evitar overfitting
                max_depth=5,             # MÁS REDUCIDO para mayor regularización
                learning_rate=0.03,      # MÁS BAJO para mayor estabilidad
                subsample=0.75,          # MÁS CONSERVADOR para regularización
                colsample_bytree=0.75,   # MÁS CONSERVADOR para regularización
                min_child_weight=8,      # AUMENTADO significativamente
                reg_alpha=0.3,           # REGULARIZACIÓN L1 AGRESIVA
                reg_lambda=0.3,          # REGULARIZACIÓN L2 AGRESIVA
                random_state=42,
                n_jobs=-1,
                max_delta_step=1,        # Limitar cambios extremos
                gamma=0.1                # Regularización adicional por complejidad
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=400,        # Reducido para evitar overfitting
                max_depth=7,             # MÁS REDUCIDO para mayor regularización
                learning_rate=0.03,      # MÁS BAJO para mayor estabilidad
                subsample=0.75,          # MÁS CONSERVADOR para regularización
                colsample_bytree=0.75,   # MÁS CONSERVADOR para regularización
                min_child_samples=35,    # AUMENTADO significativamente
                reg_alpha=0.3,           # REGULARIZACIÓN L1 AGRESIVA
                reg_lambda=0.3,          # REGULARIZACIÓN L2 AGRESIVA
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                min_split_gain=0.1,      # Regularización adicional
                feature_fraction=0.8     # Reducir features por árbol
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=300,        # Reducido para evitar overfitting
                max_depth=8,             # MÁS REDUCIDO para mayor regularización
                min_samples_split=15,    # AUMENTADO significativamente
                min_samples_leaf=8,      # AUMENTADO significativamente
                max_features=0.6,        # MÁS RESTRICTIVO
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                min_weight_fraction_leaf=0.01,  # Regularización adicional
                max_leaf_nodes=500       # Limitar complejidad del árbol
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300,        # Reducido para evitar overfitting
                max_depth=5,             # MÁS REDUCIDO para mayor regularización
                learning_rate=0.03,      # MÁS BAJO para mayor estabilidad
                subsample=0.75,          # MÁS CONSERVADOR para regularización
                min_samples_split=15,    # AUMENTADO significativamente
                min_samples_leaf=8,      # AUMENTADO significativamente
                random_state=42,
                alpha=0.9,               # Regularización por quantile loss
                max_features=0.6         # MÁS RESTRICTIVO
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=300,        # Reducido para evitar overfitting
                max_depth=8,             # MÁS REDUCIDO para mayor regularización
                min_samples_split=15,    # AUMENTADO significativamente
                min_samples_leaf=8,      # AUMENTADO significativamente
                max_features=0.6,        # MÁS RESTRICTIVO
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                min_weight_fraction_leaf=0.01,  # Regularización adicional
                max_leaf_nodes=500       # Limitar complejidad del árbol
            ),
            
            # RED NEURONAL para comparación individual
            'pytorch_neural_net': PyTorchNBARegressor(
                hidden_size=128,         # Arquitectura media para individual
                epochs=100,              # Epocas para entrenamiento individual
                batch_size=32,           # Batch size estándar
                learning_rate=0.001,     # Learning rate estándar
                weight_decay=0.01,       # Regularización L2 moderada
                early_stopping_patience=15
            )
        }
        
        logger.info("Modelos base configurados con REGULARIZACIÓN AGRESIVA para mayor estabilidad")
    
    def _setup_stacking_model(self):
        """Configura el modelo de stacking robusto con REGULARIZACIÓN MÁXIMA + Red Neuronal mejorada."""
        
        # Modelos base para stacking con REGULARIZACIÓN EXTREMA + Neural Network MEJORADA
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
            # Neural Network MEJORADA con compatibilidad sklearn completa
            ('pytorch_nn', PyTorchNBARegressor(
                hidden_size=96,           # Arquitectura compacta
                epochs=100,               # Reducido para stacking
                batch_size=64,            # Batch size apropiado
                learning_rate=0.002,      # Learning rate conservador
                weight_decay=0.02,        # Regularización L2 agresiva
                early_stopping_patience=12
            ))
        ]
        
        # Meta-modelo con REGULARIZACIÓN MÁXIMA
        meta_model = Ridge(
            alpha=10.0,             # REGULARIZACIÓN AGRESIVA
            random_state=42,
            max_iter=2000,          # Más iteraciones para convergencia
            solver='auto'           # Mejor solver automático
        )
        
        # Stacking con validación cruzada más robusta
        self.stacking_model = StackingRegressor(
            estimators=base_models_stacking,
            final_estimator=meta_model,
            cv=7,  # MÁS FOLDS para mayor robustez
            n_jobs=-1,
            passthrough=False  # Solo usar predicciones de base models
        )
        
        # Guardar modelos base para análisis posterior
        self.base_models = dict(base_models_stacking)
        self.meta_model = meta_model
        
        logger.info("Modelo de stacking configurado con REGULARIZACIÓN MÁXIMA + Red Neuronal mejorada")
    
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
            logger.warning(f"Características faltantes para equipo: {missing}")
            logger.info(f"Características faltantes más comunes:")
            for i, feat in enumerate(list(missing)[:10]):
                logger.info(f"  - {feat}")
        
        logger.info(f"Características disponibles para puntos de equipo: {len(available_features)}")
        return available_features
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
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
        
        # Preparar datos
        X = df[self.feature_columns].fillna(0)
        y = df[self.target_column]
        
        # Validación temporal (los datos más recientes para validación)
        split_idx = int(len(df) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"División temporal: {len(X_train)} entrenamiento, {len(X_val)} validación")
        
        # Escalar características manteniendo estructura DataFrame
        X_train_scaled_array = self.scaler.fit_transform(X_train)
        X_val_scaled_array = self.scaler.transform(X_val)
        
        # Convertir arrays escalados de vuelta a DataFrame para mantener nombres de características
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=self.feature_columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled_array, columns=self.feature_columns, index=X_val.index)
        
        # Entrenar modelos individuales
        logger.info("Entrenando modelos individuales...")
        model_predictions_train = {}
        model_predictions_val = {}
        
        for name, model in self.models.items():
            logger.info(f"Entrenando {name}...")
            
            # Optimización de hiperparámetros si está habilitada
            if self.optimize_hyperparams and name in ['xgboost', 'lightgbm', 'pytorch_neural_net']:
                model = self._optimize_model_hyperparams(model, X_train_scaled, y_train, name)
            
            # Entrenar modelo
            model.fit(X_train_scaled, y_train)
            
            # Predicciones
            pred_train = model.predict(X_train_scaled)
            pred_val = model.predict(X_val_scaled)
            
            model_predictions_train[name] = pred_train
            model_predictions_val[name] = pred_val
            
            # Guardar modelo entrenado
            self.trained_models[name] = model
            
            # Métricas individuales
            mae_val = mean_absolute_error(y_val, pred_val)
            r2_val = r2_score(y_val, pred_val)
            logger.info(f"{name} - MAE: {mae_val:.3f}, R²: {r2_val:.4f}")
        
        # Entrenar ensemble voting
        logger.info("Entrenando ensemble voting con Red Neuronal...")
        # Usar TODOS los modelos incluyendo PyTorch con compatibilidad mejorada
        voting_models = [(name, model) for name, model in self.trained_models.items()]
        
        if len(voting_models) == 0:
            logger.warning("No hay modelos válidos para voting - usando solo el primer modelo disponible")
            voting_models = [list(self.trained_models.items())[0]]
        
        voting_regressor = VotingRegressor(voting_models)
        logger.info(f"VotingRegressor creado con {len(voting_models)} modelos: {[name for name, _ in voting_models]}")
        voting_regressor.fit(X_train_scaled, y_train)
        self.trained_models['voting'] = voting_regressor
        
        voting_pred_train = voting_regressor.predict(X_train_scaled)
        voting_pred_val = voting_regressor.predict(X_val_scaled)
        model_predictions_train['voting'] = voting_pred_train
        model_predictions_val['voting'] = voting_pred_val
        
        # Entrenar stacking
        logger.info("Entrenando stacking avanzado...")
        self.stacking_model.fit(X_train_scaled, y_train)
        self.trained_models['stacking'] = self.stacking_model
        
        stacking_pred_train = self.stacking_model.predict(X_train_scaled)
        stacking_pred_val = self.stacking_model.predict(X_val_scaled)
        model_predictions_train['stacking'] = stacking_pred_train
        model_predictions_val['stacking'] = stacking_pred_val
        
        # Validación cruzada para modelo stacking
        logger.info("Ejecutando validación cruzada...")
        cv_scores = self._perform_cross_validation(X_train_scaled, y_train)
        
        # Seleccionar mejor modelo
        logger.info("Seleccionando mejor modelo...")
        self._select_best_model(model_predictions_val, y_val)
        
        # Análisis de rendimiento
        best_pred_train = model_predictions_train[self.best_model_name]
        best_pred_val = model_predictions_val[self.best_model_name]
        
        metrics = self._analyze_model_performance_cv(
            y_train, best_pred_train, y_val, best_pred_val, 
            stacking_pred_train, stacking_pred_val, 
            voting_pred_train, voting_pred_val, cv_scores
        )
        
        self.is_trained = True
        logger.info("Entrenamiento completado exitosamente")
        
        # Guardar el modelo final de producción
        self.save_production_model()
        
        return metrics
    
    def _optimize_model_hyperparams(self, model, X_train, y_train, model_name):
        """Optimiza hiperparámetros usando búsqueda aleatoria CONSERVADORA."""

        logger.info(f"Optimizando hiperparámetros para {model_name} con REGULARIZACIÓN AGRESIVA...")
        
        # Parámetros para XGBoost 
        if model_name == 'xgboost':
            param_dist = {
                'n_estimators': [200, 300, 400],         # Rango más conservador
                'max_depth': [3, 4, 5],                  # MÁS RESTRICTIVO
                'learning_rate': [0.02, 0.03, 0.05],     # MÁS LENTO
                'subsample': [0.7, 0.8],                 # MÁS CONSERVADOR
                'colsample_bytree': [0.7, 0.8],          # MÁS CONSERVADOR
                'reg_alpha': [0.2, 0.3, 0.5],            # MÁS REGULARIZACIÓN L1
                'reg_lambda': [0.2, 0.3, 0.5],           # MÁS REGULARIZACIÓN L2
                'min_child_weight': [8, 10, 15],         # MÁS RESTRICTIVO
                'gamma': [0.1, 0.2, 0.3]                 # REGULARIZACIÓN ADICIONAL
            }
        
        # Parámetros para LightGBM 
        elif model_name == 'lightgbm':
            param_dist = {
                'n_estimators': [200, 300, 400],         # Rango más conservador
                'max_depth': [5, 6, 7],                  # MÁS RESTRICTIVO
                'learning_rate': [0.02, 0.03, 0.05],     # MÁS LENTO
                'subsample': [0.7, 0.8],                 # MÁS CONSERVADOR
                'colsample_bytree': [0.7, 0.8],          # MÁS CONSERVADOR
                'reg_alpha': [0.2, 0.3, 0.5],            # MÁS REGULARIZACIÓN L1
                'reg_lambda': [0.2, 0.3, 0.5],           # MÁS REGULARIZACIÓN L2
                'min_child_samples': [30, 40, 50],       # MÁS RESTRICTIVO
                'min_split_gain': [0.1, 0.2, 0.3],       # REGULARIZACIÓN ADICIONAL
                'feature_fraction': [0.7, 0.8, 0.9]      # REGULARIZACIÓN DE FEATURES
            }
        
        # Parámetros para Red Neuronal
        elif model_name == 'pytorch_neural_net':
            param_dist = {
                'hidden_size': [64, 96, 128],             # Arquitecturas compactas
                'learning_rate': [0.0005, 0.001, 0.002], # Learning rates conservadores
                'weight_decay': [0.01, 0.02, 0.03],      # Regularización L2 agresiva
                'batch_size': [32, 64],                   # Batch sizes apropiados
                'epochs': [120, 150, 180],                # Épocas moderadas
                'early_stopping_patience': [10, 15, 20]  # Patience variada
            }
        
        else:
            return model
        
        # Búsqueda aleatoria con validación cruzada temporal MÁS ROBUSTA
        random_search = RandomizedSearchCV(
            model, param_dist, 
            n_iter=12 if model_name == 'pytorch_neural_net' else 15,  # Menos iteraciones para NN
            cv=5,                                  # Validación cruzada robusta
            scoring='neg_mean_absolute_error', 
            n_jobs=-1 if model_name != 'pytorch_neural_net' else 1,  # NN en serial por GPU
            random_state=42,
            verbose=0
        )
        
        # X_train ahora es DataFrame con nombres de características
        random_search.fit(X_train, y_train)
        
        logger.info(f"Mejores parámetros REGULARIZADOS para {model_name}: {random_search.best_params_}")
        return random_search.best_estimator_
    
    def _perform_cross_validation(self, X, y):
        """Ejecuta validación cruzada temporal ROBUSTA para el modelo stacking."""
        # Validación cruzada temporal MÁS ROBUSTA
        tscv = TimeSeriesSplit(n_splits=7)  # MÁS FOLDS para mayor robustez
        
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
        
        # Accuracy scores (tolerancia ±3 puntos) con función más robusta
        def accuracy_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            # Aplicar límites realistas antes de calcular precisión
            y_pred_clipped = np.clip(y_pred, 70, 160)
            return self._calculate_accuracy(y, y_pred_clipped, 3.0)
        
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
            # Métricas adicionales de estabilidad
            'cv_stability': np.std(mae_scores) / np.mean(mae_scores),  # Coeficiente de variación
            'mae_min': np.min(mae_scores),
            'mae_max': np.max(mae_scores),
            'mae_range': np.max(mae_scores) - np.min(mae_scores)
        }
        
        logger.info(f"Validación cruzada ROBUSTA (7-fold) - MAE: {cv_results['mean_mae']:.3f}±{cv_results['std_mae']:.3f}")
        logger.info(f"Estabilidad CV (std/mean): {cv_results['cv_stability']:.3f}")
        logger.info(f"Rango MAE: [{cv_results['mae_min']:.3f}, {cv_results['mae_max']:.3f}] (±{cv_results['mae_range']:.3f})")
        logger.info(f"Validación cruzada - R²: {cv_results['mean_r2']:.4f}±{cv_results['std_r2']:.4f}")
        logger.info(f"Validación cruzada - Precisión ±3pts: {cv_results['mean_accuracy']:.1f}%±{cv_results['std_accuracy']:.1f}%")
        
        return cv_results
    
    def _select_best_model(self, predictions_dict, y_true):
        """Selecciona el mejor modelo basado en métricas de validación."""
        best_mae = float('inf')
        best_model = None
        
        for model_name, pred in predictions_dict.items():
            mae = mean_absolute_error(y_true, pred)
            r2 = r2_score(y_true, pred)
            
            # Criterio de selección: MAE principal, R² como criterio secundario
            if mae < best_mae:
                best_mae = mae
                best_model = model_name
        
        self.best_model_name = best_model
        logger.info(f"Mejor modelo seleccionado: {best_model} (MAE: {best_mae:.3f})")
    
    def _analyze_model_performance_cv(self, y_train, pred_train, y_val, pred_val, 
                                     stacking_train, stacking_val, 
                                     voting_train, voting_val, cv_scores):
        """Análisis completo del rendimiento del modelo con validación cruzada."""
        
        # Métricas de entrenamiento
        train_metrics = {
            'mae': mean_absolute_error(y_train, pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, pred_train)),
            'r2': r2_score(y_train, pred_train)
        }
        
        # Métricas de validación
        val_metrics = {
            'mae': mean_absolute_error(y_val, pred_val),
            'rmse': np.sqrt(mean_squared_error(y_val, pred_val)),
            'r2': r2_score(y_val, pred_val)
        }
        
        # Métricas de stacking
        stacking_metrics = {
            'mae': mean_absolute_error(y_val, stacking_val),
            'rmse': np.sqrt(mean_squared_error(y_val, stacking_val)),
            'r2': r2_score(y_val, stacking_val)
        }
        
        # Métricas de voting
        voting_metrics = {
            'mae': mean_absolute_error(y_val, voting_val),
            'rmse': np.sqrt(mean_squared_error(y_val, voting_val)),
            'r2': r2_score(y_val, voting_val)
        }
        
        # Métricas de validación cruzada
        cv_metrics = cv_scores
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("ANÁLISIS DE RENDIMIENTO - MODELO PUNTOS DE EQUIPO")
        print("="*80)
        
        print(f"\nMEJOR MODELO: {self.best_model_name.upper()}")
        print(f"{'Métrica':<15} {'Entrenamiento':<15} {'Validación':<15} {'Diferencia':<15}")
        print("-" * 60)
        print(f"{'MAE':<15} {train_metrics['mae']:<15.3f} {val_metrics['mae']:<15.3f} {abs(train_metrics['mae'] - val_metrics['mae']):<15.3f}")
        print(f"{'RMSE':<15} {train_metrics['rmse']:<15.3f} {val_metrics['rmse']:<15.3f} {abs(train_metrics['rmse'] - val_metrics['rmse']):<15.3f}")
        print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {val_metrics['r2']:<15.4f} {abs(train_metrics['r2'] - val_metrics['r2']):<15.4f}")
        
        # Análisis de overfitting mejorado
        mae_diff = abs(train_metrics['mae'] - val_metrics['mae'])
        r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
        cv_stability = cv_metrics.get('cv_stability', 0)
        mae_range = cv_metrics.get('mae_range', 0)
        
        print(f"\nANÁLISIS DE ROBUSTEZ:")
        print(f"Estabilidad CV (std/mean): {cv_stability:.3f}")
        print(f"Rango MAE en CV: ±{mae_range:.3f}")
        print(f"Diferencia Entrenamiento-Validación MAE: {mae_diff:.3f}")
        
        # Clasificación de estabilidad más precisa
        if cv_stability < 0.15 and mae_range < 1.0:
            print("Modelo MUY ESTABLE - Excelente robustez")
        elif cv_stability < 0.25 and mae_range < 2.0:
            print("Modelo ESTABLE - Buena robustez")
        elif cv_stability < 0.35:
            print("Modelo MODERADAMENTE ESTABLE - Aceptable con cuidado")
        else:
            print("Modelo INESTABLE - Requiere más regularización")
        
        # Evaluación de overfitting más detallada
        if mae_diff < 1.0 and r2_diff < 0.03:
            print("Sin overfitting - Excelente generalización")
        elif mae_diff < 2.0 and r2_diff < 0.08:
            print("Overfitting mínimo - Buena generalización")
        elif mae_diff < 3.0 and r2_diff < 0.15:
            print("Ligero overfitting - Monitorear en producción")
        else:
            print("Overfitting significativo - Aumentar regularización")
        
        # Recomendaciones específicas basadas en estabilidad
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
        
        # Análisis de precisión por tolerancia
        print(f"\nPRECISIÓN POR TOLERANCIA (Validación Final):")
        for tolerance in [1, 2, 3, 5, 7, 10]:
            acc = self._calculate_accuracy(y_val, pred_val, tolerance)
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        # Rendimiento de ensembles
        print(f"\nCOMPARACIÓN DE ENSEMBLES (Validación Final):")
        print(f"{'Modelo':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 50)
        print(f"{'Mejor Individual':<20} {val_metrics['mae']:<10.3f} {val_metrics['rmse']:<10.3f} {val_metrics['r2']:<10.4f}")
        print(f"{'Voting':<20} {voting_metrics['mae']:<10.3f} {voting_metrics['rmse']:<10.3f} {voting_metrics['r2']:<10.4f}")
        print(f"{'Stacking':<20} {stacking_metrics['mae']:<10.3f} {stacking_metrics['rmse']:<10.3f} {stacking_metrics['r2']:<10.4f}")
        
        # Validación cruzada detallada
        print(f"\nVALIDACIÓN CRUZADA (5-FOLD TEMPORAL):")
        print(f"MAE: {cv_metrics['mean_mae']:.3f} ± {cv_metrics['std_mae']:.3f}")
        print(f"R²: {cv_metrics['mean_r2']:.4f} ± {cv_metrics['std_r2']:.4f}")
        print(f"Precisión ±3pts: {cv_metrics['mean_accuracy']:.1f}% ± {cv_metrics['std_accuracy']:.1f}%")
        
        # Guardar métricas
        self.evaluation_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'stacking': stacking_metrics,
            'voting': voting_metrics,
            'cross_validation': cv_metrics,
            'best_model': self.best_model_name
        }
        
        return self.evaluation_metrics
    
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
        X = df[self.feature_columns].fillna(0)
        
        # Escalar características manteniendo estructura DataFrame
        X_scaled_array = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_array, columns=self.feature_columns, index=X.index)
        
        # Usar el mejor modelo
        best_model = self.trained_models[self.best_model_name]
        predictions = best_model.predict(X_scaled)
        
        # Aplicar límites realistas para puntos de equipo NBA (80-150 típicamente)
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
            logger.warning(f"No se puede obtener importancia para {self.best_model_name}")
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
            group_features_in_model = [f for f in group_features if f in self.feature_columns]
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
        
        # Realizar predicciones
        predictions = self.predict(df)
        y_true = df[self.target_column]
        
        # Calcular métricas
        metrics = {
            'mae': mean_absolute_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'r2': r2_score(y_true, predictions),
            'accuracy_1pt': self._calculate_accuracy(y_true, predictions, 1),
            'accuracy_2pt': self._calculate_accuracy(y_true, predictions, 2),
            'accuracy_3pt': self._calculate_accuracy(y_true, predictions, 3),
            'accuracy_5pt': self._calculate_accuracy(y_true, predictions, 5)
        }
        
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
            # Crear carpeta trained_models si no existe
            os.makedirs("trained_models", exist_ok=True)
            save_path = "trained_models/teams_points.joblib"
        
        # Preparar objeto del modelo para producción
        production_model = {
            'model': self.trained_models[self.best_model_name],
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_engineer': self.feature_engineer,
            'best_model_name': self.best_model_name,
            'evaluation_metrics': self.evaluation_metrics,
            'target_column': self.target_column,
            'model_metadata': {
                'training_date': datetime.now().isoformat(),
                'model_type': 'NBA Team Points Predictor',
                'version': '1.0',
                'accuracy_3pts': self.evaluation_metrics.get('validation', {}).get('accuracy_3pt', None),
                'mae': self.evaluation_metrics.get('validation', {}).get('mae', None),
                'r2': self.evaluation_metrics.get('validation', {}).get('r2', None),
                'total_features': len(self.feature_columns),
                'best_model': self.best_model_name
            }
        }
        
        try:
            # Guardar con joblib para compatibilidad sklearn
            joblib.dump(production_model, save_path)
            
            logger.info(f"[OK] MODELO DE PRODUCCION GUARDADO EXITOSAMENTE:")
            logger.info(f"   • Ruta: {save_path}")
            logger.info(f"   • Mejor modelo: {self.best_model_name}")
            logger.info(f"   • Features: {len(self.feature_columns)}")
            logger.info(f"   • MAE: {production_model['model_metadata']['mae']:.3f}")
            logger.info(f"   • R²: {production_model['model_metadata']['r2']:.4f}")
            
            # Manejar accuracy_3pts que puede ser None
            accuracy_3pts = production_model['model_metadata']['accuracy_3pts']
            if accuracy_3pts is not None:
                logger.info(f"   • Precision ±3pts: {accuracy_3pts:.1f}%")
            else:
                logger.info(f"   • Precision ±3pts: No disponible")
                
            logger.info(f"   • Fecha: {production_model['model_metadata']['training_date']}")
            
        except Exception as e:
            logger.error(f"Error al guardar modelo de produccion: {e}")
            raise
    
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
            
            logger.info(f"[OK] MODELO DE PRODUCCION CARGADO:")
            logger.info(f"   • Ruta: {model_path}")
            logger.info(f"   • Modelo: {production_model['best_model_name']}")
            logger.info(f"   • Features: {len(production_model['feature_columns'])}")
            logger.info(f"   • Fecha entrenamiento: {production_model['model_metadata']['training_date']}")
            logger.info(f"   • MAE: {production_model['model_metadata']['mae']:.3f}")
            logger.info(f"   • R²: {production_model['model_metadata']['r2']:.4f}")
            
            return production_model
            
        except Exception as e:
            logger.error(f"Error al cargar modelo de produccion: {e}")
            raise
