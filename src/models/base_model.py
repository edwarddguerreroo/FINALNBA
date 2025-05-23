import pandas as pd
import numpy as np
import joblib
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# Obtener el logger configurado en el módulo principal
logger = logging.getLogger(__name__)

class BaseNBAModel(ABC):
    """
    Clase base para todos los modelos de predicción NBA.
    Proporciona funcionalidades comunes y define la interfaz.
    """
    
    def __init__(self, target_column, model_type='regression', feature_columns=None):
        """
        Inicializa el modelo base.
        
        Args:
            target_column (str): Nombre de la columna objetivo a predecir
            model_type (str): 'regression' o 'classification'
            feature_columns (list): Lista de columnas de características específicas
        """
        self.target_column = target_column
        self.model_type = model_type
        self.feature_columns = feature_columns or []
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_fitted = False
        self.validation_scores = {}
        
        # Configurar modelos por defecto
        self._setup_default_models()
        
        # Datos de entrenamiento y prueba
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _setup_default_models(self):
        """Configura los modelos por defecto según el tipo"""
        if self.model_type == 'regression':
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }
        else:  # classification
            self.models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }
    
    @abstractmethod
    def get_feature_columns(self, df):
        """
        Método abstracto para obtener las columnas de características específicas.
        Debe ser implementado por cada modelo específico.
        """
        pass
    
    @abstractmethod
    def preprocess_target(self, df):
        """
        Método abstracto para preprocesar la variable objetivo.
        Debe ser implementado por cada modelo específico.
        """
        pass
    
    def prepare_data(self, df, test_size=0.2, time_split=True):
        """
        Prepara los datos para entrenamiento y validación.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            test_size (float): Proporción de datos para test
            time_split (bool): Si usar división temporal
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Obtener características específicas
        features = self.get_feature_columns(df)
        
        # Filtrar características disponibles
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"Características faltantes para {self.target_column}: {missing}")
        
        self.feature_columns = available_features
        
        # Preparar datos
        X = df[available_features].copy()
        y = self.preprocess_target(df)
        
        # Eliminar filas con valores faltantes en el objetivo
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # División temporal o aleatoria
        if time_split and 'Date' in df.columns:
            # Usar división temporal basada en fechas
            df_valid = df[valid_mask].copy()
            cutoff_date = df_valid['Date'].quantile(1 - test_size)
            train_mask = df_valid['Date'] <= cutoff_date
            
            X_train = X[train_mask]
            X_test = X[~train_mask]
            y_train = y[train_mask]
            y_test = y[~train_mask]
        else:
            # División aleatoria
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"Datos preparados para {self.target_column}: "
                   f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Guardar scaler
        self.scalers['main'] = scaler
        
        # Convertir a DataFrame manteniendo nombres de columnas
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Guardar datos
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train, use_scaling=True):
        """
        Entrena todos los modelos configurados.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            use_scaling (bool): Si aplicar escalado a las características
        """
        logger.info(f"Entrenando modelos para {self.target_column}")
        
        # Aplicar escalado si es necesario
        if use_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers['main'] = scaler
        else:
            X_train_scaled = X_train
        
        # Entrenar cada modelo
        for model_name, model in self.models.items():
            try:
                logger.info(f"Entrenando {model_name}")
                
                # Algunos modelos manejan mejor datos sin escalar
                if model_name in ['xgboost', 'lightgbm', 'random_forest']:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Guardar importancia de características
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(
                        zip(self.feature_columns, model.feature_importances_)
                    )
                
                logger.info(f"{model_name} entrenado exitosamente")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {str(e)}")
                
        self.is_fitted = True
    
    def validate_models(self, X_test, y_test):
        """
        Valida todos los modelos entrenados.
        
        Args:
            X_test: Características de test
            y_test: Variable objetivo de test
            
        Returns:
            dict: Métricas de validación por modelo
        """
        if not self.is_fitted:
            raise ValueError("Los modelos no han sido entrenados")
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Aplicar escalado si fue usado en entrenamiento
                if 'main' in self.scalers and model_name not in ['xgboost', 'lightgbm', 'random_forest']:
                    X_test_scaled = self.scalers['main'].transform(X_test)
                    predictions = model.predict(X_test_scaled)
                else:
                    predictions = model.predict(X_test)
                
                # Calcular métricas según el tipo de modelo
                if self.model_type == 'regression':
                    mse = mean_squared_error(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    
                    results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'predictions': predictions
                    }
                    
                    logger.info(f"{model_name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
                    
                else:  # classification
                    accuracy = accuracy_score(y_test, predictions)
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'predictions': predictions,
                        'classification_report': classification_report(y_test, predictions)
                    }
                    
                    logger.info(f"{model_name} - Accuracy: {accuracy:.3f}")
                    
            except Exception as e:
                logger.error(f"Error validando {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.validation_scores = results
        return results
    
    def get_best_model(self):
        """
        Obtiene el mejor modelo basado en las métricas de validación.
        
        Returns:
            tuple: (nombre_modelo, modelo, score)
        """
        if not self.validation_scores:
            raise ValueError("No hay scores de validación disponibles")
        
        best_score = float('inf') if self.model_type == 'regression' else 0
        best_model_name = None
        
        for model_name, scores in self.validation_scores.items():
            if 'error' in scores:
                continue
                
            if self.model_type == 'regression':
                score = scores.get('rmse', float('inf'))
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                score = scores.get('accuracy', 0)
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError("No se encontró un modelo válido")
        
        return best_model_name, self.models[best_model_name], best_score
    
    def predict(self, X, model_name=None):
        """
        Realiza predicciones con el modelo especificado o el mejor.
        
        Args:
            X: Características para predicción
            model_name (str): Nombre del modelo a usar (None para el mejor)
            
        Returns:
            array: Predicciones
        """
        if not self.is_fitted:
            raise ValueError("Los modelos no han sido entrenados")
        
        if model_name is None:
            model_name, model, _ = self.get_best_model()
        else:
            model = self.models[model_name]
        
        # Aplicar escalado si es necesario
        if 'main' in self.scalers and model_name not in ['xgboost', 'lightgbm', 'random_forest']:
            X_scaled = self.scalers['main'].transform(X)
            return model.predict(X_scaled)
        else:
            return model.predict(X)
    
    def save_model(self, filepath, model_name=None):
        """
        Guarda el modelo especificado o el mejor.
        
        Args:
            filepath (str): Ruta donde guardar el modelo
            model_name (str): Nombre del modelo a guardar (None para el mejor)
        """
        if model_name is None:
            model_name, _, _ = self.get_best_model()
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers.get('main'),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'model_name': model_name,
            'validation_scores': self.validation_scores
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo {model_name} guardado en {filepath}")
    
    def load_model(self, filepath):
        """
        Carga un modelo guardado.
        
        Args:
            filepath (str): Ruta del modelo guardado
        """
        model_data = joblib.load(filepath)
        
        self.models[model_data['model_name']] = model_data['model']
        if model_data['scaler']:
            self.scalers['main'] = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.model_type = model_data['model_type']
        self.validation_scores = model_data['validation_scores']
        self.is_fitted = True
        
        logger.info(f"Modelo cargado desde {filepath}")
    
    def get_feature_importance_summary(self):
        """
        Obtiene un resumen de la importancia de características.
        
        Returns:
            pd.DataFrame: DataFrame con importancia promedio de características
        """
        if not self.feature_importance:
            return pd.DataFrame()
        
        # Calcular importancia promedio
        all_features = set()
        for importances in self.feature_importance.values():
            all_features.update(importances.keys())
        
        importance_summary = []
        for feature in all_features:
            scores = [
                importances.get(feature, 0) 
                for importances in self.feature_importance.values()
            ]
            importance_summary.append({
                'feature': feature,
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores)
            })
        
        df = pd.DataFrame(importance_summary)
        return df.sort_values('mean_importance', ascending=False)
    
    def get_train_test_data(self):
        """
        Obtiene los datos de entrenamiento y prueba.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if any(x is None for x in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Los datos no han sido preparados. Llame a prepare_data primero.")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_prediction_context(self, player_name, df, n_games=5):
        """
        Obtiene contexto específico para la predicción.
        Debe ser implementado por las clases hijas.
        
        Args:
            player_name (str): Nombre del jugador
            df (pd.DataFrame): DataFrame con los datos
            n_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Contexto de predicción específico
        """
        return {} 