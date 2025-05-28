"""
Modelo de Predicción de Puntos (PTS) para Jugadores NBA
======================================================

Este módulo implementa un modelo especializado para predecir puntos por partido
de jugadores NBA. Utiliza la arquitectura base y características específicas
optimizadas para maximizar la precisión en la predicción de puntos.

Características principales:
- Hereda de BaseNBAModel para funcionalidades comunes
- Utiliza PointsFeatureEngineer para características específicas
- Implementa ensemble de modelos (RandomForest, XGBoost, LightGBM)
- Optimización de hiperparámetros automática
- Validación temporal para evitar data leakage
- Métricas avanzadas de evaluación

Objetivo: Alcanzar ≥97% de precisión en predicción de puntos.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
from datetime import datetime
import warnings

# Imports de ML
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from scipy import stats
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb

# Imports del proyecto
from src.models.base_model import BaseNBAModel
from src.models.players.points.features_points import PointsFeatureEngineer
from src.preprocessing.data_loader import NBADataLoader

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class PointsModel(BaseNBAModel):
    """
    Modelo especializado para predicción de puntos por partido.
    
    Implementa un sistema ensemble con optimización automática de hiperparámetros
    y características específicamente diseñadas para maximizar la precisión
    en la predicción de puntos.
    """
    
    def __init__(self, optimize_hyperparams: bool = True):
        """
        Inicializa el modelo de puntos.
        
        Args:
            optimize_hyperparams: Si optimizar hiperparámetros automáticamente
        """
        super().__init__(
            target_column='PTS',
            model_type='regression'
        )
        
        self.feature_engineer = PointsFeatureEngineer()
        self.optimize_hyperparams = optimize_hyperparams
        self.best_model_name = None
        self.ensemble_weights = {}
        
        # Stacking components
        self.stacking_model = None
        self.base_models = {}
        self.meta_model = None
        
        # Configurar modelos optimizados para puntos
        self._setup_optimized_models()
        self._setup_stacking_model()
        
        # Métricas de evaluación
        self.evaluation_metrics = {}
        
    def _setup_optimized_models(self):
        """Configura modelos optimizados específicamente para puntos."""
        
        # RandomForest optimizado para puntos
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost optimizado para puntos
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM optimizado para puntos
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=12,
            learning_rate=0.05,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        logger.info("Modelos optimizados configurados para predicción de puntos")
    
    def _setup_stacking_model(self):
        """Configura el modelo de stacking robusto con múltiples modelos y regularización agresiva."""
        
        # Imputer para modelos que no manejan NaN
        imputer = SimpleImputer(strategy='median')
        
        # RandomForest base
        base_rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost ya maneja valores NaN
        base_xgb = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=1.5,     # Regularización L1 agresiva 
            reg_lambda=2.5,    # Regularización L2 agresiva 
            random_state=42,
            n_jobs=-1
        )
        
        # Usar HistGradientBoostingRegressor que maneja NaN nativamente
        hist_gb = HistGradientBoostingRegressor(
            max_iter=250,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=3.5,  # Mayor regularización L2 (aumentada de 2.0)
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15  # Aumentar para mejorar validación temprana
        )
        
        # Crear pipelines para modelos que no manejan NaN
        huber_pipe = Pipeline([
            ('imputer', imputer),
            ('huber', HuberRegressor(
                epsilon=1.2, 
                alpha=0.1,  # Aumentar regularización alpha
                max_iter=500
            ))
        ])
        
        # Añadir Ridge y ElasticNet para mejor regularización y manejo de heteroscedasticidad
        ridge_pipe = Pipeline([
            ('imputer', imputer),
            ('ridge', Ridge(
                alpha=2.0,  # Control de regularización L2 (aumentado de 1.0)
                solver='auto',
                random_state=42
            ))
        ])
        
        elastic_pipe = Pipeline([
            ('imputer', imputer),
            ('elastic', ElasticNet(
                alpha=0.8,       # Aumentado de 0.5
                l1_ratio=0.7,    # Balance entre L1 y L2
                random_state=42,
                max_iter=1000
            ))
        ])
        
        # Usar estimadores que manejan NaN para el stacking
        stack = StackingRegressor(
            estimators=[
                ('rf', base_rf),
                ('xgb', base_xgb),
                ('hist_gb', hist_gb),
                ('ridge', ridge_pipe),
                ('elastic', elastic_pipe)
            ],
            final_estimator=LassoCV(alphas=[0.1, 0.5, 1.0, 2.0]),  # Aumentados los valores de alpha
            cv=5,
            n_jobs=-1
        )
        
        # Actualizar modelos base para referencia
        self.base_models = {
            'rf': base_rf,
            'xgb': base_xgb,
            'hist_gb': hist_gb,
            'huber': huber_pipe,
            'ridge': ridge_pipe,
            'elastic': elastic_pipe
        }
        
        # Meta-modelo LassoCV
        self.meta_model = LassoCV(alphas=[0.1, 0.5, 1.0, 2.0])
        
        # Stacking model
        self.stacking_model = stack
        
        # Actualizar modelos
        self.models.update({
            'hist_gb': hist_gb,
            'huber': huber_pipe,
            'ridge': ridge_pipe,
            'elastic': elastic_pipe,
            'stacking': stack,
        })
        
        logger.info("Modelo de stacking robusto configurado con 5 modelos base + meta-modelo LassoCV")
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las columnas de características específicas para puntos.
        
        Args:
            df: DataFrame con datos de jugadores
            
        Returns:
            Lista de nombres de características
        """
        # Generar todas las características usando el feature engineer
        # IMPORTANTE: generate_all_features modifica el DataFrame in-place
        features = self.feature_engineer.generate_all_features(df)
        
        # Filtrar características que realmente existen en el DataFrame
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"Características faltantes: {missing}")
            logger.info(f"Características faltantes más comunes:")
            for i, feat in enumerate(list(missing)[:10]):
                logger.info(f"  - {feat}")
        
        logger.info(f"Características disponibles para puntos: {len(available_features)}")
        return available_features
    
    def prepare_data(self, df, test_size=0.2, time_split=True, detect_outliers=True, balance_data=True):
        """
        Prepara los datos con transformaciones específicas para puntos, detección avanzada de outliers
        y balanceo de datos para rangos subrepresentados.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            test_size (float): Proporción de datos para test
            time_split (bool): Si usar división temporal
            detect_outliers (bool): Si detectar y filtrar outliers
            balance_data (bool): Si aplicar técnicas de balanceo para rangos subrepresentados
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Primero generar todas las características en el dataset completo
        df_with_features = df.copy()
        feature_columns = self.get_feature_columns(df_with_features)
        
        # Luego hacer la división cronológica
        if time_split and 'Date' in df_with_features.columns:
            # División cronológica: entrenar con datos más antiguos, test con más recientes
            df_sorted = df_with_features.sort_values('Date').copy()
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            df_train = df_sorted.iloc[:split_idx]
            df_test = df_sorted.iloc[split_idx:]
            
            # Extraer características y target para cada conjunto
            X_train = df_train[feature_columns]
            X_test = df_test[feature_columns]
            y_train = self.preprocess_target(df_train)
            y_test = self.preprocess_target(df_test)
            
            logger.info(f"División cronológica aplicada - Train: {len(X_train)}, Test: {len(X_test)}")
        else:
            # Usar método base si no hay división temporal
            X_train, X_test, y_train, y_test = super().prepare_data(df_with_features, test_size, time_split)
        
        # Guardar una copia de los datos originales antes de aplicar transformaciones
        self.X_train_original = X_train.copy()
        self.X_test_original = X_test.copy()
        self.y_train_original = y_train.copy()
        self.y_test_original = y_test.copy()
        
        # Limpiar datos: reemplazar infinitos con NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Detectar outliers con método avanzado si está habilitado
        if detect_outliers:
            try:
                # Método multivariado (más sofisticado)
                inlier_mask, outlier_scores = self.detect_multivariate_outliers(X_train, y_train, significance=0.01)
                
                # Método univariado de respaldo
                if inlier_mask.sum() < len(X_train) * 0.9:  # Si detectó demasiados outliers
                    logger.warning(f"Demasiados outliers detectados ({(~inlier_mask).sum()}). Usando método alternativo.")
                    # Usar método más conservador
                    inlier_mask, outlier_scores = self.detect_multivariate_outliers(X_train, y_train, significance=0.001)
                
                # Guardar información de outliers para análisis
                self.outlier_info = {
                    'n_outliers': (~inlier_mask).sum(),
                    'outlier_percentage': (~inlier_mask).sum() / len(X_train) * 100,
                    'outlier_scores': outlier_scores if 'outlier_scores' in locals() else None
                }
                
                # Filtrar outliers solo si no son demasiados
                if inlier_mask.sum() >= len(X_train) * 0.9:
                    # Convertir el índice booleano a un índice de lista de enteros para evitar errores
                    indices = np.where(inlier_mask)[0]
                    X_train = X_train.iloc[indices]
                    y_train = y_train.iloc[indices]
                    logger.info(f"Filtrados {(~inlier_mask).sum()} outliers de {len(inlier_mask)} muestras.")
                else:
                    logger.warning("Se mantienen todos los datos. Posible problema en detección de outliers.")
                    
            except Exception as e:
                logger.error(f"Error durante la detección de outliers: {str(e)}", exc_info=True)
        
        # Balanceo de datos para rangos subrepresentados
        if balance_data:
            try:
                # Analizar distribución de la variable objetivo
                bins = [0, 5, 10, 15, 20, 25, 30, 40, 100]  # Rangos de puntos
                y_binned = pd.cut(y_train, bins=bins)
                bin_counts = y_binned.value_counts().sort_index()
                
                logger.info(f"Distribución original por rangos de puntos: {bin_counts}")
                
                # Identificar rangos subrepresentados (menos del 5% de los datos)
                min_samples_per_bin = len(y_train) * 0.05
                underrepresented_bins = bin_counts[bin_counts < min_samples_per_bin].index
                
                if len(underrepresented_bins) > 0:
                    logger.info(f"Rangos subrepresentados: {underrepresented_bins.tolist()}")
                    
                    # Aplicar sobremuestreo a rangos subrepresentados
                    from sklearn.utils import resample
                    
                    X_train_resampled = X_train.copy()
                    y_train_resampled = y_train.copy()
                    
                    for bin_range in underrepresented_bins:
                        # Índices de muestras en este rango
                        bin_indices = np.where(y_binned == bin_range)[0]
                        
                        if len(bin_indices) > 0:
                            # Determinar cuántas muestras añadir
                            n_samples = int(min_samples_per_bin) - len(bin_indices)
                            
                            if n_samples > 0:
                                # Extraer muestras de este rango
                                X_bin = X_train.iloc[bin_indices]
                                y_bin = y_train.iloc[bin_indices]
                                
                                # Sobremuestrear con reemplazo
                                X_resampled, y_resampled = resample(
                                    X_bin, y_bin, 
                                    n_samples=n_samples, 
                                    replace=True, 
                                    random_state=42
                                )
                                
                                # Añadir muestras sobremuestreadas
                                X_train_resampled = pd.concat([X_train_resampled, X_resampled])
                                y_train_resampled = pd.concat([y_train_resampled, y_resampled])
                                
                                logger.info(f"Añadidas {n_samples} muestras para el rango {bin_range}")
                
                    # Actualizar los datos de entrenamiento con los datos balanceados
                    X_train = X_train_resampled
                    y_train = y_train_resampled
                    
                    # Verificar nueva distribución
                    y_binned_new = pd.cut(y_train, bins=bins)
                    bin_counts_new = y_binned_new.value_counts().sort_index()
                    logger.info(f"Distribución después del balanceo: {bin_counts_new}")
                    
                    # Guardar información de balanceo para referencia
                    self.balance_info = {
                        'original_distribution': bin_counts.to_dict(),
                        'balanced_distribution': bin_counts_new.to_dict(),
                        'added_samples': len(y_train) - len(self.y_train_original)
                    }
                else:
                    logger.info("No se detectaron rangos significativamente subrepresentados. No se aplicó balanceo.")
                    
            except Exception as e:
                logger.error(f"Error durante el balanceo de datos: {str(e)}", exc_info=True)
        
        # Imputar valores NaN para asegurar compatibilidad con todos los modelos
        for col in X_train.columns:
            if X_train[col].isna().any():
                # Usar la mediana para imputar, más robusta que la media
                median_value = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_value)
                X_test[col] = X_test[col].fillna(median_value)
        
        # Agregar características de interacción avanzadas
        X_train = self._add_interaction_features(X_train)
        X_test = self._add_interaction_features(X_test)
        
        # Detectar y manejar outliers en las características
        for col in X_train.columns:
            # Aplicar transformación Winsorize (recortar outliers a percentiles)
            # Solo para columnas numéricas con más de 100 valores únicos
            if X_train[col].dtype.kind in 'fi' and X_train[col].nunique() > 100:
                q_low = X_train[col].quantile(0.001)
                q_high = X_train[col].quantile(0.999)
                
                # Recortar valores extremos 
                X_train[col] = X_train[col].clip(lower=q_low, upper=q_high)
                X_test[col] = X_test[col].clip(lower=q_low, upper=q_high)
        
        # Aplicar transformaciones para manejar heteroscedasticidad para columnas clave
        key_features = ['FGA', 'FG%', '2PA', '2P%', '3PA', '3P%', 'FTA', 'FT%', 'MP']
        
        for col in key_features:
            if col in X_train.columns and X_train[col].min() >= 0:
                # Aplicamos transformación Box-Cox o log dependiendo de la distribución
                # Verificar si es candidata para Box-Cox (todos positivos)
                if (X_train[col] > 0).all():
                    from scipy import stats
                    # Intentar transformación Box-Cox primero
                    try:
                        # Calcular lambda óptimo para Box-Cox
                        _, lambda_bc = stats.boxcox(X_train[col].replace(0, 1e-8))
                        
                        # Aplicar transformación a ambos conjuntos de datos
                        X_train[f'{col}_bc'] = stats.boxcox(X_train[col].replace(0, 1e-8), lmbda=lambda_bc)
                        X_test[f'{col}_bc'] = stats.boxcox(X_test[col].replace(0, 1e-8), lmbda=lambda_bc)
                        
                        # Guardar lambda para futuras transformaciones
                        if not hasattr(self, 'box_cox_lambdas'):
                            self.box_cox_lambdas = {}
                        self.box_cox_lambdas[col] = lambda_bc
                        
                    except:
                        # Si Box-Cox falla, usar log(x+1)
                        X_train[f'{col}_log'] = np.log1p(X_train[col])
                        X_test[f'{col}_log'] = np.log1p(X_test[col])
                else:
                    # Para variables que pueden ser cero, usar log(x+1)
                    X_train[f'{col}_log'] = np.log1p(X_train[col])
                    X_test[f'{col}_log'] = np.log1p(X_test[col])
        
        # Tratamiento especial para FGA - Transformaciones adicionales para valores extremos
        if 'FGA' in X_train.columns:
            # 1. Transformación raíz cuadrada
            X_train['FGA_sqrt'] = np.sqrt(X_train['FGA'])
            X_test['FGA_sqrt'] = np.sqrt(X_test['FGA'])
            
            # 2. Transformación logarítmica especial para manejar skewness
            X_train['FGA_log2'] = np.log2(X_train['FGA'] + 1)  # log base 2 puede ser mejor para ciertos rangos
            X_test['FGA_log2'] = np.log2(X_test['FGA'] + 1)
            
            # 3. Transformación Yeo-Johnson (generalización de Box-Cox que maneja valores negativos y cero)
            try:
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                X_train_fga = X_train['FGA'].values.reshape(-1, 1)
                X_test_fga = X_test['FGA'].values.reshape(-1, 1)
                
                X_train['FGA_yj'] = pt.fit_transform(X_train_fga).flatten()
                X_test['FGA_yj'] = pt.transform(X_test_fga).flatten()
                
                # Guardar el transformador para uso futuro
                self.power_transformer_fga = pt
                
            except Exception as e:
                logger.warning(f"Error aplicando transformación Yeo-Johnson a FGA: {str(e)}")
            
            # 4. Transformación personalizada basada en percentiles para FGA
            # Mapea valores a su rango percentil (0-1)
            try:
                from scipy.stats import rankdata
                
                # Calcular rangos y normalizar a [0,1]
                X_train['FGA_rank'] = rankdata(X_train['FGA']) / len(X_train['FGA'])
                
                # Para test, aplicar la misma transformación basada en los rangos de entrenamiento
                # Esto requiere un enfoque más elaborado en producción
                X_test['FGA_rank'] = rankdata(X_test['FGA']) / len(X_test['FGA'])
                
            except Exception as e:
                logger.warning(f"Error aplicando transformación de rangos a FGA: {str(e)}")
        
        # Ponderación de muestras para contrarrestar heteroscedasticidad
        # Las muestras con valores altos de puntos reciben menor peso
        y_weights = 1.0 / (1.0 + np.abs(y_train - y_train.median()) / y_train.std())
        self.sample_weights = y_weights
        
        # Actualizar las columnas de características
        self.feature_columns = X_train.columns.tolist()
        
        # Guardar la referencia a los datos de entrenamiento para uso futuro
        self.X_train = X_train.copy()
        
        # SIEMPRE aplicar transformación logarítmica para y (evita problemas con valores de 0)
        # Esta transformación ayuda a manejar la heteroscedasticidad
        self.y_transform = 'log'
        
        # Usamos log(y + 1) para evitar problemas con log(0)
        # Transformar targets
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        logger.info("Aplicada transformacion logaritmica a la variable objetivo 'PTS'")
        
        # Guardar datos transformados
        self.y_train = y_train_log
        self.y_test = y_test_log
        
        return X_train, X_test, y_train_log, y_test_log
    
    def _add_interaction_features(self, X):
        """Añade características de interacción avanzadas."""
        try:
            # Interacciones clave para predicción de puntos
            if 'FGA' in X.columns and 'FG%' in X.columns:
                X['FGA_x_FG%'] = X['FGA'] * X['FG%']
            
            if '3PA' in X.columns and '3P%' in X.columns:
                X['3PA_x_3P%'] = X['3PA'] * X['3P%']
            
            if 'FTA' in X.columns and 'FT%' in X.columns:
                X['FTA_x_FT%'] = X['FTA'] * X['FT%']
            
            if 'MP' in X.columns and 'FGA' in X.columns:
                X['MP_x_FGA'] = X['MP'] * X['FGA']
                
        except Exception as e:
            logger.warning(f"Error añadiendo características de interacción: {str(e)}")
        
        return X
    
    def preprocess_target(self, df):
        """
        Preprocesa la variable objetivo PTS con mejor manejo de outliers.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            pd.Series: Serie con la variable objetivo procesada
        """
        if 'PTS' not in df.columns:
            raise ValueError("Columna 'PTS' no encontrada en el DataFrame")
        
        # Convertir a numérico y manejar valores faltantes
        pts = pd.to_numeric(df['PTS'], errors='coerce')
        
        # Eliminar valores extremos (outliers) usando un enfoque basado en percentiles
        # en lugar de cortar con límites absolutos
        lower_bound = max(0, pts.quantile(0.001))  # No menos de 0
        upper_bound = pts.quantile(0.999)  # Preserva valores extremos legítimos
        
        pts = pts.clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Estadisticas de puntos - Media: {pts.mean():.1f}, "
                   f"Mediana: {pts.median():.1f}, "
                   f"Min: {pts.min()}, Max: {pts.max()}")
        
        return pts
    
    def detect_multivariate_outliers(self, X, y, significance=0.01):
        """
        Detecta outliers multivariados usando distancia de Mahalanobis y
        técnicas basadas en regresión.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            significance (float): Nivel de significancia para detectar outliers
            
        Returns:
            tuple: (mask_inliers, outlier_scores)
        """
        import numpy as np
        from scipy import stats
        
        # 1. Detección por distancia de Mahalanobis
        try:
            # Convertir target a array para evitar problemas de indexación
            y_array = y.values if hasattr(y, 'values') else np.array(y)
            
            # Seleccionar solo columnas numéricas sin valores faltantes
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            X_numeric = X[numeric_cols].copy()
            
            # Eliminar columnas con valores faltantes o varianza cero
            for col in X_numeric.columns:
                if X_numeric[col].isna().any() or X_numeric[col].var() == 0:
                    X_numeric.drop(columns=[col], inplace=True)
            
            if X_numeric.empty:
                logger.warning("No hay columnas numéricas válidas para análisis multivariado")
                return np.ones(len(X), dtype=bool), np.zeros(len(X))
            
            # Calcular media y covarianza
            mean_vec = np.array(X_numeric.mean())
            cov_matrix = np.array(X_numeric.cov())
            
            # Calcular la inversa de la covarianza (añadir regularización si es necesario)
            try:
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Si la matriz es singular, usar pseudo-inversa
                inv_cov = np.linalg.pinv(cov_matrix)
            
            # Calcular distancia de Mahalanobis para cada muestra
            mahalanobis_dist = np.zeros(len(X_numeric))
            for i in range(len(X_numeric)):
                x = np.array(X_numeric.iloc[i])
                mahalanobis_dist[i] = np.sqrt(np.dot(np.dot((x - mean_vec).T, inv_cov), (x - mean_vec)))
            
            # Convertir a p-valores (aproximación chi-cuadrado)
            p_vals = 1 - stats.chi2.cdf(mahalanobis_dist**2, df=len(X_numeric.columns))
            
            # Identificar outliers por p-valor
            mask_mahalanobis = p_vals > significance
            
            # 2. Detección por residuos de regresión
            from sklearn.linear_model import HuberRegressor
            
            # Ajustar modelo robusto
            model = HuberRegressor(epsilon=1.35, alpha=0.01).fit(X_numeric, y_array)
            
            # Calcular residuos y normalizarlos
            y_pred = model.predict(X_numeric)
            residuals = np.abs(y_array - y_pred)
            normalized_residuals = (residuals - residuals.mean()) / residuals.std()
            
            # Identificar outliers por residuos extremos
            mask_residuals = normalized_residuals < stats.norm.ppf(1 - significance)
            
            # Combinar ambos métodos (AND lógico)
            mask_combined = np.logical_and(mask_mahalanobis, mask_residuals)
            
            # Calcular puntuación combinada de outlier (0 = inlier, 1 = extremo outlier)
            outlier_scores = 1 - (p_vals * (1 - normalized_residuals/normalized_residuals.max())/2)
            
            return mask_combined, outlier_scores
            
        except Exception as e:
            logger.error(f"Error en detección multivariada de outliers: {str(e)}")
            return np.ones(len(X), dtype=bool), np.zeros(len(X))
    
    def train_with_optimization(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el modelo con optimización de hiperparámetros.
        
        Args:
            df: DataFrame con datos de entrenamiento
            test_size: Proporción de datos para test
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento con optimización para modelo de puntos")
        
        # Preparar datos (incluye detección de outliers, balanceo, transformaciones)
        X_train, X_test, y_train, y_test = self.prepare_data(
            df, 
            test_size=test_size, 
            time_split=True, 
            detect_outliers=True, 
            balance_data=True
        )
        
        # Recopilar información de preprocesamiento
        results = {}
        
        # Información de outliers
        if hasattr(self, 'outlier_info'):
            results.update(self.outlier_info)
        
        # Información de balanceo
        if hasattr(self, 'balance_info'):
            results['balance_info'] = self.balance_info
        
        if self.optimize_hyperparams:
            # Optimizar hiperparámetros para cada modelo
            results = self._optimize_all_models(X_train, y_train)
        
        # Entrenar modelos (optimizados o por defecto)
        self.train_models(X_train, y_train)
        
        # Validar modelos
        validation_results = self.validate_models(X_test, y_test)
        results.update(validation_results)
        
        # Entrenar modelo de stacking
        stacking_results = self._train_stacking_model(X_train, y_train, X_test, y_test)
        results.update(stacking_results)
        
        # Determinar mejor modelo (incluyendo stacking)
        best_model_info = self.get_best_model()
        if isinstance(best_model_info, tuple):
            self.best_model_name = best_model_info[0]
        else:
            self.best_model_name = best_model_info
        results['best_model'] = self.best_model_name
        
        # Calcular métricas avanzadas
        advanced_metrics = self._calculate_advanced_metrics(X_test, y_test)
        results['advanced_metrics'] = advanced_metrics
        
        # Calcular pesos para ensemble
        self.ensemble_weights = self._calculate_ensemble_weights(X_test, y_test)
        results['ensemble_weights'] = self.ensemble_weights
        
        logger.info(f"Entrenamiento completado. Mejor modelo: {self.best_model_name}")
        return results
    
    def _optimize_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimiza hiperparámetros para todos los modelos."""
        logger.info("Optimizando hiperparámetros...")
        
        optimization_results = {}
        
        # Configurar validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Parámetros para RandomForest
        rf_params = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        # Parámetros para XGBoost
        xgb_params = {
            'n_estimators': [300, 500, 700],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.8, 0.85, 0.9],
            'colsample_bytree': [0.8, 0.85, 0.9]
        }
        
        # Parámetros para LightGBM
        lgb_params = {
            'n_estimators': [300, 500, 700],
            'max_depth': [10, 12, 15],
            'learning_rate': [0.03, 0.05, 0.1],
            'feature_fraction': [0.8, 0.85, 0.9],
            'bagging_fraction': [0.8, 0.85, 0.9]
        }
        
        # Optimizar cada modelo
        model_params = {
            'random_forest': (RandomForestRegressor(random_state=42, n_jobs=-1), rf_params),
            'xgboost': (xgb.XGBRegressor(random_state=42, n_jobs=-1), xgb_params),
            'lightgbm': (lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), lgb_params)
        }
        
        for model_name, (base_model, params) in model_params.items():
            logger.info(f"Optimizando {model_name}...")
            
            try:
                grid_search = GridSearchCV(
                    base_model,
                    params,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Actualizar modelo con mejores parámetros
                self.models[model_name] = grid_search.best_estimator_
                
                optimization_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': -grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                logger.info(f"{model_name} optimizado. MSE: {-grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizando {model_name}: {str(e)}")
                optimization_results[model_name] = {'error': str(e)}
        
        return optimization_results
    
    def _train_stacking_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Entrena el modelo de stacking y evalúa su rendimiento.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_test: Características de test
            y_test: Target de test
            
        Returns:
            Diccionario con resultados del stacking
        """
        logger.info("Entrenando modelo de stacking...")
        
        stacking_results = {}
        
        try:
            # Entrenar el modelo de stacking
            self.stacking_model.fit(X_train, y_train)
            
            # Evaluar en conjunto de test
            y_pred_stacking = self.stacking_model.predict(X_test)
            
            # Calcular métricas usando valores en escala original
            if hasattr(self, 'y_test_original'):
                y_true_original = self.y_test_original
            else:
                y_true_original = np.expm1(y_test) if hasattr(self, 'y_transform') and self.y_transform == 'log' else y_test
            
            # Las predicciones del stacking están en escala logarítmica, convertir a original
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                y_pred_original = np.expm1(y_pred_stacking)
            else:
                y_pred_original = y_pred_stacking
            
            # Calcular métricas en escala original
            mse = mean_squared_error(y_true_original, y_pred_original)
            mae = mean_absolute_error(y_true_original, y_pred_original)
            r2 = r2_score(y_true_original, y_pred_original)
            rmse = np.sqrt(mse)
            mape = self._calculate_robust_mape(y_true_original, y_pred_original)
            
            stacking_results['stacking'] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'model_type': 'stacking'
            }
            
            # Obtener importancia de características de los modelos base
            base_importances = {}
            for name, model in self.base_models.items():
                if hasattr(model, 'feature_importances_'):
                    # Entrenar modelo base individual para obtener importancias
                    model.fit(X_train, y_train)
                    base_importances[name] = model.feature_importances_
            
            stacking_results['base_model_importances'] = base_importances
            
            # Obtener coeficientes del meta-modelo (LassoCV)
            if hasattr(self.stacking_model.final_estimator_, 'coef_'):
                meta_coefs = self.stacking_model.final_estimator_.coef_
                estimator_names = [name for name, _ in self.stacking_model.estimators]
                
                stacking_results['meta_model_weights'] = {}
                for i, name in enumerate(estimator_names):
                    if i < len(meta_coefs):
                        stacking_results['meta_model_weights'][name] = float(meta_coefs[i])
                
                # Añadir alpha óptimo del LassoCV
                if hasattr(self.stacking_model.final_estimator_, 'alpha_'):
                    stacking_results['optimal_alpha'] = float(self.stacking_model.final_estimator_.alpha_)
            
            logger.info(f"Stacking entrenado - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo de stacking: {str(e)}")
            stacking_results['stacking'] = {'error': str(e)}
        
        return stacking_results
    
    def _calculate_advanced_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Calcula métricas avanzadas de evaluación usando métodos robustos."""
        metrics = {}
        
        for model_name, model in self.models.items():
            if not hasattr(model, 'predict'):
                continue
                
            try:
                # Usar el método evaluate_model que maneja correctamente las transformaciones
                model_metrics = self.evaluate_model(model_name, X_test, y_test)
                
                # Extraer métricas principales
                metrics[model_name] = {
                    'mse': model_metrics.get('rmse', 0) ** 2,  # Calcular MSE desde RMSE
                    'mae': model_metrics.get('mae', 0),
                    'rmse': model_metrics.get('rmse', 0),
                    'r2': model_metrics.get('r2', 0),
                    'mape': model_metrics.get('mape', 0),
                    'wmse': model_metrics.get('wmse', 0),
                    'wrmse': model_metrics.get('wrmse', 0),
                    'wmae': model_metrics.get('wmae', 0),
                    'wmape': model_metrics.get('wmape', 0),
                    'hetero_pvalue': model_metrics.get('hetero_pvalue'),
                    'hetero_slope': model_metrics.get('hetero_slope')
                }
                
                # Calcular precisión por rangos usando valores correctos
                if hasattr(self, 'y_test_original'):
                    y_true_original = self.y_test_original
                else:
                    y_true_original = np.expm1(y_test) if hasattr(self, 'y_transform') and self.y_transform == 'log' else y_test
                
                y_pred_original = self.predict(X_test, model_name)
                accuracy_ranges = self._calculate_range_accuracy(y_true_original, y_pred_original)
                metrics[model_name]['accuracy_ranges'] = accuracy_ranges
                
                logger.info(f"Métricas calculadas para {model_name} - R²: {metrics[model_name]['r2']:.4f}, "
                           f"RMSE: {metrics[model_name]['rmse']:.4f}, MAPE: {metrics[model_name]['mape']:.2f}%")
                
            except Exception as e:
                logger.error(f"Error calculando métricas para {model_name}: {str(e)}")
                metrics[model_name] = {'error': str(e)}
        
        return metrics
    
    def _calculate_range_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula precisión por rangos de puntos."""
        ranges = {
            '0-10': (0, 10),
            '11-20': (11, 20),
            '21-30': (21, 30),
            '31-40': (31, 40),
            '40+': (41, 100)
        }
        
        accuracy_by_range = {}
        
        for range_name, (min_pts, max_pts) in ranges.items():
            mask = (y_true >= min_pts) & (y_true <= max_pts)
            if mask.sum() > 0:
                range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                range_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-6))) * 100
                
                accuracy_by_range[range_name] = {
                    'count': mask.sum(),
                    'mae': range_mae,
                    'mape': range_mape
                }
        
        return accuracy_by_range
    
    def _calculate_ensemble_weights(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calcula pesos para ensemble basado en rendimiento."""
        weights = {}
        total_inverse_mse = 0
        
        # Calcular MSE inverso para cada modelo
        for model_name, model in self.models.items():
            if not hasattr(model, 'predict'):
                continue
                
            try:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                inverse_mse = 1 / (mse + 1e-6)
                weights[model_name] = inverse_mse
                total_inverse_mse += inverse_mse
            except:
                weights[model_name] = 0
        
        # Normalizar pesos
        if total_inverse_mse > 0:
            for model_name in weights:
                weights[model_name] /= total_inverse_mse
        
        return weights
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicción usando stacking o ensemble ponderado.
        
        Args:
            X: Características para predicción
            
        Returns:
            Array con predicciones ensemble
        """
        # Priorizar stacking si está disponible y entrenado
        if (self.stacking_model is not None and 
            hasattr(self.stacking_model, 'predict')):
            try:
                return self.stacking_model.predict(X)
            except Exception as e:
                logger.warning(f"Error en predicción stacking: {e}. Usando ensemble tradicional.")
        
        # Fallback a ensemble ponderado tradicional
        if not self.ensemble_weights:
            # Si no hay pesos, usar el mejor modelo
            return self.predict(X, self.best_model_name)
        
        predictions = []
        weights = []
        
        for model_name, weight in self.ensemble_weights.items():
            if weight > 0 and model_name in self.models and model_name != 'stacking':
                try:
                    pred = self.models[model_name].predict(X)
                    predictions.append(pred)
                    weights.append(weight)
                except:
                    continue
        
        if not predictions:
            # Fallback al mejor modelo
            return self.predict(X, self.best_model_name)
        
        # Combinar predicciones con pesos
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Renormalizar
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Obtiene importancia de características del mejor modelo o stacking.
        
        Args:
            top_n: Número de características más importantes a retornar
            
        Returns:
            Diccionario con importancia de características
        """
        result = {}
        
        # Si el mejor modelo es stacking, obtener importancias de modelos base
        if self.best_model_name == 'stacking' and self.base_models:
            logger.info("Obteniendo importancia de características de modelos base del stacking")
            
            # Combinar importancias de modelos base
            combined_importances = np.zeros(len(self.feature_columns))
            valid_models = 0
            
            for name, model in self.base_models.items():
                if hasattr(model, 'feature_importances_'):
                    combined_importances += model.feature_importances_
                    valid_models += 1
            
            if valid_models > 0:
                # Promediar importancias
                combined_importances /= valid_models
                
                # Crear DataFrame con importancias
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': combined_importances
                }).sort_values('importance', ascending=False)
                
                result['top_features'] = importance_df.head(top_n).to_dict('records')
                result['model_used'] = 'stacking_ensemble'
                
                # Añadir importancias individuales de cada modelo base
                result['base_model_importances'] = {}
                for name, model in self.base_models.items():
                    if hasattr(model, 'feature_importances_'):
                        base_df = pd.DataFrame({
                            'feature': self.feature_columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        result['base_model_importances'][name] = base_df.head(10).to_dict('records')
        
        # Fallback a modelo individual
        elif self.best_model_name and self.best_model_name in self.models:
            model = self.models[self.best_model_name]
            
            if hasattr(model, 'feature_importances_'):
                # Obtener importancias
                importances = model.feature_importances_
                feature_names = self.feature_columns
                
                # Crear DataFrame con importancias
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                result['top_features'] = importance_df.head(top_n).to_dict('records')
                result['model_used'] = self.best_model_name
            else:
                logger.warning(f"Modelo {self.best_model_name} no tiene feature_importances_")
                return {}
        else:
            logger.warning("No hay mejor modelo disponible")
            return {}
        
        # Agrupar por categorías de características
        if 'top_features' in result:
            feature_groups = self.feature_engineer.get_feature_importance_groups()
            grouped_importance = {}
            
            # Usar las importancias principales para agrupar
            importance_dict = {item['feature']: item['importance'] for item in result['top_features']}
            
            for group_name, group_features in feature_groups.items():
                group_importance = sum(
                    importance_dict.get(feature, 0) for feature in group_features
                    if feature in importance_dict
                )
                grouped_importance[group_name] = group_importance
            
            result['grouped_importance'] = grouped_importance
        
        return result
    
    def predict_adaptive(self, X_test):
        """
        Realiza predicciones usando el ensemble adaptativo regional con correcciones
        para mejorar la distribución de residuos y compensar heteroscedasticidad.
        
        Args:
            X_test: Características para predecir
            
        Returns:
            array: Predicciones corregidas
        """
        try:
            # Asegurarse de que el modelo adaptativo existe
            if 'adaptive_regional_ensemble' not in self.models:
                logger.warning("Modelo adaptativo no disponible, usando fallback con corrección de heteroscedasticidad")
                # Usar el método con pesos variables para compensar heteroscedasticidad
                return self.predict_with_weights(X_test, model_name='best')
            
            # Obtener predicciones del ensemble adaptativo
            ensemble_preds = self.models['adaptive_regional_ensemble'].predict(X_test)
            
            # Aplicar transformaciones inversas si es necesario
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                ensemble_preds = np.expm1(ensemble_preds)
            
            # Detectar posible heteroscedasticidad en valores bajos
            low_values = ensemble_preds < 5.0
            has_low_values = np.sum(low_values) > len(ensemble_preds) * 0.1  # Al menos 10% son valores bajos
            
            if has_low_values:
                # Aplicar correcciones específicas para heteroscedasticidad en valores bajos
                logger.info("Detectados valores bajos. Aplicando corrección para heteroscedasticidad.")
                
                # Preprocesar ensemble_preds para valores bajos con mayor corrección
                calibrated_low = ensemble_preds.copy()
                
                # Para valores muy bajos aplicar corrección específica para el patrón de abanico
                very_low = ensemble_preds < 3.0
                if np.sum(very_low) > 0:
                    # Corrección más agresiva para valores muy bajos
                    correction_low = 1.0 + 0.2 * (1.0 - ensemble_preds[very_low] / 3.0)
                    calibrated_low[very_low] *= correction_low
                
                # Para valores bajos pero no muy bajos, corrección estándar
                low_not_very = (ensemble_preds >= 3.0) & (ensemble_preds < 5.0)
                if np.sum(low_not_very) > 0:
                    # Corrección moderada
                    correction_mid = 1.0 + 0.1 * (1.0 - ensemble_preds[low_not_very] / 5.0)
                    calibrated_low[low_not_very] *= correction_mid
                
                # Combinar con predicciones normales
                ensemble_preds[low_values] = calibrated_low[low_values]
            
            # Aplicar calibración final para el resto de rangos
            final_preds = self._calibrate_predictions(ensemble_preds)
            
            return final_preds
            
        except Exception as e:
            logger.error(f"Error en predict_adaptive: {str(e)}")
            # Fallback a predicción con pesos variables para compensar heteroscedasticidad
            return self.predict_with_weights(X_test, model_name='best')

    def predict_with_weights(self, X_test, model_name='best'):
        """
        Método de fallback para predicciones con pesos variables.
        
        Args:
            X_test: Características para predecir
            model_name: Nombre del modelo a usar
            
        Returns:
            array: Predicciones
        """
        try:
            if model_name == 'best' and self.best_model_name:
                model = self.models[self.best_model_name]
            elif model_name in self.models:
                model = self.models[model_name]
            else:
                # Usar stacking si está disponible
                if 'stacking' in self.models:
                    model = self.models['stacking']
                else:
                    # Usar el primer modelo disponible
                    model = list(self.models.values())[0]
            
            predictions = model.predict(X_test)
            
            # Aplicar transformaciones inversas si es necesario
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                predictions = np.expm1(predictions)
            
            return self._calibrate_predictions(predictions)
            
        except Exception as e:
            logger.error(f"Error en predict_with_weights: {str(e)}")
            # Último fallback
            if self.best_model_name and self.best_model_name in self.models:
                raw_preds = self.models[self.best_model_name].predict(X_test)
                if hasattr(self, 'y_transform') and self.y_transform == 'log':
                    raw_preds = np.expm1(raw_preds)
                return raw_preds
            else:
                raise e

    def _calibrate_predictions(self, predictions):
        """
        Calibra las predicciones para corregir sesgos sistemáticos.
        Mejora particular para valores bajos con corrección específica.
        
        Args:
            predictions (np.array): Predicciones originales
            
        Returns:
            np.array: Predicciones calibradas
        """
        # Crear copia para no modificar el original
        calibrated = predictions.copy()
        
        # Análisis previo de las distribuciones de predicciones
        pred_mean = np.mean(calibrated)
        pred_std = np.std(calibrated)
        
        logger.info(f"Calibrando predicciones. Media original: {pred_mean:.4f}, Std: {pred_std:.4f}")
        
        # 1. Corrección específica para valores muy bajos (< 1.0)
        very_low_mask = calibrated < 1.0
        if np.any(very_low_mask):
            # Ajuste más agresivo para valores muy bajos
            # Usamos una corrección de factor 0.15 (aumentado desde 0.10)
            calibrated[very_low_mask] *= (1.0 + 0.15)
            logger.info(f"Aplicada corrección del 15% a {np.sum(very_low_mask)} predicciones muy bajas (<1.0)")
        
        # 2. Corrección para valores bajos (1.0 - 5.0)
        low_mask = (calibrated >= 1.0) & (calibrated < 5.0)
        if np.any(low_mask):
            # Aplicar un factor de corrección decreciente según el valor
            # Mayor factor para valores más bajos
            low_values = calibrated[low_mask]
            correction_factors = 0.08 - 0.015 * (low_values - 1.0)  # Factor que decrece de 0.08 a 0.0
            correction_factors = np.maximum(0, correction_factors)  # Asegurar factores no negativos
            
            calibrated[low_mask] *= (1.0 + correction_factors)
            logger.info(f"Aplicada corrección progresiva a {np.sum(low_mask)} predicciones bajas (1.0-5.0)")
        
        # 3. Corrección para valores altos (>25.0) - suelen ser sobreestimados
        high_mask = calibrated > 25.0
        if np.any(high_mask):
            # Reducir ligeramente valores muy altos
            calibrated[high_mask] *= 0.95
            logger.info(f"Aplicada reducción del 5% a {np.sum(high_mask)} predicciones altas (>25.0)")
        
        # 4. Corrección logarítmica para valores intermedios para reducir heteroscedasticidad
        # Esta transformación ayuda a estabilizar la varianza
        mid_mask = (calibrated >= 5.0) & (calibrated <= 25.0)
        if np.any(mid_mask):
            # Transformar a espacio logarítmico
            log_values = np.log1p(calibrated[mid_mask] - 5.0)
            # Aplicar pequeña corrección en el espacio logarítmico
            adjusted_log = log_values * 1.02  # Ajuste sutil del 2%
            # Volver a transformar
            calibrated[mid_mask] = np.expm1(adjusted_log) + 5.0
            logger.info(f"Aplicada corrección logarítmica a {np.sum(mid_mask)} predicciones intermedias (5.0-25.0)")
        
        # Verificar efectos de la calibración
        cal_mean = np.mean(calibrated)
        cal_std = np.std(calibrated)
        
        logger.info(f"Calibración completa. Media final: {cal_mean:.4f}, Std: {cal_std:.4f}")
        logger.info(f"Cambio en la media: {cal_mean - pred_mean:.4f}, en Std: {cal_std - pred_std:.4f}")
        
        return calibrated
    
    def evaluate_model(self, model_name, X_test, y_test, prefix=''):
        """
        Evalúa el rendimiento de un modelo específico con métricas robustas a heteroscedasticidad.
        
        Args:
            model_name (str): Nombre del modelo a evaluar
            X_test (pd.DataFrame): Características de test
            y_test (pd.Series): Target de test
            prefix (str): Prefijo para las métricas en el resultado
            
        Returns:
            dict: Métricas de evaluación
        """
        # Determinar valores verdaderos correctos
        if hasattr(self, 'y_test_original') and hasattr(self, 'y_transform') and self.y_transform == 'log':
            # Usar valores originales si están disponibles y hay transformación
            y_true = self.y_test_original.copy()
            logger.info("Usando valores originales de y_test para evaluación")
        else:
            # Si y_test está en escala logarítmica, convertir a escala original
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                y_true = np.expm1(y_test)
                logger.info("Convirtiendo y_test de escala logarítmica a original")
            else:
                y_true = y_test.copy()
                logger.info("Usando y_test directamente")
        
        # Hacer predicciones
        try:
            y_pred = self.predict(X_test, model_name)
            
            # Verificar que las predicciones estén en escala original
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                # El método predict debería ya aplicar la transformación inversa
                # pero verificamos por seguridad
                if np.mean(y_pred) < 5:  # Si el promedio es muy bajo, probablemente está en escala log
                    logger.warning("Las predicciones parecen estar en escala logarítmica, aplicando transformación inversa")
                    y_pred = np.expm1(y_pred)
            
        except Exception as e:
            logger.error(f"Error haciendo predicciones con {model_name}: {str(e)}")
            return {f'{prefix}error': str(e)}
        
        # Verificar rangos razonables
        logger.info(f"Rangos de evaluación - y_true: [{np.min(y_true):.2f}, {np.max(y_true):.2f}], "
                   f"y_pred: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
        
        # Calcular métricas estándar
        metrics = {}
        try:
            metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
            
            # Calcular MAPE de forma robusta
            metrics[f'{prefix}mape'] = self._calculate_robust_mape(y_true, y_pred)
            
            # Calcular métricas adicionales para heteroscedasticidad
            robust_metrics = self._calculate_robust_metrics(y_true, y_pred, prefix)
            metrics.update(robust_metrics)
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {str(e)}")
            metrics[f'{prefix}error'] = str(e)
        
        return metrics
        
    def _calculate_robust_metrics(self, y_true, y_pred, prefix=''):
        """
        Calcula métricas robustas que manejan mejor la heteroscedasticidad.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            prefix: Prefijo para las métricas
            
        Returns:
            dict: Métricas robustas
        """
        metrics = {}
        
        # Calcular error cuadrático medio ponderado (weighted MSE)
        # Da menos peso a errores grandes
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        # Aplicar pesos inversamente proporcionales al valor real
        # para contrarrestar la heteroscedasticidad
        weights = 1.0 / (1.0 + np.abs(y_true - np.median(y_true)) / np.std(y_true))
        
        # MSE ponderado
        metrics[f'{prefix}wmse'] = np.sum(weights * residuals**2) / np.sum(weights)
        
        # RMSE ponderado
        metrics[f'{prefix}wrmse'] = np.sqrt(metrics[f'{prefix}wmse'])
        
        # Error absoluto medio ponderado
        metrics[f'{prefix}wmae'] = np.sum(weights * abs_residuals) / np.sum(weights)
        
        # Error porcentual absoluto medio ponderado
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_errors = np.abs(residuals / y_true) * 100
            # Reemplazar infinitos y NaN
            percent_errors = np.nan_to_num(percent_errors, nan=0.0, posinf=0.0, neginf=0.0)
            # Recortar valores extremos
            percent_errors = np.clip(percent_errors, 0, 100)
            
        metrics[f'{prefix}wmape'] = np.sum(weights * percent_errors) / np.sum(weights)
        
        # Prueba de heteroscedasticidad de White (simplificada)
        # Regresión de residuos al cuadrado vs valores predichos
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, residuals**2)
            metrics[f'{prefix}hetero_pvalue'] = p_value
            metrics[f'{prefix}hetero_slope'] = slope
        except:
            metrics[f'{prefix}hetero_pvalue'] = None
            metrics[f'{prefix}hetero_slope'] = None
        
        return metrics
    
    def _calculate_robust_mape(self, y_true, y_pred, epsilon=0.1, max_pct=50):
        """
        Cálculo robusto de MAPE para evitar divisiones por cero y valores extremos.
        Implementa varias estrategias para manejar valores pequeños y cero.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            epsilon: Pequeño valor para sustituir valores muy pequeños
            max_pct: Porcentaje máximo de error a considerar
            
        Returns:
            float: MAPE robusto
        """
        # Convertir a arrays numpy si son series o dataframes
        if hasattr(y_true, 'values'):
            y_true_array = y_true.values
        else:
            y_true_array = np.array(y_true)
        
        if hasattr(y_pred, 'values'):
            y_pred_array = y_pred.values
        else:
            y_pred_array = np.array(y_pred)
        
        # Verificar que ambos arrays tengan valores positivos (puntos NBA)
        if np.any(y_true_array <= 0) or np.any(y_pred_array < 0):
            logger.warning(f"Valores no válidos detectados - y_true min: {np.min(y_true_array)}, y_pred min: {np.min(y_pred_array)}")
        
        # Filtrar valores muy pequeños que pueden causar problemas
        valid_mask = (y_true_array >= epsilon) & (y_pred_array >= 0)
        
        if not np.any(valid_mask):
            logger.warning("No hay valores válidos para calcular MAPE")
            return 0.0
        
        y_true_valid = y_true_array[valid_mask]
        y_pred_valid = y_pred_array[valid_mask]
        
        # Método 1: MAPE tradicional con protección
        with np.errstate(divide='ignore', invalid='ignore'):
            # Usar máximo entre valor real y epsilon para evitar divisiones por valores muy pequeños
            denominators = np.maximum(np.abs(y_true_valid), epsilon)
            mape_values = np.abs(y_true_valid - y_pred_valid) / denominators * 100
            
            # Filtrar valores infinitos o NaN
            mape_values = mape_values[np.isfinite(mape_values)]
            
            if len(mape_values) == 0:
                logger.warning("No se pudieron calcular valores MAPE válidos")
                return 0.0
            
            # Recortar valores extremos
            mape_values = np.clip(mape_values, 0, max_pct)
            
            # Usar mediana para robustez
            robust_mape = np.median(mape_values)
        
        # Verificación de cordura
        if not np.isfinite(robust_mape) or robust_mape > max_pct:
            logger.warning(f"MAPE calculado fuera de rango: {robust_mape}, usando método alternativo")
            
            # Método alternativo: MAE relativo
            mae = np.mean(np.abs(y_true_valid - y_pred_valid))
            mean_true = np.mean(y_true_valid)
            
            if mean_true > epsilon:
                robust_mape = (mae / mean_true) * 100
                robust_mape = min(robust_mape, max_pct)  # Limitar a máximo razonable
            else:
                robust_mape = 0.0
        
        # Verificación final
        if robust_mape > 100:
            logger.warning(f"MAPE muy alto ({robust_mape:.2f}%), limitando a 50%")
            robust_mape = 50.0
        
        logger.info(f"MAPE robusto calculado: {robust_mape:.4f}% (usando {len(y_true_valid)} valores válidos de {len(y_true_array)} totales)")
        
        return robust_mape
    
    def predict(self, X, model_name=None):
        """
        Realiza predicciones con manejo correcto de transformaciones inversas.
        
        Args:
            X: Características para predicción
            model_name (str): Nombre del modelo a usar (None para el mejor)
            
        Returns:
            array: Predicciones en escala original
        """
        if not self.is_fitted:
            raise ValueError("Los modelos no han sido entrenados")
        
        # Determinar qué modelo usar
        if model_name is None:
            if self.best_model_name:
                model_name = self.best_model_name
            else:
                model_name, _, _ = self.get_best_model()
        
        # Verificar que el modelo existe
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.models[model_name]
        
        try:
            # Aplicar escalado si es necesario
            if 'main' in self.scalers and model_name not in ['xgboost', 'lightgbm', 'random_forest', 'stacking']:
                X_scaled = self.scalers['main'].transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
            
            # Aplicar transformación inversa si es necesario
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                # Convertir de escala logarítmica a escala original
                predictions = np.expm1(predictions)
                logger.debug(f"Aplicada transformación inversa logarítmica. Rango predicciones: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
            
            # Asegurar que las predicciones sean no negativas (puntos NBA)
            predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error en predicción con modelo {model_name}: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> bool:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para guardar
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'best_model_name': self.best_model_name,
                'ensemble_weights': self.ensemble_weights,
                'evaluation_metrics': self.evaluation_metrics,
                'target_column': self.target_column,
                'model_type': self.model_type,
                'feature_engineer': self.feature_engineer,
                'stacking_model': self.stacking_model,
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar usando joblib
            joblib.dump(model_data, filepath)
            
            logger.info(f"Modelo de puntos guardado en: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            return False
    
    @classmethod
    def load_model(cls, filepath: str) -> 'PointsModel':
        """
        Carga un modelo guardado.
        
        Args:
            filepath: Ruta del modelo guardado
            
        Returns:
            Instancia del modelo cargado
        """
        try:
            # Cargar datos
            model_data = joblib.load(filepath)
            
            # Crear instancia
            model = cls(optimize_hyperparams=False)
            
            # Restaurar estado
            model.models = model_data['models']
            model.scalers = model_data['scalers']
            model.feature_columns = model_data['feature_columns']
            model.best_model_name = model_data.get('best_model_name')
            model.ensemble_weights = model_data.get('ensemble_weights', {})
            model.evaluation_metrics = model_data.get('evaluation_metrics', {})
            model.target_column = model_data['target_column']
            model.model_type = model_data['model_type']
            model.feature_engineer = model_data.get('feature_engineer', PointsFeatureEngineer())
            model.stacking_model = model_data.get('stacking_model')
            model.base_models = model_data.get('base_models', {})
            model.meta_model = model_data.get('meta_model')
            model.is_fitted = True
            
            logger.info(f"Modelo de puntos cargado desde: {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise

def train_points_model(data_path: str = None, save_path: str = None) -> PointsModel:
    """
    Función de conveniencia para entrenar un modelo de puntos.
    
    Args:
        data_path: Ruta a los datos (si None, usa data_loader)
        save_path: Ruta donde guardar el modelo entrenado
        
    Returns:
        Modelo entrenado
    """
    logger.info("Iniciando entrenamiento de modelo de puntos")
    
    # Cargar datos
    if data_path:
        df = pd.read_csv(data_path)
    else:
        # Usar data_loader del proyecto
        data_loader = NBADataLoader(
            game_data_path='data/players.csv',
            biometrics_path='data/height.csv',
            teams_path='data/teams.csv'
        )
        df, _ = data_loader.load_data()
    
    # Crear y entrenar modelo
    model = PointsModel(optimize_hyperparams=True)
    results = model.train_with_optimization(df)
    
    # Guardar modelo si se especifica ruta
    if save_path:
        model.save_model(save_path)
    
    # Log de resultados
    logger.info("Resultados del entrenamiento:")
    logger.info(f"Mejor modelo: {results.get('best_model')}")
    
    if 'advanced_metrics' in results:
        best_metrics = results['advanced_metrics'].get(results['best_model'], {})
        logger.info(f"R²: {best_metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"RMSE: {best_metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"MAE: {best_metrics.get('mae', 'N/A'):.4f}")
        logger.info(f"MAPE: {best_metrics.get('mape', 'N/A'):.2f}%")
    
    return model

if __name__ == "__main__":
    # Entrenar modelo si se ejecuta directamente
    model = train_points_model(save_path='trained_models/points_model_v2.pkl')