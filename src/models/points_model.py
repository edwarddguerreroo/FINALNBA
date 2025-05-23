import pandas as pd
import numpy as np
import logging
from .base_model import BaseNBAModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LassoCV, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge

logger = logging.getLogger(__name__)

class PointsModel(BaseNBAModel):
    """
    Modelo específico para predecir puntos (PTS) de jugadores NBA.
    Hereda de BaseNBAModel y define características específicas para puntos.
    
    Mejoras implementadas:
    - Manejo de valores NaN mediante imputación
    - Corrección del MAPE para manejar valores cercanos a cero
    - Transformación logarítmica para reducir heteroscedasticidad en valores altos
    - Modelos más robustos para manejar outliers (HuberRegressor)
    - Mejor caracterización de los datos con polinomios
    - Stacking de modelos para optimizar predicciones
    """
    
    def __init__(self):
        super().__init__(target_column='PTS', model_type='regression')
        
    def _setup_default_models(self):
        """
        Configura modelos mejorados para predicción de puntos
        que manejan mejor outliers y heteroscedasticidad.
        """
        # Primero llamar al método de la clase base para inicializar los modelos estándar
        super()._setup_default_models()
        
        # Crear una pipeline de imputación para manejar NaN
        imputer = SimpleImputer(strategy='median')
        
        # Obtener modelos base ya inicializados
        base_rf = self.models['random_forest']
        
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
        
        # Actualizar modelos
        self.models.update({
            'hist_gb': hist_gb,
            'huber': huber_pipe,
            'ridge': ridge_pipe,
            'elastic': elastic_pipe,
            'stacking': stack,
        })
        
        logger.info("Modelos mejorados configurados para prediccion de puntos")
        
    def get_feature_columns(self, df):
        """
        Define las características específicas para predicción de puntos.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            list: Lista de columnas de características para puntos
        """
        # Características básicas de puntuación
        basic_features = [
            'MP',  # Minutos jugados
            'FGA', 'FG%',  # Intentos y porcentaje de tiros de campo
            '3PA', '3P%',  # Intentos y porcentaje de triples
            'FTA', 'FT%',  # Intentos y porcentaje de tiros libres
            '2PA', '2P%',  # Intentos y porcentaje de dobles
            'TS%', # True Shooting Percentage
            'pts_per_possession'
        ]
        
        # Características de eficiencia específicas para puntos (creadas por feature engineering)
        efficiency_features = [
            'pts_per_minute',
            'pts_per_fga',
            'pts_from_3p', 'pts_from_2p', 'pts_from_ft',
            'pts_prop_from_3p', 'pts_prop_from_2p', 'pts_prop_from_ft',
            'pts_per_scoring_poss', 'scoring_efficiency',
        ]
        
        # Características de ventanas móviles para puntos
        rolling_features = []
        for window in [3, 5, 10, 20]:
            rolling_features.extend([
                f'PTS_mean_{window}',
                f'PTS_std_{window}',
                f'2P%_mean_{window}',
                f'FG%_mean_{window}',
                f'TS%_mean_{window}',
                f'3P%_mean_{window}',
                f'FT%_mean_{window}',
                f'MP_mean_{window}',
                f'pts_momentum_{window}',
                f'pts_per_min_{window}',
                f'expected_pts_{window}',
                f'pts_vs_expected_{window}',
            ])
        
        # Características de rendimiento vs oponentes
        matchup_features = [
            'pts_avg_vs_opp',
            'pts_diff_vs_opp',
            'opp_pts_allowed_avg',
            'opp_pts_allowed_diff',
        ]
        
        # Características de rachas y momentum
        streak_features = [
            'pts_above_avg_streak',
            'pts_increase_streak',
        ]
        
        # Características contextuales
        context_features = [
            'is_home',
            'is_started',
            'pts_home_avg',
            'pts_away_avg',
            'pts_home_away_diff',
            'days_rest', 
            'is_back_to_back', 
            'has_overtime', 
            'overtime_periods',
            'is_high_usage'
        ]
        
        # Características físicas y de posición
        physical_features = [
            'Height_Inches',
            'Weight',
            'BMI',
            'mapped_pos',
        ]
        
        # Características de tendencias recientes
        trend_features = [
            'recent_form_pts',
            'hot_streak_pts',
            'consistency_pts',
        ]
        
        # Combinar todas las características
        all_features = (
            basic_features + 
            efficiency_features + 
            rolling_features + 
            matchup_features + 
            streak_features + 
            context_features + 
            physical_features + 
            trend_features
        )
        
        # Filtrar solo las características que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Caracteristicas disponibles para puntos: {len(available_features)}/{len(all_features)}")
        
        return available_features
    
    def _add_interaction_features(self, X):
        """
        Añade características de interacción y transformaciones no lineales.
        
        Args:
            X (pd.DataFrame): Características originales
            
        Returns:
            pd.DataFrame: Características aumentadas con interacciones
        """
        logger.info("Añadiendo características de interacción y transformaciones no lineales")
        
        # Crear copia para evitar modificar el original
        X_enhanced = X.copy()
        
        # 1. Interacciones entre MP, FGA y FG% (variables de alta importancia)
        if all(col in X.columns for col in ['MP', 'FGA', 'FG%']):
            # Asegurar que no hay valores NaN
            for col in ['MP', 'FGA', 'FG%']:
                if X_enhanced[col].isna().any():
                    logger.info(f"Imputando valores NaN en {col} antes de crear interacciones")
                    X_enhanced[col] = X_enhanced[col].fillna(X_enhanced[col].median())
            
            # Interacciones multiplicativas
            X_enhanced['MP_x_FGA'] = X_enhanced['MP'] * X_enhanced['FGA']
            X_enhanced['MP_x_FG%'] = X_enhanced['MP'] * X_enhanced['FG%']
            X_enhanced['FGA_x_FG%'] = X_enhanced['FGA'] * X_enhanced['FG%']
            X_enhanced['MP_x_FGA_x_FG%'] = X_enhanced['MP'] * X_enhanced['FGA'] * X_enhanced['FG%']
            
            # Interacciones cuadráticas
            X_enhanced['MP_squared'] = X_enhanced['MP'] ** 2
            X_enhanced['FGA_squared'] = X_enhanced['FGA'] ** 2
            X_enhanced['FG%_squared'] = X_enhanced['FG%'] ** 2
            
            # Interacciones cúbicas para capturar relaciones no lineales más complejas
            X_enhanced['MP_cubed'] = X_enhanced['MP'] ** 3
            X_enhanced['FGA_cubed'] = X_enhanced['FGA'] ** 3
            X_enhanced['FG%_cubed'] = X_enhanced['FG%'] ** 3
            
            # Características polinomiales
            X_enhanced['poly_MP_FGA'] = X_enhanced['MP_squared'] * X_enhanced['FGA']
            X_enhanced['poly_FGA_FG%'] = X_enhanced['FGA_squared'] * X_enhanced['FG%']
            X_enhanced['poly_MP_FG%'] = X_enhanced['MP_squared'] * X_enhanced['FG%']
            X_enhanced['poly_FG%2'] = X_enhanced['FG%'] ** 2
            X_enhanced['poly_FGA2'] = X_enhanced['FGA'] ** 2
            
            logger.info("Características de interacción MP, FGA, FG% creadas correctamente")
            
        # 2. Interacciones con 3P% y FT% si están disponibles
        for col1, col2 in [('3P%', 'MP'), ('3P%', 'FTA'), ('3P%', 'FT%'), 
                          ('FT%', 'MP'), ('FT%', 'FTA')]:
            if all(c in X.columns for c in [col1, col2]):
                col_name = f"{col1}_x_{col2}"
                X_enhanced[col_name] = X_enhanced[col1] * X_enhanced[col2]
                logger.info(f"Característica de interacción {col_name} creada")
                
        # 3. Ratio de eficiencia mejorado
        if all(col in X.columns for col in ['FGA', 'MP']):
            X_enhanced['FGA_per_minute'] = X_enhanced['FGA'] / X_enhanced['MP'].clip(lower=1)
            logger.info("Característica de ratio FGA por minuto creada")
            
        # 4. Aproximación de puntos esperados
        if all(col in X.columns for col in ['FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%']):
            # Estimar puntos de tiros de campo de 2
            fg2a = X_enhanced['FGA'] - X_enhanced['3PA']
            fg2_pct = (X_enhanced['FG%'] * X_enhanced['FGA'] - X_enhanced['3P%'] * X_enhanced['3PA']) / fg2a.clip(lower=0.1)
            
            # Corregir valores fuera de rango
            fg2_pct = fg2_pct.clip(lower=0, upper=1)
            
            # Calcular puntos esperados
            pts_from_2 = 2 * fg2a * fg2_pct
            pts_from_3 = 3 * X_enhanced['3PA'] * X_enhanced['3P%']
            pts_from_ft = X_enhanced['FTA'] * X_enhanced['FT%']
            
            X_enhanced['pts_per_fga_proxy'] = (pts_from_2 + pts_from_3) / X_enhanced['FGA'].clip(lower=0.1)
            X_enhanced['expected_pts'] = pts_from_2 + pts_from_3 + pts_from_ft
            X_enhanced['pts_efficiency_index'] = X_enhanced['pts_per_fga_proxy'] * X_enhanced['FGA_per_minute']
            
            logger.info("Características de eficiencia y puntos esperados creadas")
            
        # Registrar el número de características añadidas
        n_features_added = len(X_enhanced.columns) - len(X.columns)
        logger.info(f"Total de {n_features_added} características de interacción añadidas")
        
        return X_enhanced
    
    def detect_outliers(self, X, method='isolation_forest', contamination=0.05):
        """
        Detecta outliers utilizando técnicas más sofisticadas.
        
        Args:
            X (pd.DataFrame): DataFrame con características
            method (str): Método de detección ('isolation_forest', 'local_outlier_factor', 'elliptic_envelope')
            contamination (float): Proporción esperada de outliers
            
        Returns:
            np.array: Máscara booleana donde True indica que no es outlier (inlier)
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.covariance import EllipticEnvelope
        
        # Eliminar columnas con valores faltantes o sin varianza
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].isna().any() or X_clean[col].nunique() <= 1:
                X_clean.drop(columns=[col], inplace=True)
        
        # Seleccionar detector según el método
        if method == 'isolation_forest':
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=False
            )
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(
                contamination=contamination,
                support_fraction=0.7,
                random_state=42
            )
        else:
            raise ValueError(f"Método de detección '{method}' no soportado")
        
        try:
            # Para LOF, -1 son outliers, 1 son inliers
            if method == 'local_outlier_factor':
                y_pred = detector.fit_predict(X_clean)
                mask = y_pred == 1  # True para inliers
            else:
                # Ajustar detector y obtener predicciones
                detector.fit(X_clean)
                y_pred = detector.predict(X_clean)
                mask = y_pred == 1  # True para inliers
                
            # Calcular número de outliers detectados
            n_outliers = (~mask).sum()
            logger.info(f"Detección de outliers ({method}): {n_outliers} outliers encontrados de {len(X)} muestras")
            
            return mask
            
        except Exception as e:
            logger.error(f"Error en detección de outliers: {str(e)}")
            # En caso de error, no eliminar ninguna muestra
            return np.ones(len(X), dtype=bool)
    
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
        # Primero preparamos los datos base
        X_train, X_test, y_train, y_test = super().prepare_data(df, test_size, time_split)
        
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
                    inlier_mask = self.detect_outliers(X_train, method='isolation_forest', contamination=0.05)
                
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
        
        # Agregar características de interacción avanzadas (con las mejoras implementadas)
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
    
    def _transform_fga_features(self, X):
        """
        Transforma las características FGA especiales para predicciones.
        Se aplica al hacer nuevas predicciones, reutilizando las transformaciones
        del entrenamiento.
        
        Args:
            X (pd.DataFrame): DataFrame con características
        
        Returns:
            pd.DataFrame: DataFrame con características FGA transformadas
        """
        # Crear una copia para no modificar el original
        X_transformed = X.copy()
        
        # Verificar si tenemos FGA y necesitamos aplicar transformaciones
        if 'FGA' in X_transformed.columns:
            # 1. Transformación raíz cuadrada
            X_transformed['FGA_sqrt'] = np.sqrt(X_transformed['FGA'])
            
            # 2. Transformación logarítmica especial
            X_transformed['FGA_log2'] = np.log2(X_transformed['FGA'] + 1)
            
            # 3. Yeo-Johnson si el transformador fue guardado durante el entrenamiento
            if hasattr(self, 'power_transformer_fga'):
                try:
                    X_fga = X_transformed['FGA'].values.reshape(-1, 1)
                    X_transformed['FGA_yj'] = self.power_transformer_fga.transform(X_fga).flatten()
                except Exception as e:
                    logger.warning(f"Error aplicando Yeo-Johnson a FGA en predicción: {str(e)}")
                    # Crear una aproximación para evitar el error
                    X_transformed['FGA_yj'] = 0.5 * np.log1p(X_transformed['FGA'])
            else:
                # Si no tenemos el transformador, crear una aproximación
                X_transformed['FGA_yj'] = 0.5 * np.log1p(X_transformed['FGA'])
            
            # 4. Para la transformación de rango, necesitamos adaptar el enfoque
            # Esta es una aproximación simplificada que puede no ser perfecta
            if hasattr(self, 'X_train') and 'FGA' in self.X_train.columns:
                try:
                    from scipy.stats import percentileofscore
                    
                    # Calcular el percentil de cada valor FGA respecto a los datos de entrenamiento
                    X_transformed['FGA_rank'] = X_transformed['FGA'].apply(
                        lambda x: percentileofscore(self.X_train['FGA'], x) / 100.0
                    )
                except Exception as e:
                    logger.warning(f"Error aplicando transformación de rangos a FGA en predicción: {str(e)}")
                    # Crear una aproximación normalizada entre 0 y 1
                    X_transformed['FGA_rank'] = (X_transformed['FGA'] - X_transformed['FGA'].min()) / (X_transformed['FGA'].max() - X_transformed['FGA'].min() + 1e-8)
            else:
                # Si no tenemos los datos de entrenamiento, crear una aproximación normalizada
                X_transformed['FGA_rank'] = (X_transformed['FGA'] - X_transformed['FGA'].min()) / (X_transformed['FGA'].max() - X_transformed['FGA'].min() + 1e-8)
        
        return X_transformed
    
    def ensure_feature_compatibility(self, X, required_features=None):
        """
        Asegura que X contiene todas las características requeridas por el modelo.
        Si faltan características, se crean como aproximaciones o valores neutros.
        
        Args:
            X (pd.DataFrame): DataFrame con características
            required_features (list): Lista de características requeridas (None para usar self.feature_columns)
            
        Returns:
            pd.DataFrame: DataFrame con todas las características requeridas
        """
        # Si no se especifican características requeridas, usar las de entrenamiento
        if required_features is None and hasattr(self, 'feature_columns'):
            required_features = self.feature_columns
        elif required_features is None:
            # Si no hay lista de características, devolver X sin modificar
            return X
        
        # Crear copia para no modificar el original
        X_comp = X.copy()
        
        # Verificar qué características faltan
        missing_features = [f for f in required_features if f not in X_comp.columns]
        
        if missing_features:
            logger.warning(f"Faltan {len(missing_features)} características en los datos: {missing_features}")
            
            # Determinar qué tipo de características son las faltantes
            missing_derived = [f for f in missing_features if '_' in f]  # Características derivadas como 'FGA_x_FG%'
            missing_basic = [f for f in missing_features if '_' not in f]  # Características básicas
            
            logger.info(f"Características básicas faltantes: {missing_basic}")
            logger.info(f"Características derivadas faltantes: {missing_derived}")
            
            # Primero manejar características básicas faltantes (mayor prioridad)
            for feature in missing_basic:
                # Si es una característica clave conocida, usar un valor predeterminado razonable
                if feature in ['FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'MP']:
                    # Para porcentajes, usar valor medio típico
                    if '%' in feature:
                        X_comp[feature] = 0.45  # Valor medio típico para porcentajes de tiro
                        logger.info(f"Creada característica básica '{feature}' con valor promedio 0.45")
                    # Para conteos, usar valor medio bajo
                    else:
                        X_comp[feature] = 5.0  # Valor conservador para intentos/minutos
                        logger.info(f"Creada característica básica '{feature}' con valor conservador 5.0")
                else:
                    # Para otras características básicas desconocidas, usar cero
                    X_comp[feature] = 0.0
                    logger.info(f"Creada característica básica '{feature}' con valor neutro 0.0")
            
            # Crear características derivadas usando características básicas (pueden ser originales o recién creadas)
            for feature in missing_derived:
                # Características FGA derivadas
                if feature.startswith('FGA_'):
                    if feature == 'FGA_sqrt' and 'FGA' in X_comp.columns:
                        X_comp[feature] = np.sqrt(X_comp['FGA'])
                        logger.info(f"Creada característica derivada '{feature}' usando transformación")
                    elif feature == 'FGA_log2' and 'FGA' in X_comp.columns:
                        X_comp[feature] = np.log2(X_comp['FGA'] + 1)
                        logger.info(f"Creada característica derivada '{feature}' usando transformación logarítmica")
                    elif feature == 'FGA_yj' and 'FGA' in X_comp.columns:
                        X_comp[feature] = 0.5 * np.log1p(X_comp['FGA'])
                        logger.info(f"Creada característica derivada '{feature}' usando aproximación Yeo-Johnson")
                    elif feature == 'FGA_rank' and 'FGA' in X_comp.columns:
                        X_comp[feature] = (X_comp['FGA'] - X_comp['FGA'].min()) / (X_comp['FGA'].max() - X_comp['FGA'].min() + 1e-8)
                        logger.info(f"Creada característica derivada '{feature}' usando normalización de rango")
                    elif feature == 'FGA_cubed' and 'FGA' in X_comp.columns:
                        X_comp[feature] = X_comp['FGA'] ** 3
                        logger.info(f"Creada característica derivada '{feature}' usando transformación cúbica")
                    else:
                        # Para otras transformaciones de FGA desconocidas, usar aproximación
                        if 'FGA' in X_comp.columns:
                            X_comp[feature] = X_comp['FGA'] / X_comp['FGA'].mean() if X_comp['FGA'].mean() > 0 else X_comp['FGA']
                            logger.info(f"Creada característica derivada '{feature}' usando aproximación basada en FGA")
                        else:
                            X_comp[feature] = 0.5
                            logger.info(f"Creada característica derivada '{feature}' con valor predeterminado 0.5")
                
                # Características derivadas de interacción (_x_)
                elif '_x_' in feature:
                    parts = feature.split('_x_')
                    
                    if len(parts) == 2:
                        # Identificar las partes base y verificar si tienen modificadores
                        part1_base = parts[0].split('_')[0] if '_' in parts[0] else parts[0]
                        part2_base = parts[1].split('_')[0] if '_' in parts[1] else parts[1]
                        
                        # Verificar si tenemos las características base o podemos calcularlas
                        if part1_base in X_comp.columns and part2_base in X_comp.columns:
                            # Caso simple: ambas partes básicas existen
                            if parts[0] in X_comp.columns and parts[1] in X_comp.columns:
                                X_comp[feature] = X_comp[parts[0]] * X_comp[parts[1]]
                                logger.info(f"Creada interacción '{feature}' directamente de '{parts[0]}' y '{parts[1]}'")
                            # Manejo especial para partes con modificadores
                            else:
                                # Aproximar mejor según el tipo de interacción específica
                                if 'squared' in parts[0] or 'squared' in parts[1]:
                                    # Interacción con término cuadrático
                                    if 'squared' in parts[0]:
                                        base = part1_base
                                        X_comp[feature] = (X_comp[base] ** 2) * X_comp[part2_base]
                                    else:
                                        base = part2_base
                                        X_comp[feature] = X_comp[part1_base] * (X_comp[base] ** 2)
                                    logger.info(f"Creada interacción '{feature}' usando aproximación cuadrática")
                                else:
                                    # Interacción simple
                                    X_comp[feature] = X_comp[part1_base] * X_comp[part2_base]
                                    logger.info(f"Creada interacción '{feature}' usando aproximación simple con características base")
                        else:
                            # No tenemos las características base, usar valor neutro
                            X_comp[feature] = 0.0
                            logger.info(f"Creada interacción '{feature}' con valor neutro 0.0 (faltan características base)")
                    else:
                        # Formato _x_ no estándar, usar valor neutro
                        X_comp[feature] = 0.0
                        logger.info(f"Creada característica '{feature}' con valor neutro 0.0 (formato no estándar)")
                
                # Características transformadas generales
                elif '_squared' in feature or '_cubed' in feature:
                    base_feature = feature.split('_')[0]
                    if base_feature in X_comp.columns:
                        if '_squared' in feature:
                            X_comp[feature] = X_comp[base_feature] ** 2
                            logger.info(f"Creada característica '{feature}' usando transformación cuadrática")
                        elif '_cubed' in feature:
                            X_comp[feature] = X_comp[base_feature] ** 3
                            logger.info(f"Creada característica '{feature}' usando transformación cúbica")
                    else:
                        X_comp[feature] = 0.0
                        logger.info(f"Creada característica '{feature}' con valor neutro 0.0 (falta característica base)")
                
                # Características transformadas con Box-Cox o log
                elif '_bc' in feature or '_log' in feature:
                    base_feature = feature.split('_')[0]
                    if base_feature in X_comp.columns:
                        if '_bc' in feature:
                            # Aproximar transformación Box-Cox con log
                            X_comp[feature] = np.log1p(X_comp[base_feature])
                            logger.info(f"Creada característica '{feature}' usando aproximación log para Box-Cox")
                        elif '_log' in feature:
                            X_comp[feature] = np.log1p(X_comp[base_feature])
                            logger.info(f"Creada característica '{feature}' usando transformación log")
                    else:
                        X_comp[feature] = 0.0
                        logger.info(f"Creada característica '{feature}' con valor neutro 0.0 (falta característica base)")
                
                # Características polinómicas de sklearn
                elif feature.startswith('poly_'):
                    # Estas son más complejas, usar valores neutros o aproximaciones sencillas
                    # Intentar extraer las bases (por ejemplo, poly_FGA_FG% o poly_FGA_power_2)
                    if 'FGA' in feature and 'FG%' in feature:
                        if 'FGA' in X_comp.columns and 'FG%' in X_comp.columns:
                            # Aproximación simple basada en la interacción de las bases
                            X_comp[feature] = X_comp['FGA'] * X_comp['FG%']
                            logger.info(f"Creada característica polinómica '{feature}' usando aproximación simple")
                        else:
                            X_comp[feature] = 0.0
                            logger.info(f"Creada característica polinómica '{feature}' con valor neutro 0.0")
                    else:
                        X_comp[feature] = 0.0
                        logger.info(f"Creada característica polinómica '{feature}' con valor neutro 0.0")
                
                # Para otras características derivadas desconocidas, usar ceros
                else:
                    X_comp[feature] = 0.0
                    logger.info(f"Creada característica desconocida '{feature}' con valor neutro 0.0")
                    
            logger.info(f"Agregadas {len(missing_features)} características faltantes a los datos")
        
        # Asegurarse de que solo devolvemos las características requeridas en el orden correcto
        try:
            result = X_comp[required_features]
            # Verificar que no haya NaNs en el resultado final
            if result.isna().any().any():
                logger.warning(f"Detectados valores NaN en el resultado final. Imputando con medianas.")
                result = result.fillna(result.median())
                # Si aún hay NaNs (por ejemplo, en columnas completamente NaN), usar ceros
                if result.isna().any().any():
                    result = result.fillna(0.0)
            return result
        except Exception as e:
            logger.error(f"Error al seleccionar características requeridas: {str(e)}")
            logger.error(f"Características requeridas: {required_features}")
            logger.error(f"Características disponibles: {X_comp.columns.tolist()}")
            
            # Intentar recuperación: devolver todas las características que podamos
            try:
                common_features = [f for f in required_features if f in X_comp.columns]
                if common_features:
                    logger.warning(f"Devolviendo {len(common_features)}/{len(required_features)} características comunes")
                    return X_comp[common_features]
                else:
                    logger.error("No hay características comunes. Devolviendo datos originales.")
                    return X
            except:
                logger.error("Error en recuperación. Devolviendo datos originales.")
                return X
    
    def adjust_features_for_xgboost(self, X):
        """
        Ajusta las características para hacerlas compatibles con XGBoost.
        Maneja el problema 'DataFrame' object has no attribute 'dtype'.
        
        Args:
            X (pd.DataFrame): DataFrame con características
            
        Returns:
            numpy.ndarray: Array NumPy compatible con XGBoost
        """
        try:
            # Verificar si es un DataFrame
            if hasattr(X, 'values'):
                # Crear copia para evitar efectos secundarios
                X_copy = X.copy()
                
                # Verificar características faltantes conocidas relacionadas con FGA
                expected_fga_features = ['FGA_sqrt', 'FGA_log2', 'FGA_yj', 'FGA_rank']
                missing_features = [f for f in expected_fga_features if f not in X_copy.columns]
                
                if missing_features and 'FGA' in X_copy.columns:
                    logger.warning(f"Agregando características FGA faltantes: {missing_features}")
                    
                    # Crear características faltantes
                    if 'FGA_sqrt' in missing_features:
                        X_copy['FGA_sqrt'] = np.sqrt(np.maximum(0, X_copy['FGA']))
                    
                    if 'FGA_log2' in missing_features:
                        X_copy['FGA_log2'] = np.log2(X_copy['FGA'] + 1)
                    
                    if 'FGA_yj' in missing_features:
                        X_copy['FGA_yj'] = 0.5 * np.log1p(X_copy['FGA'])
                    
                    if 'FGA_rank' in missing_features:
                        # Normalización simple en rango [0,1]
                        fga_max = X_copy['FGA'].max()
                        fga_min = X_copy['FGA'].min()
                        if fga_max > fga_min:
                            X_copy['FGA_rank'] = (X_copy['FGA'] - fga_min) / (fga_max - fga_min)
                        else:
                            X_copy['FGA_rank'] = 0.5  # Valor por defecto
                
                # Verificar y manejar NaNs
                has_nans = X_copy.isna().any().any()
                if has_nans:
                    logger.warning("Detectados NaNs en ajuste para XGBoost. Imputando con medianas.")
                    X_copy = X_copy.fillna(X_copy.median())
                    # Si aún hay NaNs (columnas completamente NaN), usar ceros
                    X_copy = X_copy.fillna(0)
                
                # Convertir a array NumPy
                numpy_array = X_copy.values
                
                # Verificar una vez más valores infinitos o NaN
                if np.isnan(numpy_array).any() or np.isinf(numpy_array).any():
                    logger.warning("Detectados valores infinitos o NaN en array NumPy. Reemplazando.")
                    numpy_array = np.nan_to_num(numpy_array, nan=0.0, posinf=1e6, neginf=-1e6)
                
                return numpy_array
                
            elif isinstance(X, np.ndarray):
                # Ya es un array NumPy, verificar NaNs e infinitos
                if np.isnan(X).any() or np.isinf(X).any():
                    logger.warning("Detectados valores infinitos o NaN en array NumPy dado. Reemplazando.")
                    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
                return X
            else:
                # Intentar convertir a NumPy array genéricamente
                logger.warning(f"Tipo no reconocido ({type(X)}). Intentando conversión genérica.")
                try:
                    numpy_array = np.array(X, dtype=np.float64)
                    return np.nan_to_num(numpy_array, nan=0.0, posinf=1e6, neginf=-1e6)
                except Exception as e:
                    raise ValueError(f"No se pudo convertir a formato compatible con XGBoost: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error en adjust_features_for_xgboost: {str(e)}")
            # Último recurso: crear array de ceros con forma aproximada
            if hasattr(X, 'shape'):
                shape = X.shape
            elif hasattr(X, 'values') and hasattr(X.values, 'shape'):
                shape = X.values.shape
            else:
                shape = (1, 1)  # Forma genérica
            
            logger.warning(f"Devolviendo array de ceros con forma {shape} como último recurso")
            return np.zeros(shape)

    def predict(self, X_test, model_name='best'):
        """
        Realiza predicciones usando el modelo especificado.
        
        Args:
            X_test: Características para predecir
            model_name: Nombre del modelo a usar
            
        Returns:
            array: Predicciones
        """
        # Si tenemos disponible el ensemble adaptativo regional y es el modelo solicitado
        if model_name in ['best', 'adaptive_regional_ensemble'] and 'adaptive_regional_ensemble' in self.models:
            return self.predict_adaptive(X_test)
        
        # Si no, hay que garantizar que los datos son compatibles para todos los modelos
        
        # Preparar características especiales para los modelos
        try:
            # Manejar valores NaN en los datos de entrada
            if hasattr(X_test, 'isna') and X_test.isna().any().any():
                logger.info("Detectados NaNs en datos de entrada para predict. Imputando.")
                X_test_clean = X_test.copy()
                X_test_clean = X_test_clean.fillna(X_test_clean.median())
                X_test_clean = X_test_clean.fillna(0)  # Por si quedan NaNs
            else:
                X_test_clean = X_test
            
            # Verificar si necesitamos características de FGA
            if 'FGA' in X_test_clean.columns and not all(f in X_test_clean.columns for f in ['FGA_sqrt', 'FGA_log2', 'FGA_yj', 'FGA_rank']):
                X_test_clean = self._transform_fga_features(X_test_clean)
            
            # Para modelos propensos al error 'DataFrame' object has no attribute 'dtype'
            if model_name in ['xgboost', 'xgb_optimized', 'xgb_weighted', 'xgb_reg', 'stacking', 'voting', 'region_mid', 'region_high']:
                # Convertir características a formato compatible
                X_features = self.adjust_features_for_xgboost(X_test_clean)
                
                try:
                    # Obtener el modelo
                    model = self.models.get(model_name)
                    if model is None:
                        logger.error(f"Modelo '{model_name}' no encontrado")
                        # Usar el mejor modelo disponible como fallback
                        model_name, model, _ = self.get_best_model()
                    
                    # Realizar predicción con array NumPy
                    predictions = model.predict(X_features)
                    
                    # Aplicar transformación inversa si es necesario
                    if hasattr(self, 'y_transform') and self.y_transform == 'log':
                        predictions = np.expm1(predictions)
                    
                    return predictions
                    
                except Exception as e:
                    logger.error(f"Error en predict con {model_name}: {str(e)}")
                    # Usar implementación genérica como fallback
                    return super().predict(X_test_clean, model_name='best')
                
            else:
                # Usar la implementación original para otros modelos
                return super().predict(X_test_clean, model_name)
        
        except Exception as e:
            logger.error(f"Error general en predict: {str(e)}")
            # Último recurso: implementación básica
            return super().predict(X_test, model_name)
    
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
        # Si tenemos una transformación aplicada en y, usar los valores originales
        if hasattr(self, 'y_transform') and self.y_transform == 'log':
            # Usar los valores originales guardados
            y_true = self.y_test_original if prefix == '' else y_test
        else:
            y_true = y_test
        
        # Hacer predicciones (ya aplica transformación inversa si es necesario)
        y_pred = self.predict(X_test, model_name)
        
        # Calcular métricas estándar
        metrics = {}
        metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
        
        # Calcular MAPE de forma robusta
        metrics[f'{prefix}mape'] = self._calculate_robust_mape(y_true, y_pred)
        
        # Calcular métricas adicionales para heteroscedasticidad
        metrics.update(self._calculate_robust_metrics(y_true, y_pred, prefix))
        
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
    
    def _calculate_robust_mape(self, y_true, y_pred, epsilon=1.0, max_pct=100):
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
        
        # 1. Usar sMAPE (Symmetric MAPE) como alternativa más robusta
        # Formula: 200% * |y_true - y_pred| / (|y_true| + |y_pred| + epsilon)
        abs_diff = np.abs(y_true_array - y_pred_array)
        abs_sum = np.abs(y_true_array) + np.abs(y_pred_array) + epsilon
        smape = 200.0 * (abs_diff / abs_sum)
        
        # 2. Recortar valores extremos
        smape = np.clip(smape, 0, max_pct)
        
        # 3. Usar mediana en lugar de media para reducir influencia de valores extremos
        robust_mape = np.median(smape)
        
        # 4. Verificar que el valor sea razonable
        if robust_mape > max_pct or not np.isfinite(robust_mape):
            # Como última opción, usar MAE relativo
            mean_abs_true = np.mean(np.abs(y_true_array))
            if mean_abs_true > epsilon:
                mae = np.mean(abs_diff)
                robust_mape = (mae / mean_abs_true) * 100
            else:
                robust_mape = 0.0
        
        logger.info(f"MAPE calculado usando sMAPE con mediana: {robust_mape:.4f}%")
        
        return robust_mape

    def analyze_heteroscedasticity(self, X_test, y_test, model_name='best'):
        """
        Analiza la heteroscedasticidad en las predicciones del modelo.
        
        Args:
            X_test: Características de test
            y_test: Target de test
            model_name: Nombre del modelo a evaluar
            
        Returns:
            dict: Resultados del análisis de heteroscedasticidad
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # Hacer predicciones
        y_pred = self.predict(X_test, model_name)
        
        # Si tenemos una transformación aplicada en y, usar los valores originales
        if hasattr(self, 'y_transform') and self.y_transform == 'log':
            y_true = self.y_test_original if hasattr(self, 'y_test_original') else y_test
        else:
            y_true = y_test
            
        # Calcular residuos
        residuals = y_true - y_pred
        
        # Prueba de Breusch-Pagan para heteroscedasticidad
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            
            # Reshape para la prueba
            y_pred_reshaped = y_pred.reshape(-1, 1)
            
            # Realizar la prueba
            bp_test = het_breuschpagan(residuals, y_pred_reshaped)
            
            hetero_results = {
                'bp_lm_stat': bp_test[0],
                'bp_lm_pvalue': bp_test[1],
                'bp_f_stat': bp_test[2],
                'bp_f_pvalue': bp_test[3],
                'hetero_detected': bp_test[1] < 0.05  # p-valor < 0.05 indica heteroscedasticidad
            }
        except:
            # Si falla la prueba, usar método simplificado
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, residuals**2)
            hetero_results = {
                'slope': slope,
                'p_value': p_value,
                'hetero_detected': p_value < 0.05  # p-valor < 0.05 indica heteroscedasticidad
            }
        
        # Crear visualización de residuos vs valores predichos
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        
        # Añadir línea de tendencia para los residuos
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(y_pred, residuals)
            x_line = np.linspace(min(y_pred), max(y_pred), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r--', alpha=0.7)
            plt.text(0.05, 0.95, f'Pendiente: {slope:.4f} (p-valor: {p_value:.4f})', 
                    transform=plt.gca().transAxes)
        except:
            pass
            
        plt.title('Análisis de Heteroscedasticidad: Residuos vs Predicciones')
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuos')
        plt.grid(True, alpha=0.3)
        
        # Guardar la figura
        hetero_plot_path = 'reports/figures/heteroscedasticity_analysis.png'
        os.makedirs(os.path.dirname(hetero_plot_path), exist_ok=True)
        plt.savefig(hetero_plot_path)
        plt.close()
        
        hetero_results['plot_path'] = hetero_plot_path
        
        return hetero_results
    
    def validate_models(self, X_test, y_test):
        """
        Valida modelos con métricas específicas para puntos.
        
        Args:
            X_test: Características de test
            y_test: Variable objetivo de test
            
        Returns:
            dict: Métricas de validación por modelo
        """
        results = super().validate_models(X_test, y_test)
        
        # Si se aplicó transformación logarítmica, también calcular métricas en escala original
        if hasattr(self, 'y_transform') and self.y_transform == 'log':
            for model_name, model in self.models.items():
                try:
                    if model_name not in results or 'error' in results[model_name]:
                        continue  # Saltamos modelos que fallaron
                    
                    # Obtener predicciones en escala transformada
                    if 'main' in self.scalers and model_name not in ['xgboost', 'lightgbm', 'random_forest']:
                        X_test_scaled = self.scalers['main'].transform(X_test)
                        log_predictions = model.predict(X_test_scaled)
                    else:
                        log_predictions = model.predict(X_test)
                    
                    # Transformar predicciones a escala original
                    predictions = np.expm1(log_predictions)
                    
                    # Calcular métricas en escala original
                    mse = mean_squared_error(self.y_test_original, predictions)
                    mae = mean_absolute_error(self.y_test_original, predictions)
                    rmse = np.sqrt(mse)
                    
                    # Calcular MAPE de forma más robusta con nuestra nueva implementación
                    mape = self._calculate_robust_mape(
                        self.y_test_original.values, 
                        predictions, 
                        epsilon=1.0,  # Umbral para valores pequeños
                        max_pct=100   # Límite máximo de porcentaje de error
                    )
                    
                    # Calcular R²
                    r2 = 1 - ((self.y_test_original - predictions) ** 2).sum() / ((self.y_test_original - self.y_test_original.mean()) ** 2).sum()
                    
                    results[model_name].update({
                        'original_mse': mse,
                        'original_mae': mae,
                        'original_rmse': rmse,
                        'original_mape': mape,
                        'original_r2': r2,
                        'original_predictions': predictions
                    })
                    
                    logger.info(f"{model_name} (escala original) - RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error calculando metricas originales para {model_name}: {str(e)}")
        
        return results
    
    def get_prediction_context(self, player_name, df, n_games=5):
        """
        Obtiene contexto específico para la predicción de puntos de un jugador.
        
        Args:
            player_name (str): Nombre del jugador
            df (pd.DataFrame): DataFrame con los datos
            n_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Contexto de predicción específico para puntos
        """
        player_data = df[df['Player'] == player_name].copy()
        
        if len(player_data) == 0:
            return {}
        
        # Ordenar por fecha (más reciente primero)
        player_data = player_data.sort_values('Date', ascending=False)
        recent_games = player_data.head(n_games)
        
        context = {
            'avg_pts_recent': recent_games['PTS'].mean(),
            'avg_pts_season': player_data['PTS'].mean(),
            'avg_minutes_recent': recent_games['MP'].mean(),
            'avg_fg_pct_recent': recent_games['FG%'].mean(),
            'avg_3p_attempts_recent': recent_games['3PA'].mean(),
            'games_with_20plus': (recent_games['PTS'] >= 20).sum(),
            'games_with_30plus': (recent_games['PTS'] >= 30).sum(),
            'highest_score_recent': recent_games['PTS'].max(),
            'lowest_score_recent': recent_games['PTS'].min(),
            'scoring_consistency': recent_games['PTS'].std(),
        }
        
        # Tendencia de puntuación (mejorando/empeorando)
        if len(recent_games) >= 3:
            context['scoring_trend'] = recent_games['PTS'].iloc[:3].mean() - recent_games['PTS'].iloc[-3:].mean()
        
        return context
    
    def analyze_scoring_patterns(self, df):
        """
        Analiza patrones de puntuación en el dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Análisis de patrones de puntuación
        """
        analysis = {}
        
        if 'PTS' not in df.columns:
            return analysis
        
        # Estadísticas generales
        analysis['pts_stats'] = {
            'mean': df['PTS'].mean(),
            'median': df['PTS'].median(),
            'std': df['PTS'].std(),
            'min': df['PTS'].min(),
            'max': df['PTS'].max(),
        }
        
        # Distribución por rangos
        analysis['pts_distribution'] = {
            '0-9_pts': (df['PTS'] < 10).sum(),
            '10-19_pts': ((df['PTS'] >= 10) & (df['PTS'] < 20)).sum(),
            '20-29_pts': ((df['PTS'] >= 20) & (df['PTS'] < 30)).sum(),
            '30-39_pts': ((df['PTS'] >= 30) & (df['PTS'] < 40)).sum(),
            '40plus_pts': (df['PTS'] >= 40).sum(),
        }
        
        # Análisis por posición
        if 'mapped_pos' in df.columns:
            analysis['pts_by_position'] = df.groupby('mapped_pos')['PTS'].agg(['mean', 'std']).to_dict()
        
        # Análisis casa vs visitante
        if 'is_home' in df.columns:
            analysis['home_vs_away'] = {
                'home_avg': df[df['is_home'] == 1]['PTS'].mean(),
                'away_avg': df[df['is_home'] == 0]['PTS'].mean(),
            }
        
        # Top anotadores
        if 'Player' in df.columns:
            top_scorers = df.groupby('Player')['PTS'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False)
            analysis['top_scorers'] = top_scorers.head(10).to_dict()
        
        logger.info("Analisis de patrones de puntuacion completado")
        
        return analysis
    
    def analyze_prediction_errors(self, X_test=None, y_test=None, model_name='best'):
        """
        Analiza los errores de predicción con detalle para entender patrones.
        
        Args:
            X_test: Features de prueba (usa los almacenados si es None)
            y_test: Target de prueba (usa los almacenados si es None)
            model_name: Nombre del modelo a evaluar
            
        Returns:
            dict: Diccionario con métricas y análisis de errores
        """
        # Usar datos de test almacenados si no se proporcionan
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test_original
        
        # Obtener el modelo a evaluar
        if model_name == 'best':
            model_name, model, _ = self.get_best_model()
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        # Hacer predicciones
        y_pred = self.predict(X_test, model_name=model_name)
        
        # Calcular residuos y métricas
        residuals = y_test - y_pred
        abs_errors = np.abs(residuals)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE con manejo robusto
        mask = (y_test > 1.0)  # Solo considerar valores significativos
        if mask.sum() > 0:
            pct_errors = np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]) * 100
            pct_errors = np.clip(pct_errors, 0, 200)  # Limitar valores extremos
            mape = np.percentile(pct_errors, 99) if len(pct_errors) > 100 else np.mean(pct_errors)
        else:
            mape = None
        
        # Análisis de errores por cuartiles de la variable objetivo
        quartiles = pd.qcut(y_test, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        error_by_quartile = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'abs_error': abs_errors,
            'rel_error': abs_errors / (y_test + 1e-8) * 100,
            'quartile': quartiles
        })
        
        quartile_stats = error_by_quartile.groupby('quartile').agg({
            'abs_error': ['mean', 'median', 'std'],
            'rel_error': ['mean', 'median', 'std']
        })
        
        # Análisis de variabilidad de error por rangos
        error_variability = {}
        for range_name, mask in [
            ('0-5', y_test < 5),
            ('5-10', (y_test >= 5) & (y_test < 10)),
            ('10-20', (y_test >= 10) & (y_test < 20)),
            ('20-30', (y_test >= 20) & (y_test < 30)),
            ('30+', y_test >= 30)
        ]:
            if mask.sum() > 0:
                range_errors = abs_errors[mask]
                error_variability[range_name] = {
                    'count': len(range_errors),
                    'mean_error': range_errors.mean(),
                    'median_error': np.median(range_errors),
                    'std_error': range_errors.std(),
                    'max_error': range_errors.max()
                }
        
        # Test de normalidad en residuos
        shapiro_test = stats.shapiro(residuals) if len(residuals) <= 5000 else None
        
        # Análisis de autocorrelación en residuos (para detectar dependencias temporales)
        if hasattr(X_test, 'index') and hasattr(X_test.index, 'to_series'):
            try:
                # Intentar ordenar por fecha si está disponible
                if 'Date' in X_test.columns:
                    ordered_residuals = pd.Series(residuals, index=X_test.index).sort_index()
                    autocorr = ordered_residuals.autocorr()
                else:
                    autocorr = None
            except:
                autocorr = None
        else:
            autocorr = None
        
        # Compilar resultados en un dict
        analysis = {
            'model_name': model_name,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            },
            'error_distribution': {
                'mean_residual': residuals.mean(),
                'median_residual': np.median(residuals),
                'std_residual': residuals.std(),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals),
                'shapiro_test': shapiro_test._asdict() if shapiro_test else None,
                'autocorrelation': autocorr
            },
            'error_by_quartile': quartile_stats.to_dict(),
            'error_by_range': error_variability
        }
        
        # Crear visualizaciones
        try:
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        
        # 1. Crear figura para Residuos vs Predicciones
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='-', linewidth=1)
        ax.set_xlabel('Valores Predichos')
        ax.set_ylabel('Residuos')
        ax.set_title('Análisis de Residuos para Modelo de Puntos')
        ax.grid(True, alpha=0.3)
        
        # 2. Histograma de residuos
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Distribución de Residuos')
        plt.xlabel('Residuos')
        plt.grid(True, alpha=0.3)
        
        # 3. Box plot de errores por rangos
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='quartile', y='abs_error', data=error_by_quartile)
        plt.title('Errores Absolutos por Cuartil de Puntos')
        plt.xlabel('Cuartil de Puntos (Q1=bajo, Q4=alto)')
        plt.ylabel('Error Absoluto')
        plt.grid(True, alpha=0.3)
        
        # 4. Predicción vs Real
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
        
        # Línea de referencia y=x (predicción perfecta)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales')
        plt.grid(True, alpha=0.3)
        
        # Cerrar todas las figuras para liberar memoria
        plt.close('all')
        
        return analysis
        
    def optimize_model_hyperparams(self, param_grid=None, n_iter=30, cv=5, scoring='neg_root_mean_squared_error'):
        """
        Optimiza los hiperparámetros de los modelos mediante validación cruzada temporal.
        Enfocado en reducir RMSE y sesgo en los residuos.
        
        Args:
            param_grid: Diccionario de grids de parámetros por modelo (None para usar defaults)
            n_iter: Número de iteraciones para RandomizedSearchCV
            cv: Número de folds para validación cruzada temporal
            scoring: Métrica para optimizar ('neg_root_mean_squared_error', 'r2', etc.)
        
        Returns:
            dict: Resultados de la optimización y mejores modelos
        """
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        import joblib
        import os
        
        # Crear directorio para guardar modelos si no existe
        os.makedirs('models/optimized', exist_ok=True)
        
        # Grid de parámetros predeterminados mejorados si no se proporciona
        if param_grid is None:
            param_grid = {
                'xgboost': {
                    'n_estimators': [100, 200, 300, 500, 1000],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
                    'min_child_weight': [1, 2, 3, 4, 5],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
                    'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0, 5.0],
                    'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                },
                'hist_gb': {
                    'max_iter': [100, 200, 300, 500, 1000],
                    'max_depth': [None, 3, 5, 7, 10, 15],
                    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
                    'l2_regularization': [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
                    'max_bins': [128, 255],
                    'min_samples_leaf': [10, 20, 30, 50],
                    'max_leaf_nodes': [None, 15, 31, 63, 127],
                    'early_stopping': [True]
                },
                'lightgbm': {
                    'n_estimators': [100, 200, 300, 500, 1000],
                    'max_depth': [-1, 3, 5, 7, 9],
                    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
                    'num_leaves': [31, 63, 127, 255],
                    'min_child_samples': [5, 10, 20, 50],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0, 5.0]
                },
                'ridge': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
                },
                'elastic': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 2000, 5000],
                    'tol': [1e-4, 1e-3, 1e-2]
                }
            }
        
        # Asegurarse que tenemos datos para optimización
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            logger.error("No hay datos de entrenamiento disponibles para optimización")
            return {}
        
        # Preparar separadores para validación cruzada temporal
        if hasattr(self.X_train, 'index') and hasattr(self.X_train.index, 'is_monotonic_increasing'):
            # Si el índice es temporal (monótono creciente)
            if self.X_train.index.is_monotonic_increasing:
                # Usar TimeSeriesSplit para preservar dependencia temporal
                cv_splitter = TimeSeriesSplit(n_splits=cv)
            else:
                # División estándar si no es temporal
                from sklearn.model_selection import KFold
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            # División estándar por defecto
            from sklearn.model_selection import KFold
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Definir un scorers personalizado para penalizar el sesgo en residuos
        def bias_penalized_rmse(estimator, X, y):
            """
            Métrica personalizada que penaliza tanto el RMSE como el sesgo sistemático en residuos.
            Mayor penalización a modelos con sesgo direccional.
            """
            # Predecir en la escala transformada
            y_pred = estimator.predict(X)
            
            # Cálculo estándar RMSE
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Penalización por sesgo (residuos con dirección sistemática)
            residuals = y - y_pred
            mean_residual = np.mean(residuals)
            
            # Penalizar sesgo como porcentaje de RMSE (0 = sin sesgo)
            bias_penalty = np.abs(mean_residual) / (rmse + 1e-10)
            
            # Ajustar la métrica (mayor = mejor)
            return -rmse * (1 + bias_penalty)
        
        # Registrar scorer personalizado si se desea usar
        from sklearn.metrics import make_scorer
        custom_scorers = {
            'bias_penalized_rmse': make_scorer(bias_penalized_rmse),
            'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
            'r2': 'r2'
        }
        
        # Usar el scorer personalizado si se especifica
        if scoring == 'bias_penalized_rmse':
            scorer = custom_scorers['bias_penalized_rmse']
        else:
            scorer = scoring
        
        # Ajustar X e y para optimización
        X_train_opt = self.X_train.copy()
        y_train_opt = self.y_train.copy()
        
        results = {}
        best_models = {}
        
        # Optimizar modelos según el grid de parámetros
        for model_name, params in param_grid.items():
            if model_name in self.models:
                logger.info(f"Optimizando hiperparámetros para {model_name}...")
                
                try:
                    # Clonar modelo base para evitar modificar el original
                    from sklearn.base import clone
                    base_model = clone(self.models[model_name])
                    
                    # Crear búsqueda aleatoria
                    search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=params,
                        n_iter=n_iter,
                        cv=cv_splitter,
                        scoring=scorer,
                        n_jobs=-1,
                        random_state=42,
                        verbose=1,
                        return_train_score=True,
                        error_score='raise'
                    )
                    
                    # Ajustar la búsqueda (con manejo de pesos si están disponibles)
                    if hasattr(self, 'sample_weights') and self.sample_weights is not None:
                        # Verificar que los pesos coincidan con la longitud de los datos
                        if len(self.sample_weights) == len(y_train_opt):
                            logger.info(f"Usando pesos de muestra para {model_name}")
                            search.fit(X_train_opt, y_train_opt, sample_weight=self.sample_weights)
                        else:
                            logger.warning(f"Longitud de pesos no coincide con los datos. Optimizando sin pesos.")
                            search.fit(X_train_opt, y_train_opt)
                    else:
                        search.fit(X_train_opt, y_train_opt)
                    
                    # Guardar el mejor modelo
                    best_models[model_name] = search.best_estimator_
                    
                    # Actualizar el modelo en la instancia principal
                    self.models[model_name] = search.best_estimator_
                    
                    # Calcular métricas adicionales con el mejor modelo
                    best_y_pred = search.best_estimator_.predict(X_train_opt)
                    best_rmse = np.sqrt(mean_squared_error(y_train_opt, best_y_pred))
                    best_r2 = r2_score(y_train_opt, best_y_pred)
                    best_residuals = y_train_opt - best_y_pred
                    best_mean_residual = np.mean(best_residuals)
                    
                    # Guardar resultados
                    results[model_name] = {
                        'best_params': search.best_params_,
                        'best_score': search.best_score_,
                        'best_rmse': best_rmse,
                        'best_r2': best_r2,
                        'mean_residual': best_mean_residual,
                        'cv_results': {
                            'mean_train_score': np.mean(search.cv_results_['mean_train_score']),
                            'mean_test_score': np.mean(search.cv_results_['mean_test_score']),
                            'std_test_score': np.mean(search.cv_results_['std_test_score'])
                        }
                    }
                    
                    # Guardar el modelo optimizado
                    joblib.dump(
                        search.best_estimator_,
                        f"models/optimized/{model_name}_optimized.pkl"
                    )
                    
                    logger.info(f"Mejor score para {model_name}: {search.best_score_:.4f}")
                    logger.info(f"RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}, Sesgo medio: {best_mean_residual:.4f}")
                    logger.info(f"Mejores parámetros para {model_name}: {search.best_params_}")
                    
                except Exception as e:
                    logger.error(f"Error optimizando {model_name}: {str(e)}", exc_info=True)
                    results[model_name] = {'error': str(e)}
        
        # Agregar la combinación optimizada de modelos al resultado
        results['best_models'] = best_models
        
        return results

    def build_ensemble_model(self, base_models=None, meta_model=None, n_folds=5):
        """
        Construye un modelo de ensamblado avanzado combinando varios algoritmos base.
        Implementa stacking, blending y votación ponderada para mejorar la precisión.
        
        Args:
            base_models: Lista de nombres de modelos base a incluir (None para automático)
            meta_model: Modelo meta a usar (None para LassoCV adaptativo)
            n_folds: Número de folds para validación cruzada en el ensamblado
        
        Returns:
            object: Modelo de ensamblado entrenado
        """
        from sklearn.ensemble import VotingRegressor, StackingRegressor
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LassoCV
        
        # Si no se especifican modelos base, seleccionar automáticamente los mejores
        if base_models is None:
            # Seleccionar modelos base con base en sus métricas si están disponibles
            if hasattr(self, 'validation_results') and self.validation_results:
                # Ordenar modelos por R² o RMSE
                sorted_models = []
                for model_name, metrics in self.validation_results.items():
                    if 'r2' in metrics and not np.isnan(metrics['r2']):
                        sorted_models.append((model_name, metrics['r2']))
                
                # Ordenar por R² descendente
                sorted_models.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar los mejores 5 modelos o todos si hay menos
                base_models = [m[0] for m in sorted_models[:5]]
            else:
                # Sin métricas de validación, usar una selección predeterminada
                base_models = ['hist_gb', 'xgboost', 'ridge', 'elastic', 'linear']
        
        # Filtrar para incluir solo modelos que existen
        available_base_models = [m for m in base_models if m in self.models]
        
        if len(available_base_models) < 2:
            logger.error("No hay suficientes modelos base disponibles para ensamblado (mínimo 2)")
            return None
        
        # Preparar estimadores base con nombres válidos para scikit-learn
        base_estimators = [(f"base_{i}_{name}", self.models[name]) 
                          for i, name in enumerate(available_base_models)]
        
        # Configurar modelo meta (el que combina las predicciones de los modelos base)
        if meta_model is None:
            # Meta-regresor adaptativo con selección de características
            meta_model = LassoCV(
                alphas=[0.001, 0.01, 0.1, 0.5, 1.0, 10.0],
                cv=3,
                max_iter=2000,
                tol=0.001,
                random_state=42
            )
        
        # Configurar CV para el stacking
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Crear tres tipos de ensamblados para máximo rendimiento
        
        # 1. Ensamblado por stacking (usa predicciones de modelos base como features)
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=cv,
            n_jobs=-1,
            passthrough=False  # No incluir características originales
        )
        
        # 2. Ensamblado por votación ponderada (promedio ponderado de predicciones)
        # Asignar pesos iniciales iguales
        weights = [1.0] * len(base_estimators)
        voting = VotingRegressor(
            estimators=base_estimators,
            weights=weights
        )
        
        # 3. Ensamblado por blending (stacking + votación)
        # Primero entrenar los modelos individuales
        logger.info(f"Entrenando ensamblado con {len(available_base_models)} modelos base")
        
        # Verificar si tenemos datos para entrenamiento
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            try:
                # Entrenar stacking
                stacking.fit(self.X_train, self.y_train)
                logger.info("Ensamblado por stacking entrenado correctamente")
                
                # Entrenar votación
                voting.fit(self.X_train, self.y_train)
                logger.info("Ensamblado por votación entrenado correctamente")
                
                # Agregar los ensamblados a los modelos disponibles
                self.models['stacking_ensemble'] = stacking
                self.models['voting_ensemble'] = voting
                
                # Implementar blending manual (combinación de stacking y votación)
                # Esta es una implementación simple - se pueden explorar otras más sofisticadas
                class BlendingEnsemble:
                    def __init__(self, stacking, voting, blend_weight=0.7):
                        self.stacking = stacking
                        self.voting = voting
                        self.blend_weight = blend_weight
                    
                    def predict(self, X):
                        stacking_pred = self.stacking.predict(X)
                        voting_pred = self.voting.predict(X)
                        return self.blend_weight * stacking_pred + (1 - self.blend_weight) * voting_pred
                    
                    def fit(self, X, y):
                        # Los modelos ya están entrenados
                        return self
                
                # Crear y agregar el modelo de blending
                blending = BlendingEnsemble(stacking, voting, blend_weight=0.7)
                self.models['blending_ensemble'] = blending
                
                # Evaluar los ensamblados en datos de validación si están disponibles
                if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
                    from sklearn.metrics import mean_squared_error, r2_score
                    
                    # Evaluar stacking
                    stacking_pred = stacking.predict(self.X_test)
                    stacking_rmse = np.sqrt(mean_squared_error(self.y_test, stacking_pred))
                    stacking_r2 = r2_score(self.y_test, stacking_pred)
                    
                    # Evaluar voting
                    voting_pred = voting.predict(self.X_test)
                    voting_rmse = np.sqrt(mean_squared_error(self.y_test, voting_pred))
                    voting_r2 = r2_score(self.y_test, voting_pred)
                    
                    # Evaluar blending
                    blending_pred = blending.predict(self.X_test)
                    blending_rmse = np.sqrt(mean_squared_error(self.y_test, blending_pred))
                    blending_r2 = r2_score(self.y_test, blending_pred)
                    
                    logger.info(f"Stacking - RMSE: {stacking_rmse:.4f}, R²: {stacking_r2:.4f}")
                    logger.info(f"Voting - RMSE: {voting_rmse:.4f}, R²: {voting_r2:.4f}")
                    logger.info(f"Blending - RMSE: {blending_rmse:.4f}, R²: {blending_r2:.4f}")
                    
                    # Actualizar el modelo "best" si el blending es mejor que el actual mejor
                    if not hasattr(self, 'best_model') or blending_r2 > self.best_model['r2']:
                        self.best_model = {
                            'name': 'blending_ensemble',
                            'model': blending,
                            'rmse': blending_rmse,
                            'r2': blending_r2
                        }
                        logger.info(f"Blending ensemble establecido como mejor modelo (R²: {blending_r2:.4f})")
                
                return blending  # Devolver el modelo de blending como resultado principal
                
            except Exception as e:
                logger.error(f"Error entrenando ensamblado: {str(e)}", exc_info=True)
                return None
        else:
            logger.error("No hay datos de entrenamiento disponibles para el ensamblado")
            return None
    
    def train_models(self, X_train, y_train, use_scaling=True):
        """
        Entrena modelos mejorados para predicción de puntos, incluyendo
        ensemble avanzado y manejo optimizado de heteroscedasticidad.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            use_scaling: Si aplicar escalado
        """
        # Primero, entrenar modelos base de la clase padre
        # (esto inicializa self.models con los modelos básicos)
        super().train_models(X_train, y_train, use_scaling)
        
        # Valores originales para usar en transformación inversa si es necesario
        if hasattr(self, 'y_transform') and self.y_transform == 'log':
            y_train_original = self.y_train_original
        else:
            y_train_original = y_train
        
        # ===== NUEVAS MEJORAS: Transformaciones adaptativas =====
        logger.info("Aplicando transformaciones adaptativas por rangos")
        
        # Verificar y manejar valores NaN en los datos
        has_nans = X_train.isna().any().any()
        if has_nans:
            logger.warning(f"Detectados valores NaN en los datos de entrenamiento. Imputando con medianas.")
            # Crear una copia para no modificar los datos originales
            X_train_clean = X_train.copy()
            # Imputar NaNs con medianas por columna
            X_train_clean = X_train_clean.fillna(X_train_clean.median())
        else:
            X_train_clean = X_train
        
        # Dividir el espacio de predicción en tres regiones
        y_percentiles = np.percentile(y_train_original, [33, 66])
        low_mask = y_train_original <= y_percentiles[0]
        high_mask = y_train_original > y_percentiles[1]
        mid_mask = ~(low_mask | high_mask)
        
        # Particionamiento inteligente: crear datos para cada región
        X_low, y_low = X_train_clean.loc[low_mask], y_train.loc[low_mask]
        X_mid, y_mid = X_train_clean.loc[mid_mask], y_train.loc[mid_mask]
        X_high, y_high = X_train_clean.loc[high_mask], y_train.loc[high_mask]
        
        logger.info(f"Distribución de datos: Bajo={sum(low_mask)}, Medio={sum(mid_mask)}, Alto={sum(high_mask)}")
        
        # Verificar NaNs en cada región
        for region_name, X_region in [("bajo", X_low), ("medio", X_mid), ("alto", X_high)]:
            if X_region.isna().any().any():
                logger.warning(f"Región {region_name} contiene valores NaN después de la imputación. "
                              f"Realizando imputación específica para esta región.")
                # Imputar NaNs adicionales si quedaron
                region_medians = X_region.median()
                X_region.fillna(region_medians, inplace=True)
                # Si aún quedan NaNs (columnas completamente NaN), usar cero
                X_region.fillna(0, inplace=True)
        
        # ===== Transformación adaptativa Yeo-Johnson =====
        try:
            from sklearn.preprocessing import PowerTransformer
            
            # Transformadores específicos para cada región
            pt_low = PowerTransformer(method='yeo-johnson')
            pt_mid = PowerTransformer(method='yeo-johnson')
            pt_high = PowerTransformer(method='yeo-johnson')
            
            # Guardar las transformaciones para aplicarlas luego en predicción
            self.region_transformers = {
                'low': pt_low.fit(X_low),
                'mid': pt_mid.fit(X_mid),
                'high': pt_high.fit(X_high)
            }
            
            # Transformar datos
            X_low_trans = pd.DataFrame(pt_low.transform(X_low), columns=X_low.columns, index=X_low.index)
            X_mid_trans = pd.DataFrame(pt_mid.transform(X_mid), columns=X_mid.columns, index=X_mid.index)
            X_high_trans = pd.DataFrame(pt_high.transform(X_high), columns=X_high.columns, index=X_high.index)
            
            logger.info("Transformación Yeo-Johnson aplicada por regiones")
        except Exception as e:
            logger.warning(f"Error aplicando transformación Yeo-Johnson: {str(e)}")
            # Si falla, usar los datos originales
            X_low_trans, X_mid_trans, X_high_trans = X_low, X_mid, X_high
            self.region_transformers = None
        
        # ===== NUEVOS MODELOS ESPECIALIZADOS POR REGIÓN =====
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import HuberRegressor, ElasticNet
        import xgboost as xgb
        
        try:
            # Verificar NaNs finales antes de entrenar modelos
            for region_name, X_region in [("bajo_trans", X_low_trans), ("medio_trans", X_mid_trans), ("alto_trans", X_high_trans)]:
                if X_region.isna().any().any():
                    logger.warning(f"La región {region_name} aún contiene valores NaN. Aplicando imputación final.")
                    # Imputación final
                    X_region.fillna(X_region.median(), inplace=True)
                    # Si aún quedan NaNs, usar cero
                    X_region.fillna(0, inplace=True)
            
            # Modelo para valores bajos (0-33%)
            # Histgradient maneja NaNs nativamente
            from sklearn.ensemble import HistGradientBoostingRegressor
            
            logger.info("Entrenando modelo para valores bajos")
            low_model = HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=4,
                learning_rate=0.05,
                l2_regularization=1.0,
                max_bins=255,  # Aumentar bins para mejor precisión
                random_state=42
            )
            low_model.fit(X_low_trans, y_low)
            logger.info("Modelo especializado para valores bajos entrenado")
            
            # Modelo para valores medios (33-66%)
            logger.info("Entrenando modelo para valores medios")
            mid_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                colsample_bytree=0.8,
                reg_alpha=0.5,  # L1 regularización
                reg_lambda=1.0,  # L2 regularización
                random_state=42
            )
            mid_model.fit(X_mid_trans, y_mid)
            logger.info("Modelo especializado para valores medios entrenado")
            
            # Modelo para valores altos (66-100%)
            logger.info("Entrenando modelo para valores altos")
            high_model = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=8,
                learning_rate=0.03,
                colsample_bytree=0.7,
                reg_alpha=1.5,  # Regularización más fuerte para evitar sobreajuste
                reg_lambda=2.5,
                gamma=0.1,
                random_state=42
            )
            high_model.fit(X_high_trans, y_high)
            logger.info("Modelo especializado para valores altos entrenado")
            
            # Guardar los modelos regionales
            self.region_models = {
                'low': low_model,
                'mid': mid_model,
                'high': high_model
            }
            
            # Añadir límites de regiones para usar en predicción
            self.region_limits = {
                'low_threshold': y_percentiles[0],
                'high_threshold': y_percentiles[1]
            }
            
            # Implementar mezclado suave entre regiones
            # Función para calcular pesos según distancia a los umbrales
            def calculate_weights(y_pred):
                # Inicializar con ceros
                weights_low = np.zeros_like(y_pred, dtype=float)
                weights_mid = np.zeros_like(y_pred, dtype=float)
                weights_high = np.zeros_like(y_pred, dtype=float)
                
                # Asignar pesos basados en la posición relativa a los umbrales
                low_t = self.region_limits['low_threshold']
                high_t = self.region_limits['high_threshold']
                blend_range = (high_t - low_t) * 0.2  # 20% de overlap para suavizar
                
                # Calcular pesos para cada región con transición suave
                for i, val in enumerate(y_pred):
                    if val <= low_t - blend_range:
                        # Completamente en región baja
                        weights_low[i] = 1.0
                    elif val <= low_t + blend_range:
                        # Zona de transición bajo-medio
                        d = (val - (low_t - blend_range)) / (2 * blend_range)
                        weights_low[i] = 1.0 - d
                        weights_mid[i] = d
                    elif val <= high_t - blend_range:
                        # Completamente en región media
                        weights_mid[i] = 1.0
                    elif val <= high_t + blend_range:
                        # Zona de transición medio-alto
                        d = (val - (high_t - blend_range)) / (2 * blend_range)
                        weights_mid[i] = 1.0 - d
                        weights_high[i] = d
                    else:
                        # Completamente en región alta
                        weights_high[i] = 1.0
                
                return weights_low, weights_mid, weights_high
            
            self.calculate_region_weights = calculate_weights
            
            # Añadir modelos al diccionario principal
            self.models.update({
                'region_low': low_model,
                'region_mid': mid_model,
                'region_high': high_model
            })
            
            # Crear modelo de ensemblado adaptativo regional
            class AdaptiveRegionalEnsemble:
                def __init__(self, region_models, region_transformers, calculate_weights, region_limits):
                    self.region_models = region_models
                    self.region_transformers = region_transformers
                    self.calculate_weights = calculate_weights
                    self.region_limits = region_limits
                    
                def predict(self, X):
                    # Manejar valores NaN en entrada
                    X_clean = X.copy()
                    if hasattr(X_clean, 'isna') and X_clean.isna().any().any():
                        # Imputar NaNs
                        X_clean = X_clean.fillna(X_clean.median())
                        # Si aún quedan NaNs, usar cero
                        X_clean = X_clean.fillna(0)
                    
                    # Paso 1: Obtener predicciones base usando todos los modelos
                    X_low = X_clean.copy()
                    X_mid = X_clean.copy()
                    X_high = X_clean.copy()
                    
                    # Aplicar transformaciones si están disponibles
                    if self.region_transformers:
                        try:
                            X_low = pd.DataFrame(self.region_transformers['low'].transform(X_low), 
                                                columns=X_low.columns, index=X_low.index)
                            X_mid = pd.DataFrame(self.region_transformers['mid'].transform(X_mid), 
                                                columns=X_mid.columns, index=X_mid.index)
                            X_high = pd.DataFrame(self.region_transformers['high'].transform(X_high), 
                                                columns=X_high.columns, index=X_high.index)
                        except Exception as e:
                            # Si la transformación falla, usar datos sin transformar
                            pass
                    
                    # Predicciones iniciales
                    pred_low = self.region_models['low'].predict(X_low)
                    pred_mid = self.region_models['mid'].predict(X_mid)
                    pred_high = self.region_models['high'].predict(X_high)
                    
                    # Paso 2: Predicción inicial para calcular pesos
                    # Usar modelo mid como base para la primera estimación
                    initial_pred = pred_mid.copy()
                    
                    # Paso 3: Calcular pesos para cada región basado en esta predicción inicial
                    weights_low, weights_mid, weights_high = self.calculate_weights(initial_pred)
                    
                    # Paso 4: Combinar predicciones ponderadas
                    final_pred = (weights_low * pred_low + 
                                 weights_mid * pred_mid + 
                                 weights_high * pred_high)
                    
                    # Paso 5: Aplicar corrección de sesgo basada en análisis de residuos
                    # Ajustar predicciones muy bajas (<2 puntos)
                    very_low_idx = final_pred < 2.0
                    if np.any(very_low_idx):
                        final_pred[very_low_idx] *= 1.05  # +5% para compensar subestimación
                    
                    # Ajustar predicciones muy altas (>30 puntos)
                    very_high_idx = final_pred > 30.0
                    if np.any(very_high_idx):
                        final_pred[very_high_idx] *= 0.97  # -3% para compensar sobreestimación
                    
                    return final_pred
                    
                def fit(self, X, y):
                    # Modelos ya entrenados
                    return self
            
            # Instanciar el ensemble adaptativo regional
            adaptive_regional_ensemble = AdaptiveRegionalEnsemble(
                self.region_models,
                self.region_transformers,
                self.calculate_region_weights,
                self.region_limits
            )
            
            # Añadir al diccionario de modelos
            self.models['adaptive_regional_ensemble'] = adaptive_regional_ensemble
            
            # Establecer como modelo predeterminado (best)
            self.models['best'] = adaptive_regional_ensemble
            
            logger.info("Ensemble adaptativo regional creado y establecido como modelo predeterminado")
            
        except Exception as e:
            logger.error(f"Error entrenando modelos regionales: {str(e)}")
            # Si algo falla, asegurarse de que tengamos un fallback
            # Crear modelos adicionales especializados para puntos
            try:
                logger.info("Fallback: Entrenando modelos especializados para puntos")
                self._setup_specialized_models(X_train_clean, y_train)
            except Exception as e2:
                logger.error(f"Error en fallback: {str(e2)}")
        
        # Indicar que el modelo está ajustado
        self.is_fitted = True
        
        return self.models
    
    def analyze_feature_importance(self, X_test, y_test, model_name=None):
        """
        Analiza la importancia de las características, con especial atención a
        las transformaciones de FGA.
        
        Args:
            X_test: Características de test
            y_test: Target de test
            model_name: Nombre del modelo a evaluar (None para todos)
            
        Returns:
            dict: Resultados del análisis de importancia
        """
        import matplotlib.pyplot as plt
        
        # Si no se especifica modelo, analizar el mejor
        if model_name is None:
            model_name, model, _ = self.get_best_model()
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        # Verificar si el modelo tiene atributo de importancia de características
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            logger.warning(f"El modelo {model_name} no tiene atributo de importancia de características")
            return {}
        
        # Obtener importancia
        feature_names = X_test.columns
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # Para modelos lineales con coeficientes
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = importance.mean(axis=0)
        
        # Crear DataFrame de importancia
        if len(importance) == len(feature_names):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
        else:
            logger.warning(f"Dimensiones no coinciden: {len(importance)} vs {len(feature_names)}")
            return {}
        
        # Identificar características relacionadas con FGA
        fga_features = importance_df[importance_df['feature'].str.contains('FGA')]
        
        # Visualizar importancia de características
        plt.figure(figsize=(12, 8))
        
        # Mostrar top 15 características
        top_n = 15
        importance_df.head(top_n).set_index('feature')['importance'].plot(kind='barh')
        
        plt.title(f'Top {top_n} Características Importantes - {model_name}')
        plt.xlabel('Importancia')
        plt.tight_layout()
        
        # Guardar figura
        importance_plot_path = f'reports/figures/feature_importance_{model_name}.png'
        os.makedirs(os.path.dirname(importance_plot_path), exist_ok=True)
        plt.savefig(importance_plot_path)
        plt.close()
        
        # Visualización específica para características FGA
        if len(fga_features) > 0:
            plt.figure(figsize=(10, 6))
            fga_features.set_index('feature')['importance'].plot(kind='barh')
            plt.title(f'Importancia de Transformaciones FGA - {model_name}')
            plt.xlabel('Importancia')
            plt.tight_layout()
            
            # Guardar figura
            fga_plot_path = f'reports/figures/fga_importance_{model_name}.png'
            plt.savefig(fga_plot_path)
            plt.close()
        
        # Devolver resultados
        analysis = {
            'top_features': importance_df.head(top_n).to_dict('records'),
            'fga_features': fga_features.to_dict('records') if len(fga_features) > 0 else [],
            'importance_plot': importance_plot_path,
            'fga_plot': fga_plot_path if len(fga_features) > 0 else None
        }
        
        return analysis
    
    def cross_validate_temporal(self, X, y, n_splits=5, train_prop=0.7, gap_prop=0.05, models=None):
        """
        Realiza validación cruzada con divisiones temporales, respetando la cronología de los datos.
        Implementa un esquema de ventana expandible con gap entre train y test.
        
        Args:
            X (pd.DataFrame): Features completos
            y (pd.Series): Target completo
            n_splits (int): Número de divisiones temporales
            train_prop (float): Proporción inicial de datos para entrenamiento
            gap_prop (float): Proporción de datos a usar como gap entre train y test
            models (list): Lista de nombres de modelos a evaluar (None para todos)
            
        Returns:
            dict: Resultados de validación cruzada por modelo
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error, r2_score
        
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X debe ser un DataFrame y y debe ser una Series")
        
        # Verificar si tenemos fecha en el índice
        if not isinstance(X.index, pd.DatetimeIndex) and 'Date' in X.columns:
            # Ordenar por fecha
            date_col = X['Date'].copy()
            X = X.drop(columns=['Date'])
            X_y = pd.concat([X, y], axis=1)
            X_y = X_y.sort_values(by='Date')
            X = X_y.drop(columns=[y.name])
            y = X_y[y.name]
        elif not isinstance(X.index, pd.DatetimeIndex):
            logger.warning("No se encontró una columna de fecha. Utilizando índice actual como proxy temporal.")
        
        # Determinar los modelos a evaluar
        if models is None:
            models = list(self.models.keys())
        else:
            # Filtrar solo modelos que existen
            models = [m for m in models if m in self.models]
        
        # Inicializar resultados
        cv_results = {model_name: {'rmse': [], 'mae': [], 'r2': [], 'test_indices': []} for model_name in models}
        
        # Calcular índices para splits
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calcular tamaños iniciales
        initial_train_size = int(n_samples * train_prop / n_splits)
        gap_size = int(n_samples * gap_prop)
        
        # Preparar splits temporales
        splits = []
        for i in range(n_splits):
            # Tamaño de entrenamiento crece con cada split (ventana expandible)
            train_end = initial_train_size + i * ((n_samples - initial_train_size - gap_size * n_splits) // n_splits)
            test_start = train_end + gap_size
            test_end = test_start + (n_samples - test_start) // (n_splits - i) if i < n_splits - 1 else n_samples
            
            # Asegurar que no nos pasamos de los límites
            train_end = min(train_end, n_samples - gap_size - 1)
            test_start = min(test_start, n_samples - 1)
            test_end = min(test_end, n_samples)
            
            # Crear índices
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        # Realizar validación cruzada
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Split {i+1}/{n_splits}: Train size={len(train_idx)}, Test size={len(test_idx)}")
            
            # Obtener datos de este split
            X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
            y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
            
            # Preparar datos (aplicar transformaciones)
            X_train_proc, X_test_proc = X_train_split.copy(), X_test_split.copy()
            y_train_proc, y_test_proc = y_train_split.copy(), y_test_split.copy()
            
            # Aplicar transformación logarítmica a targets
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                y_train_proc = np.log1p(y_train_proc)
                y_test_proc = np.log1p(y_test_proc)
            
            # Entrenar y evaluar cada modelo
            for model_name in models:
                try:
                    # Clonar el modelo original para no afectar al existente
                    from sklearn.base import clone
                    model = clone(self.models[model_name])
                    
                    # Entrenar modelo en este split
                    model.fit(X_train_proc, y_train_proc)
                    
                    # Predecir
                    y_pred = model.predict(X_test_proc)
                    
                    # Si se aplicó transformación, deshacer
                    if hasattr(self, 'y_transform') and self.y_transform == 'log':
                        y_pred = np.expm1(y_pred)
                    
                    # Calcular métricas en escala original
                    rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
                    mae = mean_absolute_error(y_test_split, y_pred)
                    r2 = r2_score(y_test_split, y_pred)
                    
                    # Guardar resultados
                    cv_results[model_name]['rmse'].append(rmse)
                    cv_results[model_name]['mae'].append(mae)
                    cv_results[model_name]['r2'].append(r2)
                    cv_results[model_name]['test_indices'].append(test_idx)
                    
                    logger.info(f"  {model_name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error en CV temporal para {model_name}, split {i+1}: {str(e)}")
                    # Añadir NaN para este split
                    cv_results[model_name]['rmse'].append(np.nan)
                    cv_results[model_name]['mae'].append(np.nan)
                    cv_results[model_name]['r2'].append(np.nan)
                    cv_results[model_name]['test_indices'].append(test_idx)
        
        # Calcular métricas agregadas
        for model_name in models:
            rmse_values = np.array(cv_results[model_name]['rmse'])
            mae_values = np.array(cv_results[model_name]['mae'])
            r2_values = np.array(cv_results[model_name]['r2'])
            
            # Filtrar NaN
            valid_rmse = rmse_values[~np.isnan(rmse_values)]
            valid_mae = mae_values[~np.isnan(mae_values)]
            valid_r2 = r2_values[~np.isnan(r2_values)]
            
            if len(valid_rmse) > 0:
                cv_results[model_name]['mean_rmse'] = valid_rmse.mean()
                cv_results[model_name]['std_rmse'] = valid_rmse.std()
                cv_results[model_name]['mean_mae'] = valid_mae.mean()
                cv_results[model_name]['std_mae'] = valid_mae.std()
                cv_results[model_name]['mean_r2'] = valid_r2.mean()
                cv_results[model_name]['std_r2'] = valid_r2.std()
                
                logger.info(f"{model_name} - Promedio: RMSE={valid_rmse.mean():.3f}±{valid_rmse.std():.3f}, "
                          f"R²={valid_r2.mean():.3f}±{valid_r2.std():.3f}")
        
        # Visualizar resultados de validación cruzada
        plt.figure(figsize=(12, 8))
        
        # Preparar datos para gráfico
        model_names = []
        mean_rmses = []
        std_rmses = []
        
        for model_name in models:
            if 'mean_rmse' in cv_results[model_name]:
                model_names.append(model_name)
                mean_rmses.append(cv_results[model_name]['mean_rmse'])
                std_rmses.append(cv_results[model_name]['std_rmse'])
        
        # Ordenar por RMSE (menor a mayor)
        sorted_indices = np.argsort(mean_rmses)
        model_names = [model_names[i] for i in sorted_indices]
        mean_rmses = [mean_rmses[i] for i in sorted_indices]
        std_rmses = [std_rmses[i] for i in sorted_indices]
        
        # Crear gráfico de barras con barras de error
        bars = plt.bar(model_names, mean_rmses, yerr=std_rmses, capsize=5, alpha=0.7)
        
        # Añadir etiquetas con valores
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std_rmses[i]*0.5,
                    f'{mean_rmses[i]:.2f}±{std_rmses[i]:.2f}',
                    ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title('Comparación de Modelos (Validación Cruzada Temporal)')
        plt.xlabel('Modelo')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Guardar figura
        cv_plot_path = 'reports/figures/temporal_cv_comparison.png'
        os.makedirs(os.path.dirname(cv_plot_path), exist_ok=True)
        plt.savefig(cv_plot_path)
        plt.close()
        
        # Añadir ruta de la figura a los resultados
        cv_results['plot_path'] = cv_plot_path
        
        return cv_results 
    
    def create_run_script(self, output_file="run_points_model.py"):
        """
        Crea un script de ejecución para entrenar y evaluar el modelo de puntos
        con todas las mejoras implementadas.
        
        Args:
            output_file (str): Ruta del archivo de salida
            
        Returns:
            str: Ruta del archivo creado
        """
        script_content = """
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from src.models.points_model import PointsModel
from src.data.data_processor import DataProcessor

# Configurar logging solo si no está ya configurado
root_logger = logging.getLogger()
if not root_logger.handlers:
    # Crear directorio para logs
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'points_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

def main():
    # Crear directorios para resultados
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info("Iniciando entrenamiento del modelo de puntos mejorado")
    
    # Cargar y procesar datos
    try:
        data_processor = DataProcessor()
        df = data_processor.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No se pudieron cargar los datos")
            return
            
        logger.info(f"Datos cargados: {len(df)} registros")
        
        # Preprocesar datos
        df = data_processor.preprocess_data(df)
        
        # Inicializar modelo
        model = PointsModel()
        
        # Preparar datos con detección de outliers avanzada
        X_train, X_test, y_train, y_test = model.prepare_data(
            df, 
            test_size=0.2, 
            time_split=True,
            detect_outliers=True
        )
        
        # Entrenar modelos con pesos para manejar heteroscedasticidad
        results = model.train_models(X_train, y_train, use_scaling=True)
        
        # Validar modelos
        validation = model.validate_models(X_test, y_test)
        
        # Analizar heteroscedasticidad
        hetero_analysis = model.analyze_heteroscedasticity(X_test, y_test)
        logger.info(f"Análisis de heteroscedasticidad: {hetero_analysis}")
        
        # Realizar validación cruzada temporal
        temporal_cv = model.cross_validate_temporal(
            X=pd.concat([X_train, X_test]),
            y=pd.concat([model.y_train_original, model.y_test_original]),
            n_splits=5,
            train_prop=0.6,
            gap_prop=0.05
        )
        
        # Analizar importancia de características
        feature_importance = model.analyze_feature_importance(X_test, y_test)
        
        # Analizar errores de predicción en detalle
        error_analysis = model.analyze_prediction_errors(X_test, y_test)
        
        # Obtener y guardar el mejor modelo
        best_model_name, best_model, best_score = model.get_best_model()
        logger.info(f"Mejor modelo: {best_model_name} (Score: {best_score:.4f})")
        
        # Guardar modelo
        import joblib
        model_path = os.path.join('models', 'points_model_enhanced.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado en {model_path}")
        
        # Generar visualizaciones adicionales de residuos
        plt.figure(figsize=(12, 8))
        best_preds = model.predict(X_test, best_model_name)
        
        plt.subplot(2, 2, 1)
        plt.scatter(best_preds, model.y_test_original, alpha=0.5)
        plt.plot([0, max(model.y_test_original)], [0, max(model.y_test_original)], 'r--')
        plt.title(f'Predicciones vs Reales - {best_model_name}')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        
        plt.subplot(2, 2, 2)
        residuals = model.y_test_original - best_preds
        plt.scatter(best_preds, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuos vs Predicciones')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        
        plt.subplot(2, 2, 3)
        sns.histplot(residuals, kde=True)
        plt.title('Distribución de Residuos')
        plt.xlabel('Residuos')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(y=residuals)
        plt.title('Boxplot de Residuos')
        plt.ylabel('Residuos')
        
        plt.tight_layout()
        plt.savefig('reports/figures/residual_analysis_enhanced.png')
        plt.close()
        
        logger.info("Análisis completo. Revise las gráficas en reports/figures/")
        
    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main()
        """
        
        # Escribir el script a un archivo
        script_path = output_file
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        logger.info(f"Script de ejecución creado en {script_path}")
        
        return script_path

    def optimize_ensemble_weights(self, X_val, y_val, metric='rmse', n_trials=50):
        """
        Optimiza los pesos del ensemble mediante validación para obtener el mejor rendimiento.
        
        Args:
            X_val: Características de validación
            y_val: Variable objetivo de validación
            metric: Métrica a optimizar ('rmse', 'mae', 'bias')
            n_trials: Número de combinaciones de pesos a probar
            
        Returns:
            dict: Mejores pesos y rendimiento
        """
        logger.info(f"Optimizando pesos del ensemble usando {metric} como métrica principal")
        
        # Verificar que tenemos los modelos base necesarios
        required_models = ['xgb_optimized', 'hist_gb_robust', 'lgbm_specialized', 'random_forest']
        if not all(model in self.models for model in required_models):
            logger.error("No se encuentran todos los modelos base requeridos")
            return None
        
        # Si tenemos transformación logarítmica, usar los valores originales para evaluación
        if hasattr(self, 'y_transform') and self.y_transform == 'log' and hasattr(self, 'y_test_original'):
            y_val_original = np.expm1(y_val)  # Transformación inversa
        else:
            y_val_original = y_val
            
        # Generar predicciones de cada modelo base
        model_preds = {}
        
        for model_name in required_models:
            logger.info(f"Generando predicciones para {model_name}")
            try:
                model_pred = self.predict(X_val, model_name=model_name)
                model_preds[model_name] = model_pred
            except Exception as e:
                logger.error(f"Error obteniendo predicciones de {model_name}: {str(e)}")
                return None
                
        # Función para evaluar una combinación de pesos
        def evaluate_weights(weights):
            # Normalizar pesos
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Combinar predicciones con los pesos
            ensemble_pred = np.zeros_like(y_val_original)
            
            for i, model_name in enumerate(required_models):
                ensemble_pred += weights[i] * model_preds[model_name]
                
            # Calcular métricas
            rmse = np.sqrt(mean_squared_error(y_val_original, ensemble_pred))
            mae = mean_absolute_error(y_val_original, ensemble_pred)
            bias = np.mean(ensemble_pred - y_val_original)  # Sesgo (positivo = sobreestimación)
            
            # Retornar métrica principal
            if metric == 'rmse':
                return rmse
            elif metric == 'mae':
                return mae
            elif metric == 'bias':
                return abs(bias)  # Minimizar el sesgo absoluto
            else:
                return rmse
        
        # Generar combinaciones aleatorias de pesos y evaluarlas
        best_score = float('inf')
        best_weights = [0.25, 0.25, 0.25, 0.25]  # Pesos iguales por defecto
        
        for _ in range(n_trials):
            # Generar pesos aleatorios que suman 1
            weights = np.random.random(len(required_models))
            weights = weights / weights.sum()
            
            # Evaluar
            score = evaluate_weights(weights)
            
            # Actualizar si es mejor
            if score < best_score:
                best_score = score
                best_weights = weights
                
        # Crear y entrenar un nuevo ensemble con los mejores pesos
        from sklearn.ensemble import VotingRegressor
        
        estimators = [(name, self.models[name]) for name in required_models]
        
        # Crear un nuevo VotingRegressor con los pesos optimizados
        optimized_voting = VotingRegressor(
            estimators=estimators,
            weights=best_weights
        )
        
        # Entrenar con todos los datos disponibles
        logger.info("Entrenando ensemble con pesos optimizados")
        optimized_voting.fit(self.X_train, self.y_train)
        
        # Actualizar el BlendingEnsemble
        class OptimizedBlendingEnsemble:
            def __init__(self, voting, stacking=None, blend_weight=0.9):
                self.voting = voting
                self.stacking = stacking
                self.blend_weight = blend_weight
                
            def predict(self, X):
                voting_pred = self.voting.predict(X)
                
                # Si tenemos stacking, combinar
                if self.stacking is not None:
                    stacking_pred = self.stacking.predict(X)
                    return self.blend_weight * voting_pred + (1 - self.blend_weight) * stacking_pred
                else:
                    return voting_pred
                
            def fit(self, X, y):
                # Ya entrenado
                return self
        
        # Crear optimized_ensemble incluyendo stacking si existe
        stacking = self.models.get('stacking', None)
        optimized_ensemble = OptimizedBlendingEnsemble(optimized_voting, stacking, blend_weight=0.9)
        
        # Actualizar modelos
        self.models['optimized_voting'] = optimized_voting
        self.models['optimized_ensemble'] = optimized_ensemble
        self.models['best'] = optimized_ensemble  # Establecer como modelo por defecto
        
        # Calcular métricas finales
        final_pred = optimized_ensemble.predict(X_val)
        final_rmse = np.sqrt(mean_squared_error(y_val_original, final_pred))
        final_mae = mean_absolute_error(y_val_original, final_pred)
        final_bias = np.mean(final_pred - y_val_original)
        final_r2 = r2_score(y_val_original, final_pred)
        
        # Devolver los pesos y el rendimiento
        result = {
            'weights': dict(zip(required_models, best_weights)),
            'metrics': {
                'rmse': final_rmse,
                'mae': final_mae,
                'bias': final_bias,
                'r2': final_r2
            }
        }
        
        logger.info(f"Optimización completada. Mejores pesos: {result['weights']}")
        logger.info(f"Métricas finales: RMSE={final_rmse:.4f}, MAE={final_mae:.4f}, R²={final_r2:.4f}")
        
        return result

    def analyze_residual_distribution(self, X_test, y_test, model_name='best'):
        """
        Analiza la distribución de residuos del modelo y proporciona estadísticas detalladas.
        Identifica patrones en la distribución para guiar correcciones adaptativas.
        
        Args:
            X_test: Características de prueba
            y_test: Valores reales
            model_name: Nombre del modelo a evaluar
            
        Returns:
            dict: Estadísticas de la distribución de residuos y recomendaciones
        """
        try:
            import numpy as np
            import pandas as pd
            from scipy import stats
            
            # Obtener predicciones
            y_pred = self.predict(X_test, model_name)
            
            # Usar valores originales si hay transformación logarítmica
            if hasattr(self, 'y_transform') and self.y_transform == 'log':
                y_true = self.y_test_original if hasattr(self, 'y_test_original') else y_test
            else:
                y_true = y_test
                
            # Calcular residuos
            residuals = y_true - y_pred
            
            # Estadísticas básicas
            stats_dict = {
                'mean': float(np.mean(residuals)),
                'median': float(np.median(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals))
            }
            
            # Prueba de normalidad
            shapiro_test = stats.shapiro(residuals)
            stats_dict['shapiro_stat'] = float(shapiro_test[0])
            stats_dict['shapiro_pvalue'] = float(shapiro_test[1])
            stats_dict['is_normal'] = shapiro_test[1] > 0.05
            
            # Analizar por cuantiles para identificar patrones específicos
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            quantile_values = np.quantile(residuals, quantiles)
            
            stats_dict['quantiles'] = {str(q): float(val) for q, val in zip(quantiles, quantile_values)}
            
            # Analizar residuos por rangos de valores predichos
            y_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40]
            range_stats = []
            
            for i in range(len(y_ranges)-1):
                low, high = y_ranges[i], y_ranges[i+1]
                mask = (y_pred >= low) & (y_pred < high)
                
                if np.sum(mask) > 0:
                    range_residuals = residuals[mask]
                    range_stats.append({
                        'range': f"{low}-{high}",
                        'count': int(np.sum(mask)),
                        'mean_residual': float(np.mean(range_residuals)),
                        'median_residual': float(np.median(range_residuals)),
                        'std_residual': float(np.std(range_residuals)),
                        'fraction_negative': float(np.mean(range_residuals < 0))
                    })
            
            stats_dict['range_analysis'] = range_stats
            
            # Análisis de autocorrelación en residuos
            # Esto puede detectar patrones temporales o estructurales en los errores
            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_stat = durbin_watson(residuals)
                stats_dict['durbin_watson'] = float(dw_stat)
                stats_dict['has_autocorrelation'] = dw_stat < 1.5 or dw_stat > 2.5
            except:
                stats_dict['durbin_watson'] = None
            
            # Detección de grupos de residuos similares (clusters)
            try:
                from sklearn.cluster import KMeans
                
                # Combinar valores predichos y residuos para análisis
                cluster_data = np.column_stack((y_pred, residuals))
                
                # Detectar clusters automáticamente (3-5 clusters)
                best_silhouette = -1
                best_n_clusters = 3
                
                for n_clusters in range(3, 6):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    cluster_labels = kmeans.fit_predict(cluster_data)
                    
                    if len(set(cluster_labels)) > 1:  # Asegurarse de que hay al menos 2 clusters
                        from sklearn.metrics import silhouette_score
                        silhouette = silhouette_score(cluster_data, cluster_labels)
                        
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_n_clusters = n_clusters
                
                # Usar el mejor número de clusters
                kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(cluster_data)
                
                # Analizar cada cluster
                cluster_analysis = []
                for i in range(best_n_clusters):
                    mask = cluster_labels == i
                    cluster_residuals = residuals[mask]
                    cluster_preds = y_pred[mask]
                    
                    cluster_analysis.append({
                        'cluster_id': int(i),
                        'size': int(np.sum(mask)),
                        'mean_prediction': float(np.mean(cluster_preds)),
                        'mean_residual': float(np.mean(cluster_residuals)),
                        'std_residual': float(np.std(cluster_residuals))
                    })
                
                stats_dict['cluster_analysis'] = cluster_analysis
                stats_dict['n_clusters'] = best_n_clusters
            except Exception as e:
                stats_dict['cluster_error'] = str(e)
            
            # Generar recomendaciones basadas en el análisis
            recommendations = []
            
            # 1. Verificar sesgo general
            if abs(stats_dict['mean']) > 0.5:
                bias_direction = "positivo" if stats_dict['mean'] > 0 else "negativo"
                recommendations.append(f"Corregir sesgo {bias_direction} general de {stats_dict['mean']:.2f}")
            
            # 2. Verificar no normalidad
            if not stats_dict['is_normal']:
                if stats_dict['skewness'] > 0.5:
                    recommendations.append("Aplicar transformación para corregir asimetría positiva")
                elif stats_dict['skewness'] < -0.5:
                    recommendations.append("Aplicar transformación para corregir asimetría negativa")
            
            # 3. Revisar análisis por rangos
            for range_info in stats_dict['range_analysis']:
                if abs(range_info['mean_residual']) > 0.8:
                    direction = "sobreestimación" if range_info['mean_residual'] < 0 else "subestimación"
                    recommendations.append(f"Corregir {direction} en rango {range_info['range']} puntos")
            
            # 4. Verificar autocorrelación
            if stats_dict.get('has_autocorrelation', False):
                recommendations.append("Considerar estructura temporal/secuencial en los datos")
            
            stats_dict['recommendations'] = recommendations
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error en analyze_residual_distribution: {str(e)}")
            return {'error': str(e)}

    def _setup_specialized_models(self, X_train, y_train):
        """
        Configura modelos especializados adicionales optimizados para la predicción de puntos.
        Sirve como fallback si el ensemble adaptativo regional falla.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo
        """
        logger.info("Configurando modelos especializados para predicción de puntos")
        
        try:
            # Importar modelos necesarios
            from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
            from sklearn.linear_model import HuberRegressor, ElasticNet
            import xgboost as xgb
            
            # Asegurar que no hay valores nulos
            X_train_clean = X_train.copy()
            if hasattr(X_train_clean, 'isna') and X_train_clean.isna().any().any():
                logger.warning("Detectados valores NaN en datos para modelos fallback. Imputando...")
                X_train_clean = X_train_clean.fillna(X_train_clean.median())
                # Si aún hay NaNs (columnas completamente NaN), usar ceros
                X_train_clean = X_train_clean.fillna(0)
            
            # Modelo 1: HistGradientBoosting Regressor (maneja NaNs nativamente)
            # Este modelo es una mejor opción que GradientBoostingRegressor al manejar NaNs
            logger.info("Entrenando HistGradientBoostingRegressor robusto")
            hist_gb = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=8,
                learning_rate=0.1,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=42
            )
            # Este modelo maneja NaNs nativamente, así que podemos usar X_train directamente
            hist_gb.fit(X_train, y_train)
            self.models['hist_gb'] = hist_gb
            
            # Especificar como modelo predeterminado para tener al menos uno funcional
            self.models['best'] = hist_gb
            logger.info("HistGradientBoostingRegressor establecido como modelo predeterminado de fallback")
            
            # Modelo 2: Gradient Boosting Regressor robusto (solo si X_train_clean está libre de NaNs)
            try:
                logger.info("Entrenando GradientBoostingRegressor robusto")
                # Verificar una vez más que no hay NaNs
                if not np.isnan(X_train_clean.values).any():
                    gb_robust = GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        loss='huber',
                        alpha=0.9,
                        random_state=42
                    )
                    gb_robust.fit(X_train_clean, y_train)
                    self.models['gb_robust'] = gb_robust
                    logger.info("GradientBoostingRegressor entrenado exitosamente")
                else:
                    logger.warning("Aún hay NaNs en X_train_clean. Omitiendo GradientBoostingRegressor.")
            except Exception as e:
                logger.warning(f"Error al entrenar GradientBoostingRegressor: {str(e)}")
            
            # Modelo 3: XGBoost (maneja NaNs mejor que GradientBoostingRegressor)
            try:
                # Definir pesos para dar más importancia a valores extremos
                if hasattr(y_train, 'values'):
                    y_values = y_train.values
                else:
                    y_values = np.array(y_train)
                    
                # Calcular pesos: mayor peso para valores muy altos o muy bajos
                y_percentiles = np.percentile(y_values, [25, 75])
                weights = np.ones_like(y_values, dtype=float)
                
                # Aumentar peso de valores bajos y altos
                low_mask = y_values <= y_percentiles[0]
                high_mask = y_values >= y_percentiles[1]
                
                weights[low_mask] = 1.2  # 20% más peso
                weights[high_mask] = 1.3  # 30% más peso
                
                # XGBoost con regularización y pesos adaptativos
                logger.info("Entrenando XGBoost con pesos adaptativos")
                xgb_weighted = xgb.XGBRegressor(
                    n_estimators=250,
                    max_depth=7,
                    learning_rate=0.05,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    reg_alpha=1.0,    # L1 regularización
                    reg_lambda=1.5,   # L2 regularización
                    gamma=0.05,
                    random_state=42
                )
                # XGBoost maneja NaNs, pero puede tener problemas con demasiados
                # Asegurar que está limpio
                xgb_weighted.fit(X_train_clean, y_train, sample_weight=weights)
                self.models['xgb_weighted'] = xgb_weighted
                logger.info("XGBoost con pesos adaptativos entrenado exitosamente")
            except Exception as e:
                logger.warning(f"Error configurando XGBoost con pesos: {str(e)}")
                
                # Alternativa sin pesos
                try:
                    logger.info("Entrenando XGBoost estándar (sin pesos)")
                    xgb_reg = xgb.XGBRegressor(
                        n_estimators=250,
                        max_depth=7,
                        learning_rate=0.05,
                        colsample_bytree=0.8,
                        reg_alpha=1.0,
                        reg_lambda=1.5,
                        random_state=42
                    )
                    xgb_reg.fit(X_train_clean, y_train)
                    self.models['xgb_reg'] = xgb_reg
                    logger.info("XGBoost estándar entrenado exitosamente")
                except Exception as e_xgb:
                    logger.warning(f"Error al entrenar XGBoost estándar: {str(e_xgb)}")
            
            # Modelo 4: HuberRegressor (robusto a outliers)
            try:
                logger.info("Entrenando HuberRegressor")
                # Para HuberRegressor, garantizar que no hay NaNs ni infinitos
                X_huber = X_train_clean.copy()
                X_huber = np.nan_to_num(X_huber, nan=0.0, posinf=0.0, neginf=0.0)
                
                huber = HuberRegressor(
                    epsilon=1.5,
                    alpha=0.001,
                    max_iter=200
                )
                huber.fit(X_huber, y_train)
                self.models['huber'] = huber
                logger.info("HuberRegressor entrenado exitosamente")
            except Exception as e_huber:
                logger.warning(f"Error al entrenar HuberRegressor: {str(e_huber)}")
            
            # Crear un ensemble de votación simple como fallback
            from sklearn.ensemble import VotingRegressor
            
            # Seleccionar modelos para el ensemble
            estimators = []
            if 'gb_robust' in self.models:
                estimators.append(('gb_robust', self.models['gb_robust']))
            if 'hist_gb' in self.models:
                estimators.append(('hist_gb', self.models['hist_gb']))
            if 'xgb_weighted' in self.models:
                estimators.append(('xgb_weighted', self.models['xgb_weighted']))
            elif 'xgb_reg' in self.models:
                estimators.append(('xgb_reg', self.models['xgb_reg']))
            
            # Solo crear el VotingRegressor si tenemos al menos 2 modelos
            if len(estimators) >= 2:
                logger.info("Creando VotingRegressor con los modelos disponibles")
                voting_reg = VotingRegressor(estimators=estimators)
                voting_reg.fit(X_train_clean, y_train)
                self.models['voting_ensemble'] = voting_reg
                
                # Establecer como modelo predeterminado si no existe otro
                self.models['best'] = voting_reg
                logger.info("VotingRegressor establecido como modelo predeterminado")
            else:
                # Si no podemos crear un ensemble, usar el mejor modelo individual
                logger.info("No hay suficientes modelos para VotingRegressor. Usando mejor modelo individual.")
                if 'hist_gb' in self.models:
                    self.models['best'] = self.models['hist_gb']
                    logger.info("HistGradientBoostingRegressor establecido como modelo predeterminado")
                elif 'xgb_weighted' in self.models:
                    self.models['best'] = self.models['xgb_weighted']
                    logger.info("XGBoostRegressor establecido como modelo predeterminado")
                elif 'gb_robust' in self.models:
                    self.models['best'] = self.models['gb_robust']
                    logger.info("GradientBoostingRegressor establecido como modelo predeterminado")
            
            logger.info(f"Configurados {len(self.models)} modelos especializados como fallback")
            
        except Exception as e:
            logger.error(f"Error en _setup_specialized_models: {str(e)}")
            # Asegurar que tenemos al menos un modelo funcional
            try:
                logger.warning("Intentando configurar al menos un modelo funcional como último recurso...")
                # Intentar con HistGradientBoostingRegressor que maneja NaNs nativamente
                from sklearn.ensemble import HistGradientBoostingRegressor
                
                hist_gb_fallback = HistGradientBoostingRegressor(
                    max_iter=100,  # Reducir para asegurar entrenamiento rápido
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Entrenar con datos originales (maneja NaNs)
                hist_gb_fallback.fit(X_train, y_train)
                self.models['hist_gb_fallback'] = hist_gb_fallback
                self.models['best'] = hist_gb_fallback
                logger.info("Configurado modelo de último recurso con HistGradientBoostingRegressor")
            except Exception as last_resort_error:
                logger.error(f"ERROR CRÍTICO: No se pudo configurar ningún modelo: {str(last_resort_error)}")
                # Si falla todo, crear un modelo extremadamente simple que siempre devuelva la media
                from sklearn.dummy import DummyRegressor
                
                logger.warning("Configurando DummyRegressor como último recurso")
                dummy = DummyRegressor(strategy='mean')
                dummy.fit(np.ones((len(y_train), 1)), y_train)  # Usar característica simulada
                self.models['dummy'] = dummy
                self.models['best'] = dummy

    def predict_with_weights(self, X_test, model_name='best'):
        """
        Realiza predicciones usando un sistema de pesos variables para compensar
        la heteroscedasticidad (patrón en forma de abanico) para valores bajos.
        
        Args:
            X_test: Características para predecir
            model_name: Nombre del modelo a usar
            
        Returns:
            array: Predicciones ajustadas
        """
        # Obtener las predicciones base
        base_preds = self.predict(X_test, model_name)
        
        # Crear un sistema de pesos variables basado en el valor predicho
        # Para compensar heteroscedasticidad en forma de abanico
        
        # Paso 1: Múltiples modelos para diferentes rangos de valores
        try:
            # Si tenemos el ensemble adaptativo regional, usarlo
            if 'adaptive_regional_ensemble' in self.models:
                # Ya está calibrado
                return base_preds
            
            # Si no, intentar con enfoque de predicción ponderada
            # Predecir con múltiples modelos para formar un consenso
            models_to_use = []
            weights = []
            
            # Decidir modelos a usar basado en los disponibles
            for potential_model in ['xgb_weighted', 'gb_robust', 'hist_gb', 'huber', 'voting_ensemble']:
                if potential_model in self.models:
                    models_to_use.append(potential_model)
                    weights.append(1.0)  # Inicialmente pesos iguales
            
            # Si tenemos al menos 2 modelos, crear un ensemble ponderado
            if len(models_to_use) >= 2:
                # Obtener predicciones de cada modelo
                all_preds = {}
                for model_name in models_to_use:
                    all_preds[model_name] = self.predict(X_test, model_name)
                
                # Matriz para ponderación variable
                # Calculamos pesos por muestra basados en el valor predicho
                # Valores pequeños tienen mayor varianza -> combinar múltiples modelos
                weighted_preds = np.zeros_like(base_preds)
                
                # Calcular pesos por muestra basados en el valor predicho base
                for i, pred_value in enumerate(base_preds):
                    sample_weights = np.ones(len(models_to_use))
                    
                    # Ponderar basado en el valor (compensar heteroscedasticidad)
                    if pred_value < 5.0:
                        # Para valores pequeños, dar más peso a modelos robustos
                        for j, model_name in enumerate(models_to_use):
                            if 'huber' in model_name or 'robust' in model_name:
                                sample_weights[j] *= 1.5
                            if 'hist_gb' in model_name:  # Bueno para valores pequeños
                                sample_weights[j] *= 1.3
                        
                        # Normalizar pesos
                        sample_weights = sample_weights / np.sum(sample_weights)
                        
                        # Combinar predicciones con pesos específicos para esta muestra
                        for j, model_name in enumerate(models_to_use):
                            weighted_preds[i] += sample_weights[j] * all_preds[model_name][i]
                    else:
                        # Para valores más grandes, usar la predicción base
                        weighted_preds[i] = base_preds[i]
                
                # Aplicar calibración final sobre las predicciones ponderadas
                calibrated_preds = self._calibrate_predictions(weighted_preds)
                
                # Aplicar una corrección de varianza adicional basada en el valor predicho
                # La varianza aumenta con valores pequeños en patrón de abanico
                final_preds = np.zeros_like(calibrated_preds)
                
                # Analizar varianza por rangos
                low_range = (calibrated_preds < 5.0)
                mid_range = (calibrated_preds >= 5.0) & (calibrated_preds < 15.0)
                high_range = (calibrated_preds >= 15.0)
                
                # Ajustar aleatoriamente para reflejar la incertidumbre real (opcional)
                # Solo si queremos modelar explícitamente la heteroscedasticidad
                if np.sum(low_range) > 0:
                    # Menor variabilidad - más confianza en predicciones pequeñas
                    noise_factor = 0.05  # 5% de ruido para valores pequeños
                    noise = np.random.normal(0, noise_factor * calibrated_preds[low_range])
                    final_preds[low_range] = calibrated_preds[low_range] * (1.0 + noise)
                    
                if np.sum(mid_range) > 0:
                    # Variabilidad media para valores medios
                    final_preds[mid_range] = calibrated_preds[mid_range]
                    
                if np.sum(high_range) > 0:
                    # Mayor estabilidad para valores altos
                    final_preds[high_range] = calibrated_preds[high_range]
                
                # Asegurar predicciones no negativas
                final_preds = np.maximum(final_preds, 0.0)
                
                logger.info("Aplicada corrección para heteroscedasticidad por valores bajos")
                return final_preds
                
            else:
                # Si no hay suficientes modelos, devolver predicciones calibradas
                return self._calibrate_predictions(base_preds)
                
        except Exception as e:
            logger.error(f"Error en predict_with_weights: {str(e)}")
            # Fallback a predicciones base
            return base_preds

    def analyze_heteroscedasticity_by_range(self, X_test, y_test, model_name='best'):
        """
        Analiza específicamente la heteroscedasticidad por rangos de valores,
        poniendo especial atención en el patrón de abanico para valores bajos.
        
        Args:
            X_test: Características de prueba
            y_test: Valores reales
            model_name: Nombre del modelo a evaluar
            
        Returns:
            dict: Análisis detallado de heteroscedasticidad por rangos
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import os
        
        # Obtener las predicciones estándar y las predicciones corregidas
        standard_preds = super().predict(X_test, model_name)
        corrected_preds = self.predict_adaptive(X_test)
        
        # Usar valores originales si hay transformación logarítmica
        if hasattr(self, 'y_transform') and self.y_transform == 'log':
            y_true = self.y_test_original if hasattr(self, 'y_test_original') else y_test
        else:
            y_true = y_test
        
        # Calcular residuos para ambos tipos de predicciones
        std_residuals = y_true - standard_preds
        corrected_residuals = y_true - corrected_preds
        
        # Definir rangos para análisis detallado
        ranges = [
            (0, 2, "Muy bajo"),
            (2, 5, "Bajo"),
            (5, 10, "Medio-bajo"),
            (10, 20, "Medio"),
            (20, 30, "Alto"),
            (30, 100, "Muy alto")
        ]
        
        # Crear DataFrame para análisis
        analysis_data = pd.DataFrame({
            'y_true': y_true,
            'std_pred': standard_preds,
            'corr_pred': corrected_preds,
            'std_residual': std_residuals,
            'corr_residual': corrected_residuals
        })
        
        # Añadir columna de rango
        def get_range(value):
            for low, high, label in ranges:
                if low <= value < high:
                    return label
            return "Extremo"
        
        analysis_data['pred_range'] = analysis_data['std_pred'].apply(get_range)
        
        # Calcular estadísticas por rango
        range_stats = []
        
        for low, high, label in ranges:
            # Filtrar datos para este rango
            range_mask = (standard_preds >= low) & (standard_preds < high)
            if np.sum(range_mask) == 0:
                continue
            
            # Obtener datos para este rango
            range_data = analysis_data[range_mask]
            
            # Calcular estadísticas
            stats = {
                'range': label,
                'range_values': f"{low}-{high}",
                'count': len(range_data),
                'pct_total': len(range_data) / len(analysis_data) * 100,
                
                # Estadísticas de residuos estándar
                'std_residual_mean': range_data['std_residual'].mean(),
                'std_residual_median': range_data['std_residual'].median(),
                'std_residual_std': range_data['std_residual'].std(),
                'std_residual_skew': range_data['std_residual'].skew(),
                
                # Estadísticas de residuos corregidos
                'corr_residual_mean': range_data['corr_residual'].mean(),
                'corr_residual_median': range_data['corr_residual'].median(),
                'corr_residual_std': range_data['corr_residual'].std(),
                'corr_residual_skew': range_data['corr_residual'].skew(),
                
                # Mejora en términos de sesgo y variabilidad
                'bias_improvement': abs(range_data['std_residual'].mean()) - abs(range_data['corr_residual'].mean()),
                'var_improvement': range_data['std_residual'].std() - range_data['corr_residual'].std()
            }
            
            # Añadir una métrica de heteroscedasticidad específica
            # Mide cómo la varianza del residuo depende del valor predicho
            from scipy.stats import linregress
            
            # Para standard
            std_slope, _, std_r, std_p, _ = linregress(
                range_data['std_pred'], 
                range_data['std_residual']**2
            )
            
            # Para corregidos
            corr_slope, _, corr_r, corr_p, _ = linregress(
                range_data['corr_pred'], 
                range_data['corr_residual']**2
            )
            
            stats.update({
                'std_hetero_slope': std_slope,
                'std_hetero_r': std_r,
                'std_hetero_p': std_p,
                'corr_hetero_slope': corr_slope,
                'corr_hetero_r': corr_r,
                'corr_hetero_p': corr_p,
                'hetero_improvement': abs(std_slope) - abs(corr_slope)
            })
            
            range_stats.append(stats)
        
        # Crear visualizaciones
        # 1. Gráfico de residuos vs predicciones por rangos
        plt.figure(figsize=(12, 10))
        
        # En la parte superior: residuos estándar
        plt.subplot(2, 1, 1)
        plt.scatter(standard_preds, std_residuals, alpha=0.5, c='blue')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.title('Residuos Estándar vs Predicciones (Patrón de Abanico en Valores Bajos)')
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuos')
        plt.grid(True, alpha=0.3)
        
        # En la parte inferior: residuos corregidos
        plt.subplot(2, 1, 2)
        plt.scatter(corrected_preds, corrected_residuals, alpha=0.5, c='green')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.title('Residuos Corregidos vs Predicciones (Heteroscedasticidad Reducida)')
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuos')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfica
        hetero_plot_path = 'reports/figures/heteroscedasticity_by_range.png'
        os.makedirs(os.path.dirname(hetero_plot_path), exist_ok=True)
        plt.savefig(hetero_plot_path)
        plt.close()
        
        # 2. Gráfico de boxplots de residuos por rango
        plt.figure(figsize=(14, 8))
        
        # Convertir análisis a formato largo para facilitar visualización
        long_format = []
        
        for _, row in analysis_data.iterrows():
            long_format.append({
                'pred_range': row['pred_range'],
                'residual': row['std_residual'],
                'type': 'Estándar'
            })
            long_format.append({
                'pred_range': row['pred_range'],
                'residual': row['corr_residual'],
                'type': 'Corregido'
            })
        
        long_df = pd.DataFrame(long_format)
        
        # Ordenar rangos
        range_order = [r[2] for r in ranges]
        
        # Crear boxplot
        import seaborn as sns
        ax = sns.boxplot(x='pred_range', y='residual', hue='type', data=long_df, 
                        order=[r for r in range_order if r in long_df['pred_range'].unique()])
        
        plt.title('Distribución de Residuos por Rango de Predicción')
        plt.xlabel('Rango de Predicción')
        plt.ylabel('Residuos')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Tipo de Residuo')
        
        plt.tight_layout()
        
        # Guardar gráfica
        boxplot_path = 'reports/figures/residuals_boxplot_by_range.png'
        plt.savefig(boxplot_path)
        plt.close()
        
        # 3. Gráfico de varianza de residuos por rango (para mostrar reducción de heteroscedasticidad)
        plt.figure(figsize=(12, 6))
        
        # Preparar datos para gráfico de barras
        ranges_labels = [stat['range'] for stat in range_stats]
        std_vars = [stat['std_residual_std'] for stat in range_stats]
        corr_vars = [stat['corr_residual_std'] for stat in range_stats]
        
        x = np.arange(len(ranges_labels))
        width = 0.35
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, std_vars, width, label='Estándar')
        rects2 = ax.bar(x + width/2, corr_vars, width, label='Corregido')
        
        # Añadir etiquetas y títulos
        ax.set_ylabel('Desviación Estándar de Residuos')
        ax.set_xlabel('Rango de Predicción')
        ax.set_title('Reducción de Heteroscedasticidad por Rango')
        ax.set_xticks(x)
        ax.set_xticklabels(ranges_labels)
        ax.legend()
        
        # Función para añadir etiquetas en las barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        # Guardar gráfica
        variance_path = 'reports/figures/residual_variance_by_range.png'
        plt.savefig(variance_path)
        plt.close()
        
        # Compilar resultados
        result = {
            'range_stats': range_stats,
            'overall_improvement': {
                'bias_improvement': sum(stat['bias_improvement'] for stat in range_stats) / len(range_stats),
                'var_improvement': sum(stat['var_improvement'] for stat in range_stats) / len(range_stats),
                'hetero_improvement': sum(stat['hetero_improvement'] for stat in range_stats) / len(range_stats)
            },
            'plots': {
                'heteroscedasticity_plot': hetero_plot_path,
                'boxplot_path': boxplot_path,
                'variance_path': variance_path
            }
        }
        
        logger.info(f"Análisis de heteroscedasticidad por rangos completado. Mejora global: {result['overall_improvement']}")
        
        return result

    def train_models_two_stage(self, X_train, y_train, use_scaling=True):
        """
        Implementa un modelo de dos etapas para predicción de puntos:
        1. Primera etapa: Clasificador para determinar el rango de puntos
        2. Segunda etapa: Regresores especializados por cada rango
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo
            use_scaling: Si se debe aplicar escalado
            
        Returns:
            self: Instancia actualizada con modelos entrenados
        """
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import BayesianRidge, HuberRegressor
        import xgboost as xgb
        
        logger.info("Entrenando modelo de dos etapas con especialización por rango")
        
        # Asegurar que no hay valores nulos
        X_train_clean = X_train.copy()
        y_train_clean = y_train.copy()
        
        if hasattr(X_train_clean, 'isna') and X_train_clean.isna().any().any():
            logger.warning("Detectados valores NaN en X_train. Imputando...")
            X_train_clean = X_train_clean.fillna(X_train_clean.median())
            X_train_clean = X_train_clean.fillna(0)  # Por si quedan NaNs
        
        if hasattr(y_train_clean, 'isna') and y_train_clean.isna().any():
            logger.warning("Detectados valores NaN en y_train. Imputando...")
            y_train_clean = y_train_clean.fillna(y_train_clean.median())
        
        # Aplicar transformaciones logarítmicas y otras no lineales
        X_train_transformed = self._apply_nonlinear_transformations(X_train_clean)
        
        # Definir rangos para especialización
        # Estos umbrales se pueden ajustar según la distribución de puntos
        y_values = y_train_clean.values if hasattr(y_train_clean, 'values') else y_train_clean
        
        # Usar cuartiles para definir rangos
        q1 = np.percentile(y_values, 25)
        q2 = np.percentile(y_values, 50)
        q3 = np.percentile(y_values, 75)
        
        # Crear etiquetas para clasificación
        y_ranges = np.zeros_like(y_values, dtype=int)
        y_ranges[(y_values >= q1) & (y_values < q2)] = 1  # Rango bajo-medio
        y_ranges[(y_values >= q2) & (y_values < q3)] = 2  # Rango medio-alto
        y_ranges[y_values >= q3] = 3  # Rango alto
        
        # Guardar umbrales para predicción
        self.range_thresholds = {
            'q1': q1,
            'q2': q2,
            'q3': q3
        }
        
        # Aplicar escalado si es necesario
        if use_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_transformed)
            self.scaler = scaler
        else:
            X_train_scaled = X_train_transformed
        
        # 1. Primera etapa: Clasificador para determinar el rango
        logger.info("Entrenando clasificador de rangos...")
        range_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        range_classifier.fit(X_train_scaled, y_ranges)
        self.models['range_classifier'] = range_classifier
        
        # 2. Segunda etapa: Regresores especializados por rango
        # Entrenar un modelo para cada rango
        for range_idx, range_name in enumerate(['very_low', 'low_mid', 'mid_high', 'high']):
            # Filtrar datos para este rango
            mask = (y_ranges == range_idx)
            if mask.sum() < 50:  # Si hay muy pocos ejemplos, usar todos los datos
                logger.warning(f"Muy pocos ejemplos para rango {range_name} ({mask.sum()}). Usando todos los datos.")
                X_range = X_train_scaled
                y_range = y_train_clean
            else:
                X_range = X_train_scaled[mask]
                y_range = y_train_clean[mask]
            
            logger.info(f"Entrenando modelo especializado para rango {range_name} con {len(X_range)} ejemplos")
            
            # Seleccionar modelo apropiado según el rango
            if range_idx == 0:  # Rango muy bajo - usar Bayesian Ridge para manejar incertidumbre
                model = BayesianRidge(
                    n_iter=300,
                    alpha_1=1e-6,
                    alpha_2=1e-6,
                    lambda_1=1e-6,
                    lambda_2=1e-6
                )
            elif range_idx == 1:  # Rango bajo-medio - usar HistGradientBoosting para robustez
                model = HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=8,
                    learning_rate=0.1,
                    l2_regularization=0.1,
                    random_state=42
                )
            elif range_idx == 2:  # Rango medio-alto - usar XGBoost para precisión
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=1,
                    random_state=42
                )
            else:  # Rango alto - usar HuberRegressor para robustez a outliers
                model = HuberRegressor(
                    epsilon=1.5,
                    alpha=0.0001,
                    max_iter=1000
                )
            
            # Entrenar modelo
            model.fit(X_range, y_range)
            self.models[f'range_{range_name}'] = model
        
        # Entrenar un modelo general como fallback
        logger.info("Entrenando modelo general como fallback")
        fallback_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        fallback_model.fit(X_train_scaled, y_train_clean)
        self.models['fallback'] = fallback_model
        
        # Guardar información sobre los rangos
        self.range_info = {
            'very_low': {'threshold': 0, 'count': (y_ranges == 0).sum()},
            'low_mid': {'threshold': q1, 'count': (y_ranges == 1).sum()},
            'mid_high': {'threshold': q2, 'count': (y_ranges == 2).sum()},
            'high': {'threshold': q3, 'count': (y_ranges == 3).sum()}
        }
        
        logger.info("Modelo de dos etapas entrenado exitosamente")
        return self

    def _apply_nonlinear_transformations(self, X):
        """
        Aplica transformaciones no lineales a las características.
        
        Args:
            X: DataFrame con características
            
        Returns:
            DataFrame: Características con transformaciones aplicadas
        """
        import numpy as np
        import pandas as pd
        
        X_transformed = X.copy()
        
        # Transformaciones logarítmicas para FGA y otras variables no lineales
        if 'FGA' in X.columns:
            # Log(x+1) para manejar ceros
            X_transformed['FGA_log'] = np.log1p(X['FGA'])
            
            # Raíz cuadrada
            X_transformed['FGA_sqrt'] = np.sqrt(X['FGA'])
            
            # Transformación Box-Cox simplificada (potencia 0.5)
            X_transformed['FGA_boxcox'] = np.power(X['FGA'] + 1, 0.5)
        
        # Transformaciones para otras variables importantes
        for col in ['MP', '3PA', 'FTA']:
            if col in X.columns:
                X_transformed[f'{col}_log'] = np.log1p(X[col])
                X_transformed[f'{col}_sqrt'] = np.sqrt(X[col])
        
        # Variables de interacción importantes
        if 'FGA' in X.columns and 'MP' in X.columns:
            X_transformed['FGA_per_minute'] = X['FGA'] / np.maximum(X['MP'], 1)
        
        if 'PTS_last5' in X.columns:
            X_transformed['PTS_trend'] = X['PTS_last5'] / np.maximum(X['PTS_last10'], 1)
        
        # Características contextuales si están disponibles
        if 'days_rest' in X.columns:
            X_transformed['fatigue_factor'] = 1 / np.maximum(X['days_rest'] + 1, 1)
        
        if 'away' in X.columns:
            # Convertir booleano a numérico si es necesario
            if X['away'].dtype == bool:
                X_transformed['away'] = X['away'].astype(int)
        
        return X_transformed

    def predict_two_stage(self, X_test):
        """
        Realiza predicciones usando el modelo de dos etapas.
        
        Args:
            X_test: Características para predicción
            
        Returns:
            array: Predicciones de puntos
        """
        import numpy as np
        
        # Verificar si tenemos los modelos necesarios
        if 'range_classifier' not in self.models:
            logger.warning("Modelo de dos etapas no entrenado. Usando predict_adaptive como fallback.")
            return self.predict_adaptive(X_test)
        
        # Preparar datos
        X_test_clean = X_test.copy()
        
        # Manejar valores NaN
        if hasattr(X_test_clean, 'isna') and X_test_clean.isna().any().any():
            logger.info("Detectados NaNs en datos de entrada para predict_two_stage. Imputando.")
            X_test_clean = X_test_clean.fillna(X_test_clean.median())
            X_test_clean = X_test_clean.fillna(0)
        
        # Aplicar transformaciones no lineales
        X_test_transformed = self._apply_nonlinear_transformations(X_test_clean)
        
        # Aplicar escalado si es necesario
        if hasattr(self, 'scaler'):
            X_test_scaled = self.scaler.transform(X_test_transformed)
        else:
            X_test_scaled = X_test_transformed
        
        # 1. Predecir el rango usando el clasificador
        range_predictions = self.models['range_classifier'].predict(X_test_scaled)
        
        # También obtener probabilidades para ponderación suave
        range_probs = self.models['range_classifier'].predict_proba(X_test_scaled)
        
        # 2. Predecir puntos usando el modelo especializado para cada muestra
        predictions = np.zeros(len(X_test))
        
        # Mapeo de índices de rango a nombres de modelos
        range_models = {
            0: 'range_very_low',
            1: 'range_low_mid',
            2: 'range_mid_high',
            3: 'range_high'
        }
        
        # Enfoque de ensemble ponderado por cuartiles
        for i, (range_idx, probs) in enumerate(zip(range_predictions, range_probs)):
            # Obtener predicción del modelo principal para este rango
            main_model_name = range_models[range_idx]
            
            if main_model_name in self.models:
                main_pred = self.models[main_model_name].predict([X_test_scaled[i]])[0]
            else:
                # Si no existe el modelo para este rango, usar fallback
                main_pred = self.models['fallback'].predict([X_test_scaled[i]])[0]
            
            # Inicializar predicción ponderada
            weighted_pred = main_pred
            
            # Si la confianza del clasificador no es muy alta, mezclar con otros modelos
            max_prob = probs.max()
            if max_prob < 0.7:  # Umbral de confianza
                # Calcular predicción ponderada usando todos los modelos disponibles
                total_weight = max_prob
                weighted_pred = main_pred * max_prob
                
                # Añadir contribuciones de otros modelos según sus probabilidades
                for other_range, prob in enumerate(probs):
                    if other_range != range_idx and prob > 0.1:  # Umbral mínimo de probabilidad
                        other_model_name = range_models[other_range]
                        if other_model_name in self.models:
                            other_pred = self.models[other_model_name].predict([X_test_scaled[i]])[0]
                            weighted_pred += other_pred * prob
                            total_weight += prob
                
                # Normalizar
                if total_weight > 0:
                    weighted_pred /= total_weight
            
            predictions[i] = weighted_pred
        
        # Aplicar calibración bayesiana para manejar incertidumbre en rangos extremos
        predictions = self._apply_bayesian_calibration(predictions, range_predictions, range_probs)
        
        return predictions

    def _apply_bayesian_calibration(self, predictions, range_predictions, range_probs):
        """
        Aplica calibración bayesiana para manejar incertidumbre en rangos extremos.
        
        Args:
            predictions: Predicciones iniciales
            range_predictions: Rangos predichos
            range_probs: Probabilidades de cada rango
            
        Returns:
            array: Predicciones calibradas
        """
        import numpy as np
        
        # Parámetros de calibración por rango
        # Estos valores se pueden ajustar basados en análisis de error
        calibration_params = {
            0: {'prior_mean': 5.0, 'prior_strength': 0.3},  # Muy bajo
            1: {'prior_mean': 10.0, 'prior_strength': 0.2},  # Bajo-medio
            2: {'prior_mean': 18.0, 'prior_strength': 0.1},  # Medio-alto
            3: {'prior_mean': 30.0, 'prior_strength': 0.2}   # Alto
        }
        
        # Aplicar calibración bayesiana
        calibrated_predictions = predictions.copy()
        
        for i, (pred, range_idx) in enumerate(zip(predictions, range_predictions)):
            # Obtener parámetros para este rango
            params = calibration_params[range_idx]
            prior_mean = params['prior_mean']
            prior_strength = params['prior_strength']
            
            # Ajustar la fuerza del prior basado en la confianza del clasificador
            confidence = range_probs[i].max()
            adjusted_strength = prior_strength * (1 - confidence)
            
            # Aplicar calibración bayesiana
            # Formula: (prior_strength * prior_mean + prediction) / (prior_strength + 1)
            calibrated_predictions[i] = (adjusted_strength * prior_mean + pred) / (adjusted_strength + 1)
        
        # Asegurar que no hay valores negativos
        calibrated_predictions = np.maximum(calibrated_predictions, 0)
        
        return calibrated_predictions

    def incorporate_contextual_variables(self, X, player_data=None, schedule_data=None):
        """
        Incorpora variables contextuales como fatiga acumulada y calendarios.
        
        Args:
            X: DataFrame con características base
            player_data: DataFrame con datos históricos de jugadores (opcional)
            schedule_data: DataFrame con datos de calendario (opcional)
            
        Returns:
            DataFrame: Características aumentadas con variables contextuales
        """
        import pandas as pd
        import numpy as np
        
        X_augmented = X.copy()
        
        # Si no hay datos adicionales, devolver las características originales
        if player_data is None and schedule_data is None:
            return X_augmented
        
        # 1. Incorporar variables de fatiga si hay datos de jugadores
        if player_data is not None and 'Player' in X.columns:
            logger.info("Incorporando variables de fatiga basadas en historial de jugadores")
            
            # Asegurar que tenemos las columnas necesarias
            required_cols = ['Player', 'Date', 'MP']
            if all(col in player_data.columns for col in required_cols):
                # Convertir fecha a datetime si es necesario
                if not pd.api.types.is_datetime64_any_dtype(player_data['Date']):
                    player_data['Date'] = pd.to_datetime(player_data['Date'], errors='coerce')
                
                # Ordenar por jugador y fecha
                player_data = player_data.sort_values(['Player', 'Date'])
                
                # Calcular minutos acumulados en los últimos 7 días para cada jugador
                player_fatigue = {}
                
                for player in player_data['Player'].unique():
                    player_games = player_data[player_data['Player'] == player]
                    
                    # Calcular minutos acumulados en ventanas móviles
                    player_games['MP_last3'] = player_games['MP'].rolling(window=3, min_periods=1).sum()
                    player_games['MP_last5'] = player_games['MP'].rolling(window=5, min_periods=1).sum()
                    player_games['MP_last7'] = player_games['MP'].rolling(window=7, min_periods=1).sum()
                    
                    # Calcular días de descanso
                    player_games['days_rest'] = player_games['Date'].diff().dt.days.fillna(3)
                    
                    # Guardar últimos valores para cada jugador
                    last_row = player_games.iloc[-1]
                    player_fatigue[player] = {
                        'MP_last3': last_row['MP_last3'],
                        'MP_last5': last_row['MP_last5'],
                        'MP_last7': last_row['MP_last7'],
                        'days_rest': last_row['days_rest']
                    }
                
                # Añadir variables de fatiga a X
                for idx, row in X.iterrows():
                    player = row['Player']
                    if player in player_fatigue:
                        for var, value in player_fatigue[player].items():
                            X_augmented.loc[idx, var] = value
                    else:
                        # Valores por defecto si no hay datos
                        X_augmented.loc[idx, 'MP_last3'] = 0
                        X_augmented.loc[idx, 'MP_last5'] = 0
                        X_augmented.loc[idx, 'MP_last7'] = 0
                        X_augmented.loc[idx, 'days_rest'] = 3
                
                # Crear índices de fatiga
                X_augmented['fatigue_index'] = X_augmented['MP_last7'] / (X_augmented['days_rest'] + 1)
                X_augmented['fatigue_index'] = X_augmented['fatigue_index'].fillna(0)
        
        # 2. Incorporar variables de calendario si hay datos de calendario
        if schedule_data is not None and 'Team' in X.columns and 'Opp' in X.columns:
            logger.info("Incorporando variables contextuales de calendario")
            
            # Asegurar que tenemos las columnas necesarias
            required_cols = ['Team', 'Opp', 'Date', 'away']
            if all(col in schedule_data.columns for col in required_cols):
                # Convertir fecha a datetime si es necesario
                if not pd.api.types.is_datetime64_any_dtype(schedule_data['Date']):
                    schedule_data['Date'] = pd.to_datetime(schedule_data['Date'], errors='coerce')
                
                # Calcular variables de calendario para cada equipo
                team_schedule = {}
                
                for team in schedule_data['Team'].unique():
                    team_games = schedule_data[schedule_data['Team'] == team]
                    team_games = team_games.sort_values('Date')
                    
                    # Calcular juegos consecutivos
                    team_games['back_to_back'] = (team_games['Date'].diff().dt.days == 1).astype(int)
                    
                    # Calcular juegos en los últimos 5 días
                    team_games['games_last5d'] = 0
                    for i in range(len(team_games)):
                        if i == 0:
                            team_games.iloc[i, team_games.columns.get_loc('games_last5d')] = 0
                        else:
                            current_date = team_games.iloc[i]['Date']
                            prev_5d = team_games.iloc[:i][team_games.iloc[:i]['Date'] >= (current_date - pd.Timedelta(days=5))]
                            team_games.iloc[i, team_games.columns.get_loc('games_last5d')] = len(prev_5d)
                    
                    # Guardar últimos valores para cada equipo
                    last_row = team_games.iloc[-1]
                    team_schedule[team] = {
                        'back_to_back': last_row['back_to_back'],
                        'games_last5d': last_row['games_last5d'],
                        'away_streak': team_games['away'].rolling(window=3, min_periods=1).sum().iloc[-1]
                    }
                
                # Añadir variables de calendario a X
                for idx, row in X.iterrows():
                    team = row['Team']
                    opp = row['Opp']
                    
                    # Variables para el equipo
                    if team in team_schedule:
                        for var, value in team_schedule[team].items():
                            X_augmented.loc[idx, f'team_{var}'] = value
                    else:
                        X_augmented.loc[idx, 'team_back_to_back'] = 0
                        X_augmented.loc[idx, 'team_games_last5d'] = 0
                        X_augmented.loc[idx, 'team_away_streak'] = 0
                    
                    # Variables para el oponente
                    if opp in team_schedule:
                        for var, value in team_schedule[opp].items():
                            X_augmented.loc[idx, f'opp_{var}'] = value
                    else:
                        X_augmented.loc[idx, 'opp_back_to_back'] = 0
                        X_augmented.loc[idx, 'opp_games_last5d'] = 0
                        X_augmented.loc[idx, 'opp_away_streak'] = 0
                
                # Crear índice de ventaja/desventaja de calendario
                X_augmented['schedule_advantage'] = (
                    X_augmented['opp_games_last5d'] - X_augmented['team_games_last5d'] +
                    X_augmented['opp_back_to_back'] - X_augmented['team_back_to_back']
                )
        
        # Rellenar valores NaN en las nuevas columnas
        for col in X_augmented.columns:
            if col not in X.columns:
                X_augmented[col] = X_augmented[col].fillna(0)
        
        return X_augmented