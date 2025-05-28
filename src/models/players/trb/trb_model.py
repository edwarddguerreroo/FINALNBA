"""
Modelo de Predicción de Rebotes NBA - ARQUITECTURA CON ENSEMBLE DE TERCER NIVEL
============================================================================================

Sistema de predicción de rebotes de última generación con arquitectura híbrida optimizada:
- Redes Neuronales Profundas con Attention Mechanisms y arquitecturas especializadas
- Ensemble de modelos especializados con optimización bayesiana agresiva
- Ensemble de TERCER NIVEL con meta-modelos
- Feature Engineering automático con selección inteligente multi-nivel
- Optimización de hiperparámetros agresiva con múltiples algoritmos
- Regularización avanzada y técnicas anti-overfitting de última generación
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Ensemble Models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    VotingRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, 
    BayesianRidge, HuberRegressor,
    SGDRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Optimization
from sklearn.model_selection import (
    cross_val_score, 
    StratifiedKFold, 
    GridSearchCV,
    train_test_split,
    RandomizedSearchCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.base import BaseEstimator, RegressorMixin, clone

# Bayesian Optimization
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_lcb
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

# Hyperopt for additional optimization
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Importar ReboundsFeatureEngineer
from .features_trb import ReboundsFeatureEngineer

logger = logging.getLogger(__name__)


# Attention Layer 
class AttentionLayer(nn.Module):
    """Capa de atención ultra-avanzada con múltiples cabezas y mecanismos de gating."""
    
    def __init__(self, input_dim: int, attention_dim: int = 128, num_heads: int = 8):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        # Multi-head attention
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, input_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-head attention
        Q = self.W_q(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, -1)
        attended = self.W_o(attended)
        
        # Gating mechanism
        gate_weights = self.gate(x)
        gated_output = gate_weights * attended + (1 - gate_weights) * x
        
        # Residual connection and layer norm
        output = self.layer_norm(gated_output + x)
        output = self.dropout(output)
        
        return output

class DeepReboundsNet(nn.Module):
    """Red neuronal profunda especializada para predicción de rebotes con arquitecturas múltiples."""
    
    def __init__(self, input_dim: int, architecture: str = 'ultra_deep', 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super(DeepReboundsNet, self).__init__()
        
        self.architecture = architecture
        self.use_attention = use_attention
        
        if architecture == 'ultra_deep':
            hidden_dims = [1024, 512, 256, 128, 64, 32]
        elif architecture == 'wide':
            hidden_dims = [2048, 1024, 512, 256]
        elif architecture == 'residual':
            hidden_dims = [512, 512, 512, 256, 128]
        else:  # default
            hidden_dims = [512, 256, 128, 64]
        
        # Capa de atención ultra-avanzada
        if use_attention:
            self.attention = AttentionLayer(input_dim, attention_dim=128, num_heads=8)
        
        # Capas densas con normalización avanzada
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            
            # Conexiones residuales para arquitectura residual
            if architecture == 'residual' and i > 0 and prev_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Múltiples cabezas de salida para ensemble interno
        self.main_network = nn.Sequential(*layers)
        
        # Múltiples cabezas de predicción
        self.head1 = nn.Linear(prev_dim, 1)
        self.head2 = nn.Linear(prev_dim, 1)
        self.head3 = nn.Linear(prev_dim, 1)
        
        # Meta-cabeza que combina las predicciones
        self.meta_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        # Inicialización de pesos avanzada
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización de pesos ultra-optimizada."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if 'head' in str(m):
                    # Inicialización especial para cabezas de salida
                    nn.init.xavier_normal_(m.weight, gain=0.1)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.use_attention:
            x = self.attention(x)
        
        features = self.main_network(x)
        
        # Múltiples predicciones
        pred1 = self.head1(features)
        pred2 = self.head2(features)
        pred3 = self.head3(features)
        
        # Combinar predicciones
        combined = torch.cat([pred1, pred2, pred3], dim=1)
        final_output = self.meta_head(combined)
        
        return final_output.squeeze()

class ResidualBlock(nn.Module):
    """Bloque residual para conexiones skip."""
    
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        return F.relu(x + self.block(x))

class UltraAdvancedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble ultra-avanzado con TERCER NIVEL y optimización de hiperparámetros agresiva.
    """
    
    def __init__(self, use_deep_learning: bool = True, aggressive_optimization: bool = True,
                 third_level_ensemble: bool = True, cv_folds: int = 5, random_state: int = 42):
        self.use_deep_learning = use_deep_learning
        self.aggressive_optimization = aggressive_optimization
        self.third_level_ensemble = third_level_ensemble
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Modelos de primer nivel
        self.level1_models = {}
        self.level1_deep_models = []
        
        # Modelos de segundo nivel (meta-modelos)
        self.level2_models = {}
        
        # Modelos de tercer nivel (meta-meta-modelos)
        self.level3_models = {}
        
        self.scalers = {}
        self.feature_selectors = {}
        self.is_fitted = False
        
        # Configuración de dispositivo para PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
    
    def _create_level1_models(self) -> Dict[str, Any]:
        """Crea modelos de primer nivel optimizados."""
        
        models = {}
        
        if self.aggressive_optimization:
            # XGBoost optimizado
            models['xgb_ultra'] = xgb.XGBRegressor(
                n_estimators=800,  # Reducido para evitar overfitting
                max_depth=8,       # Reducido para mayor generalización
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                gamma=0.05,
                min_child_weight=5,
                random_state=self.random_state,
                n_jobs=-1,
                objective='reg:squarederror',
                tree_method='hist'
            )
            
            # LightGBM optimizado con parámetros más conservadores
            models['lgb_ultra'] = lgb.LGBMRegressor(
                n_estimators=600,
                max_depth=6,       # Reducido para evitar overfitting
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=10,  # Aumentado para mayor estabilidad
                min_split_gain=0.01,   # Añadido para evitar splits innecesarios
                num_leaves=31,         # Limitado para evitar overfitting
                random_state=self.random_state,
                n_jobs=-1,
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                force_col_wise=True,   # Para evitar warnings
                verbosity=-1           # Reducir warnings
            )
            
            # CatBoost optimizado
            models['cat_ultra'] = cb.CatBoostRegressor(
                iterations=600,
                depth=6,           # Reducido para mayor generalización
                learning_rate=0.05,
                subsample=0.8,
                reg_lambda=1.0,
                min_data_in_leaf=10,
                random_state=self.random_state,
                thread_count=-1,
                loss_function='MAE',
                verbose=False,
                bootstrap_type='Bernoulli'
            )
            
            # Random Forest optimizado
            models['rf_ultra'] = RandomForestRegressor(
                n_estimators=300,  # Reducido para velocidad
                max_depth=15,      # Reducido para evitar overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True
            )
            
            # Extra Trees optimizado
            models['et_ultra'] = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=False,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Gradient Boosting optimizado
            models['gb_ultra'] = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=self.random_state
            )
            
            # Modelos adicionales para diversidad
            models['ada_boost'] = AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state
            )
            
            models['bagging'] = BaggingRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Modelos lineales regularizados
            models['ridge_ultra'] = Ridge(alpha=5.0)
            models['lasso_ultra'] = Lasso(alpha=0.5)
            models['elastic_ultra'] = ElasticNet(alpha=0.5, l1_ratio=0.5)
            models['bayesian_ridge'] = BayesianRidge()
            models['huber'] = HuberRegressor(epsilon=1.35)
            
            # SVR optimizado con parámetros más conservadores
            models['svr_rbf'] = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
            
            # KNN optimizado
            models['knn'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
            
        else:
            # Modelos básicos pero efectivos
            models['xgboost'] = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=self.random_state, n_jobs=-1)
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200, 
                max_depth=6, 
                random_state=self.random_state, 
                n_jobs=-1,
                force_col_wise=True,
                verbosity=-1
            )
            models['catboost'] = cb.CatBoostRegressor(iterations=200, depth=6, random_state=self.random_state, verbose=False)
            models['random_forest'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)
            models['extra_trees'] = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)
        
        return models
    
    def _create_level1_deep_models(self, input_dim: int) -> List[DeepReboundsNet]:
        """Crea múltiples modelos de deep learning con arquitecturas diferentes."""
        models = []
        
        if self.use_deep_learning:
            # Arquitectura profunda
            models.append(DeepReboundsNet(
                input_dim=input_dim,
                architecture='ultra_deep',
                dropout_rate=0.3,
                use_attention=True
            ).to(self.device))
            
            # Arquitectura ancha
            models.append(DeepReboundsNet(
                input_dim=input_dim,
                architecture='wide',
                dropout_rate=0.4,
                use_attention=True
            ).to(self.device))
            
            # Arquitectura residual
            models.append(DeepReboundsNet(
                input_dim=input_dim,
                architecture='residual',
                dropout_rate=0.2,
                use_attention=True
            ).to(self.device))
            
            # Arquitectura sin atención (para diversidad)
            models.append(DeepReboundsNet(
                input_dim=input_dim,
                architecture='ultra_deep',
                dropout_rate=0.3,
                use_attention=False
            ).to(self.device))
        
        return models
    
    def _aggressive_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray):
        """Optimización de hiperparámetros con múltiples algoritmos."""
        if not BAYESIAN_OPT_AVAILABLE:
            logger.warning("Skopt no disponible, usando optimización básica")
            return
        
        logger.info("Iniciando optimización de hiperparámetros...")
        
        # Optimización para XGBoost con múltiples algoritmos
        self._optimize_xgboost_ultra_aggressive(X, y)
        
        # Optimización para LightGBM
        self._optimize_lightgbm_ultra_aggressive(X, y)
        
        # Optimización para CatBoost
        self._optimize_catboost_ultra_aggressive(X, y)
        
        logger.info("✅ Optimización ultra-agresiva completada")
    
    def _optimize_xgboost_ultra_aggressive(self, X: np.ndarray, y: np.ndarray):
        """Optimización agresiva para XGBoost."""
        
        # Espacio de búsqueda más conservador
        space = [
            Integer(200, 800, name='n_estimators'),      # Rango más conservador
            Integer(3, 10, name='max_depth'),            # Profundidad limitada
            Real(0.01, 0.15, name='learning_rate'),      # Learning rate más conservador
            Real(0.7, 1.0, name='subsample'),            # Subsample conservador
            Real(0.7, 1.0, name='colsample_bytree'),     # Colsample conservador
            Real(0.7, 1.0, name='colsample_bylevel'),    # Colsample por nivel conservador
            Real(0.0, 1.0, name='reg_alpha'),            # Regularización L1
            Real(0.0, 2.0, name='reg_lambda'),           # Regularización L2
            Real(0.0, 0.5, name='gamma'),                # Gamma conservador
            Integer(3, 10, name='min_child_weight')      # Min child weight conservador
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                objective='reg:squarederror',
                tree_method='hist',
                **params
            )
            
            # Validación cruzada con múltiples métricas
            mae_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
            return -mae_scores.mean()
        
        # Múltiples algoritmos de optimización
        results = []
        
        # Gaussian Process
        try:
            result_gp = gp_minimize(objective, space, n_calls=50, random_state=self.random_state,  # Reducido
                                   acquisition_func='EI')
            results.append(('GP', result_gp))
        except:
            pass
        
        # Random Forest
        try:
            result_rf = forest_minimize(objective, space, n_calls=30, random_state=self.random_state)  # Reducido
            results.append(('RF', result_rf))
        except:
            pass
        
        # Seleccionar el mejor resultado
        if results:
            best_result = min(results, key=lambda x: x[1].fun)
            best_params = dict(zip([dim.name for dim in space], best_result[1].x))
            
            self.level1_models['xgb_ultra'] = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                objective='reg:squarederror',
                tree_method='hist',
                **best_params
            )
            
            logger.info(f"XGBoost optimizado con {best_result[0]}: {best_params}")
        else:
            # Usar parámetros por defecto conservadores si falla la optimización
            self.level1_models['xgb_ultra'] = xgb.XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                gamma=0.05,
                min_child_weight=5,
                random_state=self.random_state,
                n_jobs=-1,
                objective='reg:squarederror',
                tree_method='hist'
            )
    
    def _optimize_lightgbm_ultra_aggressive(self, X: np.ndarray, y: np.ndarray):
        """Optimización agresiva para LightGBM."""
        
        space = [
            Integer(200, 800, name='n_estimators'),      # Rango más conservador
            Integer(3, 8, name='max_depth'),             # Profundidad limitada
            Real(0.01, 0.15, name='learning_rate'),      # Learning rate más conservador
            Real(0.7, 1.0, name='subsample'),            # Subsample conservador
            Real(0.7, 1.0, name='colsample_bytree'),     # Colsample conservador
            Real(0.0, 1.0, name='reg_alpha'),            # Regularización L1
            Real(0.0, 2.0, name='reg_lambda'),           # Regularización L2
            Integer(5, 25, name='min_child_samples'),     # Mínimo de muestras por hoja
            Real(0.0, 0.1, name='min_split_gain'),       # Ganancia mínima para split
            Integer(15, 63, name='num_leaves')           # Número de hojas limitado
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                force_col_wise=True,
                verbosity=-1,
                **params
            )
            
            mae_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
            return -mae_scores.mean()
        
        try:
            result = gp_minimize(objective, space, n_calls=50, random_state=self.random_state)  # Reducido para velocidad
            best_params = dict(zip([dim.name for dim in space], result.x))
            
            self.level1_models['lgb_ultra'] = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                force_col_wise=True,
                verbosity=-1,
                **best_params
            )
            
            logger.info(f"LightGBM optimizado: {best_params}")
        except Exception as e:
            logger.warning(f"Error optimizando LightGBM: {e}")
            # Usar parámetros por defecto conservadores si falla la optimización
            self.level1_models['lgb_ultra'] = lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=10,
                min_split_gain=0.01,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                force_col_wise=True,
                verbosity=-1
            )
    
    def _optimize_catboost_ultra_aggressive(self, X: np.ndarray, y: np.ndarray):
        """Optimización agresiva para CatBoost."""
        
        space = [
            Integer(200, 800, name='iterations'),        # Rango más conservador
            Integer(3, 8, name='depth'),                 # Profundidad limitada
            Real(0.01, 0.15, name='learning_rate'),      # Learning rate más conservador
            Real(0.7, 1.0, name='subsample'),            # Subsample conservador
            Real(0.0, 2.0, name='reg_lambda'),           # Regularización L2
            Integer(5, 25, name='min_data_in_leaf')      # Mínimo de datos por hoja
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = cb.CatBoostRegressor(
                random_state=self.random_state,
                thread_count=-1,
                loss_function='MAE',
                verbose=False,
                bootstrap_type='Bernoulli',
                **params
            )
            
            mae_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
            return -mae_scores.mean()
        
        try:
            result = gp_minimize(objective, space, n_calls=40, random_state=self.random_state)  # Reducido
            best_params = dict(zip([dim.name for dim in space], result.x))
            
            self.level1_models['cat_ultra'] = cb.CatBoostRegressor(
                random_state=self.random_state,
                thread_count=-1,
                loss_function='MAE',
                verbose=False,
                bootstrap_type='Bernoulli',
                **best_params
            )
            
            logger.info(f"CatBoost optimizado: {best_params}")
        except Exception as e:
            logger.warning(f"Error optimizando CatBoost: {e}")
            # Usar parámetros por defecto conservadores si falla la optimización
            self.level1_models['cat_ultra'] = cb.CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                subsample=0.8,
                reg_lambda=1.0,
                min_data_in_leaf=10,
                random_state=self.random_state,
                thread_count=-1,
                loss_function='MAE',
                verbose=False,
                bootstrap_type='Bernoulli'
            )
    
    def _train_deep_models_ultra_advanced(self, X: np.ndarray, y: np.ndarray, 
                                         X_val: np.ndarray, y_val: np.ndarray):
        """Entrena múltiples modelos de deep learning con técnicas avanzadas."""
        logger.info("🧠 Entrenando modelos de Deep Learning Avanzados...")
        
        for i, model in enumerate(self.level1_deep_models):
            logger.info(f"   Entrenando modelo Deep Learning {i+1}/{len(self.level1_deep_models)}...")
            
            # Crear datasets
            train_dataset = TensorDataset(
                torch.FloatTensor(X).to(self.device),
                torch.FloatTensor(y).to(self.device)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val).to(self.device),
                torch.FloatTensor(y_val).to(self.device)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Optimizador avanzado con diferentes configuraciones
            if i == 0:
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            elif i == 1:
                optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)
            elif i == 2:
                optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.01)
            else:
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
            
            # Scheduler avanzado
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            
            # Función de pérdida combinada
            mse_loss = nn.MSELoss()
            mae_loss = nn.L1Loss()
            
            def combined_loss(pred, target):
                return 0.7 * mse_loss(pred, target) + 0.3 * mae_loss(pred, target)
            
            # Early stopping agresivo
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 30
            
            # Entrenamiento optimizado
            for epoch in range(300):
                # Entrenamiento
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = combined_loss(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping agresivo
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validación
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = combined_loss(outputs, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Guardar mejor modelo
                    torch.save(model.state_dict(), f'best_deep_model_{i}.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"   Early stopping en época {epoch} para modelo {i+1}")
                    break
                
                if epoch % 50 == 0:
                    logger.info(f"   Modelo {i+1} - Época {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            # Cargar mejor modelo
            model.load_state_dict(torch.load(f'best_deep_model_{i}.pth'))
            
            # Evaluar modelo
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.FloatTensor(X_val).to(self.device)).cpu().numpy()
                mae = mean_absolute_error(y_val, val_pred)
                logger.info(f"   Modelo Deep Learning {i+1} - MAE: {mae:.3f}")
        
        logger.info("Modelos de Deep Learning Avanzados entrenados")
    
    def _create_level2_models(self) -> Dict[str, Any]:
        """Crea modelos de segundo nivel (meta-modelos)."""
        models = {}
        
        # Meta-modelos diversos para segundo nivel
        models['meta_xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['meta_lgb'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['meta_rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['meta_ridge'] = Ridge(alpha=1.0)
        models['meta_elastic'] = ElasticNet(alpha=0.5, l1_ratio=0.5)
        
        return models
    
    def _create_level3_models(self) -> Dict[str, Any]:
        """Crea modelos de tercer nivel (meta-meta-modelos)."""
        models = {}
        
        # Meta-meta-modelos especializados
        models['final_xgb'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['final_ridge'] = Ridge(alpha=0.1)
        models['final_elastic'] = ElasticNet(alpha=0.1, l1_ratio=0.7)
        
        # Voting regressor como meta-meta-modelo final
        models['final_voting'] = VotingRegressor([
            ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=self.random_state)),
            ('ridge', Ridge(alpha=0.1)),
            ('elastic', ElasticNet(alpha=0.1))
        ])
        
        return models

    def _select_features_multi_level(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], level: int = 1) -> Tuple[np.ndarray, List[str]]:
        """Selección inteligente de características multi-nivel."""
        logger.info(f"Seleccionando características para nivel {level}...")
        
        if level == 1:
            # Selección más agresiva para primer nivel
            k_best = min(int(X.shape[1] * 0.9), 120)
            selector = SelectKBest(score_func=f_regression, k=k_best)
        elif level == 2:
            # Selección moderada para segundo nivel
            k_best = min(int(X.shape[1] * 0.8), 80)
            selector = SelectKBest(score_func=f_regression, k=k_best)
        else:
            # Selección conservadora para tercer nivel
            k_best = min(int(X.shape[1] * 0.7), 50)
            selector = SelectKBest(score_func=f_regression, k=k_best)
        
        X_selected = selector.fit_transform(X, y)
        
        # Obtener características seleccionadas
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        self.feature_selectors[f'level_{level}'] = selector
        
        logger.info(f"Seleccionadas {len(selected_features)} características de {len(feature_names)} para nivel {level}")
        
        return X_selected, selected_features

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> 'UltraAdvancedEnsembleRegressor':
        """Entrena el ensemble avanzado de tres niveles."""
        logger.info("Iniciando entrenamiento del ENSEMBLE AVANZADO DE TERCER NIVEL...")
        
        # Dividir datos con estratificación más sofisticada
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state,
            stratify=pd.cut(y, bins=10, labels=False)  # Más bins para mejor estratificación
        )
        
        # Preparar nombres de características
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # ==================== NIVEL 1: MODELOS BASE ====================
        logger.info(" Entrenando NIVEL 1: MODELOS BASE OPTIMIZADOS")
        
        # Selección de características para nivel 1
        X_train_l1, selected_features_l1 = self._select_features_multi_level(X_train, y_train, feature_names, level=1)
        X_val_l1 = self.feature_selectors['level_1'].transform(X_val)
        
        # Escalado para nivel 1
        self.scalers['level_1'] = RobustScaler()
        X_train_l1_scaled = self.scalers['level_1'].fit_transform(X_train_l1)
        X_val_l1_scaled = self.scalers['level_1'].transform(X_val_l1)
        
        # Crear modelos de nivel 1
        self.level1_models = self._create_level1_models()
        
        # Optimización agresiva de hiperparámetros
        if self.aggressive_optimization:
            self._aggressive_hyperparameter_optimization(X_train_l1_scaled, y_train)
        
        # Entrenar modelos tradicionales de nivel 1
        logger.info("   Entrenando modelos tradicionales de nivel 1...")
        level1_predictions_train = np.zeros((len(X_train_l1_scaled), len(self.level1_models)))
        level1_predictions_val = np.zeros((len(X_val_l1_scaled), len(self.level1_models)))
        
        for i, (name, model) in enumerate(self.level1_models.items()):
            logger.info(f"      Entrenando {name}...")
            model.fit(X_train_l1_scaled, y_train)
            
            level1_predictions_train[:, i] = model.predict(X_train_l1_scaled)
            level1_predictions_val[:, i] = model.predict(X_val_l1_scaled)
            
            mae = mean_absolute_error(y_val, level1_predictions_val[:, i])
            logger.info(f"         {name} - MAE: {mae:.3f}")
        
        # Crear y entrenar modelos de deep learning de nivel 1
        if self.use_deep_learning:
            self.level1_deep_models = self._create_level1_deep_models(X_train_l1_scaled.shape[1])
            self._train_deep_models_ultra_advanced(X_train_l1_scaled, y_train, X_val_l1_scaled, y_val)
            
            # Predicciones de modelos de deep learning
            deep_predictions_train = np.zeros((len(X_train_l1_scaled), len(self.level1_deep_models)))
            deep_predictions_val = np.zeros((len(X_val_l1_scaled), len(self.level1_deep_models)))
            
            for i, model in enumerate(self.level1_deep_models):
                model.eval()
                with torch.no_grad():
                    deep_predictions_train[:, i] = model(torch.FloatTensor(X_train_l1_scaled).to(self.device)).cpu().numpy()
                    deep_predictions_val[:, i] = model(torch.FloatTensor(X_val_l1_scaled).to(self.device)).cpu().numpy()
                
                mae = mean_absolute_error(y_val, deep_predictions_val[:, i])
                logger.info(f"         Deep Learning {i+1} - MAE: {mae:.3f}")
            
            # Combinar predicciones tradicionales y de deep learning
            all_level1_train = np.column_stack([level1_predictions_train, deep_predictions_train])
            all_level1_val = np.column_stack([level1_predictions_val, deep_predictions_val])
        else:
            all_level1_train = level1_predictions_train
            all_level1_val = level1_predictions_val
        
        logger.info(f"NIVEL 1 COMPLETADO: {all_level1_train.shape[1]} predictores base")
        
        # ==================== NIVEL 2: META-MODELOS ====================
        logger.info(" Entrenando NIVEL 2: META-MODELOS AVANZADOS")
        
        # Crear modelos de nivel 2
        self.level2_models = self._create_level2_models()
        
        # Entrenar meta-modelos
        level2_predictions_train = np.zeros((len(all_level1_train), len(self.level2_models)))
        level2_predictions_val = np.zeros((len(all_level1_val), len(self.level2_models)))
        
        for i, (name, model) in enumerate(self.level2_models.items()):
            logger.info(f"   Entrenando meta-modelo {name}...")
            model.fit(all_level1_train, y_train)
            
            level2_predictions_train[:, i] = model.predict(all_level1_train)
            level2_predictions_val[:, i] = model.predict(all_level1_val)
            
            mae = mean_absolute_error(y_val, level2_predictions_val[:, i])
            logger.info(f"      {name} - MAE: {mae:.3f}")
        
        logger.info(f"NIVEL 2 COMPLETADO: {level2_predictions_train.shape[1]} meta-predictores")
        
        # ==================== NIVEL 3: META-META-MODELOS ====================
        if self.third_level_ensemble:
            logger.info(" Entrenando NIVEL 3: META-META-MODELOS ESPECIALIZADOS")
            
            # Crear modelos de nivel 3
            self.level3_models = self._create_level3_models()
            
            # Entrenar meta-meta-modelos
            level3_predictions_val = np.zeros((len(level2_predictions_val), len(self.level3_models)))
            
            for i, (name, model) in enumerate(self.level3_models.items()):
                logger.info(f"   Entrenando meta-meta-modelo {name}...")
                model.fit(level2_predictions_train, y_train)
                
                level3_predictions_val[:, i] = model.predict(level2_predictions_val)
                
                mae = mean_absolute_error(y_val, level3_predictions_val[:, i])
                logger.info(f"      {name} - MAE: {mae:.3f}")
            
            # Modelo final especializado
            logger.info("   Entrenando MODELO FINAL ESPECIALIZADO...")
            self.final_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.final_model.fit(level3_predictions_val, y_val)
            final_predictions = self.final_model.predict(level3_predictions_val)
            
            logger.info(f"NIVEL 3 COMPLETADO: {level3_predictions_val.shape[1]} meta-meta-predictores")
        else:
            # Si no hay tercer nivel, usar el mejor modelo de nivel 2
            best_l2_idx = np.argmin([mean_absolute_error(y_val, level2_predictions_val[:, i]) 
                                   for i in range(level2_predictions_val.shape[1])])
            final_predictions = level2_predictions_val[:, best_l2_idx]
            self.final_model = list(self.level2_models.values())[best_l2_idx]
        
        # ==================== EVALUACIÓN FINAL ====================
        final_mae = mean_absolute_error(y_val, final_predictions)
        final_r2 = r2_score(y_val, final_predictions)
        accuracy_1 = np.mean(np.abs(y_val - final_predictions) <= 1) * 100
        accuracy_exact = np.mean(np.abs(y_val - final_predictions) <= 0.5) * 100
        
        logger.info("RESULTADOS FINALES DEL ENSEMBLE AVANZADO:")
        logger.info(f"   MAE: {final_mae:.3f}")
        logger.info(f"   R²: {final_r2:.3f}")
        logger.info(f"   Precisión ±1: {accuracy_1:.1f}%")
        logger.info(f"   Precisión exacta: {accuracy_exact:.1f}%")
        
        # Verificar si alcanzamos el objetivo
        if accuracy_exact >= 97.0:
            logger.info("¡OBJETIVO ALCANZADO! Precisión ≥97% conseguida")
        else:
            logger.info(f"⚠️  Necesita más optimización: {accuracy_exact:.1f}% < 97.0%")
        
        self.is_fitted = True
        logger.info("ENSEMBLE AVANZADO DE TERCER NIVEL entrenado exitosamente")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones con el ensemble avanzado de tres niveles."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        # ==================== PREDICCIONES NIVEL 1 ====================
        # Seleccionar y escalar características para nivel 1
        X_l1 = self.feature_selectors['level_1'].transform(X)
        X_l1_scaled = self.scalers['level_1'].transform(X_l1)
        
        # Predicciones de modelos tradicionales de nivel 1
        level1_predictions = np.zeros((len(X_l1_scaled), len(self.level1_models)))
        
        for i, (name, model) in enumerate(self.level1_models.items()):
            level1_predictions[:, i] = model.predict(X_l1_scaled)
        
        # Predicciones de modelos de deep learning de nivel 1
        if self.use_deep_learning and self.level1_deep_models:
            deep_predictions = np.zeros((len(X_l1_scaled), len(self.level1_deep_models)))
            
            for i, model in enumerate(self.level1_deep_models):
                model.eval()
                with torch.no_grad():
                    deep_predictions[:, i] = model(torch.FloatTensor(X_l1_scaled).to(self.device)).cpu().numpy()
            
            all_level1_predictions = np.column_stack([level1_predictions, deep_predictions])
        else:
            all_level1_predictions = level1_predictions
        
        # ==================== PREDICCIONES NIVEL 2 ====================
        level2_predictions = np.zeros((len(all_level1_predictions), len(self.level2_models)))
        
        for i, (name, model) in enumerate(self.level2_models.items()):
            level2_predictions[:, i] = model.predict(all_level1_predictions)
        
        # ==================== PREDICCIONES NIVEL 3 ====================
        if self.third_level_ensemble and hasattr(self, 'level3_models'):
            level3_predictions = np.zeros((len(level2_predictions), len(self.level3_models)))
            
            for i, (name, model) in enumerate(self.level3_models.items()):
                level3_predictions[:, i] = model.predict(level2_predictions)
            
            # Predicción final especializada
            final_predictions = self.final_model.predict(level3_predictions)
        else:
            # Si no hay tercer nivel, usar modelo final de nivel 2
            final_predictions = self.final_model.predict(level2_predictions)
        
        # Asegurar que sean no negativos y enteros
        final_predictions = np.maximum(final_predictions, 0)
        final_predictions = np.round(final_predictions).astype(int)
        
        return final_predictions

class UltraSpecificReboundsPredictor:
    """
    Predictor ULTRA-ESPECÍFICO para rebotes con enfoque quirúrgico.
    
    OBJETIVO: ≥97% precisión usando solo las características más predictivas.
    ENFOQUE: Menos características, más precisión, modelos especializados.
    """
    
    def __init__(self, use_simple_ensemble: bool = True):
        """
        Inicializa el predictor ultra-específico.
        
        Args:
            use_simple_ensemble: Si usar ensemble simple pero efectivo
        """
        self.use_simple_ensemble = use_simple_ensemble
        self.models = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info("🎯 UltraSpecificReboundsPredictor inicializado (enfoque quirúrgico)")
    
    def _create_ultra_specific_models(self) -> Dict[str, Any]:
        """Crea modelos ultra-específicos optimizados para rebotes."""
        models = {}
        
        # 1. XGBoost ultra-optimizado para rebotes
        models['xgb_ultra'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. LightGBM ultra-optimizado
        models['lgb_ultra'] = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.3,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # 3. CatBoost ultra-optimizado
        models['cat_ultra'] = cb.CatBoostRegressor(
            iterations=350,
            depth=4,
            learning_rate=0.06,
            reg_lambda=0.2,
            random_state=42,
            verbose=False
        )
        
        # 4. Random Forest especializado
        models['rf_ultra'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 5. Ridge ultra-optimizado (para estabilidad)
        models['ridge_ultra'] = Ridge(
            alpha=1.0,
            random_state=42
        )
        
        return models
    
    def _create_ultra_specific_deep_model(self, input_dim: int) -> nn.Module:
        """Crea red neuronal ultra-específica para rebotes."""
        
        class UltraSpecificNet(nn.Module):
            def __init__(self, input_dim: int):
                super(UltraSpecificNet, self).__init__()
                
                # Arquitectura específica para rebotes (más simple pero efectiva)
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    
                    nn.Linear(16, 1)
                )
                
                # Inicialización específica para rebotes
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                return self.network(x)
        
        return UltraSpecificNet(input_dim)
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Entrena el predictor ultra-específico.
        
        Args:
            X: Características ultra-específicas
            y: Targets de rebotes
            feature_names: Nombres de características
            
        Returns:
            Métricas de entrenamiento
        """
        logger.info("🎯 Entrenando predictor ULTRA-ESPECÍFICO para rebotes...")
        
        self.feature_names = feature_names
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Crear modelos ultra-específicos
        self.models = self._create_ultra_specific_models()
        
        # Entrenar modelos tradicionales
        logger.info("🔧 Entrenando modelos tradicionales ultra-específicos...")
        for name, model in self.models.items():
            logger.info(f"   Entrenando {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluar
            val_pred = model.predict(X_val_scaled)
            mae = mean_absolute_error(y_val, val_pred)
            logger.info(f"      {name} - MAE: {mae:.3f}")
        
        # Entrenar modelo de deep learning ultra-específico
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("🚀 Usando GPU para entrenamiento")
        else:
            device = torch.device('cpu')
            logger.info("🖥️  Usando CPU para entrenamiento")
        
        logger.info("🧠 Entrenando red neuronal ultra-específica...")
        deep_model = self._create_ultra_specific_deep_model(X_train_scaled.shape[1])
        deep_model = deep_model.to(device)
        
        # Configuración de entrenamiento
        optimizer = optim.AdamW(deep_model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        # Convertir a tensores
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        # Entrenamiento
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(200):
            # Entrenamiento
            deep_model.train()
            optimizer.zero_grad()
            
            train_pred = deep_model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validación
            deep_model.eval()
            with torch.no_grad():
                val_pred = deep_model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                best_model_state = deep_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"   Early stopping en época {epoch}")
                    break
            
            if epoch % 50 == 0:
                logger.info(f"   Época {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        # Cargar mejor modelo
        deep_model.load_state_dict(best_model_state)
        self.models['deep_ultra'] = deep_model
        
        # Evaluar modelo deep
        deep_model.eval()
        with torch.no_grad():
            val_pred_deep = deep_model(X_val_tensor).cpu().numpy().flatten()
        mae_deep = mean_absolute_error(y_val, val_pred_deep)
        logger.info(f"      deep_ultra - MAE: {mae_deep:.3f}")
        
        # Crear ensemble final ultra-específico
        if self.use_simple_ensemble:
            logger.info("🎯 Creando ensemble ultra-específico...")
            
            # Obtener predicciones de todos los modelos
            ensemble_preds = []
            for name, model in self.models.items():
                if name == 'deep_ultra':
                    model.eval()
                    with torch.no_grad():
                        pred = model(X_val_tensor).cpu().numpy().flatten()
                else:
                    pred = model.predict(X_val_scaled)
                ensemble_preds.append(pred)
            
            ensemble_preds = np.array(ensemble_preds).T
            
            # Entrenar meta-modelo simple pero efectivo
            self.meta_model = Ridge(alpha=0.1)
            self.meta_model.fit(ensemble_preds, y_val)
            
            # Evaluar ensemble
            ensemble_pred = self.meta_model.predict(ensemble_preds)
            mae_ensemble = mean_absolute_error(y_val, ensemble_pred)
            r2_ensemble = r2_score(y_val, ensemble_pred)
            
            # Calcular precisión exacta y ±1
            exact_accuracy = np.mean(np.abs(ensemble_pred - y_val) < 0.5) * 100
            plus_minus_1_accuracy = np.mean(np.abs(ensemble_pred - y_val) <= 1.0) * 100
            
            logger.info("✅ RESULTADOS ULTRA-ESPECÍFICOS:")
            logger.info(f"   MAE: {mae_ensemble:.3f}")
            logger.info(f"   R²: {r2_ensemble:.3f}")
            logger.info(f"   Precisión exacta: {exact_accuracy:.1f}%")
            logger.info(f"   Precisión ±1: {plus_minus_1_accuracy:.1f}%")
            
            if exact_accuracy >= 97.0:
                logger.info("🎉 ¡OBJETIVO ALCANZADO! Precisión ≥97%")
            else:
                logger.info(f"⚠️  Necesita más optimización: {exact_accuracy:.1f}% < 97.0%")
        
        self.is_trained = True
        
        return {
            'mae': mae_ensemble if self.use_simple_ensemble else mae_deep,
            'r2': r2_ensemble if self.use_simple_ensemble else r2_score(y_val, val_pred_deep),
            'exact_accuracy': exact_accuracy if self.use_simple_ensemble else np.mean(np.abs(val_pred_deep - y_val) < 0.5) * 100,
            'plus_minus_1_accuracy': plus_minus_1_accuracy if self.use_simple_ensemble else np.mean(np.abs(val_pred_deep - y_val) <= 1.0) * 100
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice rebotes usando el modelo ultra-específico."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Escalar características
        X_scaled = self.scaler.transform(X)
        
        if self.use_simple_ensemble:
            # Obtener predicciones de todos los modelos
            ensemble_preds = []
            
            for name, model in self.models.items():
                if name == 'deep_ultra':
                    model.eval()
                    device = next(model.parameters()).device
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    with torch.no_grad():
                        pred = model(X_tensor).cpu().numpy().flatten()
                else:
                    pred = model.predict(X_scaled)
                ensemble_preds.append(pred)
            
            ensemble_preds = np.array(ensemble_preds).T
            
            # Predicción final del ensemble
            predictions = self.meta_model.predict(ensemble_preds)
        else:
            # Solo usar modelo deep
            model = self.models['deep_ultra']
            model.eval()
            device = next(model.parameters()).device
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            with torch.no_grad():
                predictions = model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def cross_validate_ultra_specific(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """Validación cruzada ultra-específica."""
        logger.info(f"🔬 Realizando validación cruzada ULTRA-ESPECÍFICA con {cv_folds} folds...")
        
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"  Procesando Fold {fold+1}/{cv_folds}...")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Crear y entrenar modelo temporal
            temp_predictor = UltraSpecificReboundsPredictor(use_simple_ensemble=self.use_simple_ensemble)
            temp_predictor.train(X_train_fold, y_train_fold, self.feature_names)
            
            # Predecir
            fold_pred = temp_predictor.predict(X_val_fold)
            
            # Calcular métricas
            fold_mae = mean_absolute_error(y_val_fold, fold_pred)
            fold_r2 = r2_score(y_val_fold, fold_pred)
            fold_exact = np.mean(np.abs(fold_pred - y_val_fold) < 0.5) * 100
            fold_plus_minus_1 = np.mean(np.abs(fold_pred - y_val_fold) <= 1.0) * 100
            
            fold_scores.append({
                'mae': fold_mae,
                'r2': fold_r2,
                'exact_accuracy': fold_exact,
                'plus_minus_1_accuracy': fold_plus_minus_1
            })
            
            logger.info(f"     Fold {fold+1} - MAE: {fold_mae:.3f}, R²: {fold_r2:.3f}, Exacta: {fold_exact:.1f}%")
        
        # Calcular promedios
        avg_scores = {
            'mae': np.mean([s['mae'] for s in fold_scores]),
            'r2': np.mean([s['r2'] for s in fold_scores]),
            'exact_accuracy': np.mean([s['exact_accuracy'] for s in fold_scores]),
            'plus_minus_1_accuracy': np.mean([s['plus_minus_1_accuracy'] for s in fold_scores])
        }
        
        logger.info("📊 RESULTADOS VALIDACIÓN CRUZADA ULTRA-ESPECÍFICA:")
        logger.info(f"   MAE promedio: {avg_scores['mae']:.3f}")
        logger.info(f"   R² promedio: {avg_scores['r2']:.3f}")
        logger.info(f"   Precisión exacta promedio: {avg_scores['exact_accuracy']:.1f}%")
        logger.info(f"   Precisión ±1 promedio: {avg_scores['plus_minus_1_accuracy']:.1f}%")
        
        if avg_scores['exact_accuracy'] >= 97.0:
            logger.info("🎉 ¡OBJETIVO ALCANZADO EN VALIDACIÓN CRUZADA! Precisión ≥97%")
        else:
            logger.info(f"⚠️  Necesita más optimización: {avg_scores['exact_accuracy']:.1f}% < 97.0%")
        
        return avg_scores
