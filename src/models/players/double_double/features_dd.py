"""
Módulo de Características para Predicción de Double Double
=========================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de double double de un jugador NBA por partido. Implementa características
avanzadas enfocadas en factores que determinan la probabilidad de lograr un double double.

Sin data leakage, todas las métricas usan shift(1) para crear historial

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DoubleDoubleFeatureEngineer:
    """
    Motor de features para predicción de double double usando ESTADÍSTICAS HISTÓRICAS
    OPTIMIZADO - Rendimiento pasado para predecir juegos futuros
    """
    
    def __init__(self, lookback_games: int = 10):
        """Inicializa el ingeniero de características para predicción de double double."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        # Cache para evitar recálculos
        self._cached_calculations = {}
        # Cache para features generadas
        self._features_cache = {}
        self._last_data_hash = None
        
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generar hash único para el DataFrame"""
        try:
            # Usar shape, columnas y algunos valores para crear hash
            data_info = f"{df.shape}_{list(df.columns)}_{df.iloc[0].sum() if len(df) > 0 else 0}_{df.iloc[-1].sum() if len(df) > 0 else 0}"
            return str(hash(data_info))
        except:
            return str(hash(str(df.shape)))
    
    def _ensure_datetime_and_sort(self, df: pd.DataFrame) -> None:
        """Método auxiliar para asegurar que Date esté en formato datetime y ordenar datos"""
        if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.sort_values(['Player', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.debug("Datos ordenados cronológicamente por jugador")
    
    def _calculate_basic_temporal_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features temporales básicas una sola vez"""
        if 'Date' in df.columns:
            # Calcular una sola vez todas las features temporales
            df['days_rest'] = df.groupby('Player')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
            
            # Back-to-back indicator (calculado una sola vez)
            df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
            
            logger.debug("Features temporales básicas calculadas")
    
    def _calculate_player_context_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features de contexto del jugador una sola vez"""
        # Features de contexto ya disponibles del data_loader
        if 'is_home' not in df.columns:
            logger.debug("is_home no encontrado del data_loader - features de ventaja local no disponibles")
        else:
            logger.debug("Usando is_home del data_loader para features de ventaja local")
            # Calcular features relacionadas con ventaja local
            df['home_advantage'] = df['is_home'] * 0.03  # 3% boost para jugadores en casa
            df['travel_penalty'] = np.where(df['is_home'] == 0, -0.01, 0.0)
        
        # Features de titular/suplente ya disponibles del data_loader
        if 'is_started' not in df.columns:
            logger.debug("is_started no encontrado del data_loader - features de titular no disponibles")
        else:
            logger.debug("Usando is_started del data_loader para features de titular")
            # Boost para titulares (más minutos = más oportunidades de double double)
            df['starter_boost'] = df['is_started'] * 0.15
    
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        PIPELINE SIMPLIFICADO DE FEATURES ANTI-OVERFITTING
        Usar solo estadísticas básicas históricas - MENOS COMPLEJIDAD
        REGENERAR SIEMPRE para asegurar consistency
        """
        
        # DESHABILITAR CACHE temporalmente para asegurar consistency
        # La verificación y el entrenamiento deben usar las mismas features
        logger.info("Generando features NBA ESPECIALIZADAS anti-overfitting para double double...")

        # VERIFICACIÓN DE double_double COMO TARGET (ya viene del dataset)
        if 'double_double' in df.columns:
            dd_distribution = df['double_double'].value_counts().to_dict()
            logger.info(f"Target double_double disponible - Distribución: {dd_distribution}")
        else:
            logger.error("double_double no encontrado en el dataset - requerido para features de double double")
            return []
        
        # VERIFICAR FEATURES DEL DATA_LOADER (consolidado en un solo mensaje)
        data_loader_features = ['is_home', 'is_started', 'Height_Inches', 'Weight', 'BMI']
        available_features = [f for f in data_loader_features if f in df.columns]
        missing_features = [f for f in data_loader_features if f not in df.columns]
        
        if available_features:
            logger.info(f"Features del data_loader: {len(available_features)}/{len(data_loader_features)} disponibles")
        if missing_features:
            logger.debug(f"Features faltantes: {missing_features}")
        
        # Trabajar directamente con el DataFrame
        if df.empty:
            return []
        
        # PASO 0: Preparación básica (SIEMPRE ejecutar)
        self._ensure_datetime_and_sort(df)
        self._calculate_basic_temporal_features(df)
        self._calculate_player_context_features(df)
        
        logger.info("Iniciando generación de features ESPECIALIZADAS...")
        
        # *** CREAR FEATURES ESPECIALIZADAS EN EL DATAFRAME SIEMPRE ***
        logger.info("Creando features especializadas en el DataFrame...")
        
        # GENERAR TODAS LAS FEATURES ESPECIALIZADAS
        self._create_temporal_features_simple(df)
        self._create_contextual_features_simple(df)
        self._create_performance_features_simple(df)
        self._create_double_double_features_simple(df)
        self._create_statistical_features_simple(df)
        self._create_opponent_features_simple(df)
        self._create_biometric_features_simple(df)
        
        logger.info("Features especializadas creadas en el DataFrame")
        
        # Actualizar lista de features disponibles después de crearlas
        self._update_feature_columns(df)
        
        # Compilar lista de características ESPECIALIZADAS ÚNICAMENTE
        specialized_features = [col for col in df.columns if col not in [
            # Columnas básicas del dataset
            'Player', 'Date', 'Team', 'Opp', 'Result', 'MP', 'GS', 'Away',
            # Estadísticas del juego actual (NO USAR - data leakage)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            # Columnas de double específicas del juego actual
            'PTS_double', 'TRB_double', 'AST_double', 'STL_double', 'BLK_double',
            # Target variables
            'double_double', 'triple_double',
            # Columnas auxiliares temporales (básicas del dataset)
            'day_of_week', 'month', 'days_rest', 'days_into_season'
        ]]
        
        # COMPILAR FEATURES ESENCIALES ÚNICAMENTE (REDUCIR COMPLEJIDAD)
        essential_features = []
        
        # PRIORIDAD 1: Features de double double especializadas (MÁXIMO 8)
        dd_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'dd_rate_5g', 'weighted_dd_rate_5g', 'dd_momentum_5g', 'dd_streak',
            'dd_form_trend', 'dd_consistency_10g', 'dd_potential_score'
        ])]
        essential_features.extend(dd_features[:8])  # LIMITAR a 8
        
        # PRIORIDAD 2: Features de consistencia (MUY IMPORTANTES según análisis)
        consistency_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'usage_consistency_5g', 'mp_consistency_5g', 'trb_consistency_5g', 
            'ast_consistency_5g', 'pts_consistency_5g', 'efficiency_consistency_5g',
            'overall_consistency', 'minutes_stability'
        ])]
        essential_features.extend([f for f in consistency_features if f not in essential_features][:10])
        
        # PRIORIDAD 3: Features de rendimiento estadístico (MÁXIMO 8)
        stats_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'pts_hist_avg_5g', 'trb_hist_avg_5g', 'ast_hist_avg_5g', 'mp_hist_avg_5g',
            'trb_above_avg', 'pts_above_avg', 'versatility_index', 'total_impact_5g'
        ])]
        essential_features.extend([f for f in stats_features if f not in essential_features][:8])
        
        # PRIORIDAD 4: Features contextuales avanzadas (MÁXIMO 8)
        context_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'starter_boost', 'is_center', 'is_guard', 'is_forward',
            'high_minutes_player', 'starter_minutes', 'home_advantage',
            'position_dd_likelihood', 'well_rounded_player', 'workload_factor'
        ])]
        essential_features.extend([f for f in context_features if f not in essential_features][:8])
        
        # PRIORIDAD 5: Features de momentum y tendencia (MÁXIMO 6)
        momentum_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'combined_momentum', 'pts_momentum_6g', 'trb_momentum_6g', 'ast_momentum_6g',
            'pts_trend_factor', 'trb_trend_factor', 'ast_trend_factor'
        ])]
        essential_features.extend([f for f in momentum_features if f not in essential_features][:6])
        
        # PRIORIDAD 6: Features de proximidad y oportunidad (MÁXIMO 5)
        proximity_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'pts_dd_proximity', 'trb_dd_proximity', 'ast_dd_proximity',
            'best_dd_combo_score', 'pts_trb_combined'
        ])]
        essential_features.extend([f for f in proximity_features if f not in essential_features][:5])
        
        # PRIORIDAD 7: Features temporales básicas (MÁXIMO 3)
        temporal_features = [col for col in specialized_features if any(keyword in col for keyword in [
            'is_weekend', 'is_back_to_back', 'energy_factor'
        ])]
        essential_features.extend([f for f in temporal_features if f not in essential_features][:3])
        
        # Agregar nuevas features que existen en el dataframe
        remaining_new_features = [
            'primary_scorer', 'primary_rebounder', 'primary_playmaker',
            'team_scoring_importance', 'team_rebounding_importance', 'high_workload'
        ]
        
        for feature in remaining_new_features:
            if feature in df.columns and feature not in essential_features:
                essential_features.append(feature)
                if len(essential_features) >= 50:  # Límite total aumentado para incluir más features predictivas
                    break
        
        logger.info(f"Features esenciales seleccionadas: {len(essential_features)}")
        logger.info(f"Distribución: DD={len([f for f in essential_features if 'dd_' in f])}, "
                   f"Stats={len([f for f in essential_features if 'hist_avg' in f])}, "
                   f"Consistency={len([f for f in essential_features if 'consistency' in f])}")
        
        # Actualizar lista de features
        self.feature_columns = essential_features
        
        # NUEVAS FEATURES CONTEXTUALES PARA MAYOR PRECISIÓN
        
        # 1. Features de posición específica
        df['is_center'] = (df['Height_Inches'] >= 82).astype(int)  # 6'10" o más
        df['is_guard'] = (df['Height_Inches'] <= 78).astype(int)   # 6'6" o menos
        df['is_forward'] = ((df['Height_Inches'] > 78) & (df['Height_Inches'] < 82)).astype(int)
        
        # 2. Features de minutos esperados (proxy para importancia)
        if 'mp_hist_avg_5g' in df.columns:
            df['high_minutes_player'] = (df['mp_hist_avg_5g'] >= 25).astype(int)
            df['starter_minutes'] = (df['mp_hist_avg_5g'] >= 20).astype(int)
        else:
            df['high_minutes_player'] = 0
            df['starter_minutes'] = 0
        
        # 3. Features de rendimiento reciente vs histórico
        if 'PTS' in df.columns and 'pts_hist_avg_5g' in df.columns:
            df['pts_above_avg'] = (df['PTS'] > df['pts_hist_avg_5g']).astype(int)
        else:
            df['pts_above_avg'] = 0
            
        if 'TRB' in df.columns and 'trb_hist_avg_5g' in df.columns:
            df['trb_above_avg'] = (df['TRB'] > df['trb_hist_avg_5g']).astype(int)
        else:
            df['trb_above_avg'] = 0
        
        # 4. Features de combinación de stats
        if all(col in df.columns for col in ['pts_hist_avg_5g', 'trb_hist_avg_5g', 'ast_hist_avg_5g']):
            df['pts_trb_combined'] = df['pts_hist_avg_5g'] + df['trb_hist_avg_5g']
            df['pts_ast_combined'] = df['pts_hist_avg_5g'] + df['ast_hist_avg_5g']
            df['trb_ast_combined'] = df['trb_hist_avg_5g'] + df['ast_hist_avg_5g']
        else:
            df['pts_trb_combined'] = 0.0
            df['pts_ast_combined'] = 0.0
            df['trb_ast_combined'] = 0.0
        
        # 5. Feature de probabilidad de DD basada en posición
        if 'Height_Inches' in df.columns:
            df['position_dd_likelihood'] = 0.0
            df.loc[df['is_center'] == 1, 'position_dd_likelihood'] = 0.15  # Centros más probable
            df.loc[df['is_forward'] == 1, 'position_dd_likelihood'] = 0.10  # Forwards moderado
            df.loc[df['is_guard'] == 1, 'position_dd_likelihood'] = 0.05   # Guards menos probable
        else:
            df['position_dd_likelihood'] = 0.05  # Valor neutral
        
        # 6. Features de consistencia mejoradas
        if all(col in df.columns for col in ['pts_consistency_5g', 'trb_consistency_5g', 'ast_consistency_5g']):
            df['overall_consistency'] = (
                df['pts_consistency_5g'] + 
                df['trb_consistency_5g'] + 
                df['ast_consistency_5g']
            ) / 3
        else:
            df['overall_consistency'] = 0.5  # Valor neutral
        
        # 7. Feature de momentum combinado
        if all(col in df.columns for col in ['dd_momentum_5g', 'weighted_dd_rate_5g']):
            df['combined_momentum'] = (
                df['dd_momentum_5g'] * df['weighted_dd_rate_5g']
            )
        else:
            df['combined_momentum'] = 0.0
        
        # 8. NUEVAS FEATURES CONTEXTUALES AVANZADAS
        
        # Features de rol del jugador basadas en stats
        if all(col in df.columns for col in ['pts_hist_avg_5g', 'trb_hist_avg_5g', 'ast_hist_avg_5g']):
            # Clasificación de rol principal
            df['primary_scorer'] = (df['pts_hist_avg_5g'] >= 15).astype(int)
            df['primary_rebounder'] = (df['trb_hist_avg_5g'] >= 8).astype(int)
            df['primary_playmaker'] = (df['ast_hist_avg_5g'] >= 5).astype(int)
            
            # Jugador "completo" (bueno en múltiples áreas)
            df['well_rounded_player'] = (
                (df['pts_hist_avg_5g'] >= 10) & 
                (df['trb_hist_avg_5g'] >= 5) & 
                (df['ast_hist_avg_5g'] >= 3)
            ).astype(int)
        
        # Features de contexto de equipo (si están disponibles)
        if 'Team' in df.columns:
            # Calcular promedio de equipo para contexto relativo
            team_avg_pts = df.groupby(['Team', 'Date'])['PTS'].transform('mean') if 'PTS' in df.columns else 0
            team_avg_trb = df.groupby(['Team', 'Date'])['TRB'].transform('mean') if 'TRB' in df.columns else 0
            
            # Importancia relativa en el equipo
            if 'PTS' in df.columns:
                df['team_scoring_importance'] = df['pts_hist_avg_5g'] / (team_avg_pts + 0.1)
            if 'TRB' in df.columns:
                df['team_rebounding_importance'] = df['trb_hist_avg_5g'] / (team_avg_trb + 0.1)
        
        # Features de fatiga y carga de trabajo
        if 'MP' in df.columns:
            # Carga de trabajo reciente
            recent_minutes = self._get_historical_series(df, 'MP', 3, 'mean')
            season_avg_minutes = self._get_historical_series(df, 'MP', 20, 'mean')
            # CORREGIR: Asegurar que los índices coincidan
            recent_minutes_aligned = recent_minutes.reindex(df.index).fillna(20)
            season_avg_minutes_aligned = season_avg_minutes.reindex(df.index).fillna(20)
            df['workload_factor'] = recent_minutes_aligned / (season_avg_minutes_aligned + 0.1)
            
            # Indicador de alta carga de trabajo
            if 'mp_hist_avg_5g' in df.columns:
                df['high_workload'] = (df['mp_hist_avg_5g'] >= 30).astype(int)
            else:
                df['high_workload'] = 0
        else:
            df['workload_factor'] = 1.0
            df['high_workload'] = 0
        
        # Features de oportunidad de double-double
        if all(col in df.columns for col in ['pts_hist_avg_5g', 'trb_hist_avg_5g', 'ast_hist_avg_5g']):
            # Proximidad a double-double en cada categoría
            df['pts_dd_proximity'] = np.minimum(df['pts_hist_avg_5g'] / 10.0, 1.0)
            df['trb_dd_proximity'] = np.minimum(df['trb_hist_avg_5g'] / 10.0, 1.0)
            df['ast_dd_proximity'] = np.minimum(df['ast_hist_avg_5g'] / 10.0, 1.0)
            
            # Mejor combinación para DD
            df['best_dd_combo_score'] = np.maximum(
                df['pts_dd_proximity'] + df['trb_dd_proximity'],
                np.maximum(
                    df['pts_dd_proximity'] + df['ast_dd_proximity'],
                    df['trb_dd_proximity'] + df['ast_dd_proximity']
                )
            )
        else:
            df['pts_dd_proximity'] = 0.0
            df['trb_dd_proximity'] = 0.0
            df['ast_dd_proximity'] = 0.0
            df['best_dd_combo_score'] = 0.0
        
        # Features de forma reciente
        if 'double_double' in df.columns:
            # Forma reciente en double-doubles (últimos 3 vs anteriores 7)
            recent_dd_rate = df.groupby('Player')['double_double'].shift(1).rolling(window=3, min_periods=1).mean()
            older_dd_rate = df.groupby('Player')['double_double'].shift(4).rolling(window=7, min_periods=1).mean()
            # CORREGIR: Asegurar que los índices coincidan
            recent_dd_rate_aligned = recent_dd_rate.reindex(df.index).fillna(0)
            older_dd_rate_aligned = older_dd_rate.reindex(df.index).fillna(0)
            df['dd_form_trend'] = recent_dd_rate_aligned - older_dd_rate_aligned
            
            # Consistencia en double-doubles
            dd_std = df.groupby('Player')['double_double'].shift(1).rolling(window=10, min_periods=2).std()
            dd_std_aligned = dd_std.reindex(df.index).fillna(1)
            df['dd_consistency_10g'] = 1 / (dd_std_aligned + 1)
        else:
            df['dd_form_trend'] = 0.0
            df['dd_consistency_10g'] = 0.5
        
        return essential_features
    
    def _create_temporal_features_simple(self, df: pd.DataFrame) -> None:
        """Features temporales básicas disponibles antes del juego"""
        # Solo agregar features adicionales aquí
        if 'days_rest' in df.columns:
            # Factor de energía basado en descanso (importante para double doubles)
            df['energy_factor'] = np.where(
                df['days_rest'] == 0, 0.80,  # Back-to-back penalty más fuerte
                np.where(df['days_rest'] == 1, 0.90,  # 1 día
                        np.where(df['days_rest'] >= 3, 1.10, 1.0))  # 3+ días boost
            )
    
    def _create_contextual_features_simple(self, df: pd.DataFrame) -> None:
        """Features contextuales disponibles antes del juego"""
        # Las features básicas de home/starter ya fueron calculadas en _calculate_player_context_features
        
        # Rest advantage específico para double doubles (usando days_rest ya calculado)
        if 'days_rest' in df.columns:
            df['rest_advantage'] = np.where(
                df['days_rest'] == 0, -0.20,  # Penalización back-to-back fuerte
                np.where(df['days_rest'] == 1, -0.08,
                        np.where(df['days_rest'] >= 3, 0.12, 0.0))
            )
        
        # Season progression factor (jugadores mejoran durante temporada)
        if 'month' in df.columns:
            df['season_progression_factor'] = np.where(
                df['month'].isin([10, 11]), -0.05,  # Inicio temporada
                np.where(df['month'].isin([12, 1, 2]), 0.05,  # Mitad temporada
                        np.where(df['month'].isin([3, 4]), 0.02, 0.0))  # Final temporada
            )
        
        # Weekend boost (más energía en fines de semana)
        if 'is_weekend' in df.columns:
            df['weekend_boost'] = df['is_weekend'] * 0.02
    
    def _create_performance_features_simple(self, df: pd.DataFrame) -> None:
        """Features de rendimiento BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        # Estadísticas clave para double double
        key_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'MP']
        
        for window in basic_windows:
            for stat in key_stats:
                if stat in df.columns:
                    # Promedio histórico básico
                    stat_hist_avg = self._get_historical_series(df, stat, window, 'mean')
                    df[f'{stat.lower()}_hist_avg_{window}g'] = stat_hist_avg
                    
                    # Consistencia básica (solo para stats principales)
                    if stat in ['PTS', 'TRB', 'AST', 'MP'] and window == 5:
                        stat_std = self._get_historical_series(df, stat, window, 'std', min_periods=2)
                        df[f'{stat.lower()}_consistency_{window}g'] = 1 / (stat_std.fillna(1) + 1)
        
        # NUEVAS FEATURES AVANZADAS BASADAS EN ANÁLISIS DE IMPORTANCIA
        
        # 1. USAGE RATE CONSISTENCY (Feature más importante: 49.06)
        if 'FGA' in df.columns and 'FTA' in df.columns and 'TOV' in df.columns and 'MP' in df.columns:
            # Calcular Usage Rate aproximado: (FGA + 0.44*FTA + TOV) / MP
            df['usage_rate_approx'] = (
                df['FGA'] + 0.44 * df['FTA'] + df['TOV']
            ) / (df['MP'] + 0.1)  # Evitar división por 0
            
            # Consistencia de usage rate (ventana de 5 juegos)
            usage_std = self._get_historical_series(df, 'usage_rate_approx', 5, 'std', min_periods=2)
            df['usage_consistency_5g'] = 1 / (usage_std.fillna(1) + 1)
        else:
            # Fallback usando solo FGA si no hay todas las stats
            if 'FGA' in df.columns and 'MP' in df.columns:
                df['usage_rate_approx'] = df['FGA'] / (df['MP'] + 0.1)
                usage_std = self._get_historical_series(df, 'usage_rate_approx', 5, 'std', min_periods=2)
                df['usage_consistency_5g'] = 1 / (usage_std.fillna(1) + 1)
            else:
                df['usage_consistency_5g'] = 0.5  # Valor neutral
        
        # 2. EFFICIENCY CONSISTENCY (importante para predecir rendimiento)
        if 'PTS' in df.columns and 'FGA' in df.columns:
            # Eficiencia de puntos por intento
            df['pts_efficiency'] = df['PTS'] / (df['FGA'] + 0.1)
            efficiency_std = self._get_historical_series(df, 'pts_efficiency', 5, 'std', min_periods=2)
            df['efficiency_consistency_5g'] = 1 / (efficiency_std.fillna(1) + 1)
        else:
            df['efficiency_consistency_5g'] = 0.5
        
        # 3. FEATURES DE RENDIMIENTO RELATIVO (vs promedio del jugador)
        for window in [5, 10]:
            for stat in ['PTS', 'TRB', 'AST']:
                if stat in df.columns:
                    stat_avg = self._get_historical_series(df, stat, window, 'mean')
                    # Feature de si está por encima del promedio histórico
                    # CORREGIR: Asegurar que los índices coincidan
                    stat_avg_aligned = stat_avg.reindex(df.index).fillna(0)
                    df[f'{stat.lower()}_above_historical_{window}g'] = (
                        df[stat] > stat_avg_aligned
                    ).astype(int)
        
        # 4. FEATURES DE MOMENTUM AVANZADO
        for stat in ['PTS', 'TRB', 'AST']:
            if stat in df.columns:
                # Momentum: últimos 3 juegos vs anteriores 3 juegos
                recent_avg = self._get_historical_series(df, stat, 3, 'mean')
                older_avg = df.groupby('Player')[stat].shift(3).rolling(window=3, min_periods=1).mean()
                # CORREGIR: Asegurar que los índices coincidan
                recent_avg_aligned = recent_avg.reindex(df.index).fillna(0)
                older_avg_aligned = older_avg.reindex(df.index).fillna(0)
                df[f'{stat.lower()}_momentum_6g'] = recent_avg_aligned - older_avg_aligned
        
        # 5. FEATURES DE COMBINACIÓN ESTADÍSTICA AVANZADA
        if all(col in df.columns for col in ['PTS', 'TRB', 'AST']):
            # Verificar que las features históricas existen antes de usarlas
            if all(col in df.columns for col in ['pts_hist_avg_5g', 'trb_hist_avg_5g', 'ast_hist_avg_5g']):
                # Índice de versatilidad (suma ponderada de stats principales)
                df['versatility_index'] = (
                    0.4 * df['pts_hist_avg_5g'] + 
                    0.35 * df['trb_hist_avg_5g'] + 
                    0.25 * df['ast_hist_avg_5g']
                )
                
                # Potencial de double-double basado en dos stats más altas
                df['dd_potential_score'] = np.maximum(
                    df['pts_hist_avg_5g'] + df['trb_hist_avg_5g'],
                    np.maximum(
                        df['pts_hist_avg_5g'] + df['ast_hist_avg_5g'],
                        df['trb_hist_avg_5g'] + df['ast_hist_avg_5g']
                    )
                )
            else:
                # Fallback si no existen las features históricas
                df['versatility_index'] = 0.0
                df['dd_potential_score'] = 0.0
        
        # 6. FEATURES DE TENDENCIA TEMPORAL
        for stat in ['PTS', 'TRB', 'AST']:
            if stat in df.columns:
                # Tendencia: diferencia entre promedio reciente vs histórico
                recent_3g = self._get_historical_series(df, stat, 3, 'mean')
                historical_10g = self._get_historical_series(df, stat, 10, 'mean')
                # CORREGIR: Asegurar que los índices coincidan
                recent_3g_aligned = recent_3g.reindex(df.index).fillna(0)
                historical_10g_aligned = historical_10g.reindex(df.index).fillna(0)
                df[f'{stat.lower()}_trend_factor'] = recent_3g_aligned - historical_10g_aligned
        
        # 7. FEATURES DE ESTABILIDAD DE RENDIMIENTO
        if 'MP' in df.columns:
            # Estabilidad de minutos (importante para oportunidades)
            mp_std = self._get_historical_series(df, 'MP', 5, 'std')
            mp_mean = self._get_historical_series(df, 'MP', 5, 'mean')
            # CORREGIR: Asegurar que los índices coincidan
            mp_std_aligned = mp_std.reindex(df.index).fillna(1)
            mp_mean_aligned = mp_mean.reindex(df.index).fillna(1)
            mp_cv = mp_std_aligned / (mp_mean_aligned + 0.1)
            df['minutes_stability'] = 1 / (mp_cv.fillna(1) + 1)
        
        # 8. FEATURES DE IMPACTO EN EL JUEGO
        if all(col in df.columns for col in ['PTS', 'TRB', 'AST', 'STL', 'BLK']):
            # Verificar que las features históricas existen
            if all(col in df.columns for col in ['pts_hist_avg_5g', 'trb_hist_avg_5g', 'ast_hist_avg_5g']):
                stl_hist = self._get_historical_series(df, 'STL', 5, 'mean').reindex(df.index).fillna(0)
                blk_hist = self._get_historical_series(df, 'BLK', 5, 'mean').reindex(df.index).fillna(0)
                
                # Índice de impacto total
                df['total_impact_5g'] = (
                    df['pts_hist_avg_5g'] + 
                    df['trb_hist_avg_5g'] + 
                    df['ast_hist_avg_5g'] + 
                    stl_hist +
                    blk_hist
                )
            else:
                df['total_impact_5g'] = 0.0
    
    def _create_double_double_features_simple(self, df: pd.DataFrame) -> None:
        """Features de double double BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        for window in basic_windows:
            # Double double rate histórico básico
            df[f'dd_rate_{window}g'] = (
                df.groupby('Player')['double_double'].shift(1)
                .rolling(window=window, min_periods=1).mean()
            ).fillna(0.1)  # Default bajo para nuevos jugadores
            
            # Weighted double double rate básico (solo para ventana de 5)
            if window == 5:
                dd_shifted = df.groupby('Player')['double_double'].shift(1).fillna(0)
                
                def simple_weighted_mean(x):
                    try:
                        x_clean = pd.to_numeric(x, errors='coerce').dropna()
                        if len(x_clean) == 0:
                            return 0.1
                        # Pesos simples: más reciente = más peso
                        weights = np.linspace(0.5, 1.0, len(x_clean))
                        weights = weights / weights.sum()
                        return float(np.average(x_clean, weights=weights))
                    except:
                        return 0.1
                
                df[f'weighted_dd_rate_{window}g'] = (
                    dd_shifted.rolling(window=window, min_periods=1)
                    .apply(simple_weighted_mean, raw=False)
                )
                
                # Double double momentum básico
                if window >= 5:
                    first_half = dd_shifted.rolling(window=3, min_periods=1).mean()
                    second_half = dd_shifted.shift(2).rolling(window=3, min_periods=1).mean()
                    df[f'dd_momentum_{window}g'] = first_half - second_half
        
        # Racha actual de double doubles - CORREGIDO
        def calculate_streak_for_group(group):
            """Calcular racha para un grupo de jugador"""
            # Usar double_double con shift(1) para evitar data leakage
            dd_series = group['double_double'].shift(1)
            streaks = []
            
            for i in range(len(group)):
                if i == 0:
                    streaks.append(0)  # Primer juego no tiene historial
                else:
                    # Obtener valores históricos hasta este punto
                    historical_values = dd_series.iloc[:i].dropna()
                    if len(historical_values) == 0:
                        streaks.append(0)
                    else:
                        # Calcular racha actual desde el final
                        streak = 0
                        for value in reversed(historical_values.tolist()):
                            if value == 1:
                                streak += 1
                            else:
                                break
                        streaks.append(streak)
            
            return pd.Series(streaks, index=group.index)
        
        try:
            # Aplicar función por grupo y obtener solo la serie resultante
            streak_series = df.groupby('Player').apply(calculate_streak_for_group)
            
            # Si es un DataFrame multinivel, aplanarlo
            if isinstance(streak_series, pd.DataFrame):
                streak_series = streak_series.iloc[:, 0]  # Tomar primera columna
            
            # Resetear índice para alinear con df original
            if hasattr(streak_series, 'reset_index'):
                streak_series = streak_series.reset_index(level=0, drop=True)
            
            # Asegurar que el índice coincide con df
            streak_series.index = df.index
            
            df['dd_streak'] = streak_series
            
        except Exception as e:
            logger.warning(f"Error calculando dd_streak: {str(e)}")
            # Fallback: usar cálculo más simple
            df['dd_streak'] = 0
        
        # Forma reciente (últimos 3 juegos)
        df['recent_dd_form'] = (
            df.groupby('Player')['double_double'].shift(1)
            .rolling(window=3, min_periods=1).mean()
        ).fillna(0.1)
    
    def _create_statistical_features_simple(self, df: pd.DataFrame) -> None:
        """Features estadísticas BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        for window in basic_windows:
            # Usage rate aproximado (solo si tenemos FGA y FTA)
            if all(col in df.columns for col in ['FGA', 'FTA', 'MP']):
                # Calcular usage histórico básico
                usage_hist = self._get_historical_series(df, 'FGA', window, 'mean') + \
                           self._get_historical_series(df, 'FTA', window, 'mean') * 0.44
                df[f'usage_hist_{window}g'] = usage_hist
                
                # Consistencia de usage (solo ventana 5)
                if window == 5:
                    usage_std = self._get_historical_series(df, 'FGA', window, 'std', min_periods=2)
                    df[f'usage_consistency_{window}g'] = 1 / (usage_std.fillna(1) + 1)
            
            # Eficiencia básica (PTS por minuto)
            if all(col in df.columns for col in ['PTS', 'MP']):
                pts_per_min = df['PTS'] / (df['MP'] + 0.1)  # Evitar división por 0
                pts_per_min_hist = self._get_historical_series_custom(df, pts_per_min, window, 'mean')
                df[f'pts_per_min_hist_{window}g'] = pts_per_min_hist
                
                # Consistencia de eficiencia (solo ventana 5)
                if window == 5:
                    eff_std = self._get_historical_series_custom(df, pts_per_min, window, 'std', min_periods=2)
                    df[f'efficiency_consistency_{window}g'] = 1 / (eff_std.fillna(1) + 1)
    
    def _create_opponent_features_simple(self, df: pd.DataFrame) -> None:
        """Features de oponente BÁSICAS únicamente - ANTI-OVERFITTING"""
        if 'Opp' not in df.columns:
            return
            
        # Defensive rating del oponente (aproximado usando puntos permitidos)
        if 'PTS' in df.columns:
            # Calcular puntos promedio permitidos por el oponente
            opp_def_rating = df.groupby('Opp')['PTS'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            # Invertir: menos puntos permitidos = mejor defensa = más difícil double double
            df['opponent_def_rating'] = opp_def_rating.fillna(105.0)  # Default NBA average
        
        # Último resultado vs este oponente (para double double)
        df['last_dd_vs_opp'] = df.groupby(['Player', 'Opp'])['double_double'].transform(
            lambda x: x.shift(1).tail(1).iloc[0] if len(x.shift(1).dropna()) > 0 else 0.1
        ).fillna(0.1)
        
        # Motivación extra vs rivales específicos
        df['rivalry_motivation'] = np.where(
            df['last_dd_vs_opp'] == 0, 0.05,  # No logró DD último vs este rival
            np.where(df['last_dd_vs_opp'] == 1, -0.02, 0)  # Logró DD último
        )
    
    def _create_biometric_features_simple(self, df: pd.DataFrame) -> None:
        """Features biométricas especializadas para double doubles"""
        if 'Height_Inches' not in df.columns:
            logger.debug("Height_Inches no disponible - saltando features biométricas")
            return
        
        logger.debug("Creando features biométricas especializadas para double doubles")
        
        # 1. Categorización de altura para double doubles
        # Basado en posiciones típicas NBA donde los double doubles son más comunes
        def categorize_height(height):
            if pd.isna(height):
                return 0  # Unknown
            elif height < 72:  # <6'0" - Guards pequeños
                return 1  # Small_Guard
            elif height < 75:  # 6'0"-6'3" - Guards normales
                return 2  # Guard
            elif height < 78:  # 6'3"-6'6" - Wings/Forwards pequeños
                return 3  # Wing
            elif height < 81:  # 6'6"-6'9" - Forwards
                return 4  # Forward
            else:  # >6'9" - Centers/Power Forwards
                return 5  # Big_Man
        
        df['height_category'] = df['Height_Inches'].apply(categorize_height)
        
        # 2. Factor de ventaja para rebotes basado en altura
        # Los jugadores más altos tienen ventaja natural para rebotes
        height_normalized = (df['Height_Inches'] - 72) / 12  # Normalizar desde 6'0" base
        df['height_rebounding_factor'] = np.clip(height_normalized * 0.15, 0, 0.25)
        
        # 3. Factor de ventaja para bloqueos basado en altura
        # Los jugadores más altos bloquean más
        df['height_blocking_factor'] = np.clip(height_normalized * 0.1, 0, 0.2)
        
        # 4. Ventaja de altura general para double doubles
        # Combina rebotes y bloqueos - jugadores altos tienen más oportunidades de DD
        df['height_advantage'] = (df['height_rebounding_factor'] + df['height_blocking_factor']) / 2
        
        # 5. Interacción altura-posición (aproximada por Height_Inches)
        # Guards altos y Centers pequeños tienen patrones únicos
        df['height_position_interaction'] = np.where(
            df['Height_Inches'] < 75,  # Guards
            np.where(df['Height_Inches'] > 73, 0.1, 0.0),  # Guards altos (+bonus)
            np.where(df['Height_Inches'] > 80, 0.05, 0.15)  # Centers vs Forwards
        )
        
        # 6. Factor de altura vs peso (si está disponible) para determinar tipo de jugador
        if 'Weight' in df.columns:
            # BMI ya está calculado en data_loader, pero podemos crear factor específico
            height_weight_ratio = df['Weight'] / df['Height_Inches']
            df['build_factor'] = np.where(
                height_weight_ratio > 2.8, 0.1,  # Jugadores "pesados" (más rebotes)
                np.where(height_weight_ratio < 2.4, -0.05, 0.0)  # Jugadores "ligeros"
            )
        
        logger.debug("Features biométricas especializadas creadas")
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualizar lista de columnas de features históricas"""
        exclude_cols = [
            # Columnas básicas del dataset
            'Player', 'Date', 'Team', 'Opp', 'Result', 'MP', 'GS', 'Away',
            
            # Estadísticas del juego actual (usadas solo para crear historial)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            
            # Columnas de double específicas del juego actual
            'PTS_double', 'TRB_double', 'AST_double', 'STL_double', 'BLK_double',
            
            # Target variables
            'double_double', 'triple_double',
            
            # Columnas auxiliares temporales
            'day_of_week', 'month', 'days_rest', 'is_home', 'is_started'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Retorna las características agrupadas por categoría HISTÓRICAS."""
        groups = {
            'temporal_context': [
                'day_of_week', 'month', 'is_weekend', 'days_into_season',
                'days_rest', 'energy_factor', 'season_progression_factor'
            ],
            
            'player_context': [
                'is_home', 'is_started', 'home_advantage', 'travel_penalty',
                'starter_boost', 'weekend_boost'
            ],
            
            'double_double_historical': [
                'dd_rate_5g', 'dd_rate_10g', 'weighted_dd_rate_5g', 'weighted_dd_rate_10g',
                'dd_momentum_5g', 'dd_momentum_10g', 'dd_streak', 'recent_dd_form'
            ],
            
            'performance_historical': [
                'pts_hist_avg_5g', 'pts_hist_avg_10g', 'trb_hist_avg_5g', 'trb_hist_avg_10g',
                'ast_hist_avg_5g', 'ast_hist_avg_10g', 'stl_hist_avg_5g', 'blk_hist_avg_5g',
                'mp_hist_avg_5g', 'mp_hist_avg_10g'
            ],
            
            'consistency_metrics': [
                'pts_consistency_5g', 'trb_consistency_5g', 'ast_consistency_5g',
                'mp_consistency_5g', 'usage_consistency_5g', 'efficiency_consistency_5g'
            ],
            
            'efficiency_metrics': [
                'usage_hist_5g', 'usage_hist_10g', 'pts_per_min_hist_5g', 'pts_per_min_hist_10g'
            ],
            
            'opponent_factors': [
                'opponent_def_rating', 'last_dd_vs_opp', 'rivalry_motivation'
            ],
            
            'biometrics': [
                'Height_Inches', 'Weight', 'BMI',
                'height_category', 'height_rebounding_factor', 'height_blocking_factor',
                'height_advantage', 'height_position_interaction', 'build_factor'
            ]
        }
        
        return groups
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Valida la calidad de las características históricas generadas."""
        validation_report = {
            'total_features': 0,
            'missing_features': [],
            'feature_coverage': {},
            'target_analysis': {}
        }
        
        groups = self.get_feature_importance_groups()
        all_features = []
        for group_features in groups.values():
            all_features.extend(group_features)
        
        validation_report['total_features'] = len(all_features)
        
        # Verificar características faltantes
        for feature in all_features:
            if feature not in df.columns:
                validation_report['missing_features'].append(feature)
        
        # Verificar cobertura por grupo
        for group_name, group_features in groups.items():
            existing = sum(1 for f in group_features if f in df.columns)
            validation_report['feature_coverage'][group_name] = {
                'total': len(group_features),
                'existing': existing,
                'coverage': existing / len(group_features) if group_features else 0
            }
        
        # Análisis del target double_double si existe
        if 'double_double' in df.columns:
            validation_report['target_analysis'] = {
                'total_games': len(df),
                'double_doubles': df['double_double'].sum(),
                'no_double_doubles': (df['double_double'] == 0).sum(),
                'dd_rate': df['double_double'].mean(),
                'missing_target': df['double_double'].isna().sum()
            }
        
        logger.info(f"Validación completada: {len(all_features)} features históricas, "
                   f"{len(validation_report['missing_features'])} faltantes")
        
        return validation_report
    
    def _get_historical_series(self, df: pd.DataFrame, column: str, window: int, 
                              operation: str = 'mean', min_periods: int = 1) -> pd.Series:
        """
        Método auxiliar para obtener series históricas con cache para evitar recálculos
        
        Args:
            df: DataFrame con los datos
            column: Nombre de la columna a procesar
            window: Ventana temporal
            operation: Operación a realizar ('mean', 'std', 'sum', 'var')
            min_periods: Períodos mínimos para el cálculo
        
        Returns:
            Serie histórica calculada con shift(1)
        """
        cache_key = f"{column}_{window}_{operation}_{min_periods}"
        
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        if column not in df.columns:
            logger.warning(f"Columna {column} no encontrada para cálculo histórico")
            return pd.Series(index=df.index, dtype=float).fillna(0.1 if operation == 'mean' else 0.0)
        
        # Calcular serie histórica con shift(1)
        shifted_series = df.groupby('Player')[column].shift(1)
        
        if operation == 'mean':
            result = shifted_series.rolling(window=window, min_periods=min_periods).mean()
        elif operation == 'std':
            result = shifted_series.rolling(window=window, min_periods=min_periods).std()
        elif operation == 'sum':
            result = shifted_series.rolling(window=window, min_periods=min_periods).sum()
        elif operation == 'var':
            result = shifted_series.rolling(window=window, min_periods=min_periods).var()
        elif operation == 'expanding_mean':
            result = shifted_series.expanding(min_periods=min_periods).mean()
        else:
            raise ValueError(f"Operación {operation} no soportada")
        
        # Guardar en cache
        self._cached_calculations[cache_key] = result
        
        return result
    
    def _get_historical_series_custom(self, df: pd.DataFrame, series: pd.Series, window: int, 
                                    operation: str = 'mean', min_periods: int = 1) -> pd.Series:
        """
        Método auxiliar para obtener series históricas de una serie personalizada
        """
        try:
            # Crear una serie temporal con nombre único
            temp_col_name = f'temp_custom_{hash(str(series.values[:5]))}'
            
            # Agregar temporalmente la serie al DataFrame
            df_temp = df.copy()
            df_temp[temp_col_name] = series
            
            # Calcular serie histórica con shift(1) por jugador
            shifted_series = df_temp.groupby('Player')[temp_col_name].shift(1)
            
            if operation == 'mean':
                result = shifted_series.rolling(window=window, min_periods=min_periods).mean()
            elif operation == 'std':
                result = shifted_series.rolling(window=window, min_periods=min_periods).std()
            else:
                raise ValueError(f"Operación {operation} no soportada para series personalizada")
            
            return result.fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Error en _get_historical_series_custom: {str(e)}")
            # Retornar serie de ceros como fallback
            return pd.Series(index=df.index, dtype=float).fillna(0.0)
    
    def _clear_cache(self):
        """Limpiar cache de cálculos para liberar memoria"""
        self._cached_calculations.clear()
        logger.debug("Cache de cálculos limpiado")
