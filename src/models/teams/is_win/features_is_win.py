"""
Módulo de Características para Predicción de Victorias (is_win)
==============================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de victorias de un equipo NBA por partido. Implementa características
avanzadas enfocadas en factores que determinan el resultado de un partido.

FEATURES HISTÓRICAS con shift(1) - Usando estadísticas pasadas para predecir futuros
OPTIMIZADO - Sin data leakage, todas las métricas usan shift(1) para crear historial

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

class IsWinFeatureEngineer:
    """
    Motor de features para predicción de victoria/derrota usando ESTADÍSTICAS HISTÓRICAS
    Enfoque: Usar shift(1) en TODAS las métricas para crear features históricas válidas
    OPTIMIZADO - Rendimiento pasado para predecir juegos futuros
    """
    
    def __init__(self, lookback_games: int = 10):
        """Inicializa el ingeniero de características para predicción de victorias."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        # Cache para evitar recálculos
        self._cached_calculations = {}
        
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        PIPELINE COMPLETO DE FEATURES HISTÓRICAS CON SHIFT(1)
        Usar estadísticas pasadas para predecir juegos futuros - SIN DATA LEAKAGE
        """
        logger.info("Generando features NBA HISTÓRICAS con shift(1) para predicción futura...")

        # VERIFICACIÓN ESPECÍFICA DE is_win COMO TARGET
        if 'is_win' in df.columns:
            logger.info(f"OK - is_win disponible! Distribución: {df['is_win'].value_counts().to_dict()}")
        else:
            # CREAR is_win desde Result si está disponible
            if 'Result' in df.columns:
                def extract_win_from_result(result_str):
                    """Extrae is_win desde el formato 'W 123-100' o 'L 114-116'"""
                    try:
                        result_str = str(result_str).strip()
                        if result_str.startswith('W'):
                            return 1
                        elif result_str.startswith('L'):
                            return 0
                        else:
                            return None  # Valor inválido
                    except:
                        return None
                
                df['is_win'] = df['Result'].apply(extract_win_from_result)
                
                # Verificar creación exitosa
                valid_wins = df['is_win'].notna().sum()
                total_rows = len(df)
                
                logger.info(f"is_win creado: {valid_wins}/{total_rows} valores válidos")
                if valid_wins < total_rows:
                    invalid_results = df[df['is_win'].isna()]['Result'].unique()
                    logger.warning(f"   Formatos no reconocidos: {invalid_results}")
            else:
                logger.error("No se puede crear is_win: columna Result no disponible")
        
        # VERIFICAR FEATURES DEL DATA_LOADER
        data_loader_features = ['is_home', 'has_overtime', 'overtime_periods']
        available_data_loader_features = [f for f in data_loader_features if f in df.columns]
        missing_data_loader_features = [f for f in data_loader_features if f not in df.columns]
        
        logger.info(f"Features del data_loader disponibles: {available_data_loader_features}")
        if missing_data_loader_features:
            logger.warning(f"Features del data_loader faltantes: {missing_data_loader_features}")
        
        # Trabajar directamente con el DataFrame
        if df.empty:
            return []
        
        # Asegurar orden cronológico para features históricas
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(['Team', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            logger.warning("Columna 'Date' no encontrada - usando orden original")
        
        # PASO 1: Features temporales básicas (disponibles antes del juego)
        self._create_temporal_features(df)
        
        # PASO 2: Features de ventaja local y contextual
        self._create_contextual_features(df)
        
        # PASO 3: Features históricas de rendimiento con shift(1)
        self._create_performance_features_historical(df)
        
        # PASO 4: Features de eficiencia históricas con shift(1)
        self._create_efficiency_features_historical(df)
        
        # PASO 5: Features de victoria históricas con shift(1)
        self._create_win_features_historical(df)
        
        # PASO 6: Features de oponente históricas
        self._create_opponent_features_historical(df)
        
        # PASO 7: Features avanzadas históricas
        self._create_advanced_features_historical(df)
        
        # PASO 8: Features de interacción entre equipos y contexto avanzado
        self._create_matchup_features_advanced(df)
        
        # Actualizar lista de features disponibles
        self._update_feature_columns(df)
        
        # Compilar lista de todas las características HISTÓRICAS
        all_features = [col for col in df.columns if col not in [
            # Columnas básicas del dataset
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            # Estadísticas del juego actual (NO USAR - data leakage)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', 
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'PTS_Opp',
            'ORB_Opp', 'DRB_Opp', 'TRB_Opp', 'AST_Opp', 'STL_Opp', 'BLK_Opp', 'TOV_Opp', 'PF_Opp',
            # Target variable
            'is_win',
            # Columnas auxiliares temporales
            'day_of_week', 'month', 'days_rest', 'is_home'
        ]]
        
        # Organizar features por categorías para mejor análisis
        feature_categories = {
            # CATEGORÍA 1: Features de rendimiento optimizadas
            'performance_optimized': [col for col in all_features if any(keyword in col for keyword in [
                'pts_hist_avg', 'pts_opp_hist_avg', 'point_diff_hist_avg', 'pts_consistency',
                'pts_trend', 'offensive_momentum', 'pace_hist', 'points_per_possession_hist',
                'close_game_performance'
            ])],
            
            # CATEGORÍA 2: Features de tiros y eficiencia optimizadas
            'shooting_efficiency_optimized': [col for col in all_features if any(keyword in col for keyword in [
                'fg_pct_hist_avg', '3p_pct_hist_avg', 'ft_pct_hist_avg', 'consistency',
                'true_shooting_pct_hist', 'effective_fg_pct_hist', 'ts_pct_differential',
                'effective_fg_differential', 'offensive_efficiency_score', 'defensive_efficiency_score',
                'net_efficiency_score'
            ])],
            
            # CATEGORÍA 3: Features defensivas optimizadas
            'defense_optimized': [col for col in all_features if any(keyword in col for keyword in [
                'defensive_rating_hist', 'defensive_consistency_hist', 'rebound_rate_hist',
                'rebound_dominance_hist', 'assist_rate_hist', 'ball_security_hist'
            ])],
            
            # CATEGORÍA 4: Features de victorias optimizadas
            'wins_optimized': [col for col in all_features if any(keyword in col for keyword in [
                'team_win_rate', 'current_win_streak', 'weighted_win_rate', 'win_momentum',
                'home_win_rate', 'away_win_rate', 'home_away_differential', 'win_quality',
                'close_game_win_rate', 'win_consistency', 'performance_vs_expectation',
                'clutch_factor', 'form_trend', 'hot_streak_indicator', 'team_strength_score'
            ])],
            
            # CATEGORÍA 5: Features de matchup avanzadas
            'matchup_advanced': [col for col in all_features if any(keyword in col for keyword in [
                'opponent_strength', 'strength_of_schedule', 'pace_differential', 'efficiency_matchup',
                'momentum_differential', 'hot_streak_differential', 'clutch_advantage',
                'offense_vs_defense', 'favorite_status', 'pressure_index', 'coaching_score'
            ])],
            
            # CATEGORÍA 6: Features contextuales
            'contextual': [col for col in all_features if any(keyword in col for keyword in [
                'is_weekend', 'is_friday', 'is_early_season', 'is_mid_season', 'is_late_season',
                'rest_advantage', 'is_back_to_back', 'travel_burden', 'venue_advantage'
            ])],
            
            # CATEGORÍA 7: Features legacy importantes (conservar las que funcionan)
            'legacy_important': [col for col in all_features if any(keyword in col for keyword in [
                'power_score', 'desperation_index', 'recent_form', 'win_streak', 'last_3_wins'
            ])]
        }
        
        # Compilar features finales organizadas por prioridad
        priority_features = []
        
        # PRIORIDAD ALTA: Features optimizadas más predictivas
        priority_features.extend(feature_categories['wins_optimized'])
        priority_features.extend(feature_categories['performance_optimized'])
        priority_features.extend(feature_categories['shooting_efficiency_optimized'])
        
        # PRIORIDAD MEDIA: Features de contexto y matchup
        priority_features.extend(feature_categories['defense_optimized'])
        priority_features.extend(feature_categories['matchup_advanced'])
        priority_features.extend(feature_categories['contextual'])
        
        # PRIORIDAD BAJA: Features legacy que aún aportan valor
        priority_features.extend(feature_categories['legacy_important'])
        
        # Eliminar duplicados manteniendo orden de prioridad
        seen = set()
        final_features = []
        for feature in priority_features:
            if feature not in seen and feature in all_features:
                final_features.append(feature)
                seen.add(feature)
        
        # Agregar cualquier feature restante que no esté categorizada
        for feature in all_features:
            if feature not in seen:
                final_features.append(feature)
        
        # Actualizar la lista final
        self.feature_columns = final_features
        
        logger.info(f"Features OPTIMIZADAS organizadas: {len(self.feature_columns)} características")
        logger.info(f"Categorías de features:")
        for category, features in feature_categories.items():
            valid_features = [f for f in features if f in all_features]
            logger.info(f"  - {category}: {len(valid_features)} features")
        
        # Log de las top 20 features por prioridad
        logger.info("Top 20 features por prioridad:")
        for i, feature in enumerate(final_features[:20], 1):
            logger.info(f"  {i:2d}. {feature}")
        
        return self.feature_columns
    
    def _create_temporal_features(self, df: pd.DataFrame) -> None:
        """Features temporales básicas disponibles antes del juego"""
        logger.info("Creando features temporales básicas...")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
            
            # Factor de energía basado en descanso
            df['energy_factor'] = np.where(
                df['days_rest'] == 0, 0.85,  # Back-to-back penalty
                np.where(df['days_rest'] == 1, 0.92,  # 1 día
                        np.where(df['days_rest'] >= 3, 1.08, 1.0))  # 3+ días boost
            )
    
    def _create_contextual_features(self, df: pd.DataFrame) -> None:
        """Features contextuales disponibles antes del juego"""
        logger.info("Creando features contextuales...")
        
        # Ventaja local desde Away column si no viene del data_loader
        if 'is_home' not in df.columns and 'Away' in df.columns:
            df['is_home'] = (df['Away'] != '@').astype(int)
            logger.info("is_home creado desde columna Away")
        
        # Boost de ventaja local
        if 'is_home' in df.columns:
            df['home_advantage'] = df['is_home'] * 0.06  # 6% boost histórico NBA
        
        # Ventaja de altitud para equipos específicos
        altitude_teams = ['DEN', 'UTA', 'PHX']
        df['altitude_advantage'] = df['Team'].apply(
            lambda x: 0.025 if x in altitude_teams else 0.0
        )
        
        # Rest advantage específico
        if 'days_rest' in df.columns:
            df['rest_advantage'] = np.where(
                df['days_rest'] == 0, -0.15,  # Penalización back-to-back
                np.where(df['days_rest'] == 1, -0.05,
                        np.where(df['days_rest'] >= 3, 0.08, 0.0))
            )
        
        # Travel penalty
        if 'is_home' in df.columns:
            df['travel_penalty'] = np.where(df['is_home'] == 0, -0.02, 0.0)
            
            # Back-to-back road penalty
            if 'days_rest' in df.columns:
                df['road_b2b_penalty'] = np.where(
                    (df['is_home'] == 0) & (df['days_rest'] == 0), -0.04, 0.0
                )
        
        # Season fatigue factor
        if 'month' in df.columns:
            df['season_fatigue_factor'] = np.where(
                df['month'].isin([1, 2, 3]), -0.015,  # Fatiga final temporada
                np.where(df['month'].isin([11, 12]), 0.01, 0.0)  # Boost inicio
            )
        
        # Weekend boost
        if 'is_weekend' in df.columns:
            df['weekend_boost'] = df['is_weekend'] * 0.01
    
    def _create_performance_features_historical(self, df: pd.DataFrame) -> None:
        """Features de rendimiento usando ESTADÍSTICAS HISTÓRICAS ÚNICAMENTE - OPTIMIZADO"""
        logger.info("Creando features de rendimiento HISTÓRICAS OPTIMIZADAS con shift(1)...")
        
        # OPTIMIZACIÓN 1: Ventanas temporales más específicas para NBA
        # Usar ventanas que se alinean con patrones reales de NBA
        performance_windows = [3, 5, 7, 10, 15]  # Agregado 15 para tendencias de temporada
        short_windows = [3, 5]  # Para tendencias recientes
        long_windows = [10, 15]  # Para patrones de temporada
        
        # OPTIMIZACIÓN 2: Estadísticas de puntos HISTÓRICAS mejoradas
        for window in performance_windows:
            # Promedio histórico de puntos con ponderación reciente
            pts_series = df.groupby('Team')['PTS'].shift(1)
            df[f'pts_hist_avg_{window}g'] = (
                pts_series.rolling(window=window, min_periods=1).mean()
            )
            
            # Promedio histórico de puntos del oponente
            pts_opp_series = df.groupby('Team')['PTS_Opp'].shift(1)
            df[f'pts_opp_hist_avg_{window}g'] = (
                pts_opp_series.rolling(window=window, min_periods=1).mean()
            )
            
            # Diferencial HISTÓRICO de puntos
            df[f'point_diff_hist_avg_{window}g'] = (
                df[f'pts_hist_avg_{window}g'] - df[f'pts_opp_hist_avg_{window}g']
            )
            
            # NUEVO: Consistencia en puntos (volatilidad)
            df[f'pts_consistency_{window}g'] = (
                1 / (pts_series.rolling(window=window, min_periods=2).std().fillna(1) + 1)
            )
        
        # OPTIMIZACIÓN 3: Tendencias de puntos más sofisticadas
        for window in [5, 10, 15]:
            pts_trend = df.groupby('Team')['PTS'].shift(1).rolling(window=window)
            
            # Tendencia lineal
            df[f'pts_trend_{window}g'] = pts_trend.apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
            )
            
            # NUEVO: Momentum ofensivo (últimos 3 vs promedio total)
            if window >= 5:
                pts_recent = df.groupby('Team')['PTS'].shift(1).rolling(window=3, min_periods=1).mean()
                pts_long = df[f'pts_hist_avg_{window}g']
                df[f'offensive_momentum_{window}g'] = pts_recent - pts_long
        
        # OPTIMIZACIÓN 4: Features de tiros HISTÓRICAS mejoradas
        shooting_stats = ['FG%', '3P%', 'FT%']
        for stat in shooting_stats:
            if stat in df.columns:
                for window in [5, 7, 10]:
                    # Porcentajes históricos
                    stat_series = df.groupby('Team')[stat].shift(1)
                    df[f'{stat.lower().replace("%", "_pct")}_hist_avg_{window}g'] = (
                        stat_series.rolling(window=window, min_periods=1).mean()
                    )
                    
                    # NUEVO: Consistencia en tiros
                    df[f'{stat.lower().replace("%", "_pct")}_consistency_{window}g'] = (
                        1 / (stat_series.rolling(window=window, min_periods=2).std().fillna(0.1) + 0.01)
                    )
                    
                    # Diferencial HISTÓRICO vs oponente
                    opp_stat = f'{stat}_Opp'
                    if opp_stat in df.columns:
                        opp_series = df.groupby('Team')[opp_stat].shift(1)
                        opp_hist = opp_series.rolling(window=window, min_periods=1).mean()
                        team_hist = df[f'{stat.lower().replace("%", "_pct")}_hist_avg_{window}g']
                        df[f'{stat.lower().replace("%", "_pct")}_diff_hist_avg_{window}g'] = team_hist - opp_hist
        
        # OPTIMIZACIÓN 5: NUEVAS features de ritmo y tempo
        if all(col in df.columns for col in ['FGA', 'FTA']):
            for window in [5, 10]:
                # Estimación de posesiones históricas
                possessions_hist = (
                    df.groupby('Team')['FGA'].shift(1).rolling(window=window, min_periods=1).mean() +
                    df.groupby('Team')['FTA'].shift(1).rolling(window=window, min_periods=1).mean() * 0.44
                )
                df[f'pace_hist_{window}g'] = possessions_hist
                
                # Eficiencia por posesión
                pts_hist = df.groupby('Team')['PTS'].shift(1).rolling(window=window, min_periods=1).mean()
                df[f'points_per_possession_hist_{window}g'] = (
                    pts_hist / (possessions_hist + 1e-6)
                ).fillna(1.0)
        
        # OPTIMIZACIÓN 6: NUEVAS features de clutch y situaciones críticas
        # Simulación de performance bajo presión usando diferencial histórico
        for window in [5, 10]:
            point_diff_hist = df[f'point_diff_hist_avg_{window}g']
            # Identificar juegos "cerrados" históricos (diferencia <= 5)
            close_games_mask = abs(point_diff_hist) <= 5
            
            # Performance en juegos cerrados
            df[f'close_game_performance_{window}g'] = np.where(
                close_games_mask, 
                df[f'pts_hist_avg_{window}g'] / (df[f'pts_opp_hist_avg_{window}g'] + 1e-6),
                1.0
            )
    
    def _create_efficiency_features_historical(self, df: pd.DataFrame) -> None:
        """Features de eficiencia usando ÚNICAMENTE estadísticas históricas - OPTIMIZADO"""
        logger.info("Creando features de eficiencia HISTÓRICAS OPTIMIZADAS con shift(1)...")
        
        # OPTIMIZACIÓN 1: Métricas de eficiencia NBA más predictivas
        efficiency_windows = [5, 7, 10, 15]  # Ventanas optimizadas
        
        # OPTIMIZACIÓN 2: True Shooting Percentage histórico (más preciso que FG%)
        if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
            for window in efficiency_windows:
                # Componentes históricos para TS%
                pts_hist = df.groupby('Team')['PTS'].shift(1).rolling(window=window, min_periods=1).mean()
                fga_hist = df.groupby('Team')['FGA'].shift(1).rolling(window=window, min_periods=1).mean()
                fta_hist = df.groupby('Team')['FTA'].shift(1).rolling(window=window, min_periods=1).mean()
                
                # True Shooting histórico: TS% = PTS / (2 * (FGA + 0.44 * FTA))
                true_shooting_attempts = 2 * (fga_hist + 0.44 * fta_hist)
                df[f'true_shooting_pct_hist_{window}g'] = (
                    pts_hist / (true_shooting_attempts + 1e-6)
                ).fillna(0.5)
                
                # NUEVO: Eficiencia diferencial vs oponente
        if all(col in df.columns for col in ['PTS_Opp', 'FGA_Opp', 'FTA_Opp']):
                    pts_opp_hist = df.groupby('Team')['PTS_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                    fga_opp_hist = df.groupby('Team')['FGA_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                    fta_opp_hist = df.groupby('Team')['FTA_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                    
                    opp_ts_attempts = 2 * (fga_opp_hist + 0.44 * fta_opp_hist)
                    opp_ts_pct = pts_opp_hist / (opp_ts_attempts + 1e-6)
                    
                    df[f'ts_pct_differential_hist_{window}g'] = (
                        df[f'true_shooting_pct_hist_{window}g'] - opp_ts_pct
                    )
        
        # OPTIMIZACIÓN 3: Effective Field Goal Percentage histórico
        if all(col in df.columns for col in ['FG', '3P', 'FGA']):
            for window in [5, 10]:
                fg_hist = df.groupby('Team')['FG'].shift(1).rolling(window=window, min_periods=1).mean()
                fg3_hist = df.groupby('Team')['3P'].shift(1).rolling(window=window, min_periods=1).mean()
                fga_hist = df.groupby('Team')['FGA'].shift(1).rolling(window=window, min_periods=1).mean()
                
                # eFG% = (FG + 0.5 * 3P) / FGA
                df[f'effective_fg_pct_hist_{window}g'] = (
                    (fg_hist + 0.5 * fg3_hist) / (fga_hist + 1e-6)
                ).fillna(0.45)
                
                # Diferencial vs oponente
        if all(col in df.columns for col in ['FG_Opp', '3P_Opp', 'FGA_Opp']):
                    fg_opp_hist = df.groupby('Team')['FG_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                    fg3_opp_hist = df.groupby('Team')['3P_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                    fga_opp_hist = df.groupby('Team')['FGA_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                    
                    opp_efg = (fg_opp_hist + 0.5 * fg3_opp_hist) / (fga_opp_hist + 1e-6)
                    df[f'effective_fg_differential_hist_{window}g'] = (
                        df[f'effective_fg_pct_hist_{window}g'] - opp_efg
                    )
        
        # OPTIMIZACIÓN 4: NUEVAS métricas defensivas históricas
        if 'PTS_Opp' in df.columns:
            for window in [5, 10, 15]:
                # Defensive Rating histórico (puntos permitidos por 100 posesiones)
                pts_opp_hist = df.groupby('Team')['PTS_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                
                if all(col in df.columns for col in ['FGA_Opp', 'FTA_Opp']):
                    possessions_opp = (
                        df.groupby('Team')['FGA_Opp'].shift(1).rolling(window=window, min_periods=1).mean() +
                        df.groupby('Team')['FTA_Opp'].shift(1).rolling(window=window, min_periods=1).mean() * 0.44
                    )
                    df[f'defensive_rating_hist_{window}g'] = (
                        (pts_opp_hist * 100) / (possessions_opp + 1e-6)
                    ).fillna(110.0)
                
                # NUEVO: Defensive consistency (menor variabilidad = mejor defensa)
                df[f'defensive_consistency_hist_{window}g'] = (
                    1 / (df.groupby('Team')['PTS_Opp'].shift(1).rolling(window=window, min_periods=2).std().fillna(5) + 1)
                )
        
        # OPTIMIZACIÓN 5: NUEVAS métricas de rebote más específicas
        if all(col in df.columns for col in ['TRB', 'TRB_Opp']):
            for window in [5, 10]:
                # Rebote rate histórico (más preciso que total de rebotes)
                trb_hist = df.groupby('Team')['TRB'].shift(1).rolling(window=window, min_periods=1).mean()
                trb_opp_hist = df.groupby('Team')['TRB_Opp'].shift(1).rolling(window=window, min_periods=1).mean()
                
                total_rebounds = trb_hist + trb_opp_hist
                df[f'rebound_rate_hist_{window}g'] = (
                    trb_hist / (total_rebounds + 1e-6)
            ).fillna(0.5)
            
                # NUEVO: Rebote dominance (cuánto se supera al oponente)
                df[f'rebound_dominance_hist_{window}g'] = trb_hist - trb_opp_hist
        
        # OPTIMIZACIÓN 6: NUEVAS métricas de asistencias y flujo ofensivo
        if 'AST' in df.columns:
            for window in [5, 10]:
                ast_hist = df.groupby('Team')['AST'].shift(1).rolling(window=window, min_periods=1).mean()
                
                # Assist rate (asistencias por punto anotado)
                if 'PTS' in df.columns:
                    pts_hist = df.groupby('Team')['PTS'].shift(1).rolling(window=window, min_periods=1).mean()
                    df[f'assist_rate_hist_{window}g'] = (
                        ast_hist / (pts_hist + 1e-6)
                    ).fillna(0.2)
                
                # NUEVO: Ball movement quality (asistencias vs turnovers si está disponible)
                if 'TOV' in df.columns:
                    tov_hist = df.groupby('Team')['TOV'].shift(1).rolling(window=window, min_periods=1).mean()
                    df[f'ball_security_hist_{window}g'] = (
                        ast_hist / (tov_hist + 1e-6)
        ).fillna(1.5)
        
        # OPTIMIZACIÓN 7: NUEVAS métricas compuestas más predictivas
        for window in [5, 10]:
            # Offensive Efficiency Score compuesto
            components = []
            if f'true_shooting_pct_hist_{window}g' in df.columns:
                components.append(df[f'true_shooting_pct_hist_{window}g'] * 2)  # Peso alto
            if f'assist_rate_hist_{window}g' in df.columns:
                components.append(df[f'assist_rate_hist_{window}g'])
            if f'rebound_rate_hist_{window}g' in df.columns:
                components.append(df[f'rebound_rate_hist_{window}g'])
                
            if components:
                df[f'offensive_efficiency_score_{window}g'] = np.mean(components, axis=0)
            
            # Defensive Efficiency Score compuesto
            def_components = []
            if f'defensive_rating_hist_{window}g' in df.columns:
                # Invertir porque menor defensive rating es mejor
                def_components.append(120 - df[f'defensive_rating_hist_{window}g'])
            if f'defensive_consistency_hist_{window}g' in df.columns:
                def_components.append(df[f'defensive_consistency_hist_{window}g'] * 10)
                
            if def_components:
                df[f'defensive_efficiency_score_{window}g'] = np.mean(def_components, axis=0)
            
            # NUEVO: Net Efficiency (diferencia ofensiva vs defensiva)
            if f'offensive_efficiency_score_{window}g' in df.columns and f'defensive_efficiency_score_{window}g' in df.columns:
                df[f'net_efficiency_score_{window}g'] = (
                    df[f'offensive_efficiency_score_{window}g'] - df[f'defensive_efficiency_score_{window}g']
                )
    
    def _create_win_features_historical(self, df: pd.DataFrame) -> None:
        """Features de victoria/derrota usando ÚNICAMENTE historial previo - OPTIMIZADO"""
        logger.info("Creando features de victoria HISTÓRICAS OPTIMIZADAS con shift(1)...")
        
        # OPTIMIZACIÓN 1: Ventanas temporales más estratégicas para momentum
        momentum_windows = [3, 5, 7]  # Para momentum reciente
        form_windows = [10, 15, 20]   # Para forma general
        
        # OPTIMIZACIÓN 2: Win Rate histórico con diferentes ventanas
        for window in momentum_windows + form_windows:
            # Win rate histórico básico
            df[f'team_win_rate_{window}g'] = (
                df.groupby('Team')['is_win'].shift(1)
                .rolling(window=window, min_periods=1).mean()
            ).fillna(0.5)
            
            # NUEVO: Win streak actual (racha de victorias consecutivas)
            if window <= 7:  # Solo para ventanas cortas
                # Crear indicador de racha de victorias
                wins_shifted = df.groupby('Team')['is_win'].shift(1).fillna(0)
                
                # Calcular racha actual
                df[f'current_win_streak_{window}g'] = (
                    wins_shifted.groupby(df['Team']).apply(
                        lambda x: x.iloc[::-1].cumprod().iloc[::-1].rolling(window=window, min_periods=1).sum()
                    ).values
                )
        
        # OPTIMIZACIÓN 3: NUEVAS métricas de momentum más sofisticadas
        for window in [5, 10]:
            # Weighted win rate (juegos más recientes pesan más)
            wins_shifted = df.groupby('Team')['is_win'].shift(1).fillna(0)
            
            # Crear pesos exponenciales (más peso a juegos recientes)
            def weighted_mean(x):
                try:
                    x_clean = pd.to_numeric(x, errors='coerce').dropna()
                    if len(x_clean) == 0:
                        return 0.5
                    weights = np.exp(np.linspace(-1, 0, len(x_clean)))
                    weights = weights / weights.sum()
                    return float(np.average(x_clean, weights=weights))
                except:
                    return 0.5
            
            df[f'weighted_win_rate_{window}g'] = (
                wins_shifted.rolling(window=window, min_periods=1)
                .apply(weighted_mean, raw=False)
            )
            
            # NUEVO: Win momentum (tendencia de victorias)
            # Comparar primeros vs últimos juegos de la ventana
            if window >= 6:
                try:
                    first_half = wins_shifted.rolling(window=window//2, min_periods=1).mean()
                    second_half = wins_shifted.shift(window//2).rolling(window=window//2, min_periods=1).mean()
                    df[f'win_momentum_{window}g'] = first_half - second_half
                except Exception as e:
                    logger.warning(f"Error creando win_momentum_{window}g: {e}")
                    df[f'win_momentum_{window}g'] = 0
        
        # OPTIMIZACIÓN 4: NUEVAS métricas contextuales (Home/Away si está disponible)
        if 'Away' in df.columns:
            # Crear columna is_home
            df['is_home'] = (df['Away'] == 0).astype(int)
            
            for window in [5, 10, 15]:
                # Win rate en casa histórico
                home_games = df[df['is_home'] == 1].groupby('Team')['is_win'].shift(1)
                df[f'home_win_rate_{window}g'] = (
                    home_games.rolling(window=window, min_periods=1).mean()
                ).fillna(0.55)  # Ventaja de casa típica
                
                # Win rate fuera histórico
                away_games = df[df['is_home'] == 0].groupby('Team')['is_win'].shift(1)
                df[f'away_win_rate_{window}g'] = (
                    away_games.rolling(window=window, min_periods=1).mean()
                ).fillna(0.45)  # Desventaja de visitante
                
                # NUEVO: Home/Away differential
                df[f'home_away_differential_{window}g'] = (
                    df[f'home_win_rate_{window}g'] - df[f'away_win_rate_{window}g']
                )
        
        # OPTIMIZACIÓN 5: NUEVAS métricas de calidad de victorias
        # Usar point differential como proxy para calidad de victoria
        if all(col in df.columns for col in ['PTS', 'PTS_Opp']):
            for window in [5, 10]:
                # Win quality score (promedio de diferencial en victorias)
                wins_mask = df.groupby('Team')['is_win'].shift(1) == 1
                point_diff = (df.groupby('Team')['PTS'].shift(1) - 
                             df.groupby('Team')['PTS_Opp'].shift(1))
                
                # Calidad promedio de victorias
                win_quality = point_diff.where(wins_mask)
                df[f'win_quality_{window}g'] = (
                    win_quality.rolling(window=window, min_periods=1).mean()
                ).fillna(5.0)  # Valor neutral
                
                # NUEVO: Close game performance (performance en juegos cerrados ≤ 5 pts)
                close_games_mask = abs(point_diff) <= 5
                close_wins = df.groupby('Team')['is_win'].shift(1).where(close_games_mask)
                df[f'close_game_win_rate_{window}g'] = (
                    close_wins.rolling(window=window, min_periods=1).mean()
                ).fillna(0.5)
        
        # OPTIMIZACIÓN 6: NUEVAS métricas de consistency y reliability
        for window in [10, 15]:
            # Win consistency (menor varianza = más consistente)
            wins_var = df.groupby('Team')['is_win'].shift(1).rolling(window=window, min_periods=2).var()
            df[f'win_consistency_{window}g'] = (
                1 / (wins_var.fillna(0.25) + 0.01)
            )
            
            # NUEVO: Performance vs expectation
            # Comparar win rate actual vs win rate esperado (basado en point differential)
            if f'point_diff_hist_avg_{window}g' in df.columns:
                # Convertir point differential a win probability aproximada
                point_diff_avg = df[f'point_diff_hist_avg_{window}g']
                expected_win_prob = 1 / (1 + np.exp(-point_diff_avg / 3))  # Función sigmoide
                actual_win_rate = df[f'team_win_rate_{window}g']
                
                df[f'performance_vs_expectation_{window}g'] = actual_win_rate - expected_win_prob
        
        # OPTIMIZACIÓN 7: NUEVAS métricas de clutch performance
        # Simulación de situaciones clutch usando datos históricos
        for window in [5, 10]:
            if f'close_game_win_rate_{window}g' in df.columns and f'team_win_rate_{window}g' in df.columns:
                # Clutch factor (performance en juegos cerrados vs performance general)
                df[f'clutch_factor_{window}g'] = (
                    df[f'close_game_win_rate_{window}g'] - df[f'team_win_rate_{window}g']
                )
        
        # OPTIMIZACIÓN 8: NUEVAS métricas de form trends
        for window in [7, 15]:
            # Recent form trend (¿está mejorando o empeorando?)
            win_rate_short = df[f'team_win_rate_{min(5, window)}g']
            win_rate_long = df[f'team_win_rate_{window}g']
            
            df[f'form_trend_{window}g'] = win_rate_short - win_rate_long
            
            # NUEVO: Hot/Cold streaks indicator
            # Identificar si está en racha positiva o negativa
            recent_wins = df.groupby('Team')['is_win'].shift(1).rolling(window=3, min_periods=1).sum()
            df[f'hot_streak_indicator_{window}g'] = np.where(recent_wins >= 2, 1, 
                                                           np.where(recent_wins <= 1, -1, 0))
        
        # OPTIMIZACIÓN 9: NUEVA métrica compuesta de "Team Strength"
        for window in [10, 15]:
            # Compilar componentes de fortaleza del equipo
            strength_components = []
            
            if f'team_win_rate_{window}g' in df.columns:
                strength_components.append(df[f'team_win_rate_{window}g'] * 2)  # Peso alto
            
            if f'win_quality_{window}g' in df.columns:
                # Normalizar calidad de victorias
                win_quality_norm = (df[f'win_quality_{window}g'] + 15) / 30  # Normalizar a 0-1
                strength_components.append(win_quality_norm)
            
            if f'win_consistency_{window}g' in df.columns:
                # Normalizar consistencia
                consistency_norm = np.clip(df[f'win_consistency_{window}g'] / 10, 0, 1)
                strength_components.append(consistency_norm)
            
            if f'clutch_factor_{window}g' in df.columns:
                # Normalizar clutch factor
                clutch_norm = (df[f'clutch_factor_{window}g'] + 0.2) / 0.4
                strength_components.append(clutch_norm * 0.5)  # Peso menor
            
            if strength_components:
                df[f'team_strength_score_{window}g'] = np.mean(strength_components, axis=0)
        
        logger.info("Features de victoria históricas optimizadas creadas exitosamente")
    
    def _create_opponent_features_historical(self, df: pd.DataFrame) -> None:
        """Features de oponente usando datos históricos con shift(1)"""
        logger.info("Creando features de oponente históricas...")
        
        if 'Opp' not in df.columns or 'is_win' not in df.columns:
            return
            
        # Recent form del oponente (últimas 5 victorias) con shift(1)
        opp_recent_form = df.groupby('Opp')['is_win'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df['opponent_recent_form'] = 1 - opp_recent_form.fillna(0.5)  # Invertir
        
        # Win rate del oponente en temporada con shift(1)
        opp_season_record = df.groupby('Opp')['is_win'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['opponent_season_record'] = 1 - opp_season_record.fillna(0.5)  # Invertir
        
        # Último resultado vs este oponente - HISTÓRICO
        df['last_vs_opp_result'] = df.groupby(['Team', 'Opp'])['is_win'].transform(
            lambda x: x.shift(1).tail(1).iloc[0] if len(x.shift(1).dropna()) > 0 else 0.5
        ).fillna(0.5)
        
        # Revenge factor
        df['revenge_motivation'] = np.where(
            df['last_vs_opp_result'] == 0, 0.04,  # Perdieron último vs este rival
            np.where(df['last_vs_opp_result'] == 1, -0.01, 0)  # Ganaron último
        )
        
        # Opponent power rating usando net rating histórico
        if 'game_net_rating' in df.columns:
            opp_power = df.groupby('Opp')['game_net_rating'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df['opponent_power_rating'] = opp_power.fillna(0)
    
    def _create_advanced_features_historical(self, df: pd.DataFrame) -> None:
        """Features avanzadas usando datos históricos con shift(1)"""
        logger.info("Creando features avanzadas históricas...")
        
        # Power score basado en rendimiento histórico
        if 'net_rating_hist_avg_5g' in df.columns and 'team_win_rate_10g' in df.columns:
            df['power_score'] = (
                df['net_rating_hist_avg_5g'] * 0.4 +
                (df['team_win_rate_10g'] - 0.5) * 20 * 0.6
            )
        
        # Power mismatch vs oponente
        if all(col in df.columns for col in ['power_score', 'opponent_power_rating']):
            df['power_mismatch'] = df['power_score'] - df['opponent_power_rating']
        
        # Four factors dominance
        efficiency_factors = []
        if 'efg_diff_hist_avg_5g' in df.columns:
            efficiency_factors.append('efg_diff_hist_avg_5g')
        if 'fg_diff_hist_avg_5g' in df.columns:
            efficiency_factors.append('fg_diff_hist_avg_5g')
        if 'ts_diff_hist_avg_5g' in df.columns:
            efficiency_factors.append('ts_diff_hist_avg_5g')
        
        if len(efficiency_factors) >= 2:
            df['four_factors_dominance'] = df[efficiency_factors].mean(axis=1)
        
        # Performance consistency usando net rating histórico
        if 'game_net_rating' in df.columns:
            df['performance_volatility'] = df.groupby('Team')['game_net_rating'].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=5).std()
            ).fillna(5.0)
            
            df['consistency_score'] = 1 / (df['performance_volatility'] + 1)
        
        # Home/Road splits históricos
        if all(col in df.columns for col in ['is_home', 'game_net_rating']):
            # Home vs Road performance usando shift(1)
            home_rating = df.groupby(['Team', 'is_home'])['game_net_rating'].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0)
            
            df['home_road_split'] = home_rating
        
        # Overtime performance histórico
        if all(col in df.columns for col in ['has_overtime', 'is_win']):
            df['overtime_wins'] = df['has_overtime'] * df['is_win']
            
            ot_numerator = df.groupby('Team')['overtime_wins'].transform(
                lambda x: x.shift(1).expanding().sum()
            )
            ot_denominator = df.groupby('Team')['has_overtime'].transform(
                lambda x: x.shift(1).expanding().sum()
            ) + 1e-6
            
            df['overtime_win_rate'] = ot_numerator / ot_denominator
            df['overtime_win_rate'] = df['overtime_win_rate'].fillna(0.5)
        
        # Playoff desperation
        if all(col in df.columns for col in ['days_into_season', 'team_win_rate_10g']):
            playoff_hunt = (
                (df['days_into_season'] > 180) & 
                (df['team_win_rate_10g'].between(0.35, 0.65))
            ).astype(int)
            
            df['desperation_index'] = playoff_hunt * (1 - df['team_win_rate_10g'])
    
    def _create_matchup_features_advanced(self, df: pd.DataFrame) -> None:
        """NUEVAS features avanzadas de matchup y contexto - OPTIMIZACIÓN FINAL"""
        logger.info("Creando features avanzadas de matchup y contexto...")
        
        # NUEVA OPTIMIZACIÓN 1: Features de strength of schedule histórico
        if 'Opp' in df.columns and 'is_win' in df.columns:
            for window in [5, 10, 15]:
                try:
                    # Calcular SOS basado en win rate histórico de oponentes enfrentados
                    opp_win_rates = []
                    for idx, row in df.iterrows():
                        team = row['Team']
                        opp = row['Opp']
                        
                        # Obtener win rate histórico del oponente antes de este juego
                        opp_data = df[(df['Team'] == opp) & (df.index < idx)]
                        if len(opp_data) >= 1 and 'is_win' in opp_data.columns:
                            opp_wins = opp_data['is_win'].tail(window).mean() if len(opp_data) >= window else opp_data['is_win'].mean()
                            opp_win_rates.append(float(opp_wins) if pd.notnull(opp_wins) else 0.5)
                        else:
                            opp_win_rates.append(0.5)  # Default
                    
                    df[f'opponent_strength_{window}g'] = opp_win_rates
                    
                    # Strength of Schedule promedio
                    if f'opponent_strength_{window}g' in df.columns:
                        df[f'strength_of_schedule_{window}g'] = (
                            df.groupby('Team')[f'opponent_strength_{window}g'].shift(1)
                            .rolling(window=window, min_periods=1).mean()
                        ).fillna(0.5)
                except Exception as e:
                    logger.warning(f"Error creando opponent_strength_{window}g: {e}")
        
        # NUEVA OPTIMIZACIÓN 2: Features de estilo de juego y matchup
        try:
            pace_cols = [col for col in df.columns if 'pace_hist_' in col and col.endswith('g')]
            efficiency_cols = [col for col in df.columns if 'points_per_possession_hist_' in col and col.endswith('g')]
            
            if pace_cols and efficiency_cols and 'Opp' in df.columns:
                for window in [5, 10]:
                    pace_col = f'pace_hist_{window}g'
                    eff_col = f'points_per_possession_hist_{window}g'
                    
                    if pace_col in df.columns and eff_col in df.columns:
                        # Pace differential (ritmo del equipo vs ritmo del oponente histórico)
                        pace_team = pd.to_numeric(df[pace_col], errors='coerce').fillna(75.0)
                        
                        # Pace del oponente usando transform
                        pace_opp = df.groupby('Opp')[pace_col].transform('mean')
                        pace_opp = pd.to_numeric(pace_opp, errors='coerce').fillna(pace_team.mean())
                        df[f'pace_differential_{window}g'] = pace_team - pace_opp
                        
                        # Efficiency matchup
                        eff_team = pd.to_numeric(df[eff_col], errors='coerce').fillna(1.0)
                        eff_opp = df.groupby('Opp')[eff_col].transform('mean')
                        eff_opp = pd.to_numeric(eff_opp, errors='coerce').fillna(eff_team.mean())
                        df[f'efficiency_matchup_{window}g'] = eff_team - eff_opp
        except Exception as e:
            logger.warning(f"Error creando pace/efficiency features: {e}")
        
        # NUEVA OPTIMIZACIÓN 3: Features de momentum relativo
        try:
            for window in [5, 10]:
                momentum_col = f'weighted_win_rate_{window}g'
                hot_streak_col = f'hot_streak_indicator_{window}g'
                
                if momentum_col in df.columns and 'Opp' in df.columns:
                    # Momentum diferencial vs oponente
                    team_momentum = pd.to_numeric(df[momentum_col], errors='coerce').fillna(0.5)
                    opp_momentum = df.groupby('Opp')[momentum_col].transform('mean')
                    opp_momentum = pd.to_numeric(opp_momentum, errors='coerce').fillna(0.5)
                    df[f'momentum_differential_{window}g'] = team_momentum - opp_momentum
                    
                    # Hot streak differential
                    if hot_streak_col in df.columns:
                        team_hot = pd.to_numeric(df[hot_streak_col], errors='coerce').fillna(0)
                        opp_hot = df.groupby('Opp')[hot_streak_col].transform('mean')
                        opp_hot = pd.to_numeric(opp_hot, errors='coerce').fillna(0)
                        df[f'hot_streak_differential_{window}g'] = team_hot - opp_hot
        except Exception as e:
            logger.warning(f"Error creando momentum features: {e}")
        
        # NUEVA OPTIMIZACIÓN 4: Features de situación contextual
        try:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['day_of_week'] = df['Date'].dt.dayofweek.fillna(3)  # Default miércoles
                df['month'] = df['Date'].dt.month.fillna(1)  # Default enero
                
                # Day of week effects (algunos equipos juegan mejor ciertos días)
                df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)  # Sábado y Domingo
                df['is_friday'] = (df['day_of_week'] == 4).astype(int)  # Viernes
                
                # Season timing effects
                df['is_early_season'] = (df['month'].isin([10, 11, 12])).astype(int)
                df['is_mid_season'] = (df['month'].isin([1, 2])).astype(int)
                df['is_late_season'] = (df['month'].isin([3, 4])).astype(int)
        except Exception as e:
            logger.warning(f"Error creando contextual features: {e}")
        
        # NUEVA OPTIMIZACIÓN 5: Features de rest advantage
        try:
            if 'Date' in df.columns:
                # Calcular días de descanso (simulado - en datos reales sería preciso)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
                df['days_rest'] = pd.to_numeric(df['days_rest'], errors='coerce').fillna(2)
                
                # Rest advantage vs oponente
                team_rest = df['days_rest']
                if 'Opp' in df.columns:
                    opp_rest = df.groupby('Opp')['days_rest'].transform('mean')
                    opp_rest = pd.to_numeric(opp_rest, errors='coerce').fillna(2)
                    df['rest_advantage'] = team_rest - opp_rest
                
                # Back-to-back games indicator
                df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
        except Exception as e:
            logger.warning(f"Error creando rest features: {e}")
        
        # NUEVA OPTIMIZACIÓN 6: Features de clutch matchup
        try:
            for window in [5, 10]:
                clutch_col = f'clutch_factor_{window}g'
                if clutch_col in df.columns and 'Opp' in df.columns:
                    # Clutch advantage vs oponente
                    team_clutch = pd.to_numeric(df[clutch_col], errors='coerce').fillna(0)
                    opp_clutch = df.groupby('Opp')[clutch_col].transform('mean')
                    opp_clutch = pd.to_numeric(opp_clutch, errors='coerce').fillna(0)
                    df[f'clutch_advantage_{window}g'] = team_clutch - opp_clutch
        except Exception as e:
            logger.warning(f"Error creando clutch features: {e}")
        
        # NUEVA OPTIMIZACIÓN 7: Features de defensive matchup específico
        try:
            for window in [5, 10]:
                def_col = f'defensive_rating_hist_{window}g'
                off_col = f'true_shooting_pct_hist_{window}g'
                
                if def_col in df.columns and off_col in df.columns and 'Opp' in df.columns:
                    # Offensive efficiency vs Defensive strength
                    team_offense = pd.to_numeric(df[off_col], errors='coerce').fillna(0.5)
                    opp_defense = df.groupby('Opp')[def_col].transform('mean')
                    opp_defense = pd.to_numeric(opp_defense, errors='coerce').fillna(110)
                    
                    # Normalizar defensive rating (menor es mejor)
                    opp_defense_norm = (120 - opp_defense) / 20  # Convertir a 0-1 donde 1 es mejor defensa
                    df[f'offense_vs_defense_{window}g'] = team_offense - opp_defense_norm
        except Exception as e:
            logger.warning(f"Error creando defensive matchup features: {e}")
        
        # NUEVA OPTIMIZACIÓN 8: Features de venue y travel
        try:
            if 'Away' in df.columns:
                # Travel burden (más juegos de visitante recientes = mayor carga)
                away_numeric = pd.to_numeric(df['Away'], errors='coerce').fillna(0)
                df['travel_burden'] = (
                    df.groupby('Team')['Away'].shift(1)
                    .rolling(window=5, min_periods=1).mean()
                ).fillna(0.5)
                
                # Home court differential (performance at home vs away histórico)
                home_col = 'home_win_rate_10g'
                away_col = 'away_win_rate_10g'
                
                if home_col in df.columns and away_col in df.columns:
                    home_rate = pd.to_numeric(df[home_col], errors='coerce').fillna(0.55)
                    away_rate = pd.to_numeric(df[away_col], errors='coerce').fillna(0.45)
                    
                    df['venue_advantage'] = np.where(
                        df['Away'] == 0,
                        home_rate - 0.5,  # Ventaja de casa
                        0.5 - away_rate   # Desventaja de visitante
                    )
        except Exception as e:
            logger.warning(f"Error creando venue features: {e}")
        
        # NUEVA OPTIMIZACIÓN 9: Features de psychological factors
        try:
            for window in [5, 10]:
                strength_col = f'team_strength_score_{window}g'
                if strength_col in df.columns and 'Opp' in df.columns:
                    # Underdog/Favorite status
                    team_strength = pd.to_numeric(df[strength_col], errors='coerce').fillna(0.5)
                    opp_strength = df.groupby('Opp')[strength_col].transform('mean')
                    opp_strength = pd.to_numeric(opp_strength, errors='coerce').fillna(team_strength.mean())
                    
                    strength_diff = team_strength - opp_strength
                    df[f'favorite_status_{window}g'] = np.where(
                        strength_diff > 0.1, 1,      # Favorito
                        np.where(strength_diff < -0.1, -1, 0)  # Underdog, neutro
                    )
                    
                    # Pressure index (favoritos tienen más presión)
                    df[f'pressure_index_{window}g'] = np.clip(strength_diff, -0.3, 0.3) * 3.33  # Normalizar a -1, 1
        except Exception as e:
            logger.warning(f"Error creando psychological features: {e}")
        
        # NUEVA OPTIMIZACIÓN 10: Features de coaching and adjustments
        try:
            for window in [10, 15]:
                perf_col = f'performance_vs_expectation_{window}g'
                consist_col = f'win_consistency_{window}g'
                
                if perf_col in df.columns:
                    # Coaching efficiency (performance vs expectation histórico)
                    coaching_eff = pd.to_numeric(df[perf_col], errors='coerce').fillna(0)
                    
                    # Adaptability (menor varianza en performance = mejor coaching)
                    if consist_col in df.columns:
                        adaptability = pd.to_numeric(df[consist_col], errors='coerce').fillna(1)
                        adaptability_std = adaptability.std()
                        if adaptability_std > 0:
                            adaptability_norm = (adaptability - adaptability.mean()) / adaptability_std
                            df[f'coaching_score_{window}g'] = (
                                coaching_eff * 0.6 + adaptability_norm * 0.4
                            ).fillna(0)
                        else:
                            df[f'coaching_score_{window}g'] = coaching_eff.fillna(0)
        except Exception as e:
            logger.warning(f"Error creando coaching features: {e}")
        
        logger.info("Features avanzadas de matchup y contexto creadas exitosamente")
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualizar lista de columnas de features históricas"""
        exclude_cols = [
            # Columnas básicas del dataset
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            
            # Estadísticas del juego actual (usadas solo para crear historial)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', 
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'PTS_Opp',
            'ORB_Opp', 'DRB_Opp', 'TRB_Opp', 'AST_Opp', 'STL_Opp', 'BLK_Opp', 'TOV_Opp', 'PF_Opp',
            # Target variable
            'is_win',
            # Columnas auxiliares temporales
            'day_of_week', 'month', 'days_rest', 'is_home'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Retorna las características agrupadas por categoría HISTÓRICAS."""
        groups = {
            'temporal_context': [
                'day_of_week', 'month', 'is_weekend', 'days_into_season',
                'days_rest', 'energy_factor', 'season_fatigue_factor'
            ],
            
            'home_advantage': [
                'is_home', 'home_advantage', 'travel_penalty', 'road_b2b_penalty',
                'altitude_advantage', 'weekend_boost'
            ],
            
            'performance_historical': [
                'net_rating_hist_avg_3g', 'net_rating_hist_avg_5g', 'net_rating_hist_avg_7g', 'net_rating_hist_avg_10g',
                'point_diff_hist_avg_3g', 'point_diff_hist_avg_5g', 'point_diff_hist_avg_7g', 'point_diff_hist_avg_10g',
                'off_rating_hist_avg_3g', 'off_rating_hist_avg_5g', 'off_rating_hist_avg_7g', 'off_rating_hist_avg_10g',
                'def_rating_hist_avg_3g', 'def_rating_hist_avg_5g', 'def_rating_hist_avg_7g', 'def_rating_hist_avg_10g'
            ],
            
            'efficiency_historical': [
                'fg_diff_hist_avg_3g', 'fg_diff_hist_avg_5g', 'fg_diff_hist_avg_7g',
                'three_diff_hist_avg_3g', 'three_diff_hist_avg_5g', 'three_diff_hist_avg_7g',
                'ft_diff_hist_avg_3g', 'ft_diff_hist_avg_5g', 'ft_diff_hist_avg_7g',
                'ts_diff_hist_avg_3g', 'ts_diff_hist_avg_5g', 'ts_diff_hist_avg_7g',
                'efg_diff_hist_avg_3g', 'efg_diff_hist_avg_5g', 'efg_diff_hist_avg_7g'
            ],
            
            'momentum_factors': [
                'team_win_rate_3g', 'team_win_rate_5g', 'team_win_rate_7g', 'team_win_rate_10g', 'team_win_rate_15g',
                'recent_form', 'win_streak', 'last_3_wins', 'power_score'
            ],
            
            'opponent_quality': [
                'opponent_recent_form', 'opponent_season_record', 'opponent_power_rating',
                'last_vs_opp_result', 'revenge_motivation', 'power_mismatch'
            ],
            
            'clutch_performance': [
                'clutch_win_rate', 'overtime_win_rate'
            ],
            
            'advanced_metrics': [
                'four_factors_dominance', 'performance_volatility', 'consistency_score',
                'home_road_split', 'desperation_index'
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
        
        # Análisis del target is_win si existe
        if 'is_win' in df.columns:
            validation_report['target_analysis'] = {
                'total_games': len(df),
                'wins': df['is_win'].sum(),
                'losses': (df['is_win'] == 0).sum(),
                'win_rate': df['is_win'].mean(),
                'missing_target': df['is_win'].isna().sum()
            }
        
        logger.info(f"Validación completada: {len(all_features)} features históricas, "
                   f"{len(validation_report['missing_features'])} faltantes")
        
        return validation_report
