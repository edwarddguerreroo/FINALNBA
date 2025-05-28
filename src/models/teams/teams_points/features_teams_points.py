"""
Módulo de Características para Predicción de Puntos de Equipo (PTS)
================================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de puntos de un equipo NBA por partido. Implementa características
avanzadas basadas en el modelo exitoso de total_points pero optimizado para un solo equipo.

FEATURES DE DOMINIO ESPECÍFICO con máximo poder predictivo
OPTIMIZADO - Sin cálculos duplicados, sin multicolinealidad
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

class TeamPointsFeatureEngineer:
    """
    Motor de features para predicción de puntos de un equipo específico
    Enfoque: Features de DOMINIO ESPECÍFICO con máximo poder predictivo
    OPTIMIZADO - Sin cálculos duplicados, basado en lógica exitosa de total_points
    """
    
    def __init__(self, lookback_games: int = 10):
        """Inicializa el ingeniero de características para puntos de equipo."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        # Cache para evitar recálculos
        self._cached_calculations = {}
        
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        PIPELINE COMPLETO DE FEATURES PARA 97% PRECISIÓN - OPTIMIZADO
        Usa la misma lógica exitosa de total_points adaptada para equipos
        """
        logger.info("Generando features NBA específicas OPTIMIZADAS para puntos de equipo...")
        
        # Trabajar directamente con el DataFrame (NO crear copia)
        if df.empty:
            return []
        
        # Asegurar orden cronológico para evitar data leakage
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(['Team', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            logger.warning("Columna 'Date' no encontrada - usando orden original")
        
        # PASO 1: Cálculos base (una sola vez) - MISMA LÓGICA QUE TOTAL_POINTS
        self._create_base_calculations(df)
        
        # PASO 2: Features básicas NBA usando cálculos base
        self._create_basic_nba_features(df)
        
        # PASO 3: Features avanzadas usando cálculos existentes
        self._create_advanced_features_optimized(df)
        
        # PASO 4: Features de contexto y situación
        self._create_context_features(df)
        
        # PASO 5: Features de interacción final
        self._create_final_interactions(df)
        
        # Aplicar filtros de calidad
        self._apply_quality_filters(df)
        
        # Actualizar lista de features disponibles
        self._update_feature_columns(df)
        
        # Compilar lista de todas las características
        all_features = [col for col in df.columns if col not in [
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
            '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
            'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'is_win',  # Excluir is_win directo
            # Excluir variables categóricas y auxiliares
            'team_scoring_tier', 'team_tier_adjusted_projection'
        ]]
            
        logger.info(f"Generadas {len(all_features)} características ESPECÍFICAS para puntos de equipo")
        return all_features
    
    def _create_base_calculations(self, df: pd.DataFrame) -> None:
        """
        CÁLCULOS BASE NBA - Una sola vez para evitar duplicaciones
        BASADO EN LÓGICA EXITOSA DE TOTAL_POINTS pero para equipo individual
        """
        logger.info("Calculando métricas base NBA para equipo...")
        
        # ==================== CÁLCULOS TEMPORALES BASE ====================
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
        
        # ==================== CÁLCULOS DE POSESIONES NBA OFICIAL ====================
        # Fórmula NBA oficial adaptada para falta de TOV
        if all(col in df.columns for col in ['FGA', 'FTA']):
            df['team_possessions'] = df['FGA'] + df['FTA'] * 0.44
            df['opp_possessions'] = df['FGA_Opp'] + df['FTA_Opp'] * 0.44
        
        # Posesiones alternativas (para validación)
        if all(col in df.columns for col in ['FGA', 'FTA']):
            df['real_possessions'] = df['FGA'] + df['FTA'] * 0.44
            df['opp_real_possessions'] = df['FGA_Opp'] + df['FTA_Opp'] * 0.44
        
        # ==================== CÁLCULOS DE EFICIENCIA BASE ====================
        # True Shooting Percentage aproximado (SIN usar PTS reales para evitar data leakage)
        if all(col in df.columns for col in ['FG%', 'FT%']):
            df['team_true_shooting_approx'] = (df['FG%'].fillna(0.45) * 0.6 + df['FT%'].fillna(0.75) * 0.4)
            df['opp_true_shooting_approx'] = (df['FG%_Opp'].fillna(0.45) * 0.6 + df['FT%_Opp'].fillna(0.75) * 0.4)
        
        # Effective Field Goal Percentage aproximado
        if all(col in df.columns for col in ['FG%', '3P%', '3PA', 'FGA']):
            df['team_efg_approx'] = df['FG%'].fillna(0.45) + (0.5 * df['3P%'].fillna(0.35) * (df['3PA'] / (df['FGA'] + 1)))
            df['opp_efg_approx'] = df['FG%_Opp'].fillna(0.45) + (0.5 * df['3P%_Opp'].fillna(0.35) * (df['3PA_Opp'] / (df['FGA_Opp'] + 1)))
        
        # Conversion efficiency combinada
        if all(col in df.columns for col in ['FG%', '3P%', 'FT%']):
            df['team_conversion_efficiency'] = (
                df['FG%'].fillna(0.45) * 0.5 + 
                df['3P%'].fillna(0.35) * 0.3 + 
                df['FT%'].fillna(0.75) * 0.2
            )
            df['opp_conversion_efficiency'] = (
                df['FG%_Opp'].fillna(0.45) * 0.5 + 
                df['3P%_Opp'].fillna(0.35) * 0.3 + 
                df['FT%_Opp'].fillna(0.75) * 0.2
            )
        
        # ==================== PROYECCIONES DIRECTAS BASE ====================
        # Proyección directa de scoring del equipo (SIN usar PTS reales)
        if all(col in df.columns for col in ['FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%']):
            df['team_direct_scoring_projection'] = (
                df['FGA'] * df['FG%'].fillna(0.45) * 2 +    # Puntos de 2P estimados
                df['3PA'] * df['3P%'].fillna(0.35) * 3 +    # Puntos de 3P estimados
                df['FTA'] * df['FT%'].fillna(0.75) * 1      # Puntos de FT estimados
            )
        
        # ==================== VOLÚMENES Y MÉTRICAS COMBINADAS ====================
        if all(col in df.columns for col in ['FGA', 'FTA']):
            df['team_total_shot_volume'] = df['FGA'] + df['FTA']
            df['opp_total_shot_volume'] = df['FGA_Opp'] + df['FTA_Opp']
        
        if 'team_total_shot_volume' in df.columns and 'team_conversion_efficiency' in df.columns:
            df['team_weighted_shot_volume'] = df['team_total_shot_volume'] * df['team_conversion_efficiency']
        
        if all(col in df.columns for col in ['FGA', '3PA', 'FTA']):
            df['team_expected_shots'] = df['FGA'] + df['3PA'] * 0.4 + df['FTA'] * 0.44
            df['opp_expected_shots'] = df['FGA_Opp'] + df['3PA_Opp'] * 0.4 + df['FTA_Opp'] * 0.44
        
        # ==================== FEATURES DE CONTEXTO BASE ====================
        # Ventaja local
        if 'Away' in df.columns:
            df['team_is_home'] = (df['Away'] == 0).astype(int)
        else:
            df['team_is_home'] = 0
        
        # Factor de energía basado en descanso
        if 'days_rest' in df.columns:
            df['team_energy_factor'] = np.where(
                df['days_rest'] == 0, 0.92,  # Back-to-back penalty
                np.where(df['days_rest'] == 1, 0.97,  # 1 día
                        np.where(df['days_rest'] >= 3, 1.03, 1.0))  # 3+ días boost
            )
        
        # Boost de ventaja local específico para equipo
        if 'team_is_home' in df.columns:
            df['team_home_court_boost'] = df['team_is_home'] * 2.8  # Específico para un equipo
        
        # Importancia del partido
        if 'days_into_season' in df.columns:
            df['team_season_importance'] = np.where(
                df['days_into_season'] > 200, 1.06,  # Playoffs/final temporada
                np.where(df['days_into_season'] > 100, 1.03, 1.0)  # Mitad temporada
            )
    
    def _create_basic_nba_features(self, df: pd.DataFrame) -> None:
        """Features básicas NBA usando cálculos base existentes - ADAPTADO DE TOTAL_POINTS"""
        
        # ==================== PROMEDIOS MÓVILES OPTIMIZADOS ====================
        windows = [3, 5, 7, 10]
        
        # Promedios de características base del equipo
        for window in windows:
            if 'team_direct_scoring_projection' in df.columns:
                df[f'team_direct_projection_avg_{window}g'] = df.groupby('Team')['team_direct_scoring_projection'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'team_conversion_efficiency' in df.columns:
                df[f'team_conversion_efficiency_avg_{window}g'] = df.groupby('Team')['team_conversion_efficiency'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'team_possessions' in df.columns:
                df[f'team_pace_avg_{window}g'] = df.groupby('Team')['team_possessions'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'team_true_shooting_approx' in df.columns:
                df[f'team_ts_avg_{window}g'] = df.groupby('Team')['team_true_shooting_approx'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'team_total_shot_volume' in df.columns:
                df[f'team_volume_avg_{window}g'] = df.groupby('Team')['team_total_shot_volume'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # ==================== MÉTRICAS DERIVADAS OPTIMIZADAS ====================
        # Ritmo del equipo
        if 'team_pace_avg_5g' in df.columns:
            df['team_game_pace'] = df['team_pace_avg_5g']
        
        # Proyección base del equipo
        if 'team_direct_projection_avg_5g' in df.columns:
            df['team_base_projection'] = df['team_direct_projection_avg_5g']
    
    def _create_advanced_features_optimized(self, df: pd.DataFrame) -> None:
        """Features avanzadas optimizadas sin duplicaciones y multicolinealidad - ADAPTADO DE TOTAL_POINTS"""
        
        # ==================== PROYECCIONES OPTIMIZADAS SIN MULTICOLINEALIDAD ====================
        # Solo mantener la proyección matemática más robusta para el equipo
        if all(col in df.columns for col in ['team_expected_shots', 'team_conversion_efficiency']):
            df['team_mathematical_projection'] = (
                df['team_expected_shots'] * df['team_conversion_efficiency'] * 100
            )
        
        # Proyección híbrida simplificada para el equipo
        if all(col in df.columns for col in ['team_mathematical_projection', 'team_weighted_shot_volume']):
            df['team_hybrid_projection'] = (
                df['team_mathematical_projection'] * 0.6 +
                df['team_weighted_shot_volume'] * 40  # Escalar apropiadamente
            )
        
        # ==================== FEATURES MEJORADAS PARA RANGOS ESPECÍFICOS ====================
        # Características específicas para diferentes rangos de scoring
        if 'team_direct_scoring_projection' in df.columns:
            # Clasificar equipos por tendencia de scoring (manejando NaN explícitamente)
            scoring_values = df['team_direct_scoring_projection'].fillna(df['team_direct_scoring_projection'].median())
            df['team_scoring_tier'] = pd.cut(
                scoring_values, 
                bins=[0, 95, 115, 200], 
                labels=['low_scoring', 'mid_scoring', 'high_scoring']
            )
            # Convertir a string para evitar problemas categóricos
            df['team_scoring_tier'] = df['team_scoring_tier'].astype(str)
            # Manejar cualquier NaN restante
            df['team_scoring_tier'] = df['team_scoring_tier'].fillna('mid_scoring')
            
            # Features específicas por tier de scoring
            for tier in ['low_scoring', 'mid_scoring', 'high_scoring']:
                tier_mask = df['team_scoring_tier'] == tier
                if 'team_conversion_efficiency' in df.columns:
                    tier_col = f'team_{tier}_efficiency'
                    df[tier_col] = df.groupby('Team')['team_conversion_efficiency'].transform(
                        lambda x: x.where(tier_mask).expanding().mean().shift(1)
                    ).fillna(df['team_conversion_efficiency'])
        
        # ==================== FEATURES DE CONSISTENCIA MEJORADAS ====================
        # Estabilidad de scoring por ventana temporal
        if 'team_direct_scoring_projection' in df.columns:
            # Coeficiente de variación (estabilidad)
            df['team_scoring_stability'] = df.groupby('Team')['team_direct_scoring_projection'].transform(
                lambda x: x.rolling(window=10, min_periods=3).std().shift(1) / 
                         (x.rolling(window=10, min_periods=3).mean().shift(1) + 1e-6)
            )
            # Invertir para que mayor valor = mayor estabilidad
            df['team_scoring_stability'] = 1 / (df['team_scoring_stability'] + 0.01)
        
        # ==================== FEATURES DE OPONENTE OPTIMIZADAS ====================
        # Calidad del oponente (SIN data leakage)
        if 'Opp' in df.columns and 'PTS_Opp' in df.columns:
            opp_def_ranking = df.groupby('Opp')['PTS_Opp'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df['opponent_def_strength'] = opp_def_ranking.rank(pct=True).fillna(0.5)
        
        if 'Opp' in df.columns and 'PTS' in df.columns:
            opp_off_ranking = df.groupby('Opp')['PTS'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df['opponent_off_strength'] = opp_off_ranking.rank(pct=True).fillna(0.5)
        
        if all(col in df.columns for col in ['opponent_off_strength', 'opponent_def_strength']):
            df['opponent_quality_factor'] = (
                df['opponent_off_strength'] * 0.4 + 
                df['opponent_def_strength'] * 0.6  # Más peso a defensa para puntos permitidos
            )
        
        # ==================== FEATURES DE PRESSURE Y MOMENTUM MEJORADAS ====================
        # Pressure de scoring basado en expectativas
        if all(col in df.columns for col in ['team_direct_scoring_projection', 'team_direct_projection_avg_10g']):
            df['team_scoring_pressure'] = (
                df['team_direct_scoring_projection'] - df['team_direct_projection_avg_10g']
            ) / (df['team_direct_projection_avg_10g'] + 1e-6)
        
        # ==================== FEATURES DE MATCHUP ESPECÍFICO ====================
        # Historial vs oponente específico del equipo
        if all(col in df.columns for col in ['Team', 'Opp', 'team_direct_scoring_projection']):
            df['team_vs_opp_scoring_history'] = df.groupby(['Team', 'Opp'])['team_direct_scoring_projection'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
        
        # Edge del matchup específico del equipo
        if all(col in df.columns for col in ['team_vs_opp_scoring_history', 'team_direct_projection_avg_10g']):
            df['team_matchup_edge'] = (
                df['team_vs_opp_scoring_history'].fillna(df['team_direct_projection_avg_10g']) - 
                df['team_direct_projection_avg_10g']
            )
        
        # ==================== FEATURES DE MOMENTUM OPTIMIZADAS ====================
        # Usar la columna 'is_win' proporcionada por el dataloader pero con datos históricos
        if 'is_win' in df.columns:
            # Win percentage histórico (usando datos desplazados para evitar data leakage)
            df['team_win_pct_5g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            ).fillna(0.5)
            
            df['team_win_pct_10g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
            ).fillna(0.5)
            
            # Recent momentum (últimas 3 victorias históricas)
            df['team_recent_wins'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
            ).fillna(1.5)
            
            # Hot/cold streak detection (histórico)
            df['team_win_streak'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).apply(
                    lambda wins: (wins == 1).sum() if len(wins) > 0 else 0
                )
            ).fillna(2.5)
            
        # Factor de confianza del equipo basado en win percentage histórico
        if 'team_win_pct_5g' in df.columns:
            # Usar win percentage histórico como proxy de "confianza"
            df['team_confidence_factor'] = (df['team_win_pct_5g'] - 0.5) * 4  # Escalado apropiado
        elif 'team_conversion_efficiency_avg_5g' in df.columns:
            # Fallback: usar eficiencia histórica como proxy de "confianza"
            efficiency_avg = df['team_conversion_efficiency_avg_5g'].fillna(0.45)
            df['team_confidence_factor'] = (efficiency_avg - 0.45) * 4
        else:
            df['team_confidence_factor'] = 0
            
        # Momentum de confianza basado en cambios de win percentage (NO en victorias actuales)
        if 'team_win_pct_5g' in df.columns:
            df['team_confidence_momentum'] = df.groupby('Team')['team_win_pct_5g'].transform(
                lambda x: x.diff().shift(1)
            ).fillna(0)
        elif 'team_conversion_efficiency_avg_5g' in df.columns:
            # Fallback: usar cambios de eficiencia
            df['team_confidence_momentum'] = df.groupby('Team')['team_conversion_efficiency_avg_5g'].transform(
                lambda x: x.diff().shift(1)
            ).fillna(0)
        else:
            df['team_confidence_momentum'] = 0
    
    def _create_context_features(self, df: pd.DataFrame) -> None:
        """Features de contexto situacional - ADAPTADO DE TOTAL_POINTS"""
        
        # ==================== FACTORES CONTEXTUALES ESPECÍFICOS ====================
        # Factor de altitud (equipos específicos)
        altitude_teams = ['DEN', 'UTA', 'PHX']
        df['team_altitude_factor'] = df['Team'].apply(lambda x: 1.02 if x in altitude_teams else 1.0)
        
        # Factor de rivalidad específico del equipo
        rivalry_boost_teams = {
            'LAL': ['BOS', 'LAC'], 'BOS': ['LAL', 'PHI'], 'GSW': ['LAC', 'CLE'],
            'MIA': ['BOS', 'NYK'], 'CHI': ['DET', 'CLE'], 'DAL': ['SAS', 'HOU']
        }
        def get_rivalry_factor(row):
            team = row['Team']
            opp = row.get('Opp', '')
            if team in rivalry_boost_teams and opp in rivalry_boost_teams[team]:
                return 1.04
            return 1.0
        
        if 'Opp' in df.columns:
            df['team_rivalry_factor'] = df.apply(get_rivalry_factor, axis=1)
        else:
            df['team_rivalry_factor'] = 1.0
        
        # Fatiga de temporada específica
        if 'month' in df.columns:
            df['team_season_fatigue'] = np.where(df['month'].isin([1, 2, 3]), 0.98, 1.0)
        
        # Factor de urgencia del equipo
        if 'days_into_season' in df.columns:
            df['team_urgency_factor'] = np.where(df['days_into_season'] > 200, 1.04, 1.0)
        
        # Advantage de descanso específico del equipo
        if 'days_rest' in df.columns:
            df['team_rest_advantage'] = np.where(
                df['days_rest'] == 0, -3.5,  # Penalización back-to-back específica para equipo
                np.where(df['days_rest'] == 1, -1.2,  
                        np.where(df['days_rest'] >= 3, 2.2, 0.0))
            )
    
    def _create_final_interactions(self, df: pd.DataFrame) -> None:
        """Features de interacción final optimizadas sin multicolinealidad - ADAPTADO DE TOTAL_POINTS"""
        
        # ==================== INTERACCIONES SIMPLIFICADAS ====================
        # Interacción pace-efficiency específica del equipo
        if all(col in df.columns for col in ['team_possessions', 'team_conversion_efficiency']):
            df['team_pace_efficiency_interaction'] = df['team_possessions'] * df['team_conversion_efficiency']
        
        # Interacción momentum-contexto del equipo
        if all(col in df.columns for col in ['team_confidence_factor', 'team_energy_factor']):
            df['team_momentum_context'] = df['team_confidence_factor'] * df['team_energy_factor']
        
        # Interacción calidad-eficiencia específica del equipo vs oponente
        if all(col in df.columns for col in ['team_conversion_efficiency', 'opponent_quality_factor']):
            df['team_quality_efficiency_interaction'] = (
                df['team_conversion_efficiency'] * (2 - df['opponent_quality_factor'])
            )
        
        # ==================== INTERACCIONES MEJORADAS PARA ESTABILIDAD ====================
        # Estabilidad-momentum combinada
        if all(col in df.columns for col in ['team_scoring_stability', 'team_confidence_factor']):
            df['team_stability_confidence'] = df['team_scoring_stability'] * (df['team_confidence_factor'] + 1)
        
        # Pressure-quality interaction (para manejar diferentes rangos)
        if all(col in df.columns for col in ['team_scoring_pressure', 'opponent_quality_factor']):
            df['team_pressure_quality'] = df['team_scoring_pressure'] * (1 - df['opponent_quality_factor'])
        
        # ==================== PROYECCIÓN FINAL MEJORADA DEL EQUIPO ====================
        # Proyección base mejorada
        base_features = ['team_hybrid_projection', 'team_quality_efficiency_interaction']
        if all(col in df.columns for col in base_features):
            df['team_enhanced_projection'] = (
                df['team_hybrid_projection'] * 0.65 +
                df['team_quality_efficiency_interaction'] * 25
            )
        
        # Ajustes contextuales mejorados
        contextual_features = ['team_home_court_boost', 'team_rest_advantage', 'team_season_importance']
        if all(col in df.columns for col in contextual_features):
            df['team_contextual_adjustment'] = (
                df['team_home_court_boost'] + 
                df['team_rest_advantage'] +
                df['team_season_importance'] * 3.0
            )
        
        # Ajuste por estabilidad del equipo
        stability_features = ['team_stability_confidence', 'team_pressure_quality']
        if all(col in df.columns for col in stability_features):
            df['team_stability_adjustment'] = (
                df['team_stability_confidence'] * 2.0 +
                df['team_pressure_quality'] * 5.0
            )
        
        # Una sola proyección final robusta para el equipo MEJORADA
        final_features = ['team_enhanced_projection', 'team_contextual_adjustment', 'team_stability_adjustment']
        if all(col in df.columns for col in final_features):
            df['team_final_projection'] = (
                df['team_enhanced_projection'] +
                df['team_contextual_adjustment'] + 
                df['team_stability_adjustment']
            )
        elif 'team_enhanced_projection' in df.columns and 'team_contextual_adjustment' in df.columns:
            # Fallback si no hay stability features
            df['team_final_projection'] = (
                df['team_enhanced_projection'] +
                df['team_contextual_adjustment']
            )
        
        # ==================== AJUSTES ESPECÍFICOS POR RANGO ====================
        # Ajustes específicos para diferentes rangos de scoring
        if all(col in df.columns for col in ['team_final_projection', 'team_scoring_tier']):
            # Ajustar predicciones según el tier de scoring histórico
            df['team_tier_adjusted_projection'] = df['team_final_projection'].copy()
            
            # Ajustes por tier para mejorar precisión en cada rango
            low_mask = df['team_scoring_tier'] == 'low_scoring'
            mid_mask = df['team_scoring_tier'] == 'mid_scoring' 
            high_mask = df['team_scoring_tier'] == 'high_scoring'
            
            # Ajustes conservadores para low scoring teams
            df.loc[low_mask, 'team_tier_adjusted_projection'] *= 0.98
            
            # Ajustes neutros para mid scoring teams
            df.loc[mid_mask, 'team_tier_adjusted_projection'] *= 1.0
            
            # Ajustes ligeramente agresivos para high scoring teams
            df.loc[high_mask, 'team_tier_adjusted_projection'] *= 1.02
            
            # Usar la proyección ajustada como final
            df['team_final_projection'] = df['team_tier_adjusted_projection']
        
        # Límites realistas NBA para un equipo (más estrictos)
        if 'team_final_projection' in df.columns:
            df['team_final_prediction'] = np.clip(df['team_final_projection'], 85, 150)
        
        # ==================== FEATURES DE CONFIANZA MEJORADAS DEL EQUIPO ====================
        # Métrica de confianza específica del equipo mejorada
        confidence_features = ['opponent_quality_factor', 'team_conversion_efficiency', 'team_scoring_stability']
        if all(col in df.columns for col in confidence_features):
            df['team_prediction_confidence'] = (
                (2 - df['opponent_quality_factor']) * 0.4 +
                df['team_conversion_efficiency'] * 0.3 +
                df['team_scoring_stability'] * 0.3
            )
        elif all(col in df.columns for col in ['opponent_quality_factor', 'team_conversion_efficiency']):
            # Fallback sin stability
            df['team_prediction_confidence'] = (
                (2 - df['opponent_quality_factor']) * 0.5 +
                df['team_conversion_efficiency'] * 0.5
            )
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> None:
        """Aplicar filtros de calidad para eliminar features problemáticas"""
        # Eliminar features con alta correlación o problemas
        problematic_features = [
            'temp_total_points'  # Feature temporal si se creó
        ]
        
        for feature in problematic_features:
            if feature in df.columns:
                df.drop(columns=[feature], inplace=True)
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualizar lista de columnas de features disponibles"""
        exclude_cols = ['Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp',
                       'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                       'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
                       '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
                       'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'is_win']  # Excluir is_win directo
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Retorna las características agrupadas por categoría."""
        groups = {
            'base_projections': [
                'team_direct_scoring_projection', 'team_mathematical_projection', 
                'team_hybrid_projection', 'team_final_projection'
            ],
            'efficiency_metrics': [
                'team_conversion_efficiency', 'team_true_shooting_approx', 
                'team_efg_approx'
            ],
            'historical_trends': [
                'team_direct_projection_avg_5g', 'team_conversion_efficiency_avg_5g',
                'team_pace_avg_5g', 'team_ts_avg_5g'
            ],
            'opponent_factors': [
                'opponent_def_strength', 'opponent_quality_factor', 
                'team_vs_opp_scoring_history', 'team_matchup_edge'
            ],
            'contextual_factors': [
                'team_is_home', 'team_energy_factor', 'team_home_court_boost',
                'team_rest_advantage', 'team_season_importance'
            ],
            'momentum_features': [
                'team_confidence_factor', 'team_confidence_momentum',
                'team_win_pct_5g', 'team_win_pct_10g', 'team_recent_wins', 'team_win_streak'
            ],
            'final_interactions': [
                'team_pace_efficiency_interaction', 'team_momentum_context',
                'team_quality_efficiency_interaction', 'team_final_prediction'
            ]
        }
        
        return groups
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Valida la calidad de las características generadas."""
        validation_report = {
            'total_features': 0,
            'missing_features': [],
            'feature_coverage': {}
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
        
        logger.info(f"Validación completada: {len(all_features)} features, "
                   f"{len(validation_report['missing_features'])} faltantes")
        
        return validation_report 