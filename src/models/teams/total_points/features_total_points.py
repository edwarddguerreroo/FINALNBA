import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class TotalPointsFeatureEngine:
    """
    Motor de features ELITE para predicción de PUNTOS TOTALES NBA
    Enfoque: Features de DOMINIO ESPECÍFICO con máximo poder predictivo
    Objetivo: >97% precisión con ingeniería de características avanzada
    """
    
    def __init__(self, lookback_games: int = 10):
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_features(self, teams_data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features ELITE específicas del dominio NBA
        """
        df = teams_data.copy()
        
        # 1. FEATURES DE PUNTUACIÓN DIRECTA - Máximo poder predictivo
        df = self._create_direct_scoring_features(df)
        
        # 2. FEATURES DE RITMO Y POSESIONES - Críticas para total de puntos
        df = self._create_pace_possession_features(df)
        
        # 3. FEATURES DE EFICIENCIA AVANZADA - Métricas NBA específicas
        df = self._create_advanced_efficiency_features(df)
        
        # 4. FEATURES DE MATCHUP ESPECÍFICO - Historial detallado
        df = self._create_detailed_matchup_features(df)
        
        # 5. FEATURES DE CONTEXTO NBA - Factores reales del juego
        df = self._create_nba_context_features(df)
        
        # 6. FEATURES DE MOMENTUM Y STREAKS - Tendencias críticas
        df = self._create_momentum_streak_features(df)
        
        # 7. FEATURES DE SITUACIÓN DE JUEGO - Factores específicos
        df = self._create_game_situation_features(df)
        
        # 8. FEATURES DE INTERACCIÓN AVANZADA
        df = self._create_advanced_interactions(df)
        
        # 9. PROYECCIÓN INTELIGENTE FINAL
        df = self._create_intelligent_projection(df)
        
        # 10. FEATURES AVANZADAS NBA (después de crear features básicas)
        df = self._create_advanced_nba_features(df)
        
        # 11. FEATURES ULTRA-ESPECÍFICAS FINALES PARA 97% PRECISIÓN
        # Basadas en las correlaciones más altas detectadas
        
        # Proyección híbrida ultra-optimizada (combinando las mejores correlaciones)
        df['ultimate_scoring_projection'] = (
            df['direct_scoring_projection'] * 0.4 +      # 0.7136 correlación
            df['opp_direct_scoring_projection'] * 0.4 +  # 0.7136 correlación
            df['weighted_shot_volume'] * 0.2             # 0.6943 correlación
        )
        
        # Proyección matemática ultra-precisa
        df['mathematical_total_projection'] = (
            df['total_expected_shots'] * df['combined_conversion_efficiency'] * 1.1
        )
        
        # Proyección ensemble de múltiples métodos
        df['ensemble_projection_v1'] = (
            df['ultimate_scoring_projection'] * 0.5 +
            df['mathematical_total_projection'] * 0.3 +
            df['final_ultra_projection'] * 0.2
        )
        
        # Proyección con factor de confianza ultra-alto
        df['high_confidence_projection'] = (
            df['ensemble_projection_v1'] * df['opponent_quality_factor'] * 
            df['combined_conversion_efficiency']
        )
        
        # Proyección final con todos los ajustes
        df['master_projection'] = (
            df['high_confidence_projection'] + 
            df['home_advantage_factor'] + 
            df['rest_advantage'] +
            df['game_importance_factor'] * 5.0 +
            df['projection_trend'] * 3.0
        )
        
        # Límites ultra-realistas NBA
        df['final_master_prediction'] = np.clip(df['master_projection'], 185, 275)
        
        # 12. FEATURES DE VALIDACIÓN Y CONFIANZA
        # Confianza en la proyección basada en consistencia
        df['projection_confidence_score'] = (
            abs(df['ultimate_scoring_projection'] - df['mathematical_total_projection']) / 
            (df['ultimate_scoring_projection'] + 1)
        )
        df['projection_confidence_score'] = 1 - np.clip(df['projection_confidence_score'], 0, 1)
        
        # Factor de certeza basado en múltiples métricas
        df['certainty_factor'] = (
            df['projection_confidence_score'] * 0.4 +
            df['opponent_quality_factor'] * 0.3 +
            df['combined_conversion_efficiency'] * 0.3
        )
        
        # Proyección final con factor de certeza
        df['ultra_final_projection'] = (
            df['master_projection'] * df['certainty_factor']
        )
        
        print(f"Features FINALES v2.0 creadas: {len([col for col in df.columns if col not in ['Team', 'Date', 'Opp', 'PTS', 'PTS_Opp']])} features")
        print(f"🎯 Features clave para 97%: ultimate_scoring_projection, mathematical_total_projection, master_projection")
        
        # Limpiar y validar
        df = self._clean_and_validate_features(df)
        
        return df
    
    def _create_direct_scoring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de puntuación directa - Máximo poder predictivo"""
        
        # MÚLTIPLES VENTANAS TEMPORALES para capturar patrones
        windows = [3, 5, 7, 10]
        
        for window in windows:
            # Puntos anotados por ventana
            df[f'pts_avg_{window}g'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Puntos permitidos por ventana
            df[f'pts_allowed_avg_{window}g'] = df.groupby('Team')['PTS_Opp'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # TOTAL DE PUNTOS por ventana (suma directa)
            df[f'total_pts_avg_{window}g'] = df[f'pts_avg_{window}g'] + df[f'pts_allowed_avg_{window}g']
        
        # TENDENCIAS DE PUNTUACIÓN (últimos 3 vs promedio 10)
        df['pts_trend_short'] = df['pts_avg_3g'] - df['pts_avg_10g']
        df['pts_allowed_trend_short'] = df['pts_allowed_avg_3g'] - df['pts_allowed_avg_10g']
        df['total_pts_trend'] = df['pts_trend_short'] + df['pts_allowed_trend_short']
        
        # VOLATILIDAD DE PUNTUACIÓN (coeficiente de variación)
        df['pts_volatility'] = df.groupby('Team')['PTS'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().shift(1) / 
                     (x.rolling(window=5, min_periods=1).mean().shift(1) + 1)
        )
        
        # CONSISTENCIA OFENSIVA (inverso de volatilidad)
        df['offensive_consistency'] = 1 / (df['pts_volatility'] + 0.01)
        
        return df
    
    def _create_pace_possession_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de ritmo y posesiones - Críticas para total de puntos"""
        
        # POSESIONES ESTIMADAS (fórmula NBA oficial)
        df['possessions'] = df['FGA'] - df['FG'] + df['FTA'] * 0.44 + df['FG'] * 0.56
        df['opp_possessions'] = df['FGA_Opp'] - df['FG_Opp'] + df['FTA_Opp'] * 0.44 + df['FG_Opp'] * 0.56
        
        # RITMO PROMEDIO (posesiones por partido)
        for window in [3, 5, 10]:
            df[f'pace_avg_{window}g'] = df.groupby('Team')['possessions'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            df[f'opp_pace_avg_{window}g'] = df.groupby('Team')['opp_possessions'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # RITMO COMBINADO (suma de ambos equipos)
        df['combined_pace_3g'] = df['pace_avg_3g'] + df['opp_pace_avg_3g']
        df['combined_pace_5g'] = df['pace_avg_5g'] + df['opp_pace_avg_5g']
        
        # EFICIENCIA POR POSESIÓN
        df['points_per_possession'] = df.groupby('Team').apply(
            lambda x: (x['PTS'] / x['possessions']).rolling(window=5, min_periods=1).mean().shift(1)
        ).reset_index(level=0, drop=True)
        
        # FACTOR DE ACELERACIÓN (cambio en ritmo)
        df['pace_acceleration'] = df['pace_avg_3g'] - df['pace_avg_10g']
        
        return df
    
    def _create_advanced_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de eficiencia avanzada - Métricas NBA específicas"""
        
        # TRUE SHOOTING PERCENTAGE (métrica oficial NBA)
        df['true_shooting_pct'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['opp_true_shooting_pct'] = df['PTS_Opp'] / (2 * (df['FGA_Opp'] + 0.44 * df['FTA_Opp']))
        
        # PROMEDIOS DE TRUE SHOOTING
        for window in [3, 5, 10]:
            df[f'ts_pct_avg_{window}g'] = df.groupby('Team')['true_shooting_pct'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            df[f'opp_ts_pct_avg_{window}g'] = df.groupby('Team')['opp_true_shooting_pct'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # EFFECTIVE FIELD GOAL PERCENTAGE
        df['efg_pct'] = (df['FG'] + 0.5 * df['3P']) / df['FGA']
        df['opp_efg_pct'] = (df['FG_Opp'] + 0.5 * df['3P_Opp']) / df['FGA_Opp']
        
        # PROMEDIOS DE EFG%
        df['efg_pct_avg_5g'] = df.groupby('Team')['efg_pct'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # DIFERENCIAL DE EFICIENCIA (clave para total de puntos)
        df['efficiency_differential'] = df['ts_pct_avg_5g'] - df['opp_ts_pct_avg_5g']
        
        # FACTOR DE TIROS DE TRES (impacto en puntuación)
        df['three_point_rate'] = df['3PA'] / df['FGA']
        df['three_point_rate_avg'] = df.groupby('Team')['three_point_rate'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        return df
    
    def _create_detailed_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de matchup específico - Historial detallado"""
        
        # RENDIMIENTO HISTÓRICO CONTRA OPONENTE ESPECÍFICO
        df['vs_opp_pts_history'] = df.groupby(['Team', 'Opp'])['PTS'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['vs_opp_allowed_history'] = df.groupby(['Team', 'Opp'])['PTS_Opp'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['vs_opp_total_history'] = df['vs_opp_pts_history'] + df['vs_opp_allowed_history']
        
        # VENTAJA/DESVENTAJA ESPECÍFICA
        df['matchup_offensive_edge'] = df['vs_opp_pts_history'] - df['pts_avg_10g']
        df['matchup_defensive_edge'] = df['pts_allowed_avg_10g'] - df['vs_opp_allowed_history']
        df['matchup_total_edge'] = df['matchup_offensive_edge'] + df['matchup_defensive_edge']
        
        # RITMO ESPECÍFICO DEL MATCHUP
        df['matchup_pace_history'] = df.groupby(['Team', 'Opp'])['combined_pace_5g'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # EFICIENCIA ESPECÍFICA DEL MATCHUP
        df['matchup_efficiency_history'] = df.groupby(['Team', 'Opp'])['efficiency_differential'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # TENDENCIA OVERTIME ESPECÍFICA
        if 'has_overtime' in df.columns:
            df['matchup_overtime_tendency'] = df.groupby(['Team', 'Opp'])['has_overtime'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
        else:
            df['matchup_overtime_tendency'] = 0.05
        
        return df
    
    def _create_nba_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de contexto NBA - Factores reales del juego"""
        
        # FACTOR DE DESCANSO (crítico en NBA)
        if 'days_rest' not in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
        
        # ENERGÍA BASADA EN DESCANSO (impacto real en puntuación)
        df['energy_factor'] = np.where(
            df['days_rest'] == 0, 0.92,  # Back-to-back penalty
            np.where(df['days_rest'] == 1, 0.97,  # 1 día
                    np.where(df['days_rest'] >= 3, 1.03, 1.0))  # 3+ días boost
        )
        
        # VENTAJA LOCAL (impacto específico en puntuación)
        if 'is_home' not in df.columns:
            df['is_home'] = (df['Away'] == 0).astype(int)
        
        df['home_court_boost'] = df['is_home'] * 2.5  # Boost promedio NBA
        
        # FACTOR DE ALTITUD (si disponible)
        # Simplificado: algunos equipos juegan en altitud
        altitude_teams = ['DEN', 'UTA', 'PHX']  # Equipos en altitud
        df['altitude_factor'] = df['Team'].apply(lambda x: 1.02 if x in altitude_teams else 1.0)
        
        return df
    
    def _create_momentum_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum y streaks - Tendencias críticas"""
        
        # MOMENTUM RECIENTE (win% últimos juegos)
        if 'is_win' in df.columns:
            for window in [3, 5, 7]:
                df[f'win_pct_{window}g'] = df.groupby('Team')['is_win'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        else:
            # Crear is_win si no existe
            df['is_win'] = (df['PTS'] > df['PTS_Opp']).astype(int)
            df['win_pct_5g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )
        
        # STREAK ACTUAL (racha de victorias/derrotas)
        df['current_streak'] = df.groupby('Team')['is_win'].transform(
            lambda x: x.shift(1).groupby((x.shift(1) != x.shift(1).shift(1)).cumsum()).cumcount() + 1
        )
        
        # MOMENTUM DE PUNTUACIÓN (tendencia reciente vs histórica)
        df['scoring_momentum'] = (df['pts_avg_3g'] - df['pts_avg_10g']) / df['pts_avg_10g']
        df['defensive_momentum'] = (df['pts_allowed_avg_10g'] - df['pts_allowed_avg_3g']) / df['pts_allowed_avg_10g']
        
        # FACTOR DE CONFIANZA (basado en momentum)
        df['confidence_factor'] = (df['win_pct_5g'] - 0.5) * 2  # Normalizado [-1, 1]
        
        return df
    
    def _create_game_situation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de situación de juego - Factores específicos"""
        
        # FACTOR DE PRESIÓN (diferencia de records)
        df['pressure_differential'] = abs(df['win_pct_5g'] - 0.5) * 2
        
        # IMPORTANCIA DEL PARTIDO (basado en momento de la temporada)
        df['Date'] = pd.to_datetime(df['Date'])
        season_start = df['Date'].min()
        df['days_into_season'] = (df['Date'] - season_start).dt.days
        df['season_importance'] = np.where(
            df['days_into_season'] > 200, 1.05,  # Playoffs/final temporada
            np.where(df['days_into_season'] > 100, 1.02, 1.0)  # Mitad temporada
        )
        
        # FACTOR DE RIVALIDAD (simplificado)
        rivalry_pairs = [
            ('LAL', 'BOS'), ('LAL', 'GSW'), ('BOS', 'PHI'), 
            ('MIA', 'BOS'), ('LAC', 'LAL'), ('GSW', 'LAC')
        ]
        df['rivalry_factor'] = df.apply(
            lambda row: 1.03 if (row['Team'], row['Opp']) in rivalry_pairs or 
                              (row['Opp'], row['Team']) in rivalry_pairs else 1.0, axis=1
        )
        
        return df
    
    def _create_advanced_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de interacción avanzada - Combinaciones poderosas"""
        
        # INTERACCIÓN RITMO-EFICIENCIA
        if 'combined_pace_advanced_5g' in df.columns and 'true_shooting_pct_5g' in df.columns:
            df['pace_efficiency_interaction'] = df['combined_pace_advanced_5g'] * df['true_shooting_pct_5g']
        
        # INTERACCIÓN MOMENTUM-CONTEXTO
        if 'confidence_factor' in df.columns and 'energy_factor' in df.columns and 'home_court_boost' in df.columns:
            df['momentum_context_combo'] = df['confidence_factor'] * df['energy_factor'] * df['home_court_boost']
        
        # INTERACCIÓN MATCHUP-SITUACIÓN
        if 'matchup_total_edge' in df.columns and 'season_importance' in df.columns and 'rivalry_factor' in df.columns:
            df['matchup_situation_combo'] = df['matchup_total_edge'] * df['season_importance'] * df['rivalry_factor']
        
        # FACTOR DE EXPLOSIVIDAD (capacidad de partidos de muchos puntos)
        df['explosiveness_factor'] = (
            df['pts_avg_3g'] * df['combined_pace_5g'] * df['ts_pct_avg_5g'] / 1000
        )
        
        # FACTOR DE DEFENSIVIDAD (capacidad de limitar puntos)
        df['defensive_factor'] = (
            (1 - df['opp_ts_pct_avg_5g']) * (1 - df['pts_allowed_avg_5g'] / 120)
        )
        
        # BALANCE OFENSIVO-DEFENSIVO
        df['offensive_defensive_balance'] = df['explosiveness_factor'] / (df['defensive_factor'] + 0.1)
        
        return df
    
    def _create_intelligent_projection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proyección inteligente final - Combinando todos los factores"""
        
        # PROYECCIÓN BASE (múltiples ventanas)
        df['base_projection_3g'] = df['total_pts_avg_3g']
        df['base_projection_5g'] = df['total_pts_avg_5g']
        df['base_projection_10g'] = df['total_pts_avg_10g']
        
        # AJUSTE POR MATCHUP
        matchup_adjustment = (
            df['matchup_total_edge'].fillna(0) * 0.7 +
            df['matchup_pace_history'].fillna(df['combined_pace_5g']) * 0.3
        )
        
        # AJUSTE POR CONTEXTO
        context_adjustment = (
            df['energy_factor'] * 3 +
            df['home_court_boost'] +
            df['altitude_factor'] * 2 +
            df['season_importance'] * 2 +
            df['rivalry_factor'] * 3
        )
        
        # AJUSTE POR MOMENTUM
        momentum_adjustment = (
            df['scoring_momentum'] * 5 +
            df['confidence_factor'] * 3 +
            df['total_pts_trend'] * 0.5
        )
        
        # AJUSTE POR OVERTIME
        overtime_adjustment = df['matchup_overtime_tendency'] * 20
        
        # PROYECCIÓN FINAL INTELIGENTE (promedio ponderado de ventanas)
        base_weighted = (
            df['base_projection_3g'] * 0.5 +
            df['base_projection_5g'] * 0.3 +
            df['base_projection_10g'] * 0.2
        )
        
        df['intelligent_total_projection'] = (
            base_weighted + 
            matchup_adjustment + 
            context_adjustment + 
            momentum_adjustment + 
            overtime_adjustment
        )
        
        # CONFIANZA EN LA PROYECCIÓN
        df['projection_confidence'] = (
            df['offensive_consistency'] * 0.3 +
            (1 - df['pts_volatility']) * 0.3 +
            abs(df['confidence_factor']) * 0.2 +
            (df['matchup_total_edge'].fillna(0) / 10 + 0.5) * 0.2
        )
        
        return df
    
    def _create_advanced_nba_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas específicas del dominio NBA
        Enfoque en características que realmente impactan los puntos totales
        VERSIÓN 2.0 - Optimizada basada en análisis de correlación
        """
        print("🔬 Generando features avanzadas NBA específicas v2.0...")
        
        # ==================== FEATURES ULTRA-ESPECÍFICAS NBA PARA 97% PRECISIÓN ====================
        
        # 1. FEATURES DE PUNTUACIÓN DIRECTA ULTRA-ESPECÍFICAS (MEJORADAS)
        # Proyección directa basada en eficiencia real NBA
        df['direct_scoring_projection'] = (
            df['FGA'] * df['FG%'].fillna(0.45) * 2 +  # Puntos de 2P
            df['3PA'] * df['3P%'].fillna(0.35) * 3 +  # Puntos de 3P
            df['FTA'] * df['FT%'].fillna(0.75) * 1    # Puntos de FT
        )
        
        df['opp_direct_scoring_projection'] = (
            df['FGA_Opp'] * df['FG%_Opp'].fillna(0.45) * 2 +
            df['3PA_Opp'] * df['3P%_Opp'].fillna(0.35) * 3 +
            df['FTA_Opp'] * df['FT%_Opp'].fillna(0.75) * 1
        )
        
        # TOTAL DIRECTO (suma de ambos equipos)
        df['total_direct_projection'] = df['direct_scoring_projection'] + df['opp_direct_scoring_projection']
        
        # NUEVAS FEATURES BASADAS EN ANÁLISIS DE CORRELACIÓN
        
        # 2. FEATURES DE PROYECCIÓN MATEMÁTICA ULTRA-PRECISAS
        # Proyección basada en tiros esperados vs reales
        df['expected_shots_team'] = df['FGA'] + df['3PA'] * 0.4 + df['FTA'] * 0.44
        df['expected_shots_opp'] = df['FGA_Opp'] + df['3PA_Opp'] * 0.4 + df['FTA_Opp'] * 0.44
        df['total_expected_shots'] = df['expected_shots_team'] + df['expected_shots_opp']
        
        # Eficiencia de conversión ultra-específica
        df['conversion_efficiency'] = (
            (df['FG%'].fillna(0.45) + df['3P%'].fillna(0.35) + df['FT%'].fillna(0.75)) / 3
        )
        df['opp_conversion_efficiency'] = (
            (df['FG%_Opp'].fillna(0.45) + df['3P%_Opp'].fillna(0.35) + df['FT%_Opp'].fillna(0.75)) / 3
        )
        df['combined_conversion_efficiency'] = (df['conversion_efficiency'] + df['opp_conversion_efficiency']) / 2
        
        # 3. FEATURES DE VOLUMEN ULTRA-ESPECÍFICAS (CORRELACIÓN 0.5594)
        # Volumen total de tiros (predictor directo de puntos totales)
        df['total_shot_volume'] = df['FGA'] + df['FGA_Opp'] + df['FTA'] + df['FTA_Opp']
        
        # Volumen ponderado por eficiencia
        df['weighted_shot_volume'] = (
            df['total_shot_volume'] * df['combined_conversion_efficiency']
        )
        
        # Intensidad de tiros de alto valor
        df['high_value_shot_intensity'] = (
            (df['3PA'] + df['3PA_Opp']) * 3 + (df['FTA'] + df['FTA_Opp']) * 1
        ) / (df['total_shot_volume'] + 1)
        
        # 4. FEATURES DE PROYECCIÓN HÍBRIDA ULTRA-AVANZADAS
        # Combinando las mejores correlaciones
        df['hybrid_projection_v1'] = (
            df['direct_scoring_projection'] * 0.7 +  # Mejor correlación (0.7136)
            df['total_shot_volume'] * 0.3            # Segunda mejor (0.5594)
        )
        
        df['hybrid_projection_v2'] = (
            df['total_direct_projection'] * 0.6 +
            df['weighted_shot_volume'] * 0.4
        )
        
        # 5. FEATURES DE CALIDAD DEL OPONENTE MEJORADAS (CORRELACIÓN 0.4089)
        # Ranking defensivo del oponente (basado en puntos permitidos)
        opp_def_ranking = df.groupby('Opp')['PTS_Opp'].rolling(10, min_periods=3).mean().reset_index(0, drop=True)
        df['opp_defensive_ranking'] = opp_def_ranking.rank(pct=True).fillna(0.5)
        
        # Ranking ofensivo del oponente
        opp_off_ranking = df.groupby('Opp')['PTS'].rolling(10, min_periods=3).mean().reset_index(0, drop=True)
        df['opp_offensive_ranking'] = opp_off_ranking.rank(pct=True).fillna(0.5)
        
        # Factor de calidad total del oponente MEJORADO
        df['opponent_quality_factor'] = (
            df['opp_offensive_ranking'] * 0.6 + 
            df['opp_defensive_ranking'] * 0.4
        )
        
        # Factor de dificultad del matchup
        df['matchup_difficulty'] = abs(df['opponent_quality_factor'] - 0.5) * 2
        
        # 6. FEATURES DE EFICIENCIA ULTRA-ESPECÍFICAS
        # Eficiencia de puntos por tiro MEJORADA
        df['points_per_shot'] = df['direct_scoring_projection'] / (df['FGA'] + 0.44 * df['FTA'] + 1)
        df['opp_points_per_shot'] = df['opp_direct_scoring_projection'] / (df['FGA_Opp'] + 0.44 * df['FTA_Opp'] + 1)
        
        # Eficiencia combinada ULTRA-ESPECÍFICA
        df['combined_points_per_shot'] = df['points_per_shot'] + df['opp_points_per_shot']
        
        # Eficiencia relativa (vs promedio liga)
        league_avg_efficiency = 1.1  # Aproximado NBA
        df['efficiency_vs_league'] = df['combined_points_per_shot'] / league_avg_efficiency
        
        # 7. FEATURES DE RITMO ULTRA-ESPECÍFICAS (CRÍTICAS PARA TOTAL DE PUNTOS)
        # Posesiones reales estimadas (fórmula NBA oficial simplificada)
        df['real_possessions'] = df['FGA'] + df['FTA'] * 0.44 - df['FG'] * 0.1
        df['opp_real_possessions'] = df['FGA_Opp'] + df['FTA_Opp'] * 0.44 - df['FG_Opp'] * 0.1
        
        # Pace total del juego MEJORADO
        df['total_game_pace'] = df['real_possessions'] + df['opp_real_possessions']
        
        # Pace ajustado por calidad del oponente
        df['adjusted_pace'] = df['total_game_pace'] * df['opponent_quality_factor']
        
        # 8. FEATURES DE PROYECCIÓN FINAL ULTRA-ESPECÍFICAS
        # Proyección matemática ULTRA-PRECISA
        df['ultra_precise_projection'] = (
            df['hybrid_projection_v1'] * 0.4 +
            df['hybrid_projection_v2'] * 0.3 +
            df['weighted_shot_volume'] * 0.2 +
            df['adjusted_pace'] * 0.1
        )
        
        # Proyección con factor de confianza
        df['confidence_weighted_projection'] = (
            df['ultra_precise_projection'] * df['opponent_quality_factor']
        )
        
        # 9. FEATURES DE SITUACIÓN DE JUEGO AVANZADAS
        # Factor de ventaja local específico MEJORADO
        df['home_advantage_factor'] = df['is_home'] * 2.5  # Ventaja promedio NBA
        
        # Factor de descanso avanzado MEJORADO
        df['rest_advantage'] = np.where(
            df['days_rest'] == 0, -3.0,  # Penalización back-to-back
            np.where(df['days_rest'] == 1, -1.0,  # Penalización 1 día
                    np.where(df['days_rest'] >= 3, 2.0, 0.0))  # Bonus 3+ días
        )
        
        # Factor de importancia del partido MEJORADO
        df['game_importance_factor'] = np.where(
            df['days_into_season'] > 200, 1.08,  # Playoffs/final temporada
            np.where(df['days_into_season'] > 150, 1.04,  # Recta final
                    np.where(df['days_into_season'] > 100, 1.02, 1.0))  # Mitad temporada
        )
        
        # 10. FEATURES DE INTERACCIÓN CRÍTICAS ULTRA-AVANZADAS
        # Interacción volumen x eficiencia x calidad oponente
        df['volume_efficiency_quality_interaction'] = (
            df['total_shot_volume'] * 
            df['combined_conversion_efficiency'] * 
            df['opponent_quality_factor']
        )
        
        # Interacción proyección x contexto
        df['projection_context_interaction'] = (
            df['ultra_precise_projection'] * 
            df['home_advantage_factor'] * 
            df['game_importance_factor']
        )
        
        # 11. FEATURES DE TENDENCIAS ULTRA-ESPECÍFICAS
        # Promedios móviles de proyección directa MEJORADOS
        for window in [3, 5, 7, 10]:
            df[f'direct_projection_avg_{window}g'] = df.groupby('Team')['direct_scoring_projection'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            df[f'total_projection_avg_{window}g'] = df.groupby('Team')['total_direct_projection'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # Tendencia de proyección (últimos 3 vs últimos 10)
        df['projection_trend'] = (
            df['direct_projection_avg_3g'] - df['direct_projection_avg_10g']
        ) / (df['direct_projection_avg_10g'] + 1)
        
        # 12. FEATURES DE PROYECCIÓN FINAL ULTRA-OPTIMIZADAS
        # Proyección ensemble ULTRA-ESPECÍFICA
        df['final_ultra_projection'] = (
            df['ultra_precise_projection'] * 0.35 +
            df['confidence_weighted_projection'] * 0.25 +
            df['volume_efficiency_quality_interaction'] * 0.20 +
            df['projection_context_interaction'] * 0.15 +
            df['total_projection_avg_5g'] * 0.05
        )
        
        # Proyección con ajustes contextuales FINALES
        df['context_adjusted_prediction'] = (
            df['final_ultra_projection'] + 
            df['home_advantage_factor'] + 
            df['rest_advantage'] +
            df['projection_trend'] * 2.0
        )
        
        # Límites realistas NBA OPTIMIZADOS
        df['final_prediction'] = np.clip(df['context_adjusted_prediction'], 180, 280)
        
        print(f"Features avanzadas v2.0 creadas: {len([col for col in df.columns if col not in ['Team', 'Date', 'Opp', 'PTS', 'PTS_Opp']])} features")
        
        return df
    
    def _clean_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida features eliminando redundancia"""
        
        # Columnas a excluir - SOLO las que causan data leakage real
        original_cols = [
            # Identificadores y metadatos
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            
            # TARGET DIRECTO (data leakage crítico)
            'PTS', 'PTS_Opp', 'total_points', 'total_score',
            
            # TIROS CONVERTIDOS (correlación directa con puntos)
            'FG', '2P', '3P', 'FT',  # Estos suman directamente a PTS
            'FG_Opp', '2P_Opp', '3P_Opp', 'FT_Opp',  # Estos suman directamente a PTS_Opp
            
            # ESTADÍSTICAS DE PUNTOS AGREGADAS (data leakage)
            'PTS_home_avg', 'PTS_away_avg', 'PTS_Opp_home_avg', 'PTS_Opp_away_avg',
            
            # Resultados conocidos después del partido (data leakage)
            'is_win',
            
            # Columnas intermedias que ya se usan en features derivadas
            'days_rest', 'days_into_season'
        ]
        
        # Seleccionar features numéricas ÚTILES (porcentajes, intentos, métricas derivadas)
        feature_cols = [col for col in df.columns 
                       if col not in original_cols 
                       and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        # PRIMER FILTRO: Eliminar correlaciones > 0.85 (filtro inicial)
        if len(feature_cols) > 1:
            correlation_matrix = df[feature_cols].corr().abs()
            
            # Encontrar pares con correlación > 0.85
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.85:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            # Eliminar features redundantes (mantener la primera de cada par)
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                features_to_remove.add(feat2)  # Eliminar la segunda feature
            
            # Actualizar lista de features
            feature_cols = [col for col in feature_cols if col not in features_to_remove]
        else:
            features_to_remove = set()
        
        # Limpiar valores problemáticos
        for col in feature_cols:
            # Reemplazar infinitos y NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
            
            # Clip outliers extremos (1% y 99%)
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)
        
        # Guardar features iniciales
        self.feature_columns = feature_cols
        
        print(f"✅ Features optimizadas: {len(feature_cols)} características NBA")
        print(f"🗑️  Features eliminadas por correlación >85%: {len(features_to_remove)}")
        print(f"📊 Top features: {feature_cols[:10]}")
        print(f"🚫 EXCLUIDOS: PTS, tiros convertidos (FG, 2P, 3P, FT)")
        print(f"✅ INCLUIDOS: Porcentajes (FG%, 2P%, 3P%, FT%), intentos (FGA, 2PA, 3PA, FTA), métricas derivadas")
        
        return df
    
    def apply_final_correlation_filter(self, df: pd.DataFrame, correlation_threshold: float = 0.95) -> pd.DataFrame:
        """
        Aplica filtro final más estricto eliminando correlaciones >95%
        Mantiene columnas requeridas para el funcionamiento del modelo
        """
        
        # Columnas críticas que NO se pueden eliminar
        critical_features = [
            'pts_avg_5g', 'pts_allowed_avg_5g', 'total_pts_avg_5g', 
            'intelligent_total_projection', 'is_home', 'energy_factor',
            'combined_pace_5g', 'efficiency_differential', 'confidence_factor'
        ]
        
        # Features disponibles para filtrado
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) < 2:
            print("⚠️  Muy pocas features para aplicar filtro de correlación")
            return df
        
        # Calcular matriz de correlación
        correlation_matrix = df[available_features].corr().abs()
        
        # Encontrar pares con correlación > threshold
        high_corr_pairs_95 = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if corr_val > correlation_threshold:
                    feat1 = correlation_matrix.columns[i]
                    feat2 = correlation_matrix.columns[j]
                    high_corr_pairs_95.append((feat1, feat2, corr_val))
        
        # Determinar features a eliminar (protegiendo las críticas)
        features_to_remove_95 = set()
        
        for feat1, feat2, corr_val in high_corr_pairs_95:
            # Si ambas son críticas, mantener ambas
            if feat1 in critical_features and feat2 in critical_features:
                continue
            # Si una es crítica, eliminar la otra
            elif feat1 in critical_features:
                features_to_remove_95.add(feat2)
            elif feat2 in critical_features:
                features_to_remove_95.add(feat1)
            # Si ninguna es crítica, eliminar la segunda (arbitrario)
            else:
                features_to_remove_95.add(feat2)
        
        # Actualizar lista de features finales
        final_features = [col for col in available_features if col not in features_to_remove_95]
        
        # Asegurar que tenemos features mínimas críticas
        for critical_feat in critical_features:
            if critical_feat in df.columns and critical_feat not in final_features:
                final_features.append(critical_feat)
        
        # Actualizar feature_columns
        self.feature_columns = final_features
        
        print(f"FILTRO FINAL APLICADO (correlación >{correlation_threshold*100}%):")
        print(f"Features antes del filtro: {len(available_features)}")
        print(f"Features eliminadas: {len(features_to_remove_95)}")
        print(f"Features finales: {len(final_features)}")
        
        if high_corr_pairs_95:
            print(f"⚠️  Correlaciones altas detectadas y resueltas:")
            for feat1, feat2, corr_val in high_corr_pairs_95[:5]:  # Mostrar solo las primeras 5
                action = "ELIMINADA" if feat2 in features_to_remove_95 else "MANTENIDA (crítica)"
                print(f"   • {feat1} ↔ {feat2}: {corr_val:.3f} → {feat2} {action}")
            if len(high_corr_pairs_95) > 5:
                print(f"   ... y {len(high_corr_pairs_95) - 5} más")
        else:
            print(f"✅ No se detectaron correlaciones >{correlation_threshold*100}%")
        
        print(f"Features críticas protegidas: {[f for f in critical_features if f in final_features]}")
        print(f"Features finales: {final_features[:15]}...")
        
        return df
    
    def prepare_prediction_features(self, team1: str, team2: str, teams_data: pd.DataFrame, 
                                  is_team1_home: bool = True) -> np.ndarray:
        """Prepara features para predicción de un partido específico"""
        
        try:
            # Crear features completas
            df_with_features = self.create_features(teams_data)
            df_with_features = self.apply_final_correlation_filter(df_with_features, correlation_threshold=0.95)
            
            # Obtener últimos datos de cada equipo
            team1_data = df_with_features[df_with_features['Team'] == team1].iloc[-1:]
            team2_data = df_with_features[df_with_features['Team'] == team2].iloc[-1:]
            
            if team1_data.empty or team2_data.empty:
                raise ValueError(f"No hay datos suficientes para {team1} o {team2}")
            
            # Verificar que todas las features existen en los datos
            available_features = [col for col in self.feature_columns if col in df_with_features.columns]
            missing_features = [col for col in self.feature_columns if col not in df_with_features.columns]
            
            if missing_features:
                print(f"⚠️  Features faltantes (se rellenarán con 0): {missing_features[:5]}...")
            
            # Extraer features disponibles
            team1_features = []
            team2_features = []
            
            for feature_name in self.feature_columns:
                try:
                    if feature_name in df_with_features.columns:
                        team1_val = team1_data[feature_name].iloc[0] if not team1_data[feature_name].isna().iloc[0] else 0
                        team2_val = team2_data[feature_name].iloc[0] if not team2_data[feature_name].isna().iloc[0] else 0
                    else:
                        # Feature faltante, usar valor por defecto
                        if 'is_home' in feature_name:
                            team1_val = 1 if is_team1_home else 0
                            team2_val = 0 if is_team1_home else 1
                        elif any(keyword in feature_name for keyword in ['pts_avg', 'total_pts', 'projection']):
                            team1_val = 0
                            team2_val = 0
                        else:
                            team1_val = 0
                            team2_val = 0
                except Exception as e:
                    print(f"⚠️  Error procesando feature {feature_name}: {e}")
                    team1_val = 0
                    team2_val = 0
                    # Feature faltante, usar valor por defecto
                    if 'is_home' in feature_name:
                        team1_val = 1 if is_team1_home else 0
                        team2_val = 0 if is_team1_home else 1
                    elif any(keyword in feature_name for keyword in ['avg', 'pct', 'efficiency']):
                        team1_val = 0.5  # Valor neutro para porcentajes/promedios
                        team2_val = 0.5
                    else:
                        team1_val = 0
                        team2_val = 0
                
                team1_features.append(team1_val)
                team2_features.append(team2_val)
                        
        except Exception as e:
                    print(f"⚠️  Error procesando feature {feature_name}: {e}")
                    # Usar valores por defecto seguros
                    team1_features.append(0)
                    team2_features.append(0)
        
        # COMBINAR FEATURES DE FORMA INTELIGENTE PARA TOTAL DE PUNTOS
        combined_features = []
        for i, feature_name in enumerate(self.feature_columns):
            try:
                if any(keyword in feature_name for keyword in ['pts_avg', 'total_pts', 'projection']):
                    # SUMAR capacidades de puntuación (clave para total de puntos)
                    combined_features.append(team1_features[i] + team2_features[i])
                elif any(keyword in feature_name for keyword in ['allowed', 'defensive']):
                    # Promediar capacidades defensivas
                    combined_features.append((team1_features[i] + team2_features[i]) / 2)
                elif 'is_home' in feature_name:
                    # Ajustar ventaja local
                    combined_features.append(1 if is_team1_home else 0)
                elif any(keyword in feature_name for keyword in ['pace', 'combined', 'possessions']):
                    # SUMAR métricas de ritmo (más posesiones = más puntos)
                    combined_features.append(team1_features[i] + team2_features[i])
                elif any(keyword in feature_name for keyword in ['efficiency', 'ts_pct', 'efg']):
                    # Promediar eficiencias
                    combined_features.append((team1_features[i] + team2_features[i]) / 2)
                elif any(keyword in feature_name for keyword in ['vs_opp', 'matchup', 'home_avg', 'away_avg']):
                    # Para matchups y stats casa/visitante, usar promedio ponderado favoreciendo team1
                    combined_features.append((team1_features[i] * 0.6 + team2_features[i] * 0.4))
                elif any(keyword in feature_name for keyword in ['momentum', 'streak', 'confidence', 'wins']):
                    # Para momentum, usar promedio
                    combined_features.append((team1_features[i] + team2_features[i]) / 2)
                elif any(keyword in feature_name for keyword in ['energy', 'boost', 'factor', 'advantage']):
                    # Para factores contextuales, usar promedio
                    combined_features.append((team1_features[i] + team2_features[i]) / 2)
                elif any(keyword in feature_name for keyword in ['trend', 'differential', 'consistency']):
                    # Para tendencias y diferenciales, usar promedio
                    combined_features.append((team1_features[i] + team2_features[i]) / 2)
                else:
                    # Para otras métricas, promediar
                    combined_features.append((team1_features[i] + team2_features[i]) / 2)
            except Exception as e:
                print(f"⚠️  Error combinando feature {feature_name}: {e}")
                combined_features.append(0)  # Valor por defecto seguro
            
        # Verificar que no hay valores problemáticos
        combined_features = np.array(combined_features)
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1.0, neginf=-1.0)
        try:
            return combined_features.reshape(1, -1)
        
        except Exception as e:
            print(f"❌ Error en prepare_prediction_features: {e}")

            # Retornar features por defecto si todo falla
            default_features = np.zeros((1, len(self.feature_columns)))
            return default_features
        
    def get_feature_importance_names(self) -> List[str]:
        """Retorna nombres de features para análisis de importancia"""
        return self.feature_columns
