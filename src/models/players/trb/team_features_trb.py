"""
Características de Equipos para Predicción de Rebotes (TRB) - OPTIMIZADO
======================================================================

Este módulo genera características de equipos basadas en datos reales
partido a partido para mejorar la predicción de rebotes individuales.

CARACTERÍSTICAS CLAVE BASADAS EN EVIDENCIA EMPÍRICA:
- Pace del equipo (más posesiones = más oportunidades de rebote)
- Defensive Rating del oponente (peor defensa = más tiros fallados)
- Four Factors del equipo y oponente (eFG%, TOV%, ORB%, FTr%)
- Tendencias recientes del equipo
- Matchup específico entre equipos
- Contexto del juego (local/visitante, back-to-back)

Objetivo: Proporcionar contexto de equipo para alcanzar ≥97% precisión en rebotes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TeamReboundingFeatures:
    """
    Generador de características de equipos para predicción de rebotes.
    
    Extrae información valiosa de los datos reales de equipos partido a partido
    para proporcionar contexto que mejore la predicción de rebotes individuales.
    """
    
    def __init__(self):
        """Inicializa el generador de características de equipos."""
        self.team_stats_cache = {}
        self.opponent_stats_cache = {}
        
        logger.debug("TeamReboundingFeatures inicializado")
    

    
    def _validate_dataframes(self, player_df: pd.DataFrame, teams_df: pd.DataFrame):
        """Valida que los DataFrames tengan las columnas necesarias."""
        # Columnas requeridas en datos de jugadores
        player_required = ['Player', 'Date', 'Team', 'Opp']
        missing_player = [col for col in player_required if col not in player_df.columns]
        if missing_player:
            raise ValueError(f"Columnas faltantes en datos de jugadores: {missing_player}")
        
        # Columnas requeridas en datos de equipos (basadas en estructura real verificada)
        team_required = ['Team', 'Date', 'Opp', 'FGA', 'FG', '3PA', '3P', 'FTA', 'FT', 
                        'FGA_Opp', 'FG_Opp', '3PA_Opp', '3P_Opp', 'PTS']
        missing_team = [col for col in team_required if col not in teams_df.columns]
        if missing_team:
            logger.warning(f"Columnas faltantes en datos de equipos: {missing_team}")
            logger.info("Continuando con las columnas disponibles...")
    
    def _prepare_teams_data(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Prepara y enriquece los datos de equipos."""
        df = teams_df.copy()
        
        # Convertir fechas
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Calcular métricas avanzadas
        df = self._calculate_advanced_team_metrics(df)
        
        # Ordenar por equipo y fecha
        df = df.sort_values(['Team', 'Date'])
        
        return df
    
    def _calculate_advanced_team_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas avanzadas de equipos enfocadas en rebotes."""
        df_enhanced = df.copy()
        
        # 1. MÉTRICAS DE TIROS FALLADOS (OPORTUNIDADES DE REBOTE)
        # Tiros fallados propios (oportunidades de rebote ofensivo) - con validación
        df_enhanced['team_missed_shots'] = np.maximum(0, df_enhanced['FGA'] - df_enhanced['FG'])
        df_enhanced['team_missed_3pt'] = np.maximum(0, df_enhanced['3PA'] - df_enhanced['3P'])
        df_enhanced['team_missed_2pt'] = np.maximum(0, 
            (df_enhanced['FGA'] - df_enhanced['3PA']) - (df_enhanced['FG'] - df_enhanced['3P'])
        )
        
        # Tiros fallados del oponente (oportunidades de rebote defensivo) - con validación
        df_enhanced['opp_missed_shots'] = np.maximum(0, df_enhanced['FGA_Opp'] - df_enhanced['FG_Opp'])
        df_enhanced['opp_missed_3pt'] = np.maximum(0, df_enhanced['3PA_Opp'] - df_enhanced['3P_Opp'])
        df_enhanced['opp_missed_2pt'] = np.maximum(0,
            (df_enhanced['FGA_Opp'] - df_enhanced['3PA_Opp']) - (df_enhanced['FG_Opp'] - df_enhanced['3P_Opp'])
        )
        
        # Total de oportunidades de rebote por juego
        df_enhanced['total_rebound_opportunities'] = df_enhanced['team_missed_shots'] + df_enhanced['opp_missed_shots']
        df_enhanced['total_long_rebound_opps'] = df_enhanced['team_missed_3pt'] + df_enhanced['opp_missed_3pt']
        df_enhanced['total_close_rebound_opps'] = df_enhanced['team_missed_2pt'] + df_enhanced['opp_missed_2pt']
        
        # 2. MÉTRICAS DE EFICIENCIA DE TIRO (CALIDAD DE OPORTUNIDADES)
        # Porcentajes de tiro del equipo
        df_enhanced['team_fg_pct'] = np.where(df_enhanced['FGA'] > 0, df_enhanced['FG'] / df_enhanced['FGA'], 0.46)
        df_enhanced['team_3pt_pct'] = np.where(df_enhanced['3PA'] > 0, df_enhanced['3P'] / df_enhanced['3PA'], 0.35)
        df_enhanced['team_2pt_pct'] = np.where(
            (df_enhanced['FGA'] - df_enhanced['3PA']) > 0,
            (df_enhanced['FG'] - df_enhanced['3P']) / (df_enhanced['FGA'] - df_enhanced['3PA']),
            0.52
        )
        
        # Porcentajes de tiro del oponente
        df_enhanced['opp_fg_pct'] = np.where(df_enhanced['FGA_Opp'] > 0, df_enhanced['FG_Opp'] / df_enhanced['FGA_Opp'], 0.46)
        df_enhanced['opp_3pt_pct'] = np.where(df_enhanced['3PA_Opp'] > 0, df_enhanced['3P_Opp'] / df_enhanced['3PA_Opp'], 0.35)
        df_enhanced['opp_2pt_pct'] = np.where(
            (df_enhanced['FGA_Opp'] - df_enhanced['3PA_Opp']) > 0,
            (df_enhanced['FG_Opp'] - df_enhanced['3P_Opp']) / (df_enhanced['FGA_Opp'] - df_enhanced['3PA_Opp']),
            0.52
        )
        
        # 3. MÉTRICAS DE PACE Y TEMPO (EXPOSICIÓN A OPORTUNIDADES)
        # Estimación de posesiones por equipo (usando columnas disponibles con validación)
        
        # Validar y obtener columnas auxiliares con valores por defecto seguros
        fta_team = df_enhanced['FTA'] if 'FTA' in df_enhanced.columns else df_enhanced['FGA'] * 0.25
        fta_opp = df_enhanced['FTA_Opp'] if 'FTA_Opp' in df_enhanced.columns else df_enhanced['FGA_Opp'] * 0.25
        orb_team = df_enhanced['ORB'] if 'ORB' in df_enhanced.columns else df_enhanced['FGA'] * 0.1
        orb_opp = df_enhanced['ORB_Opp'] if 'ORB_Opp' in df_enhanced.columns else df_enhanced['FGA_Opp'] * 0.1
        tov_team = df_enhanced['TOV'] if 'TOV' in df_enhanced.columns else df_enhanced['FGA'] * 0.15
        tov_opp = df_enhanced['TOV_Opp'] if 'TOV_Opp' in df_enhanced.columns else df_enhanced['FGA_Opp'] * 0.15
        
        # Calcular posesiones con validación de valores positivos
        df_enhanced['team_possessions'] = np.maximum(
            df_enhanced['FGA'] + 0.44 * fta_team - orb_team + tov_team,
            df_enhanced['FGA'] * 0.8  # Mínimo 80% de FGA como posesiones
        )
        
        df_enhanced['opp_possessions'] = np.maximum(
            df_enhanced['FGA_Opp'] + 0.44 * fta_opp - orb_opp + tov_opp,
            df_enhanced['FGA_Opp'] * 0.8  # Mínimo 80% de FGA como posesiones
        )
        
        # Pace del juego (posesiones por 48 minutos)
        df_enhanced['game_pace'] = (df_enhanced['team_possessions'] + df_enhanced['opp_possessions']) / 2
        df_enhanced['team_pace'] = df_enhanced['team_possessions']
        df_enhanced['opp_pace'] = df_enhanced['opp_possessions']
        
        # 4. MÉTRICAS DEFENSIVAS (CALIDAD DE LA DEFENSA)
        # Rating defensivo aproximado (puntos permitidos por 100 posesiones)
        pts_opp = df_enhanced['PTS_Opp'] if 'PTS_Opp' in df_enhanced.columns else df_enhanced['PTS'] * 0.95
        
        df_enhanced['defensive_rating_approx'] = np.where(
            df_enhanced['team_possessions'] > 0,
            (pts_opp / df_enhanced['team_possessions']) * 100,
            110.0
        )
        
        # Rating ofensivo aproximado
        df_enhanced['offensive_rating_approx'] = np.where(
            df_enhanced['team_possessions'] > 0,
            (df_enhanced['PTS'] / df_enhanced['team_possessions']) * 100,
            110.0
        )
        
        # 5. MÉTRICAS DE ESTILO DE JUEGO
        # Tendencia de tiros de 3 puntos
        df_enhanced['team_3pt_rate'] = np.where(df_enhanced['FGA'] > 0, df_enhanced['3PA'] / df_enhanced['FGA'], 0.4)
        df_enhanced['opp_3pt_rate'] = np.where(df_enhanced['FGA_Opp'] > 0, df_enhanced['3PA_Opp'] / df_enhanced['FGA_Opp'], 0.4)
        
        # Agresividad ofensiva (intentos de tiro por posesión)
        df_enhanced['team_shot_aggressiveness'] = np.where(
            df_enhanced['team_possessions'] > 0,
            df_enhanced['FGA'] / df_enhanced['team_possessions'],
            0.85
        )
        
        # 6. MÉTRICAS DE REBOTE CONTEXTUALES
        # Oportunidades de rebote ponderadas por pace
        df_enhanced['pace_adjusted_rebound_opps'] = df_enhanced['total_rebound_opportunities'] * (df_enhanced['game_pace'] / 95.0)
        
        # Calidad de oportunidades de rebote (más tiros fallados = más oportunidades)
        total_shots = df_enhanced['FGA'] + df_enhanced['FGA_Opp']
        df_enhanced['rebound_opportunity_quality'] = np.where(
            total_shots > 0,
            df_enhanced['total_rebound_opportunities'] / total_shots,
            0.54  # Promedio NBA de tiros fallados
        )
        
        logger.debug("Métricas avanzadas de equipos calculadas")
        return df_enhanced
    
    def _add_team_features_to_games(self, player_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Añade características de equipos a cada juego de jugador."""
        enhanced_df = player_df.copy()
        
        # Convertir fechas si es necesario
        if not pd.api.types.is_datetime64_any_dtype(enhanced_df['Date']):
            enhanced_df['Date'] = pd.to_datetime(enhanced_df['Date'], errors='coerce')
        
        # Características de equipos a añadir
        team_features = [
            # Oportunidades de rebote
            'team_missed_shots', 'opp_missed_shots', 'total_rebound_opportunities',
            'total_long_rebound_opps', 'total_close_rebound_opps',
            
            # Eficiencia de tiro
            'team_fg_pct', 'team_3pt_pct', 'opp_fg_pct', 'opp_3pt_pct',
            
            # Pace y tempo
            'game_pace', 'team_pace', 'opp_pace', 'pace_adjusted_rebound_opps',
            
            # Métricas defensivas
            'defensive_rating_approx', 'offensive_rating_approx',
            
            # Estilo de juego
            'team_3pt_rate', 'opp_3pt_rate', 'team_shot_aggressiveness',
            
            # Calidad de oportunidades
            'rebound_opportunity_quality'
        ]
        
        # Inicializar columnas con valores por defecto
        for feature in team_features:
            enhanced_df[feature] = np.nan
        
        # Para cada juego del jugador, buscar datos del equipo
        for idx, row in enhanced_df.iterrows():
            game_date = row['Date']
            player_team = row['Team']
            opponent_team = row['Opp']
            
            # Buscar el juego del equipo del jugador
            team_game = teams_df[
                (teams_df['Team'] == player_team) & 
                (teams_df['Date'] == game_date) &
                (teams_df['Opp'] == opponent_team)
            ]
            
            if len(team_game) > 0:
                team_data = team_game.iloc[0]
                
                # Añadir características del equipo con validación
                for feature in team_features:
                    if feature in team_data.index and pd.notna(team_data[feature]):
                        enhanced_df.loc[idx, feature] = team_data[feature]
        
        # Rellenar valores faltantes con promedios de liga
        default_values = {
            'team_missed_shots': 46.0, 'opp_missed_shots': 46.0, 'total_rebound_opportunities': 92.0,
            'total_long_rebound_opps': 44.0, 'total_close_rebound_opps': 48.0,
            'team_fg_pct': 0.46, 'team_3pt_pct': 0.35, 'opp_fg_pct': 0.46, 'opp_3pt_pct': 0.35,
            'game_pace': 95.0, 'team_pace': 95.0, 'opp_pace': 95.0, 'pace_adjusted_rebound_opps': 92.0,
            'defensive_rating_approx': 110.0, 'offensive_rating_approx': 110.0,
            'team_3pt_rate': 0.4, 'opp_3pt_rate': 0.4, 'team_shot_aggressiveness': 0.85,
            'rebound_opportunity_quality': 0.54
        }
        
        for feature, default_val in default_values.items():
            enhanced_df[feature] = enhanced_df[feature].fillna(default_val)
        
        # Calcular características derivadas
        enhanced_df = self._calculate_derived_team_features(enhanced_df)
        
        logger.debug(f"Características de equipos añadidas a {len(enhanced_df)} juegos")
        return enhanced_df
    
    def _calculate_derived_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula características derivadas basadas en datos de equipos."""
        df_derived = df.copy()
        
        # 1. VENTAJAS COMPARATIVAS
        # Ventaja en eficiencia de tiro
        df_derived['fg_pct_advantage'] = df_derived['team_fg_pct'] - df_derived['opp_fg_pct']
        df_derived['3pt_pct_advantage'] = df_derived['team_3pt_pct'] - df_derived['opp_3pt_pct']
        
        # Ventaja en pace
        df_derived['pace_advantage'] = df_derived['team_pace'] - df_derived['opp_pace']
        
        # Ventaja defensiva (menor rating defensivo es mejor)
        df_derived['defensive_advantage'] = 110.0 - df_derived['defensive_rating_approx']
        
        # 2. ÍNDICES COMPUESTOS PARA REBOTES
        # Índice de oportunidades de rebote ofensivo (basado en tiros fallados propios)
        df_derived['offensive_rebound_index'] = (
            df_derived['team_missed_shots'] * 
            (1 + df_derived['team_3pt_rate'] * 0.3)  # Bonus por tiros de 3 (rebotes largos)
        )
        
        # Índice de oportunidades de rebote defensivo (basado en tiros fallados del oponente)
        df_derived['defensive_rebound_index'] = (
            df_derived['opp_missed_shots'] * 
            (1 + df_derived['opp_3pt_rate'] * 0.2)  # Menor bonus para rebotes defensivos de 3
        )
        
        # Índice total de oportunidades ponderado por pace
        df_derived['total_rebound_index'] = (
            (df_derived['offensive_rebound_index'] + df_derived['defensive_rebound_index']) *
            (df_derived['game_pace'] / 95.0)  # Ajuste por pace
        )
        
        # 3. MÉTRICAS DE CONTEXTO DE JUEGO
        # Juego de alto scoring (más posesiones = más oportunidades)
        df_derived['high_scoring_game'] = (
            (df_derived['game_pace'] > 100) | 
            ((df_derived['team_fg_pct'] + df_derived['opp_fg_pct']) > 0.95)
        ).astype(int)
        
        # Juego defensivo (pocas oportunidades pero de alta calidad)
        df_derived['defensive_game'] = (
            (df_derived['game_pace'] < 90) & 
            (df_derived['defensive_rating_approx'] < 105)
        ).astype(int)
        
        # Juego de tiros de 3 (más rebotes largos)
        df_derived['three_point_heavy_game'] = (
            (df_derived['team_3pt_rate'] + df_derived['opp_3pt_rate']) > 0.8
        ).astype(int)
        
        # 4. FACTORES DE AJUSTE POR ESTILO DE JUEGO
        # Multiplicador por estilo de juego del equipo
        df_derived['team_style_multiplier'] = np.clip(
            1.0 + 
            (df_derived['team_3pt_rate'] - 0.4) * 0.2 +  # Ajuste por tiros de 3
            (df_derived['game_pace'] - 95.0) / 95.0 * 0.1,  # Ajuste por pace
            0.7, 1.5  # Limitar entre 0.7 y 1.5 para evitar valores extremos
        )
        
        # Factor de calidad del oponente (peor defensa = más oportunidades)
        df_derived['opponent_quality_factor'] = np.where(
            df_derived['defensive_rating_approx'] > 115,  # Defensa muy débil
            1.15,  # Bonus mayor contra defensa muy débil
            np.where(df_derived['defensive_rating_approx'] > 110,  # Defensa débil
                     1.05,  # Bonus menor contra defensa débil
                     np.where(df_derived['defensive_rating_approx'] < 105,  # Defensa fuerte
                              0.9, 1.0))  # Penalización contra defensa fuerte
        )
        
        # 5. MÉTRICAS FINALES PARA REBOTES
        # Oportunidades ajustadas por todos los factores
        df_derived['final_rebound_opportunities'] = (
            df_derived['total_rebound_index'] * 
            df_derived['team_style_multiplier'] * 
            df_derived['opponent_quality_factor']
        )
        
        # Calidad esperada de rebotes (combinando todos los factores)
        df_derived['expected_rebound_quality'] = (
            df_derived['rebound_opportunity_quality'] * 
            df_derived['team_style_multiplier'] * 
            (1 + df_derived['fg_pct_advantage'] * 0.1)  # Bonus por mejor eficiencia
        )
        
        logger.debug("Características derivadas de equipos calculadas")
        return df_derived
    
    def generate_team_context_features(self, player_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """
        Método principal para generar todas las características de contexto de equipos.
        
        Args:
            player_df: DataFrame con datos de jugadores
            teams_df: DataFrame con datos de equipos
            
        Returns:
            DataFrame con características de equipos añadidas
        """
        logger.info("🏀 Generando características de contexto de equipos para rebotes...")
        
        # Validar datos
        self._validate_dataframes(player_df, teams_df)
        
        # Preparar datos de equipos
        teams_prepared = self._prepare_teams_data(teams_df)
        
        # Añadir características de equipos
        enhanced_df = self._add_team_features_to_games(player_df, teams_prepared)
        
        # Calcular tendencias de equipos (últimos 5 juegos)
        enhanced_df = self._add_team_trends(enhanced_df, teams_prepared)
        
        logger.info("✅ Características de contexto de equipos generadas exitosamente")
        return enhanced_df
    
    def _generate_team_context_features_silent(self, player_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """
        Versión silenciosa para generar características de contexto de equipos (sin logging verboso).
        
        Args:
            player_df: DataFrame con datos de jugadores
            teams_df: DataFrame con datos de equipos
            
        Returns:
            DataFrame con características de equipos añadidas
        """
        # Validar datos
        self._validate_dataframes(player_df, teams_df)
        
        # Preparar datos de equipos
        teams_prepared = self._prepare_teams_data(teams_df)
        
        # Añadir características de equipos
        enhanced_df = self._add_team_features_to_games(player_df, teams_prepared)
        
        # Calcular tendencias de equipos (últimos 5 juegos)
        enhanced_df = self._add_team_trends(enhanced_df, teams_prepared)
        
        return enhanced_df
    
    def _add_team_trends(self, player_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Añade tendencias recientes de equipos (últimos 5 juegos)."""
        enhanced_df = player_df.copy()
        
        # Características de tendencia a calcular
        trend_features = ['team_missed_shots', 'opp_missed_shots', 'game_pace', 'defensive_rating_approx']
        
        for feature in trend_features:
            enhanced_df[f'{feature}_trend_5g'] = np.nan
            enhanced_df[f'{feature}_avg_5g'] = np.nan
        
        # Para cada equipo, calcular tendencias de forma más eficiente
        for team in enhanced_df['Team'].unique():
            team_games = teams_df[teams_df['Team'] == team].sort_values('Date').copy()
            
            if len(team_games) == 0:
                continue
                
            # Calcular tendencias para todas las características disponibles
            for feature in trend_features:
                if feature in team_games.columns:
                    # Promedio móvil de 5 juegos
                    team_games[f'{feature}_avg_5g'] = team_games[feature].rolling(5, min_periods=1).mean()
                    
                    # Tendencia (pendiente) de 5 juegos
                    team_games[f'{feature}_trend_5g'] = team_games[feature].rolling(5, min_periods=1).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
            
            # Merge optimizado con datos del jugador
            team_player_games = enhanced_df[enhanced_df['Team'] == team].copy()
            
            for idx, row in team_player_games.iterrows():
                game_date = row['Date']
                
                # Buscar el juego más cercano anterior o igual (más eficiente)
                team_game_trends = team_games[team_games['Date'] <= game_date]
                
                if len(team_game_trends) > 0:
                    latest_trends = team_game_trends.iloc[-1]
                    
                    for feature in trend_features:
                        avg_col = f'{feature}_avg_5g'
                        trend_col = f'{feature}_trend_5g'
                        
                        if avg_col in latest_trends.index and pd.notna(latest_trends[avg_col]):
                            enhanced_df.loc[idx, avg_col] = latest_trends[avg_col]
                        
                        if trend_col in latest_trends.index and pd.notna(latest_trends[trend_col]):
                            enhanced_df.loc[idx, trend_col] = latest_trends[trend_col]
        
        # Rellenar valores faltantes
        trend_defaults = {
            'team_missed_shots_avg_5g': 46.0, 'team_missed_shots_trend_5g': 0.0,
            'opp_missed_shots_avg_5g': 46.0, 'opp_missed_shots_trend_5g': 0.0,
            'game_pace_avg_5g': 95.0, 'game_pace_trend_5g': 0.0,
            'defensive_rating_approx_avg_5g': 110.0, 'defensive_rating_approx_trend_5g': 0.0
        }
        
        for feature, default_val in trend_defaults.items():
            enhanced_df[feature] = enhanced_df[feature].fillna(default_val)
        
        logger.debug("Tendencias de equipos añadidas")
        return enhanced_df
    