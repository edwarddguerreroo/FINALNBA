"""
Ingeniería de Características para Predicción de Rebotes (TRB) - ENFOQUE FUNDAMENTAL
==================================================================================

Feature Engineering enfocado en los factores fundamentales de los rebotes NBA:
1. OPORTUNIDADES DE REBOTE: Tiros fallados del equipo y oponente
2. CAPACIDAD FÍSICA: Altura, posición, ventaja física
3. TIEMPO DE EXPOSICIÓN: Minutos jugados y contexto temporal

OBJETIVO: Generar características que capturen la realidad del baloncesto.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)

class ReboundsFeatureEngineer:
    """
    Ingeniero de características enfocado en los fundamentos de los rebotes NBA.
    """
    
    def __init__(self):
        """Inicializa el ingeniero de características fundamental."""
        
        # Características prohibidas (data leakage)
        self.forbidden_features = {
            'TRB', 'ORB', 'DRB', 'TRB%', 'ORB%', 'DRB%', 
            'REB', 'OREB', 'DREB', 'REBR', 'OREBR', 'DREBR'
        }
        
        # Pesos por posición para rebotes (basado en realidad NBA y dataset actual)
        self.position_rebounding_weights = {
            'C': 3.5,      # Centers - máxima capacidad (497 en análisis)
            'F': 2.8,      # Forwards - alta capacidad (463 en análisis)  
            'C-F': 3.3,    # Center-Forward híbridos (352 en análisis)
            'F-C': 3.2,    # Forward-Center híbridos (234 en análisis)
            'F-G': 1.8,    # Forward-Guard híbridos (32 en análisis)
            'G-F': 1.5,    # Guard-Forward híbridos (21 en análisis)
            'G': 1.0,      # Guards - mínima capacidad (0 en análisis)
        }
        
        logger.info("ReboundsFeatureEngineer inicializado con enfoque fundamental")
    
    def _get_opponent_real_data(self, df: pd.DataFrame, teams_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Obtiene datos reales del oponente desde el DataFrame de equipos."""
        df_features = df.copy()
        
        # INICIALIZAR COLUMNAS CON VALORES POR DEFECTO
        df_features['has_real_opp_data'] = 0
        df_features['opp_real_fga'] = 85.0
        df_features['opp_real_fg'] = 39.0
        df_features['opp_real_3pa'] = 34.0
        df_features['opp_real_3p'] = 12.0
        df_features['opp_real_fta'] = 20.0
        df_features['opp_real_ft'] = 15.0
        df_features['opp_real_missed_shots'] = 46.0
        df_features['opp_real_missed_3pt'] = 22.0
        df_features['opp_real_missed_2pt'] = 24.0
        df_features['opp_real_missed_ft'] = 5.0
        df_features['opp_real_fg_pct'] = 0.46
        df_features['opp_real_3pt_pct'] = 0.35
        df_features['opp_real_ft_pct'] = 0.75
        df_features['team_real_fga'] = 85.0
        df_features['team_real_fg'] = 39.0
        df_features['team_real_3pa'] = 34.0
        df_features['team_real_3p'] = 12.0
        
        if teams_df is None:
            logger.warning("No hay datos de equipos disponibles, usando estimaciones")
            return df_features
        
        # Preparar datos de equipos
        teams_df = teams_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(teams_df['Date']):
            teams_df['Date'] = pd.to_datetime(teams_df['Date'], errors='coerce')
        
        # Para cada juego del jugador, buscar los datos reales del equipo oponente
        for idx, row in df_features.iterrows():
            game_date = pd.to_datetime(row['Date'])
            player_team = row['Team']
            opponent_team = row['Opp']
            
            # Buscar el juego del equipo del jugador en la misma fecha
            # El oponente del jugador aparece como 'Team' en el dataset de equipos
            team_game = teams_df[
                (teams_df['Team'] == player_team) & 
                (teams_df['Date'] == game_date) &
                (teams_df['Opp'] == opponent_team)
            ]
            
            if len(team_game) > 0:
                team_data = team_game.iloc[0]
                
                # USAR DATOS REALES DEL OPONENTE (columnas _Opp)
                df_features.loc[idx, 'opp_real_fga'] = team_data.get('FGA_Opp', 85.0)
                df_features.loc[idx, 'opp_real_fg'] = team_data.get('FG_Opp', 39.0)
                df_features.loc[idx, 'opp_real_3pa'] = team_data.get('3PA_Opp', 34.0)
                df_features.loc[idx, 'opp_real_3p'] = team_data.get('3P_Opp', 12.0)
                df_features.loc[idx, 'opp_real_fta'] = team_data.get('FTA_Opp', 20.0)
                df_features.loc[idx, 'opp_real_ft'] = team_data.get('FT_Opp', 15.0)
                
                # Calcular tiros fallados reales del oponente
                df_features.loc[idx, 'opp_real_missed_shots'] = team_data.get('FGA_Opp', 85.0) - team_data.get('FG_Opp', 39.0)
                df_features.loc[idx, 'opp_real_missed_3pt'] = team_data.get('3PA_Opp', 34.0) - team_data.get('3P_Opp', 12.0)
                df_features.loc[idx, 'opp_real_missed_2pt'] = (
                    (team_data.get('FGA_Opp', 85.0) - team_data.get('3PA_Opp', 34.0)) - 
                    (team_data.get('FG_Opp', 39.0) - team_data.get('3P_Opp', 12.0))
                )
                df_features.loc[idx, 'opp_real_missed_ft'] = team_data.get('FTA_Opp', 20.0) - team_data.get('FT_Opp', 15.0)
                
                # Porcentajes reales del oponente
                df_features.loc[idx, 'opp_real_fg_pct'] = team_data.get('FG%_Opp', 0.46)
                df_features.loc[idx, 'opp_real_3pt_pct'] = team_data.get('3P%_Opp', 0.35)
                df_features.loc[idx, 'opp_real_ft_pct'] = team_data.get('FT%_Opp', 0.75)
                
                # Datos del equipo propio también
                df_features.loc[idx, 'team_real_fga'] = team_data.get('FGA', 85.0)
                df_features.loc[idx, 'team_real_fg'] = team_data.get('FG', 39.0)
                df_features.loc[idx, 'team_real_3pa'] = team_data.get('3PA', 34.0)
                df_features.loc[idx, 'team_real_3p'] = team_data.get('3P', 12.0)
                
                # Indicador de datos reales
                df_features.loc[idx, 'has_real_opp_data'] = 1
                
        logger.debug(f"Datos reales del oponente obtenidos para {df_features['has_real_opp_data'].sum()}/{len(df_features)} juegos")
        
        return df_features

    def _calculate_missed_shots_opportunities(self, df: pd.DataFrame, teams_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calcula las oportunidades de rebote basadas en tiros fallados REALES."""
        df_features = df.copy()
        
        # 1. TIROS FALLADOS PROPIOS (Oportunidades de rebote ofensivo)
        if 'FGA' in df.columns and 'FG' in df.columns:
            df_features['own_missed_shots'] = df['FGA'] - df['FG']
            df_features['own_fg_pct'] = np.where(df['FGA'] > 0, df['FG'] / df['FGA'], 0.45)
            
            # Promedio móvil de tiros fallados (últimos 5 juegos)
            df_features['own_missed_shots_avg_5g'] = df_features['own_missed_shots'].rolling(5, min_periods=1).mean()
            df_features['own_missed_shots_trend_5g'] = df_features['own_missed_shots'].rolling(5, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        # 2. TIROS DE 3 PUNTOS FALLADOS PROPIOS (Rebotes largos)
        if '3PA' in df.columns and '3P' in df.columns:
            df_features['own_missed_3pt'] = df['3PA'] - df['3P']
            df_features['own_3pt_pct'] = np.where(df['3PA'] > 0, df['3P'] / df['3PA'], 0.35)
            
            # Los tiros de 3 generan rebotes más largos (más oportunidades para todos)
            df_features['long_rebound_opportunities'] = df_features['own_missed_3pt'] * 1.5
        
        # 3. TIROS DE 2 PUNTOS FALLADOS PROPIOS (Rebotes cerca del aro)
        if 'FGA' in df.columns and '3PA' in df.columns and 'FG' in df.columns and '3P' in df.columns:
            df_features['2pt_attempts'] = df['FGA'] - df['3PA']
            df_features['2pt_made'] = df['FG'] - df['3P']
            df_features['own_missed_2pt'] = df_features['2pt_attempts'] - df_features['2pt_made']
            
            # Los tiros de 2 puntos cerca del aro favorecen a jugadores altos
            df_features['close_rebound_opportunities'] = df_features['own_missed_2pt'] * 1.2
        
        # 4. OBTENER DATOS REALES DEL OPONENTE
        df_features = self._get_opponent_real_data(df_features, teams_df)
        
        # 5. TOTAL DE OPORTUNIDADES DE REBOTE (USANDO DATOS REALES)
        df_features['total_rebound_opportunities'] = (
            df_features.get('own_missed_shots', 0) + 
            df_features.get('opp_real_missed_shots', 46.0)
        )
        
        df_features['total_long_opportunities'] = (
            df_features.get('own_missed_3pt', 0) + 
            df_features.get('opp_real_missed_3pt', 22.0)
        )
        
        df_features['total_close_opportunities'] = (
            df_features.get('own_missed_2pt', 0) + 
            df_features.get('opp_real_missed_2pt', 24.0)
        )
        
        # 6. MÉTRICAS AVANZADAS CON DATOS REALES
        # Calidad defensiva del oponente (más tiros fallados = peor defensa = más oportunidades)
        df_features['opp_defensive_weakness'] = np.where(
            df_features['opp_real_fga'] > 0,
            df_features['opp_real_missed_shots'] / df_features['opp_real_fga'],
            0.54  # Promedio NBA de tiros fallados
        )
        
        # Tendencia de tiros de 3 del oponente (afecta tipo de rebotes)
        df_features['opp_3pt_tendency'] = np.where(
            df_features['opp_real_fga'] > 0,
            df_features['opp_real_3pa'] / df_features['opp_real_fga'],
            0.40  # Promedio NBA de tiros de 3
        )
        
        # Índice de oportunidades ponderado por calidad del oponente
        df_features['quality_weighted_opportunities'] = (
            df_features['total_rebound_opportunities'] * 
            (1 + df_features['opp_defensive_weakness'] * 0.5)  # Bonus por defensa débil
        )
        
        logger.debug(f"Calculadas oportunidades de rebote basadas en tiros fallados REALES")
        
        return df_features
    
    def _calculate_physical_advantages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula ventajas físicas CRÍTICAS para rebotes basadas en análisis real NBA.
        
        CARACTERÍSTICAS FÍSICAS CRÍTICAS (según análisis):
        1. ALTURA: Predictor superior (~0.6-0.7 correlación con RPG)
        2. ENVERGADURA: Extiende radio de rebote (~0.4-0.6 correlación)
        3. PESO: Mantiene posición y boxeo (~0.3-0.5 correlación)
        4. POSICIÓN: Centers (497) y Forwards (463) dominan rebotes
        5. SALTO VERTICAL: Supera oponentes (~0.2-0.4 correlación)
        """
        df_features = df.copy()
        
        # ==================== 1. ALTURA (PREDICTOR SUPERIOR) ====================
        
        if 'Height_Inches' in df_features.columns:
            df_features['height_inches'] = df_features['Height_Inches'].fillna(
                df_features.get('Pos', 'F').map({
                    'C': 84.0,      # 7'0" promedio para Centers (Zubac, Embiid)
                    'F': 80.0,      # 6'8" promedio para Forwards (Antetokounmpo, Sabonis)
                    'C-F': 82.5,    # 6'10.5" promedio para Center-Forward híbridos
                    'F-C': 82.0,    # 6'10" promedio para Forward-Center híbridos
                    'F-G': 78.0,    # 6'6" promedio para Forward-Guard híbridos
                    'G-F': 77.0,    # 6'5" promedio para Guard-Forward híbridos
                    'G': 75.0,      # 6'3" promedio para Guards
                }).fillna(79.0)
            )
        else:
            # Fallback: usar altura por posición como proxy
            df_features['height_inches'] = df_features.get('Pos', 'F').map({
                'C': 84.0, 'F': 80.0, 'C-F': 82.5, 'F-C': 82.0,
                'F-G': 78.0, 'G-F': 77.0, 'G': 75.0
            }).fillna(79.0)
        
        # VENTAJA DE ALTURA EXPONENCIAL (crítica para rebotes)
        df_features['height_advantage_raw'] = (df_features['height_inches'] - 78.0) / 6.0  # Normalizado
        
        # Categorías de altura específicas para rebotes
        df_features['height_tier_elite'] = (df_features['height_inches'] >= 84).astype(int)  # 7'0"+ (Zubac, Embiid)
        df_features['height_tier_very_tall'] = (df_features['height_inches'] >= 81).astype(int)  # 6'9"+ (Jokić, Sabonis)
        df_features['height_tier_tall'] = (df_features['height_inches'] >= 78).astype(int)  # 6'6"+ (Antetokounmpo)
        df_features['height_tier_average'] = (df_features['height_inches'] >= 75).astype(int)  # 6'3"+
        
        # Multiplicador de altura para rebotes (exponencial como en análisis)
        df_features['height_rebounding_multiplier'] = np.where(
            df_features['height_inches'] >= 84, 3.0,      # Elite (Zubac, Embiid)
            np.where(df_features['height_inches'] >= 81, 2.5,  # Very Tall (Jokić, Sabonis)
                     np.where(df_features['height_inches'] >= 78, 2.0,  # Tall (Antetokounmpo)
                              np.where(df_features['height_inches'] >= 75, 1.5, 1.0)))  # Average/Short
        )
        
        # ==================== 2. ENVERGADURA (RADIO DE REBOTE) ====================
        
        # Estimar envergadura basada en altura (típicamente altura + 2-4 pulgadas)
        # La mayoría de datasets no incluyen envergadura, así que la estimamos
        df_features['wingspan_inches'] = df_features['height_inches'] + np.where(
            df_features['height_inches'] >= 82, 4.0,  # Jugadores altos tienen mayor diferencia
            np.where(df_features['height_inches'] >= 78, 3.0, 2.0)
        )
        
        # Ventaja de envergadura para rebotes
        df_features['wingspan_advantage'] = (df_features['wingspan_inches'] - 80.0) / 8.0  # Normalizado
        
        # Categorías de envergadura (según análisis: Embiid ~7'6", Antetokounmpo 7'3")
        df_features['wingspan_elite'] = (df_features['wingspan_inches'] >= 90).astype(int)  # 7'6"+ (Embiid)
        df_features['wingspan_very_long'] = (df_features['wingspan_inches'] >= 87).astype(int)  # 7'3"+ (Antetokounmpo)
        df_features['wingspan_long'] = (df_features['wingspan_inches'] >= 84).astype(int)  # 7'0"+
        
        # Multiplicador de envergadura para rebotes
        df_features['wingspan_rebounding_multiplier'] = np.where(
            df_features['wingspan_inches'] >= 90, 2.5,    # Elite wingspan
            np.where(df_features['wingspan_inches'] >= 87, 2.0,  # Very long
                     np.where(df_features['wingspan_inches'] >= 84, 1.5, 1.0))  # Long/Average
        )
        
        # ==================== 3. PESO (FISICALIDAD Y BOXEO) ====================
        
        # Usar datos de peso ya integrados en el dataset principal
        if 'Weight' in df_features.columns:
            df_features['weight_lbs'] = df_features['Weight'].fillna(
                df_features.get('Pos', 'F').map({
                    'C': 260.0,     # Jokić: 284, Embiid: 280, promedio ~260
                    'F': 230.0,     # Antetokounmpo: ~242, Sabonis: ~240, promedio ~230
                    'C-F': 250.0,   # Center-Forward híbridos
                    'F-C': 245.0,   # Forward-Center híbridos
                    'F-G': 210.0,   # Forward-Guard híbridos
                    'G-F': 200.0,   # Guard-Forward híbridos
                    'G': 185.0,     # Guards promedio ~185
                }).fillna(220.0)
            )
        else:
            # Fallback: estimar peso basado en posición
            df_features['weight_lbs'] = df_features.get('Pos', 'F').map({
                'C': 260.0, 'F': 230.0, 'C-F': 250.0, 'F-C': 245.0,
                'F-G': 210.0, 'G-F': 200.0, 'G': 185.0
            }).fillna(220.0)
        
        # Ventaja de peso para fisicalidad
        df_features['weight_advantage'] = (df_features['weight_lbs'] - 215.0) / 50.0  # Normalizado
        
        # Categorías de peso (según análisis: Jokić 284, Embiid 280, Duren 250)
        df_features['weight_elite'] = (df_features['weight_lbs'] >= 270).astype(int)  # Elite (Jokić, Embiid)
        df_features['weight_heavy'] = (df_features['weight_lbs'] >= 240).astype(int)  # Heavy (Duren, Zubac)
        df_features['weight_solid'] = (df_features['weight_lbs'] >= 210).astype(int)  # Solid
        
        # Multiplicador de peso para rebotes (especialmente defensivos)
        df_features['weight_rebounding_multiplier'] = np.where(
            df_features['weight_lbs'] >= 270, 2.0,       # Elite weight (Jokić, Embiid)
            np.where(df_features['weight_lbs'] >= 240, 1.5,  # Heavy (Duren)
                     np.where(df_features['weight_lbs'] >= 210, 1.2, 1.0))  # Solid/Light
        )
        
        # ==================== 4. POSICIÓN (PREDICTOR FUERTE) ====================
        
        if 'Pos' in df.columns:
            df_features['position_rebounding_weight'] = df['Pos'].map(
                self.position_rebounding_weights
            ).fillna(2.0)
            
            # Indicadores específicos de posición (según análisis: 497 C, 463 F dominan)
            df_features['is_center'] = (df['Pos'] == 'C').astype(int)  # Centers puros (497)
            df_features['is_forward'] = (df['Pos'] == 'F').astype(int)  # Forwards puros (463)
            df_features['is_power_forward'] = (df['Pos'] == 'F').astype(int)  # Power Forwards (alias para F)
            df_features['is_guard'] = (df['Pos'] == 'G').astype(int)    # Guards puros (0)
            
            # Híbridos importantes (352 C-F, 234 F-C según análisis)
            df_features['is_center_forward'] = (df['Pos'] == 'C-F').astype(int)  # 352 en análisis
            df_features['is_forward_center'] = (df['Pos'] == 'F-C').astype(int)  # 234 en análisis
            df_features['is_forward_guard'] = (df['Pos'] == 'F-G').astype(int)   # 32 en análisis
            df_features['is_guard_forward'] = (df['Pos'] == 'G-F').astype(int)   # 21 en análisis
            
            # Agrupaciones lógicas
            df_features['is_big_man'] = (df_features['is_center'] | df_features['is_center_forward'] | df_features['is_forward_center']).astype(int)
            df_features['is_versatile_big'] = (df_features['is_center_forward'] | df_features['is_forward_center']).astype(int)
            df_features['has_forward_skills'] = (df_features['is_forward'] | df_features['is_center_forward'] | df_features['is_forward_center'] | df_features['is_forward_guard']).astype(int)
            df_features['has_guard_skills'] = (df_features['is_guard'] | df_features['is_guard_forward'] | df_features['is_forward_guard']).astype(int)
        
        # ==================== 5. SALTO VERTICAL (ATLETISMO) ====================
        
        # Estimar salto vertical basado en posición y atletismo
        # La mayoría de datasets no incluyen salto vertical, así que lo estimamos
        df_features['vertical_leap_inches'] = df_features.get('Pos', 'F').map({
            'C': 28.0,      # Promedio para Centers
            'F': 31.0,      # Promedio para Forwards (Antetokounmpo alto)
            'C-F': 29.0,    # Center-Forward híbridos
            'F-C': 29.5,    # Forward-Center híbridos
            'F-G': 33.0,    # Forward-Guard híbridos (más atléticos)
            'G-F': 34.0,    # Guard-Forward híbridos (más atléticos)
            'G': 35.0,      # Guards (más atléticos pero menos altura)
        }).fillna(31.0)  # Promedio NBA general
        
        # Categorías de atletismo
        df_features['athletic_elite'] = (df_features['vertical_leap_inches'] >= 36).astype(int)  # Elite
        df_features['athletic_high'] = (df_features['vertical_leap_inches'] >= 32).astype(int)  # High
        df_features['athletic_average'] = (df_features['vertical_leap_inches'] >= 28).astype(int)  # Average
        
        # Multiplicador de atletismo (especialmente para rebotes ofensivos)
        df_features['athleticism_rebounding_multiplier'] = np.where(
            df_features['vertical_leap_inches'] >= 36, 1.5,  # Elite (supera oponentes)
            np.where(df_features['vertical_leap_inches'] >= 32, 1.2,  # High
                     np.where(df_features['vertical_leap_inches'] >= 28, 1.0, 0.8))  # Average/Low
        )
        
        # ==================== 6. ÍNDICES FÍSICOS COMPUESTOS ====================
        
        # ÍNDICE FÍSICO PRINCIPAL (combina altura, envergadura, peso)
        df_features['physical_rebounding_index'] = (
            df_features['height_rebounding_multiplier'] * 0.4 +      # 40% altura (predictor superior)
            df_features['wingspan_rebounding_multiplier'] * 0.3 +    # 30% envergadura
            df_features['weight_rebounding_multiplier'] * 0.2 +      # 20% peso
            df_features['athleticism_rebounding_multiplier'] * 0.1   # 10% atletismo
        )
        
        # ÍNDICE DE DOMINANCIA FÍSICA (para rebotes disputados)
        df_features['physical_dominance_index'] = (
            df_features['height_advantage_raw'] * 
            df_features['wingspan_advantage'] * 
            df_features['weight_advantage']
        )
        
        # ESPECIALIZACIÓN POR TIPO DE REBOTE
        # Rebotes defensivos: Altura + Peso + Posicionamiento
        df_features['defensive_rebounding_physical_index'] = (
            df_features['height_rebounding_multiplier'] * 0.5 +
            df_features['weight_rebounding_multiplier'] * 0.3 +
            df_features['position_rebounding_weight'] / 3.5 * 0.2  # Normalizado
        )
        
        # Rebotes ofensivos: Atletismo + Agresividad + Envergadura
        df_features['offensive_rebounding_physical_index'] = (
            df_features['athleticism_rebounding_multiplier'] * 0.4 +
            df_features['wingspan_rebounding_multiplier'] * 0.3 +
            df_features['height_rebounding_multiplier'] * 0.3
        )
        
        # ==================== 7. CARACTERÍSTICAS ESPECÍFICAS POR POSICIÓN ====================
        
        # Ventaja específica para Centers (497 en análisis)
        df_features['center_rebounding_advantage'] = (
            df_features['is_center'] * 
            df_features['physical_rebounding_index'] * 1.5  # Bonus para Centers
        )
        
        # Ventaja específica para Forwards (463 en análisis)
        df_features['forward_rebounding_advantage'] = (
            df_features['is_forward'] * 
            df_features['physical_rebounding_index'] * 1.3  # Bonus para Forwards
        )
        
        # Ventaja para híbridos versátiles (352 C-F, 234 F-C)
        df_features['versatile_big_advantage'] = (
            df_features['is_versatile_big'] * 
            df_features['physical_rebounding_index'] * 1.2  # Bonus para versátiles
        )
        
        # Ventaja para Center-Forward (352 en análisis)
        df_features['center_forward_advantage'] = (
            df_features['is_center_forward'] * 
            df_features['physical_rebounding_index'] * 1.25  # Bonus específico
        )
        
        # Ventaja para Forward-Center (234 en análisis)
        df_features['forward_center_advantage'] = (
            df_features['is_forward_center'] * 
            df_features['physical_rebounding_index'] * 1.2  # Bonus específico
        )
        
        logger.debug(f"Calculadas ventajas físicas CRÍTICAS para rebotes")
        logger.debug(f"   Altura promedio: {df_features['height_inches'].mean():.1f} pulgadas")
        logger.debug(f"   Envergadura promedio: {df_features['wingspan_inches'].mean():.1f} pulgadas")
        logger.debug(f"   Peso promedio: {df_features['weight_lbs'].mean():.1f} lbs")
        logger.debug(f"   Índice físico promedio: {df_features['physical_rebounding_index'].mean():.2f}")
        
        return df_features
    
    def _calculate_exposure_and_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula tiempo de exposición y contexto de juego."""
        df_features = df.copy()
        
        # 1. TIEMPO DE EXPOSICIÓN BÁSICO
        if 'MP' in df.columns:
            df_features['minutes_played'] = df['MP']
            
            # Normalizar minutos (48 minutos = juego completo)
            df_features['minutes_pct'] = df_features['minutes_played'] / 48.0
            
            # Categorías de minutos
            df_features['starter'] = (df_features['minutes_played'] >= 25).astype(int)
            df_features['key_player'] = (df_features['minutes_played'] >= 30).astype(int)
            df_features['bench_player'] = (df_features['minutes_played'] < 20).astype(int)
            
            # Tendencia de minutos (últimos 5 juegos)
            df_features['minutes_trend_5g'] = df_features['minutes_played'].rolling(5, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Consistencia en minutos
            df_features['minutes_consistency'] = 1 / (1 + df_features['minutes_played'].rolling(5, min_periods=1).std().fillna(5))
        
        # 2. CONTEXTO DE DESCANSO
        if 'Date' in df.columns:
            df_features['Date'] = pd.to_datetime(df_features['Date'])
            
            # CALCULAR DÍAS DE DESCANSO POR JUGADOR (NO GLOBAL)
            df_features['days_rest'] = 2  # Valor por defecto
            df_features['back_to_back'] = 0
            df_features['well_rested'] = 0
            
            # Procesar por jugador para evitar contaminar datos entre jugadores
            if 'Player' in df_features.columns:
                for player in df_features['Player'].unique():
                    player_mask = df_features['Player'] == player
                    player_data = df_features[player_mask].sort_values('Date')
                    
                    if len(player_data) > 1:
                        days_diff = player_data['Date'].diff().dt.days.fillna(2)
                        
                        # Asignar días de descanso solo para este jugador usando índices correctos
                        df_features.loc[player_data.index, 'days_rest'] = days_diff.values
                        df_features.loc[player_data.index, 'back_to_back'] = (days_diff <= 1).astype(int).values
                        df_features.loc[player_data.index, 'well_rested'] = (days_diff >= 3).astype(int).values
        
        # 3. CONTEXTO HOME/AWAY (usar columna is_home ya procesada por data_loader)
        if 'is_home' in df.columns:
            df_features['is_home'] = df['is_home']
        elif 'Away' in df.columns:
            # Away es '@' cuando es visitante, '' cuando es local
            df_features['is_home'] = (df['Away'] != '@').astype(int)
        else:
            df_features['is_home'] = 1  # Default: asumir local
        
        # 4. ACTIVIDAD DEFENSIVA (proxy para esfuerzo)
        activity_stats = []
        for stat in ['STL', 'BLK', 'PF']:
            if stat in df.columns:
                activity_stats.append(stat)
        
        if len(activity_stats) >= 2:
            df_features['defensive_activity'] = df[activity_stats].sum(axis=1)
            df_features['high_effort_game'] = (df_features['defensive_activity'] >= 3).astype(int)
        
        logger.debug(f"Calculado tiempo de exposición y contexto")
        
        return df_features
    
    def _create_rebounding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características específicas para rebotes combinando todos los factores."""
        df_features = df.copy()
        
        # 1. OPORTUNIDADES PONDERADAS POR CAPACIDAD FÍSICA
        if 'total_rebound_opportunities' in df_features.columns and 'physical_rebounding_index' in df_features.columns:
            df_features['weighted_rebound_opportunities'] = (
                df_features['total_rebound_opportunities'] * 
                df_features['physical_rebounding_index']
            )
            
            df_features['weighted_close_opportunities'] = (
                df_features.get('total_close_opportunities', 0) * 
                df_features['physical_rebounding_index'] * 1.2  # Favorece más a jugadores altos
            )
            
            df_features['weighted_long_opportunities'] = (
                df_features.get('total_long_opportunities', 0) * 
                df_features['physical_rebounding_index'] * 0.8  # Menos dependiente de altura
            )
        
        # 2. OPORTUNIDADES POR MINUTO (TASA DE EXPOSICIÓN)
        if 'minutes_played' in df_features.columns and 'total_rebound_opportunities' in df_features.columns:
            df_features['opportunities_per_minute'] = np.where(
                df_features['minutes_played'] > 0,
                df_features['total_rebound_opportunities'] / df_features['minutes_played'],
                0
            )
            
            df_features['weighted_opportunities_per_minute'] = np.where(
                df_features['minutes_played'] > 0,
                df_features.get('weighted_rebound_opportunities', 0) / df_features['minutes_played'],
                0
            )
        
        # 3. ÍNDICE DE REBOTE ESPERADO
        base_rebounding_rate = 0.12  # ~12% de oportunidades se convierten en rebotes para jugador promedio
        
        if all(col in df_features.columns for col in ['weighted_rebound_opportunities', 'minutes_pct']):
            df_features['expected_rebounds'] = (
                df_features['weighted_rebound_opportunities'] * 
                base_rebounding_rate * 
                df_features.get('minutes_pct', 1.0)
            )
        
        # 4. FACTORES DE AJUSTE POR CONTEXTO
        context_multiplier = 1.0
        
        # Ajuste por descanso
        if 'back_to_back' in df_features.columns:
            context_multiplier *= np.where(df_features['back_to_back'] == 1, 0.95, 1.0)
        
        if 'well_rested' in df_features.columns:
            context_multiplier *= np.where(df_features['well_rested'] == 1, 1.05, 1.0)
        
        # Ajuste por local/visitante
        if 'is_home' in df_features.columns:
            context_multiplier *= np.where(df_features['is_home'] == 1, 1.02, 0.98)
        
        # Aplicar ajustes de contexto
        if 'expected_rebounds' in df_features.columns:
            df_features['context_adjusted_rebounds'] = df_features['expected_rebounds'] * context_multiplier
        
        # 5. CARACTERÍSTICAS DE MOMENTUM
        if 'minutes_played' in df_features.columns:
            # Momentum de minutos (indica confianza del entrenador)
            df_features['minutes_momentum'] = (
                df_features['minutes_played'].rolling(3, min_periods=1).mean() - 
                df_features['minutes_played'].expanding().mean()
            )
            
            # Estabilidad en el rol
            df_features['role_stability'] = df_features.get('minutes_consistency', 1.0)
        
        logger.debug(f"Creadas características específicas de rebotes")
        
        return df_features
    
    def _convert_categorical_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte variables categóricas a numéricas."""
        df_numeric = df.copy()
        
        # Convertir columnas categóricas comunes
        categorical_columns = ['Team', 'Opp', 'Pos', 'Player']
        
        for col in categorical_columns:
            if col in df_numeric.columns:
                if df_numeric[col].dtype == 'object':
                    # Usar label encoding para convertir a números
                    unique_values = df_numeric[col].unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    df_numeric[col] = df_numeric[col].map(value_map)
                    logger.debug(f"Convertida columna categórica '{col}' a numérica ({len(unique_values)} valores únicos)")
        
        # Convertir cualquier otra columna object a numérica
        for col in df_numeric.columns:
            if df_numeric[col].dtype == 'object':
                try:
                    # Intentar conversión directa a numérico
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                    logger.debug(f"   Convertida columna '{col}' a numérica")
                except:
                    # Si falla, usar label encoding
                    unique_values = df_numeric[col].dropna().unique()
                    if len(unique_values) > 0:
                        value_map = {val: idx for idx, val in enumerate(unique_values)}
                        df_numeric[col] = df_numeric[col].map(value_map)
                        logger.debug(f"   Convertida columna categórica '{col}' a numérica usando label encoding")
        
        # Rellenar valores NaN con 0
        df_numeric = df_numeric.fillna(0)
        
        return df_numeric

    def generate_game_by_game_features(self, df: pd.DataFrame, target_col: str = 'TRB', 
                                      min_games: int = 10, teams_df: Optional[pd.DataFrame] = None) -> tuple:
        """
        Genera características juego por juego para entrenamiento.
        
        Args:
            df: DataFrame con datos históricos
            target_col: Columna objetivo (TRB)
            min_games: Mínimo de juegos históricos requeridos
            teams_df: DataFrame opcional con datos de equipos
            
        Returns:
            Tuple (X, y, feature_names) con características y targets
        """
        logger.info("🏀 Generando características FUNDAMENTALES juego por juego...")
        
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
        
        # Ordenar por jugador y fecha
        df_sorted = df.sort_values(['Player', 'Date']).copy()
        
        X_list = []
        y_list = []
        
        total_players = len(df_sorted['Player'].unique())
        processed_players = 0
        
        # Procesar cada jugador
        for player in df_sorted['Player'].unique():
            player_data = df_sorted[df_sorted['Player'] == player].copy()
            
            if len(player_data) < min_games:
                continue
            
            processed_players += 1
            
            # Log progreso cada 50 jugadores
            if processed_players % 50 == 0 or processed_players == total_players:
                logger.info(f"📊 Procesando jugadores: {processed_players}/{total_players}")
            
            # Generar características fundamentales para este jugador (sin logging repetitivo)
            player_features = self._generate_fundamental_features_silent(player_data, teams_df)
            
            # Para cada juego (excepto los primeros min_games-1)
            for i in range(min_games-1, len(player_features)):
                current_game = player_features.iloc[i]
                
                # EXTRAER TARGET ANTES DE FILTRAR CARACTERÍSTICAS
                original_game = player_data.iloc[i]
                target_value = original_game[target_col]
                
                # Crear vector de características FUNDAMENTALES
                feature_vector = {}
                
                # CARACTERÍSTICAS FUNDAMENTALES ESPECÍFICAS
                fundamental_features = [
                    # CARACTERÍSTICAS FÍSICAS CRÍTICAS (NUEVA PRIORIDAD MÁXIMA)
                    'height_inches', 'wingspan_inches', 'weight_lbs', 'vertical_leap_inches',
                    'height_advantage_raw', 'wingspan_advantage', 'weight_advantage',
                    'height_rebounding_multiplier', 'wingspan_rebounding_multiplier', 'weight_rebounding_multiplier',
                    'athleticism_rebounding_multiplier', 'physical_rebounding_index', 'physical_dominance_index',
                    'height_tier_elite', 'height_tier_very_tall', 'height_tier_tall',
                    'wingspan_elite', 'wingspan_very_long', 'wingspan_long',
                    'weight_elite', 'weight_heavy', 'weight_solid',
                    'athletic_elite', 'athletic_high', 'athletic_average',
                    'defensive_rebounding_physical_index', 'offensive_rebounding_physical_index',
                    'center_rebounding_advantage', 'forward_rebounding_advantage', 'versatile_big_advantage',
                    'center_forward_advantage', 'forward_center_advantage',
                    'is_center_forward', 'is_forward_center', 'is_forward_guard', 'is_guard_forward',
                    'has_forward_skills', 'has_guard_skills',
                    
                    # CARACTERÍSTICAS ESTADÍSTICAS CRÍTICAS (NUEVA PRIORIDAD ALTA)
                    'historical_rpg_10g', 'historical_rpg_5g', 'historical_rpg_3g', 'season_rpg_avg',
                    'rpg_trend_10g', 'rpg_trend_5g', 'rpg_consistency_10g', 'rpg_consistency_5g',
                    'rpg_max_10g', 'rpg_min_10g', 'rpg_range_10g', 'rpg_percentile_75', 'rpg_percentile_25',
                    'historical_oreb_avg', 'historical_dreb_avg', 'oreb_dreb_ratio',
                    'offensive_rebounding_specialist', 'defensive_rebounding_specialist',
                    'historical_mpg_10g', 'historical_mpg_5g', 'season_mpg_avg', 'mpg_trend_10g', 'mpg_consistency',
                    'starter_role', 'key_player_role', 'star_player_role', 'bench_player_role',
                    'rebounds_per_minute', 'rebounds_per_36min',
                    'historical_bpg_10g', 'historical_bpg_5g', 'elite_shot_blocker', 'good_shot_blocker', 'average_shot_blocker',
                    'historical_spg_10g', 'historical_spg_5g', 'high_defensive_activity',
                    'historical_pf_10g', 'physical_aggressiveness',
                    'historical_fg_pct_10g', 'historical_fg_pct_5g', 'elite_fg_pct', 'high_fg_pct', 'paint_presence',
                    'elite_rebounding_index', 'total_consistency_index', 'positive_momentum_index',
                    
                    # OPORTUNIDADES (40% peso) - AHORA CON DATOS REALES
                    'own_missed_shots', 'own_missed_3pt', 'own_missed_2pt',
                    'total_rebound_opportunities', 'total_long_opportunities', 'total_close_opportunities',
                    'weighted_rebound_opportunities', 'weighted_close_opportunities', 'weighted_long_opportunities',
                    'opportunities_per_minute', 'weighted_opportunities_per_minute',
                    
                    # DATOS REALES DEL OPONENTE (CRÍTICOS)
                    'opp_real_missed_shots', 'opp_real_missed_3pt', 'opp_real_missed_2pt',
                    'opp_real_fg_pct', 'opp_real_3pt_pct', 'has_real_opp_data',
                    'opp_defensive_weakness', 'opp_3pt_tendency', 'quality_weighted_opportunities',
                    
                    # CARACTERÍSTICAS DE EQUIPOS (CRÍTICAS PARA CONTEXTO)
                    'team_missed_shots', 'opp_missed_shots', 'game_pace', 'team_pace',
                    'defensive_rating_approx', 'offensive_rating_approx', 'team_3pt_rate', 'opp_3pt_rate',
                    'total_rebound_index', 'offensive_rebound_index', 'defensive_rebound_index',
                    'final_rebound_opportunities', 'opponent_quality_factor', 'team_style_multiplier',
                    'high_scoring_game', 'defensive_game', 'three_point_heavy_game',
                    
                    # CARACTERÍSTICAS HÍBRIDAS (COMBINAN JUGADOR + EQUIPO)
                    'player_team_missed_shots_ratio', 'player_adjusted_total_opportunities',
                    'pace_adjusted_exposure', 'pace_adjusted_opps_per_minute',
                    'context_amplified_physical_advantage', 'physical_dominance_index',
                    'hybrid_expected_rebounds', 'prediction_confidence_factor',
                    'hybrid_momentum', 'positive_trend_indicator',
                    
                    # CAPACIDAD FÍSICA (35% peso)
                    'position_rebounding_weight', 'is_center', 'is_power_forward', 'is_big_man', 'is_guard',
                    'height_rebounding_multiplier', 'physical_rebounding_index',
                    
                    # EXPOSICIÓN (15% peso)
                    'minutes_played', 'minutes_pct', 'starter', 'key_player', 'bench_player',
                    'minutes_trend_5g', 'minutes_consistency', 'minutes_momentum',
                    
                    # CONTEXTO (10% peso)
                    'days_rest', 'back_to_back', 'well_rested', 'is_home',
                    'expected_rebounds', 'context_adjusted_rebounds'
                ]
                
                # Usar solo características fundamentales disponibles
                for feature in fundamental_features:
                    if feature in player_features.columns:
                        feature_vector[feature] = current_game[feature]
                
                # Agregar algunas características básicas si están disponibles
                basic_features = ['FGA', 'FG', '3PA', '3P', 'STL', 'BLK', 'PF']
                for feature in basic_features:
                    if feature in player_features.columns and feature not in self.forbidden_features:
                        feature_vector[feature] = current_game[feature]
                
                X_list.append(feature_vector)
                y_list.append(target_value)
        
        if not X_list:
            raise ValueError("No se pudieron generar características")
        
        # Convertir a DataFrame
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        
        # CONVERSIÓN CRÍTICA: Convertir variables categóricas a numéricas
        logger.info("🔧 Convirtiendo variables categóricas a numéricas...")
        X = self._convert_categorical_to_numeric(X)
        
        # Limpiar características finales
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Verificar que todas las columnas son numéricas
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Eliminando columnas no numéricas: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)
        
        feature_names = list(X.columns)
        
        logger.info(f"✅ Generadas {len(feature_names)} características FUNDAMENTALES para {len(X)} juegos")
        logger.info(f"📊 Distribución objetivo - Min: {y.min()}, Max: {y.max()}, Media: {y.mean():.2f}")
        
        # Mostrar características más importantes
        opportunity_features = [f for f in feature_names if any(x in f for x in ['missed', 'opportunity', 'rebound'])]
        physical_features = [f for f in feature_names if any(x in f for x in ['position', 'height', 'physical', 'big_man', 'center'])]
        exposure_features = [f for f in feature_names if any(x in f for x in ['minutes', 'starter', 'key_player'])]
        
        logger.info(f"🎯 Características de oportunidades: {len(opportunity_features)}")
        logger.info(f"💪 Características físicas: {len(physical_features)}")
        logger.info(f"⏱️  Características de exposición: {len(exposure_features)}")
        
        return X.values, y, feature_names
    
    def _generate_fundamental_features_silent(self, df: pd.DataFrame, teams_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Genera características fundamentales sin logging verboso (para uso interno)."""
        # Crear copia para no modificar original
        df_features = df.copy()
        
        # Verificar columnas requeridas
        required_cols = ['Player', 'Date']
        missing_cols = [col for col in required_cols if col not in df_features.columns]
        if missing_cols:
            raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")
        
        # Convertir Date a datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
            df_features['Date'] = pd.to_datetime(df_features['Date'], errors='coerce')
        
        # 1. Calcular oportunidades de rebote (40% del peso)
        df_features = self._calculate_missed_shots_opportunities(df_features, teams_df)
        
        # 2. Calcular ventajas físicas (35% del peso)
        df_features = self._calculate_physical_advantages(df_features)
        
        # 3. Calcular exposición y contexto (15% del peso)
        df_features = self._calculate_exposure_and_context(df_features)
        
        # 4. Crear características específicas de rebotes (10% del peso)
        df_features = self._create_rebounding_features(df_features)
        
        # 5. CARACTERÍSTICAS ESTADÍSTICAS CRÍTICAS (NUEVO - BASADO EN ANÁLISIS)
        df_features = self._calculate_advanced_statistical_features(df_features)
        
        # ELIMINAR CARACTERÍSTICAS PROHIBIDAS (DESPUÉS de calcular estadísticas históricas)
        forbidden_in_features = [col for col in df_features.columns if col in self.forbidden_features]
        if forbidden_in_features:
            df_features = df_features.drop(columns=forbidden_in_features)
        
        # INTEGRAR CARACTERÍSTICAS DE EQUIPOS (CRÍTICO PARA PRECISIÓN)
        if teams_df is not None:
            try:
                from .team_features_trb import TeamReboundingFeatures
                
                # Crear instancia solo una vez y reutilizar
                if not hasattr(self, '_team_feature_generator'):
                    self._team_feature_generator = TeamReboundingFeatures()
                
                df_features = self._team_feature_generator._generate_team_context_features_silent(df_features, teams_df)
                
                # Crear características híbridas que combinen datos individuales y de equipos
                df_features = self._create_hybrid_team_player_features(df_features)
                
            except Exception as e:
                logger.warning(f"Error integrando características de equipos: {e}")
                logger.debug("Continuando sin características de equipos...")
        else:
            logger.debug("No hay datos de equipos disponibles")
        
        # LIMPIAR DATOS FINALES
        df_features = df_features.fillna(0)
        df_features = df_features.replace([np.inf, -np.inf], 0)
        
        return df_features
    
    def generate_fundamental_features(self, df: pd.DataFrame, teams_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Genera características fundamentales para rebotes."""
        logger.info("Generando características fundamentales para rebotes...")
        
        # Crear copia para no modificar original
        df_features = df.copy()
        
        # Verificar columnas requeridas
        required_cols = ['Player', 'Date']
        missing_cols = [col for col in required_cols if col not in df_features.columns]
        if missing_cols:
            raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")
        
        # Convertir Date a datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
            df_features['Date'] = pd.to_datetime(df_features['Date'], errors='coerce')
        
        # 1. Calcular oportunidades de rebote (40% del peso)
        df_features = self._calculate_missed_shots_opportunities(df_features, teams_df)
        logger.debug("Oportunidades de rebote calculadas")
        
        # 2. Calcular ventajas físicas (35% del peso)
        df_features = self._calculate_physical_advantages(df_features)
        logger.debug("Ventajas físicas calculadas")
        
        # 3. Calcular exposición y contexto (15% del peso)
        df_features = self._calculate_exposure_and_context(df_features)
        logger.debug("Exposición y contexto calculados")
        
        # 4. Crear características específicas de rebotes (10% del peso)
        df_features = self._create_rebounding_features(df_features)
        logger.debug("Características específicas de rebotes creadas")
        
        # 5. CARACTERÍSTICAS ESTADÍSTICAS CRÍTICAS (NUEVO - BASADO EN ANÁLISIS)
        df_features = self._calculate_advanced_statistical_features(df_features)
        logger.debug("Características estadísticas críticas calculadas")
        
        # ELIMINAR CARACTERÍSTICAS PROHIBIDAS (DESPUÉS de calcular estadísticas históricas)
        forbidden_in_features = [col for col in df_features.columns if col in self.forbidden_features]
        if forbidden_in_features:
            logger.debug(f"Eliminando características prohibidas: {forbidden_in_features}")
            df_features = df_features.drop(columns=forbidden_in_features)
        
        # INTEGRAR CARACTERÍSTICAS DE EQUIPOS (CRÍTICO PARA PRECISIÓN)
        if teams_df is not None:
            try:
                from .team_features_trb import TeamReboundingFeatures
                
                # Crear instancia solo una vez y reutilizar
                if not hasattr(self, '_team_feature_generator'):
                    self._team_feature_generator = TeamReboundingFeatures()
                
                df_features = self._team_feature_generator.generate_team_context_features(df_features, teams_df)
                logger.debug("✅ Características de equipos integradas exitosamente")
                
                # Crear características híbridas que combinen datos individuales y de equipos
                df_features = self._create_hybrid_team_player_features(df_features)
                logger.debug("Características híbridas jugador-equipo creadas")
                
            except Exception as e:
                logger.warning(f"Error integrando características de equipos: {e}")
                logger.debug("Continuando sin características de equipos...")
        else:
            logger.debug("No hay datos de equipos disponibles")
        
        # LIMPIAR DATOS FINALES
        df_features = df_features.fillna(0)
        df_features = df_features.replace([np.inf, -np.inf], 0)
        
        # Contar características generadas
        feature_cols = [col for col in df_features.columns if col not in ['Player', 'Date', 'Team', 'Opp']]
        
        # Log final consolidado
        logger.info(f"✅ Características fundamentales generadas: {len(feature_cols)} para {df_features.shape[0]} juegos")
        logger.debug(f"Forma final del dataset: {df_features.shape}")
        
        return df_features
    
    def _create_hybrid_team_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características híbridas que combinan datos individuales y de equipos."""
        df_hybrid = df.copy()
        
        # 1. OPORTUNIDADES HÍBRIDAS (Combinando datos individuales y de equipos)
        if all(col in df_hybrid.columns for col in ['own_missed_shots', 'team_missed_shots']):
            # Proporción de tiros fallados del jugador vs equipo
            df_hybrid['player_team_missed_shots_ratio'] = np.where(
                df_hybrid['team_missed_shots'] > 0,
                df_hybrid['own_missed_shots'] / df_hybrid['team_missed_shots'],
                0.0
            )
            
            # Oportunidades totales ajustadas por participación del jugador
            df_hybrid['player_adjusted_total_opportunities'] = (
                df_hybrid.get('total_rebound_opportunities', 0) * 
                (1 + df_hybrid['player_team_missed_shots_ratio'] * 0.2)
            )
        
        # 2. CONTEXTO DE PACE AJUSTADO POR MINUTOS
        if all(col in df_hybrid.columns for col in ['minutes_played', 'game_pace']):
            # Exposición a oportunidades ajustada por pace del juego
            df_hybrid['pace_adjusted_exposure'] = (
                df_hybrid['minutes_played'] / 48.0 * 
                df_hybrid.get('game_pace', 95.0) / 95.0
            )
            
            # Oportunidades por minuto ajustadas por pace
            df_hybrid['pace_adjusted_opps_per_minute'] = (
                df_hybrid.get('opportunities_per_minute', 0) * 
                df_hybrid.get('game_pace', 95.0) / 95.0
            )
        
        # 3. VENTAJA FÍSICA EN CONTEXTO DE EQUIPO
        if all(col in df_hybrid.columns for col in ['physical_rebounding_index', 'defensive_rating_approx']):
            # Ventaja física amplificada contra defensa débil
            df_hybrid['context_amplified_physical_advantage'] = (
                df_hybrid['physical_rebounding_index'] * 
                df_hybrid.get('opponent_quality_factor', 1.0)
            )
            
            # Índice de dominancia física en el contexto del juego
            df_hybrid['physical_dominance_index'] = (
                df_hybrid['physical_rebounding_index'] * 
                df_hybrid.get('total_rebound_index', 1.0) / 100.0
            )
        
        # 4. MÉTRICAS DE EFICIENCIA CONTEXTUAL
        if all(col in df_hybrid.columns for col in ['expected_rebounds', 'final_rebound_opportunities']):
            # Eficiencia esperada combinando factores individuales y de equipo
            df_hybrid['hybrid_expected_rebounds'] = (
                df_hybrid['expected_rebounds'] * 0.6 + 
                df_hybrid.get('final_rebound_opportunities', 0) * 0.4 / 100.0
            )
            
            # Factor de confianza basado en datos disponibles
            df_hybrid['prediction_confidence_factor'] = (
                df_hybrid.get('has_real_opp_data', 0) * 0.3 +  # Datos reales del oponente
                (df_hybrid.get('minutes_consistency', 0) > 0.8) * 0.2 +  # Consistencia en minutos
                (df_hybrid.get('is_big_man', 0)) * 0.3 +  # Ventaja posicional
                0.2  # Base
            )
        
        # 5. CARACTERÍSTICAS DE MOMENTUM HÍBRIDAS
        if 'minutes_momentum' in df_hybrid.columns and 'team_missed_shots_trend_5g' in df_hybrid.columns:
            # Momentum combinado jugador-equipo
            df_hybrid['hybrid_momentum'] = (
                df_hybrid['minutes_momentum'] * 0.6 + 
                df_hybrid.get('team_missed_shots_trend_5g', 0) * 0.4
            )
            
            # Indicador de tendencia positiva
            df_hybrid['positive_trend_indicator'] = (
                (df_hybrid['minutes_momentum'] > 0) & 
                (df_hybrid.get('team_missed_shots_trend_5g', 0) > 0)
            ).astype(int)
        
        logger.debug("Características híbridas jugador-equipo creadas")
        return df_hybrid

    def _create_ultra_specific_rebound_features(self, player_data: pd.DataFrame, 
                                              game_data: pd.Series, teams_df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Crea características ULTRA-ESPECÍFICAS de rebotes basadas en la mecánica real del baloncesto.
        
        FUNDAMENTOS REALES DE LOS REBOTES:
        1. OPORTUNIDADES = Tiros fallados del equipo + oponente
        2. POSICIONAMIENTO = Altura + Posición + Minutos en cancha
        3. AGRESIVIDAD = Faltas + Robos + Bloqueos (intensidad física)
        4. EFICIENCIA = Ratio rebotes/oportunidades históricas
        5. CONTEXTO = Ritmo del juego + diferencia de score + fatiga
        """
        
        features = {}
        
        # ==================== OPORTUNIDADES DE REBOTE REALES ====================
        
        # 1. Tiros fallados del equipo (rebotes ofensivos)
        if 'FGA' in player_data.columns and 'FG' in player_data.columns:
            team_missed_shots = (player_data['FGA'] - player_data['FG']).rolling(5, min_periods=1).mean()
            features['team_missed_shots_avg'] = team_missed_shots.iloc[-1] if len(team_missed_shots) > 0 else 0
            
            # Tiros fallados por partido del equipo
            features['team_missed_shots_per_game'] = features['team_missed_shots_avg'] * 5  # Estimación equipo completo
        else:
            features['team_missed_shots_avg'] = 0
            features['team_missed_shots_per_game'] = 0
        
        # 2. Tiros de 3 puntos fallados (rebotes largos)
        if '3PA' in player_data.columns and '3P' in player_data.columns:
            three_point_misses = (player_data['3PA'] - player_data['3P']).rolling(3, min_periods=1).mean()
            features['three_point_misses_avg'] = three_point_misses.iloc[-1] if len(three_point_misses) > 0 else 0
            
            # Los rebotes de 3 puntos son más largos y favorecen a jugadores altos
            features['long_rebound_opportunities'] = features['three_point_misses_avg'] * 1.5
        else:
            features['three_point_misses_avg'] = 0
            features['long_rebound_opportunities'] = 0
        
        # 3. Tiros libres fallados (rebotes controlados)
        if 'FTA' in player_data.columns and 'FT' in player_data.columns:
            ft_misses = (player_data['FTA'] - player_data['FT']).rolling(3, min_periods=1).mean()
            features['free_throw_misses_avg'] = ft_misses.iloc[-1] if len(ft_misses) > 0 else 0
        else:
            features['free_throw_misses_avg'] = 0
        
        # 4. OPORTUNIDADES TOTALES DE REBOTE
        features['total_rebound_opportunities'] = (
            features['team_missed_shots_per_game'] + 
            features['long_rebound_opportunities'] + 
            features['free_throw_misses_avg']
        )
        
        # ==================== CAPACIDAD FÍSICA PARA REBOTES ====================
        
        # 5. Índice de altura/posición (fundamental para rebotes)
        height_bonus = self._get_height_advantage_for_player(game_data.get('Player', ''))
        position_bonus = self._get_position_rebound_bonus(game_data.get('Pos', ''))
        
        features['physical_rebound_index'] = height_bonus * position_bonus
        features['height_advantage'] = height_bonus
        features['position_rebound_factor'] = position_bonus
        
        # 6. Tiempo de exposición (minutos jugados)
        if 'MP' in player_data.columns:
            minutes_avg = player_data['MP'].rolling(5, min_periods=1).mean()
            features['minutes_exposure'] = minutes_avg.iloc[-1] if len(minutes_avg) > 0 else 0
            
            # Exposición ajustada por capacidad física
            features['effective_exposure'] = features['minutes_exposure'] * features['physical_rebound_index']
        else:
            features['minutes_exposure'] = 0
            features['effective_exposure'] = 0
        
        # ==================== AGRESIVIDAD Y INTENSIDAD ====================
        
        # 7. Índice de agresividad (correlaciona con rebotes)
        aggressiveness = 0
        if 'PF' in player_data.columns:  # Faltas personales
            pf_avg = player_data['PF'].rolling(5, min_periods=1).mean()
            aggressiveness += pf_avg.iloc[-1] if len(pf_avg) > 0 else 0
        
        if 'STL' in player_data.columns:  # Robos
            stl_avg = player_data['STL'].rolling(5, min_periods=1).mean()
            aggressiveness += stl_avg.iloc[-1] * 2 if len(stl_avg) > 0 else 0  # Robos valen más
        
        if 'BLK' in player_data.columns:  # Bloqueos
            blk_avg = player_data['BLK'].rolling(5, min_periods=1).mean()
            aggressiveness += blk_avg.iloc[-1] * 3 if len(blk_avg) > 0 else 0  # Bloqueos valen más
        
        features['aggressiveness_index'] = aggressiveness
        
        # 8. Intensidad defensiva (correlaciona con rebotes defensivos)
        defensive_intensity = features['aggressiveness_index'] * features['physical_rebound_index']
        features['defensive_intensity'] = defensive_intensity
        
        # ==================== EFICIENCIA HISTÓRICA DE REBOTES ====================
        
        # 9. Ratio rebotes/oportunidades histórico
        if 'TRB' in player_data.columns and len(player_data) >= 3:
            historical_rebounds = player_data['TRB'].rolling(10, min_periods=3).mean()
            features['historical_rebound_avg'] = historical_rebounds.iloc[-1]
            
            # Eficiencia de rebotes (rebotes por minuto jugado)
            if features['minutes_exposure'] > 0:
                features['rebound_efficiency'] = features['historical_rebound_avg'] / features['minutes_exposure']
            else:
                features['rebound_efficiency'] = 0
            
            # Tendencia de rebotes (¿está mejorando o empeorando?)
            recent_rebounds = player_data['TRB'].tail(3).mean()
            older_rebounds = player_data['TRB'].head(max(1, len(player_data)-3)).mean()
            features['rebound_trend'] = recent_rebounds - older_rebounds
            
        else:
            features['historical_rebound_avg'] = 0
            features['rebound_efficiency'] = 0
            features['rebound_trend'] = 0
        
        # ==================== CONTEXTO DEL JUEGO ====================
        
        # 10. Ritmo del juego (más posesiones = más rebotes)
        if teams_df is not None:
            pace_factor = self._get_game_pace_factor(game_data, teams_df)
            features['game_pace_factor'] = pace_factor
            
            # Rebotes esperados por ritmo
            features['pace_adjusted_rebounds'] = features['total_rebound_opportunities'] * pace_factor
        else:
            features['game_pace_factor'] = 1.0
            features['pace_adjusted_rebounds'] = features['total_rebound_opportunities']
        
        # 11. Factor de fatiga (juegos consecutivos)
        fatigue_factor = self._calculate_fatigue_factor(player_data, game_data)
        features['fatigue_factor'] = fatigue_factor
        
        # 12. Ventaja de local/visitante para rebotes
        is_home = not game_data.get('@', False)
        features['home_court_advantage'] = 1.1 if is_home else 0.9
        
        # ==================== CARACTERÍSTICAS HÍBRIDAS ULTRA-ESPECÍFICAS ====================
        
        # 13. PREDICTOR PRINCIPAL: Rebotes esperados basado en fundamentos
        expected_rebounds = (
            features['total_rebound_opportunities'] * 0.4 +  # 40% oportunidades
            features['physical_rebound_index'] * 2.0 +       # 30% capacidad física  
            features['effective_exposure'] * 0.3 +           # 20% exposición
            features['aggressiveness_index'] * 0.5           # 10% agresividad
        ) * features['game_pace_factor'] * features['home_court_advantage'] * features['fatigue_factor']
        
        features['expected_rebounds_fundamental'] = max(0, expected_rebounds)
        
        # 14. Rebotes ofensivos vs defensivos (diferentes mecánicas)
        offensive_rebound_factor = features['aggressiveness_index'] * 0.3  # Más agresividad para ofensivos
        defensive_rebound_factor = features['physical_rebound_index'] * 0.7  # Más altura para defensivos
        
        features['offensive_rebound_potential'] = offensive_rebound_factor
        features['defensive_rebound_potential'] = defensive_rebound_factor
        
        # 15. Factor de consistencia (jugadores consistentes son más predecibles)
        if 'TRB' in player_data.columns and len(player_data) >= 5:
            rebound_std = player_data['TRB'].rolling(10, min_periods=5).std()
            consistency = 1.0 / (1.0 + rebound_std.iloc[-1]) if len(rebound_std) > 0 else 0.5
            features['consistency_factor'] = consistency
        else:
            features['consistency_factor'] = 0.5
        
        # 16. PREDICTOR FINAL ULTRA-ESPECÍFICO
        final_prediction_base = (
            features['expected_rebounds_fundamental'] * features['consistency_factor'] +
            features['historical_rebound_avg'] * 0.3 +
            features['rebound_trend'] * 0.1
        )
        
        features['ultra_specific_rebound_prediction'] = max(0, final_prediction_base)
        
        # ==================== CARACTERÍSTICAS DE INTERACCIÓN ESPECÍFICAS ====================
        
        # 17. Interacciones multiplicativas específicas de rebotes
        features['height_x_minutes'] = features['height_advantage'] * features['minutes_exposure']
        features['aggressiveness_x_opportunities'] = features['aggressiveness_index'] * features['total_rebound_opportunities']
        features['efficiency_x_pace'] = features['rebound_efficiency'] * features['game_pace_factor']
        features['physical_x_fatigue'] = features['physical_rebound_index'] * features['fatigue_factor']
        
        # 18. Ratios específicos de rebotes
        if features['total_rebound_opportunities'] > 0:
            features['rebound_capture_rate'] = features['historical_rebound_avg'] / features['total_rebound_opportunities']
        else:
            features['rebound_capture_rate'] = 0
        
        if features['minutes_exposure'] > 0:
            features['rebounds_per_minute'] = features['historical_rebound_avg'] / features['minutes_exposure']
        else:
            features['rebounds_per_minute'] = 0
        
        # 19. Características específicas por tipo de rebote
        features['long_rebound_specialist'] = features['height_advantage'] * features['long_rebound_opportunities']
        features['close_rebound_specialist'] = features['aggressiveness_index'] * (features['total_rebound_opportunities'] - features['long_rebound_opportunities'])
        
        # 20. Factor de dominancia en rebotes
        dominance_factor = (
            features['physical_rebound_index'] * 
            features['aggressiveness_index'] * 
            features['rebound_efficiency']
        )
        features['rebound_dominance_factor'] = dominance_factor
        
        return features

    def _get_height_advantage_for_player(self, height_inches: float) -> float:
        """Calcula ventaja de altura específica para rebotes basada en altura directa."""
        if pd.isna(height_inches):
            return 1.5  # Valor por defecto conservador
            
        # Ventaja exponencial por altura para rebotes (usando datos ya procesados)
        if height_inches >= 84:  # 7'0" o más (Zubac, Embiid)
            return 3.0
        elif height_inches >= 81:  # 6'9" - 6'11" (Jokić, Sabonis)
            return 2.5
        elif height_inches >= 78:  # 6'6" - 6'8" (Antetokounmpo)
            return 2.0
        elif height_inches >= 75:  # 6'3" - 6'5"
            return 1.5
        else:  # Menos de 6'3"
            return 1.0
    
    def _get_position_rebound_bonus(self, position: str) -> float:
        """Calcula bonus de posición específico para rebotes."""
        if not position:
            return 1.5
        
        position = str(position).upper()
        
        # Bonus específico por posición para rebotes
        if 'C' in position:  # Centro
            return 3.0
        elif 'PF' in position or 'F-C' in position:  # Power Forward
            return 2.5
        elif 'SF' in position or 'F' in position:  # Small Forward
            return 2.0
        elif 'SG' in position or 'G-F' in position:  # Shooting Guard
            return 1.5
        elif 'PG' in position or 'G' in position:  # Point Guard
            return 1.0
        else:
            return 1.5
    
    def _get_game_pace_factor(self, game_data: pd.Series, teams_df: pd.DataFrame) -> float:
        """Calcula factor de ritmo del juego para rebotes."""
        try:
            team = game_data.get('Team', '')
            opponent = game_data.get('Opp', '')
            
            if teams_df is not None and len(teams_df) > 0:
                # Obtener pace de ambos equipos
                team_pace = teams_df[teams_df['Team'] == team]['Pace'].values
                opp_pace = teams_df[teams_df['Team'] == opponent]['Pace'].values
                
                if len(team_pace) > 0 and len(opp_pace) > 0:
                    avg_pace = (team_pace[0] + opp_pace[0]) / 2
                    # Normalizar pace (100 es promedio NBA)
                    pace_factor = avg_pace / 100.0
                    return max(0.8, min(1.3, pace_factor))  # Limitar entre 0.8 y 1.3
            
            return 1.0  # Factor neutro si no hay datos
            
        except Exception:
            return 1.0
    
    def _calculate_fatigue_factor(self, player_data: pd.DataFrame, game_data: pd.Series) -> float:
        """Calcula factor de fatiga basado en juegos consecutivos y minutos."""
        try:
            if 'Date' not in player_data.columns or len(player_data) < 2:
                return 1.0
            
            # Ordenar por fecha
            sorted_data = player_data.sort_values('Date')
            
            # Calcular días de descanso
            last_game_date = sorted_data['Date'].iloc[-1]
            second_last_date = sorted_data['Date'].iloc[-2]
            
            days_rest = (last_game_date - second_last_date).days
            
            # Factor de fatiga basado en descanso
            if days_rest >= 3:
                rest_factor = 1.1  # Bien descansado
            elif days_rest == 2:
                rest_factor = 1.0  # Normal
            elif days_rest == 1:
                rest_factor = 0.95  # Poco descanso
            else:  # Back-to-back
                rest_factor = 0.85  # Fatigado
            
            # Ajustar por minutos jugados recientes
            if 'MP' in player_data.columns:
                recent_minutes = sorted_data['MP'].tail(3).mean()
                if recent_minutes > 35:
                    rest_factor *= 0.95  # Penalizar muchos minutos
                elif recent_minutes < 20:
                    rest_factor *= 1.05  # Bonus por pocos minutos
            
            return max(0.7, min(1.2, rest_factor))
            
        except Exception:
            return 1.0

    def _create_optimized_features_for_game(self, player_data: pd.DataFrame, 
                                           game_data: pd.Series, teams_df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Crea características optimizadas para un juego específico con enfoque ULTRA-ESPECÍFICO en rebotes.
        """
        
        # 1. CARACTERÍSTICAS ULTRA-ESPECÍFICAS DE REBOTES (PRIORIDAD MÁXIMA)
        ultra_specific_features = self._create_ultra_specific_rebound_features(
            player_data, game_data, teams_df
        )
        
        # 2. Características básicas de rendimiento
        basic_features = self._create_basic_performance_features(player_data)
        
        # 3. Características de tendencias temporales
        trend_features = self._create_trend_features(player_data)
        
        # 4. Características de contexto de juego
        context_features = self._create_game_context_features(player_data, game_data)
        
        # 5. Características de equipos (si están disponibles)
        team_features = {}
        if teams_df is not None:
            try:
                from .team_features_trb import TeamReboundingFeatures
                team_feature_engineer = TeamReboundingFeatures()
                team_features = team_feature_engineer.create_team_context_features(
                    game_data, teams_df
                )
            except Exception as e:
                logger.warning(f"Error generando características de equipos: {e}")
        
        # 6. COMBINAR TODAS LAS CARACTERÍSTICAS CON PRIORIDAD A ULTRA-ESPECÍFICAS
        all_features = {}
        
        # Primero añadir características ultra-específicas (máxima prioridad)
        all_features.update(ultra_specific_features)
        
        # Luego añadir otras características
        all_features.update(basic_features)
        all_features.update(trend_features)
        all_features.update(context_features)
        all_features.update(team_features)
        
        # 7. CARACTERÍSTICAS HÍBRIDAS FINALES ESPECÍFICAS DE REBOTES
        hybrid_features = self._create_final_hybrid_rebound_features(all_features)
        all_features.update(hybrid_features)
        
        # 8. Validar y limpiar características
        cleaned_features = self._validate_and_clean_features(all_features)
        
        logger.debug(f"Características ultra-específicas generadas: {len(cleaned_features)}")
        logger.debug(f"Características principales: {list(ultra_specific_features.keys())[:10]}")
        
        return cleaned_features

    def _create_final_hybrid_rebound_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Crea características híbridas finales ultra-específicas para rebotes.
        Estas son las características más importantes que combinan todos los factores.
        """
        
        hybrid_features = {}
        
        # ==================== PREDICTORES PRINCIPALES HÍBRIDOS ====================
        
        # 1. PREDICTOR MAESTRO DE REBOTES (combina todos los factores fundamentales)
        master_predictor = 0
        
        if 'ultra_specific_rebound_prediction' in features:
            master_predictor += features['ultra_specific_rebound_prediction'] * 0.4
        
        if 'expected_rebounds_fundamental' in features:
            master_predictor += features['expected_rebounds_fundamental'] * 0.3
        
        if 'historical_rebound_avg' in features:
            master_predictor += features['historical_rebound_avg'] * 0.2
        
        if 'rebound_dominance_factor' in features:
            master_predictor += features['rebound_dominance_factor'] * 0.1
        
        hybrid_features['master_rebound_predictor'] = max(0, master_predictor)
        
        # 2. ÍNDICE DE OPORTUNIDAD TOTAL (oportunidades × capacidad)
        opportunity_index = 0
        if 'total_rebound_opportunities' in features and 'physical_rebound_index' in features:
            opportunity_index = features['total_rebound_opportunities'] * features['physical_rebound_index']
        
        hybrid_features['total_opportunity_index'] = opportunity_index
        
        # 3. FACTOR DE EXPOSICIÓN EFECTIVA (minutos × capacidad × agresividad)
        effective_exposure = 0
        if all(key in features for key in ['minutes_exposure', 'physical_rebound_index', 'aggressiveness_index']):
            effective_exposure = (
                features['minutes_exposure'] * 
                features['physical_rebound_index'] * 
                (1 + features['aggressiveness_index'] * 0.1)
            )
        
        hybrid_features['effective_exposure_factor'] = effective_exposure
        
        # 4. PREDICTOR DE REBOTES OFENSIVOS ESPECÍFICO
        offensive_predictor = 0
        if all(key in features for key in ['aggressiveness_index', 'team_missed_shots_per_game', 'minutes_exposure']):
            offensive_predictor = (
                features['aggressiveness_index'] * 
                features['team_missed_shots_per_game'] * 
                (features['minutes_exposure'] / 48.0)  # Normalizar por minutos totales
            )
        
        hybrid_features['offensive_rebound_predictor'] = offensive_predictor
        
        # 5. PREDICTOR DE REBOTES DEFENSIVOS ESPECÍFICO
        defensive_predictor = 0
        if all(key in features for key in ['physical_rebound_index', 'long_rebound_opportunities', 'defensive_intensity']):
            defensive_predictor = (
                features['physical_rebound_index'] * 
                features['long_rebound_opportunities'] * 
                (1 + features['defensive_intensity'] * 0.05)
            )
        
        hybrid_features['defensive_rebound_predictor'] = defensive_predictor
        
        # ==================== FACTORES DE AJUSTE CONTEXTUAL ====================
        
        # 6. FACTOR DE CONTEXTO TOTAL
        context_factor = 1.0
        if 'game_pace_factor' in features:
            context_factor *= features['game_pace_factor']
        if 'home_court_advantage' in features:
            context_factor *= features['home_court_advantage']
        if 'fatigue_factor' in features:
            context_factor *= features['fatigue_factor']
        
        hybrid_features['total_context_factor'] = context_factor
        
        # 7. PREDICTOR AJUSTADO POR CONTEXTO
        if 'master_rebound_predictor' in hybrid_features:
            context_adjusted = hybrid_features['master_rebound_predictor'] * context_factor
            hybrid_features['context_adjusted_predictor'] = context_adjusted
        
        # ==================== CARACTERÍSTICAS DE ESPECIALIZACIÓN ====================
        
        # 8. ESPECIALISTA EN REBOTES LARGOS (3 puntos)
        long_specialist = 0
        if all(key in features for key in ['height_advantage', 'long_rebound_opportunities', 'position_rebound_factor']):
            long_specialist = (
                features['height_advantage'] * 
                features['long_rebound_opportunities'] * 
                features['position_rebound_factor']
            )
        
        hybrid_features['long_rebound_specialist_score'] = long_specialist
        
        # 9. ESPECIALISTA EN REBOTES CERCANOS (dentro del área)
        close_specialist = 0
        if all(key in features for key in ['aggressiveness_index', 'physical_rebound_index']):
            close_opportunities = features.get('total_rebound_opportunities', 0) - features.get('long_rebound_opportunities', 0)
            close_specialist = (
                features['aggressiveness_index'] * 
                features['physical_rebound_index'] * 
                close_opportunities
            )
        
        hybrid_features['close_rebound_specialist_score'] = close_specialist
        
        # ==================== PREDICTORES DE CONSISTENCIA ====================
        
        # 10. PREDICTOR DE CONSISTENCIA TOTAL
        consistency_predictor = 0
        if all(key in features for key in ['consistency_factor', 'rebound_efficiency', 'historical_rebound_avg']):
            consistency_predictor = (
                features['consistency_factor'] * 
                features['rebound_efficiency'] * 
                features['historical_rebound_avg']
            )
        
        hybrid_features['consistency_predictor'] = consistency_predictor
        
        # 11. FACTOR DE TENDENCIA AJUSTADA
        trend_adjusted = 0
        if all(key in features for key in ['rebound_trend', 'historical_rebound_avg']):
            if features['historical_rebound_avg'] > 0:
                trend_strength = features['rebound_trend'] / features['historical_rebound_avg']
                trend_adjusted = features['historical_rebound_avg'] * (1 + trend_strength * 0.2)
            else:
                trend_adjusted = features.get('rebound_trend', 0)
        
        hybrid_features['trend_adjusted_prediction'] = max(0, trend_adjusted)
        
        # ==================== PREDICTOR FINAL ULTRA-ESPECÍFICO ====================
        
        # 12. PREDICTOR FINAL COMBINADO (la característica más importante)
        final_predictor = 0
        
        # Combinar todos los predictores con pesos específicos
        predictors = [
            ('context_adjusted_predictor', 0.35),
            ('offensive_rebound_predictor', 0.15),
            ('defensive_rebound_predictor', 0.15),
            ('consistency_predictor', 0.15),
            ('trend_adjusted_prediction', 0.10),
            ('long_rebound_specialist_score', 0.05),
            ('close_rebound_specialist_score', 0.05)
        ]
        
        for predictor_name, weight in predictors:
            if predictor_name in hybrid_features:
                final_predictor += hybrid_features[predictor_name] * weight
        
        hybrid_features['final_ultra_specific_predictor'] = max(0, final_predictor)
        
        # ==================== CARACTERÍSTICAS DE VALIDACIÓN ====================
        
        # 13. SCORE DE CONFIANZA EN LA PREDICCIÓN
        confidence_score = 0.5  # Base
        
        if 'consistency_factor' in features:
            confidence_score += features['consistency_factor'] * 0.3
        
        if 'rebound_efficiency' in features and features['rebound_efficiency'] > 0:
            confidence_score += min(0.2, features['rebound_efficiency'] * 10)
        
        if 'historical_rebound_avg' in features and features['historical_rebound_avg'] > 0:
            confidence_score += 0.2
        
        hybrid_features['prediction_confidence_score'] = min(1.0, confidence_score)
        
        # 14. FACTOR DE CALIDAD DE DATOS
        data_quality = 0.5
        available_features = len([k for k, v in features.items() if v != 0])
        total_possible = 50  # Número aproximado de características importantes
        
        data_quality = min(1.0, available_features / total_possible)
        hybrid_features['data_quality_factor'] = data_quality
        
        return hybrid_features



    def _calculate_advanced_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características estadísticas CRÍTICAS para rebotes basadas en análisis NBA.
        
        CARACTERÍSTICAS ESTADÍSTICAS CRÍTICAS (según análisis):
        1. RPG HISTÓRICO: Correlación más alta (~0.8-0.9) - predictor básico
        2. OREB% y DREB%: Alta correlación (~0.6-0.8) - eficiencia en rebotes
        3. MINUTOS POR JUEGO: Correlación moderada (~0.4-0.6) - oportunidades
        4. BLOQUES Y ROBOS: Correlación moderada (~0.3-0.5) - actividad defensiva
        5. FG%: Correlación moderada (~0.2-0.4) - presencia en pintura
        """
        df_features = df.copy()
        
        # ==================== 1. RPG HISTÓRICO (PREDICTOR BÁSICO) ====================
        
        if 'TRB' in df.columns:
            # RPG promedio histórico (últimas 10 partidos)
            df_features['historical_rpg_10g'] = df_features['TRB'].rolling(10, min_periods=3).mean()
            df_features['historical_rpg_5g'] = df_features['TRB'].rolling(5, min_periods=2).mean()
            df_features['historical_rpg_3g'] = df_features['TRB'].rolling(3, min_periods=1).mean()
            
            # RPG promedio de temporada (expandiendo)
            df_features['season_rpg_avg'] = df_features['TRB'].expanding(min_periods=1).mean()
            
            # Tendencia de RPG (¿está mejorando o empeorando?)
            df_features['rpg_trend_10g'] = df_features['TRB'].rolling(10, min_periods=3).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            df_features['rpg_trend_5g'] = df_features['TRB'].rolling(5, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Consistencia en rebotes (desviación estándar)
            df_features['rpg_consistency_10g'] = 1 / (1 + df_features['TRB'].rolling(10, min_periods=3).std().fillna(3))
            df_features['rpg_consistency_5g'] = 1 / (1 + df_features['TRB'].rolling(5, min_periods=2).std().fillna(2))
            
            # Máximo y mínimo histórico
            df_features['rpg_max_10g'] = df_features['TRB'].rolling(10, min_periods=1).max()
            df_features['rpg_min_10g'] = df_features['TRB'].rolling(10, min_periods=1).min()
            df_features['rpg_range_10g'] = df_features['rpg_max_10g'] - df_features['rpg_min_10g']
            
            # Percentiles de rendimiento
            df_features['rpg_percentile_75'] = df_features['TRB'].rolling(20, min_periods=5).quantile(0.75)
            df_features['rpg_percentile_25'] = df_features['TRB'].rolling(20, min_periods=5).quantile(0.25)
        
        # ==================== 2. PORCENTAJES DE REBOTE (EFICIENCIA) ====================
        
        # Si tenemos datos de rebotes ofensivos y defensivos separados
        if 'ORB' in df.columns and 'DRB' in df.columns:
            # Porcentajes de rebote históricos
            df_features['historical_oreb_avg'] = df_features['ORB'].rolling(10, min_periods=3).mean()
            df_features['historical_dreb_avg'] = df_features['DRB'].rolling(10, min_periods=3).mean()
            
            # Ratio ofensivo vs defensivo
            df_features['oreb_dreb_ratio'] = np.where(
                df_features['historical_dreb_avg'] > 0,
                df_features['historical_oreb_avg'] / df_features['historical_dreb_avg'],
                0
            )
            
            # Especialización en tipo de rebote
            df_features['offensive_rebounding_specialist'] = (df_features['historical_oreb_avg'] >= 2.0).astype(int)
            df_features['defensive_rebounding_specialist'] = (df_features['historical_dreb_avg'] >= 8.0).astype(int)
        
        # ==================== 3. MINUTOS POR JUEGO (OPORTUNIDADES) ====================
        
        if 'MP' in df.columns:
            # Minutos promedio históricos
            df_features['historical_mpg_10g'] = df_features['MP'].rolling(10, min_periods=3).mean()
            df_features['historical_mpg_5g'] = df_features['MP'].rolling(5, min_periods=2).mean()
            df_features['season_mpg_avg'] = df_features['MP'].expanding(min_periods=1).mean()
            
            # Tendencia de minutos (indica confianza del entrenador)
            df_features['mpg_trend_10g'] = df_features['MP'].rolling(10, min_periods=3).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Consistencia en minutos
            df_features['mpg_consistency'] = 1 / (1 + df_features['MP'].rolling(10, min_periods=3).std().fillna(5))
            
            # Categorías de rol basadas en minutos
            df_features['starter_role'] = (df_features['historical_mpg_10g'] >= 25).astype(int)
            df_features['key_player_role'] = (df_features['historical_mpg_10g'] >= 30).astype(int)
            df_features['star_player_role'] = (df_features['historical_mpg_10g'] >= 35).astype(int)
            df_features['bench_player_role'] = (df_features['historical_mpg_10g'] < 20).astype(int)
            
            # Eficiencia de rebotes por minuto
            if 'TRB' in df.columns:
                df_features['rebounds_per_minute'] = np.where(
                    (df_features['historical_mpg_10g'] > 0) & (df_features['historical_rpg_10g'].notna()),
                    df_features['historical_rpg_10g'] / df_features['historical_mpg_10g'],
                    0
                )
                
                # Rebotes por 36 minutos (normalizado)
                df_features['rebounds_per_36min'] = df_features['rebounds_per_minute'] * 36
        
        # ==================== 4. ACTIVIDAD DEFENSIVA (AJETREO) ====================
        
        # Bloques por juego (presencia defensiva)
        if 'BLK' in df.columns:
            df_features['historical_bpg_10g'] = df_features['BLK'].rolling(10, min_periods=3).mean()
            df_features['historical_bpg_5g'] = df_features['BLK'].rolling(5, min_periods=2).mean()
            
            # Categorías de bloqueo
            df_features['elite_shot_blocker'] = (df_features['historical_bpg_10g'] >= 1.5).astype(int)  # Como Embiid
            df_features['good_shot_blocker'] = (df_features['historical_bpg_10g'] >= 1.0).astype(int)  # Como Zubac
            df_features['average_shot_blocker'] = (df_features['historical_bpg_10g'] >= 0.5).astype(int)
        
        # Robos por juego (actividad defensiva)
        if 'STL' in df.columns:
            df_features['historical_spg_10g'] = df_features['STL'].rolling(10, min_periods=3).mean()
            df_features['historical_spg_5g'] = df_features['STL'].rolling(5, min_periods=2).mean()
            
            # Actividad defensiva alta
            df_features['high_defensive_activity'] = (df_features['historical_spg_10g'] >= 1.0).astype(int)  # Como Jokić
        
        # Faltas personales (agresividad/fisicalidad)
        if 'PF' in df.columns:
            df_features['historical_pf_10g'] = df_features['PF'].rolling(10, min_periods=3).mean()
            
            # Índice de agresividad física
            df_features['physical_aggressiveness'] = (
                df_features.get('historical_bpg_10g', 0) * 3 +  # Bloqueos valen más
                df_features.get('historical_spg_10g', 0) * 2 +  # Robos valen medio
                df_features['historical_pf_10g'] * 1            # Faltas indican fisicalidad
            )
        
        # ==================== 5. PRESENCIA EN PINTURA (FG%) ====================
        
        if 'FG%' in df.columns:
            df_features['historical_fg_pct_10g'] = df_features['FG%'].rolling(10, min_periods=3).mean()
            df_features['historical_fg_pct_5g'] = df_features['FG%'].rolling(5, min_periods=2).mean()
            
            # Alto FG% indica presencia en pintura (como Duren: 69%, Zubac: ~61%)
            df_features['elite_fg_pct'] = (df_features['historical_fg_pct_10g'] >= 0.65).astype(int)  # Como Duren
            df_features['high_fg_pct'] = (df_features['historical_fg_pct_10g'] >= 0.55).astype(int)  # Como Zubac
            df_features['paint_presence'] = df_features['historical_fg_pct_10g']  # Proxy para presencia en pintura
        
        # ==================== 6. MÉTRICAS COMPUESTAS AVANZADAS ====================
        
        # Índice de reboteador élite (combina múltiples factores)
        elite_rebounding_factors = []
        
        if 'historical_rpg_10g' in df_features.columns:
            factor = df_features['historical_rpg_10g'].fillna(0) / 15.0  # Normalizado por 15 RPG
            elite_rebounding_factors.append(factor)
        
        if 'rebounds_per_36min' in df_features.columns:
            factor = df_features['rebounds_per_36min'].fillna(0) / 20.0  # Normalizado por 20 per 36
            elite_rebounding_factors.append(factor)
        
        if 'physical_aggressiveness' in df_features.columns:
            factor = df_features['physical_aggressiveness'].fillna(0) / 5.0  # Normalizado
            elite_rebounding_factors.append(factor)
        
        if 'paint_presence' in df_features.columns:
            factor = df_features['paint_presence'].fillna(0)  # Ya normalizado (0-1)
            elite_rebounding_factors.append(factor)
        
        if elite_rebounding_factors:
            df_features['elite_rebounding_index'] = np.mean(elite_rebounding_factors, axis=0)
        else:
            df_features['elite_rebounding_index'] = 0.0
        
        # Índice de consistencia total
        consistency_factors = []
        
        if 'rpg_consistency_10g' in df_features.columns:
            factor = df_features['rpg_consistency_10g'].fillna(0.5)
            consistency_factors.append(factor)
        
        if 'mpg_consistency' in df_features.columns:
            factor = df_features['mpg_consistency'].fillna(0.5)
            consistency_factors.append(factor)
        
        if consistency_factors:
            df_features['total_consistency_index'] = np.mean(consistency_factors, axis=0)
        else:
            df_features['total_consistency_index'] = 0.5
        
        # Índice de momentum (tendencias positivas)
        momentum_factors = []
        
        if 'rpg_trend_5g' in df_features.columns:
            factor = np.where(df_features['rpg_trend_5g'].fillna(0) > 0, 1, 0)
            momentum_factors.append(factor)
        
        if 'mpg_trend_10g' in df_features.columns:
            factor = np.where(df_features['mpg_trend_10g'].fillna(0) > 0, 1, 0)
            momentum_factors.append(factor)
        
        if momentum_factors:
            df_features['positive_momentum_index'] = np.mean(momentum_factors, axis=0)
        else:
            df_features['positive_momentum_index'] = 0.0
        
        logger.debug(f"Calculadas características estadísticas CRÍTICAS para rebotes")
        if 'historical_rpg_10g' in df_features.columns:
            logger.debug(f"   RPG histórico promedio: {df_features['historical_rpg_10g'].mean():.2f}")
        if 'rebounds_per_36min' in df_features.columns:
            logger.debug(f"   Rebotes por 36min promedio: {df_features['rebounds_per_36min'].mean():.2f}")
        if 'elite_rebounding_index' in df_features.columns:
            logger.debug(f"   Índice de reboteador élite promedio: {df_features['elite_rebounding_index'].mean():.3f}")
        
        return df_features

    def generate_ultra_specific_rebound_features(self, df: pd.DataFrame, target_col: str = 'TRB', 
                                                min_games: int = 10, teams_df: Optional[pd.DataFrame] = None) -> tuple:
        """
        Genera SOLO las características ULTRA-ESPECÍFICAS más predictivas para rebotes.
        
        ENFOQUE QUIRÚRGICO: Solo las 15-20 características más poderosas.
        Objetivo: ≥97% precisión eliminando ruido y características irrelevantes.
        
        CARACTERÍSTICAS ULTRA-ESPECÍFICAS SELECCIONADAS:
        1. RPG histórico (predictor #1 - correlación ~0.85-0.9)
        2. Altura física real (predictor #2 - correlación ~0.7-0.8) 
        3. Minutos por juego (predictor #3 - correlación ~0.6-0.7)
        4. Oportunidades de rebote reales (predictor #4 - correlación ~0.5-0.6)
        5. Posición específica (predictor #5 - correlación ~0.4-0.5)
        """
        logger.info("🎯 Generando características ULTRA-ESPECÍFICAS para rebotes (enfoque quirúrgico)...")
        
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
        
        # Ordenar por jugador y fecha
        df_sorted = df.sort_values(['Player', 'Date']).copy()
        
        X_list = []
        y_list = []
        
        total_players = len(df_sorted['Player'].unique())
        processed_players = 0
        
        # Procesar cada jugador
        for player in df_sorted['Player'].unique():
            player_data = df_sorted[df_sorted['Player'] == player].copy()
            
            if len(player_data) < min_games:
                continue
            
            processed_players += 1
            
            # Log progreso cada 50 jugadores
            if processed_players % 50 == 0 or processed_players == total_players:
                logger.info(f"🔬 Procesando jugadores (ultra-específico): {processed_players}/{total_players}")
            
            # Para cada juego (excepto los primeros min_games-1)
            for i in range(min_games-1, len(player_data)):
                current_game = player_data.iloc[i]
                historical_data = player_data.iloc[:i+1]  # Incluir juego actual para cálculos históricos
                
                # EXTRAER TARGET
                target_value = current_game[target_col]
                
                # ==================== CARACTERÍSTICAS ULTRA-ESPECÍFICAS ====================
                
                feature_vector = {}
                
                # 1. RPG HISTÓRICO (PREDICTOR #1 - MÁS IMPORTANTE)
                if len(historical_data) >= 3:
                    # RPG últimos 10 juegos (sin incluir juego actual)
                    hist_without_current = historical_data.iloc[:-1] if len(historical_data) > 1 else historical_data
                    feature_vector['rpg_last_10'] = hist_without_current[target_col].tail(10).mean()
                    feature_vector['rpg_last_5'] = hist_without_current[target_col].tail(5).mean()
                    feature_vector['rpg_last_3'] = hist_without_current[target_col].tail(3).mean()
                    feature_vector['rpg_season_avg'] = hist_without_current[target_col].mean()
                    
                    # Consistencia en rebotes (crítico para predicción)
                    feature_vector['rpg_consistency'] = 1.0 / (1.0 + hist_without_current[target_col].tail(10).std())
                    
                    # Tendencia reciente (¿mejorando o empeorando?)
                    if len(hist_without_current) >= 5:
                        recent_5 = hist_without_current[target_col].tail(5).values
                        if len(recent_5) > 1:
                            trend = np.polyfit(range(len(recent_5)), recent_5, 1)[0]
                            feature_vector['rpg_trend'] = trend
                        else:
                            feature_vector['rpg_trend'] = 0.0
                    else:
                        feature_vector['rpg_trend'] = 0.0
                else:
                    feature_vector['rpg_last_10'] = 0.0
                    feature_vector['rpg_last_5'] = 0.0
                    feature_vector['rpg_last_3'] = 0.0
                    feature_vector['rpg_season_avg'] = 0.0
                    feature_vector['rpg_consistency'] = 0.5
                    feature_vector['rpg_trend'] = 0.0
                
                # 2. ALTURA FÍSICA REAL (PREDICTOR #2 - CRÍTICO)
                if 'Height_Inches' in current_game.index:
                    height = current_game['Height_Inches']
                elif 'Pos' in current_game.index:
                    # Estimar altura por posición
                    pos_height_map = {'C': 84.0, 'F': 80.0, 'C-F': 82.5, 'F-C': 82.0, 
                                    'F-G': 78.0, 'G-F': 77.0, 'G': 75.0}
                    height = pos_height_map.get(current_game['Pos'], 79.0)
                else:
                    height = 79.0
                
                feature_vector['height_inches'] = height
                feature_vector['height_advantage'] = (height - 78.0) / 6.0  # Normalizado
                feature_vector['is_very_tall'] = 1 if height >= 82 else 0  # 6'10"+
                feature_vector['is_elite_height'] = 1 if height >= 84 else 0  # 7'0"+
                
                # 3. MINUTOS POR JUEGO (PREDICTOR #3 - EXPOSICIÓN)
                if 'MP' in current_game.index:
                    current_minutes = current_game['MP']
                    feature_vector['minutes_current'] = current_minutes
                    feature_vector['minutes_pct'] = current_minutes / 48.0
                    feature_vector['is_starter'] = 1 if current_minutes >= 25 else 0
                    feature_vector['is_key_player'] = 1 if current_minutes >= 30 else 0
                    
                    # Minutos históricos
                    if len(historical_data) >= 3:
                        hist_minutes = historical_data['MP'].iloc[:-1] if len(historical_data) > 1 else historical_data['MP']
                        feature_vector['minutes_avg_10'] = hist_minutes.tail(10).mean()
                        feature_vector['minutes_consistency'] = 1.0 / (1.0 + hist_minutes.tail(10).std())
                    else:
                        feature_vector['minutes_avg_10'] = current_minutes
                        feature_vector['minutes_consistency'] = 0.5
                else:
                    feature_vector['minutes_current'] = 25.0
                    feature_vector['minutes_pct'] = 0.52
                    feature_vector['is_starter'] = 1
                    feature_vector['is_key_player'] = 0
                    feature_vector['minutes_avg_10'] = 25.0
                    feature_vector['minutes_consistency'] = 0.5
                
                # 4. OPORTUNIDADES DE REBOTE REALES (PREDICTOR #4)
                # Tiros fallados propios (rebotes ofensivos)
                if 'FGA' in current_game.index and 'FG' in current_game.index:
                    own_missed = current_game['FGA'] - current_game['FG']
                    feature_vector['own_missed_shots'] = own_missed
                    
                    # Histórico de tiros fallados
                    if len(historical_data) >= 3:
                        hist_fga = historical_data['FGA'].iloc[:-1] if len(historical_data) > 1 else historical_data['FGA']
                        hist_fg = historical_data['FG'].iloc[:-1] if len(historical_data) > 1 else historical_data['FG']
                        hist_missed = (hist_fga - hist_fg).tail(10).mean()
                        feature_vector['own_missed_avg'] = hist_missed
                    else:
                        feature_vector['own_missed_avg'] = own_missed
                else:
                    feature_vector['own_missed_shots'] = 8.0  # Promedio NBA
                    feature_vector['own_missed_avg'] = 8.0
                
                # Tiros de 3 fallados (rebotes largos)
                if '3PA' in current_game.index and '3P' in current_game.index:
                    missed_3pt = current_game['3PA'] - current_game['3P']
                    feature_vector['missed_3pt'] = missed_3pt
                    feature_vector['long_rebound_opps'] = missed_3pt * 1.5  # Rebotes largos
                else:
                    feature_vector['missed_3pt'] = 4.0
                    feature_vector['long_rebound_opps'] = 6.0
                
                # Obtener datos reales del oponente si están disponibles
                if teams_df is not None:
                    opp_missed = self._get_opponent_missed_shots(current_game, teams_df)
                    feature_vector['opp_missed_shots'] = opp_missed
                    feature_vector['total_rebound_opps'] = feature_vector['own_missed_shots'] + opp_missed
                else:
                    feature_vector['opp_missed_shots'] = 46.0  # Promedio NBA
                    feature_vector['total_rebound_opps'] = feature_vector['own_missed_shots'] + 46.0
                
                # 5. POSICIÓN ESPECÍFICA (PREDICTOR #5)
                if 'Pos' in current_game.index:
                    pos = current_game['Pos']
                    feature_vector['is_center'] = 1 if pos == 'C' else 0
                    feature_vector['is_forward'] = 1 if pos == 'F' else 0
                    feature_vector['is_center_forward'] = 1 if pos == 'C-F' else 0
                    feature_vector['is_forward_center'] = 1 if pos == 'F-C' else 0
                    feature_vector['is_big_man'] = 1 if pos in ['C', 'C-F', 'F-C'] else 0
                    
                    # Peso por posición para rebotes
                    pos_weights = {'C': 3.5, 'F': 2.8, 'C-F': 3.3, 'F-C': 3.2, 
                                 'F-G': 1.8, 'G-F': 1.5, 'G': 1.0}
                    feature_vector['position_weight'] = pos_weights.get(pos, 2.0)
                else:
                    feature_vector['is_center'] = 0
                    feature_vector['is_forward'] = 1
                    feature_vector['is_center_forward'] = 0
                    feature_vector['is_forward_center'] = 0
                    feature_vector['is_big_man'] = 0
                    feature_vector['position_weight'] = 2.0
                
                # 6. CARACTERÍSTICAS HÍBRIDAS ULTRA-ESPECÍFICAS
                # Capacidad física total
                feature_vector['physical_capacity'] = (
                    feature_vector['height_advantage'] * 
                    feature_vector['position_weight'] / 3.5
                )
                
                # Exposición efectiva (minutos × capacidad física)
                feature_vector['effective_exposure'] = (
                    feature_vector['minutes_pct'] * 
                    feature_vector['physical_capacity']
                )
                
                # Oportunidades ponderadas por capacidad
                feature_vector['weighted_opportunities'] = (
                    feature_vector['total_rebound_opps'] * 
                    feature_vector['physical_capacity']
                )
                
                # Eficiencia histórica de rebotes
                if feature_vector['minutes_avg_10'] > 0:
                    feature_vector['rebound_efficiency'] = feature_vector['rpg_last_10'] / feature_vector['minutes_avg_10']
                else:
                    feature_vector['rebound_efficiency'] = 0.0
                
                # Predictor maestro ultra-específico
                feature_vector['master_predictor'] = (
                    feature_vector['rpg_last_5'] * 0.4 +           # 40% histórico reciente
                    feature_vector['weighted_opportunities'] * 0.02 +  # 20% oportunidades
                    feature_vector['effective_exposure'] * 8.0 +   # 20% exposición
                    feature_vector['rebound_efficiency'] * 20.0 +  # 15% eficiencia
                    feature_vector['rpg_trend'] * 2.0              # 5% tendencia
                )
                
                X_list.append(feature_vector)
                y_list.append(target_value)
        
        if not X_list:
            raise ValueError("No se pudieron generar características ultra-específicas")
        
        # Convertir a DataFrame
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        
        # Limpiar datos
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        feature_names = list(X.columns)
        
        logger.info(f"✅ Generadas {len(feature_names)} características ULTRA-ESPECÍFICAS para {len(X)} juegos")
        logger.info(f"📊 Características principales: {feature_names[:10]}")
        logger.info(f"🎯 Distribución objetivo - Min: {y.min()}, Max: {y.max()}, Media: {y.mean():.2f}")
        
        return X.values, y, feature_names
    
    def _get_opponent_missed_shots(self, game_data: pd.Series, teams_df: pd.DataFrame) -> float:
        """Obtiene tiros fallados reales del oponente para un juego específico."""
        try:
            game_date = pd.to_datetime(game_data['Date'])
            player_team = game_data['Team']
            opponent_team = game_data['Opp']
            
            # Buscar el juego del equipo del jugador
            team_game = teams_df[
                (teams_df['Team'] == player_team) & 
                (teams_df['Date'] == game_date) &
                (teams_df['Opp'] == opponent_team)
            ]
            
            if len(team_game) > 0:
                team_data = team_game.iloc[0]
                # Tiros fallados del oponente
                opp_fga = team_data.get('FGA_Opp', 85.0)
                opp_fg = team_data.get('FG_Opp', 39.0)
                return max(0, opp_fga - opp_fg)
            
            return 46.0  # Promedio NBA si no hay datos
            
        except Exception:
            return 46.0  # Promedio NBA en caso de error

    def generate_specific_features(self, df: pd.DataFrame, target_col: str = 'TRB', 
                                  min_games: int = 10, teams_df: Optional[pd.DataFrame] = None) -> tuple:
        """
        Genera únicamente las características específicas solicitadas para rebotes.
        
        Args:
            df: DataFrame con datos históricos
            target_col: Columna objetivo (TRB)
            min_games: Mínimo de juegos históricos requeridos
            teams_df: DataFrame opcional con datos de equipos
            
        Returns:
            Tuple (X, y, feature_names) con características específicas y targets
        """
        logger.info("🎯 Generando características ESPECÍFICAS solicitadas para rebotes...")
        
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
        
        # Ordenar por jugador y fecha
        df_sorted = df.sort_values(['Player', 'Date']).copy()
        
        X_list = []
        y_list = []
        
        total_players = len(df_sorted['Player'].unique())
        processed_players = 0
        
        # Procesar cada jugador
        for player in df_sorted['Player'].unique():
            player_data = df_sorted[df_sorted['Player'] == player].copy()
            
            if len(player_data) < min_games:
                continue
            
            processed_players += 1
            
            # Log progreso cada 50 jugadores
            if processed_players % 50 == 0 or processed_players == total_players:
                logger.info(f"📊 Procesando jugadores: {processed_players}/{total_players}")
            
            # Generar características para este jugador
            player_features = self._generate_specific_features_for_player(player_data, teams_df)
            
            # Para cada juego (excepto los primeros min_games-1)
            for i in range(min_games-1, len(player_features)):
                current_game = player_features.iloc[i]
                original_game = player_data.iloc[i]
                
                # EXTRAER TARGET
                target_value = original_game[target_col]
                
                # Crear vector de características específicas
                feature_vector = self._extract_specific_features(current_game)
                
                X_list.append(feature_vector)
                y_list.append(target_value)
        
        if not X_list:
            raise ValueError("No se pudieron generar características específicas")
        
        # Convertir a DataFrame
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        
        # Limpiar datos
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        feature_names = list(X.columns)
        
        logger.info(f"✅ Generadas {len(feature_names)} características específicas para {len(X)} juegos")
        logger.info(f"📊 Distribución objetivo - Min: {y.min()}, Max: {y.max()}, Media: {y.mean():.2f}")
        
        return X.values, y, feature_names
    
    def _generate_specific_features_for_player(self, player_data: pd.DataFrame, teams_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Genera características específicas para un jugador."""
        df_features = player_data.copy()
        
        # Convertir Date a datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
            df_features['Date'] = pd.to_datetime(df_features['Date'], errors='coerce')
        
        # 1. CARACTERÍSTICAS FÍSICAS
        df_features = self._calculate_physical_advantages(df_features)
        
        # 2. CARACTERÍSTICAS ESTADÍSTICAS
        df_features = self._calculate_advanced_statistical_features(df_features)
        
        # 3. OPORTUNIDADES DE REBOTE
        df_features = self._calculate_missed_shots_opportunities(df_features, teams_df)
        
        # 4. CARACTERÍSTICAS DE EXPOSICIÓN
        df_features = self._calculate_exposure_and_context(df_features)
        
        # 5. CARACTERÍSTICAS DE EQUIPOS (si están disponibles)
        if teams_df is not None:
            try:
                from .team_features_trb import TeamReboundingFeatures
                if not hasattr(self, '_team_feature_generator'):
                    self._team_feature_generator = TeamReboundingFeatures()
                df_features = self._team_feature_generator._generate_team_context_features_silent(df_features, teams_df)
                df_features = self._create_hybrid_team_player_features(df_features)
            except Exception as e:
                logger.warning(f"Error integrando características de equipos: {e}")
        
        # Limpiar datos
        df_features = df_features.fillna(0)
        df_features = df_features.replace([np.inf, -np.inf], 0)
        
        return df_features
    
    def _extract_specific_features(self, game_data: pd.Series) -> Dict[str, float]:
        """Extrae únicamente las características específicas solicitadas."""
        features = {}
        
        # CARACTERÍSTICAS FÍSICAS
        physical_features = [
            'height_inches', 'wingspan_inches', 'weight_lbs',
            'height_rebounding_multiplier', 'wingspan_rebounding_multiplier', 'weight_rebounding_multiplier',
            'physical_rebounding_index', 'defensive_rebounding_physical_index', 'offensive_rebounding_physical_index',
            'center_rebounding_advantage', 'forward_rebounding_advantage', 'versatile_big_advantage',
            'center_forward_advantage', 'forward_center_advantage',
            'is_center_forward', 'is_forward_center'
        ]
        
        # CARACTERÍSTICAS ESTADÍSTICAS (ESPECÍFICAS DE REBOTES)
        statistical_features = [
            'historical_rpg_10g', 'historical_rpg_5g', 'historical_rpg_3g', 'season_rpg_avg',
            'rpg_trend_10g', 'rpg_trend_5g', 'rpg_consistency_10g', 'rpg_consistency_5g',
            'historical_oreb_avg', 'historical_dreb_avg', 'oreb_dreb_ratio',
            'offensive_rebounding_specialist', 'defensive_rebounding_specialist',
            'historical_mpg_10g', 'historical_mpg_5g', 'season_mpg_avg',
            'mpg_trend_10g', 'mpg_consistency',
            'rebounds_per_minute', 'rebounds_per_36min',
            'elite_rebounding_index'
        ]
        
        # OPORTUNIDADES DE REBOTE
        opportunity_features = [
            'own_missed_shots', 'own_missed_3pt', 'own_missed_2pt',
            'total_rebound_opportunities', 'total_long_opportunities', 'total_close_opportunities',
            'weighted_rebound_opportunities', 'weighted_close_opportunities', 'weighted_long_opportunities',
            'opportunities_per_minute', 'weighted_opportunities_per_minute'
        ]
        
        # DATOS DEL OPONENTE (ESPECÍFICOS DE REBOTES)
        opponent_features = [
            'opp_real_missed_shots', 'opp_real_missed_3pt', 'opp_real_missed_2pt',
            'opp_3pt_tendency', 'quality_weighted_opportunities'
        ]
        
        # CARACTERÍSTICAS DE EQUIPOS (ESPECÍFICAS DE REBOTES)
        team_features = [
            'team_missed_shots', 'opp_missed_shots', 'game_pace', 'team_pace',
            'team_3pt_rate', 'opp_3pt_rate',
            'total_rebound_index', 'offensive_rebound_index', 'defensive_rebound_index',
            'final_rebound_opportunities'
        ]
        
        # CARACTERÍSTICAS HÍBRIDAS (JUGADOR + EQUIPO, ESPECÍFICAS DE REBOTES)
        hybrid_features = [
            'player_team_missed_shots_ratio', 'player_adjusted_total_opportunities',
            'pace_adjusted_exposure', 'pace_adjusted_opps_per_minute'
        ]
        
        # CAPACIDAD POSICIONAL DE REBOTES
        positional_features = [
            'position_rebounding_weight', 'is_center', 'is_power_forward', 'is_big_man'
        ]
        
        # EXPOSICIÓN (ESPECÍFICA DE REBOTES)
        exposure_features = [
            'minutes_played', 'minutes_pct', 'starter',
            'minutes_trend_5g', 'minutes_consistency'
        ]
        
        # Combinar todas las características específicas
        all_specific_features = (
            physical_features + statistical_features + opportunity_features +
            opponent_features + team_features + hybrid_features +
            positional_features + exposure_features
        )
        
        # Extraer solo las características que existen en los datos
        for feature in all_specific_features:
            if feature in game_data.index:
                features[feature] = game_data[feature]
            else:
                # Valores por defecto para características faltantes
                features[feature] = self._get_default_value_for_feature(feature)
        
        return features
    
    def _get_default_value_for_feature(self, feature_name: str) -> float:
        """Obtiene valor por defecto para características faltantes."""
        
        # Valores por defecto específicos por tipo de característica
        defaults = {
            # Físicas
            'height_inches': 79.0,
            'wingspan_inches': 81.0,
            'weight_lbs': 220.0,
            'height_rebounding_multiplier': 2.0,
            'wingspan_rebounding_multiplier': 1.5,
            'weight_rebounding_multiplier': 1.2,
            'physical_rebounding_index': 1.5,
            'defensive_rebounding_physical_index': 1.5,
            'offensive_rebounding_physical_index': 1.3,
            'center_rebounding_advantage': 0.0,
            'forward_rebounding_advantage': 0.0,
            'versatile_big_advantage': 0.0,
            'center_forward_advantage': 0.0,
            'forward_center_advantage': 0.0,
            'is_center_forward': 0,
            'is_forward_center': 0,
            
            # Estadísticas
            'historical_rpg_10g': 5.0,
            'historical_rpg_5g': 5.0,
            'historical_rpg_3g': 5.0,
            'season_rpg_avg': 5.0,
            'rpg_trend_10g': 0.0,
            'rpg_trend_5g': 0.0,
            'rpg_consistency_10g': 0.5,
            'rpg_consistency_5g': 0.5,
            'historical_oreb_avg': 1.5,
            'historical_dreb_avg': 3.5,
            'oreb_dreb_ratio': 0.4,
            'offensive_rebounding_specialist': 0,
            'defensive_rebounding_specialist': 0,
            'historical_mpg_10g': 25.0,
            'historical_mpg_5g': 25.0,
            'season_mpg_avg': 25.0,
            'mpg_trend_10g': 0.0,
            'mpg_consistency': 0.8,
            'rebounds_per_minute': 0.2,
            'rebounds_per_36min': 7.2,
            'elite_rebounding_index': 0.3,
            
            # Oportunidades
            'own_missed_shots': 8.0,
            'own_missed_3pt': 4.0,
            'own_missed_2pt': 4.0,
            'total_rebound_opportunities': 54.0,
            'total_long_opportunities': 26.0,
            'total_close_opportunities': 28.0,
            'weighted_rebound_opportunities': 81.0,
            'weighted_close_opportunities': 33.6,
            'weighted_long_opportunities': 20.8,
            'opportunities_per_minute': 2.16,
            'weighted_opportunities_per_minute': 3.24,
            
            # Oponente
            'opp_real_missed_shots': 46.0,
            'opp_real_missed_3pt': 22.0,
            'opp_real_missed_2pt': 24.0,
            'opp_3pt_tendency': 0.40,
            'quality_weighted_opportunities': 54.0,
            
            # Equipos
            'team_missed_shots': 46.0,
            'opp_missed_shots': 46.0,
            'game_pace': 95.0,
            'team_pace': 95.0,
            'team_3pt_rate': 0.40,
            'opp_3pt_rate': 0.40,
            'total_rebound_index': 100.0,
            'offensive_rebound_index': 100.0,
            'defensive_rebound_index': 100.0,
            'final_rebound_opportunities': 54.0,
            
            # Híbridas
            'player_team_missed_shots_ratio': 0.17,
            'player_adjusted_total_opportunities': 54.0,
            'pace_adjusted_exposure': 0.52,
            'pace_adjusted_opps_per_minute': 2.16,
            
            # Posicionales
            'position_rebounding_weight': 2.0,
            'is_center': 0,
            'is_power_forward': 0,
            'is_big_man': 0,
            
            # Exposición
            'minutes_played': 25.0,
            'minutes_pct': 0.52,
            'starter': 1,
            'minutes_trend_5g': 0.0,
            'minutes_consistency': 0.8
        }
        
        return defaults.get(feature_name, 0.0)

# Mantener compatibilidad con código existente
AdvancedReboundsFeatureEngineer = ReboundsFeatureEngineer 