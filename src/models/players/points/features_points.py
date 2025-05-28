"""
Módulo de Características para Predicción de Puntos (PTS)
========================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de puntos en jugadores NBA. Implementa características
avanzadas basadas en:

1. Estadísticas históricas del jugador
2. Tendencias recientes y momentum
3. Factores contextuales (local/visitante, rival, etc.)
4. Métricas de eficiencia ofensiva
5. Características temporales y de carga de trabajo

Arquitectura modular que permite fácil extensión y mantenimiento.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PointsFeatureEngineer:
    """
    Ingeniero de características especializado en predicción de puntos.
    
    Genera características avanzadas específicamente diseñadas para maximizar
    la precisión en la predicción de puntos por partido.
    """
    
    def __init__(self):
        """Inicializa el ingeniero de características para puntos."""
        self.feature_groups = {
            'basic_stats': [],
            'shooting_efficiency': [],
            'historical_trends': [],
            'momentum_features': [],
            'contextual_factors': [],
            'opponent_impact': [],
            'temporal_features': [],
            'advanced_metrics': []
        }
        
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        Genera todas las características para predicción de puntos.
        
        Args:
            df: DataFrame con datos de jugadores (se modifica in-place)
            
        Returns:
            Lista de nombres de características generadas
        """
        logger.info("Generando características para predicción de puntos...")
        
        # Asegurar que el DataFrame esté ordenado por jugador y fecha
        if not df.empty:
            df.sort_values(['Player', 'Date'], inplace=True)
        
        # 1. Características básicas de anotación
        self._add_basic_scoring_features(df)
        
        # 2. Eficiencia de tiro
        self._add_shooting_efficiency_features(df)
        
        # 3. Tendencias históricas
        self._add_historical_trends(df)
        
        # 4. Momentum y rachas
        self._add_momentum_features(df)
        
        # 5. Factores contextuales
        self._add_contextual_features(df)
        
        # 6. Impacto del rival
        self._add_opponent_impact_features(df)
        
        # 7. Características temporales
        self._add_temporal_features(df)
        
        # 8. Métricas avanzadas
        self._add_advanced_metrics(df)
        
        # Compilar lista de todas las características
        all_features = []
        for group_features in self.feature_groups.values():
            all_features.extend(group_features)
            
        logger.info(f"Generadas {len(all_features)} características para puntos")
        return all_features
    
    def _add_basic_scoring_features(self, df: pd.DataFrame) -> None:
        """Añade características básicas de anotación."""
        features = []
        
        # Estadísticas básicas de tiro (SOLO intentos y porcentajes, NO anotados)
        basic_features = ['FGA', 'FG%', '2PA', '2P%', 
                         '3PA', '3P%', 'FTA', 'FT%', 'TS%']
        
        for feature in basic_features:
            if feature in df.columns:
                features.append(feature)
        
        # Minutos jugados (crucial para puntos)
        if 'MP' in df.columns:
            features.append('MP')
            
        # Intentos de tiro por minuto
        if 'FGA' in df.columns and 'MP' in df.columns:
            df['FGA_per_minute'] = df['FGA'] / (df['MP'] + 1e-6)
            features.append('FGA_per_minute')
                    
        self.feature_groups['basic_stats'] = features
    
    def _add_shooting_efficiency_features(self, df: pd.DataFrame) -> None:
        """Añade características de eficiencia de tiro."""
        features = []
        
        # True Shooting Percentage ya está en básicas
        
        # NO calcular eFG% ya que requiere FG y 3P (data leakage)
        
        # Distribución de tiros
        if all(col in df.columns for col in ['2PA', '3PA', 'FGA']):
            df['2P_rate'] = df['2PA'] / (df['FGA'] + 1e-6)
            df['3P_rate'] = df['3PA'] / (df['FGA'] + 1e-6)
            features.extend(['2P_rate', '3P_rate'])
        
        # Tasa de tiros libres
        if all(col in df.columns for col in ['FTA', 'FGA']):
            df['FT_rate'] = df['FTA'] / (df['FGA'] + 1e-6)
            features.append('FT_rate')
        
        # NO incluir puntos desde tiros libres ni triples (data leakage)
        
        self.feature_groups['shooting_efficiency'] = features
    
    def _add_historical_trends(self, df: pd.DataFrame) -> None:
        """Añade tendencias históricas del jugador."""
        features = []
        
        # Promedios móviles de diferentes ventanas
        windows = [3, 5, 10, 15]
        
        for window in windows:
            # NO incluir promedios móviles de puntos (data leakage)
            
            # Promedio móvil de FGA
            if 'FGA' in df.columns:
                col_name = f'FGA_avg_{window}g'
                df[col_name] = df.groupby('Player')['FGA'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                features.append(col_name)
            
            # Promedio móvil de FG%
            if 'FG%' in df.columns:
                col_name = f'FG%_avg_{window}g'
                df[col_name] = df.groupby('Player')['FG%'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                features.append(col_name)
            
            # Promedio móvil de minutos
            if 'MP' in df.columns:
                col_name = f'MP_avg_{window}g'
                df[col_name] = df.groupby('Player')['MP'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                features.append(col_name)
        
        # NO incluir estadísticas de puntos (data leakage)
        
        self.feature_groups['historical_trends'] = features
    
    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        """Añade características de momentum y rachas."""
        features = []
        
        # NO incluir rachas basadas en puntos (data leakage)
        
        # Momentum de FG% (últimos 3 vs últimos 10)
        if all(f'FG%_avg_{w}g' in df.columns for w in [3, 10]):
            df['FG%_momentum'] = df['FG%_avg_3g'] - df['FG%_avg_10g']
            features.append('FG%_momentum')
        
        # Días de descanso
        df['days_rest'] = df.groupby('Player')['Date'].diff().dt.days.fillna(2)
        features.append('days_rest')
        
        # Indicador de back-to-back
        df['is_b2b'] = (df['days_rest'] <= 1).astype(int)
        features.append('is_b2b')
        
        self.feature_groups['momentum_features'] = features
    
    def _add_contextual_features(self, df: pd.DataFrame) -> None:
        """Añade factores contextuales."""
        features = []
        
        # Local vs Visitante
        if 'is_home' in df.columns:
            features.append('is_home')
            
            # NO incluir promedios de puntos por ubicación (data leakage)
        
        # Titular vs suplente
        if 'is_started' in df.columns:
            features.append('is_started')
            
            # NO incluir promedios de puntos por rol (data leakage)
        
        # Mes de la temporada
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
            features.append('month')
            
            # Parte de la temporada (inicio, medio, final)
            # Mapear meses a partes de temporada de forma más simple
            def map_season_part(month):
                if month in [10, 11, 12]:  # Oct-Dic (inicio)
                    return 0
                elif month in [1, 2]:      # Ene-Feb (medio)
                    return 1
                elif month in [3, 4]:      # Mar-Abr (final)
                    return 2
                else:                      # Mayo+ (playoffs/offseason)
                    return 3
            
            df['season_part'] = df['month'].apply(map_season_part)
            features.append('season_part')
        
        self.feature_groups['contextual_factors'] = features
    
    def _add_opponent_impact_features(self, df: pd.DataFrame) -> None:
        """Añade características del impacto del equipo rival."""
        features = []
        
        if 'Opp' in df.columns:
            # NO incluir promedio de puntos vs rival (data leakage)
            
            # Número de veces que ha enfrentado a este rival
            df['games_vs_opp'] = df.groupby(['Player', 'Opp']).cumcount()
            features.append('games_vs_opp')
        
        # Aquí se podrían añadir estadísticas defensivas del rival
        # si estuvieran disponibles en el dataset
        
        self.feature_groups['opponent_impact'] = features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> None:
        """Añade características temporales."""
        features = []
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Día de la semana
            df['day_of_week'] = df['Date'].dt.dayofweek
            features.append('day_of_week')
            
            # Fin de semana
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
            features.append('is_weekend')
            
            # Número de juego en la temporada
            df['game_number'] = df.groupby('Player').cumcount() + 1
            features.append('game_number')
            
            # Juegos en los últimos 7 días
            df['games_last_7d'] = df.groupby('Player').apply(
                lambda group: self._count_games_in_period(group, 7)
            ).reset_index(level=0, drop=True)
            features.append('games_last_7d')
            
            # Juegos en los últimos 14 días
            df['games_last_14d'] = df.groupby('Player').apply(
                lambda group: self._count_games_in_period(group, 14)
            ).reset_index(level=0, drop=True)
            features.append('games_last_14d')
        
        self.feature_groups['temporal_features'] = features
    
    def _add_advanced_metrics(self, df: pd.DataFrame) -> None:
        """Añade métricas avanzadas."""
        features = []
        
        # Usage Rate aproximado (FGA + 0.44*FTA + TOV) / MP
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'MP']):
            df['usage_rate_approx'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / (df['MP'] + 1e-6)
            features.append('usage_rate_approx')
        
        # Game Score (métrica de rendimiento general)
        if 'GmSc' in df.columns:
            features.append('GmSc')
            
            # Promedio móvil de Game Score
            df['GmSc_avg_5g'] = df.groupby('Player')['GmSc'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )
            features.append('GmSc_avg_5g')
        
        # Plus/Minus si está disponible
        if '+/-' in df.columns:
            features.append('+/-')
            
            df['plus_minus_avg_5g'] = df.groupby('Player')['+/-'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )
            features.append('plus_minus_avg_5g')
        
        # Ratio de asistencias a pérdidas
        if all(col in df.columns for col in ['AST', 'TOV']):
            df['ast_tov_ratio'] = df['AST'] / (df['TOV'] + 1e-6)
            features.append('ast_tov_ratio')
        
        # Eficiencia por minuto (SIN puntos para evitar data leakage)
        if all(col in df.columns for col in ['TRB', 'AST', 'STL', 'BLK', 'TOV', 'MP']):
            df['efficiency_per_min'] = (df['TRB'] + df['AST'] + 
                                       df['STL'] + df['BLK'] - df['TOV']) / (df['MP'] + 1e-6)
            features.append('efficiency_per_min')
        
        self.feature_groups['advanced_metrics'] = features
    
    def _calculate_streak(self, condition_series: pd.Series) -> pd.Series:
        """Calcula rachas consecutivas basadas en una condición."""
        # Crear grupos de rachas consecutivas
        groups = (condition_series != condition_series.shift()).cumsum()
        
        # Calcular la longitud de cada racha
        streak_lengths = condition_series.groupby(groups).cumsum()
        
        # Solo mantener rachas positivas (donde la condición es True)
        result = streak_lengths.where(condition_series, 0)
        
        # Desplazar una posición para evitar data leakage
        return result.shift(1).fillna(0)
    
    def _calculate_above_average_streak(self, group: pd.DataFrame) -> pd.Series:
        """Calcula racha de juegos por encima del promedio personal."""
        if 'PTS' not in group.columns or len(group) < 2:
            return pd.Series(0, index=group.index)
        
        # Calcular promedio móvil expandido (excluyendo el juego actual)
        expanding_avg = group['PTS'].expanding().mean().shift(1)
        
        # Condición: puntos actuales > promedio histórico
        above_avg = group['PTS'] > expanding_avg
        
        return self._calculate_streak(above_avg)
    
    def _count_games_in_period(self, group: pd.DataFrame, days: int) -> pd.Series:
        """Cuenta juegos en un período específico."""
        if 'Date' not in group.columns:
            return pd.Series(0, index=group.index)
        
        result = []
        for i, current_date in enumerate(group['Date']):
            if i == 0:
                result.append(0)
            else:
                # Contar juegos en los últimos 'days' días
                cutoff_date = current_date - timedelta(days=days)
                recent_games = group.iloc[:i]
                count = (recent_games['Date'] >= cutoff_date).sum()
                result.append(count)
        
        return pd.Series(result, index=group.index)
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Retorna las características agrupadas por categoría."""
        return self.feature_groups.copy()
    
    def get_core_features(self) -> List[str]:
        """Retorna las características más importantes para puntos (SIN data leakage)."""
        core_features = []
        
        # Características básicas esenciales (SOLO intentos y porcentajes)
        core_features.extend(['MP', 'FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'TS%'])
        
        # Tendencias históricas clave (SOLO intentos y porcentajes)
        core_features.extend(['FGA_avg_5g', 'FGA_avg_10g', 'FG%_avg_5g', 'MP_avg_5g'])
        
        # Factores contextuales importantes
        core_features.extend(['is_home', 'is_started', 'days_rest'])
        
        # Distribución de tiros (sin eFG% que usa anotados)
        core_features.extend(['2P_rate', '3P_rate', 'FT_rate'])
        
        # Filtrar solo las que existen
        existing_features = []
        all_generated = []
        for group_features in self.feature_groups.values():
            all_generated.extend(group_features)
        
        for feature in core_features:
            if feature in all_generated:
                existing_features.append(feature)
        
        return existing_features
