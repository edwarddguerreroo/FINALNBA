import pandas as pd
import numpy as np
import logging
from .base_model import BaseNBAModel

logger = logging.getLogger(__name__)

class ReboundsModel(BaseNBAModel):
    """
    Modelo específico para predecir rebotes totales (TRB) de jugadores NBA.
    Hereda de BaseNBAModel y define características específicas para rebotes.
    """
    
    def __init__(self):
        super().__init__(target_column='TRB', model_type='regression')
        
    def get_feature_columns(self, df):
        """
        Define las características específicas para predicción de rebotes.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            list: Lista de columnas de características para rebotes
        """
        # Características básicas de rebotes
        basic_features = [
            'MP',  # Minutos jugados
            'ORB', 'DRB',  # Rebotes ofensivos y defensivos
            'FGA', 'FG%',  # Intentos de tiro (influyen en rebotes)
        ]
        
        # Características de eficiencia específicas para rebotes
        efficiency_features = [
            'trb_per_minute',
            'orb_per_minute',
            'drb_per_minute', 
            'orb_drb_ratio',
            'trb_rate_5',
            'orb_rate_5',
            'drb_rate_5',
        ]
        
        # Características de ventanas móviles para rebotes
        rolling_features = []
        for window in [3, 5, 10, 20]:
            rolling_features.extend([
                f'TRB_mean_{window}',
                f'TRB_std_{window}',
                f'ORB_mean_{window}',
                f'DRB_mean_{window}',
                f'MP_mean_{window}',
            ])
        
        # Características físicas (muy importantes para rebotes)
        physical_features = [
            'Height_Inches',
            'Weight',
            'BMI',
            'trb_per_height',
            'trb_per_weight',
        ]
        
        # Características de posición (cruciales para rebotes)
        position_features = [
            'mapped_pos',
            'trb_vs_position_avg',
        ]
        
        # Características contextuales
        context_features = [
            'is_home',
            'is_started',
            'is_win',
            'trb_home_away_diff',
            'trb_win_loss_diff',
        ]
        
        # Características vs oponentes
        matchup_features = [
            'trb_vs_opp_diff',
        ]
        
        # Características de tendencias recientes
        trend_features = [
            'recent_form_trb',
            'consistency_trb',
            'hot_streak_trb',
        ]
        
        # Combinar todas las características
        all_features = (
            basic_features + 
            efficiency_features + 
            rolling_features + 
            physical_features + 
            position_features + 
            context_features + 
            matchup_features + 
            trend_features
        )
        
        # Filtrar solo las características que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Características disponibles para rebotes: {len(available_features)}/{len(all_features)}")
        
        return available_features
    
    def preprocess_target(self, df):
        """
        Preprocesa la variable objetivo TRB.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            pd.Series: Serie con la variable objetivo procesada
        """
        if 'TRB' not in df.columns:
            raise ValueError("Columna 'TRB' no encontrada en el DataFrame")
        
        # Convertir a numérico y manejar valores faltantes
        trb = pd.to_numeric(df['TRB'], errors='coerce')
        
        # Eliminar valores extremos (outliers)
        # Definir límites razonables para rebotes en NBA
        trb = trb.clip(lower=0, upper=35)  # Máximo histórico aprox. 34 rebotes (Wilt)
        
        logger.info(f"Estadísticas de rebotes - Media: {trb.mean():.1f}, "
                   f"Mediana: {trb.median():.1f}, "
                   f"Min: {trb.min()}, Max: {trb.max()}")
        
        return trb
    
    def get_prediction_context(self, player_name, df, n_games=5):
        """
        Obtiene contexto específico para la predicción de rebotes de un jugador.
        
        Args:
            player_name (str): Nombre del jugador
            df (pd.DataFrame): DataFrame con los datos
            n_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Contexto de predicción específico para rebotes
        """
        player_data = df[df['Player'] == player_name].copy()
        
        if len(player_data) == 0:
            return {}
        
        # Ordenar por fecha (más reciente primero)
        player_data = player_data.sort_values('Date', ascending=False)
        recent_games = player_data.head(n_games)
        
        context = {
            'avg_trb_recent': recent_games['TRB'].mean(),
            'avg_trb_season': player_data['TRB'].mean(),
            'avg_minutes_recent': recent_games['MP'].mean(),
            'rebounding_consistency': recent_games['TRB'].std(),
            'games_with_10plus_rebounds': (recent_games['TRB'] >= 10).sum(),
            'games_with_15plus_rebounds': (recent_games['TRB'] >= 15).sum(),
            'highest_rebounds_recent': recent_games['TRB'].max(),
            'lowest_rebounds_recent': recent_games['TRB'].min(),
        }
        
        # Separar rebotes ofensivos y defensivos si están disponibles
        if 'ORB' in recent_games.columns and 'DRB' in recent_games.columns:
            context.update({
                'avg_orb_recent': recent_games['ORB'].mean(),
                'avg_drb_recent': recent_games['DRB'].mean(),
                'orb_drb_ratio_recent': recent_games['ORB'].sum() / max(recent_games['DRB'].sum(), 1),
            })
        
        # Tendencia de rebotes (mejorando/empeorando)
        if len(recent_games) >= 3:
            context['rebounding_trend'] = recent_games['TRB'].iloc[:3].mean() - recent_games['TRB'].iloc[-3:].mean()
        
        return context
    
    def analyze_rebounding_patterns(self, df):
        """
        Analiza patrones de rebotes en el dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Análisis de patrones de rebotes
        """
        analysis = {}
        
        if 'TRB' not in df.columns:
            return analysis
        
        # Estadísticas generales
        analysis['trb_stats'] = {
            'mean': df['TRB'].mean(),
            'median': df['TRB'].median(),
            'std': df['TRB'].std(),
            'min': df['TRB'].min(),
            'max': df['TRB'].max(),
        }
        
        # Distribución por rangos
        analysis['trb_distribution'] = {
            '0-4_rebounds': (df['TRB'] < 5).sum(),
            '5-9_rebounds': ((df['TRB'] >= 5) & (df['TRB'] < 10)).sum(),
            '10-14_rebounds': ((df['TRB'] >= 10) & (df['TRB'] < 15)).sum(),
            '15-19_rebounds': ((df['TRB'] >= 15) & (df['TRB'] < 20)).sum(),
            '20plus_rebounds': (df['TRB'] >= 20).sum(),
        }
        
        # Análisis por posición
        if 'mapped_pos' in df.columns:
            analysis['trb_by_position'] = df.groupby('mapped_pos')['TRB'].agg(['mean', 'std']).to_dict()
        
        # Análisis de rebotes ofensivos vs defensivos
        if 'ORB' in df.columns and 'DRB' in df.columns:
            analysis['orb_vs_drb'] = {
                'orb_mean': df['ORB'].mean(),
                'drb_mean': df['DRB'].mean(),
                'orb_drb_ratio_avg': df['ORB'].sum() / max(df['DRB'].sum(), 1),
                'players_with_more_orb': (df['ORB'] > df['DRB']).sum(),
            }
        
        # Análisis casa vs visitante
        if 'is_home' in df.columns:
            analysis['home_vs_away'] = {
                'home_avg': df[df['is_home'] == 1]['TRB'].mean(),
                'away_avg': df[df['is_home'] == 0]['TRB'].mean(),
            }
        
        # Top reboteadores
        if 'Player' in df.columns:
            top_rebounders = df.groupby('Player')['TRB'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False)
            analysis['top_rebounders'] = top_rebounders.head(10).to_dict()
        
        # Correlación con altura y peso
        if 'Height_Inches' in df.columns:
            analysis['height_correlation'] = df['TRB'].corr(df['Height_Inches'])
        
        if 'Weight' in df.columns:
            analysis['weight_correlation'] = df['TRB'].corr(df['Weight'])
        
        logger.info("Análisis de patrones de rebotes completado")
        
        return analysis
    
    def get_position_rebounding_insights(self, df):
        """
        Obtiene insights específicos de rebotes por posición.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Insights de rebotes por posición
        """
        insights = {}
        
        if 'mapped_pos' not in df.columns or 'TRB' not in df.columns:
            return insights
        
        position_stats = df.groupby('mapped_pos').agg({
            'TRB': ['mean', 'std', 'max'],
            'ORB': 'mean' if 'ORB' in df.columns else lambda x: 0,
            'DRB': 'mean' if 'DRB' in df.columns else lambda x: 0,
            'MP': 'mean'
        }).round(2)
        
        insights['position_stats'] = position_stats.to_dict()
        
        # Identificar posiciones más efectivas en rebotes
        avg_by_pos = df.groupby('mapped_pos')['TRB'].mean().sort_values(ascending=False)
        insights['best_rebounding_positions'] = avg_by_pos.to_dict()
        
        # Eficiencia de rebotes por minuto por posición
        if 'MP' in df.columns:
            df_temp = df[df['MP'] > 0].copy()
            df_temp['trb_per_min'] = df_temp['TRB'] / df_temp['MP']
            efficiency_by_pos = df_temp.groupby('mapped_pos')['trb_per_min'].mean().sort_values(ascending=False)
            insights['rebounding_efficiency_by_position'] = efficiency_by_pos.to_dict()
        
        return insights 