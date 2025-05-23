import pandas as pd
import numpy as np
import logging
from .base_model import BaseNBAModel

logger = logging.getLogger(__name__)

class AssistsModel(BaseNBAModel):
    """
    Modelo específico para predecir asistencias (AST) de jugadores NBA.
    Hereda de BaseNBAModel y define características específicas para asistencias.
    """
    
    def __init__(self):
        super().__init__(target_column='AST', model_type='regression')
        
    def get_feature_columns(self, df):
        """
        Define las características específicas para predicción de asistencias.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            list: Lista de columnas de características para asistencias
        """
        # Características básicas de asistencias
        basic_features = [
            'MP',  # Minutos jugados
            'TOV',  # Pérdidas de balón (relacionado con asistencias)
            'PF',   # Faltas personales
        ]
        
        # Características de eficiencia específicas para asistencias
        efficiency_features = [
            'ast_per_minute',
            'ast_to_tov_ratio',
            'ast_vs_position_avg',
            'playmaking_rating',
        ]
        
        # Características de ventanas móviles para asistencias
        rolling_features = []
        for window in [3, 5, 10, 20]:
            rolling_features.extend([
                f'AST_mean_{window}',
                f'AST_std_{window}',
                f'TOV_mean_{window}',
                f'MP_mean_{window}',
            ])
        
        # Características de posición (muy importantes para asistencias)
        position_features = [
            'mapped_pos',
            'is_guard',  # Los bases suelen tener más asistencias
        ]
        
        # Características contextuales
        context_features = [
            'is_home',
            'is_started',
            'is_win',
            'ast_home_away_diff',
            'ast_win_loss_diff',
        ]
        
        # Características vs oponentes
        matchup_features = [
            'ast_vs_opp_diff',
        ]
        
        # Características de tendencias recientes
        trend_features = [
            'recent_form_ast',
            'consistency_ast',
            'hot_streak_ast',
        ]
        
        # Características de tempo de equipo (si están disponibles)
        team_features = [
            'team_pace',
            'team_off_rating',
            'possessions_per_game',
        ]
        
        # Combinar todas las características
        all_features = (
            basic_features + 
            efficiency_features + 
            rolling_features + 
            position_features + 
            context_features + 
            matchup_features + 
            trend_features +
            team_features
        )
        
        # Filtrar solo las características que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Características disponibles para asistencias: {len(available_features)}/{len(all_features)}")
        
        return available_features
    
    def preprocess_target(self, df):
        """
        Preprocesa la variable objetivo AST.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            pd.Series: Serie con la variable objetivo procesada
        """
        if 'AST' not in df.columns:
            raise ValueError("Columna 'AST' no encontrada en el DataFrame")
        
        # Convertir a numérico y manejar valores faltantes
        ast = pd.to_numeric(df['AST'], errors='coerce')
        
        # Eliminar valores extremos (outliers)
        # Definir límites razonables para asistencias en NBA
        ast = ast.clip(lower=0, upper=25)  # Máximo histórico aprox. 24 asistencias (Scott Skiles)
        
        logger.info(f"Estadísticas de asistencias - Media: {ast.mean():.1f}, "
                   f"Mediana: {ast.median():.1f}, "
                   f"Min: {ast.min()}, Max: {ast.max()}")
        
        return ast
    
    def get_prediction_context(self, player_name, df, n_games=5):
        """
        Obtiene contexto específico para la predicción de asistencias de un jugador.
        
        Args:
            player_name (str): Nombre del jugador
            df (pd.DataFrame): DataFrame con los datos
            n_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Contexto de predicción específico para asistencias
        """
        player_data = df[df['Player'] == player_name].copy()
        
        if len(player_data) == 0:
            return {}
        
        # Ordenar por fecha (más reciente primero)
        player_data = player_data.sort_values('Date', ascending=False)
        recent_games = player_data.head(n_games)
        
        context = {
            'avg_ast_recent': recent_games['AST'].mean(),
            'avg_ast_season': player_data['AST'].mean(),
            'avg_minutes_recent': recent_games['MP'].mean(),
            'playmaking_consistency': recent_games['AST'].std(),
            'games_with_5plus_assists': (recent_games['AST'] >= 5).sum(),
            'games_with_10plus_assists': (recent_games['AST'] >= 10).sum(),
            'highest_assists_recent': recent_games['AST'].max(),
            'lowest_assists_recent': recent_games['AST'].min(),
        }
        
        # Ratio asistencias/pérdidas si está disponible
        if 'TOV' in recent_games.columns:
            context.update({
                'avg_turnovers_recent': recent_games['TOV'].mean(),
                'ast_to_tov_ratio_recent': recent_games['AST'].sum() / max(recent_games['TOV'].sum(), 1),
            })
        
        # Información de posición
        if 'mapped_pos' in recent_games.columns:
            most_common_pos = recent_games['mapped_pos'].mode()
            if len(most_common_pos) > 0:
                context['primary_position'] = most_common_pos.iloc[0]
                context['is_likely_point_guard'] = most_common_pos.iloc[0] == 'G'
        
        # Tendencia de asistencias (mejorando/empeorando)
        if len(recent_games) >= 3:
            context['playmaking_trend'] = recent_games['AST'].iloc[:3].mean() - recent_games['AST'].iloc[-3:].mean()
        
        return context
    
    def analyze_playmaking_patterns(self, df):
        """
        Analiza patrones de asistencias en el dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Análisis de patrones de asistencias
        """
        analysis = {}
        
        if 'AST' not in df.columns:
            return analysis
        
        # Estadísticas generales
        analysis['ast_stats'] = {
            'mean': df['AST'].mean(),
            'median': df['AST'].median(),
            'std': df['AST'].std(),
            'min': df['AST'].min(),
            'max': df['AST'].max(),
        }
        
        # Distribución por rangos
        analysis['ast_distribution'] = {
            '0-2_assists': (df['AST'] < 3).sum(),
            '3-5_assists': ((df['AST'] >= 3) & (df['AST'] < 6)).sum(),
            '6-9_assists': ((df['AST'] >= 6) & (df['AST'] < 10)).sum(),
            '10-14_assists': ((df['AST'] >= 10) & (df['AST'] < 15)).sum(),
            '15plus_assists': (df['AST'] >= 15).sum(),
        }
        
        # Análisis por posición
        if 'mapped_pos' in df.columns:
            analysis['ast_by_position'] = df.groupby('mapped_pos')['AST'].agg(['mean', 'std']).to_dict()
        
        # Análisis de ratio asistencias/pérdidas
        if 'TOV' in df.columns:
            df_valid = df[(df['AST'] > 0) & (df['TOV'] > 0)].copy()
            df_valid['ast_tov_ratio'] = df_valid['AST'] / df_valid['TOV']
            
            analysis['ast_tov_analysis'] = {
                'avg_ast_tov_ratio': df_valid['ast_tov_ratio'].mean(),
                'players_with_positive_ratio': (df_valid['ast_tov_ratio'] > 1).sum(),
                'best_ast_tov_ratio': df_valid['ast_tov_ratio'].max(),
                'worst_ast_tov_ratio': df_valid['ast_tov_ratio'].min(),
            }
        
        # Análisis casa vs visitante
        if 'is_home' in df.columns:
            analysis['home_vs_away'] = {
                'home_avg': df[df['is_home'] == 1]['AST'].mean(),
                'away_avg': df[df['is_home'] == 0]['AST'].mean(),
            }
        
        # Análisis de titulares vs suplentes
        if 'is_started' in df.columns:
            analysis['starter_vs_bench'] = {
                'starter_avg': df[df['is_started'] == 1]['AST'].mean(),
                'bench_avg': df[df['is_started'] == 0]['AST'].mean(),
            }
        
        # Top asistidores
        if 'Player' in df.columns:
            top_playmakers = df.groupby('Player')['AST'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False)
            analysis['top_playmakers'] = top_playmakers.head(10).to_dict()
        
        logger.info("Análisis de patrones de asistencias completado")
        
        return analysis
    
    def get_position_playmaking_insights(self, df):
        """
        Obtiene insights específicos de asistencias por posición.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Insights de asistencias por posición
        """
        insights = {}
        
        if 'mapped_pos' not in df.columns or 'AST' not in df.columns:
            return insights
        
        position_stats = df.groupby('mapped_pos').agg({
            'AST': ['mean', 'std', 'max'],
            'TOV': 'mean' if 'TOV' in df.columns else lambda x: 0,
            'MP': 'mean'
        }).round(2)
        
        insights['position_stats'] = position_stats.to_dict()
        
        # Identificar posiciones más efectivas en asistencias
        avg_by_pos = df.groupby('mapped_pos')['AST'].mean().sort_values(ascending=False)
        insights['best_playmaking_positions'] = avg_by_pos.to_dict()
        
        # Eficiencia de asistencias por minuto por posición
        if 'MP' in df.columns:
            df_temp = df[df['MP'] > 0].copy()
            df_temp['ast_per_min'] = df_temp['AST'] / df_temp['MP']
            efficiency_by_pos = df_temp.groupby('mapped_pos')['ast_per_min'].mean().sort_values(ascending=False)
            insights['playmaking_efficiency_by_position'] = efficiency_by_pos.to_dict()
        
        # Ratio asistencias/pérdidas por posición
        if 'TOV' in df.columns:
            df_temp = df[(df['AST'] > 0) & (df['TOV'] > 0)].copy()
            df_temp['ast_tov_ratio'] = df_temp['AST'] / df_temp['TOV']
            ratio_by_pos = df_temp.groupby('mapped_pos')['ast_tov_ratio'].mean().sort_values(ascending=False)
            insights['ast_tov_ratio_by_position'] = ratio_by_pos.to_dict()
        
        return insights
    
    def identify_playmaker_types(self, df):
        """
        Identifica diferentes tipos de armadores basado en sus patrones de asistencias.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Clasificación de tipos de armadores
        """
        types = {}
        
        if 'Player' not in df.columns or 'AST' not in df.columns:
            return types
        
        # Calcular estadísticas por jugador
        player_stats = df.groupby('Player').agg({
            'AST': ['mean', 'std'],
            'MP': 'mean',
            'TOV': 'mean' if 'TOV' in df.columns else lambda x: 0,
        }).round(2)
        
        player_stats.columns = ['avg_ast', 'ast_consistency', 'avg_mp', 'avg_tov']
        
        # Calcular asistencias por 36 minutos para normalizar
        player_stats['ast_per_36'] = (player_stats['avg_ast'] / player_stats['avg_mp'] * 36).clip(0, 20)
        
        # Clasificar tipos de armadores
        conditions = [
            (player_stats['avg_ast'] >= 8, 'Elite Playmaker'),
            (player_stats['avg_ast'] >= 6, 'Primary Playmaker'),
            (player_stats['avg_ast'] >= 4, 'Secondary Playmaker'),
            (player_stats['avg_ast'] >= 2, 'Role Player'),
        ]
        
        for condition, label in conditions:
            players_in_category = player_stats[condition].index.tolist()
            if players_in_category:
                types[label] = {
                    'players': players_in_category,
                    'count': len(players_in_category),
                    'avg_assists': player_stats[condition]['avg_ast'].mean(),
                    'avg_consistency': player_stats[condition]['ast_consistency'].mean(),
                }
        
        return types 