import pandas as pd
import numpy as np
import logging
from .base_model import BaseNBAModel

logger = logging.getLogger(__name__)

class ThreesModel(BaseNBAModel):
    """
    Modelo específico para predecir triples anotados (3P) de jugadores NBA.
    Hereda de BaseNBAModel y define características específicas para triples.
    """
    
    def __init__(self):
        super().__init__(target_column='3P', model_type='regression')
        
    def get_feature_columns(self, df):
        """
        Define las características específicas para predicción de triples.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            list: Lista de columnas de características para triples
        """
        # Características básicas de triples
        basic_features = [
            'MP',  # Minutos jugados
            '3PA', '3P%',  # Intentos y porcentaje de triples
            'FGA', 'FG%',  # Intentos totales y porcentaje general
        ]
        
        # Características de eficiencia específicas para triples
        efficiency_features = [
            '3p_per_minute',
            '3p_per_attempt',
            '3p_vs_position_avg',
            '3p_attempt_rate',
            '3p_volume_rating',
        ]
        
        # Características de ventanas móviles para triples
        rolling_features = []
        for window in [3, 5, 10, 20]:
            rolling_features.extend([
                f'3P_mean_{window}',
                f'3P_std_{window}',
                f'3PA_mean_{window}',
                f'3P%_mean_{window}',
                f'MP_mean_{window}',
            ])
        
        # Características de posición (importantes para triples)
        position_features = [
            'mapped_pos',
            'is_guard',  # Los guards suelen tirar más triples
            'is_forward',
            'is_center',
        ]
        
        # Características contextuales
        context_features = [
            'is_home',
            'is_started',
            'is_win',
            '3p_home_away_diff',
            '3p_win_loss_diff',
        ]
        
        # Características vs oponentes
        matchup_features = [
            '3p_vs_opp_diff',
            'opp_3p_defense_rating',
        ]
        
        # Características de tendencias recientes
        trend_features = [
            'recent_form_3p',
            'hot_streak_3p',
            'consistency_3p',
            '3p_momentum',
        ]
        
        # Características de tempo y estilo de juego
        pace_features = [
            'team_pace',
            'team_3pa_per_game',
            'usage_rate',
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
            pace_features
        )
        
        # Filtrar solo las características que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Características disponibles para triples: {len(available_features)}/{len(all_features)}")
        
        return available_features
    
    def preprocess_target(self, df):
        """
        Preprocesa la variable objetivo 3P.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            pd.Series: Serie con la variable objetivo procesada
        """
        if '3P' not in df.columns:
            raise ValueError("Columna '3P' no encontrada en el DataFrame")
        
        # Convertir a numérico y manejar valores faltantes
        threes = pd.to_numeric(df['3P'], errors='coerce')
        
        # Eliminar valores extremos (outliers)
        # Definir límites razonables para triples en NBA
        threes = threes.clip(lower=0, upper=15)  # Máximo histórico aprox. 14 triples (Klay Thompson)
        
        logger.info(f"Estadísticas de triples - Media: {threes.mean():.1f}, "
                   f"Mediana: {threes.median():.1f}, "
                   f"Min: {threes.min()}, Max: {threes.max()}")
        
        return threes
    
    def get_prediction_context(self, player_name, df, n_games=5):
        """
        Obtiene contexto específico para la predicción de triples de un jugador.
        
        Args:
            player_name (str): Nombre del jugador
            df (pd.DataFrame): DataFrame con los datos
            n_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Contexto de predicción específico para triples
        """
        player_data = df[df['Player'] == player_name].copy()
        
        if len(player_data) == 0:
            return {}
        
        # Ordenar por fecha (más reciente primero)
        player_data = player_data.sort_values('Date', ascending=False)
        recent_games = player_data.head(n_games)
        
        context = {
            'avg_3p_recent': recent_games['3P'].mean(),
            'avg_3p_season': player_data['3P'].mean(),
            'avg_3pa_recent': recent_games['3PA'].mean() if '3PA' in recent_games.columns else 0,
            'avg_3p_pct_recent': recent_games['3P%'].mean() if '3P%' in recent_games.columns else 0,
            'avg_minutes_recent': recent_games['MP'].mean(),
            'three_point_consistency': recent_games['3P'].std(),
            'games_with_3plus_threes': (recent_games['3P'] >= 3).sum(),
            'games_with_5plus_threes': (recent_games['3P'] >= 5).sum(),
            'highest_threes_recent': recent_games['3P'].max(),
            'games_without_threes': (recent_games['3P'] == 0).sum(),
        }
        
        # Información de volumen de intentos
        if '3PA' in recent_games.columns:
            context.update({
                'attempts_per_game_recent': recent_games['3PA'].mean(),
                'attempts_consistency': recent_games['3PA'].std(),
                'max_attempts_recent': recent_games['3PA'].max(),
            })
        
        # Información de eficiencia
        if '3P%' in recent_games.columns:
            context.update({
                'shooting_pct_recent': recent_games['3P%'].mean(),
                'shooting_consistency': recent_games['3P%'].std(),
                'best_shooting_game': recent_games['3P%'].max(),
                'worst_shooting_game': recent_games['3P%'].min(),
            })
        
        # Tendencia de triples (mejorando/empeorando)
        if len(recent_games) >= 3:
            context['three_point_trend'] = recent_games['3P'].iloc[:3].mean() - recent_games['3P'].iloc[-3:].mean()
        
        # Hot streak detection
        if len(recent_games) >= 3:
            context['is_hot_from_three'] = (recent_games['3P'].head(3) >= 2).all()
            context['consecutive_games_with_threes'] = 0
            for i, threes in enumerate(recent_games['3P']):
                if threes > 0:
                    context['consecutive_games_with_threes'] = i + 1
                else:
                    break
        
        return context
    
    def analyze_three_point_patterns(self, df):
        """
        Analiza patrones de triples en el dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Análisis de patrones de triples
        """
        analysis = {}
        
        if '3P' not in df.columns:
            return analysis
        
        # Estadísticas generales
        analysis['3p_stats'] = {
            'mean': df['3P'].mean(),
            'median': df['3P'].median(),
            'std': df['3P'].std(),
            'min': df['3P'].min(),
            'max': df['3P'].max(),
        }
        
        # Distribución por rangos
        analysis['3p_distribution'] = {
            '0_threes': (df['3P'] == 0).sum(),
            '1-2_threes': ((df['3P'] >= 1) & (df['3P'] <= 2)).sum(),
            '3-4_threes': ((df['3P'] >= 3) & (df['3P'] <= 4)).sum(),
            '5-6_threes': ((df['3P'] >= 5) & (df['3P'] <= 6)).sum(),
            '7plus_threes': (df['3P'] >= 7).sum(),
        }
        
        # Análisis por posición
        if 'mapped_pos' in df.columns:
            analysis['3p_by_position'] = df.groupby('mapped_pos')['3P'].agg(['mean', 'std']).to_dict()
        
        # Análisis de eficiencia
        if '3PA' in df.columns and '3P%' in df.columns:
            df_attempts = df[df['3PA'] > 0].copy()
            
            analysis['shooting_efficiency'] = {
                'avg_attempts_per_game': df_attempts['3PA'].mean(),
                'avg_shooting_percentage': df_attempts['3P%'].mean(),
                'high_volume_shooters': (df_attempts['3PA'] >= 6).sum(),
                'efficient_shooters': (df_attempts['3P%'] >= 0.35).sum(),
                'volume_and_efficiency': ((df_attempts['3PA'] >= 6) & (df_attempts['3P%'] >= 0.35)).sum(),
            }
        
        # Análisis casa vs visitante
        if 'is_home' in df.columns:
            analysis['home_vs_away'] = {
                'home_avg': df[df['is_home'] == 1]['3P'].mean(),
                'away_avg': df[df['is_home'] == 0]['3P'].mean(),
            }
        
        # Análisis de titulares vs suplentes
        if 'is_started' in df.columns:
            analysis['starter_vs_bench'] = {
                'starter_avg': df[df['is_started'] == 1]['3P'].mean(),
                'bench_avg': df[df['is_started'] == 0]['3P'].mean(),
            }
        
        # Top tiradores de triples
        if 'Player' in df.columns:
            top_shooters = df.groupby('Player')['3P'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False)
            analysis['top_three_point_shooters'] = top_shooters.head(10).to_dict()
        
        logger.info("Análisis de patrones de triples completado")
        
        return analysis
    
    def get_shooting_style_insights(self, df):
        """
        Obtiene insights sobre estilos de tiro de triples.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Insights sobre estilos de tiro
        """
        insights = {}
        
        if '3P' not in df.columns or '3PA' not in df.columns:
            return insights
        
        # Filtrar solo jugadores con intentos
        df_shooters = df[df['3PA'] > 0].copy()
        
        # Calcular estadísticas por jugador
        player_stats = df_shooters.groupby('Player').agg({
            '3P': 'mean',
            '3PA': 'mean',
            '3P%': 'mean',
            'MP': 'mean'
        }).round(3)
        
        # Calcular rate stats
        player_stats['3p_per_game'] = player_stats['3P']
        player_stats['3pa_per_game'] = player_stats['3PA']
        player_stats['3p_rate'] = player_stats['3PA'] / player_stats['MP'] * 36  # Per 36 minutes
        
        # Clasificar tipos de tiradores
        insights['shooter_categories'] = {}
        
        # High volume shooters (>6 attempts per game)
        high_volume = player_stats[player_stats['3pa_per_game'] >= 6]
        insights['shooter_categories']['high_volume'] = {
            'count': len(high_volume),
            'avg_attempts': high_volume['3pa_per_game'].mean(),
            'avg_percentage': high_volume['3P%'].mean(),
            'top_players': high_volume.sort_values('3pa_per_game', ascending=False).head(5).index.tolist()
        }
        
        # High efficiency shooters (>40% on reasonable volume)
        high_efficiency = player_stats[(player_stats['3P%'] >= 0.40) & (player_stats['3pa_per_game'] >= 3)]
        insights['shooter_categories']['high_efficiency'] = {
            'count': len(high_efficiency),
            'avg_attempts': high_efficiency['3pa_per_game'].mean(),
            'avg_percentage': high_efficiency['3P%'].mean(),
            'top_players': high_efficiency.sort_values('3P%', ascending=False).head(5).index.tolist()
        }
        
        # Elite shooters (high volume + high efficiency)
        elite = player_stats[(player_stats['3P%'] >= 0.37) & (player_stats['3pa_per_game'] >= 6)]
        insights['shooter_categories']['elite'] = {
            'count': len(elite),
            'avg_attempts': elite['3pa_per_game'].mean(),
            'avg_percentage': elite['3P%'].mean(),
            'players': elite.index.tolist()
        }
        
        return insights
    
    def detect_hot_streaks(self, df, min_games=3, min_threes_per_game=2):
        """
        Detecta rachas calientes de triples en el dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            min_games (int): Mínimo de juegos consecutivos para considerar racha
            min_threes_per_game (int): Mínimo de triples por juego para la racha
            
        Returns:
            dict: Información sobre rachas calientes detectadas
        """
        streaks = {}
        
        if 'Player' not in df.columns or '3P' not in df.columns or 'Date' not in df.columns:
            return streaks
        
        for player in df['Player'].unique():
            player_data = df[df['Player'] == player].sort_values('Date')
            
            if len(player_data) < min_games:
                continue
            
            # Detectar rachas
            hot_games = player_data['3P'] >= min_threes_per_game
            current_streak = 0
            max_streak = 0
            streak_start = None
            best_streak_period = None
            
            for i, is_hot in enumerate(hot_games):
                if is_hot:
                    if current_streak == 0:
                        streak_start = i
                    current_streak += 1
                    if current_streak > max_streak:
                        max_streak = current_streak
                        best_streak_period = (streak_start, i)
                else:
                    current_streak = 0
            
            if max_streak >= min_games:
                streak_data = player_data.iloc[best_streak_period[0]:best_streak_period[1]+1]
                streaks[player] = {
                    'max_streak_length': max_streak,
                    'avg_threes_in_streak': streak_data['3P'].mean(),
                    'total_threes_in_streak': streak_data['3P'].sum(),
                    'streak_dates': (streak_data['Date'].min(), streak_data['Date'].max()),
                    'streak_shooting_pct': streak_data['3P%'].mean() if '3P%' in streak_data.columns else None
                }
        
        # Resumen de rachas encontradas
        if streaks:
            streak_summary = {
                'total_players_with_streaks': len(streaks),
                'longest_streak': max(s['max_streak_length'] for s in streaks.values()),
                'avg_streak_length': np.mean([s['max_streak_length'] for s in streaks.values()]),
                'best_streak_player': max(streaks.keys(), key=lambda x: streaks[x]['max_streak_length'])
            }
            streaks['summary'] = streak_summary
        
        return streaks 