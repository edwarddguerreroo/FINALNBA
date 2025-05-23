import pandas as pd
import numpy as np
import logging
from .base_model import BaseNBAModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class DoubleDoubleModel(BaseNBAModel):
    """
    Modelo específico para predecir doble-dobles y triple-dobles de jugadores NBA.
    Hereda de BaseNBAModel y define características específicas para estas categorías.
    """
    
    def __init__(self, target='double_double'):
        """
        Inicializa el modelo para doble-dobles o triple-dobles.
        
        Args:
            target (str): 'double_double' o 'triple_double'
        """
        if target not in ['double_double', 'triple_double']:
            raise ValueError("target debe ser 'double_double' o 'triple_double'")
        
        super().__init__(target_column=target, model_type='classification')
        self.target = target
        
    def get_feature_columns(self, df):
        """
        Define las características específicas para predicción de doble-dobles/triple-dobles.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            list: Lista de columnas de características
        """
        # Características básicas de estadísticas principales
        basic_features = [
            'MP',  # Minutos jugados
            'PTS', 'TRB', 'AST', 'STL', 'BLK',  # Estadísticas principales
            'FGA', 'FG%',  # Eficiencia de tiro
        ]
        
        # Características específicas de doble-doble/triple-doble
        dd_features = [
            'PTS_double', 'TRB_double', 'AST_double', 'STL_double', 'BLK_double',  # Indicadores de 10+
        ]
        
        # Características de ventanas móviles para estadísticas principales
        rolling_features = []
        for window in [3, 5, 10, 20]:
            rolling_features.extend([
                f'PTS_mean_{window}', f'PTS_std_{window}',
                f'TRB_mean_{window}', f'TRB_std_{window}',
                f'AST_mean_{window}', f'AST_std_{window}',
                f'STL_mean_{window}', f'BLK_mean_{window}',
                f'MP_mean_{window}',
                f'double_double_rate_{window}',
                f'triple_double_rate_{window}',
            ])
        
        # Características de eficiencia y producción
        efficiency_features = [
            'pts_per_minute',
            'trb_per_minute',
            'ast_per_minute',
            'production_rate',  # Combinación de estadísticas principales
            'versatility_score',  # Qué tan versátil es el jugador
        ]
        
        # Características de posición (muy importantes para doble-dobles)
        position_features = [
            'mapped_pos',
            'Height_Inches',
            'Weight',
            'BMI',
            'is_guard',
            'is_forward',
            'is_center',
        ]
        
        # Características contextuales
        context_features = [
            'is_home',
            'is_started',
            'is_win',
            'usage_rate',
        ]
        
        # Características de tendencias y rachas
        trend_features = [
            'recent_dd_streak',
            'recent_td_streak',
            'dd_consistency',
            'td_consistency',
            'hot_streak_indicator',
        ]
        
        # Características vs oponentes
        matchup_features = [
            'pts_vs_opp_diff',
            'trb_vs_opp_diff',
            'ast_vs_opp_diff',
        ]
        
        # Combinar todas las características
        all_features = (
            basic_features + 
            dd_features +
            rolling_features + 
            efficiency_features + 
            position_features + 
            context_features + 
            trend_features +
            matchup_features
        )
        
        # Filtrar solo las características que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Características disponibles para {self.target}: {len(available_features)}/{len(all_features)}")
        
        return available_features
    
    def preprocess_target(self, df):
        """
        Preprocesa la variable objetivo (double_double o triple_double).
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            pd.Series: Serie con la variable objetivo procesada
        """
        if self.target not in df.columns:
            raise ValueError(f"Columna '{self.target}' no encontrada en el DataFrame")
        
        # Convertir a numérico y manejar valores faltantes
        target_values = pd.to_numeric(df[self.target], errors='coerce').fillna(0)
        
        # Asegurar que sean valores binarios (0 o 1)
        target_values = target_values.astype(int).clip(0, 1)
        
        logger.info(f"Estadísticas de {self.target} - "
                   f"Total: {len(target_values)}, "
                   f"Positivos: {target_values.sum()} ({target_values.mean()*100:.1f}%)")
        
        return target_values
    
    def get_prediction_context(self, player_name, df, n_games=5):
        """
        Obtiene contexto específico para la predicción de doble-dobles/triple-dobles.
        
        Args:
            player_name (str): Nombre del jugador
            df (pd.DataFrame): DataFrame con los datos
            n_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Contexto de predicción específico
        """
        player_data = df[df['Player'] == player_name].copy()
        
        if len(player_data) == 0:
            return {}
        
        # Ordenar por fecha (más reciente primero)
        player_data = player_data.sort_values('Date', ascending=False)
        recent_games = player_data.head(n_games)
        
        context = {
            'avg_pts_recent': recent_games['PTS'].mean(),
            'avg_trb_recent': recent_games['TRB'].mean(),
            'avg_ast_recent': recent_games['AST'].mean(),
            'avg_minutes_recent': recent_games['MP'].mean(),
            'dd_rate_recent': recent_games['double_double'].mean() if 'double_double' in recent_games.columns else 0,
            'td_rate_recent': recent_games['triple_double'].mean() if 'triple_double' in recent_games.columns else 0,
            'dd_rate_season': player_data['double_double'].mean() if 'double_double' in player_data.columns else 0,
            'td_rate_season': player_data['triple_double'].mean() if 'triple_double' in player_data.columns else 0,
        }
        
        # Información sobre las estadísticas que más contribuyen
        if 'double_double' in recent_games.columns:
            # Analizar qué estadísticas alcanzan más frecuentemente el umbral de 10
            for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                if stat in recent_games.columns:
                    context[f'{stat.lower()}_double_rate'] = (recent_games[stat] >= 10).mean()
        
        # Rachas actuales
        if len(recent_games) >= 3:
            # Racha de doble-dobles
            dd_streak = 0
            for dd in recent_games['double_double'] if 'double_double' in recent_games.columns else []:
                if dd == 1:
                    dd_streak += 1
                else:
                    break
            context['current_dd_streak'] = dd_streak
            
            # Racha de triple-dobles
            td_streak = 0
            for td in recent_games['triple_double'] if 'triple_double' in recent_games.columns else []:
                if td == 1:
                    td_streak += 1
                else:
                    break
            context['current_td_streak'] = td_streak
        
        # Información de posición y rol
        if 'mapped_pos' in recent_games.columns:
            most_common_pos = recent_games['mapped_pos'].mode()
            if len(most_common_pos) > 0:
                context['primary_position'] = most_common_pos.iloc[0]
        
        return context
    
    def analyze_double_double_patterns(self, df):
        """
        Analiza patrones de doble-dobles y triple-dobles en el dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            dict: Análisis de patrones
        """
        analysis = {}
        
        # Estadísticas generales
        if 'double_double' in df.columns:
            analysis['dd_stats'] = {
                'total_games': len(df),
                'total_double_doubles': df['double_double'].sum(),
                'dd_rate': df['double_double'].mean(),
                'players_with_dd': df[df['double_double'] == 1]['Player'].nunique() if 'Player' in df.columns else 0,
            }
        
        if 'triple_double' in df.columns:
            analysis['td_stats'] = {
                'total_triple_doubles': df['triple_double'].sum(),
                'td_rate': df['triple_double'].mean(),
                'players_with_td': df[df['triple_double'] == 1]['Player'].nunique() if 'Player' in df.columns else 0,
            }
        
        # Análisis por posición
        if 'mapped_pos' in df.columns and 'double_double' in df.columns:
            dd_by_pos = df.groupby('mapped_pos')['double_double'].agg(['mean', 'sum', 'count'])
            analysis['dd_by_position'] = dd_by_pos.to_dict()
        
        if 'mapped_pos' in df.columns and 'triple_double' in df.columns:
            td_by_pos = df.groupby('mapped_pos')['triple_double'].agg(['mean', 'sum', 'count'])
            analysis['td_by_position'] = td_by_pos.to_dict()
        
        # Análisis de combinaciones de estadísticas más comunes
        if all(col in df.columns for col in ['PTS_double', 'TRB_double', 'AST_double']):
            # Combinaciones para doble-dobles
            dd_combinations = df[df['double_double'] == 1].copy()
            if len(dd_combinations) > 0:
                combo_counts = {}
                stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
                for _, row in dd_combinations.iterrows():
                    combo = tuple(sorted([stat for stat in stats if f'{stat}_double' in df.columns and row.get(f'{stat}_double', 0) == 1]))
                    if len(combo) >= 2:  # Al menos doble-doble
                        combo_counts[combo] = combo_counts.get(combo, 0) + 1
                
                # Convertir tuplas a strings para JSON serialization
                combo_counts_str = {str(combo): count for combo, count in combo_counts.items()}
                analysis['most_common_dd_combinations'] = dict(sorted(combo_counts_str.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Top performers
        if 'Player' in df.columns:
            if 'double_double' in df.columns:
                top_dd_players = df.groupby('Player')['double_double'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
                analysis['top_dd_players'] = top_dd_players.head(10).to_dict()
            
            if 'triple_double' in df.columns:
                top_td_players = df.groupby('Player')['triple_double'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
                analysis['top_td_players'] = top_td_players.head(10).to_dict()
        
        # Análisis de casa vs visitante
        if 'is_home' in df.columns:
            if 'double_double' in df.columns:
                analysis['dd_home_vs_away'] = {
                    'home_rate': df[df['is_home'] == 1]['double_double'].mean(),
                    'away_rate': df[df['is_home'] == 0]['double_double'].mean(),
                }
            
            if 'triple_double' in df.columns:
                analysis['td_home_vs_away'] = {
                    'home_rate': df[df['is_home'] == 1]['triple_double'].mean(),
                    'away_rate': df[df['is_home'] == 0]['triple_double'].mean(),
                }
        
        logger.info("Análisis de patrones de doble-dobles/triple-dobles completado")
        
        return analysis
    
    def predict_with_probability(self, X, model_name=None):
        """
        Realiza predicciones con probabilidades para clasificación.
        
        Args:
            X: Características para predicción
            model_name (str): Nombre del modelo a usar (None para el mejor)
            
        Returns:
            tuple: (predicciones, probabilidades)
        """
        if not self.is_fitted:
            raise ValueError("Los modelos no han sido entrenados")
        
        if model_name is None:
            model_name, model, _ = self.get_best_model()
        else:
            model = self.models[model_name]
        
        # Aplicar escalado si es necesario
        if 'main' in self.scalers and model_name not in ['xgboost', 'lightgbm', 'random_forest']:
            X_scaled = self.scalers['main'].transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
        else:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
        
        return predictions, probabilities
    
    def get_feature_importance_for_dd(self):
        """
        Obtiene importancia de características específicamente para doble-dobles/triple-dobles.
        
        Returns:
            dict: Importancia de características con interpretación específica
        """
        importance_df = self.get_feature_importance_summary()
        
        if importance_df.empty:
            return {}
        
        # Categorizar características por tipo
        categories = {
            'statistical_production': [],
            'efficiency_metrics': [],
            'position_physical': [],
            'contextual_factors': [],
            'recent_trends': []
        }
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['mean_importance']
            
            if any(stat in feature for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']):
                categories['statistical_production'].append((feature, importance))
            elif any(metric in feature for metric in ['per_minute', 'rate', 'efficiency']):
                categories['efficiency_metrics'].append((feature, importance))
            elif any(factor in feature for factor in ['Height', 'Weight', 'pos', 'guard', 'forward', 'center']):
                categories['position_physical'].append((feature, importance))
            elif any(context in feature for context in ['home', 'started', 'win']):
                categories['contextual_factors'].append((feature, importance))
            else:
                categories['recent_trends'].append((feature, importance))
        
        # Ordenar cada categoría por importancia
        for category in categories:
            categories[category] = sorted(categories[category], key=lambda x: x[1], reverse=True)
        
        return categories
    
    def identify_dd_td_candidates(self, df, min_games=10):
        """
        Identifica jugadores candidatos a doble-dobles/triple-dobles basado en patrones.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            min_games (int): Mínimo de juegos para considerar al jugador
            
        Returns:
            dict: Candidatos clasificados por probabilidad
        """
        candidates = {}
        
        if 'Player' not in df.columns:
            return candidates
        
        # Calcular estadísticas por jugador
        player_stats = df.groupby('Player').agg({
            'PTS': ['mean', 'std'],
            'TRB': ['mean', 'std'],
            'AST': ['mean', 'std'],
            'STL': ['mean', 'std'],
            'BLK': ['mean', 'std'],
            'MP': 'mean',
            'double_double': ['mean', 'sum'] if 'double_double' in df.columns else lambda x: [0, 0],
            'triple_double': ['mean', 'sum'] if 'triple_double' in df.columns else lambda x: [0, 0],
        })
        
        # Filtrar jugadores con suficientes juegos
        games_per_player = df['Player'].value_counts()
        qualified_players = games_per_player[games_per_player >= min_games].index
        
        for player in qualified_players:
            player_data = player_stats.loc[player]
            
            # Calcular score de potencial para doble-doble
            dd_potential = 0
            stats_near_10 = 0
            
            for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                avg = player_data[stat]['mean']
                if avg >= 10:
                    dd_potential += 2
                elif avg >= 8:
                    dd_potential += 1.5
                    stats_near_10 += 1
                elif avg >= 6:
                    dd_potential += 1
                    stats_near_10 += 1
            
            # Calcular score de potencial para triple-doble
            td_potential = dd_potential * 0.3  # Base más baja para triple-doble
            stats_above_8 = sum(1 for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK'] 
                               if player_data[stat]['mean'] >= 8)
            
            if stats_above_8 >= 3:
                td_potential += 2
            elif stats_above_8 >= 2:
                td_potential += 1
            
            # Factores adicionales
            minutes_factor = min(player_data['MP']['mean'] / 30, 1.2)  # Bonus por minutos
            dd_potential *= minutes_factor
            td_potential *= minutes_factor
            
            candidates[player] = {
                'dd_potential_score': round(dd_potential, 2),
                'td_potential_score': round(td_potential, 2),
                'stats_near_double_digits': stats_near_10,
                'current_dd_rate': player_data['double_double']['mean'] if 'double_double' in df.columns else 0,
                'current_td_rate': player_data['triple_double']['mean'] if 'triple_double' in df.columns else 0,
                'avg_stats': {
                    stat: round(player_data[stat]['mean'], 1) 
                    for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']
                }
            }
        
        # Ordenar por potencial
        candidates = dict(sorted(candidates.items(), 
                                key=lambda x: x[1]['dd_potential_score'], 
                                reverse=True))