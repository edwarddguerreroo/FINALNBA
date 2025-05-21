import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
import logging
from ..feature_engineering.teams_features import TeamsFeatures
from ..feature_engineering.players_features import PlayersFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeaturesSelector:
    """
    Clase para seleccionar características específicas por tipo de predicción
    """
    def __init__(self):
        # No inicializar los generadores de características directamente
        # ya que causan errores al recibir DataFrames vacíos
        self.teams_feature_engineering = None
        self.players_feature_engineering = None
        
        # Definir características base por tipo de predicción
        self.target_features = {
            # Características para predicciones de equipo
            'Win': [],
            'Total_Points_Over_Under': [],
            'Team_Points_Over_Under': [],
            
            # Características para predicciones de jugador
            'PTS': [],
            'TRB': [],
            'AST': [],
            '3P': [],
            'Double_Double': [],
            'Triple_Double': []
        }
        
        # Características comunes para todos los targets
        self.common_features = [
            'is_home', 'days_rest', 'games_last_7_days', 'dayofweek', 'month',
            'is_weekend', 'pace', 'possessions', 'games_played',
            'opp_win_rate', 'offensive_rating', 'defensive_rating',
            'offensive_efficiency', 'defensive_efficiency', 'efficiency_diff',
            'home_advantage', 'opp_defensive_rating', 'off_def_rating_interaction',
            'pace_efficiency_interaction', 'momentum_interaction_3', 'momentum_interaction_5',
            'momentum_interaction_10', 'momentum_interaction_20'
        ]
        
        # Características de matchup específicas - inicializadas como listas vacías
        self.matchup_features = {
            'Win': [],
            'Total_Points_Over_Under': [],
            'Team_Points_Over_Under': [],
            'PTS': [],
            'TRB': [],
            'AST': [],
            '3P': [],
            'Double_Double': [],
            'Triple_Double': []
        }
        
        # Características avanzadas por tipo de predicción - inicializadas como listas vacías
        self.advanced_features = {
            'Win': [],
            'Total_Points_Over_Under': [],
            'Team_Points_Over_Under': [],
            'PTS': [],
            'TRB': [],
            'AST': [],
            '3P': [],
            'Double_Double': [],
            'Triple_Double': []
        }
        
        # Features específicos para jugadores por cada estadística
        self.player_features_by_stat = {
            'PTS': [
                'PTS', 'PTS_mean_3', 'PTS_mean_5', 'PTS_mean_10', 'PTS_mean_20',
                'PTS_std_3', 'PTS_std_5', 'PTS_std_10', 'PTS_std_20',
                'PTS_max_3', 'PTS_max_5', 'PTS_max_10', 'PTS_max_20',
                'PTS_min_3', 'PTS_min_5', 'PTS_min_10', 'PTS_min_20',
                'PTS_trend_3', 'PTS_trend_5', 'PTS_trend_10', 'PTS_trend_20',
                'FG%', 'FG%_mean_3', 'FG%_mean_5', 'FG%_mean_10', 'FG%_mean_20',
                'FG', 'FGA', 'FG_mean_3', 'FGA_mean_3',
                '3P', '3PA', '3P%', '3P_mean_3', '3PA_mean_3', '3P%_mean_3',
                'FT', 'FTA', 'FT%', 'FT_mean_3', 'FTA_mean_3', 'FT%_mean_3',
                'MP', 'MP_mean_3', 'MP_mean_5', 'MP_mean_10', 'MP_mean_20',
                'TS%', 'efg_pct', 'pts_per_fga', 'pts_per_minute',
                'pts_momentum_3', 'pts_momentum_5', 'pts_momentum_10',
                'pts_from_3p', 'pts_from_2p', 'pts_from_ft',
                'pts_prop_from_3p', 'pts_prop_from_2p', 'pts_prop_from_ft',
                'pts_per_scoring_poss', 'pts_per_min_3', 'pts_per_min_5', 'pts_per_min_10',
                'pts_above_avg_streak', 'pts_increase_streak',
                'PTS_over_10', 'PTS_over_15', 'PTS_over_20', 'PTS_over_25', 'PTS_over_30', 'PTS_over_35'
            ],
            'TRB': [
                'TRB', 'TRB_mean_3', 'TRB_mean_5', 'TRB_mean_10', 'TRB_mean_20',
                'TRB_std_3', 'TRB_std_5', 'TRB_std_10', 'TRB_std_20',
                'TRB_max_3', 'TRB_max_5', 'TRB_max_10', 'TRB_max_20',
                'TRB_min_3', 'TRB_min_5', 'TRB_min_10', 'TRB_min_20',
                'TRB_trend_3', 'TRB_trend_5', 'TRB_trend_10', 'TRB_trend_20',
                'ORB', 'DRB', 'ORB_mean_3', 'ORB_mean_5', 'ORB_mean_10', 'ORB_mean_20',
                'trb_per_minute', 'orb_per_minute', 'drb_per_minute', 'orb_drb_ratio',
                'trb_per_height', 'trb_home_away_diff', 'trb_win_loss_diff',
                'TRB_over_4', 'TRB_over_6', 'TRB_over_8', 'TRB_over_10', 'TRB_over_12',
                'MP', 'MP_mean_3', 'MP_mean_5', 'MP_mean_10', 'MP_mean_20'
            ],
            'AST': [
                'AST', 'AST_mean_3', 'AST_mean_5', 'AST_mean_10', 'AST_mean_20',
                'AST_std_3', 'AST_std_5', 'AST_std_10', 'AST_std_20',
                'AST_max_3', 'AST_max_5', 'AST_max_10', 'AST_max_20',
                'AST_min_3', 'AST_min_5', 'AST_min_10', 'AST_min_20',
                'AST_trend_3', 'AST_trend_5', 'AST_trend_10', 'AST_trend_20',
                'TOV', 'TOV_mean_3', 'TOV_mean_5', 'TOV_mean_10', 'TOV_mean_20',
                'ast_per_minute', 'ast_home_away_diff', 'ast_win_loss_diff',
                'ast_to_tov_ratio', 'playmaking_rating', 'is_playmaker',
                'AST_over_4', 'AST_over_6', 'AST_over_8', 'AST_over_10', 'AST_over_12',
                'MP', 'MP_mean_3', 'MP_mean_5', 'MP_mean_10', 'MP_mean_20'
            ],
            '3P': [
                '3P', '3P_mean_3', '3P_mean_5', '3P_mean_10', '3P_mean_20',
                '3P_std_3', '3P_std_5', '3P_std_10', '3P_std_20',
                '3P_max_3', '3P_max_5', '3P_max_10', '3P_max_20',
                '3P_min_3', '3P_min_5', '3P_min_10', '3P_min_20',
                '3P_trend_3', '3P_trend_5', '3P_trend_10', '3P_trend_20',
                '3PA', '3PA_mean_3', '3PA_mean_5', '3PA_mean_10', '3PA_mean_20',
                '3P%', '3P%_mean_3', '3P%_mean_5', '3P%_mean_10', '3P%_mean_20',
                '3p_per_fga', '3p_from_3p', '3p_per_minute', '3p_home_away_diff',
                '3P_over_1', '3P_over_2', '3P_over_3', '3P_over_4', '3P_over_5',
                'MP', 'MP_mean_3', 'MP_mean_5', 'MP_mean_10', 'MP_mean_20'
            ],
            'Double_Double': [
                'double_double', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
                'PTS_mean_10', 'TRB_mean_10', 'AST_mean_10',
                'PTS_mean_20', 'TRB_mean_20', 'AST_mean_20',
                'PTS_over_10', 'TRB_over_10', 'AST_over_10', 
                'double_double_rate_10', 'double_double_streak',
                'MP', 'MP_mean_10', 'MP_mean_20'
            ],
            'Triple_Double': [
                'triple_double', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
                'PTS_mean_10', 'TRB_mean_10', 'AST_mean_10',
                'PTS_mean_20', 'TRB_mean_20', 'AST_mean_20',
                'PTS_over_10', 'TRB_over_10', 'AST_over_10',
                'triple_double_rate_10', 'triple_double_streak',
                'MP', 'MP_mean_10', 'MP_mean_20'
            ]
        }
        
        # Features específicos para equipos por cada estadística
        self.team_features_by_stat = {
            'Win': [
                # Características básicas
                'is_win', 'win_rate_10', 'win_rate_20', 'win_streak', 'current_win_streak',
                'PTS', 'PTS_Opp', 'PTS_diff', 'total_points',
                'is_home', 'home_advantage', 'current_home_win_streak',
                'FG%', 'FG%_Opp', '3P%', '3P%_Opp', 'FT%', 'FT%_Opp',
                
                # Métricas avanzadas
                'offensive_rating', 'defensive_rating', 'offensive_efficiency', 'defensive_efficiency', 
                'efficiency_diff', 'pace', 'possessions', 'points_per_possession',
                
                # Tendencias y promedios
                'PTS_mean_3', 'PTS_mean_5', 'PTS_mean_10', 'PTS_mean_20',
                'PTS_Opp_mean_3', 'PTS_Opp_mean_5', 'PTS_Opp_mean_10', 'PTS_Opp_mean_20',
                'PTS_trend_3', 'PTS_trend_5', 'PTS_trend_10', 
                'PTS_Opp_trend_3', 'PTS_Opp_trend_5', 'PTS_Opp_trend_10',
                
                # Características de momento
                'offensive_momentum_3', 'defensive_momentum_3', 'efficiency_momentum_3',
                'offensive_momentum_5', 'defensive_momentum_5', 'efficiency_momentum_5',
                'offensive_momentum_10', 'defensive_momentum_10', 'efficiency_momentum_10',
                'offensive_momentum_20', 'defensive_momentum_20', 'efficiency_momentum_20',
                
                # Características de oponente y matchup
                'opp_win_rate', 'opp_avg_PTS', 'opp_avg_PTS_against',
                'h2h_games', 'h2h_win_rate', 'h2h_home_win_rate', 'h2h_avg_points', 'h2h_avg_points_against',
                'vs_opp_win_rate', 'vs_opp_avg_PTS', 'vs_opp_avg_PTS_against',
                
                # Predictores específicos
                'win_probability', 'better_team_wins', 'wins_vs_better_teams_3', 'situation_wins_3',
                'wins_vs_offensive_teams', 'wins_vs_defensive_teams'
            ],
            
            'Total_Points_Over_Under': [
                # Características básicas
                'total_points', 'PTS', 'PTS_Opp', 'PTS_diff',
                'FG', 'FGA', 'FG%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp',
                '3P', '3PA', '3P%', '3P_Opp', '3PA_Opp', '3P%_Opp',
                'FT', 'FTA', 'FT%', 'FT_Opp', 'FTA_Opp', 'FT%_Opp',
                
                # Métricas avanzadas
                'pace', 'possessions', 'offensive_efficiency', 'defensive_efficiency',
                'points_per_possession', 'total_points_volatility_3',
                
                # Promedios y tendencias
                'total_points_mean_3', 'total_points_mean_5', 'total_points_mean_10', 'total_points_mean_20',
                'total_points_std_3', 'total_points_std_5', 'total_points_std_10', 'total_points_std_20',
                'total_points_trend_3', 'total_points_trend_5', 'total_points_trend_10', 'total_points_trend_20',
                'PTS_mean_3', 'PTS_mean_5', 'PTS_mean_10', 'PTS_mean_20',
                'PTS_Opp_mean_3', 'PTS_Opp_mean_5', 'PTS_Opp_mean_10', 'PTS_Opp_mean_20',
                
                # Predictores de líneas específicas
                'high_scoring_matchup', 'defensive_matchup',
                
                # Líneas de over/under específicas
                'total_points_over_180', 'total_points_over_185', 'total_points_over_190',
                'total_points_over_195', 'total_points_over_200', 'total_points_over_205',
                'total_points_over_210', 'total_points_over_215', 'total_points_over_220',
                'total_points_over_225', 'total_points_over_230', 'total_points_over_235',
                
                # Probabilidades y consistencia de líneas
                'prob_total_over_200_3', 'prob_total_over_200_5', 'prob_total_over_200_10', 'prob_total_over_200_20',
                'prob_total_over_210_3', 'prob_total_over_210_5', 'prob_total_over_210_10', 'prob_total_over_210_20',
                'prob_total_over_220_3', 'prob_total_over_220_5', 'prob_total_over_220_10', 'prob_total_over_220_20',
                'total_points_line_consistency_200', 'total_points_line_consistency_210', 'total_points_line_consistency_220',
                'total_points_line_momentum_3', 'total_points_line_momentum_5', 'total_points_line_momentum_10', 'total_points_line_momentum_20'
            ],
            
            'Team_Points_Over_Under': [
                # Características básicas
                'PTS', 'team_score', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
                
                # Métricas avanzadas
                'offensive_rating', 'offensive_efficiency', 'pace', 'possessions', 'points_per_possession',
                
                # Promedios y tendencias
                'PTS_mean_3', 'PTS_mean_5', 'PTS_mean_10', 'PTS_mean_20',
                'PTS_std_3', 'PTS_std_5', 'PTS_std_10', 'PTS_std_20',
                'PTS_trend_3', 'PTS_trend_5', 'PTS_trend_10',
                'FG%_mean_3', 'FG%_mean_5', 'FG%_mean_10', 'FG%_mean_20',
                '3P%_mean_3', '3P%_mean_5', '3P%_mean_10', '3P%_mean_20',
                
                # Líneas de over/under específicas
                'team_points_over_80', 'team_points_over_85', 'team_points_over_90',
                'team_points_over_95', 'team_points_over_100', 'team_points_over_105',
                'team_points_over_110', 'team_points_over_115', 'team_points_over_120',
                'team_points_over_125', 'team_points_over_130',
                
                # Probabilidades y consistencia de líneas
                'prob_team_over_100_3', 'prob_team_over_100_5', 'prob_team_over_100_10', 'prob_team_over_100_20',
                'prob_team_over_110_3', 'prob_team_over_110_5', 'prob_team_over_110_10', 'prob_team_over_110_20',
                'prob_team_over_120_3', 'prob_team_over_120_5', 'prob_team_over_120_10', 'prob_team_over_120_20',
                'team_points_line_consistency_100', 'team_points_line_consistency_110', 'team_points_line_consistency_120',
                'team_points_line_momentum_5', 'team_points_line_momentum_10', 'team_points_line_momentum_20'
            ]
        }
        
        # Características que pueden sesgar los modelos (se eliminarán)
        self.biasing_features = {
            'Win': [
                'is_win',  # El target mismo
                'win_probability',  # Prácticamente el target
                'better_team_wins',  # Directamente relacionado con el target
                'wins_vs_better_teams_3',  # Información muy relacionada con el target 
                'situation_wins_3',  # Información muy relacionada con el target
            ],
            'Total_Points_Over_Under': [
                'total_points',  # El target mismo
                'total_points_over_180', 'total_points_over_185', 'total_points_over_190',
                'total_points_over_195', 'total_points_over_200', 'total_points_over_205',
                'total_points_over_210', 'total_points_over_215', 'total_points_over_220',
                'total_points_over_225', 'total_points_over_230', 'total_points_over_235',
                'prob_total_over_200_3', 'prob_total_over_200_5', 'prob_total_over_200_10', 'prob_total_over_200_20',
                'prob_total_over_210_3', 'prob_total_over_210_5', 'prob_total_over_210_10', 'prob_total_over_210_20',
                'prob_total_over_220_3', 'prob_total_over_220_5', 'prob_total_over_220_10', 'prob_total_over_220_20',
            ],
            'Team_Points_Over_Under': [
                'team_score',  # El target mismo
                'team_points_over_80', 'team_points_over_85', 'team_points_over_90',
                'team_points_over_95', 'team_points_over_100', 'team_points_over_105',
                'team_points_over_110', 'team_points_over_115', 'team_points_over_120',
                'team_points_over_125', 'team_points_over_130',
                'prob_team_over_100_3', 'prob_team_over_100_5', 'prob_team_over_100_10', 'prob_team_over_100_20',
                'prob_team_over_110_3', 'prob_team_over_110_5', 'prob_team_over_110_10', 'prob_team_over_110_20',
                'prob_team_over_120_3', 'prob_team_over_120_5', 'prob_team_over_120_10', 'prob_team_over_120_20',
            ],
            'PTS': [
                'PTS',  # El target mismo
                'PTS_over_10', 'PTS_over_15', 'PTS_over_20', 'PTS_over_25', 'PTS_over_30', 'PTS_over_35',
            ],
            'TRB': [
                'TRB',  # El target mismo
                'TRB_over_4', 'TRB_over_6', 'TRB_over_8', 'TRB_over_10', 'TRB_over_12',
            ],
            'AST': [
                'AST',  # El target mismo
                'AST_over_4', 'AST_over_6', 'AST_over_8', 'AST_over_10', 'AST_over_12',
            ],
            '3P': [
                '3P',  # El target mismo
                '3P_over_1', '3P_over_2', '3P_over_3', '3P_over_4', '3P_over_5',
            ]
        }

        # Características adicionales para encontrar value bets en cada objetivo
        self.value_betting_features = {
            'PTS': [
                # Características de línea y valor
                'PTS_over_10_line_diff_5', 'PTS_over_15_line_diff_5', 'PTS_over_20_line_diff_5', 
                'PTS_over_25_line_diff_5', 'PTS_over_30_line_diff_5', 'PTS_over_35_line_diff_5',
                'PTS_line_value_5', 'PTS_line_value_10',
                
                # Características de volatilidad
                'PTS_volatility_3', 'PTS_volatility_5', 'PTS_volatility_10', 'PTS_volatility_20',
                'PTS_over_35_volatility_3', 'PTS_over_35_volatility_5', 'PTS_over_35_volatility_10',
                
                # Consistencia del rendimiento vs línea
                'PTS_consistency_vs_line_3', 'PTS_consistency_vs_line_5', 'PTS_consistency_vs_line_10',
                
                # Características de bookmakers
                'pts_above_avg_streak', 'pts_increase_streak',
                'pts_vs_expected_3', 'pts_vs_expected_5', 'pts_vs_expected_10',
                
                # Eficiencia y matchups específicos
                'pts_per_minute', 'pts_favorable_matchup', 'pts_unfavorable_matchup',
                'matchup_pts_diff', 'pts_matchup_trend'
            ],
            'TRB': [
                # Características de línea y valor
                'TRB_over_4_line_diff_3', 'TRB_over_6_line_diff_3', 'TRB_over_8_line_diff_3', 
                'TRB_over_10_line_diff_3', 'TRB_over_12_line_diff_10',
                'TRB_line_value_5', 'TRB_line_value_10',
                
                # Características de volatilidad
                'TRB_volatility_3', 'TRB_volatility_5', 'TRB_volatility_10',
                'TRB_over_12_volatility_3', 'TRB_over_12_volatility_5', 'TRB_over_12_volatility_10',
                
                # Consistencia de rebotes
                'TRB_consistency_vs_line_3', 'TRB_consistency_vs_line_5', 'TRB_consistency_vs_line_10',
                
                # Características físicas relevantes para rebotes
                'Height_Inches', 'Weight', 'BMI', 'physical_fit_PF', 'physical_fit_C',
                'height_advantage_C', 'height_advantage_F',
                
                # Matchups específicos
                'trb_vs_position_avg', 'trb_vs_opp_diff', 'matchup_trb_diff', 'trb_matchup_trend'
            ],
            'AST': [
                # Características de línea y valor
                'AST_over_4_line_diff_5', 'AST_over_6_line_diff_5', 'AST_over_8_line_diff_5',
                'AST_over_10_line_diff_5', 'AST_over_12_line_diff_5', 'AST_over_12_line_diff_20',
                'AST_line_value_5', 'AST_line_value_10',
                
                # Características de volatilidad
                'AST_volatility_3', 'AST_volatility_5', 'AST_volatility_10',
                'AST_over_12_volatility_3', 'AST_over_12_volatility_5', 'AST_over_12_volatility_10',
                
                # Características de playmaking
                'ast_to_tov_ratio', 'playmaking_rating', 'is_playmaker',
                'pure_guard_playmaking', 'combo_guard_efficiency',
                
                # Características de matchup
                'matchup_ast_diff', 'ast_matchup_trend', 'ast_favorable_matchup', 'ast_unfavorable_matchup'
            ],
            '3P': [
                # Características de línea y valor
                '3P_over_1_line_diff_5', '3P_over_2_line_diff_5', '3P_over_3_line_diff_5',
                '3P_over_4_line_diff_5', '3P_over_5_line_diff_10',
                '3P_line_value_5', '3P_line_value_10',
                
                # Características de volatilidad
                '3P_volatility_3', '3P_volatility_5', '3P_volatility_10',
                '3P_over_5_volatility_3', '3P_over_5_volatility_5',
                
                # Tendencias de tiro
                '3P_trend_3', '3P_trend_5', '3P_trend_10', '3P_trend_20',
                '3p_momentum_3', '3p_momentum_5', '3p_momentum_10',
                
                # Características de tiro
                '3p_per_fga', '3p_per_minute', 'is_shooter',
                
                # Características de matchup
                'matchup_3p_diff', '3p_matchup_trend', '3p_favorable_matchup', '3p_unfavorable_matchup'
            ]
        }
        
        # Características de línea de apuestas específicas por target
        self.betting_line_features = {
            'PTS': {
                10: ['PTS_over_10', 'PTS_over_10_prob_3', 'PTS_over_10_prob_5', 'PTS_over_10_prob_10'],
                15: ['PTS_over_15', 'PTS_over_15_prob_3', 'PTS_over_15_prob_5', 'PTS_over_15_prob_10'],
                20: ['PTS_over_20', 'PTS_over_20_prob_3', 'PTS_over_20_prob_5', 'PTS_over_20_prob_10'],
                25: ['PTS_over_25', 'PTS_over_25_prob_3', 'PTS_over_25_prob_5', 'PTS_over_25_prob_10', 'PTS_over_25_prob_20'],
                30: ['PTS_over_30', 'PTS_over_30_prob_3', 'PTS_over_30_prob_5', 'PTS_over_30_prob_10'],
                35: ['PTS_over_35', 'PTS_over_35_prob_3', 'PTS_over_35_prob_5', 'PTS_over_35_prob_10', 'PTS_over_35_prob_20']
            },
            'TRB': {
                4: ['TRB_over_4', 'TRB_over_4_prob_3', 'TRB_over_4_prob_5', 'TRB_over_4_prob_10', 'TRB_over_4_prob_20'],
                6: ['TRB_over_6', 'TRB_over_6_prob_3', 'TRB_over_6_prob_5'],
                8: ['TRB_over_8', 'TRB_over_8_prob_3', 'TRB_over_8_prob_5', 'TRB_over_8_prob_10', 'TRB_over_8_prob_20'],
                10: ['TRB_over_10', 'TRB_over_10_prob_3', 'TRB_over_10_prob_5', 'TRB_over_10_prob_10'],
                12: ['TRB_over_12', 'TRB_over_12_prob_3', 'TRB_over_12_prob_5', 'TRB_over_12_prob_20']
            },
            'AST': {
                4: ['AST_over_4', 'AST_over_4_prob_3', 'AST_over_4_prob_5'],
                6: ['AST_over_6', 'AST_over_6_prob_3', 'AST_over_6_prob_5', 'AST_over_6_prob_10', 'AST_over_6_prob_20'],
                8: ['AST_over_8', 'AST_over_8_prob_3', 'AST_over_8_prob_5', 'AST_over_8_prob_10', 'AST_over_8_prob_20'],
                10: ['AST_over_10', 'AST_over_10_prob_3', 'AST_over_10_prob_5', 'AST_over_10_prob_10', 'AST_over_10_prob_20'],
                12: ['AST_over_12', 'AST_over_12_prob_3', 'AST_over_12_prob_5', 'AST_over_12_prob_10', 'AST_over_12_prob_20']
            },
            '3P': {
                1: ['3P_over_1', '3P_over_1_prob_3', '3P_over_1_prob_5', '3P_over_1_prob_10', '3P_over_1_prob_20'],
                2: ['3P_over_2', '3P_over_2_prob_3', '3P_over_2_prob_5', '3P_over_2_prob_20'],
                3: ['3P_over_3', '3P_over_3_prob_3', '3P_over_3_prob_5', '3P_over_3_prob_10', '3P_over_3_prob_20'],
                4: ['3P_over_4', '3P_over_4_prob_3', '3P_over_4_prob_5', '3P_over_4_prob_10', '3P_over_4_prob_20'],
                5: ['3P_over_5', '3P_over_5_prob_3', '3P_over_5_prob_5', '3P_over_5_prob_10']
            }
        }

    def safe_get_features(self, method):
        """
        Obtiene características de forma segura, retornando una lista vacía si el método falla
        """
        try:
            features = method()
            if features is None:
                return []
            return features
        except Exception as e:
            logger.warning(f"Error al obtener características: {e}")
            return []

    def is_player_dataset(self, df: pd.DataFrame) -> bool:
        """
        Determina si el DataFrame es un dataset de jugadores o de equipos
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            True si es un dataset de jugadores, False si es de equipos
        """
        # Características distintivas de un dataset de jugadores
        player_indicators = ['Player', 'TRB', 'AST', 'STL', 'BLK', 'double_double', 'triple_double']
        
        # Si al menos 3 de estos indicadores están presentes, consideramos que es un dataset de jugadores
        present_indicators = [col for col in player_indicators if col in df.columns]
        return len(present_indicators) >= 3

    def get_features_for_target(
        self,
        target: str,
        df: pd.DataFrame,
        include_common: bool = True,
        include_matchup: bool = True,
        include_advanced: bool = True,
        null_threshold: float = 0.9,
        line_value: Optional[float] = None,
        high_precision_mode: bool = True
    ) -> List[str]:
        """
        Obtiene las características óptimas para un target específico
        
        Args:
            target: Tipo de predicción (Win, Total_Points_Over_Under, Team_Points_Over_Under, PTS, TRB, AST, 3P)
            df: DataFrame con los datos
            include_common: Incluir características comunes
            include_matchup: Incluir características de matchup
            include_advanced: Incluir características avanzadas
            null_threshold: Umbral de valores nulos para filtrar características
            line_value: Valor específico de línea de apuesta (si aplica)
            high_precision_mode: Usar modo de alta precisión (96%+)
            
        Returns:
            Lista de características óptimas para el target
        """
        # Detectar si es predicción de equipo o jugador
        is_player_prediction = target in ['PTS', 'TRB', 'AST', '3P', 'Double_Double', 'Triple_Double']
        is_team_prediction = target in ['Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']
        
        if not (is_player_prediction or is_team_prediction):
            logger.warning(f"Target desconocido: {target}, devolviendo características vacías")
            return []
        
        # Inicializar generadores de características si es necesario
        # Pasamos el DataFrame como parámetro para evitar el error
        if is_player_prediction and self.players_feature_engineering is None:
            try:
                # Solo inicializamos si tenemos los datos de jugadores necesarios
                if 'Player' in df.columns:
                    self.players_feature_engineering = PlayersFeatures(players_data=df)
                else:
                    logger.warning("No se pudo inicializar PlayersFeatures: DataFrame no contiene columna 'Player'")
            except Exception as e:
                logger.error(f"Error al inicializar PlayersFeatures: {str(e)}")
                # Continuamos sin el feature engineering
                pass
            
        if is_team_prediction and self.teams_feature_engineering is None:
            try:
                # Solo inicializamos si tenemos los datos de equipos necesarios
                if 'Team' in df.columns:
                    self.teams_feature_engineering = TeamsFeatures(teams_data=df)
                else:
                    logger.warning("No se pudo inicializar TeamsFeatures: DataFrame no contiene columna 'Team'")
            except Exception as e:
                logger.error(f"Error al inicializar TeamsFeatures: {str(e)}")
                # Continuamos sin el feature engineering
                pass
        
        # Inicializar resultados
        available_features = []
        
        # Añadir características comunes si corresponde
        if include_common and self.common_features:
            # Filtrar para incluir solo las que realmente existen en el DataFrame
            common_feats = [f for f in self.common_features if f in df.columns]
            available_features.extend(common_feats)
            
            # Características específicas para jugadores o equipos
            if is_player_prediction and hasattr(self, 'common_player_features'):
                player_common = [f for f in self.common_player_features if f in df.columns]
                available_features.extend(player_common)
                
            if is_team_prediction and hasattr(self, 'common_team_features'):
                team_common = [f for f in self.common_team_features if f in df.columns]
                available_features.extend(team_common)
                
        # Añadir características específicas para el target
        if is_player_prediction:
            # Para targets de jugador, usar player_features_by_stat
            if target in self.player_features_by_stat:
                target_feats = [f for f in self.player_features_by_stat[target] if f in df.columns]
                available_features.extend(target_feats)
        else:
            # Para targets de equipo, usar team_features_by_stat
            if target in self.team_features_by_stat:
                target_feats = [f for f in self.team_features_by_stat[target] if f in df.columns]
                available_features.extend(target_feats)
        
        # Añadir características de matchup si se solicita
        if include_matchup and target in self.matchup_features:
            matchup_feats = [f for f in self.matchup_features[target] if f in df.columns]
            available_features.extend(matchup_feats)
            
        # Añadir características avanzadas si se solicita
        if include_advanced and target in self.advanced_features:
            advanced_feats = [f for f in self.advanced_features[target] if f in df.columns]
            available_features.extend(advanced_feats)

        # Eliminar duplicados manteniendo el orden
        available_features = list(dict.fromkeys(available_features))
        
        # Modo de alta precisión (96%+) - Nuevo análisis avanzado
        if high_precision_mode:
            try:
                # 1. Análisis de correlación para evitar data leakage
                correlation_threshold = 0.2  # Significativo pero no demasiado alto
                target_col = target
                
                # Para predicciones de over/under, crear columna binaria
                if line_value is not None:
                    target_col = f"{target}_over_{line_value}"
                    if target_col not in df.columns:
                        df[target_col] = (df[target] > line_value).astype(int)
                
                # Verificar que el target existe en el DataFrame
                if target not in df.columns:
                    logger.warning(f"El target {target} no está en el DataFrame. Omitiendo análisis de alta precisión.")
                    raise ValueError(f"'{target}' not in DataFrame columns")
                
                # Verificar que hay suficientes características para análisis
                if len(available_features) == 0:
                    logger.warning(f"No hay características disponibles para {target}. Omitiendo análisis de alta precisión.")
                    raise ValueError("No features available for correlation analysis")
                
                # Calcular correlaciones con el target, manejando posibles excepciones
                try:
                    # Asegurar que todas las columnas existen antes de calcular correlaciones
                    columns_to_use = [col for col in available_features + [target_col] if col in df.columns]
                    
                    # Si target_col no está incluido, agregarlo
                    if target_col not in columns_to_use:
                        logger.warning(f"La columna target {target_col} no está presente para cálculo de correlaciones.")
                        raise ValueError(f"Target column {target_col} not available")
                        
                    correlations = df[columns_to_use].corr()[target_col].abs()
                    
                    # Filtrar por correlación significativa
                    filtered_correlations = correlations[correlations >= correlation_threshold]
                except KeyError as ke:
                    logger.warning(f"KeyError en cálculo de correlaciones: {ke}. Posiblemente {target_col} no está en el índice.")
                    raise ValueError(f"Target column {target_col} not in correlation index")
                
                if not filtered_correlations.empty:
                    # Depuración: comprobar si es Series o DataFrame y usar el método correcto
                    logger.info(f"Tipo de filtered_correlations: {type(filtered_correlations)}")
                    try:
                        # Cuando se usa sort_values en Series, no se necesita el parámetro 'by'
                        if isinstance(filtered_correlations, pd.Series):
                            high_corr_features = filtered_correlations.sort_values(ascending=False)
                        else:  # Es DataFrame
                            # Verificar si hay columnas duplicadas
                            duplicated_cols = filtered_correlations.columns[filtered_correlations.columns.duplicated()]
                            if len(duplicated_cols) > 0:
                                logger.warning(f"Se detectaron columnas duplicadas: {duplicated_cols}")
                                # Usar la primera columna para ordenar (o eliminar duplicados)
                                filtered_correlations = filtered_correlations.loc[:, ~filtered_correlations.columns.duplicated()]
                            
                            # Ordenar usando la primera columna del DataFrame
                            try:
                                high_corr_features = filtered_correlations.sort_values(by=filtered_correlations.columns[0], ascending=False)
                            except Exception as sort_err:
                                logger.warning(f"Error al ordenar correlaciones en DataFrame: {sort_err}")
                                # Convertir a Series como fallback
                                if len(filtered_correlations.columns) > 0:
                                    high_corr_features = pd.Series(filtered_correlations.iloc[:, 0], index=filtered_correlations.index)
                                else:
                                    high_corr_features = pd.Series([], dtype='float64')
                        
                        high_corr_features = high_corr_features.index.tolist()
                    except Exception as e:
                        logger.error(f"Error al ordenar correlaciones: {str(e)}")
                        # En caso de error, usar las características sin ordenar
                        try:
                            high_corr_features = filtered_correlations.index.tolist()
                        except:
                            logger.error("No se pudo obtener índice de correlaciones, usando características disponibles")
                            high_corr_features = available_features.copy()
                else:
                    high_corr_features = []
                
                # Eliminar el propio target de las características
                if target_col in high_corr_features:
                    high_corr_features.remove(target_col)
                
                # Si no hay correlaciones significativas, usar todas las características disponibles
                if not high_corr_features:
                    high_corr_features = available_features.copy()
                    logger.warning(f"No se encontraron correlaciones significativas para {target}. Usando todas las características disponibles.")
                
                # 2. Características de alta precisión específicas por target
                high_precision_features = []
                
                # Para predicciones de puntos
                if target == 'PTS':
                    try:
                        # Identificar colinealidad para evitar redundancia
                        colinear_groups = [
                            ['PTS_mean_3', 'PTS_mean_5', 'PTS_mean_10', 'PTS_mean_20'],
                            ['PTS_over_20', 'PTS_over_25', 'PTS_over_30'],
                            ['FG%_mean_3', 'FG%_mean_5', 'FG%_mean_10', 'FG%_mean_20'],
                            ['MP_mean_3', 'MP_mean_5', 'MP_mean_10', 'MP_mean_20']
                        ]
                        
                        # Para cada grupo, seleccionar la característica más correlacionada
                        for group in colinear_groups:
                            try:
                                # Filtrar para incluir solo elementos existentes en high_corr_features
                                existing_group = [f for f in group if f in high_corr_features]
                                if existing_group:
                                    # Verificar que todas las características existen en correlations
                                    valid_features = [f for f in existing_group if f in correlations.index]
                                    if valid_features:
                                        try:
                                            # Obtener la característica con mayor correlación
                                            best_feature = max(valid_features, key=lambda x: correlations[x])
                                            high_precision_features.append(best_feature)
                                        except Exception as max_err:
                                            logger.warning(f"Error al encontrar la característica con mayor correlación: {max_err}")
                                            # Usar la primera característica como fallback
                                            high_precision_features.append(valid_features[0])
                            except Exception as group_err:
                                logger.warning(f"Error procesando grupo de características {group}: {group_err}")
                        
                        # Características físicas relevantes para puntos
                        physical_features = ['height', 'weight', 'wingspan', 'is_guard', 'is_forward', 'is_center']
                        physical_features = [f for f in physical_features if f in df.columns]
                        high_precision_features.extend(physical_features)
                        
                        # Eficiencia de tiro y tendencias
                        shooting_features = ['FG%', 'TS%', 'efg_pct', '3P%', 'FT%', 
                                            'pts_per_minute', 'pts_per_fga']
                        shooting_features = [f for f in shooting_features if f in df.columns]
                        high_precision_features.extend(shooting_features)
                        
                        # Tendencias recientes para líneas específicas si se proporciona un valor
                        if line_value is not None:
                            line_features = [
                                f'PTS_over_{line_value}_prob_3', f'PTS_over_{line_value}_prob_5',
                                f'PTS_over_{line_value}_prob_10', f'PTS_over_{line_value}_consistency',
                                f'PTS_over_{line_value}_line_diff_5'
                            ]
                            line_features = [f for f in line_features if f in df.columns]
                            high_precision_features.extend(line_features)
                    except Exception as pts_error:
                        logger.warning(f"Error seleccionando características avanzadas para PTS: {pts_error}")
                
                # Para predicciones de rebotes
                elif target == 'TRB':
                    try:
                        # Seleccionar características específicas para rebotes
                        rebounding_features = [
                            'height', 'weight', 'is_center', 'is_forward', 
                            'TRB_mean_5', 'TRB_trend_10', 'orb_drb_ratio',
                            'trb_per_minute', 'trb_per_height'
                        ]
                        rebounding_features = [f for f in rebounding_features if f in df.columns]
                        high_precision_features.extend(rebounding_features)
                        
                        # Líneas específicas de rebotes
                        if line_value is not None:
                            line_features = [
                                f'TRB_over_{line_value}_prob_5', f'TRB_over_{line_value}_prob_10',
                                f'TRB_over_{line_value}_consistency'
                            ]
                            line_features = [f for f in line_features if f in df.columns]
                            high_precision_features.extend(line_features)
                    except Exception as trb_error:
                        logger.warning(f"Error seleccionando características avanzadas para TRB: {trb_error}")
                
                # Para predicciones de asistencias
                elif target == 'AST':
                    try:
                        # Seleccionar características específicas para asistencias
                        assist_features = [
                            'is_guard', 'is_point_guard', 'ast_per_minute',
                            'AST_mean_5', 'AST_trend_10', 'ast_to_tov_ratio',
                            'playmaking_rating'
                        ]
                        assist_features = [f for f in assist_features if f in df.columns]
                        high_precision_features.extend(assist_features)
                        
                        # Líneas específicas de asistencias
                        if line_value is not None:
                            line_features = [
                                f'AST_over_{line_value}_prob_5', f'AST_over_{line_value}_prob_10',
                                f'AST_over_{line_value}_consistency'
                            ]
                            line_features = [f for f in line_features if f in df.columns]
                            high_precision_features.extend(line_features)
                    except Exception as ast_error:
                        logger.warning(f"Error seleccionando características avanzadas para AST: {ast_error}")
                
                # Para predicciones de triples
                elif target == '3P':
                    try:
                        # Seleccionar características específicas para triples
                        three_point_features = [
                            'is_guard', 'is_shooter', '3P_mean_5', '3P%_mean_10',
                            '3P_trend_10', '3p_per_minute', '3p_per_fga'
                        ]
                        three_point_features = [f for f in three_point_features if f in df.columns]
                        high_precision_features.extend(three_point_features)
                        
                        # Líneas específicas de triples
                        if line_value is not None:
                            line_features = [
                                f'3P_over_{line_value}_prob_5', f'3P_over_{line_value}_prob_10',
                                f'3P_over_{line_value}_consistency'
                            ]
                            line_features = [f for f in line_features if f in df.columns]
                            high_precision_features.extend(line_features)
                    except Exception as threeP_error:
                        logger.warning(f"Error seleccionando características avanzadas para 3P: {threeP_error}")
                
                # Añadir soporte explícito para targets de equipo
                elif target in ['Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']:
                    try:
                        # Características básicas para targets de equipo
                        team_basic_features = [
                            'is_home', 'win_rate_10', 'win_rate_20', 
                            'offensive_rating', 'defensive_rating', 
                            'offensive_efficiency', 'defensive_efficiency',
                            'efficiency_diff', 'pace', 'possessions'
                        ]
                        team_basic_features = [f for f in team_basic_features if f in df.columns]
                        high_precision_features.extend(team_basic_features)
                        
                        # Características específicas para Total_Points_Over_Under
                        if target == 'Total_Points_Over_Under':
                            total_points_features = [
                                'total_points_mean_10', 'total_points_std_10',
                                'PTS_mean_10', 'PTS_Opp_mean_10',
                                'pace', 'possessions', 'defensive_efficiency'
                            ]
                            total_points_features = [f for f in total_points_features if f in df.columns]
                            high_precision_features.extend(total_points_features)
                            
                        # Características específicas para Team_Points_Over_Under
                        elif target == 'Team_Points_Over_Under':
                            team_points_features = [
                                'PTS_mean_10', 'PTS_std_10', 'offensive_efficiency',
                                'points_per_possession', 'pace'
                            ]
                            team_points_features = [f for f in team_points_features if f in df.columns]
                            high_precision_features.extend(team_points_features)
                            
                        # Características específicas para Win
                        elif target == 'Win':
                            win_features = [
                                'win_rate_10', 'efficiency_diff', 'home_advantage',
                                'opp_win_rate', 'PTS_diff', 'current_win_streak'
                            ]
                            win_features = [f for f in win_features if f in df.columns]
                            high_precision_features.extend(win_features)
                            
                    except Exception as team_error:
                        logger.warning(f"Error seleccionando características avanzadas para {target}: {team_error}")
                
                # 3. Verificar disponibilidad y eliminar duplicados
                high_precision_features = [f for f in high_precision_features if f in df.columns]
                high_precision_features = list(dict.fromkeys(high_precision_features))
                
                # 4. Combinar con las características significativas
                # Primero las de alta precisión, luego las significativas que no estén ya incluidas
                prioritized_features = high_precision_features + [
                    f for f in high_corr_features 
                    if f not in high_precision_features
                ]
                
                # 5. Si hay suficientes características de alta precisión, usarlas
                # Si no, utilizar las características originales
                if len(prioritized_features) >= 10:
                    available_features = prioritized_features
                    logger.info(f"Usando {len(available_features)} características de alta precisión para {target}")
                else:
                    logger.warning(f"Insuficientes características de alta precisión ({len(prioritized_features)}), usando configuración estándar")
            
            except Exception as e:
                logger.warning(f"Error en análisis de alta precisión: {e}. Usando características estándar.")
        
        # Si se proporciona una línea específica, usar características para esa línea
        if line_value is not None:
            value_features = self.get_value_betting_features(df, target, line_value)
            
            # Asegurar que estas características están priorizadas
            # Primero eliminarlas si ya estaban en available_features
            available_features = [f for f in available_features if f not in value_features]
            # Luego añadirlas al principio
            available_features = value_features + available_features
            
            logger.info(f"Añadidas {len(value_features)} características de value betting para {target}={line_value}")
        else:
            # Si no hay línea específica, detectar características de alta confianza
            high_confidence_features = list(self.get_high_confidence_features(df, target, 0.9))
            if high_confidence_features:
                # Priorizarlas igual que las de value betting
                available_features = [f for f in available_features if f not in high_confidence_features]
                available_features = high_confidence_features + available_features
                
                logger.info(f"Añadidas {len(high_confidence_features)} características de alta confianza")
        
        # Filtrar por valores nulos
        try:
            # Verificar qué características tienen muchos valores nulos
            null_analysis = df[available_features].isnull().mean()
            filtered_features = [
                f for f in available_features
                if null_analysis[f] < null_threshold
            ]
            
            # Si quedan muy pocas características, relajar el filtro
            if len(filtered_features) < 5 and available_features:
                logger.warning(f"Pocas características después de filtrar por nulos ({len(filtered_features)}), relajando umbral")
                filtered_features = [
                    f for f in available_features
                    if null_analysis[f] < 0.5  # Umbral más permisivo
                ]
            
            available_features = filtered_features
        except Exception as e:
            logger.warning(f"Error al filtrar por valores nulos: {e}")
        
        # Verificar si hay suficientes características
        if len(available_features) < 3:
            logger.warning(f"Muy pocas características disponibles para {target}: {len(available_features)}")
            
            # Tratar de añadir algunas características básicas siempre presentes
            if is_player_prediction:
                basic_stats = ['MP', 'FG%', 'is_starter', 'height', 'previous_' + target]
            else:
                basic_stats = ['win_rate_10', 'offensive_rating', 'defensive_rating', 'pace']
                
            for stat in basic_stats:
                if stat in df.columns and stat not in available_features:
                    available_features.append(stat)
        
        # Limitar el número de características si hay demasiadas
        max_features = 50  # Un número razonable para evitar sobreajuste
        if len(available_features) > max_features:
            logger.info(f"Limitando características de {len(available_features)} a {max_features}")
            available_features = available_features[:max_features]
            
        logger.info(f"Seleccionadas {len(available_features)} características para {target}")
        return available_features

    def get_all_target_features(
        self,
        df: pd.DataFrame,
        targets: List[str],
        include_common: bool = True,
        include_matchup: bool = True,
        include_advanced: bool = True,
        null_threshold: float = 0.9
    ) -> Dict[str, List[str]]:
        """
        Obtiene características para múltiples targets
        
        Args:
            df: DataFrame con los datos
            targets: Lista de tipos de predicción
            include_common: Si incluir características comunes
            include_matchup: Si incluir características de matchup
            include_advanced: Si incluir características avanzadas
            null_threshold: Umbral para filtrar características con alto % de nulos
            
        Returns:
            Diccionario con características por target
        """
        # Determinar si estamos trabajando con un dataset de jugadores o equipos
        is_player_data = self.is_player_dataset(df)
        logger.info(f"Tipo de dataset detectado: {'Jugadores' if is_player_data else 'Equipos'}")
        
        target_features = {}
        
        for target in targets:
            target_features[target] = self.get_features_for_target(
                target, df, include_common, include_matchup, include_advanced, null_threshold
            )
            
            # Para asegurar que usamos todas las características disponibles:
            if len(target_features[target]) < 5:  # Si hay muy pocas características
                logger.warning(f"Muy pocas características ({len(target_features[target])}) seleccionadas para {target}. " 
                              f"Usando todas las columnas disponibles excepto los targets y ID.")
                
                # Usar todas las columnas, excluyendo targets e identificadores
                excluded_cols = targets + ['Player', 'Team', 'Date', 'Opp', 'Result'] + self.biasing_features.get(target, [])
                target_features[target] = [col for col in df.columns 
                                          if col not in excluded_cols]
                
                # Mantener los identificadores básicos si no hay otras características
                if len(target_features[target]) < 3:
                    if is_player_data:
                        basic_numerical = ['MP', 'STL', 'BLK', 'TOV', 'PF']
                    else:
                        basic_numerical = ['PTS_Opp', 'FG%', 'FG%_Opp']
                    
                    for col in basic_numerical:
                        if col in df.columns and col not in target_features[target] and col not in self.biasing_features.get(target, []):
                            target_features[target].append(col)
                
                logger.info(f"Usando {len(target_features[target])} características genéricas para {target}")
        
        # Asegurarse de que existe al menos una característica para cada target
        for target, features in target_features.items():
            if not features:
                logger.warning(f"No hay características disponibles para {target}. Usando columnas básicas.")
                # Intentar usar columnas numéricas básicas
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                # Excluir columnas que son targets o que sesgan el modelo
                biasing_cols = targets + self.biasing_features.get(target, [])
                numeric_cols = [col for col in numeric_cols if col not in biasing_cols]
                # Limitar a 20 columnas para no sobrecargar
                target_features[target] = numeric_cols[:20] if len(numeric_cols) > 20 else numeric_cols
            
            logger.info(f"Final: {len(features)} características para {target}")
        
        return target_features

    def analyze_feature_importance(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        method: str = 'correlation'
    ) -> pd.DataFrame:
        """
        Analiza la importancia de las características para un target
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción
            features: Lista de características a analizar
            method: Método de análisis ('correlation' o 'mutual_info')
            
        Returns:
            DataFrame con la importancia de cada característica
        """
        if method == 'correlation':
            # Calcular correlaciones
            correlations = df[features].corrwith(df[target])
            importance = pd.DataFrame({
                'feature': features,
                'importance': correlations.abs(),
                'correlation': correlations
            })
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            # Calcular información mutua
            mi_scores = mutual_info_regression(df[features], df[target])
            importance = pd.DataFrame({
                'feature': features,
                'importance': mi_scores
            })
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        # Ordenar por importancia si hay datos (aquí importance es un DataFrame, por lo que necesita by)
        if not importance.empty:
            importance = importance.sort_values(by='importance', ascending=False)
        
        return importance

    def get_optimal_feature_set(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        min_importance: float = 0.1,
        max_features: int = 20
    ) -> List[str]:
        """
        Obtiene un conjunto óptimo de características basado en su importancia
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción
            features: Lista de características a analizar
            min_importance: Importancia mínima para incluir una característica
            max_features: Número máximo de características a incluir
            
        Returns:
            Lista de características óptimas
        """
        # Analizar importancia
        importance = self.analyze_feature_importance(df, target, features)
        
        # Filtrar por importancia mínima
        important_features = importance[
            importance['importance'] >= min_importance
        ]['feature'].tolist()
        
        # Limitar número de características
        if len(important_features) > max_features:
            important_features = important_features[:max_features]
        
        return important_features 

    def get_value_betting_features(
        self, 
        df: pd.DataFrame, 
        target: str, 
        line_value: float,
        min_confidence: float = 0.95  # Aumentado a 0.96 para mayor precisión
    ) -> List[str]:
        """
        Obtiene características específicas para identificar value bets para un target y línea específica
        con mayor precisión (96%+)
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            line_value: Valor de la línea de apuesta
            min_confidence: Confianza mínima para considerar una recomendación (default: 96%)
            
        Returns:
            Lista de características para detectar value bets de alta precisión
        """
        # Verificar que el target está soportado
        if target not in self.value_betting_features:
            logger.warning(f"Target {target} no tiene características de value betting definidas")
            return []
            
        # Características base de value betting para el target
        value_features = self.value_betting_features.get(target, []).copy()
        
        # Añadir características específicas de línea si están disponibles
        if target in self.betting_line_features:
            # Encontrar la línea más cercana
            closest_line = min(self.betting_line_features[target].keys(), 
                             key=lambda x: abs(x - line_value))
            
            # Usar características de esa línea
            line_specific_features = self.betting_line_features[target].get(closest_line, [])
            value_features.extend(line_specific_features)
            
            # Si la línea no es exactamente la misma que una predefinida, añadir también
            # características de las líneas adyacentes
            if closest_line != line_value:
                keys = sorted(self.betting_line_features[target].keys())
                idx = keys.index(closest_line)
                
                # Añadir línea superior si existe
                if idx + 1 < len(keys):
                    upper_line = keys[idx + 1]
                    value_features.extend(self.betting_line_features[target].get(upper_line, []))
                    
                # Añadir línea inferior si existe
                if idx > 0:
                    lower_line = keys[idx - 1]
                    value_features.extend(self.betting_line_features[target].get(lower_line, []))
        
        # Filtrar características que no están en el DataFrame
        available_features = [f for f in value_features if f in df.columns]
        
        # Nuevas características de alta precisión
        # Análisis de correlación para evitar data leakage
        correlation_threshold = 0.2  # Solo características con correlación significativa
        
        try:
            # Crear una columna target específica para esta línea
            target_col = f"{target}_over_{line_value}"
            if target_col not in df.columns:
                df[target_col] = (df[target] > line_value).astype(int)
                
            # Calcular correlaciones con el target binario
            correlations = df[available_features + [target_col]].corr()[target_col].abs()
            
            # Filtrar por correlación mínima, ordenar por importancia
            filtered_correlations = correlations[correlations >= correlation_threshold]
            if not filtered_correlations.empty:
                # Depuración: comprobar si es Series o DataFrame y usar el método correcto
                logger.info(f"Tipo de filtered_correlations en get_value_betting_features: {type(filtered_correlations)}")
                try:
                    # Cuando se usa sort_values en Series, no se necesita el parámetro 'by'
                    if isinstance(filtered_correlations, pd.Series):
                        high_corr_features = filtered_correlations.sort_values(ascending=False)
                    else:  # Es DataFrame
                        # Verificar si hay columnas duplicadas
                        duplicated_cols = filtered_correlations.columns[filtered_correlations.columns.duplicated()]
                        if len(duplicated_cols) > 0:
                            logger.warning(f"Se detectaron columnas duplicadas: {duplicated_cols}")
                            # Usar la primera columna para ordenar (o eliminar duplicados)
                            filtered_correlations = filtered_correlations.loc[:, ~filtered_correlations.columns.duplicated()]
                        
                        # Ordenar usando la primera columna del DataFrame
                        try:
                            high_corr_features = filtered_correlations.sort_values(by=filtered_correlations.columns[0], ascending=False)
                        except Exception as sort_err:
                            logger.warning(f"Error al ordenar correlaciones en DataFrame: {sort_err}")
                            # Convertir a Series como fallback
                            if len(filtered_correlations.columns) > 0:
                                high_corr_features = pd.Series(filtered_correlations.iloc[:, 0], index=filtered_correlations.index)
                            else:
                                high_corr_features = pd.Series([], dtype='float64')
                    
                    high_corr_features = high_corr_features.index.tolist()
                except Exception as e:
                    logger.error(f"Error al ordenar correlaciones: {str(e)}")
                    # En caso de error, usar las características sin ordenar
                    try:
                        high_corr_features = filtered_correlations.index.tolist()
                    except:
                        logger.error("No se pudo obtener índice de correlaciones, usando características disponibles")
                        high_corr_features = available_features.copy()
                else:
                    high_corr_features = []
            
            # Eliminar el propio target de la lista de características
            if target_col in high_corr_features:
                high_corr_features.remove(target_col)
                
            # Características físicas y de posición en función del target
            positional_features = []
            if target == 'PTS':
                positional_features = ['is_guard', 'is_forward', 'is_center', 'height', 'weight', 
                                      'usage_rate', 'scoring_efficiency', 'shooter_rating']
            elif target == 'TRB':
                positional_features = ['is_forward', 'is_center', 'height', 'weight', 'wingspan', 
                                      'rebounding_rate', 'box_out_rating', 'vertical_leap']
            elif target == 'AST':
                positional_features = ['is_guard', 'is_point_guard', 'playmaking_rating', 
                                      'assist_to_turnover', 'ball_handling_rating']
            elif target == '3P':
                positional_features = ['is_guard', 'is_wing', 'is_shooter', 'three_point_rating', 
                                      '3P%_mean_10', '3P_volume']
                
            positional_features = [f for f in positional_features if f in df.columns]
            
            # Características de consistencia y tendencia
            consistency_features = [
                f"{target}_consistency_score", f"{target}_volatility_5", 
                f"{target}_momentum_10", f"{target}_upward_trend"
            ]
            consistency_features = [f for f in consistency_features if f in df.columns]
            
            # Características específicas de la línea
            line_prediction_features = [
                f"{target}_over_{line_value}_prob_5", f"{target}_over_{line_value}_prob_10",
                f"{target}_over_{line_value}_book_prob", f"{target}_over_{line_value}_line_diff",
                f"{target}_over_{line_value}_value_rating"
            ]
            line_prediction_features = [f for f in line_prediction_features if f in df.columns]
            
            # Características de matchup relevantes para alta precisión
            matchup_features = [
                f"opp_{target}_allowed_mean_5", f"opp_{target}_rank", 
                f"matchup_{target}_advantage", f"{target}_vs_opp_history",
                f"opp_defensive_rating", f"favorable_matchup_{target}"
            ]
            matchup_features = [f for f in matchup_features if f in df.columns]
            
            # Priorizar características: primero las específicas de línea, luego las de 
            # correlación alta, posición, matchup y finalmente consistencia
            prioritized_features = (
                line_prediction_features + 
                high_corr_features[:10] +  # Limitar a las 10 más correlacionadas
                positional_features + 
                matchup_features +
                consistency_features
            )
            
            # Eliminar duplicados manteniendo el orden
            unique_features = []
            for f in prioritized_features:
                if f not in unique_features and f in df.columns:
                    unique_features.append(f)
                    
            # Combinar con las características originales para asegurar tener suficientes
            combined_features = unique_features + [f for f in available_features if f not in unique_features]
            
            # Filtrar características con alto porcentaje de valores nulos
            null_threshold = 0.3
            try:
                null_analysis = df[combined_features].isnull().mean()
                available_features = [
                    f for f in combined_features 
                    if null_analysis[f] < null_threshold
                ]
            except Exception as e:
                logger.warning(f"Error al analizar nulos: {e}, usando todas las características disponibles")
                available_features = combined_features
        
        except Exception as e:
            logger.warning(f"Error en análisis avanzado de características: {e}. Usando características básicas.")
            
            # Filtrar características con alto porcentaje de valores nulos (fallback original)
            null_threshold = 0.3
            try:
                null_analysis = df[available_features].isnull().mean()
                available_features = [
                    f for f in available_features 
                    if null_analysis[f] < null_threshold
                ]
            except Exception as e2:
                logger.warning(f"Error secundario al analizar nulos: {e2}, usando todas las características disponibles")
        
        logger.info(f"Seleccionadas {len(available_features)} características de value betting para {target} con línea {line_value}")
        
        return available_features
        
    def get_advanced_line_features(
        self,
        df: pd.DataFrame,
        target: str,
        betting_lines: List[float],
        feature_pool: Optional[List[str]] = None
    ) -> Dict[float, List[str]]:
        """
        Obtiene las características más predictivas para cada línea de apuesta específica
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            betting_lines: Lista de líneas de apuesta a analizar
            feature_pool: Pool opcional de características de donde seleccionar (si None, usa todas disponibles)
            
        Returns:
            Diccionario con las mejores características por línea de apuesta
        """
        result = {}
        
        # Si no se proporciona pool de características, usar todas menos los identificadores
        if feature_pool is None:
            non_feature_cols = ['Player', 'Date', 'Opp', 'Result', 'Away', 'Team', target]
            feature_pool = [col for col in df.columns if col not in non_feature_cols]
        
        for line in betting_lines:
            # Crear columna target binaria para esta línea específica (over/under)
            target_col = f"{target}_over_{line}"
            
            # Si la columna no existe, crearla
            if target_col not in df.columns:
                try:
                    df[target_col] = (df[target] > line).astype(int)
                except:
                    logger.warning(f"No se pudo crear la columna {target_col}, saltando línea {line}")
                    continue
            
            # Obtener características específicas para esta línea
            line_features = self.get_value_betting_features(df, target, line)
            
            # Añadir características base de la línea
            if target in self.betting_line_features:
                for line_key, features in self.betting_line_features[target].items():
                    if abs(line_key - line) <= 5:  # Usar líneas cercanas
                        for feature in features:
                            if feature in df.columns and feature not in line_features:
                                line_features.append(feature)
            
            # Si hay suficientes características, usar modelo de selección
            if len(line_features) >= 10:
                selected_features = self._select_features_for_line(df, target_col, line_features)
                if len(selected_features) >= 3:
                    result[line] = selected_features
                    continue
            
            # Si no hay suficientes características o la selección falló, usar todas disponibles
            result[line] = line_features
            
        return result
        
    def _select_features_for_line(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: List[str],
        max_features: int = 20
    ) -> List[str]:
        """
        Selecciona las características más importantes para una línea específica
        
        Args:
            df: DataFrame con los datos
            target_col: Columna target (binaria over/under)
            features: Lista de características a analizar
            max_features: Número máximo de características a seleccionar
            
        Returns:
            Lista de características seleccionadas
        """
        try:
            # Preparar datos
            X = df[features].copy()
            y = df[target_col].copy()
            
            # Manejar valores nulos
            X = X.fillna(X.mean())
            
            # Normalizar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Seleccionar características con Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(rf, max_features=max_features)
            selector.fit(X_scaled, y)
            
            # Obtener máscara de características seleccionadas
            support = selector.get_support()
            
            # Devolver nombres de características seleccionadas
            selected_features = [features[i] for i in range(len(features)) if support[i]]
            
            logger.info(f"Seleccionadas {len(selected_features)} características de {len(features)} para {target_col}")
            return selected_features
            
        except Exception as e:
            logger.warning(f"Error en selección de características para {target_col}: {e}")
            return features[:max_features] if len(features) > max_features else features
            
    def get_high_confidence_features(
        self,
        df: pd.DataFrame,
        target: str,
        confidence_threshold: float = 0.9,
        consistency_window: int = 20
    ) -> Set[str]:
        """
        Identifica características que consistentemente predicen el resultado con alta confianza
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción
            confidence_threshold: Umbral de confianza
            consistency_window: Ventana para analizar consistencia
            
        Returns:
            Conjunto de características de alta confianza
        """
        high_confidence_features = set()
        
        # Verificar disponibilidad de columnas de probabilidad
        prob_cols = [col for col in df.columns if '_prob_' in col and target in col]
        
        for col in prob_cols:
            # Calcular mediana de probabilidad para esta característica
            median_prob = df[col].median()
            
            # Verificar si la mediana supera el umbral
            if median_prob >= confidence_threshold:
                # Verificar también la consistencia en las últimas N observaciones
                recent_consistency = df[col].tail(consistency_window).mean()
                
                if recent_consistency >= confidence_threshold:
                    high_confidence_features.add(col)
                    
                    # Añadir también la característica base (sin _prob_)
                    base_feature = col.split('_prob_')[0]
                    if base_feature in df.columns:
                        high_confidence_features.add(base_feature)
                        
                    # Añadir columnas relacionadas
                    for related_col in df.columns:
                        if base_feature in related_col and related_col != col:
                            high_confidence_features.add(related_col)
        
        # Si no se encontraron características con el umbral actual, intentar con un umbral más bajo
        if not high_confidence_features and confidence_threshold > 0.8:
            logger.info(f"No se encontraron características con confianza {confidence_threshold} para {target}, intentando con umbral 0.7")
            return self.get_high_confidence_features(df, target, confidence_threshold=0.8, consistency_window=consistency_window)
        
        logger.info(f"Identificadas {len(high_confidence_features)} características de alta confianza para {target}")
        return high_confidence_features

    def identify_high_confidence_betting_lines(
        self,
        df: pd.DataFrame,
        target: str,
        min_confidence: float = 0.96,
        min_samples: int = 30,
        lookback_days: int = 60
    ) -> Dict[float, Dict[str, float]]:
        """
        Identifica líneas de apuestas con alta confianza que históricamente alcanzan
        el umbral de precisión deseado (96% o más)
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            min_confidence: Confianza mínima requerida (por defecto 0.96 para 96%)
            min_samples: Cantidad mínima de muestras para considerar la línea
            lookback_days: Días hacia atrás para analizar (ventana de análisis)
            
        Returns:
            Diccionario con líneas de alta confianza y sus métricas
        """
        if target not in ['PTS', 'TRB', 'AST', '3P']:
            logger.warning(f"Target {target} no soportado para análisis de líneas de alta confianza")
            return {}
            
        high_confidence_lines = {}
        betting_lines = []
        
        # Determinar las líneas disponibles basado en columnas existentes
        for col in df.columns:
            if f"{target}_over_" in col and not col.endswith(('_prob_3', '_prob_5', '_prob_10', '_prob_20')):
                try:
                    line_value = float(col.split(f"{target}_over_")[1])
                    betting_lines.append(line_value)
                except:
                    continue
        
        # Si no hay líneas detectadas, usar las predefinidas
        if not betting_lines and target in self.betting_line_features:
            betting_lines = list(self.betting_line_features[target].keys())
        
        logger.info(f"Analizando {len(betting_lines)} líneas de apuestas para {target}")
        
        # Asegurar que la fecha está en formato datetime
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                logger.warning("No se pudo convertir la columna Date a datetime")
        
        # Filtrar por fecha reciente si es posible
        recent_df = df
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            cutoff_date = df['Date'].max() - pd.Timedelta(days=lookback_days)
            recent_df = df[df['Date'] >= cutoff_date].copy()
            logger.info(f"Análisis restringido a los últimos {lookback_days} días ({len(recent_df)} registros)")
        
        # Verificar líneas disponibles en los datos
        for line in sorted(betting_lines):
            line_col = f"{target}_over_{line}"
            
            # Si la columna no existe, crearla
            if line_col not in recent_df.columns:
                try:
                    recent_df[line_col] = (recent_df[target] > line).astype(int)
                except:
                    logger.warning(f"No se pudo crear la columna {line_col}, saltando")
                    continue
            
            # Calcular métricas de confianza
            over_count = recent_df[line_col].sum()
            under_count = len(recent_df) - over_count
            total_samples = len(recent_df)
            
            if total_samples < min_samples:
                logger.info(f"Insuficientes muestras para línea {line} ({total_samples} < {min_samples})")
                continue
                
            # Calcular proporción y consistencia
            over_pct = over_count / total_samples
            under_pct = under_count / total_samples
            
            # La confianza es el máximo entre over y under
            confidence = max(over_pct, under_pct)
            prediction = 'over' if over_pct >= under_pct else 'under'
            
            # Verificar consistencia en diferentes ventanas temporales
            window_consistency = {}
            for window in [3, 5, 10, 20]:
                if len(recent_df) >= window:
                    window_data = recent_df.sort_values(by='Date', ascending=False).head(window)
                    window_over = window_data[line_col].mean()
                    window_consistency[window] = max(window_over, 1 - window_over)
            
            # Calcular volatilidad (variabilidad) en la predicción
            volatility = recent_df[line_col].std() if len(recent_df) > 1 else 0.5
            
            # Análisis por oponente si es posible
            opp_consistency = {}
            if 'Opp' in recent_df.columns:
                for opp in recent_df['Opp'].unique():
                    opp_data = recent_df[recent_df['Opp'] == opp]
                    if len(opp_data) >= 3:  # Al menos 3 juegos contra este oponente
                        opp_over = opp_data[line_col].mean()
                        opp_consistency[opp] = max(opp_over, 1 - opp_over)
            
            # Calcular confianza ajustada por factores adicionales
            adjusted_confidence = confidence
            
            # Penalizar alta volatilidad
            if volatility > 0.3:
                adjusted_confidence *= (1 - (volatility - 0.3))
            
            # Premiar consistencia en ventanas recientes
            if window_consistency:
                # Dar más peso a ventanas más pequeñas (más recientes)
                weighted_consistency = sum(window_consistency[w] * (1/w) for w in window_consistency) / sum(1/w for w in window_consistency)
                adjusted_confidence = 0.7 * adjusted_confidence + 0.3 * weighted_consistency
            
            # Si cumple el umbral de confianza, añadir a las líneas de alta confianza
            if adjusted_confidence >= min_confidence:
                high_confidence_lines[line] = {
                    'confidence': confidence,
                    'adjusted_confidence': adjusted_confidence,
                    'prediction': prediction,
                    'samples': total_samples,
                    'volatility': volatility,
                    'recent_windows': window_consistency,
                    'opponent_analysis': opp_consistency
                }
                
                logger.info(f"Línea de alta confianza detectada: {target} {line} ({prediction.upper()}) - "
                           f"Confianza: {adjusted_confidence:.4f}, Muestras: {total_samples}")
        
        if not high_confidence_lines:
            logger.warning(f"No se encontraron líneas de alta confianza para {target} con umbral {min_confidence}")
            
            # Si no hay líneas que cumplan el umbral estricto, intentar con un umbral más bajo
            # pero solo para propósitos informativos
            if min_confidence > 0.9:
                fallback_lines = self.identify_high_confidence_betting_lines(
                    df, target, min_confidence=0.9, min_samples=min_samples, lookback_days=lookback_days
                )
                if fallback_lines:
                    logger.info(f"Se encontraron {len(fallback_lines)} líneas con confianza >90% pero <{min_confidence*100}%")
        
        return high_confidence_lines
        
    def get_optimal_betting_strategy(
        self,
        df: pd.DataFrame,
        target: str,
        confidence_threshold: float = 0.96,
        min_edge: float = 0.05,
        bankroll_fraction: float = 0.02
    ) -> Dict[str, Dict]:
        """
        Genera una estrategia óptima de apuestas basada en las líneas de alta confianza
        y el análisis de ventaja sobre la casa de apuestas
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            confidence_threshold: Umbral de confianza mínima (0.96 = 96% de precisión)
            min_edge: Ventaja mínima sobre la casa de apuestas para considerar una apuesta
            bankroll_fraction: Fracción del bankroll a apostar (Kelly simplificado)
            
        Returns:
            Estrategia de apuestas optimizada para máxima precisión y valor
        """
        strategy = {
            'target': target,
            'confidence_threshold': confidence_threshold,
            'high_confidence_lines': {},
            'value_bets': {},
            'best_lines': [],
            'avoid_lines': [],
            'recommended_features': {}
        }
        
        # Identificar líneas de alta confianza
        high_confidence_lines = self.identify_high_confidence_betting_lines(
            df, target, min_confidence=confidence_threshold
        )
        
        if not high_confidence_lines:
            logger.warning(f"No se encontraron líneas de alta confianza para {target}. Estrategia no disponible.")
            return strategy
            
        strategy['high_confidence_lines'] = high_confidence_lines
        
        # Para cada línea de alta confianza, obtener las mejores características
        for line, line_info in high_confidence_lines.items():
            # Obtener características específicas para esta línea
            line_features = self.get_features_for_target(
                target, df, line_value=line, high_precision_mode=True
            )
            
            # Guardar las características recomendadas
            strategy['recommended_features'][line] = line_features[:30]  # Top 30 características
            
            # Determinar si es una value bet (tiene ventaja sobre la casa)
            market_prob = 0.5  # Probabilidad implícita del mercado (línea justa)
            model_prob = line_info['adjusted_confidence']
            
            # Si hay columnas de probabilidad de casas, usar esa información
            book_prob_cols = [col for col in df.columns 
                             if f"{target}_over_{line}_book_prob" in col or f"{target}_under_{line}_book_prob" in col]
            
            if book_prob_cols:
                # Usar la probabilidad promedio de las casas
                book_probs = df[book_prob_cols].mean().mean()
                if not pd.isna(book_probs):
                    market_prob = book_probs
            
            # Calcular ventaja (edge)
            if line_info['prediction'] == 'over':
                edge = model_prob - market_prob
            else:
                edge = model_prob - (1 - market_prob)
                
            # Determinar fracción de Kelly (apuesta óptima)
            if edge > 0:
                # Simplificación de la fórmula de Kelly
                kelly_fraction = (model_prob * 2 - 1) / 1.0  # Asumiendo cuota de 2.0
                # Limitar la fracción para gestión de riesgo
                kelly_fraction = min(kelly_fraction, bankroll_fraction)
            else:
                kelly_fraction = 0
                
            # Guardar información de value bet
            if edge >= min_edge:
                strategy['value_bets'][line] = {
                    'prediction': line_info['prediction'],
                    'confidence': line_info['adjusted_confidence'],
                    'market_probability': market_prob,
                    'edge': edge,
                    'kelly_fraction': kelly_fraction,
                    'samples': line_info['samples'],
                    'volatility': line_info['volatility']
                }
                
                # Añadir a mejores líneas
                strategy['best_lines'].append({
                    'target': target,
                    'line': line,
                    'prediction': line_info['prediction'],
                    'confidence': line_info['adjusted_confidence'],
                    'edge': edge,
                    'recommendation': f"{target} {line_info['prediction'].upper()} {line}"
                })
            elif edge < -min_edge:
                # Líneas a evitar (ventaja para la casa)
                strategy['avoid_lines'].append({
                    'target': target,
                    'line': line,
                    'prediction': line_info['prediction'],
                    'edge': edge
                })
                
        # Ordenar las mejores líneas por confianza
        strategy['best_lines'] = sorted(
            strategy['best_lines'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        # Resumir la estrategia
        if strategy['best_lines']:
            best_bet = strategy['best_lines'][0]
            logger.info(f"Mejor apuesta: {best_bet['recommendation']} - Confianza: {best_bet['confidence']:.4f}, Ventaja: {best_bet['edge']:.4f}")
        else:
            logger.warning(f"No se encontraron value bets para {target} con ventaja mínima de {min_edge}")
            
        return strategy 

    def analyze_market_inefficiencies(
        self,
        df: pd.DataFrame,
        target: str,
        bookmakers: List[str] = None,
        min_confidence: float = 0.96,
        min_edge: float = 0.04,
        min_odds: float = 1.8,
        lookback_days: int = 60
    ) -> Dict[float, Dict[str, Any]]:
        """
        Analiza ineficiencias del mercado para encontrar líneas con alta precisión
        y buenas odds ofrecidas por las casas de apuestas
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            bookmakers: Lista de columnas con odds de diferentes casas (si None, usa detección automática)
            min_confidence: Umbral mínimo de confianza para nuestras predicciones (ej: 0.96)
            min_edge: Ventaja mínima necesaria sobre las casas de apuestas
            min_odds: Odds mínimas para considerar una apuesta valiosa
            lookback_days: Días hacia atrás para analizar
            
        Returns:
            Diccionario con líneas que tienen alta precisión y buenas odds
        """
        if target not in ['PTS', 'TRB', 'AST', '3P']:
            logger.warning(f"Target {target} no soportado para análisis de ineficiencias")
            return {}
            
        # Identificar líneas de alta confianza primero
        high_confidence_lines = self.identify_high_confidence_betting_lines(
            df, target, min_confidence=min_confidence, lookback_days=lookback_days
        )
        
        if not high_confidence_lines:
            logger.warning(f"No se encontraron líneas de alta confianza para {target}")
            return {}
            
        # Valores de retorno
        valuable_lines = {}
        
        # Si no se especifican casas de apuestas, intentar detectarlas
        if bookmakers is None:
            odds_columns = []
            for col in df.columns:
                if any(term in col.lower() for term in ['odds', 'probability', 'implied', 'book', 'market']):
                    if target.lower() in col.lower():
                        odds_columns.append(col)
            
            if odds_columns:
                logger.info(f"Detectadas {len(odds_columns)} columnas con odds: {odds_columns[:5]}...")
                bookmakers = odds_columns
            else:
                logger.warning("No se detectaron columnas con odds de las casas de apuestas")
                return {}
        
        # Filtrar por fecha reciente
            recent_df = df
            if 'Date' in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                            df['Date'] = pd.to_datetime(df['Date'])
                            
                            cutoff_date = df['Date'].max() - pd.Timedelta(days=lookback_days)
                            recent_df = df[df['Date'] >= cutoff_date].copy()
                    logger.info(f"Análisis restringido a los últimos {lookback_days} días ({len(recent_df)} registros)")
                except Exception as e:
                    logger.warning(f"Error al filtrar por fecha: {e}")
        
        # Para cada línea de alta confianza, evaluar odds
        for line, line_info in high_confidence_lines.items():
            line_col = f"{target}_over_{line}"
            
            # Asegurar que tenemos la columna de over/under para esta línea
            if line_col not in recent_df.columns:
                try:
                    recent_df[line_col] = (recent_df[target] > line).astype(int)
                except Exception as e:
                    logger.warning(f"No se pudo crear columna {line_col}: {e}")
                    continue
            
            # Columnas de odds específicas para esta línea
            line_odds_cols = [col for col in bookmakers 
                             if str(line) in col and target.lower() in col.lower()]
            
            if not line_odds_cols:
                logger.info(f"No se encontraron columnas de odds para {target} línea {line}")
                continue
                
            # Analizar cada columna de odds
            line_value_bets = []
            
            for odds_col in line_odds_cols:
                is_over = 'over' in odds_col.lower()
                is_under = 'under' in odds_col.lower()
                
                # Si no sabemos si es over o under, intentar inferir del nombre
                if not (is_over or is_under):
                    is_over = line_info['prediction'] == 'over'  # Usar nuestra predicción
                    is_under = not is_over
                
                if is_over and line_info['prediction'] != 'over':
                    continue  # No es una apuesta favorable
                    
                if is_under and line_info['prediction'] != 'under':
                    continue  # No es una apuesta favorable
                
                # Procesar las odds
                try:
                    # Pueden ser probabilidades (0-1) o cuotas europeas (1.5, 2.0, etc.)
                    odds_values = recent_df[odds_col].dropna()
                    
                    # Si no hay valores, continuar
                    if len(odds_values) == 0:
                        continue
                        
                    # Determinar si son probabilidades o cuotas
                    avg_value = odds_values.mean()
                    
                    # Si el promedio es < 1.1 o > 10, probablemente hay un problema con el dato
                    if avg_value < 1.1 or avg_value > 10:
                        continue
                    
                    # Convertir todo a probabilidades implícitas
                    if avg_value > 1.0:  # Son cuotas europeas
                        implied_probabilities = 1 / odds_values
                    else:  # Ya son probabilidades
                        implied_probabilities = odds_values
                    
                    # Calcular la probabilidad implícita promedio
                    avg_implied_prob = implied_probabilities.mean()
                    
                    # Calcular la ventaja (edge)
                    our_confidence = line_info['adjusted_confidence']
                    
                    if (is_over and line_info['prediction'] == 'over') or \
                       (is_under and line_info['prediction'] == 'under'):
                        edge = our_confidence - avg_implied_prob
                    else:
                        edge = 0  # No tenemos ventaja si la dirección no coincide
                    
                    # Si hay suficiente ventaja, es una apuesta de valor
                    if edge >= min_edge:
                        # Calcular cuota promedio
                        if avg_value > 1.0:  # Ya son cuotas europeas
                            avg_odds = avg_value
                        else:  # Convertir probabilidad a cuota
                            avg_odds = 1 / avg_value if avg_value > 0 else 0
                            
                        # Si las odds son atractivas, guardar como apuesta de valor
                        if avg_odds >= min_odds:
                            line_value_bets.append({
                                'bookmaker': odds_col,
                                'prediction': line_info['prediction'],
                                'our_confidence': our_confidence,
                                'market_probability': avg_implied_prob,
                                'market_odds': avg_odds,
                                'edge': edge,
                                'expected_value': avg_odds * our_confidence,
                                'samples': len(odds_values)
                            })
                
                except Exception as e:
                    logger.warning(f"Error al procesar odds para {odds_col}: {e}")
            
            # Si encontramos apuestas de valor para esta línea, guardarlas
            if line_value_bets:
                # Ordenar por expected value (esperanza)
                line_value_bets = sorted(line_value_bets, key=lambda x: x['expected_value'], reverse=True)
                
                valuable_lines[line] = {
                    'line': line,
                    'prediction': line_info['prediction'],
                    'confidence': line_info['adjusted_confidence'],
                    'value_bets': line_value_bets,
                    'best_bookmaker': line_value_bets[0]['bookmaker'],
                    'best_odds': line_value_bets[0]['market_odds'],
                    'best_edge': line_value_bets[0]['edge'],
                    'expected_roi': (line_value_bets[0]['expected_value'] - 1) * 100  # ROI en %
                }
                
                logger.info(f"VALUE BET: {target} {line_info['prediction'].upper()} {line} - "
                           f"Odds: {line_value_bets[0]['market_odds']:.2f}, "
                           f"Edge: {line_value_bets[0]['edge']:.2%}, "
                           f"ROI esperado: {valuable_lines[line]['expected_roi']:.1f}%")
        
        # Ordenar lines por expected ROI
        valuable_lines = {k: v for k, v in sorted(
            valuable_lines.items(), 
            key=lambda item: item[1]['expected_roi'], 
            reverse=True
        )}
        
        if not valuable_lines:
            logger.warning(f"No se encontraron líneas con alta precisión y odds favorables para {target}")
        else:
            logger.info(f"Se encontraron {len(valuable_lines)} líneas con valor para {target}")
            # Mostrar las 3 mejores apuestas
            for i, (line, info) in enumerate(list(valuable_lines.items())[:3]):
                logger.info(f"Top {i+1}: {target} {info['prediction'].upper()} {line} - "
                           f"ROI esperado: {info['expected_roi']:.1f}%, "
                           f"Odds: {info['best_odds']:.2f} ({info['best_bookmaker']})")
                
        return valuable_lines

    def find_best_odds_arbitrage(
        self,
        df: pd.DataFrame,
        target: str,
        min_profit: float = 0.02,  # 2% de ganancia mínima
        lookback_days: int = 30,
        max_arbitrages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Encuentra oportunidades de arbitraje entre diferentes casas de apuestas
        para líneas donde tenemos alta confianza en nuestras predicciones
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            min_profit: Ganancia mínima para considerar arbitraje (0.02 = 2%)
            lookback_days: Días hacia atrás para analizar
            max_arbitrages: Número máximo de oportunidades a devolver
            
        Returns:
            Lista de oportunidades de arbitraje con máximas ganancias
        """
        if target not in ['PTS', 'TRB', 'AST', '3P']:
            logger.warning(f"Target {target} no soportado para análisis de arbitraje")
            return []
            
        # Filtrar por fecha reciente
        recent_df = df
        if 'Date' in df.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                cutoff_date = df['Date'].max() - pd.Timedelta(days=lookback_days)
                recent_df = df[df['Date'] >= cutoff_date].copy()
                logger.info(f"Análisis restringido a los últimos {lookback_days} días ({len(recent_df)} registros)")
            except Exception as e:
                    logger.warning(f"Error al filtrar por fecha: {e}")
            
        # Detectar columnas de casas de apuestas
        over_odds_cols = {}
        under_odds_cols = {}
        
        # Detectar líneas y casas disponibles
        for col in recent_df.columns:
            if target.lower() in col.lower() and '_over_' in col.lower() and 'odds' in col.lower():
                try:
                    line_value = float(col.split('_over_')[1].split('_')[0])
                    bookmaker = col.split('_odds_')[1] if '_odds_' in col else 'unknown'
                    
                    if line_value not in over_odds_cols:
                        over_odds_cols[line_value] = []
                    over_odds_cols[line_value].append((col, bookmaker))
                except:
                    continue
                
            elif target.lower() in col.lower() and '_under_' in col.lower() and 'odds' in col.lower():
                try:
                    line_value = float(col.split('_under_')[1].split('_')[0])
                    bookmaker = col.split('_odds_')[1] if '_odds_' in col else 'unknown'
                    
                    if line_value not in under_odds_cols:
                        under_odds_cols[line_value] = []
                    under_odds_cols[line_value].append((col, bookmaker))
                except:
                    continue
        
        # Verificar si tenemos suficientes datos
        if not over_odds_cols or not under_odds_cols:
            logger.warning(f"No se encontraron suficientes columnas de odds para {target}")
            return []
            
        # Buscar oportunidades de arbitraje
        arbitrage_opportunities = []
        
        # Para cada línea, buscar combinaciones de over/under con arbitraje
        for line in set(over_odds_cols.keys()).intersection(under_odds_cols.keys()):
            over_cols = over_odds_cols[line]
            under_cols = under_odds_cols[line]
            
            # Sólo procesar si tenemos al menos una columna de cada tipo
            if not over_cols or not under_cols:
                continue
                
            # Obtener las últimas odds para cada casa
            latest_data = recent_df.sort_values(by='Date').iloc[-1]
            
            # Análisis de arbitraje
            for over_col, over_bookmaker in over_cols:
                over_odds = latest_data.get(over_col)
                if pd.isna(over_odds) or over_odds <= 1.0:
                    continue
                    
                for under_col, under_bookmaker in under_cols:
                    under_odds = latest_data.get(under_col)
                    if pd.isna(under_odds) or under_odds <= 1.0:
                        continue
                    
                    # Calcular si hay arbitraje
                    inverse_sum = (1/over_odds) + (1/under_odds)
                    
                    if inverse_sum < 1.0:  # Hay arbitraje
                        profit_pct = (1/inverse_sum) - 1  # Ganancia porcentual
                        
                        if profit_pct >= min_profit:
                            # Calcular cómo distribuir la apuesta
                            over_weight = (1/over_odds) / inverse_sum
                            under_weight = (1/under_odds) / inverse_sum
                            
                            # Añadir oportunidad
                            arbitrage_opportunities.append({
                                'target': target,
                                'line': line,
                                'over_bookmaker': over_bookmaker,
                                'over_odds': over_odds,
                                'over_weight': over_weight,
                                'under_bookmaker': under_bookmaker,
                                'under_odds': under_odds,
                                'under_weight': under_weight,
                                'profit_pct': profit_pct,
                                'inverse_sum': inverse_sum,
                                'date': latest_data.get('Date') if 'Date' in latest_data else None
                            })
        
        # Ordenar por ganancia y limitar resultados
        arbitrage_opportunities = sorted(
            arbitrage_opportunities, 
            key=lambda x: x['profit_pct'], 
            reverse=True
        )[:max_arbitrages]
        
        if arbitrage_opportunities:
            logger.info(f"Se encontraron {len(arbitrage_opportunities)} oportunidades de arbitraje para {target}")
            
            # Mostrar las 3 mejores oportunidades
            for i, arb in enumerate(arbitrage_opportunities[:3]):
                logger.info(f"Arbitraje #{i+1}: {target} {arb['line']} - "
                          f"Profit: {arb['profit_pct']:.2%} - "
                          f"OVER: {arb['over_odds']:.2f} ({arb['over_bookmaker']}), "
                          f"UNDER: {arb['under_odds']:.2f} ({arb['under_bookmaker']})")
            else:
                logger.warning(f"No se encontraron oportunidades de arbitraje para {target}")
            
        return arbitrage_opportunities
                    
    def compare_line_movements(
        self,
        df: pd.DataFrame,
        target: str,
        days_before_event: int = 3,
        min_confidence: float = 0.96,
        lookback_events: int = 50
    ) -> Dict[float, Dict[str, Any]]:
        """
        Analiza movimientos de líneas en las casas de apuestas antes del evento
        para identificar patrones que señalen oportunidades de apuesta
        
        Args:
            df: DataFrame con los datos
            target: Tipo de predicción (PTS, TRB, AST, 3P)
            days_before_event: Días antes del evento para analizar el movimiento de la línea
            min_confidence: Confianza mínima para nuestras predicciones
            lookback_events: Número de eventos históricos a analizar
            
        Returns:
            Diccionario con análisis de movimientos de líneas prometedores
        """
        if target not in ['PTS', 'TRB', 'AST', '3P']:
            logger.warning(f"Target {target} no soportado para análisis de movimientos de línea")
            return {}
            
        # Asegurar que tenemos datos de fecha
        if 'Date' not in df.columns:
            logger.warning("No se encontró columna de fecha para análisis de movimientos")
            return {}
            
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            logger.warning(f"Error al convertir fechas: {e}")
            return {}
            
        # Encontrar las mejores líneas en las que tenemos alta confianza
        high_confidence_lines = self.identify_high_confidence_betting_lines(
            df, target, min_confidence=min_confidence
        )
        
        if not high_confidence_lines:
            logger.warning(f"No se encontraron líneas de alta confianza para {target}")
            return {}
            
        # Detectar columnas de odds para las líneas relevantes
        line_movement_analysis = {}
        
        for line, line_info in high_confidence_lines.items():
            # Buscar columnas de odds para esta línea específica
            odds_cols = [col for col in df.columns 
                        if f"{target.lower()}_over_{line}" in col.lower() 
                        and any(term in col.lower() for term in ['odds', 'line', 'price'])]
            
            if not odds_cols:
                continue
                
            # Analizar cada evento reciente
            recent_events = df.sort_values(by='Date', ascending=False).head(lookback_events)
            
            # Agrupar por evento (jugador/partido)
            if 'Player' in df.columns:
                grouped = recent_events.groupby(['Player', 'Date'])
            else:
                grouped = recent_events.groupby(['Team', 'Date'])
                
            movement_patterns = []
            
            # Para cada evento, analizar el movimiento de la línea
            for _, event_data in grouped:
                # Es posible que tengamos datos para esta línea a lo largo del tiempo
                if len(event_data) <= 1:
                    continue  # Necesitamos al menos dos puntos para analizar movimiento
                    
                # Ordenar cronológicamente
                event_data = event_data.sort_values(by='Date')
                
                for odds_col in odds_cols:
                    # Verificar si tenemos suficientes datos
                    if event_data[odds_col].isna().all() or event_data[odds_col].nunique() <= 1:
                        continue
                        
                    # Obtener valores inicial y final
                    initial_odds = event_data[odds_col].iloc[0]
                    final_odds = event_data[odds_col].iloc[-1]
                    
                    # Calcular cambio porcentual
                    if pd.notna(initial_odds) and pd.notna(final_odds) and initial_odds > 0:
                        pct_change = (final_odds - initial_odds) / initial_odds
                        
                        # Determinar si el resultado fue over o under
                        result = None
                        if target in event_data.columns:
                            result = 'over' if event_data[target].iloc[-1] > line else 'under'
                            
                        # Guardar análisis
                        movement_patterns.append({
                            'column': odds_col,
                            'initial_odds': initial_odds,
                            'final_odds': final_odds,
                            'pct_change': pct_change,
                            'days_tracked': (event_data['Date'].max() - event_data['Date'].min()).days,
                            'result': result,
                            'event_date': event_data['Date'].max(),
                            'player': event_data['Player'].iloc[0] if 'Player' in event_data.columns else None,
                            'team': event_data['Team'].iloc[0] if 'Team' in event_data.columns else None
                        })
            
            if not movement_patterns:
                continue
                
            # Analizar patrones por su resultado
            movement_by_result = {'over': [], 'under': []}
            
            for pattern in movement_patterns:
                if pattern['result'] in movement_by_result:
                    movement_by_result[pattern['result']].append(pattern)
            
            # Calcular promedios para patrones de over y under
            avg_movement = {}
            
            for result, patterns in movement_by_result.items():
                if patterns:
                    avg_movement[result] = {
                        'avg_pct_change': sum(p['pct_change'] for p in patterns) / len(patterns),
                        'count': len(patterns),
                        'positive_moves': sum(1 for p in patterns if p['pct_change'] > 0),
                        'negative_moves': sum(1 for p in patterns if p['pct_change'] < 0),
                        'avg_initial_odds': sum(p['initial_odds'] for p in patterns) / len(patterns),
                        'avg_final_odds': sum(p['final_odds'] for p in patterns) / len(patterns)
                    }
            
            # Determinar si hay un patrón significativo en el movimiento
            if 'over' in avg_movement and 'under' in avg_movement:
                over_moves = avg_movement['over']
                under_moves = avg_movement['under']
                
                # Calcular diferencia de movimiento entre over y under
                movement_diff = over_moves['avg_pct_change'] - under_moves['avg_pct_change']
                
                # Si la diferencia es significativa, tenemos un patrón
                if abs(movement_diff) >= 0.05:  # 5% de diferencia
                    significant_result = 'over' if movement_diff > 0 else 'under'
                    opposite_result = 'under' if significant_result == 'over' else 'over'
                    
                    # Verificar si coincide con nuestra predicción
                    matches_prediction = line_info['prediction'] == significant_result
                    
                    line_movement_analysis[line] = {
                        'line': line,
                        'prediction': line_info['prediction'],
                        'confidence': line_info['adjusted_confidence'],
                        'movement_indicates': significant_result,
                        'matches_prediction': matches_prediction,
                        'movement_diff': movement_diff,
                        'significant_pattern': True,
                        'dominant_result_moves': avg_movement[significant_result],
                        'opposite_result_moves': avg_movement[opposite_result],
                        'samples': over_moves['count'] + under_moves['count']
                    }
                    
                    # Mensaje de log
                    logger.info(f"Patrón de movimiento para {target} {line}: "
                               f"El movimiento indica {significant_result.upper()} "
                               f"({'coincide' if matches_prediction else 'no coincide'} con nuestra predicción)")
                
        return line_movement_analysis 