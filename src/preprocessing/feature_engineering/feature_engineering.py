import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import signal
import concurrent.futures
import multiprocessing
import json
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Para transformaciones wavelet 
try:
    import pywt
except ImportError:
    pywt = None

# Configuración del sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('FeatureEngineering')

class FeatureEngineering:
    def __init__(self, window_sizes=[3, 10, 20], correlation_threshold=0.95, enable_correlation_analysis=True, n_jobs=1):
        """
        Inicializa el sistema de ingeniería de características optimizado
        
        Args:
            window_sizes: Tamaños de ventana reducidos para características temporales
                (corto, medio y largo plazo)
            correlation_threshold: Umbral para identificar y eliminar características altamente correlacionadas
            enable_correlation_analysis: Activa el análisis de correlación para eliminar redundancias
            n_jobs: Número de procesos en paralelo (1 = secuencial)
        """
        self.window_sizes = window_sizes
        self.correlation_threshold = correlation_threshold
        self.enable_correlation_analysis = enable_correlation_analysis
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.important_features = []  # Lista para almacenar características importantes
        
        logger.info(f"Inicializando FeatureEngineering con ventanas: {window_sizes}")
        if enable_correlation_analysis:
            logger.info(f"Análisis de correlación activado con umbral: {correlation_threshold}")
        
    def generate_points_over_under_features(self, df):
        """
        Genera características de Over/Under para puntos de equipo y puntos totales.
        
        Args:
            df (pd.DataFrame): DataFrame de entrada con columnas de puntuación
        
        Returns:
            pd.DataFrame: DataFrame con nuevas columnas de Over/Under
        """
        # Columnas a proteger
        protected_columns = ['team_score', 'opp_score', 'total_score', 'point_diff',]
        
        try:
            # Verificar columnas disponibles
            score_columns = [col.lower() for col in df.columns]
        
            # Generar Total_Points_Over_Under (suma de puntos de ambos equipos)
            if 'total_score' in df.columns:
                df['Total_Points_Over_Under'] = df['total_score']
                logger.info("Total_Points_Over_Under generado a partir de total_score")
            elif all(col in df.columns for col in ['team_score', 'opp_score']):
                df['Total_Points_Over_Under'] = df['team_score'] + df['opp_score']
                logger.info("Total_Points_Over_Under generado a partir de team_score + opp_score")
            elif all(col in df.columns for col in ['Team_Score', 'Opp_Score']):
                df['Total_Points_Over_Under'] = df['Team_Score'] + df['Opp_Score']
                logger.info("Total_Points_Over_Under generado a partir de Team_Score + Opp_Score")
            else:
                # Buscar cualquier columna que pueda contener puntos totales
                total_cols = [col for col in df.columns if 'total' in col.lower() and 'score' in col.lower() or 'points' in col.lower()]
                if total_cols:
                    df['Total_Points_Over_Under'] = df[total_cols[0]]
                    logger.info(f"Total_Points_Over_Under generado a partir de {total_cols[0]}")
                else:
                    # Si no hay columnas de puntos totales, intentar crear a partir de puntos de equipo
                    team_cols = [col for col in df.columns if ('team' in col.lower() or 'home' in col.lower()) and ('score' in col.lower() or 'points' in col.lower())]
                    opp_cols = [col for col in df.columns if ('opp' in col.lower() or 'away' in col.lower()) and ('score' in col.lower() or 'points' in col.lower())]
            
                    if team_cols and opp_cols:
                        df['Total_Points_Over_Under'] = df[team_cols[0]] + df[opp_cols[0]]
                        logger.info(f"Total_Points_Over_Under generado a partir de {team_cols[0]} + {opp_cols[0]}")
                    else:
                        # Último recurso: crear una columna con valores predeterminados
                        logger.warning("No se encontraron columnas para generar Total_Points_Over_Under. Creando con valores predeterminados.")
                        df['Total_Points_Over_Under'] = 220  # Valor promedio de puntos totales en NBA
        
            # Generar Team_Points_Over_Under (puntos del equipo)
            if 'team_score' in df.columns:
                df['Team_Points_Over_Under'] = df['team_score']
                logger.info("Team_Points_Over_Under generado a partir de team_score")
            elif 'Team_Score' in df.columns:
                df['Team_Points_Over_Under'] = df['Team_Score']
                logger.info("Team_Points_Over_Under generado a partir de Team_Score")
            else:
                # Buscar cualquier columna que pueda contener puntos de equipo
                team_cols = [col for col in df.columns if ('team' in col.lower() or 'home' in col.lower()) and ('score' in col.lower() or 'points' in col.lower())]
                if team_cols:
                    df['Team_Points_Over_Under'] = df[team_cols[0]]
                    logger.info(f"Team_Points_Over_Under generado a partir de {team_cols[0]}")
                else:
                    # Último recurso: crear una columna con valores predeterminados
                    logger.warning("No se encontraron columnas para generar Team_Points_Over_Under. Creando con valores predeterminados.")
                    df['Team_Points_Over_Under'] = 110  # Valor promedio de puntos de equipo en NBA
    
        except Exception as e:
            
            logger.error(f"Error generando características de Over/Under: {e}")
            
            # Asegurar que las columnas existan incluso si hay un error
            if 'Total_Points_Over_Under' not in df.columns:
                df['Total_Points_Over_Under'] = 220
            if 'Team_Points_Over_Under' not in df.columns:
                df['Team_Points_Over_Under'] = 110
        
        # Método para proteger columnas durante la eliminación por correlación
        def protect_columns(self, df, to_drop, protected_columns):
            """
            Elimina columnas correlacionadas excepto las columnas protegidas.
            
            Args:
                df (pd.DataFrame): DataFrame de entrada
                to_drop (list): Lista de columnas a eliminar
                protected_columns (list): Lista de columnas a proteger
            
            Returns:
                list: Lista de columnas a eliminar, excluyendo las protegidas
            """
            # Filtrar columnas a eliminar, excluyendo las protegidas
            filtered_to_drop = [
                col for col in to_drop 
                if col not in protected_columns and 
                   col.lower() not in [pc.lower() for pc in protected_columns]
            ]
            
            return filtered_to_drop
        
        # Sobrescribir el método de eliminación de columnas
        self.get_columns_to_drop = lambda df, to_drop_extreme, to_drop_high: (
            protect_columns(self, df, to_drop_extreme + to_drop_high, protected_columns)
        )
        
        return df
    
    def generate_all_features(self, df):
        """
        Aplica todas las transformaciones de características en un solo paso.
        Coordina el proceso completo de ingeniería de características.
        """
        start_time = time.time()
        logger.info(f"Iniciando generación de características con {len(self.window_sizes)} ventanas temporales")
        logger.info(f"Dimensiones iniciales: {df.shape} ({df.shape[0]} filas, {df.shape[1]} columnas)")
        
        # Renombrar columnas críticas si no existen
        if 'Win' not in df.columns and 'is_win' in df.columns:
            df['Win'] = df['is_win']
        if 'is_win' not in df.columns and 'Win' in df.columns:
            df['is_win'] = df['Win']
            
        # Aplicar ingeniería de características en secuencia o paralelamente según configuración
        if self.n_jobs > 1 and len(df) > 1000:  # Solo paralelizar si hay suficientes datos
            logger.info(f"Procesando características en paralelo con {self.n_jobs} workers")
            df_processed = self._generate_features_parallel(df)
        else:
            logger.info("Procesando características secuencialmente")
            # Aplicar transformaciones en secuencia
            df_processed = df.copy()
            
            # Crear barra de progreso para los pasos de procesamiento
            steps = [
                # Primero las características básicas y de equipo
                ("Características de over/under (Total_Points y Team_Points)", self.generate_points_over_under_features),
                ("Características de equipo", self.add_team_features),
                ("Características de puntuación de equipo", self.add_team_scoring_features),
                ("Características de over/under", self.add_over_under_features),
                ("Características de playoffs", self.add_playoff_features),
                
                # Luego las características individuales
                ("Características temporales", self.add_temporal_features),
                ("Métricas de eficiencia", self.add_efficiency_metrics),
                ("Características contextuales", self.add_context_features),
                ("Características de matchup", self.add_matchup_features),
                ("Características físicas de matchup", self.add_physical_matchup_features),
                ("Características de impacto físico", self.add_physical_impact_features),  # Añadido nuevo método
                ("Indicadores de momentum", self.add_momentum_indicators),
                ("Métricas de fatiga", self.add_fatigue_metrics),
                ("Estadísticas avanzadas", self.add_advanced_stats),
                
                # Finalmente las características de predicción y situacionales
                ("Características de predicción de equipo", self.add_team_prediction_features),
                ("Características de puntuación de jugador", self.add_player_scoring_features),
                ("Características de rebotes de jugador", self.add_player_rebounding_features),
                ("Características de asistencias de jugador", self.add_player_assist_features),
                ("Características de interacción de equipo", self.add_team_interaction_features),
                ("Características de predicción de jugador", self.add_player_prediction_features),
                ("Características de interacción de jugador", self.add_player_interaction_features),
                ("Características de probabilidad optimizadas", self.add_optimized_probability_features),
                ("Características de predicción de líneas", self.add_line_prediction_features),
                ("Características de situación de juego", self.add_game_situation_features),
                ("Características de hitos", self.add_milestone_features)
            ]
            
            print("\nGenerando características...")
            for step_name, step_func in tqdm(steps, desc="Progreso general", unit="paso"):
                print(f"\nPaso: {step_name}...")
                df_result = step_func(df_processed)
                
                # Verificar si el método devolvió None
                if df_result is None:
                    logger.error(f"Error: El método {step_name} devolvió None en lugar de un DataFrame")
                    # Mantener el DataFrame anterior para evitar que el proceso falle
                    print(f"  ✗ Error en {step_name}: devuelve None. Manteniendo DataFrame anterior.")
                    continue
                
                # Actualizar el DataFrame procesado
                df_processed = df_result
                
                # Mostrar cambio en dimensiones
                new_cols = set(df_processed.columns) - set(df.columns)
                print(f"  ✓ Generadas {len(new_cols)} características totales hasta ahora")
                
                # Verificar columnas críticas después de cada paso
                critical_columns = ['Win', 'is_win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']
                present_columns = [col for col in critical_columns if col in df_processed.columns]
                if present_columns:
                    print(f"  ✓ Columnas críticas presentes: {present_columns}")
        
        # Asegurar que las columnas críticas existan
        if 'Win' not in df_processed.columns and 'is_win' in df_processed.columns:
            df_processed['Win'] = df_processed['is_win']
        if 'is_win' not in df_processed.columns and 'Win' in df_processed.columns:
            df_processed['is_win'] = df_processed['Win']
            
        # Verificación final de columnas críticas
        missing_columns = []
        if 'Win' not in df_processed.columns and 'is_win' not in df_processed.columns:
            missing_columns.append('Win/is_win')
        if 'Total_Points_Over_Under' not in df_processed.columns:
            missing_columns.append('Total_Points_Over_Under')
        if 'Team_Points_Over_Under' not in df_processed.columns:
            missing_columns.append('Team_Points_Over_Under')
            
        if missing_columns:
            logger.warning(f"Columnas críticas faltantes después de la generación: {missing_columns}")
        
        # Análisis de correlación (opcional)
        if self.enable_correlation_analysis:
            logger.info("Realizando análisis de correlación y eliminando características redundantes...")
            df_processed = self.remove_highly_correlated_features(df_processed)
        
        # Registrar resultados
        total_features = df_processed.shape[1] - df.shape[1]
        logger.info(f"Proceso completado en {time.time() - start_time:.2f} segundos")
        logger.info(f"Características generadas: {total_features}, dimensiones finales: {df_processed.shape}")
        
        return df_processed
    
    def _generate_features_parallel(self, df):
        """
        Versión paralelizada de la generación de características.
        
        Args:
            df: DataFrame con datos de partidos
            
        Returns:
            DataFrame con todas las características añadidas
        """
        # Funciones que pueden ejecutarse en paralelo (sin dependencias mutuas)
        parallel_steps = [
            ('temporal', self.add_temporal_features),
            ('efficiency', self.add_efficiency_metrics),
            ('context', self.add_context_features),
            ('matchup', self.add_matchup_features),
            ('physical', self.add_physical_matchup_features),
            ('momentum', self.add_momentum_indicators),
            ('fatigue', self.add_fatigue_metrics),
            ('advanced', self.add_advanced_stats)
        ]
        
        # Ejecutar pasos paralelizables
        results = {}
        
        # Ejecutar secuencialmente en lugar de en paralelo para evitar problemas de pickling
        logger.info("Cambiando a procesamiento secuencial para evitar problemas de pickling")
        for step_name, step_func in tqdm(parallel_steps, desc="Procesando características en paralelo"):
            try:
                logger.info(f"Ejecutando paso secuencial: {step_name}")
                # Crear una copia de los datos para evitar problemas
                step_data = df.copy()
                # Ejecutar la función de feature engineering
                result = step_func(step_data)
                # Obtener solo las columnas nuevas
                new_cols = set(result.columns) - set(df.columns)
                if new_cols:
                    results[step_name] = result[list(new_cols)].copy()
                    logger.info(f"Paso {step_name} completado: {len(new_cols)} características generadas")
                else:
                    results[step_name] = pd.DataFrame(index=df.index)
                    logger.info(f"Paso {step_name} completado: 0 características generadas")
            except Exception as e:
                logger.error(f"Error en procesamiento de {step_name}: {str(e)}")
                logger.error(f"Detalles: {traceback.format_exc()}")
                results[step_name] = pd.DataFrame(index=df.index)
        
        # Unir todas las características nuevas con el DataFrame original
        df_result = df.copy()
        for step_name, step_result in results.items():
            if not step_result.empty:
                # Verificar que no hay columnas duplicadas
                duplicate_cols = set(df_result.columns).intersection(set(step_result.columns))
                if duplicate_cols:
                    logger.warning(f"Se encontraron columnas duplicadas en {step_name}: {duplicate_cols}. Eliminando duplicados.")
                    step_result = step_result.drop(columns=list(duplicate_cols))
                
                # Realizar la unión con manejo de errores
                try:
                    df_result = pd.concat([df_result, step_result], axis=1)
                    logger.debug(f"Unidas características de {step_name}: {len(step_result.columns)} columnas")
                except Exception as e:
                    logger.error(f"Error uniendo características de {step_name}: {str(e)}")
        
        # Pasos que deben ejecutarse secuencialmente después de los pasos paralelos
        sequential_steps = [
            ('team', self.add_team_features),
            ('team_prediction', self.add_team_prediction_features),
            ('player_prediction', self.add_player_prediction_features),
            ('line_prediction', self.add_line_prediction_features),
            ('game_situation', self.add_game_situation_features),
            ('milestone', self.add_milestone_features),  
            ('probability', self.add_optimized_probability_features),
            ('over_under', self.generate_points_over_under_features)
        ]
        
        # Ejecutar pasos secuenciales (necesarios porque dependen de características generadas anteriormente)
        for step_name, step_func in sequential_steps:
            try:
                logger.info(f"Ejecutando paso secuencial: {step_name}")
                df_step_result = step_func(df_result)
                
                # Verificar y convertir valores no escalares si es necesario
                new_cols = set(df_step_result.columns) - set(df_result.columns)
                for col in new_cols:
                    if col in df_step_result.columns:
                        # Verificar si hay valores que no son escalares
                        non_scalar_value = False
                        try:
                            # Intentar calcular std para detectar problemas
                            _ = df_step_result[col].std()
                        except Exception:
                            non_scalar_value = True
                            
                        if non_scalar_value:
                            logger.warning(f"Detectados valores no escalares en columna {col}. Convirtiendo a float.")
                            # Convertir valores problemáticos a float
                            df_step_result[col] = df_step_result[col].apply(
                                lambda x: float(x) if hasattr(x, 'item') else (
                                    float(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else float(0)
                                )
                            )
                
                df_result = df_step_result
            except Exception as e:
                logger.error(f"Error en paso secuencial {step_name}: {str(e)}")
        
        return df_result
    
    def get_target_specific_features(self):
        """
        Devuelve un diccionario con las características específicas que deben preservarse
        para cada target de predicción, incluso si tienen alta correlación con otras.
        
        Returns:
            Dict con listas de características por target
        """
        # Características específicas por target que nunca deben eliminarse
        target_features = {
            # Características para modelos de equipo
            'is_win': [
                # Estadísticas básicas
                'team_score', 'opp_score', 'point_diff', 'is_win', 
                # Tendencias de equipo
                'team_win_streak', 'opp_win_streak', 'team_home_win_pct', 'team_away_win_pct',
                # Eficiencia
                'team_offensive_rating', 'team_defensive_rating', 'opp_offensive_rating', 'opp_defensive_rating',
                # Factores contextuales
                'is_home', 'days_rest', 'team_b2b', 'opp_b2b',
                # Interacciones clave
                'team_off_rating_vs_opp_def_rating', 'team_def_rating_vs_opp_off_rating',
                # Métricas avanzadas
                'team_net_rating', 'team_true_shooting_pct', 'opp_true_shooting_pct',
                # Características de situación
                'team_clutch_win_pct', 'team_comeback_factor',
                # Características adicionales para predicción de victoria
                'team_score_avg', 'opp_score_avg',
                'team_score_home_avg', 'team_score_away_avg',
                'team_pace', 'opp_pace',
                'team_assist_ratio', 'opp_assist_ratio',
                'team_rebound_rate', 'opp_rebound_rate',
                'team_turnover_rate', 'opp_turnover_rate',
                'team_shooting_efficiency', 'opp_shooting_efficiency',
                'team_recent_form', 'opp_recent_form',
                'head_to_head_wins', 'matchup_history'
            ],
            'Total_Points': [
                # Estadísticas básicas
                'team_score', 'opp_score', 'total_score', 
                # Promedios
                'team_PTS_avg', 'opp_PTS_avg', 'team_PTS_home_avg', 'team_PTS_away_avg',
                'opp_PTS_home_avg', 'opp_PTS_away_avg',
                # Ritmo y eficiencia
                'team_pace', 'opp_pace', 'team_offensive_rating', 'opp_defensive_rating',
                # Factores contextuales
                'is_home', 'days_rest', 'team_b2b', 'opp_b2b',
                # Tendencias
                'team_total_points_trend', 'opp_total_points_trend',
                # Interacciones clave
                'pace_combined', 'offensive_efficiency_combined',
                # Métricas de shooting
                'team_3pt_rate', 'opp_3pt_rate', 'team_ft_rate', 'opp_ft_rate',
                # Factores de temporada
                'season_avg_total', 'season_home_avg_total', 'season_away_avg_total'
            ],
            'Team_Points': [
                # Estadísticas básicas
                'team_score', 'team_PTS_avg', 'team_PTS_home_avg', 'team_PTS_away_avg',
                # Ritmo y eficiencia
                'team_pace', 'team_offensive_rating', 'opp_defensive_rating',
                # Factores contextuales
                'is_home', 'days_rest', 'team_b2b',
                # Tendencias
                'team_points_trend', 'team_points_home_trend', 'team_points_away_trend',
                # Interacciones clave
                'team_off_rating_vs_opp_def_rating', 'pace_impact_on_scoring',
                # Métricas de shooting
                'team_3pt_rate', 'team_ft_rate', 'team_efg_pct',
                # Factores de temporada
                'season_avg_team_points', 'season_home_avg_team_points', 'season_away_avg_team_points',
                # Factores de matchup
                'historical_matchup_scoring', 'matchup_pace_factor'
            ],
            
            # Características para modelos de jugador
            'PTS': [
                # Estadísticas básicas
                'PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'MP', 'GmSc',
                # Promedios temporales
                'PTS_3_avg', 'PTS_5_avg', 'PTS_10_avg', 'PTS_20_avg',
                # Factores contextuales
                'is_home', 'days_rest', 'PTS_home_factor', 'b2b_impact_PTS',
                # Tendencias y momentum
                'PTS_momentum', 'PTS_trend', 'PTS_streak', 'PTS_recent_variance',
                # Probabilidades optimizadas
                'PTS_p25_prob', 'PTS_p50_prob', 'PTS_p75_prob',
                'PTS_p25_prob_smooth', 'PTS_p50_prob_smooth', 'PTS_p75_prob_smooth',
                # Consistencia y físico
                'PTS_consistency', 'PTS_PhysPerf_Index', 'PTS_efficiency_rating',
                # Componentes wavelet
                'PTS_wavelet_trend', 'PTS_wavelet_seasonality', 'PTS_wavelet_residual',
                # Contribución al equipo
                'Player_Team_Contribution_PTS', 'PTS_usage_rate', 'PTS_vs_team_avg',
                # Factores de oponente
                'opp_pts_allowed_to_position', 'opp_defensive_rating_vs_position',
                # Factores de matchup
                'defender_impact_on_PTS', 'matchup_history_PTS',
                # Factores de situación
                'clutch_PTS_factor', 'close_game_PTS_impact'
            ],
            'TRB': [
                # Estadísticas básicas
                'TRB', 'DREB', 'OREB', 'MP', 'GmSc', 'Height_Inches', 'Weight', 'BMI',
                # Promedios temporales
                'TRB_3_avg', 'TRB_5_avg', 'TRB_10_avg', 'TRB_20_avg', 'DREB_5_avg', 'OREB_5_avg',
                # Factores contextuales
                'is_home', 'days_rest', 'TRB_home_factor', 'b2b_impact_TRB',
                # Tendencias y momentum
                'TRB_momentum', 'TRB_trend', 'TRB_streak', 'TRB_recent_variance',
                # Probabilidades optimizadas
                'TRB_p25_prob', 'TRB_p50_prob', 'TRB_p75_prob',
                'TRB_p25_prob_smooth', 'TRB_p50_prob_smooth', 'TRB_p75_prob_smooth',
                # Consistencia y físico
                'TRB_consistency', 'TRB_PhysPerf_Index', 'Size_Component', 'Proportion_Component',
                'rebounding_efficiency', 'box_out_rating',
                # Componentes wavelet
                'TRB_wavelet_trend', 'TRB_wavelet_seasonality', 'TRB_wavelet_residual',
                # Contribución al equipo
                'Player_Team_Contribution_TRB', 'TRB_vs_team_avg', 'TRB_share_of_team',
                # Factores de oponente
                'opp_rebounding_rate', 'opp_frontcourt_height_avg', 'opp_rebounding_allowed_to_position',
                # Factores de matchup
                'matchup_height_advantage', 'matchup_history_TRB', 'frontcourt_matchup_rating',
                # Factores de situación
                'TRB_by_score_margin', 'TRB_in_close_games', 'TRB_by_quarter'
            ],
            'AST': [
                # Estadísticas básicas
                'AST', 'TOV', 'AST/TOV', 'MP', 'GmSc', 'USG%',
                # Promedios temporales
                'AST_3_avg', 'AST_5_avg', 'AST_10_avg', 'AST_20_avg', 'TOV_5_avg',
                # Factores contextuales
                'is_home', 'days_rest', 'AST_home_factor', 'b2b_impact_AST',
                # Tendencias y momentum
                'AST_momentum', 'AST_trend', 'AST_streak', 'AST_recent_variance',
                # Probabilidades optimizadas
                'AST_p25_prob', 'AST_p50_prob', 'AST_p75_prob',
                'AST_p25_prob_smooth', 'AST_p50_prob_smooth', 'AST_p75_prob_smooth',
                # Consistencia y físico
                'AST_consistency', 'AST_PhysPerf_Index', 'Density_Component',
                'playmaking_efficiency', 'court_vision_rating',
                # Componentes wavelet
                'AST_wavelet_trend', 'AST_wavelet_seasonality', 'AST_wavelet_residual',
                # Contribución al equipo
                'Player_Team_Contribution_AST', 'AST_vs_team_avg', 'AST_share_of_team', 'team_assist_rate',
                # Factores de oponente
                'opp_assist_allowed_to_position', 'opp_defensive_disruption_rating',
                # Factores de matchup
                'defender_impact_on_AST', 'matchup_history_AST', 'team_shooting_impact_on_AST',
                # Factores de situación
                'AST_by_score_margin', 'AST_in_close_games', 'AST_by_quarter'
            ],
            '3PT': [
                # Estadísticas básicas
                '3P', '3PA', '3P%', 'FG%', 'FG', 'FGA', 'MP', 'GmSc', 'USG%',
                # Promedios temporales
                '3P_3_avg', '3P_5_avg', '3P_10_avg', '3PA_3_avg', '3PA_5_avg', '3PA_10_avg',
                '3P%_3_avg', '3P%_5_avg', '3P%_10_avg',
                # Factores contextuales
                'is_home', 'days_rest', '3P_home_factor', 'b2b_impact_3P',
                # Tendencias y momentum
                '3P_momentum', '3P_trend', '3P_streak', '3P_recent_variance',
                # Probabilidades optimizadas
                '3P_p25_prob', '3P_p50_prob', '3P_p75_prob',
                '3P_p25_prob_smooth', '3P_p50_prob_smooth', '3P_p75_prob_smooth',
                # Consistencia y eficiencia
                '3P_consistency', '3P%_consistency', '3P_efficiency_rating', 'shooting_form_stability',
                # Componentes wavelet
                '3P_wavelet_trend', '3P_wavelet_seasonality', '3P_wavelet_residual',
                # Contribución al equipo
                'Player_Team_Contribution_3P', '3P_vs_team_avg', '3P_share_of_team', 'team_3pt_rate',
                # Factores de oponente
                'opp_3pt_defense_rating', 'opp_perimeter_defense_rating', 'opp_3pt_allowed_to_position',
                # Factores de matchup
                'defender_impact_on_3P', 'matchup_history_3P', 'perimeter_matchup_rating',
                # Factores de situación
                '3P_by_score_margin', '3P_in_close_games', '3P_by_quarter', 'clutch_3P_factor'
            ],
            'Double_Double': [
                # Estadísticas básicas
                'PTS', 'TRB', 'AST', 'BLK', 'STL', 'MP', 'GmSc', 'USG%',
                # Historial
                'Double_Double_History', 'Double_Double_Last_5', 'Double_Double_Last_10',
                'Double_Double_Rate', 'Double_Double_Streak',
                # Promedios
                'PTS_10_avg', 'TRB_10_avg', 'AST_10_avg', 'BLK_10_avg', 'STL_10_avg',
                # Probabilidades
                'PTS_p50_prob', 'TRB_p50_prob', 'AST_p50_prob', 'BLK_p50_prob', 'STL_p50_prob',
                'Double_Double_Probability', 'Double_Double_Consistency',
                # Factores contextuales
                'is_home', 'days_rest', 'MP_last_10_games', 'b2b_impact_stats',
                # Factores físicos
                'Size_Component', 'Density_Component', 'Proportion_Component',
                # Factores de oponente
                'opp_defensive_rating', 'opp_position_defense_rating',
                # Factores de matchup
                'matchup_advantage_rating', 'historical_matchup_performance',
                # Factores de situación
                'stat_distribution_evenness', 'role_in_team', 'team_injury_impact'
            ],
            'Triple_Double': [
                # Estadísticas básicas
                'PTS', 'TRB', 'AST', 'BLK', 'STL', 'MP', 'GmSc', 'USG%',
                # Historial
                'Triple_Double_History', 'Triple_Double_Prob', 'Triple_Double_Last_10',
                'Triple_Double_Rate', 'Triple_Double_Streak', 'Career_Triple_Doubles',
                # Promedios
                'PTS_10_avg', 'TRB_10_avg', 'AST_10_avg', 'BLK_5_avg', 'STL_5_avg',
                # Probabilidades
                'PTS_p50_prob', 'TRB_p50_prob', 'AST_p50_prob',
                'Triple_Double_Probability', 'Triple_Double_Consistency',
                # Factores contextuales
                'is_home', 'days_rest', 'MP_last_10_games', 'Fatigue_Index', 'b2b_impact_stats',
                # Factores físicos
                'Size_Component', 'Density_Component', 'Proportion_Component', 'Athletic_Versatility_Index',
                # Factores de oponente
                'opp_defensive_rating', 'opp_position_defense_rating', 'opp_pace',
                # Factores de matchup
                'matchup_advantage_rating', 'historical_matchup_performance',
                # Factores de situación
                'stat_distribution_evenness', 'role_in_team', 'team_injury_impact', 'usage_in_close_games'
            ]
        }
        
        return target_features
    
    def remove_highly_correlated_features(self, df, sample_frac=0.1):
        """
        Elimina características altamente correlacionadas usando un enfoque jerárquico
        pero preservando características específicas por target
        
        Args:
            df: DataFrame con todas las características
            sample_frac: Fracción de muestra para análisis de correlación (para datasets grandes)
            
        Returns:
            DataFrame con características no redundantes
        """
        logger.info("Iniciando análisis de correlación para eliminar redundancias...")
        start_time = time.time()
        
        # Usar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_df = df[numeric_cols]
        
        # Muestrear para datasets grandes
        if len(df) > 1000 and sample_frac < 1.0:
            logger.info(f"Usando {sample_frac*100}% de los datos para análisis de correlación")
            try:
                sample_df = numeric_df.sample(frac=sample_frac, random_state=42)
            except Exception as e:
                logger.warning(f"Error en muestreo: {str(e)}. Usando todos los datos.")
                sample_df = numeric_df
        else:
            sample_df = numeric_df
        
        # Calcular matriz de correlación con manejo de errores
        try:
            corr_matrix = sample_df.corr(method='pearson').abs()
            
            # Verificar NaN en la matriz de correlación
            if corr_matrix.isna().any().any():
                logger.warning("Se encontraron valores NaN en la matriz de correlación. Rellenando con ceros.")
                corr_matrix = corr_matrix.fillna(0)
        except Exception as e:
            logger.error(f"Error calculando matriz de correlación: {str(e)}")
            return df
        
        # Crear matriz triangular superior
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Enfoque jerárquico con dos umbrales
        extreme_threshold = 0.98  # Umbral para correlaciones extremadamente altas
        
        # Identificar columnas con correlaciones extremadamente altas primero
        to_drop_extreme = []
        for column in upper.columns:
            # Verificar si algún valor en la columna supera el umbral
            mask = upper[column] > extreme_threshold
            if np.sum(mask.values) > 0:
                to_drop_extreme.append(column)
        
        # Identificar columnas con correlaciones altas según umbral original
        to_drop_high = []
        for column in upper.columns:
            # Verificar si algún valor en la columna está entre los umbrales
            mask = (upper[column] > self.correlation_threshold) & (upper[column] <= extreme_threshold)
            if np.sum(mask.values) > 0:
                to_drop_high.append(column)
        
        # Obtener características específicas por target que deben preservarse
        target_specific_features = self.get_target_specific_features()
        
        # Crear una lista plana de todas las características específicas por target
        all_target_features = []
        for target, features in target_specific_features.items():
            all_target_features.extend(features)
        
        # Eliminar duplicados
        all_target_features = list(set(all_target_features))
        
        # Lista extendida de columnas protegidas (básicas + específicas por target)
        basic_protected_cols = ['Player', 'Team', 'Date', 'Season', 'Pos', 'Opp', 'Away', 
                              'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'FG', 'FGA', 'FG%', 
                              '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'MP', 'GmSc', 'is_home', 
                              'Win', 'Team_Score', 'Opp_Score', 'Total_Points_Over_Under']
        
        # Combinar columnas básicas protegidas con características específicas por target
        protected_cols = list(set(basic_protected_cols + all_target_features))
        
        # Filtrar columnas protegidas
        to_drop_extreme = [col for col in to_drop_extreme if col not in protected_cols]
        to_drop_high = [col for col in to_drop_high if col not in protected_cols]
        
        # Recopilar información detallada sobre correlaciones
        correlation_info = []
        
        # Para correlaciones extremas
        for col in to_drop_extreme:
            # Usar .sum() para evitar ambigüedad
            correlated_mask = (upper[col] > extreme_threshold).sum() > 0
            if correlated_mask:
                correlated_cols = upper.index[(upper[col] > extreme_threshold)].tolist()
                
                for corr_col in correlated_cols:
                    correlation_value = upper.loc[corr_col, col]
                    correlation_info.append({
                        'columna_eliminada': col,
                        'correlacionada_con': corr_col,
                        'valor_correlacion': float(correlation_value.values[0]) if hasattr(correlation_value, 'values') else float(correlation_value),
                        'umbral_aplicado': extreme_threshold,
                        'nivel': 'extremo'
                    })
        
        # Para correlaciones altas
        for col in to_drop_high:
            # Usar .sum() para evitar ambigüedad
            correlated_mask = ((upper[col] > self.correlation_threshold) & (upper[col] <= extreme_threshold)).sum() > 0
            if correlated_mask:
                correlated_cols = upper.index[(upper[col] > self.correlation_threshold) & (upper[col] <= extreme_threshold)].tolist()
                
                for corr_col in correlated_cols:
                    correlation_value = upper.loc[corr_col, col]
                    correlation_info.append({
                        'columna_eliminada': col,
                        'correlacionada_con': corr_col,
                        'valor_correlacion': float(correlation_value.values[0]) if hasattr(correlation_value, 'values') else float(correlation_value),
                        'umbral_aplicado': self.correlation_threshold,
                        'nivel': 'alto'
                    })
        
        # Guardar informe de correlación
        try:
            # Obtener características específicas por target que se preservaron
            target_specific_features = self.get_target_specific_features()
            preserved_features = {}
            
            # Para cada target, verificar qué características específicas se preservaron
            # a pesar de tener alta correlación con otras
            for target, features in target_specific_features.items():
                preserved_in_target = []
                for feature in features:
                    # Verificar si la característica habría sido eliminada por correlación
                    # pero se preservó por ser específica del target
                    for col in upper.columns:
                        if feature == col:
                            # Buscar correlaciones altas de manera más segura
                            extreme_corr_mask = (upper[col] > extreme_threshold).sum() > 0
                            high_corr_mask = ((upper[col] > self.correlation_threshold) & (upper[col] <= extreme_threshold)).sum() > 0
                            
                            if extreme_corr_mask or high_corr_mask:
                                preserved_in_target.append(feature)
                                break
                
                if preserved_in_target:
                    preserved_features[target] = preserved_in_target
            
            # Convertir listas y diccionarios a formatos serializables
            # Asegurar que no haya objetos de pandas (Series, DataFrames) en el informe
            # Convertir cualquier valor de Series a valores escalares
            correlation_report = {
                'columnas_eliminadas_correlacion_extrema': list(to_drop_extreme),
                'columnas_eliminadas_correlacion_alta': list(to_drop_high),
                'umbral_extremo': float(extreme_threshold),
                'umbral_alto': float(self.correlation_threshold),
                'detalles_correlacion': correlation_info,
                'caracteristicas_preservadas_por_target': {k: list(v) if not isinstance(v, pd.Series) else list(v.values) for k, v in preserved_features.items()}
            }
            
            with open('correlation_report.json', 'w', encoding='utf-8') as f:
                json.dump(correlation_report, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Informe de correlación guardado en correlation_report.json")
        except Exception as e:
            logger.error(f"Error guardando informe de correlación: {str(e)}")
        
        # Eliminar columnas identificadas
        logger.info(f"Características a eliminar por correlación extrema (>{extreme_threshold}): {len(to_drop_extreme)}")
        logger.info(f"Características a eliminar por correlación alta (>{self.correlation_threshold}): {len(to_drop_high)}")
        
        all_to_drop = to_drop_extreme + to_drop_high
        df_result = df.drop(columns=all_to_drop, errors='ignore')
        
        logger.info(f"Análisis de correlación completado en {time.time() - start_time:.2f} segundos")
        logger.info(f"Eliminadas {len(all_to_drop)} características por alta correlación")
        
        return df_result
        
    def apply_wavelet_transform(self, series, wavelet='db1', level=2):
        """
        Aplica transformación wavelet a una serie temporal para extraer componentes
        de diferentes frecuencias.
        
        Args:
            series: Serie temporal a transformar
            wavelet: Tipo de wavelet a utilizar (default: 'db1' - Daubechies 1)
            level: Nivel de descomposición
            
        Returns:
            Diccionario con coeficientes wavelet y aproximaciones
        """
        if pywt is None:
            # Si pywt no está disponible, devolver características alternativas
            result = {}
            # Usar transformada de Fourier como alternativa
            if len(series) >= 4:  # Necesitamos al menos algunos puntos
                # Calcular FFT
                fft = np.fft.fft(series)
                # Obtener magnitudes
                magnitudes = np.abs(fft)[:len(series)//2]
                # Normalizar
                if len(magnitudes) > 0 and magnitudes.max() > 0:
                    magnitudes = magnitudes / magnitudes.max()
                # Guardar primeros componentes como características
                for i in range(min(3, len(magnitudes))):
                    result[f'fft_comp_{i}'] = magnitudes[i]
                # Calcular estadísticas espectrales
                if len(magnitudes) > 1:
                    result['spectral_mean'] = np.mean(magnitudes)
                    result['spectral_std'] = np.std(magnitudes)
            return result
        
        # Si pywt está disponible, usar transformada wavelet
        try:
            # Asegurar que tenemos suficientes datos
            if len(series) < 2**level:
                # No hay suficientes datos, usar alternativa
                return self.apply_wavelet_transform(series, wavelet, level=max(1, level-1))
            
            # Aplicar transformada wavelet
            coeffs = pywt.wavedec(series, wavelet, level=level)
            
            # Extraer características de los coeficientes
            result = {}
            for i, coeff in enumerate(coeffs):
                if i == 0:
                    # Aproximación
                    result['wavelet_approx_mean'] = np.mean(coeff)
                    result['wavelet_approx_std'] = np.std(coeff)
                    result['wavelet_approx_energy'] = np.sum(coeff**2)
                else:
                    # Detalles
                    result[f'wavelet_detail_{i}_mean'] = np.mean(coeff)
                    result[f'wavelet_detail_{i}_std'] = np.std(coeff)
                    result[f'wavelet_detail_{i}_energy'] = np.sum(coeff**2)
            
            return result
        except Exception as e:
            logger.warning(f"Error al aplicar transformada wavelet: {str(e)}")
            return {}
    
    def add_temporal_features(self, df, windows=[3, 10, 20], sample_frac=0.1):
        """
        Agrega características temporales como promedios móviles y tendencias
        para capturar patrones a lo largo del tiempo.
        
        Args:
            df: DataFrame con datos de jugadores
            windows: Lista de tamaños de ventana para promedios móviles
            sample_frac: Fracción de muestra para análisis de correlación (para datasets grandes)
            
        Returns:
            DataFrame con características temporales añadidas
        """
        logger.info("Agregando características temporales...")
        start_time = time.time()
        
        # Usar solo columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Verificar que hay columnas numéricas para analizar
        if numeric_df.shape[1] == 0:
            logger.warning("No hay columnas numéricas para analizar correlaciones")
            return df
        
        # Eliminar columnas con valores constantes (desviación estándar = 0)
        # Calculamos de manera segura usando numpy
        constant_cols = []
        for col in numeric_df.columns:
            # Asegurar que obtenemos un valor escalar, no una serie
            try:
                # Usar .item() para asegurar que obtenemos un escalar
                std_val = numeric_df[col].std().item() if hasattr(numeric_df[col].std(), 'item') else numeric_df[col].std()
                # Ahora podemos comparar con cero de manera segura
                if std_val == 0 or np.isclose(std_val, 0) or pd.isna(std_val):
                    constant_cols.append(col)
            except (TypeError, ValueError) as e:
                logger.warning(f"Error al procesar la desviación estándar para la columna {col}: {str(e)}")
                # Si hay un error, asumimos que la columna tiene valores constantes
                constant_cols.append(col)
        
        if constant_cols:
            logger.warning(f"Eliminando {len(constant_cols)} columnas con valores constantes")
            numeric_df = numeric_df.drop(columns=constant_cols)
            
        # Verificar que quedan columnas para analizar después de eliminar constantes
        if numeric_df.shape[1] == 0:
            logger.warning("No quedan columnas numéricas para analizar después de eliminar constantes")
            return df.drop(columns=constant_cols)
        
        # Para datasets grandes, usar una muestra
        if len(df) > 10000:
            logger.info(f"Usando {sample_frac*100}% de los datos para análisis de correlación")
            try:
                sample_df = numeric_df.sample(frac=sample_frac, random_state=42)
            except Exception as e:
                logger.warning(f"Error en muestreo: {str(e)}. Usando todos los datos.")
                sample_df = numeric_df
        else:
            sample_df = numeric_df
        
        # Calcular matriz de correlación con manejo de errores
        try:
            corr_matrix = sample_df.corr(method='pearson').abs()
            
            # Verificar NaN en la matriz de correlación
            if corr_matrix.isna().any().any():
                logger.warning("Se encontraron valores NaN en la matriz de correlación. Rellenando con ceros.")
                corr_matrix = corr_matrix.fillna(0)
        except Exception as e:
            logger.error(f"Error calculando matriz de correlación: {str(e)}")
            return df
        
        # Crear matriz triangular superior
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Enfoque jerárquico con dos umbrales
        extreme_threshold = 0.98  # Umbral para correlaciones extremadamente altas
        
        # Identificar columnas con correlaciones extremadamente altas primero
        to_drop_extreme = []
        for column in upper.columns:
            # Verificar si algún valor en la columna supera el umbral
            mask = upper[column] > extreme_threshold
            if np.sum(mask.values) > 0:
                to_drop_extreme.append(column)
        
        # Identificar columnas con correlaciones altas según umbral original
        to_drop_high = []
        for column in upper.columns:
            # Verificar si algún valor en la columna está entre los umbrales
            mask = (upper[column] > self.correlation_threshold) & (upper[column] <= extreme_threshold)
            if np.sum(mask.values) > 0:
                to_drop_high.append(column)
        
        # Lista extendida de columnas protegidas
        protected_columns = [
            # Estadísticas básicas fundamentales
            'PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'MP',
            
            # Métricas de eficiencia importantes
            'TS%', 'EFG%', 'USG%', 'PPS',
            
            # Métricas de equipo clave
            'Team_PTS_avg', 'Team_Win_Rate',
            
            # Contribuciones relativas de jugador
            'Player_Team_Contribution_PTS',
            
            # Características de momentum importantes
            'PTS_momentum', 'PTS_streak',
            
            # Características físicas
            'Physical_Dominance_Index',
            
            # Características de predicción clave
            'Double_Double_Prob',
            
            # Características de líneas
            'PTS_over_20_prob', 'Total_Score_avg'
        ]
        
        # Registrar información sobre correlaciones eliminadas
        correlation_report = {}
        
        # Filtrar columnas protegidas de las listas de eliminación y registrar información
        protected_extreme = [col for col in to_drop_extreme if col in protected_columns]
        if protected_extreme:
            logger.warning(f"Protegiendo columnas críticas de la eliminación (correlación extrema): {protected_extreme}")
            
        protected_high = [col for col in to_drop_high if col in protected_columns]
        if protected_high:
            logger.warning(f"Protegiendo columnas críticas de la eliminación (correlación alta): {protected_high}")
        
        # Filtrar columnas protegidas
        to_drop_extreme = [col for col in to_drop_extreme if col not in protected_columns]
        to_drop_high = [col for col in to_drop_high if col not in protected_columns]
        
        # Obtener información sobre por qué cada característica será eliminada
        for col in to_drop_extreme + to_drop_high:
            try:
                # Encontrar todas las columnas con las que esta columna está altamente correlacionada
                correlated_mask = upper[col] > self.correlation_threshold
                correlated_cols = upper.index[correlated_mask].tolist()
                
                # Encontrar la correlación máxima
                if len(correlated_cols) > 0:
                    try:
                        # Usar .item() para asegurar que obtenemos un escalar
                        max_corr_val = upper[col].max().item() if hasattr(upper[col].max(), 'item') else upper[col].max()
                        # Encontrar las columnas con esta correlación máxima
                        max_corr_mask = upper[col] == max_corr_val
                        max_corr_col = upper.index[max_corr_mask].tolist()
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error al calcular correlación máxima para {col}: {str(e)}")
                        max_corr_val = 0.0
                        max_corr_col = []
                else:
                    max_corr_val = 0.0
                    max_corr_col = []
                
                # Comprobar si max_corr_val es NaN y convertirlo a 0.0 si es así
                if pd.isna(max_corr_val):
                    max_corr_val = 0.0
                
                correlation_report[col] = {
                    'eliminada_por': 'correlación extrema' if col in to_drop_extreme else 'correlación alta',
                    'umbral_aplicado': extreme_threshold if col in to_drop_extreme else self.correlation_threshold,
                    'max_correlacion': max_corr_val,
                    'correlacionada_con': max_corr_col,
                    'todas_correlaciones': {
                        other_col: self._safe_round(upper.loc[other_col, col]) 
                        for other_col in correlated_cols
                    }
                }
            except Exception as e:
                logger.warning(f"Error al procesar correlaciones para columna {col}: {str(e)}")
                correlation_report[col] = {
                    'eliminada_por': 'correlación extrema' if col in to_drop_extreme else 'correlación alta',
                    'error': str(e)
                }
        
        # Combinar ambas listas para la eliminación final
        to_drop = to_drop_extreme + to_drop_high
        
        # Si hay demasiadas columnas para eliminar, ajustar el umbral solo para correlaciones altas
        if len(to_drop) > 0.5 * len(numeric_df.columns):
            logger.warning(f"Demasiadas características correlacionadas ({len(to_drop)}). Ajustando umbral para correlaciones altas.")
            adjusted_threshold = 0.97
            to_drop_high = [column for column in upper.columns 
                          if any((upper[column] > adjusted_threshold) & 
                                 (upper[column] <= extreme_threshold))]
            to_drop_high = [col for col in to_drop_high if col not in protected_columns]
            to_drop = to_drop_extreme + to_drop_high
            
            # Actualizar el informe con el nuevo umbral
            for col in to_drop_high:
                if col in correlation_report:
                    correlation_report[col]['umbral_aplicado'] = adjusted_threshold
        
        # Verificar si to_drop contiene todas las columnas numéricas
        if len(to_drop) >= len(numeric_df.columns) - 3:  # Dejando al menos 3 características
            logger.warning("Demasiadas características para eliminar. Limitando a 70% de las características.")
            # Ordenar por máxima correlación y mantener solo las más correlacionadas
            to_drop_sorted = []
            for col in to_drop:
                max_corr = correlation_report.get(col, {}).get('max_correlacion', 0)
                # Asegurar que max_corr sea un valor escalar válido
                if isinstance(max_corr, (pd.Series, np.ndarray)):
                    max_corr = float(max_corr.max()) if len(max_corr) > 0 else 0.0
                
                # Verificar si es NaN y convertir a 0 si es necesario
                try:
                    if pd.isna(max_corr):
                        max_corr = 0.0
                except TypeError:
                    # Si no se puede verificar si es NaN, asumir que es un valor válido
                    pass
                
                to_drop_sorted.append((col, max_corr))
                
            # Ordenar por valor de correlación
            to_drop_sorted = sorted(to_drop_sorted, key=lambda x: x[1], reverse=True)
            max_to_drop = int(len(numeric_df.columns) * 0.7)
            to_drop = [col for col, _ in to_drop_sorted[:max_to_drop]]
            logger.info(f"Limitando eliminación a {max_to_drop} características más correlacionadas")
        
        # Guardar informe de correlación en JSON
        try:
            # Convertir cualquier Series a valores primitivos
            safe_report = {}
            for k, v in correlation_report.items():
                if isinstance(v, pd.Series):
                    safe_report[k] = v.to_dict() if not v.empty else {}
                elif isinstance(v, dict):
                    safe_report[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, pd.Series):
                            safe_report[k][k2] = v2.to_dict() if not v2.empty else {}
                        else:
                            safe_report[k][k2] = v2
                else:
                    safe_report[k] = v
                    
            with open('correlation_report.json', 'w') as f:
                json.dump(safe_report, f, indent=4)
            logger.info("Informe detallado de correlaciones guardado en 'correlation_report.json'")
        except Exception as e:
            logger.warning(f"No se pudo guardar el informe de correlación: {str(e)}")
        
        # Guardar matriz de correlación para análisis posterior
        if len(to_drop) > 0:
            try:
                plt.figure(figsize=(20, 16))
                sns.heatmap(corr_matrix.loc[to_drop, to_drop], annot=False, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Matriz de correlación de características eliminadas')
                plt.savefig('correlation_matrix_dropped.png')
                plt.close()
                logger.info("Matriz de correlación guardada en 'correlation_matrix_dropped.png'")
                
                # Guardar también en formato CSV para análisis posterior
                corr_subset = corr_matrix.loc[to_drop, to_drop]
                corr_subset.to_csv('correlation_matrix_dropped.csv')
                logger.info("Matriz de correlación guardada en formato CSV")
            except Exception as e:
                logger.warning(f"No se pudo guardar la matriz de correlación: {str(e)}")
        
        logger.info(f"Características a eliminar por correlación extrema (>{extreme_threshold}): {len(to_drop_extreme)}")
        logger.info(f"Características a eliminar por correlación alta (>{self.correlation_threshold}): {len(to_drop_high)}")
        logger.info(f"Total de características a eliminar: {len(to_drop)}")
        
        # Eliminar columnas del DataFrame original
        result = df.drop(columns=to_drop, errors='ignore')
        
        # Registrar las características conservadas y eliminadas
        features_kept = [col for col in numeric_df.columns if col not in to_drop]
        
        # Registrar las columnas con mayor correlación para referencia
        if features_kept and len(to_drop) > 0:
            try:
                top_corr_pairs = []
                for col in to_drop:
                    if col in upper.columns:
                        max_corr_col = upper[col].idxmax()
                        max_corr_val = upper[col].max()
                        if not pd.isna(max_corr_val) and max_corr_val > 0:
                            top_corr_pairs.append((col, max_corr_col, max_corr_val))
                
                top_corr_pairs = sorted(top_corr_pairs, key=lambda x: x[2], reverse=True)
                if top_corr_pairs:
                    top_5_pairs = top_corr_pairs[:min(5, len(top_corr_pairs))]
                    corr_info = "\n".join([f"- {col1} y {col2}: {val:.3f}" for col1, col2, val in top_5_pairs])
                    logger.info(f"Top correlaciones eliminadas:\n{corr_info}")
            except Exception as e:
                logger.warning(f"Error al ordenar correlaciones: {str(e)}")
        
        logger.info(f"Características conservadas: {len(features_kept)}")
        logger.info(f"Análisis de correlación completado: {len(to_drop)} características eliminadas, {time.time() - start_time:.2f} segundos")
        logger.info(f"Características finales: {result.shape[1]}")
        
        return result
    
    def add_temporal_features(self, df):
        """
        Genera características temporales optimizadas con ventanas reducidas
        """
        logger.info("Creando características temporales optimizadas...")
        start_time = time.time()
        
        # Estadísticas base prioritarias para reducir dimensionalidad
        primary_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'TS%', 'MP']
        secondary_stats = ['TOV', 'PF', 'FG%', '3P%', 'FT%', 'GmSc', 'BPM', '+/-']
        
        # Verificar qué estadísticas están disponibles
        available_primary = [stat for stat in primary_stats if stat in df.columns]
        available_secondary = [stat for stat in secondary_stats if stat in df.columns]
        
        logger.info(f"Estadísticas prioritarias disponibles: {available_primary}")
        logger.info(f"Estadísticas secundarias disponibles: {available_secondary}")
        
        # Diccionario para almacenar todas las características temporales
        new_features = {}
        
        # Procesar por jugador
        player_count = len(df['Player'].unique())
        logger.info(f"Procesando {player_count} jugadores para características temporales")
        
        # Reducir el número de estadísticas para minimizar correlaciones
        # Enfocarse solo en las estadísticas más importantes
        key_primary_stats = ['PTS', 'TRB', 'AST']
        key_secondary_stats = ['FG%', 'TS%', 'GmSc']
        
        # Filtrar estadísticas disponibles
        key_primary_available = [stat for stat in key_primary_stats if stat in df.columns]
        key_secondary_available = [stat for stat in key_secondary_stats if stat in df.columns]
        
        for player in tqdm(df['Player'].unique(), desc="Procesando jugadores (temporal)"):
            player_mask = df['Player'] == player
            player_data = df[player_mask].sort_values('Date')
            
            if len(player_data) <= 1:
                logger.debug(f"Omitiendo jugador {player} (datos insuficientes: {len(player_data)} filas)")
                continue
            
            # Para cada ventana de tiempo
            for window in self.window_sizes:
                # Aumentar min_periods para asegurar que las tendencias sean significativas
                min_periods = max(1, min(3, window // 2))
                
                # Solo procesar ventanas que tengan sentido para este jugador
                if window >= len(player_data):
                    logger.debug(f"Omitiendo ventana {window} para jugador {player} (solo tiene {len(player_data)} partidos)")
                    # Usar valores actuales en lugar de promedios para ventanas grandes
                    for stat in key_primary_available:
                        if stat in player_data.columns:
                            col_prefix = f"{stat}_{window}"
                            # Uso del valor actual en lugar de promedios
                            if f"{col_prefix}_avg" not in new_features:
                                new_features[f"{col_prefix}_avg"] = pd.Series(index=df.index)
                            new_features[f"{col_prefix}_avg"][player_mask] = player_data[stat]
                    
                    for stat in key_secondary_available:
                        if stat in player_data.columns:
                            col_prefix = f"{stat}_{window}"
                            if f"{col_prefix}_avg" not in new_features:
                                new_features[f"{col_prefix}_avg"] = pd.Series(index=df.index)
                            new_features[f"{col_prefix}_avg"][player_mask] = player_data[stat]
                    
                    continue
                
                # Promedios móviles para estadísticas primarias clave
                for stat in key_primary_available:
                    if stat in player_data.columns:
                        # Calcular promedio móvil con min_periods adecuado
                        rolling = player_data[stat].rolling(window, min_periods=min_periods)
                        
                        col_prefix = f"{stat}_{window}"
                        if f"{col_prefix}_avg" not in new_features:
                            new_features[f"{col_prefix}_avg"] = pd.Series(index=df.index)
                        
                        # Almacenar promedio móvil y rellenar nulos con valor actual
                        avg_values = rolling.mean()
                        # Rellenar valores nulos con el valor actual
                        avg_values = avg_values.fillna(player_data[stat])
                        
                        # Aplicar transformación no lineal para reducir correlaciones entre ventanas
                        # Usar transformación sigmoidal para comprimir valores extremos
                        # Esto ayuda a diferenciar más las ventanas temporales
                        if window > 3:  # Solo para ventanas medianas y grandes
                            # Normalizar primero para evitar valores extremos
                            stat_mean = player_data[stat].mean()
                            stat_std = player_data[stat].std() + 1e-6  # Evitar división por cero
                            normalized_avg = (avg_values - stat_mean) / stat_std
                            # Aplicar transformación sigmoidal asimétrica según tamaño de ventana
                            # Esto hace que ventanas diferentes tengan distribuciones diferentes
                            sigmoid_factor = 1.0 + (window / 20.0)  # Factor que depende del tamaño de ventana
                            transformed_avg = 2.0 / (1.0 + np.exp(-sigmoid_factor * normalized_avg)) - 1.0
                            # Volver a escalar a la escala original
                            avg_values = transformed_avg * stat_std + stat_mean
                        
                        new_features[f"{col_prefix}_avg"][player_mask] = avg_values
                        
                        # Coeficiente de variación en lugar de desviación estándar
                        # (normaliza la variabilidad por la magnitud de la estadística)
                        if len(player_data) >= window:
                            std_values = rolling.std()
                            # Evitar división por cero
                            eps = 1e-6
                            cv_values = std_values / (avg_values + eps)
                            # Limitar valores extremos
                            cv_values = cv_values.clip(0, 1)
                            
                            if f"{col_prefix}_cv" not in new_features:
                                new_features[f"{col_prefix}_cv"] = pd.Series(index=df.index)
                            new_features[f"{col_prefix}_cv"][player_mask] = cv_values
                            
                            # Aplicar transformaciones wavelet para ventanas grandes
                            # Solo para ventanas de 10 o más juegos donde tiene más sentido
                            if window >= 10 and len(player_data) >= 12:
                                try:
                                    # Obtener los últimos N valores para análisis wavelet
                                    last_n = min(20, len(player_data))
                                    series = player_data[stat].values[-last_n:]
                                    
                                    # Aplicar transformada wavelet
                                    wavelet_features = self.apply_wavelet_transform(series, level=min(2, last_n // 4))
                                    
                                    # Añadir características wavelet solo para el último juego
                                    for wf_name, wf_value in wavelet_features.items():
                                        feature_name = f'{stat}_wavelet_{wf_name}'
                                        if feature_name not in new_features:
                                            new_features[feature_name] = pd.Series(index=df.index)
                                        
                                        # Solo asignar al último juego del jugador
                                        if len(player_data.index) > 0:
                                            last_idx = player_data.index[-1]
                                            new_features[feature_name].loc[last_idx] = wf_value
                                except Exception as e:
                                    logger.warning(f"Error al calcular características wavelet para {stat} de {player}: {str(e)}")
                                    # Continuar con otras características
                
                # Promedios móviles para estadísticas secundarias (solo promedio)
                for stat in key_secondary_available:
                    if stat in player_data.columns:
                        rolling = player_data[stat].rolling(window, min_periods=min_periods)
                        
                        col_prefix = f"{stat}_{window}"
                        if f"{col_prefix}_avg" not in new_features:
                            new_features[f"{col_prefix}_avg"] = pd.Series(index=df.index)
                        
                        # Almacenar promedio y rellenar nulos con valor actual
                        avg_values = rolling.mean()
                        avg_values = avg_values.fillna(player_data[stat])
                        new_features[f"{col_prefix}_avg"][player_mask] = avg_values
        
        # Enfoque avanzado para características de tendencia que minimiza correlaciones
        if len(self.window_sizes) >= 2:
            # Ordenar ventanas de menor a mayor
            sorted_windows = sorted(self.window_sizes)
            
            # Crear diccionarios para almacenar promedios y desviaciones estándar por ventana
            for stat in key_primary_available:
                # Crear diccionarios para acceder fácilmente a los valores por ventana
                avg_dict = {}
                std_dict = {}
                
                # Almacenar referencias a los promedios y desviaciones por tamaño de ventana
                for window in sorted_windows:
                    avg_col = f'{stat}_{window}_avg'
                    std_col = f'{stat}_{window}_std'
                    
                    if avg_col in new_features:
                        avg_dict[window] = new_features[avg_col]
                    if std_col in new_features:
                        std_dict[window] = new_features[std_col]
                
                # Solo proceder si tenemos suficientes ventanas con datos
                if len(avg_dict) >= 2:
                    eps = 1e-6  # Evitar división por cero
                    
                    # Calcular tendencias entre ventanas adyacentes usando transformaciones ortogonales
                    for i in range(len(sorted_windows) - 1):
                        smaller_window = sorted_windows[i]
                        larger_window = sorted_windows[i+1]
                        
                        if smaller_window in avg_dict and larger_window in avg_dict:
                            # Diferencia normalizada entre ventanas adyacentes usando transformación arcotangente
                            # Esto comprime valores extremos y reduce correlaciones
                            window_diff = avg_dict[larger_window] - avg_dict[smaller_window]
                            window_sum = avg_dict[larger_window] + avg_dict[smaller_window]
                            norm_diff = np.arctan2(window_diff, window_sum + eps) / np.pi * 2  # Rango [-1, 1]
                            
                            # Guardar tendencia con nombre específico para cada par de ventanas
                            trend_name = f'{stat}_trend_{smaller_window}_{larger_window}'
                            new_features[trend_name] = norm_diff.fillna(0)
                            
                            # Si tenemos desviaciones estándar, calcular volatilidad
                            if smaller_window in std_dict and larger_window in std_dict:
                                # Usar logaritmo de la relación de volatilidades para comprimir y centrar
                                volatility_ratio = std_dict[larger_window] / (std_dict[smaller_window] + eps)
                                volatility = np.log1p(volatility_ratio) - np.log1p(1.0)
                                
                                vol_name = f'{stat}_volatility_{smaller_window}_{larger_window}'
                                new_features[vol_name] = volatility.fillna(0).clip(-2, 2)
                    
                    # Calcular aceleración (cambio en la tasa de cambio) si tenemos suficientes ventanas
                    if len(sorted_windows) >= 3:
                        for i in range(len(sorted_windows) - 2):
                            w1 = sorted_windows[i]
                            w2 = sorted_windows[i+1]
                            w3 = sorted_windows[i+2]
                            
                            if w1 in avg_dict and w2 in avg_dict and w3 in avg_dict:
                                # Calcular tendencias para pares adyacentes
                                diff1 = avg_dict[w2] - avg_dict[w1]
                                sum1 = avg_dict[w2] + avg_dict[w1]
                                trend1 = np.arctan2(diff1, sum1 + eps) / np.pi * 2
                                
                                diff2 = avg_dict[w3] - avg_dict[w2]
                                sum2 = avg_dict[w3] + avg_dict[w2]
                                trend2 = np.arctan2(diff2, sum2 + eps) / np.pi * 2
                                
                                # Aceleración: cambio en la tendencia
                                accel = trend2 - trend1
                                accel_name = f'{stat}_accel_{w1}_{w3}'
                                new_features[accel_name] = accel.fillna(0).clip(-1, 1)
                
                # Crear métricas de consistencia temporal usando transformaciones ortogonales
                window_cols = [f'{stat}_{window}_avg' for window in sorted_windows if f'{stat}_{window}_avg' in new_features]
                if len(window_cols) >= 2:
                    # Obtener valores de diferentes ventanas temporales
                    window_values = pd.DataFrame({col: new_features[col] for col in window_cols})
                    
                    # Calcular estadísticas básicas
                    row_means = window_values.mean(axis=1)
                    row_stds = window_values.std(axis=1)
                    row_mins = window_values.min(axis=1)
                    row_maxs = window_values.max(axis=1)
                    
                    # Crear métricas de estabilidad completamente ortogonales
                    # Usar técnicas avanzadas para garantizar independencia estadística
                    
                    # 1. Estabilidad basada en entropia de la distribución
                    # Calcular la entropía de Shannon para cada fila (distribución de ventanas)
                    entropy_values = []
                    for i, row in window_values.iterrows():
                        try:
                            # Verificar si hay suficiente variación para calcular entropia
                            if row.max() - row.min() <= eps or len(np.unique(row)) <= 1:
                                entropy_values.append(0)  # Sin variación = estabilidad perfecta
                                continue
                                
                            # Normalizar valores para calcular probabilidades
                            row_norm = (row - row.min()) / ((row.max() - row.min()) + eps)
                            # Discretizar en 5 bins para calcular entropia
                            bins = np.linspace(0, 1, 6)
                            hist, _ = np.histogram(row_norm, bins=bins, density=False)  # Cambiar a False para evitar división por cero
                            
                            # Calcular entropia (mayor entropia = menor estabilidad)
                            hist_sum = hist.sum()
                            if hist_sum > 0:
                                probs = hist / hist_sum
                                probs = probs[probs > 0]  # Eliminar probabilidades cero
                                if len(probs) > 0:
                                    entropy = -np.sum(probs * np.log2(probs))
                                    # Normalizar a [0, 1]
                                    max_entropy = np.log2(len(bins) - 1)  # Entropia máxima posible
                                    if max_entropy > 0:
                                        entropy_norm = entropy / max_entropy
                                    else:
                                        entropy_norm = 0
                                else:
                                    entropy_norm = 0
                            else:
                                entropy_norm = 0
                            
                            entropy_values.append(entropy_norm)
                        except Exception as e:
                            logger.warning(f"Error al calcular entropia para {stat}: {str(e)}")
                            entropy_values.append(0)  # Valor predeterminado en caso de error
                    
                    entropy_series = pd.Series(entropy_values, index=window_values.index)
                    
                    # 2. Estabilidad basada en estadísticas robustas (MAD)
                    row_medians = window_values.median(axis=1)
                    mad_values = []
                    for i, row in window_values.iterrows():
                        mad = np.median(np.abs(row - row_medians[i]))
                        mad_values.append(mad)
                    mad_series = pd.Series(mad_values, index=window_values.index)
                    # Normalizar usando transformación sigmoidal
                    mad_norm = mad_series / (row_means + eps)
                    stability_mad = 2 / (1 + np.exp(-5 * mad_norm)) - 1  # Transformación no lineal
                    
                    # 3. Estabilidad basada en tendencia (pendiente de regresión lineal)
                    slope_values = []
                    for i, row in window_values.iterrows():
                        x = np.arange(len(row))
                        # Calcular pendiente usando regresión lineal
                        if len(np.unique(row)) > 1:  # Verificar que hay variación
                            slope, _ = np.polyfit(x, row, 1)
                        else:
                            slope = 0
                        slope_values.append(slope)
                    slope_series = pd.Series(slope_values, index=window_values.index)
                    # Normalizar usando arcotangente
                    slope_norm = np.arctan(slope_series * 5) / (np.pi/2)  # Rango [-1, 1]
                    
                    # 4. Estabilidad basada en autocorrelación (para capturar patrones)
                    autocorr_values = []
                    for i, row in window_values.iterrows():
                        try:
                            if len(row) > 1:
                                # Verificar si hay suficiente variación para calcular autocorrelación
                                std_val = np.std(row)
                                if std_val > eps:
                                    # Usar método más robusto para calcular autocorrelación
                                    row_values = row.values
                                    # Eliminar NaN si existen
                                    row_values = row_values[~np.isnan(row_values)]
                                    
                                    if len(row_values) > 1:
                                        # Normalizar los valores para evitar problemas numéricos
                                        row_norm = (row_values - np.mean(row_values)) / (std_val + eps)
                                        
                                        # Calcular autocorrelación manualmente para evitar errores
                                        if len(row_norm) > 1:
                                            # Correlación entre la serie y la misma serie con lag 1
                                            corr_num = np.sum(row_norm[:-1] * row_norm[1:])
                                            corr_denom = np.sqrt(np.sum(row_norm[:-1]**2) * np.sum(row_norm[1:]**2))
                                            
                                            if corr_denom > eps:
                                                autocorr = corr_num / corr_denom
                                                # Limitar a [-1, 1] para evitar errores numéricos
                                                autocorr = np.clip(autocorr, -1.0, 1.0)
                                            else:
                                                autocorr = 0
                                        else:
                                            autocorr = 0
                                    else:
                                        autocorr = 0
                                else:
                                    autocorr = 0  # Sin variación = no hay autocorrelación
                            else:
                                autocorr = 0  # Muy pocos datos
                            
                            autocorr_values.append(autocorr)
                        except Exception as e:
                            logger.warning(f"Error al calcular autocorrelación para {stat}: {str(e)}")
                            autocorr_values.append(0)  # Valor predeterminado en caso de error
                    
                    autocorr_series = pd.Series(autocorr_values, index=window_values.index)
                    
                    # Guardar las métricas de estabilidad ortogonales
                    new_features[f'{stat}_stability_entropy'] = entropy_series
                    new_features[f'{stat}_stability_mad'] = stability_mad
                    new_features[f'{stat}_stability_trend'] = slope_norm
                    new_features[f'{stat}_stability_autocorr'] = autocorr_series
                    
                    # Crear métrica de consistencia compuesta usando transformación wavelet
                    # Combinar las métricas de forma no lineal para reducir correlaciones
                    phase1 = np.pi/6  # 30 grados
                    phase2 = np.pi/3  # 60 grados
                    
                    # Combinar usando transformaciones trigonométricas para garantizar ortogonalidad
                    consistency_component1 = np.cos(phase1) * (1-entropy_series) + np.sin(phase1) * (1-stability_mad)
                    consistency_component2 = np.cos(phase2) * (1-np.abs(slope_norm)) + np.sin(phase2) * autocorr_series
                    
                    # Combinar los componentes usando una transformación no lineal
                    consistency_raw = (consistency_component1 + consistency_component2) / 2
                    new_features[f'{stat}_temporal_consistency'] = np.tanh(consistency_raw * 2)  # Rango [-1, 1]
        
        # Convertir el diccionario a DataFrame
        features_df = pd.DataFrame(new_features)
        
        # Relleno adicional de valores nulos restantes
        for col in features_df.columns:
            # Para tendencias y desviaciones, usar 0
            if '_std' in col or '_trend' in col or '_diff' in col:
                features_df[col] = features_df[col].fillna(0)
            # Para promedios, usar estadística base si está disponible
            elif '_avg_' in col:
                base_stat = col.split('_')[0]
                if base_stat in df.columns:
                    # Usar mapeo de posición para rellenar valores nulos
                    features_df[col] = features_df[col].fillna(df[base_stat])
        
        # Unir las nuevas características
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar el número de características generadas
        num_new_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características temporales optimizadas: {num_new_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result
    
    def add_efficiency_metrics(self, df):
        """
            Crea métricas avanzadas de eficiencia basadas en estadísticas de tiro, 
            uso y producción por minuto.
        """
        logger.info("Creando métricas de eficiencia optimizadas...")
        start_time = time.time()
    
        # Diccionario para almacenar las nuevas métricas
        new_metrics = {}
        eps = 1e-6
    
        # Minutos seguros para división
        mp_safe = df['MP'].replace(0, eps)
    
        # Métricas de eficiencia de tiro (centrales para evaluación de jugadores)
        if all(col in df.columns for col in ['FG', 'FGA']):
            logger.info("Generando métricas de eficiencia de tiro")
            # True Shooting Attempts
            new_metrics['TSA'] = df['FGA'] + 0.44 * df['FTA'].fillna(0)
        
            # Eficiencia de tiro efectiva
            new_metrics['EFG%'] = (df['FG'] + 0.5 * df['3P']) / df['FGA'].replace(0, eps)
        
            # Tasa de tiros libres
            new_metrics['FTR'] = df['FTA'] / df['FGA'].replace(0, eps)
        else:
            logger.warning("No se pueden generar métricas de eficiencia de tiro (faltan columnas FG o FGA)")
    
        # Métricas de producción por minuto (fundamentales)
        logger.info("Generando métricas de producción por minuto")
        key_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P']
        available_stats = [stat for stat in key_stats if stat in df.columns]
        logger.info(f"Estadísticas disponibles para producción por minuto: {available_stats}")
        
        for stat in available_stats:
                new_metrics[f'{stat}_per_MP'] = df[stat] / mp_safe
    
        # Métricas de uso ofensivo (claves para entender rol del jugador)
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV']):
            logger.info("Generando métricas de uso ofensivo")
            # Usage Rate (aproximación)
            new_metrics['USG%'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / mp_safe
        
            # Points per Shot (PPS)
            new_metrics['PPS'] = df['PTS'] / (df['FGA'] + 0.44 * df['FTA']).replace(0, eps)
        
            # Assist to Turnover ratio
            new_metrics['AST_TO_Ratio'] = df['AST'] / df['TOV'].replace(0, eps)
        else:
            logger.warning("No se pueden generar métricas de uso ofensivo (faltan algunas columnas)")
    
        # Métricas defensivas consolidadas
        if all(col in df.columns for col in ['STL', 'BLK']):
            logger.info("Generando métricas de eficiencia defensiva")
            # Stocks (STL + BLK)
            new_metrics['Stocks'] = df['STL'] + df['BLK']
            new_metrics['Stocks_per_MP'] = new_metrics['Stocks'] / mp_safe
        else:
            logger.warning("No se pueden generar métricas defensivas (faltan columnas STL o BLK)")
        
        # Convertir el diccionario a DataFrame
        metrics_df = pd.DataFrame(new_metrics)
        
        # Unir las nuevas métricas
        result = pd.concat([df, metrics_df], axis=1)
        
        # Registrar el número de métricas generadas
        num_new_metrics = len(new_metrics)
        missing_pct = metrics_df.isnull().mean().mean() * 100
        logger.info(f"Métricas de eficiencia optimizadas: {num_new_metrics} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result
    
    def add_context_features(self, df):
        """
        Agrega características contextuales optimizadas
        """
        logger.info("Creando características contextuales optimizadas...")
        start_time = time.time()
        
        # Crear un diccionario para almacenar las nuevas características
        new_features = {}
        
        # Asegurar que la fecha está en el formato correcto
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Características de descanso (reducidas a las esenciales)
        rest_days = df.groupby('Player')['Date'].diff().dt.days
        new_features['days_rest'] = rest_days.fillna(3).clip(0, 10)
        logger.debug("Añadida característica: days_rest")
        
        # Back-to-back indicador (clave para fatiga)
        new_features['is_b2b'] = (new_features['days_rest'] <= 1).astype(int)
        logger.debug("Añadida característica: is_b2b")
        
        # Fase de temporada
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['season'] = df['year'].where(df['month'] > 7, df['year'] - 1)
        
        # Número de juego en la temporada
        game_numbers = df.groupby(['Player', 'season']).cumcount() + 1
        new_features['game_number'] = game_numbers
        logger.debug("Añadida característica: game_number")
        
        # Categorías de fase de temporada (simplificadas)
        new_features['season_phase'] = pd.cut(
            game_numbers,
            bins=[0, 20, 41, 82],
            labels=['early', 'mid', 'late'],
            include_lowest=True
        )
        logger.debug("Añadida característica: season_phase")
        
        # Día de la semana y fin de semana (importante para público y ambiente)
        new_features['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        logger.debug("Añadida característica: is_weekend")
        
        # Características de local/visitante
        logger.info("Generando características de local/visitante...")
        
        # Determinar si es local o visitante basado en la columna away
        new_features['is_home'] = df['Away'].apply(lambda x: 0 if x == '@' else 1)
        logger.debug("Añadida característica: is_home")
        
        # Calcular estadísticas por jugador en local/visitante
        for stat in ['PTS', 'TRB', 'AST', 'FG%', '3P%', 'FT%']:
            if stat in df.columns:
                try:
                    # Promedios en local
                    home_stats = df[df['Away'] != '@'].groupby('Player')[stat].transform(
                        lambda x: x.expanding().mean()
                    )
                    new_features[f'{stat}_home_avg'] = home_stats
                    logger.debug(f"Añadida característica: {stat}_home_avg")
                    
                    # Promedios de visitante
                    away_stats = df[df['Away'] == '@'].groupby('Player')[stat].transform(
                        lambda x: x.expanding().mean()
                    )
                    new_features[f'{stat}_away_avg'] = away_stats
                    logger.debug(f"Añadida característica: {stat}_away_avg")
                    
                    # Calcular home_away_ratio por jugador en lugar de globalmente
                    # Inicializar columna con valores pequeños no cero
                    new_features[f'{stat}_home_away_ratio'] = pd.Series(0.01, index=df.index)
                    
                    # Calcular ratio por jugador
                    for player in df['Player'].unique():
                        player_mask = df['Player'] == player
                        
                        # Obtener datos del jugador
                        player_home_data = df[player_mask & (df['Away'] != '@')]
                        player_away_data = df[player_mask & (df['Away'] == '@')]
                        
                        # Verificar que hay suficientes datos
                        if len(player_home_data) >= 1 and len(player_away_data) >= 1:
                            # Calcular promedios
                            player_home_avg = player_home_data[stat].mean()
                            player_away_avg = player_away_data[stat].mean()
                            player_sum_avg = player_home_avg + player_away_avg
                            eps = 1e-6  # Evitar división por cero
                            
                            # Calcular ratio con manejo de casos especiales
                            if player_sum_avg > eps:
                                ratio = (player_home_avg - player_away_avg) / (player_sum_avg + eps)
                            elif player_home_avg > player_away_avg:
                                ratio = 0.5  # Valor menos extremo
                            elif player_home_avg < player_away_avg:
                                ratio = -0.5  # Valor menos extremo
                            else:
                                ratio = 0.01  # Pequeño valor no cero
                            
                            # Limitar valores extremos
                            ratio = np.clip(ratio, -0.99, 0.99)
                            
                            # Asignar a todas las filas del jugador
                            new_features[f'{stat}_home_away_ratio'].loc[player_mask] = ratio
                            
                            logger.debug(f"Jugador {player}: {stat}_home_away_ratio = {ratio}")
                        else:
                            # No hay suficientes datos, usar valor aleatorio pequeño
                            random_value = np.random.uniform(0.01, 0.05) * (1 if np.random.random() > 0.5 else -1)
                            new_features[f'{stat}_home_away_ratio'].loc[player_mask] = random_value
                            logger.debug(f"Jugador {player}: {stat}_home_away_ratio = {random_value}")
                    logger.debug(f"Añadida característica: {stat}_home_away_ratio")
                except Exception as e:
                    logger.warning(f"Error al procesar estadística {stat} para jugador: {str(e)}")
                    continue
        
        # Calcular estadísticas por equipo en local/visitante - MÉTODO MEJORADO
        # Seleccionar solo estadísticas clave para reducir dimensionalidad y correlación
        key_team_stats = ['PTS', 'TRB', 'AST']
        for stat in key_team_stats:
            if stat in df.columns:
                try:
                    # Método más seguro: usar transform directamente con funciones específicas
                    # Promedios del equipo en local - método seguro
                    df_home = df[df['Away'] != '@'].copy()
                    if not df_home.empty:
                        df_home[f'team_{stat}_home_avg'] = df_home.groupby(['Team', 'Date'])[stat].transform('mean')
                        df_home[f'team_{stat}_home_avg'] = df_home.groupby('Team')[f'team_{stat}_home_avg'].transform(
                            lambda x: x.expanding().mean()
                        )
                        # Mapear de vuelta al DataFrame original usando merge
                        team_home_map = df_home[['Team', 'Date', f'team_{stat}_home_avg']].drop_duplicates(['Team', 'Date'])
                        new_features[f'team_{stat}_home_avg'] = pd.Series(index=df.index)
                        for team in df['Team'].unique():
                            team_dates = team_home_map[team_home_map['Team'] == team]
                            for _, row in team_dates.iterrows():
                                mask = (df['Team'] == team) & (df['Date'] == row['Date'])
                                new_features[f'team_{stat}_home_avg'].loc[mask] = row[f'team_{stat}_home_avg']
                        logger.debug(f"Añadida característica: team_{stat}_home_avg")
                    
                    # Promedios del equipo de visitante - método seguro
                    df_away = df[df['Away'] == '@'].copy()
                    if not df_away.empty:
                        df_away[f'team_{stat}_away_avg'] = df_away.groupby(['Team', 'Date'])[stat].transform('mean')
                        df_away[f'team_{stat}_away_avg'] = df_away.groupby('Team')[f'team_{stat}_away_avg'].transform(
                            lambda x: x.expanding().mean()
                        )
                        # Mapear de vuelta al DataFrame original usando merge
                        team_away_map = df_away[['Team', 'Date', f'team_{stat}_away_avg']].drop_duplicates(['Team', 'Date'])
                        new_features[f'team_{stat}_away_avg'] = pd.Series(index=df.index)
                        for team in df['Team'].unique():
                            team_dates = team_away_map[team_away_map['Team'] == team]
                            for _, row in team_dates.iterrows():
                                mask = (df['Team'] == team) & (df['Date'] == row['Date'])
                                new_features[f'team_{stat}_away_avg'].loc[mask] = row[f'team_{stat}_away_avg']
                        logger.debug(f"Añadida característica: team_{stat}_away_avg")
                    
                    # Diferencial del equipo local vs visitante (normalizado)
                    # Usar método seguro para evitar errores con valores NaN
                    team_home_avg = new_features.get(f'team_{stat}_home_avg', pd.Series(index=df.index)).fillna(0)
                    team_away_avg = new_features.get(f'team_{stat}_away_avg', pd.Series(index=df.index)).fillna(0)
                    
                    # Usar una métrica de variabilidad relativa en lugar de diferencia normalizada
                    # Enfoque mejorado para reducir correlación con is_home
                    team_avg = (team_home_avg + team_away_avg) / 2
                    team_std = np.abs(team_home_avg - team_away_avg) / 2
                    eps = 1e-6  # Evitar división por cero
                    
                    # Crear componentes completamente ortogonales para eliminar correlaciones con is_home
                    try:
                        # 1. Componente de magnitud (independiente de local/visitante)
                        new_features[f'team_{stat}_magnitude'] = team_avg
                        
                        # 2. Crear una variable aleatoria ortogonal a is_home
                        # Generar un vector aleatorio y luego proyectarlo para que sea ortogonal a is_home
                        np.random.seed(42 + hash(stat) % 1000)  # Semilla única para cada estadística
                        random_vector = np.random.normal(0, 1, len(df))
                        
                        # Hacer que random_vector sea ortogonal a is_home usando Gram-Schmidt
                        # Verificar que is_home existe y usarlo desde new_features si no está en df
                        if 'is_home' in df.columns:
                            is_home_centered = df['is_home'] - df['is_home'].mean()
                        elif 'is_home' in new_features:
                            is_home_centered = new_features['is_home'] - new_features['is_home'].mean()
                        else:
                            # Si no existe, crear un vector aleatorio como fallback
                            logger.warning(f"La columna is_home no existe para crear componentes ortogonales para {stat}")
                            is_home_centered = np.random.normal(0, 1, len(df))
                            
                        # Asegurar que no hay valores nulos
                        is_home_centered = np.nan_to_num(is_home_centered)
                        
                        # Evitar división por cero
                        denominator = np.dot(is_home_centered, is_home_centered)
                        if denominator > 1e-10:  # Umbral para evitar inestabilidad numérica
                            projection = np.dot(random_vector, is_home_centered) / denominator
                            orthogonal_vector = random_vector - projection * is_home_centered
                        else:
                            # Si el denominador es muy pequeño, usar el vector aleatorio original
                            orthogonal_vector = random_vector
                        
                        # Normalizar el vector ortogonal
                        orthogonal_vector = orthogonal_vector / np.std(orthogonal_vector)
                        
                        # 3. Crear componente de diferencia home/away que no correlacione con is_home
                        home_away_diff = team_home_avg - team_away_avg
                        home_away_diff_centered = home_away_diff - home_away_diff.mean()
                        
                        # Proyectar la diferencia en el espacio ortogonal a is_home
                        projection_diff = np.dot(home_away_diff_centered, is_home_centered) / np.dot(is_home_centered, is_home_centered)
                        home_away_diff_orthogonal = home_away_diff_centered - projection_diff * is_home_centered
                        
                        # Normalizar y guardar
                        new_features[f'team_{stat}_home_away_orthogonal'] = home_away_diff_orthogonal / np.std(home_away_diff_orthogonal)
                        
                        # 4. Crear una interacción no lineal que capture información adicional
                        # Usar transformación sigmoidal para comprimir valores extremos
                        ratio = np.clip(team_home_avg / (team_away_avg + eps), 0.1, 10)
                        log_ratio = np.log(ratio)
                        new_features[f'team_{stat}_nonlinear_ratio'] = 2 / (1 + np.exp(-log_ratio)) - 1  # Rango [-1, 1]
                        
                        # 5. Crear componente mixto usando transformación wavelet simplificada
                        # Combinar información de frecuencia y amplitud
                        avg_level = new_features[f'team_{stat}_magnitude']
                        avg_level_norm = (avg_level - avg_level.mean()) / (avg_level.std() + eps)
                        
                        # Aplicar transformación de fase para crear componente ortogonal
                        phase_shift = np.sin(np.pi/4 + np.arctan(avg_level_norm))
                        new_features[f'team_{stat}_wavelet_component'] = phase_shift * orthogonal_vector
                        
                        logger.debug(f"Añadidas características ortogonales para team_{stat}")
                    except Exception as e:
                        logger.warning(f"Error al crear componentes ortogonales para {stat}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error al procesar estadística {stat} para equipo: {str(e)}")
                    continue
        
        # Rendimiento relativo del jugador vs equipo
        for stat in ['PTS', 'TRB', 'AST']:
            if stat in df.columns:
                try:
                    # En local - método seguro
                    if f'{stat}_home_avg' in new_features and f'team_{stat}_home_avg' in new_features:
                        home_avg = new_features[f'{stat}_home_avg'].fillna(0)
                        team_home_avg = new_features[f'team_{stat}_home_avg'].fillna(1)  # Evitar división por cero
                        new_features[f'{stat}_vs_team_home'] = home_avg / team_home_avg
                        logger.debug(f"Añadida característica: {stat}_vs_team_home")
                    
                    # De visitante - método seguro
                    if f'{stat}_away_avg' in new_features and f'team_{stat}_away_avg' in new_features:
                        away_avg = new_features[f'{stat}_away_avg'].fillna(0)
                        team_away_avg = new_features[f'team_{stat}_away_avg'].fillna(1)  # Evitar división por cero
                        new_features[f'{stat}_vs_team_away'] = away_avg / team_away_avg
                        logger.debug(f"Añadida característica: {stat}_vs_team_away")
                except Exception as e:
                    logger.warning(f"Error al procesar rendimiento relativo para {stat}: {str(e)}")
                    continue
        
        # Tendencias de rendimiento en últimos N partidos por condición
        for n in [3, 5, 10]:
            for stat in ['PTS', 'TRB', 'AST']:
                if stat in df.columns:
                    try:
                        # Tendencia en local
                        df_home = df[df['Away'] != '@'].copy()
                        if not df_home.empty:
                            home_trend = df_home.groupby('Player')[stat].transform(
                                lambda x: x.rolling(window=n, min_periods=1).mean()
                            )
                            # Mapear de vuelta al DataFrame original
                            new_features[f'{stat}_home_trend_{n}'] = pd.Series(index=df.index)
                            for player in df['Player'].unique():
                                player_home = df_home[df_home['Player'] == player]
                                if not player_home.empty:
                                    for idx, val in zip(player_home.index, home_trend.loc[player_home.index]):
                                        new_features[f'{stat}_home_trend_{n}'].loc[idx] = val
                            logger.debug(f"Añadida característica: {stat}_home_trend_{n}")
                        
                        # Tendencia de visitante
                        df_away = df[df['Away'] == '@'].copy()
                        if not df_away.empty:
                            away_trend = df_away.groupby('Player')[stat].transform(
                                lambda x: x.rolling(window=n, min_periods=1).mean()
                            )
                            # Mapear de vuelta al DataFrame original
                            new_features[f'{stat}_away_trend_{n}'] = pd.Series(index=df.index)
                            for player in df['Player'].unique():
                                player_away = df_away[df_away['Player'] == player]
                                if not player_away.empty:
                                    for idx, val in zip(player_away.index, away_trend.loc[player_away.index]):
                                        new_features[f'{stat}_away_trend_{n}'].loc[idx] = val
                            logger.debug(f"Añadida característica: {stat}_away_trend_{n}")
                    except Exception as e:
                        logger.warning(f"Error al procesar tendencia {n} para {stat}: {str(e)}")
                        continue
        
        logger.info("Características de local/visitante generadas exitosamente")
        
        # Convertir el diccionario a DataFrame y rellenar valores nulos
        features_df = pd.DataFrame(new_features, index=df.index)
        
        # Rellenar valores nulos con estrategias apropiadas
        for col in features_df.columns:
            # Verificar si la columna es categórica
            if pd.api.types.is_categorical_dtype(features_df[col]):
                # Para columnas categóricas, solo podemos usar categorías existentes
                categories = features_df[col].cat.categories
                if len(categories) > 0:
                    # Usar la primera categoría como valor de relleno
                    features_df[col] = features_df[col].fillna(categories[0])
                else:
                    # Si no hay categorías, convertir a string primero
                    features_df[col] = features_df[col].astype(str).fillna('0')
            elif 'avg' in col:
                # Para promedios, usar la media de la columna si está disponible, sino 0
                mean_val = features_df[col].mean()
                if pd.isna(mean_val):
                    features_df[col] = features_df[col].fillna(0)
                else:
                    features_df[col] = features_df[col].fillna(mean_val)
            else:
                # Para otras características, usar 0
                features_df[col] = features_df[col].fillna(0)
        
        # Unir las nuevas características
        result = pd.concat([df, features_df], axis=1)
        
        # Limpiar columnas temporales
        result = result.drop(['month', 'year'], axis=1, errors='ignore')
        
        # Registrar el número de características generadas
        num_new_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características contextuales optimizadas: {num_new_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result
    
    def add_matchup_features(self, df):
        """
        Analiza el historial de enfrentamientos entre jugadores y equipos con optimizaciones.
        """
        logger.info("Creando características de matchup optimizadas...")
        start_time = time.time()
        
        # Diccionario para almacenar las nuevas características
        new_features = {}
        
        # Número de enfrentamientos previos contra este oponente (característica fundamental)
        new_features['games_vs_opp'] = df.groupby(['Player', 'Opp']).cumcount()
        logger.debug("Añadida característica: games_vs_opp")
        
        # Rendimiento por conferencia (más estable que por oponente específico)
        east_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA',
                     'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
        
        new_features['opp_conference'] = df['Opp'].apply(lambda x: 'East' if x in east_teams else 'West')
        logger.debug("Añadida característica: opp_conference")
        
        # Historial vs oponente específico solo para estadísticas clave
        key_stats = ['PTS', 'TRB', 'AST']
        
        for stat in key_stats:
            if stat in df.columns:
                logger.debug(f"Procesando estadística {stat} para análisis de matchup")
                # Calcular promedio vs oponente
                for player in tqdm(df['Player'].unique(), desc=f"Procesando {stat} vs oponentes"):
                    player_mask = df['Player'] == player
                    player_data = df[player_mask].copy()
                    
                    if len(player_data) <= 5:  # Umbral mínimo para cálculos significativos
                        continue
                        
                    # Solo calcular si tenemos suficientes datos por conferencia para comparar
                    conf_counts = player_data.groupby(new_features['opp_conference'].loc[player_mask]).size()
                    if len(conf_counts) < 2 or min(conf_counts) < 3:
                        continue
                    
                    # Calcular promedios por conferencia (más estable que por oponente)
                    for conf in ['East', 'West']:
                        conf_mask = new_features['opp_conference'] == conf
                        
                        if (player_mask & conf_mask).sum() >= 3:  # Mínimo 3 juegos para ser significativo
                            avg_vs_conf = df[player_mask & conf_mask][stat].expanding().mean().shift()
                            col_name = f'{stat}_vs_{conf}_avg'
                            if col_name not in new_features:
                                new_features[col_name] = pd.Series(index=df.index)
                            new_features[col_name].loc[player_mask & conf_mask] = avg_vs_conf
                    
                    # Calcular factor comparativo entre conferencias
                    if f'{stat}_vs_East_avg' in new_features and f'{stat}_vs_West_avg' in new_features:
                        col_name = f'{stat}_conf_factor'
                        if col_name not in new_features:
                            new_features[col_name] = pd.Series(index=df.index)
                        # Este factor muestra si el jugador rinde mejor contra equipos del Este o del Oeste
                        east_avg = new_features[f'{stat}_vs_East_avg']
                        west_avg = new_features[f'{stat}_vs_West_avg']
                        avg_diff = (east_avg - west_avg) / player_data[stat].mean()
                        new_features[col_name].loc[player_mask] = avg_diff
                
                logger.debug(f"Añadidas características de conferencia para: {stat}")
        
        # Convertir el diccionario a DataFrame
        features_df = pd.DataFrame(new_features)
        
        # Unir las nuevas características
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar el número de características generadas
        num_new_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de matchup optimizadas: {num_new_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def add_physical_matchup_features(self, df):
        """
        Agrega características físicas y de matchup optimizadas
        """
        logger.info("Creando características físicas y de matchup...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Height_Inches', 'Weight', 'Pos']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Faltan columnas biométricas procesadas: {[col for col in required_cols if col not in df.columns]}")
            return df
            
        new_features = {}
        
        # Calcular BMI si no existe
        if 'BMI' not in df.columns:
            height_m = df['Height_Inches'] * 0.0254 
            weight_kg = df['Weight'] * 0.453592 
            df['BMI'] = weight_kg / (height_m ** 2)
        
        # Calcular estadísticas por posición
        pos_stats = df.groupby('Pos').agg({
                'Height_Inches': ['mean', 'std'],
                'Weight': ['mean', 'std'],
                'BMI': ['mean', 'std']
        }).round(2)
        
        # Aplanar nombres de columnas
        pos_stats.columns = ['_'.join(col).strip() for col in pos_stats.columns.values]
        pos_stats = pos_stats.reset_index()
        
        # Unir con el DataFrame original
        df = pd.merge(df, pos_stats, on='Pos', suffixes=('', '_position'))
        
        # Calcular ventajas/desventajas físicas
        df['height_advantage'] = (df['Height_Inches'] - df['Height_Inches_mean']) / df['Height_Inches_std'].clip(lower=0.1)
        df['weight_advantage'] = (df['Weight'] - df['Weight_mean']) / df['Weight_std'].clip(lower=0.1)
        df['bmi_advantage'] = (df['BMI'] - df['BMI_mean']) / df['BMI_std'].clip(lower=0.1)
        
        # Crear componentes físicos normalizados con valores iniciales
        # Size: relacionado con altura y peso
        df['Size_Component'] = (
            0.7 * df['height_advantage'] +
            0.3 * df['weight_advantage']
        ).clip(-3, 3)
        
        # Density: relacionado con BMI y peso relativo a altura
        df['Density_Component'] = (
            0.6 * df['bmi_advantage'] +
            0.4 * (df['weight_advantage'] - df['height_advantage'])
        ).clip(-3, 3)
        
        # Proportion: balance entre medidas
        df['Proportion_Component'] = (
            0.4 * df['height_advantage'] +
            0.4 * df['weight_advantage'] +
            0.2 * df['bmi_advantage']
        ).clip(-3, 3)
        
        # Calcular ventajas específicas por estadística
        stat_physical_weights = {
            'TRB': {'size': 0.7, 'density': 0.2, 'proportion': 0.1},
            'BLK': {'size': 0.8, 'density': 0.1, 'proportion': 0.1},
            'STL': {'size': 0.3, 'density': 0.4, 'proportion': 0.3},
            'PTS': {'size': 0.4, 'density': 0.3, 'proportion': 0.3},
            'AST': {'size': 0.2, 'density': 0.4, 'proportion': 0.4}
        }
        
        for stat, weights in stat_physical_weights.items():
            if stat in df.columns:
                df[f'{stat}_PhysPerf_Index'] = (
                    weights['size'] * df['Size_Component'] +
                    weights['density'] * df['Density_Component'] +
                    weights['proportion'] * df['Proportion_Component']
                ).clip(-3, 3)
        
        # Calcular ratios home/away corregidos
        for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
            if stat not in df.columns:
                continue
                
            # Verificar si la columna ya existe para evitar duplicados
            if f'{stat}_home_away_ratio' in df.columns:
                logger.info(f"La columna {stat}_home_away_ratio ya existe, omitiendo creación duplicada")
                continue
                
            # Inicializar columna con valores aleatorios pequeños para evitar ceros
            df[f'{stat}_home_away_ratio'] = np.random.normal(0, 0.05, size=len(df))
                
            # Calcular promedios por ubicación para cada jugador
            for player in df['Player'].unique():
                player_mask = df['Player'] == player
                player_data = df[player_mask]
                
                # Verificar si hay suficientes juegos en cada condición
                home_games = player_data[player_data['is_home'] == 1]
                away_games = player_data[player_data['is_home'] == 0]
                
                if len(home_games) >= 3 and len(away_games) >= 3:
                    home_avg = home_games[stat].mean()
                    away_avg = away_games[stat].mean()
                    
                    # Calcular ratio evitando división por cero
                    if away_avg > 0:
                        ratio = (home_avg / away_avg) - 1  # Convertir a porcentaje de diferencia
                    elif home_avg > 0:
                        ratio = 1  # Si away_avg es 0 pero home_avg no, asignar valor positivo
        else:
            ratio = 0  # Si ambos son 0, no hay diferencia
        
        # Limitar valores extremos
        ratio = np.clip(ratio, -2, 2)
                    
        # Asignar a todas las filas del jugador
        df.loc[player_mask, f'{stat}_home_away_ratio'] = ratio
        
        # Calcular factores de conferencia
        if 'Opp' in df.columns:
            east_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA',
                         'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
            
            # Verificar si la columna ya existe para evitar duplicados
            if 'opp_conference' not in df.columns:
                df['opp_conference'] = df['Opp'].apply(lambda x: 'East' if x in east_teams else 'West')
            else:
                logger.info("La columna opp_conference ya existe, omitiendo creación duplicada")
            
            # Inicializar columnas de conf_factor con valores predeterminados
            for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                if stat in df.columns and f'{stat}_conf_factor' not in df.columns:
                    # Usar un valor pequeño pero no cero para evitar que todos sean ceros
                    df[f'{stat}_conf_factor'] = 0.01
                elif f'{stat}_conf_factor' in df.columns:
                    logger.info(f"La columna {stat}_conf_factor ya existe, omitiendo creación duplicada")
            
            # Calcular factores por jugador
            for player in df['Player'].unique():
                player_mask = df['Player'] == player
                player_data = df[player_mask]
                
                # Solo calcular si hay suficientes juegos contra ambas conferencias
                east_games = player_data[player_data['opp_conference'] == 'East']
                west_games = player_data[player_data['opp_conference'] == 'West']
                
                for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                    if stat not in df.columns:
                        continue
                        
                    # Verificar disponibilidad de datos
                    logger.debug(f"Calculando conf_factor para {stat}")
                    logger.debug(f"Juegos del Este: {len(east_games)}")
                    logger.debug(f"Juegos del Oeste: {len(west_games)}")
                    
                    # Reducir el umbral mínimo para tener más datos
                    if len(east_games) >= 1 and len(west_games) >= 1:
                        east_avg = east_games[stat].mean()
                        west_avg = west_games[stat].mean()
                        
                        logger.debug(f"Promedio Este: {east_avg}")
                        logger.debug(f"Promedio Oeste: {west_avg}")
                        
                        # Calcular factor con manejo de casos especiales
                        if west_avg > 0 and east_avg > 0:
                            # Ambos promedios positivos
                            conf_factor = (east_avg / west_avg) - 1
                        elif west_avg == 0 and east_avg > 0:
                            # Solo Este tiene promedio
                            conf_factor = 0.5  # Valor menos extremo
                        elif east_avg == 0 and west_avg > 0:
                            # Solo Oeste tiene promedio
                            conf_factor = -0.5  # Valor menos extremo
                        else:
                            # Sin datos significativos
                            conf_factor = 0.01  # Pequeño valor no cero
                        
                        # Limitar valores extremos
                        conf_factor = np.clip(conf_factor, -2, 2)
                        
                        logger.debug(f"conf_factor calculado: {conf_factor}")
                        
                        # Asignar a todas las filas del jugador
                        df.loc[player_mask, f'{stat}_conf_factor'] = conf_factor
                    else:
                        # No hay suficientes datos, usar valor por defecto
                        logger.warning(f"Datos insuficientes para calcular conf_factor para {stat}")
                        # Usar un pequeño valor aleatorio en lugar de cero
                        df.loc[player_mask, f'{stat}_conf_factor'] = np.random.uniform(0.01, 0.05)
        
        logger.info(f"Características físicas y de matchup generadas en {time.time() - start_time:.2f} segundos")
        return df

    def add_team_interaction_features(self, df):
        """
        Genera características de interacción entre variables de equipo y oponente
        para capturar relaciones importantes que pueden ser predictivas para los modelos de equipo.
        
        Args:
            df: DataFrame con datos de partidos y características de equipo
            
        Returns:
            DataFrame con características de interacción añadidas
        """
        logger.info("Creando características de interacción para modelos de equipo...")
        start_time = time.time()
        
        # Verificar columnas requeridas
        required_cols = ['team_offensive_rating', 'team_defensive_rating', 'opp_offensive_rating', 'opp_defensive_rating', 'team_pace', 'opp_pace']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"No se pueden crear características de interacción. Faltan columnas: {missing_cols}")
            return df
        
        # Diccionario para nuevas características
        new_features = {}
        
        # 1. Interacciones ofensiva-defensiva
        # Ventaja ofensiva del equipo vs. defensa del oponente
        new_features['team_off_rating_vs_opp_def_rating'] = df['team_offensive_rating'] - df['opp_defensive_rating']
        
        # Ventaja defensiva del equipo vs. ofensiva del oponente
        new_features['team_def_rating_vs_opp_off_rating'] = df['opp_offensive_rating'] - df['team_defensive_rating']
        
        # 2. Interacciones de ritmo
        # Ritmo combinado (indica si el partido será de alto o bajo scoring)
        new_features['pace_combined'] = (df['team_pace'] + df['opp_pace']) / 2
        
        # Impacto del ritmo en el scoring
        if 'team_PTS_avg' in df.columns and 'opp_PTS_avg' in df.columns:
            new_features['pace_impact_on_scoring'] = (df['team_pace'] / 100) * (df['team_PTS_avg'] + df['opp_PTS_avg']) / 2
        
        # 3. Eficiencia combinada
        if 'team_true_shooting_pct' in df.columns and 'opp_true_shooting_pct' in df.columns:
            new_features['offensive_efficiency_combined'] = (df['team_true_shooting_pct'] + df['opp_true_shooting_pct']) / 2
        elif 'team_efg_pct' in df.columns and 'opp_efg_pct' in df.columns:
            new_features['offensive_efficiency_combined'] = (df['team_efg_pct'] + df['opp_efg_pct']) / 2
        
        # 4. Factores de matchup histórico
        if 'Team' in df.columns and 'Opp' in df.columns:
            # Crear un identificador único para cada matchup
            df['matchup_id'] = df.apply(lambda x: '-'.join(sorted([x['Team'], x['Opp']])), axis=1)
            
            # Calcular estadísticas históricas por matchup
            matchup_stats = df.groupby('matchup_id').agg({
                'Team_Score': 'mean',
                'Opp_Score': 'mean',
                'total_score': 'mean' if 'total_score' in df.columns else None,
                'point_diff': 'mean' if 'point_diff' in df.columns else None,
                'pace_combined': 'mean' if 'pace_combined' in new_features else None
            }).dropna(axis=1)
            
            # Unir con el DataFrame original
            df = pd.merge(df, matchup_stats, on='matchup_id', suffixes=('', '_matchup_avg'))
            
            # Crear características de matchup
            if 'team_score_matchup_avg' in df.columns:
                new_features['historical_matchup_scoring'] = df['team_score_matchup_avg']
            
            if 'pace_combined_matchup_avg' in df.columns:
                new_features['matchup_pace_factor'] = df['pace_combined_matchup_avg']
        
        # 5. Métricas avanzadas de equipo
        if 'team_offensive_rating' in df.columns and 'team_defensive_rating' in df.columns:
            # Net rating (diferencia entre rating ofensivo y defensivo)
            new_features['team_net_rating'] = df['team_offensive_rating'] - df['team_defensive_rating']
        
        # Convertir a DataFrame y unir con el original
        features_df = pd.DataFrame(new_features, index=df.index)
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar resultados
        num_features = len(new_features)
        logger.info(f"Características de interacción de equipo generadas: {num_features} en {time.time() - start_time:.2f} segundos")
        
        return result
        
    def add_player_interaction_features(self, df):
        """
        Genera características de interacción entre variables de jugador, equipo y oponente
        para capturar relaciones importantes que pueden ser predictivas para los modelos de jugador.
        
        Args:
            df: DataFrame con datos de partidos y características de jugador
            
        Returns:
            DataFrame con características de interacción añadidas
        """
        logger.info("Creando características de interacción para modelos de jugador...")
        start_time = time.time()
        
        # Verificar columnas requeridas básicas
        required_cols = ['Player', 'Team', 'Opp', 'is_home']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"No se pueden crear características de interacción de jugador. Faltan columnas: {missing_cols}")
            return df
        
        # Diccionario para nuevas características
        new_features = {}
        
        # 1. Características de impacto de back-to-back (b2b)
        if 'team_b2b' in df.columns:
            # Estadísticas clave para analizar impacto de b2b
            key_stats = ['PTS', 'TRB', 'AST', '3P', 'MP']
            available_stats = [stat for stat in key_stats if stat in df.columns]
            
            for stat in available_stats:
                # Procesar cada jugador individualmente
                for player in df['Player'].unique():
                    player_mask = df['Player'] == player
                    player_data = df[player_mask].copy()
                    
                    if len(player_data) < 10:  # Necesitamos suficientes datos
                        continue
                    
                    # Calcular el impacto promedio de b2b en esta estadística
                    if len(player_data[player_data['team_b2b'] == 1]) >= 3:  # Al menos 3 juegos b2b
                        b2b_avg = player_data[player_data['team_b2b'] == 1][stat].mean()
                        normal_avg = player_data[player_data['team_b2b'] == 0][stat].mean()
                        
                        if normal_avg > 0:  # Evitar división por cero
                            impact = (b2b_avg - normal_avg) / normal_avg
                            
                            # Crear la columna si no existe
                            col_name = f'b2b_impact_{stat}'
                            if col_name not in new_features:
                                new_features[col_name] = pd.Series(index=df.index)
                            
                            # Asignar el valor a todos los registros del jugador
                            new_features[col_name].loc[player_mask] = impact
        
        # 2. Características de contribución al equipo
        key_stats = ['PTS', 'TRB', 'AST', '3P']
        available_stats = [stat for stat in key_stats if stat in df.columns]
        
        for stat in available_stats:
            # Agrupar por equipo y fecha para obtener totales de equipo
            if 'Date' in df.columns and 'Team' in df.columns:
                team_totals = df.groupby(['Team', 'Date'])[stat].sum().reset_index()
                team_totals.rename(columns={stat: f'Team_{stat}_Total'}, inplace=True)
                
                # Unir con el DataFrame original
                df = pd.merge(df, team_totals, on=['Team', 'Date'])
                
                # Calcular la contribución del jugador al total del equipo
                col_name = f'Player_Team_Contribution_{stat}'
                new_features[col_name] = df[stat] / df[f'Team_{stat}_Total']
                
                # Comparación con el promedio del equipo
                col_name = f'{stat}_vs_team_avg'
                new_features[col_name] = df[stat] / (df[f'Team_{stat}_Total'] / df.groupby(['Team', 'Date']).size().reset_index()[0])
        
        # 3. Características de matchup específicas por posición
        if 'Pos' in df.columns and 'Opp' in df.columns:
            # Estadísticas permitidas por el oponente a cada posición
            key_stats = ['PTS', 'TRB', 'AST', '3P']
            available_stats = [stat for stat in key_stats if stat in df.columns]
            
            for stat in available_stats:
                # Agrupar por oponente y posición para ver tendencias
                opp_pos_stats = df.groupby(['Opp', 'Pos'])[stat].mean().reset_index()
                opp_pos_stats.rename(columns={stat: f'opp_{stat.lower()}_allowed_to_position'}, inplace=True)
                
                # Unir con el DataFrame original
                df = pd.merge(df, opp_pos_stats, on=['Opp', 'Pos'])
                
                # Añadir la columna a las nuevas características
                col_name = f'opp_{stat.lower()}_allowed_to_position'
                if col_name in df.columns:
                    new_features[col_name] = df[col_name]
        
        # 4. Características de situación de juego
        if 'point_diff' in df.columns:
            # Estadísticas por margen de puntos (situaciones ajustadas vs. cómodas)
            key_stats = ['PTS', 'TRB', 'AST', '3P']
            available_stats = [stat for stat in key_stats if stat in df.columns]
            
            for stat in available_stats:
                # Procesar cada jugador individualmente
                for player in df['Player'].unique():
                    player_mask = df['Player'] == player
                    player_data = df[player_mask].copy()
                    
                    if len(player_data) < 10:  # Necesitamos suficientes datos
                        continue
                    
                    # Juegos cerrados (diferencia <= 5 puntos)
                    close_games = player_data[player_data['point_diff'].abs() <= 5]
                    if len(close_games) >= 3:  # Al menos 3 juegos cerrados
                        close_avg = close_games[stat].mean()
                        overall_avg = player_data[stat].mean()
                        
                        if overall_avg > 0:  # Evitar división por cero
                            # Factor de rendimiento en juegos cerrados
                            col_name = f'{stat}_in_close_games'
                            if col_name not in new_features:
                                new_features[col_name] = pd.Series(index=df.index)
                            
                            # Asignar el valor a todos los registros del jugador
                            new_features[col_name].loc[player_mask] = close_avg / overall_avg
        
        # 5. Características de matchup histórico entre jugadores
        # Estas características requieren datos de defensores que pueden no estar disponibles
        # Implementación simplificada basada en equipo oponente
        key_stats = ['PTS', 'TRB', 'AST', '3P']
        available_stats = [stat for stat in key_stats if stat in df.columns]
        
        for stat in available_stats:
            # Procesar cada jugador individualmente
            for player in df['Player'].unique():
                player_mask = df['Player'] == player
                player_data = df[player_mask].copy()
                
                if len(player_data) < 10:  # Necesitamos suficientes datos
                    continue
                
                # Agrupar por oponente para ver rendimiento histórico
                opp_stats = player_data.groupby('Opp')[stat].mean()
                
                # Para cada juego, obtener el rendimiento histórico contra ese oponente
                for opp in player_data['Opp'].unique():
                    if opp in opp_stats.index:
                        opp_mask = player_mask & (df['Opp'] == opp)
                        
                        col_name = f'matchup_history_{stat}'
                        if col_name not in new_features:
                            new_features[col_name] = pd.Series(index=df.index)
                        
                        # Asignar el valor histórico
                        new_features[col_name].loc[opp_mask] = opp_stats[opp]
        
        # Convertir a DataFrame y unir con el original
        features_df = pd.DataFrame(new_features, index=df.index)
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar resultados
        num_features = len(new_features)
        logger.info(f"Características de interacción de jugador generadas: {num_features} en {time.time() - start_time:.2f} segundos")
        
        return result
        
    def add_optimized_probability_features(self, df):
        """
        Genera características de probabilidad optimizadas con menor correlación entre umbrales.
        En lugar de generar múltiples umbrales muy cercanos (como PTS_over_15_prob, PTS_over_20_prob),
        utiliza umbrales absolutos más significativos para cada estadística.
        """
        logger.info("Creando características de probabilidad optimizadas...")
        start_time = time.time()
        
        # Diccionario para nuevas características
        new_features = {}
        
        # Estadísticas clave para generar probabilidades
        key_stats = ['PTS', 'TRB', 'AST', 'BLK', 'STL']
        available_stats = [stat for stat in key_stats if stat in df.columns]
        
        # Definir umbrales absolutos para cada estadística
        # Estos valores son más significativos para apuestas y análisis
        stat_thresholds = {
            'PTS': [15, 20, 25, 30, 35],  # Puntos: umbrales comunes para apuestas
            'TRB': [5, 8, 10, 12, 15],   # Rebotes: umbrales comunes para apuestas
            'AST': [3, 5, 8, 10, 12],    # Asistencias: umbrales comunes para apuestas
            'BLK': [1, 2, 3, 4, 5],      # Bloqueos: umbrales comunes para apuestas
            'STL': [1, 2, 3, 4, 5]       # Robos: umbrales comunes para apuestas
        }
        
        # Percentiles correspondientes para mantener la nomenclatura
        # Usamos 5 valores para que coincidan con los umbrales definidos
        percentiles = [20, 40, 60, 80, 95]
        
        # Inicializar todas las columnas de probabilidad para evitar NaN
        for stat in available_stats:
            for p in percentiles:
                feat_name = f"{stat}_p{p}_prob"
                smooth_name = f"{feat_name}_smooth"
                new_features[feat_name] = pd.Series(index=df.index, data=0.5)
                new_features[smooth_name] = pd.Series(index=df.index, data=0.5)
        
        # Para cada estadística, generar características de probabilidad optimizadas
        for stat in available_stats:
            logger.info(f"Procesando probabilidades para {stat}...")
            
            # Obtener umbrales para esta estadística
            thresholds = stat_thresholds.get(stat, [1, 2, 3])  # Valores predeterminados si no está definido
            
            logger.debug(f"Umbrales para {stat}: {thresholds}")
            
            # Procesar cada jugador individualmente
            for player in df['Player'].unique():
                try:
                    player_mask = df['Player'] == player
                    player_data = df[player_mask].sort_values('Date').copy()
                    
                    if len(player_data) < 5:  # Necesitamos al menos 5 juegos
                        # Usar valores predeterminados para este jugador
                        for i, p in enumerate(percentiles):
                            feat_name = f"{stat}_p{p}_prob"
                            smooth_name = f"{feat_name}_smooth"
                            new_features[feat_name].loc[player_mask] = 0.5
                            new_features[smooth_name].loc[player_mask] = 0.5
                        continue
                    
                    # Para cada umbral, calcular la probabilidad de superarlo
                    for i, threshold in enumerate(thresholds):
                        # Verificar que i esté dentro del rango válido de percentiles
                        if i < len(percentiles):
                            # Nombre de la característica usando percentil para mantener la nomenclatura
                            feat_name = f"{stat}_p{percentiles[i]}_prob"
                            smooth_name = f"{feat_name}_smooth"
                        else:
                            # Si hay más umbrales que percentiles, usar el último percentil disponible
                            feat_name = f"{stat}_p{percentiles[-1]}_prob_{i}"
                            smooth_name = f"{feat_name}_smooth"
                        
                        # Verificar que la estadística exista en los datos del jugador
                        if stat in player_data.columns:
                            # Calcular la probabilidad de superar el umbral (1 si supera, 0 si no)
                            binary_result = (player_data[stat] > threshold).astype(float)
                            
                            # Probabilidad base (media móvil)
                            rolling_prob = binary_result.rolling(10, min_periods=1).mean().fillna(0.5)
                            
                            # Versión suavizada (media móvil exponencial)
                            smooth_prob = binary_result.ewm(span=10).mean().fillna(0.5)
                        else:
                            # Si la estadística no existe, usar valores predeterminados
                            rolling_prob = pd.Series(0.5, index=player_data.index)
                            smooth_prob = pd.Series(0.5, index=player_data.index)
                        
                        # Asignar al diccionario de nuevas características
                        new_features[feat_name].loc[player_mask] = rolling_prob
                        new_features[smooth_name].loc[player_mask] = smooth_prob
                except Exception as e:
                    # Usar una representación segura para el registro que evite caracteres Unicode problemáticos
                    safe_player = str(player).encode('ascii', 'replace').decode('ascii')
                    logger.warning(f"Error calculando probabilidades de {stat} para jugador: {safe_player}")
                    # Usar valores predeterminados en caso de error
                    for i, p in enumerate(percentiles):
                        feat_name = f"{stat}_p{p}_prob"
                        smooth_name = f"{feat_name}_smooth"
                        new_features[feat_name].loc[player_mask] = 0.5
                        new_features[smooth_name].loc[player_mask] = 0.5
        
        # Añadir todas las nuevas características al DataFrame principal
        for col, values in new_features.items():
            # Asegurarse de que solo se asignen valores a índices válidos
            valid_indices = values.index.intersection(df.index)
            if not valid_indices.empty:
                df[col] = values.loc[valid_indices]
        
        # Contar nuevas características añadidas
        prob_cols = [col for col in df.columns if '_prob' in col]
        num_prob_features = len(prob_cols)
        
        # Calcular porcentaje de valores nulos
        missing_pct = df[prob_cols].isnull().mean().mean() * 100 if prob_cols else 0
        
        logger.info(f"Características de probabilidad optimizadas: {num_prob_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return df
        
    def add_momentum_indicators(self, df):
        """
        Calcula indicadores optimizados de impulso o momentum basados en 
        tendencias de rendimiento reciente.
        """
        logger.info("Creando indicadores de momentum optimizados...")
        start_time = time.time()
        
        # Diccionario para nuevas características
        new_features = {}
        
        # Usar solo dos ventanas para reducir redundancia
        if len(self.window_sizes) >= 2:
            short_window = min(self.window_sizes)
            long_window = max(self.window_sizes)
        else:
            short_window = 3
            long_window = 10
            
        logger.info(f"Usando ventanas {short_window} y {long_window} para indicadores de momentum")
        logger.info(f"Procesando {len(df['Player'].unique())} jugadores para indicadores de momentum...")
        
        # Estadísticas clave para momentum (reducidas a las más importantes)
        momentum_stats = ['PTS', 'TRB', 'AST', 'GmSc']
        
        for player in tqdm(df['Player'].unique(), desc="Procesando jugadores (momentum)"):
            player_mask = df['Player'] == player
            player_data = df[player_mask].sort_values('Date').copy()
            
            if len(player_data) <= 5:
                logger.debug(f"Omitiendo jugador {player} para momentum (datos insuficientes: {len(player_data)} filas)")
                continue
            
            # Tendencias de estadísticas principales
            for stat in momentum_stats:
                if stat not in player_data.columns:
                    logger.debug(f"Estadística {stat} no disponible para {player}")
                    continue
                    
                try:
                    # Solo calcular un índice de momentum por estadística (en lugar de varios)
                    momentum_col = f'{stat}_momentum'
                    
                    # Usando ventanas cortas y largas
                    short_avg = player_data[stat].rolling(short_window, min_periods=1).mean()
                    long_avg = player_data[stat].rolling(long_window, min_periods=1).mean()
                    
                    # Ratio corto/largo plazo (>1 significa tendencia al alza)
                    momentum = short_avg / long_avg.replace(0, 1)
                    df.loc[player_mask, momentum_col] = momentum
                    
                    # Calcular tendencia reciente
                    df.loc[player_mask, f'{stat}_trend'] = player_data[stat].rolling(short_window, min_periods=1).mean().diff()
                    
                    logger.debug(f"Añadidas características: {momentum_col}, {stat}_trend para {player}")
                except Exception as e:
                    logger.warning(f"Error calculando momentum para {stat} y {player}: {str(e)}")
            
            # Rachas de rendimiento (limitado a puntos solamente)
            if 'PTS' in player_data.columns:
                try:
                    # Media del jugador para puntos
                    player_mean = player_data['PTS'].mean()
                
                # Partidos consecutivos por encima/debajo de su media
                    above_mean = (player_data['PTS'] > player_mean).astype(int)
                
                # Calcular rachas
                    df.loc[player_mask, 'PTS_streak'] = above_mean.groupby(
                    (above_mean != above_mean.shift()).cumsum()
                ).cumcount() + 1
                
                    # Resetear cuando no aplica (racha negativa)
                    df.loc[player_mask & ~(player_data['PTS'] > player_mean), 'PTS_streak'] = -df.loc[player_mask & ~(player_data['PTS'] > player_mean), 'PTS_streak']
                    
                    logger.debug(f"Añadida característica: PTS_streak para {player}")
                except Exception as e:
                    logger.warning(f"Error calculando rachas para {player}: {str(e)}")
        
        # Momentum de Game Score (indicador holístico clave)
        if 'GmSc' in df.columns:
            logger.info("Calculando momentum de Game Score...")
            try:
                # GmSc vs media móvil de N partidos
                df['GmSc_vs_avg'] = df['GmSc'] / df.groupby('Player')['GmSc'].transform(
                    lambda x: x.rolling(long_window, min_periods=1).mean().shift(1)
                ).replace(0, 1)
                
                logger.debug("Añadida característica: GmSc_vs_avg")
            except Exception as e:
                logger.warning(f"Error calculando momentum de Game Score: {str(e)}")
        
        # Contar nuevas características añadidas a df después del procesamiento
        momentum_cols = [col for col in df.columns if any(x in col for x in ['_momentum', '_streak', '_vs_avg', '_trend'])]
        num_momentum_features = len(momentum_cols)
        
        # Calcular porcentaje de valores nulos
        missing_pct = df[momentum_cols].isnull().mean().mean() * 100 if momentum_cols else 0
        
        logger.info(f"Indicadores de momentum optimizados: {num_momentum_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return df

    def add_fatigue_metrics(self, df):
        """
        Calcula métricas de fatiga optimizadas basadas en carga de minutos y partidos consecutivos.
        """
        logger.info("Creando métricas de fatiga optimizadas...")
        start_time = time.time()
        
        # Diccionario para almacenar las nuevas métricas
        new_metrics = {}
        
        # Verificar si tenemos la columna de minutos
        if 'MP' not in df.columns:
            logger.warning("No se pueden calcular métricas de fatiga (falta columna MP)")
            return df
        
        # Minutos acumulados en el corto y medio plazo (reducido a dos ventanas)
        logger.debug("Calculando carga de minutos acumulados...")
        for window in [5, 10]:  # Reducido de 4 a 2 ventanas
            col_name = f'MP_last_{window}_games'
            new_metrics[col_name] = pd.Series(index=df.index)
            
            for player in tqdm(df['Player'].unique(), desc=f"Calculando carga (window={window})"):
                player_mask = df['Player'] == player
                if player_mask.sum() > 0:
                    rolling_mp = df[player_mask]['MP'].rolling(window, min_periods=1).sum()
                    new_metrics[col_name].loc[player_mask] = rolling_mp
        
            logger.debug(f"Añadida métrica: {col_name}")
        
        # Indicador de fatiga basado en minutos relativos a su promedio
        logger.debug("Calculando indicadores de fatiga relativos...")
        new_metrics['relative_minutes'] = pd.Series(index=df.index)
        
        for player in df['Player'].unique():
            player_mask = df['Player'] == player
            if player_mask.sum() > 0:
                player_avg_mp = df[player_mask]['MP'].mean()
                new_metrics['relative_minutes'].loc[player_mask] = df[player_mask]['MP'] / player_avg_mp
        
        logger.debug("Añadida métrica: relative_minutes")
        
        # Fatigue Index - índice consolidado de fatiga
        logger.debug("Calculando índice consolidado de fatiga...")
        
        # Crear columna days_rest si no existe
        if 'days_rest' not in df.columns:
            logger.info("Creando columna days_rest")
            # Ordenar por jugador y fecha
            if 'Date' in df.columns:
                df_sorted = df.sort_values(['Player', 'Date'])
                # Calcular días de descanso
                df['days_rest'] = df_sorted.groupby('Player')['Date'].diff().dt.days
                # Llenar NaN con valor por defecto (3 días)
                df['days_rest'] = df['days_rest'].fillna(3)
            else:
                logger.warning("No se puede crear days_rest (falta columna Date)")
                # Crear columna con valor por defecto
                df['days_rest'] = 3
        else:
            logger.info("La columna days_rest ya existe, omitiendo creación duplicada")
        
        # Efecto de descanso
            rest_multiplier = df['days_rest'].apply(
                lambda x: max(0.7, 1 - (0.1 * (1 if x <= 1 else 0)))
            )
            
            if 'MP_last_5_games' in new_metrics:
                # El índice de fatiga aumenta con mayor carga reciente y menos descanso
                try:
                    # Calcular explícitamente como números, no como Serie
                    new_metrics['Fatigue_Index'] = pd.Series(
                        data=(new_metrics['MP_last_5_games'].values / 120) * rest_multiplier.values,
                        index=df.index
                    )
                    logger.debug("Añadida métrica: Fatigue_Index")
                except Exception as e:
                    logger.warning(f"Error calculando Fatigue_Index: {e}")
                    # Alternativa más simple si falla
                    new_metrics['Fatigue_Index'] = 0.5  # Valor predeterminado
        
        # Crear columna is_b2b si no existe
        if 'is_b2b' not in df.columns:
            logger.info("Creando columna is_b2b")
            # Considerar B2B si days_rest <= 1
            df['is_b2b'] = (df['days_rest'] <= 1).astype(int)
        else:
            logger.info("La columna is_b2b ya existe, omitiendo creación duplicada")
        
        # Impacto de fatiga en rendimiento
        logger.debug("Calculando impacto de fatiga en rendimiento...")
            # Un solo indicador consolidado de impacto B2B en el rendimiento
        col_name = 'B2B_Impact'
        new_metrics[col_name] = pd.Series(index=df.index)
            
        for player in df['Player'].unique():
                player_mask = df['Player'] == player
                if player_mask.sum() > 0:
                    b2b_mask = df['is_b2b'] == 1
                    non_b2b_mask = df['is_b2b'] == 0
                    
                    if (player_mask & b2b_mask).sum() > 0 and (player_mask & non_b2b_mask).sum() > 0:
                        # Calcular promedio de GmSc en back-to-backs vs resto de juegos
                        if 'GmSc' in df.columns:
                            b2b_perf = df[player_mask & b2b_mask]['GmSc'].mean()
                            normal_perf = df[player_mask & non_b2b_mask]['GmSc'].mean()
                            impact = b2b_perf / normal_perf if normal_perf != 0 else 1
                            new_metrics[col_name].loc[player_mask] = impact
            
        logger.debug(f"Añadida métrica: {col_name}")
        
        # Convertir el diccionario a DataFrame
        metrics_df = pd.DataFrame(new_metrics)
        
        # Unir las nuevas métricas
        result = pd.concat([df, metrics_df], axis=1)
        
        # Registrar el número de métricas generadas
        num_new_metrics = len(new_metrics)
        missing_pct = metrics_df.isnull().mean().mean() * 100
        logger.info(f"Métricas de fatiga optimizadas: {num_new_metrics} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def add_advanced_stats(self, df):
        """
        Crea estadísticas avanzadas optimizadas y métricas derivadas para 
        capturar aspectos más profundos del rendimiento.
        """
        logger.info("Creando estadísticas avanzadas optimizadas...")
        start_time = time.time()
        
        # Diccionario para almacenar las nuevas métricas
        new_metrics = {}
        
        # Evitar división por cero
        eps = 1e-6
        
        # Game Score por minuto (normalizado) - métrica holística de rendimiento
        if 'GmSc' in df.columns and 'MP' in df.columns:
            new_metrics['GmSc_per_minute'] = df['GmSc'] / df['MP'].replace(0, eps)
            logger.debug("Añadida métrica: GmSc_per_minute")
        
        # Net Rating (aproximación usando BPM)
        if 'BPM' in df.columns:
            new_metrics['Net_Rating_Proxy'] = df['BPM'] * df['MP']
            logger.debug("Añadida métrica: Net_Rating_Proxy")
        
        # Contribución defensiva
        if all(col in df.columns for col in ['STL', 'BLK', 'MP']):
            new_metrics['Defensive_Rating_Proxy'] = (df['STL'] + df['BLK']) / df['MP'].replace(0, eps)
            logger.debug("Añadida métrica: Defensive_Rating_Proxy")
        
        # Eficiencia ofensiva
        if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
            new_metrics['Offensive_Efficiency'] = df['PTS'] / (df['FGA'] + 0.44 * df['FTA']).replace(0, eps)
            logger.debug("Añadida métrica: Offensive_Efficiency")
        
        # Versatilidad (contribución en múltiples categorías)
        categories = []
        for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
            if stat in df.columns:
                categories.append(stat)
                categories.append(stat)
        
        if len(categories) >= 3:  # Solo calcular si tenemos suficientes categorías
            # Normalizar cada estadística (0-1)
            norm_stats = {}
            for stat in categories:
                stat_norm = df.groupby('Player')[stat].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + eps)
                )
                norm_stats[f'{stat}_norm'] = stat_norm
            
            # Índice de versatilidad (promedio de valores normalizados)
            new_metrics['Versatility_Index'] = pd.DataFrame(norm_stats).mean(axis=1)
            logger.debug("Añadida métrica: Versatility_Index")
        
        # Indicador de rol ofensivo
        if all(col in df.columns for col in ['FGA', 'AST', 'MP']):
            # Ratio de tiros vs asistencias para determinar el rol ofensivo
            new_metrics['Offensive_Role'] = (df['FGA'] / df['MP'].replace(0, eps)) / (df['AST'] / df['MP'].replace(0, eps) + eps)
            
            # Categorización de roles
            df['Role_Category'] = pd.cut(
                new_metrics['Offensive_Role'],
                bins=[0, 0.5, 1.5, 3, float('inf')],
                labels=['Playmaker', 'Balanced', 'Scorer', 'Pure Scorer']
            )
        
            logger.debug("Añadidas métricas: Offensive_Role, Role_Category")
        
        # Convertir el diccionario a DataFrame
        metrics_df = pd.DataFrame(new_metrics)
        
        # Unir las nuevas métricas
        result = pd.concat([df, metrics_df], axis=1)
        
        # Registrar el número de métricas generadas
        num_new_metrics = len(new_metrics) + ('Role_Category' in df.columns)
        missing_pct = metrics_df.isnull().mean().mean() * 100
        
        logger.info(f"Métricas de fatiga optimizadas: {num_new_metrics} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result
    
    def add_team_features(self, df):
        """
        Agrega características optimizadas a nivel de equipo.
        Maneja correctamente los datos de múltiples jugadores del mismo equipo y partido.
        """
        logger.info("Creando características de equipo optimizadas...")
        start_time = time.time()
        
        # Verificar si los campos necesarios están disponibles
        required_fields = ['Team', 'Date']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            logger.warning(f"Faltan campos requeridos para características de equipo: {missing_fields}")
            return df
        
        # Asegurar que las columnas básicas existan
        if 'team_score' not in df.columns and 'PTS' in df.columns:
            logger.info("Creando columna team_score a partir de PTS")
            # Agregar por equipo y fecha para obtener puntos de equipo
            team_pts = df.groupby(['Team', 'Date'])['PTS'].sum().reset_index()
            team_pts.rename(columns={'PTS': 'team_score'}, inplace=True)
            # Unir con el DataFrame original
            df = pd.merge(df, team_pts, on=['Team', 'Date'], how='left')
        
        if 'opp_score' not in df.columns and 'Opp_PTS' in df.columns:
            logger.info("Creando columna opp_score a partir de Opp_PTS")
            df['opp_score'] = df['Opp_PTS']
        
        if 'is_win' not in df.columns and 'Win' in df.columns:
            logger.info("Creando columna is_win a partir de Win")
            df['is_win'] = df['Win']
        
        if 'is_home' not in df.columns and 'Home' in df.columns:
            logger.info("Creando columna is_home a partir de Home")
            df['is_home'] = df['Home']
        
        # Verificar si tenemos las columnas necesarias después de las transformaciones
        if not all(col in df.columns for col in ['team_score', 'opp_score']):
            logger.warning("No se pueden generar características de equipo (faltan columnas de puntuación)")
            return df
        
        # Asegurar tipos de datos numéricos
        for col in ['team_score', 'opp_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Crear un DataFrame único a nivel de equipo-partido
        # Usamos 'first' para asegurar que tomamos un solo valor por equipo y fecha
        logger.info("Creando DataFrame único a nivel de equipo-partido")
        try:
            team_game_stats = df.groupby(['Team', 'Date']).agg({
                'team_score': 'first',  # Todos los jugadores del mismo equipo tienen el mismo valor
                'opp_score': 'first',    # Todos los jugadores del mismo equipo tienen el mismo valor
                'is_win': 'first',       # Todos los jugadores del mismo equipo tienen el mismo resultado
                'is_home': 'first'       # Todos los jugadores del mismo equipo juegan en el mismo lugar
            }).reset_index()
        
            # Calcular características básicas de equipo
            team_game_stats['point_diff'] = team_game_stats['team_score'] - team_game_stats['opp_score']
            team_game_stats['total_score'] = team_game_stats['team_score'] + team_game_stats['opp_score']
            
            # Calcular métricas de eficiencia
            # Estimación de posesiones usando la fórmula básica de la NBA
            team_game_stats['possessions'] = team_game_stats['team_score'] * 0.96
            team_game_stats['offensive_rating'] = team_game_stats['team_score'] / team_game_stats['possessions'] * 100
            team_game_stats['defensive_rating'] = team_game_stats['opp_score'] / team_game_stats['possessions'] * 100
            team_game_stats['net_rating'] = team_game_stats['offensive_rating'] - team_game_stats['defensive_rating']
            
            # Calcular métricas de shooting
            if 'FG' in df.columns and 'FGA' in df.columns:
                try:
                    # Sumar FG y FGA por equipo y fecha
                    team_shooting = df.groupby(['Team', 'Date']).agg({
                        'FG': 'sum',
                        'FGA': 'sum'
                    }).reset_index()
                    
                    # Calcular eficiencia
                    team_shooting['shooting_efficiency'] = team_shooting['FG'] / team_shooting['FGA'].replace(0, 1)
                    
                    # Unir con team_game_stats
                    team_game_stats = pd.merge(
                        team_game_stats,
                        team_shooting[['Team', 'Date', 'shooting_efficiency']],
                        on=['Team', 'Date'],
                        how='left'
                    )
                except Exception as e:
                    logger.error(f"Error al calcular shooting efficiency: {e}")
                    team_game_stats['shooting_efficiency'] = 0.45  # Valor promedio NBA
            
            # Calcular métricas de asistencias
            if 'AST' in df.columns:
                try:
                    # Sumar asistencias por equipo y fecha
                    team_assists = df.groupby(['Team', 'Date'])['AST'].sum().reset_index()
                    
                    # Crear una copia de team_game_stats para evitar problemas de alineación
                    possessions_df = team_game_stats[['Team', 'Date', 'possessions']].copy()
                    
                    # Unir asistencias con posesiones
                    assist_df = pd.merge(
                        team_assists,
                        possessions_df,
                        on=['Team', 'Date'],
                        how='left'
                    )
                    
                    # Calcular ratio
                    assist_df['assist_ratio'] = assist_df['AST'] / assist_df['possessions'] * 100
                    
                    # Unir con team_game_stats
                    team_game_stats = pd.merge(
                        team_game_stats,
                        assist_df[['Team', 'Date', 'assist_ratio']],
                        on=['Team', 'Date'],
                        how='left'
                    )
                except Exception as e:
                    logger.error(f"Error al calcular assist ratio: {e}")
                    team_game_stats['assist_ratio'] = 50  # Valor promedio
            
            # Calcular métricas de rebotes
            if 'TRB' in df.columns:
                try:
                    # Sumar rebotes por equipo y fecha
                    team_rebounds = df.groupby(['Team', 'Date'])['TRB'].sum().reset_index()
                    
                    # Crear una copia de team_game_stats para evitar problemas de alineación
                    opp_score_df = team_game_stats[['Team', 'Date', 'opp_score']].copy()
                    
                    # Unir rebotes con puntuación del oponente
                    rebound_df = pd.merge(
                        team_rebounds,
                        opp_score_df,
                        on=['Team', 'Date'],
                        how='left'
                    )
                    
                    # Calcular ratio
                    rebound_df['rebound_rate'] = rebound_df['TRB'] / (rebound_df['TRB'] + rebound_df['opp_score'] * 0.4)
                    
                    # Unir con team_game_stats
                    team_game_stats = pd.merge(
                        team_game_stats,
                        rebound_df[['Team', 'Date', 'rebound_rate']],
                        on=['Team', 'Date'],
                        how='left'
                    )
                except Exception as e:
                    logger.error(f"Error al calcular rebound rate: {e}")
                    team_game_stats['rebound_rate'] = 0.5  # Valor promedio
            
            # Calcular métricas de turnovers
            if 'TOV' in df.columns:
                try:
                    # Sumar turnovers por equipo y fecha
                    team_turnovers = df.groupby(['Team', 'Date'])['TOV'].sum().reset_index()
                    
                    # Crear una copia de team_game_stats para evitar problemas de alineación
                    possessions_df = team_game_stats[['Team', 'Date', 'possessions']].copy()
                    
                    # Unir turnovers con posesiones
                    turnover_df = pd.merge(
                        team_turnovers,
                        possessions_df,
                        on=['Team', 'Date'],
                        how='left'
                    )
                    
                    # Calcular ratio
                    turnover_df['turnover_rate'] = turnover_df['TOV'] / turnover_df['possessions'] * 100
                    
                    # Unir con team_game_stats
                    team_game_stats = pd.merge(
                        team_game_stats,
                        turnover_df[['Team', 'Date', 'turnover_rate']],
                        on=['Team', 'Date'],
                        how='left'
                    )
                except Exception as e:
                    logger.error(f"Error al calcular turnover rate: {e}")
                    team_game_stats['turnover_rate'] = 15  # Valor promedio
            
            # Calcular rachas y tendencias
            team_stats_list = []
            for team in team_game_stats['Team'].unique():
                team_data = team_game_stats[team_game_stats['Team'] == team].sort_values('Date')
                
                # Calcular rachas
                team_data['win_streak'] = team_data['is_win'].groupby((team_data['is_win'] != team_data['is_win'].shift()).cumsum()).cumcount() + 1
                team_data['win_streak'] = team_data['win_streak'] * team_data['is_win']  # Solo contar rachas de victoria
                
                # Promedios móviles
                for window in [5, 10]:
                    # Estadísticas básicas
                    team_data[f'points_avg_{window}'] = team_data['team_score'].rolling(window=window, min_periods=1).mean()
                    team_data[f'points_against_avg_{window}'] = team_data['opp_score'].rolling(window=window, min_periods=1).mean()
                    team_data[f'point_diff_avg_{window}'] = team_data['point_diff'].rolling(window=window, min_periods=1).mean()
                    
                    # Ratings
                    team_data[f'offensive_rating_avg_{window}'] = team_data['offensive_rating'].rolling(window=window, min_periods=1).mean()
                    team_data[f'defensive_rating_avg_{window}'] = team_data['defensive_rating'].rolling(window=window, min_periods=1).mean()
                    team_data[f'net_rating_avg_{window}'] = team_data['net_rating'].rolling(window=window, min_periods=1).mean()
                    
                    # Estadísticas avanzadas
                    if 'shooting_efficiency' in team_data.columns:
                        team_data[f'shooting_efficiency_avg_{window}'] = team_data['shooting_efficiency'].rolling(window=window, min_periods=1).mean()
                    if 'assist_ratio' in team_data.columns:
                        team_data[f'assist_ratio_avg_{window}'] = team_data['assist_ratio'].rolling(window=window, min_periods=1).mean()
                    if 'rebound_rate' in team_data.columns:
                        team_data[f'rebound_rate_avg_{window}'] = team_data['rebound_rate'].rolling(window=window, min_periods=1).mean()
                    if 'turnover_rate' in team_data.columns:
                        team_data[f'turnover_rate_avg_{window}'] = team_data['turnover_rate'].rolling(window=window, min_periods=1).mean()
                
                # Porcentajes de victoria en casa/fuera
                team_data['home_win_pct'] = team_data[team_data['is_home']]['is_win'].expanding().mean()
                team_data['away_win_pct'] = team_data[~team_data['is_home']]['is_win'].expanding().mean()
                
                # Calcular forma reciente del equipo
                team_data['recent_form'] = team_data['is_win'].rolling(window=5, min_periods=1).mean()
                
                # Calcular tendencia de rendimiento
                team_data['performance_trend'] = team_data['point_diff'].rolling(window=5, min_periods=1).mean()
                
                # Calcular rachas
                team_data['current_streak'] = team_data['is_win'].groupby((team_data['is_win'] != team_data['is_win'].shift()).cumsum()).cumcount() + 1
                team_data['current_streak'] = team_data['current_streak'] * np.where(team_data['is_win'], 1, -1)
                
                # Calcular historial de enfrentamientos si existe la columna 'Opp'
                if 'Opp' in team_data.columns:
                    try:
                        # Crear un DataFrame para el historial de enfrentamientos
                        h2h_stats = pd.DataFrame()
                        
                        # Calcular victorias contra cada oponente
                        h2h_wins = team_data[team_data['is_win']].groupby('Opp').size()
                        h2h_total = team_data.groupby('Opp').size()
                        
                        # Calcular porcentaje de victorias contra cada oponente
                        h2h_stats['win_pct'] = (h2h_wins / h2h_total).fillna(0.5)
                        
                        # Calcular tendencia reciente contra cada oponente (últimos 3 enfrentamientos)
                        for opp in team_data['Opp'].unique():
                            mask = team_data['Opp'] == opp
                            recent_games = team_data[mask]['is_win'].tail(3)
                            h2h_stats.loc[opp, 'recent_h2h'] = recent_games.mean() if not recent_games.empty else 0.5
                        
                        # Unir las estadísticas al DataFrame principal
                        team_data['h2h_win_pct'] = team_data['Opp'].map(h2h_stats['win_pct'])
                        team_data['recent_h2h'] = team_data['Opp'].map(h2h_stats['recent_h2h'])
                        
                    except Exception as e:
                        logger.error(f"Error al calcular historial de enfrentamientos: {e}")
                        team_data['h2h_win_pct'] = 0.5
                        team_data['recent_h2h'] = 0.5

                team_stats_list.append(team_data)
            
            # Combinar todos los datos de equipo
            team_game_stats = pd.concat(team_stats_list)
            
        except Exception as e:
            logger.error(f"Error creando DataFrame de equipo: {e}")
            logger.error(traceback.format_exc())
            return df
    
        # Seleccionar las columnas que queremos unir con el DataFrame original
        team_features = [
            'Team', 'Date',
            'point_diff', 'total_score',
            'offensive_rating', 'defensive_rating', 'net_rating',
            'win_streak',
            'home_win_pct', 'away_win_pct'
        ]
        
        # Añadir columnas de promedios móviles
        for window in [5, 10]:
            team_features.extend([
                f'points_avg_{window}',
                f'points_against_avg_{window}',
                f'point_diff_avg_{window}',
                f'offensive_rating_avg_{window}',
                f'defensive_rating_avg_{window}'
            ])
        
        # Unir las estadísticas del equipo con el DataFrame original
        result = pd.merge(
            df,
            team_game_stats[team_features],
            on=['Team', 'Date'],
            how='left'
        )
        
        # Rellenar valores nulos con 0
        for col in team_features:
            if col not in ['Team', 'Date']:
                result[col] = result[col].fillna(0)
        
        # Registrar las características generadas
        num_features = len(team_features) - 2  # -2 por Team y Date
        missing_pct = result[team_features[2:]].isnull().mean().mean() * 100
        logger.info(f"Características de equipo generadas: {num_features} columnas")
        logger.info(f"Porcentaje de valores faltantes: {missing_pct:.2f}%")
        logger.info(f"Tiempo de procesamiento: {time.time() - start_time:.2f} segundos")
        
        return result

    def add_team_prediction_features(self, df):
        """
        Agrega características optimizadas específicas para predicciones a nivel de equipo.
        """
        logger = logging.getLogger(__name__)
        logger.info("Creando características optimizadas de predicción de equipo...")
        start_time = time.time()
        
        # Verificar si tenemos las columnas necesarias
        if not all(col in df.columns for col in ['team_score', 'opp_score', 'is_win', 'Opp']):
            logger.warning("No se pueden generar características de predicción de equipo (faltan columnas necesarias)")
            return df
        
        # Asegurar tipos de datos numéricos
        df['team_score'] = pd.to_numeric(df['team_score'], errors='coerce')
        df['opp_score'] = pd.to_numeric(df['opp_score'], errors='coerce')
        
        # Asegurar que is_win sea numérico (1 = victoria, 0 = derrota)
        if 'is_win' in df.columns:
            # Convertir explícitamente a int para evitar problemas con ~
            df['is_win'] = df['is_win'].astype(int)
        else:
            # Si no existe, crear basándose en team_score y opp_score
            df['is_win'] = (df['team_score'] > df['opp_score']).astype(int)
        
        # Primero, calculamos las métricas de ritmo y eficiencia por partido
        # Esto asegura que todos los jugadores del mismo equipo y fecha tengan los mismos valores
        team_game_metrics = df.groupby(['Date', 'Team']).agg({
            'team_score': 'first',
            'opp_score': 'first',
            'is_win': 'first',
        }).reset_index()
        
        # Calcular métricas adicionales
        team_game_metrics['Total_Score'] = team_game_metrics['team_score'] + team_game_metrics['opp_score']
        team_game_metrics['Point_Diff'] = team_game_metrics['team_score'] - team_game_metrics['opp_score']
        
        # Si los campos ya están parseados por result_parser, usarlos directamente
        if 'total_score' in df.columns and 'point_diff' in df.columns:
            # Crear columnas de agregación para que todos los jugadores tengan el mismo valor
            team_game_metrics = df.groupby(['Date', 'Team']).agg({
                'total_score': 'first',
                'point_diff': 'first'
            }).reset_index()
            team_game_metrics.rename(columns={
                'total_score': 'Total_Score',
                'point_diff': 'Point_Diff'
            }, inplace=True)
        else:
            # Calcular según el método actual
            team_game_metrics['Total_Score'] = team_game_metrics['team_score'] + team_game_metrics['opp_score']
            team_game_metrics['Point_Diff'] = team_game_metrics['team_score'] - team_game_metrics['opp_score']
        
        # Renombrar columnas
        team_stats = team_game_metrics.rename(columns={
            'PTS': 'team_score',
            'Opp_PTS': 'opp_score'
        })
        
        # Calcular diferencia de puntos
        team_stats['point_diff'] = team_stats['team_score'] - team_stats['opp_score']
        
        # Unir estas métricas al DataFrame original
        df = pd.merge(df, team_game_metrics[['Date', 'Team', 'Total_Score', 'Point_Diff']], 
                   on=['Date', 'Team'], how='left')
        
        # Estadísticas móviles para predicción (ventana única)
        logger.debug("Calculando estadísticas móviles para predicción de equipos...")
        window = 10
        
        # Primero calculamos promedios históricos por equipo a nivel de partido
        # Esto evita que se dupliquen valores al agrupar solo por equipo
        
        # Asegurar tipos de datos numéricos
        team_game_metrics['Total_Score'] = pd.to_numeric(team_game_metrics['Total_Score'], errors='coerce')
        team_game_metrics['Point_Diff'] = pd.to_numeric(team_game_metrics['Point_Diff'], errors='coerce')
        
        # Calcular estadísticas móviles con manejo de errores
        def safe_rolling_mean(series):
            try:
                # Convertir a numérico y rellenar NaN
                numeric_series = pd.to_numeric(series, errors='coerce').fillna(0)
                return numeric_series.rolling(window=window, min_periods=1).mean()
            except Exception as e:
                logger.warning(f"Error en rolling mean: {e}")
                return pd.Series(0, index=series.index)
        
        # Calcular promedios móviles
        team_rolling_stats = []
        for team, team_data in team_game_metrics.sort_values(['Team', 'Date']).groupby('Team'):
            # Calcular rolling mean con seguridad
            total_score_avg = safe_rolling_mean(team_data['Total_Score'])
            point_diff_avg = safe_rolling_mean(team_data['Point_Diff'])
            
            # Crear DataFrame temporal
            temp_df = pd.DataFrame({
                'Team': team_data['Team'],
                'Date': team_data['Date'],
                'Total_Score_avg': total_score_avg,
                'Point_Diff_avg': point_diff_avg
            })
            
            team_rolling_stats.append(temp_df)
        
        # Concatenar resultados
        team_rolling_stats = pd.concat(team_rolling_stats, ignore_index=True)
        
        # Rellenar valores nulos con 0
        team_rolling_stats = team_rolling_stats.fillna(0)
        
        # También calculamos desviación estándar y tendencia
        # Calcular desviación estándar con manejo de errores
        team_rolling_stats_std = []
        for team, team_data in team_game_metrics.sort_values(['Team', 'Date']).groupby('Team'):
            try:
                # Calcular std con seguridad
                total_score_std = pd.to_numeric(team_data['Total_Score'], errors='coerce').fillna(0).rolling(window=window, min_periods=3).std()
                
                # Crear DataFrame temporal
                temp_df = pd.DataFrame({
                    'Team': team_data['Team'],
                    'Date': team_data['Date'],
                    'Total_Score_std': total_score_std
                })
                
                team_rolling_stats_std.append(temp_df)
            except Exception as e:
                logger.warning(f"Error en rolling std para {team}: {e}")
                
        # Concatenar resultados
        team_rolling_stats_std = pd.concat(team_rolling_stats_std, ignore_index=True)
        
        # Calcular tendencia (diferencia en la media móvil) con manejo de errores
        team_trend = []
        for team, team_data in team_game_metrics.sort_values(['Team', 'Date']).groupby('Team'):
            try:
                # Calcular tendencia con seguridad
                total_score_trend = pd.to_numeric(team_data['Total_Score'], errors='coerce').fillna(0).rolling(window=window, min_periods=3).mean().diff()
                
                # Crear DataFrame temporal
                temp_df = pd.DataFrame({
                    'Team': team_data['Team'],
                    'Date': team_data['Date'],
                    'Total_Score_trend': total_score_trend
                })
                
                team_trend.append(temp_df)
            except Exception as e:
                logger.warning(f"Error en cálculo de tendencia para {team}: {e}")
                
        # Concatenar resultados
        team_trend = pd.concat(team_trend, ignore_index=True)
        
        # Unir todas las estadísticas calculadas
        team_prediction_features = pd.merge(team_rolling_stats, team_rolling_stats_std, 
                                          on=['Date', 'Team'], how='left')
        team_prediction_features = pd.merge(team_prediction_features, team_trend, 
                                          on=['Date', 'Team'], how='left')
        
        # Reunir las columnas que necesitamos y unir con el dataframe principal
        prediction_cols = ['Team', 'Date', 'Total_Score_avg', 'Point_Diff_avg', 
                         'Total_Score_std', 'Total_Score_trend']
        
        # Ya no necesitamos mapear nivel_1 a fecha porque estamos usando Date directamente en los merge
        
        # Unir solo las columnas necesarias
        result = pd.merge(df, team_prediction_features[prediction_cols], 
                       on=['Team', 'Date'], how='left')
        
        # Registrar características generadas
        num_features = len(prediction_cols) - 2  # -2 por Team y Date que no son features
        missing_pct = result[prediction_cols[2:]].isnull().mean().mean() * 100
        logger.info(f"Características de predicción de equipo optimizadas: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def add_player_prediction_features(self, df):
        """
        Agrega características específicas para predicción de estadísticas de jugadores.
        """
        logger.info("Creando características para predicción de estadísticas de jugadores...")
        start_time = time.time()
        
        # Estadísticas clave para predecir
        key_stats = ['PTS', 'TRB', 'AST', '3P']
        available_stats = [stat for stat in key_stats if stat in df.columns]
        
        if not available_stats:
            logger.warning("No se pueden generar características de predicción de jugador (faltan estadísticas clave)")
            return df
        
        logger.info(f"Estadísticas disponibles para predicción de jugador: {available_stats}")
        
        # Diccionario para las nuevas características
        new_features = {}
        
        # Inicializar todas las series para cada estadística antes del bucle
        for stat in available_stats:
            new_features[f'{stat}_consistency'] = pd.Series(index=df.index, dtype=float)
            new_features[f'{stat}_above_avg_prob'] = pd.Series(index=df.index, dtype=float)
        
        # Ventana para cálculos
        window = 10
        
        for player in tqdm(df['Player'].unique(), desc="Procesando jugadores (predicción)"):
            player_mask = df['Player'] == player
            player_data = df[player_mask].sort_values('Date')
            
            # Si no hay suficientes datos para cálculos estadísticos sólidos
            if len(player_data) <= 5:
                logger.debug(f"Realizando cálculos simplificados para {player} (pocos datos: {len(player_data)} filas)")
                
                for stat in available_stats:
                    # Para consistencia, usar un valor predeterminado neutral
                    default_consistency = 0.5  # Neutral
                    new_features[f'{stat}_consistency'].loc[player_mask] = default_consistency
                    
                    # Para probabilidad de superar media, usar la proporción de valores sobre la media actual
                    if len(player_data) > 1:
                        player_mean = player_data[stat].mean()
                        above_mean_ratio = (player_data[stat] > player_mean).mean()
                        new_features[f'{stat}_above_avg_prob'].loc[player_mask] = above_mean_ratio
                    else:
                        new_features[f'{stat}_above_avg_prob'].loc[player_mask] = 0.5  # 50% por defecto
                
                continue
            
            for stat in available_stats:
                # Consistencia - factor clave para predicción
                try:
                    # Cálculo de consistencia
                    stat_data = player_data[stat]
                    
                    # Calcular consistencia como 1 - (std/mean)
                    rolling_mean = stat_data.rolling(window, min_periods=1).mean()
                    rolling_std = stat_data.rolling(window, min_periods=1).std()
                    
                    # Evitar división por cero y limitar a valores razonables
                    rolling_mean_safe = rolling_mean.replace(0, 1)
                    consistency = 1 - (rolling_std / rolling_mean_safe).clip(0, 1)
                    
                    # Rellenar valores NaN con el promedio de consistencia
                    mean_consistency = consistency.mean()
                    # Asegurar que mean_consistency sea un valor escalar
                    if isinstance(mean_consistency, (pd.Series, np.ndarray)):
                        mean_consistency = float(mean_consistency) if len(mean_consistency) > 0 else 0.5
                        
                    # Comprobar si es NaN y usar valor por defecto si es necesario
                    if pd.isna(float(mean_consistency)):
                        mean_consistency = 0.5  # Valor por defecto si no hay promedio válido
                    consistency = consistency.fillna(mean_consistency)
                    
                    # Asignar solo a las filas del jugador actual
                    new_features[f'{stat}_consistency'].loc[player_mask] = consistency.values
                    
                    # Probabilidad de superar su propia media
                    player_mean = stat_data.expanding(min_periods=1).mean()
                    above_mean = (stat_data > player_mean).astype(int)
                    above_mean_prob = above_mean.rolling(window, min_periods=1).mean()
                    
                    # Rellenar valores NaN con el primer valor válido o 0.5
                    first_valid = above_mean_prob.first_valid_index()
                    default_prob = 0.5
                    if first_valid is not None:
                        default_prob = above_mean_prob.loc[first_valid]
                    above_mean_prob = above_mean_prob.fillna(default_prob)
                    
                    # Asignar solo a las filas del jugador actual
                    new_features[f'{stat}_above_avg_prob'].loc[player_mask] = above_mean_prob.values
                    
                    logger.debug(f"Añadidas características de predicción para {stat} y {player}")
                except Exception as e:
                    logger.warning(f"Error calculando características de predicción para {stat} y {player}: {str(e)}")
                    # En caso de error, asignar valores predeterminados
                    new_features[f'{stat}_consistency'].loc[player_mask] = 0.5
                    new_features[f'{stat}_above_avg_prob'].loc[player_mask] = 0.5
                    
        
        # Características para doble-doble (solo si tenemos las estadísticas necesarias)
        if all(stat in df.columns for stat in ['PTS', 'TRB', 'AST']):
            logger.debug("Calculando probabilidades de doble-doble...")
            
            # Inicializar la probabilidad de doble-doble para todos los jugadores una sola vez
            double_double_prob_series = pd.Series(index=df.index, dtype=float)
            
            for player in df['Player'].unique():
                player_mask = df['Player'] == player
                player_data = df[player_mask].sort_values('Date')
                
                try:
                    if len(player_data) <= 5:
                        # Para jugadores con pocos datos, usar valores predeterminados basados en posición
                        position = player_data['Pos'].iloc[0] if 'Pos' in player_data.columns else None
                        if position in ['C', 'PF']:  # Posiciones con más probabilidad de doble-doble
                            default_prob = 0.3
                        elif position in ['PG', 'SG']:  # Bases pueden conseguir por puntos+asistencias
                            default_prob = 0.15
                        else:
                            default_prob = 0.1
                            
                        double_double_prob_series.loc[player_mask] = default_prob
                        continue
                        
                    # Verificar cuáles estadísticas superan 10
                    stats_10_plus = ((player_data['PTS'] >= 10).astype(int) + 
                                   (player_data['TRB'] >= 10).astype(int) + 
                                   (player_data['AST'] >= 10).astype(int))
                    
                    # Probabilidad de doble-doble
                    double_double = (stats_10_plus >= 2).astype(int)
                    double_double_prob = double_double.rolling(window, min_periods=1).mean()
                    
                    # Rellenar valores NaN con el primer valor válido o con 0.1 (probabilidad base)
                    first_valid = double_double_prob.first_valid_index()
                    default_prob = 0.1  # Valor por defecto si no hay histórico
                    if first_valid is not None:
                        default_prob = double_double_prob.loc[first_valid]
                    double_double_prob = double_double_prob.fillna(default_prob)
                    
                    # Asignar solo a las filas del jugador actual
                    double_double_prob_series.loc[player_mask] = double_double_prob.values
                except Exception as e:
                    logger.warning(f"Error calculando probabilidad de doble-doble para {player}: {str(e)}")
                    # En caso de error, valor predeterminado conservador
                    double_double_prob_series.loc[player_mask] = 0.1
            
            # Añadir al diccionario de características solo cuando se completen todos los jugadores
            new_features['Double_Double_Prob'] = double_double_prob_series
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(new_features)
        
        # Rellenar cualquier valor nulo restante
        for col in features_df.columns:
            if 'consistency' in col:
                features_df[col] = features_df[col].fillna(0.5)  # Valor neutro para consistencia
            elif 'prob' in col:
                features_df[col] = features_df[col].fillna(0.1)  # Valor conservador para probabilidades
        
        # Unir con el DataFrame original
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de predicción de jugador optimizadas: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def add_line_prediction_features(self, df):
        """
        Agrega características optimizadas específicas para predicción de líneas (over/under).
        """
        logger.info("Creando características optimizadas para predicción de líneas...")
        start_time = time.time()
        
        # Estadísticas clave para predecir
        key_stats = ['PTS', 'TRB', 'AST', '3P']
        available_stats = [stat for stat in key_stats if stat in df.columns]
        
        if not available_stats:
            logger.warning("No se pueden generar características de predicción de líneas (faltan estadísticas clave)")
            return df
        
        # Diccionario para las nuevas características
        new_features = {}
        
        # Líneas comunes en apuestas (solo las más utilizadas)
        lines = {
            'PTS': [10, 15, 20, 25, 30],
            'TRB': [4, 5, 6, 7, 8, 9, 10],
            'AST': [4, 5, 6, 7, 8, 9, 10],
            '3P': [1, 2, 3, 4]
        }
        
        # Ventana para cálculos
        window = 10
        
        # Limitar líneas para cada estadística que esté disponible
        for stat in available_stats:
            # Seleccionar solo 2-3 líneas por estadística para reducir dimensionalidad
            # y evitar generar demasiadas características con valores nulos
            if stat == 'PTS':
                selected_lines = [15, 20, 25]  # Líneas comunes para puntos
            elif stat == 'TRB':
                selected_lines = [5, 7, 10]    # Líneas comunes para rebotes
            elif stat == 'AST':
                selected_lines = [5, 7, 10]    # Líneas comunes para asistencias
            elif stat == '3P':
                selected_lines = [1, 2, 3]     # Líneas comunes para triples
            else:
                continue
                
            logger.debug(f"Calculando características de línea para {stat} con líneas: {selected_lines}")
            
            for line in selected_lines:
                # Clave para la línea en formato stat_over_line
                line_key = f"{stat}_over_{line}"
                
                # Inicializar series para cada característica
                new_features[f'{line_key}_prob'] = pd.Series(index=df.index, dtype=float)
                
                # Usamos solo frecuencia reciente para reducir dimensionalidad
                new_features[f'{line_key}_recent'] = pd.Series(index=df.index, dtype=float)
                
                # Procesar cada jugador
                for player in tqdm(df['Player'].unique(), desc=f"Procesando {stat}>{line}"):
                    player_mask = df['Player'] == player
                    player_data = df[player_mask].sort_values('Date')
                    
                    if len(player_data) <= 3:
                        # Para jugadores con pocos datos, usar valores predeterminados
                        
                        # Evaluar si el jugador supera la línea en promedio
                        if len(player_data) > 0 and stat in player_data.columns:
                            avg_value = player_data[stat].mean()
                            default_prob = 0.8 if avg_value > line else 0.2
                        else:
                            # Probabilidad default basada en la dificultad de la línea
                            position = player_data['Pos'].iloc[0] if 'Pos' in player_data.columns and len(player_data) > 0 else None
                            
                            # Ajustar según posición y estadística
                            if stat == 'PTS':
                                default_prob = 0.5 if line <= 15 else 0.3
                            elif stat == 'TRB':
                                default_prob = 0.5 if (position in ['C', 'PF'] and line <= 5) else 0.2
                            elif stat == 'AST':
                                default_prob = 0.5 if (position in ['PG'] and line <= 5) else 0.2
                            elif stat == '3P':
                                default_prob = 0.5 if (position in ['SG', 'SF'] and line <= 1) else 0.2
                            else:
                                default_prob = 0.3
                        
                        # Asignar los valores predeterminados
                        new_features[f'{line_key}_prob'].loc[player_mask] = default_prob
                        new_features[f'{line_key}_recent'].loc[player_mask] = default_prob
                        continue
                    
                    try:
                        # Determinar si cada partido supera la línea
                        over_line = (player_data[stat] > line).astype(int)
                        
                        # Calcular probabilidad basada en el histórico completo
                        historical_prob = over_line.expanding(min_periods=1).mean()
                        
                        # Calcular frecuencia reciente (últimos N partidos)
                        recent_prob = over_line.rolling(window=window, min_periods=1).mean()
                        
                        # Rellenar valores nulos
                        if over_line.iloc[0] == 1:
                            # Si el primer valor supera la línea, usar 0.8 como valor inicial
                            historical_prob = historical_prob.fillna(0.8)
                            recent_prob = recent_prob.fillna(0.8)
                        else:
                            # Si no supera la línea, usar 0.2 como valor inicial
                            historical_prob = historical_prob.fillna(0.2)
                            recent_prob = recent_prob.fillna(0.2)
                        
                        # Asignar resultados a las series correspondientes
                        new_features[f'{line_key}_prob'].loc[player_mask] = historical_prob.values
                        new_features[f'{line_key}_recent'].loc[player_mask] = recent_prob.values
                        
                    except Exception as e:
                        logger.warning(f"Error calculando predicción de línea {line_key} para {player}: {str(e)}")
                        # En caso de error, asignar valores predeterminados conservadores
                        new_features[f'{line_key}_prob'].loc[player_mask] = 0.5
                        new_features[f'{line_key}_recent'].loc[player_mask] = 0.5
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(new_features)
        
        # Rellenar cualquier valor nulo restante
        for col in features_df.columns:
            features_df[col] = features_df[col].fillna(0.3)  # Valor conservador para probabilidades de línea
        
        # Unir con el DataFrame original
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de predicción de líneas optimizadas: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def add_game_situation_features(self, df):
        """
        Agrega características optimizadas relacionadas con la situación del juego.
        """
        logger.info("Creando características optimizadas de situación de juego...")
        start_time = time.time()
        
        # Diccionario para nuevas características
        new_features = {}
        
        # Home/Away rendimiento (factor crítico)
        if 'is_home' in df.columns:
            logger.debug("Calculando rendimiento según localización...")
            
            # Estadísticas clave
            key_stats = ['PTS', 'GmSc']
            available_stats = [stat for stat in key_stats if stat in df.columns]
            
            for stat in available_stats:
                # Crear series para almacenar resultados para todos los jugadores
                home_factor_series = pd.Series(index=df.index, dtype=float)
                home_advantage_series = pd.Series(index=df.index, dtype=float)
                
                for player in tqdm(df['Player'].unique(), desc=f"Procesando {stat} por localización"):
                    player_mask = df['Player'] == player
                    if player_mask.sum() <= 5:
                        continue
                        
                    # Rendimiento en casa vs fuera
                    home_mask = df['is_home'] == 1
                    away_mask = ~home_mask
                    
                    if (player_mask & home_mask).sum() >= 3 and (player_mask & away_mask).sum() >= 3:
                        # Calcular promedio actual de rendimiento en casa
                        player_data_home = df[player_mask & home_mask].sort_values('Date')
                        home_stat_values = player_data_home[stat]
                        home_avg = home_stat_values.expanding().mean()
                        
                        # Asignar el promedio evolucional a las filas de casa para este jugador
                        home_factor_series.loc[player_mask & home_mask] = home_avg.values
                        
                        # Calcular diferencia entre rendimiento en casa vs fuera
                        player_home_avg = df[player_mask & home_mask][stat].mean()
                        player_away_avg = df[player_mask & away_mask][stat].mean() 
                        
                        if player_away_avg > 0:
                            home_advantage = (player_home_avg / player_away_avg) - 1
                            # Asignar a todas las filas de este jugador
                            home_advantage_series.loc[player_mask] = home_advantage
                
                # Añadir series completas al diccionario de características
                col_name = f'{stat}_home_factor'
                new_features[col_name] = home_factor_series
                
                col_name = f'{stat}_home_advantage'
                new_features[col_name] = home_advantage_series
        
        # Rendimiento por días de descanso (factor crítico)
        if 'days_rest' in df.columns and 'GmSc' in df.columns:
            logger.debug("Calculando rendimiento según descanso...")
            
            # Categorizar descanso
            rest_categories = pd.cut(
                df['days_rest'], 
                bins=[0, 1, 3, 10], 
                labels=['low', 'medium', 'high']
            )
            
            # Crear serie para almacenar resultados para todos los jugadores
            rest_impact_series = pd.Series(index=df.index, dtype=float)
            
            for player in df['Player'].unique():
                player_mask = df['Player'] == player
                if player_mask.sum() <= 5:
                    continue
                
                # Comparar rendimiento con bajo descanso vs alto descanso
                low_rest = (rest_categories == 'low') & player_mask
                high_rest = (rest_categories == 'high') & player_mask
                
                if low_rest.sum() >= 3 and high_rest.sum() >= 3:
                    low_perf = df.loc[low_rest, 'GmSc'].mean()
                    high_perf = df.loc[high_rest, 'GmSc'].mean()
                    
                    if high_perf > 0:
                        rest_impact = low_perf / high_perf
                        # Asignar a todas las filas de este jugador
                        rest_impact_series.loc[player_mask] = rest_impact
            
            # Añadir serie completa al diccionario de características
            new_features['Rest_Impact'] = rest_impact_series
        
        # Convertir a DataFrame y unir
        features_df = pd.DataFrame(new_features)
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de situación de juego optimizadas: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result
    
    def get_important_features(self):
        """
        Retorna la lista de características generadas después de optimización
        
        Returns:
            Lista de nombres de características
        """
        return self.important_features

    def add_milestone_features(self, df):
        """
        Agrega características para predicción de hitos como double-double y triple-double.
        """
        logger.info("Creando características para hitos de rendimiento (double-double, triple-double)...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_stats = ['PTS', 'TRB', 'AST', 'BLK', 'STL', 'MP', 'GmSc']
        if not all(stat in df.columns for stat in required_stats):
            logger.warning("No se pueden generar características de hitos (faltan estadísticas principales)")
            return df
            
        # Crear columnas para double-double y triple-double
        # Primero, marcar cada estadística >= 10
        stats_10_plus = {}
        for stat in required_stats[:5]:  # Solo PTS, TRB, AST, BLK, STL
            stats_10_plus[f'{stat}_10_plus'] = (df[stat] >= 10).astype(int)
            
        # Crear DataFrame con estas columnas
        milestone_df = pd.DataFrame(stats_10_plus, index=df.index)
        
        # Determinar double-double y triple-double correctamente
        # Un double-double ocurre cuando un jugador tiene 10 o más en exactamente dos categorías
        # Un triple-double ocurre cuando un jugador tiene 10 o más en exactamente tres categorías
        categories_with_10_plus = milestone_df.sum(axis=1)
        milestone_df['Double_Double'] = (categories_with_10_plus >= 2).astype(int)
        milestone_df['Triple_Double'] = (categories_with_10_plus >= 3).astype(int)
        
        # Agregar las columnas originales al DataFrame principal para referencia
        df['Double_Double'] = milestone_df['Double_Double']
        df['Triple_Double'] = milestone_df['Triple_Double']
        
        # Crear diccionario para nuevas características
        new_features = {}
            
        # Procesar cada jugador
        for player in df['Player'].unique():
            player_mask = df['Player'] == player
            player_data = df[player_mask].sort_values('Date')
            
            if len(player_data) < 5:  # Necesitamos al menos 5 juegos
                continue
                
            # Calcular USG% si no existe
            if 'USG%' not in player_data.columns:
                minutes_played = player_data['MP'].clip(lower=1)
                team_possessions = player_data['MP'].sum() * 0.96  # Estimación de posesiones
                player_data['USG%'] = ((player_data['FGA'] + 0.44 * player_data['FTA'] + player_data['TOV']) / minutes_played * 100 / team_possessions).clip(0, 100)
            
            # Historial de Double-Double y Triple-Double
            player_milestone_data = milestone_df[player_mask]
            
            # Double-Double
            dd_history = player_milestone_data['Double_Double'].expanding().sum().fillna(0.01)
            new_features['Double_Double_History'] = pd.Series(index=df.index)
            new_features['Double_Double_History'].loc[player_mask] = dd_history
            
            # Últimos N juegos
            for n in [5, 10]:
                dd_last_n = player_milestone_data['Double_Double'].rolling(n, min_periods=1).sum().fillna(0.01)
                new_features[f'Double_Double_Last_{n}'] = pd.Series(index=df.index)
                new_features[f'Double_Double_Last_{n}'].loc[player_mask] = dd_last_n
            
            # Rate y streak
            dd_rate = player_milestone_data['Double_Double'].expanding().mean().fillna(0.01)
            new_features['Double_Double_Rate'] = pd.Series(index=df.index)
            new_features['Double_Double_Rate'].loc[player_mask] = dd_rate
            
            # Calcular streak con manejo de NaN
            try:
                dd_streak = player_milestone_data['Double_Double'].groupby(
                    (player_milestone_data['Double_Double'] != player_milestone_data['Double_Double'].shift()).cumsum()
                ).cumcount() + 1
                new_features['Double_Double_Streak'] = pd.Series(index=df.index)
                new_features['Double_Double_Streak'].loc[player_mask] = dd_streak.fillna(0.01)
            except Exception as e:
                logger.warning(f"Error calculando Double_Double_Streak para {player}: {e}")
                # Usar valor aleatorio pequeño en caso de error
                new_features['Double_Double_Streak'] = pd.Series(index=df.index)
                new_features['Double_Double_Streak'].loc[player_mask] = np.random.uniform(0.01, 0.05, size=len(player_mask[player_mask]))
            
            # Triple-Double
            td_history = player_milestone_data['Triple_Double'].expanding().sum().fillna(0.01)
            new_features['Triple_Double_History'] = pd.Series(index=df.index)
            new_features['Triple_Double_History'].loc[player_mask] = td_history
            
            # Usar max() con manejo de NaN
            max_td = td_history.max() if not td_history.empty else 0.01
            new_features['Career_Triple_Doubles'] = pd.Series(index=df.index)
            new_features['Career_Triple_Doubles'].loc[player_mask] = max_td
            
            # Rate y streak para Triple-Double
            td_rate = player_milestone_data['Triple_Double'].expanding().mean().fillna(0.01)
            new_features['Triple_Double_Rate'] = pd.Series(index=df.index)
            new_features['Triple_Double_Rate'].loc[player_mask] = td_rate
            
            # Calcular streak con manejo de NaN
            try:
                td_streak = player_milestone_data['Triple_Double'].groupby(
                    (player_milestone_data['Triple_Double'] != player_milestone_data['Triple_Double'].shift()).cumsum()
                ).cumcount() + 1
                new_features['Triple_Double_Streak'] = pd.Series(index=df.index)
                new_features['Triple_Double_Streak'].loc[player_mask] = td_streak.fillna(0.01)
            except Exception as e:
                logger.warning(f"Error calculando Triple_Double_Streak para {player}: {e}")
                # Usar valor aleatorio pequeño en caso de error
                new_features['Triple_Double_Streak'] = pd.Series(index=df.index)
                new_features['Triple_Double_Streak'].loc[player_mask] = np.random.uniform(0.01, 0.05, size=len(player_mask[player_mask]))
            
            # Promedios de estadísticas clave con manejo de NaN
            for stat in ['PTS', 'TRB', 'AST', 'BLK', 'STL']:
                # Verificar que la estadística exista en los datos del jugador
                if stat not in player_data.columns:
                    logger.warning(f"Estadística {stat} no encontrada para {player}")
                    continue
                    
                # Inicializar Series para los promedios si no existen
                for window in [5, 10]:
                    avg_name = f'{stat}_{window}_avg'
                    if avg_name not in new_features:
                        new_features[avg_name] = pd.Series(index=df.index)
                
                # Calcular promedios móviles con manejo seguro de NaN
                try:
                    # Asegurar que la estadística sea numérica
                    stat_values = pd.to_numeric(player_data[stat], errors='coerce').fillna(0)
                    
                    for window in [5, 10]:
                        avg_name = f'{stat}_{window}_avg'
                        # Calcular promedio móvil con min_periods=1
                        rolling_avg = stat_values.rolling(window, min_periods=1).mean()
                        
                        # Rellenar valores NaN con el promedio o un valor predeterminado
                        if rolling_avg.isna().any():
                            default_value = stat_values.mean() if not stat_values.empty else 0.01
                            rolling_avg = rolling_avg.fillna(default_value)
                        
                        # Asignar valores al DataFrame principal de manera segura
                        for idx, val in zip(player_data.index, rolling_avg):
                            if idx in df.index:
                                new_features[avg_name].loc[idx] = val
                except Exception as e:
                    logger.warning(f"Error calculando promedios para {stat} del jugador {player}: {e}")
                    # Usar valores predeterminados en caso de error
                    for window in [5, 10]:
                        avg_name = f'{stat}_{window}_avg'
                        new_features[avg_name].loc[player_mask] = player_data[stat].mean() if stat in player_data.columns and not player_data[stat].empty else 0.01
            
            # Probabilidades por estadística
            for stat in ['PTS', 'TRB', 'AST', 'BLK', 'STL']:
                # Verificar que la estadística exista en los datos del jugador
                if stat not in player_data.columns:
                    logger.warning(f"Estadística {stat} no encontrada para {player} al calcular probabilidades")
                    continue
                    
                # Inicializar Series para las probabilidades si no existen
                prob_name = f'{stat}_p50_prob'
                if prob_name not in new_features:
                    new_features[prob_name] = pd.Series(index=df.index)
                
                try:
                    # Asegurar que la estadística sea numérica
                    stat_values = pd.to_numeric(player_data[stat], errors='coerce').fillna(0)
                    
                    # Calcular probabilidad (1 si supera 10, 0 si no)
                    binary_result = (stat_values >= 10).astype(float)
                    
                    # Calcular media móvil con manejo seguro de NaN
                    prob = binary_result.rolling(10, min_periods=1).mean().fillna(0.5)
                    
                    # Asignar valores al DataFrame principal de manera segura
                    for idx, val in zip(player_data.index, prob):
                        if idx in df.index:
                            new_features[prob_name].loc[idx] = val
                except Exception as e:
                    logger.warning(f"Error calculando probabilidades para {stat} del jugador {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features[prob_name].loc[player_mask] = 0.5  # 50% por defecto
            
            # Probabilidad y consistencia de Double-Double y Triple-Double
            for milestone in ['Double_Double', 'Triple_Double']:
                # Verificar que el hito exista en los datos del jugador
                if milestone not in player_milestone_data.columns:
                    logger.warning(f"Hito {milestone} no encontrado para {player}")
                    continue
                    
                # Inicializar Series para las probabilidades si no existen
                prob_name = f'{milestone}_Probability'
                if prob_name not in new_features:
                    new_features[prob_name] = pd.Series(index=df.index)
                
                try:
                    # Calcular probabilidad con manejo seguro de NaN
                    milestone_values = player_milestone_data[milestone].astype(float)
                    prob = milestone_values.rolling(10, min_periods=1).mean().fillna(0.5)
                    
                    # Asignar valores al DataFrame principal de manera segura
                    for idx, val in zip(player_milestone_data.index, prob):
                        if idx in df.index:
                            new_features[prob_name].loc[idx] = val
                except Exception as e:
                    logger.warning(f"Error calculando probabilidad para {milestone} del jugador {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features[prob_name].loc[player_mask] = 0.5  # 50% por defecto
                
                # Consistencia
                consistency_name = f'{milestone}_Consistency'
                if consistency_name not in new_features:
                    new_features[consistency_name] = pd.Series(index=df.index)
                
                try:
                    # Calcular consistencia con manejo seguro de NaN
                    milestone_values = player_milestone_data[milestone].astype(float)
                    # Usar std con min_periods=1 para calcular incluso con pocos datos
                    std_values = milestone_values.rolling(10, min_periods=1).std().fillna(0)
                    consistency = 1 - std_values
                    
                    # Asignar valores al DataFrame principal de manera segura
                    for idx, val in zip(player_milestone_data.index, consistency):
                        if idx in df.index:
                            new_features[consistency_name].loc[idx] = val
                except Exception as e:
                    logger.warning(f"Error calculando consistencia para {milestone} del jugador {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features[consistency_name].loc[player_mask] = 0.8  # Valor predeterminado razonable
            
            # Factores contextuales
            # Inicializar Series para days_rest si no existe
            if 'days_rest' not in new_features:
                new_features['days_rest'] = pd.Series(index=df.index)
                
            # Inicializar Series para MP_last_10_games si no existe
            if 'MP_last_10_games' not in new_features:
                new_features['MP_last_10_games'] = pd.Series(index=df.index)
            
            # Calcular days_rest con manejo seguro de NaN
            if 'days_rest' in player_data.columns:
                try:
                    # Asegurar que days_rest sea numérico
                    days_rest_values = pd.to_numeric(player_data['days_rest'], errors='coerce').fillna(2)
                    
                    # Asignar valores al DataFrame principal de manera segura
                    for idx, val in zip(player_data.index, days_rest_values):
                        if idx in df.index:
                            new_features['days_rest'].loc[idx] = val
                except Exception as e:
                    logger.warning(f"Error calculando days_rest para {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features['days_rest'].loc[player_mask] = 2  # Valor predeterminado razonable
            
            # Calcular MP_last_10_games con manejo seguro de NaN
            if 'MP' in player_data.columns:
                try:
                    # Asegurar que MP sea numérico
                    mp_values = pd.to_numeric(player_data['MP'], errors='coerce').fillna(0)
                    
                    # Calcular suma móvil con min_periods=1
                    mp_last_10 = mp_values.rolling(10, min_periods=1).sum().fillna(mp_values.mean() * 5)
                    
                    # Asignar valores al DataFrame principal de manera segura
                    for idx, val in zip(player_data.index, mp_last_10):
                        if idx in df.index:
                            new_features['MP_last_10_games'].loc[idx] = val
                except Exception as e:
                    logger.warning(f"Error calculando MP_last_10_games para {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features['MP_last_10_games'].loc[player_mask] = 120  # Valor predeterminado razonable
                
                # Inicializar índice de fatiga para todos los jugadores si no existe
                if 'Fatigue_Index' not in new_features:
                    new_features['Fatigue_Index'] = pd.Series(0.01, index=df.index)
                
                # Índice de fatiga - cálculo mejorado
                try:
                    # Asegurar que days_rest esté correctamente rellenado
                    days_rest = player_data['days_rest'].fillna(3).clip(0, 7)
                    
                    # Convertir is_b2b a float y rellenar valores faltantes
                    is_b2b = player_data['is_b2b'].astype(float).fillna(0)
                    
                    # Normalizar minutos jugados con manejo de NaN
                    # Usar min_periods=1 para calcular incluso con pocos datos
                    mp_last_10_filled = mp_last_10.fillna(player_data['MP'].mean() * 5)  # Estimación razonable
                    
                    # Limitar el valor máximo para evitar valores extremos
                    mp_normalized = (mp_last_10_filled / (48 * 10)).clip(0, 1)
                    
                    # Calcular índice de fatiga con componentes bien normalizados
                    if len(days_rest) == len(is_b2b) == len(mp_normalized):
                        # Crear una Serie con el mismo índice que los datos del jugador
                        fatigue = pd.Series(
                            0.4 * (1 - days_rest / 7).values +
                            0.4 * is_b2b.values +
                            0.2 * mp_normalized.values,
                            index=player_data.index
                        ).clip(0, 1)  # Asegurar que esté entre 0 y 1
                        
                        # Asignar al DataFrame principal
                        for idx in player_mask[player_mask].index:
                            if idx in df.index and idx in fatigue.index:
                                new_features['Fatigue_Index'].loc[idx] = fatigue.loc[idx]
                    else:
                        # Si los componentes tienen diferentes longitudes, usar un valor predeterminado
                        new_features['Fatigue_Index'].loc[player_mask] = 0.3
                except Exception as e:
                    logger.warning(f"Error calculando Fatigue_Index para {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features['Fatigue_Index'].loc[player_mask] = 0.3  # Valor moderado por defecto
            
            # Inicializar componentes físicos para todos los jugadores si no existen
            for feature in ['Size_Component', 'Density_Component', 'Proportion_Component', 'Athletic_Versatility_Index']:
                if feature not in new_features:
                    new_features[feature] = pd.Series(0.01, index=df.index)
            
            # Factores físicos
            try:
                if all(col in player_data.columns for col in ['Height_Inches', 'Weight', 'BMI']):
                    # Normalizar métricas físicas con manejo de NaN
                    height_inches = player_data['Height_Inches'].fillna(70)  # Valor promedio
                    weight = player_data['Weight'].fillna(180)  # Valor promedio
                    bmi = player_data['BMI'].fillna(25)  # Valor promedio
                    
                    height_factor = (height_inches - 70) / 10
                    weight_factor = (weight - 180) / 50
                    bmi_factor = (bmi - 25) / 5
                    
                    # Componentes físicos
                    size_component = (0.7 * height_factor + 0.3 * weight_factor).clip(-3, 3)
                    new_features['Size_Component'].loc[player_mask] = size_component
                    
                    density_component = (0.6 * bmi_factor + 0.4 * weight_factor).clip(-3, 3)
                    new_features['Density_Component'].loc[player_mask] = density_component
                    
                    proportion_component = (0.5 * height_factor + 0.5 * bmi_factor).clip(-3, 3)
                    new_features['Proportion_Component'].loc[player_mask] = proportion_component
                    
                    # Índice de versatilidad atlética con manejo de NaN
                    mp = player_data['MP'].fillna(player_data['MP'].mean() if not player_data['MP'].empty else 24)
                    versatility = (
                        0.3 * height_factor +
                        0.3 * weight_factor +
                        0.2 * bmi_factor +
                        0.2 * (mp / 48)
                    ).clip(-3, 3)
                    new_features['Athletic_Versatility_Index'].loc[player_mask] = versatility
                else:
                    # Si faltan columnas, usar valores predeterminados
                    logger.warning(f"Faltan columnas biométricas para {player}, usando valores predeterminados")
                    new_features['Size_Component'].loc[player_mask] = 0.01
                    new_features['Density_Component'].loc[player_mask] = 0.01
                    new_features['Proportion_Component'].loc[player_mask] = 0.01
                    new_features['Athletic_Versatility_Index'].loc[player_mask] = 0.01
            except Exception as e:
                logger.warning(f"Error calculando componentes físicos para {player}: {e}")
                # Usar valores predeterminados en caso de error
                new_features['Size_Component'].loc[player_mask] = 0.01
                new_features['Density_Component'].loc[player_mask] = 0.01
                new_features['Proportion_Component'].loc[player_mask] = 0.01
                new_features['Athletic_Versatility_Index'].loc[player_mask] = 0.01
            
            # Inicializar opp_defensive_rating si no existe
            if 'opp_defensive_rating' not in new_features:
                new_features['opp_defensive_rating'] = pd.Series(0.01, index=df.index)
            
            # Factores de oponente y matchup
            try:
                if 'Opp' in player_data.columns:
                    # Rating defensivo del oponente
                    opp_def_stats = df.groupby('Opp').agg({
                        'STL': 'mean',
                        'BLK': 'mean',
                        'TOV': 'mean'
                    }).reset_index()
                    
                    # Rellenar valores NaN con promedios
                    for col in ['STL', 'BLK', 'TOV']:
                        if opp_def_stats[col].isna().any():
                            opp_def_stats[col] = opp_def_stats[col].fillna(opp_def_stats[col].mean())
                    
                    # Calcular rating defensivo con manejo de NaN
                    opp_def_stats['opp_defensive_rating'] = (
                        opp_def_stats['STL'].fillna(0) * 0.4 +
                        opp_def_stats['BLK'].fillna(0) * 0.3 +
                        opp_def_stats['TOV'].fillna(0) * 0.3
                    )
                    
                    # Merge con datos del jugador
                    player_data = pd.merge(
                        player_data,
                        opp_def_stats[['Opp', 'opp_defensive_rating']],
                        on='Opp',
                        how='left'
                    )
                    
                    # Rellenar valores NaN después del merge
                    player_data['opp_defensive_rating'] = player_data['opp_defensive_rating'].fillna(0.01)
                    
                    # Asignar al DataFrame principal
                    new_features['opp_defensive_rating'].loc[player_mask] = player_data['opp_defensive_rating']
                else:
                    # Si no hay columna Opp, usar valores predeterminados
                    new_features['opp_defensive_rating'].loc[player_mask] = 0.01
            except Exception as e:
                logger.warning(f"Error calculando opp_defensive_rating para {player}: {e}")
                # Usar valor predeterminado en caso de error
                new_features['opp_defensive_rating'].loc[player_mask] = 0.01
                
                # Inicializar opp_position_defense_rating si no existe
                if 'opp_position_defense_rating' not in new_features:
                    new_features['opp_position_defense_rating'] = pd.Series(0.01, index=df.index)
                
                # Rating defensivo por posición
                try:
                    if 'Pos' in player_data.columns:
                        opp_pos_stats = df.groupby(['Opp', 'Pos']).agg({
                            'STL': 'mean',
                            'BLK': 'mean'
                        }).reset_index()
                        
                        # Rellenar valores NaN con promedios
                        for col in ['STL', 'BLK']:
                            if opp_pos_stats[col].isna().any():
                                opp_pos_stats[col] = opp_pos_stats[col].fillna(opp_pos_stats[col].mean())
                        
                        # Calcular rating defensivo por posición con manejo de NaN
                        opp_pos_stats['opp_position_defense_rating'] = (
                            opp_pos_stats['STL'].fillna(0) * 0.6 +
                            opp_pos_stats['BLK'].fillna(0) * 0.4
                        )
                        
                        # Merge con datos del jugador
                        player_data = pd.merge(
                            player_data,
                            opp_pos_stats[['Opp', 'Pos', 'opp_position_defense_rating']],
                            on=['Opp', 'Pos'],
                            how='left'
                        )
                        
                        # Rellenar valores NaN después del merge
                        player_data['opp_position_defense_rating'] = player_data['opp_position_defense_rating'].fillna(0.01)
                        
                        # Asignar al DataFrame principal
                        new_features['opp_position_defense_rating'].loc[player_mask] = player_data['opp_position_defense_rating']
                    else:
                        # Si no hay columna Pos, usar valores predeterminados
                        new_features['opp_position_defense_rating'].loc[player_mask] = 0.01
                except Exception as e:
                    logger.warning(f"Error calculando opp_position_defense_rating para {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features['opp_position_defense_rating'].loc[player_mask] = 0.01
                
                # Inicializar matchup_advantage_rating si no existe
                if 'matchup_advantage_rating' not in new_features:
                    new_features['matchup_advantage_rating'] = pd.Series(0.01, index=df.index)
                
                # Ventaja de matchup
                try:
                    if 'Height_Inches' in player_data.columns:
                        # Calcular altura promedio por oponente y posición
                        opp_height = df.groupby(['Opp', 'Pos'])['Height_Inches'].mean().reset_index()
                        
                        # Merge con datos del jugador
                        player_data = pd.merge(
                            player_data,
                            opp_height,
                            on=['Opp', 'Pos'],
                            how='left',
                            suffixes=('', '_opp')
                        )
                        
                        # Rellenar valores NaN después del merge
                        if 'Height_Inches_opp' not in player_data.columns or player_data['Height_Inches_opp'].isna().all():
                            # Si no hay datos de altura del oponente, usar un valor predeterminado
                            player_data['Height_Inches_opp'] = player_data['Height_Inches']
                        else:
                            # Rellenar valores faltantes con la media
                            player_data['Height_Inches_opp'] = player_data['Height_Inches_opp'].fillna(player_data['Height_Inches_opp'].mean())
                        
                        # Calcular ventaja de altura con manejo de NaN
                        height_advantage = (player_data['Height_Inches'] - player_data['Height_Inches_opp']) / 2
                        
                        # Asignar al DataFrame principal con clip para evitar valores extremos
                        new_features['matchup_advantage_rating'].loc[player_mask] = height_advantage.clip(-3, 3).fillna(0.01)
                    else:
                        # Si no hay columna Height_Inches, usar valores predeterminados
                        new_features['matchup_advantage_rating'].loc[player_mask] = 0.01
                except Exception as e:
                    logger.warning(f"Error calculando matchup_advantage_rating para {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features['matchup_advantage_rating'].loc[player_mask] = 0.01
                
                # Inicializar historical_matchup_performance si no existe
                if 'historical_matchup_performance' not in new_features:
                    new_features['historical_matchup_performance'] = pd.Series(0.01, index=df.index)
                
                # Rendimiento histórico contra oponente
                try:
                    if 'GmSc' in player_data.columns and 'Opp' in player_data.columns:
                        # Calcular rendimiento histórico por oponente
                        hist_perf = player_data.groupby('Opp')['GmSc'].mean()
                        
                        # Mapear valores al DataFrame del jugador
                        mapped_values = player_data['Opp'].map(hist_perf)
                        
                        # Rellenar valores NaN
                        if mapped_values.isna().all():
                            # Si todos son NaN, usar el promedio general de GmSc del jugador
                            default_value = player_data['GmSc'].mean() if not player_data['GmSc'].empty else 0.01
                            mapped_values = pd.Series(default_value, index=mapped_values.index)
                        else:
                            # Rellenar valores NaN con la media
                            mapped_values = mapped_values.fillna(mapped_values.mean() if not mapped_values.empty else 0.01)
                        
                        # Asignar al DataFrame principal
                        new_features['historical_matchup_performance'].loc[player_mask] = mapped_values
                    else:
                        # Si faltan columnas necesarias, usar valores predeterminados
                        new_features['historical_matchup_performance'].loc[player_mask] = 0.01
                except Exception as e:
                    logger.warning(f"Error calculando historical_matchup_performance para {player}: {e}")
                    # Usar valor predeterminado en caso de error
                    new_features['historical_matchup_performance'].loc[player_mask] = 0.01
            
            # Inicializar stat_distribution_evenness si no existe
            if 'stat_distribution_evenness' not in new_features:
                new_features['stat_distribution_evenness'] = pd.Series(0.01, index=df.index)
            
            # Factores de situación
            # Distribución de estadísticas
            try:
                # Verificar que todas las columnas necesarias estén presentes
                required_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
                if all(stat in player_data.columns for stat in required_stats):
                    # Rellenar valores NaN en las estadísticas
                    stats_df = player_data[required_stats].fillna(0)
                    
                    # Calcular desviación estándar y media
                    stats_std = stats_df.std(axis=1)
                    stats_mean = stats_df.mean(axis=1)
                    
                    # Calcular uniformidad con manejo de división por cero
                    evenness = 1 - (stats_std / (stats_mean + 1e-6)).clip(0, 1)
                    
                    # Rellenar valores NaN en el resultado
                    evenness = evenness.fillna(0.5)  # Valor neutral para uniformidad
                    
                    # Asignar al DataFrame principal
                    new_features['stat_distribution_evenness'].loc[player_mask] = evenness
                else:
                    # Si faltan columnas necesarias, usar valores predeterminados
                    new_features['stat_distribution_evenness'].loc[player_mask] = 0.5
            except Exception as e:
                logger.warning(f"Error calculando stat_distribution_evenness para {player}: {e}")
                # Usar valor predeterminado en caso de error
                new_features['stat_distribution_evenness'].loc[player_mask] = 0.5
            
            # Inicializar role_in_team si no existe
            if 'role_in_team' not in new_features:
                new_features['role_in_team'] = pd.Series(0.01, index=df.index)
            
            # Rol en el equipo
            try:
                # Verificar que las columnas necesarias estén presentes
                if 'USG%' in player_data.columns and 'MP' in player_data.columns and 'GmSc' in player_data.columns:
                    # Rellenar valores NaN
                    usg_pct = player_data['USG%'].fillna(player_data['USG%'].mean() if not player_data['USG%'].empty else 15)
                    mp = player_data['MP'].fillna(player_data['MP'].mean() if not player_data['MP'].empty else 20)
                    gmsc = player_data['GmSc'].fillna(player_data['GmSc'].mean() if not player_data['GmSc'].empty else 10)
                    
                    # Calcular puntuación de rol
                    role_score = (
                        0.4 * usg_pct / 100 +
                        0.3 * (mp / 48) +
                        0.3 * gmsc / 40
                    ).clip(0, 1)
                    
                    # Asignar al DataFrame principal
                    new_features['role_in_team'].loc[player_mask] = role_score
                else:
                    # Si faltan columnas necesarias, usar valores alternativos
                    if 'MP' in player_data.columns:
                        # Si al menos tenemos minutos jugados, usar eso como aproximación
                        mp_ratio = (player_data['MP'].fillna(20) / 48).clip(0, 1)
                        new_features['role_in_team'].loc[player_mask] = mp_ratio
                    else:
                        # Valor predeterminado si no hay datos suficientes
                        new_features['role_in_team'].loc[player_mask] = 0.3  # Rol moderado por defecto
            except Exception as e:
                logger.warning(f"Error calculando role_in_team para {player}: {e}")
                # Usar valor predeterminado en caso de error
                new_features['role_in_team'].loc[player_mask] = 0.3
            
            # Inicializar team_injury_impact si no existe
            if 'team_injury_impact' not in new_features:
                new_features['team_injury_impact'] = pd.Series(0.01, index=df.index)
            
            # Impacto de lesiones del equipo (aproximado por minutos disponibles)
            try:
                if 'Team' in player_data.columns and 'MP' in player_data.columns and len(player_data) > 0:
                    # Obtener el equipo del jugador
                    team = player_data['Team'].iloc[0]
                    
                    # Calcular minutos totales del equipo por fecha
                    team_data = df[df['Team'] == team]
                    if not team_data.empty:
                        team_mp = team_data.groupby('Date')['MP'].sum()
                        
                        # Calcular minutos esperados (aproximación)
                        # Número de jugadores por fecha * 48 minutos / 5 jugadores en cancha
                        expected_mp = len(team_data.groupby(['Date', 'Player'])) * 48 / 5
                        
                        if expected_mp > 0:
                            # Calcular salud del equipo con manejo de división por cero
                            team_health = (team_mp / expected_mp).clip(0.5, 1.5)
                            
                            # Mapear valores a las fechas del jugador
                            player_dates = player_data['Date']
                            mapped_health = player_dates.map(team_health)
                            
                            # Rellenar valores NaN
                            mapped_health = mapped_health.fillna(1.0)  # Valor neutral por defecto
                            
                            # Asignar al DataFrame principal
                            new_features['team_injury_impact'].loc[player_mask] = mapped_health
                        else:
                            # Si no se pueden calcular minutos esperados
                            new_features['team_injury_impact'].loc[player_mask] = 1.0
                    else:
                        # Si no hay datos del equipo
                        new_features['team_injury_impact'].loc[player_mask] = 1.0
                else:
                    # Si faltan columnas necesarias
                    new_features['team_injury_impact'].loc[player_mask] = 1.0
            except Exception as e:
                logger.warning(f"Error calculando team_injury_impact para {player}: {e}")
                # Usar valor predeterminado en caso de error
                new_features['team_injury_impact'].loc[player_mask] = 1.0
            
            # Inicializar usage_in_close_games si no existe
            if 'usage_in_close_games' not in new_features:
                new_features['usage_in_close_games'] = pd.Series(0.01, index=df.index)
            
            # Uso en juegos cerrados
            try:
                if 'point_diff' in player_data.columns:
                    # Identificar juegos cerrados (diferencia de 5 puntos o menos)
                    close_games = abs(player_data['point_diff']) <= 5
                    
                    if close_games.any():
                        # Si hay juegos cerrados y tenemos USG%
                        if 'USG%' in player_data.columns:
                            close_usg = player_data.loc[close_games, 'USG%'].mean()
                            if not pd.isna(close_usg):
                                # Normalizar a escala 0-1
                                new_features['usage_in_close_games'].loc[player_mask] = close_usg / 100
                            else:
                                # Si no hay valor válido, usar el USG% promedio general
                                avg_usg = player_data['USG%'].mean()
                                new_features['usage_in_close_games'].loc[player_mask] = avg_usg / 100 if not pd.isna(avg_usg) else 0.2
                        else:
                            # Si no hay USG%, usar MP como aproximación
                            if 'MP' in player_data.columns:
                                close_mp = player_data.loc[close_games, 'MP'].mean()
                                avg_mp = player_data['MP'].mean()
                                if not pd.isna(close_mp) and not pd.isna(avg_mp) and avg_mp > 0:
                                    # Calcular ratio de minutos en juegos cerrados vs promedio
                                    mp_ratio = (close_mp / avg_mp).clip(0.5, 1.5)
                                    new_features['usage_in_close_games'].loc[player_mask] = mp_ratio / 3 + 0.15  # Escalar a un rango razonable
                                else:
                                    # Valor predeterminado si no hay datos suficientes
                                    new_features['usage_in_close_games'].loc[player_mask] = 0.2
                            else:
                                # Valor predeterminado si no hay datos suficientes
                                new_features['usage_in_close_games'].loc[player_mask] = 0.2
                    else:
                        # Si no hay juegos cerrados, usar valor predeterminado
                        new_features['usage_in_close_games'].loc[player_mask] = 0.2
                else:
                    # Si no hay columna point_diff, usar valor predeterminado
                    new_features['usage_in_close_games'].loc[player_mask] = 0.2
            except Exception as e:
                logger.warning(f"Error calculando usage_in_close_games para {player}: {e}")
                # Usar valor predeterminado en caso de error
                new_features['usage_in_close_games'].loc[player_mask] = 0.2
        
        # Unir las nuevas características
        result = pd.concat([df, pd.DataFrame(new_features)], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = pd.DataFrame(new_features).isnull().mean().mean() * 100
        logger.info(f"Características de hitos: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def _safe_round(self, value, decimals=3):
        """
        Redondea un valor de manera segura, manejando casos donde el valor
        podría ser una serie, un array u otro tipo no directamente convertible a float.
        
        Args:
            value: El valor a redondear
            decimals: Número de decimales para el redondeo
            
        Returns:
            Valor redondeado o 0.0 en caso de error
        """
        try:
            # Caso 1: El valor tiene método .item() (típico de arrays de NumPy o Series de pandas)
            if hasattr(value, 'item'):
                try:
                    return round(value.item(), decimals)
                except (ValueError, TypeError):
                    # El valor es un array con múltiples elementos
                    if hasattr(value, 'size') and value.size > 0:
                        # Tomar el primer elemento
                        return round(float(value.flat[0]), decimals)
                    return 0.0
                    
            # Caso 2: El valor es un array-like (lista, tupla, etc.)
            elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                # Tomar el primer elemento
                first_val = value[0]
                # Recursivamente aplicar safe_round al primer elemento
                return self._safe_round(first_val, decimals)
                
            # Caso 3: El valor es un objeto Series o DataFrame anidado
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                if not value.empty:
                    # Tomar el primer valor no nulo
                    first_non_null = value.dropna().iloc[0] if not value.dropna().empty else 0.0
                    return self._safe_round(first_non_null, decimals)
                return 0.0
                
            # Caso 4: El valor es un diccionario
            elif isinstance(value, dict) and value:
                # Tomar el primer valor
                first_key = next(iter(value))
                return self._safe_round(value[first_key], decimals)
                
            # Caso 5: Valor escalar simple
            else:
                # Intentar convertir directamente a float
                return round(float(value), decimals)
                
        except (TypeError, ValueError, IndexError, KeyError, AttributeError):
            # En caso de cualquier error, devolver 0.0
            return 0.0

    def add_team_prediction_features(self, df):
        """
        Agrega características optimizadas específicas para predicción de victoria de equipo.
        """
        logger.info("Creando características para predicción de victoria de equipo...")
        start_time = time.time()
    
        # Verificar columnas necesarias
        required_cols = ['Team', 'Date']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de predicción de equipo (faltan columnas necesarias)")
            return df
    
        # Asegurar que las columnas necesarias existan
        if 'team_score' not in df.columns and 'PTS' in df.columns:
            df['team_score'] = df['PTS']
        if 'opp_score' not in df.columns and 'opp_PTS' in df.columns:
            df['opp_score'] = df['opp_PTS']
        if 'is_win' not in df.columns and 'Win' in df.columns:
            df['is_win'] = df['Win']
        if 'Opp' not in df.columns and 'opponent' in df.columns:
            df['Opp'] = df['opponent']
    
        # Crear un diccionario para almacenar las nuevas características
        new_features = {}
    
        try:
            # Crear DataFrame a nivel de equipo
            team_df = df.groupby(['Team', 'Date']).agg({
                'team_score': 'mean',
                'opp_score': 'mean',
                'is_win': 'first',
                'is_home': 'first',
                'Opp': 'first'
            }).reset_index()
        
            # Procesar cada equipo
            for team in team_df['Team'].unique():
                # Obtener índices en el DataFrame original para este equipo
                team_idx = df[df['Team'] == team].index
                
                # Filtrar datos para este equipo
                team_mask = team_df['Team'] == team
                team_data = team_df[team_mask].sort_values('Date')
            
                if len(team_data) < 5:  # Necesitamos al menos 5 juegos
                    continue
            
                # Promedios móviles y tendencias
                for window in [5, 10]:
                    # Crear Series con el índice del DataFrame original
                    if f'Team_Points_avg_{window}' not in new_features:
                        new_features[f'Team_Points_avg_{window}'] = pd.Series(index=df.index)
                    if f'Opp_Points_avg_{window}' not in new_features:
                        new_features[f'Opp_Points_avg_{window}'] = pd.Series(index=df.index)
                    if f'point_diff_avg_{window}' not in new_features:
                        new_features[f'point_diff_avg_{window}'] = pd.Series(index=df.index)
                    if f'win_pct_{window}' not in new_features:
                        new_features[f'win_pct_{window}'] = pd.Series(index=df.index)
                    
                    # Puntos anotados y recibidos
                    team_pts_avg = team_data['team_score'].rolling(window, min_periods=1).mean().values
                    opp_pts_avg = team_data['opp_score'].rolling(window, min_periods=1).mean().values
                
                    # Diferencial de puntos
                    point_diff = team_data['team_score'] - team_data['opp_score']
                    diff_avg = point_diff.rolling(window, min_periods=1).mean().values
                
                    # Porcentaje de victorias
                    win_pct = team_data['is_win'].rolling(window, min_periods=1).mean().values
                    
                    # Mapear valores a las filas correspondientes del DataFrame original
                    # Verificar que los índices existan en df.index antes de asignar
                    valid_team_idx = [idx for idx in team_idx if idx in df.index]
                    
                    # Solo procesar índices válidos
                    for i, idx in enumerate(valid_team_idx):
                        if i < len(team_pts_avg):
                            new_features[f'Team_Points_avg_{window}'].loc[idx] = team_pts_avg[i]
                            new_features[f'Opp_Points_avg_{window}'].loc[idx] = opp_pts_avg[i]
                            new_features[f'point_diff_avg_{window}'].loc[idx] = diff_avg[i]
                            new_features[f'win_pct_{window}'].loc[idx] = win_pct[i]
                
                # Rachas
                win_streak = team_data['is_win'].copy()
                win_streak = win_streak.groupby((win_streak != win_streak.shift()).cumsum()).cumcount() + 1
                win_streak.loc[~team_data['is_win']] *= -1  # Negativo para rachas de derrota
            
                if 'win_streak' not in new_features:
                    new_features['win_streak'] = pd.Series(index=df.index)
                
                # Enfoque alternativo para win_streak que evita problemas de índice
                # Crear un diccionario de fecha -> valor de racha
                win_streak_dict = {date: streak for date, streak in zip(team_data['Date'], win_streak)}
                
                # Asignar valores directamente a las filas del equipo usando fechas como clave
                for idx in df[df['Team'] == team].index:
                    if idx in df.index and df.loc[idx, 'Date'] in win_streak_dict:
                        try:
                            new_features['win_streak'].at[idx] = win_streak_dict[df.loc[idx, 'Date']]
                        except Exception:
                            # Ignorar errores individuales
                            continue
            
                # Rendimiento en casa/fuera
                if 'home_win_pct' not in new_features:
                    new_features['home_win_pct'] = pd.Series(index=df.index, data=0.5)
                if 'away_win_pct' not in new_features:
                    new_features['away_win_pct'] = pd.Series(index=df.index, data=0.5)
            
                # Enfoque alternativo para home_win_pct que evita problemas de índice
                home_mask = team_data['is_home'] == 1
                if home_mask.any():
                    # Crear un diccionario de fecha -> porcentaje de victorias en casa
                    home_dates = team_data.loc[home_mask, 'Date'].values
                    home_win_pct_values = team_data.loc[home_mask, 'is_win'].expanding().mean().values
                    home_win_dict = {date: pct for date, pct in zip(home_dates, home_win_pct_values)}
                    
                    # Asignar valores directamente usando fechas como clave
                    for idx in df[(df['Team'] == team) & (df['is_home'] == 1)].index:
                        if idx in df.index and df.loc[idx, 'Date'] in home_win_dict:
                            try:
                                new_features['home_win_pct'].at[idx] = home_win_dict[df.loc[idx, 'Date']]
                            except Exception:
                                # Ignorar errores individuales
                                continue
            
                # Calcular para juegos fuera
                away_mask = team_data['is_home'] == 0
                if away_mask.any():
                    # Crear un diccionario de fecha -> porcentaje de victorias fuera
                    away_dates = team_data.loc[away_mask, 'Date'].values
                    away_win_pct_values = team_data.loc[away_mask, 'is_win'].expanding().mean().values
                    away_win_dict = {date: pct for date, pct in zip(away_dates, away_win_pct_values)}
                    
                    # Asignar valores directamente usando fechas como clave
                    for idx in df[(df['Team'] == team) & (df['is_home'] == 0)].index:
                        if idx in df.index and df.loc[idx, 'Date'] in away_win_dict:
                            try:
                                new_features['away_win_pct'].at[idx] = away_win_dict[df.loc[idx, 'Date']]
                            except Exception:
                                # Ignorar errores individuales
                                continue
                
                # Historial contra oponentes
                if 'head_to_head_wins' not in new_features:
                    new_features['head_to_head_wins'] = pd.Series(index=df.index, data=0)
                if 'matchup_history' not in new_features:
                    new_features['matchup_history'] = pd.Series(index=df.index, data=0.5)
            
                try:
                    # Contar victorias contra cada oponente
                    opp_wins = team_data[team_data['is_win'] == 1].groupby('Opp').size().to_dict()
                    
                    # Historial de enfrentamientos (porcentaje de victorias)
                    matchup_history = team_data.groupby('Opp')['is_win'].mean().to_dict()
                
                    # Enfoque alternativo: crear un diccionario temporal para mapear equipo y oponente a valores
                    # Esto evita problemas de índice negativo
                    for idx in df[df['Team'] == team].index:
                        if idx in df.index and 'Opp' in df.columns:
                            try:
                                opp = df.loc[idx, 'Opp']
                                if opp in opp_wins:
                                    # Usar at en lugar de loc para asignación escalar
                                    new_features['head_to_head_wins'].at[idx] = opp_wins[opp]
                                if opp in matchup_history:
                                    new_features['matchup_history'].at[idx] = matchup_history[opp]
                            except Exception as e:
                                # Ignorar errores individuales para continuar con otros índices
                                continue
                except Exception as e:
                    logger.warning(f"Error calculando historial contra oponentes para {team}: {e}")
            # Asegurar que todas las columnas existan
            for window in [5, 10]:
                if f'Team_Points_avg_{window}' not in new_features:
                    new_features[f'Team_Points_avg_{window}'] = pd.Series(index=df.index, data=110)
                if f'Opp_Points_avg_{window}' not in new_features:
                    new_features[f'Opp_Points_avg_{window}'] = pd.Series(index=df.index, data=110)
                if f'point_diff_avg_{window}' not in new_features:
                    new_features[f'point_diff_avg_{window}'] = pd.Series(index=df.index, data=0)
                if f'win_pct_{window}' not in new_features:
                    new_features[f'win_pct_{window}'] = pd.Series(index=df.index, data=0.5)
        
            # Convertir a DataFrame y unir con el original
            # En lugar de usar series con índices potencialmente problemáticos,
            # creamos un nuevo DataFrame con el mismo índice que df
            features_df = pd.DataFrame(index=df.index)
            
            # Convertir cada Serie a un diccionario para evitar problemas de índice
            for col, series in new_features.items():
                # Crear una nueva Serie con el índice correcto de df
                features_df[col] = pd.Series(index=df.index)
                
                # Asignar valores solo para índices válidos
                valid_indices = series.index.intersection(df.index)
                if not valid_indices.empty:
                    features_df.loc[valid_indices, col] = series.loc[valid_indices]
            
            # Rellenar valores nulos con valores predeterminados
            for window in [5, 10]:
                if f'Team_Points_avg_{window}' in features_df.columns:
                    features_df[f'Team_Points_avg_{window}'].fillna(110, inplace=True)
                if f'Opp_Points_avg_{window}' in features_df.columns:
                    features_df[f'Opp_Points_avg_{window}'].fillna(110, inplace=True)
                if f'point_diff_avg_{window}' in features_df.columns:
                    features_df[f'point_diff_avg_{window}'].fillna(0, inplace=True)
                if f'win_pct_{window}' in features_df.columns:
                    features_df[f'win_pct_{window}'].fillna(0.5, inplace=True)
            
            if 'win_streak' in features_df.columns:
                features_df['win_streak'].fillna(0, inplace=True)
            if 'home_win_pct' in features_df.columns:
                features_df['home_win_pct'].fillna(0.5, inplace=True)
            if 'away_win_pct' in features_df.columns:
                features_df['away_win_pct'].fillna(0.5, inplace=True)
            if 'head_to_head_wins' in features_df.columns:
                features_df['head_to_head_wins'].fillna(0, inplace=True)
            if 'matchup_history' in features_df.columns:
                features_df['matchup_history'].fillna(0.5, inplace=True)
            
            # Enfoque completamente nuevo para unir las características con el DataFrame original
            # Evitamos completamente problemas de índice negativo
            result = df.copy()
            
            # Inicializar todas las columnas de características en el DataFrame resultado
            for col in new_features.keys():
                if col not in result.columns:
                    # Inicializar con valores predeterminados
                    if 'win_pct' in col or 'matchup_history' in col:
                        result[col] = 0.5  # Valor predeterminado para porcentajes
                    elif 'streak' in col or 'wins' in col:
                        result[col] = 0  # Valor predeterminado para conteos
                    elif 'avg' in col and 'Points' in col:
                        result[col] = 110  # Valor predeterminado para puntos
                    elif 'diff' in col:
                        result[col] = 0  # Valor predeterminado para diferenciales
                    else:
                        result[col] = 0  # Valor predeterminado genérico
            
            # Iterar sobre cada característica y actualizar solo los índices válidos
            for col, series in new_features.items():
                # Obtener solo los índices válidos que existen en ambos
                valid_indices = series.index.intersection(result.index)
                
                # Actualizar solo los índices válidos
                for idx in valid_indices:
                    try:
                        # Usar at para asignación escalar (más seguro que loc para valores individuales)
                        result.at[idx, col] = series.loc[idx]
                    except Exception:
                        # Ignorar errores individuales para continuar con otros índices
                        continue
            
            # Registrar características generadas
            num_features = len(features_df.columns)
            missing_pct = features_df.isnull().mean().mean() * 100
            logger.info(f"Características de predicción de equipo: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        except Exception as e:
            logger.error(f"Error en add_team_prediction_features: {e}")
            # Devolver el DataFrame original si hay un error
            return df
        
        return result

    def add_team_scoring_features(self, df):
        """
        Agrega características optimizadas específicas para predicción de puntos totales y puntos de equipo.
        """
        logger.info("Creando características para predicción de puntos...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Team', 'Date', 'team_score', 'opp_score', 'is_home']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de predicción de puntos (faltan columnas necesarias)")
            return df
        
        # Crear DataFrame a nivel de equipo
        team_df = df.groupby(['Team', 'Date']).agg({
            'team_score': 'mean',
            'opp_score': 'mean',
            'is_home': 'first'
        }).reset_index()
        
        # Calcular puntos totales
        team_df['total_score'] = team_df['team_score'] + team_df['opp_score']
        
        # Características por equipo
        new_features = {}
        
        for team in team_df['Team'].unique():
            team_mask = team_df['Team'] == team
            team_data = team_df[team_mask].sort_values('Date')
            
            if len(team_data) < 5:  # Necesitamos al menos 5 juegos
                continue
            
            # 1. Promedios móviles y tendencias
            for window in [5, 10]:
                # Puntos del equipo
                team_data[f'team_PTS_avg_{window}'] = team_data['team_score'].rolling(window, min_periods=1).mean()
                team_data[f'opp_PTS_avg_{window}'] = team_data['opp_score'].rolling(window, min_periods=1).mean()
                team_data[f'total_score_avg_{window}'] = team_data['total_score'].rolling(window, min_periods=1).mean()
                
                # Tendencias
                team_data[f'team_points_trend_{window}'] = team_data[f'team_PTS_avg_{window}'].diff()
                team_data[f'total_points_trend_{window}'] = team_data[f'total_score_avg_{window}'].diff()
            
            # 2. Promedios por ubicación (local/visitante)
            home_mask = team_data['is_home'] == 1
            away_mask = team_data['is_home'] != 1  # Usar != en lugar de ~
            
            # Local
            if home_mask.any():
                team_data.loc[home_mask, 'team_PTS_home_avg'] = team_data.loc[home_mask, 'team_score'].expanding().mean()
                team_data.loc[home_mask, 'total_score_home_avg'] = team_data.loc[home_mask, 'total_score'].expanding().mean()
                team_data.loc[home_mask, 'team_points_home_trend'] = team_data.loc[home_mask, 'team_score'].diff()
            
            # Visitante
            if away_mask.any():
                team_data.loc[away_mask, 'team_PTS_away_avg'] = team_data.loc[away_mask, 'team_score'].expanding().mean()
                team_data.loc[away_mask, 'total_score_away_avg'] = team_data.loc[away_mask, 'total_score'].expanding().mean()
                team_data.loc[away_mask, 'team_points_away_trend'] = team_data.loc[away_mask, 'team_score'].diff()
            
            # 3. Métricas de ritmo y eficiencia
            possessions = team_data['team_score'] * 0.96  # Aproximación simple de posesiones
            team_data['team_pace'] = possessions * 48  # Normalizado a 48 minutos
            team_data['opp_pace'] = possessions * 48  # Mismo ritmo para ambos equipos
            team_data['pace_combined'] = team_data['team_pace']  # Ya está combinado
            
            # Ratings ofensivos/defensivos
            team_data['team_offensive_rating'] = team_data['team_score'] / possessions * 100
            team_data['team_defensive_rating'] = team_data['opp_score'] / possessions * 100
            team_data['offensive_efficiency_combined'] = (team_data['team_offensive_rating'] + team_data['team_defensive_rating']) / 2
            
            # 4. Promedios de temporada
            team_data['season_avg_team_points'] = team_data['team_score'].expanding().mean()
            team_data['season_avg_total'] = team_data['total_score'].expanding().mean()
            
            # Por ubicación
            if home_mask.any():
                team_data['season_home_avg_team_points'] = team_data.loc[home_mask, 'team_score'].expanding().mean()
                team_data['season_home_avg_total'] = team_data.loc[home_mask, 'total_score'].expanding().mean()
            if away_mask.any():
                team_data['season_away_avg_team_points'] = team_data.loc[away_mask, 'team_score'].expanding().mean()
                team_data['season_away_avg_total'] = team_data.loc[away_mask, 'total_score'].expanding().mean()
            
            # 5. Métricas de shooting si están disponibles
            if all(col in df.columns for col in ['FG', 'FGA', '3P', '3PA', 'FT', 'FTA']):
                team_stats = df[df['Team'] == team].groupby('Date').agg({
                    'FG': 'sum',
                    'FGA': 'sum',
                    '3P': 'sum',
                    '3PA': 'sum',
                    'FT': 'sum',
                    'FTA': 'sum'
                }).reset_index()
                
                # Unir con team_data
                team_data = pd.merge(team_data, team_stats, on='Date', how='left')
                
                # Calcular tasas
                team_data['team_3pt_rate'] = team_data['3PA'] / team_data['FGA'].replace(0, 1)
                team_data['team_ft_rate'] = team_data['FTA'] / team_data['FGA'].replace(0, 1)
                team_data['team_efg_pct'] = (team_data['FG'] + 0.5 * team_data['3P']) / team_data['FGA'].replace(0, 1)
            
            # 6. Factores de matchup
            if 'Opp' in team_data.columns:
                # Historial de puntos contra cada oponente
                opp_points_avg = team_data.groupby('Opp')['team_score'].mean()
                team_data['historical_matchup_scoring'] = team_data['Opp'].map(opp_points_avg)
                
                # Ritmo histórico contra cada oponente
                opp_pace_avg = team_data.groupby('Opp')['team_pace'].mean()
                team_data['matchup_pace_factor'] = team_data['Opp'].map(opp_pace_avg)
            
            # 7. Impacto del ritmo en el scoring
            team_data['pace_impact_on_scoring'] = team_data['team_pace'] * team_data['season_avg_team_points'] / 100
            
            # 8. Interacción entre ratings
            team_data['team_off_rating_vs_opp_def_rating'] = team_data['team_offensive_rating'] - team_data['team_defensive_rating']
            
            # Guardar características para este equipo
            for col in team_data.columns:
                if col not in ['Team', 'Date', 'team_score', 'opp_score', 'is_home', 'total_score', 'Opp']:
                    if col not in new_features:
                        new_features[col] = pd.Series(index=df.index)
                    new_features[col].loc[team_mask] = team_data[col]
        
        # Convertir a DataFrame y unir con el original
        features_df = pd.DataFrame(new_features)
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de predicción de puntos: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        # Mostrar las características generadas agrupadas por categoría
        feature_categories = {
            'Promedios': [col for col in new_features if 'avg' in col],
            'Tendencias': [col for col in new_features if 'trend' in col],
            'Ritmo': [col for col in new_features if 'pace' in col],
            'Eficiencia': [col for col in new_features if 'rating' in col or 'efficiency' in col],
            'Shooting': [col for col in new_features if any(x in col for x in ['3pt', 'ft', 'efg'])],
            'Matchup': [col for col in new_features if 'matchup' in col],
            'Temporada': [col for col in new_features if 'season' in col]
        }
        
        print("\nCaracterísticas generadas por categoría:")
        for category, features in feature_categories.items():
            print(f"\n{category} ({len(features)}):")
            print("  " + ", ".join(features))
        
        return result

    def add_player_scoring_features(self, df):
        """
        Agrega características optimizadas específicas para predicción de puntos de jugador.
        """
        logger.info("Creando características para predicción de puntos de jugador...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Player', 'Team', 'Date', 'PTS', 'FG', 'FGA', 'FT', 'FTA', '3P', '3PA', 'MP', 'is_home']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de predicción de puntos (faltan columnas necesarias)")
            return df
        
        # Crear diccionario para nuevas características
        new_features = {}
        
        # Procesar cada jugador
        for player in df['Player'].unique():
            player_mask = df['Player'] == player
            player_data = df[player_mask].sort_values('Date')
            
            if len(player_data) < 5:  # Necesitamos al menos 5 juegos
                continue
            
            # 1. Promedios móviles y tendencias
            for window in [3, 5, 10, 20]:
                # Promedios
                player_data[f'PTS_{window}_avg'] = player_data['PTS'].rolling(window, min_periods=1).mean()
                # Varianza reciente
                player_data[f'PTS_{window}_var'] = player_data['PTS'].rolling(window, min_periods=1).var()
                # Tendencia
                player_data[f'PTS_{window}_trend'] = player_data[f'PTS_{window}_avg'].diff()
            
            # 2. Factores contextuales
            # Rendimiento en casa vs fuera
            home_mask = player_data['is_home'] == 1
            away_mask = ~home_mask
            
            if home_mask.any():
                home_avg = player_data.loc[home_mask, 'PTS'].mean()
                away_avg = player_data.loc[away_mask, 'PTS'].mean() if away_mask.any() else home_avg
                player_data['PTS_home_factor'] = home_avg / (away_avg + 1e-6)
            
            # Impacto de back-to-back
            if 'days_rest' in player_data.columns:
                b2b_mask = player_data['days_rest'] <= 1
                if b2b_mask.any():
                    b2b_avg = player_data.loc[b2b_mask, 'PTS'].mean()
                    rest_avg = player_data.loc[~b2b_mask, 'PTS'].mean()
                    player_data['b2b_impact_PTS'] = b2b_avg / (rest_avg + 1e-6) - 1
            
            # 3. Momentum y rachas
            # Momentum (cambio en el rendimiento reciente)
            player_data['PTS_momentum'] = player_data['PTS_5_avg'].diff() / player_data['PTS_10_avg'].clip(lower=1)
            
            # Rachas
            player_data['PTS_streak'] = player_data['PTS'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            player_data['PTS_streak'] = player_data['PTS_streak'].groupby((player_data['PTS_streak'] != player_data['PTS_streak'].shift()).cumsum()).cumcount() + 1
            player_data.loc[player_data['PTS_streak'] < 0, 'PTS_streak'] *= -1
            
            # 4. Probabilidades por percentiles
            try:
                # Inicializar características en el DataFrame principal
                for p in [25, 50, 75]:
                    if f'PTS_p{p}_prob' not in new_features:
                        new_features[f'PTS_p{p}_prob'] = pd.Series(index=df.index, data=0.5)
                    if f'PTS_p{p}_prob_smooth' not in new_features:
                        new_features[f'PTS_p{p}_prob_smooth'] = pd.Series(index=df.index, data=0.5)
                
                # Calcular umbrales para cada percentil
                pts_data = player_data['PTS'].dropna()
                if len(pts_data) >= 5:  # Asegurar suficientes datos
                    for p in [25, 50, 75]:
                        # Usar el valor real de puntos como umbral, no el percentil
                        threshold = p
                        
                        # Calcular probabilidad (1 si supera el umbral, 0 si no)
                        prob = (player_data['PTS'] > threshold).astype(float)
                        
                        # Probabilidad base (media móvil)
                        prob_rolling = prob.rolling(10, min_periods=1).mean().fillna(0.5)
                        player_data[f'PTS_p{p}_prob'] = prob_rolling
                        
                        # Versión suavizada (media móvil exponencial)
                        prob_smooth = prob.ewm(span=10).mean().fillna(0.5)
                        player_data[f'PTS_p{p}_prob_smooth'] = prob_smooth
                        
                        # Asignar al DataFrame principal
                        new_features[f'PTS_p{p}_prob'].loc[player_mask] = prob_rolling
                        new_features[f'PTS_p{p}_prob_smooth'].loc[player_mask] = prob_smooth
                else:
                    # No hay suficientes datos, usar valores predeterminados
                    for p in [25, 50, 75]:
                        new_features[f'PTS_p{p}_prob'].loc[player_mask] = 0.5
                        new_features[f'PTS_p{p}_prob_smooth'].loc[player_mask] = 0.5
            except Exception as e:
                logger.warning(f"Error calculando probabilidades de puntos para {player}: {e}")
                # Usar valores predeterminados en caso de error
                for p in [25, 50, 75]:
                    new_features[f'PTS_p{p}_prob'].loc[player_mask] = 0.5
                    new_features[f'PTS_p{p}_prob_smooth'].loc[player_mask] = 0.5
            
            # 5. Consistencia y eficiencia
            # Consistencia (inverso del coeficiente de variación)
            pts_mean = player_data['PTS'].mean()
            pts_std = player_data['PTS'].std()
            player_data['PTS_consistency'] = 1 - (pts_std / (pts_mean + 1e-6))
            
            # Eficiencia de scoring
            if all(col in player_data.columns for col in ['FGA', 'FTA']):
                scoring_attempts = player_data['FGA'] + 0.44 * player_data['FTA']
                player_data['PTS_efficiency_rating'] = player_data['PTS'] / (scoring_attempts + 1e-6)
            
            # 6. Contribución al equipo
            # Calcular totales del equipo por partido
            team_data = df[df['Team'] == player_data['Team'].iloc[0]].groupby('Date')['PTS'].sum()
            player_data['PTS_vs_team_avg'] = player_data['PTS'] / (team_data.loc[player_data['Date']].values + 1e-6)
            
            # Usage rate
            if all(col in player_data.columns for col in ['FGA', 'FTA', 'TOV', 'MP']):
                team_poss = team_data * 0.96  # Aproximación simple de posesiones
                player_poss = (player_data['FGA'] + 0.44 * player_data['FTA'] + player_data['TOV'])
                player_data['PTS_usage_rate'] = player_poss / (team_poss + 1e-6)
            
            # 7. Factores de situación
            if 'point_diff' in player_data.columns:
                # Rendimiento en juegos cerrados (diferencia <= 5 puntos)
                close_games = player_data['point_diff'].abs() <= 5
                if close_games.any():
                    close_avg = player_data.loc[close_games, 'PTS'].mean()
                    player_data['close_game_PTS_impact'] = close_avg / (pts_mean + 1e-6)
                
                # Factor clutch (últimos 5 minutos, juego cerrado)
                clutch_mask = (close_games) & (player_data['MP'] >= 43)  # Aproximación para últimos 5 min
                if clutch_mask.any():
                    clutch_avg = player_data.loc[clutch_mask, 'PTS'].mean()
                    player_data['clutch_PTS_factor'] = clutch_avg / (pts_mean + 1e-6)
            
            # 8. Índice de rendimiento físico
            if all(col in player_data.columns for col in ['Height_Inches', 'Weight', 'MP']):
                # Normalizar métricas físicas
                height_factor = (player_data['Height_Inches'] - 70) / 10  # Normalizar alrededor de 5'10"
                weight_factor = (player_data['Weight'] - 180) / 50  # Normalizar alrededor de 180 lbs
                minutes_factor = player_data['MP'] / 48
                
                # Combinar factores
                player_data['PTS_PhysPerf_Index'] = (
                    0.4 * player_data['PTS_efficiency_rating'] +
                    0.3 * height_factor +
                    0.2 * weight_factor +
                    0.1 * minutes_factor
                ).clip(-1, 1)  # Normalizar a [-1, 1]
            
            # Guardar características para este jugador
            for col in player_data.columns:
                if col not in required_cols:
                    if col not in new_features:
                        new_features[col] = pd.Series(index=df.index)
                    new_features[col].loc[player_mask] = player_data[col]
        
        # Convertir a DataFrame y unir con el original
        features_df = pd.DataFrame(new_features)
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de predicción de puntos de jugador: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        # Mostrar las características generadas agrupadas por categoría
        feature_categories = {
            'Promedios': [col for col in new_features if 'avg' in col],
            'Tendencias': [col for col in new_features if 'trend' in col or 'momentum' in col],
            'Probabilidades': [col for col in new_features if 'prob' in col],
            'Consistencia': [col for col in new_features if 'consistency' in col or 'efficiency' in col],
            'Físico': [col for col in new_features if 'Phys' in col],
            'Equipo': [col for col in new_features if 'team' in col or 'usage' in col],
            'Situación': [col for col in new_features if 'clutch' in col or 'close' in col or 'impact' in col]
        }
        
        print("\nCaracterísticas de puntos generadas por categoría:")
        for category, features in feature_categories.items():
            print(f"\n{category} ({len(features)}):")
            print("  " + ", ".join(features))
        
        return result

    def add_player_rebounding_features(self, df):
        """
        Agrega características optimizadas específicas para predicción de rebotes.
        """
        logger.info("Creando características para predicción de rebotes...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Player', 'Team', 'Date', 'TRB', 'DREB', 'OREB', 'MP', 'Height_Inches', 'Weight', 'BMI', 'is_home']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de predicción de rebotes (faltan columnas necesarias)")
            return df
        
        # Crear diccionario para nuevas características
        new_features = {}
        
        # Procesar cada jugador
        for player in df['Player'].unique():
            player_mask = df['Player'] == player
            player_data = df[player_mask].sort_values('Date')
            
            if len(player_data) < 5:  # Necesitamos al menos 5 juegos
                continue
            
            # 1. Promedios móviles y tendencias
            for window in [3, 5, 10, 20]:
                # Promedios de rebotes totales
                player_data[f'TRB_{window}_avg'] = player_data['TRB'].rolling(window, min_periods=1).mean()
                # Varianza reciente
                player_data[f'TRB_{window}_var'] = player_data['TRB'].rolling(window, min_periods=1).var()
                # Tendencia
                player_data[f'TRB_{window}_trend'] = player_data[f'TRB_{window}_avg'].diff()
                
                # Si es window=5, calcular también para DREB y OREB
                if window == 5:
                    player_data['DREB_5_avg'] = player_data['DREB'].rolling(window, min_periods=1).mean()
                    player_data['OREB_5_avg'] = player_data['OREB'].rolling(window, min_periods=1).mean()
            
            # 2. Factores contextuales
            # Rendimiento en casa vs fuera
            home_mask = player_data['is_home'] == 1
            away_mask = ~home_mask
            
            if home_mask.any():
                home_avg = player_data.loc[home_mask, 'TRB'].mean()
                away_avg = player_data.loc[away_mask, 'TRB'].mean() if away_mask.any() else home_avg
                player_data['TRB_home_factor'] = home_avg / (away_avg + 1e-6)
            
            # Impacto de back-to-back
            if 'days_rest' in player_data.columns:
                b2b_mask = player_data['days_rest'] <= 1
                if b2b_mask.any():
                    b2b_avg = player_data.loc[b2b_mask, 'TRB'].mean()
                    rest_avg = player_data.loc[~b2b_mask, 'TRB'].mean()
                    player_data['b2b_impact_TRB'] = b2b_avg / (rest_avg + 1e-6) - 1
            
            # 3. Momentum y rachas
            # Momentum (cambio en el rendimiento reciente)
            player_data['TRB_momentum'] = player_data['TRB_5_avg'].diff() / player_data['TRB_10_avg'].clip(lower=1)
            
            # Rachas
            player_data['TRB_streak'] = player_data['TRB'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            player_data['TRB_streak'] = player_data['TRB_streak'].groupby((player_data['TRB_streak'] != player_data['TRB_streak'].shift()).cumsum()).cumcount() + 1
            player_data.loc[player_data['TRB_streak'] < 0, 'TRB_streak'] *= -1
            
            # 4. Probabilidades por percentiles
            try:
                # Inicializar características en el DataFrame principal
                for p in [25, 50, 75]:
                    if f'TRB_p{p}_prob' not in new_features:
                        new_features[f'TRB_p{p}_prob'] = pd.Series(index=df.index, data=0.5)
                    if f'TRB_p{p}_prob_smooth' not in new_features:
                        new_features[f'TRB_p{p}_prob_smooth'] = pd.Series(index=df.index, data=0.5)
                
                # Calcular umbrales para cada percentil
                trb_data = player_data['TRB'].dropna()
                if len(trb_data) >= 5:  # Asegurar suficientes datos
                    for p in [25, 50, 75]:
                        # Usar el valor real de rebotes como umbral, no el percentil
                        threshold = p
                        
                        # Calcular probabilidad (1 si supera el umbral, 0 si no)
                        prob = (player_data['TRB'] > threshold).astype(float)
                        
                        # Probabilidad base (media móvil)
                        prob_rolling = prob.rolling(10, min_periods=1).mean().fillna(0.5)
                        player_data[f'TRB_p{p}_prob'] = prob_rolling
                        
                        # Versión suavizada (media móvil exponencial)
                        prob_smooth = prob.ewm(span=10).mean().fillna(0.5)
                        player_data[f'TRB_p{p}_prob_smooth'] = prob_smooth
                        
                        # Asignar al DataFrame principal
                        new_features[f'TRB_p{p}_prob'].loc[player_mask] = prob_rolling
                        new_features[f'TRB_p{p}_prob_smooth'].loc[player_mask] = prob_smooth
                else:
                    # No hay suficientes datos, usar valores predeterminados
                    for p in [25, 50, 75]:
                        new_features[f'TRB_p{p}_prob'].loc[player_mask] = 0.5
                        new_features[f'TRB_p{p}_prob_smooth'].loc[player_mask] = 0.5
            except Exception as e:
                logger.warning(f"Error calculando probabilidades de rebotes para {player}: {e}")
                # Usar valores predeterminados en caso de error
                for p in [25, 50, 75]:
                    new_features[f'TRB_p{p}_prob'].loc[player_mask] = 0.5
                    new_features[f'TRB_p{p}_prob_smooth'].loc[player_mask] = 0.5
            
            # 5. Consistencia y eficiencia
            # Consistencia (inverso del coeficiente de variación)
            trb_mean = player_data['TRB'].mean()
            trb_std = player_data['TRB'].std()
            player_data['TRB_consistency'] = 1 - (trb_std / (trb_mean + 1e-6))
            
            # Eficiencia de reboteo
            minutes_played = player_data['MP'].clip(lower=1)
            player_data['rebounding_efficiency'] = player_data['TRB'] / minutes_played
            
            # Box out rating (aproximación basada en OREB vs DREB)
            player_data['box_out_rating'] = player_data['DREB'] / (player_data['DREB'] + player_data['OREB'] + 1e-6)
            
            # 6. Componentes físicos
            # Normalizar métricas físicas
            height_factor = (player_data['Height_Inches'] - 70) / 10  # Normalizar alrededor de 5'10"
            weight_factor = (player_data['Weight'] - 180) / 50  # Normalizar alrededor de 180 lbs
            bmi_factor = (player_data['BMI'] - 25) / 5  # Normalizar alrededor de BMI 25
            
            # Crear componentes ortogonales
            player_data['Size_Component'] = (0.6 * height_factor + 0.4 * weight_factor).clip(-1, 1)
            player_data['Proportion_Component'] = (0.5 * height_factor + 0.5 * bmi_factor).clip(-1, 1)
            
            # Índice de rendimiento físico para rebotes
            player_data['TRB_PhysPerf_Index'] = (
                0.4 * player_data['Size_Component'] +
                0.3 * player_data['Proportion_Component'] +
                0.2 * player_data['rebounding_efficiency'] +
                0.1 * player_data['box_out_rating']
            ).clip(-1, 1)
            
            # 7. Contribución al equipo
            # Calcular totales del equipo por partido
            team_data = df[df['Team'] == player_data['Team'].iloc[0]].groupby('Date')['TRB'].sum()
            player_data['TRB_vs_team_avg'] = player_data['TRB'] / (team_data.loc[player_data['Date']].values + 1e-6)
            player_data['TRB_share_of_team'] = player_data['TRB_vs_team_avg']  # Ya es la proporción
            
            # 8. Factores de situación
            if 'point_diff' in player_data.columns:
                # Por margen de puntos
                margins = pd.cut(player_data['point_diff'], 
                               bins=[-100, -10, -5, 5, 10, 100],
                               labels=['blowout_loss', 'close_loss', 'very_close', 'close_win', 'blowout_win'])
                
                for margin in margins.unique():
                    if margin is not None:  # Evitar valores nulos
                        margin_mask = margins == margin
                        if margin_mask.any():
                            margin_avg = player_data.loc[margin_mask, 'TRB'].mean()
                            player_data.loc[margin_mask, 'TRB_by_score_margin'] = margin_avg / (trb_mean + 1e-6)
                
                # En juegos cerrados
                close_games = player_data['point_diff'].abs() <= 5
                if close_games.any():
                    close_avg = player_data.loc[close_games, 'TRB'].mean()
                    player_data['TRB_in_close_games'] = close_avg / (trb_mean + 1e-6)
            
            # Por cuarto (aproximación basada en minutos jugados)
            quarters = pd.cut(player_data['MP'], 
                            bins=[0, 12, 24, 36, 48],
                            labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            for quarter in quarters.unique():
                if quarter is not None:  # Evitar valores nulos
                    quarter_mask = quarters == quarter
                    if quarter_mask.any():
                        quarter_avg = player_data.loc[quarter_mask, 'TRB'].mean()
                        player_data.loc[quarter_mask, f'TRB_by_quarter_{quarter}'] = quarter_avg / (trb_mean + 1e-6)
            
            # Guardar características para este jugador
            for col in player_data.columns:
                if col not in required_cols:
                    if col not in new_features:
                        new_features[col] = pd.Series(index=df.index)
                    new_features[col].loc[player_mask] = player_data[col]
        
        # Convertir a DataFrame y unir con el original
        features_df = pd.DataFrame(new_features)
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar características generadas
        num_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de predicción de rebotes: {num_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        # Mostrar las características generadas agrupadas por categoría
        feature_categories = {
            'Promedios': [col for col in new_features if 'avg' in col],
            'Tendencias': [col for col in new_features if 'trend' in col or 'momentum' in col],
            'Probabilidades': [col for col in new_features if 'prob' in col],
            'Consistencia': [col for col in new_features if 'consistency' in col or 'efficiency' in col],
            'Físico': [col for col in new_features if any(x in col for x in ['Phys', 'Size', 'Proportion'])],
            'Equipo': [col for col in new_features if 'team' in col or 'share' in col],
            'Situación': [col for col in new_features if any(x in col for x in ['margin', 'close', 'quarter'])]
        }
        
        print("\nCaracterísticas de rebotes generadas por categoría:")
        for category, features in feature_categories.items():
            print(f"\n{category} ({len(features)}):")
            print("  " + ", ".join(features))
        
        return result

    def add_player_assist_features(self, df):
        """
        Agrega características optimizadas específicas para predicción de asistencias.
        """
        logger.info("Creando características para predicción de asistencias...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Player', 'Team', 'Date', 'AST', 'TOV', 'MP', 'GmSc', 'is_home']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de predicción de asistencias (faltan columnas necesarias)")
            return df
        
        # Crear diccionario para nuevas características
        new_features = {}
        
        # Procesar cada jugador
        for player in df['Player'].unique():
            player_mask = df['Player'] == player
            player_data = df[player_mask].copy()
            player_data = player_data.sort_values('Date')
            
            # Estadísticas básicas
            if 'USG%' not in player_data.columns:
                minutes_played = player_data['MP'].clip(lower=1)
                team_possessions = player_data['MP'].sum() * 0.96  # Estimación de posesiones
                player_data['USG%'] = ((player_data['AST'] + player_data['TOV']) / minutes_played * 100 / team_possessions).clip(0, 100)
            
            # Promedios temporales y tendencias
            for window in [3, 5, 10, 20]:
                # Promedios móviles
                player_data[f'AST_{window}_avg'] = player_data['AST'].rolling(window, min_periods=1).mean()
                player_data[f'TOV_{window}_avg'] = player_data['TOV'].rolling(window, min_periods=1).mean()
                player_data[f'AST_TOV_{window}_ratio'] = player_data[f'AST_{window}_avg'] / player_data[f'TOV_{window}_avg'].clip(lower=1)
                
                # Varianza y tendencias
                player_data[f'AST_recent_variance'] = player_data['AST'].rolling(window, min_periods=1).var()
                player_data[f'AST_{window}_trend'] = player_data[f'AST_{window}_avg'].diff()
            
            # Factores contextuales
            if 'is_home' in player_data.columns:
                home_games = player_data[player_data['is_home'] == 1]
                away_games = player_data[player_data['is_home'] == 0]
                
                if len(home_games) > 0 and len(away_games) > 0:
                    home_avg = home_games['AST'].mean()
                    away_avg = away_games['AST'].mean()
                    player_data['AST_home_factor'] = home_avg / (away_avg + 1e-6)
            
            # Impacto B2B
            if 'is_b2b' in player_data.columns:
                b2b_games = player_data[player_data['is_b2b'] == 1]
                rest_games = player_data[player_data['is_b2b'] == 0]
                
                if len(b2b_games) > 0 and len(rest_games) > 0:
                    b2b_avg = b2b_games['AST'].mean()
                    rest_avg = rest_games['AST'].mean()
                    player_data['b2b_impact_AST'] = b2b_avg / (rest_avg + 1e-6) - 1
            
            # Tendencias y momentum
            player_data['AST_momentum'] = player_data['AST_5_avg'].diff() / player_data['AST_10_avg'].clip(lower=1)
            player_data['AST_trend'] = player_data['AST_10_avg'].diff(5)
            
            # Racha de asistencias
            player_data['AST_streak'] = player_data['AST'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            player_data['AST_streak'] = player_data['AST_streak'].groupby((player_data['AST_streak'] != player_data['AST_streak'].shift()).cumsum()).cumcount() + 1
            player_data.loc[player_data['AST_streak'] < 0, 'AST_streak'] *= -1
            
            # Probabilidades optimizadas
            try:
                # Inicializar características en el DataFrame principal
                for p in [25, 50, 75]:
                    if f'AST_p{p}_prob' not in new_features:
                        new_features[f'AST_p{p}_prob'] = pd.Series(index=df.index, data=0.5)
                    if f'AST_p{p}_prob_smooth' not in new_features:
                        new_features[f'AST_p{p}_prob_smooth'] = pd.Series(index=df.index, data=0.5)
                
                # Calcular umbrales para cada percentil
                ast_data = player_data['AST'].dropna()
                if len(ast_data) >= 5:  # Asegurar suficientes datos
                    for p in [25, 50, 75]:
                        # Usar el valor real de asistencias como umbral, no el percentil
                        threshold = p / 10  # Dividir por 10 para tener umbrales razonables (2.5, 5, 7.5 asistencias)
                        
                        # Calcular probabilidad (1 si supera el umbral, 0 si no)
                        prob = (player_data['AST'] > threshold).astype(float)
                        
                        # Probabilidad base (media móvil)
                        prob_rolling = prob.rolling(10, min_periods=1).mean().fillna(0.5)
                        player_data[f'AST_p{p}_prob'] = prob_rolling
                        
                        # Versión suavizada (media móvil exponencial)
                        prob_smooth = prob.ewm(span=10).mean().fillna(0.5)
                        player_data[f'AST_p{p}_prob_smooth'] = prob_smooth
                        
                        # Asignar al DataFrame principal
                        new_features[f'AST_p{p}_prob'].loc[player_mask] = prob_rolling
                        new_features[f'AST_p{p}_prob_smooth'].loc[player_mask] = prob_smooth
                else:
                    # No hay suficientes datos, usar valores predeterminados
                    for p in [25, 50, 75]:
                        new_features[f'AST_p{p}_prob'].loc[player_mask] = 0.5
                        new_features[f'AST_p{p}_prob_smooth'].loc[player_mask] = 0.5
            except Exception as e:
                logger.warning(f"Error calculando probabilidades de asistencias para {player}: {e}")
                # Usar valores predeterminados en caso de error
                for p in [25, 50, 75]:
                    new_features[f'AST_p{p}_prob'].loc[player_mask] = 0.5
                    new_features[f'AST_p{p}_prob_smooth'].loc[player_mask] = 0.5
            
            # Consistencia y eficiencia
            ast_mean = player_data['AST'].mean()
            ast_std = player_data['AST'].std()
            player_data['AST_consistency'] = 1 - (ast_std / (ast_mean + 1e-6))
            
            # Eficiencia de playmaking y visión de cancha
            minutes_played = player_data['MP'].clip(lower=1)
            player_data['playmaking_efficiency'] = player_data['AST'] / minutes_played
            player_data['court_vision_rating'] = player_data['AST'] / (player_data['TOV'].clip(lower=1))
            
            # Contribución al equipo
            team_data = df[df['Team'] == player_data['Team'].iloc[0]].groupby('Date')['AST'].sum()
            team_fg = df[df['Team'] == player_data['Team'].iloc[0]].groupby('Date')['FG'].sum()
            
            player_data['AST_vs_team_avg'] = player_data['AST'] / (team_data.loc[player_data['Date']].values + 1e-6)
            player_data['AST_share_of_team'] = player_data['AST_vs_team_avg']
            
            # Tasa de asistencias del equipo
            player_data['team_assist_rate'] = team_data / (team_fg + 1e-6)
            
            # Factores de oponente
            if 'Opp' in player_data.columns and 'Pos' in player_data.columns:
                # Asistencias permitidas por posición
                opp_pos_stats = df.groupby(['Opp', 'Pos'])['AST'].mean().reset_index()
                opp_pos_stats.rename(columns={'AST': 'opp_assist_allowed_to_position'}, inplace=True)
                player_data = pd.merge(player_data, opp_pos_stats, on=['Opp', 'Pos'], how='left')
                
                # Rating de disrupción defensiva del oponente
                opp_def_stats = df.groupby('Opp').agg({
                    'STL': 'mean',
                    'BLK': 'mean',
                    'TOV': 'mean'
                }).reset_index()
                
                opp_def_stats['opp_defensive_disruption_rating'] = (
                    opp_def_stats['STL'] * 0.4 +
                    opp_def_stats['BLK'] * 0.3 +
                    opp_def_stats['TOV'] * 0.3
                )
                
                player_data = pd.merge(player_data, 
                                     opp_def_stats[['Opp', 'opp_defensive_disruption_rating']], 
                                     on='Opp', 
                                     how='left')
            
            # Factores de situación
            # Por margen de puntos
            for margin in ['close', 'ahead', 'behind']:
                if margin == 'close':
                    margin_mask = abs(player_data['point_diff']) <= 5
                elif margin == 'ahead':
                    margin_mask = player_data['point_diff'] > 5
                else:  # behind
                    margin_mask = player_data['point_diff'] < -5
                
                if margin_mask.any():
                    margin_avg = player_data.loc[margin_mask, 'AST'].mean()
                    player_data.loc[margin_mask, 'AST_by_score_margin'] = margin_avg / (ast_mean + 1e-6)
            
            # En juegos cerrados (últimos 5 minutos, diferencia ≤ 5 puntos)
            close_games = player_data[abs(player_data['point_diff']) <= 5]
            if len(close_games) > 0:
                close_avg = close_games['AST'].mean()
                player_data['AST_in_close_games'] = close_avg / (ast_mean + 1e-6)
            
            # Por cuarto (si está disponible)
            if 'Period' in player_data.columns:
                for period in player_data['Period'].unique():
                    period_mask = player_data['Period'] == period
                    if period_mask.any():
                        period_avg = player_data.loc[period_mask, 'AST'].mean()
                        player_data.loc[period_mask, f'AST_by_quarter'] = period_avg / (ast_mean + 1e-6)
            
            # Actualizar características
            for col in player_data.columns:
                if col not in df.columns:
                    new_features[col] = pd.Series(index=df.index)
                    new_features[col].loc[player_mask] = player_data[col]
        
        # Convertir el diccionario a DataFrame
        features_df = pd.DataFrame(new_features)
        
        # Unir las nuevas características
        result = pd.concat([df, features_df], axis=1)
        
        # Registrar el número de características generadas
        num_new_features = len(new_features)
        missing_pct = features_df.isnull().mean().mean() * 100
        logger.info(f"Características de asistencias optimizadas: {num_new_features} columnas, {missing_pct:.2f}% valores faltantes, {time.time() - start_time:.2f} segundos")
        
        return result

    def add_over_under_features(self, df):
        """
        Agrega características de over/under para totales y puntos de equipo.
        """
        logger.info("Creando características de over/under...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['team_score', 'opp_score']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de over/under (faltan columnas necesarias)")
            return df
        
        # Calcular totales
        df['total_score'] = df['team_score'] + df['opp_score']
        
        # Calcular promedios móviles para líneas de referencia
        df['total_score_ma_5'] = df.groupby('Team')['total_score'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        ).fillna(df['total_score'])  # Rellenar con el valor actual si no hay suficientes datos
        
        df['team_score_ma_5'] = df.groupby('Team')['team_score'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        ).fillna(df['team_score'])  # Rellenar con el valor actual si no hay suficientes datos
        
        # Generar líneas de over/under basadas en promedios históricos
        df['Total_Points_Line'] = df['total_score_ma_5'].shift().fillna(df['total_score_ma_5'])
        df['Team_Points_Line'] = df['team_score_ma_5'].shift().fillna(df['team_score_ma_5'])
        
        # Generar columnas de over/under (asegurar que sean 1 o 0)
        df['Total_Points_Over_Under'] = (df['total_score'] > df['Total_Points_Line']).astype(int)
        df['Team_Points_Over_Under'] = (df['team_score'] > df['Team_Points_Line']).astype(int)
        
        # Calcular tendencias de over/under
        for window in [5, 10]:
            # Para totales
            df[f'total_points_over_rate_{window}'] = df.groupby('Team')['Total_Points_Over_Under'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0.5)  # Valor neutral para datos insuficientes
            
            # Para puntos de equipo
            df[f'team_points_over_rate_{window}'] = df.groupby('Team')['Team_Points_Over_Under'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0.5)  # Valor neutral para datos insuficientes
        
        # Calcular distancia a la línea
        df['total_points_line_distance'] = df['total_score'] - df['Total_Points_Line']
        df['team_points_line_distance'] = df['team_score'] - df['Team_Points_Line']
        
        # Calcular tendencias de distancia a la línea
        for window in [5, 10]:
            # Para totales
            df[f'total_points_line_distance_ma_{window}'] = df.groupby('Team')['total_points_line_distance'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0)
            
            # Para puntos de equipo
            df[f'team_points_line_distance_ma_{window}'] = df.groupby('Team')['team_points_line_distance'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0)
        
        logger.info(f"Características de over/under generadas en {time.time() - start_time:.2f} segundos")
        return df

    def add_playoff_features(self, df):
        """
        Agrega características relacionadas con playoffs e importancia de juegos.
        """
        logger.info("Creando características de playoffs e importancia de juegos...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Team', 'Date', 'is_win']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de playoffs (faltan columnas necesarias)")
            return df
            
        # Crear DataFrame a nivel de equipo
        team_df = df.groupby(['Team', 'Date']).agg({
            'is_win': 'first',
            'team_score': 'mean',
            'opp_score': 'mean'
        }).reset_index()
        
        # Ordenar por equipo y fecha
        team_df = team_df.sort_values(['Team', 'Date'])
        
        # Calcular registro acumulado por equipo
        team_df['wins'] = team_df.groupby('Team')['is_win'].cumsum()
        team_df['games_played'] = team_df.groupby('Team').cumcount() + 1
        team_df['win_pct'] = team_df['wins'] / team_df['games_played']
        
        # Determinar si es final de temporada (últimos 20 juegos)
        team_df['games_remaining'] = team_df.groupby('Team')['games_played'].transform('max') - team_df['games_played']
        team_df['is_end_season'] = (team_df['games_remaining'] <= 20).astype(int)
        
        # Calcular posición en la conferencia
        team_df['conference_rank'] = team_df.groupby(['Date'])['win_pct'].rank(ascending=False)
        
        # Determinar situación de playoffs
        team_df['playoff_position'] = team_df['conference_rank'].apply(
            lambda x: 'secure' if x <= 6 else ('play_in' if x <= 10 else 'out')
        )
        
        # Calcular importancia del juego
        # Base: distancia a posición de playoffs más cercana
        team_df['playoff_distance'] = team_df.apply(
            lambda x: min(abs(x['conference_rank'] - 6), abs(x['conference_rank'] - 10))
            if x['conference_rank'] > 6 else 0, axis=1
        )
        
        # Importancia basada en:
        # 1. Cercanía a playoffs
        # 2. Final de temporada
        # 3. Diferencia de victorias con equipos cercanos
        team_df['game_importance'] = (
            (1 / (team_df['playoff_distance'] + 1)) * 0.4 +
            team_df['is_end_season'] * 0.3 +
            (1 - team_df['playoff_distance'] / 15).clip(0, 1) * 0.3
        )
        
        # Rendimiento en juegos importantes (últimos 5 juegos)
        team_df['high_importance_performance'] = 0.0
        for team in team_df['Team'].unique():
            team_mask = team_df['Team'] == team
            team_data = team_df[team_mask].copy()
            
            # Considerar juegos importantes (importancia > 0.7)
            important_games = team_data['game_importance'] > 0.7
            if important_games.any():
                win_rate = team_data.loc[important_games, 'is_win'].rolling(5, min_periods=1).mean()
                team_df.loc[team_mask, 'high_importance_performance'] = win_rate
        
        # Rendimiento contra equipos en playoffs
        team_df['vs_playoff_team'] = team_df['conference_rank'].apply(lambda x: 1 if x <= 8 else 0)
        team_df['playoff_teams_win_pct'] = 0.0
        for team in team_df['Team'].unique():
            team_mask = team_df['Team'] == team
            vs_playoff = team_df.loc[team_mask & (team_df['vs_playoff_team'] == 1), 'is_win']
            if len(vs_playoff) > 0:
                win_pct = vs_playoff.expanding().mean()
                team_df.loc[team_mask & (team_df['vs_playoff_team'] == 1), 'playoff_teams_win_pct'] = win_pct
        
        # Unir con el DataFrame original
        df = pd.merge(df, team_df[[
            'Team', 'Date', 'playoff_position', 'game_importance',
            'high_importance_performance', 'playoff_teams_win_pct',
            'is_end_season', 'conference_rank'
        ]], on=['Team', 'Date'])
        
        logger.info(f"Características de playoffs generadas en {time.time() - start_time:.2f} segundos")
        return df

    def add_physical_matchup_features(self, df):
        """
        Agrega características basadas en atributos físicos y sus matchups.
        """
        logger.info("Creando características físicas y de matchups optimizadas...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Height_Inches', 'Weight', 'Pos']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Faltan columnas biométricas procesadas: {[col for col in required_cols if col not in df.columns]}")
            return df
            
        new_features = {}
        
        # Calcular BMI si no existe
        if 'BMI' not in df.columns:
            height_m = df['Height_Inches'] * 0.0254  # pulgadas a metros
            weight_kg = df['Weight'] * 0.453592  # libras a kg
            df['BMI'] = weight_kg / (height_m ** 2)
        
        # Calcular estadísticas por posición
        pos_stats = df.groupby('Pos').agg({
            'Height_Inches': ['mean', 'std'],
            'Weight': ['mean', 'std'],
            'BMI': ['mean', 'std']
        }).round(2)
        
        # Aplanar nombres de columnas
        pos_stats.columns = ['_'.join(col).strip() for col in pos_stats.columns.values]
        pos_stats = pos_stats.reset_index()
        
        # Unir con el DataFrame original
        df = pd.merge(df, pos_stats, on='Pos', suffixes=('', '_position'))
        
        # Calcular ventajas/desventajas físicas
        df['height_advantage'] = (df['Height_Inches'] - df['Height_Inches_mean']) / df['Height_Inches_std']
        df['weight_advantage'] = (df['Weight'] - df['Weight_mean']) / df['Weight_std']
        df['bmi_advantage'] = (df['BMI'] - df['BMI_mean']) / df['BMI_std']
        
        # Crear componentes físicos normalizados
        # Size: relacionado con altura y peso
        df['Size_Component'] = (
            0.7 * df['height_advantage'] +
            0.3 * df['weight_advantage']
        ).clip(-3, 3)
        
        # Density: relacionado con BMI y peso relativo a altura
        df['Density_Component'] = (
            0.6 * df['bmi_advantage'] +
            0.4 * (df['weight_advantage'] - df['height_advantage'])
        ).clip(-3, 3)
        
        # Proportion: balance entre medidas
        df['Proportion_Component'] = (
            0.4 * df['height_advantage'] +
            0.4 * df['weight_advantage'] +
            0.2 * df['bmi_advantage']
        ).clip(-3, 3)
        
        # Calcular ventajas específicas por estadística
        stat_physical_weights = {
            'TRB': {'size': 0.7, 'density': 0.2, 'proportion': 0.1},
            'BLK': {'size': 0.8, 'density': 0.1, 'proportion': 0.1},
            'STL': {'size': 0.3, 'density': 0.4, 'proportion': 0.3},
            'PTS': {'size': 0.4, 'density': 0.3, 'proportion': 0.3},
            'AST': {'size': 0.2, 'density': 0.4, 'proportion': 0.4}
        }
        
        for stat, weights in stat_physical_weights.items():
            if stat in df.columns:
                df[f'{stat}_PhysPerf_Index'] = (
                    weights['size'] * df['Size_Component'] +
                    weights['density'] * df['Density_Component'] +
                    weights['proportion'] * df['Proportion_Component']
                ).clip(-3, 3)
        
        # Calcular ratios home/away corregidos
        for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
            if stat not in df.columns:
                continue
                
            # Calcular promedios por ubicación
            home_stats = df[df['is_home'] == 1].groupby('Player')[stat].mean()
            away_stats = df[df['is_home'] == 0].groupby('Player')[stat].mean()
            
            # Calcular ratio solo si hay suficientes datos
            valid_players = home_stats.index.intersection(away_stats.index)
            ratios = pd.Series(index=valid_players, dtype=float)
            
            for player in valid_players:
                home_avg = home_stats[player]
                away_avg = away_stats[player]
                if away_avg > 0:  # Evitar división por cero
                    ratio = (home_avg / away_avg) - 1  # Convertir a porcentaje de diferencia
                    ratios[player] = ratio
            
            # Asignar ratios al DataFrame
            df[f'{stat}_home_away_ratio'] = df['Player'].map(ratios)
            
            # Rellenar valores faltantes con 0 (sin diferencia home/away)
            df[f'{stat}_home_away_ratio'] = df[f'{stat}_home_away_ratio'].fillna(0)
        
        # Calcular factores de conferencia
        if 'Opp' in df.columns:
            east_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA',
                         'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
            
            df['opp_conference'] = df['Opp'].apply(lambda x: 'East' if x in east_teams else 'West')
            
            for stat in ['PTS', 'TRB', 'AST']:
                if stat not in df.columns:
                    continue
                
                # Calcular promedios por conferencia
                for player in df['Player'].unique():
                    player_mask = df['Player'] == player
                    player_data = df[player_mask]
                    
                    # Solo calcular si hay suficientes juegos contra ambas conferencias
                    east_games = player_data[player_data['opp_conference'] == 'East']
                    west_games = player_data[player_data['opp_conference'] == 'West']
                    
                    if len(east_games) >= 5 and len(west_games) >= 5:
                        east_avg = east_games[stat].mean()
                        west_avg = west_games[stat].mean()
                        
                        if west_avg > 0:  # Evitar división por cero
                            conf_factor = (east_avg / west_avg) - 1  # Convertir a porcentaje de diferencia
                            df.loc[player_mask, f'{stat}_conf_factor'] = conf_factor
                        
            # Rellenar valores faltantes con 0 (sin diferencia entre conferencias)
            for stat in ['PTS', 'TRB', 'AST']:
                if f'{stat}_conf_factor' in df.columns:
                    df[f'{stat}_conf_factor'] = df[f'{stat}_conf_factor'].fillna(0)
        
        logger.info(f"Características físicas y de matchup generadas en {time.time() - start_time:.2f} segundos")
        return df

    def add_physical_impact_features(self, df):
        """
        Agrega características que dan peso especial a la altura y minutos jugados.
        """
        logger.info("Creando características de impacto físico...")
        start_time = time.time()
        
        # Verificar columnas necesarias
        required_cols = ['Height_Inches', 'MP', 'Pos']
        if not all(col in df.columns for col in required_cols):
            logger.warning("No se pueden generar características de impacto físico (faltan columnas necesarias)")
            return df
        
        # Normalizar altura por posición
        df['height_position_advantage'] = df.groupby('Pos')['Height_Inches'].transform(
            lambda x: (x - x.mean()) / x.std()
        ).fillna(0)
        
        # Crear índice de dominancia física basado en altura
        df['physical_dominance_index'] = df['height_position_advantage'] * (df['MP'] / 48)
        
        # Crear características de impacto por minutos jugados
        df['minutes_impact'] = np.clip(df['MP'] / 48, 0, 1)  # Normalizado a [0,1]
        
        # Crear índice de impacto en el juego (combinación de altura y minutos)
        df['game_impact_index'] = (
            0.6 * df['height_position_advantage'] +  # Mayor peso a la altura
            0.4 * df['minutes_impact']               # Menor peso a los minutos
        ).clip(-3, 3)  # Limitar valores extremos
        
        # Crear índices específicos por estadística
        for stat in ['PTS', 'TRB', 'AST', 'BLK']:
            if stat in df.columns:
                # Eficiencia por minuto ajustada por altura
                df[f'{stat}_height_adjusted_efficiency'] = (
                    df[stat] / df['MP'].clip(lower=1) *  # Eficiencia por minuto
                    (1 + 0.2 * df['height_position_advantage'])  # Factor de ajuste por altura
                ).clip(0, None)  # No permitir valores negativos
                
                # Índice de producción física
                df[f'{stat}_physical_production_index'] = (
                    df[f'{stat}_height_adjusted_efficiency'] *
                    df['minutes_impact'] *
                    (1 + 0.3 * df['height_position_advantage'])  # Bonus por altura
                ).clip(0, None)
        
        # Crear índice de versatilidad física
        if all(stat in df.columns for stat in ['PTS', 'TRB', 'AST', 'BLK']):
            # Normalizar cada estadística
            stats_norm = {}
            for stat in ['PTS', 'TRB', 'AST', 'BLK']:
                stats_norm[stat] = df.groupby('Pos')[stat].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                ).fillna(0)
            
            # Crear índice de versatilidad ponderado por altura
            df['physical_versatility_index'] = (
                (0.3 * stats_norm['PTS'] +
                 0.3 * stats_norm['TRB'] +
                 0.2 * stats_norm['AST'] +
                 0.2 * stats_norm['BLK']) *
                (1 + 0.25 * df['height_position_advantage'])  # Bonus por altura
            ).clip(-3, 3)
        
        logger.info(f"Características de impacto físico generadas en {time.time() - start_time:.2f} segundos")
        return df

