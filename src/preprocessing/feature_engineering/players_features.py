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
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
from itertools import combinations


# Configuración del sistema de logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("players_features_engineering.log", mode='w', encoding='utf-8'),  # Usar UTF-8 para el archivo
        logging.StreamHandler()
    ]
)

# Clase para manejar problemas de codificación en el logging de Windows
class SafeLogFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            try:
                # Intenta sanitizar cualquier mensaje con caracteres Unicode problemáticos
                record.msg = record.msg.encode('ascii', 'replace').decode('ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                record.msg = "Mensaje con caracteres no ASCII (filtrado)"
        return True

# Aplicar el filtro al logger
logger = logging.getLogger('PlayersFeatures')
logger.addFilter(SafeLogFilter())

# Reducir verbosidad de los warnings de pandas
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger('PlayersFeatures')

class PlayersFeatures:
    def __init__(self, players_data, window_sizes=[3, 5, 10, 20], correlation_threshold=0.95, enable_correlation_analysis=True, n_jobs=1):
        """
        Inicializa el sistema de ingeniería de características de jugadores
        
        Args:
            players_data (pd.DataFrame): DataFrame con los datos de los jugadores
            window_sizes (list): Lista de tamaños de ventana para las características temporales
            correlation_threshold (float): Umbral de correlación para las características
            enable_correlation_analysis (bool): Si True, se realizará análisis de correlación
            n_jobs (int): Número de procesos en paralelo para el procesamiento
        """
        self.players_data = players_data.copy()
        # Asegurar que los tamaños de ventana sean enteros positivos
        self.window_sizes = [int(w) for w in window_sizes if int(w) > 0]
        if not self.window_sizes:
            self.window_sizes = [3, 5, 10, 20]  # valores por defecto
        self.window_sizes.sort()  # ordenar de menor a mayor
        
        self.correlation_threshold = correlation_threshold
        self.enable_correlation_analysis = enable_correlation_analysis
        self.n_jobs = n_jobs
        self.betting_lines = {
            'PTS': [10, 15, 20, 25, 30, 35], 
            'TRB': [4, 6, 8, 10, 12], 
            'AST': [4, 6, 8, 10, 12], 
            '3P': [1, 2, 3, 4, 5],
            'Double_Double': [0.5],  # Para predicciones binarias de doble-doble
            'Triple_Double': [0.5]   # Para predicciones binarias de triple-doble
        }
        
        # Asegurar que las fechas estén en formato datetime
        if 'Date' in self.players_data.columns:
            self.players_data['Date'] = pd.to_datetime(self.players_data['Date'])
            
        logger.info(f"Inicializado sistema de características con {len(self.players_data)} registros y ventanas {self.window_sizes}")
    
    def _preprocess_data(self):
        """
        Realiza el preprocesamiento inicial de los datos
        """
        logger.info("Iniciando preprocesamiento de datos")
        
        # Verificar y eliminar duplicados
        before_drop = len(self.players_data)
        if 'Player' in self.players_data.columns and 'Date' in self.players_data.columns:
            logger.info("Verificando duplicados por jugador y fecha")
            
            # Asegurar que Date está en formato datetime
            self.players_data['Date'] = pd.to_datetime(self.players_data['Date'])
            
            # Contar duplicados
            duplicates = self.players_data.duplicated(subset=['Player', 'Date'], keep=False)
            n_duplicates = duplicates.sum()
            
            if n_duplicates > 0:
                logger.warning(f"Se encontraron {n_duplicates} filas duplicadas por jugador y fecha")
                
                # Mostrar algunos ejemplos de duplicados
                duplicate_examples = self.players_data[duplicates].sort_values(['Player', 'Date']).head(10)
                logger.debug(f"Ejemplos de duplicados:\n{duplicate_examples[['Player', 'Date']]}")
                
                # Eliminar duplicados manteniendo la primera ocurrencia
                self.players_data = self.players_data.drop_duplicates(subset=['Player', 'Date'], keep='first')
                after_drop = len(self.players_data)
                logger.info(f"Se eliminaron {before_drop - after_drop} duplicados")
        
        # Ordenar datos por jugador y fecha
        self.players_data.sort_values(['Player', 'Date'], inplace=True)
        
        logger.info("Preprocesamiento completado")
        return self.players_data
    
    
    def _safe_rolling(self, series, window, operation='mean', min_periods=1):
        """
        Aplica una operación rolling de manera segura
        
        Args:
            series (pd.Series): Serie de datos
            window (int): Tamaño de la ventana
            operation (str): Operación a realizar ('mean', 'std', 'sum', 'max', 'min')
            min_periods (int): Número mínimo de observaciones requeridas
            
        Returns:
            pd.Series: Resultado de la operación rolling
        """
        try:
            # Convertir la serie a numérico para evitar problemas con tipos
            series = pd.to_numeric(series, errors='coerce')
            
            # Reemplazar valores infinitos con NaN para evitar problemas
            series = series.replace([np.inf, -np.inf], np.nan)
            
            window = int(window)
            if window <= 0:
                logger.warning(f"Tamaño de ventana inválido ({window}), usando valor por defecto de 3")
                window = 3
                
            if operation == 'mean':
                result = series.rolling(window=window, min_periods=min_periods).mean()
            elif operation == 'std':
                
                result = series.rolling(window=window, min_periods=min_periods).std()
                
                result = result.fillna(0)
            elif operation == 'sum':
                result = series.rolling(window=window, min_periods=min_periods).sum()
            elif operation == 'max':
                result = series.rolling(window=window, min_periods=min_periods).max()
            elif operation == 'min':
                result = series.rolling(window=window, min_periods=min_periods).min()
            else:
                logger.warning(f"Operación rolling '{operation}' no reconocida, usando mean")
                result = series.rolling(window=window, min_periods=min_periods).mean()
                
            # Manejar valores NaN e infinitos en el resultado
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # Rellena NaN con valor apropiado según la operación
            if operation == 'std':
                result = result.fillna(0)  # Para std, usar 0 para valores faltantes
            elif operation in ['max', 'min']:
                # Para max/min, usar el valor extremo correspondiente
                result = result.fillna(series.mean())
            else:
                # Para otras operaciones, usar la media de la serie
                result = result.fillna(series.mean())
                
            return result
        except Exception as e:
            logger.error(f"Error en operación rolling {operation}: {str(e)}")
            # En caso de error, devolver una serie de ceros con el mismo índice
            return pd.Series(0, index=series.index)
    
    def generate_features(self, save_path=None):
        """
        Genera todas las características para jugadores y devuelve el DataFrame final
        """
        logger.info("Iniciando proceso de generación de características para jugadores")
        
        try:
            # Primero realizamos el preprocesamiento
            self._preprocess_data()
            
            # Crear características rolling
            self.create_rolling_features()
            
            # Crear características temporales y situacionales
            self._create_temporal_features()
            
            # Crear características físicas del jugador
            self._create_physical_features()
            
            # Crear características de posición y rol del jugador
            self._create_position_role_features()
            
            # Crear características de titular vs suplente
            self._create_starter_features()
            
            # Crear características específicas para predicción de puntos
            self._create_pts_prediction_features()
            
            # Crear características específicas para predicción de rebotes
            self._create_trb_prediction_features()
            
            # Crear características específicas para predicción de asistencias
            self._create_ast_prediction_features()
            
            # Crear características específicas para predicción de triples
            self._create_3p_prediction_features()
            
            # Crear características para líneas de apuestas
            self._create_betting_line_features()
            
            # Crear características para doble-doble y triple-doble
            self._create_double_triple_features()
            
            # Crear características de matchup para jugadores
            self._create_matchup_features()
            
            # Crear características de eficiencia y productividad
            self._create_efficiency_features()
            
            
            # Verificar y corregir valores de porcentajes
            percentage_cols = ['FG%', '2P%', '3P%', 'FT%', 'TS%']
            for col in percentage_cols:
                if col in self.players_data.columns:
                    # Comprobar si hay valores muy pequeños (como 0.00667 en lugar de 0.667)
                    small_values = (self.players_data[col] > 0) & (self.players_data[col] < 0.1)
                    if small_values.any():
                        logger.info(f"Corrigiendo valores pequeños en la columna {col} ({small_values.sum()} valores)")
                        # Multiplicar por 100 los valores muy pequeños para corregirlos
                        self.players_data.loc[small_values, col] = self.players_data.loc[small_values, col] * 100
              
            # Verificar y manejar valores NaN
            nan_cols = []
            for col in self.players_data.columns:
                if self.players_data[col].isna().any():
                    nan_cols.append(col)
                    # Comprobar si es una columna categórica para evitar errores
                    if pd.api.types.is_categorical_dtype(self.players_data[col]):
                        # Para columnas categóricas, convertir a string primero
                        self.players_data[col] = self.players_data[col].astype(str).fillna('MISSING')
                    else:
                        # Para columnas no categóricas, rellenar con 0
                        self.players_data[col] = self.players_data[col].fillna(0)
            
            if nan_cols:
                logger.warning(f"Se rellenaron valores NaN en {len(nan_cols)} columnas")
                logger.debug(f"Columnas con NaN rellenados: {nan_cols[:10]} ...")
            
            # Filtrar características altamente correlacionadas si está habilitado
            if self.enable_correlation_analysis:
                logger.info(f"Filtrando características altamente correlacionadas (umbral: {self.correlation_threshold})")
                
                # Definir columnas base que deben preservarse
                base_columns = [
                    'Player', 'Date', 'Team', 'Away', 'Opp', 'Result', 'GS', 'MP',
                    'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
                    'FT', 'FTA', 'FT%', 'TS%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                    'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', '+/-', 'Pos', 'is_win',
                    'team_score', 'opp_score', 'total_score', 'point_diff', 'has_overtime',
                    'overtime_periods', 'is_home', 'Height_Inches', 'Weight', 'BMI', 'is_started'
                ]
                
                # Filtrar solo columnas que existen en el DataFrame
                base_columns_present = [col for col in base_columns if col in self.players_data.columns]
                
                # Crear una copia del DataFrame original para preservar columnas importantes
                original_df = self.players_data.copy()
                
                # Obtener todas las columnas no esenciales
                non_essential_cols = [col for col in self.players_data.columns if col not in base_columns_present]
                
                # Seleccionar solo columnas numéricas para análisis de correlación
                numeric_non_essential = self.players_data[non_essential_cols].select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_non_essential) > 1:
                    # Filtrar características correlacionadas solo en columnas no esenciales
                    logger.info(f"Aplicando filtro de correlación a {len(numeric_non_essential)} características numéricas no esenciales")
                    filtered_data = self.filter_correlated_features(self.players_data[numeric_non_essential])
                    
                    # Verificar cuántas columnas se eliminaron
                    removed_count = len(numeric_non_essential) - len(filtered_data.columns)
                    
                    if removed_count > 0:
                        logger.info(f"Se eliminaron {removed_count} características correlacionadas")
                        
                        # Reconstruir DataFrame con el orden correcto
                        # 1. Primero las columnas base
                        # 2. Luego las columnas filtradas
                        result_df = pd.concat([
                            original_df[base_columns_present], 
                            filtered_data
                        ], axis=1)
                        
                        # Verificar que no se perdieron filas
                        if len(result_df) != len(original_df):
                            logger.error(f"Error: La reconstrucción del DataFrame resultó en un cambio de longitud: {len(result_df)} vs {len(original_df)}")
                            # En caso de error, mantener el DataFrame original
                            self.players_data = original_df
                        else:
                            # Todo bien, usar el DataFrame reconstruido
                            self.players_data = result_df
                    else:
                        logger.info("No se eliminaron columnas durante el filtrado de correlación")
                else:
                    logger.info("No hay suficientes columnas numéricas no esenciales para analizar correlación")
            else:
                logger.info("Análisis de correlación desactivado, omitiendo filtrado")
            
            # Reordenar las columnas para que las columnas base vayan primero
            logger.info("Reordenando columnas del DataFrame final")
            
            # Definir orden final de columnas
            final_column_order = [
                'Player', 'Date', 'Team', 'Away', 'Opp', 'Result', 'GS', 'MP',
                'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
                'FT', 'FTA', 'FT%', 'TS%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', '+/-', 'Pos', 'is_win',
                'team_score', 'opp_score', 'total_score', 'point_diff', 'has_overtime',
                'overtime_periods', 'is_home', 'Height_Inches', 'Weight', 'BMI', 'is_started'
            ]
            
            # Filtrar las columnas base que existen en el DataFrame
            present_base_columns = [col for col in final_column_order if col in self.players_data.columns]
            
            # Obtener el resto de columnas (características generadas)
            generated_columns = [col for col in self.players_data.columns if col not in present_base_columns]
            
            # Reordenar las columnas: primero las columnas base, luego el resto
            new_column_order = present_base_columns + generated_columns
            
            # Aplicar el nuevo orden
            self.players_data = self.players_data[new_column_order]
            
            # Guardar en archivo si se especificó una ruta
            if save_path:
                self.players_data.to_csv(save_path, index=False)
                logger.info(f"DataFrame con características guardado en {save_path}")
            
            return self.players_data
            
        except Exception as e:
            logger.error(f"Error en generación de características: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            raise
            
    def _create_pts_prediction_features(self):
        """
        Crea características específicas para predecir los puntos (PTS) del jugador
        """
        logger.info("Creando características específicas para predicción de puntos")
        
        try:
            # Verificar que tenemos la columna PTS
            if 'PTS' not in self.players_data.columns:
                logger.error("Columna PTS no encontrada, abortando creación de características")
                return self.players_data
                
            # 1. Características de scoring por cuartos (si están disponibles)
            quarter_cols = [col for col in self.players_data.columns if 'PTS_Q' in col]
            if quarter_cols:
                # Calcular tendencias de puntuación por cuarto
                for col in quarter_cols:
                    try:
                        quarter = col.split('_')[1]  # Extraer el número de cuarto
                        # Calcular proporción de puntos en este cuarto
                        self.players_data[f'pts_prop_{quarter}'] = (self.players_data[col] / self.players_data['PTS'].clip(1)).clip(0, 1)
                    except Exception as e:
                        logger.error(f"Error al crear características de cuartos: {str(e)}")
                    
            # 2. Efectividad de tiro desglosada
            if all(col in self.players_data.columns for col in ['FG', 'FGA', '3P', '3PA', 'FT', 'FTA']):
                # Puntos por intento de tiro
                try:
                    # Puntos por tiro de campo
                    self.players_data['pts_per_fga'] = (self.players_data['PTS'] / self.players_data['FGA'].clip(1)).clip(0, 4)
                    
                    # Distribución de puntos por tipo de tiro
                    self.players_data['pts_from_3p'] = self.players_data['3P'] * 3
                    self.players_data['pts_from_2p'] = self.players_data['FG'] * 2 - self.players_data['pts_from_3p']
                    self.players_data['pts_from_ft'] = self.players_data['FT']
                    
                    # Proporciones de puntos por tipo
                    total_pts = self.players_data['PTS'].clip(1)
                    self.players_data['pts_prop_from_3p'] = (self.players_data['pts_from_3p'] / total_pts).clip(0, 1)
                    self.players_data['pts_prop_from_2p'] = (self.players_data['pts_from_2p'] / total_pts).clip(0, 1)
                    self.players_data['pts_prop_from_ft'] = (self.players_data['pts_from_ft'] / total_pts).clip(0, 1)
                    
                    # Eficiencia de tiro
                    self.players_data['pts_per_scoring_poss'] = (self.players_data['PTS'] / 
                                                            (self.players_data['FG'] + (self.players_data['FTA'] * 0.44)).clip(1)).clip(0, 3)
                except Exception as e:
                    logger.error(f"Error al crear características de efectividad de tiro: {str(e)}")
            
            # 3. Características de rendimiento por tipo de defensa (si existe info del oponente)
            if 'Opp' in self.players_data.columns:
                # Calcular estadísticas de puntos contra cada oponente
                try:
                    # Para cada jugador, calcular promedio contra cada oponente
                    for player in tqdm(self.players_data['Player'].unique(), desc="Calculando rendimiento vs. oponentes"):
                        player_mask = self.players_data['Player'] == player
                        for opp in self.players_data.loc[player_mask, 'Opp'].unique():
                            # Filtrar partidos de este jugador contra este oponente
                            mask = (self.players_data['Player'] == player) & (self.players_data['Opp'] == opp)
                            
                            if mask.sum() >= 2:  # Solo si hay suficientes partidos
                                pts_vs_opp = self.players_data.loc[mask, 'PTS'].mean()
                                # Asignar a todas las filas de este jugador contra este oponente
                                self.players_data.loc[mask, 'pts_avg_vs_opp'] = pts_vs_opp
                                
                                # Calcular diferencia respecto al promedio general del jugador
                                avg_pts = self.players_data.loc[player_mask, 'PTS'].mean()
                                self.players_data.loc[mask, 'pts_diff_vs_opp'] = pts_vs_opp - avg_pts
                except Exception as e:
                    logger.error(f"Error al crear características de rendimiento contra oponentes: {str(e)}")
            
            # 4. Características de momentum y rachas de anotación
            for window in self.window_sizes:
                try:
                    # Momentum ofensivo (tendencia de puntuación)
                    if f'PTS_mean_{window}' in self.players_data.columns:
                        pts_mean_col = f'PTS_mean_{window}'
                        
                        # Calcular una ventana más grande para comparar
                        larger_window = next((w for w in self.window_sizes if w > window), window*2)
                        pts_larger_mean_col = f'PTS_mean_{larger_window}'
                        
                        # Si existe la ventana más grande, calcular momentum
                        if pts_larger_mean_col in self.players_data.columns:
                            self.players_data[f'pts_momentum_{window}'] = (
                                self.players_data[pts_mean_col] - self.players_data[pts_larger_mean_col]
                            ).clip(-20, 20)
                except Exception as e:
                    logger.error(f"Error al crear características de momentum para ventana {window}: {str(e)}")
            
            # 5. Características de rachas de puntuación
            try:
                # Para cada jugador, calcular rachas de puntuación
                for player in tqdm(self.players_data['Player'].unique(), desc="Calculando rachas de puntuación"):
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data.loc[player_mask].sort_values('Date').copy()
                    
                    if len(player_data) >= 3:  # Solo con suficientes partidos
                        # Identificar partidos con más/menos puntos que el promedio
                        player_avg = player_data['PTS'].mean()
                        above_avg = player_data['PTS'] > player_avg
                        
                        # Rachas de puntuación por encima del promedio
                        streaks = []
                        current_streak = 0
                        
                        for above in above_avg:
                            if above:
                                current_streak += 1
                            else:
                                current_streak = 0
                            streaks.append(current_streak)
                        
                        self.players_data.loc[player_data.index, 'pts_above_avg_streak'] = streaks
                        
                        # Identificar partidos con aumento de puntuación consecutivo
                        pts_increase = player_data['PTS'].diff() > 0
                        
                        # Rachas de aumentos consecutivos
                        increase_streaks = []
                        current_inc_streak = 0
                        
                        for inc in pts_increase:
                            if inc:
                                current_inc_streak += 1
                            else:
                                current_inc_streak = 0
                            increase_streaks.append(current_inc_streak)
                        
                        self.players_data.loc[player_data.index, 'pts_increase_streak'] = increase_streaks
            except Exception as e:
                logger.error(f"Error al crear características de rachas de puntuación: {str(e)}")
            
            # 6. Características basadas en el oponente
            if 'Opp' in self.players_data.columns:
                try:
                    # Calcular promedio de puntos concedidos por cada oponente
                    opp_pts_allowed = self.players_data.groupby('Opp')['PTS'].mean().to_dict()
                    
                    # Asignar a cada fila
                    self.players_data['opp_pts_allowed_avg'] = self.players_data['Opp'].map(
                        lambda x: opp_pts_allowed.get(x, self.players_data['PTS'].mean())
                    )
                    
                    # Calcular diferencia respecto al promedio de la liga
                    league_avg = self.players_data['PTS'].mean()
                    self.players_data['opp_pts_allowed_diff'] = self.players_data['opp_pts_allowed_avg'] - league_avg
                except Exception as e:
                    logger.error(f"Error al crear características basadas en oponente: {str(e)}")
            
            # 7. Relación entre minutos jugados y puntos
            if 'MP' in self.players_data.columns:
                try:
                    # Puntos por minuto
                    self.players_data['pts_per_minute'] = (self.players_data['PTS'] / self.players_data['MP'].clip(1)).clip(0, 2)
                    
                    # Eficiencia de puntuación ajustada por minutos
                    for window in self.window_sizes:
                        if f'MP_mean_{window}' in self.players_data.columns and f'PTS_mean_{window}' in self.players_data.columns:
                            # Calcular puntos por minuto en la ventana
                            self.players_data[f'pts_per_min_{window}'] = (
                                self.players_data[f'PTS_mean_{window}'] / 
                                self.players_data[f'MP_mean_{window}'].clip(1)
                            ).clip(0, 2)
                            
                            # Predecir puntos basado en minutos esperados
                            self.players_data[f'expected_pts_{window}'] = (
                                self.players_data[f'pts_per_min_{window}'] * 
                                self.players_data['MP']
                            ).clip(0, 60)
                            
                            # Diferencia entre puntos reales y esperados
                            self.players_data[f'pts_vs_expected_{window}'] = (
                                self.players_data['PTS'] - 
                                self.players_data[f'expected_pts_{window}']
                            ).clip(-30, 30)
                except Exception as e:
                    logger.error(f"Error al crear características de puntos por minuto: {str(e)}")
            
            # 8. Características basadas en casa/fuera
            if 'is_home' in self.players_data.columns:
                try:
                    # Calcular promedios de puntos en casa vs. fuera por jugador
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        
                        # Promedio en casa
                        home_mask = player_mask & (self.players_data['is_home'] == 1)
                        if home_mask.sum() > 0:
                            home_avg = self.players_data.loc[home_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, 'pts_home_avg'] = home_avg
                        
                        # Promedio fuera
                        away_mask = player_mask & (self.players_data['is_home'] == 0)
                        if away_mask.sum() > 0:
                            away_avg = self.players_data.loc[away_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, 'pts_away_avg'] = away_avg
                        
                        # Diferencia casa-fuera
                        if home_mask.sum() > 0 and away_mask.sum() > 0:
                            self.players_data.loc[player_mask, 'pts_home_away_diff'] = home_avg - away_avg
                except Exception as e:
                    logger.error(f"Error al crear características de puntos casa/fuera: {str(e)}")
            
            # 9. Características de puntuación en función del resultado
            if 'is_win' in self.players_data.columns:
                try:
                    # Calcular promedios de puntos en victorias vs. derrotas por jugador
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        
                        # Promedio en victorias
                        win_mask = player_mask & (self.players_data['is_win'] == 1)
                        if win_mask.sum() > 0:
                            win_avg = self.players_data.loc[win_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, 'pts_win_avg'] = win_avg
                        
                        # Promedio en derrotas
                        loss_mask = player_mask & (self.players_data['is_win'] == 0)
                        if loss_mask.sum() > 0:
                            loss_avg = self.players_data.loc[loss_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, 'pts_loss_avg'] = loss_avg
                        
                        # Diferencia victorias-derrotas
                        if win_mask.sum() > 0 and loss_mask.sum() > 0:
                            self.players_data.loc[player_mask, 'pts_win_loss_diff'] = win_avg - loss_avg
                except Exception as e:
                    logger.error(f"Error al crear características de puntos por resultado: {str(e)}")
            
            # Rellenar valores nulos en las nuevas columnas
            pts_cols = [col for col in self.players_data.columns if col.startswith('pts_')]
            for col in pts_cols:
                if self.players_data[col].isnull().any():
                    self.players_data[col] = self.players_data[col].fillna(self.players_data[col].mean())
            
            logger.info("Características específicas para predicción de puntos creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error general al crear características de predicción de puntos: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            
        return self.players_data
    
    def create_rolling_features(self):
        """
        Crea características de ventana móvil para cada jugador
        """
        logger.info(f"Creando características de ventana móvil con ventanas: {self.window_sizes}")
        
        try:
            # Columnas numéricas para crear características móviles
            base_cols = [
                'PTS', 'TRB', 'AST', '3P', 'FG', 'FGA', 'FG%', '3PA', '3P%', 'FT', 
                'FTA', 'FT%', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF', 'MP'
            ]
            
            # Filtrar columnas que realmente existen en el DataFrame
            numeric_cols = [col for col in base_cols if col in self.players_data.columns]
            logger.debug(f"Columnas base para características rolling: {len(numeric_cols)}")
            
            # Crear características móviles para cada jugador
            players = self.players_data['Player'].unique()
            all_dfs = []
            
            for player in tqdm(players, desc="Procesando jugadores"):
                try:
                    # Crear una copia del DataFrame del jugador
                    player_data = self.players_data[self.players_data['Player'] == player].copy()
                    player_data = player_data.sort_values('Date')
                    
                    # Asegurar índice consecutivo
                    player_data = player_data.reset_index(drop=True)
                    
                    # Diccionario para almacenar todas las nuevas características
                    new_features = {}
                    
                    # Para cada ventana y columna numérica
                    for window in self.window_sizes:
                        for col in numeric_cols:
                            try:
                                # Calcular todas las características para esta columna y ventana
                                new_features[f'{col}_mean_{window}'] = self._safe_rolling(player_data[col], window, 'mean')
                                # Calcular std de forma segura y limitar a valores razonables
                                std_values = self._safe_rolling(player_data[col], window, 'std')
                                # Limitar valores std a un rango razonable para evitar valores extremos
                                # Usar un límite superior basado en la media de la columna
                                col_mean = player_data[col].mean()
                                std_limit = max(col_mean * 2, 1.0) if col_mean > 0 else 10.0
                                std_values = std_values.clip(0, std_limit)
                                new_features[f'{col}_std_{window}'] = std_values
                                new_features[f'{col}_max_{window}'] = self._safe_rolling(player_data[col], window, 'max')
                                new_features[f'{col}_min_{window}'] = self._safe_rolling(player_data[col], window, 'min')
                            except Exception as e:
                                logger.error(f"Error al calcular características rolling para {col}, ventana {window}: {str(e)}")
                    
                    # Calcular tendencias después de tener todas las medias
                    for window in self.window_sizes:
                        for col in numeric_cols:
                            mean_col = f'{col}_mean_{window}'
                            if mean_col in new_features:
                                try:
                                    # Convertir a numérico de forma segura
                                    col_values = pd.to_numeric(player_data[col], errors='coerce')
                                    mean_values = pd.to_numeric(new_features[mean_col], errors='coerce')
                                    
                                    # Reemplazar valores infinitos o NaN
                                    col_values = col_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    mean_values = mean_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    
                                    # Calcular diferencia entre valores actuales y medias móviles
                                    trend = col_values - mean_values
                                    
                                    # Determinar un límite de tendencia razonable basado en la naturaleza de la columna
                                    if col == 'PTS':
                                        trend_limit = 20.0  # Para puntos
                                    elif col in ['TRB', 'AST']:
                                        trend_limit = 10.0  # Para rebotes y asistencias
                                    elif col == '3P':
                                        trend_limit = 5.0   # Para triples
                                    elif '%' in col:
                                        trend_limit = 0.3   # Para porcentajes
                                    else:
                                        trend_limit = 10.0  # Límite predeterminado
                                    
                                    # Limitar valores extremos
                                    trend = trend.clip(-trend_limit, trend_limit)
                                    
                                    # Guardar la tendencia limitada
                                    new_features[f'{col}_trend_{window}'] = trend
                                except Exception as e:
                                    logger.error(f"Error al calcular tendencia para {col}, ventana {window}: {str(e)}")
                                    # En caso de error, crear una columna de ceros
                                    new_features[f'{col}_trend_{window}'] = pd.Series(0, index=player_data.index)
                    
                    # Verificar que todas las características tienen la misma longitud
                    lengths = {k: len(v) for k, v in new_features.items()}
                    if len(set(lengths.values())) > 1:
                        logger.warning(f"Longitudes inconsistentes en características para jugador {player}: {lengths}")
                        # Ajustar a la longitud más común
                        most_common_length = max(set(lengths.values()), key=list(lengths.values()).count)
                        for k, v in list(new_features.items()):
                            if len(v) != most_common_length:
                                logger.warning(f"Ajustando longitud de {k} de {len(v)} a {most_common_length}")
                                if len(v) > most_common_length:
                                    new_features[k] = v[:most_common_length]
                                else:
                                    # Rellenar con ceros
                                    padding = [0] * (most_common_length - len(v))
                                    new_features[k] = list(v) + padding
                    
                    # Asegurar que la longitud coincide con player_data
                    for k, v in list(new_features.items()):
                        if len(v) != len(player_data):
                            logger.warning(f"Longitud de {k} ({len(v)}) no coincide con player_data ({len(player_data)})")
                            if len(v) > len(player_data):
                                new_features[k] = v[:len(player_data)]
                            else:
                                # Eliminar característica si es demasiado corta
                                logger.warning(f"Eliminando característica {k} por longitud insuficiente")
                                del new_features[k]
                    
                    # Crear DataFrame con las nuevas características
                    features_df = pd.DataFrame(new_features, index=player_data.index)
                    
                    # Agregar todas las nuevas características de una vez
                    player_data = pd.concat([player_data, features_df], axis=1)
                    all_dfs.append(player_data)
                    
                except Exception as e:
                    logger.error(f"Error al procesar jugador {player}: {str(e)}")
                    # Agregar el jugador sin características adicionales para no perder datos
                    all_dfs.append(self.players_data[self.players_data['Player'] == player])
            
            # Combinar todos los DataFrames
            if all_dfs:
                self.players_data = pd.concat(all_dfs).sort_index()
                
                # Asegurar que no hay fragmentación
                self.players_data = self.players_data.copy()
                
                logger.info(f"Creadas nuevas características de ventana móvil")
            
            return self.players_data
            
        except Exception as e:
            logger.error(f"Error en create_rolling_features: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            # Devolver el DataFrame original
            return self.players_data
    
    def plot_feature_importance(self, model, X, y, feature_names=None, top_n=20):
        """
        Grafica la importancia de las características para un modelo y genera JSON con correlaciones
        
        Args:
            model: Modelo entrenado (debe tener atributo feature_importances_)
            X (np.array): Características
            y (np.array): Variable objetivo
            feature_names (list): Lista de nombres de características
            top_n (int): Número de características principales a mostrar
            
        Returns:
            pd.DataFrame: DataFrame con importancia de características
        """
        if not hasattr(model, 'feature_importances_'):
            logger.error("El modelo no tiene atributo feature_importances_")
            return None
            
        # Obtener importancia de características
        importances = model.feature_importances_
        
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns
            else:
                feature_names = [f"Feature {i}" for i in range(X.shape[1])]
                
        # Crear DataFrame de importancia
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Generar gráfico
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Características por Importancia')
        plt.tight_layout()
        
        # Guardar gráfico como imagen
        plot_path = f"player_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de importancia guardado en {plot_path}")
        
        # Calcular y guardar correlaciones
        try:
            # Seleccionar top_n características por importancia
            top_features = importance_df.head(top_n)['Feature'].tolist()
            
            # Obtener datos correspondientes a las características más importantes
            if hasattr(X, 'loc'):
                # Para DataFrames de pandas
                top_X = X.loc[:, top_features]
            else:
                # Para arrays de numpy
                top_indices = [list(feature_names).index(feat) for feat in top_features if feat in feature_names]
                top_X = X[:, top_indices]
                top_X = pd.DataFrame(top_X, columns=[feature_names[i] for i in top_indices])
            
            # Calcular matriz de correlación
            corr_matrix = top_X.corr().round(3).fillna(0)
            
            # Convertir a diccionario para JSON
            corr_dict = {
                'features': top_features,
                'correlations': corr_matrix.to_dict()
            }
            
            # Guardar como JSON
            json_path = f"player_feature_correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(corr_dict, f, indent=4)
            
            logger.info(f"Correlaciones guardadas en {json_path}")
        except Exception as e:
            logger.error(f"Error al calcular y guardar correlaciones: {str(e)}")
            logger.error(traceback.format_exc())
        
        return importance_df
    
    def _create_temporal_features(self):
        """
        Crea características temporales y de calendario más avanzadas
        """
        logger.info("Creando características temporales avanzadas")
        
        try:
            # Asegurar que tenemos un índice numérico
            self.players_data = self.players_data.reset_index(drop=True)
            
            # Características de calendario básicas
            self.players_data['dayofweek'] = self.players_data['Date'].dt.dayofweek
            self.players_data['month'] = self.players_data['Date'].dt.month
            self.players_data['is_weekend'] = self.players_data['dayofweek'].isin([5, 6]).astype(int)
            
            # Características de temporada más específicas
            self.players_data['day_of_month'] = self.players_data['Date'].dt.day
            self.players_data['week_of_year'] = self.players_data['Date'].dt.isocalendar().week
            self.players_data['quarter_of_year'] = self.players_data['Date'].dt.quarter
            
            # Períodos específicos de la temporada NBA
            # Crear mapeo directo para cada mes a su fase
            season_phases = {
                1: 'mid_season',    # Enero
                2: 'mid_season',    # Febrero
                3: 'late_season',   # Marzo
                4: 'late_season',   # Abril
                5: 'playoffs',      # Mayo
                6: 'playoffs',      # Junio
                7: 'off_season',    # Julio
                8: 'off_season',    # Agosto
                9: 'off_season',    # Septiembre
                10: 'early_season',  # Octubre
                11: 'early_season',  # Noviembre
                12: 'mid_season'     # Diciembre
            }
            
            # Aplicar el mapeo directamente
            self.players_data['season_phase'] = self.players_data['month'].map(season_phases)
            
            # Convertir fase a variables dummy
            try:
                season_phase_dummies = pd.get_dummies(
                    self.players_data['season_phase'], 
                    prefix='phase',
                    dummy_na=False
                )
                self.players_data = pd.concat([self.players_data, season_phase_dummies], axis=1)
                logger.info("Variables dummy de fase de temporada creadas correctamente")
            except Exception as e:
                logger.error(f"Error al crear dummies de fase de temporada: {str(e)}")
            
            # Indicador de día festivo (aproximación basada en fechas comunes)
            holiday_dates = [
                (12, 25),  # Navidad
                (1, 1),    # Año Nuevo
                (1, 16),   # Martin Luther King Jr. Day (aproximación)
                (2, 14),   # San Valentín
                (7, 4),    # Día de la Independencia
                (11, 11),  # Día de los Veteranos
                (11, 24),  # Día de Acción de Gracias (aproximación)
            ]
            
            self.players_data['is_holiday'] = self.players_data.apply(
                lambda row: 1 if (row['month'], row['day_of_month']) in holiday_dates else 0, 
                axis=1
            )
            
            # Días desde último partido
            self.players_data['days_rest'] = self.players_data.groupby('Player')['Date'].diff().dt.days.fillna(0)
            
            # Categorización de descanso
            rest_bins = [-1, 1, 2, 3, 100]  # -1 para incluir valores de 0
            rest_labels = ['no_rest', 'short_rest', 'medium_rest', 'long_rest']
            
            self.players_data['rest_category'] = pd.cut(
                self.players_data['days_rest'],
                bins=rest_bins,
                labels=rest_labels,
                include_lowest=True
            )
            
            # Convertir categoría a variables dummy
            try:
                rest_dummies = pd.get_dummies(
                    self.players_data['rest_category'], 
                    prefix='rest',
                    dummy_na=False
                )
                self.players_data = pd.concat([self.players_data, rest_dummies], axis=1)
                logger.info("Variables dummy de categoría de descanso creadas correctamente")
            except Exception as e:
                logger.error(f"Error al crear dummies de categoría de descanso: {str(e)}")
            
            # Local o visitante
            if 'is_home' not in self.players_data.columns and 'Tm' in self.players_data.columns and 'Opp' in self.players_data.columns:
                try:
                    # Detectar partidos en casa vs. fuera
                    self.players_data['is_home'] = (self.players_data['Tm'] == self.players_data['Opp']).astype(int)
                except Exception as e:
                    logger.error(f"Error al crear columna is_home: {str(e)}")
                    self.players_data['is_home'] = 0  # Valor por defecto
            
            # Patrones de viaje (distancia desde último partido)
            if 'is_home' in self.players_data.columns:
                try:
                    # Crear variable para rastrear cambios de ubicación
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        player_data = self.players_data.loc[player_mask].sort_values('Date')
                        
                        # Inicializar rastreo de viaje
                        travel_changes = []
                        prev_home = None
                        
                        # Para cada partido, determinar si hubo cambio de ubicación
                        for idx, row in player_data.iterrows():
                            current_home = row['is_home']
                            
                            if prev_home is not None:
                                # 0 = sin cambio, 1 = cambio de local/visitante
                                travel_changes.append(1 if current_home != prev_home else 0)
                            else:
                                travel_changes.append(0)  # Primer partido, no hay cambio
                            
                            prev_home = current_home
                        
                        # Asignar cambios al DataFrame
                        self.players_data.loc[player_data.index, 'location_change'] = travel_changes
                        
                        # Calcular viajes consecutivos (rachas de partidos fuera de casa)
                        away_streaks = []
                        current_away_streak = 0
                        
                        for is_home in player_data['is_home']:
                            if is_home == 0:  # Partido fuera
                                current_away_streak += 1
                            else:
                                current_away_streak = 0
                            away_streaks.append(current_away_streak)
                        
                        self.players_data.loc[player_data.index, 'away_streak'] = away_streaks
                        
                        # Crear indicador de "road trip" (3+ partidos seguidos fuera)
                        self.players_data.loc[player_data.index, 'is_road_trip'] = (
                            self.players_data.loc[player_data.index, 'away_streak'] >= 3
                        ).astype(int)
                
                except Exception as e:
                    logger.error(f"Error al calcular patrones de viaje: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                # Características avanzadas de patrones de viaje
                try:
                    # Calcular efectos de cambio de zona horaria (aproximación simplificada)
                    # Usando la conferencia como aproximación para la zona horaria
                    if 'Conf' in self.players_data.columns:
                        for player in self.players_data['Player'].unique():
                            player_mask = self.players_data['Player'] == player
                            player_data = self.players_data.loc[player_mask].sort_values('Date')
                            
                            # Seguimiento de cambios de conferencia como aproximación para cambios de zona horaria
                            timezone_changes = []
                            prev_conf = None
                            
                            for idx, row in player_data.iterrows():
                                current_conf = row.get('Conf')
                                
                                if prev_conf is not None and pd.notna(current_conf) and pd.notna(prev_conf):
                                    # Cambio Este a Oeste o viceversa (aprox. 3 horas de diferencia)
                                    if current_conf != prev_conf:
                                        timezone_changes.append(1)
                                    else:
                                        timezone_changes.append(0)
                                else:
                                    timezone_changes.append(0)
                                
                                prev_conf = current_conf
                            
                            # Asignar al DataFrame
                            if timezone_changes:
                                self.players_data.loc[player_data.index, 'timezone_change'] = timezone_changes
                                
                                # Crear indicador de "jet lag" (cambio de zona horaria reciente)
                                # Mayor impacto si ha habido poco descanso tras el cambio
                                jet_lag = []
                                for i, (tz_change, rest) in enumerate(zip(timezone_changes, player_data['days_rest'])):
                                    if i > 0 and timezone_changes[i-1] == 1 and rest < 3:
                                        # Efectos de jet lag significativos
                                        jet_lag.append(1)
                                    else:
                                        jet_lag.append(0)
                                
                                self.players_data.loc[player_data.index, 'jet_lag_effect'] = jet_lag
                except Exception as e:
                    logger.error(f"Error al calcular efectos de cambio de zona horaria: {str(e)}")
            
            # Partidos en últimos 7 días (fatiga)
            games_last_7_days = []
            
            # Procesar cada jugador por separado
            for player in self.players_data['Player'].unique():
                player_data = self.players_data[self.players_data['Player'] == player].copy()
                player_data = player_data.sort_values('Date')
                
                # Calcular partidos en los últimos 7 días
                games_7d = []
                for date in player_data['Date']:
                    mask = (player_data['Date'] <= date) & (player_data['Date'] > date - pd.Timedelta(days=7))
                    games_7d.append(len(player_data[mask]))
                
                player_data['games_last_7_days'] = games_7d
                
                # Calcular partidos en diferentes ventanas temporales
                for days in [3, 5, 10, 14]:
                    games_in_window = []
                    for date in player_data['Date']:
                        mask = (player_data['Date'] <= date) & (player_data['Date'] > date - pd.Timedelta(days=days))
                        games_in_window.append(len(player_data[mask]))
                    
                    player_data[f'games_last_{days}_days'] = games_in_window
                
                # Índice de densidad de calendario (partidos recientes ponderados)
                # Más peso a partidos muy recientes para capturar fatiga aguda
                density_index = []
                for i, date in enumerate(player_data['Date']):
                    # Pesos decrecientes para 1, 2, 3, 4, 5 días atrás
                    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
                    density = 0
                    
                    for days_ago, weight in enumerate(weights, 1):
                        prev_date = date - pd.Timedelta(days=days_ago)
                        if any(player_data['Date'] == prev_date):
                            density += weight
                    
                    density_index.append(min(density, 2.0))  # Limitar a un máximo razonable
                
                player_data['schedule_density_index'] = density_index
                
                # Categorizar densidad
                player_data['density_category'] = pd.cut(
                    player_data['schedule_density_index'],
                    bins=[-0.1, 0.5, 1.0, 1.5, 2.1],
                    labels=['low', 'medium', 'high', 'extreme'],
                    include_lowest=True
                )
                
                # Añadir tendencias de carga de juego (minutos)
                if 'MP' in player_data.columns:
                    # Calcular tendencia de minutos recientes (últimos 3 partidos comparados con media de temporada)
                    player_data['MP'] = pd.to_numeric(player_data['MP'], errors='coerce')
                    player_data['MP_7day_avg'] = player_data['MP'].rolling(3, min_periods=1).mean()
                    overall_avg_mp = player_data['MP'].mean()
                    
                    if pd.notna(overall_avg_mp) and overall_avg_mp > 0:
                        player_data['minutes_trend'] = (player_data['MP_7day_avg'] / overall_avg_mp - 1) * 100
                        player_data['minutes_trend'] = player_data['minutes_trend'].clip(-50, 50)
                        
                        # Categorizar tendencia de minutos
                        player_data['minutes_trend_category'] = pd.cut(
                            player_data['minutes_trend'],
                            bins=[-50, -15, -5, 5, 15, 50],
                            labels=['sharp_decrease', 'decrease', 'stable', 'increase', 'sharp_increase'],
                            include_lowest=True
                        )
                
                # Añadir indicadores de momentum/rachas
                if 'PTS' in player_data.columns:
                    # Calcular si el jugador viene de buenas o malas actuaciones
                    player_data['PTS'] = pd.to_numeric(player_data['PTS'], errors='coerce')
                    player_data['PTS_3game_avg'] = player_data['PTS'].rolling(3, min_periods=1).mean()
                    player_data['PTS_10game_avg'] = player_data['PTS'].rolling(10, min_periods=3).mean()
                    
                    # Calcular momentum (si el rendimiento reciente es mejor que el de largo plazo)
                    player_data['scoring_momentum'] = (player_data['PTS_3game_avg'] / player_data['PTS_10game_avg'].clip(lower=1) - 1) * 100
                    player_data['scoring_momentum'] = player_data['scoring_momentum'].clip(-30, 30)
                    
                    # Categorizar momentum
                    player_data['momentum_category'] = pd.cut(
                        player_data['scoring_momentum'],
                        bins=[-30, -10, -3, 3, 10, 30],
                        labels=['cold_streak', 'cooling', 'neutral', 'warming', 'hot_streak'],
                        include_lowest=True
                    )
                    
                    # Identificar rachas significativas
                    # Definir umbral para buen partido (por encima de la media del jugador)
                    pts_mean = player_data['PTS'].mean()
                    pts_std = player_data['PTS'].std()
                    if pd.notna(pts_mean) and pd.notna(pts_std) and pts_std > 0:
                        good_game_threshold = pts_mean + 0.5 * pts_std
                        bad_game_threshold = pts_mean - 0.5 * pts_std
                        
                        # Marcar buenos/malos partidos
                        player_data['good_scoring_game'] = (player_data['PTS'] >= good_game_threshold).astype(int)
                        player_data['bad_scoring_game'] = (player_data['PTS'] <= bad_game_threshold).astype(int)
                        
                        # Calcular rachas (consecutivos buenos/malos partidos)
                        streaks = {}
                        for streak_type in ['good', 'bad']:
                            current_streak = 0
                            streak_list = []
                            
                            # Iterar sobre cada juego en orden cronológico
                            for game_idx, is_streak_game in enumerate(player_data[f'{streak_type}_scoring_game']):
                                if is_streak_game == 1:
                                    # Si es un buen/mal juego, incrementar la racha actual
                                    current_streak += 1
                                else:
                                    # Si no es un buen/mal juego, reiniciar la racha
                                    current_streak = 0
                                
                                # Guardar el valor actual de la racha
                                streak_list.append(current_streak)
                            
                            # Asignar la lista de rachas al diccionario
                            streaks[f'{streak_type}_game_streak'] = streak_list
                            
                            # Actualizar los datos del jugador con las rachas
                            player_data[f'{streak_type}_game_streak'] = streak_list
                        
                        # Añadir columnas de rachas
                        player_data['good_game_streak'] = streaks['good_game_streak']
                        player_data['bad_game_streak'] = streaks['bad_game_streak']
                        
                        # Indicadores de racha activa significativa
                        player_data['in_hot_streak'] = (player_data['good_game_streak'] >= 3).astype(int)
                        player_data['in_cold_streak'] = (player_data['bad_game_streak'] >= 3).astype(int)
                
                # Añadir características de días de la semana y efectos de fin de semana
                # Calcular rendimiento medio por día de la semana para cada jugador
                if 'PTS' in player_data.columns and len(player_data) > 10:
                    for day in range(7):
                        day_mask = player_data['dayofweek'] == day
                        if day_mask.sum() >= 3:  # Al menos 3 partidos en este día
                            day_avg = player_data.loc[day_mask, 'PTS'].mean()
                            overall_avg = player_data['PTS'].mean()
                            
                            if pd.notna(day_avg) and pd.notna(overall_avg) and overall_avg > 0:
                                # Rendimiento relativo en este día comparado con su promedio
                                player_data.loc[day_mask, f'day{day}_performance'] = (day_avg / overall_avg - 1) * 100
                    
                    # Rendimiento en fin de semana vs. días laborables
                    weekend_mask = player_data['is_weekend'] == 1
                    weekday_mask = player_data['is_weekend'] == 0
                    
                    if weekend_mask.sum() >= 3 and weekday_mask.sum() >= 3:
                        weekend_avg = player_data.loc[weekend_mask, 'PTS'].mean()
                        weekday_avg = player_data.loc[weekday_mask, 'PTS'].mean()
                        
                        if pd.notna(weekend_avg) and pd.notna(weekday_avg) and weekday_avg > 0:
                            # Diferencia porcentual entre rendimiento en fin de semana vs. días laborables
                            weekend_diff = (weekend_avg / weekday_avg - 1) * 100
                            player_data['weekend_effect'] = weekend_diff
                            
                            # Categorizar efecto de fin de semana
                            if abs(weekend_diff) > 10:
                                if weekend_diff > 0:
                                    player_data['weekend_performer'] = 1
                                    player_data['weekday_performer'] = 0
                                else:
                                    player_data['weekend_performer'] = 0
                                    player_data['weekday_performer'] = 1
                            else:
                                player_data['weekend_performer'] = 0
                                player_data['weekday_performer'] = 0
                
                games_last_7_days.append(player_data)
            
            # Combinar todos los resultados
            self.players_data = pd.concat(games_last_7_days).sort_index()
            
            # Convertir categoría de densidad a variables dummy
            try:
                density_dummies = pd.get_dummies(
                    self.players_data['density_category'], 
                    prefix='density',
                    dummy_na=False
                )
                self.players_data = pd.concat([self.players_data, density_dummies], axis=1)
                logger.info("Variables dummy de densidad de calendario creadas correctamente")
            except Exception as e:
                logger.error(f"Error al crear dummies de densidad: {str(e)}")
            
            # Partidos jugados en la temporada
            self.players_data['games_played'] = self.players_data.groupby('Player').cumcount() + 1
            
            # Calcular el progreso de la temporada
            for player in self.players_data['Player'].unique():
                player_mask = self.players_data['Player'] == player
                player_games = self.players_data.loc[player_mask, 'games_played']
                max_games = player_games.max()
                self.players_data.loc[player_mask, 'season_progress'] = player_games / max(1, max_games)
            
            # Categorizar progreso de temporada
            progress_bins = [0, 0.25, 0.5, 0.75, 1.01]  # 1.01 para incluir 1.0
            progress_labels = ['start', 'early_mid', 'late_mid', 'end']
            
            self.players_data['progress_category'] = pd.cut(
                self.players_data['season_progress'],
                bins=progress_bins,
                labels=progress_labels,
                include_lowest=True
            )
            
            # Convertir categoría a variables dummy
            try:
                progress_dummies = pd.get_dummies(
                    self.players_data['progress_category'], 
                    prefix='progress',
                    dummy_na=False
                )
                self.players_data = pd.concat([self.players_data, progress_dummies], axis=1)
                logger.info("Variables dummy de progreso de temporada creadas correctamente")
            except Exception as e:
                logger.error(f"Error al crear dummies de progreso: {str(e)}")
            
            # Características de temporada específicas (All-Star Break, playoffs, etc.)
            try:
                # Aproximación para el All-Star Break (mediados de febrero)
                self.players_data['is_post_asg'] = ((self.players_data['month'] > 2) | 
                                             ((self.players_data['month'] == 2) & 
                                              (self.players_data['day_of_month'] > 15))).astype(int)
                
                # Última parte de la temporada (últimos 20 juegos aproximadamente)
                self.players_data['is_season_end'] = (self.players_data['season_progress'] > 0.75).astype(int)
                
                # Primera parte de la temporada (primeros 20 juegos aproximadamente)
                self.players_data['is_season_start'] = (self.players_data['season_progress'] < 0.25).astype(int)
                
                # Indicador de partido en seguidilla (back-to-back)
                self.players_data['is_back_to_back'] = (self.players_data['days_rest'] == 0).astype(int)
                
                # Indicador de tercer partido en 4 noches
                third_in_four = []
                
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data.loc[player_mask].sort_values('Date')
                    
                    # Inicializar
                    third_in_four_days = [0] * len(player_data)
                    
                    for i in range(len(player_data) - 1, 1, -1):  # Empezar desde el final
                        date = player_data.iloc[i]['Date']
                        date_minus_4 = date - pd.Timedelta(days=4)
                        
                        # Contar juegos en ventana de 4 días
                        games_in_window = sum((player_data['Date'] <= date) & 
                                           (player_data['Date'] > date_minus_4))
                        
                        if games_in_window >= 3:
                            third_in_four_days[i] = 1
                    
                    # Asignar al DataFrame
                    third_in_four.extend(third_in_four_days)
                
                if len(third_in_four) == len(self.players_data):
                    self.players_data['is_third_in_four'] = third_in_four
                
                # Índice compuesto de fatiga basado en múltiples factores
                if all(col in self.players_data.columns for col in 
                       ['games_last_7_days', 'days_rest', 'is_back_to_back', 'schedule_density_index']):
                    
                    self.players_data['fatigue_index'] = (
                        (self.players_data['games_last_7_days'] * 0.3) + 
                        ((3 - self.players_data['days_rest'].clip(0, 3)) * 0.3) + 
                        (self.players_data['is_back_to_back'] * 1.5) + 
                        (self.players_data['schedule_density_index'] * 0.4))
                    
                    # Normalizar a escala 0-10
                    self.players_data['fatigue_index'] = (self.players_data['fatigue_index'] / 5 * 10).clip(0, 10)
                    
                    # Categorizar índice de fatiga
                    fatigue_bins = [0, 2, 4, 6, 8, 10]
                    fatigue_labels = ['fresh', 'rested', 'normal', 'tired', 'exhausted']
                    
                    self.players_data['fatigue_category'] = pd.cut(
                        self.players_data['fatigue_index'],
                        bins=fatigue_bins,
                        labels=fatigue_labels,
                        include_lowest=True
                    )
            except Exception as e:
                logger.error(f"Error al crear características específicas de temporada: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Convertir otras categorías a variables dummy si existen
            try:
                for col in ['minutes_trend_category', 'momentum_category', 'fatigue_category']:
                    if col in self.players_data.columns and not self.players_data[col].isna().all():
                        dummies = pd.get_dummies(
                            self.players_data[col], 
                            prefix=col.split('_')[0],  # Usar prefijo más limpio
                            dummy_na=False
                        )
                        self.players_data = pd.concat([self.players_data, dummies], axis=1)
                        logger.info(f"Variables dummy de {col} creadas correctamente")
            except Exception as e:
                logger.error(f"Error al convertir categorías adicionales a dummies: {str(e)}")
            
            logger.info("Características temporales creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error al crear características temporales: {str(e)}")
            logger.error(traceback.format_exc())
            
        return self.players_data
    
    def _create_betting_line_features(self):
        """
        Crea características específicas para líneas de apuestas
        """
        logger.info("Creando características para líneas de apuestas")
        
        try:

            # Validar existencia de columnas de medias y desviaciones antes de procesar
            required_cols = {}
            for stat in self.betting_lines.keys():
                if stat in ['Double_Double', 'Triple_Double']:
                    col_name = stat.lower()
                    # Omitir validación de Double_Double y Triple_Double aquí
                    # Se manejarán específicamente en _create_double_triple_features
                    continue
                else:
                    col_name = stat
                
                if col_name not in self.players_data.columns:
                    logger.warning(f"La columna {col_name} no existe en el DataFrame, omitiendo líneas de apuesta")
                    continue
                
                required_cols[stat] = []
                for window in self.window_sizes:
                    mean_col = f'{col_name}_mean_{window}'
                    std_col = f'{col_name}_std_{window}'
                    if mean_col in self.players_data.columns and std_col in self.players_data.columns:
                        required_cols[stat].append(window)
                    else:
                        logger.warning(f"Columnas {mean_col} o {std_col} no existen, omitiendo ventana {window} para {stat}")


            # Diccionario para almacenar todas las nuevas características
            new_features = {}
            
            # Procesar cada línea de apuesta para cada estadística
            for stat, thresholds in self.betting_lines.items():
                # Omitir Double_Double y Triple_Double aquí, se manejan por separado
                if stat in ['Double_Double', 'Triple_Double']:
                    continue
                    
                if stat not in self.players_data.columns:
                    logger.warning(f"La columna {stat} no existe en el DataFrame, omitiendo líneas de apuesta")
                    continue
                
                # Para cada umbral, calcular la probabilidad histórica de superarlo
                for threshold in thresholds:
                    # Crear la característica binaria
                    over_col = f'{stat}_over_{threshold}'
                    self.players_data[over_col] = (self.players_data[stat] > threshold).astype(int)
                    
                    # Para cada ventana, calcular la tasa de superación móvil
                    for window in self.window_sizes:
                        # Verificar si ya tenemos las medias
                        if f'{stat}_mean_{window}' in self.players_data.columns:
                            # Crear un DataFrame temporal para los cálculos
                            temp_df = pd.DataFrame({
                                'Player': self.players_data['Player'],
                                'mean': self.players_data[f'{stat}_mean_{window}']
                            })
                            
                            # Calcular la diferencia con la línea
                            temp_df['line_diff'] = temp_df['mean'] - threshold
                            
                            # Guardar la diferencia
                            self.players_data[f'{stat}_line_diff_{threshold}_{window}'] = temp_df['line_diff'].values
                            
                            # Probabilidad de superar la línea
                            self.players_data[f'{stat}_prob_over_{threshold}_{window}'] = (temp_df['line_diff'] > 0).astype(float).values
                            
                            # Calcular estadísticas por jugador
                            grouped = temp_df.groupby('Player')['line_diff']
                            
                            # Volatilidad con manejo robusto de valores
                            try:
                                # Convertir a numérico y aplicar rolling std de forma segura
                                volatility = grouped.transform(
                                    lambda x: pd.to_numeric(x, errors='coerce')
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0)
                                          .rolling(window=min(window, len(x)), min_periods=1)
                                          .std()
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0)
                                )
                                
                                # Limitar a rango razonable según la estadística
                                if stat == 'PTS':
                                    volatility = volatility.clip(0, 15)
                                elif stat in ['TRB', 'AST']:
                                    volatility = volatility.clip(0, 8)
                                elif stat == '3P':
                                    volatility = volatility.clip(0, 5)
                                else:
                                    volatility = volatility.clip(0, 10)
                                    
                                self.players_data[f'{stat}_line_volatility_{threshold}_{window}'] = volatility.values
                            except Exception as e:
                                logger.error(f"Error al calcular {stat}_line_volatility_{threshold}_{window}: {str(e)}")
                                # Valor por defecto en caso de error
                                self.players_data[f'{stat}_line_volatility_{threshold}_{window}'] = np.zeros(len(self.players_data))
                            
                            # Consistencia para superar la línea
                            try:
                                # Calcular tasa de superación de forma segura
                                over_rates = self.players_data.groupby('Player')[over_col].transform(
                                    lambda x: pd.to_numeric(x, errors='coerce')
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0)
                                          .rolling(window=min(window, len(x)), min_periods=1)
                                          .mean()
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0.5)
                                )
                                
                                # Calcular consistencia (qué tan constante es la tasa)
                                over_std = self.players_data.groupby('Player')[over_col].transform(
                                    lambda x: pd.to_numeric(x, errors='coerce')
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0)
                                          .rolling(window=min(window, len(x)), min_periods=1)
                                          .std()
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0)
                                )
                                
                                # Una baja desviación estándar significa alta consistencia
                                consistency = 1 - over_std.clip(0, 1)
                                
                                self.players_data[f'{stat}_line_consistency_{threshold}_{window}'] = consistency
                                self.players_data[f'{stat}_over_rate_{threshold}_{window}'] = over_rates
                                
                            except Exception as e:
                                logger.error(f"Error al calcular consistencia para {stat}_line_{threshold}_{window}: {str(e)}")
                                # Valores por defecto en caso de error
                                self.players_data[f'{stat}_line_consistency_{threshold}_{window}'] = 0.5
                                self.players_data[f'{stat}_over_rate_{threshold}_{window}'] = 0.5
                            
                            # Rachas para superar la línea
                            try:
                                for player in self.players_data['Player'].unique():
                                    player_mask = self.players_data['Player'] == player
                                    player_data = self.players_data.loc[player_mask].sort_values('Date').copy()
                                    
                                    # Inicializar rachas
                                    streaks = []
                                    current_streak = 0
                                    
                                    # Calcular racha para cada partido
                                    for over in player_data[over_col]:
                                        if over == 1:
                                            current_streak += 1
                                        else:
                                            current_streak = 0
                                        streaks.append(current_streak)
                                    
                                    # Asignar rachas al DataFrame
                                    self.players_data.loc[player_data.index, f'{stat}_over_streak_{threshold}'] = streaks
                            except Exception as e:
                                logger.error(f"Error al calcular rachas para {stat}_over_{threshold}: {str(e)}")
                                # Valor por defecto en caso de error
                                self.players_data[f'{stat}_over_streak_{threshold}'] = 0
                    
                    try:
                        if stat in self.players_data.columns:
                            # Calcular margen sobre la línea, normalizado por la desviación estándar
                            for window in self.window_sizes:
                                stat_mean_col = f'{stat}_mean_{window}'
                                stat_std_col = f'{stat}_std_{window}'
                                
                                if stat_mean_col in self.players_data.columns and stat_std_col in self.players_data.columns:
                                    # Calcular z-score respecto a la línea de forma segura
                                    # Primero convertir a valores numéricos y eliminar valores problemáticos
                                    means = pd.to_numeric(self.players_data[stat_mean_col], errors='coerce')
                                    stds = pd.to_numeric(self.players_data[stat_std_col], errors='coerce')
                                    
                                    # Reemplazar valores problemáticos
                                    means = means.replace([np.inf, -np.inf], np.nan).fillna(threshold)
                                    # Aumentamos el valor mínimo de std para evitar divisiones por valores muy pequeños
                                    stds = stds.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1.0)
                                    
                                    # Asegurar que el denominador no sea demasiado pequeño
                                    # Cálculo de edge de forma segura
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        edge_raw = (means - threshold) / stds
                                    
                                    # Reemplazar valores inválidos manualmente
                                    edge = np.where(np.isfinite(edge_raw), edge_raw, 0)
                                    edge = np.clip(edge, -5, 5)
                                    
                                    # Asignar al DataFrame
                                    self.players_data[f'{stat}_edge_{threshold}_{window}'] = edge
                    except Exception as e:
                        logger.error(f"Error al calcular edge para {stat}_{threshold}: {str(e)}")
                        logger.error(traceback.format_exc())
                    
                    # 2. Características situacionales (casa/visitante, victoria/derrota)
                    try:
                        # Rendimiento sobre la línea en casa vs visitante
                        if 'is_home' in self.players_data.columns:
                            for player in self.players_data['Player'].unique():
                                player_mask = self.players_data['Player'] == player
                                
                                # Filtrar partidos en casa y fuera
                                home_mask = player_mask & (self.players_data['is_home'] == True)
                                away_mask = player_mask & (self.players_data['is_home'] == False)
                                
                                if home_mask.sum() >= 3 and away_mask.sum() >= 3:
                                    # Calcular tasa de superación en casa (con manejo seguro de valores)
                                    home_over_rate = self.players_data.loc[home_mask, over_col].astype(float).mean()
                                    if not np.isfinite(home_over_rate):
                                        home_over_rate = 0.5
                                        
                                    # Calcular tasa de superación fuera (con manejo seguro de valores)
                                    away_over_rate = self.players_data.loc[away_mask, over_col].astype(float).mean()
                                    if not np.isfinite(away_over_rate):
                                        away_over_rate = 0.5
                                    
                                    # Asignar a todos los partidos del jugador
                                    self.players_data.loc[player_mask, f'{stat}_home_over_rate_{threshold}'] = home_over_rate
                                    self.players_data.loc[player_mask, f'{stat}_away_over_rate_{threshold}'] = away_over_rate
                                    
                                    # Calcular la diferencia de forma segura
                                    home_away_diff = home_over_rate - away_over_rate
                                    self.players_data.loc[player_mask, f'{stat}_home_away_diff_{threshold}'] = home_away_diff
                        
                        # Rendimiento sobre la línea en victorias vs derrotas
                        if 'is_win' in self.players_data.columns:
                            for player in self.players_data['Player'].unique():
                                player_mask = self.players_data['Player'] == player
                                
                                # Filtrar victorias y derrotas
                                win_mask = player_mask & (self.players_data['is_win'] == True)
                                loss_mask = player_mask & (self.players_data['is_win'] == False)
                                
                                if win_mask.sum() >= 3 and loss_mask.sum() >= 3:
                                    # Calcular tasa de superación en victorias
                                    win_over_rate = self.players_data.loc[win_mask, over_col].astype(float).mean()
                                    if not np.isfinite(win_over_rate):
                                        win_over_rate = 0.5
                                    
                                    # Calcular tasa de superación en derrotas
                                    loss_over_rate = self.players_data.loc[loss_mask, over_col].astype(float).mean()
                                    if not np.isfinite(loss_over_rate):
                                        loss_over_rate = 0.5
                                    
                                    # Asignar a todos los partidos del jugador
                                    self.players_data.loc[player_mask, f'{stat}_win_over_rate_{threshold}'] = win_over_rate
                                    self.players_data.loc[player_mask, f'{stat}_loss_over_rate_{threshold}'] = loss_over_rate
                                    
                                    # Diferencia
                                    win_loss_diff = win_over_rate - loss_over_rate
                                    self.players_data.loc[player_mask, f'{stat}_win_loss_diff_{threshold}'] = win_loss_diff
                    except Exception as e:
                        logger.error(f"Error al calcular características situacionales para {stat}_{threshold}: {str(e)}")
                        logger.error(traceback.format_exc())
            
            # 3. Doble-Doble y Triple-Doble
            for stat in ['Double_Double', 'Triple_Double']:
                if stat in self.betting_lines:
                    thresholds = self.betting_lines[stat]
                    
                    # Usar la columna correspondiente
                    col_name = stat.lower()
                    if col_name in self.players_data.columns:
                        for threshold in thresholds:
                            # Crear variable sobre umbral
                            over_col = f'{col_name}_over_{threshold}'
                            self.players_data[over_col] = (self.players_data[col_name] > threshold).astype(int)
                            
                            # Calcular tasa de logro para diferentes ventanas
                            for window in self.window_sizes:
                                # Usar la función safe_rolling para el cálculo
                                self.players_data[f'{col_name}_rate_{window}'] = self.players_data.groupby('Player')[col_name].transform(
                                    lambda x: self._safe_rolling(x, window=window, operation='mean', min_periods=1)
                                )
            
            logger.info("Características para líneas de apuestas creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error al crear características para líneas de apuestas: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.players_data
    
    def _create_double_triple_features(self):
        """
        Crea características DERIVADAS para predicciones de doble-doble y triple-doble.
        """
        logger.info("Creando características derivadas para doble-doble y triple-doble")
        
        try:
            # 1. Calcular dobles y triples dobles base
            stats_for_double = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
            available_stats = [stat for stat in stats_for_double if stat in self.players_data.columns]
            
            # Asegurar que las estadísticas base sean numéricas
            for stat in available_stats:
                self.players_data[stat] = pd.to_numeric(self.players_data[stat], errors='coerce').fillna(0)
            
            # Crear columnas X_double
            for stat in available_stats:
                x_double_col = f'{stat}_double'
                self.players_data[x_double_col] = (self.players_data[stat] >= 10).astype(int)
                logger.info(f"Columna {x_double_col} creada")
            
            # Calcular double_double y triple_double
            double_cols = [f'{stat}_double' for stat in available_stats]
            self.players_data['double_double'] = (self.players_data[double_cols].sum(axis=1) >= 2).astype(int)
            self.players_data['triple_double'] = (self.players_data[double_cols].sum(axis=1) >= 3).astype(int)
            
            # Verificación y logging inicial
            dd_count = self.players_data['double_double'].sum()
            td_count = self.players_data['triple_double'].sum()
            logger.info(f"Se identificaron {dd_count} doble-dobles y {td_count} triple-dobles")
            
            # Verificar casos con PTS=0
            zero_pts_dd = self.players_data[(self.players_data['PTS'] < 1) & (self.players_data['double_double'] == 1)]
            if not zero_pts_dd.empty:
                logger.info(f"VALIDACIÓN: Hay {len(zero_pts_dd)} doble-dobles sin puntos (legítimos)")
            
            # 2. Cálculo de tendencias temporales (tasas, medias, desviaciones)
            # Características para doble-doble
            for window in self.window_sizes:
                try:
                    # Tasa de doble-doble por jugador a lo largo del tiempo
                    self.players_data[f'double_double_rate_{window}'] = self.players_data.groupby('Player')['double_double'].transform(
                        lambda x: pd.to_numeric(x, errors='coerce')
                                .rolling(window=min(window, len(x)), min_periods=1)
                                .mean()
                                .fillna(0)
                    )
                    
                    # Media móvil para double_double
                    if f'double_double_mean_{window}' not in self.players_data.columns:
                        self.players_data[f'double_double_mean_{window}'] = self.players_data[f'double_double_rate_{window}']
                    
                    # Desviación estándar móvil para double_double
                    if f'double_double_std_{window}' not in self.players_data.columns:
                        self.players_data[f'double_double_std_{window}'] = self.players_data.groupby('Player')['double_double'].transform(
                            lambda x: pd.to_numeric(x, errors='coerce')
                                    .rolling(window=min(window, len(x)), min_periods=1)
                                    .std()
                                    .fillna(0)
                        )
                    
                    # 3. Rachas de doble-dobles
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        player_data = self.players_data.loc[player_mask].sort_values('Date').copy()
                        streaks = []
                        current_streak = 0
                        for dd in player_data['double_double']:
                            if dd == 1:
                                current_streak += 1
                            else:
                                current_streak = 0
                            streaks.append(current_streak)
                        self.players_data.loc[player_data.index, 'double_double_streak'] = streaks
                        
                    # 4. Características de triple-doble similares
                    self.players_data[f'triple_double_rate_{window}'] = self.players_data.groupby('Player')['triple_double'].transform(
                        lambda x: pd.to_numeric(x, errors='coerce')
                                .rolling(window=min(window, len(x)), min_periods=1)
                                .mean()
                                .fillna(0)
                    )
                    
                    if f'triple_double_mean_{window}' not in self.players_data.columns:
                        self.players_data[f'triple_double_mean_{window}'] = self.players_data[f'triple_double_rate_{window}']
                    
                    if f'triple_double_std_{window}' not in self.players_data.columns:
                        self.players_data[f'triple_double_std_{window}'] = self.players_data.groupby('Player')['triple_double'].transform(
                            lambda x: pd.to_numeric(x, errors='coerce')
                                    .rolling(window=min(window, len(x)), min_periods=1)
                                    .std()
                                    .fillna(0)
                        )
                    
                    # Rachas de triple-dobles
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        player_data = self.players_data.loc[player_mask].sort_values('Date').copy()
                        streaks = []
                        current_streak = 0
                        for td in player_data['triple_double']:
                            if td == 1:
                                current_streak += 1
                            else:
                                current_streak = 0
                            streaks.append(current_streak)
                        self.players_data.loc[player_data.index, 'triple_double_streak'] = streaks
                    
                except Exception as e:
                    logger.error(f"Error al calcular características derivadas con ventana {window}: {str(e)}")
            
            # 5. Características para pronósticos basados en umbrales de apuestas
            for stat in ['Double_Double', 'Triple_Double']:
                col_name = stat.lower()
                if col_name not in self.players_data.columns:
                    continue
                    
                # Verificar si el stat existe en betting_lines
                if stat not in self.betting_lines:
                    continue
                    
                for threshold in self.betting_lines[stat]:
                    try:
                        # Columna sobre umbral (sin modificar las columnas base)
                        over_col = f'{col_name}_over_{threshold}'
                        self.players_data[over_col] = (self.players_data[col_name] > threshold).astype(int)
                        
                        # Probabilidades por ventana
                        for window in self.window_sizes:
                            prob_col = f'{col_name}_prob_over_{threshold}_{window}'
                            try:
                                self.players_data[prob_col] = self.players_data.groupby('Player')[over_col].transform(
                                    lambda x: pd.to_numeric(x, errors='coerce')
                                            .rolling(window=min(window, len(x)), min_periods=1)
                                            .mean()
                                            .fillna(0)
                                )
                                
                                # Tendencia respecto a ventanas más grandes
                                if window < self.window_sizes[-1]:
                                    next_window = next((w for w in self.window_sizes if w > window), None)
                                    if next_window:
                                        next_prob_col = f'{col_name}_prob_over_{threshold}_{next_window}'
                                        
                                        if next_prob_col in self.players_data.columns:
                                            trend_col = f'{col_name}_trend_{threshold}_{window}'
                                            self.players_data[trend_col] = (
                                                self.players_data[prob_col] - 
                                                self.players_data[next_prob_col]
                                            ).clip(-1, 1).fillna(0)
                            except Exception as e:
                                logger.error(f"Error al calcular {prob_col}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error al crear características de pronóstico para {col_name}: {str(e)}")

            # 6. Rellenar valores nulos en columnas derivadas
            double_triple_cols = [col for col in self.players_data.columns 
                                 if ('double' in col or 'triple' in col) and col not in ['double_double', 'triple_double']]
            for col in double_triple_cols:
                if self.players_data[col].isnull().any():
                    self.players_data[col] = self.players_data[col].fillna(0)

            logger.info("Características derivadas para doble-doble y triple-doble creadas correctamente")
        
        except Exception as e:
            logger.error(f"Error al crear características derivadas: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
        
        return self.players_data
    
    def _create_physical_features(self):
        """
        Crea características basadas en los atributos físicos de los jugadores
        """
        logger.info("Creando características físicas del jugador")
        
        try:
            # Verificar si tenemos columnas con datos físicos
            physical_cols = [col for col in self.players_data.columns if col in 
                           ['Height_Inches', 'Weight', 'BMI']]
            
            logger.info(f"Columnas físicas encontradas: {physical_cols}")
            
            if not physical_cols:
                logger.warning("No se encontraron columnas con datos físicos. No se crearán características físicas.")
                return
            
            # Procesamiento de altura
            if 'Height_Inches' in self.players_data.columns:
                try:
                    # Convertir a numérico asegurando que se manejan correctamente los valores
                    self.players_data['Height_Inches'] = pd.to_numeric(self.players_data['Height_Inches'], errors='coerce')
                    
                    # Categorizar por altura
                    height_bins = [0, 72, 76, 80, 84, 100]  # Bins en pulgadas (6', 6'4", 6'8", 7')
                    height_labels = ['short', 'average', 'tall', 'very_tall', 'extremely_tall']
                    
                    self.players_data['height_category'] = pd.cut(
                        self.players_data['Height_Inches'], 
                        bins=height_bins, 
                        labels=height_labels,
                        include_lowest=True
                    )
                    
                    # Convertir categoría a variables dummy
                    if not self.players_data['height_category'].isna().all():
                        try:
                            height_dummies = pd.get_dummies(
                                self.players_data['height_category'], 
                                prefix='height',
                                dummy_na=False
                            )
                            self.players_data = pd.concat([self.players_data, height_dummies], axis=1)
                            logger.info("Categorías de altura creadas correctamente")
                        except Exception as e:
                            logger.error(f"Error al crear dummies de altura: {str(e)}")
                            
                    # Agregar atributos de altura más específicos
                    # Normalizar altura (z-score) para tener idea de cuán alta/baja es la persona respecto a la media
                    height_mean = self.players_data['Height_Inches'].mean()
                    height_std = self.players_data['Height_Inches'].std()
                    if height_std > 0:
                        self.players_data['height_z_score'] = ((self.players_data['Height_Inches'] - height_mean) / height_std).clip(-3, 3)
                    else:
                        self.players_data['height_z_score'] = 0
                        
                    # Altura al cuadrado (para capturar efectos no lineales)
                    self.players_data['height_squared'] = (self.players_data['Height_Inches'] ** 2) / 1000  # Normalizado
                    
                    # Altura relativa por posición
                    if 'Pos' in self.players_data.columns:
                        for pos in self.players_data['Pos'].unique():
                            if pd.notna(pos):
                                pos_mask = self.players_data['Pos'] == pos
                                if pos_mask.sum() > 5:  # Suficientes jugadores para calcular
                                    pos_height_mean = self.players_data.loc[pos_mask, 'Height_Inches'].mean()
                                    pos_height_std = self.players_data.loc[pos_mask, 'Height_Inches'].std()
                                    
                                    if pd.notna(pos_height_mean) and pos_height_std > 0:
                                        # Altura en relación con la posición (percentil dentro de su posición)
                                        rel_height = (self.players_data.loc[pos_mask, 'Height_Inches'] - pos_height_mean) / pos_height_std
                                        self.players_data.loc[pos_mask, f'height_rel_to_{pos}'] = rel_height.clip(-3, 3)
                                        
                                        # Ventaja/desventaja de altura para la posición
                                        # Un valor alto indica una ventaja significativa
                                        self.players_data.loc[pos_mask, f'height_advantage_{pos}'] = (
                                            (self.players_data.loc[pos_mask, 'Height_Inches'] / pos_height_mean) - 1
                                        ).clip(-0.15, 0.15)
                    
                except Exception as e:
                    logger.error(f"Error al procesar altura (Height_Inches): {str(e)}")
            
            # Procesamiento de peso
            if 'Weight' in self.players_data.columns:
                try:
                    # Convertir peso a numérico
                    self.players_data['Weight'] = pd.to_numeric(self.players_data['Weight'], errors='coerce')
                    
                    # Categorizar por peso
                    weight_bins = [0, 180, 200, 220, 240, 300]  # en libras
                    weight_labels = ['light', 'average', 'heavy', 'very_heavy', 'extremely_heavy']
                    
                    self.players_data['weight_category'] = pd.cut(
                        self.players_data['Weight'], 
                        bins=weight_bins, 
                        labels=weight_labels,
                        include_lowest=True
                    )
                    
                    # Convertir categoría a variables dummy
                    if not self.players_data['weight_category'].isna().all():
                        try:
                            weight_dummies = pd.get_dummies(
                                self.players_data['weight_category'], 
                                prefix='weight',
                                dummy_na=False
                            )
                            self.players_data = pd.concat([self.players_data, weight_dummies], axis=1)
                            logger.info("Categorías de peso creadas correctamente")
                        except Exception as e:
                            logger.error(f"Error al crear dummies de peso: {str(e)}")
                            
                    # Agregar atributos de peso más específicos
                    # Normalizar peso (z-score)
                    weight_mean = self.players_data['Weight'].mean()
                    weight_std = self.players_data['Weight'].std()
                    if weight_std > 0:
                        self.players_data['weight_z_score'] = ((self.players_data['Weight'] - weight_mean) / weight_std).clip(-3, 3)
                    else:
                        self.players_data['weight_z_score'] = 0
                        
                    # Peso al cuadrado (para capturar efectos no lineales)
                    self.players_data['weight_squared'] = (self.players_data['Weight'] ** 2) / 10000  # Normalizado
                    
                    # Peso relativo por posición
                    if 'Pos' in self.players_data.columns:
                        for pos in self.players_data['Pos'].unique():
                            if pd.notna(pos):
                                pos_mask = self.players_data['Pos'] == pos
                                if pos_mask.sum() > 5:  # Suficientes jugadores para calcular
                                    pos_weight_mean = self.players_data.loc[pos_mask, 'Weight'].mean()
                                    pos_weight_std = self.players_data.loc[pos_mask, 'Weight'].std()
                                    
                                    if pd.notna(pos_weight_mean) and pos_weight_std > 0:
                                        # Peso en relación con la posición
                                        rel_weight = (self.players_data.loc[pos_mask, 'Weight'] - pos_weight_mean) / pos_weight_std
                                        self.players_data.loc[pos_mask, f'weight_rel_to_{pos}'] = rel_weight.clip(-3, 3)
                                        
                                        # Ventaja/desventaja de peso para la posición
                                        self.players_data.loc[pos_mask, f'weight_advantage_{pos}'] = (
                                            (self.players_data.loc[pos_mask, 'Weight'] / pos_weight_mean) - 1
                                        ).clip(-0.2, 0.2)
                    
                except Exception as e:
                    logger.error(f"Error al procesar peso (Weight): {str(e)}")
            
            # Procesamiento de BMI
            if 'BMI' in self.players_data.columns:
                try:
                    # Convertir BMI a numérico
                    self.players_data['BMI'] = pd.to_numeric(self.players_data['BMI'], errors='coerce')
                    
                    # Si BMI no está disponible pero tenemos altura y peso, calcularlo
                    if self.players_data['BMI'].isna().all() and 'Height_Inches' in self.players_data.columns and 'Weight' in self.players_data.columns:
                        # BMI = (peso en libras * 703) / (altura en pulgadas)^2
                        height_squared = np.maximum(1, self.players_data['Height_Inches'] ** 2)  # Evitar división por cero
                        self.players_data['BMI'] = (self.players_data['Weight'] * 703) / height_squared
                    
                    # Limitar a valores razonables
                    self.players_data['BMI'] = self.players_data['BMI'].clip(15, 40)
                    
                    # Categorizar BMI
                    bmi_bins = [0, 18.5, 25, 30, 40]
                    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
                    
                    self.players_data['bmi_category'] = pd.cut(
                        self.players_data['BMI'], 
                        bins=bmi_bins, 
                        labels=bmi_labels,
                        include_lowest=True
                    )
                    
                    # Convertir categoría a variables dummy
                    if not self.players_data['bmi_category'].isna().all():
                        try:
                            bmi_dummies = pd.get_dummies(
                                self.players_data['bmi_category'], 
                                prefix='bmi',
                                dummy_na=False
                            )
                            self.players_data = pd.concat([self.players_data, bmi_dummies], axis=1)
                            logger.info("Categorías de BMI creadas correctamente")
                        except Exception as e:
                            logger.error(f"Error al crear dummies de BMI: {str(e)}")
                            
                    # Agregar características de BMI más avanzadas
                    # BMI al cuadrado (efectos no lineales)
                    self.players_data['bmi_squared'] = (self.players_data['BMI'] ** 2) / 100  # Normalizado
                    
                    # BMI categorizado por posición (si tenemos posición)
                    if 'Pos' in self.players_data.columns:
                        for pos in self.players_data['Pos'].unique():
                            if pd.notna(pos):
                                pos_mask = self.players_data['Pos'] == pos
                                if pos_mask.sum() > 5:  # Suficientes jugadores para calcular
                                    pos_bmi_mean = self.players_data.loc[pos_mask, 'BMI'].mean()
                                    if pd.notna(pos_bmi_mean) and pos_bmi_mean > 0:
                                        # BMI relativo a la posición (qué tan por encima/debajo de la media estás)
                                        self.players_data.loc[pos_mask, f'bmi_rel_to_{pos}'] = (
                                            self.players_data.loc[pos_mask, 'BMI'] / pos_bmi_mean
                                        ).clip(0.8, 1.2)
                        
                except Exception as e:
                    logger.error(f"Error al procesar BMI: {str(e)}")
            
            # Crear características de relación entre métricas físicas
            try:
                if 'Height_Inches' in self.players_data.columns and 'Weight' in self.players_data.columns:
                    # Índice de estructura física (peso/altura) - útil para identificar jugadores más compactos vs. estilizados
                    height_safe = np.maximum(60, self.players_data['Height_Inches'])  # Asegurar valor mínimo para división segura
                    self.players_data['weight_height_ratio'] = (self.players_data['Weight'] / height_safe).clip(1.5, 4)
                    
                    # Categorizar esta relación
                    ratio_bins = [0, 2.3, 2.6, 3.0, 4.0]
                    ratio_labels = ['lean', 'average_build', 'solid', 'strong']
                    
                    self.players_data['build_category'] = pd.cut(
                        self.players_data['weight_height_ratio'], 
                        bins=ratio_bins, 
                        labels=ratio_labels,
                        include_lowest=True
                    )
                    
                    # Convertir categoría a variables dummy
                    if not self.players_data['build_category'].isna().all():
                        build_dummies = pd.get_dummies(
                            self.players_data['build_category'], 
                            prefix='build',
                            dummy_na=False
                        )
                        self.players_data = pd.concat([self.players_data, build_dummies], axis=1)
                        logger.info("Categorías de estructura física creadas correctamente")
                        
                    # Crear características más avanzadas de estructura física
                    # Nuevo índice: Índice de potencia (combinación raíz cuadrada de (altura × peso))
                    self.players_data['power_index'] = np.sqrt(self.players_data['Height_Inches'] * self.players_data['Weight']) / 10
                    
                    # Índice de masa muscular (aproximado): weight / height^1.5 
                    # Una mejor aproximación que BMI para atletas con mucha masa muscular
                    height_power = np.power(np.maximum(60, self.players_data['Height_Inches']), 1.5)  # Evitar potencias negativas
                    self.players_data['muscle_mass_index'] = (
                        self.players_data['Weight'] / height_power
                    ).clip(0.1, 1.0)
                    
                    # Índice de área corporal (estima superficie): √(altura × peso / 3600)
                    self.players_data['body_surface_index'] = np.sqrt(
                        (self.players_data['Height_Inches'] * self.players_data['Weight']) / 3600
                    ).clip(0.8, 2.0)
                    
                    # Ectomorfia (delgadez relativa): 0.732 * altura / raíz cúbica del peso
                    height_cm = self.players_data['Height_Inches'] * 2.54  # Convertir a cm
                    weight_kg = self.players_data['Weight'] * 0.453592  # Convertir a kg
                    
                    # Evitar valores con raíz cúbica problemática
                    valid_weight = weight_kg.clip(30, 200)
                    weight_cuberoot = np.cbrt(valid_weight)
                    
                    self.players_data['ectomorphy'] = (0.732 * height_cm / weight_cuberoot).clip(0.5, 9)
                    
                    # Mesomorfia (musculatura relativa): aproximación básica usando BMI y altura
                    if 'BMI' in self.players_data.columns:
                        self.players_data['mesomorphy'] = (
                            (self.players_data['BMI'] / 20) * (self.players_data['Height_Inches'] / 70)
                        ).clip(0.5, 9)
                        
                    # Endomorfia (gordura relativa): aproximación usando peso y altura
                    self.players_data['endomorphy'] = (
                        (self.players_data['Weight'] / self.players_data['Height_Inches']) / 2.3
                    ).clip(0.5, 9)
                    
                    # Añadir índices de idoneidad física por posición
                    if 'Pos' in self.players_data.columns:
                        # Definir características ideales por posición (basadas en conocimiento baloncesto)
                        position_ideals = {
                            'PG': {'height': 74, 'agility': 0.9, 'power': 0.5},  # Point Guard
                            'SG': {'height': 76, 'agility': 0.8, 'power': 0.6},  # Shooting Guard
                            'SF': {'height': 79, 'agility': 0.7, 'power': 0.7},  # Small Forward
                            'PF': {'height': 81, 'agility': 0.6, 'power': 0.8},  # Power Forward
                            'C': {'height': 84, 'agility': 0.5, 'power': 0.9}    # Center
                        }
                        
                        # Crear una aproximación de agilidad basada en altura y peso
                        # Jugadores más altos y pesados tienden a ser menos ágiles
                        height_factor = (80 - self.players_data['Height_Inches']) / 10  # Mayor si es más bajo
                        weight_factor = (230 - self.players_data['Weight']) / 50  # Mayor si es más ligero
                        self.players_data['agility_approx'] = (height_factor + weight_factor + 1) / 3
                        self.players_data['agility_approx'] = self.players_data['agility_approx'].clip(0.4, 1.0)
                        
                        # Aproximación de potencia basada en peso y altura
                        self.players_data['power_approx'] = (self.players_data['power_index'] / 25).clip(0.4, 1.0)
                        
                        # Para cada posición definida, calcular índice de idoneidad física
                        for pos_code, ideal in position_ideals.items():
                            # Calcular similitud con las características ideales
                            height_match = 1 - np.abs(self.players_data['Height_Inches'] - ideal['height']) / 10
                            agility_match = 1 - np.abs(self.players_data['agility_approx'] - ideal['agility']) / 0.5
                            power_match = 1 - np.abs(self.players_data['power_approx'] - ideal['power']) / 0.5
                            
                            # Combinar en un índice de idoneidad para la posición
                            self.players_data[f'physical_fit_{pos_code}'] = (
                                (height_match * 0.5) + (agility_match * 0.3) + (power_match * 0.2)
                            ).clip(0, 1)
                    
                    # Índice de eficiencia física (basado en rendimiento estadístico vs físico)
                    if 'PTS' in self.players_data.columns and 'TRB' in self.players_data.columns:
                        # Calcular producción por unidad de masa corporal
                        # Detecta jugadores que producen mucho a pesar de su tamaño
                        self.players_data['pts_per_weight'] = (self.players_data['PTS'] / self.players_data['Weight']) * 100
                        self.players_data['trb_per_weight'] = (self.players_data['TRB'] / self.players_data['Weight']) * 100
                        
                        # Eficiencia física general (producción total por unidad de masa)
                        if 'AST' in self.players_data.columns:
                            self.players_data['physical_efficiency'] = (
                                (self.players_data['PTS'] + self.players_data['TRB'] * 1.5 + self.players_data['AST']) / 
                                self.players_data['Weight']
                            ) * 10
                            self.players_data['physical_efficiency'] = self.players_data['physical_efficiency'].clip(0.2, 2.0)
                            
                    # Calcular estimación del porcentaje de grasa corporal basado en BMI y edad
                    if 'BMI' in self.players_data.columns and 'Age' in self.players_data.columns:
                        # Fórmula simplificada para estimar grasa corporal basada en BMI y edad
                        # %Grasa = (1.2 * BMI) + (0.23 * edad) - 16.2
                        # Ajustada para atletas con menor grasa que población general
                        self.players_data['body_fat_est'] = (
                            (1.0 * self.players_data['BMI']) + 
                            (0.1 * self.players_data['Age']) - 
                            16.0
                        ).clip(5, 25)  # Limita a rango realista para jugadores NBA
                        
                        # Crear categorías de grasa corporal
                        fat_bins = [0, 8, 12, 16, 25]
                        fat_labels = ['very_lean', 'lean', 'average', 'above_average']
                        
                        self.players_data['fat_category'] = pd.cut(
                            self.players_data['body_fat_est'],
                            bins=fat_bins,
                            labels=fat_labels,
                            include_lowest=True
                        )
                        
                        # Convertir a dummies
                        if not self.players_data['fat_category'].isna().all():
                            fat_dummies = pd.get_dummies(
                                self.players_data['fat_category'],
                                prefix='fat',
                                dummy_na=False
                            )
                            self.players_data = pd.concat([self.players_data, fat_dummies], axis=1)
                    
            except Exception as e:
                logger.error(f"Error al crear características de relación física: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Crear características para la posición si está disponible
            if 'Pos' in self.players_data.columns:
                try:
                    # Crear variables dummy para posiciones
                    position_dummies = pd.get_dummies(
                        self.players_data['Pos'], 
                        prefix='pos',
                        dummy_na=False
                    )
                    self.players_data = pd.concat([self.players_data, position_dummies], axis=1)
                    logger.info("Variables dummy de posición creadas correctamente")
                    
                    # Categorías más amplias basadas en las posiciones específicas disponibles
                    # G, G-F -> Guards
                    # F, F-G, F-C -> Forwards
                    # C, C-F -> Centers
                    
                    # Identificar guards (base y escolta)
                    self.players_data['is_guard'] = self.players_data['Pos'].isin(['G', 'G-F', 'PG', 'SG']).astype(int)
                    
                    # Identificar forwards (alero)
                    self.players_data['is_forward'] = self.players_data['Pos'].isin(['F', 'F-G', 'F-C', 'SF', 'PF']).astype(int)
                    
                    # Identificar centers (pivots)
                    self.players_data['is_center'] = self.players_data['Pos'].isin(['C', 'C-F']).astype(int)
                    
                    # Métricas de adecuación física para la posición jugada
                    # Basada en altura, peso e índices físicos calculados anteriormente
                    if 'Height_Inches' in self.players_data.columns and 'Weight' in self.players_data.columns:
                        # Guards: valora altura moderada, menor peso
                        guard_score = (
                            (1 - np.abs(self.players_data['Height_Inches'] - 75) / 10) * 0.5 +
                            (1 - (self.players_data['Weight'] / 230)) * 0.5
                        )
                        self.players_data['guard_physique_score'] = guard_score.clip(0, 1)
                        
                        # Forwards: valora altura intermedia, peso intermedio
                        forward_score = (
                            (1 - np.abs(self.players_data['Height_Inches'] - 80) / 10) * 0.5 +
                            (1 - np.abs(self.players_data['Weight'] - 225) / 50) * 0.5
                        )
                        self.players_data['forward_physique_score'] = forward_score.clip(0, 1)
                        
                        # Centers: valora mayor altura y peso
                        center_score = (
                            (self.players_data['Height_Inches'] / 84) * 0.6 +
                            (self.players_data['Weight'] / 250) * 0.4
                        )
                        self.players_data['center_physique_score'] = center_score.clip(0, 1)
                        
                        # Crear índice de alineación física con posición actual
                        self.players_data['physique_position_alignment'] = (
                            self.players_data['guard_physique_score'] * self.players_data['is_guard'] +
                            self.players_data['forward_physique_score'] * self.players_data['is_forward'] +
                            self.players_data['center_physique_score'] * self.players_data['is_center']
                        )
                    
                except Exception as e:
                    logger.error(f"Error al crear características de posición: {str(e)}")
                    logger.error(traceback.format_exc())
            
            logger.info("Características físicas creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error general al crear características físicas: {str(e)}")
            logger.error(traceback.format_exc())
            
        return self.players_data
    
    def _create_position_role_features(self):
        """
        Crea características relacionadas con la posición y el rol del jugador
        """
        logger.info("Creando características de posición y rol del jugador")
        
        try:
            # Verificar si tenemos datos de posición
            if 'Pos' not in self.players_data.columns:
                logger.warning("No se encontró columna de posición. No se crearán características de posición/rol.")
                return
            
            # 1. Identificar roles ofensivos basados en estadísticas
            # Shooter: alto % de puntos desde 3P
            if 'pts_prop_from_3p' in self.players_data.columns:
                self.players_data['is_shooter'] = (self.players_data['pts_prop_from_3p'] > 0.4).astype(int)
            elif all(col in self.players_data.columns for col in ['3P', 'PTS']):
                # Alternativa si no tenemos la proporción calculada
                self.players_data['is_shooter'] = ((self.players_data['3P'] * 3 / self.players_data['PTS'].clip(1)) > 0.4).astype(int)
            
            # Playmaker: alto número de asistencias
            if 'AST' in self.players_data.columns:
                self.players_data['is_playmaker'] = (self.players_data['AST'] > 5).astype(int)
            
            # Rebounder: alto número de rebotes
            if 'TRB' in self.players_data.columns:
                self.players_data['is_rebounder'] = (self.players_data['TRB'] > 8).astype(int)
            
            # Scorer: alto número de puntos
            if 'PTS' in self.players_data.columns:
                self.players_data['is_scorer'] = (self.players_data['PTS'] > 20).astype(int)
            
            # Defensive specialist: alto número de robos + tapones
            if all(col in self.players_data.columns for col in ['STL', 'BLK']):
                self.players_data['is_defender'] = ((self.players_data['STL'] + self.players_data['BLK']) > 3).astype(int)
            
            # 2. Identificar jugadores de impacto vs complementarios
            # Jugador de impacto: usa muchas posesiones (FGA + FTA * 0.44 + TOV)
            if all(col in self.players_data.columns for col in ['FGA', 'FTA']):
                # Cálculo básico de posesiones usadas
                poss_used = self.players_data['FGA'] + 0.44 * self.players_data['FTA']
                
                if 'TOV' in self.players_data.columns:
                    poss_used += self.players_data['TOV']
                
                # Categorizar uso de posesiones
                self.players_data['is_high_usage'] = (poss_used > 20).astype(int)
                self.players_data['is_low_usage'] = (poss_used < 10).astype(int)
            
            # 3. Características específicas por posición combinadas con estadísticas
            if 'Pos' in self.players_data.columns:
                # Para cada posición principal, crear características relevantes según el rol
                
                # Guards (G, G-F): Énfasis en asistencias, triples y eficiencia
                if all(col in self.players_data.columns for col in ['AST', 'TOV']):
                    # Ratio asistencias/pérdidas - importante para bases
                    self.players_data['ast_to_tov_ratio'] = (self.players_data['AST'] / self.players_data['TOV'].clip(1)).clip(0, 10)
                    
                    # Categorías por posición específica
                    mask_pure_guard = self.players_data['Pos'] == 'G'
                    mask_combo_guard = self.players_data['Pos'] == 'G-F'
                    
                    # Bases puros: mayor expectativa de asistencias
                    if mask_pure_guard.sum() > 0:
                        self.players_data.loc[mask_pure_guard, 'pure_guard_playmaking'] = (
                            (self.players_data.loc[mask_pure_guard, 'ast_to_tov_ratio'] > 3.5).astype(int)
                        )
                    
                    # Combo guards: balance entre anotación y asistencias
                    if mask_combo_guard.sum() > 0 and 'PTS' in self.players_data.columns:
                        self.players_data.loc[mask_combo_guard, 'combo_guard_efficiency'] = (
                            ((self.players_data.loc[mask_combo_guard, 'PTS'] > 15) & 
                            (self.players_data.loc[mask_combo_guard, 'AST'] > 3)).astype(int)
                        )
                
                # Forwards (F, F-G, F-C): Versatilidad, anotación, eficiencia
                if all(col in self.players_data.columns for col in ['PTS', 'FGA']):
                    # Eficiencia de anotación
                    self.players_data['scoring_efficiency'] = (self.players_data['PTS'] / self.players_data['FGA'].clip(1)).clip(0, 3)
                    
                    # Por cada tipo de forward
                    mask_pure_forward = self.players_data['Pos'] == 'F'
                    mask_scoring_forward = self.players_data['Pos'] == 'F-G'
                    mask_big_forward = self.players_data['Pos'] == 'F-C'
                    
                    # Forward puro: versatilidad
                    if mask_pure_forward.sum() > 0 and all(col in self.players_data.columns for col in ['TRB', 'AST']):
                        self.players_data.loc[mask_pure_forward, 'forward_versatility'] = (
                            ((self.players_data.loc[mask_pure_forward, 'PTS'] > 12) & 
                            (self.players_data.loc[mask_pure_forward, 'TRB'] > 5) &
                            (self.players_data.loc[mask_pure_forward, 'AST'] > 2)).astype(int)
                        )
                    
                    # Forward anotador (F-G): especialista en anotación
                    if mask_scoring_forward.sum() > 0 and '3P' in self.players_data.columns:
                        self.players_data.loc[mask_scoring_forward, 'wing_scorer_rating'] = (
                            ((self.players_data.loc[mask_scoring_forward, 'PTS'] > 18) & 
                            (self.players_data.loc[mask_scoring_forward, '3P'] > 1.5)).astype(int)
                        )
                    
                    # Ala-pívot (F-C): interior con capacidad de rebote
                    if mask_big_forward.sum() > 0 and 'TRB' in self.players_data.columns:
                        self.players_data.loc[mask_big_forward, 'big_forward_rating'] = (
                            ((self.players_data.loc[mask_big_forward, 'PTS'] > 12) & 
                            (self.players_data.loc[mask_big_forward, 'TRB'] > 7)).astype(int)
                        )
                
                # Centers (C, C-F): Rebotes, tapones, eficiencia interior
                if all(col in self.players_data.columns for col in ['BLK', 'TRB']):
                    mask_pure_center = self.players_data['Pos'] == 'C'
                    mask_stretch_center = self.players_data['Pos'] == 'C-F'
                    
                    # Pivots puros: protección de aro y rebote
                    if mask_pure_center.sum() > 0:
                        self.players_data.loc[mask_pure_center, 'classic_center_rating'] = (
                            ((self.players_data.loc[mask_pure_center, 'BLK'] > 1.5) & 
                            (self.players_data.loc[mask_pure_center, 'TRB'] > 9)).astype(int)
                        )
                    
                    # Pivots más versátiles: balance entre interior y exterior
                    if mask_stretch_center.sum() > 0 and '3P' in self.players_data.columns:
                        self.players_data.loc[mask_stretch_center, 'stretch_center_rating'] = (
                            ((self.players_data.loc[mask_stretch_center, 'TRB'] > 7) & 
                            (self.players_data.loc[mask_stretch_center, '3P'] > 0.5)).astype(int)
                        )
            
            logger.info("Creación de características de posición y rol completada")
            
        except Exception as e:
            logger.error(f"Error al crear características de posición y rol: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
    
    def _create_trb_prediction_features(self):
        """
        Crea características específicas para la predicción de rebotes totales (TRB) con mejoras en validación y tasas móviles.
        """
        logger.info("Creando características específicas para predicción de rebotes")
        
        try:
            # Verificar columnas necesarias
            if 'TRB' not in self.players_data.columns or 'MP' not in self.players_data.columns:
                logger.warning("Faltan columnas necesarias para el análisis de rebotes (TRB, MP)")
                return self.players_data
            
            # Validar y rellenar NaN
            self.players_data['TRB'] = pd.to_numeric(self.players_data['TRB'], errors='coerce').fillna(0)
            self.players_data['MP'] = pd.to_numeric(self.players_data['MP'], errors='coerce').fillna(0)
            if 'Pos' in self.players_data.columns:
                self.players_data['Pos'] = self.players_data['Pos'].fillna('F')
            if 'is_home' in self.players_data.columns:
                self.players_data['is_home'] = self.players_data['is_home'].fillna(0)
            if 'is_win' in self.players_data.columns:
                self.players_data['is_win'] = self.players_data['is_win'].fillna(0)
            if 'Opp' in self.players_data.columns:
                self.players_data['Opp'] = self.players_data['Opp'].fillna('UNK')
            
            # Ajustar MP para evitar división por 0
            mp_default = self.players_data.groupby('Player')['MP'].transform('mean').clip(1)
            mp_adjusted = self.players_data['MP'].where(self.players_data['MP'] > 0, mp_default)
            
            # 1. Características básicas de rebotes
            self.players_data['trb_per_minute'] = (self.players_data['TRB'] / mp_adjusted).clip(0, 2)
            self.players_data['trb_rate_5'] = self.players_data.groupby('Player')['TRB'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            ).clip(0, 20).fillna(0)
            
            if 'ORB' in self.players_data.columns and 'DRB' in self.players_data.columns:
                self.players_data['ORB'] = pd.to_numeric(self.players_data['ORB'], errors='coerce').fillna(0)
                self.players_data['DRB'] = pd.to_numeric(self.players_data['DRB'], errors='coerce').fillna(0)
                self.players_data['orb_per_minute'] = (self.players_data['ORB'] / mp_adjusted).clip(0, 1)
                self.players_data['drb_per_minute'] = (self.players_data['DRB'] / mp_adjusted).clip(0, 1.5)
                self.players_data['orb_drb_ratio'] = (self.players_data['ORB'] / self.players_data['DRB'].clip(1)).clip(0, 5)
                self.players_data['orb_rate_5'] = self.players_data.groupby('Player')['ORB'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                ).clip(0, 10).fillna(0)
                self.players_data['drb_rate_5'] = self.players_data.groupby('Player')['DRB'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                ).clip(0, 15).fillna(0)
            else:
                logger.warning("No se encontraron columnas ORB o DRB para características adicionales")
            
            # 2. Características por posición
            if 'Pos' in self.players_data.columns:
                pos_mapping = {
                    'PG': 'G', 'SG': 'G', 'G': 'G', 'G-F': 'G-F',
                    'SF': 'F', 'PF': 'F', 'F': 'F', 'F-G': 'F-G', 'F-C': 'F-C',
                    'C': 'C', 'C-F': 'C-F'
                }
                self.players_data['mapped_pos'] = self.players_data['Pos'].map(pos_mapping).fillna('F')
                
                position_avg_trb = {}
                for pos in ['G', 'G-F', 'F', 'F-G', 'F-C', 'C', 'C-F']:
                    pos_mask = self.players_data['mapped_pos'] == pos
                    if pos_mask.sum() >= 2:  # Reducir umbral
                        position_avg_trb[pos] = self.players_data.loc[pos_mask, 'TRB'].mean()
                
                self.players_data['trb_vs_position_avg'] = self.players_data.apply(
                    lambda row: row['TRB'] / position_avg_trb.get(row['mapped_pos'], 5.0)
                    if row['mapped_pos'] in position_avg_trb and position_avg_trb[row['mapped_pos']] > 0 else 1.0,
                    axis=1
                ).clip(0, 3)
            
            # 3. Características físicas
            if 'Height_Inches' in self.players_data.columns:
                self.players_data['Height_Inches'] = pd.to_numeric(self.players_data['Height_Inches'], errors='coerce').fillna(78)
                self.players_data['trb_per_height'] = (self.players_data['TRB'] / self.players_data['Height_Inches'].clip(60)).clip(0, 0.5)
            else:
                logger.warning("No se encontró columna Height_Inches para características físicas")
            
            if 'Weight' in self.players_data.columns:
                self.players_data['Weight'] = pd.to_numeric(self.players_data['Weight'], errors='coerce').fillna(200)
                self.players_data['trb_per_weight'] = (self.players_data['TRB'] * 10 / self.players_data['Weight'].clip(150)).clip(0, 1)
            else:
                logger.warning("No se encontró columna Weight para características físicas")
            
            # 4. Factores contextuales
            if 'is_home' in self.players_data.columns:
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    if player_mask.sum() < 5:  
                        continue
                    home_mask = self.players_data['is_home'] == 1
                    away_mask = self.players_data['is_home'] == 0
                    if (player_mask & home_mask).sum() > 0 and (player_mask & away_mask).sum() > 0:
                        home_avg = self.players_data.loc[player_mask & home_mask, 'TRB'].mean()
                        away_avg = self.players_data.loc[player_mask & away_mask, 'TRB'].mean()
                        self.players_data.loc[player_mask, 'trb_home_away_diff'] = (home_avg - away_avg).clip(-5, 5)
            
            if 'is_win' in self.players_data.columns:
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    if player_mask.sum() < 5:
                        continue
                    win_mask = self.players_data['is_win'] == 1
                    loss_mask = self.players_data['is_win'] == 0
                    if (player_mask & win_mask).sum() > 0 and (player_mask & loss_mask).sum() > 0:
                        win_avg = self.players_data.loc[player_mask & win_mask, 'TRB'].mean()
                        loss_avg = self.players_data.loc[player_mask & loss_mask, 'TRB'].mean()
                        self.players_data.loc[player_mask, 'trb_win_loss_diff'] = (win_avg - loss_avg).clip(-5, 5)
            
            # 5. Rebotes contra oponentes específicos
            if 'Opp' in self.players_data.columns:
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data[player_mask].copy()
                    if len(player_data) < 5:
                        continue
                    player_avg_trb = player_data['TRB'].mean()
                    for opp in player_data['Opp'].unique():
                        opp_mask = player_data['Opp'] == opp
                        if opp_mask.sum() >= 1:  # Reducir umbral
                            opp_avg_trb = player_data.loc[opp_mask, 'TRB'].mean()
                            diff = (opp_avg_trb - player_avg_trb).clip(-5, 5)
                            self.players_data.loc[player_mask & (self.players_data['Opp'] == opp), 'trb_vs_opp_diff'] = diff
            
            # Rellenar valores faltantes
            for col in ['trb_home_away_diff', 'trb_win_loss_diff', 'trb_vs_opp_diff']:
                if col in self.players_data.columns:
                    self.players_data[col] = self.players_data[col].fillna(0)
            
            logger.info("Características de rebotes creadas correctamente")
        
        except Exception as e:
            logger.error(f"Error al crear características para predicción de rebotes: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.players_data
    
    def _create_ast_prediction_features(self):
        """
        Crea características específicas para la predicción de asistencias (AST)
        """
        logger.info("Creando características específicas para predicción de asistencias")
        
        try:
            
            # Verificar columnas necesarias
            if 'AST' not in self.players_data.columns or 'MP' not in self.players_data.columns:
                logger.warning("Faltan columnas necesarias para el análisis de asistencias")
                return
            
            # Ajustar MP para evitar división por 0
            mp_default = self.players_data.groupby('Player')['MP'].transform('mean').clip(1)
            mp_adjusted = self.players_data['MP'].where(self.players_data['MP'] > 0, mp_default)

            # Asistencias por minuto (normaliza por tiempo de juego)
            self.players_data['ast_per_minute'] = (self.players_data['AST'] / self.players_data['MP'].clip(1)).clip(0, 1)
            
            # Ratio asistencias/pérdidas si TOV está disponible
            if 'TOV' in self.players_data.columns:
                self.players_data['ast_to_tov_ratio'] = (self.players_data['AST'] / self.players_data['TOV'].clip(1)).clip(0, 10)
            
            # 2. Características por posición
            if 'Pos' in self.players_data.columns:
                pos_mapping = {
                    'PG': 'G', 'SG': 'G', 'G': 'G', 'G-F': 'G-F',
                    'SF': 'F', 'PF': 'F', 'F': 'F', 'F-G': 'F-G', 'F-C': 'F-C',
                    'C': 'C', 'C-F': 'C-F'
                }
                self.players_data['mapped_pos'] = self.players_data['Pos'].map(pos_mapping).fillna('F')
                
                position_avg_ast = {}
                for pos in ['G', 'G-F', 'F', 'F-G', 'F-C', 'C', 'C-F']:
                    pos_mask = self.players_data['mapped_pos'] == pos
                    if pos_mask.sum() >= 5:  
                        position_avg_ast[pos] = self.players_data.loc[pos_mask, 'AST'].mean()
                
                self.players_data['ast_vs_position_avg'] = self.players_data.apply(
                    lambda row: row['AST'] / position_avg_ast.get(row['mapped_pos'], 3.0)
                    if row['mapped_pos'] in position_avg_ast and position_avg_ast[row['mapped_pos']] > 0 else 1.0,
                    axis=1
                ).clip(0, 5)
                
                if 'is_guard' in self.players_data.columns:
                    self.players_data['is_guard'] = pd.to_numeric(self.players_data['is_guard'], errors='coerce').fillna(0)
                    guard_avg_ast = self.players_data[self.players_data['is_guard'] == 1]['AST'].mean()
                    guard_avg_ast = max(guard_avg_ast, 3.0)  
                    self.players_data['playmaking_rating'] = ((self.players_data['AST'] / mp_adjusted * 36) / guard_avg_ast).clip(0, 3)
                else:
                    logger.warning("No se encontró columna is_guard para calcular playmaking_rating")
            
            # 3. Factores contextuales
            # Diferencia entre casa/visitante
            if 'is_home' in self.players_data.columns:
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    if player_mask.sum() < 5:  # Solo si hay suficientes datos
                        continue
                        
                    home_mask = self.players_data['is_home'] == True
                    away_mask = self.players_data['is_home'] == False
                    
                    if (player_mask & home_mask).sum() > 0 and (player_mask & away_mask).sum() > 0:
                        home_avg = self.players_data.loc[player_mask & home_mask, 'AST'].mean()
                        away_avg = self.players_data.loc[player_mask & away_mask, 'AST'].mean()
                        
                        # Diferencia entre casa y fuera
                        self.players_data.loc[player_mask, 'ast_home_away_diff'] = home_avg - away_avg
            
            # Diferencia entre victorias/derrotas
            if 'is_win' in self.players_data.columns:
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    if player_mask.sum() < 5:  # Solo si hay suficientes datos
                        continue
                        
                    win_mask = self.players_data['is_win'] == True
                    loss_mask = self.players_data['is_win'] == False
                    
                    if (player_mask & win_mask).sum() > 0 and (player_mask & loss_mask).sum() > 0:
                        win_avg = self.players_data.loc[player_mask & win_mask, 'AST'].mean()
                        loss_avg = self.players_data.loc[player_mask & loss_mask, 'AST'].mean()
                        
                        # Diferencia entre victorias y derrotas
                        self.players_data.loc[player_mask, 'ast_win_loss_diff'] = win_avg - loss_avg
            
            # 4. Asistencias contra oponentes específicos
            for player in self.players_data['Player'].unique():
                player_mask = self.players_data['Player'] == player
                player_data = self.players_data[player_mask].copy()
                
                if len(player_data) < 5:  # Omitir jugadores con pocos datos
                    continue
                    
                # Media general del jugador
                player_avg_ast = player_data['AST'].mean()
                
                # Para cada oponente
                for opp in player_data['Opp'].unique():
                    opp_mask = player_data['Opp'] == opp
                    if opp_mask.sum() >= 2:  # Al menos 2 partidos contra este oponente
                        opp_avg_ast = player_data.loc[opp_mask, 'AST'].mean()
                        
                        # Diferencia vs promedio
                        diff = opp_avg_ast - player_avg_ast
                        self.players_data.loc[player_mask & (self.players_data['Opp'] == opp), 'ast_vs_opp_diff'] = diff
            
            # 5. Correlación entre asistencias y otros stats
            if all(col in self.players_data.columns for col in ['PTS', 'TRB']):
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data[player_mask].copy()
                    
                    if len(player_data) < 5:
                        continue
                    
                    # Correlación entre asistencias y puntos
                    corr_pts_ast = player_data[['PTS', 'AST']].corr().iloc[0, 1]
                    self.players_data.loc[player_mask, 'pts_ast_correlation'] = corr_pts_ast
                    
                    # Correlación entre asistencias y rebotes
                    corr_trb_ast = player_data[['TRB', 'AST']].corr().iloc[0, 1]
                    self.players_data.loc[player_mask, 'trb_ast_correlation'] = corr_trb_ast
            
            # Rellenar valores faltantes
            for col in ['ast_home_away_diff', 'ast_win_loss_diff', 'ast_vs_opp_diff', 
                       'pts_ast_correlation', 'trb_ast_correlation']:
                if col in self.players_data.columns:
                    self.players_data[col] = self.players_data[col].fillna(0)
            
            logger.info("Características de asistencias creadas correctamente")
        except Exception as e:
            logger.error(f"Error al crear características para predicción de asistencias: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.players_data
    
    def _create_3p_prediction_features(self):
        """
        Crea características específicas para predecir los triples (3P) del jugador
        """
        logger.info("Creando características específicas para predicción de triples")
        
        try:
            # Verificar que tenemos la columna 3P
            if '3P' not in self.players_data.columns:
                logger.error("Columna 3P no encontrada, abortando creación de características")
                return self.players_data
                
            # 1. Características de scoring por cuartos (si están disponibles)
            quarter_cols = [col for col in self.players_data.columns if '3P_Q' in col]
            if quarter_cols:
                # Calcular tendencias de puntuación por cuarto
                for col in quarter_cols:
                    try:
                        quarter = col.split('_')[1]  # Extraer el número de cuarto
                        # Calcular proporción de puntos en este cuarto
                        self.players_data[f'3p_prop_{quarter}'] = (self.players_data[col] / self.players_data['3P'].clip(1)).clip(0, 1)
                    except Exception as e:
                        logger.error(f"Error al crear características de cuartos: {str(e)}")
                    
            # 2. Efectividad de tiro desglosada
            if all(col in self.players_data.columns for col in ['FG', 'FGA', '3P', '3PA', 'FT', 'FTA']):
                # Puntos por intento de tiro
                try:
                    # Puntos por tiro de campo
                    self.players_data['3p_per_fga'] = (self.players_data['3P'] / self.players_data['FGA'].clip(1)).clip(0, 4)
                    
                    # Distribución de puntos por tipo de tiro
                    self.players_data['3p_from_3p'] = self.players_data['3P'] * 3
                    self.players_data['3p_from_2p'] = self.players_data['FG'] * 2 - self.players_data['3p_from_3p']
                    self.players_data['3p_from_ft'] = self.players_data['FT']
                    
                    # Proporciones de puntos por tipo
                    total_3p = self.players_data['3P'].clip(1)
                    self.players_data['3p_prop_from_3p'] = (self.players_data['3p_from_3p'] / total_3p).clip(0, 1)
                    self.players_data['3p_prop_from_2p'] = (self.players_data['3p_from_2p'] / total_3p).clip(0, 1)
                    self.players_data['3p_prop_from_ft'] = (self.players_data['3p_from_ft'] / total_3p).clip(0, 1)
                    
                    # Eficiencia de tiro
                    self.players_data['3p_per_scoring_poss'] = (self.players_data['3P'] / 
                                                            (self.players_data['FG'] + (self.players_data['FTA'] * 0.44)).clip(1)).clip(0, 3)
                except Exception as e:
                    logger.error(f"Error al crear características de efectividad de tiro: {str(e)}")
            
            # 3. Características de rendimiento por tipo de defensa (si existe info del oponente)
            if 'Opp' in self.players_data.columns:
                # Calcular estadísticas de puntos contra cada oponente
                try:
                    # Para cada jugador, calcular promedio contra cada oponente
                    for player in tqdm(self.players_data['Player'].unique(), desc="Calculando rendimiento vs. oponentes"):
                        player_mask = self.players_data['Player'] == player
                        for opp in self.players_data.loc[player_mask, 'Opp'].unique():
                            # Filtrar partidos de este jugador contra este oponente
                            mask = (self.players_data['Player'] == player) & (self.players_data['Opp'] == opp)
                            
                            if mask.sum() >= 2:  # Solo si hay suficientes partidos
                                pts_vs_opp = self.players_data.loc[mask, 'PTS'].mean()
                                # Asignar a todas las filas de este jugador contra este oponente
                                self.players_data.loc[mask, 'pts_avg_vs_opp'] = pts_vs_opp
                                
                                # Calcular diferencia respecto al promedio general del jugador
                                avg_pts = self.players_data.loc[player_mask, 'PTS'].mean()
                                self.players_data.loc[mask, 'pts_diff_vs_opp'] = pts_vs_opp - avg_pts
                except Exception as e:
                    logger.error(f"Error al crear características de rendimiento contra oponentes: {str(e)}")
            
            # 4. Características de momentum y rachas de anotación
            for window in self.window_sizes:
                try:
                    # Momentum ofensivo (tendencia de puntuación)
                    if f'3P_mean_{window}' in self.players_data.columns:
                        pts_mean_col = f'3P_mean_{window}'
                        
                        # Calcular una ventana más grande para comparar
                        larger_window = next((w for w in self.window_sizes if w > window), window*2)
                        pts_larger_mean_col = f'3P_mean_{larger_window}'
                        
                        # Si existe la ventana más grande, calcular momentum
                        if pts_larger_mean_col in self.players_data.columns:
                            self.players_data[f'3p_momentum_{window}'] = (
                                self.players_data[pts_mean_col] - self.players_data[pts_larger_mean_col]
                            ).clip(-20, 20)
                except Exception as e:
                    logger.error(f"Error al crear características de momentum para ventana {window}: {str(e)}")
            
            # 5. Características de rachas de puntuación
            try:
                # Para cada jugador, calcular rachas de puntuación
                for player in tqdm(self.players_data['Player'].unique(), desc="Calculando rachas de puntuación"):
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data.loc[player_mask].sort_values('Date').copy()
                    
                    if len(player_data) >= 3:  # Solo con suficientes partidos
                        # Identificar partidos con más/menos puntos que el promedio
                        player_avg = player_data['PTS'].mean()
                        above_avg = player_data['PTS'] > player_avg
                        
                        # Rachas de puntuación por encima del promedio
                        streaks = []
                        current_streak = 0
                        
                        for above in above_avg:
                            if above:
                                current_streak += 1
                            else:
                                current_streak = 0
                            streaks.append(current_streak)
                        
                        self.players_data.loc[player_data.index, '3p_above_avg_streak'] = streaks
                        
                        # Identificar partidos con aumento de puntuación consecutivo
                        pts_increase = player_data['PTS'].diff() > 0
                        
                        # Rachas de aumentos consecutivos
                        increase_streaks = []
                        current_inc_streak = 0
                        
                        for inc in pts_increase:
                            if inc:
                                current_inc_streak += 1
                            else:
                                current_inc_streak = 0
                            increase_streaks.append(current_inc_streak)
                        
                        self.players_data.loc[player_data.index, '3p_increase_streak'] = increase_streaks
            except Exception as e:
                logger.error(f"Error al crear características de rachas de puntuación: {str(e)}")
            
            # 6. Características basadas en el oponente
            if 'Opp' in self.players_data.columns:
                try:
                    # Calcular promedio de puntos concedidos por cada oponente
                    opp_pts_allowed = self.players_data.groupby('Opp')['PTS'].mean().to_dict()
                    
                    # Asignar a cada fila
                    self.players_data['opp_pts_allowed_avg'] = self.players_data['Opp'].map(
                        lambda x: opp_pts_allowed.get(x, self.players_data['PTS'].mean())
                    )
                    
                    # Calcular diferencia respecto al promedio de la liga
                    league_avg = self.players_data['PTS'].mean()
                    self.players_data['opp_pts_allowed_diff'] = self.players_data['opp_pts_allowed_avg'] - league_avg
                except Exception as e:
                    logger.error(f"Error al crear características basadas en oponente: {str(e)}")
            
            # 7. Relación entre minutos jugados y puntos
            if 'MP' in self.players_data.columns:
                try:
                    # Puntos por minuto
                    self.players_data['3p_per_minute'] = (self.players_data['3P'] / self.players_data['MP'].clip(1)).clip(0, 2)
                    
                    # Eficiencia de puntuación ajustada por minutos
                    for window in self.window_sizes:
                        if f'MP_mean_{window}' in self.players_data.columns and f'3P_mean_{window}' in self.players_data.columns:
                            # Calcular puntos por minuto en la ventana
                            self.players_data[f'3p_per_min_{window}'] = (
                                self.players_data[f'3P_mean_{window}'] / 
                                self.players_data[f'MP_mean_{window}'].clip(1)
                            ).clip(0, 2)
                            
                            # Predecir puntos basado en minutos esperados
                            self.players_data[f'expected_3p_{window}'] = (
                                self.players_data[f'3p_per_min_{window}'] * 
                                self.players_data['MP']
                            ).clip(0, 60)
                            
                            # Diferencia entre puntos reales y esperados
                            self.players_data[f'3p_vs_expected_{window}'] = (
                                self.players_data['3P'] - 
                                self.players_data[f'expected_3p_{window}']
                            ).clip(-30, 30)
                except Exception as e:
                    logger.error(f"Error al crear características de puntos por minuto: {str(e)}")
            
            # 8. Características basadas en casa/fuera
            if 'is_home' in self.players_data.columns:
                try:
                    # Calcular promedios de puntos en casa vs. fuera por jugador
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        
                        # Promedio en casa
                        home_mask = player_mask & (self.players_data['is_home'] == 1)
                        if home_mask.sum() > 0:
                            home_avg = self.players_data.loc[home_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, '3p_home_avg'] = home_avg
                        
                        # Promedio fuera
                        away_mask = player_mask & (self.players_data['is_home'] == 0)
                        if away_mask.sum() > 0:
                            away_avg = self.players_data.loc[away_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, '3p_away_avg'] = away_avg
                        
                        # Diferencia casa-fuera
                        if home_mask.sum() > 0 and away_mask.sum() > 0:
                            self.players_data.loc[player_mask, '3p_home_away_diff'] = home_avg - away_avg
                except Exception as e:
                    logger.error(f"Error al crear características de puntos casa/fuera: {str(e)}")
            
            # 9. Características de puntuación en función del resultado
            if 'is_win' in self.players_data.columns:
                try:
                    # Calcular promedios de puntos en victorias vs. derrotas por jugador
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        
                        # Promedio en victorias
                        win_mask = player_mask & (self.players_data['is_win'] == 1)
                        if win_mask.sum() > 0:
                            win_avg = self.players_data.loc[win_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, '3p_win_avg'] = win_avg
                        
                        # Promedio en derrotas
                        loss_mask = player_mask & (self.players_data['is_win'] == 0)
                        if loss_mask.sum() > 0:
                            loss_avg = self.players_data.loc[loss_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask, '3p_loss_avg'] = loss_avg
                        
                        # Diferencia victorias-derrotas
                        if win_mask.sum() > 0 and loss_mask.sum() > 0:
                            self.players_data.loc[player_mask, '3p_win_loss_diff'] = win_avg - loss_avg
                except Exception as e:
                    logger.error(f"Error al crear características de puntos por resultado: {str(e)}")
            
            # Rellenar valores nulos en las nuevas columnas
            pts_cols = [col for col in self.players_data.columns if col.startswith('3p_')]
            for col in pts_cols:
                if self.players_data[col].isnull().any():
                    self.players_data[col] = self.players_data[col].fillna(self.players_data[col].mean())
            
            logger.info("Características específicas para predicción de triples creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error general al crear características de predicción de triples: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            
        return self.players_data
    
    def _create_matchup_features(self):
        """
        Crea características basadas en enfrentamientos entre jugadores y equipos
        """
        logger.info("Creando características de matchup para jugadores")
        
        try:
            # Verificar que tenemos las columnas necesarias
            if not all(col in self.players_data.columns for col in ['Player', 'Team', 'Opp']):
                logger.warning("Faltan columnas básicas para análisis de matchups")
                return self.players_data
            
            # Diccionario con equipos por conferencia
            eastern_conference = [
                'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DET', 'IND', 
                'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS'
            ]
            
            western_conference = [
                'DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN',
                'NOP', 'OKC', 'PHO', 'POR', 'SAC', 'SAS', 'UTA'
            ]
            
            # Crear diccionario para mapear equipos a conferencias
            team_to_conference = {}
            for team in eastern_conference:
                team_to_conference[team] = 'East'
            for team in western_conference:
                team_to_conference[team] = 'West'
            
            # Agregar columna de conferencia si no existe
            if 'conference' not in self.players_data.columns:
                # Mapear equipos a conferencias
                self.players_data['conference'] = self.players_data['Team'].map(team_to_conference)
                self.players_data['opp_conference'] = self.players_data['Opp'].map(team_to_conference)
                logger.info("Columna de conferencia creada para equipos y oponentes")
            
            # 1. Rendimiento contra equipos específicos
            player_ids = self.players_data['Player'].unique()
            
            # Procesamiento por jugador
            for player in tqdm(player_ids, desc="Analizando matchups por jugador"):
                player_mask = self.players_data['Player'] == player
                player_data = self.players_data[player_mask].copy()
                
                if len(player_data) < 5:  # Skip si hay pocos datos
                    continue
                
                # Obtener estadísticas promedio del jugador
                player_avg_pts = player_data['PTS'].mean() if 'PTS' in player_data.columns else 0
                player_avg_trb = player_data['TRB'].mean() if 'TRB' in player_data.columns else 0
                player_avg_ast = player_data['AST'].mean() if 'AST' in player_data.columns else 0
                player_avg_3p = player_data['3P'].mean() if '3P' in player_data.columns else 0
                
                # Para cada oponente contra el que ha jugado
                for opp in player_data['Opp'].unique():
                    opp_mask = player_data['Opp'] == opp
                    
                    # Solo si hay suficientes partidos
                    if opp_mask.sum() >= 2:
                        # Calcular estadísticas contra este oponente
                        if 'PTS' in player_data.columns:
                            opp_avg_pts = player_data.loc[opp_mask, 'PTS'].mean()
                            self.players_data.loc[player_mask & (self.players_data['Opp'] == opp), 'matchup_pts_diff'] = opp_avg_pts - player_avg_pts
                        
                        if 'TRB' in player_data.columns:
                            opp_avg_trb = player_data.loc[opp_mask, 'TRB'].mean()
                            self.players_data.loc[player_mask & (self.players_data['Opp'] == opp), 'matchup_trb_diff'] = opp_avg_trb - player_avg_trb
                        
                        if 'AST' in player_data.columns:
                            opp_avg_ast = player_data.loc[opp_mask, 'AST'].mean()
                            self.players_data.loc[player_mask & (self.players_data['Opp'] == opp), 'matchup_ast_diff'] = opp_avg_ast - player_avg_ast
                        
                        if '3P' in player_data.columns:
                            opp_avg_3p = player_data.loc[opp_mask, '3P'].mean()
                            self.players_data.loc[player_mask & (self.players_data['Opp'] == opp), 'matchup_3p_diff'] = opp_avg_3p - player_avg_3p
            
            # 2. Rendimiento por conferencia del oponente (Este vs Oeste)
            # Usamos la columna de conferencia que acabamos de crear
            if all(col in self.players_data.columns for col in ['conference', 'opp_conference']):
                # Procesar jugador por jugador
                for player in player_ids:
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data[player_mask].copy()
                    
                    if len(player_data) < 5:
                        continue
                    
                    # Separar por conferencia del oponente
                    east_mask = player_data['opp_conference'] == 'East'
                    west_mask = player_data['opp_conference'] == 'West'
                    
                    # Si hay suficientes partidos contra ambas conferencias
                    if east_mask.sum() >= 3 and west_mask.sum() >= 3:
                        # Calcular diferencias para estadísticas principales
                        for stat in ['PTS', 'TRB', 'AST', '3P']:
                            if stat in player_data.columns:
                                east_avg = player_data.loc[east_mask, stat].mean()
                                west_avg = player_data.loc[west_mask, stat].mean()
                                
                                # Asignar la diferencia Este-Oeste
                                self.players_data.loc[player_mask, f'{stat.lower()}_east_west_diff'] = east_avg - west_avg
            
            # 3. Análisis de matchups favorables/desfavorables
            # Basado en desviación respecto a la media
            for player in player_ids:
                player_mask = self.players_data['Player'] == player
                
                # Usar las columnas de diferencia calculadas anteriormente
                for stat in ['pts', 'trb', 'ast', '3p']:
                    diff_col = f'matchup_{stat}_diff'
                    
                    if diff_col in self.players_data.columns:
                        # Obtener diferencias disponibles para este jugador
                        diffs = self.players_data.loc[player_mask, diff_col].dropna()
                        
                        if len(diffs) >= 5:  # Solo si hay suficientes datos
                            # Calcular umbral para matchups favorables/desfavorables
                            threshold = diffs.std()
                            
                            # Identificar matchups favorables/desfavorables
                            self.players_data.loc[player_mask, f'{stat}_favorable_matchup'] = (
                                self.players_data.loc[player_mask, diff_col] > threshold
                            ).astype(int)
                            
                            self.players_data.loc[player_mask, f'{stat}_unfavorable_matchup'] = (
                                self.players_data.loc[player_mask, diff_col] < -threshold
                            ).astype(int)
            
            # 4. Análisis de defensa del oponente por posición
            if 'Pos' in self.players_data.columns:
                # Calcular efectividad defensiva de cada equipo contra cada posición
                positions = ['G', 'G-F', 'F', 'F-G', 'F-C', 'C', 'C-F']
                
                for pos in positions:
                    # Filtrar jugadores de esta posición
                    pos_mask = self.players_data['Pos'] == pos
                    
                    if pos_mask.sum() > 0:
                        # Calcular promedio de puntos por posición
                        pos_avg_pts = self.players_data.loc[pos_mask, 'PTS'].mean() if 'PTS' in self.players_data.columns else 0
                        
                        # Para cada equipo oponente
                        for opp in self.players_data['Opp'].unique():
                            # Puntos promedio concedidos a esta posición
                            opp_pos_mask = pos_mask & (self.players_data['Opp'] == opp)
                            
                            if opp_pos_mask.sum() >= 5:  # Solo si hay suficientes partidos
                                opp_pos_avg = self.players_data.loc[opp_pos_mask, 'PTS'].mean()
                                
                                # Diferencia respecto a la media de la posición
                                diff = opp_pos_avg - pos_avg_pts
                                
                                # Asignar a todos los jugadores de esta posición contra este oponente
                                self.players_data.loc[pos_mask & (self.players_data['Opp'] == opp), f'opp_def_vs_{pos}'] = diff
            
            # 5. Tendencias recientes en matchups
            if 'Date' in self.players_data.columns:
                # Ordenar por fecha para análisis de tendencias
                self.players_data = self.players_data.sort_values(['Player', 'Date'])
                
                # Para cada jugador y oponente, analizar tendencias recientes
                for player in player_ids:
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data[player_mask].copy()
                    
                    if len(player_data) < 5:
                        continue
                    
                    # Para cada oponente que ha enfrentado más de una vez
                    for opp in player_data['Opp'].value_counts()[player_data['Opp'].value_counts() > 1].index:
                        opp_mask = player_data['Opp'] == opp
                        opp_games = player_data[opp_mask].sort_values('Date')
                        
                        if len(opp_games) >= 2:  # Al menos 2 partidos
                            # Calcular tendencias para estadísticas principales
                            for stat in ['PTS', 'TRB', 'AST', '3P']:
                                if stat in opp_games.columns:
                                    # Calcular diferencia entre partidos más recientes
                                    recent_diff = opp_games[stat].diff().dropna()
                                    
                                    if len(recent_diff) > 0:
                                        # Asignar al partido más reciente
                                        most_recent_idx = opp_games.index[-1]
                                        self.players_data.loc[most_recent_idx, f'{stat.lower()}_matchup_trend'] = recent_diff.iloc[-1]
            
            # 6. Análisis de rendimiento por división (aplicando agrupaciones más específicas)
            divisions = {
                'Atlantic': ['BOS', 'BRK', 'NYK', 'PHI', 'TOR'],
                'Central': ['CHI', 'CLE', 'DET', 'IND', 'MIL'],
                'Southeast': ['ATL', 'CHO', 'MIA', 'ORL', 'WAS'],
                'Northwest': ['DEN', 'MIN', 'OKC', 'POR', 'UTA'],
                'Pacific': ['GSW', 'LAC', 'LAL', 'PHO', 'SAC'],
                'Southwest': ['DAL', 'HOU', 'MEM', 'NOP', 'SAS']
            }
            
            # Crear mapeo de equipo a división
            team_to_division = {}
            for division, teams in divisions.items():
                for team in teams:
                    team_to_division[team] = division
            
            # Agregar columna de división para el oponente
            self.players_data['opp_division'] = self.players_data['Opp'].map(team_to_division)
            
            # Analizar rendimiento por división para cada jugador
            for player in player_ids:
                player_mask = self.players_data['Player'] == player
                player_data = self.players_data[player_mask].copy()
                
                if len(player_data) < 10:  # Necesitamos más datos para este análisis
                    continue
                
                # Para cada división, calcular el rendimiento medio
                division_stats = {}
                
                for division in divisions.keys():
                    div_mask = player_data['opp_division'] == division
                    if div_mask.sum() >= 3:  # Al menos 3 partidos contra esta división
                        for stat in ['PTS', 'TRB', 'AST', '3P']:
                            if stat in player_data.columns:
                                avg_stat = player_data.loc[div_mask, stat].mean()
                                division_stats[(division, stat)] = avg_stat
                
                # Calcular diferencias respecto a la media global del jugador
                for stat in ['PTS', 'TRB', 'AST', '3P']:
                    if stat in player_data.columns:
                        avg_stat = player_data[stat].mean()
                        
                        for division in divisions.keys():
                            if (division, stat) in division_stats:
                                div_avg = division_stats[(division, stat)]
                                diff = div_avg - avg_stat
                                
                                # Asignar a los partidos contra esta división
                                div_mask = (self.players_data['Player'] == player) & (self.players_data['opp_division'] == division)
                                self.players_data.loc[div_mask, f'{stat.lower()}_vs_{division.lower()}_diff'] = diff
            
            # Rellenar valores faltantes en columnas nuevas
            matchup_cols = [col for col in self.players_data.columns if 'matchup' in col]
            for col in matchup_cols:
                if col in self.players_data.columns:
                    self.players_data[col] = self.players_data[col].fillna(0)
            
            logger.info("Características de matchup creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error al crear características de matchup: {str(e)}")
            logger.error(traceback.format_exc())
            
        return self.players_data
    
    def _create_efficiency_features(self):
        """
        Crea características de eficiencia y productividad para jugadores
        """
        logger.info("Creando características de eficiencia y productividad")
        
        try:
            # 1. Eficiencia de tiro
            # True Shooting Percentage (TS%)
            if all(col in self.players_data.columns for col in ['PTS', 'FGA', 'FTA']):
                # Denominador seguro
                ts_denom = 2 * (self.players_data['FGA'] + 0.44 * self.players_data['FTA']).clip(lower=1)
                self.players_data['ts_pct'] = (self.players_data['PTS'] / ts_denom).clip(0, 1).fillna(0.5)
                
            # Effective Field Goal Percentage (eFG%)
            if all(col in self.players_data.columns for col in ['FG', '3P', 'FGA']):
                efg_denom = self.players_data['FGA'].clip(lower=1)
                self.players_data['efg_pct'] = ((self.players_data['FG'] + 0.5 * self.players_data['3P']) / efg_denom).clip(0, 1).fillna(0.45)
            
            # 2. Productividad por minuto
            for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P']:
                if all(col in self.players_data.columns for col in [stat, 'MP']):
                    # Estadística por minuto jugado
                    self.players_data[f'{stat.lower()}_per_minute'] = (
                        self.players_data[stat] / self.players_data['MP'].clip(1)
                    ).clip(0, 2).fillna(0)  # Limitar a valores razonables
            
            # 3. Eficiencia ofensiva general
            if all(col in self.players_data.columns for col in ['PTS', 'FGA', 'TOV']):
                # Puntos por posesión (estimada)
                possessions = self.players_data['FGA'] + self.players_data['TOV'] * 0.44
                self.players_data['pts_per_possession'] = (
                    self.players_data['PTS'] / possessions.clip(1)
                ).clip(0, 2).fillna(1.0)
            
            # 4. Índice de polivalencia
            if all(col in self.players_data.columns for col in ['PTS', 'TRB', 'AST', 'STL', 'BLK']):
                # Crear un índice que valora la versatilidad del jugador
                self.players_data['versatility_index'] = (
                    (self.players_data['PTS'] / 20) +
                    (self.players_data['TRB'] / 10) +
                    (self.players_data['AST'] / 10) +
                    (self.players_data['STL'] / 2) +
                    (self.players_data['BLK'] / 2)
                ).clip(0, 5).fillna(1.0)
            
            # 5. Relación asistencias/pérdidas
            if all(col in self.players_data.columns for col in ['AST', 'TOV']):
                self.players_data['ast_to_tov_ratio'] = (
                    self.players_data['AST'] / self.players_data['TOV'].clip(1)
                ).clip(0, 10).fillna(1.0)
            
            # 6. PIE (Player Impact Estimate) - simplificado
            if all(col in self.players_data.columns for col in ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FGM', 'TOV']):
                # Versión simplificada de PIE
                self.players_data['player_impact'] = (
                    self.players_data['PTS'] +
                    self.players_data['TRB'] +
                    self.players_data['AST'] +
                    self.players_data['STL'] +
                    self.players_data['BLK'] -
                    (self.players_data['FGA'] - self.players_data['FG']) -
                    self.players_data['TOV']
                ).clip(0, 70).fillna(20)
            
            # 7. Índice de impacto defensivo
            if all(col in self.players_data.columns for col in ['STL', 'BLK', 'TRB']):
                self.players_data['defensive_impact'] = (
                    self.players_data['STL'] +
                    self.players_data['BLK'] * 1.5 +
                    self.players_data['TRB'] * 0.5
                ).clip(0, 20).fillna(5)
            
            # 8. Índice Box Plus/Minus simplificado
            if all(col in self.players_data.columns for col in ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']):
                self.players_data['box_plus_minus'] = (
                    self.players_data['PTS'] * 0.8 +
                    self.players_data['TRB'] * 0.3 +
                    self.players_data['AST'] * 0.7 +
                    self.players_data['STL'] * 1.5 +
                    self.players_data['BLK'] * 1.5 -
                    self.players_data['TOV'] * 1.0 -
                    self.players_data['PF'] * 0.5
                ).clip(-10, 20).fillna(0)
            
            # 9. Score ratio (Proporción de la puntuación del equipo)
            if all(col in self.players_data.columns for col in ['PTS', 'Team_Score']):
                self.players_data['scoring_ratio'] = (
                    self.players_data['PTS'] / self.players_data['Team_Score'].clip(50)
                ).clip(0, 0.7).fillna(0.1)
            
            # 10. Índice de consistencia (basado en la desviación estándar de ventanas)
            for window in self.window_sizes:
                pts_std_col = f'PTS_std_{window}'
                if pts_std_col in self.players_data.columns:
                    # Mayor consistencia = menor desviación estándar
                    self.players_data[f'consistency_index_{window}'] = (
                        1 / (self.players_data[pts_std_col].clip(1) * 0.2)
                    ).clip(0, 2).fillna(1.0)
            
            logger.info("Características de eficiencia y productividad creadas correctamente")
        except Exception as e:
            logger.error(f"Error al crear características de eficiencia: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.players_data
    
    def _safe_linregress(self, x, y):
        """
        Realiza una regresión lineal de manera segura, manejando casos problemáticos
        
        Args:
            x (np.array): Variable independiente
            y (np.array): Variable dependiente
            
        Returns:
            float: Pendiente de la regresión, o 0 si hay algún problema
        """
        try:
            # Verificar valores válidos
            if len(x) < 2 or len(y) < 2:
                return 0
                
            # Comprobar NaNs o Infs
            if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
                return 0
                
            # Comprobar varianza suficiente (evita división por cero)
            if np.std(x) < 0.001 or np.std(y) < 0.001:
                return 0
                
            # Calcular regresión
            slope, _, _, _, _ = stats.linregress(x, y)
            
            # Verificar resultado válido
            if np.isnan(slope) or np.isinf(slope):
                return 0
                
            # Limitar valores extremos
            return np.clip(slope, -100, 100)
            
        except Exception as e:
            logger.debug(f"Error en regresión lineal: {str(e)}")
            return 0
            
    def filter_correlated_features(self, X, threshold=None):
        """
        Filtra características altamente correlacionadas
        
        Args:
            X (pd.DataFrame): DataFrame con características
            threshold (float): Umbral de correlación (si None, usa self.correlation_threshold)
            
        Returns:
            pd.DataFrame: DataFrame con características filtradas
        """
        if not self.enable_correlation_analysis:
            return X
            
        if threshold is None:
            threshold = self.correlation_threshold
            
        try:
            logger.info(f"Filtrando características con correlación > {threshold}")
            
            # Asegurar que X tiene índice consecutivo
            X = X.reset_index(drop=True)
            
            # Si X está vacío o tiene solo una columna, devolverlo tal cual
            if X.empty or len(X.columns) <= 1:
                return X
            
            # Definir columnas esenciales que no deben eliminarse
            essential_cols = [
                'Player', 'Date', 'Team', 'Away', 'Opp', 'Result', 'GS', 'MP',
                'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
                'FT', 'FTA', 'FT%', 'TS%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', '+/-', 'Pos', 'is_win',
                'team_score', 'opp_score', 'total_score', 'point_diff', 'has_overtime',
                'overtime_periods', 'is_home', 'Height_Inches', 'Weight', 'BMI', 'is_started'
            ]
            present_essential_cols = [col for col in essential_cols if col in X.columns]
            
            # Columnas que se pueden considerar para eliminación (excluir las esenciales)
            non_essential_cols = [col for col in X.columns if col not in present_essential_cols]
            
            # Si todas las columnas son esenciales, devolver el DataFrame original
            if not non_essential_cols:
                logger.info("No hay columnas no esenciales para filtrar")
                return X
            
            # Calcular la matriz de correlación de manera segura
            try:
                # Primero verificar que no hay valores problemáticos
                X_clean = X.copy()
                for col in X_clean.columns:
                    if X_clean[col].dtype in ['float64', 'int64']:
                        # Reemplazar infinitos con NaN
                        X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
                        # Rellenar NaN con 0
                        X_clean[col] = X_clean[col].fillna(0)
                
                # Calcular correlación
                corr_matrix = X_clean.corr().abs()
            except Exception as e:
                logger.error(f"Error al calcular matriz de correlación: {str(e)}")
                logger.error(traceback.format_exc())
                return X
            
            # Matriz triangular superior
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Encontrar características para eliminar (solo de las no esenciales)
            to_drop = []
            
            # Considerar solo correlaciones con columnas no esenciales
            for column in non_essential_cols:
                # Verificar correlaciones de esta columna con todas las demás
                if any(upper[column] > threshold):
                    # Comprobar si está correlacionada con alguna columna esencial
                    correlated_with_essential = False
                    for essential_col in present_essential_cols:
                        if essential_col in upper.index and upper.loc[essential_col, column] > threshold:
                            correlated_with_essential = True
                            break
                    
                    # Si está correlacionada con una columna esencial, eliminarla
                    if correlated_with_essential:
                        to_drop.append(column)
                    else:
                        # Verificar si está correlacionada con otras columnas no esenciales
                        for other_col in non_essential_cols:
                            if column != other_col and other_col in upper.columns:
                                if upper.loc[column, other_col] > threshold:
                                    # Entre dos columnas correlacionadas no esenciales, 
                                    # preferir eliminar la que tenga menor varianza
                                    if column not in to_drop and other_col not in to_drop:
                                        col_var = X[column].var()
                                        other_var = X[other_col].var()
                                        if col_var <= other_var:
                                            to_drop.append(column)
                                        else:
                                            to_drop.append(other_col)
            
            # Eliminar duplicados de la lista de columnas a eliminar
            to_drop = list(set(to_drop))
            
            logger.info(f"Eliminando {len(to_drop)} características correlacionadas de {X.shape[1]} totales")
            logger.info(f"Columnas a eliminar: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")
            logger.info(f"Columnas esenciales preservadas: {present_essential_cols}")
            
            # Si no hay nada que eliminar, devolver el DataFrame original
            if not to_drop:
                return X
            
            # Eliminar características correlacionadas
            X_filtered = X.drop(columns=to_drop)
            
            # Asegurar que el índice es consecutivo
            X_filtered = X_filtered.reset_index(drop=True)
            
            # Verificar que la longitud no cambió
            if len(X_filtered) != len(X):
                logger.error(f"Error: La longitud cambió después de filtrar características correlacionadas: {len(X_filtered)} vs {len(X)}")
                return X  # Devolver el original si hay problemas
            
            return X_filtered
            
        except Exception as e:
            logger.error(f"Error en filter_correlated_features: {str(e)}")
            logger.error(traceback.format_exc())
            # En caso de error, devolver el DataFrame original
            return X
    
    def _create_starter_features(self):
        """
        Crea características basadas en si el jugador fue titular (is_started) o suplente.
        Analiza patrones de rendimiento en diferentes roles.
        """
        logger.info("Creando características de titular vs suplente")
        
        try:
            # Verificar si tenemos la columna is_started
            if 'is_started' not in self.players_data.columns:
                logger.error("No se encontró columna is_started, no se pueden crear características de titular")
                return
            
            # Asegurar que is_started sea binaria (0 o 1)
            self.players_data['is_started'] = pd.to_numeric(self.players_data['is_started'], errors='coerce').fillna(0).astype(int)
            
            # Estadísticas principales a analizar
            main_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'FG%', '3P%', 'FT%']
            available_stats = [stat for stat in main_stats if stat in self.players_data.columns]
            
            # 1. Diferencia de rendimiento como titular vs suplente
            for stat in available_stats:
                # Convertir a numérico
                self.players_data[stat] = pd.to_numeric(self.players_data[stat], errors='coerce').fillna(0)
                
                # Para cada jugador, calcular promedio como titular y como suplente
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    player_data = self.players_data[player_mask]
                    
                    # Solo procesar si hay suficientes partidos en ambos roles
                    starter_data = player_data[player_data['is_started'] == 1]
                    bench_data = player_data[player_data['is_started'] == 0]
                    
                    if len(starter_data) >= 3 and len(bench_data) >= 3:
                        # Calcular promedios
                        starter_avg = starter_data[stat].mean()
                        bench_avg = bench_data[stat].mean()
                        
                        # Diferencia y ratio
                        if bench_avg > 0:
                            starter_impact = ((starter_avg / bench_avg) - 1) * 100  # % de mejora como titular
                        else:
                            starter_impact = 0 if starter_avg == 0 else 100  # Si bench_avg es 0, evitar división por cero
                        
                        # Asignar a todos los registros del jugador
                        self.players_data.loc[player_mask, f'{stat}_starter_impact'] = starter_impact
                                    
            # 2. Consistencia en diferentes roles
            for stat in ['PTS', 'TRB', 'AST']:
                if stat in self.players_data.columns:
                    # Para cada jugador, calcular desviación estándar como titular y como suplente
                    for player in self.players_data['Player'].unique():
                        player_mask = self.players_data['Player'] == player
                        
                        # Filtrar datos de titular y suplente
                        starter_data = self.players_data.loc[player_mask & (self.players_data['is_started'] == 1), stat]
                        bench_data = self.players_data.loc[player_mask & (self.players_data['is_started'] == 0), stat]
                        
                        if len(starter_data) >= 5:
                            starter_std = starter_data.std()
                            starter_mean = starter_data.mean()
                            if starter_mean > 0:
                                starter_cv = (starter_std / starter_mean) * 100  # Coeficiente de variación
                                self.players_data.loc[player_mask, f'{stat}_starter_cv'] = starter_cv
                        
                        if len(bench_data) >= 5:
                            bench_std = bench_data.std()
                            bench_mean = bench_data.mean()
                            if bench_mean > 0:
                                bench_cv = (bench_std / bench_mean) * 100  # Coeficiente de variación
                                self.players_data.loc[player_mask, f'{stat}_bench_cv'] = bench_cv
            
            # 3. Tendencias a lo largo del tiempo en cada rol
            for window in self.window_sizes:
                for stat in ['PTS', 'TRB', 'AST']:
                    if stat in self.players_data.columns:
                        # Crear características de tendencia para cada rol
                        self.players_data[f'{stat}_as_starter_{window}'] = 0
                        self.players_data[f'{stat}_as_bench_{window}'] = 0
                        
                        # Procesar jugador por jugador
                        for player in self.players_data['Player'].unique():
                            player_mask = self.players_data['Player'] == player
                            player_data = self.players_data.loc[player_mask].sort_values('Date')
                            
                            # Calcular tendencias como titular
                            starter_mask = player_data['is_started'] == 1
                            if starter_mask.sum() >= 3:
                                self.players_data.loc[player_mask & starter_mask, f'{stat}_as_starter_{window}'] = \
                                    player_data.loc[starter_mask, stat].rolling(window=min(window, starter_mask.sum()), 
                                                                             min_periods=1).mean()
                            
                            # Calcular tendencias como suplente
                            bench_mask = player_data['is_started'] == 0
                            if bench_mask.sum() >= 3:
                                self.players_data.loc[player_mask & bench_mask, f'{stat}_as_bench_{window}'] = \
                                    player_data.loc[bench_mask, stat].rolling(window=min(window, bench_mask.sum()), 
                                                                           min_periods=1).mean()
            
            # 4. Características de predicción para líneas de apuestas según rol
            for stat, thresholds in self.betting_lines.items():
                if stat in ['PTS', 'TRB', 'AST', '3P'] and stat in self.players_data.columns:
                    for threshold in thresholds:
                        # Calcular ratio de superación del umbral como titular vs suplente
                        starter_over_col = f'{stat}_starter_over_{threshold}'
                        bench_over_col = f'{stat}_bench_over_{threshold}'
                        
                        # Inicializar columnas
                        self.players_data[starter_over_col] = 0
                        self.players_data[bench_over_col] = 0
                        
                        # Calcular por jugador
                        for player in self.players_data['Player'].unique():
                            player_mask = self.players_data['Player'] == player
                            
                            # Datos como titular
                            starter_data = self.players_data.loc[player_mask & (self.players_data['is_started'] == 1)]
                            if len(starter_data) >= 5:
                                starter_over_rate = (starter_data[stat] > threshold).mean()
                                self.players_data.loc[player_mask, starter_over_col] = starter_over_rate
                            
                            # Datos como suplente
                            bench_data = self.players_data.loc[player_mask & (self.players_data['is_started'] == 0)]
                            if len(bench_data) >= 5:
                                bench_over_rate = (bench_data[stat] > threshold).mean()
                                self.players_data.loc[player_mask, bench_over_col] = bench_over_rate
                            
                            # Diferencia entre tasas
                            if len(starter_data) >= 5 and len(bench_data) >= 5:
                                self.players_data.loc[player_mask, f'{stat}_role_diff_{threshold}'] = \
                                    self.players_data.loc[player_mask, starter_over_col] - \
                                    self.players_data.loc[player_mask, bench_over_col]
            
            # 5. Características de minutos jugados según rol
            if 'MP' in self.players_data.columns:
                # Media de minutos como titular y como suplente
                for player in self.players_data['Player'].unique():
                    player_mask = self.players_data['Player'] == player
                    
                    # Minutos como titular
                    starter_mp = self.players_data.loc[player_mask & (self.players_data['is_started'] == 1), 'MP']
                    if len(starter_mp) >= 3:
                        self.players_data.loc[player_mask, 'MP_as_starter_avg'] = starter_mp.mean()
                    
                    # Minutos como suplente
                    bench_mp = self.players_data.loc[player_mask & (self.players_data['is_started'] == 0), 'MP']
                    if len(bench_mp) >= 3:
                        self.players_data.loc[player_mask, 'MP_as_bench_avg'] = bench_mp.mean()
                    
                    # Ratio entre ambos
                    if len(starter_mp) >= 3 and len(bench_mp) >= 3 and bench_mp.mean() > 0:
                        self.players_data.loc[player_mask, 'MP_starter_to_bench_ratio'] = \
                            starter_mp.mean() / bench_mp.mean()
            
            # 6. Identificar "spark plugs" (jugadores que rinden mejor saliendo del banquillo)
            self.players_data['is_spark_plug'] = 0
            for player in self.players_data['Player'].unique():
                player_mask = self.players_data['Player'] == player
                
                # Verificar si hay suficientes partidos en ambos roles
                starter_games = self.players_data.loc[player_mask & (self.players_data['is_started'] == 1)]
                bench_games = self.players_data.loc[player_mask & (self.players_data['is_started'] == 0)]
                
                if len(starter_games) >= 5 and len(bench_games) >= 5:
                    # Comparar rendimiento por minuto para neutralizar diferencia de minutos
                    if 'MP' in self.players_data.columns and 'PTS' in self.players_data.columns:
                        # Puntos por minuto
                        starter_ppm = (starter_games['PTS'] / starter_games['MP'].clip(lower=1)).mean()
                        bench_ppm = (bench_games['PTS'] / bench_games['MP'].clip(lower=1)).mean()
                        
                        # Si rinde mejor saliendo del banquillo
                        if bench_ppm > starter_ppm * 1.1:  # Al menos 10% mejor como suplente
                            self.players_data.loc[player_mask, 'is_spark_plug'] = 1
            
            logger.info("Características de titular vs suplente creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error al crear características de titular vs suplente: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.players_data