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


# Configuración del sistema de logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("teams_features_engineering.log", mode='w'),  # Sobrescribe el archivo en cada ejecución
        logging.StreamHandler()
    ]
)

# Reducir verbosidad de los warnings de pandas
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger('TeamsFeatures')

class TeamsFeatures:
    def __init__(self, teams_data, window_sizes=[3, 5, 10, 20], correlation_threshold=0.95, enable_correlation_analysis=True, n_jobs=1):
        """
        Inicializa el sistema de ingeniería de características de equipos
        
        Args:
            teams_data (pd.DataFrame): DataFrame con los datos de los partidos
            window_sizes (list): Lista de tamaños de ventana para las características temporales
            correlation_threshold (float): Umbral de correlación para las características
            enable_correlation_analysis (bool): Si True, se realizará análisis de correlación
            n_jobs (int): Número de procesos en paralelo para el procesamiento
        """
        self.teams_data = teams_data.copy()
        # Asegurar que los tamaños de ventana sean enteros positivos
        self.window_sizes = [int(w) for w in window_sizes if int(w) > 0]
        if not self.window_sizes:
            self.window_sizes = [3, 5, 10, 20]  # valores por defecto si no hay válidos
        self.window_sizes.sort()  # ordenar de menor a mayor
        
        self.correlation_threshold = correlation_threshold
        self.enable_correlation_analysis = enable_correlation_analysis
        self.n_jobs = n_jobs       
        
        # Asegurar que las fechas estén en formato datetime
        if 'Date' in self.teams_data.columns:
            self.teams_data['Date'] = pd.to_datetime(self.teams_data['Date'])
            
        # Calcular total_points (suma de puntos de ambos equipos)
        self.teams_data['total_points'] = self.teams_data['PTS'] + self.teams_data['PTS_Opp']
        
        logger.info(f"Inicializado sistema de características con {len(self.teams_data)} registros y ventanas {self.window_sizes}")
        
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
                # Para std es importante asegurar que hay suficientes valores no-NaN
                # y manejar los casos donde hay poca variación
                result = series.rolling(window=window, min_periods=min_periods).std()
                
                # Reemplazar NaN con 0 para series con poca variación
                # Si la desviación estándar no se puede calcular (pocos datos o todos iguales)
                # es razonable usar 0 como valor predeterminado
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
    
    def _preprocess_data(self):
        """
        Realiza el preprocesamiento inicial de los datos
        """
        logger.info("Iniciando preprocesamiento de datos")
        
        # Convertir columnas de porcentajes a valores decimales si contienen '%'
        for col in self.teams_data.columns:
            if isinstance(self.teams_data[col].iloc[0], str) and '%' in self.teams_data[col].iloc[0]:
                self.teams_data[col] = self.teams_data[col].str.replace('%', '').astype(float) / 100
        
        # Verificar y eliminar duplicados
        before_drop = len(self.teams_data)
        if 'Team' in self.teams_data.columns and 'Date' in self.teams_data.columns:
            logger.info("Verificando duplicados por equipo y fecha")
            
            # Asegurar que Date está en formato datetime
            self.teams_data['Date'] = pd.to_datetime(self.teams_data['Date'])
            
            # Contar duplicados
            duplicates = self.teams_data.duplicated(subset=['Team', 'Date'], keep=False)
            n_duplicates = duplicates.sum()
            
            if n_duplicates > 0:
                logger.warning(f"Se encontraron {n_duplicates} filas duplicadas por equipo y fecha")
                
                # Mostrar algunos ejemplos de duplicados
                duplicate_examples = self.teams_data[duplicates].sort_values(['Team', 'Date']).head(10)
                logger.debug(f"Ejemplos de duplicados:\n{duplicate_examples[['Team', 'Date']]}")
                
                # Eliminar duplicados manteniendo la primera ocurrencia
                self.teams_data = self.teams_data.drop_duplicates(subset=['Team', 'Date'], keep='first')
                after_drop = len(self.teams_data)
                logger.info(f"Se eliminaron {before_drop - after_drop} duplicados")
        
        # Ordenar datos por equipo y fecha
        self.teams_data.sort_values(['Team', 'Date'], inplace=True)
        
        # Crear columnas para diferenciales
        self._create_differential_features()
        
        logger.info("Preprocesamiento completado")
        return self.teams_data
    
    def _create_differential_features(self):
        """
        Crea características de diferenciales entre equipos
        """
        logger.info("Creando características diferenciales")
        
        # Diferenciales básicos
        self.teams_data['PTS_diff'] = self.teams_data['PTS'] - self.teams_data['PTS_Opp']
        self.teams_data['FG_diff'] = self.teams_data['FG'] - self.teams_data['FG_Opp']
        self.teams_data['FGA_diff'] = self.teams_data['FGA'] - self.teams_data['FGA_Opp']
        self.teams_data['FG%_diff'] = self.teams_data['FG%'] - self.teams_data['FG%_Opp']
        self.teams_data['2P_diff'] = self.teams_data['2P'] - self.teams_data['2P_Opp']
        self.teams_data['2PA_diff'] = self.teams_data['2PA'] - self.teams_data['2PA_Opp']
        self.teams_data['2P%_diff'] = self.teams_data['2P%'] - self.teams_data['2P%_Opp']
        self.teams_data['3P_diff'] = self.teams_data['3P'] - self.teams_data['3P_Opp']
        self.teams_data['3PA_diff'] = self.teams_data['3PA'] - self.teams_data['3PA_Opp']
        self.teams_data['3P%_diff'] = self.teams_data['3P%'] - self.teams_data['3P%_Opp']
        self.teams_data['FT_diff'] = self.teams_data['FT'] - self.teams_data['FT_Opp']
        self.teams_data['FTA_diff'] = self.teams_data['FTA'] - self.teams_data['FTA_Opp']
        self.teams_data['FT%_diff'] = self.teams_data['FT%'] - self.teams_data['FT%_Opp']
        
        # Eficiencias ofensivas y defensivas
        self.teams_data['offensive_efficiency'] = self.teams_data['PTS'] / self.teams_data['FGA'] if 'FGA' in self.teams_data.columns else 0
        self.teams_data['defensive_efficiency'] = self.teams_data['PTS_Opp'] / self.teams_data['FGA_Opp'] if 'FGA_Opp' in self.teams_data.columns else 0
        self.teams_data['efficiency_diff'] = self.teams_data['offensive_efficiency'] - self.teams_data['defensive_efficiency']
        
    def create_rolling_features(self):
        """
        Crea características de ventana móvil para cada equipo
        """
        logger.info(f"Creando características de ventana móvil con ventanas: {self.window_sizes}")
        
        try:
            # Columnas numéricas para crear características móviles
            base_cols = [
                'PTS', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
                'PTS_Opp', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
                'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'total_points'
            ]
            
            # Filtrar columnas que realmente existen en el DataFrame
            numeric_cols = [col for col in base_cols if col in self.teams_data.columns]
            logger.debug(f"Columnas base para características rolling: {len(numeric_cols)}")
            
            # Crear características móviles para cada equipo
            teams = self.teams_data['Team'].unique()
            all_dfs = []
            
            for team in tqdm(teams, desc="Procesando equipos"):
                try:
                    # Crear una copia del DataFrame del equipo
                    team_data = self.teams_data[self.teams_data['Team'] == team].copy()
                    team_data = team_data.sort_values('Date')
                    
                    # Asegurar índice consecutivo
                    team_data = team_data.reset_index(drop=True)
                    
                    # Diccionario para almacenar todas las nuevas características
                    new_features = {}
                    
                    # Para cada ventana y columna numérica
                    for window in self.window_sizes:
                        for col in numeric_cols:
                            try:
                                # Calcular todas las características para esta columna y ventana
                                new_features[f'{col}_mean_{window}'] = self._safe_rolling(team_data[col], window, 'mean')
                                # Calcular std de forma segura y limitar a valores razonables
                                std_values = self._safe_rolling(team_data[col], window, 'std')
                                # Limitar valores std a un rango razonable para evitar valores extremos
                                # Usar un límite superior basado en la media de la columna
                                col_mean = team_data[col].mean()
                                std_limit = max(col_mean * 2, 1.0) if col_mean > 0 else 10.0
                                std_values = std_values.clip(0, std_limit)
                                new_features[f'{col}_std_{window}'] = std_values
                                new_features[f'{col}_max_{window}'] = self._safe_rolling(team_data[col], window, 'max')
                                new_features[f'{col}_min_{window}'] = self._safe_rolling(team_data[col], window, 'min')
                            except Exception as e:
                                logger.error(f"Error al calcular características rolling para {col}, ventana {window}: {str(e)}")
                    
                    # Calcular tendencias después de tener todas las medias
                    for window in self.window_sizes:
                        for col in numeric_cols:
                            mean_col = f'{col}_mean_{window}'
                            if mean_col in new_features:
                                try:
                                    # Convertir a numérico de forma segura
                                    col_values = pd.to_numeric(team_data[col], errors='coerce')
                                    mean_values = pd.to_numeric(new_features[mean_col], errors='coerce')
                                    
                                    # Reemplazar valores infinitos o NaN
                                    col_values = col_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    mean_values = mean_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    
                                    # Calcular diferencia entre valores actuales y medias móviles
                                    trend = col_values - mean_values
                                    
                                    # Determinar un límite de tendencia razonable basado en la naturaleza de la columna
                                    if 'PTS' in col or 'total_points' in col:
                                        trend_limit = 30.0  # Para puntos
                                    elif 'FG' in col or '3P' in col:
                                        trend_limit = 15.0  # Para tiros
                                    elif 'FT' in col:
                                        trend_limit = 10.0  # Para tiros libres
                                    elif '%' in col:
                                        trend_limit = 0.3   # Para porcentajes
                                    else:
                                        trend_limit = 20.0  # Límite predeterminado
                                    
                                    # Limitar valores extremos
                                    trend = trend.clip(-trend_limit, trend_limit)
                                    
                                    # Guardar la tendencia limitada
                                    new_features[f'{col}_trend_{window}'] = trend
                                except Exception as e:
                                    logger.error(f"Error al calcular tendencia para {col}, ventana {window}: {str(e)}")
                                    # En caso de error, crear una columna de ceros
                                    new_features[f'{col}_trend_{window}'] = pd.Series(0, index=team_data.index)
                    
                    # Características adicionales de racha
                    if 'is_win' in team_data.columns:
                        for window in [10, 20]:
                            try:
                                new_features[f'win_rate_{window}'] = self._safe_rolling(team_data['is_win'], window, 'mean')
                            except Exception as e:
                                logger.error(f"Error al calcular win_rate_{window}: {str(e)}")
                        
                        try:
                            new_features['win_streak'] = self._safe_rolling(team_data['is_win'], 10, 'sum')
                        except Exception as e:
                            logger.error(f"Error al calcular win_streak: {str(e)}")
                    
                    # Verificar que todas las características tienen la misma longitud
                    lengths = {k: len(v) for k, v in new_features.items()}
                    if len(set(lengths.values())) > 1:
                        logger.warning(f"Longitudes inconsistentes en características para equipo {team}: {lengths}")
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
                    
                    # Asegurar que la longitud coincide con team_data
                    for k, v in list(new_features.items()):
                        if len(v) != len(team_data):
                            logger.warning(f"Longitud de {k} ({len(v)}) no coincide con team_data ({len(team_data)})")
                            if len(v) > len(team_data):
                                new_features[k] = v[:len(team_data)]
                            else:
                                # Eliminar característica si es demasiado corta
                                logger.warning(f"Eliminando característica {k} por longitud insuficiente")
                                del new_features[k]
                    
                    # Crear DataFrame con las nuevas características
                    features_df = pd.DataFrame(new_features, index=team_data.index)
                    
                    # Agregar todas las nuevas características de una vez
                    team_data = pd.concat([team_data, features_df], axis=1)
                    all_dfs.append(team_data)
                    
                except Exception as e:
                    logger.error(f"Error al procesar equipo {team}: {str(e)}")
                    # Agregar el equipo sin características adicionales para no perder datos
                    all_dfs.append(self.teams_data[self.teams_data['Team'] == team])
            
            # Combinar todos los DataFrames
            if all_dfs:
                self.teams_data = pd.concat(all_dfs).sort_index()
                
                # Asegurar que no hay fragmentación
                self.teams_data = self.teams_data.copy()
                
                logger.info(f"Creadas nuevas características de ventana móvil")
            
            return self.teams_data
            
        except Exception as e:
            logger.error(f"Error en create_rolling_features: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            # Devolver el DataFrame original
            return self.teams_data
    
    def create_opponent_features(self):
        """
        Crea características basadas en oponentes
        """
        logger.info("Creando características basadas en oponentes")
        
        # Crear un diccionario para almacenar métricas históricas de cada equipo
        team_metrics = {}
        teams = self.teams_data['Team'].unique()
        
        for team in teams:
            team_data = self.teams_data[self.teams_data['Team'] == team].sort_values('Date')
            # Calcular métricas promedio para cada equipo
            team_metrics[team] = {
                'avg_PTS': team_data['PTS'].mean(),
                'avg_PTS_against': team_data['PTS_Opp'].mean(),
                'win_rate': team_data['is_win'].mean(),
                'home_win_rate': team_data[team_data['is_home'] == True]['is_win'].mean() if 'is_home' in team_data.columns else 0.5,
                'away_win_rate': team_data[team_data['is_home'] == False]['is_win'].mean() if 'is_home' in team_data.columns else 0.5
            }
        
        # Agregar métricas del oponente como características
        self.teams_data['opp_avg_PTS'] = self.teams_data['Opp'].map(lambda x: team_metrics.get(x, {}).get('avg_PTS', 0))
        self.teams_data['opp_avg_PTS_against'] = self.teams_data['Opp'].map(lambda x: team_metrics.get(x, {}).get('avg_PTS_against', 0))
        self.teams_data['opp_win_rate'] = self.teams_data['Opp'].map(lambda x: team_metrics.get(x, {}).get('win_rate', 0.5))
        
        # Características de ventaja de local/visitante
        self.teams_data['home_advantage'] = self.teams_data.apply(
            lambda row: team_metrics.get(row['Team'], {}).get('home_win_rate', 0.5) if row.get('is_home', True) else 1 - team_metrics.get(row['Opp'], {}).get('home_win_rate', 0.5),
            axis=1
        )
        
        # Históricas contra este oponente específico
        all_dfs = []
        
        for team in tqdm(teams, desc="Procesando históricas por oponente"):
            for opp in teams:
                if team == opp:
                    continue
                    
                # Filtrar partidos entre estos dos equipos
                matchups = self.teams_data[(self.teams_data['Team'] == team) & (self.teams_data['Opp'] == opp)].sort_values('Date')
                
                if len(matchups) > 0:
                    # Calcular estadísticas históricas contra este oponente
                    matchups['vs_opp_win_rate'] = matchups['is_win'].expanding().mean()
                    matchups['vs_opp_avg_PTS'] = matchups['PTS'].expanding().mean()
                    matchups['vs_opp_avg_PTS_against'] = matchups['PTS_Opp'].expanding().mean()
                    
                    all_dfs.append(matchups)
        
        if all_dfs:
            self.teams_data = pd.concat(all_dfs).reset_index(drop=True)
            
        logger.info("Características de oponentes creadas")
        return self.teams_data
        
    
    def create_overtime_features(self):
        """
        Crea características relacionadas con partidos en tiempo extra
        """
        logger.info("Creando características de tiempo extra")
        
        # Características de tiempo extra
        if 'has_overtime' in self.teams_data.columns:
            # Probabilidad histórica de tiempo extra por equipo
            team_ot_rates = self.teams_data.groupby('Team')['has_overtime'].mean().to_dict()
            self.teams_data['team_overtime_rate'] = self.teams_data['Team'].map(team_ot_rates)
            
            # Probabilidad de victoria en tiempos extra
            ot_games = self.teams_data[self.teams_data['has_overtime'] == True]
            if len(ot_games) > 0:
                team_ot_win_rates = ot_games.groupby('Team')['is_win'].mean().to_dict()
                self.teams_data['team_overtime_win_rate'] = self.teams_data['Team'].map(
                    lambda x: team_ot_win_rates.get(x, 0.5)
                )
            
            # Para cada ventana, calcular frecuencia de tiempos extra
            for window in self.window_sizes:
                self.teams_data[f'overtime_rate_{window}'] = self.teams_data.groupby('Team')['has_overtime'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        logger.info("Características de tiempo extra creadas")
        return self.teams_data
    
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
                return X
            
            # Matriz triangular superior
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Encontrar características para eliminar
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            logger.info(f"Eliminando {len(to_drop)} características correlacionadas de {X.shape[1]} totales")
            
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
            # En caso de error, devolver el DataFrame original
            return X
    
    def _create_temporal_features(self):
        """
        Crea características temporales y de calendario
        """
        logger.info("Creando características temporales")
        
        try:
            # Asegurar que tenemos un índice numérico
            self.teams_data = self.teams_data.reset_index(drop=True)
            
            # Características de calendario
            self.teams_data['dayofweek'] = self.teams_data['Date'].dt.dayofweek
            self.teams_data['month'] = self.teams_data['Date'].dt.month
            self.teams_data['is_weekend'] = self.teams_data['dayofweek'].isin([5, 6]).astype(int)
            
            # Días desde último partido
            self.teams_data['days_rest'] = self.teams_data.groupby('Team')['Date'].diff().dt.days.fillna(0)
            
            # Partidos en últimos 7 días (fatiga)
            games_last_7_days = []
            
            # Procesar cada equipo por separado
            for team in self.teams_data['Team'].unique():
                team_data = self.teams_data[self.teams_data['Team'] == team].copy()
                team_data = team_data.sort_values('Date')
                
                # Calcular partidos en los últimos 7 días
                games_7d = []
                for date in team_data['Date']:
                    mask = (team_data['Date'] <= date) & (team_data['Date'] > date - pd.Timedelta(days=7))
                    games_7d.append(len(team_data[mask]))
                
                team_data['games_last_7_days'] = games_7d
                games_last_7_days.append(team_data)
            
            # Combinar todos los resultados
            self.teams_data = pd.concat(games_last_7_days).sort_index()
            
            # Características de temporada
            self.teams_data['games_played'] = self.teams_data.groupby('Team').cumcount() + 1
            
            # Calcular el progreso de la temporada de manera segura
            for team in self.teams_data['Team'].unique():
                team_mask = self.teams_data['Team'] == team
                team_games = self.teams_data.loc[team_mask, 'games_played']
                min_games = team_games.min()
                max_games = team_games.max()
                games_range = max(1, max_games - min_games)  # Evitar división por cero
                
                self.teams_data.loc[team_mask, 'season_progress'] = (
                    (team_games - min_games) / games_range
                )
            
            logger.info("Características temporales creadas")
            
        except Exception as e:
            logger.error(f"Error al crear características temporales: {str(e)}")
            raise
    
    def _create_advanced_momentum_features(self):
        """
        Crea características avanzadas de momentum y rachas
        """
        logger.info("Creando características de momentum")
        
        try:
            # Asegurar que tenemos un índice numérico
            self.teams_data = self.teams_data.reset_index(drop=True)
            
            # Procesar cada equipo por separado
            all_teams = []
            
            for team in self.teams_data['Team'].unique():
                # Obtener datos del equipo
                team_data = self.teams_data[self.teams_data['Team'] == team].copy()
                team_data = team_data.sort_values('Date')
                
                # Rachas actuales
                if 'is_win' in team_data.columns:
                    # Racha de victorias
                    win_streaks = []
                    current_streak = 0
                    
                    for win in team_data['is_win']:
                        if win:
                            current_streak += 1
                        else:
                            current_streak = 0
                        win_streaks.append(current_streak)
                    
                    team_data['current_win_streak'] = win_streaks
                    
                    # Racha de victorias en casa
                    if 'is_home' in team_data.columns:
                        home_games = team_data[team_data['is_home'] == 1]
                        if not home_games.empty:
                            home_win_streaks = []
                            current_home_streak = 0
                            
                            for idx, row in team_data.iterrows():
                                if row['is_home'] == 1:
                                    if row['is_win']:
                                        current_home_streak += 1
                                    else:
                                        current_home_streak = 0
                                home_win_streaks.append(current_home_streak)
                            
                            team_data['current_home_win_streak'] = home_win_streaks
                
                # Momentum ofensivo y defensivo
                for window in self.window_sizes:
                    # Momentum ofensivo
                    if 'PTS' in team_data.columns:
                        pts_mean_recent = self._safe_rolling(team_data['PTS'], window, 'mean')
                        pts_mean_older = self._safe_rolling(team_data['PTS'], window*2, 'mean')
                        team_data[f'offensive_momentum_{window}'] = pts_mean_recent - pts_mean_older
                    
                    # Momentum defensivo
                    if 'PTS_Opp' in team_data.columns:
                        pts_opp_mean_older = self._safe_rolling(team_data['PTS_Opp'], window*2, 'mean')
                        pts_opp_mean_recent = self._safe_rolling(team_data['PTS_Opp'], window, 'mean')
                        team_data[f'defensive_momentum_{window}'] = pts_opp_mean_older - pts_opp_mean_recent
                    
                    # Momentum de eficiencia
                    if f'offensive_momentum_{window}' in team_data.columns and f'defensive_momentum_{window}' in team_data.columns:
                        team_data[f'efficiency_momentum_{window}'] = (
                            team_data[f'offensive_momentum_{window}'] + 
                            team_data[f'defensive_momentum_{window}']
                        )
                
                # Rachas de puntuación
                if 'PTS' in team_data.columns:
                    # Mejora en puntuación
                    pts_diff = team_data['PTS'].diff()
                    team_data['pts_improvement'] = (pts_diff > 0).astype(int)
                    
                    # Racha de mejora
                    scoring_streaks = []
                    current_scoring_streak = 0
                    
                    for improvement in team_data['pts_improvement'].fillna(0):
                        if improvement:
                            current_scoring_streak += 1
                        else:
                            current_scoring_streak = 0
                        scoring_streaks.append(current_scoring_streak)
                    
                    team_data['current_scoring_streak'] = scoring_streaks
                
                all_teams.append(team_data)
            
            # Combinar todos los equipos
            self.teams_data = pd.concat(all_teams).sort_index()
            
            # Desfragmentar
            self.teams_data = self.teams_data.copy()
            
            logger.info("Características de momentum creadas")
            
        except Exception as e:
            logger.error(f"Error al crear características de momentum: {str(e)}")
            raise
    
    def _create_advanced_matchup_features(self):
        """
        Crea características avanzadas de enfrentamientos entre equipos
        """
        logger.info("Creando características avanzadas de enfrentamientos")
        
        # Crear características históricas de enfrentamientos
        teams = self.teams_data['Team'].unique()
        all_matchups = []
        
        for team in tqdm(teams, desc="Procesando enfrentamientos"):
            for opp in teams:
                if team == opp:
                    continue
                    
                # Filtrar partidos entre estos equipos
                matchups = self.teams_data[
                    ((self.teams_data['Team'] == team) & (self.teams_data['Opp'] == opp)) |
                    ((self.teams_data['Team'] == opp) & (self.teams_data['Opp'] == team))
                ].sort_values('Date')
                
                if len(matchups) > 0:
                    # Calcular estadísticas de enfrentamientos
                    team_stats = []
                    for idx, row in matchups.iterrows():
                        current_team = row['Team']
                        current_opp = row['Opp']
                        
                        # Filtrar partidos anteriores entre estos equipos
                        previous_matchups = matchups[matchups['Date'] < row['Date']]
                        
                        if len(previous_matchups) > 0:
                            # Estadísticas cuando el equipo actual era local
                            team_home = previous_matchups[
                                (previous_matchups['Team'] == current_team) & 
                                (previous_matchups['is_home'] == True)
                            ]
                            
                            # Estadísticas cuando el equipo actual era visitante
                            team_away = previous_matchups[
                                (previous_matchups['Team'] == current_team) & 
                                (previous_matchups['is_home'] == False)
                            ]
                            
                            stats = {
                                'h2h_games': len(previous_matchups),
                                'h2h_wins': len(previous_matchups[previous_matchups['Team'] == current_team]),
                                'h2h_home_wins': len(team_home[team_home['is_win'] == True]),
                                'h2h_away_wins': len(team_away[team_away['is_win'] == True]),
                                'h2h_avg_points': previous_matchups[previous_matchups['Team'] == current_team]['PTS'].mean(),
                                'h2h_avg_points_against': previous_matchups[previous_matchups['Team'] == current_team]['PTS_Opp'].mean(),
                                'h2h_avg_margin': previous_matchups[previous_matchups['Team'] == current_team]['PTS_diff'].mean()
                            }
                        else:
                            stats = {
                                'h2h_games': 0,
                                'h2h_wins': 0,
                                'h2h_home_wins': 0,
                                'h2h_away_wins': 0,
                                'h2h_avg_points': row['PTS'],
                                'h2h_avg_points_against': row['PTS_Opp'],
                                'h2h_avg_margin': row['PTS'] - row['PTS_Opp']
                            }
                            
                        team_stats.append({
                            'Team': current_team,
                            'Opp': current_opp,
                            'Date': row['Date'],
                            **stats
                        })
                    
                    matchup_df = pd.DataFrame(team_stats)
                    all_matchups.append(matchup_df)
        
        if all_matchups:
            matchup_features = pd.concat(all_matchups)
            
            # Unir características de enfrentamientos con el DataFrame principal
            self.teams_data = self.teams_data.merge(
                matchup_features,
                on=['Team', 'Opp', 'Date'],
                how='left'
            )
            
            # Calcular características adicionales de enfrentamientos
            self.teams_data['h2h_win_rate'] = self.teams_data['h2h_wins'] / self.teams_data['h2h_games'].replace(0, 1)
            self.teams_data['h2h_home_win_rate'] = self.teams_data.apply(
                lambda x: x['h2h_home_wins'] / max(1, x['h2h_games']) if x['is_home'] else 
                         x['h2h_away_wins'] / max(1, x['h2h_games']),
                axis=1
            )
        
        logger.info("Características avanzadas de enfrentamientos creadas")
        
    def _create_advanced_efficiency_features(self):
        """
        Crea características avanzadas de eficiencia y ritmo de juego
        """
        logger.info("Creando características avanzadas de eficiencia")
        
        try:
            # Implementación directa de offensive_rating y defensive_rating
            # En lugar de depender de fórmulas complejas, usar cálculos más simples
            
            # Verificar que las columnas requeridas existen
            required_cols = ['PTS', 'PTS_Opp', 'FGA', 'FGA_Opp', 'FTA', 'FTA_Opp']
            missing_cols = [col for col in required_cols if col not in self.teams_data.columns]
            if missing_cols:
                logger.warning(f"Faltan columnas para cálculos de eficiencia: {missing_cols}")
                # Si faltan columnas, crear ratings con valores por defecto
                self.teams_data['offensive_rating'] = 100.0
                self.teams_data['defensive_rating'] = 100.0
                self.teams_data['net_rating'] = 0.0
                return
                
            # 1. Implementación más básica y robusta de posesiones
            # Estimar posesiones usando múltiples componentes
            possessions = self.teams_data['FGA'].copy()  # Base: intentos de tiro
            
            # Añadir componente de tiros libres (evitando división)
            if 'FTA' in self.teams_data.columns:
                possessions = possessions + (0.44 * self.teams_data['FTA'])
                
            # Añadir componente de rebotes
            if 'ORB' in self.teams_data.columns and 'DRB' in self.teams_data.columns:
                possessions = possessions + (self.teams_data['ORB'] + self.teams_data['DRB']) * 0.1
                
            # Añadir componente de pérdidas de balón
            if 'TOV' in self.teams_data.columns:
                possessions = possessions + self.teams_data['TOV']
                
            # Asegurar un mínimo de posesiones
            possessions = possessions.clip(lower=20)  # Mínimo razonable de posesiones
            
            # Guardar para uso posterior
            self.teams_data['possessions'] = possessions
            
            # 2. Cálculo directo de offensive_rating (limitar a rango razonable)
            try:
                # Intentar calcular usando fórmula normal
                self.teams_data['offensive_rating'] = (self.teams_data['PTS'] * 100 / possessions).clip(50, 150)
            except Exception as e:
                logger.error(f"Error en cálculo de offensive_rating: {str(e)}")
                # En caso de error, usar una aproximación más simple
                self.teams_data['offensive_rating'] = (self.teams_data['PTS'] * 0.8).clip(50, 150)
            
            # 3. Cálculo directo de defensive_rating (limitar a rango razonable)
            try:
                self.teams_data['defensive_rating'] = (self.teams_data['PTS_Opp'] * 100 / possessions).clip(50, 150)
            except Exception as e:
                logger.error(f"Error en cálculo de defensive_rating: {str(e)}")
                # En caso de error, usar una aproximación más simple
                self.teams_data['defensive_rating'] = (self.teams_data['PTS_Opp'] * 0.8).clip(50, 150)
                
            # 4. Manejo explícito de valores problemáticos
            # Reemplazar infinitos y nulos
            for col in ['offensive_rating', 'defensive_rating']:
                # Identificar valores problemáticos
                inf_mask = np.isinf(self.teams_data[col])
                nan_mask = pd.isna(self.teams_data[col])
                
                if inf_mask.any() or nan_mask.any():
                    logger.warning(f"Encontrados {inf_mask.sum()} infinitos y {nan_mask.sum()} NaN en {col}")
                    
                    # Calcular valor medio para usar como sustituto
                    mean_value = self.teams_data[col][~inf_mask & ~nan_mask].mean()
                    if pd.isna(mean_value) or np.isinf(mean_value):
                        mean_value = 100.0  # Valor por defecto si la media también es problemática
                    
                    # Sustituir valores problemáticos con la media o valor por defecto
                    self.teams_data.loc[inf_mask | nan_mask, col] = mean_value
                    
                    logger.info(f"Valores problemáticos en {col} sustituidos con {mean_value}")
            
            # 5. Calcular net_rating
            try:
                # Asegurarse de que offensive_rating y defensive_rating son numéricos
                off_rating = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
                def_rating = pd.to_numeric(self.teams_data['defensive_rating'], errors='coerce')
                
                # Calcular net_rating de forma segura
                net_rating = off_rating - def_rating
                
                # Limitar a valores razonables (-50 a 50)
                net_rating = net_rating.clip(-50, 50)
                
                # Reemplazar valores problemáticos
                net_rating = net_rating.replace([np.inf, -np.inf], 0).fillna(0)
                
                # Asignar al DataFrame
                self.teams_data['net_rating'] = net_rating
                
                logger.info("net_rating calculado correctamente")
            except Exception as e:
                logger.error(f"Error al calcular net_rating: {str(e)}")
                # En caso de error, usar valor por defecto
                self.teams_data['net_rating'] = 0.0
            
            # 6. Calcular pace (ritmo de juego) de forma robusta
            try:
                # Verificar columnas necesarias
                if all(col in self.teams_data.columns for col in ['possessions', 'MP']):
                    # Asegurar que los minutos son positivos
                    minutes = pd.to_numeric(self.teams_data['MP'], errors='coerce').fillna(240).clip(lower=1)
                    
                    # Fórmula de pace: posesiones por 48 minutos
                    pace = 48 * (self.teams_data['possessions'] / (minutes / 5))
                    
                    # Limitar a valores razonables
                    pace = pace.clip(75, 125)
                    
                    # Manejar valores problemáticos
                    pace = pace.replace([np.inf, -np.inf], 100).fillna(100)
                    
                    # Asignar al DataFrame
                    self.teams_data['pace'] = pace
                    
                    logger.info("pace calculado correctamente")
                else:
                    logger.warning("Faltan columnas para calcular pace, usando valor por defecto")
                    self.teams_data['pace'] = 100.0
            except Exception as e:
                logger.error(f"Error al calcular pace: {str(e)}")
                self.teams_data['pace'] = 100.0
            
            # Completar resto de métricas de eficiencia básicas
            try:
                # True Shooting Percentage (TS%)
                if all(col in self.teams_data.columns for col in ['PTS', 'FGA', 'FTA']):
                    # Denominador seguro
                    ts_denom = 2 * (self.teams_data['FGA'] + 0.44 * self.teams_data['FTA']).clip(lower=1)
                    self.teams_data['ts_pct'] = (self.teams_data['PTS'] / ts_denom).clip(0, 1).fillna(0.5)
                    
                # Effective Field Goal Percentage (eFG%)
                if all(col in self.teams_data.columns for col in ['FG', '3P', 'FGA']):
                    efg_denom = self.teams_data['FGA'].clip(lower=1)
                    self.teams_data['efg_pct'] = ((self.teams_data['FG'] + 0.5 * self.teams_data['3P']) / efg_denom).clip(0, 1).fillna(0.45)
                
                logger.info("Métricas de eficiencia básicas calculadas correctamente")
            except Exception as e:
                logger.error(f"Error al calcular métricas de eficiencia básicas: {str(e)}")
            
            logger.info("Características avanzadas de eficiencia creadas correctamente")
            
        except Exception as e:
            logger.error(f"Error general en características de eficiencia: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            
            # En caso de error, crear columnas requeridas con valores por defecto
            self.teams_data['offensive_rating'] = 100.0
            self.teams_data['defensive_rating'] = 100.0
            self.teams_data['net_rating'] = 0.0
    
    def _create_advanced_situational_features(self):
        """
        Crea características avanzadas para situaciones específicas
        """
        logger.info("Creando características situacionales")
        
        # Rendimiento por día de la semana
        for day in range(7):
            # Crear la máscara de forma segura
            mask = self.teams_data['dayofweek'] == day
            self._safe_team_feature_calculation(mask, f'day_{day}', 'is_win')
        
        # Rendimiento por mes
        for month in range(1, 13):
            # Crear la máscara de forma segura
            mask = self.teams_data['month'] == month
            self._safe_team_feature_calculation(mask, f'month_{month}', 'is_win')
        
        # Rendimiento con diferentes días de descanso
        for rest_days in [0, 1, 2, 3]:
            try:
                # Convertir a numérico de forma segura
                days_rest = pd.to_numeric(self.teams_data['days_rest'], errors='coerce')
                days_rest = days_rest.replace([np.inf, -np.inf], np.nan).fillna(1)
                
                # Crear la máscara de forma segura
                mask = days_rest == rest_days
                self._safe_team_feature_calculation(mask, f'rest_{rest_days}', 'is_win')
            except Exception as e:
                logger.error(f"Error al procesar días de descanso {rest_days}: {str(e)}")
                # Valores por defecto en caso de error
                self.teams_data[f'win_rate_rest_{rest_days}'] = 0.5
                self.teams_data[f'avg_points_rest_{rest_days}'] = self.teams_data['PTS'].mean()
        
        # Rendimiento en juegos cerrados (diferencia <= 5 puntos)
        try:
            # Crear máscara para juegos cerrados
            pts_diff = pd.to_numeric(self.teams_data['PTS_diff'], errors='coerce')
            pts_diff = pts_diff.replace([np.inf, -np.inf], np.nan).fillna(0)
            close_games = abs(pts_diff) <= 5
            
            # Calcular solo win_rate para juegos cerrados (no avg_points)
            if close_games.sum() > 0:
                win_values = pd.to_numeric(self.teams_data.loc[close_games, 'is_win'], errors='coerce')
                win_values = win_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                self.teams_data.loc[close_games, 'close_game_win_rate'] = self.teams_data.loc[close_games].groupby('Team')['is_win'].transform('mean')
                
                # Propagar los valores al resto de filas del mismo equipo
                for team in self.teams_data['Team'].unique():
                    team_mask = self.teams_data['Team'] == team
                    team_close_mask = close_games & team_mask
                    
                    if team_close_mask.any():
                        team_win_rate = self.teams_data.loc[team_close_mask, 'close_game_win_rate'].mean()
                        self.teams_data.loc[team_mask, 'close_game_win_rate'] = team_win_rate
            else:
                self.teams_data['close_game_win_rate'] = 0.5
        except Exception as e:
            logger.error(f"Error al calcular características para juegos cerrados: {str(e)}")
            self.teams_data['close_game_win_rate'] = 0.5
        
        # Rendimiento en overtime
        if 'has_overtime' in self.teams_data.columns:
            try:
                # Convertir a booleano de forma segura
                ot_games = self.teams_data['has_overtime'] == True
                self._safe_team_feature_calculation(ot_games, 'overtime', 'is_win')
            except Exception as e:
                logger.error(f"Error al calcular características para overtime: {str(e)}")
                self.teams_data['overtime_win_rate'] = 0.5
                self.teams_data['avg_points_overtime'] = self.teams_data['PTS'].mean()
        
        logger.info("Características situacionales creadas")
    
    def _create_advanced_differential_features(self):
        """
        Crea características avanzadas de diferenciales
        """
        logger.info("Creando características avanzadas de diferenciales")
        
        # Diferenciales básicos
        self._create_differential_features()
        
        # Diferenciales avanzados
        self.teams_data['ts_pct_diff'] = (
            self.teams_data['PTS'] / (2 * (self.teams_data['FGA'] + 0.44 * self.teams_data['FTA'])) -
            self.teams_data['PTS_Opp'] / (2 * (self.teams_data['FGA_Opp'] + 0.44 * self.teams_data['FTA_Opp']))
        )
        
        self.teams_data['efg_pct_diff'] = (
            (self.teams_data['FG'] + 0.5 * self.teams_data['3P']) / self.teams_data['FGA'] -
            (self.teams_data['FG_Opp'] + 0.5 * self.teams_data['3P_Opp']) / self.teams_data['FGA_Opp']
        )
        
        self.teams_data['pace_diff'] = (
            (self.teams_data['FGA'] + 0.44 * self.teams_data['FTA']) -
            (self.teams_data['FGA_Opp'] + 0.44 * self.teams_data['FTA_Opp'])
        )
        
        # Diferenciales de tiro por zona
        self.teams_data['paint_points_diff'] = (
            self.teams_data['2P'] * 2 -
            self.teams_data['2P_Opp'] * 2
        )
        
        self.teams_data['three_points_diff'] = (
            self.teams_data['3P'] * 3 -
            self.teams_data['3P_Opp'] * 3
        )
        
        self.teams_data['ft_points_diff'] = (
            self.teams_data['FT'] -
            self.teams_data['FT_Opp']
        )
        
        logger.info("Características avanzadas de diferenciales creadas")
    
    def _create_win_prediction_features(self):
        """
        Crea características específicas para la predicción de victoria
        """
        logger.info("Creando características específicas para predicción de victoria")
        
        try:
            # Asegurar que tenemos un índice numérico
            self.teams_data = self.teams_data.reset_index(drop=True)
            
            # Calcular win_rate para cada equipo si no existe
            if 'is_win' in self.teams_data.columns and 'win_rate' not in self.teams_data.columns:
                self.teams_data['win_rate'] = self.teams_data.groupby('Team')['is_win'].transform('mean')
            
            # Calcular opp_win_rate si no existe
            if 'Opp' in self.teams_data.columns and 'win_rate' in self.teams_data.columns and 'opp_win_rate' not in self.teams_data.columns:
                # Crear un mapeo de equipos a tasas de victoria
                team_win_rates = self.teams_data.groupby('Team')['win_rate'].first().to_dict()
                # Aplicar el mapeo a los oponentes
                self.teams_data['opp_win_rate'] = self.teams_data['Opp'].map(team_win_rates).fillna(0.5)
            
            # Probabilidad de victoria basada en diferencial de ratings
            if 'net_rating' in self.teams_data.columns:
                self.teams_data['win_prob_by_rating'] = 1 / (1 + np.exp(-self.teams_data['net_rating'] / 100))
            
            # Factores de victoria históricos
            if 'PTS_diff' in self.teams_data.columns:
                pts_diff_std = self.teams_data.groupby('Team')['PTS_diff'].transform('std')
                self.teams_data['win_margin_consistency'] = 1 / (pts_diff_std + 1)  # Mayor consistencia = mejor predictor
            
            # Victoria contra equipos mejores/peores
            if all(col in self.teams_data.columns for col in ['win_rate', 'opp_win_rate', 'is_win']):
                self.teams_data['better_team_wins'] = self.teams_data.apply(
                    lambda row: 1 if (row['win_rate'] > row['opp_win_rate'] and row['is_win']) or 
                                    (row['win_rate'] < row['opp_win_rate'] and not row['is_win']) else 0,
                    axis=1
                )
            
            # Rachas específicas por equipo
            all_teams = []
            
            for team in self.teams_data['Team'].unique():
                team_data = self.teams_data[self.teams_data['Team'] == team].copy()
                team_data = team_data.sort_values('Date')
                
                # Procesar solo si tenemos las columnas necesarias
                if all(col in team_data.columns for col in ['Date', 'is_win', 'win_rate', 'opp_win_rate']):
                    for window in self.window_sizes:
                        # Inicializar columnas
                        team_data[f'wins_vs_better_teams_{window}'] = np.nan
                        team_data[f'situation_wins_{window}'] = np.nan
                        
                        # Calcular para cada fila
                        for i in range(len(team_data)):
                            if i == 0:
                                continue
                                
                            # Obtener datos históricos hasta este punto
                            history = team_data.iloc[:i]
                            
                            # Victorias contra equipos con mejor récord
                            try:
                                # Convertir a numérico de forma segura
                                history_win_rate = pd.to_numeric(history['win_rate'], errors='coerce')
                                history_opp_win_rate = pd.to_numeric(history['opp_win_rate'], errors='coerce')
                                
                                # Reemplazar valores problemáticos
                                history_win_rate = history_win_rate.replace([np.inf, -np.inf], np.nan).fillna(0.5)
                                history_opp_win_rate = history_opp_win_rate.replace([np.inf, -np.inf], np.nan).fillna(0.5)
                                
                                # Filtrar equipos con mejor récord con manejo seguro de valores
                                better_teams_mask = history_opp_win_rate > history_win_rate
                                better_teams_history = history[better_teams_mask]
                                
                                if len(better_teams_history) > 0:
                                    # Convertir is_win a numérico para seguridad
                                    is_win_values = pd.to_numeric(better_teams_history['is_win'], errors='coerce')
                                    is_win_values = is_win_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    
                                    # Calcular media y limitar a [0,1]
                                    wins_vs_better = is_win_values.mean()
                                    wins_vs_better = max(0, min(1, wins_vs_better))
                                    
                                    # Asignar valor a la columna
                                    team_data.iloc[i, team_data.columns.get_loc(f'wins_vs_better_teams_{window}')] = wins_vs_better
                                else:
                                    # Si no hay datos de equipos mejores, usar un valor por defecto de 0.5
                                    team_data.iloc[i, team_data.columns.get_loc(f'wins_vs_better_teams_{window}')] = 0.5
                            except Exception as e:
                                logger.error(f"Error al calcular wins_vs_better_teams_{window} para índice {i}: {str(e)}")
                                # Usar valor por defecto en caso de error
                                team_data.iloc[i, team_data.columns.get_loc(f'wins_vs_better_teams_{window}')] = 0.5
                            
                            # Victorias en situaciones similares (home/away)
                            if 'is_home' in team_data.columns:
                                is_home = team_data.iloc[i]['is_home']
                                similar_situations = history[history['is_home'] == is_home]
                                if len(similar_situations) > 0:
                                    situation_wins = similar_situations['is_win'].mean()
                                    team_data.iloc[i, team_data.columns.get_loc(f'situation_wins_{window}')] = situation_wins
                
                # Características de clutch
                if 'PTS_diff' in team_data.columns and 'is_win' in team_data.columns:
                    close_games = abs(team_data['PTS_diff']) <= 5
                    if close_games.any():
                        team_data['clutch_win_rate'] = team_data.loc[close_games, 'is_win'].mean()
                        team_data['clutch_game_frequency'] = close_games.mean()
                
                # Victorias contra tipos específicos de equipos
                if all(col in team_data.columns for col in ['is_win', 'offensive_rating', 'defensive_rating']):
                    try:
                        # Calcular promedios globales de forma segura
                        off_values = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
                        off_values = off_values.replace([np.inf, -np.inf], np.nan).fillna(100.0)
                        avg_off_rating = off_values.mean()
                        
                        def_values = pd.to_numeric(self.teams_data['defensive_rating'], errors='coerce')
                        def_values = def_values.replace([np.inf, -np.inf], np.nan).fillna(100.0)
                        avg_def_rating = def_values.mean()
                        
                        # Inicializar columnas
                        team_data['wins_vs_offensive_teams'] = np.nan
                        team_data['wins_vs_defensive_teams'] = np.nan
                        
                        # Calcular para cada fila
                        for i in range(len(team_data)):
                            if i == 0:
                                # Para la primera fila, usar valores por defecto
                                team_data.iloc[i, team_data.columns.get_loc('wins_vs_offensive_teams')] = 0.5
                                team_data.iloc[i, team_data.columns.get_loc('wins_vs_defensive_teams')] = 0.5
                                continue
                                
                            # Obtener datos históricos hasta este punto
                            history = team_data.iloc[:i]
                            
                            # Victorias contra equipos ofensivos
                            try:
                                # Convertir a numérico de forma segura
                                history_off_rating = pd.to_numeric(history['offensive_rating'], errors='coerce')
                                history_off_rating = history_off_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
                                
                                # Filtrar equipos ofensivos
                                offensive_teams = history[history_off_rating > avg_off_rating]
                                if len(offensive_teams) > 0:
                                    # Calcular tasa de victoria de forma segura
                                    is_win_values = pd.to_numeric(offensive_teams['is_win'], errors='coerce')
                                    is_win_values = is_win_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    win_rate = is_win_values.mean()
                                    win_rate = max(0, min(1, win_rate))
                                    team_data.iloc[i, team_data.columns.get_loc('wins_vs_offensive_teams')] = win_rate
                                else:
                                    team_data.iloc[i, team_data.columns.get_loc('wins_vs_offensive_teams')] = 0.5
                            except Exception as e:
                                logger.error(f"Error al calcular wins_vs_offensive_teams para índice {i}: {str(e)}")
                                team_data.iloc[i, team_data.columns.get_loc('wins_vs_offensive_teams')] = 0.5
                            
                            # Victorias contra equipos defensivos
                            try:
                                # Convertir a numérico de forma segura
                                history_def_rating = pd.to_numeric(history['defensive_rating'], errors='coerce')
                                history_def_rating = history_def_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
                                
                                # Filtrar equipos defensivos (rating defensivo bajo = mejor defensa)
                                defensive_teams = history[history_def_rating < avg_def_rating]
                                if len(defensive_teams) > 0:
                                    # Calcular tasa de victoria de forma segura
                                    is_win_values = pd.to_numeric(defensive_teams['is_win'], errors='coerce')
                                    is_win_values = is_win_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                                    win_rate = is_win_values.mean()
                                    win_rate = max(0, min(1, win_rate))
                                    team_data.iloc[i, team_data.columns.get_loc('wins_vs_defensive_teams')] = win_rate
                                else:
                                    team_data.iloc[i, team_data.columns.get_loc('wins_vs_defensive_teams')] = 0.5
                            except Exception as e:
                                logger.error(f"Error al calcular wins_vs_defensive_teams para índice {i}: {str(e)}")
                                team_data.iloc[i, team_data.columns.get_loc('wins_vs_defensive_teams')] = 0.5
                    except Exception as e:
                        logger.error(f"Error al calcular victorias contra tipos específicos de equipos: {str(e)}")
                        # Crear columnas con valores por defecto en caso de error
                        team_data['wins_vs_offensive_teams'] = 0.5
                        team_data['wins_vs_defensive_teams'] = 0.5
                
                all_teams.append(team_data)
            
            # Combinar todos los equipos
            if all_teams:
                self.teams_data = pd.concat(all_teams).sort_index()
                
                # Desfragmentar
                self.teams_data = self.teams_data.copy()
            
            logger.info("Características de predicción de victoria creadas")
            
        except Exception as e:
            logger.error(f"Error al crear características de predicción de victoria: {str(e)}")
            raise
    
    def _create_total_points_features(self):
        """
        Crea características específicas para la predicción de puntos totales
        """
        logger.info("Creando características específicas para predicción de puntos totales")
        
        # Factores que afectan el total de puntos
        try:
            # Convertir a numérico y manejar valores problemáticos para pace
            pace = pd.to_numeric(self.teams_data['pace'], errors='coerce')
            pace = pace.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            pace = pace.clip(75, 125)
            
            # Calcular pace promedio por oponente de forma segura
            opp_pace = self.teams_data.groupby('Opp')['pace'].transform(
                lambda x: pd.to_numeric(x, errors='coerce')
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(100.0)
                      .clip(75, 125)
                      .mean()
            )
            
            # Calcular combined_pace
            self.teams_data['combined_pace'] = ((pace + opp_pace) / 2).clip(75, 125)
        except Exception as e:
            logger.error(f"Error al calcular combined_pace: {str(e)}")
            self.teams_data['combined_pace'] = 100.0
        
        # Eficiencia combinada
        try:
            # Convertir a numérico y manejar valores problemáticos para offensive_rating
            off_rating = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
            off_rating = off_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            off_rating = off_rating.clip(80, 130)
            
            # Calcular offensive_rating promedio por oponente de forma segura
            opp_off_rating = self.teams_data.groupby('Opp')['offensive_rating'].transform(
                lambda x: pd.to_numeric(x, errors='coerce')
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(100.0)
                      .clip(80, 130)
                      .mean()
            )
            
            # Calcular combined_efficiency
            self.teams_data['combined_efficiency'] = ((off_rating + opp_off_rating) / 2).clip(80, 130)
        except Exception as e:
            logger.error(f"Error al calcular combined_efficiency: {str(e)}")
            self.teams_data['combined_efficiency'] = 100.0
            
        # Variabilidad de puntuación
        for window in self.window_sizes:
            try:
                # Convertir a numérico y manejar valores problemáticos
                total_pts = pd.to_numeric(self.teams_data['total_points'], errors='coerce')
                total_pts = total_pts.replace([np.inf, -np.inf], np.nan).fillna(200.0)  # Valor por defecto razonable
                
                # Limitar a un rango razonable antes de cálculos
                total_pts = total_pts.clip(120, 300)
                
                # Volatilidad de puntuación total con manejo robusto de valores
                self.teams_data[f'total_points_volatility_{window}'] = self.teams_data.groupby('Team')['total_points'].transform(
                    lambda x: pd.to_numeric(x, errors='coerce')
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(200.0)
                          .clip(120, 300)
                          .rolling(window=window, min_periods=1)
                          .std()
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0)
                          .clip(0, 30)  # Limitar volatilidad a un máximo razonable
                )
                
                # Tendencia de puntuación total con manejo robusto de valores
                self.teams_data[f'total_points_trend_{window}'] = self.teams_data.groupby('Team')['total_points'].transform(
                    lambda x: pd.to_numeric(x, errors='coerce')
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(200.0)
                          .clip(120, 300)
                          .rolling(window=window, min_periods=1)
                          .mean()
                          .diff(periods=window)
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0)
                          .clip(-30, 30)  # Limitar tendencia a un rango razonable
                )
                
                logger.debug(f"Calculados volatilidad y tendencia para ventana {window}")
            except Exception as e:
                logger.error(f"Error al calcular volatilidad y tendencia para ventana {window}: {str(e)}")
                # Valores por defecto en caso de error
                self.teams_data[f'total_points_volatility_{window}'] = 0
                self.teams_data[f'total_points_trend_{window}'] = 0
        
        # Características de ritmo de juego
        try:
            # Convertir a numérico y manejar valores problemáticos
            possessions = pd.to_numeric(self.teams_data['possessions'], errors='coerce')
            mp = pd.to_numeric(self.teams_data['MP'], errors='coerce')
            total_points = pd.to_numeric(self.teams_data['total_points'], errors='coerce')
            
            # Reemplazar valores inválidos
            possessions = possessions.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            mp = mp.replace([np.inf, -np.inf], np.nan).fillna(240.0)  # 48 min × 5 jugadores
            total_points = total_points.replace([np.inf, -np.inf], np.nan).fillna(200.0)
            
            # Limitar a rangos razonables
            possessions = possessions.clip(40, 160)
            mp = mp.clip(200, 300)  # Incluye posible tiempo extra
            total_points = total_points.clip(120, 300)
            
            # Calcular métricas evitando división por cero
            self.teams_data['possessions_per_minute'] = (possessions / mp.clip(lower=1)).clip(0.2, 1.0)
            self.teams_data['points_per_possession'] = (total_points / possessions.clip(lower=1)).clip(0.5, 2.0)
        except Exception as e:
            logger.error(f"Error al calcular métricas de ritmo de juego: {str(e)}")
            # Valores por defecto en caso de error
            self.teams_data['possessions_per_minute'] = 0.4  # Valor típico
            self.teams_data['points_per_possession'] = 1.1  # Valor típico
        
        # Eficiencia por tipo de tiro
        try:
            # Convertir a numérico y manejar valores problemáticos
            ts_pct = pd.to_numeric(self.teams_data['ts_pct'], errors='coerce')
            ts_pct = ts_pct.replace([np.inf, -np.inf], np.nan).fillna(0.5)
            ts_pct = ts_pct.clip(0.3, 0.8)
            
            # Calcular TS% promedio por oponente de forma segura
            opp_ts_pct = self.teams_data.groupby('Opp')['ts_pct'].transform(
                lambda x: pd.to_numeric(x, errors='coerce')
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(0.5)
                      .clip(0.3, 0.8)
                      .mean()
            )
            
            # Calcular promedio combinado
            self.teams_data['combined_ts_pct'] = ((ts_pct + opp_ts_pct) / 2).clip(0.3, 0.8)
        except Exception as e:
            logger.error(f"Error al calcular combined_ts_pct: {str(e)}")
            # Valor por defecto en caso de error
            self.teams_data['combined_ts_pct'] = 0.55  # Valor típico
        
        # Factores situacionales que afectan la puntuación
        try:
            # Calcular de forma segura los umbrales
            off_mean = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
            off_mean = off_mean.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            off_mean = off_mean.mean()
            
            def_mean = pd.to_numeric(self.teams_data['defensive_rating'], errors='coerce')
            def_mean = def_mean.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            def_mean = def_mean.mean()
            
            # Preparar ratings para comparación
            off_rating = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
            off_rating = off_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            
            # Calcular rating ofensivo promedio por oponente de forma segura
            opp_off_rating = self.teams_data.groupby('Opp')['offensive_rating'].transform(
                lambda x: pd.to_numeric(x, errors='coerce')
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(100.0)
                      .mean()
            )
            
            # Preparar defensive_rating para comparación
            def_rating = pd.to_numeric(self.teams_data['defensive_rating'], errors='coerce')
            def_rating = def_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            
            # Calcular rating defensivo promedio por oponente de forma segura
            opp_def_rating = self.teams_data.groupby('Opp')['defensive_rating'].transform(
                lambda x: pd.to_numeric(x, errors='coerce')
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(100.0)
                      .mean()
            )
            
            # Crear los flags
            self.teams_data['high_scoring_matchup'] = (
                (off_rating > off_mean) & 
                (opp_off_rating > off_mean)
            ).astype(int)
            
            self.teams_data['defensive_matchup'] = (
                (def_rating < def_mean) & 
                (opp_def_rating < def_mean)
            ).astype(int)
        except Exception as e:
            logger.error(f"Error al calcular flags de matchup: {str(e)}")
            # Valores por defecto en caso de error
            self.teams_data['high_scoring_matchup'] = 0
            self.teams_data['defensive_matchup'] = 0
        
        logger.info("Características de predicción de puntos totales creadas")
        
    def _create_team_points_features(self):
        """
        Crea características específicas para la predicción de puntos por equipo
        """
        logger.info("Creando características específicas para predicción de puntos por equipo")
        
        try:
            # Asegurar que tenemos un índice numérico
            self.teams_data = self.teams_data.reset_index(drop=True)
            
            # Procesar cada equipo por separado
            all_teams = []
            
            for team in self.teams_data['Team'].unique():
                try:
                    team_data = self.teams_data[self.teams_data['Team'] == team].copy()
                    team_data = team_data.sort_values('Date')
                    
                    # Eficiencia ofensiva ajustada por oponente
                    if all(col in team_data.columns for col in ['offensive_rating', 'defensive_rating']):
                        try:
                            # Calcular rating defensivo promedio de manera segura
                            avg_def_rating = self.teams_data['defensive_rating'].mean()
                            if pd.isna(avg_def_rating) or np.isinf(avg_def_rating):
                                avg_def_rating = 100.0
                                
                            # Calcular eficiencia ajustada de manera segura
                            def_ratio = team_data['defensive_rating'] / avg_def_rating
                            def_ratio = def_ratio.clip(0.5, 2.0).fillna(1.0)  # Limitar rango y manejar nulos
                            
                            team_data['adj_offensive_rating'] = team_data['offensive_rating'] / def_ratio
                            team_data['adj_offensive_rating'] = team_data['adj_offensive_rating'].clip(50, 150).fillna(100)
                            
                            logger.debug(f"Calculado adj_offensive_rating para equipo {team}")
                        except Exception as e:
                            logger.error(f"Error al calcular adj_offensive_rating para {team}: {str(e)}")
                            team_data['adj_offensive_rating'] = team_data['offensive_rating'].fillna(100)
                    
                    # Características de matchup específicas
                    if all(col in team_data.columns for col in ['offensive_rating', 'defensive_rating']):
                        try:
                            # 1. Calcular matchup_scoring_expectation de forma robusta
                            # Expectativa basada en ratings (evitar divisiones por cero)
                            off_rating = pd.to_numeric(team_data['offensive_rating'], errors='coerce').fillna(100)
                            def_rating = pd.to_numeric(team_data['defensive_rating'], errors='coerce').fillna(100)
                            
                            # Usar un factor de escala en lugar de división directa
                            expectation = off_rating * (100 / def_rating.clip(50, 150))
                            
                            # Limitar a valores razonables
                            team_data['matchup_scoring_expectation'] = expectation.clip(50, 150).fillna(100)
                            
                            # 2. Calcular opp_defensive_rating de forma robusta
                            # Crear un diccionario de ratings defensivos medios por equipo
                            team_def_ratings = {}
                            for t in self.teams_data['Team'].unique():
                                team_def = self.teams_data[self.teams_data['Team'] == t]['defensive_rating']
                                # Calcular media de forma segura
                                if not team_def.empty and not team_def.isnull().all():
                                    # Filtrar valores infinitos y NaN
                                    valid_ratings = team_def[~team_def.isnull() & ~np.isinf(team_def)]
                                    if not valid_ratings.empty:
                                        team_def_ratings[t] = valid_ratings.mean()
                                    else:
                                        team_def_ratings[t] = 100.0
                                else:
                                    team_def_ratings[t] = 100.0
                            
                            # Mapear ratings defensivos a oponentes
                            team_data['opp_defensive_rating'] = team_data['Opp'].map(
                                lambda x: team_def_ratings.get(x, 100.0)
                            )
                            
                            # Calcular ventaja de puntuación de forma segura
                            try:
                                # Convertir a numérico de forma segura
                                off_rating = pd.to_numeric(team_data['offensive_rating'], errors='coerce')
                                def_rating = pd.to_numeric(team_data['opp_defensive_rating'], errors='coerce')
                                
                                # Reemplazar valores problemáticos
                                off_rating = off_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
                                def_rating = def_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
                                
                                # Calcular la diferencia y limitar a un rango razonable
                                team_data['scoring_advantage'] = (off_rating - def_rating).clip(-30, 30)
                            except Exception as e:
                                logger.error(f"Error al calcular scoring_advantage: {str(e)}")
                                team_data['scoring_advantage'] = 0.0
                            
                            logger.debug(f"Calculadas características de matchup para equipo {team}")
                        except Exception as e:
                            logger.error(f"Error al calcular características de matchup para {team}: {str(e)}")
                            # Crear columnas con valores por defecto en caso de error
                            team_data['matchup_scoring_expectation'] = 100.0
                            team_data['opp_defensive_rating'] = 100.0
                            team_data['scoring_advantage'] = 0.0
                    
                    all_teams.append(team_data)
                except Exception as e:
                    logger.error(f"Error al procesar equipo {team}: {str(e)}")
                    # Añadir el equipo sin modificaciones para no perder datos
                    all_teams.append(self.teams_data[self.teams_data['Team'] == team])
            
            # Combinar todos los equipos
            if all_teams:
                self.teams_data = pd.concat(all_teams).sort_index()
                
                # Desfragmentar
                self.teams_data = self.teams_data.copy()
                
                logger.info("Características de predicción de puntos por equipo creadas")
            
        except Exception as e:
            logger.error(f"Error al crear características de predicción de puntos por equipo: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
    
    def _create_advanced_interaction_features(self):
        """
        Crea características de interacción avanzadas entre diferentes métricas
        """
        logger.info("Creando características de interacción avanzadas")
        
        # Interacciones de eficiencia
        try:
            # Convertir a numérico de forma segura
            off_rating = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
            def_rating = pd.to_numeric(self.teams_data['defensive_rating'], errors='coerce')
            
            # Reemplazar valores problemáticos
            off_rating = off_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            def_rating = def_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            
            # Limitar a rangos razonables antes de multiplicar
            off_rating = off_rating.clip(80, 130)
            def_rating = def_rating.clip(80, 130)
            
            # Calcular la interacción y normalizar
            self.teams_data['off_def_rating_interaction'] = (off_rating * def_rating) / 10000
            
            # Limitar el resultado final a un rango razonable
            self.teams_data['off_def_rating_interaction'] = self.teams_data['off_def_rating_interaction'].clip(0.64, 1.69)
        except Exception as e:
            logger.error(f"Error al calcular off_def_rating_interaction: {str(e)}")
            # Valor por defecto en caso de error
            self.teams_data['off_def_rating_interaction'] = 1.0
        
        # Interacciones de tiro
        try:
            # Convertir a numérico y manejar valores problemáticos
            ts_pct = pd.to_numeric(self.teams_data['ts_pct'], errors='coerce')
            efg_pct = pd.to_numeric(self.teams_data['efg_pct'], errors='coerce')
            
            # Reemplazar valores inválidos
            ts_pct = ts_pct.replace([np.inf, -np.inf], np.nan).fillna(0.5)
            efg_pct = efg_pct.replace([np.inf, -np.inf], np.nan).fillna(0.45)
            
            # Limitar a rangos razonables antes de multiplicar (0-1 para porcentajes)
            ts_pct = ts_pct.clip(0.3, 0.8)
            efg_pct = efg_pct.clip(0.3, 0.7)
            
            # Calcular la interacción
            self.teams_data['shooting_efficiency_interaction'] = ts_pct * efg_pct
            
            # Limitar el resultado final a un rango razonable (producto de dos valores entre 0-1)
            self.teams_data['shooting_efficiency_interaction'] = self.teams_data['shooting_efficiency_interaction'].clip(0.09, 0.56)
        except Exception as e:
            logger.error(f"Error al calcular shooting_efficiency_interaction: {str(e)}")
            # Valor por defecto en caso de error (valor medio aproximado)
            self.teams_data['shooting_efficiency_interaction'] = 0.25
        
        # Interacciones de ritmo y eficiencia
        try:
            # Convertir a numérico y manejar valores problemáticos
            pace = pd.to_numeric(self.teams_data['pace'], errors='coerce')
            offensive_rating = pd.to_numeric(self.teams_data['offensive_rating'], errors='coerce')
            
            # Reemplazar valores inválidos
            pace = pace.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            offensive_rating = offensive_rating.replace([np.inf, -np.inf], np.nan).fillna(100.0)
            
            # Limitar a rangos razonables antes de multiplicar
            pace = pace.clip(50, 150)
            offensive_rating = offensive_rating.clip(80, 130)
            
            # Calcular la interacción y normalizar
            self.teams_data['pace_efficiency_interaction'] = (pace * offensive_rating) / 100
            
            # Limitar el resultado final a un rango razonable
            self.teams_data['pace_efficiency_interaction'] = self.teams_data['pace_efficiency_interaction'].clip(40, 200)
        except Exception as e:
            logger.error(f"Error al calcular pace_efficiency_interaction: {str(e)}")
            # Valor por defecto en caso de error
            self.teams_data['pace_efficiency_interaction'] = 100.0
        
        # Interacciones situacionales
        try:
            # Convertir a numérico y manejar valores problemáticos
            days_rest = pd.to_numeric(self.teams_data['days_rest'], errors='coerce')
            win_rate = pd.to_numeric(self.teams_data['win_rate'], errors='coerce')
            
            # Reemplazar valores inválidos
            days_rest = days_rest.replace([np.inf, -np.inf], np.nan).fillna(1)
            win_rate = win_rate.replace([np.inf, -np.inf], np.nan).fillna(0.5)
            
            # Limitar a rangos razonables antes de multiplicar
            days_rest = days_rest.clip(0, 7)  # Máximo 7 días de descanso
            win_rate = win_rate.clip(0.1, 0.9)  # Evitar extremos en win_rate
            
            # Calcular la interacción completamente individualizada por equipo
            rest_perf = {}
            
            # Para cada equipo, calcular una interacción específica con variabilidad
            teams = self.teams_data['Team'].unique()
            for team in teams:
                team_mask = self.teams_data['Team'] == team
                team_data = self.teams_data[team_mask].copy()
                
                # Ordenar por fecha para asegurar secuencia temporal correcta
                if 'Date' in team_data.columns:
                    team_data = team_data.sort_values('Date')
                
                # Obtener datos específicos para este equipo
                team_days_rest = pd.to_numeric(team_data['days_rest'], errors='coerce').fillna(1).clip(0, 7)
                team_win_rate = pd.to_numeric(team_data['win_rate'], errors='coerce').fillna(0.5).clip(0.1, 0.9)
                
                # Calcular promedios específicos de victorias por días de descanso
                team_rest_stats = {}
                
                # Recopilar estadísticas históricas de rendimiento por días de descanso
                for rest_day in range(4):  # 0-3 días de descanso
                    rest_col = f'win_rate_rest_{rest_day}'
                    if rest_col in team_data.columns:
                        # Tomar media específica para este equipo y nivel de descanso
                        valid_vals = pd.to_numeric(team_data[rest_col], errors='coerce')
                        valid_vals = valid_vals[~valid_vals.isna() & ~np.isinf(valid_vals)]
                        if len(valid_vals) > 0:
                            team_rest_stats[rest_day] = valid_vals.mean()
                
                # Calcular factores de ajuste específicos para este equipo
                # Esto crea una función personalizada que mapea días de descanso a rendimiento
                
                # 1. Si tenemos estadísticas de rendimiento por días de descanso, usar para crear curva
                if len(team_rest_stats) >= 2:
                    # Normalizar estadísticas al rango [0.8, 1.2] para factorizar win_rate
                    rest_factors = {}
                    mean_win_rate = team_win_rate.mean()
                    if mean_win_rate == 0:
                        mean_win_rate = 0.5  # Evitar división por cero
                    
                    for rest_day, win_rate_val in team_rest_stats.items():
                        # Calcular factor relativo al promedio
                        if mean_win_rate > 0:
                            factor = win_rate_val / mean_win_rate
                            # Limitar a un rango razonable
                            factor = max(0.8, min(1.2, factor))
                            rest_factors[rest_day] = factor
                        else:
                            rest_factors[rest_day] = 1.0
                    
                    # 2. Aplicar factor específico a cada registro según días descanso
                    for i, idx in enumerate(team_data.index):
                        current_rest = team_days_rest.iloc[i]
                        current_win_rate = team_win_rate.iloc[i]
                        
                        # Redondear a días enteros para aplicar factores
                        rest_int = min(3, max(0, int(round(current_rest))))
                        
                        # Si tenemos factor para esos días, aplicarlo
                        if rest_int in rest_factors:
                            factor = rest_factors[rest_int]
                        else:
                            # Interpolar el factor si no tenemos valor exacto
                            available_days = sorted(list(rest_factors.keys()))
                            if not available_days:
                                factor = 1.0  # Caso extremo, no hay datos
                            elif rest_int < min(available_days):
                                factor = rest_factors[min(available_days)]  # Usar menor disponible
                            elif rest_int > max(available_days):
                                factor = rest_factors[max(available_days)]  # Usar mayor disponible
                            else:
                                # Interpolar entre valores disponibles
                                lower_day = max([d for d in available_days if d <= rest_int])
                                upper_day = min([d for d in available_days if d >= rest_int])
                                if lower_day == upper_day:
                                    factor = rest_factors[lower_day]
                                else:
                                    # Interpolación lineal
                                    weight = (rest_int - lower_day) / (upper_day - lower_day)
                                    factor = (1 - weight) * rest_factors[lower_day] + weight * rest_factors[upper_day]
                        
                        # Añadir componente de variabilidad basado en tendencia
                        if i > 0:
                            # Variación basada en índice para crear tendencia
                            variation = 0.05 * np.sin(i / 5)  # Seno para crear ondulación
                        else:
                            variation = 0
                        
                        # Aplicar el factor y variación al win_rate
                        adjusted_rate = current_win_rate * factor * (1 + variation)
                        
                        # Limitar al rango [0,1]
                        adjusted_rate = max(0, min(1, adjusted_rate))
                        
                        # Guardar el valor final
                        rest_perf[idx] = adjusted_rate
                else:
                    # Si no tenemos suficientes datos históricos, usar fórmula base con variación
                    for i, idx in enumerate(team_data.index):
                        current_rest = team_days_rest.iloc[i]
                        current_win_rate = team_win_rate.iloc[i]
                        
                        # Normalizar los días de descanso [0,1]
                        normalized_rest = current_rest / 7
                        
                        # Peso base para días de descanso: 0.8 a 1.0
                        weight = 0.8 + 0.2 * normalized_rest
                        
                        # Añadir variación temporal para evitar constantes
                        if i > 0:
                            variation = 0.05 * np.sin(i / 5)  # Variación sinusoidal
                        else:
                            variation = 0
                            
                        # Aplicar peso y variación
                        adjusted_rate = current_win_rate * weight * (1 + variation)
                        
                        # Limitar al rango [0,1]
                        adjusted_rate = max(0, min(1, adjusted_rate))
                        
                        # Guardar el valor final
                        rest_perf[idx] = adjusted_rate
            
            # Convertir el diccionario a Series y asignar al DataFrame
            if rest_perf:
                self.teams_data['rest_performance_interaction'] = pd.Series(rest_perf)
                
                # Verificar valores faltantes (por si hay equipos omitidos)
                missing_mask = self.teams_data['rest_performance_interaction'].isnull()
                if missing_mask.any():
                    logger.warning(f"Faltan {missing_mask.sum()} valores en rest_performance_interaction. Rellenando.")
                    
                    # Usar cálculo base para rellenar faltantes
                    normalized_days_rest = days_rest / 7  # Normalizar a [0,1]
                    weight = 0.8 + 0.2 * normalized_days_rest  # Rango [0.8, 1.0]
                    # La interacción siempre estará en [0,1]
                    missing_values = win_rate * weight
                    self.teams_data.loc[missing_mask, 'rest_performance_interaction'] = missing_values[missing_mask]
                
                # Clip final para asegurar rango [0,1]
                self.teams_data['rest_performance_interaction'] = self.teams_data['rest_performance_interaction'].clip(0, 1)
                
                # Verificar que no hay valores constantes por equipo
                constant_teams = []
                for team in teams:
                    team_vals = self.teams_data.loc[self.teams_data['Team'] == team, 'rest_performance_interaction']
                    if len(team_vals) > 5 and team_vals.nunique() == 1:
                        constant_teams.append(team)
                
                # Si hay equipos con valores constantes, introducir variabilidad
                if constant_teams:
                    logger.warning(f"Detectados {len(constant_teams)} equipos con valores constantes en rest_performance_interaction. Corrigiendo.")
                    for team in constant_teams:
                        team_mask = self.teams_data['Team'] == team
                        team_size = team_mask.sum()
                        # Tomar el valor constante
                        constant_val = self.teams_data.loc[team_mask, 'rest_performance_interaction'].iloc[0]
                        # Generar valores con variación usando seno y ruido aleatorio pequeño
                        indices = np.arange(team_size)
                        # Variación sinusoidal con un poco de ruido
                        variation = 0.08 * np.sin(indices / 5) + 0.02 * np.random.randn(team_size)
                        # Aplicar variación al valor constante
                        new_values = constant_val * (1 + variation)
                        # Limitar al rango [0,1]
                        new_values = np.clip(new_values, 0, 1)
                        # Asignar al DataFrame
                        self.teams_data.loc[team_mask, 'rest_performance_interaction'] = new_values
            else:
                # Cálculo alternativo como fallback con variabilidad
                normalized_days_rest = days_rest / 7  # Normalizar a [0,1]
                weight = 0.8 + 0.2 * normalized_days_rest  # Rango [0.8, 1.0]
                
                # Añadir variabilidad por equipo
                result_values = []
                for team in teams:
                    team_mask = self.teams_data['Team'] == team
                    team_size = team_mask.sum()
                    team_win_rate = win_rate[team_mask]
                    team_weight = weight[team_mask]
                    
                    # Índices para este equipo
                    indices = np.arange(team_size)
                    # Variación sinusoidal
                    variation = 0.08 * np.sin(indices / 5) + 0.02 * np.random.randn(team_size)
                    
                    # Calcular valores con variación
                    team_values = team_win_rate * team_weight * (1 + variation)
                    team_values = np.clip(team_values, 0, 1)  # Limitar a [0,1]
                    
                    # Agregar valores a la lista
                    for idx, val in zip(self.teams_data[team_mask].index, team_values):
                        result_values.append((idx, val))
                
                # Crear Series y asignar al DataFrame
                if result_values:
                    rest_perf_series = pd.Series({idx: val for idx, val in result_values})
                    self.teams_data['rest_performance_interaction'] = rest_perf_series
                else:
                    # Último recurso
                    self.teams_data['rest_performance_interaction'] = (win_rate * weight).clip(0, 1)
        except Exception as e:
            logger.error(f"Error al calcular rest_performance_interaction: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            # Valor por defecto en caso de error (equilibrado)
            self.teams_data['rest_performance_interaction'] = 0.5
        
        # Interacciones de momentum
        for window in self.window_sizes:
            momentum_col = f'offensive_momentum_{window}'
            def_momentum_col = f'defensive_momentum_{window}'
            interaction_col = f'momentum_interaction_{window}'
            
            if momentum_col in self.teams_data.columns and def_momentum_col in self.teams_data.columns:
                try:
                    # Convertir a numérico y manejar valores problemáticos
                    off_momentum = pd.to_numeric(self.teams_data[momentum_col], errors='coerce')
                    def_momentum = pd.to_numeric(self.teams_data[def_momentum_col], errors='coerce')
                    
                    # Reemplazar valores inválidos
                    off_momentum = off_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
                    def_momentum = def_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Limitar a rangos razonables antes de la operación
                    off_momentum = off_momentum.clip(-20, 20)
                    def_momentum = def_momentum.clip(-20, 20)
                    
                    # Calcular la interacción (usando negativo de defensive_momentum)
                    self.teams_data[interaction_col] = off_momentum * (-def_momentum)
                    
                    # Limitar el resultado final a un rango razonable
                    self.teams_data[interaction_col] = self.teams_data[interaction_col].clip(-200, 200)
                except Exception as e:
                    logger.error(f"Error al calcular {interaction_col}: {str(e)}")
                    # Valor por defecto en caso de error
                    self.teams_data[interaction_col] = 0.0
        
        logger.info("Características de interacción avanzadas creadas")
        
    def _create_advanced_time_series_features(self):
        """
        Crea características avanzadas de series temporales
        """
        logger.info("Creando características avanzadas de series temporales")
        
        # Características de tendencia exponencial
        alpha = 0.1  # Factor de suavizado
        for col in ['PTS', 'PTS_Opp', 'offensive_rating', 'defensive_rating']:
            if col in self.teams_data.columns:
                try:
                    # Preparar la serie para EMA y EMV
                    series = pd.to_numeric(self.teams_data[col], errors='coerce')
                    
                    # Reemplazar infinitos y NaN por valores razonables según la columna
                    if col in ['offensive_rating', 'defensive_rating']:
                        default_value = 100.0
                        valid_range = (80, 130)
                    elif col in ['PTS', 'PTS_Opp']:
                        default_value = series.mean() if not series.isnull().all() else 100.0
                        valid_range = (70, 150)
                    else:
                        default_value = 0.0
                        valid_range = (-50, 50)
                    
                    # Limpiar valores
                    series = series.replace([np.inf, -np.inf], np.nan).fillna(default_value)
                    series = series.clip(valid_range[0], valid_range[1])
                    
                    # Aplicar EMA (Exponential Moving Average) por equipo
                    self.teams_data[f'{col}_ema'] = self.teams_data.groupby('Team')[col].transform(
                        lambda x: pd.to_numeric(x, errors='coerce')
                              .replace([np.inf, -np.inf], np.nan)
                              .ffill()  # Reemplazar fillna(method='ffill')
                              .fillna(default_value)
                              .ewm(alpha=alpha, ignore_na=True)
                              .mean()
                    )
                    
                    # Aplicar EMV (Exponential Moving Volatility) por equipo
                    self.teams_data[f'{col}_emv'] = self.teams_data.groupby('Team')[col].transform(
                        lambda x: pd.to_numeric(x, errors='coerce')
                              .replace([np.inf, -np.inf], np.nan)
                              .ffill()  # Reemplazar fillna(method='ffill')
                              .fillna(default_value)
                              .ewm(alpha=alpha, ignore_na=True)
                              .std()
                    )
                    
                    # Asegurar que no hay valores NaN o infinitos en el resultado
                    self.teams_data[f'{col}_ema'] = self.teams_data[f'{col}_ema'].replace(
                        [np.inf, -np.inf], np.nan).fillna(default_value)
                    self.teams_data[f'{col}_emv'] = self.teams_data[f'{col}_emv'].replace(
                        [np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Limitar a rangos razonables
                    self.teams_data[f'{col}_ema'] = self.teams_data[f'{col}_ema'].clip(valid_range[0], valid_range[1])
                    self.teams_data[f'{col}_emv'] = self.teams_data[f'{col}_emv'].clip(0, 30)
                    
                    logger.debug(f"Calculado EMA y EMV para {col}")
                except Exception as e:
                    logger.error(f"Error al calcular EMA y EMV para {col}: {str(e)}")
                    # Usar valores por defecto en caso de error
                    self.teams_data[f'{col}_ema'] = default_value
                    self.teams_data[f'{col}_emv'] = 0
        
        # Características de cambio de tendencia
        for window in self.window_sizes:
            # Detectar cambios de tendencia usando diferencias
            for col in ['PTS', 'PTS_Opp', 'offensive_rating', 'defensive_rating']:
                if col in self.teams_data.columns:
                    trend_col = f'{col}_trend_{window}'
                    if trend_col in self.teams_data.columns:
                        try:
                            # Preparar la columna de tendencia
                            trend_series = pd.to_numeric(self.teams_data[trend_col], errors='coerce')
                            trend_series = trend_series.replace([np.inf, -np.inf], np.nan).fillna(0)
                            
                            # Cambio en la tendencia (diferencia de primer orden)
                            self.teams_data[f'{col}_trend_change_{window}'] = self.teams_data.groupby('Team')[trend_col].transform(
                                lambda x: pd.to_numeric(x, errors='coerce')
                                      .replace([np.inf, -np.inf], np.nan)
                                      .ffill()  # Reemplazar fillna(method='ffill')
                                      .fillna(0)
                                      .diff()
                                      .fillna(0)
                            )
                            
                            # Aceleración de la tendencia (diferencia de segundo orden)
                            change_col = f'{col}_trend_change_{window}'
                            self.teams_data[f'{col}_trend_acceleration_{window}'] = self.teams_data.groupby('Team')[change_col].transform(
                                lambda x: pd.to_numeric(x, errors='coerce')
                                      .replace([np.inf, -np.inf], np.nan)
                                      .ffill()  # Reemplazar fillna(method='ffill')
                                      .fillna(0)
                                      .diff()
                                      .fillna(0)
                            )
                            
                            # Limitar a rangos razonables
                            self.teams_data[f'{col}_trend_change_{window}'] = self.teams_data[f'{col}_trend_change_{window}'].clip(-20, 20)
                            self.teams_data[f'{col}_trend_acceleration_{window}'] = self.teams_data[f'{col}_trend_acceleration_{window}'].clip(-10, 10)
                            
                            logger.debug(f"Calculados cambios de tendencia para {col} con ventana {window}")
                        except Exception as e:
                            logger.error(f"Error al calcular cambios de tendencia para {col} con ventana {window}: {str(e)}")
                            # Valores por defecto en caso de error
                            self.teams_data[f'{col}_trend_change_{window}'] = 0
                            self.teams_data[f'{col}_trend_acceleration_{window}'] = 0
        
        logger.info("Características avanzadas de series temporales creadas")
    
    def generate_features(self, save_path=None):
        """
        Genera todas las características y devuelve el DataFrame final
        """
        logger.info("Iniciando proceso de generación de características")
        
        try:
            # Preprocesamiento inicial y características temporales
            self._preprocess_data()
            logger.debug("Preprocesamiento completado")
            
            self._create_temporal_features()
            logger.debug("Características temporales creadas")
            
            # Crear características básicas requeridas primero
            self._create_advanced_efficiency_features()  # Esto crea defensive_rating
            logger.debug("Características de eficiencia creadas")
            self.teams_data = self.teams_data.copy()
            
            # Resto de características
            self.create_rolling_features()
            logger.debug("Características rolling creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_advanced_momentum_features()
            logger.debug("Características de momentum creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_advanced_matchup_features()
            logger.debug("Características de matchup creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_advanced_situational_features()
            logger.debug("Características situacionales creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_advanced_differential_features()
            logger.debug("Características diferenciales creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_win_prediction_features()
            logger.debug("Características de predicción de victoria creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_total_points_features()
            logger.debug("Características de puntos totales creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_team_points_features()
            logger.debug("Características de puntos por equipo creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_advanced_interaction_features()
            logger.debug("Características de interacción creadas")
            self.teams_data = self.teams_data.copy()
            
            self._create_advanced_time_series_features()
            logger.debug("Características de series temporales creadas")
            self.teams_data = self.teams_data.copy()
            
            self.create_opponent_features()
            logger.debug("Características de oponentes creadas")
            self.teams_data = self.teams_data.copy()
            
            self.create_betting_features()
            logger.debug("Características de apuestas creadas")
            self.teams_data = self.teams_data.copy()
            
            self.create_overtime_features()
            logger.debug("Características de tiempo extra creadas")
            self.teams_data = self.teams_data.copy()
            
            # Manejar valores NaN
            logger.info(f"Manejando valores NaN. Forma antes: {self.teams_data.shape}")
            
            # Asegurar que el índice es numérico y consecutivo
            self.teams_data = self.teams_data.reset_index(drop=True)
            logger.debug(f"Índice reseteado. Forma: {self.teams_data.shape}")
            
            # Verificar columnas con muchos NaN
            try:
                null_percentages = self.teams_data.isnull().mean() * 100
                high_null_cols = null_percentages[null_percentages > 50].index.tolist()
                
                # Asegurar que no eliminamos características requeridas
                required_cols = ['Team', 'Date', 'PTS', 'PTS_Opp', 'total_points', 
                                'win_probability', 'offensive_rating', 'defensive_rating']
                high_null_cols = [col for col in high_null_cols if col not in required_cols]
                
                if high_null_cols:
                    logger.warning(f"Eliminando {len(high_null_cols)} columnas con >50% valores nulos")
                    # Verificar que estas columnas existen antes de eliminarlas
                    missing_cols = [col for col in high_null_cols if col not in self.teams_data.columns]
                    if missing_cols:
                        logger.error(f"Columnas a eliminar que no existen: {missing_cols}")
                        high_null_cols = [col for col in high_null_cols if col in self.teams_data.columns]
                    
                    # Eliminar columnas una por una para evitar errores
                    for col in high_null_cols:
                        try:
                            logger.debug(f"Eliminando columna con muchos nulos: {col}")
                            self.teams_data = self.teams_data.drop(columns=[col])
                        except Exception as e:
                            logger.error(f"Error al eliminar columna {col}: {str(e)}")
                    
                    logger.debug(f"Columnas con muchos nulos eliminadas. Forma: {self.teams_data.shape}")
            except Exception as e:
                logger.error(f"Error al procesar columnas con muchos nulos: {str(e)}")
                logger.error(f"Traza de error: {traceback.format_exc()}")
            
            # Rellenar valores NaN restantes
            numeric_cols = self.teams_data.select_dtypes(include=[np.number]).columns
            logger.debug(f"Columnas numéricas encontradas: {len(numeric_cols)}")
            
            # Imprimir información sobre valores NaN en columnas numéricas
            null_counts = self.teams_data[numeric_cols].isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if not cols_with_nulls.empty:
                logger.debug(f"Columnas numéricas con NaN antes de rellenar: {cols_with_nulls.to_dict()}")
            
            # Rellenar valores NaN columna por columna en lugar de todas a la vez
            logger.debug("Rellenando valores NaN columna por columna")
            for col in numeric_cols:
                try:
                    if self.teams_data[col].isnull().any():
                        logger.debug(f"Rellenando NaN en columna: {col}")
                        self.teams_data[col] = self.teams_data[col].fillna(0)
                except Exception as e:
                    logger.error(f"Error al rellenar columna {col}: {str(e)}")
                    # Intentar eliminar la columna problemática
                    try:
                        logger.warning(f"Eliminando columna problemática: {col}")
                        self.teams_data = self.teams_data.drop(columns=[col])
                    except Exception as drop_error:
                        logger.error(f"Error al eliminar columna {col}: {str(drop_error)}")
            
            logger.debug("Valores NaN en columnas numéricas rellenados")
            
            # Rellenar valores no numéricos
            non_numeric_cols = self.teams_data.select_dtypes(exclude=[np.number]).columns
            logger.debug(f"Columnas no numéricas encontradas: {len(non_numeric_cols)}")
            
            for col in non_numeric_cols:
                try:
                    if self.teams_data[col].isnull().any():
                        default_value = self.teams_data[col].mode()[0] if not self.teams_data[col].mode().empty else ""
                        logger.debug(f"Rellenando NaN en columna no numérica: {col}")
                        self.teams_data[col] = self.teams_data[col].fillna(default_value)
                except Exception as e:
                    logger.error(f"Error al rellenar columna no numérica {col}: {str(e)}")
                    # Intentar eliminar la columna problemática
                    try:
                        logger.warning(f"Eliminando columna problemática: {col}")
                        self.teams_data = self.teams_data.drop(columns=[col])
                    except Exception as drop_error:
                        logger.error(f"Error al eliminar columna {col}: {str(drop_error)}")
            
            logger.debug("Valores NaN en columnas no numéricas rellenados")
            
            # Asegurar que no hay valores NaN
            if self.teams_data.isnull().any().any():
                logger.warning("Todavía hay valores NaN después de la imputación. Rellenando con 0.")
                # Identificar columnas con NaN restantes
                null_counts = self.teams_data.isnull().sum()
                cols_with_nulls = null_counts[null_counts > 0]
                logger.debug(f"Columnas con NaN restantes: {cols_with_nulls.to_dict()}")
                
                # Rellenar todos los NaN restantes
                self.teams_data = self.teams_data.fillna(0)
                logger.debug("Todos los valores NaN rellenados")
            
            # Asegurar que el índice está bien
            self.teams_data = self.teams_data.reset_index(drop=True)
            logger.debug(f"Índice reseteado nuevamente. Forma: {self.teams_data.shape}")
            
            logger.info(f"Valores NaN manejados. Forma después: {self.teams_data.shape}")
            
            # Filtrar características correlacionadas si está habilitado
            if self.enable_correlation_analysis:
                logger.debug("Iniciando análisis de correlación")
                try:
                    # Seleccionar solo columnas numéricas excepto las requeridas
                    required_cols = ['Team', 'Date', 'PTS', 'PTS_Opp', 'total_points', 
                                   'win_probability', 'offensive_rating', 'defensive_rating']
                    numeric_cols = [col for col in self.teams_data.select_dtypes(include=[np.number]).columns 
                                if col not in required_cols]
                    logger.debug(f"Columnas numéricas para análisis de correlación: {len(numeric_cols)}")
                    
                    if numeric_cols:
                        # Crear una copia del DataFrame original
                        original_df = self.teams_data.copy()
                        logger.debug(f"DataFrame original copiado. Forma: {original_df.shape}")
                        
                        # Extraer datos numéricos para filtrar correlación
                        numeric_data = original_df[numeric_cols].copy()
                        logger.debug(f"Datos numéricos extraídos. Forma: {numeric_data.shape}")
                        
                        # Verificar si hay valores problemáticos
                        inf_check = np.isinf(numeric_data).any().any()
                        nan_check = numeric_data.isnull().any().any()
                        if inf_check or nan_check:
                            logger.warning(f"Datos numéricos contienen infinitos: {inf_check} o NaN: {nan_check}")
                            # Limpiar datos
                            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(0)
                            logger.debug("Datos numéricos limpiados")
                        
                        # Filtrar características correlacionadas
                        logger.debug("Llamando a filter_correlated_features")
                        filtered_numeric_data = self.filter_correlated_features(numeric_data)
                        logger.debug(f"Datos filtrados recibidos. Forma: {filtered_numeric_data.shape}")
                        
                        # Mantener columnas no numéricas y requeridas
                        kept_cols = [col for col in original_df.columns if col not in numeric_cols]
                        logger.debug(f"Columnas a mantener: {len(kept_cols)}")
                        
                        # Construir el nuevo DataFrame paso a paso
                        result_parts = []
                        
                        # Agregar columnas mantenidas
                        if kept_cols:
                            kept_df = original_df[kept_cols]
                            logger.debug(f"DataFrame de columnas mantenidas. Forma: {kept_df.shape}")
                            result_parts.append(kept_df)
                            
                        # Agregar columnas numéricas filtradas
                        if not filtered_numeric_data.empty:
                            # Asegurar que los índices coinciden
                            if len(filtered_numeric_data) != len(original_df):
                                logger.error(f"Error de longitud en filtered_numeric_data: {len(filtered_numeric_data)} vs {len(original_df)}")
                                # Si hay error de longitud, usar el DataFrame original
                                self.teams_data = original_df
                                logger.debug("Usando DataFrame original debido a error de longitud")
                                return self.teams_data
                            
                            logger.debug(f"Agregando datos numéricos filtrados. Forma: {filtered_numeric_data.shape}")
                            result_parts.append(filtered_numeric_data)
                        
                        # Concatenar todas las partes si hay al menos una
                        if result_parts:
                            logger.debug(f"Concatenando {len(result_parts)} partes")
                            
                            # Verificar índices antes de concatenar
                            for i, part in enumerate(result_parts):
                                logger.debug(f"Parte {i}: Forma {part.shape}, Índice min {part.index.min()}, max {part.index.max()}")
                            
                            try:
                                # Usar concat con índices explícitamente reseteados
                                parts_reset = [part.reset_index(drop=True) for part in result_parts]
                                result_df = pd.concat(parts_reset, axis=1)
                                logger.debug(f"Concatenación exitosa. Forma: {result_df.shape}")
                            except Exception as e:
                                logger.error(f"Error durante la concatenación: {str(e)}")
                                logger.debug("Usando DataFrame original debido a error en concatenación")
                                self.teams_data = original_df
                                return self.teams_data
                            
                            # Verificar que el resultado tiene las dimensiones correctas
                            if len(result_df) != len(original_df):
                                logger.error(f"Error de longitud en el resultado final: {len(result_df)} vs {len(original_df)}")
                                # Si hay error, usar el DataFrame original
                                self.teams_data = original_df
                                logger.debug("Usando DataFrame original debido a error de longitud en el resultado")
                            else:
                                # Todo bien, usar el resultado filtrado
                                self.teams_data = result_df
                                logger.debug(f"Usando resultado filtrado. Forma: {self.teams_data.shape}")
                        else:
                            # Si no hay partes para concatenar, mantener el DataFrame original
                            logger.warning("No hay columnas para concatenar después del filtrado")
                            logger.debug("Manteniendo DataFrame original")
                    else:
                        logger.debug("No hay columnas numéricas para filtrar")
                    
                    # Desfragmentar una última vez
                    self.teams_data = self.teams_data.copy()
                    logger.debug("DataFrame desfragmentado")
                    
                except Exception as e:
                    logger.error(f"Error durante el filtrado de características correlacionadas: {str(e)}")
                    logger.error(f"Traza de error: {traceback.format_exc()}")
                    # No hacer nada más, continuar con el DataFrame actual
            
            # Verificación final de columnas requeridas
            required_cols = ['Team', 'Date', 'PTS', 'PTS_Opp', 'total_points', 
                           'win_probability', 'offensive_rating', 'defensive_rating']
            
            missing_cols = [col for col in required_cols if col not in self.teams_data.columns]
            if missing_cols:
                logger.warning(f"Columnas requeridas faltantes después del procesamiento: {missing_cols}")
                
                # Recrear columnas faltantes
                for col in missing_cols:
                    logger.warning(f"Recreando columna requerida: {col}")
                    if col == 'win_probability':
                        self.teams_data[col] = 0.5
                    elif col == 'offensive_rating' or col == 'defensive_rating':
                        self.teams_data[col] = 100.0
                    elif col == 'total_points' and 'PTS' in self.teams_data.columns and 'PTS_Opp' in self.teams_data.columns:
                        self.teams_data[col] = self.teams_data['PTS'] + self.teams_data['PTS_Opp']
                    else:
                        self.teams_data[col] = 0
            
            # Verificar valores nulos en columnas requeridas
            for col in required_cols:
                if col in self.teams_data.columns and self.teams_data[col].isnull().any():
                    logger.warning(f"Columna requerida {col} contiene valores nulos. Rellenando.")
                    if col == 'win_probability':
                        self.teams_data[col] = self.teams_data[col].fillna(0.5)
                    elif col == 'offensive_rating' or col == 'defensive_rating':
                        self.teams_data[col] = self.teams_data[col].fillna(100.0)
                    elif col == 'total_points':
                        self.teams_data[col] = self.teams_data[col].fillna(0)
                    elif col in ['Team', 'Date']:
                        # No deberían ser nulos, pero por si acaso
                        logger.error(f"Columna crítica {col} contiene valores nulos.")
                    else:
                        self.teams_data[col] = self.teams_data[col].fillna(0)
            
            logger.info(f"Proceso de generación de características completado. DataFrame final: {self.teams_data.shape}")
            
            # Verificación final para columnas std_*
            logger.info("Realizando verificación final para columnas de desviación estándar...")
            std_cols = [col for col in self.teams_data.columns if '_std_' in col]
            if std_cols:
                for col in std_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        neg_mask = (values < 0)
                        
                        if inf_mask.any() or nan_mask.any() or neg_mask.any():
                            logger.warning(f"Columna {col} tiene valores problemáticos: {inf_mask.sum()} infinitos, {nan_mask.sum()} NaN, {neg_mask.sum()} negativos")
                            
                            # Determinar un límite superior razonable basado en la naturaleza de la columna
                            if 'PTS' in col or 'total_points' in col:
                                upper_limit = 30.0  # Límite para puntos
                            elif 'FG' in col or 'FGA' in col or '3P' in col or '3PA' in col:
                                upper_limit = 15.0  # Límite para tiros
                            elif 'FT' in col or 'FTA' in col:
                                upper_limit = 10.0  # Límite para tiros libres
                            elif '%' in col:
                                upper_limit = 0.3   # Límite para porcentajes
                            else:
                                upper_limit = 20.0  # Límite general
                            
                            # Corregir valores problemáticos
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, upper_limit)
                            logger.info(f"Valores en columna {col} limitados al rango [0, {upper_limit}]")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        # En caso de error grave, reemplazar toda la columna con ceros
                        self.teams_data[col] = 0
                        logger.warning(f"Columna {col} reemplazada con ceros debido a error")
            
            # Verificación final para columnas trend_*
            logger.info("Realizando verificación final para columnas de tendencia...")
            trend_cols = [col for col in self.teams_data.columns if '_trend_' in col]
            if trend_cols:
                for col in trend_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        
                        if inf_mask.any() or nan_mask.any():
                            logger.warning(f"Columna {col} tiene valores problemáticos: {inf_mask.sum()} infinitos, {nan_mask.sum()} NaN")
                            
                            # Determinar límites razonables basados en la naturaleza de la columna
                            if 'PTS' in col or 'total_points' in col:
                                limit = 30.0  # Límite para puntos
                            elif 'FG' in col or 'FGA' in col or '3P' in col or '3PA' in col:
                                limit = 15.0  # Límite para tiros
                            elif 'FT' in col or 'FTA' in col:
                                limit = 10.0  # Límite para tiros libres
                            elif '%' in col:
                                limit = 0.3   # Límite para porcentajes
                            else:
                                limit = 20.0  # Límite general
                            
                            # Corregir valores problemáticos
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-limit, limit)
                            logger.info(f"Valores en columna {col} limitados al rango [{-limit}, {limit}]")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        # En caso de error grave, reemplazar toda la columna con ceros
                        self.teams_data[col] = 0
                        logger.warning(f"Columna {col} reemplazada con ceros debido a error")
            
            # Verificación final para columnas volatility_*
            logger.info("Realizando verificación final para columnas de volatilidad...")
            volatility_cols = [col for col in self.teams_data.columns if '_volatility_' in col]
            if volatility_cols:
                for col in volatility_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        neg_mask = (values < 0)
                        
                        if inf_mask.any() or nan_mask.any() or neg_mask.any():
                            logger.warning(f"Columna {col} tiene valores problemáticos: {inf_mask.sum()} infinitos, {nan_mask.sum()} NaN, {neg_mask.sum()} negativos")
                            
                            # Corregir valores problemáticos - volatilidad siempre debe ser >= 0
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 30)
                            logger.info(f"Valores en columna {col} limitados al rango [0, 30]")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        # En caso de error grave, reemplazar toda la columna con ceros
                        self.teams_data[col] = 0
                        logger.warning(f"Columna {col} reemplazada con ceros debido a error")
            
            # Verificación final para columnas wins_vs_*
            logger.info("Realizando verificación final para columnas de victorias contra tipos de equipos...")
            wins_vs_cols = [col for col in self.teams_data.columns if 'wins_vs_' in col]
            if wins_vs_cols:
                for col in wins_vs_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        out_of_range = (values < 0) | (values > 1)  # Fuera del rango [0,1]
                        
                        if inf_mask.any() or nan_mask.any() or out_of_range.any():
                            logger.warning(f"Columna {col} tiene valores problemáticos: {inf_mask.sum()} infinitos, {nan_mask.sum()} NaN, {out_of_range.sum()} fuera de rango")
                            
                            # Corregir valores problemáticos - tasas de victoria deben estar en [0,1]
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 1)
                            logger.info(f"Valores en columna {col} limitados al rango [0, 1]")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        # En caso de error grave, reemplazar toda la columna con 0.5
                        self.teams_data[col] = 0.5
                        logger.warning(f"Columna {col} reemplazada con 0.5 debido a error")
            
            # Verificación final para columnas de consistencia
            logger.info("Realizando verificación final para columnas de consistencia...")
            consistency_cols = [col for col in self.teams_data.columns if '_consistency_' in col]
            if consistency_cols:
                for col in consistency_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        out_of_range = (values < 0) | (values > 1)  # Fuera del rango [0,1]
                        
                        if inf_mask.any() or nan_mask.any() or out_of_range.any():
                            logger.warning(f"Columna {col} tiene valores problemáticos: {inf_mask.sum()} infinitos, {nan_mask.sum()} NaN, {out_of_range.sum()} fuera de rango")
                            
                            # Corregir valores problemáticos - consistencia debe estar en [0,1]
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 1)
                            logger.info(f"Valores en columna {col} limitados al rango [0, 1]")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        # En caso de error grave, reemplazar toda la columna con 0.5
                        self.teams_data[col] = 0.5
                        logger.warning(f"Columna {col} reemplazada con 0.5 debido a error")
            
            # Verificación final para columnas de racha
            logger.info("Realizando verificación final para columnas de racha...")
            streak_cols = [col for col in self.teams_data.columns if '_streak_' in col]
            if streak_cols:
                for col in streak_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        neg_mask = (values < 0)
                        
                        if inf_mask.any() or nan_mask.any() or neg_mask.any():
                            logger.warning(f"Columna {col} tiene valores problemáticos: {inf_mask.sum()} infinitos, {nan_mask.sum()} NaN, {neg_mask.sum()} negativos")
                            
                            # Corregir valores problemáticos - rachas siempre son >= 0
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 20)
                            logger.info(f"Valores en columna {col} limitados al rango [0, 20]")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        # En caso de error grave, reemplazar toda la columna con ceros
                        self.teams_data[col] = 0
                        logger.warning(f"Columna {col} reemplazada con ceros debido a error")
            
            # Verificación final para características de días de descanso
            logger.info("Realizando verificación final para características relacionadas con días de descanso...")
            rest_cols = [col for col in self.teams_data.columns if 'rest_' in col and col != 'days_rest']
            if rest_cols:
                for col in rest_cols:
                    try:
                        # Convertir a numérico de forma segura
                        values = pd.to_numeric(self.teams_data[col], errors='coerce')
                        
                        # Determinar si es una tasa (0-1) o un valor numérico general
                        is_interaction = col == 'rest_performance_interaction'
                        is_rate = 'win_rate' in col or 'rate' in col or is_interaction
                        
                        # Identificar valores problemáticos
                        inf_mask = np.isinf(values)
                        nan_mask = np.isnan(values)
                        
                        if is_rate or is_interaction:
                            # Para tasas e interacciones, valores válidos están en [0,1]
                            invalid_mask = (values < 0) | (values > 1)
                            valid_range = [0, 1]
                            default_value = 0.5
                        else:
                            # Para puntos, rangos típicos en NBA
                            invalid_mask = (values < 70) | (values > 150)
                            valid_range = [70, 150]
                            default_value = 100
                        
                        # Contar problemas
                        problem_count = inf_mask.sum() + nan_mask.sum() + invalid_mask.sum()
                        
                        if problem_count > 0:
                            logger.warning(f"Columna {col} tiene {problem_count} valores problemáticos: " 
                                          f"{inf_mask.sum()} infinitos, {nan_mask.sum()} NaN, "
                                          f"{invalid_mask.sum()} fuera de rango")
                            
                            # Corregir valores problemáticos
                            self.teams_data[col] = values.replace([np.inf, -np.inf], np.nan).fillna(default_value).clip(valid_range[0], valid_range[1])
                            logger.info(f"Valores en columna {col} limitados al rango [{valid_range[0]}, {valid_range[1]}]")
                            
                            # Verificar si hay equipos sin datos
                            teams_with_constant = []
                            for team in self.teams_data['Team'].unique():
                                team_vals = self.teams_data.loc[self.teams_data['Team'] == team, col]
                                if team_vals.nunique() == 1:
                                    teams_with_constant.append(team)
                            
                            if teams_with_constant:
                                logger.warning(f"Equipos con valor constante en {col}: {len(teams_with_constant)}")
                    except Exception as e:
                        logger.error(f"Error al verificar columna {col}: {str(e)}")
                        
                        # En caso de error grave, asignar valor por defecto según tipo
                        if 'win_rate' in col or col == 'rest_performance_interaction':
                            self.teams_data[col] = 0.5
                        elif 'avg_points' in col:
                            self.teams_data[col] = 100
                        else:
                            self.teams_data[col] = 0
                        
                        logger.warning(f"Columna {col} reemplazada con valor por defecto debido a error")
            
            # Guardar en archivo si se especificó una ruta
            if save_path:
                self.teams_data.to_csv(save_path, index=False)
                logger.info(f"DataFrame con características guardado en {save_path}")
            
            return self.teams_data
            
        except Exception as e:
            logger.error(f"Error en generación de características: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            raise
    
    def plot_feature_importance(self, model, X, y, feature_names=None, top_n=20):
        """
        Calcula la importancia de las características para un modelo y genera JSON con correlaciones
        sin visualizaciones gráficas
        
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
            import json
            json_path = f"feature_correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(corr_dict, f, indent=4)
            
            logger.info(f"Correlaciones guardadas en {json_path}")
        except Exception as e:
            logger.error(f"Error al calcular y guardar correlaciones: {str(e)}")
            logger.error(traceback.format_exc())
        
        return importance_df

    def validate_features(self, features_df):
        """
        Valida las características generadas
        
        Args:
            features_df (pd.DataFrame): DataFrame con características
            
        Returns:
            bool: True si la validación es exitosa
        """
        logger.info("Validando características generadas")
        
        try:
            # Verificar que el DataFrame no está vacío
            if features_df.empty:
                logger.error("El DataFrame de características está vacío")
                return False
            
            # Verificar duplicados
            if 'Team' in features_df.columns and 'Date' in features_df.columns:
                # Asegurar que Date es datetime
                features_df['Date'] = pd.to_datetime(features_df['Date'], errors='coerce')
                
                # Contar duplicados
                duplicates = features_df.duplicated(subset=['Team', 'Date'], keep=False)
                duplicate_count = duplicates.sum()
                
                if duplicate_count > 0:
                    logger.warning(f"Se encontraron {duplicate_count} duplicados por equipo y fecha")
                    logger.warning("Eliminando duplicados y manteniendo la primera ocurrencia")
                    
                    # Eliminar duplicados
                    features_df = features_df.drop_duplicates(subset=['Team', 'Date'], keep='first')
                    logger.warning(f"Forma después de eliminar duplicados: {features_df.shape}")
            
            # Verificar columnas requeridas
            required_features = [
                'Team', 'Date', 'PTS', 'PTS_Opp', 'total_points', 
                'win_probability', 'offensive_rating', 'defensive_rating'
            ]
            
            missing_features = [col for col in required_features if col not in features_df.columns]
            if missing_features:
                logger.error(f"Faltan características requeridas: {missing_features}")
                return False
            
            # Convertir tipos de datos según sea necesario
            logger.debug("Convirtiendo tipos de datos")
            try:
                # Asegurar que Date es datetime
                if 'Date' in features_df.columns:
                    features_df['Date'] = pd.to_datetime(features_df['Date'])
                
                # Asegurar que las columnas numéricas son float64
                numeric_features = ['PTS', 'PTS_Opp', 'total_points', 'win_probability', 
                                  'offensive_rating', 'defensive_rating']
                for col in numeric_features:
                    if col in features_df.columns:
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype('float64')
                
                # Asegurar que Team es string
                if 'Team' in features_df.columns:
                    features_df['Team'] = features_df['Team'].astype(str)
            except Exception as e:
                logger.error(f"Error al convertir tipos de datos: {str(e)}")
                return False
            
            # Verificar valores infinitos solo en columnas numéricas
            logger.debug("Verificando valores infinitos")
            numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                try:
                    # Convertir a numpy array para manejar tipos no numéricos
                    col_data = np.array(features_df[col], dtype=float)
                    inf_mask = np.isinf(col_data)
                    if inf_mask.any():
                        logger.warning(f"Columna {col} tiene valores infinitos")
                        features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
                except Exception as e:
                    logger.warning(f"No se pudo verificar valores infinitos en columna {col}: {str(e)}")
            
            # Verificar valores nulos en columnas requeridas
            logger.debug("Verificando valores nulos en columnas requeridas")
            for col in required_features:
                null_count = features_df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Columna {col} tiene {null_count} valores nulos")
                    # Rellenar valores nulos según el tipo de columna
                    if col == 'win_probability':
                        features_df[col] = features_df[col].fillna(0.5)
                    elif col in ['offensive_rating', 'defensive_rating']:
                        features_df[col] = features_df[col].fillna(100.0)
                    elif col in ['PTS', 'PTS_Opp', 'total_points']:
                        features_df[col] = features_df[col].fillna(0)
            
            # Verificar que las fechas están ordenadas para cada equipo
            logger.debug("Verificando orden de fechas")
            if 'Date' in features_df.columns and 'Team' in features_df.columns:
                # Ordenar por equipo y fecha para asegurar consistencia
                features_df = features_df.sort_values(['Team', 'Date'])
                
                # Verificar si las fechas son monotónicas después del ordenamiento
                date_order_check = features_df.groupby('Team')['Date'].is_monotonic_increasing
                if not all(date_order_check):
                    logger.error("Las fechas no están ordenadas correctamente para todos los equipos después del ordenamiento")
                    return False
            
            # Verificar rangos válidos para columnas específicas
            logger.debug("Verificando rangos de valores")
            if 'win_probability' in features_df.columns:
                wp = features_df['win_probability']
                if (wp < 0).any() or (wp > 1).any():
                    logger.warning("win_probability tiene valores fuera del rango [0,1], ajustando")
                    features_df['win_probability'] = wp.clip(0, 1)
            
            logger.info("Validación completada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en validación de características: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            return False

    def _safe_team_feature_calculation(self, mask, feature_prefix, value_col, default_value=None):
        """
        Calcula características por equipo de forma segura y propaga los valores
        
        Args:
            mask (pd.Series): Máscara booleana para filtrar filas
            feature_prefix (str): Prefijo para el nombre de la característica
            value_col (str): Columna de la que obtener los valores
            default_value (float, optional): Valor por defecto si no hay datos. Si es None, se usa la media de value_col.
            
        Returns:
            tuple: (win_rate_col_name, mean_col_name) - Nombres de las columnas creadas
        """
        # Nombres de columnas a crear
        win_rate_col = f'{feature_prefix}_win_rate'
        mean_col = f'{feature_prefix}_avg_points'
        
        try:
            # Verificar si hay suficientes datos
            if mask.sum() > 0:
                # Calcular win_rate
                if value_col == 'is_win':
                    win_values = pd.to_numeric(self.teams_data.loc[mask, value_col], errors='coerce')
                    win_values = win_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Calcular por equipo inicialmente solo para las filas que cumplen la condición
                    self.teams_data.loc[mask, win_rate_col] = self.teams_data.loc[mask].groupby('Team')[value_col].transform('mean')
                    
                    # Calcular también promedio de puntos
                    pts_values = pd.to_numeric(self.teams_data.loc[mask, 'PTS'], errors='coerce')
                    pts_values = pts_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    self.teams_data.loc[mask, mean_col] = self.teams_data.loc[mask].groupby('Team')['PTS'].transform('mean')
                else:
                    # Si value_col no es 'is_win', calculamos solo promedios del valor especificado
                    values = pd.to_numeric(self.teams_data.loc[mask, value_col], errors='coerce')
                    values = values.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    self.teams_data.loc[mask, mean_col] = self.teams_data.loc[mask].groupby('Team')[value_col].transform('mean')
                
                # Propagar los valores al resto de filas del mismo equipo
                for team in self.teams_data['Team'].unique():
                    team_mask = self.teams_data['Team'] == team
                    team_condition_mask = mask & team_mask
                    
                    if team_condition_mask.any():
                        # Propagar win_rate si se calculó
                        if value_col == 'is_win' and win_rate_col in self.teams_data.columns:
                            team_win_rate = self.teams_data.loc[team_condition_mask, win_rate_col].mean()
                            self.teams_data.loc[team_mask, win_rate_col] = team_win_rate
                        
                        # Propagar avg_points
                        if mean_col in self.teams_data.columns:
                            team_avg = self.teams_data.loc[team_condition_mask, mean_col].mean()
                            self.teams_data.loc[team_mask, mean_col] = team_avg
            else:
                # Si no hay datos con esta condición, usar valores por defecto
                if value_col == 'is_win':
                    self.teams_data[win_rate_col] = 0.5  # Por defecto, 50% de probabilidad
                
                if default_value is None:
                    # Si no se especificó un valor por defecto, usar la media de la columna
                    default_value = self.teams_data[value_col].mean() if value_col in self.teams_data.columns else 0
                
                self.teams_data[mean_col] = default_value
                
            logger.debug(f"Características {feature_prefix} calculadas correctamente")
            return win_rate_col, mean_col
            
        except Exception as e:
            logger.error(f"Error al calcular características {feature_prefix}: {str(e)}")
            # Valores por defecto en caso de error
            if value_col == 'is_win':
                self.teams_data[win_rate_col] = 0.5
            
            if default_value is None:
                # Si no se especificó un valor por defecto, usar la media de la columna
                default_value = self.teams_data[value_col].mean() if value_col in self.teams_data.columns else 0
                
            self.teams_data[mean_col] = default_value
            return win_rate_col, mean_col

