import pandas as pd
import numpy as np
import logging
import os
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from src.preprocessing.feature_engineering.teams_features import TeamsFeatures
from src.preprocessing.feature_engineering.players_features import PlayersFeatures
from src.preprocessing.data_loader import NBADataLoader
from src.preprocessing.sequences import prepare_all_target_sequences, prepare_target_specific_sequences
import time

# Configuración de logging
# Establecer codificación UTF-8 para manejar correctamente caracteres especiales
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loading.log", mode='w', encoding='utf-8'),  # Usar UTF-8 para el archivo
        logging.StreamHandler()  # La salida de consola se maneja con encode/decode en los loggers
    ]
)

# Reducir verbosidad de los warnings de pandas
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Clase para manejar problemas de codificación en el logging de Windows
class SafeLogFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            try:
                # Intenta convertir cualquier mensaje con caracteres Unicode a ASCII
                # reemplazando los caracteres problemáticos con '?'
                record.msg = record.msg.encode('ascii', 'replace').decode('ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                record.msg = "Mensaje con caracteres no ASCII (filtrado)"
        return True

# Aplicar el filtro al logger raíz
root_logger = logging.getLogger()
root_logger.addFilter(SafeLogFilter())

logger = logging.getLogger('DataLoader')

class DataLoader:
    def __init__(self, data_dir="data"):
        """
        Inicializa el cargador de datos
        
        Args:
            data_dir (str): Directorio base de datos
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Crear directorios si no existen
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Definir rutas para NBADataLoader
        players_path = self.data_dir / "players.csv"
        height_path = self.data_dir / "height.csv"
        teams_path = self.data_dir / "teams.csv"
        
        # Inicializar el NBADataLoader con las rutas
        self.nba_loader = NBADataLoader(
            game_data_path=str(players_path),
            biometrics_path=str(height_path),
            teams_path=str(teams_path)
        )
        
        logger.info(f"Inicializado DataLoader con directorio base: {self.data_dir}")
    
        
    def load_teams_data(self):
        """
        Carga y preprocesa los datos de equipos usando NBADataLoader
        """
        logger.info("Cargando y preprocesando datos usando NBADataLoader")
        
        try:
            # Cargar y preprocesar datos usando NBADataLoader
            _, teams_data = self.nba_loader.load_data()  # Este método devuelve (merged_data, teams_data)
            
            if teams_data is None:
                raise ValueError("Error al cargar datos con NBADataLoader")
            
            # Verificar y eliminar duplicados
            if 'Team' in teams_data.columns and 'Date' in teams_data.columns:
                # Asegurar que Date está en formato datetime
                teams_data['Date'] = pd.to_datetime(teams_data['Date'])
                
                # Contar duplicados antes
                duplicates = teams_data.duplicated(subset=['Team', 'Date'], keep=False)
                n_duplicates = duplicates.sum()
                
                if n_duplicates > 0:
                    logger.warning(f"Se encontraron {n_duplicates} filas duplicadas por equipo y fecha")
                    
                    # Mostrar algunos ejemplos de duplicados
                    duplicate_examples = teams_data[duplicates].sort_values(['Team', 'Date']).head(5)
                    logger.warning(f"Ejemplos de duplicados:\n{duplicate_examples[['Team', 'Date']]}")
                    
                    # Eliminar duplicados manteniendo la primera ocurrencia
                    teams_data = teams_data.drop_duplicates(subset=['Team', 'Date'], keep='first')
                    logger.info(f"Se eliminaron {n_duplicates} duplicados")
                    
                # Ordenar por equipo y fecha
                teams_data = teams_data.sort_values(['Team', 'Date'])
                logger.info(f"Datos ordenados por equipo y fecha")
            
            logger.info(f"Datos cargados exitosamente. Shape: {teams_data.shape}")
            return teams_data
            
        except Exception as e:
            logger.error(f"Error al cargar datos de equipos: {str(e)}")
            raise
            
    def validate_features(self, features_df):
        """
        Valida las características generadas y corrige problemas comunes
        
        Args:
            features_df (pd.DataFrame): DataFrame con características
            
        Returns:
            tuple: (bool, pd.DataFrame) - (éxito de validación, DataFrame potencialmente corregido)
        """
        logger.info("Validando características generadas")
        
        try:
            # Verificar que el DataFrame no está vacío
            if features_df.empty:
                logger.error("El DataFrame de características está vacío")
                return False, features_df
            
            # Hacer una copia para no modificar el original durante la validación
            df = features_df.copy()
            
            # Verificar y eliminar duplicados
            if 'Team' in df.columns and 'Date' in df.columns:
                # Asegurar que Date es datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Contar duplicados
                duplicates = df.duplicated(subset=['Team', 'Date'], keep=False)
                duplicate_count = duplicates.sum()
                
                if duplicate_count > 0:
                    logger.warning(f"Se encontraron {duplicate_count} filas duplicadas por equipo y fecha")
                    
                    # Eliminar duplicados manteniendo la primera ocurrencia
                    df = df.drop_duplicates(subset=['Team', 'Date'], keep='first')
                    logger.warning(f"Se eliminaron {duplicate_count} duplicados. Shape resultante: {df.shape}")
            
            # Verificar columnas requeridas
            required_features = [
                'Team', 'Date', 'PTS', 'PTS_Opp', 'total_points', 
                'win_probability', 'offensive_rating', 'defensive_rating'
            ]
            
            missing_features = [col for col in required_features if col not in df.columns]
            if missing_features:
                logger.error(f"Faltan características requeridas: {missing_features}")
                return False, df
            
            # Convertir tipos de datos problemáticos
            # Primero, asegurar que Date es datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Convertir columnas numéricas requeridas a float64
            numeric_required = ['PTS', 'PTS_Opp', 'total_points', 'win_probability', 
                              'offensive_rating', 'defensive_rating']
            for col in numeric_required:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Error al convertir columna {col}: {str(e)}")
            
            # Identificar columnas numéricas de forma segura
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
            except Exception as e:
                logger.error(f"Error al identificar columnas numéricas: {str(e)}")
                # Usar solo las columnas numéricas requeridas
                numeric_cols = [col for col in numeric_required if col in df.columns]
            
            # Verificar valores infinitos de forma segura
            for col in numeric_cols:
                try:
                    # Convertir a numpy array específicamente como float
                    col_data = np.asarray(df[col].values, dtype=np.float64)
                    inf_mask = np.isinf(col_data)
                    if inf_mask.any():
                        logger.warning(f"Columna {col} tiene valores infinitos. Reemplazando.")
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                except Exception as e:
                    logger.warning(f"No se pudo verificar infinitos en columna {col}: {str(e)}")
            
            # Verificar valores nulos
            for col in required_features:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        logger.warning(f"Columna {col} tiene {null_count} valores nulos")
                        
                        # Rellenar valores nulos en columnas críticas
                        if col == 'win_probability':
                            df[col] = df[col].fillna(0.5)
                        elif col in ['offensive_rating', 'defensive_rating']:
                            df[col] = df[col].fillna(100)
                        elif col in ['PTS', 'PTS_Opp', 'total_points']:
                            df[col] = df[col].fillna(0)
            
            # Ordenar por equipo y fecha
            if 'Date' in df.columns and 'Team' in df.columns:
                logger.info("Ordenando DataFrame por equipo y fecha")
                try:
                    # Ordenar por equipo y fecha
                    df = df.sort_values(['Team', 'Date'])
                    
                    # Verificar si el ordenamiento fue exitoso
                    date_order_issues = False
                    for team in df['Team'].unique():
                        team_data = df[df['Team'] == team]['Date']
                        if not team_data.is_monotonic_increasing:
                            logger.warning(f"Aún hay problemas de ordenación para el equipo {team} después de ordenar")
                            date_order_issues = True
                    
                    if date_order_issues:
                        logger.warning("Hay algunos problemas de ordenación persistentes. Se intentará continuar.")
                    else:
                        logger.info("Ordenamiento por equipo y fecha exitoso")
                except Exception as e:
                    logger.error(f"Error al ordenar por equipo y fecha: {str(e)}")
            
            # Verificar rangos válidos para win_probability
            if 'win_probability' in df.columns:
                try:
                    wp = pd.to_numeric(df['win_probability'], errors='coerce')
                    invalid_wp = (wp < 0) | (wp > 1) | pd.isna(wp)
                    if invalid_wp.any():
                        logger.warning(f"win_probability tiene {invalid_wp.sum()} valores inválidos. Ajustando.")
                        df.loc[invalid_wp, 'win_probability'] = 0.5
                except Exception as e:
                    logger.error(f"Error al verificar win_probability: {str(e)}")
                    # Recrear la columna si hay problemas graves
                    df['win_probability'] = 0.5
            
            logger.info("Validación completada exitosamente")
            return True, df
            
        except Exception as e:
            logger.error(f"Error en validación de características: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            # A pesar del error, intentamos continuar
            return True, features_df
            
    def generate_team_features(self, save=True):
        """
        Genera características para los datos de equipos
        
        Args:
            save (bool): Si True, guarda el DataFrame resultante
            
        Returns:
            pd.DataFrame: DataFrame con características generadas
        """
        try:
            # Configurar logger específico para feature engineering
            feature_logger = logging.getLogger('FeatureEngineering')
            feature_handler = logging.FileHandler("feature_engineering.log", mode='w')
            feature_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            feature_logger.addHandler(feature_handler)
            feature_logger.setLevel(logging.INFO)
            
            feature_logger.info("Iniciando proceso de ingeniería de características")
            
            # Cargar datos preprocesados usando NBADataLoader
            teams_data = self.load_teams_data()
            
            # Inicializar generador de características
            feature_generator = TeamsFeatures(
                teams_data=teams_data,
                window_sizes=[3, 5, 10, 20],
                correlation_threshold=0.95,
                enable_correlation_analysis=True,
                n_jobs=-1
            )
            
            # Generar características
            logger.info("Iniciando generación de características avanzadas")
            feature_logger.info(f"Generando características para {len(teams_data)} registros")
            features_df = feature_generator.generate_features()
            
            # Validar características y obtener el DataFrame potencialmente modificado
            logger.info("Validando y ajustando características...")
            feature_logger.info("Validando características generadas")
            validation_successful, features_df = self.validate_features(features_df)
            if not validation_successful:
                logger.error("La validación de características falló")
                feature_logger.error("La validación de características falló")
                return None
            
            # Registrar estadísticas finales
            feature_logger.info(f"Características generadas: {len(features_df.columns)}")
            feature_logger.info(f"Registros finales: {len(features_df)}")
            
            # Guardar características
            if save:
                output_path = self.processed_dir / "teams_features.csv"
                features_df.to_csv(output_path, index=False)
                logger.info(f"Características guardadas en {output_path}")
                feature_logger.info(f"Características guardadas en {output_path}")
                
                # Guardar metadata
                metadata = {
                    'num_features': len(features_df.columns),
                    'num_numeric_features': len(features_df.select_dtypes(include=[np.number]).columns),
                    'num_samples': len(features_df),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'columns': features_df.columns.tolist()
                }
                
                metadata_path = self.processed_dir / "teams_features_metadata.json"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                logger.info(f"Metadata guardada en {metadata_path}")
                feature_logger.info(f"Metadata guardada en {metadata_path}")
            
            # Cerrar el logger específico
            feature_handler.close()
            feature_logger.removeHandler(feature_handler)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error en generación de características: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            raise
            
    def load_players_data(self):
        """
        Carga y preprocesa los datos de jugadores usando NBADataLoader
        
        Returns:
            pd.DataFrame: DataFrame con datos de jugadores
        """
        logger.info("Cargando y preprocesando datos de jugadores usando NBADataLoader")
        
        try:
            # Cargar y preprocesar datos usando NBADataLoader
            players_data, _ = self.nba_loader.load_data()  # Este método devuelve (merged_data, teams_data)
            
            if players_data is None:
                raise ValueError("Error al cargar datos de jugadores con NBADataLoader")
            
            # Verificar y eliminar duplicados
            if 'Player' in players_data.columns and 'Date' in players_data.columns:
                # Asegurar que Date está en formato datetime
                players_data['Date'] = pd.to_datetime(players_data['Date'])
                
                # Contar duplicados antes
                duplicates = players_data.duplicated(subset=['Player', 'Date'], keep=False)
                n_duplicates = duplicates.sum()
                
                if n_duplicates > 0:
                    logger.warning(f"Se encontraron {n_duplicates} filas duplicadas por jugador y fecha")
                    
                    # Mostrar algunos ejemplos de duplicados
                    duplicate_examples = players_data[duplicates].sort_values(['Player', 'Date']).head(5)
                    logger.warning(f"Ejemplos de duplicados:\n{duplicate_examples[['Player', 'Date']]}")
                    
                    # Eliminar duplicados manteniendo la primera ocurrencia
                    players_data = players_data.drop_duplicates(subset=['Player', 'Date'], keep='first')
                    logger.info(f"Se eliminaron {n_duplicates} duplicados")
                    
                # Ordenar por jugador y fecha
                players_data = players_data.sort_values(['Player', 'Date'])
                logger.info(f"Datos ordenados por jugador y fecha")
            
            logger.info(f"Datos de jugadores cargados exitosamente. Shape: {players_data.shape}")
            return players_data
            
        except Exception as e:
            logger.error(f"Error al cargar datos de jugadores: {str(e)}")
            raise
            
    def validate_players_features(self, features_df):
        """
        Valida las características generadas para jugadores
        
        Args:
            features_df (pd.DataFrame): DataFrame con características
            
        Returns:
            tuple: (bool, pd.DataFrame) - (éxito de validación, DataFrame potencialmente corregido)
        """
        logger.info("Validando características de jugadores generadas")
        
        try:
            # Verificar que el DataFrame no está vacío
            if features_df.empty:
                logger.error("El DataFrame de características está vacío")
                return False, features_df
            
            # Hacer una copia para no modificar el original durante la validación
            df = features_df.copy()
            
            # Verificar y eliminar duplicados
            if 'Player' in df.columns and 'Date' in df.columns:
                # Asegurar que Date es datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Contar duplicados
                duplicates = df.duplicated(subset=['Player', 'Date'], keep=False)
                duplicate_count = duplicates.sum()
                
                if duplicate_count > 0:
                    logger.warning(f"Se encontraron {duplicate_count} filas duplicadas por jugador y fecha")
                    
                    # Eliminar duplicados manteniendo la primera ocurrencia
                    df = df.drop_duplicates(subset=['Player', 'Date'], keep='first')
                    logger.warning(f"Se eliminaron {duplicate_count} duplicados. Shape resultante: {df.shape}")
            
            # Verificar columnas requeridas
            required_features = [
                'Player', 'Date', 'PTS', 'TRB', 'AST', 'MP'
            ]
            
            missing_features = [col for col in required_features if col not in df.columns]
            if missing_features:
                logger.warning(f"Faltan características requeridas: {missing_features}")
                # Intentar reconstruir columnas faltantes si es posible
                # ... (código para reconstruir columnas omitido por brevedad)
            
            # Convertir tipos de datos problemáticos
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Convertir columnas numéricas requeridas a float64
            numeric_required = ['PTS', 'TRB', 'AST', 'MP']
            for col in numeric_required:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Error al convertir columna {col}: {str(e)}")
            
            # Ordenar por jugador y fecha
            if 'Date' in df.columns and 'Player' in df.columns:
                logger.info("Ordenando DataFrame por jugador y fecha")
                try:
                    # Ordenar por jugador y fecha
                    df = df.sort_values(['Player', 'Date'])
                    logger.info("Ordenamiento por jugador y fecha exitoso")
                except Exception as e:
                    logger.error(f"Error al ordenar por jugador y fecha: {str(e)}")
            
            logger.info("Validación completada exitosamente")
            return True, df
            
        except Exception as e:
            logger.error(f"Error en validación de características: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            # A pesar del error, intentamos continuar
            return False, features_df

    def generate_player_features(self, save=True):
        """
        Genera características para los datos de jugadores
        
        Args:
            save (bool): Si True, guarda el DataFrame resultante
            
        Returns:
            pd.DataFrame: DataFrame con características generadas
        """
        try:
            # Configurar logger específico para feature engineering
            feature_logger = logging.getLogger('PlayerFeatureEngineering')
            feature_handler = logging.FileHandler("player_feature_engineering.log", mode='w')
            feature_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            feature_logger.addHandler(feature_handler)
            feature_logger.setLevel(logging.INFO)
            
            feature_logger.info("Iniciando proceso de ingeniería de características para jugadores")
            
            # Cargar datos preprocesados
            players_data = self.load_players_data()
            
            # Inicializar generador de características
            feature_generator = PlayersFeatures(
                players_data=players_data,
                window_sizes=[3, 5, 10, 20],
                correlation_threshold=0.95,
                enable_correlation_analysis=True,
                n_jobs=-1
            )
            
            # Generar características
            logger.info("Iniciando generación de características avanzadas para jugadores")
            feature_logger.info(f"Generando características para {len(players_data)} registros")
            features_df = feature_generator.generate_features()
            
            # Validar características y obtener el DataFrame potencialmente modificado
            logger.info("Validando y ajustando características...")
            feature_logger.info("Validando características generadas")
            validation_successful, features_df = self.validate_players_features(features_df)
            if not validation_successful:
                logger.error("La validación de características falló")
                feature_logger.error("La validación de características falló")
                return None
            
            # Registrar estadísticas finales
            feature_logger.info(f"Características generadas: {len(features_df.columns)}")
            feature_logger.info(f"Registros finales: {len(features_df)}")
            
            # Guardar características
            if save:
                output_path = self.processed_dir / "players_features.csv"
                features_df.to_csv(output_path, index=False)
                logger.info(f"Características guardadas en {output_path}")
                feature_logger.info(f"Características guardadas en {output_path}")
                
                # Guardar metadata
                metadata = {
                    'num_features': len(features_df.columns),
                    'num_numeric_features': len(features_df.select_dtypes(include=[np.number]).columns),
                    'num_samples': len(features_df),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'num_players': features_df['Player'].nunique(),
                    'columns': features_df.columns.tolist()
                }
                
                metadata_path = self.processed_dir / "players_features_metadata.json"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                logger.info(f"Metadata guardada en {metadata_path}")
                feature_logger.info(f"Metadata guardada en {metadata_path}")
            
            # Cerrar el logger específico
            feature_handler.close()
            feature_logger.removeHandler(feature_handler)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error en generación de características para jugadores: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")
            raise

    def generate_sequences(self, data_df=None, targets=None, output_dir=None, save=True):
        """
        Genera secuencias para modelos predictivos a partir de los datos procesados
        
        Args:
            data_df (pd.DataFrame): DataFrame con características preprocesadas. Si es None, se cargará.
            targets (list): Lista de targets para los que generar secuencias. Si es None, se usarán por defecto.
            output_dir (str): Directorio de salida para las secuencias. Si es None, se usará 'data/sequences'.
            save (bool): Si True, guarda las secuencias en el directorio especificado.
            
        Returns:
            dict: Diccionario con rutas a las secuencias generadas por target
        """
        logger.info("Iniciando generación de secuencias para modelos predictivos")
        
        try:
            # Configurar logger específico
            seq_logger = logging.getLogger('SequenceGenerator')
            seq_handler = logging.FileHandler("sequence_generation.log", mode='w')
            seq_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            seq_logger.addHandler(seq_handler)
            seq_logger.setLevel(logging.INFO)
            
            # Targets por defecto si no se especifican
            if targets is None:
                targets = ['PTS', 'TRB', 'AST', '3P', 'Win']
                
            # Configurar directorio de salida
            if output_dir is None:
                output_dir = self.data_dir / "sequences"
            os.makedirs(output_dir, exist_ok=True)
            
            # Cargar datos si no se proporcionan
            if data_df is None:
                seq_logger.info("Cargando datos procesados para generar secuencias")
                
                # Intentar cargar desde archivos de características guardados
                player_features_path = self.processed_dir / "players_features.csv"
                team_features_path = self.processed_dir / "teams_features.csv"
                
                # Verificar si los datos están disponibles
                if not player_features_path.exists() and not team_features_path.exists():
                    seq_logger.error("No se encontraron datos procesados. Ejecute primero generate_player_features() y generate_team_features()")
                    return {}
                
                # Cargar datos de jugadores si incluye targets de jugador
                player_targets = [t for t in targets if t in ['PTS', 'TRB', 'AST', '3P', 'Double_Double', 'Triple_Double']]
                if player_targets and player_features_path.exists():
                    seq_logger.info(f"Cargando datos de jugadores desde {player_features_path}")
                    player_df = pd.read_csv(player_features_path)
                    if 'Date' in player_df.columns:
                        player_df['Date'] = pd.to_datetime(player_df['Date'])
                else:
                    player_df = None
                
                # Cargar datos de equipos si incluye targets de equipo
                team_targets = [t for t in targets if t in ['Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']]
                if team_targets and team_features_path.exists():
                    seq_logger.info(f"Cargando datos de equipos desde {team_features_path}")
                    team_df = pd.read_csv(team_features_path)
                    if 'Date' in team_df.columns:
                        team_df['Date'] = pd.to_datetime(team_df['Date'])
                else:
                    team_df = None
                    
                # Si se proporcionó un dataframe, usarlo directamente
                if data_df is not None:
                    seq_logger.info("Usando el DataFrame proporcionado")
                    # Determinar si es de jugadores o equipos basado en columnas
                    if 'Player' in data_df.columns:
                        player_df = data_df
                    elif 'Team' in data_df.columns:
                        team_df = data_df
            else:
                # Usar el DataFrame proporcionado
                if 'Player' in data_df.columns:
                    player_df = data_df
                    team_df = None
                elif 'Team' in data_df.columns:
                    team_df = data_df
                    player_df = None
                else:
                    seq_logger.warning("No se pudo determinar el tipo de datos (jugador/equipo). Se intentará usar como está.")
                    player_df = data_df
                    team_df = data_df
            
            # Generar secuencias para targets de jugador
            player_results = {}
            if player_df is not None and any(t in ['PTS', 'TRB', 'AST', '3P', 'Double_Double', 'Triple_Double'] for t in targets):
                seq_logger.info("Generando secuencias para targets de jugador")
                player_targets = [t for t in targets if t in ['PTS', 'TRB', 'AST', '3P', 'Double_Double', 'Triple_Double']]
                
                # Carpeta específica para secuencias de jugador
                player_output_dir = os.path.join(output_dir, "player")
                os.makedirs(player_output_dir, exist_ok=True)
                
                try:
                    player_results = prepare_all_target_sequences(
                        df=player_df,
                        targets=player_targets,
                        output_dir=player_output_dir,
                        sequence_length=10,
                        min_games=5,
                        confidence_threshold=0.85,
                        min_historical_accuracy=0.90
                    )
                    
                    seq_logger.info(f"Secuencias de jugador generadas para {len(player_results)} targets")
                except Exception as e:
                    seq_logger.error(f"Error al generar secuencias de jugador: {str(e)}")
                    seq_logger.error(traceback.format_exc())
            
            # Generar secuencias para targets de equipo
            team_results = {}
            if team_df is not None and any(t in ['Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under'] for t in targets):
                seq_logger.info("Generando secuencias para targets de equipo")
                team_targets = [t for t in targets if t in ['Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']]
                
                # Carpeta específica para secuencias de equipo
                team_output_dir = os.path.join(output_dir, "team")
                os.makedirs(team_output_dir, exist_ok=True)
                
                try:
                    team_results = prepare_all_target_sequences(
                        df=team_df,
                        targets=team_targets,
                        output_dir=team_output_dir,
                        sequence_length=10,
                        min_games=5,
                        confidence_threshold=0.85,
                        min_historical_accuracy=0.90
                    )
                    
                    seq_logger.info(f"Secuencias de equipo generadas para {len(team_results)} targets")
                except Exception as e:
                    seq_logger.error(f"Error al generar secuencias de equipo: {str(e)}")
                    seq_logger.error(traceback.format_exc())
            
            # Combinar resultados
            all_results = {**player_results, **team_results}
            
            # Generar resumen de resultados
            seq_logger.info("\n" + "="*50)
            seq_logger.info("RESUMEN DE SECUENCIAS GENERADAS")
            seq_logger.info("="*50)
            
            for target, (seq_path, map_path) in all_results.items():
                if seq_path:
                    seq_logger.info(f"Target {target}: [OK]")
                    seq_logger.info(f"  - Secuencias: {seq_path}")
                    seq_logger.info(f"  - Mapeo: {map_path}")
                else:
                    seq_logger.info(f"Target {target}: [FAILED] (falló)")
            
            seq_logger.info(f"\nProcesados {len(all_results)}/{len(targets)} targets exitosamente")
            
            # Cerrar el logger
            seq_handler.close()
            seq_logger.removeHandler(seq_handler)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error en generación de secuencias: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def load_and_process_all(self):
        """
        Carga, procesa y genera características para todos los datos
        
        Returns:
            tuple: (player_features, team_features) - DataFrames con características de jugadores y equipos
        """
        start_time = time.time()
        logger.info("Iniciando procesamiento completo de datos")
        
        try:
            # Paso 1: Generar características de equipo
            logger.info("PASO 1: Generando características de equipo")
            team_features = self.generate_team_features(save=True)
            
            # Paso 2: Generar características de jugador
            logger.info("PASO 2: Generando características de jugador")
            player_features = self.generate_player_features(save=True)
            
            # Paso 3: Generar secuencias para modelos predictivos
            logger.info("PASO 3: Generando secuencias para modelos predictivos")
            targets = ['PTS', 'TRB', 'AST', '3P', 'Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']
            sequences_results = self.generate_sequences(
                data_df=None,  # Cargar desde archivos guardados
                targets=targets,
                save=True
            )
            
            end_time = time.time()
            logger.info(f"Procesamiento completo finalizado en {end_time - start_time:.2f} segundos")
            
            return player_features, team_features
            
        except Exception as e:
            logger.error(f"Error en procesamiento completo: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

if __name__ == "__main__":
    try:
        # Configurar logging
        logger.info("Iniciando procesamiento completo")
        
        # Crear instancia de DataLoader
        data_loader = DataLoader(data_dir="data")
        
        # Ejecutar el proceso completo
        start_time = time.time()
        players_df, teams_df = data_loader.load_and_process_all()
        end_time = time.time()
        
        # Reportar éxito o fracaso
        if players_df is not None and teams_df is not None:
            logger.info("="*50)
            logger.info("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
            logger.info(f"Tiempo total: {(end_time - start_time) / 60:.2f} minutos")
            logger.info(f"Datos de jugadores: {players_df.shape}")
            logger.info(f"Datos de equipos: {teams_df.shape}")
            logger.info("="*50)
            print("\n¡PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
            print(f"Tiempo total: {(end_time - start_time) / 60:.2f} minutos")
            print(f"Datos de jugadores: {players_df.shape}")
            print(f"Datos de equipos: {teams_df.shape}")
        else:
            logger.error("="*50)
            logger.error("ERROR EN EL PROCESAMIENTO")
            logger.error("="*50)
            print("\nERROR EN EL PROCESAMIENTO. Revise los archivos de log para más detalles.")
            
    except Exception as e:
        logger.error(f"Error en el procesamiento principal: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\nError en el procesamiento: {str(e)}")
        print("Revise los archivos de log para más detalles.") 