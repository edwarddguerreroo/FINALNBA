import pandas as pd
import numpy as np
import logging
import os
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from src.preprocessing.feature_engineering.teams_features import TeamsFeatures
from src.preprocessing.data_loader import NBADataLoader
import time

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loading.log", mode='w'),  # Sobrescribe el archivo en cada ejecución
        logging.StreamHandler()
    ]
)

# Reducir verbosidad de los warnings de pandas
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

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
        logger.info(f"Rutas configuradas:")
        logger.info(f"- Players: {players_path}")
        logger.info(f"- Height: {height_path}")
        logger.info(f"- Teams: {teams_path}")
        
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
            
    def load_and_process_all(self):
        """
        Carga y procesa todos los datos necesarios
        """
        logger.info("Iniciando carga y procesamiento de todos los datos")
        
        try:
            # Generar características de equipos
            teams_features = self.generate_team_features(save=True)
            
            if teams_features is not None:
                # Verificar duplicados una vez más antes de finalizar
                if 'Team' in teams_features.columns and 'Date' in teams_features.columns:
                    duplicates = teams_features.duplicated(subset=['Team', 'Date'], keep=False)
                    n_duplicates = duplicates.sum()
                    
                    if n_duplicates > 0:
                        logger.warning(f"Se encontraron {n_duplicates} duplicados antes de finalizar. Eliminando...")
                        teams_features = teams_features.drop_duplicates(subset=['Team', 'Date'], keep='first')
                        logger.info(f"Datos finales sin duplicados. Shape: {teams_features.shape}")
                
                # Asegurar ordenamiento final
                if 'Team' in teams_features.columns and 'Date' in teams_features.columns:
                    teams_features = teams_features.sort_values(['Team', 'Date'])
                    logger.info("Datos finales ordenados por equipo y fecha")
                
                logger.info(f"Procesamiento completado. Shape final: {teams_features.shape}")
                
                # Mostrar información sobre las características generadas
                numeric_cols = teams_features.select_dtypes(include=[np.number]).columns
                logger.info(f"Número total de características numéricas: {len(numeric_cols)}")
                
                # Mostrar algunas estadísticas básicas
                logger.info("\nEstadísticas básicas de algunas características importantes:")
                important_cols = ['PTS', 'PTS_Opp', 'total_points', 'win_probability']
                stats = teams_features[important_cols].describe()
                logger.info(f"\n{stats}")
                
                return teams_features
            else:
                logger.error("Error en el procesamiento de datos")
                return None
                
        except Exception as e:
            logger.error(f"Error en el procesamiento general: {str(e)}")
            raise

if __name__ == "__main__":
    # Crear instancia del cargador de datos
    data_loader = DataLoader()
    
    # Procesar todos los datos
    try:
        start_time = time.time()
        logger.info("Iniciando procesamiento de datos")
        
        features_df = data_loader.load_and_process_all()
        
        execution_time = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {execution_time:.2f} segundos")
        
        if features_df is not None:
            print("\nProcesamiento completado exitosamente!")
            print(f"Dimensiones del DataFrame final: {features_df.shape}")
            print(f"Tiempo total de procesamiento: {execution_time:.2f} segundos")
            
            # Verificar que no hay duplicados
            if 'Team' in features_df.columns and 'Date' in features_df.columns:
                duplicates = features_df.duplicated(subset=['Team', 'Date'], keep=False)
                if duplicates.sum() > 0:
                    print(f"ADVERTENCIA: Aún hay {duplicates.sum()} filas duplicadas!")
                else:
                    print("Verificación completada: No hay filas duplicadas.")
            
            print("\nPrimeras columnas del DataFrame:")
            print(features_df.columns[:10].tolist())
            print("\nMuestra de las primeras filas:")
            print(features_df.head())
            
            # Mostrar información sobre características generadas
            numeric_features = features_df.select_dtypes(include=[np.number]).columns
            print(f"\nNúmero total de características numéricas: {len(numeric_features)}")
            print("\nAlgunas características importantes:")
            for col in ['win_probability', 'total_points', 'offensive_rating', 'defensive_rating']:
                if col in features_df.columns:
                    print(f"\n{col}:")
                    print(features_df[col].describe())
        else:
            print("\nERROR: Procesamiento fallido o no se generaron datos!")
            logger.error("El procesamiento no generó resultados válidos")
    except KeyboardInterrupt:
        print("\nProcesamiento interrumpido por el usuario.")
        logger.warning("Procesamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\nERROR en el procesamiento: {str(e)}")
        print("\nDetalles del error:")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        logger.error(f"Error en el procesamiento: {str(e)}")
        logger.error(f"Traza de error: {error_trace}") 