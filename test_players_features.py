import pandas as pd
import numpy as np
import logging
import os
import time
import traceback
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Reducir logging de matplotlib
logging.getLogger('PIL').setLevel(logging.WARNING)  # Reducir logging de PIL

from src.preprocessing.feature_engineering.players_features import PlayersFeatures
from src.preprocessing.data_loader import NBADataLoader

# Configurar el logger principal
logger = logging.getLogger('PlayersFeatureTester')
logger.setLevel(logging.INFO)

# Crear un manejador de consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Crear un formateador
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Agregar el manejador al logger
logger.addHandler(console_handler)

# Desactivar la propagación al logger raíz
logger.propagate = False

# Reducir verbosidad de los warnings de pandas
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class PlayerFeatureTester:
    def __init__(self, data_dir="data"):
        """
        Inicializa el tester de características de jugadores
        
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
        
        logger.info(f"Inicializado PlayerFeatureTester con directorio base: {self.data_dir}")
        logger.info(f"Rutas configuradas:")
        logger.info(f"- Players: {players_path}")
        logger.info(f"- Height: {height_path}")
        logger.info(f"- Teams: {teams_path}")
    
    def load_players_data(self):
        """
        Carga y preprocesa los datos de jugadores usando NBADataLoader
        
        Returns:
            pd.DataFrame: DataFrame con datos de jugadores
        """
        logger.info("Cargando y preprocesando datos usando NBADataLoader")
        
        try:
            # Cargar y preprocesar datos usando NBADataLoader
            players_data, _ = self.nba_loader.load_data()  # Este método devuelve (merged_data, teams_data)
            
            if players_data is None:
                raise ValueError("Error al cargar datos con NBADataLoader")
            
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
            
            logger.info(f"Datos cargados exitosamente. Shape: {players_data.shape}")
            
            # Mostrar un resumen de los datos
            logger.info(f"Columnas disponibles: {players_data.columns.tolist()}")
            logger.info(f"Número de jugadores únicos: {players_data['Player'].nunique()}")
            
            # Mostrar estadísticas básicas para algunas columnas importantes
            num_cols = ['PTS', 'TRB', 'AST', '3P', 'MP']
            present_cols = [col for col in num_cols if col in players_data.columns]
            if present_cols:
                logger.info("Estadísticas básicas:")
                for col in present_cols:
                    stats = players_data[col].describe()
                    logger.info(f"{col}:\n{stats}")
            
            return players_data
            
        except Exception as e:
            logger.error(f"Error al cargar datos de jugadores: {str(e)}")
            logger.error(traceback.format_exc())
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
                for missing_col in missing_features:
                    # Para PTS, intentar reconstruir desde variables derivadas o relacionadas
                    if missing_col == 'PTS':
                        if 'pts_from_2p' in df.columns and 'pts_from_3p' in df.columns and 'pts_from_ft' in df.columns:
                            logger.info("Reconstruyendo PTS a partir de pts_from_2p, pts_from_3p y pts_from_ft")
                            df['PTS'] = df['pts_from_2p'] + df['pts_from_3p'] + df['pts_from_ft']
                        elif '2P' in df.columns and '3P' in df.columns and 'FT' in df.columns:
                            logger.info("Reconstruyendo PTS a partir de 2P, 3P y FT")
                            df['PTS'] = df['2P'] * 2 + df['3P'] * 3 + df['FT']
                        elif 'FG' in df.columns and '3P' in df.columns and 'FT' in df.columns:
                            logger.info("Reconstruyendo PTS a partir de FG, 3P y FT")
                            df['PTS'] = (df['FG'] * 2 - df['3P']) + (df['3P'] * 3) + df['FT']
                        else:
                            # Intentar usar el promedio de PTS_mean si está disponible
                            pts_mean_cols = [col for col in df.columns if 'PTS_mean_' in col]
                            if pts_mean_cols:
                                logger.info(f"Reconstruyendo PTS a partir del promedio de {pts_mean_cols}")
                                df['PTS'] = df[pts_mean_cols].mean(axis=1)
                            else:
                                logger.error("No se pudo reconstruir PTS")
                    
                    # Para TRB, intentar reconstruir desde componentes o promedios
                    elif missing_col == 'TRB':
                        if 'ORB' in df.columns and 'DRB' in df.columns:
                            logger.info("Reconstruyendo TRB a partir de ORB y DRB")
                            df['TRB'] = df['ORB'] + df['DRB']
                        else:
                            # Intentar usar el promedio de TRB_mean si está disponible
                            trb_mean_cols = [col for col in df.columns if 'TRB_mean_' in col]
                            if trb_mean_cols:
                                logger.info(f"Reconstruyendo TRB a partir del promedio de {trb_mean_cols}")
                                df['TRB'] = df[trb_mean_cols].mean(axis=1)
                            else:
                                logger.error("No se pudo reconstruir TRB")
                    
                    # Para AST, intentar reconstruir desde promedios
                    elif missing_col == 'AST':
                        # Intentar usar el promedio de AST_mean si está disponible
                        ast_mean_cols = [col for col in df.columns if 'AST_mean_' in col]
                        if ast_mean_cols:
                            logger.info(f"Reconstruyendo AST a partir del promedio de {ast_mean_cols}")
                            df['AST'] = df[ast_mean_cols].mean(axis=1)
                        else:
                            logger.error("No se pudo reconstruir AST")
                    
                    # Para MP, intentar reconstruir desde promedios
                    elif missing_col == 'MP':
                        # Intentar usar el promedio de MP_mean si está disponible
                        mp_mean_cols = [col for col in df.columns if 'MP_mean_' in col]
                        if mp_mean_cols:
                            logger.info(f"Reconstruyendo MP a partir del promedio de {mp_mean_cols}")
                            df['MP'] = df[mp_mean_cols].mean(axis=1)
                        else:
                            logger.error("No se pudo reconstruir MP")
                
                # Verificar de nuevo las columnas faltantes después de la reconstrucción
                still_missing = [col for col in required_features if col not in df.columns]
                if still_missing:
                    # Si faltan columnas de fecha o jugador, es un error crítico
                    if 'Player' in still_missing or 'Date' in still_missing:
                        logger.error(f"Faltan columnas críticas después de reconstrucción: {still_missing}")
                        return False, df
                    # Para otras columnas, crear con valores por defecto
                    else:
                        for col in still_missing:
                            logger.warning(f"Creando columna {col} con valores por defecto")
                            df[col] = 0
            
            # Convertir tipos de datos problemáticos
            # Primero, asegurar que Date es datetime
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
                        if col in ['PTS', 'TRB', 'AST', 'MP']:
                            df[col] = df[col].fillna(0)
            
            # Ordenar por jugador y fecha
            if 'Date' in df.columns and 'Player' in df.columns:
                logger.info("Ordenando DataFrame por jugador y fecha")
                try:
                    # Ordenar por jugador y fecha
                    df = df.sort_values(['Player', 'Date'])
                    
                    # Verificar si el ordenamiento fue exitoso
                    date_order_issues = False
                    for player in df['Player'].unique()[:10]:  # Verificar solo los primeros 10 jugadores
                        player_data = df[df['Player'] == player]['Date']
                        if not player_data.is_monotonic_increasing:
                            logger.warning(f"Aún hay problemas de ordenación para el jugador {player} después de ordenar")
                            date_order_issues = True
                    
                    if date_order_issues:
                        logger.warning("Hay algunos problemas de ordenación persistentes. Se intentará continuar.")
                    else:
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
            
            # Visualizar distribuciones de algunas características importantes
            if save:
                self.visualize_key_features(features_df)
            
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
    
    def visualize_key_features(self, df):
        """
        Crea visualizaciones para características clave
        
        Args:
            df (pd.DataFrame): DataFrame con características
        """
        logger.info("Creando visualizaciones para características clave")
        
        try:
            # Crear directorio para visualizaciones
            viz_dir = self.processed_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Crear una copia del DataFrame para no modificar el original
            df_viz = df.copy()
            
            # Convertir columnas de fecha a datetime
            if 'Date' in df_viz.columns:
                df_viz['Date'] = pd.to_datetime(df_viz['Date'], errors='coerce')
            
            # 1. Distribución de características principales
            key_stats = ['PTS', 'TRB', 'AST', '3P', 'MP']
            present_stats = [stat for stat in key_stats if stat in df_viz.columns]
            
            if present_stats:
                plt.figure(figsize=(15, 12))
                for i, stat in enumerate(present_stats, 1):
                    plt.subplot(len(present_stats), 1, i)
                    # Convertir explícitamente a float64
                    data = pd.to_numeric(df_viz[stat], errors='coerce').astype('float64')
                    sns.histplot(data.dropna(), kde=True)
                    plt.title(f'Distribución de {stat}')
                
                plt.subplots_adjust(hspace=0.4)
                plt.savefig(viz_dir / "key_stats_distribution.png")
                plt.close()
                logger.info("Visualización de distribuciones principales creada")
            
            # 2. Correlación entre características principales
            if len(present_stats) > 1:
                plt.figure(figsize=(10, 8))
                
                # Columnas a proteger
                protected_columns = [
                    'Player', 'Date', 'Team', 'Away', 'Opp', 'Result', 'GS', 'MP',
                    'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
                    'FT', 'FTA', 'FT%', 'TS%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                    'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', '+/-', 'Pos', 'is_win',
                    'team_score', 'opp_score', 'total_score', 'point_diff', 'has_overtime',
                    'overtime_periods', 'is_home', 'Height_Inches', 'Weight', 'BMI', 'is_started'
                ]
                
                # Filtrar columnas numéricas que no están en la lista de protegidas
                numeric_cols = df_viz.select_dtypes(include=[np.number]).columns
                correlation_cols = [col for col in numeric_cols if col not in protected_columns]
                
                if len(correlation_cols) > 1:
                    # Convertir todas las columnas a float64 de manera segura
                    numeric_df = df_viz[correlation_cols].apply(
                        lambda x: pd.to_numeric(x, errors='coerce')
                    ).astype('float64')
                    
                    # Calcular correlación solo si hay suficientes datos
                    if not numeric_df.empty and numeric_df.shape[1] > 1:
                        corr_matrix = numeric_df.corr()
                        
                        # Crear máscara para valores NaN
                        mask = np.isnan(corr_matrix)
                        
                        # Crear el heatmap
                        sns.heatmap(
                            corr_matrix,
                            annot=True,
                            cmap='coolwarm',
                            vmin=-1,
                            vmax=1,
                            mask=mask,
                            fmt='.2f'
                        )
                        plt.title('Correlación entre características numéricas')
                        plt.tight_layout()
                        plt.savefig(viz_dir / "key_stats_correlation.png")
                        plt.close()
                        logger.info("Visualización de correlación entre características numéricas creada")
                    else:
                        logger.warning("No hay suficientes columnas numéricas para calcular correlaciones")
                else:
                    logger.warning("No hay suficientes columnas numéricas para calcular correlaciones")
            
            # 3. Tendencias de PTS a lo largo del tiempo para jugadores destacados
            if 'PTS' in df_viz.columns and 'Player' in df_viz.columns and 'Date' in df_viz.columns:
                # Convertir PTS a float64
                df_viz['PTS'] = pd.to_numeric(df_viz['PTS'], errors='coerce').astype('float64')
                top_scorers = df_viz.groupby('Player')['PTS'].mean().nlargest(5).index.tolist()
                
                plt.figure(figsize=(15, 8))
                for player in top_scorers:
                    player_data = df_viz[df_viz['Player'] == player].sort_values('Date')
                    plt.plot(player_data['Date'], player_data['PTS'], label=player)
                
                plt.title('Tendencia de puntos para jugadores destacados')
                plt.xlabel('Fecha')
                plt.ylabel('Puntos')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / "top_scorers_trend.png")
                plt.close()
                logger.info("Visualización de tendencia de puntos para jugadores destacados creada")
            
            # 4. Distribución de características derivadas
            derived_features = []
            for col in df_viz.columns:
                if '_mean_' in col or '_std_' in col or '_trend_' in col:
                    if any(stat in col for stat in key_stats):
                        derived_features.append(col)
            
            if derived_features:
                sample_derived = derived_features[:min(6, len(derived_features))]
                
                plt.figure(figsize=(15, 15))
                for i, feat in enumerate(sample_derived, 1):
                    plt.subplot(len(sample_derived), 1, i)
                    # Convertir explícitamente a float64
                    data = pd.to_numeric(df_viz[feat], errors='coerce').astype('float64')
                    sns.histplot(data.dropna(), kde=True)
                    plt.title(f'Distribución de {feat}')
                
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(viz_dir / "derived_features_distribution.png")
                plt.close()
                logger.info("Visualización de distribución de características derivadas creada")
            
            # 5. Distribución de dobles-dobles y triples-dobles
            double_double_cols = [col for col in df_viz.columns if 'double_double' in col]
            triple_double_cols = [col for col in df_viz.columns if 'triple_double' in col]
            
            if double_double_cols or triple_double_cols:
                plt.figure(figsize=(12, 10))
                i = 1
                
                for col in double_double_cols + triple_double_cols:
                    if df_viz[col].nunique() <= 10:
                        plt.subplot(len(double_double_cols) + len(triple_double_cols), 1, i)
                        # Convertir explícitamente a numérico primero
                        numeric_data = pd.to_numeric(df_viz[col], errors='coerce')
                        # Luego convertir a entero para asegurar valores discretos
                        int_data = numeric_data.fillna(0).astype(int)
                        # Finalmente convertir a categórico
                        cat_data = int_data.astype('category')
                        sns.countplot(x=cat_data)
                        plt.title(f'Distribución de {col}')
                        i += 1
                
                if i > 1:
                    plt.subplots_adjust(hspace=0.6)
                    plt.savefig(viz_dir / "double_triple_distribution.png")
                    plt.close()
                    logger.info("Visualización de distribución de dobles-dobles y triples-dobles creada")
            
            logger.info("Visualizaciones de características clave completadas")
            
        except Exception as e:
            logger.error(f"Error al crear visualizaciones: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Crear instancia del tester
    tester = PlayerFeatureTester()
    
    # Procesar los datos
    try:
        start_time = time.time()
        logger.info("Iniciando procesamiento de datos de jugadores")
        
        features_df = tester.generate_player_features(save=True)
        
        execution_time = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {execution_time:.2f} segundos")
        
        if features_df is not None:
            print("\nProcesamiento completado exitosamente!")
            print(f"Dimensiones del DataFrame final: {features_df.shape}")
            print(f"Número de jugadores: {features_df['Player'].nunique()}")
            print(f"Tiempo total de procesamiento: {execution_time:.2f} segundos")
            
            # Verificar que no hay duplicados
            if 'Player' in features_df.columns and 'Date' in features_df.columns:
                duplicates = features_df.duplicated(subset=['Player', 'Date'], keep=False)
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
            for col in ['PTS', 'TRB', 'AST', '3P']:
                if col in features_df.columns:
                    print(f"\n{col}:")
                    print(features_df[col].describe())
            
            # Mostrar ejemplo de características derivadas
            print("\nEjemplo de características derivadas generadas:")
            derived_cols = [col for col in features_df.columns if any(x in col for x in ['_mean_', '_std_', '_trend_'])][:5]
            if derived_cols:
                for col in derived_cols:
                    print(f"{col}")
            
            # Mostrar características de doble-doble y triple-doble
            print("\nCaracterísticas de doble-doble y triple-doble:")
            double_triple_cols = [col for col in features_df.columns if any(x in col for x in ['double_double', 'triple_double'])]
            if double_triple_cols:
                for col in double_triple_cols:
                    print(f"{col}")
        else:
            print("\nERROR: Procesamiento fallido o no se generaron datos!")
            logger.error("El procesamiento no generó resultados válidos")
    except KeyboardInterrupt:
        print("\nProcesamiento interrumpido por el usuario.")
        logger.warning("Procesamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\nERROR en el procesamiento: {str(e)}")
        print("\nDetalles del error:")
        error_trace = traceback.format_exc()
        print(error_trace)
        logger.error(f"Error en el procesamiento: {str(e)}")
        logger.error(f"Traza de error: {error_trace}") 