import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tqdm import tqdm
import logging
from ..utils.features_selector import FeaturesSelector

# Configurar logger
logger = logging.getLogger(__name__)

# Configurar pandas para evitar el warning de downcasting
pd.set_option('future.no_silent_downcasting', True)

# Líneas fijas 
BETTING_LINES = {
    # Líneas de equipo
    'Win': [0.5],  # Para predicciones binarias de victoria
    'Total_Points_Over_Under': [190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240],  # Total de puntos en juego
    'Team_Points_Over_Under': [90, 95, 100, 105, 110, 115, 120, 125, 130],  # Puntos de equipo
    
    # Líneas de jugador
    'PTS': [10, 15, 20, 25, 30, 35],  # Puntos
    'TRB': [4, 6, 8, 9, 10, 12],  # Rebotes
    'AST': [4, 6, 8, 9, 10, 12],  # Asistencias
    '3P': [1, 2, 3, 4],  # Triples
    'Double_Double': [0.5],  # Para predicciones binarias de doble-doble
    'Triple_Double': [0.5]   # Para predicciones binarias de triple-doble
}

# Mapeo de modelos a sus líneas correspondientes
MODEL_LINES = {
    # Modelos de equipo
    'win_predictor': ['Win'],
    'total_points_predictor': ['Total_Points_Over_Under'],
    'team_points_predictor': ['Team_Points_Over_Under'],
    
    # Modelos de jugador
    'pts_predictor': ['PTS'],
    'trb_predictor': ['TRB'],
    'ast_predictor': ['AST'],
    '3p_predictor': ['3P'],
    'double_double_predictor': ['Double_Double'],
    'triple_double_predictor': ['Triple_Double']
}

# Definir qué modelos son de equipo y cuáles de jugador
TEAM_MODELS = ['win_predictor', 'total_points_predictor', 'team_points_predictor']
PLAYER_MODELS = ['pts_predictor', 'trb_predictor', 'ast_predictor', '3p_predictor', 
                'double_double_predictor', 'triple_double_predictor']

class SequenceGenerator:
    """
    Generador de secuencias temporales para modelos predictivos
    """
    def __init__(
        self,
        sequence_length: int = 10,
        target_columns: List[str] = None,
        feature_columns: List[str] = None,
        categorical_columns: List[str] = ['Pos', 'Team', 'Opp'],
        model_type: str = None,  # Parámetro para especificar tipo de modelo
        confidence_threshold: float = 0.85,
        min_historical_accuracy: float = 0.93,
        min_samples: int = 15
    ):
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.feature_columns = feature_columns if feature_columns is not None else []
        
        # Si se especifica un modelo_type, usar las líneas correspondientes a ese modelo únicamente
        if model_type and model_type in MODEL_LINES:
            self.target_columns = MODEL_LINES[model_type]
            self.betting_lines = {stat: BETTING_LINES[stat] for stat in MODEL_LINES[model_type]}
            logger.info(f"Configurando generador para modelo específico: {model_type}")
            logger.info(f"Target seleccionado: {self.target_columns}")
        
        # Si no hay modelo_type pero se proporcionan targets, usar esos targets
        elif target_columns:
            # Asegurar que solo hay un target (para cada modelo individual)
            if len(target_columns) > 1:
                logger.warning(f"ADVERTENCIA: Se han proporcionado múltiples targets {target_columns}, pero se recomienda un único target por modelo.")
                logger.warning(f"Usando solo el primer target: {target_columns[0]}")
                self.target_columns = [target_columns[0]]
            else:
                self.target_columns = target_columns
                
            # Configurar líneas de apuestas para el target seleccionado
            self.betting_lines = {stat: BETTING_LINES[stat] for stat in self.target_columns if stat in BETTING_LINES}
            
            if not self.betting_lines:
                logger.error(f"ERROR: El target {self.target_columns[0]} no tiene líneas de apuestas configuradas.")
                self.betting_lines = {}
        
        # Caso por defecto (no debería usarse en producción)
        else:
            logger.warning("ADVERTENCIA: No se ha especificado ni model_type ni target_columns. Usando PTS como target por defecto.")
            self.target_columns = ['PTS']
            self.betting_lines = {stat: BETTING_LINES[stat] for stat in self.target_columns if stat in BETTING_LINES}
        
        logger.info(f"Modelo: {model_type if model_type else 'no especificado'}")
        logger.info(f"Target: {self.target_columns[0]}")
        logger.info(f"Líneas de apuestas configuradas: {self.betting_lines}")
        
        self.categorical_columns = categorical_columns
        self.confidence_threshold = confidence_threshold
        self.min_historical_accuracy = min_historical_accuracy
        self.min_samples = min_samples
        
        # Diccionarios para codificación categórica
        self.categorical_encoders = {}
        
        # Historial de precisión por línea
        self.line_accuracy_history = {
            stat: {line: {'correct': 0, 'total': 0} for line in lines}
            for stat, lines in self.betting_lines.items()
        }
        
        # No inicializar el selector de características aquí para evitar problemas
        # Se inicializará bajo demanda en generate_sequences
        self.features_selector = None

    def _analyze_historical_accuracy(self, player_data, stat, line):
        """
        Analiza la precisión histórica y confianza de una línea para un jugador dado.
        
        Args:
            player_data: DataFrame con los datos históricos del jugador
            stat: Estadística a analizar (PTS, TRB, AST, etc.)
            line: Línea de apuesta a analizar
            
        Returns:
            accuracy: Precisión histórica (0.0-1.0)
            confidence: Nivel de confianza (0.0-1.0)
        
        Returns:
            Tuple[float, float]: (precisión histórica, confianza)
        """
        # Valores por defecto
        historical_accuracy = 0.0
        confidence = 0.0
        
        # Manejar casos especiales
        if len(player_data) == 0:
            return historical_accuracy, confidence
        
        try:
            # Manejar diferentes tipos de estadísticas
            if stat == 'Win':
                # Para estadísticas de victoria de equipo
                over_under_results = player_data['team_score'] > line
            else:
                # Para estadísticas regulares de jugador
                if stat not in player_data.columns:
                    return historical_accuracy, confidence
                
                over_under_results = player_data[stat] > line
            
            # Calcular porcentaje de over/under
            over_pct = np.mean(over_under_results)
            under_pct = 1 - over_pct
            
            # Determinar si la tendencia es over o under
            prediction = 'over' if over_pct >= 0.5 else 'under'
            
            # Calcular precisión histórica (qué tan consistente es el patrón)
            historical_accuracy = max(over_pct, under_pct)
            
            # Calcular confianza basada en la consistencia y el tamaño de la muestra
            # Ajustar por tamaño de muestra (más muestras = más confianza)
            sample_factor = min(1.0, len(over_under_results) / 20)  # Factor máximo de 1.0 con 20+ muestras
            
            # Calcular la distancia a la línea
            if len(over_under_results) > 0:
                mean_val = np.mean(over_under_results)
                line_distance = abs(mean_val - line) / (mean_val + 1e-6)  # Evitar división por cero
                line_factor = min(1.0, line_distance * 2)  # Transformar a [0, 1.0]
            else:
                line_factor = 0.5
            
            # Combinar factores para calcular confianza
            base_confidence = 0.5 + abs(over_pct - 0.5) * 2  # Escalar a [0.5, 1.5]
            confidence = base_confidence * sample_factor * line_factor
        
        except Exception as e:
            print(f"Error en _analyze_historical_accuracy: {e}")
        
        return historical_accuracy, confidence

    def _find_best_betting_line(
        self,
        player_data: pd.DataFrame,
        stat: str
    ) -> Tuple[float, bool, float, float]:
        """
        Encuentra la línea más segura para apostar
        
        Returns:
            Tuple[float, bool, float, float]: (línea, is_over, precisión, confianza)
        """
        best_line = None
        best_is_over = None
        best_accuracy = 0.0
        best_confidence = 0.0
        
        for line in self.betting_lines[stat]:
            historical_accuracy, confidence = self._analyze_historical_accuracy(
                player_data, stat, line
            )
        
            # Solo considerar líneas que cumplen con nuestros umbrales
            if (historical_accuracy >= self.min_historical_accuracy and 
                confidence >= self.confidence_threshold and 
                historical_accuracy > best_accuracy):
                
                best_accuracy = historical_accuracy
                best_confidence = confidence
                best_line = line
                # Calcular de manera explícita si es over o under
                over_count = (player_data[stat] > line).sum()
                total_count = len(player_data)
                best_is_over = over_count / total_count > 0.5
        
        return best_line, best_is_over, best_accuracy, best_confidence
        
    def generate_sequences(
        self,
        df: pd.DataFrame,
        min_games: int = 5,
        null_threshold: float = 0.9,
        use_target_specific_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Genera secuencias para predicciones individuales por línea
        
        Args:
            df: DataFrame con datos procesados
            min_games: Número mínimo de juegos para procesar
            null_threshold: Umbral para filtrar características con alto % de nulos
            use_target_specific_features: Si usar subconjuntos específicos de características por target
            
        Returns:
            Tuple con (secuencias, targets, categóricos, valores_línea, insights)
        """
        # Determinar si es un modelo de equipo o jugador
        is_team_model = self.model_type in TEAM_MODELS if self.model_type else False
        
        # Asegurar tipos de datos correctos
        df = self._ensure_numeric(df, self.feature_columns + self.target_columns)
        
        # Ordenar los datos por equipo/jugador y fecha para mantener el orden temporal
        if 'Team' in df.columns and len(df['Team'].unique()) > 1:
            # Es un DataFrame de equipos - ordenar por equipo y fecha
            logger.debug("Ordenando DataFrame por equipo y fecha")
            df = df.sort_values(by=['Team', 'Date'])
        elif 'Player' in df.columns and len(df['Player'].unique()) > 1:
            # Es un DataFrame de jugadores - ordenar por jugador y fecha
            logger.debug("Ordenando DataFrame por jugador y fecha")
            df = df.sort_values(by=['Player', 'Date'])
        else:
            # Solo ordenar por fecha
            logger.debug("Ordenando DataFrame solo por fecha")
            df = df.sort_values(by='Date')
                
        # Crear codificadores categóricos si no existen
        if not self.categorical_encoders:
            self._create_categorical_encoders(df)
            
        # Inicializar características
        if use_target_specific_features:
            try:
                # Usar el feature selector para obtener características específicas para cada target
                # Reutilizamos el selector existente o creamos uno nuevo si no existe
                if not hasattr(self, 'features_selector') or self.features_selector is None:
                    logger.info("Inicializando FeaturesSelector...")
                    self.features_selector = FeaturesSelector()
                
                # Obtener características específicas por target usando el selector
                target_specific_features = self.features_selector.get_all_target_features(
                    df=df,
                    targets=self.target_columns,
                    include_common=True,
                    include_matchup=True,
                    include_advanced=True,
                    null_threshold=null_threshold
                )
            except Exception as e:
                logger.error(f"Error al obtener características específicas: {str(e)}")
                logger.warning("Usando todas las columnas numéricas como características")
                # En caso de error, usar todas las columnas numéricas
                numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
                target_specific_features = {stat: numeric_columns for stat in self.target_columns}
        else:
            # Si no se solicitan características específicas, usar todas las numéricas
            logger.warning("No se solicitaron características específicas. Usando todas las columnas numéricas.")
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            target_specific_features = {stat: numeric_columns for stat in self.target_columns}
        
        sequences = []
        targets = []
        categorical = []
        line_values = []
        betting_insights = {}
        
        # Para guardar las características utilizadas para cada target
        features_used = {}
        
        # Verificar que tenemos exactamente un target para este modelo
        if len(self.target_columns) != 1:
            logger.warning(f"ADVERTENCIA: Se han especificado múltiples targets ({self.target_columns}) para un modelo individual. Se recomienda un target por modelo.")
            logger.warning(f"Usando solo el primer target: {self.target_columns[0]}")
        
        # Usar solo el primer target (o el único si solo hay uno)
        stat = self.target_columns[0]
        
        # Verificar que el target existe en las líneas de apuestas
        if stat not in self.betting_lines:
            logger.error(f"ERROR: El target '{stat}' no tiene líneas de apuestas definidas en BETTING_LINES")
            return np.array([]), np.array([]), np.array([]), np.array([]), {}
        
        # Obtener las características específicas para este único target
        stat_features = target_specific_features[stat]
        
        # Guardar qué características se están utilizando para cada target
        features_used[stat] = stat_features.copy()
        
        logger.info(f"Generando secuencias para el target: {stat}")
        logger.info(f"Usando {len(stat_features)} características específicas para este target")
        logger.info(f"Ejemplo de características: {stat_features[:10]}..." if len(stat_features) > 10 else f"Características: {stat_features}")
        
        if is_team_model:
            # GENERACIÓN DE SECUENCIAS A NIVEL DE EQUIPO
            logger.info(f"\nGenerando secuencias a nivel de EQUIPO para {stat}")
            
            # Verificar si tenemos la columna Team
            if 'Team' not in df.columns:
                logger.warning("ADVERTENCIA: La columna 'Team' no está presente en el DataFrame. Usando todo el dataset como un único equipo.")
                # Crear un único grupo con todos los datos
                team_data = df
                
                # Procesar como si fuera un solo equipo
                if len(team_data) < min_games:
                    logger.warning(f"Insuficientes juegos ({len(team_data)}) para el mínimo requerido ({min_games})")
                    return np.array([]), np.array([]), np.array([]), np.array([]), {}
                
                # Ordenar datos y seleccionar los últimos partidos 
                team_data = team_data.sort_values(by='Date')
                
                # Analizar cada línea disponible para este target
                for line in self.betting_lines[stat]:
                    # Crear secuencias para este equipo virtual y línea
                    for i in range(len(team_data) - self.sequence_length):
                        sequence = team_data.iloc[i:i+self.sequence_length]
                        target_row = team_data.iloc[i+self.sequence_length]
                
                        # Extraer características
                        feature_sequence = []
                        features_present = []
                        
                        for col in stat_features:
                            if col in sequence.columns:
                                values = sequence[col].values
                                feature_sequence.append(values)
                                features_present.append(col)
                            else:
                                feature_sequence.append(np.zeros(self.sequence_length))
                        
                        # Actualizar el registro de características utilizadas
                        features_used[stat] = features_present
                        
                        # Convertir a array
                        feature_sequence = np.array(feature_sequence, dtype=np.float32).T
                        
                        # Generar target binario
                        if stat == 'Win':
                            try:
                                # Intentar diferentes formas de obtener valor de Win
                                if 'Win' in target_row:
                                    target_val = 1 if bool(target_row['Win']) else 0
                                elif 'is_win' in target_row:
                                    target_val = 1 if bool(target_row['is_win']) else 0
                                elif 'Result' in target_row:
                                    # Posiblemente Result contenga W/L
                                    target_val = 1 if str(target_row['Result']).strip().upper().startswith('W') else 0
                                else:
                                    # Si no hay columna directa, intentar inferir por los puntos
                                    if 'PTS' in target_row and 'PTS_Opp' in target_row:
                                        pts = float(target_row['PTS']) if not pd.isna(target_row['PTS']) else 0
                                        pts_opp = float(target_row['PTS_Opp']) if not pd.isna(target_row['PTS_Opp']) else 0
                                        target_val = 1 if pts > pts_opp else 0
                                    else:
                                        logger.warning(f"No se pudo determinar Win/Loss para equipo virtual")
                                        continue
                                
                                # Asegurar que el valor es binario (0 o 1)
                                target_val = 1 if target_val else 0
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error de tipo de datos procesando target Win para equipo virtual: {str(e)}")
                                continue
                            except Exception as e:
                                logger.error(f"Error procesando target Win para equipo virtual: {str(e)}")
                                continue
                        elif stat == 'Total_Points_Over_Under':
                            try:
                                # Intentar diferentes formas de obtener el total de puntos
                                if 'total_score' in target_row:
                                    target_val = float(target_row['total_score'])
                                elif 'total_points' in target_row:
                                    target_val = float(target_row['total_points'])
                                elif 'PTS' in target_row and 'PTS_Opp' in target_row:
                                    target_val = float(target_row['PTS']) + float(target_row['PTS_Opp'])
                                else:
                                    logger.warning(f"No se pudo determinar el total de puntos para {team}")
                                    continue
                                    
                                # Verificar que el valor no sea nulo o negativo
                                if pd.isna(target_val) or target_val < 0:
                                    logger.warning(f"Valor de puntos totales inválido para {team}: {target_val}")
                                    continue
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error de tipo de datos procesando target Total_Points_Over_Under para {team}: {str(e)}")
                                continue
                            except Exception as e:
                                logger.error(f"Error procesando target Total_Points_Over_Under para {team}: {str(e)}")
                                continue
                        elif stat == 'Team_Points_Over_Under':
                            try:
                                # Intentar diferentes formas de obtener los puntos del equipo
                                if 'team_score' in target_row:
                                    target_val = float(target_row['team_score'])
                                elif 'PTS' in target_row:
                                    target_val = float(target_row['PTS'])
                                elif 'points' in target_row:
                                    target_val = float(target_row['points'])
                                else:
                                    logger.warning(f"No se pudo determinar los puntos del equipo virtual")
                                    continue
                                    
                                # Verificar que el valor no sea nulo o negativo
                                if pd.isna(target_val) or target_val < 0:
                                    logger.warning(f"Valor de puntos inválido para equipo virtual: {target_val}")
                                    continue
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error de tipo de datos procesando target Team_Points_Over_Under para equipo virtual: {str(e)}")
                                continue
                            except Exception as e:
                                logger.error(f"Error procesando target Team_Points_Over_Under para equipo virtual: {str(e)}")
                                continue
                        else:
                            logger.warning(f"Target no soportado: {stat}")
                            continue
                        
                        is_over = target_val > line
                        
                        # Extraer categóricos
                        cat_values = []
                        if self.categorical_columns:  # Solo extraer si hay columnas categóricas definidas
                            for col in self.categorical_columns:
                                if col in target_row:
                                    # Convertir a string para asegurar compatibilidad con el encoder
                                    val_str = str(target_row[col])
                                    cat_idx = self.categorical_encoders.get(col, {}).get(val_str, 0)
                                    cat_values.append(cat_idx)
                                else:
                                    cat_values.append(0)
                        else:
                            # Si no hay columnas categóricas, agregar un valor por defecto
                            cat_values.append(0)
                        
                        # Guardar secuencia
                        sequences.append(feature_sequence)
                        targets.append(1 if is_over else 0)
                        categorical.append(cat_values)
                        line_values.append([line])
                        
                        # Guardar insight
                        key = f"virtual_team_{stat}_{line}_seq{i}"
                        betting_insights[key] = {
                            'team': 'virtual_team',
                            'stat': stat,
                            'line': line,
                            'prediction': 'OVER' if is_over else 'UNDER',
                            'features_count': len(features_present)
                        }
            else:
                # Agrupar por equipo
                team_groups = df.groupby('Team')
                for team, team_data in tqdm(team_groups, desc=f"Procesando equipos para {stat}"):
                    if len(team_data) < min_games:
                        continue
                    
                    # Ordenar datos y seleccionar los últimos partidos
                    team_data = team_data.sort_values(by='Date')
                    
                    # Analizar cada línea disponible para este target
                    for line in self.betting_lines[stat]:
                        # Crear secuencias para este equipo y línea
                        for i in range(len(team_data) - self.sequence_length):
                            sequence = team_data.iloc[i:i+self.sequence_length]
                            target_row = team_data.iloc[i+self.sequence_length]
                            
                            # Extraer características
                            feature_sequence = []
                            features_present = []
                            
                            for col in stat_features:
                                if col in sequence.columns:
                                    values = sequence[col].values
                                    feature_sequence.append(values)
                                    features_present.append(col)
                                else:
                                    feature_sequence.append(np.zeros(self.sequence_length))
                            
                            # Actualizar el registro de características utilizadas
                            features_used[stat] = features_present
                            
                            # Convertir a array
                            feature_sequence = np.array(feature_sequence, dtype=np.float32).T
                            
                            # Generar target binario
                            if stat == 'Win':
                                try:
                                    # Intentar diferentes formas de obtener valor de Win
                                    if 'Win' in target_row:
                                        target_val = 1 if bool(target_row['Win']) else 0
                                    elif 'is_win' in target_row:
                                        target_val = 1 if bool(target_row['is_win']) else 0
                                    elif 'Result' in target_row:
                                        # Posiblemente Result contenga W/L
                                        target_val = 1 if str(target_row['Result']).strip().upper().startswith('W') else 0
                                    else:
                                        # Si no hay columna directa, intentar inferir por los puntos
                                        if 'PTS' in target_row and 'PTS_Opp' in target_row:
                                            pts = float(target_row['PTS']) if not pd.isna(target_row['PTS']) else 0
                                            pts_opp = float(target_row['PTS_Opp']) if not pd.isna(target_row['PTS_Opp']) else 0
                                            target_val = 1 if pts > pts_opp else 0
                                        else:
                                            logger.warning(f"No se pudo determinar Win/Loss para equipo virtual")
                                            continue
                                
                                    # Asegurar que el valor es binario (0 o 1)
                                    target_val = 1 if target_val else 0
                                except (TypeError, ValueError) as e:
                                    logger.error(f"Error de tipo de datos procesando target Win para equipo virtual: {str(e)}")
                                    continue
                                except Exception as e:
                                    logger.error(f"Error procesando target Win para equipo virtual: {str(e)}")
                                    continue
                            elif stat == 'Total_Points_Over_Under':
                                try:
                                    # Intentar diferentes formas de obtener el total de puntos
                                    if 'total_score' in target_row:
                                        target_val = float(target_row['total_score'])
                                    elif 'total_points' in target_row:
                                        target_val = float(target_row['total_points'])
                                    elif 'PTS' in target_row and 'PTS_Opp' in target_row:
                                        target_val = float(target_row['PTS']) + float(target_row['PTS_Opp'])
                                    else:
                                        logger.warning(f"No se pudo determinar el total de puntos para {team}")
                                        continue
                                    
                                    # Verificar que el valor no sea nulo o negativo
                                    if pd.isna(target_val) or target_val < 0:
                                        logger.warning(f"Valor de puntos totales inválido para {team}: {target_val}")
                                        continue
                                except (TypeError, ValueError) as e:
                                    logger.error(f"Error de tipo de datos procesando target Total_Points_Over_Under para {team}: {str(e)}")
                                    continue
                                except Exception as e:
                                    logger.error(f"Error procesando target Total_Points_Over_Under para {team}: {str(e)}")
                                continue
                            elif stat == 'Team_Points_Over_Under':
                                try:
                                    # Intentar diferentes formas de obtener los puntos del equipo
                                    if 'team_score' in target_row:
                                        target_val = float(target_row['team_score'])
                                    elif 'PTS' in target_row:
                                        target_val = float(target_row['PTS'])
                                    elif 'points' in target_row:
                                        target_val = float(target_row['points'])
                                    else:
                                        logger.warning(f"No se pudo determinar los puntos del equipo virtual")
                                        continue
                                    
                                    # Verificar que el valor no sea nulo o negativo
                                    if pd.isna(target_val) or target_val < 0:
                                        logger.warning(f"Valor de puntos inválido para equipo virtual: {target_val}")
                                        continue
                                except (TypeError, ValueError) as e:
                                    logger.error(f"Error de tipo de datos procesando target Team_Points_Over_Under para equipo virtual: {str(e)}")
                                    continue
                                except Exception as e:
                                    logger.error(f"Error procesando target Team_Points_Over_Under para equipo virtual: {str(e)}")
                                continue
                            else:
                                logger.warning(f"Target no soportado: {stat}")
                                continue
                            
                            is_over = target_val > line
                            
                            # Extraer categóricos
                            cat_values = []
                            if self.categorical_columns:  # Solo extraer si hay columnas categóricas definidas
                                for col in self.categorical_columns:
                                    if col in target_row:
                                        # Convertir a string para asegurar compatibilidad con el encoder
                                        val_str = str(target_row[col])
                                        cat_idx = self.categorical_encoders.get(col, {}).get(val_str, 0)
                                        cat_values.append(cat_idx)
                                    else:
                                        cat_values.append(0)
                            else:
                                # Si no hay columnas categóricas, agregar un valor por defecto
                                cat_values.append(0)
                            
                            # Guardar secuencia
                            sequences.append(feature_sequence)
                            targets.append(1 if is_over else 0)
                            categorical.append(cat_values)
                            line_values.append([line])
                            
                            # Guardar insight
                            key = f"{team}_{stat}_{line}"
                            betting_insights[key] = {
                                'team': team,
                                'stat': stat,
                                'line': line,
                                'prediction': 'OVER' if is_over else 'UNDER',
                                'features_count': len(features_present)
                            }
        else:
            # GENERACIÓN DE SECUENCIAS A NIVEL DE JUGADOR
            logger.info(f"\nGenerando secuencias a nivel de JUGADOR para {stat}")
            
            # Verificar si tenemos la columna Player
            if 'Player' not in df.columns:
                logger.warning("ADVERTENCIA: La columna 'Player' no está presente en el DataFrame. Usando todo el dataset como un único jugador.")
                # Crear un único grupo con todos los datos
                player_data = df
                
                # Procesar como si fuera un solo jugador
                if len(player_data) < min_games:
                    logger.warning(f"Insuficientes juegos ({len(player_data)}) para el mínimo requerido ({min_games})")
                    return np.array([]), np.array([]), np.array([]), np.array([]), {}
                
                # Ordenar datos y seleccionar los últimos partidos
                player_data = player_data.sort_values(by='Date')
                
                # Analizar cada línea disponible para este target
                for line in self.betting_lines[stat]:
                    historical_accuracy, confidence = self._analyze_historical_accuracy(
                        player_data, stat, line
                    )
                    
                    # Si cumple con los umbrales de confianza
                    if (historical_accuracy >= self.min_historical_accuracy and 
                        confidence >= self.confidence_threshold):
                        
                        # Crear secuencias para este jugador virtual y línea
                        for i in range(len(player_data) - self.sequence_length):
                            sequence = player_data.iloc[i:i+self.sequence_length]
                            target_row = player_data.iloc[i+self.sequence_length]
                            
                            # Extraer características
                            feature_sequence = []
                            features_present = []
                            
                            for col in stat_features:
                                if col in sequence.columns:
                                    values = sequence[col].values
                                    feature_sequence.append(values)
                                    features_present.append(col)
                                else:
                                    feature_sequence.append(np.zeros(self.sequence_length))
                            
                            # Actualizar el registro de características utilizadas
                            features_used[stat] = features_present
                            
                            # Convertir a array
                            feature_sequence = np.array(feature_sequence, dtype=np.float32).T
                            
                            # Generar target binario
                            target_val = target_row[stat]
                            is_over = target_val > line
                            
                            # Extraer categóricos
                            cat_values = []
                            if self.categorical_columns:  # Solo extraer si hay columnas categóricas definidas
                                for col in self.categorical_columns:
                                    if col in target_row:
                                        # Convertir a string para asegurar compatibilidad con el encoder
                                        val_str = str(target_row[col])
                                        cat_idx = self.categorical_encoders.get(col, {}).get(val_str, 0)
                                        cat_values.append(cat_idx)
                                    else:
                                        cat_values.append(0)
                            else:
                                # Si no hay columnas categóricas, agregar un valor por defecto
                                cat_values.append(0)
                            
                            # Guardar secuencia
                            sequences.append(feature_sequence)
                            targets.append(1 if is_over else 0)
                            categorical.append(cat_values)
                            line_values.append([line])
                            
                            # Guardar insight
                            key = f"virtual_player_{stat}_{line}_seq{i}"
                            betting_insights[key] = {
                                'player': 'virtual_player',
                                'stat': stat,
                                'line': line,
                                'historical_accuracy': historical_accuracy,
                                'confidence': confidence,
                                'prediction': 'OVER' if is_over else 'UNDER',
                                'features_count': len(features_present)
                            }
            else:
                # Agrupar por jugador
                player_groups = df.groupby('Player')
                for player, player_data in tqdm(player_groups, desc=f"Procesando jugadores para {stat}"):
                    if len(player_data) < min_games:
                        continue
                    
                    # Ordenar datos y seleccionar los últimos partidos
                    player_data = player_data.sort_values(by='Date')
                    
                    # Analizar cada línea disponible para este target
                    for line in self.betting_lines[stat]:
                        historical_accuracy, confidence = self._analyze_historical_accuracy(
                            player_data, stat, line
                        )
                        
                        # Si cumple con los umbrales de confianza
                        if (historical_accuracy >= self.min_historical_accuracy and 
                            confidence >= self.confidence_threshold):
                            
                            # Crear secuencias para este jugador y línea
                            for i in range(len(player_data) - self.sequence_length):
                                sequence = player_data.iloc[i:i+self.sequence_length]
                                target_row = player_data.iloc[i+self.sequence_length]
                                
                                # Extraer características
                                feature_sequence = []
                                features_present = []
                                
                                for col in stat_features:
                                    if col in sequence.columns:
                                        values = sequence[col].values
                                        feature_sequence.append(values)
                                        features_present.append(col)
                                    else:
                                        feature_sequence.append(np.zeros(self.sequence_length))
                                
                                # Actualizar el registro de características utilizadas
                                features_used[stat] = features_present
                                
                                # Convertir a array
                                feature_sequence = np.array(feature_sequence, dtype=np.float32).T
                                
                                # Generar target binario
                                target_val = target_row[stat]
                                is_over = target_val > line
                                
                                # Extraer categóricos
                                cat_values = []
                                if self.categorical_columns:  # Solo extraer si hay columnas categóricas definidas
                                    for col in self.categorical_columns:
                                        if col in target_row:
                                            # Convertir a string para asegurar compatibilidad con el encoder
                                            val_str = str(target_row[col])
                                            cat_idx = self.categorical_encoders.get(col, {}).get(val_str, 0)
                                            cat_values.append(cat_idx)
                                        else:
                                            cat_values.append(0)
                                else:
                                    # Si no hay columnas categóricas, agregar un valor por defecto
                                    cat_values.append(0)
                                
                                # Guardar secuencia
                                sequences.append(feature_sequence)
                                targets.append(1 if is_over else 0)
                                categorical.append(cat_values)
                                line_values.append([line])
                                
                                # Guardar insight
                                key = f"{player}_{stat}_{line}"
                                betting_insights[key] = {
                                    'player': player,
                                    'stat': stat,
                                    'line': line,
                                    'historical_accuracy': historical_accuracy,
                                    'confidence': confidence,
                                    'prediction': 'OVER' if is_over else 'UNDER',
                                    'features_count': len(features_present)
                                }
        
        # Convertir listas a arrays
        if not sequences:
            logger.warning(f"ADVERTENCIA: No se generaron secuencias para {stat} que cumplan con los criterios")
            return np.array([]), np.array([]), np.array([]), np.array([]), {}
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        categorical = np.array(categorical, dtype=np.int64)
        line_values = np.array(line_values, dtype=np.float32)
        
        # Registrar las características efectivamente utilizadas
        betting_insights['feature_info'] = {
            'target': stat,
            'total_features_available': len(stat_features),
            'features_effectively_used': len(features_used[stat]),
            'feature_list': features_used[stat][:30] + ['...'] if len(features_used[stat]) > 30 else features_used[stat]
        }
        
        logger.info(f"Generadas {len(sequences)} secuencias para {stat}")
        logger.info(f"Dimensiones de secuencias: {sequences.shape} (secuencias, longitud_secuencia, características)")
        logger.info(f"Se utilizaron {sequences.shape[2]} características de las {len(stat_features)} disponibles")
        
        return sequences, targets, categorical, line_values, betting_insights

    def create_datasets(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        categorical: np.ndarray,
        line_values: np.ndarray,
        train_split: float = 0.7,
        val_split: float = 0.15,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Crea DataLoaders con datos balanceados específicos para el tipo de modelo
        """
        # Calcular índices de división
        n_samples = len(sequences)
        train_size = int(train_split * n_samples)
        val_size = int(val_split * n_samples)
        
        # Crear índices para la división
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Dividir índices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Logging detallado para diagnóstico
        logger.info(f"Generando datasets con las siguientes características:")
        logger.info(f"Tamaño total de secuencias: {len(sequences)}")
        logger.info(f"Tamaño de train_indices: {len(train_indices)}")
        logger.info(f"Tamaño de val_indices: {len(val_indices)}")
        logger.info(f"Tamaño de test_indices: {len(test_indices)}")
        
        # Verificar formas de los datos
        logger.debug(f"Forma de sequences: {sequences.shape}")
        logger.debug(f"Forma de targets: {targets.shape}")
        logger.debug(f"Forma de categorical: {categorical.shape}")
        
        # Crear datasets con las líneas específicas del modelo
        try:
            train_dataset = NBASequenceDatasetWithLines(
                sequences[train_indices],
                targets[train_indices],
                categorical[train_indices],
                self.betting_lines,  # Usar las líneas específicas del modelo
                None,  # No se proporcionan player_ids ni player_stats
                None  # No se proporcionan player_stats
            )
            logger.info("Train dataset creado exitosamente")
        except Exception as e:
            logger.error(f"Error creando train dataset: {e}")
            logger.error(f"Detalles de train_indices: {train_indices}")
            logger.error(f"Detalles de sequences[train_indices]: {sequences[train_indices]}")
            raise
        
        try:
            val_dataset = NBASequenceDatasetWithLines(
                sequences[val_indices],
                targets[val_indices],
                categorical[val_indices],
                self.betting_lines,
                None,  # No se proporcionan player_ids ni player_stats
                None  # No se proporcionan player_stats
            )
            logger.info("Validation dataset creado exitosamente")
        except Exception as e:
            logger.error(f"Error creando validation dataset: {e}")
            logger.error(f"Detalles de val_indices: {val_indices}")
            logger.error(f"Detalles de sequences[val_indices]: {sequences[val_indices]}")
            raise
        
        try:
            test_dataset = NBASequenceDatasetWithLines(
                sequences[test_indices],
                targets[test_indices],
                categorical[test_indices],
                self.betting_lines,
                None,  # No se proporcionan player_ids ni player_stats
                None  # No se proporcionan player_stats
            )
            logger.info("Test dataset creado exitosamente")
        except Exception as e:
            logger.error(f"Error creando test dataset: {e}")
            logger.error(f"Detalles de test_indices: {test_indices}")
            logger.error(f"Detalles de sequences[test_indices]: {sequences[test_indices]}")
            raise
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader

    def _ensure_numeric(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Asegura que todas las columnas en el DataFrame son numéricas.
        Convierte tipos de datos no numéricos y maneja valores especiales.
        
        Args:
            df: DataFrame a procesar
            columns: Lista de columnas a convertir

        Returns:
            DataFrame con columnas numéricas
        """
        df_copy = df.copy()
        
        # Excluir columnas categóricas (no se deben convertir a numéricas)
        numeric_columns = [col for col in columns if col not in self.categorical_columns]
        
        for col in numeric_columns:
            if col in df_copy.columns:
                try:
                    # Si la columna es numérica, no hacer nada
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        continue
                    
                    # Si es de tipo objeto, intentar extraer números
                    if df_copy[col].dtype == 'object':
                        df_copy[col] = df_copy[col].apply(lambda x: self._extract_numeric(x))
                    
                    # Convertir a float
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"No se pudo convertir columna {col} a numérica: {e}")
                    # Si falla, dejar la columna como está
        
        return df_copy
    
    def _create_categorical_encoders(self, df: pd.DataFrame):
        """Crea encoders para variables categóricas"""
        self.categorical_encoders = {}
        
        for col in self.categorical_columns:
            if col in df.columns:
                # Convertir a string para evitar problemas con tipos de datos
                unique_values = df[col].astype(str).unique()
                # Crear un mapeo de valor a índice
                self.categorical_encoders[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }
                logger.info(f"Creado encoder para columna categórica: {col} con {len(unique_values)} valores únicos")
            else:
                logger.warning(f"Columna categórica '{col}' no encontrada en el dataset")

    def analyze_player_betting_lines(
        self,
        player_data: pd.DataFrame,
        stat_type: str,
        window_size: int = 10,
        confidence_threshold: float = 0.7,
        min_games: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analiza el historial reciente de cada jugador para determinar las líneas de apuestas más adecuadas
        
        Args:
            player_data: DataFrame con datos históricos de jugadores
            stat_type: Tipo de estadística a analizar ('PTS', 'TRB', 'AST', etc.)
            window_size: Tamaño de ventana para análisis reciente
            confidence_threshold: Umbral de confianza para recomendaciones
            min_games: Mínimo de partidos para considerar análisis válido
            
        Returns:
            Diccionario con líneas recomendadas por jugador
        """
        if stat_type not in self.betting_lines:
            raise ValueError(f"Tipo de estadística no soportado: {stat_type}")
        
        # Líneas disponibles para este tipo de estadística
        available_lines = self.betting_lines[stat_type]
        
        # Resultado a devolver
        player_lines = {}
        
        # Agrupar datos por jugador
        for player, player_games in player_data.groupby('Player'):
            # Verificar que haya suficientes partidos
            if len(player_games) < min_games:
                continue
            
            # Ordenar cronológicamente
            player_games = player_games.sort_values(by='Date')
            
            # Extraer la estadística relevante
            if stat_type not in player_games.columns:
                continue
            
            values = player_games[stat_type].values
            
            # Calcular estadísticas básicas
            avg_value = np.mean(values[-window_size:]) if len(values) >= window_size else np.mean(values)
            std_value = np.std(values[-window_size:]) if len(values) >= window_size else np.std(values)
            median_value = np.median(values[-window_size:]) if len(values) >= window_size else np.median(values)
            
            # Calcular frecuencias de superar cada línea
            line_stats = {}
            for line in available_lines:
                over_count = np.sum(values > line)
                over_freq = over_count / len(values)
                
                # Calcular tendencia reciente
                if len(values) >= window_size:
                    recent_values = values[-window_size:]
                    recent_over_count = np.sum(recent_values > line)
                    recent_over_freq = recent_over_count / len(recent_values)
                    trend = recent_over_freq - over_freq
                else:
                    recent_over_freq = over_freq
                    trend = 0
                
                # Calcular confianza para esta línea
                # Alta confianza si la frecuencia está lejos de 0.5 (muy consistente over o under)
                confidence = abs(recent_over_freq - 0.5) * 2  # Mapear [0, 0.5] a [0, 1]
                
                # Determinar si es una buena línea para apostar
                is_good_bet = confidence >= confidence_threshold
                bet_type = "OVER" if recent_over_freq > 0.5 else "UNDER"
                
                line_stats[str(line)] = {
                    'frequency': float(over_freq),
                    'recent_frequency': float(recent_over_freq),
                    'trend': float(trend),
                    'confidence': float(confidence),
                    'is_good_bet': is_good_bet,
                    'bet_type': bet_type
                }
            
            # Encontrar la mejor línea para apostar
            best_line = None
            best_confidence = 0
            best_bet_type = None
            
            for line, stats in line_stats.items():
                if stats['is_good_bet'] and stats['confidence'] > best_confidence:
                    best_line = float(line)
                    best_confidence = stats['confidence']
                    best_bet_type = stats['bet_type']
            
            # Guardar resultados para este jugador
            player_lines[player] = {
                'avg_value': float(avg_value),
                'std_value': float(std_value),
                'median_value': float(median_value),
                'num_games': int(len(values)),
                'line_stats': line_stats,
                'best_line': best_line,
                'best_confidence': float(best_confidence) if best_line is not None else 0.0,
                'best_bet_type': best_bet_type,
                'recommended_lines': [line for line, stats in line_stats.items() if stats['is_good_bet']]
            }
        
        return player_lines

    def generate_betting_recommendations(
        self,
        player_data: pd.DataFrame,
        stat_types: List[str] = None,
        confidence_threshold: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Genera recomendaciones de apuestas para múltiples tipos de estadísticas
        
        Args:
            player_data: DataFrame con datos históricos de jugadores
            stat_types: Lista de tipos de estadísticas a analizar (None para usar todas)
            confidence_threshold: Umbral de confianza para recomendaciones
            
        Returns:
            Diccionario con recomendaciones por tipo de estadística
        """
        if stat_types is None:
            stat_types = list(self.betting_lines.keys())
        
        recommendations = {}
        
        for stat_type in stat_types:
            if stat_type not in self.betting_lines:
                continue
            
            # Analizar líneas para este tipo de estadística
            player_lines = self.analyze_player_betting_lines(
                player_data, 
                stat_type, 
                confidence_threshold=confidence_threshold
            )
            
            # Convertir a lista de recomendaciones
            stat_recommendations = []
            
            for player, data in player_lines.items():
                if data['best_line'] is not None:
                    stat_recommendations.append({
                        'player': player,
                        'line': data['best_line'],
                        'bet_type': data['best_bet_type'],
                        'confidence': data['best_confidence'],
                        'avg_value': data['avg_value'],
                        'std_value': data['std_value'],
                        'num_games': data['num_games'],
                        'edge': abs(data['avg_value'] - data['best_line'])
                    })
            
            # Ordenar por confianza
            stat_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            recommendations[stat_type] = stat_recommendations
        
        return recommendations

    def _extract_numeric(self, x):
        """
        Extrae valores numéricos de diversos tipos de entrada.
        
        Args:
            x: Valor a convertir a numérico
            
        Returns:
            Valor numérico o NaN si no se puede extraer
        """
        # Manejar diferentes tipos de entrada
        if isinstance(x, (pd.DataFrame, pd.Series)):
            # Intentar extraer un valor numérico
            if hasattr(x, 'values') and len(x.values) > 0:
                x = x.values[0]
            else:
                return np.nan
        
        # Convertir a cadena y luego a numérico
        try:
            return pd.to_numeric(str(x).replace(',', '.'), errors='coerce')
        except:
            return np.nan

    def save_feature_mapping(self, features_used: Dict, output_path: str):
        """
        Guarda el mapeo de características utilizadas por target y línea para su uso posterior en los modelos
        
        Args:
            features_used: Diccionario con las características utilizadas por cada target
            output_path: Ruta donde guardar el mapeo de características
        """
        import json
        import os
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Estructura para guardar
        feature_mapping = {
            'model_type': self.model_type,
            'target_column': self.target_columns[0] if self.target_columns else None,
            'features_used': features_used,
            'betting_lines': self.betting_lines,
            'categorical_columns': self.categorical_columns,
            'sequence_length': self.sequence_length
        }
        
        # Guardar como JSON
        with open(output_path, 'w') as f:
            json.dump(feature_mapping, f, indent=4)
        
        logger.info(f"Mapeo de características guardado en: {output_path}")
        
    def load_feature_mapping(self, input_path: str) -> Dict:
        """
        Carga el mapeo de características para usarlo con modelos entrenados
        
        Args:
            input_path: Ruta al archivo de mapeo de características
            
        Returns:
            Diccionario con el mapeo de características por target y línea
        """
        import json
        
        with open(input_path, 'r') as f:
            feature_mapping = json.load(f)
        
        logger.info(f"Mapeo de características cargado desde: {input_path}")
        
        # Actualizar configuración del generador con los valores cargados
        if 'model_type' in feature_mapping:
            self.model_type = feature_mapping['model_type']
        
        if 'target_column' in feature_mapping and feature_mapping['target_column']:
            self.target_columns = [feature_mapping['target_column']]
        
        if 'betting_lines' in feature_mapping:
            self.betting_lines = feature_mapping['betting_lines']
        
        if 'categorical_columns' in feature_mapping:
            self.categorical_columns = feature_mapping['categorical_columns']
        
        if 'sequence_length' in feature_mapping:
            self.sequence_length = feature_mapping['sequence_length']
        
        return feature_mapping

    def generate_and_save_sequences(
        self,
        df: pd.DataFrame,
        output_dir: str,
        min_games: int = 5,
        null_threshold: float = 0.9,
        use_target_specific_features: bool = True
    ):
        """
        Genera secuencias para el target especificado y las guarda junto con su mapeo de características
        
        Args:
            df: DataFrame con los datos
            output_dir: Directorio donde guardar las secuencias y mapeo
            min_games: Número mínimo de juegos para procesar
            null_threshold: Umbral para filtrar características con alto % de nulos
            use_target_specific_features: Si usar características específicas por target
        """
        import os
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar secuencias
        sequences, targets, categorical, line_values, insights = self.generate_sequences(
            df, min_games, null_threshold, use_target_specific_features
        )
        
        if len(sequences) == 0:
            logger.warning("No se generaron secuencias. No se guardará nada.")
            return
        
        # Determinar el nombre base para los archivos
        target_name = self.target_columns[0] if self.target_columns else "unknown_target"
        model_type = self.model_type if self.model_type else "custom_model"
        base_filename = f"{model_type}_{target_name}"
        
        # Guardar secuencias
        sequences_path = os.path.join(output_dir, f"{base_filename}_sequences.npz")
        save_sequences(
            sequences=sequences,
            targets=targets,
            categorical=categorical,
            output_path=sequences_path,
            feature_names=insights.get('feature_info', {}).get('feature_list', []),
            line_values=line_values
        )
        
        # Guardar mapeo de características
        mapping_path = os.path.join(output_dir, f"{base_filename}_feature_mapping.json")
        self.save_feature_mapping(
            features_used=insights.get('feature_info', {}),
            output_path=mapping_path
        )
        
        logger.info(f"Secuencias y mapeo guardados en {output_dir}")
        logger.info(f"Archivos generados: {base_filename}_sequences.npz y {base_filename}_feature_mapping.json")
        
        return sequences_path, mapping_path

class NBASequenceDataset(Dataset):
    """Dataset personalizado para secuencias de NBA"""
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        categorical: np.ndarray
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.categorical = torch.LongTensor(categorical)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Asegurar que se devuelva una tupla, no una lista
        return (
            self.features[idx],
            self.categorical[idx]
        ), self.targets[idx]

class NBASequenceDatasetWithLines(Dataset):
    """Dataset personalizado para secuencias de NBA con múltiples valores de línea"""
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        categorical: np.ndarray,
        betting_lines: Dict[str, List[float]],
        player_ids: Optional[List[str]] = None,
        player_stats: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            features: Array de características secuenciales [N, seq_len, features]
            targets: Array de valores reales [N]
            categorical: Array de variables categóricas [N, num_categorical]
            betting_lines: Diccionario con líneas específicas para este modelo
            player_ids: Lista de IDs de jugadores correspondientes a cada muestra (opcional)
            player_stats: DataFrame con estadísticas de jugadores para personalizar líneas (opcional)
        """
        self.features = torch.FloatTensor(features)
        self.categorical = torch.LongTensor(categorical)
        self.player_ids = player_ids
        
        # Procesar targets para las líneas específicas de este modelo
        self.betting_lines = betting_lines
        self.targets = {}
        
        # Si tenemos información de jugadores, personalizar líneas
        self.use_player_specific_lines = player_ids is not None and player_stats is not None
        self.player_specific_lines = {}
        
        if self.use_player_specific_lines:
            self._generate_player_specific_lines(player_stats)
        
        # Para cada tipo de línea específica del modelo
        for stat_type, lines in betting_lines.items():
            stat_targets = []
            for line in lines:
                # Crear target binario para cada línea (1 si supera la línea, 0 si no)
                binary_target = (targets >= line).astype(np.float32)
                stat_targets.append(binary_target)
            # Concatenar todos los targets para este tipo de estadística
            self.targets[stat_type] = torch.FloatTensor(np.column_stack(stat_targets))
    
    def _generate_player_specific_lines(self, player_stats: pd.DataFrame):
        """
        Genera líneas específicas por jugador basadas en su rendimiento reciente
        """
        for player_id in set(self.player_ids):
            if player_id not in player_stats.index:
                continue
                
            player_data = player_stats.loc[player_id]
            
            # Para cada tipo de estadística, determinar líneas personalizadas
            for stat_type in self.betting_lines.keys():
                if stat_type not in player_data:
                    continue
                    
                # Obtener promedio reciente y desviación estándar
                avg_value = player_data[f"{stat_type}_avg_10"] if f"{stat_type}_avg_10" in player_data else player_data[stat_type]
                std_value = player_data[f"{stat_type}_std_10"] if f"{stat_type}_std_10" in player_data else 1.0
                
                # Generar líneas personalizadas alrededor del promedio
                # Usando las líneas estándar como referencia para elegir la más cercana
                custom_lines = []
                for line in self.betting_lines[stat_type]:
                    # Usar la línea estándar más cercana al rendimiento del jugador
                    # para mantener compatibilidad con el modelo
                    custom_lines.append(line)
                
                # Guardar líneas personalizadas
                if player_id not in self.player_specific_lines:
                    self.player_specific_lines[player_id] = {}
                self.player_specific_lines[player_id][stat_type] = custom_lines
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Retorna un item del dataset con los targets para las líneas específicas del modelo
        
        Returns:
            Tuple con (features, categorical, line_values, targets_dict)
            donde targets_dict contiene solo los targets relevantes para este modelo
        """
        # Si tenemos líneas específicas por jugador, usarlas
        if self.use_player_specific_lines and idx < len(self.player_ids):
            player_id = self.player_ids[idx]
            if player_id in self.player_specific_lines:
                # Construir targets específicos para este jugador
                player_targets = {}
                for stat_type, lines in self.player_specific_lines[player_id].items():
                    if stat_type in self.targets:
                        player_targets[stat_type] = self.targets[stat_type][idx]
                
                # Si tenemos targets específicos, usarlos
                if player_targets:
                    return (
                        self.features[idx],
                        self.categorical[idx],
                        player_targets
                    )
        
        # Si no hay líneas específicas o no se encontró el jugador, usar las generales
        return (
            self.features[idx],
            self.categorical[idx],
            {stat_type: targets[idx] for stat_type, targets in self.targets.items()}
        )

def create_data_loaders(
    sequences: np.ndarray,
    targets: np.ndarray,
    categorical: np.ndarray,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento, validación y prueba
    
    Args:
        sequences: Array de secuencias (batch, seq_len, features)
        targets: Array de objetivos (batch, targets)
        categorical: Array de índices categóricos (batch, cat_features)
        batch_size: Tamaño del batch
        train_split: Proporción de datos para entrenamiento
        val_split: Proporción de datos para validación
        shuffle: Si se deben mezclar los datos
        
    Returns:
        Tuple con (train_loader, val_loader, test_loader)
    """
    # Calcular índices de división
    n_samples = len(sequences)
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    
    # Crear índices para la división
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    # Dividir índices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Crear datasets
    train_dataset = NBASequenceDataset(
        sequences[train_indices],
        targets[train_indices],
        categorical[train_indices]
    )
    
    val_dataset = NBASequenceDataset(
        sequences[val_indices],
        targets[val_indices],
        categorical[val_indices]
    )
    
    test_dataset = NBASequenceDataset(
        sequences[test_indices],
        targets[test_indices],
        categorical[test_indices]
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def save_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    categorical: np.ndarray,
    output_path: str,
    feature_names: List[str] = None,
    line_values: np.ndarray = None
):
    """
    Guarda secuencias generadas en un archivo .npz para uso posterior.
    
    Args:
        sequences: Array de secuencias
        targets: Array de targets
        categorical: Array de valores categóricos
        output_path: Ruta donde guardar el archivo
        feature_names: Nombres de las características (opcional)
        line_values: Valores de línea para over/under (opcional)
    """
    # Preparar diccionario con los arrays
    save_dict = {
        'sequences': sequences,
        'targets': targets,
        'categorical': categorical
    }
    
    # Añadir line_values si se proporcionan
    if line_values is not None:
        save_dict['line_values'] = line_values
        
    # Añadir feature_names si se proporcionan
    if feature_names is not None:
        # Guardar como array de strings
        save_dict['feature_names'] = np.array(feature_names, dtype=object)
    
    # Guardar en formato npz
    np.savez_compressed(output_path, **save_dict)
    print(f"Secuencias guardadas en: {output_path}")

def load_sequences(input_path: str) -> Dict[str, np.ndarray]:
    """
    Carga secuencias desde un archivo .npz guardado.
    
    Args:
        input_path: Ruta al archivo .npz
        
    Returns:
        Dict con los arrays cargados (sequences, targets, categorical, etc.)
    """
    try:
        # Cargar archivo npz
        data = np.load(input_path, allow_pickle=True)
        
        # Convertir a diccionario
        result = {}
        for key in data.files:
            # Si es feature_names, convertir a lista de strings
            if key == 'feature_names':
                result[key] = data[key].tolist()
            else:
                result[key] = data[key]
                
        print(f"Secuencias cargadas desde: {input_path}")
        
        # Mostrar información sobre los datos cargados
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}: forma {value.shape}, tipo {value.dtype}")
            else:
                print(f"  - {key}: {type(value)}")
                
        return result
        
    except Exception as e:
        print(f"Error cargando secuencias desde {input_path}: {e}")
        raise

def create_data_loaders_from_splits(
    sequences: np.ndarray,
    targets: np.ndarray,
    categorical: np.ndarray,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    shuffle: bool = True,
    line_values: np.ndarray = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Versión actualizada de create_data_loaders con soporte para line_values.
    
    Args:
        sequences: Array de secuencias
        targets: Array de targets
        categorical: Array de valores categóricos
        batch_size: Tamaño del batch
        test_size: Proporción del conjunto de prueba
        val_size: Proporción del conjunto de validación
        shuffle: Si se deben mezclar los datos
        line_values: Valores de línea para over/under (opcional)
        
    Returns:
        Tuple con (train_loader, val_loader, test_loader)
    """
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        sequences, targets, categorical, test_size=test_size, random_state=42, shuffle=shuffle
    )
    
    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val, cat_train, cat_val = train_test_split(
        X_train, y_train, cat_train, 
        test_size=val_size/(1-test_size),  # Ajustar para que sea proporcional al tamaño de entrenamiento
        random_state=42, 
        shuffle=shuffle
    )
    
    # Crear datasets basados en si se proporcionaron valores de línea
    if line_values is not None:
        # También dividir line_values en los conjuntos correspondientes
        line_train, line_test = train_test_split(
            line_values, test_size=test_size, random_state=42, shuffle=shuffle
        )
        
        line_train, line_val = train_test_split(
            line_train, 
            test_size=val_size/(1-test_size),
            random_state=42, 
            shuffle=shuffle
        )
        
        # Usar el dataset con soporte para valores de línea
        # Crear un diccionario de líneas de apuestas simplificado
        first_key = next(iter(BETTING_LINES))
        simplified_betting_lines = {first_key: BETTING_LINES[first_key]}
        
        # Crear datasets con el diccionario simplificado
        train_dataset = NBASequenceDatasetWithLines(
            X_train, y_train, cat_train, simplified_betting_lines, None, None
        )
        
        val_dataset = NBASequenceDatasetWithLines(
            X_val, y_val, cat_val, simplified_betting_lines, None, None
        )
        
        test_dataset = NBASequenceDatasetWithLines(
            X_test, y_test, cat_test, simplified_betting_lines, None, None
        )
    else:
        # Usar el dataset sin valores de línea
        train_dataset = NBASequenceDataset(
            X_train, y_train, cat_train
        )
        
        val_dataset = NBASequenceDataset(
            X_val, y_val, cat_val
        )
        
        test_dataset = NBASequenceDataset(
            X_test, y_test, cat_test
        )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def prepare_target_specific_sequences(
    df: pd.DataFrame, 
    target: str,
    model_type: str = None,
    output_dir: str = "data/sequences",
    sequence_length: int = 10,
    min_games: int = 5,
    confidence_threshold: float = 0.85,
    min_historical_accuracy: float = 0.90,
    null_threshold: float = 0.9
) -> Tuple[str, str]:
    """
    Función para preparar secuencias específicas para un target con FeaturesSelector
    
    Args:
        df: DataFrame con los datos
        target: Target específico ('PTS', 'TRB', 'AST', '3P', 'Win', etc.)
        model_type: Tipo de modelo (opcional)
        output_dir: Directorio donde guardar las secuencias
        sequence_length: Longitud de las secuencias
        min_games: Número mínimo de juegos para procesar un jugador/equipo
        confidence_threshold: Umbral de confianza para líneas de apuestas
        min_historical_accuracy: Precisión histórica mínima para líneas
        null_threshold: Umbral para filtrar características con alto % de nulos
        
    Returns:
        Tuple con (ruta_secuencias, ruta_mapeo)
    """
    import os
    
    # Determinar automáticamente el tipo de modelo si no se especifica
    if model_type is None:
        if target in ['Win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']:
            if target == 'Win':
                model_type = 'win_predictor'
            elif target == 'Total_Points_Over_Under':
                model_type = 'total_points_predictor'
            elif target == 'Team_Points_Over_Under':
                model_type = 'team_points_predictor'
        else:
            # Modelos de jugador
            model_type = f"{target.lower()}_predictor"
    
    logger.info(f"Preparando secuencias para target '{target}' con modelo '{model_type}'")
    
    # Crear el generador de secuencias
    generator = SequenceGenerator(
        sequence_length=sequence_length,
        target_columns=[target],
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        min_historical_accuracy=min_historical_accuracy
    )
    
    # Crear subdirectorio específico para este target
    target_dir = os.path.join(output_dir, target.lower())
    os.makedirs(target_dir, exist_ok=True)
    
    # Generar y guardar secuencias
    sequences_path, mapping_path = generator.generate_and_save_sequences(
        df=df,
        output_dir=target_dir,
        min_games=min_games,
        null_threshold=null_threshold
    )
    
    logger.info(f"Secuencias para '{target}' generadas y guardadas en '{target_dir}'")
    return sequences_path, mapping_path

def prepare_all_target_sequences(
    df: pd.DataFrame,
    targets: List[str] = None,
    output_dir: str = "data/sequences",
    sequence_length: int = 10,
    min_games: int = 5,
    confidence_threshold: float = 0.85,
    min_historical_accuracy: float = 0.90,
    null_threshold: float = 0.9
) -> Dict[str, Tuple[str, str]]:
    """
    Genera secuencias para múltiples targets
    
    Args:
        df: DataFrame con los datos
        targets: Lista de targets a procesar (si es None, usa todos los disponibles)
        output_dir: Directorio base donde guardar las secuencias
        sequence_length: Longitud de las secuencias
        min_games: Número mínimo de juegos para procesar un jugador/equipo
        confidence_threshold: Umbral de confianza para líneas de apuestas
        min_historical_accuracy: Precisión histórica mínima para líneas
        null_threshold: Umbral para filtrar características con alto % de nulos
        
    Returns:
        Diccionario con rutas de secuencias por target
    """
    import os
    
    # Si no se especifican targets, usar todos los disponibles en BETTING_LINES
    if targets is None:
        targets = list(BETTING_LINES.keys())
    
    # Crear directorio base si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Procesar cada target por separado
    for target in targets:
        try:
            logger.info(f"Procesando target: {target}")
            sequences_path, mapping_path = prepare_target_specific_sequences(
                df=df,
                target=target,
                output_dir=output_dir,
                sequence_length=sequence_length,
                min_games=min_games,
                confidence_threshold=confidence_threshold,
                min_historical_accuracy=min_historical_accuracy,
                null_threshold=null_threshold
            )
            
            results[target] = (sequences_path, mapping_path)
            logger.info(f"Target {target} procesado exitosamente")
        except Exception as e:
            logger.error(f"Error procesando target {target}: {e}")
            continue
    
    logger.info(f"Procesados {len(results)}/{len(targets)} targets exitosamente")
    return results


