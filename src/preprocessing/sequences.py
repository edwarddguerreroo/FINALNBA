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

# Configurar logger
logger = logging.getLogger(__name__)

# Configurar pandas para evitar el warning de downcasting
pd.set_option('future.no_silent_downcasting', True)

# Líneas fijas 
BETTING_LINES = {
    'PTS': [10, 15, 20, 25, 30, 35],
    'TRB': [4, 6, 8, 9, 10],
    'AST': [4, 6, 8, 9, 10],
    '3P': [1, 2, 3, 4],
    'Win': [0.5],  # Para predicciones binarias de victoria
    'Total_Points_Over_Under': [190, 195, 200, 205, 210, 215, 220, 225, 230, 235],  # Total de puntos en juego
    'Team_Points_Over_Under': [90, 95, 100, 105, 110, 115, 120, 125, 130],  # Puntos de equipo
    'Double_Double': [0.5],  # Para predicciones binarias de doble-doble
    'Triple_Double': [0.5]   # Para predicciones binarias de triple-doble
}

# Mapeo de modelos a sus líneas correspondientes
MODEL_LINES = {
    'pts_predictor': ['PTS'],
    'trb_predictor': ['TRB'],
    'ast_predictor': ['AST'],
    '3p_predictor': ['3P'],
    'win_predictor': ['Win'],
    'total_points_predictor': ['Total_Points_Over_Under'],
    'team_points_predictor': ['Team_Points_Over_Under'],
    'double_double_predictor': ['Double_Double'],
    'triple_double_predictor': ['Triple_Double']
}

class SequenceGenerator:
    """
    Generador de secuencias temporales para modelos predictivos
    """
    def __init__(
        self,
        sequence_length: int = 10,
        target_columns: List[str] = ['PTS', 'TRB', 'AST'],
        feature_columns: List[str] = None,
        categorical_columns: List[str] = ['Pos', 'Team', 'Opp'],
        model_type: str = None,  # Nuevo parámetro para especificar tipo de modelo
        confidence_threshold: float = 0.85,
        min_historical_accuracy: float = 0.93,
        min_samples: int = 15
    ):
        self.sequence_length = sequence_length
        self.model_type = model_type
        
        # Determinar líneas específicas para este modelo
        if model_type and model_type in MODEL_LINES:
            self.target_columns = MODEL_LINES[model_type]
            self.betting_lines = {stat: BETTING_LINES[stat] for stat in MODEL_LINES[model_type]}
        else:
            self.target_columns = target_columns
            self.betting_lines = {stat: BETTING_LINES[stat] for stat in target_columns if stat in BETTING_LINES}
        
        print(f"Modelo: {model_type if model_type else 'general'}")
        print(f"Líneas de apuestas configuradas: {self.betting_lines}")
        
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
        null_threshold: float = 0.9,  # Umbral para filtrar características con muchos nulos
        use_target_specific_features: bool = True  # Usar características específicas por tipo de predicción
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Genera secuencias para predicciones individuales por línea
        
        Args:
            df: DataFrame con datos procesados
            min_games: Número mínimo de juegos para procesar un jugador
            null_threshold: Umbral para filtrar características con alto % de nulos
            use_target_specific_features: Si usar subconjuntos específicos de características por target
            
        Returns:
            Tuple con (secuencias, targets, categóricos, valores_línea, insights)
        """
        # Asegurar tipos de datos correctos
        df = self._ensure_numeric(df, self.feature_columns + self.target_columns)
        
        # Ordenar por jugador y fecha
        if 'Date' in df.columns:
            df = df.sort_values(['Player', 'Date'])
        
        # Crear codificadores categóricos si no existen
        if not self.categorical_encoders:
            self._create_categorical_encoders(df)
        
        # Análisis y gestión de valores nulos
        null_analysis = {}
        for col in self.feature_columns:
            if col in df.columns:
                def safe_null_percentage(series):
                    try:
                        # Manejar Series anidadas o complejas
                        if isinstance(series.iloc[0], (pd.Series, pd.DataFrame)):
                            # Intentar aplanar Series anidadas
                            series = series.apply(lambda x: x.values[0] if hasattr(x, 'values') else np.nan)
                        
                        # Calcular porcentaje de nulos
                        total_count = len(series)
                        null_count = series.isnull().sum()
                        
                        # Asegurar que null_count sea un escalar
                        if isinstance(null_count, pd.Series):
                            null_count = null_count.iloc[0] if len(null_count) > 0 else 0
                            
                        return float(null_count) / total_count if total_count > 0 else 0.0
                    except Exception as e:
                        print(f"Warning: Could not calculate null percentage. Error: {e}")
                        return 1.0
                
                null_pct = safe_null_percentage(df[col])
                null_analysis[col] = null_pct
        
        # Ordenar características por % de nulos (de menor a mayor)
        sorted_features = sorted(null_analysis.items(), key=lambda x: x[1])
        
        # Filtrar características con alto porcentaje de valores nulos
        usable_features = [col for col, pct in sorted_features if pct < null_threshold and col in df.columns]
        
        print(f"Usando {len(usable_features)}/{len(self.feature_columns)} características (ignorando {len(self.feature_columns) - len(usable_features)} con >{null_threshold*100:.0f}% de valores nulos)")
        
        # Importar las características específicas por target desde feature_engineering
        try:
            from src.preprocessing.feature_engineering import FeatureEngineering
            fe = FeatureEngineering()
            fe_target_features = fe.get_target_specific_features()
            
            # Mapeo de características más relevantes por tipo de predicción
            target_specific_features = {}
            
            # Usar las características definidas en feature_engineering si están disponibles
            for target, features in fe_target_features.items():
                # Filtrar solo las características que están disponibles en nuestro conjunto de datos
                target_specific_features[target] = [col for col in features if col in usable_features]
                
            # Caso especial: Manejar Win/is_win como el mismo target
            if 'Win' in self.target_columns and 'is_win' in target_specific_features and 'Win' not in target_specific_features:
                target_specific_features['Win'] = target_specific_features['is_win']
                print(f"Usando características de 'is_win' para el target 'Win'")
            elif 'is_win' in self.target_columns and 'Win' in target_specific_features and 'is_win' not in target_specific_features:
                target_specific_features['is_win'] = target_specific_features['Win']
                print(f"Usando características de 'Win' para el target 'is_win'")
                
            # Manejar targets de puntos totales (nombres diferentes en features vs. targets)    
            if 'Total_Points_Over_Under' in self.target_columns and 'Total_Points' in target_specific_features:
                target_specific_features['Total_Points_Over_Under'] = target_specific_features['Total_Points']
                print(f"Usando características de 'Total_Points' para el target 'Total_Points_Over_Under'")
                
            if 'Team_Points_Over_Under' in self.target_columns and 'Team_Points' in target_specific_features:
                target_specific_features['Team_Points_Over_Under'] = target_specific_features['Team_Points']
                print(f"Usando características de 'Team_Points' para el target 'Team_Points_Over_Under'")
                
            # Asegurar que tenemos entradas para todos los targets que necesitamos
            for target in self.target_columns:
                if target not in target_specific_features or len(target_specific_features[target]) < 10:
                    # Si no tenemos suficientes características específicas, usar heurística basada en nombres
                    if target == 'PTS':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['PTS', 'FG', '3P', 'FT', 'Usage', 'Offensive', 'Scoring'])]
                    elif target == 'TRB':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['TRB', 'ORB', 'DRB', 'Rebound', 'Height', 'Physical'])]
                    elif target == 'AST':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['AST', 'Playmaker', 'TOV', 'AST_TO', 'Pass'])]
                    elif target == '3P':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['3P', 'shooting', 'Shot', 'outside', 'perimeter'])]
                    elif target == 'Win':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['Win', 'Team', 'Rating', 'Net', 'Point_Diff'])]
                    elif target == 'Double_Double':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['PTS', 'TRB', 'AST', 'Double'])]
                    elif target == 'Triple_Double':
                        target_specific_features[target] = [col for col in usable_features if any(x in col for x in ['PTS', 'TRB', 'AST', 'Triple'])]
                    else:
                        # Para cualquier otro target, usar todas las características disponibles
                        target_specific_features[target] = usable_features
            
            # Mostrar un resumen de las características específicas por target
            print(f"\nCaracterísticas específicas por target cargadas desde feature_engineering.py:")
            for target, features in target_specific_features.items():
                if target in self.target_columns:
                    # Mostrar solo para los targets que estamos usando
                    feature_categories = {}
                    
                    # Categorizar las características para una mejor visualización
                    for feature in features:
                        if 'avg' in feature or 'mean' in feature:
                            category = "Promedios"
                        elif 'trend' in feature or 'momentum' in feature:
                            category = "Tendencias"
                        elif 'prob' in feature:
                            category = "Probabilidades"
                        elif 'consistency' in feature:
                            category = "Consistencia"
                        elif 'PhysPerf' in feature or 'Physical' in feature:
                            category = "Físico"
                        elif 'vs_team' in feature or 'Contribution' in feature:
                            category = "Contribución al equipo"
                        elif 'matchup' in feature or 'opp_' in feature:
                            category = "Matchup"
                        elif 'in_close_games' in feature or 'clutch' in feature:
                            category = "Situación de juego"
                        elif 'b2b_impact' in feature or 'Fatigue' in feature:
                            category = "Fatiga"
                        else:
                            category = "Otras"
                            
                        if category not in feature_categories:
                            feature_categories[category] = []
                        feature_categories[category].append(feature)
                    
                    # Mostrar un resumen por categoría
                    print(f"\n  Target: {target} - {len(features)} características")
                    for category, cat_features in feature_categories.items():
                        print(f"    - {category}: {len(cat_features)} características")
                        # Mostrar hasta 3 ejemplos de cada categoría
                        if len(cat_features) > 0:
                            examples = cat_features[:min(3, len(cat_features))]
                            print(f"      Ejemplos: {', '.join(examples)}")
                    
                    # Mostrar las 5 características más importantes (si hay suficientes)
                    if len(features) >= 5:
                        key_features = [f for f in features if target in f][:5]
                        if len(key_features) < 5:
                            # Complementar con otras características si no hay suficientes específicas
                            key_features.extend(features[:5-len(key_features)])
                        print(f"    Top 5 características clave: {', '.join(key_features)}")
            print("\n")

        except Exception as e:
            print(f"No se pudieron cargar características específicas desde feature_engineering: {e}")
            # Fallback a la selección basada en patrones de nombres
            target_specific_features = {
                'PTS': [col for col in usable_features if any(x in col for x in ['PTS', 'FG', '3P', 'FT', 'Usage', 'Offensive', 'Scoring'])],
                'TRB': [col for col in usable_features if any(x in col for x in ['TRB', 'ORB', 'DRB', 'Rebound', 'Height', 'Physical'])],
                'AST': [col for col in usable_features if any(x in col for x in ['AST', 'Playmaker', 'TOV', 'AST_TO', 'Pass'])],
                '3P': [col for col in usable_features if any(x in col for x in ['3P', 'shooting', 'Shot', 'outside', 'perimeter'])],
                'Win': [col for col in usable_features if any(x in col for x in ['Win', 'Team', 'Rating', 'Net', 'Point_Diff'])],
                'Double_Double': [col for col in usable_features if any(x in col for x in ['PTS', 'TRB', 'AST', 'Double'])]
            }
        
        # Asegurar un conjunto mínimo de características para cada tipo
        for key in target_specific_features:
            if len(target_specific_features[key]) < 30:  # Asegurar al menos 30 características
                # Complementar con características generales más importantes
                general_features = [col for col in usable_features if col not in target_specific_features[key]]
                target_specific_features[key].extend(general_features[:30-len(target_specific_features[key])])
        
        # Rellenar valores nulos en características usables
        for col in usable_features:
            # Manejar tipos de datos complejos
            try:
                # Intentar convertir a numérico, manejando diferentes tipos
                if isinstance(df[col].iloc[0], (list, np.ndarray)):
                    # Si es una lista o array, intentar convertir el primer elemento
                    df[col] = df[col].apply(lambda x: pd.to_numeric(x[0] if len(x) > 0 else 0, errors='coerce'))
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Error convirtiendo columna {col}: {e}")
                df[col] = 0
            
            # Usar .sum() > 0 para evitar ambigüedad en Series
            null_count = df[col].isnull().sum()
            # Asegurar que null_count sea un escalar
            if isinstance(null_count, pd.Series):
                null_count = null_count.iloc[0] if len(null_count) > 0 else 0
                
            if null_count > 0:
                # Para estadísticas base, usar el valor base si existe
                if '_avg_' in col or '_std_' in col or '_trend_' in col:
                    base_stat = col.split('_')[0]
                    if base_stat in df.columns:
                        # Forma recomendada sin usar inplace=True
                        base_mean = pd.to_numeric(df[base_stat], errors='coerce').mean()
                        df[col] = df[col].fillna(base_mean)
                    else:
                        df[col] = df[col].fillna(0)
                # Para porcentajes, usar la media
                elif '%' in col:
                    df[col] = df[col].fillna(df[col].mean())
                # Para características de predicción
                elif 'prob' in col or 'consistency' in col:
                    df[col] = df[col].fillna(0.5)  # Valor neutral
                # Para el resto, usar 0
                else:
                    df[col] = df[col].fillna(0)
        
        sequences = []
        targets = []
        categorical = []
        line_values = []
        betting_insights = {}
        
        # Determinar si estamos generando secuencias para un modelo de equipo o de jugador
        team_level_targets = ['Win', 'is_win', 'Total_Points_Over_Under', 'Team_Points_Over_Under']
        is_team_model = any(target in team_level_targets for target in self.target_columns)
        
        if is_team_model:
            # GENERACIÓN DE SECUENCIAS A NIVEL DE EQUIPO
            print(f"\nGenerando secuencias a nivel de EQUIPO para {self.target_columns}")
            
            # Para modelos de equipo, agrupar por equipo y fecha
            if 'Team' in df.columns and 'Date' in df.columns:
                # Crear un DataFrame a nivel de equipo
                team_df = df.groupby(['Team', 'Date']).agg({
                    # Agregar características relevantes a nivel de equipo
                    'Win': 'first',  # Victoria o derrota
                    'team_score': 'mean',  # Puntos del equipo
                    'opp_score': 'mean',  # Puntos del oponente
                    'total_score': 'mean',  # Total de puntos en el partido
                    # Agregar otras métricas de equipo que puedan ser útiles
                }).reset_index()
                
                # Procesar cada equipo
                teams_total = team_df['Team'].nunique()
                print(f"Procesando secuencias para {teams_total} equipos...")
                
                # Crear la barra de progreso para el procesamiento de equipos
                team_groups = list(team_df.groupby('Team'))
                for team_idx, (team, team_data) in enumerate(tqdm(team_groups, desc="Generando secuencias de equipo", unit="equipo")):
                    if len(team_data) < min_games:
                        continue
                    
                    # Ordenar por fecha
                    team_data = team_data.sort_values('Date')
                    
                    # Analizar cada línea para este equipo
                    for stat in self.target_columns:
                        if stat not in self.betting_lines:
                            continue
                        
                        # Determinar características a usar para este tipo de predicción
                        if use_target_specific_features and stat in target_specific_features:
                            # Usar conjunto específico de características para este target
                            stat_features = [col for col in target_specific_features[stat] if col in team_data.columns]
                            
                            # Si no hay suficientes características, usar un conjunto básico garantizado
                            if len(stat_features) < 5:
                                basic_team_features = [col for col in ['Win', 'is_win', 'team_score', 'opp_score', 'total_score', 
                                                                      'is_home', 'days_rest', 'point_diff'] 
                                                              if col in team_data.columns]
                                stat_features.extend([col for col in basic_team_features if col not in stat_features])
                                print(f"Usando {len(stat_features)} características básicas para {stat} (incluyendo características garantizadas)")
                        else:
                            # Usar todas las características disponibles en team_data
                            stat_features = [col for col in usable_features if col in team_data.columns]
                        
                        # Analizar cada línea disponible para esta estadística
                        for line in self.betting_lines[stat]:
                            # Para modelos de equipo, no necesitamos análisis histórico tan complejo
                            # Simplemente generamos secuencias para todas las líneas
                            
                            # Crear secuencias para este equipo y línea
                            for i in range(len(team_data) - self.sequence_length):
                                sequence = team_data.iloc[i:i+self.sequence_length]
                                target_row = team_data.iloc[i+self.sequence_length]
                                
                                # Extraer características disponibles
                                feature_sequence = []
                                for col in stat_features:
                                    if col in sequence.columns:
                                        try:
                                            # Usar .values para evitar advertencias de indexación
                                            values = sequence[col].values
                                            
                                            # Verificar que los valores sean numéricos
                                            if not np.issubdtype(values.dtype, np.number):
                                                # Intentar convertir a numérico
                                                try:
                                                    values = pd.to_numeric(sequence[col], errors='coerce').fillna(0).values
                                                except:
                                                    # Si falla, usar ceros
                                                    values = np.zeros(self.sequence_length)
                                            
                                            # Verificar que sea un array 1D simple
                                            if any(isinstance(v, (list, np.ndarray, pd.Series, pd.DataFrame)) for v in values if v is not None):
                                                # Extraer valores numéricos de arrays anidados
                                                numeric_values = []
                                                for v in values:
                                                    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                                                        # Tomar el primer elemento numérico
                                                        if isinstance(v[0], (int, float)):
                                                            numeric_values.append(float(v[0]))
                                                        else:
                                                            numeric_values.append(0.0)
                                                    elif isinstance(v, (pd.Series, pd.DataFrame)):
                                                        # Extraer el primer valor numérico
                                                        if len(v) > 0:
                                                            try:
                                                                numeric_values.append(float(v.iloc[0]))
                                                            except:
                                                                numeric_values.append(0.0)
                                                    elif isinstance(v, (int, float)):
                                                        numeric_values.append(float(v))
                                                    else:
                                                        numeric_values.append(0.0)
                                                
                                                # Asegurar que tengamos la longitud correcta
                                                if len(numeric_values) == self.sequence_length:
                                                    values = np.array(numeric_values, dtype=np.float32)
                                                else:
                                                    # Si la longitud no coincide, usar ceros
                                                    print(f"Advertencia: Longitud incorrecta después de procesar {col}: {len(numeric_values)} != {self.sequence_length}")
                                                    values = np.zeros(self.sequence_length, dtype=np.float32)
                                            
                                            # Asegurar que los valores son un array unidimensional
                                            values = np.asarray(values, dtype=np.float32).flatten()
                                            
                                            # Truncar o rellenar para tener exactamente la longitud deseada
                                            if len(values) > self.sequence_length:
                                                values = values[:self.sequence_length]
                                            elif len(values) < self.sequence_length:
                                                # Rellenar con ceros
                                                values = np.pad(values, (0, self.sequence_length - len(values)), 'constant')
                                            
                                            # Asegurar que no hay valores NaN o infinitos
                                            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                                            
                                            feature_sequence.append(values)
                                        except Exception as e:
                                            print(f"Error procesando columna {col}: {e}")
                                            feature_sequence.append(np.zeros(self.sequence_length, dtype=np.float32))
                                    else:
                                        # Si la característica no está disponible, usar ceros
                                        feature_sequence.append(np.zeros(self.sequence_length, dtype=np.float32))
                                
                                try:
                                    # Verificar que todas las secuencias tienen la misma longitud
                                    if not all(len(fs) == self.sequence_length for fs in feature_sequence):
                                        # Normalizar longitudes
                                        for i, fs in enumerate(feature_sequence):
                                            if len(fs) != self.sequence_length:
                                                feature_sequence[i] = np.zeros(self.sequence_length, dtype=np.float32)
                                    
                                    # Convertir a array con manejo de errores
                                    feature_sequence = np.array(feature_sequence, dtype=np.float32).T
                                except ValueError as e:
                                    print(f"Error creando array para secuencia: {e}")
                                    # Crear un array de ceros como fallback
                                    feature_sequence = np.zeros((self.sequence_length, len(stat_features)), dtype=np.float32)
                                
                                # Generar target binario para esta línea específica
                                if stat == 'Win':
                                    target_val = 1 if target_row['Win'] else 0
                                    is_over = target_val > line
                                elif stat == 'Total_Points_Over_Under':
                                    target_val = target_row['total_score']
                                    is_over = target_val > line
                                elif stat == 'Team_Points_Over_Under':
                                    target_val = target_row['team_score']
                                    is_over = target_val > line
                                else:
                                    # Para cualquier otro stat a nivel de equipo
                                    if stat in target_row:
                                        target_val = target_row[stat]
                                        is_over = target_val > line
                                    else:
                                        continue  # No tenemos este stat, saltar
                                
                                # Extraer categóricos (como el oponente)
                                cat_values = []
                                for col in self.categorical_columns:
                                    if col in target_row:
                                        try:
                                            cat_idx = self.categorical_encoders.get(col, {}).get(target_row[col], 0)
                                            cat_values.append(cat_idx)
                                        except Exception:
                                            cat_values.append(0)
                                    else:
                                        cat_values.append(0)
                                
                                # Asegurar que cat_values tenga la longitud correcta
                                if len(cat_values) < len(self.categorical_columns):
                                    cat_values.extend([0] * (len(self.categorical_columns) - len(cat_values)))
                                elif len(cat_values) > len(self.categorical_columns):
                                    cat_values = cat_values[:len(self.categorical_columns)]
                                
                                # Guardar la secuencia
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
                                    'historical_accuracy': 0.8,  # Valor simplificado para equipos
                                    'confidence': 0.8,  # Valor simplificado para equipos
                                    'prediction': 'OVER' if is_over else 'UNDER'
                                }
        else:
            # GENERACIÓN DE SECUENCIAS A NIVEL DE JUGADOR (código original)
            print(f"\nGenerando secuencias a nivel de JUGADOR para {self.target_columns}")
        
        # Generar secuencias por jugador
        players_total = df['Player'].nunique()
        print(f"Procesando secuencias para {players_total} jugadores...")
        
        # Crear la barra de progreso para el procesamiento de jugadores
        player_groups = list(df.groupby('Player'))
        for player_idx, (player, player_data) in enumerate(tqdm(player_groups, desc="Generando secuencias de jugador", unit="jugador")):
            if len(player_data) < min_games:
                continue
                
            player_insights = {}
            
            # Analizar cada línea individualmente
            for stat in self.target_columns:
                if stat not in self.betting_lines:
                    continue
                
                # Determinar características a usar para este tipo de predicción
                if use_target_specific_features and stat in target_specific_features:
                    # Usar conjunto específico de características para este target
                    stat_features = target_specific_features[stat]
                    # Reducir el volumen de logs imprimiendo solo para el primer jugador y cada 10 jugadores
                    if player_idx == 0 or player_idx % 10 == 0:
                        print(f"Usando {len(stat_features)} características específicas para {stat}")
                else:
                    # Usar todas las características disponibles
                    stat_features = usable_features
                
                # Analizar cada línea disponible para esta estadística
                for line in self.betting_lines[stat]:
                    historical_accuracy, confidence = self._analyze_historical_accuracy(
                        player_data, stat, line
                    )
                    
                    # Si cumple con los umbrales de confianza
                    if (historical_accuracy >= self.min_historical_accuracy and 
                        confidence >= self.confidence_threshold):
                        
                        # Crear secuencias específicas para esta línea
                        for i in range(len(player_data) - self.sequence_length):
                            sequence = player_data.iloc[i:i+self.sequence_length]
                            target_row = player_data.iloc[i+self.sequence_length]
                            
                            # Extraer características disponibles (específicas o todas)
                            feature_sequence = []
                            for col in stat_features:
                                if col in sequence.columns:
                                    try:
                                    # Usar .values para evitar advertencias de indexación
                                        values = sequence[col].values
                                        
                                        # Verificar que los valores sean numéricos
                                        if not np.issubdtype(values.dtype, np.number):
                                            # Intentar convertir a numérico
                                            try:
                                                values = pd.to_numeric(sequence[col], errors='coerce').fillna(0).values
                                            except:
                                                # Si falla, usar ceros
                                                values = np.zeros(self.sequence_length)
                                        
                                        # Verificar que sea un array 1D simple
                                        if any(isinstance(v, (list, np.ndarray, pd.Series, pd.DataFrame)) for v in values if v is not None):
                                            # Extraer valores numéricos de arrays anidados
                                            numeric_values = []
                                            for v in values:
                                                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                                                    # Tomar el primer elemento numérico
                                                    if isinstance(v[0], (int, float)):
                                                        numeric_values.append(float(v[0]))
                                                    else:
                                                        numeric_values.append(0.0)
                                                elif isinstance(v, (pd.Series, pd.DataFrame)):
                                                    # Extraer el primer valor numérico
                                                    if len(v) > 0:
                                                        try:
                                                            numeric_values.append(float(v.iloc[0]))
                                                        except:
                                                            numeric_values.append(0.0)
                                                    else:
                                                        numeric_values.append(0.0)
                                                elif isinstance(v, (int, float)):
                                                    numeric_values.append(float(v))
                                                else:
                                                    numeric_values.append(0.0)
                                            
                                            # Asegurar que tengamos la longitud correcta
                                            if len(numeric_values) == self.sequence_length:
                                                values = np.array(numeric_values, dtype=np.float32)
                                            else:
                                                # Si la longitud no coincide, usar ceros
                                                print(f"Advertencia: Longitud incorrecta después de procesar {col}: {len(numeric_values)} != {self.sequence_length}")
                                                values = np.zeros(self.sequence_length, dtype=np.float32)
                                        
                                        # Asegurar que los valores son un array unidimensional
                                        values = np.asarray(values, dtype=np.float32).flatten()
                                        
                                        # Truncar o rellenar para tener exactamente la longitud deseada
                                        if len(values) > self.sequence_length:
                                            values = values[:self.sequence_length]
                                        elif len(values) < self.sequence_length:
                                            # Rellenar con ceros
                                            values = np.pad(values, (0, self.sequence_length - len(values)), 'constant')
                                        
                                        # Asegurar que no hay valores NaN o infinitos
                                        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                                        
                                        feature_sequence.append(values)
                                    except Exception as e:
                                        print(f"Error procesando columna {col}: {e}")
                                        feature_sequence.append(np.zeros(self.sequence_length, dtype=np.float32))
                                else:
                                    # Si la característica no está disponible, usar ceros
                                    feature_sequence.append(np.zeros(self.sequence_length, dtype=np.float32))
                            
                            try:
                                # Verificar que todas las secuencias tienen la misma longitud
                                if not all(len(fs) == self.sequence_length for fs in feature_sequence):
                                    # Normalizar longitudes
                                    for i, fs in enumerate(feature_sequence):
                                        if len(fs) != self.sequence_length:
                                            feature_sequence[i] = np.zeros(self.sequence_length, dtype=np.float32)
                                
                                # Convertir a array con manejo de errores
                                feature_sequence = np.array(feature_sequence, dtype=np.float32).T
                            except ValueError as e:
                                print(f"Error creando array para secuencia: {e}")
                                # Crear un array de ceros como fallback
                                feature_sequence = np.zeros((self.sequence_length, len(stat_features)), dtype=np.float32)
                            
                            # Generar target binario para esta línea específica
                            target_val = target_row[stat]
                            is_over = target_val > line
                            
                            # Extraer categóricos
                            cat_values = []
                            for col in self.categorical_columns:
                                if col in target_row:
                                    try:
                                        cat_idx = self.categorical_encoders.get(col, {}).get(target_row[col], 0)
                                        cat_values.append(cat_idx)
                                    except Exception as e:
                                        print(f"Error al obtener índice categórico para {col}: {e}")
                                        cat_values.append(0)
                                else:
                                    cat_values.append(0)
                            
                            # Asegurar que cat_values tenga la longitud correcta
                            if len(cat_values) < len(self.categorical_columns):
                                cat_values.extend([0] * (len(self.categorical_columns) - len(cat_values)))
                            elif len(cat_values) > len(self.categorical_columns):
                                cat_values = cat_values[:len(self.categorical_columns)]
                            
                            # Guardar la secuencia
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
                                'prediction': 'OVER' if is_over else 'UNDER'
                            }
        
        # Convertir listas a arrays
        if not sequences:
            # No hay suficientes secuencias que cumplan con los criterios
            print("ADVERTENCIA: No se generaron secuencias que cumplan con los criterios de confianza")
            # Asegurar que tengamos al menos algunas características básicas
            if not usable_features:
                # Si no hay características específicas, usar características básicas
                # Verificar qué columnas tenemos disponibles
                existing_columns = list(df.columns)
                
                # Características básicas que se deben incluir para casi toda predicción
                basic_features = ['team_score', 'opp_score', 'Win', 'is_home']
                empty_feature_dim = len(basic_features)
            else:
                # Usar el número de características disponibles
                empty_feature_dim = len(usable_features)
            
            # Crear una secuencia vacía pero con la estructura correcta
            empty_seq_len = self.sequence_length
            
            # Inicializar con un tamaño 0 pero estructura correcta
            sequences = np.zeros((0, empty_seq_len, empty_feature_dim), dtype=np.float32)
            targets = np.array([], dtype=np.float32)
            categorical = np.zeros((0, len(self.categorical_columns) if self.categorical_columns else 0), dtype=np.int64)
            line_values = np.zeros((0, 1), dtype=np.float32)
            
            print(f"Devolviendo arrays vacíos con estructura: features({empty_seq_len}, {empty_feature_dim})")
        else:
            # Asegurar forma uniforme de las secuencias
            def ensure_uniform_shape(seq_list):
                """
                Asegura que todas las secuencias tengan la misma forma para poder
                convertirlas a un array de NumPy, manejando correctamente secuencias
                de 3 dimensiones.
                """
                if not seq_list:
                    return np.array([], dtype=np.float32)
                
                # Verificar y unificar la forma de cada secuencia
                uniform_seqs = []
                
                # Primero, encontramos la forma objetivo (shape)
                # Buscamos una secuencia válida (no None y con shape completo)
                target_shape = None
                for seq in seq_list:
                    if seq is not None and hasattr(seq, 'shape') and len(seq.shape) > 1:
                        # Tomamos la dimensión más grande en cada eje
                        if target_shape is None:
                            target_shape = list(seq.shape)
                        else:
                            for i in range(min(len(target_shape), len(seq.shape))):
                                target_shape[i] = max(target_shape[i], seq.shape[i])
                
                # Si no encontramos una secuencia válida, devolvemos un array vacío
                if target_shape is None:
                    print("Error: No se encontró ninguna secuencia válida")
                    return np.array([], dtype=np.float32)
                
                # Asegurar que todas las secuencias tengan la misma forma
                for i, seq in enumerate(seq_list):
                    try:
                        # Si la secuencia es None o vacía, crear un array de ceros
                        if seq is None or (hasattr(seq, 'size') and seq.size == 0):
                            uniform_seqs.append(np.zeros(target_shape, dtype=np.float32))
                            continue
                            
                        # Convertir a numpy array si no lo es
                        if not isinstance(seq, np.ndarray):
                            seq = np.array(seq, dtype=np.float32)
                        
                        # Si tiene menos dimensiones que target_shape, expandir
                        while len(seq.shape) < len(target_shape):
                            seq = np.expand_dims(seq, axis=-1)
                        
                        # Si tiene más dimensiones, aplanar las extras
                        if len(seq.shape) > len(target_shape):
                            # Determinar qué dimensiones mantener y cuáles aplanar
                            new_shape = list(seq.shape[:len(target_shape)-1])
                            # Aplanar las dimensiones restantes
                            flattened_dim = np.prod(seq.shape[len(target_shape)-1:])
                            new_shape.append(flattened_dim)
                            seq = seq.reshape(new_shape)
                        
                        # Redimensionar para que coincida con target_shape
                        current_shape = list(seq.shape)
                        if current_shape != target_shape:
                            # Crear un nuevo array con la forma objetivo
                            new_seq = np.zeros(target_shape, dtype=np.float32)
                            
                            # Copiar los datos de la secuencia original, hasta donde sea posible
                            slices = tuple(slice(0, min(current_shape[i], target_shape[i])) 
                                           for i in range(len(target_shape)))
                            new_seq[slices] = seq[slices]
                            
                            uniform_seqs.append(new_seq)
                        else:
                            uniform_seqs.append(seq)
                    except Exception as e:
                        print(f"Error procesando secuencia {i}: {e}")
                        # En caso de error, añadir un array de ceros con la forma objetivo
                        uniform_seqs.append(np.zeros(target_shape, dtype=np.float32))
                
                # Convertir a array de NumPy
                try:
                    return np.array(uniform_seqs, dtype=np.float32)
                except Exception as e:
                    print(f"Error al crear el array uniforme: {e}")
                    # Último recurso: devolver array vacío
                    return np.array([], dtype=np.float32)
            
            sequences = ensure_uniform_shape(sequences)
            targets = np.array(targets, dtype=np.float32)
            
            # Asegurar que todos los valores categóricos tengan la misma longitud
            if categorical:
                # Verificar que todos los valores categóricos tengan la misma longitud
                expected_length = len(self.categorical_columns)
                normalized_categorical = []
                
                for cat in categorical:
                    # Asegurar que cada valor categórico sea una lista con la longitud correcta
                    if isinstance(cat, (list, np.ndarray, pd.Series)):
                        # Convertir a lista y manejar Series
                        cat_list = list(cat)
                    elif cat is not None:
                        # Manejar valores escalares
                        cat_list = [cat]
                    else:
                        cat_list = []
                    
                    # Ajustar la longitud
                    if len(cat_list) < expected_length:
                        cat_list.extend([0] * (expected_length - len(cat_list)))
                    
                    # Truncar si es necesario
                    cat_list = cat_list[:expected_length]
                    
                    # Convertir a enteros, manejando valores no numéricos
                    cat_list = []
                    for x in cat_list:
                        try:
                            # Intentar convertir a entero
                            cat_list.append(int(x) if x is not None else 0)
                        except (ValueError, TypeError):
                            # Si no se puede convertir, usar 0
                            logger.warning(f"No se pudo convertir {x} a entero. Usando 0.")
                            cat_list.append(0)
                    
                    # Rellenar con ceros si es necesario
                    while len(cat_list) < expected_length:
                        cat_list.append(0)
                    
                    normalized_categorical.append(cat_list)
                
                # Convertir a numpy array con manejo de errores
                try:
                    categorical = np.array(normalized_categorical, dtype=np.int64)
                except Exception as e:
                    logger.error(f"Error convirtiendo valores categóricos: {e}")
                    logger.error(f"Detalles de categorical: {categorical}")
                    # Último recurso: crear un array de ceros
                    categorical = np.zeros((len(normalized_categorical), expected_length), dtype=np.int64)
            else:
                # Si no hay valores categóricos, crear un array de ceros
                categorical = np.zeros((len(sequences), len(self.categorical_columns)), dtype=np.int64)
            
            line_values = np.array(line_values, dtype=np.float32)
            
            print(f"Generadas {len(sequences)} secuencias con {sequences.shape[2]} características")
        
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
        """Asegura que las columnas sean numéricas con tipo float32"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                try:
                    # Manejar caso de DataFrame o Series anidado de manera más robusta
                    def extract_numeric(x):
                        # Manejar diferentes tipos de entrada
                        if isinstance(x, (pd.DataFrame, pd.Series)):
                            # Intentar extraer un valor numérico
                            if hasattr(x, 'values') and len(x.values) > 0:
                                x = x.values[0]
                            else:
                                return 0.0
                        
                        # Convertir a cadena y luego a numérico
                        try:
                            return pd.to_numeric(str(x).replace(',', '.'), errors='coerce')
                        except:
                            return 0.0
                    
                    # Aplicar la función de extracción - usar apply con funciones definidas fuera
                    # para evitar el warning de Series.__getitem__
                    extract_numeric_vectorized = np.vectorize(extract_numeric)
                    df[col] = extract_numeric_vectorized(df[col].values)
                    
                    # Convertir a float32, manejando NaN
                    df[col] = df[col].fillna(0.0).astype('float32')
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to numeric. Error: {e}")
                    # Si falla, llenar con 0
                    df[col] = 0.0
        return df
    
    def _create_categorical_encoders(self, df: pd.DataFrame):
        """Crea diccionarios para codificar variables categóricas"""
        for col in self.categorical_columns:
            if col in df.columns:
                unique_values = df[col].unique()
                self.categorical_encoders[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }

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
            
            # Ordenar por fecha
            player_games = player_games.sort_values('Date')
            
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
