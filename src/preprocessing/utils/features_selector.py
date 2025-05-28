import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeaturesSelector:


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