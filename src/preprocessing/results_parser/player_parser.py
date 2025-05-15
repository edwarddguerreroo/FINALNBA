import re
import numpy as np
import pandas as pd

class ResultParser:
    """
    Parser robusto para procesar resultados de partidos NBA
    """
    def __init__(self):
        # Patrón regex para extraer información del resultado
        # Actualizado para manejar tiempos adicionales (OT, 2OT, etc.)
        self.pattern = r'^([WL])\s+(\d+)-(\d+)(?:\s+\((\d*OT)\))?$'
    
    def parse_result(self, result_str):
        """
        Parsea un string de resultado (ej: 'W 123-100', 'L 114-116', 'W 133-129 (OT)', 'L 124-128 (2OT)')
        
        Args:
            result_str (str): String con el resultado del partido
            
        Returns:
            dict: Diccionario con la información parseada
                - is_win (int): 1 si es victoria, 0 si es derrota
                - team_score (int): Puntos anotados por el equipo
                - opp_score (int): Puntos anotados por el oponente
                - total_score (int): Suma total de puntos
                - point_diff (int): Diferencia de puntos (positiva si es victoria)
                - overtime (str): Tipo de tiempo adicional (OT, 2OT, etc.) o None si no hay
                - has_overtime (int): 1 si hubo tiempo adicional, 0 si no
                - overtime_periods (int): Número de periodos adicionales (0, 1, 2, etc.)
        """
        try:
            # Limpiar el string
            result_str = str(result_str).strip()
            
            # Aplicar regex
            match = re.match(self.pattern, result_str)
            if not match:
                return self._create_null_result()
            
            # Extraer componentes
            groups = match.groups()
            outcome, score1, score2 = groups[0], groups[1], groups[2]
            overtime = groups[3] if len(groups) > 3 and groups[3] else None
            score1, score2 = int(score1), int(score2)
            
            # Determinar victoria/derrota
            is_win = 1 if outcome == 'W' else 0
            
            # Asignar puntuaciones
            if is_win:
                team_score, opp_score = score1, score2
            else:
                team_score, opp_score = score1, score2
            
            # Calcular métricas adicionales
            total_score = team_score + opp_score
            point_diff = team_score - opp_score
            
            return {
                'is_win': is_win,
                'team_score': team_score,
                'opp_score': opp_score,
                'total_score': total_score,
                'point_diff': point_diff,
                'has_overtime': 1 if overtime else 0,
                'overtime_periods': int(overtime[0]) if overtime and overtime[0].isdigit() else 1 if overtime else 0
            }
            
        except Exception as e:
            print(f"Error parseando resultado '{result_str}': {str(e)}")
            return self._create_null_result()
    
    def _create_null_result(self):
        """Crea un resultado nulo cuando hay error de parseo"""
        return {
            'is_win': np.nan,
            'team_score': np.nan,
            'opp_score': np.nan,
            'total_score': np.nan,
            'point_diff': np.nan,
            'has_overtime': 0,
            'overtime_periods': 0
        }
    
    def parse_dataframe(self, df, result_column='Result'):
        """
        Parsea una columna de resultados en un DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame con los resultados
            result_column (str): Nombre de la columna con los resultados
            
        Returns:
            pd.DataFrame: DataFrame original con columnas adicionales de resultado
        """
        # Verificar que la columna existe
        if result_column not in df.columns:
            raise ValueError(f"Columna '{result_column}' no encontrada en el DataFrame")
        
        # Parsear cada resultado
        parsed_results = df[result_column].apply(self.parse_result)
        
        # Convertir lista de diccionarios a DataFrame
        result_df = pd.DataFrame(parsed_results.tolist())
        
        # Agregar columnas al DataFrame original
        for col in result_df.columns:
            df[col] = result_df[col]
        
        return df
