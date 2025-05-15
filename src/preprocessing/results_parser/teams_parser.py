import re
import numpy as np
import pandas as pd
import os

class TeamsParser:
    """
    Parser para procesar resultados de partidos NBA desde la perspectiva de equipos
    """
    def __init__(self):
        # Patrón regex para extraer información del resultado
        # Ejemplo: L 152-157, L 148-149 (OT)
        self.pattern = r'^([WL])\s+(\d+)-(\d+)(?:\s+\((\d*OT)\))?$'
    
    def parse_result(self, result_str):
        """
        Parsea un string de resultado (ej: 'W 123-100', 'L 114-116', 'W 133-129 (OT)', 'L 124-128 (2OT)')
        
        Args:
            result_str (str): String con el resultado del partido
            
        Returns:
            dict: Diccionario con la información parseada
                - is_win (int): 1 si es victoria, 0 si es derrota
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
            outcome = groups[0]
            overtime = groups[3] if len(groups) > 3 and groups[3] else None
            
            # Determinar victoria/derrota
            is_win = 1 if outcome == 'W' else 0
            
            return {
                'is_win': is_win,
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
    
    @staticmethod
    def load_teams_data(data_path='data/teams.csv'):
        """
        Carga y parsea el archivo teams.csv
        
        Args:
            data_path (str): Ruta al archivo teams.csv
            
        Returns:
            pd.DataFrame: DataFrame con los resultados parseados
        """
        # Verificar que el archivo existe
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo {data_path}")
        
        # Cargar el archivo
        df = pd.read_csv(data_path)
        
        # Crear instancia del parser
        parser = TeamsParser()
        
        # Parsear los resultados
        return parser.parse_dataframe(df)

if __name__ == '__main__':
    # Ejemplo de uso
    try:
        teams_df = TeamsParser.load_teams_data()
        print("\nColumnas disponibles:")
        print(teams_df.columns.tolist())
        print("\nPrimeras 5 filas:")
        print(teams_df[['Team', 'Date', 'Result', 'is_win', 'has_overtime', 'overtime_periods']].head())
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")