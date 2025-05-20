import pandas as pd
import numpy as np
import logging
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("double_double.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DoubleDouble')


def calculate_double_triple_doubles(df):
    """
    Calcula doble-dobles y triple-dobles para cada jugador en el DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con estadísticas de jugadores
        
    Returns:
        pd.DataFrame: DataFrame con columnas adicionales para doble-doble y triple-doble
    """
    try:
        # Estadísticas a considerar para doble-doble y triple-doble
        stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
        available_stats = [stat for stat in stats if stat in df.columns]
        
        # Asegurar que las estadísticas sean numéricas
        for stat in available_stats:
            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
        
        # Crear columnas X_double
        for stat in available_stats:
            x_double_col = f'{stat}_double'
            df[x_double_col] = (df[stat] >= 10).astype(int)
        
        # Calcular double_double y triple_double
        double_cols = [f'{stat}_double' for stat in available_stats]
        # Usar sum(axis=1) para contar cuántas estadísticas superan el umbral
        df['double_double'] = (df[double_cols].sum(axis=1) >= 2).astype(int)
        df['triple_double'] = (df[double_cols].sum(axis=1) >= 3).astype(int)
        
        # Verificación y logging
        dd_count = df['double_double'].sum()
        td_count = df['triple_double'].sum()
        logger.info(f"Se identificaron {dd_count} doble-dobles y {td_count} triple-dobles")
        
        # Verificar casos con PTS=0
        zero_pts_dd = df[(df['PTS'] < 1) & (df['double_double'] == 1)]
        if not zero_pts_dd.empty:
            logger.info(f"VALIDACIÓN: Hay {len(zero_pts_dd)} doble-dobles sin puntos (legítimos)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error al calcular doble-dobles y triple-dobles: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # Devolver el DataFrame original en caso de error