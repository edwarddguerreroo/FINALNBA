import pandas as pd
import logging
import os
from src.preprocessing.data_loader import NBADataLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loader_test.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DataLoaderTest')

def main():
    """
    Función principal para probar el módulo data_loader.py
    """
    # Rutas a los archivos de datos
    game_data_path = 'data/players.csv'
    biometrics_path = 'data/height.csv'
    teams_path = 'data/teams.csv'

    # Verificar que los archivos existan
    for path in [game_data_path, biometrics_path, teams_path]:
        if not os.path.exists(path):
            logger.error(f"El archivo {path} no existe.")
            return

    # Inicializar el cargador de datos
    data_loader = NBADataLoader(game_data_path, biometrics_path, teams_path)

    # Cargar y procesar datos
    try:
        merged_data, teams_data = data_loader.load_data()
        logger.info(f"Datos cargados y procesados correctamente. Total de registros: {len(merged_data)}")
        
        # Verificar que se hayan calculado doble-dobles y triple-dobles
        if 'double_double' in merged_data.columns and 'triple_double' in merged_data.columns:
            dd_count = merged_data['double_double'].sum()
            td_count = merged_data['triple_double'].sum()
            logger.info(f"Se identificaron {dd_count} doble-dobles y {td_count} triple-dobles.")
        else:
            logger.error("No se encontraron columnas de doble-doble o triple-doble en los datos procesados.")
        
        # Guardar el DataFrame procesado en un archivo CSV
        output_path = 'data/processed_data.csv'
        merged_data.to_csv(output_path, index=False)
        logger.info(f"DataFrame procesado guardado en {output_path}")
    except Exception as e:
        logger.error(f"Error al cargar o procesar datos: {str(e)}")

if __name__ == "__main__":
    main() 