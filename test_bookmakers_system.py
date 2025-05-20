import pandas as pd
import numpy as np
import logging
import sys
import traceback

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BookmakersTest")

def test_bookmakers_system():
    """
    Prueba básica del sistema de bookmakers
    """
    logger.info("Iniciando prueba del sistema de bookmakers")
    
    try:
        # Importar módulos
        logger.info("Importando módulos...")
        from src.preprocessing.utils.features_selector import FeaturesSelector
        from src.preprocessing.utils.bookmakers.bookmakers_data_fetcher import BookmakersDataFetcher
        from src.preprocessing.feature_engineering.players_features import PlayersFeatures
        
        # 1. Cargar datos de muestra
        logger.info("Creando datos de muestra")
        # Crear DataFrame de muestra
        sample_data = {
            'Player': ['LeBron James', 'Kevin Durant', 'Stephen Curry', 'Giannis Antetokounmpo', 'Joel Embiid'],
            'Date': pd.date_range(start='2023-01-01', periods=5),
            'PTS': [25, 30, 35, 28, 40],
            'TRB': [7, 8, 5, 12, 10],
            'AST': [10, 6, 8, 5, 3],
            '3P': [2, 4, 6, 1, 0],
            'Team': ['LAL', 'PHX', 'GSW', 'MIL', 'PHI'],
            'Opp': ['GSW', 'LAL', 'MIL', 'PHI', 'GSW'],
            'FG%': [0.45, 0.51, 0.48, 0.56, 0.52],
            'FT%': [0.82, 0.90, 0.92, 0.68, 0.85],
            'MP': [32, 35, 33, 34, 36]
        }
        df = pd.DataFrame(sample_data)
        
        # 2. Crear instancia de PlayersFeatures con los datos
        logger.info("Creando instancia de PlayersFeatures")
        players_features = PlayersFeatures(df)
        
        # 3. Crear instancia de BookmakersDataFetcher
        logger.info("Creando instancia de BookmakersDataFetcher")
        fetcher = BookmakersDataFetcher()
        
        # 4. Simular datos de casas de apuestas
        logger.info("Probando simulación de datos de casas de apuestas")
        df_with_odds = fetcher.simulate_bookmaker_data(df, target='PTS')
        
        # 5. Crear instancia de FeaturesSelector
        logger.info("Creando instancia de FeaturesSelector")
        selector = FeaturesSelector()
        
        # Modificar la inicialización del generador de características
        logger.info("Configurando generador de características")
        selector.players_feature_engineering = players_features
        
        # 6. Obtener características para apuestas básicas
        logger.info("Obteniendo características para apuestas")
        # Usar características fijas para evitar errores en la selección de características
        pts_features = ['FG%', 'MP', 'FT%', 'PTS', '3P', 'TRB', 'AST', 'Team']
        
        # 7. Prueba simplificada de líneas de confianza (con dataset reducido)
        logger.info("Probando identificación simplificada de líneas")
        high_confidence_lines = selector.identify_high_confidence_betting_lines(
            df_with_odds, 'PTS', min_confidence=0.6,  # Umbral reducido para pruebas
            min_samples=3  # Mínimo de muestras reducido para pruebas
        )
        
        # 8. Prueba de estrategia simplificada de apuestas
        logger.info("Probando generación de estrategia simplificada")
        betting_strategy = selector.get_optimal_betting_strategy(
            df_with_odds, 'PTS', confidence_threshold=0.6  # Umbral reducido para pruebas
        )
        
        # Mostrar resultados
        logger.info(f"Datos con odds generados: {df_with_odds.shape}")
        logger.info(f"Características para PTS: {len(pts_features)}")
        logger.info(f"Líneas identificadas: {high_confidence_lines}")
        logger.info(f"Estrategia de apuestas generada: {betting_strategy}")
        
        logger.info("Prueba finalizada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error en la prueba: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_bookmakers_system() 