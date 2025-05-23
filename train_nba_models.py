#!/usr/bin/env python3
"""
Script principal para entrenar todos los modelos específicos de predicción NBA.

Este script entrena modelos separados para:
- Puntos (PTS)
- Rebotes (TRB) 
- Asistencias (AST)
- Triples (3P)
- Doble-dobles
- Triple-dobles

Autor: Sistema de Predicción NBA
Fecha: 2024
"""

import logging
import sys
import os
import argparse
from datetime import datetime

# Configurar logging
# Verificar si ya existe un logger configurado
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"nba_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configurado correctamente")
else:
    logger = logging.getLogger(__name__)
    logger.info("Usando configuración de logging existente")

# Importar nuestros módulos
from src.models.model_trainer import NBAModelTrainer

def main():
    """Función principal para entrenar todos los modelos NBA."""
    parser = argparse.ArgumentParser(description='Entrenador de Modelos NBA')
    parser.add_argument('--game-data', default='data/processed_data.csv',
                       help='Ruta a datos de partidos procesados')
    parser.add_argument('--biometrics', default='data/height.csv',
                       help='Ruta a datos biométricos')
    parser.add_argument('--teams', default='data/teams.csv',
                       help='Ruta a datos de equipos')
    parser.add_argument('--output-dir', default='trained_models',
                       help='Directorio para guardar modelos entrenados')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporción de datos para test')
    parser.add_argument('--time-split', action='store_true',
                       help='Usar división temporal en lugar de aleatoria')
    parser.add_argument('--regenerate-features', action='store_true',
                       help='Regenerar características desde cero')
    parser.add_argument('--player-analysis', type=str,
                       help='Generar análisis específico para un jugador')
    
    args = parser.parse_args()
    
    logger.info("INICIANDO ENTRENAMIENTO DE MODELOS NBA")
    logger.info("=" * 60)
    
    try:
        # Configurar rutas de datos
        data_paths = {
            'game_data': args.game_data,
            'biometrics': args.biometrics,
            'teams': args.teams
        }
        
        # Validar que existen los archivos
        for name, path in data_paths.items():
            if not os.path.exists(path):
                logger.error(f"ERROR: Archivo {name} no encontrado: {path}")
                sys.exit(1)
        
        logger.info("OK: Archivos de datos validados")
        
        # Inicializar entrenador
        trainer = NBAModelTrainer(data_paths, output_dir=args.output_dir)
        
        # Cargar y preparar datos
        logger.info("Cargando y preparando datos...")
        trainer.load_and_prepare_data(regenerate_features=args.regenerate_features)
        
        # Entrenar todos los modelos
        logger.info("Iniciando entrenamiento de modelos...")
        training_results = trainer.train_all_models(
            test_size=args.test_size,
            use_time_split=args.time_split
        )
        
        # Mostrar resumen de resultados
        logger.info("\n" + "=" * 60)
        logger.info("RESUMEN DE ENTRENAMIENTO")
        logger.info("=" * 60)
        
        successful_models = 0
        failed_models = 0
        
        for model_name, results in training_results.items():
            if 'error' in results:
                logger.error(f"ERROR: {model_name.upper()}: {results['error']}")
                failed_models += 1
            else:
                logger.info(f"OK: {model_name.upper()}:")
                logger.info(f"   - Mejor algoritmo: {results['best_model']}")
                logger.info(f"   - Score: {results['best_score']:.3f}")
                logger.info(f"   - Características: {results['features_count']}")
                logger.info(f"   - Datos entrenamiento: {results['train_samples']}")
                logger.info(f"   - Datos test: {results['test_samples']}")
                successful_models += 1
        
        logger.info(f"\nMODELOS EXITOSOS: {successful_models}")
        logger.info(f"MODELOS FALLIDOS: {failed_models}")
        
        # Generar reporte de análisis
        logger.info("\nGenerando reporte de análisis...")
        report = trainer.generate_predictions_report(player_name=args.player_analysis)
        
        # Guardar reporte
        report_file = os.path.join(args.output_dir, "analysis_report.json")
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Reporte guardado en: {report_file}")
        
        # Comparación de modelos
        comparison = trainer.get_model_comparison()
        if not comparison.empty:
            comparison_file = os.path.join(args.output_dir, "model_comparison.csv")
            comparison.to_csv(comparison_file, index=False)
            logger.info(f"Comparación de modelos guardada en: {comparison_file}")
        
        # Exportar para apuestas
        betting_file = trainer.export_predictions_for_betting()
        logger.info(f"Datos para apuestas exportados en: {betting_file}")
        
        logger.info("\n" + "=" * 60)
        logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        # Mostrar estadísticas por modelo
        if args.player_analysis:
            logger.info(f"\nANALISIS ESPECIFICO PARA: {args.player_analysis}")
            player_analysis = report.get('player_analysis', {})
            if 'error' not in player_analysis:
                logger.info(f"   - Juegos totales: {player_analysis.get('total_games', 0)}")
                logger.info(f"   - Juegos analizados: {player_analysis.get('recent_games_analyzed', 0)}")
                
                # Mostrar contexto de puntos
                pts_context = player_analysis.get('points_context', {})
                if pts_context:
                    logger.info("   PUNTOS:")
                    logger.info(f"      - Promedio reciente: {pts_context.get('avg_pts_recent', 0):.1f}")
                    logger.info(f"      - Promedio temporada: {pts_context.get('avg_pts_season', 0):.1f}")
                
                # Mostrar contexto de rebotes
                reb_context = player_analysis.get('rebounds_context', {})
                if reb_context:
                    logger.info("   REBOTES:")
                    logger.info(f"      - Promedio reciente: {reb_context.get('avg_trb_recent', 0):.1f}")
                    logger.info(f"      - Promedio temporada: {reb_context.get('avg_trb_season', 0):.1f}")
                
                # Mostrar contexto de asistencias
                ast_context = player_analysis.get('assists_context', {})
                if ast_context:
                    logger.info("   ASISTENCIAS:")
                    logger.info(f"      - Promedio reciente: {ast_context.get('avg_ast_recent', 0):.1f}")
                    logger.info(f"      - Promedio temporada: {ast_context.get('avg_ast_season', 0):.1f}")
        
        # Información de archivos generados
        logger.info(f"\nARCHIVOS GENERADOS EN '{args.output_dir}':")
        for model_name in trainer.models.keys():
            model_file = f"{model_name}_model.pkl"
            if os.path.exists(os.path.join(args.output_dir, model_file)):
                logger.info(f"   OK: {model_file}")
        
        logger.info(f"   OK: training_summary.json")
        logger.info(f"   OK: analysis_report.json")
        logger.info(f"   OK: model_comparison.csv")
        logger.info(f"   OK: betting_predictions_*.json")
        
    except Exception as e:
        logger.error(f"ERROR CRITICO: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def show_usage_examples():
    """Muestra ejemplos de uso del script."""
    print("\nEJEMPLOS DE USO:")
    print("=" * 50)
    print("\n1. Entrenamiento básico:")
    print("   python train_nba_models.py")
    
    print("\n2. Con división temporal y análisis de jugador:")
    print("   python train_nba_models.py --time-split --player-analysis 'LeBron James'")
    
    print("\n3. Regenerar características y datos personalizados:")
    print("   python train_nba_models.py --regenerate-features \\")
    print("                               --game-data mi_datos.csv \\")
    print("                               --output-dir mis_modelos")
    
    print("\n4. Configuración completa:")
    print("   python train_nba_models.py --game-data data/processed_data.csv \\")
    print("                               --biometrics data/height.csv \\")
    print("                               --teams data/teams.csv \\")
    print("                               --output-dir trained_models \\")
    print("                               --test-size 0.25 \\")
    print("                               --time-split \\")
    print("                               --regenerate-features \\")
    print("                               --player-analysis 'Stephen Curry'")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_usage_examples()
    else:
        main() 