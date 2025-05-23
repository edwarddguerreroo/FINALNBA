#!/usr/bin/env python3
"""
Script para ejecutar análisis avanzado de modelos NBA.
"""

import argparse
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from src.models.model_trainer import NBAModelTrainer
from src.analysis.advanced_metrics import NBAAdvancedAnalytics

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Función principal para ejecutar análisis avanzado."""
    parser = argparse.ArgumentParser(description='Análisis Avanzado de Modelos NBA')
    
    parser.add_argument('--models-dir', default='trained_models',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--data-dir', default='data',
                       help='Directorio con datos procesados')
    parser.add_argument('--output-dir', default='analysis_output',
                       help='Directorio para guardar análisis')
    parser.add_argument('--player', type=str,
                       help='Analizar jugador específico')
    
    args = parser.parse_args()
    
    try:
        # Inicializar analizador
        analyzer = NBAAdvancedAnalytics(output_dir=args.output_dir)
        
        # Cargar datos
        logger.info("Cargando datos...")
        data_paths = {
            'game_data': os.path.join(args.data_dir, 'processed_data.csv'),
            'biometrics': os.path.join(args.data_dir, 'height.csv'),
            'teams': os.path.join(args.data_dir, 'teams.csv')
        }
        
        trainer = NBAModelTrainer(data_paths, output_dir=args.models_dir)
        trainer.load_and_prepare_data()
        
        # Entrenar modelos si no existen
        logger.info("Verificando modelos...")
        trainer.train_all_models()
        
        # Generar análisis avanzado para cada modelo
        logger.info("Generando análisis avanzado de modelos...")
        all_diagnostics = {}
        all_metrics = {}
        all_visualizations = {}
        
        for model_name, model in trainer.get_trained_models().items():
            logger.info(f"Analizando modelo: {model_name}")
            
            # Obtener datos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = model.get_train_test_data()
            
            # Obtener el mejor modelo
            best_model_name, best_model, _ = model.get_best_model()
            
            # Preparar datos para predicción (para evitar problemas de características)
            try:
                # Verificar si X_test tiene las características requeridas por el modelo
                if hasattr(best_model, 'feature_names_in_'):
                    expected_features = best_model.feature_names_in_
                    has_all_features = all(feat in X_test.columns for feat in expected_features)
                    
                    if not has_all_features:
                        logger.warning(f"X_test no contiene todas las características requeridas por el modelo")
                        
                        # Intentar recrear las características de interacción si es posible
                        if hasattr(model, '_add_interaction_features'):
                            logger.info(f"Recreando características de interacción para {model_name}")
                            X_train_prepared = model._add_interaction_features(X_train)
                            X_test_prepared = model._add_interaction_features(X_test)
                        else:
                            # Si no hay función de preparación específica, usar solo características básicas
                            basic_features = ['MP', 'FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'is_home', 'is_started']
                            available_basic = [f for f in basic_features if f in X_test.columns]
                            X_train_prepared = X_train[available_basic]
                            X_test_prepared = X_test[available_basic]
                    else:
                        X_train_prepared = X_train
                        X_test_prepared = X_test
                else:
                    # Si el modelo no especifica las características, usamos todas
                    X_train_prepared = X_train
                    X_test_prepared = X_test
                
                # Si estamos trabajando con XGBoost, necesitamos asegurarnos de que los nombres coincidan
                if 'xgboost' in str(best_model.__class__).lower() and hasattr(best_model, 'get_booster'):
                    # Crear un nuevo modelo y copiar los parámetros, pero con las características actuales
                    try:
                        from xgboost import XGBRegressor, XGBClassifier
                        import copy
                        
                        # Determinar si es regresión o clasificación
                        if hasattr(best_model, 'objective') and 'binary' in best_model.objective:
                            temp_model = XGBClassifier(**best_model.get_params())
                        else:
                            temp_model = XGBRegressor(**best_model.get_params())
                        
                        # Entrenar con los datos actuales (ajuste ligero)
                        # Usar parámetros compatibles con la versión de XGBoost
                        params = {
                            'eval_metric': 'rmse',
                            'verbose': False
                        }
                        
                        # Primero intentar con early_stopping_rounds
                        try:
                            temp_model.fit(
                                X_train_prepared, y_train, 
                                eval_set=[(X_test_prepared, y_test)],
                                early_stopping_rounds=5,
                                **params
                            )
                        except TypeError:
                            # Si falla, intentar con early_stopping (versiones más nuevas)
                            try:
                                params['early_stopping'] = True
                                params['eval_metric'] = 'rmse'
                                temp_model.fit(
                                    X_train_prepared, y_train, 
                                    eval_set=[(X_test_prepared, y_test)],
                                    **params
                                )
                            except TypeError:
                                # Si sigue fallando, entrenar sin early stopping
                                temp_model.fit(X_train_prepared, y_train)
                        
                        # Usar este modelo para cálculos
                        best_model = temp_model
                        logger.info(f"Reentrenado modelo XGBoost para {model_name} con las características actuales")
                    except Exception as xgb_e:
                        logger.error(f"Error reentrenando XGBoost: {str(xgb_e)}")
                        # Intentar un enfoque más simple como último recurso
                        try:
                            # Reentrenar sin parámetros adicionales
                            temp_model = XGBRegressor()
                            temp_model.fit(X_train_prepared, y_train)
                            best_model = temp_model
                            logger.info(f"Reentrenado modelo XGBoost básico para {model_name}")
                        except Exception as basic_xgb_e:
                            logger.error(f"Error en reentrenamiento básico: {str(basic_xgb_e)}")
            except Exception as e:
                logger.error(f"Error preparando datos para {model_name}: {str(e)}")
                X_train_prepared = X_train
                X_test_prepared = X_test
            
            # Calcular métricas avanzadas (R1, AUC, MCC, etc.)
            logger.info(f"Calculando métricas avanzadas para {model_name}...")
            advanced_metrics = analyzer.calculate_advanced_metrics(
                best_model, X_train_prepared, X_test_prepared, y_train, y_test, model_name
            )
            all_metrics[model_name] = advanced_metrics
            
            # Generar visualizaciones en PNG
            logger.info(f"Generando visualizaciones PNG para {model_name}...")
            visualizations = analyzer.generate_advanced_visualizations(
                best_model, X_train_prepared, X_test_prepared, y_train, y_test, model_name
            )
            all_visualizations[model_name] = visualizations
            
            # Mantener compatibilidad con diagnósticos anteriores
            diagnostics = analyzer.generate_model_diagnostics(
                best_model, X_train_prepared, X_test_prepared, y_train, y_test
            )
            all_diagnostics[model_name] = diagnostics
        
        # Analizar patrones de jugadores
        logger.info("Analizando patrones de jugadores...")
        df = trainer.get_processed_data()
        
        if args.player:
            player_df = df[df['Player'] == args.player].copy()
            if len(player_df) > 0:
                patterns = analyzer.analyze_player_patterns(player_df)
                logger.info(f"Análisis completado para {args.player}")
            else:
                logger.warning(f"No se encontraron datos para {args.player}")
        else:
            patterns = analyzer.analyze_player_patterns(df)
            logger.info("Análisis general de jugadores completado")
        
        # Generar reporte final
        logger.info("Generando reporte final...")
        report_path = analyzer.generate_advanced_report(
            model_diagnostics=all_diagnostics,
            pattern_analysis=patterns
        )
        
        logger.info(f"Reporte generado en: {report_path}")
        
        # Guardar todas las métricas y resultados en formato JSON
        metrics_path = os.path.join(args.output_dir, 'advanced_metrics_complete.json')
        complete_metrics = {
            'advanced_metrics': all_metrics,
            'visualizations': all_visualizations,
            'model_diagnostics': all_diagnostics,
            'pattern_analysis': patterns
        }
        
        import json
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(complete_metrics, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Métricas completas guardadas en: {metrics_path}")
        
        # Mostrar resumen detallado
        logger.info("\nRESUMEN DE ANÁLISIS AVANZADO")
        logger.info("=" * 80)
        
        for model_name, metrics in all_metrics.items():
            logger.info(f"\nMODELO: {model_name.upper()}")
            logger.info("-" * 50)
            
            if 'accuracy' in metrics:  # Modelo de clasificación
                logger.info("MÉTRICAS DE CLASIFICACIÓN:")
                logger.info(f"   * Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"   * Precision: {metrics['precision']:.4f}")
                logger.info(f"   * Recall: {metrics['recall']:.4f}")
                logger.info(f"   * F1-Score: {metrics['f1_score']:.4f}")
                logger.info(f"   * Matthews Corr Coef: {metrics['matthews_corr_coef']:.4f}")
                logger.info(f"   * AUC-ROC: {metrics['auc_roc']:.4f}")
                logger.info(f"   * AUC-PR: {metrics['auc_pr']:.4f}")
                logger.info(f"   * Brier Score: {metrics['brier_score']:.4f}")
                if metrics.get('log_loss'):
                    logger.info(f"   * Log Loss: {metrics['log_loss']:.4f}")
            else:  # Modelo de regresión
                logger.info("MÉTRICAS DE REGRESIÓN:")
                if 'rmse' in metrics:
                    logger.info(f"   * RMSE: {metrics['rmse']:.4f}")
                if 'mae' in metrics:
                    logger.info(f"   * MAE: {metrics['mae']:.4f}")
                if 'r2' in metrics:
                    logger.info(f"   * R²: {metrics['r2']:.4f}")
                if 'r2_adjusted' in metrics:
                    logger.info(f"   * R² Ajustado (R1): {metrics['r2_adjusted']:.4f}")
                if metrics.get('mape'):
                    logger.info(f"   * MAPE: {metrics['mape']:.2f}%")
                if 'pearson_corr' in metrics:
                    logger.info(f"   * Correlación Pearson: {metrics['pearson_corr']:.4f}")
                if metrics.get('cv_rmse'):
                    logger.info(f"   * CV-RMSE: {metrics['cv_rmse']:.4f}")
                
                # Test de normalidad
                if 'shapiro_test' in metrics:
                    shapiro = metrics['shapiro_test']
                    normal_dist = "Sí" if shapiro['p_value'] > 0.05 else "No"
                    logger.info(f"   * Residuos normales: {normal_dist} (p={shapiro['p_value']:.4f})")
            
            # Visualizaciones generadas
            if model_name in all_visualizations:
                viz_count = len(all_visualizations[model_name])
                logger.info(f"   * Visualizaciones generadas: {viz_count}")
                for viz_name, viz_path in all_visualizations[model_name].items():
                    logger.info(f"      - {viz_name}: {viz_path}")
        
        # Resumen de análisis de patrones
        if 'streaks' in patterns:
            logger.info(f"\nANÁLISIS DE PATRONES")
            logger.info("-" * 50)
            logger.info("Rachas más Largas de Doble-Dobles:")
            for _, streak in patterns['streaks']['longest_streaks'].head().iterrows():
                logger.info(f"   * {streak['Player']}: {streak['streak_length']} juegos")
        
        logger.info(f"\nANÁLISIS COMPLETADO")
        logger.info(f"Archivos generados en: {args.output_dir}")
        logger.info(f"Reporte principal: {report_path}")
        logger.info(f"Métricas completas: {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error durante el análisis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 