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
import io
import codecs
from sklearn.base import BaseEstimator

# Importar clone con manejo de compatibilidad entre versiones
try:
    from sklearn.base import clone
except ImportError:
    try:
        from sklearn.utils.metaestimators import clone
    except ImportError:
        # Implementación básica de clone si no está disponible
        def clone(estimator):
            from copy import deepcopy
            return deepcopy(estimator)

from xgboost import XGBRegressor, XGBClassifier

# Configurar stdout para UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Crear handler personalizado para consola que maneja Unicode
class SafeStreamHandler(logging.StreamHandler):
    """Handler que maneja caracteres Unicode de forma segura"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Escapar caracteres problemáticos para la consola
            safe_msg = msg.encode('ascii', 'replace').decode('ascii')
            self.stream.write(safe_msg + self.terminator)
            self.flush()
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Si falla, usar representación ASCII
            try:
                msg = self.format(record)
                ascii_msg = msg.encode('ascii', 'ignore').decode('ascii')
                self.stream.write(ascii_msg + self.terminator)
                self.flush()
            except Exception:
                # Último recurso: mensaje de error básico
                self.stream.write("Error de codificación en mensaje de log\n")
                self.flush()

# Configurar logging
def setup_logging():
    """Configura el logging con manejo robusto de Unicode"""
    # Crear directorio de logs si no existe
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar handlers
    log_file = os.path.join(log_dir, f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Handler para archivo (con UTF-8)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Handler para consola (con manejo seguro de Unicode)
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes para evitar duplicación
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Añadir nuevos handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Configurar logging al importar el módulo
logger = setup_logging().getChild(__name__)

def safe_log_info(message):
    """Función auxiliar para logging seguro de mensajes con caracteres especiales"""
    try:
        logger.info(message)
    except UnicodeEncodeError:
        # Escapar caracteres problemáticos
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        logger.info(f"[Mensaje con caracteres especiales]: {safe_message}")

def safe_log_player_info(player_name, value, description=""):
    """Función específica para logging seguro de información de jugadores"""
    try:
        # Intentar encoding/decoding para detectar caracteres problemáticos
        test_encode = player_name.encode('ascii')
        logger.info(f"   * {player_name}: {value} {description}")
    except UnicodeEncodeError:
        # Reemplazar caracteres problemáticos con equivalentes ASCII
        import unicodedata
        
        # Normalizar y convertir caracteres acentuados
        normalized = unicodedata.normalize('NFD', player_name)
        ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Si aún hay problemas, usar transliteración manual
        replacements = {
            'ć': 'c', 'č': 'c', 'ž': 'z', 'š': 's', 'đ': 'd',
            'Ć': 'C', 'Č': 'C', 'Ž': 'Z', 'Š': 'S', 'Đ': 'D',
            'ñ': 'n', 'Ñ': 'N', 'ü': 'u', 'Ü': 'U',
            'ö': 'o', 'Ö': 'O', 'ä': 'a', 'Ä': 'A'
        }
        
        for original, replacement in replacements.items():
            ascii_name = ascii_name.replace(original, replacement)
        
        logger.info(f"   * {ascii_name}: {value} {description}")
        
        # También log del nombre original en el archivo (que soporta UTF-8)
        try:
            file_logger = logging.getLogger('file_only')
            if not file_logger.handlers:
                log_file = os.path.join("logs", f"utf8_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                fh = logging.FileHandler(log_file, encoding='utf-8')
                fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                file_logger.addHandler(fh)
                file_logger.setLevel(logging.INFO)
            
            file_logger.info(f"   * {player_name}: {value} {description}")
        except:
            pass  # Si falla el logging a archivo, continuar

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
            'game_data': os.path.join(args.data_dir, 'processed\players_features.csv'),
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
            
            try:
                # Verificar si el modelo tiene datos preparados
                X_train, X_test, y_train, y_test = model.get_train_test_data()
            except ValueError as e:
                # Si no tiene datos, prepararlos nuevamente
                logger.warning(f"Datos no preparados para {model_name}, preparando nuevamente...")
                X_train, X_test, y_train, y_test = model.prepare_data(
                    trainer.get_processed_data(), 
                    test_size=0.2, 
                    time_split=True
                )
            
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
                        # Preparar un wrapper para el modelo que use arrays NumPy
                        class SafeLearningCurveWrapper(BaseEstimator):
                            def __init__(self, model):
                                self.model = model
                                self.fitted_model = None
                            
                            def fit(self, X, y):
                                # Convertir a array NumPy para evitar el error dtype
                                if hasattr(X, 'values'):
                                    X_array = X.values
                                else:
                                    X_array = np.array(X)
                                
                                if hasattr(y, 'values'):
                                    y_array = y.values
                                else:
                                    y_array = np.array(y)
                                
                                # Reemplazar NaNs si existen
                                X_array = np.nan_to_num(X_array, nan=0.0)
                                
                                # Clonar el modelo para no modificar el original
                                self.fitted_model = clone(self.model)
                                self.fitted_model.fit(X_array, y_array)
                                return self
                            
                            def predict(self, X):
                                # Convertir a array NumPy
                                if hasattr(X, 'values'):
                                    X_array = X.values
                                else:
                                    X_array = np.array(X)
                                
                                # Reemplazar NaNs si existen
                                X_array = np.nan_to_num(X_array, nan=0.0)
                                
                                return self.fitted_model.predict(X_array)
                            
                            # Añadir método score para compatibilidad con learning_curve
                            def score(self, X, y):
                                # Realizar predicción
                                y_pred = self.predict(X)
                                
                                # Convertir y a array NumPy si es necesario
                                if hasattr(y, 'values'):
                                    y_array = y.values
                                else:
                                    y_array = np.array(y)
                                
                                # Calcular R² (coeficiente de determinación)
                                from sklearn.metrics import r2_score
                                return r2_score(y_array, y_pred)
                        
                        try:
                            # Convertir datos a arrays NumPy ANTES de usar con XGBoost
                            if hasattr(X_train_prepared, 'values'):
                                X_train_array = X_train_prepared.values
                            else:
                                X_train_array = np.array(X_train_prepared)
                            
                            if hasattr(X_test_prepared, 'values'):
                                X_test_array = X_test_prepared.values  
                            else:
                                X_test_array = np.array(X_test_prepared)
                            
                            if hasattr(y_train, 'values'):
                                y_train_array = y_train.values
                            else:
                                y_train_array = np.array(y_train)
                            
                            if hasattr(y_test, 'values'):
                                y_test_array = y_test.values
                            else:
                                y_test_array = np.array(y_test)
                            
                            # Limpiar NaNs
                            X_train_array = np.nan_to_num(X_train_array, nan=0.0)
                            X_test_array = np.nan_to_num(X_test_array, nan=0.0)
                            y_train_array = np.nan_to_num(y_train_array, nan=0.0)
                            y_test_array = np.nan_to_num(y_test_array, nan=0.0)
                            
                            # Determinar si es regresión o clasificación
                            if hasattr(best_model, 'objective') and 'binary' in str(best_model.objective):
                                temp_model = XGBClassifier(**best_model.get_params())
                            else:
                                temp_model = XGBRegressor(**best_model.get_params())
                            
                            # Primero intentar con early_stopping_rounds
                            try:
                                # Verificar versión de XGBoost para usar los parámetros correctos
                                import xgboost
                                xgb_version = xgboost.__version__
                                logger.info(f"Versión de XGBoost: {xgb_version}")
                                
                                # En versiones más nuevas se usa early_stopping como parámetro
                                if int(xgb_version.split('.')[0]) >= 1:
                                    logger.info("Usando parámetro early_stopping para XGBoost >= 1.0")
                                    temp_model.fit(
                                        X_train_array, y_train_array, 
                                        eval_set=[(X_test_array, y_test_array)],
                                        early_stopping=True,
                                        eval_metric='rmse',
                                        verbose=False
                                    )
                                else:
                                    # En versiones anteriores se usa early_stopping_rounds
                                    logger.info("Usando parámetro early_stopping_rounds para XGBoost < 1.0")
                                    temp_model.fit(
                                        X_train_array, y_train_array, 
                                        eval_set=[(X_test_array, y_test_array)],
                                        early_stopping_rounds=5,
                                        eval_metric='rmse',
                                        verbose=False
                                    )
                                logger.info(f"Reentrenado XGBoost exitosamente con arrays NumPy para {model_name}")
                                best_model = temp_model
                            except TypeError as type_e:
                                logger.warning(f"Error con early_stopping_rounds: {repr(str(type_e))}")
                                # Si falla, intentar sin parámetros de early stopping
                                try:
                                    logger.info("Intentando entrenar XGBoost sin early stopping")
                                    temp_model.fit(X_train_array, y_train_array)
                                    logger.info(f"Reentrenado XGBoost básico con arrays NumPy para {model_name}")
                                    best_model = temp_model
                                except Exception as basic_e:
                                    logger.error(f"Error en reentrenamiento básico con arrays NumPy: {repr(str(basic_e))}")
                                    # Usar el modelo original si todo falla
                                    pass
                        except Exception as wrapper_error:
                            logger.error(f"Error general en el reentrenamiento con arrays NumPy: {repr(str(wrapper_error))}")
                            # Intentar un enfoque más simple como último recurso
                            try:
                                # Reentrenar sin parámetros adicionales
                                temp_model = XGBRegressor()
                                temp_model.fit(X_train_array, y_train_array)
                                best_model = temp_model
                                logger.info(f"Reentrenado modelo XGBoost básico para {model_name}")
                            except Exception as basic_xgb_e:
                                logger.error(f"Error en reentrenamiento básico: {repr(str(basic_xgb_e))}")
                                # Usar modelo original como último recurso
                                pass
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
                if 'auc_roc' in metrics:
                    logger.info(f"   * AUC-ROC: {metrics['auc_roc']:.4f}")
                if 'auc_pr' in metrics:
                    logger.info(f"   * AUC-PR: {metrics['auc_pr']:.4f}")
                if 'brier_score' in metrics:
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
            safe_log_info(f"\nANÁLISIS DE PATRONES")
            safe_log_info("-" * 50)
            safe_log_info("Rachas más Largas de Doble-Dobles:")
            for _, streak in patterns['streaks']['longest_streaks'].head().iterrows():
                safe_log_player_info(streak['Player'], streak['streak_length'], "juegos")
        
        safe_log_info(f"\nANÁLISIS COMPLETADO")
        safe_log_info(f"Archivos generados en: {args.output_dir}")
        safe_log_info(f"Reporte principal: {report_path}")
        safe_log_info(f"Métricas completas: {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error durante el análisis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 