"""
Test Real - Modelo de Predicción de Rebotes NBA con Stacking Avanzado
===================================================================

Script de prueba usando el data loader real con stacking de múltiples modelos
y características de interacción para alcanzar ≥97% precisión.

Objetivo: Validar el pipeline optimizado con datos reales NBA.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Función principal del test con stacking avanzado."""
    
    logger.info("🏀 INICIANDO TEST REAL - STACKING AVANZADO DE REBOTES NBA")
    
    try:
        # Importar módulos necesarios
        from src.preprocessing.data_loader import NBADataLoader
        from src.models.players.trb.features_trb import ReboundsFeatureEngineer
        from src.models.players.trb.trb_model import ReboundsPredictor
        
        # 1. CARGAR DATOS REALES
        logger.info("📊 Cargando datos reales NBA...")
        
        # Rutas a los archivos de datos
        game_data_path = "data/players.csv"
        biometrics_path = "data/height.csv"
        teams_path = "data/teams.csv"
        
        # Inicializar data loader con rutas correctas
        data_loader = NBADataLoader(game_data_path, biometrics_path, teams_path)
        
        # Cargar datos completos
        players_df, teams_df = data_loader.load_data()
        
        # Tomar muestra de 3000 registros para prueba
        sample_size = 2000
        if len(players_df) > sample_size:
            # Muestreo estratificado por rebotes
            players_df['trb_bin'] = pd.cut(players_df['TRB'], bins=5, labels=False)
            sample_df = players_df.groupby('trb_bin', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // 5), random_state=42)
            ).reset_index(drop=True)
            sample_df = sample_df.drop('trb_bin', axis=1)
        else:
            sample_df = players_df.copy()
        
        logger.info(f"   ✅ Muestra final: {len(sample_df)} registros")
        logger.info(f"   📈 Jugadores únicos: {sample_df['Player'].nunique()}")
        logger.info(f"   📅 Rango de fechas: {sample_df['Date'].min()} a {sample_df['Date'].max()}")
        logger.info(f"   🎯 Rebotes - Min: {sample_df['TRB'].min()}, Max: {sample_df['TRB'].max()}, Promedio: {sample_df['TRB'].mean():.1f}")
        
        # 2. GENERAR CARACTERÍSTICAS OPTIMIZADAS CON INTERACCIONES
        logger.info("🔧 Preparando datos para generación de características...")
        
        # Verificar columnas necesarias
        required_cols = ['Player', 'Date', 'Team', 'Opp', 'TRB']
        missing_cols = [col for col in required_cols if col not in sample_df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Importar y usar feature engineer optimizado
        logger.info("✅ Módulo de características importado correctamente")
        feature_engineer = ReboundsFeatureEngineer()
        logger.info("ReboundsFeatureEngineer OPTIMIZADO con características de interacción inicializado")
        
        # Generar características juego por juego con equipos
        logger.info("🔧 Generando características optimizadas con interacciones...")
        X, y, feature_names = feature_engineer.generate_game_by_game_features(
            sample_df, 
            target_col='TRB',
            min_games=3,  # Reducido para tener más datos
            teams_df=teams_df
        )
        
        logger.info("✅ Características generadas con datos reales:")
        logger.info(f"   - Muestras: {len(X)}")
        logger.info(f"   - Características: {len(feature_names)}")
        logger.info(f"   - Target promedio: {y.mean():.2f} rebotes")
        logger.info(f"   - Target rango: {y.min()}-{y.max()} rebotes")
        logger.info(f"   - Target std: {y.std():.2f}")
        
        # Análisis de características por grupo
        interaction_features = [f for f in feature_names if 'x_' in f or 'interaction' in f]
        opportunity_features = [f for f in feature_names if any(x in f for x in ['missed', 'opportunity', 'mp', 'minutes'])]
        activity_features = [f for f in feature_names if any(x in f for x in ['stl', 'blk', 'activity', 'defense'])]
        team_features = [f for f in feature_names if any(x in f for x in ['team_', 'opp_', 'game_pace', 'matchup'])]
        
        logger.info("Distribución por grupo:")
        logger.info(f"  Interacciones: {len(interaction_features)}")
        logger.info(f"  Oportunidades: {len(opportunity_features)}")
        logger.info(f"  Actividad: {len(activity_features)}")
        logger.info(f"  Equipos: {len(team_features)}")
        
        # 3. ENTRENAR MODELO ULTRA-AVANZADO DE TERCER NIVEL
        logger.info("🚀 Entrenando modelo ULTRA-AVANZADO de tercer nivel...")
        
        # Crear predictor ultra-avanzado con ensemble de tercer nivel
        predictor = ReboundsPredictor(
            use_deep_learning=True,
            aggressive_optimization=True,
            third_level_ensemble=True
        )
        
        # Entrenar modelo ultra-avanzado
        training_metrics = predictor.train(X, y, feature_names)
        
        logger.info("🏆 RESULTADOS DEL ENSEMBLE ULTRA-AVANZADO DE TERCER NIVEL:")
        logger.info(f"   MAE: {training_metrics['mae_mean']:.3f} ± {training_metrics['mae_std']:.3f} rebotes")
        logger.info(f"   R²: {training_metrics['r2_mean']:.3f} ± {training_metrics['r2_std']:.3f}")
        logger.info(f"   Precisión ±1 rebote: {training_metrics['accuracy_mean']:.1f}% ± {training_metrics['accuracy_std']:.1f}%")
        logger.info(f"   Precisión exacta (±0.5): {training_metrics['exact_accuracy_mean']:.1f}% ± {training_metrics['exact_accuracy_std']:.1f}%")
        
        # 4. VALIDACIÓN CRUZADA AVANZADA (ya realizada en train)
        logger.info("✅ Validación cruzada completada durante el entrenamiento")
        cv_metrics = training_metrics  # Ya son las métricas de CV
        
        logger.info("📊 RESULTADOS DE VALIDACIÓN CRUZADA:")
        logger.info(f"   MAE promedio: {cv_metrics['mae_mean']:.3f} ± {cv_metrics['mae_std']:.3f}")
        logger.info(f"   Precisión promedio: {cv_metrics['accuracy_mean']:.1f}% ± {cv_metrics['accuracy_std']:.1f}%")
        logger.info(f"   Precisión exacta promedio: {cv_metrics['exact_accuracy_mean']:.1f}% ± {cv_metrics['exact_accuracy_std']:.1f}%")
        
        # 5. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS (simplificado)
        logger.info("🔍 Análisis de características completado durante el entrenamiento")
        
        # 6. EVALUACIÓN FINAL
        logger.info("🏆 EVALUACIÓN FINAL DEL SISTEMA:")
        
        # Calcular métricas objetivo
        target_mae = 1.0  # MAE objetivo < 1.0 rebote
        target_accuracy_exact = 97.0  # Precisión exacta objetivo ≥ 97%
        target_accuracy_1 = 95.0  # Precisión ±1 objetivo ≥ 95%
        
        mae_achieved = training_metrics['mae_mean'] <= target_mae
        accuracy_exact_achieved = training_metrics['exact_accuracy_mean'] >= target_accuracy_exact
        accuracy_1_achieved = training_metrics['accuracy_mean'] >= target_accuracy_1
        
        logger.info(f"   🎯 MAE < {target_mae}: {'✅' if mae_achieved else '❌'} ({training_metrics['mae_mean']:.3f})")
        logger.info(f"   🎯 Precisión exacta ≥ {target_accuracy_exact}%: {'✅' if accuracy_exact_achieved else '❌'} ({training_metrics['exact_accuracy_mean']:.1f}%)")
        logger.info(f"   🎯 Precisión ±1 ≥ {target_accuracy_1}%: {'✅' if accuracy_1_achieved else '❌'} ({training_metrics['accuracy_mean']:.1f}%)")
        
        # Evaluación del objetivo principal
        if accuracy_exact_achieved and mae_achieved:
            logger.info("🎉 ¡OBJETIVO ALCANZADO! Sistema listo para producción")
        elif training_metrics['exact_accuracy_mean'] >= 95.0:
            logger.info("🔥 Muy cerca del objetivo - Sistema casi listo")
        elif training_metrics['accuracy_mean'] >= 90.0:
            logger.info("⚡ Buen rendimiento - Necesita optimización adicional")
        else:
            logger.info("⚠️  Necesita más optimización para alcanzar ≥97% precisión exacta")
        
        # 7. RECOMENDACIONES DE MEJORA
        logger.info("💡 RECOMENDACIONES PARA OPTIMIZACIÓN:")
        
        if training_metrics['mae_mean'] > target_mae:
            logger.info("   - Ajustar hiperparámetros del meta-modelo")
            logger.info("   - Añadir más modelos base especializados")
        
        if training_metrics['exact_accuracy_mean'] < target_accuracy_exact:
            logger.info("   - Implementar ensemble de tercer nivel")
            logger.info("   - Añadir características específicas por jugador")
            logger.info("   - Usar datos de tracking avanzado si están disponibles")
            logger.info("   - Optimizar thresholds de predicción")
        
        if training_metrics['accuracy_mean'] < 90.0:
            logger.info("   - Revisar feature engineering")
            logger.info("   - Aumentar tamaño del dataset")
            logger.info("   - Implementar técnicas de data augmentation")
        
        logger.info("✅ TEST COMPLETADO EXITOSAMENTE")
        
        return {
            'training_metrics': training_metrics,
            'cv_metrics': cv_metrics,
            'feature_count': len(feature_names),
            'interaction_count': len(interaction_features),
            'samples': len(X),
            'target_achieved': accuracy_exact_achieved and mae_achieved
        }
        
    except Exception as e:
        logger.error(f"❌ Error en el test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    results = main() 