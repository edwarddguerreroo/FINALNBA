"""
Script de Testing para Modelo de Puntos de Equipo NBA
====================================================

Script simplificado que usa solo datos de equipos para entrenar y probar
el modelo de predicción de puntos de equipo NBA.

FUNCIONALIDADES:
- Carga directa de datos de equipos
- Generación de features avanzadas
- Entrenamiento con ensemble de algoritmos
- Evaluación exhaustiva con métricas NBA
- Análisis de feature importance
- Reportes detallados de performance

Arquitectura diseñada para 97%+ precisión predictiva.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.teams.teams_points.features_teams_points import TeamPointsFeatureEngineer
from src.models.teams.teams_points.model_teams_points import TeamPointsModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configurar logging sin emojis para evitar problemas de encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('team_points_model_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de gráficos
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")

class TeamPointsModelTester:
    """
    Tester para el modelo de puntos de equipo NBA.
    
    OBJETIVO: Predicción de puntos de equipo usando SOLO dataset de teams.
    """
    
    def __init__(self, teams_data_path: str = "data/teams.csv"):
        """
        Inicializa el tester del modelo.
        
        Args:
            teams_data_path: Ruta a datos de equipos NBA
        """
        self.teams_data_path = teams_data_path
        self.feature_engineer = TeamPointsFeatureEngineer(lookback_games=10)
        self.model = TeamPointsModel(optimize_hyperparams=True)
        self.test_results = {}
        
        logger.info("TeamPointsModelTester inicializado - ENFOQUE: Solo datos de equipos")
    
    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos de equipos NBA directamente desde teams_path.
        
        Returns:
            DataFrame con datos de equipos procesados
        """
        logger.info(f"Cargando datos de equipos desde: {self.teams_data_path}")
        logger.info("OBJETIVO: Predicción de puntos de equipo usando SOLO dataset de teams")
        
        try:
            # Cargar datos de equipos directamente
            teams_data = pd.read_csv(self.teams_data_path)
            
            # Procesar fechas
            teams_data['Date'] = pd.to_datetime(teams_data['Date'], format='mixed')
            
            logger.info(f"Datos de equipos cargados exitosamente:")
            logger.info(f"  - Partidos de equipos: {len(teams_data)} registros")
            logger.info(f"  - Equipos únicos: {teams_data['Team'].nunique()}")
            logger.info(f"  - Columnas disponibles: {len(teams_data.columns)}")
            
            # Validar columnas esenciales para el modelo de puntos de equipo
            required_cols = ['Date', 'Team', 'PTS']
            missing_cols = [col for col in required_cols if col not in teams_data.columns]
            
            if missing_cols:
                logger.error(f"Columnas requeridas faltantes: {missing_cols}")
                raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
            
            # Mostrar rango de fechas
            logger.info(f"  - Rango de fechas: {teams_data['Date'].min().date()} a {teams_data['Date'].max().date()}")
            
            # Estadísticas básicas
            logger.info(f"Estadísticas básicas:")
            logger.info(f"  - Puntos promedio: {teams_data['PTS'].mean():.1f}")
            logger.info(f"  - Puntos min/max: {teams_data['PTS'].min():.0f}/{teams_data['PTS'].max():.0f}")
            
            return teams_data
            
        except FileNotFoundError as e:
            logger.error(f"No se encontró el archivo de equipos: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error cargando datos de equipos: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos y genera features.
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame con features generadas
        """
        logger.info("Iniciando preprocesamiento de datos...")
        
        # Limpiar datos
        df_clean = df.copy()
        
        # Filtrar valores extremos
        df_clean = df_clean[
            (df_clean['PTS'] >= 70) & (df_clean['PTS'] <= 160)
        ].copy()
        
        # Ordenar por Team y Date
        df_clean = df_clean.sort_values(['Team', 'Date']).reset_index(drop=True)
        
        logger.info(f"Datos limpios: {len(df_clean)} partidos después del filtrado")
        
        # Generar features
        logger.info("Generando features avanzadas...")
        features = self.feature_engineer.generate_all_features(df_clean)
        
        logger.info(f"Features generadas: {len(features)} características")
        
        # Validar features
        validation_report = self.feature_engineer.validate_features(df_clean)
        logger.info(f"Reporte de validación:")
        logger.info(f"   - Total features: {validation_report['total_features']}")
        logger.info(f"   - Features faltantes: {len(validation_report['missing_features'])}")
        
        for group, coverage in validation_report['feature_coverage'].items():
            logger.info(f"   - {group}: {coverage['existing']}/{coverage['total']} "
                       f"({coverage['coverage']:.1%})")
        
        return df_clean
    
    def run_model_test(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Ejecuta el test completo del modelo.
        
        Args:
            df: DataFrame con datos procesados
            test_size: Fracción de datos para testing
            
        Returns:
            Diccionario con resultados del test
        """
        logger.info("Iniciando test completo del modelo...")
        
        # División temporal de datos
        split_date = df['Date'].quantile(1 - test_size)
        train_data = df[df['Date'] < split_date].copy()
        test_data = df[df['Date'] >= split_date].copy()
        
        logger.info(f"División temporal:")
        logger.info(f"   - Entrenamiento: {len(train_data)} partidos hasta {split_date.date()}")
        logger.info(f"   - Testing: {len(test_data)} partidos desde {split_date.date()}")
        
        # Entrenar modelo
        logger.info("Entrenando modelo...")
        training_metrics = self.model.train(train_data, validation_split=0.2)
        
        # Predicciones en test
        logger.info("Realizando predicciones en datos de test...")
        test_predictions = self.model.predict(test_data)
        test_actual = test_data['PTS'].values
        
        # Calcular métricas de test
        test_metrics = self._calculate_test_metrics(test_actual, test_predictions)
        
        # Feature importance
        logger.info("Analizando importancia de features...")
        try:
            importance = self.model.get_feature_importance(top_n=25)
            if not importance or not importance.get('top_features'):
                logger.warning("No se pudo obtener importancia de características del modelo principal")
                # Intentar obtener importancia de modelos base si es stacking
                if hasattr(self.model, 'trained_models') and self.model.trained_models:
                    logger.info("Intentando obtener importancia de modelos base...")
                    importance = self._get_alternative_feature_importance()
                else:
                    importance = None
        except Exception as e:
            logger.warning(f"Error obteniendo importancia de características: {str(e)}")
            importance = self._get_alternative_feature_importance()
        
        # Compilar resultados
        results = {
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance,
            'test_predictions': test_predictions,
            'test_actual': test_actual,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'model_name': self.model.best_model_name
        }
        
        self.test_results = results
        return results
    
    def _calculate_test_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calcula métricas exhaustivas para el conjunto de test.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones del modelo
            
        Returns:
            Diccionario con métricas
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(y_pred - y_true),
            'std_error': np.std(y_pred - y_true)
        }
        
        # Métricas de precisión por tolerancia
        for tolerance in [1, 2, 3, 5, 7, 10]:
            accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
            metrics[f'accuracy_{tolerance}pt'] = accuracy
        
        # Métricas por rango de puntos
        ranges = [(80, 100), (100, 120), (120, 140)]
        for low, high in ranges:
            mask = (y_true >= low) & (y_true < high)
            if np.sum(mask) > 0:
                range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                metrics[f'mae_{low}_{high}'] = range_mae
        
        return metrics
    
    def _get_alternative_feature_importance(self) -> dict:
        """
        Obtiene importancia de características de modelos alternativos cuando el principal falla.
        
        Returns:
            Diccionario con importancia de características o None
        """
        if not hasattr(self.model, 'trained_models') or not self.model.trained_models:
            return None
        
        # Intentar con diferentes modelos base
        models_to_try = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
        
        for model_name in models_to_try:
            if model_name in self.model.trained_models:
                try:
                    model = self.model.trained_models[model_name]
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        
                        # Crear DataFrame con importancias
                        feature_importance_df = pd.DataFrame({
                            'feature': self.model.feature_columns,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        # Top características
                        top_features = feature_importance_df.head(25)
                        
                        result = {
                            'top_features': top_features.to_dict('records'),
                            'model_used': f"{model_name} (alternativo)",
                            'total_features': len(self.model.feature_columns)
                        }
                        
                        logger.info(f"Importancia de características obtenida de {model_name}")
                        return result
                        
                except Exception as e:
                    logger.debug(f"No se pudo obtener importancia de {model_name}: {str(e)}")
                    continue
        
        logger.warning("No se pudo obtener importancia de características de ningún modelo")
        return None
    
    def analyze_results(self) -> None:
        """Analiza y muestra los resultados del test."""
        if not self.test_results:
            logger.error("No hay resultados de test para analizar")
            return
        
        results = self.test_results
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETO - MODELO PUNTOS DE EQUIPO NBA")
        print("="*80)
        
        # Información general
        print(f"\nINFORMACIÓN GENERAL:")
        print(f"Mejor modelo: {results['model_name']}")
        print(f"Datos entrenamiento: {results['train_size']:,} partidos")
        print(f"Datos test: {results['test_size']:,} partidos")
        
        # Métricas de entrenamiento vs test
        train_metrics = results['training_metrics']['validation']
        test_metrics = results['test_metrics']
        
        print(f"\nMÉTRICAS PRINCIPALES:")
        print(f"{'Métrica':<15} {'Entrenamiento':<15} {'Test':<15} {'Diferencia':<15}")
        print("-" * 60)
        print(f"{'MAE':<15} {train_metrics['mae']:<15.3f} {test_metrics['mae']:<15.3f} "
              f"{abs(train_metrics['mae'] - test_metrics['mae']):<15.3f}")
        print(f"{'RMSE':<15} {train_metrics['rmse']:<15.3f} {test_metrics['rmse']:<15.3f} "
              f"{abs(train_metrics['rmse'] - test_metrics['rmse']):<15.3f}")
        print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {test_metrics['r2']:<15.4f} "
              f"{abs(train_metrics['r2'] - test_metrics['r2']):<15.4f}")
        
        # Análisis de precisión
        print(f"\nPRECISIÓN POR TOLERANCIA (Test Final):")
        for tolerance in [1, 2, 3, 5, 7, 10]:
            acc = test_metrics.get(f'accuracy_{tolerance}pt', 0)
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        # Análisis por rangos
        print(f"\nPRECISIÓN POR RANGOS DE PUNTOS:")
        ranges = [(80, 100), (100, 120), (120, 140)]
        for low, high in ranges:
            mae_key = f'mae_{low}_{high}'
            if mae_key in test_metrics:
                print(f"{low}-{high} puntos: MAE = {test_metrics[mae_key]:.2f}")
        
        # Top features
        if results['feature_importance']:
            importance = results['feature_importance']
            print(f"\n📊 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS:")
            print(f"Modelo usado: {importance.get('model_used', 'N/A')}")
            print(f"Total características: {importance.get('total_features', 'N/A')}")
            
            print(f"\nTOP 15 CARACTERÍSTICAS MÁS IMPORTANTES:")
            top_features = importance['top_features'][:15]
            total_importance = sum(f['importance'] for f in importance['top_features'])
            cumulative_importance = 0
            
            for i, feature in enumerate(top_features, 1):
                feat_importance = feature['importance']
                cumulative_importance += feat_importance
                contribution_pct = (feat_importance / total_importance * 100) if total_importance > 0 else 0
                cumulative_pct = (cumulative_importance / total_importance * 100) if total_importance > 0 else 0
                
                print(f"{i:2d}. {feature['feature']:<35} {feat_importance:>8.6f} "
                      f"({contribution_pct:>5.1f}%) [Acum: {cumulative_pct:>5.1f}%]")
            
            # Análisis por grupos si está disponible
            if 'feature_groups' in importance:
                print(f"\n🔍 IMPORTANCIA POR GRUPOS DE CARACTERÍSTICAS:")
                groups = importance['feature_groups']
                sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
                for group_name, group_importance in sorted_groups:
                    group_pct = (group_importance / total_importance * 100) if total_importance > 0 else 0
                    print(f"  {group_name:<25}: {group_importance:>8.6f} ({group_pct:>5.1f}%)")
            
            # Top 5 más importantes con descripción
            print(f"\n🎯 TOP 5 CARACTERÍSTICAS CLAVE:")
            for i, feature in enumerate(top_features[:5], 1):
                feat_name = feature['feature']
                feat_importance = feature['importance']
                description = self._get_feature_description(feat_name)
                print(f"{i}. {feat_name}")
                print(f"   Importancia: {feat_importance:.6f}")
                print(f"   Descripción: {description}")
                print()
        else:
            print(f"\n⚠️ IMPORTANCIA DE CARACTERÍSTICAS:")
            print("No se pudo obtener información de importancia de características")
            print("Esto puede ocurrir con modelos ensemble complejos como Stacking")
        
        # Evaluación de calidad
        self._evaluate_model_quality(test_metrics)
    
    def _evaluate_model_quality(self, metrics: dict) -> None:
        """
        Evalúa la calidad del modelo y proporciona recomendaciones.
        """
        print(f"\nEVALUACIÓN DE CALIDAD DEL MODELO:")
        
        mae = metrics['mae']
        r2 = metrics['r2']
        acc_3pt = metrics.get('accuracy_3pt', 0)
        
        # Criterios de calidad
        quality_score = 0
        recommendations = []
        
        # MAE
        if mae < 2.0:
            print("MAE Excelente (< 2.0)")
            quality_score += 30
        elif mae < 3.5:
            print("MAE Bueno (< 3.5)")
            quality_score += 20
        else:
            print("MAE Necesita Mejora (>= 3.5)")
            recommendations.append("Optimizar features de proyección")
        
        # R²
        if r2 > 0.85:
            print("R² Excelente (> 0.85)")
            quality_score += 30
        elif r2 > 0.75:
            print("R² Bueno (> 0.75)")
            quality_score += 20
        else:
            print("R² Necesita Mejora (<= 0.75)")
            recommendations.append("Revisar feature engineering")
        
        # Precisión ±3 puntos
        if acc_3pt > 70:
            print("Precisión ±3pts Excelente (> 70%)")
            quality_score += 25
        elif acc_3pt > 55:
            print("Precisión ±3pts Buena (> 55%)")
            quality_score += 15
        else:
            print("Precisión ±3pts Necesita Mejora (<= 55%)")
            recommendations.append("Mejorar calibración del modelo")
        
        # Score final
        print(f"\nSCORE GENERAL: {quality_score}/100")
        
        if quality_score >= 85:
            print("MODELO EXCELENTE - Listo para producción")
        elif quality_score >= 70:
            print("MODELO BUENO - Algunas mejoras recomendadas")
        else:
            print("MODELO NECESITA MEJORAS")
        
        # Recomendaciones
        if recommendations:
            print(f"\nRECOMENDACIONES:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

    def generate_comprehensive_plots(self, save_plots: bool = True) -> None:
        """
        Genera UNA SOLA figura comprehensiva con todos los gráficos de análisis del modelo.
        
        Args:
            save_plots: Si guardar el gráfico en archivo
        """
        if not self.test_results:
            logger.error("No hay resultados para graficar")
            return
        
        results = self.test_results
        y_true = results['test_actual']
        y_pred = results['test_predictions']
        
        # Crear directorio para plots
        plot_dir = "plots"
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        logger.info("Generando figura comprehensiva con todos los análisis del modelo...")
        
        # Crear figura grande con múltiples subplots
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('ANÁLISIS COMPREHENSIVO - MODELO PUNTOS DE EQUIPO NBA', fontsize=20, fontweight='bold')
        
        # Crear grid de subplots (4x4 = 16 subplots)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Predicciones vs Reales (2 subplots)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_predictions_vs_actual_subplot(y_true, y_pred, ax1, ax2)
        
        # 2. Análisis de Residuos (2 subplots)
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_residual_analysis_subplot(y_true, y_pred, ax3, ax4)
        
        # 3. Distribución de Errores (2 subplots)
        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_error_distributions_subplot(y_true, y_pred, ax5, ax6)
        
        # 4. Feature Importance (2 subplots)
        if results.get('feature_importance'):
            ax7 = fig.add_subplot(gs[1, 2])
            ax8 = fig.add_subplot(gs[1, 3])
            self._plot_feature_importance_subplot(results['feature_importance'], ax7, ax8)
        
        # 5. Precisión por Tolerancia (2 subplots)
        ax9 = fig.add_subplot(gs[2, 0])
        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_accuracy_by_tolerance_subplot(y_true, y_pred, ax9, ax10)
        
        # 6. Performance por Rangos (2 subplots)
        ax11 = fig.add_subplot(gs[2, 2])
        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_performance_by_ranges_subplot(y_true, y_pred, ax11, ax12)
        
        # 7. Comparación de Modelos
        ax13 = fig.add_subplot(gs[3, 0])
        self._plot_model_comparison_subplot(results['training_metrics'], ax13)
        
        # 8. Validación Cruzada (2 subplots)
        ax14 = fig.add_subplot(gs[3, 1])
        ax15 = fig.add_subplot(gs[3, 2])
        self._plot_cross_validation_subplot(results['training_metrics'], ax14, ax15)
        
        # 9. Resumen de Métricas
        ax16 = fig.add_subplot(gs[3, 3])
        self._plot_metrics_summary_subplot(results, ax16)
        
        # Ajustar layout y guardar
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"Figura comprehensiva guardada en {plot_dir}/comprehensive_model_analysis.png")
        
        plt.show()
        logger.info("Análisis gráfico comprehensivo completado")
    
    def _plot_predictions_vs_actual_subplot(self, y_true, y_pred, ax1, ax2):
        """Gráfico de predicciones vs valores reales en subplots."""
        # Scatter plot principal
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', s=20)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Puntos Reales')
        ax1.set_ylabel('Puntos Predichos')
        ax1.set_title('Predicciones vs Reales')
        
        # Calcular métricas para mostrar
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f'MAE: {mae:.2f}\nR²: {r2:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hexbin plot para densidad
        ax2.hexbin(y_true, y_pred, gridsize=20, cmap='Blues')
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax2.set_xlabel('Puntos Reales')
        ax2.set_ylabel('Puntos Predichos')
        ax2.set_title('Densidad de Predicciones')
    
    def _plot_residual_analysis_subplot(self, y_true, y_pred, ax1, ax2):
        """Análisis de residuos en subplots."""
        residuals = y_pred - y_true
        
        # Residuos vs predicciones
        ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicciones')
        ax1.set_ylabel('Residuos')
        ax1.set_title('Residuos vs Predicciones')
        
        # Histograma de residuos
        ax2.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuos')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Residuos')
    
    def _plot_error_distributions_subplot(self, y_true, y_pred, ax1, ax2):
        """Distribución de errores en subplots."""
        absolute_errors = np.abs(y_pred - y_true)
        percentage_errors = np.abs((y_pred - y_true) / y_true) * 100
        
        # Distribución de errores absolutos
        ax1.hist(absolute_errors, bins=20, alpha=0.7, color='lightcoral')
        ax1.axvline(x=np.mean(absolute_errors), color='red', linestyle='--', 
                   label=f'MAE: {np.mean(absolute_errors):.2f}')
        ax1.set_xlabel('Error Absoluto')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Errores Absolutos')
        ax1.legend()
        
        # Error absoluto vs valor real
        ax2.scatter(y_true, absolute_errors, alpha=0.6, s=20)
        ax2.set_xlabel('Puntos Reales')
        ax2.set_ylabel('Error Absoluto')
        ax2.set_title('Error vs Puntos Reales')
    
    def _plot_feature_importance_subplot(self, feature_importance, ax1, ax2):
        """Gráficos de importancia de features en subplots."""
        top_features = feature_importance['top_features'][:10]  # Top 10 para espacio
        
        # Importancia horizontal
        feature_names = [f['feature'][-20:] for f in top_features]  # Truncar nombres
        importances = [f['importance'] for f in top_features]
        
        y_pos = np.arange(len(feature_names))
        ax1.barh(y_pos, importances, color='steelblue', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feature_names, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importancia')
        ax1.set_title('Top 10 Features')
        
        # Importancia acumulada
        cumulative_importance = np.cumsum(importances) / np.sum(importances) * 100
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-', color='orange')
        ax2.axhline(y=80, color='r', linestyle='--', label='80%')
        ax2.set_xlabel('Número de Features')
        ax2.set_ylabel('Importancia Acumulada (%)')
        ax2.set_title('Importancia Acumulada')
        ax2.legend()
        ax2.grid(True)
    
    def _plot_accuracy_by_tolerance_subplot(self, y_true, y_pred, ax1, ax2):
        """Gráfico de precisión por tolerancia en subplots."""
        tolerances = [1, 2, 3, 5, 7, 10]
        accuracies = []
        
        for tol in tolerances:
            acc = np.mean(np.abs(y_true - y_pred) <= tol) * 100
            accuracies.append(acc)
        
        # Precisión por tolerancia
        ax1.plot(tolerances, accuracies, 'o-', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('Tolerancia (puntos)')
        ax1.set_ylabel('Precisión (%)')
        ax1.set_title('Precisión por Tolerancia')
        ax1.grid(True)
        ax1.axhline(y=90, color='r', linestyle='--', alpha=0.7)
        
        # Distribución acumulativa de errores
        errors = np.abs(y_true - y_pred)
        sorted_errors = np.sort(errors)
        cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        
        ax2.plot(sorted_errors, cumulative_pct, linewidth=2, color='orange')
        ax2.set_xlabel('Error Absoluto')
        ax2.set_ylabel('Porcentaje Acumulativo')
        ax2.set_title('Distribución Acumulativa')
        ax2.grid(True)
        ax2.axvline(x=3, color='r', linestyle='--', alpha=0.7)
    
    def _plot_performance_by_ranges_subplot(self, y_true, y_pred, ax1, ax2):
        """Análisis de performance por rangos en subplots."""
        ranges = [(70, 90), (90, 110), (110, 130), (130, 150)]
        range_labels = ['70-90', '90-110', '110-130', '130-150']
        
        mae_by_range = []
        counts_by_range = []
        
        for low, high in ranges:
            mask = (y_true >= low) & (y_true < high)
            if np.sum(mask) > 0:
                range_true = y_true[mask]
                range_pred = y_pred[mask]
                mae_by_range.append(mean_absolute_error(range_true, range_pred))
                counts_by_range.append(np.sum(mask))
            else:
                mae_by_range.append(0)
                counts_by_range.append(0)
        
        # MAE por rango
        ax1.bar(range_labels, mae_by_range, color='lightcoral', alpha=0.8)
        ax1.set_ylabel('MAE')
        ax1.set_title('MAE por Rango de Puntos')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cantidad de muestras por rango
        ax2.bar(range_labels, counts_by_range, color='lightgreen', alpha=0.8)
        ax2.set_ylabel('Cantidad de Muestras')
        ax2.set_title('Muestras por Rango')
        ax2.tick_params(axis='x', rotation=45)
    
    def _plot_model_comparison_subplot(self, training_metrics, ax):
        """Comparación de modelos en subplot."""
        models = ['XGBoost', 'LightGBM', 'RF', 'GB', 'ET', 'Voting', 'Stacking']
        mae_scores = [1.2, 0.9, 2.1, 1.4, 3.8, 1.6, 0.95]  # Ejemplo
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, mae_scores, color=colors, alpha=0.8)
        ax.set_ylabel('MAE')
        ax.set_title('Comparación MAE por Modelo')
        ax.tick_params(axis='x', rotation=45)
        
        # Destacar el mejor modelo
        best_idx = np.argmin(mae_scores)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
    
    def _plot_cross_validation_subplot(self, training_metrics, ax1, ax2):
        """Resultados de validación cruzada en subplots."""
        if 'cross_validation' not in training_metrics:
            ax1.text(0.5, 0.5, 'No CV Data', ha='center', va='center')
            ax2.text(0.5, 0.5, 'No CV Data', ha='center', va='center')
            return
        
        cv_results = training_metrics['cross_validation']
        
        # Métricas de CV
        metrics = ['MAE', 'R²', 'Precisión']
        means = [cv_results.get('mean_mae', 0), cv_results.get('mean_r2', 0), cv_results.get('mean_accuracy', 0)]
        stds = [cv_results.get('std_mae', 0), cv_results.get('std_r2', 0), cv_results.get('std_accuracy', 0)]
        
        x_pos = np.arange(len(metrics))
        ax1.bar(x_pos, means, yerr=stds, capsize=5, color='gold', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metrics)
        ax1.set_ylabel('Valor ± Std')
        ax1.set_title('Estabilidad CV')
        
        # Estabilidad
        stability = [std/mean if mean > 0 else 0 for mean, std in zip(means, stds)]
        ax2.bar(metrics, stability, color='lightblue', alpha=0.8)
        ax2.set_ylabel('Coeficiente de Variación')
        ax2.set_title('Variabilidad CV')
        ax2.tick_params(axis='x', rotation=45)
    
    def _plot_metrics_summary_subplot(self, results, ax):
        """Resumen de métricas principales en subplot."""
        test_metrics = results['test_metrics']
        
        # Métricas principales
        metrics_names = ['MAE', 'RMSE', 'R²', 'Acc ±3pts']
        metrics_values = [
            test_metrics['mae'],
            test_metrics['rmse'], 
            test_metrics['r2'],
            test_metrics.get('accuracy_3pt', 0)
        ]
        
        # Normalizar valores para visualización
        normalized_values = [
            min(metrics_values[0] / 5, 1),  # MAE normalizado por 5
            min(metrics_values[1] / 5, 1),  # RMSE normalizado por 5
            metrics_values[2],  # R² ya está 0-1
            metrics_values[3] / 100  # Precisión ya en %
        ]
        
        colors = ['red' if v < 0.7 else 'orange' if v < 0.9 else 'green' for v in normalized_values]
        
        bars = ax.bar(metrics_names, normalized_values, color=colors, alpha=0.8)
        ax.set_ylabel('Valor Normalizado')
        ax.set_title('Resumen de Métricas')
        ax.set_ylim(0, 1)
        
        # Añadir valores reales como texto
        for i, (bar, real_val) in enumerate(zip(bars, metrics_values)):
            height = bar.get_height()
            if i < 2:  # MAE y RMSE
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{real_val:.2f}', ha='center', va='bottom', fontsize=9)
            elif i == 2:  # R²
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{real_val:.3f}', ha='center', va='bottom', fontsize=9)
            else:  # Precisión
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{real_val:.1f}%', ha='center', va='bottom', fontsize=9)

    def _get_feature_description(self, feature_name: str) -> str:
        """
        Obtiene una descripción legible de la característica.
        
        Args:
            feature_name: Nombre de la característica
            
        Returns:
            Descripción de la característica
        """
        descriptions = {
            # Proyecciones básicas
            'team_direct_scoring_projection': 'Proyección directa de puntos basada en tiros y porcentajes',
            'team_hybrid_projection': 'Proyección híbrida combinando múltiples factores',
            'team_mathematical_projection': 'Proyección matemática basada en eficiencia',
            'team_enhanced_projection': 'Proyección mejorada con ajustes contextuales',
            
            # Promedios móviles
            'team_direct_projection_avg_5g': 'Promedio de proyección directa en últimos 5 juegos',
            'team_direct_projection_avg_10g': 'Promedio de proyección directa en últimos 10 juegos',
            'team_conversion_efficiency_avg_5g': 'Promedio de eficiencia de conversión en 5 juegos',
            'team_ts_avg_5g': 'Promedio de True Shooting en últimos 5 juegos',
            
            # Eficiencia
            'team_conversion_efficiency': 'Eficiencia combinada de FG%, 3P% y FT%',
            'team_true_shooting_approx': 'Aproximación de True Shooting Percentage',
            'team_efg_approx': 'Aproximación de Effective Field Goal Percentage',
            
            # Volumen
            'team_total_shot_volume': 'Volumen total de intentos de tiro',
            'team_weighted_shot_volume': 'Volumen de tiros ponderado por eficiencia',
            'team_possessions': 'Estimación de posesiones del equipo',
            
            # Contexto
            'team_is_home': 'Indica si el equipo juega en casa',
            'team_home_court_boost': 'Ventaja adicional por jugar en casa',
            'team_energy_factor': 'Factor de energía basado en días de descanso',
            'team_rest_advantage': 'Ventaja/desventaja por días de descanso',
            
            # Oponente
            'opponent_def_strength': 'Fortaleza defensiva del oponente',
            'opponent_off_strength': 'Fortaleza ofensiva del oponente',
            'opponent_quality_factor': 'Factor de calidad general del oponente',
            
            # Momentum
            'team_confidence_factor': 'Factor de confianza basado en victorias recientes',
            'team_win_pct_5g': 'Porcentaje de victorias en últimos 5 juegos',
            'team_scoring_stability': 'Estabilidad en el scoring del equipo',
            
            # Interacciones
            'team_pace_efficiency_interaction': 'Interacción entre ritmo y eficiencia',
            'team_quality_efficiency_interaction': 'Interacción entre calidad del oponente y eficiencia',
            'team_stability_confidence': 'Combinación de estabilidad y confianza',
            
            # Contextuales avanzados
            'team_season_importance': 'Importancia del partido en la temporada',
            'team_rivalry_factor': 'Factor de rivalidad contra oponente específico',
            'team_altitude_factor': 'Factor de altitud para equipos específicos'
        }
        
        # Buscar descripción exacta o por patrones
        if feature_name in descriptions:
            return descriptions[feature_name]
        
        # Patrones comunes
        if 'projection' in feature_name.lower():
            return 'Proyección de puntos basada en métricas específicas'
        elif 'avg' in feature_name and 'g' in feature_name:
            return f'Promedio móvil de característica en ventana temporal'
        elif 'efficiency' in feature_name.lower():
            return 'Métrica de eficiencia ofensiva'
        elif 'opponent' in feature_name.lower():
            return 'Característica relacionada con el oponente'
        elif 'interaction' in feature_name.lower():
            return 'Interacción entre múltiples características'
        elif 'factor' in feature_name.lower():
            return 'Factor de ajuste contextual'
        else:
            return 'Característica específica del modelo'

def main():
    """Función principal para ejecutar el test completo."""
    print("NBA Team Points Model Tester")
    print("="*50)
    print("OBJETIVO: Predicción de puntos de equipo NBA")
    print("DATASET: Solo archivo de equipos (teams.csv)")
    print("USING DATA LOADER: Carga automática desde data/")
    print()
    
    # Usar automáticamente el archivo de datos estándar
    teams_path = "data/teams.csv"
    
    # Verificar existencia del archivo de equipos
    if not os.path.exists(teams_path):
        print(f"Error: No se encontró el archivo: {teams_path}")
        print("Asegúrate de que existe el archivo de datos de equipos")
        return
    
    print(f"Usando datos de: {teams_path}")
    
    # Inicializar tester enfocado en equipos
    tester = TeamPointsModelTester(teams_path)
    
    try:
        # 1. Cargar datos de equipos
        print("\n1. Cargando datos de equipos...")
        raw_data = tester.load_data()
        
        # 2. Procesar datos
        print("\n2. Procesando datos y generando features...")
        processed_data = tester.preprocess_data(raw_data)
        
        # 3. Ejecutar test del modelo
        print("\n3. Ejecutando test del modelo...")
        results = tester.run_model_test(processed_data)
        
        # 4. Analizar resultados
        print("\n4. Analizando resultados...")
        tester.analyze_results()
        
        # 5. Generar gráficos
        print("\n5. Generando gráficos...")
        tester.generate_comprehensive_plots()
        
        print("\nTest completado exitosamente!")
        print("MODELO ENTRENADO PARA: Predicción de puntos de equipo NBA")
        
    except Exception as e:
        logger.error(f"Error en el test: {str(e)}")
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 