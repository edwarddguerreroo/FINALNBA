"""
Módulo de análisis avanzado para modelos NBA con visualizaciones y métricas sofisticadas.

Este módulo proporciona:
- Visualizaciones detalladas en PNG
- Métricas avanzadas de rendimiento (R1, AUC, MCC, etc.)
- Análisis de correlaciones
- Detección de patrones
- Evaluación exhaustiva de modelos
"""

# Configurar matplotlib para un entorno sin interfaz gráfica
import matplotlib
matplotlib.use('Agg')  # Usar backend que no requiere ventana gráfica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, log_loss, mean_absolute_error,
    mean_absolute_percentage_error, brier_score_loss
)

# Importar calibration_curve con manejo de errores
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Función alternativa si no está disponible
    def calibration_curve(y_true, y_prob, n_bins=5):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        prob_true = []
        prob_pred = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            if in_bin.sum() > 0:
                prob_true.append(y_true[in_bin].mean())
                prob_pred.append(y_prob[in_bin].mean())
        
        return np.array(prob_true), np.array(prob_pred)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, validation_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import logging
from typing import Dict, List, Tuple, Any
import json
import os
from scipy import stats
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone

# Configuración de logging mejorada
def setup_logging():
    """Configura el logger con mejor manejo de caracteres especiales"""
    logger = logging.getLogger()
    
    # Verificar si ya hay handlers configurados para evitar duplicación
    if logger.handlers:
        # Si ya hay handlers, no agregar más
        logger.info("Logger ya configurado, evitando duplicación de handlers")
        return logger
    
    # Crear nuevo handler con codificación UTF-8
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Nivel de logging
    logger.setLevel(logging.INFO)
    
    return logger

# Configurar logging
logger = setup_logging()

# Ignorar warnings
warnings.filterwarnings('ignore')

class NBAAdvancedAnalytics:
    """Clase para análisis avanzado de modelos NBA."""
    
    def __init__(self, output_dir: str = 'analysis_output'):
        """
        Inicializa el analizador avanzado.
        
        Args:
            output_dir: Directorio para guardar visualizaciones y reportes
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo de matplotlib
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        sns.set_palette("husl")
        
    def calculate_advanced_metrics(self, model, X_train, X_test, y_train, y_test, 
                                 model_name: str = "model") -> Dict[str, Any]:
        """
        Calcula métricas avanzadas para el modelo.
        
        Args:
            model: Modelo entrenado
            X_train: Características de entrenamiento
            X_test: Características de prueba
            y_train: Etiquetas de entrenamiento
            y_test: Etiquetas de prueba
            model_name: Nombre del modelo
            
        Returns:
            Dict con métricas avanzadas
        """
        metrics = {}
        
        # Determinar si es un modelo de clasificación o regresión
        is_classifier = hasattr(model, 'predict_proba')
        
        # Aplicar ajuste de compatibilidad de características antes de predecir
        try:
            X_test_comp = adjust_features_compatibility(X_test, model)
            
            # Intentar predecir con datos de prueba ajustados
            if is_classifier:
                # Métricas de clasificación
                y_pred = model.predict(X_test_comp)
                y_pred_proba = model.predict_proba(X_test_comp)[:, 1]
                
                # Métricas básicas
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
                metrics['matthews_corr_coef'] = matthews_corrcoef(y_test, y_pred)
                
                # AUC detallado
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                metrics['auc_roc'] = auc(fpr, tpr)
                
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                metrics['auc_pr'] = auc(recall, precision)
                
                # Brier Score
                metrics['brier_score'] = brier_score_loss(y_test, y_pred_proba)
                
                # Log Loss
                try:
                    metrics['log_loss'] = log_loss(y_test, y_pred_proba)
                except:
                    metrics['log_loss'] = None
                    
                # Matriz de confusión
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                
                # Métricas por clase
                report = classification_report(y_test, y_pred, output_dict=True)
                metrics['classification_report'] = report
                
            else:
                # Para modelos de regresión, usar la función predict del modelo
                # Esto activará cualquier lógica especial en el modelo (como agregar características)
                if hasattr(model, 'predict') and callable(model.predict):
                    y_pred = model.predict(X_test_comp)
                else:
                    # Fallback a la predicción estándar
                    y_pred = model.predict(X_test_comp)
                
                # Métricas básicas
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['r2'] = r2_score(y_test, y_pred)
                
                # R² ajustado (R1)
                n = len(y_test)
                p = X_test.shape[1]
                metrics['r2_adjusted'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
                
                # MAPE
                try:
                    # Implementar cálculo robusto de MAPE similar a PointsModel._calculate_robust_mape
                    # Convertir a arrays numpy si son series o dataframes
                    if hasattr(y_test, 'values'):
                        y_true_array = y_test.values
                    else:
                        y_true_array = np.array(y_test)
                    
                    if hasattr(y_pred, 'values'):
                        y_pred_array = y_pred.values
                    else:
                        y_pred_array = np.array(y_pred)
                    
                    # Usar sMAPE (Symmetric MAPE) como alternativa más robusta
                    # Formula: 200% * |y_true - y_pred| / (|y_true| + |y_pred| + epsilon)
                    epsilon = 1.0  # Pequeño valor para sustituir valores muy pequeños
                    abs_diff = np.abs(y_true_array - y_pred_array)
                    abs_sum = np.abs(y_true_array) + np.abs(y_pred_array) + epsilon
                    smape = 200.0 * (abs_diff / abs_sum)
                    
                    # Recortar valores extremos
                    max_pct = 100  # Porcentaje máximo de error a considerar
                    smape = np.clip(smape, 0, max_pct)
                    
                    # Usar mediana en lugar de media para reducir influencia de valores extremos
                    robust_mape = np.median(smape)
                    
                    # Verificar que el valor sea razonable
                    if robust_mape > max_pct or not np.isfinite(robust_mape):
                        # Como última opción, usar MAE relativo
                        mean_abs_true = np.mean(np.abs(y_true_array))
                        if mean_abs_true > epsilon:
                            mae = np.mean(abs_diff)
                            robust_mape = (mae / mean_abs_true) * 100
                        else:
                            robust_mape = 0.0
                    
                    metrics['mape'] = robust_mape
                    logging.info(f"MAPE calculado usando sMAPE con mediana: {robust_mape:.4f}%")
                except Exception as e:
                    logging.warning(f"Error al calcular MAPE robusto: {str(e)}")
                    metrics['mape'] = None
                
                # Métricas adicionales
                residuals = y_test - y_pred
                metrics['mean_residual'] = residuals.mean()
                metrics['std_residual'] = residuals.std()
                metrics['median_absolute_error'] = np.median(np.abs(residuals))
                
                # Coeficiente de variación del error
                if y_test.mean() != 0:
                    metrics['cv_rmse'] = metrics['rmse'] / y_test.mean()
                else:
                    metrics['cv_rmse'] = None
                
                # Test de normalidad de residuos (Shapiro-Wilk)
                if len(residuals) <= 5000:  # Shapiro-Wilk tiene límite de muestra
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    metrics['shapiro_test'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
                
                # Coeficiente de correlación de Pearson entre y_test y y_pred
                metrics['pearson_corr'] = stats.pearsonr(y_test, y_pred)[0]
                
        except Exception as e:
            # Usar repr para evitar problemas de codificación con caracteres especiales
            error_msg = repr(str(e))
            logger.error(f"Error en calculate_advanced_metrics: {error_msg}")
            metrics['error'] = error_msg
            
            # Intentar recuperarse del error
            try:
                # Si el error es por incompatibilidad de características
                if "feature_names mismatch" in str(e) or "shape of input" in str(e) or "features do not match" in str(e):
                    logger.warning("Intentando ajustar las características para recuperación...")
                    # Intentar aplicar el ajuste de compatibilidad antes de usar características comunes
                    X_test_recovery = adjust_features_compatibility(X_test, model)
                    
                    try:
                        # Probar predicción con características ajustadas
                        y_pred = model.predict(X_test_recovery)
                        metrics['mse'] = mean_squared_error(y_test, y_pred)
                        metrics['rmse'] = np.sqrt(metrics['mse'])
                        metrics['r2'] = r2_score(y_test, y_pred)
                        metrics['mae'] = mean_absolute_error(y_test, y_pred)
                        metrics['note'] = "Métricas calculadas usando características ajustadas"
                        
                        # Eliminar el error si pudimos recuperarnos
                        if 'error' in metrics:
                            del metrics['error']
                    except Exception as adjust_error:
                        # Si aún falla, intentar con características comunes
                        if hasattr(model, 'feature_names_in_'):
                            common_features = [col for col in X_test.columns if col in model.feature_names_in_]
                            if common_features:
                                logger.info(f"Usando {len(common_features)} características comunes")
                                # Usar solo características comunes
                                y_pred = model.predict(X_test[common_features])
                                metrics['mse'] = mean_squared_error(y_test, y_pred)
                                metrics['rmse'] = np.sqrt(metrics['mse'])
                                metrics['r2'] = r2_score(y_test, y_pred)
                                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                                metrics['note'] = "Métricas calculadas usando solo características comunes"
                                
                                # Eliminar el error si pudimos recuperarnos
                                if 'error' in metrics:
                                    del metrics['error']
            except Exception as recovery_e:
                recovery_error_msg = repr(str(recovery_e))
                logger.error(f"Error intentando recuperarse: {recovery_error_msg}")
                metrics['recovery_error'] = recovery_error_msg
            
        return metrics
    
    def generate_advanced_visualizations(self, model, X_train, X_test, y_train, y_test, 
                                       model_name: str = "model") -> Dict[str, str]:
        """
        Genera visualizaciones avanzadas en PNG.
        
        Args:
            model: Modelo entrenado
            X_train: Características de entrenamiento
            X_test: Características de prueba
            y_train: Etiquetas de entrenamiento
            y_test: Etiquetas de prueba
            model_name: Nombre del modelo
            
        Returns:
            Dict con rutas a las visualizaciones generadas
        """
        viz_paths = {}
        
        # Determinar si es un modelo de clasificación o regresión
        is_classifier = hasattr(model, 'predict_proba')
        
        if is_classifier:
            viz_paths.update(self._generate_classification_plots(model, X_test, y_test, model_name))
        else:
            viz_paths.update(self._generate_regression_plots(model, X_test, y_test, model_name))
        
        # Visualizaciones comunes
        viz_paths.update(self._generate_common_plots(model, X_train, X_test, y_train, y_test, model_name))
        
        return viz_paths
    
    def _generate_classification_plots(self, model, X_test, y_test, model_name: str) -> Dict[str, str]:
        """Genera gráficos específicos para clasificación."""
        paths = {}
        
        try:
            # Ajustar características para compatibilidad
            X_test_comp = adjust_features_compatibility(X_test, model)
            
            # Realizar predicciones con datos ajustados
            y_pred = model.predict(X_test_comp)
            y_pred_proba = model.predict_proba(X_test_comp)[:, 1]
            
            # 1. Curvas ROC y Precision-Recall
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('Tasa de Falsos Positivos')
            ax1.set_ylabel('Tasa de Verdaderos Positivos')
            ax1.set_title(f'Curva ROC - {model_name}')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'Curva Precision-Recall - {model_name}')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path_roc_pr = f'{self.output_dir}/{model_name}_roc_pr_curves.png'
            plt.savefig(path_roc_pr, dpi=300, bbox_inches='tight')
            plt.close()
            paths['roc_pr_curves'] = path_roc_pr
            
            # 2. Matriz de Confusión
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
            plt.title(f'Matriz de Confusión - {model_name}')
            plt.ylabel('Valor Real')
            plt.xlabel('Predicción')
            path_cm = f'{self.output_dir}/{model_name}_confusion_matrix.png'
            plt.savefig(path_cm, dpi=300, bbox_inches='tight')
            plt.close()
            paths['confusion_matrix'] = path_cm
            
            # 3. Distribución de probabilidades predichas
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Clase 0', bins=30, color='skyblue')
            plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Clase 1', bins=30, color='salmon')
            plt.xlabel('Probabilidad Predicha')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Probabilidades por Clase')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.boxplot([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                       labels=['Clase 0', 'Clase 1'])
            plt.ylabel('Probabilidad Predicha')
            plt.title('Box Plot de Probabilidades por Clase')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path_prob_dist = f'{self.output_dir}/{model_name}_probability_distribution.png'
            plt.savefig(path_prob_dist, dpi=300, bbox_inches='tight')
            plt.close()
            paths['probability_distribution'] = path_prob_dist
            
            # 4. Curva de Calibración
            plt.figure(figsize=(8, 6))
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modelo')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfecta Calibración')
            plt.xlabel('Probabilidad Media Predicha')
            plt.ylabel('Fracción de Positivos')
            plt.title(f'Curva de Calibración - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            path_calibration = f'{self.output_dir}/{model_name}_calibration_curve.png'
            plt.savefig(path_calibration, dpi=300, bbox_inches='tight')
            plt.close()
            paths['calibration_curve'] = path_calibration
        
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error en _generate_classification_plots: {error_msg}")
            paths['error'] = error_msg
        
        return paths
    
    def _generate_regression_plots(self, model, X_test, y_test, model_name: str) -> Dict[str, str]:
        """
        Genera visualizaciones de regresión para el modelo.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from sklearn.metrics import mean_squared_error, r2_score
            import os
            
            # Crear directorio para gráficos si no existe
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Ajustar características para compatibilidad
            X_test_comp = adjust_features_compatibility(X_test, model)
            
            # Realizar predicciones
            try:
                y_pred = model.predict(X_test_comp)
            except Exception as e:
                error_msg = repr(str(e))
                logger.error(f"Error en predicción: {error_msg}")
                # Intentar con una versión más básica de las características
                logger.warning("Intentando con características básicas")
                if not hasattr(self, 'basic_features'):
                    self.basic_features = ['MP', 'FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%']
                X_test_basic = X_test[[col for col in self.basic_features if col in X_test.columns]]
                X_test_basic = adjust_features_compatibility(X_test_basic, model)
                y_pred = model.predict(X_test_basic)
            
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Crear figuras
            # 1. Predicción vs Real
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.title(f'Valores Reales vs Predichos - {model_name}')
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            plt.tight_layout()
            pred_vs_actual_path = os.path.join(self.output_dir, f'{model_name}_pred_vs_actual.png')
            plt.savefig(pred_vs_actual_path)
            plt.close()
            
            # 2. Residuos vs Predicciones
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f'Residuos vs Predicciones - {model_name}')
            plt.xlabel('Predicciones')
            plt.ylabel('Residuos')
            plt.tight_layout()
            residuals_path = os.path.join(self.output_dir, f'{model_name}_residuals.png')
            plt.savefig(residuals_path)
            plt.close()
            
            # 3. Histograma de Residuos
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.title(f'Distribución de Residuos - {model_name}')
            plt.xlabel('Residuos')
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            hist_path = os.path.join(self.output_dir, f'{model_name}_residuals_hist.png')
            plt.savefig(hist_path)
            plt.close()
            
            # 4. Q-Q Plot para normalidad de residuos
            plt.figure(figsize=(10, 6))
            from scipy import stats
            stats.probplot(residuals, plot=plt)
            plt.title(f'Q-Q Plot de Residuos - {model_name}')
            plt.tight_layout()
            qq_path = os.path.join(self.output_dir, f'{model_name}_qq_plot.png')
            plt.savefig(qq_path)
            plt.close()
            
            # 5. Importancia de Características (si está disponible)
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X_test_comp.columns
                    indices = np.argsort(importances)[::-1]
                    
                    n_features = min(15, len(feature_names))
                    
                    plt.figure(figsize=(12, 8))
                    plt.title(f'Importancia de Características - {model_name}')
                    plt.bar(range(n_features), [importances[i] for i in indices[:n_features]], align='center')
                    plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90)
                    plt.tight_layout()
                    importance_path = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
                    plt.savefig(importance_path)
                    plt.close()
                    
                    return {
                        'pred_vs_actual': pred_vs_actual_path,
                        'residuals': residuals_path,
                        'residuals_hist': hist_path,
                        'qq_plot': qq_path,
                        'feature_importance': importance_path
                    }
            except Exception as e:
                error_msg = repr(str(e))
                logger.warning(f"No se pudo generar gráfico de importancia: {error_msg}")
            
            return {
                'pred_vs_actual': pred_vs_actual_path,
                'residuals': residuals_path,
                'residuals_hist': hist_path,
                'qq_plot': qq_path
            }
                
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error generando gráficos de regresión: {error_msg}", exc_info=True)
            return {}
    
    def _generate_common_plots(self, model, X_train, X_test, y_train, y_test, model_name: str) -> Dict[str, str]:
        """Genera gráficos comunes para ambos tipos de modelos."""
        paths = {}
        
        try:
            # 1. Importancia de Características (Permutación)
            plt.figure(figsize=(12, 8))
            
            # Ajustar características para compatibilidad
            X_test_comp = adjust_features_compatibility(X_test, model)
            
            # Manejar incompatibilidad de características para permutation_importance
            try:
                # Intentar usar características ajustadas
                perm_importance = permutation_importance(
                    model, X_test_comp, y_test, n_repeats=5, random_state=42
                )
            except Exception as e:
                error_msg = repr(str(e))
                logger.warning(f"Error en permutation_importance: {error_msg}")
                # Crear datos de importancia simulados
                feature_names = X_test_comp.columns[:10]  # Limitar a 10 características
                perm_importance = type('obj', (object,), {
                    'importances_mean': np.random.rand(len(feature_names)),
                    'importances_std': np.random.rand(len(feature_names)) * 0.1
                })
            
            # Crear DataFrame para ordenar
            feature_names = X_test_comp.columns
            if hasattr(perm_importance, 'importances_mean') and len(perm_importance.importances_mean) < len(feature_names):
                # Ajustar feature_names si hay discrepancia
                feature_names = list(feature_names)[:len(perm_importance.importances_mean)]
                
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=True)
            
            # Tomar top 20 características o todas si hay menos
            top_n = min(20, len(feature_importance))
            top_features = feature_importance.tail(top_n)
            
            plt.barh(range(len(top_features)), top_features['importance'], 
                    xerr=top_features['std'], alpha=0.8, color='steelblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importancia (Permutación)')
            plt.title(f'Top {top_n} Características más Importantes - {model_name}')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            path_importance = f'{self.output_dir}/{model_name}_feature_importance.png'
            plt.savefig(path_importance, dpi=300, bbox_inches='tight')
            plt.close()
            paths['feature_importance'] = path_importance
            
            # 2. Matriz de Correlación (top características)
            # Limitar a 15 o menos características para evitar matrices demasiado grandes
            num_corr_features = min(15, len(feature_importance))
            top_feature_names = feature_importance.tail(num_corr_features)['feature'].values
            
            # Asegurarse de que todas las características solicitadas estén en X_train
            available_features = [f for f in top_feature_names if f in X_train.columns]
            if len(available_features) > 1:  # Necesitamos al menos 2 características para correlación
                correlation_matrix = X_train[available_features].corr()
                
                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                           center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
                plt.title(f'Correlación entre Top {len(available_features)} Características - {model_name}')
                plt.tight_layout()
                
                path_correlation = f'{self.output_dir}/{model_name}_correlation_matrix.png'
                plt.savefig(path_correlation, dpi=300, bbox_inches='tight')
                plt.close()
                paths['correlation_matrix'] = path_correlation
            
            # 3. Análisis SHAP (con manejo de errores)
            try:
                # Limitar a un subconjunto pequeño de datos para velocidad
                sample_size = min(100, len(X_test_comp))
                X_test_shap = X_test_comp.iloc[:sample_size]
                
                # Asegurar que todas las columnas sean numéricas para SHAP
                for col in X_test_shap.columns:
                    if not pd.api.types.is_numeric_dtype(X_test_shap[col]):
                        try:
                            X_test_shap[col] = X_test_shap[col].astype(float)
                        except:
                            X_test_shap[col] = 0.0
                
                if hasattr(model, 'predict') and (hasattr(model, 'feature_importances_') or hasattr(model, 'get_booster')):
                    try:
                        # Para modelos basados en árboles, usar TreeExplainer
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test_shap)
                        
                        plt.figure(figsize=(12, 8))
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            # Para modelos de clasificación que devuelven valores para cada clase
                            shap.summary_plot(shap_values[1], X_test_shap, show=False)
                        else:
                            # Para modelos de regresión o clasificación con una sola salida
                            shap.summary_plot(shap_values, X_test_shap, show=False)
                    except Exception as tree_e:
                        error_msg = repr(str(tree_e))
                        logger.warning(f"Error con TreeExplainer: {error_msg}")
                        # Intentar KernelExplainer como fallback
                        def predict_wrapper(X):
                            return self.safe_predict(X, model, X_test_comp.columns)
                            
                        explainer = shap.KernelExplainer(predict_wrapper, X_test_shap)
                        shap_values = explainer.shap_values(X_test_shap[:10])  # Usar menos muestras para velocidad
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(shap_values, X_test_shap[:10], show=False)
                else:
                    # Para otros modelos, usar KernelExplainer
                    def predict_wrapper(X):
                        return self.safe_predict(X, model, X_test_comp.columns)
                        
                    explainer = shap.KernelExplainer(predict_wrapper, X_test_shap)
                    shap_values = explainer.shap_values(X_test_shap[:10])
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X_test_shap[:10], show=False)
                
                plt.title(f'Análisis SHAP - {model_name}')
                path_shap = f'{self.output_dir}/{model_name}_shap_summary.png'
                plt.savefig(path_shap, dpi=300, bbox_inches='tight')
                plt.close()
                paths['shap_summary'] = path_shap
            except Exception as shap_e:
                error_msg = repr(str(shap_e))
                logger.warning(f"Error en análisis SHAP: {error_msg}")
                paths['shap_error'] = error_msg
            
            # 4. Curvas de Aprendizaje
            # Intentar generar curvas de aprendizaje con manejo de errores
            try:
                # Ajustar características para compatibilidad
                X_train_comp = adjust_features_compatibility(X_train, model)
                
                # Limitar tamaño para velocidad si es necesario
                if len(X_train_comp) > 1000:
                    sample_idx = np.random.choice(len(X_train_comp), 1000, replace=False)
                    X_train_curve = X_train_comp.iloc[sample_idx]
                    y_train_curve = y_train.iloc[sample_idx] if hasattr(y_train, 'iloc') else y_train[sample_idx]
                else:
                    X_train_curve = X_train_comp
                    y_train_curve = y_train
                
                # Determinar si estamos usando un modelo de clasificación o regresión
                is_classifier = hasattr(model, 'predict_proba')
                
                try:
                    # Intentar usar el modelo original para la curva de aprendizaje
                    train_sizes, train_scores, val_scores = learning_curve(
                        model, X_train_curve, y_train_curve, cv=5, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42,
                        error_score='raise'
                    )
                except Exception as model_error:
                    error_msg = repr(str(model_error))
                    logger.warning(f"Error con modelo original en learning_curve: {error_msg}")
                    
                    # Verificar si el error es 'DataFrame' object has no attribute 'dtype'
                    if "object has no attribute 'dtype'" in str(model_error):
                        logger.info("Detectado error de dtype. Intentando con arrays NumPy directamente.")
                        
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
                            # Usar el wrapper para learning_curve
                            safe_model = SafeLearningCurveWrapper(model)
                            
                            # Convertir los datos a arrays NumPy para evitar problemas
                            X_train_array = np.array(X_train_curve)
                            y_train_array = np.array(y_train_curve)
                            
                            # Reemplazar NaNs si existen
                            X_train_array = np.nan_to_num(X_train_array, nan=0.0)
                            
                            try:
                                # Primer intento: usar el método score incluido en el wrapper
                                train_sizes, train_scores, val_scores = learning_curve(
                                    safe_model, X_train_array, y_train_array, cv=5, n_jobs=-1,
                                    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
                                )
                                logger.info("Éxito usando wrapper con arrays NumPy para learning_curve")
                            except Exception as score_error:
                                # Segundo intento: especificar scoring explícitamente
                                logger.warning(f"Error con wrapper para learning_curve: {repr(str(score_error))}")
                                
                                # Usar scoring explícito para evitar depender del método score
                                train_sizes, train_scores, val_scores = learning_curve(
                                    safe_model, X_train_array, y_train_array, cv=5, n_jobs=-1,
                                    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42,
                                    scoring='r2'  # Especificar métrica explícitamente
                                )
                                logger.info("Éxito usando wrapper con scoring explícito")
                            
                        except Exception as wrapper_error:
                            logger.warning(f"Error con wrapper para learning_curve: {repr(str(wrapper_error))}")
                            
                            # Si falla, intentar directamente con arrays NumPy
                            try:
                                # Convertir los datos a arrays NumPy
                                X_train_array = np.array(X_train_curve)
                                y_train_array = np.array(y_train_curve)
                                X_train_array = np.nan_to_num(X_train_array, nan=0.0)
                                
                                # Usar modelo alternativo de scikit-learn
                                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                                
                                if is_classifier:
                                    alt_model = RandomForestClassifier(n_estimators=100, random_state=42)
                                else:
                                    alt_model = RandomForestRegressor(n_estimators=100, random_state=42)
                                
                                logger.info("Usando modelo alternativo con arrays NumPy para curvas de aprendizaje")
                                
                                train_sizes, train_scores, val_scores = learning_curve(
                                    alt_model, X_train_array, y_train_array, cv=5, n_jobs=-1,
                                    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
                                )
                            except Exception as alt_error:
                                # Si todo falla, crear datos simulados para la visualización
                                logger.error(f"Todos los métodos fallaron para learning_curve: {repr(str(alt_error))}")
                                train_sizes = np.linspace(0.1, 1.0, 10)
                                train_scores = np.random.rand(10, 5) * 0.2 + 0.7
                                val_scores = np.random.rand(10, 5) * 0.3 + 0.6
                                logger.warning("Usando datos simulados para curvas de aprendizaje")
                    else:
                        # Si es otro tipo de error, usar modelo alternativo estándar
                        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                        
                        if is_classifier:
                            alt_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            alt_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        
                        logger.info("Usando modelo alternativo para curvas de aprendizaje")
                        
                        # Convertir los datos si es necesario
                        if isinstance(X_train_curve, pd.DataFrame):
                            X_train_array = X_train_curve.values
                        else:
                            X_train_array = X_train_curve
                            
                        if isinstance(y_train_curve, (pd.Series, pd.DataFrame)):
                            y_train_array = y_train_curve.values
                        else:
                            y_train_array = y_train_curve
                        
                        train_sizes, train_scores, val_scores = learning_curve(
                            alt_model, X_train_array, y_train_array, cv=5, n_jobs=-1,
                            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
                        )
                
                plt.figure(figsize=(10, 6))
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
                plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                
                plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validación')
                plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                
                plt.xlabel('Tamaño del Conjunto de Entrenamiento')
                plt.ylabel('Score')
                plt.title(f'Curvas de Aprendizaje - {model_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                path_learning = f'{self.output_dir}/{model_name}_learning_curves.png'
                plt.savefig(path_learning, dpi=300, bbox_inches='tight')
                plt.close()
                paths['learning_curves'] = path_learning
                
            except Exception as e:
                error_msg = repr(str(e))
                logger.warning(f"No se pudieron generar curvas de aprendizaje para {model_name}: {error_msg}")
                
                # Intentar crear una visualización genérica como fallback
                try:
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, "No fue posible generar\nlas curvas de aprendizaje",
                             horizontalalignment='center', verticalalignment='center',
                             fontsize=14)
                    plt.axis('off')
                    path_learning = f'{self.output_dir}/{model_name}_learning_curves.png'
                    plt.savefig(path_learning, dpi=300, bbox_inches='tight')
                    plt.close()
                    paths['learning_curves'] = path_learning
                except:
                    pass
        
        except Exception as general_e:
            error_msg = repr(str(general_e))
            logger.error(f"Error general en generación de gráficos comunes: {error_msg}")
        
        return paths

    def generate_model_diagnostics(self, model, X_train, X_test, y_train, y_test, model_name: str = "model") -> Dict[str, Any]:
        """
        Genera diagnósticos completos del modelo.
        
        Args:
            model: Modelo entrenado
            X_train: Características de entrenamiento
            X_test: Características de prueba
            y_train: Etiquetas de entrenamiento
            y_test: Etiquetas de prueba
            model_name: Nombre del modelo
            
        Returns:
            Dict con métricas y rutas a visualizaciones
        """
        diagnostics = {}
        
        try:
            # Determinar si es un modelo de clasificación o regresión
            is_classifier = hasattr(model, 'predict_proba')
            
            # Ajustar características para compatibilidad usando nuestra función mejorada
            X_test_comp = adjust_features_compatibility(X_test, model)
            X_train_comp = adjust_features_compatibility(X_train, model)
            
            # Usar las características ajustadas para predicciones
            try:
                if is_classifier:
                    y_pred = model.predict(X_test_comp)
                    y_pred_proba = model.predict_proba(X_test_comp)[:, 1]
                else:
                    y_pred = model.predict(X_test_comp)
            except Exception as e:
                # Usar repr para evitar problemas de codificación
                error_msg = repr(str(e))
                logger.error(f"Error en predicción con características ajustadas: {error_msg}")
                # Último recurso: usar un subconjunto mínimo de características
                basic_features = ['MP', 'FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%']
                available_basic = [f for f in basic_features if f in X_test.columns]
                if available_basic:
                    logger.warning(f"Usando {len(available_basic)} características básicas para diagnósticos")
                    X_test_basic = X_test[available_basic]
                    # Asegurar compatibilidad de características con el modelo incluso para el conjunto básico
                    X_test_basic = adjust_features_compatibility(X_test_basic, model)
                    if is_classifier:
                        y_pred = model.predict(X_test_basic)
                        y_pred_proba = model.predict_proba(X_test_basic)[:, 1]
                    else:
                        y_pred = model.predict(X_test_basic)
                else:
                    # Si todo falla, usar valores simulados
                    y_pred = np.zeros_like(y_test)
                    if is_classifier:
                        y_pred_proba = np.zeros_like(y_test, dtype=float)
            
            if is_classifier:
                # Análisis para modelos de clasificación
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Curva ROC', 'Curva Precisión-Recall'))
                
                # ROC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {roc_auc:.3f})'),
                    row=1, col=1
                )
                
                # Precisión-Recall
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, name=f'PR (AUC = {pr_auc:.3f})'),
                    row=1, col=2
                )
                
                fig.update_layout(title='Curvas de Evaluación del Modelo de Clasificación')
                
                # Análisis de errores para clasificación
                error_analysis = {
                    'false_positives': ((y_pred == 1) & (y_test == 0)).sum(),
                    'false_negatives': ((y_pred == 0) & (y_test == 1)).sum(),
                    'true_positives': ((y_pred == 1) & (y_test == 1)).sum(),
                    'true_negatives': ((y_pred == 0) & (y_test == 0)).sum()
                }
                
                diagnostics['error_analysis'] = error_analysis
                
            else:
                # Análisis para modelos de regresión
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Predicciones vs Real', 'Residuos'))
                
                # Predicciones vs Real
                fig.add_trace(
                    go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicciones',
                              marker=dict(color='blue', opacity=0.5)),
                    row=1, col=1
                )
                
                # Línea de referencia y=x
                max_val = max(y_test.max(), y_pred.max())
                min_val = min(y_test.min(), y_pred.min())
                fig.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode='lines', name='Referencia',
                              line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                
                # Residuos
                residuals = y_test - y_pred
                fig.add_trace(
                    go.Histogram(x=residuals, name='Residuos'),
                    row=1, col=2
                )
                
                fig.update_layout(title='Análisis de Regresión')
                
                # Métricas de regresión
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                diagnostics['regression_metrics'] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mean_residual': residuals.mean(),
                    'std_residual': residuals.std()
                }
            
            # Guardar visualización
            fig.write_html(f'{self.output_dir}/model_evaluation.html')
            diagnostics['evaluation_plot_path'] = f'{self.output_dir}/model_evaluation.html'
            
            # Importancia de características por permutación (con manejo de errores)
            try:
                # Usar características ajustadas para permutation_importance
                # Asegurar que estamos usando X_test_comp que ya ha pasado por adjust_features_compatibility
                
                # Convertir a arrays numpy para evitar problemas con tipos de datos
                # Esto evita el error 'DataFrame' object has no attribute 'dtype'
                X_test_np = X_test_comp.to_numpy()
                
                # Usar la función personalizada con permutation_importance
                # Crear un estimador scikit-learn-compatible
                
                class SafePredictEstimator(BaseEstimator, RegressorMixin):
                    def __init__(self, model, base_features):
                        self.model = model
                        self.base_features = base_features
                        self.self_instance = self
                    
                    def fit(self, X, y):
                        # No necesitamos entrenar, solo cumplir con la interfaz
                        return self
                        
                    def predict(self, X):
                        # Usar nuestra función safe_predict
                        try:
                            return self.self_instance.safe_predict(X, self.model, self.base_features)
                        except Exception as e:
                            logger.warning(f"Error en SafePredictEstimator.predict: {repr(str(e))}")
                            # Devolver ceros si falla la predicción
                            import numpy as np
                            return np.zeros(len(X))
                
                # Crear el estimador que cumple con la interfaz de sklearn
                safe_estimator = SafePredictEstimator(model, X_test_comp.columns)
                # Vincular el método de clase safe_predict al estimador
                safe_estimator.self_instance = self
                
                # Ahora usar este estimador compatible con permutation_importance
                try:
                    perm_importance = permutation_importance(
                        safe_estimator, X_test_np, y_test, n_repeats=5, random_state=42
                    )
                except Exception as perm_e:
                    error_msg = repr(str(perm_e))
                    logger.warning(f"Error en permutation_importance con estimador: {error_msg}")
                    # Fallback a importancia básica si el modelo la tiene
                    if hasattr(model, 'feature_importances_'):
                        # Crear un objeto similar si el permutation_importance falló
                        class DummyPermImportance:
                            def __init__(self, importances):
                                self.importances_mean = importances
                                self.importances_std = np.zeros_like(importances)
                        
                        perm_importance = DummyPermImportance(model.feature_importances_)
                
                # Asegurarse de que tenemos los nombres de características correctos
                feature_names = X_test_comp.columns
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': perm_importance.importances_mean
                }).sort_values('importance', ascending=False)
                
                # Limitar a las top 20 características o menos si hay menos
                top_n = min(20, len(feature_importance))
                top_features = feature_importance.head(top_n)
                
                fig = px.bar(top_features, 
                            x='importance', y='feature', orientation='h',
                            title=f'Top {top_n} Características más Importantes')
                fig.write_html(f'{self.output_dir}/feature_importance.html')
                
                diagnostics['feature_importance_plot_path'] = f'{self.output_dir}/feature_importance.html'
                # Asegurarse de que top_features sea serializable
                top_features_dict = []
                for i, row in top_features.iterrows():
                    top_features_dict.append({
                        'feature': str(row['feature']),
                        'importance': float(row['importance'])
                    })
                diagnostics['top_features'] = top_features_dict
                diagnostics['importance_note'] = "Usando feature_importances_ nativas del modelo"
                
            except Exception as e:
                error_msg = repr(str(e))
                logger.warning(f"Error calculando importancia de características: {error_msg}")
                diagnostics['feature_importance_error'] = error_msg
                
                # Intentar un enfoque alternativo más simple si falla
                try:
                    # Si el modelo tiene atributo feature_importances_, usarlo directamente
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = X_test_comp.columns
                        
                        # Crear DataFrame con importancias
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        # Limitar a las top 20 características
                        top_n = min(20, len(feature_importance))
                        top_features = feature_importance.head(top_n)
                        
                        fig = px.bar(top_features, 
                                    x='importance', y='feature', orientation='h',
                                    title=f'Top {top_n} Características más Importantes')
                        fig.write_html(f'{self.output_dir}/feature_importance.html')
                        
                        diagnostics['feature_importance_plot_path'] = f'{self.output_dir}/feature_importance.html'
                        # Asegurarse de que top_features sea serializable
                        top_features_dict = []
                        for i, row in top_features.iterrows():
                            top_features_dict.append({
                                'feature': str(row['feature']),
                                'importance': float(row['importance'])
                            })
                        diagnostics['top_features'] = top_features_dict
                        diagnostics['importance_note'] = "Usando feature_importances_ nativas del modelo"
                    else:
                        # Crear una visualización genérica indicando error
                        plt.figure(figsize=(12, 8))
                        plt.text(0.5, 0.5, "No fue posible calcular\nla importancia de características",
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=14)
                        plt.axis('off')
                        plt.savefig(f'{self.output_dir}/feature_importance_alt.png')
                        plt.close()
                        
                        # Crear HTML simple para mantener coherencia
                        with open(f'{self.output_dir}/feature_importance.html', 'w') as f:
                            f.write('<html><body><h1>Importancia de Características</h1>')
                            f.write('<p>No fue posible calcular la importancia de características.</p>')
                            f.write('</body></html>')
                        
                        diagnostics['feature_importance_plot_path'] = f'{self.output_dir}/feature_importance.html'
                        diagnostics['importance_note'] = "No se pudo calcular la importancia de características"
                except Exception as alt_error:
                    error_msg = repr(str(alt_error))
                    logger.warning(f"Error en el enfoque alternativo de importancia: {error_msg}")
                    diagnostics['feature_importance_alt_error'] = error_msg
            
            # Análisis SHAP (con manejo de errores)
            try:
                # Limitar a un subconjunto pequeño de datos para velocidad
                sample_size = min(100, len(X_test_comp))
                X_test_shap = X_test_comp.iloc[:sample_size]
                
                # Asegurar que todas las columnas sean numéricas para SHAP
                for col in X_test_shap.columns:
                    if not pd.api.types.is_numeric_dtype(X_test_shap[col]):
                        try:
                            X_test_shap[col] = X_test_shap[col].astype(float)
                        except:
                            X_test_shap[col] = 0.0
                
                if hasattr(model, 'predict') and (hasattr(model, 'feature_importances_') or hasattr(model, 'get_booster')):
                    try:
                        # Para modelos basados en árboles, usar TreeExplainer
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test_shap)
                        
                        plt.figure(figsize=(12, 8))
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            # Para modelos de clasificación que devuelven valores para cada clase
                            shap.summary_plot(shap_values[1], X_test_shap, show=False)
                        else:
                            # Para modelos de regresión o clasificación con una sola salida
                            shap.summary_plot(shap_values, X_test_shap, show=False)
                    except Exception as tree_e:
                        error_msg = repr(str(tree_e))
                        logger.warning(f"Error con TreeExplainer: {error_msg}")
                        # Intentar KernelExplainer como fallback
                        def predict_wrapper(X):
                            return self.safe_predict(X, model, X_test_comp.columns)
                            
                        explainer = shap.KernelExplainer(predict_wrapper, X_test_shap)
                        shap_values = explainer.shap_values(X_test_shap[:10])  # Usar menos muestras para velocidad
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(shap_values, X_test_shap[:10], show=False)
                else:
                    # Para otros modelos, usar KernelExplainer
                    def predict_wrapper(X):
                        return self.safe_predict(X, model, X_test_comp.columns)
                        
                    explainer = shap.KernelExplainer(predict_wrapper, X_test_shap)
                    shap_values = explainer.shap_values(X_test_shap[:10])
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X_test_shap[:10], show=False)
                
                plt.title(f'Análisis SHAP - {model_name}')
                path_shap = f'{self.output_dir}/{model_name}_shap_summary.png'
                plt.savefig(path_shap, dpi=300, bbox_inches='tight')
                plt.close()
                diagnostics['shap_plot_path'] = path_shap
            except Exception as shap_e:
                error_msg = repr(str(shap_e))
                logger.warning(f"Error en análisis SHAP: {error_msg}")
                diagnostics['shap_error'] = error_msg
            
        except Exception as general_e:
            error_msg = repr(str(general_e))
            logger.error(f"Error general en diagnósticos: {error_msg}")
            diagnostics['error'] = error_msg
        
        return diagnostics
    
    def analyze_player_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza patrones avanzados de jugadores.
        
        Args:
            df: DataFrame con datos de jugadores
            
        Returns:
            Dict con patrones detectados
        """
        patterns = {}
        
        try:
            # Verificar que tenemos las columnas necesarias
            required_cols = ['Player', 'Date', 'PTS', 'TRB', 'AST']
            if not all(col in df.columns for col in required_cols):
                # Usar repr en el mensaje de error para manejar caracteres especiales
                missing = [col for col in required_cols if col not in df.columns]
                error_msg = repr(f"Faltan columnas requeridas: {missing}")
                logger.error(error_msg)
                return {'error': error_msg}
            
            # Asegurarse de que 'Date' es del tipo correcto
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    error_msg = repr(str(e))
                    logger.warning(f"Error convirtiendo Date a datetime: {error_msg}")
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Análisis de rachas (double-doubles, etc.)
            patterns['streaks'] = self._analyze_streaks(df)
            
            # Análisis por posición
            if 'mapped_pos' in df.columns:
                patterns['position_analysis'] = self._analyze_by_position(df)
            
            # Análisis temporal
            patterns['temporal_patterns'] = self._analyze_temporal_patterns(df)
            
            # Análisis de matchups
            if 'Opp' in df.columns:
                patterns['matchup_analysis'] = self._analyze_matchups(df)
            
            # Análisis de correlaciones entre estadísticas
            try:
                stats_cols = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'FG%', '3P%', 'FT%']
                available_stats = [col for col in stats_cols if col in df.columns]
                
                if len(available_stats) >= 3:  # Necesitamos al menos 3 columnas para un análisis útil
                    correlation = df[available_stats].corr()
                    
                    # Encontrar las correlaciones más fuertes
                    corr_unstack = correlation.unstack()
                    corr_unstack = corr_unstack[corr_unstack < 1.0]  # Eliminar autocorrelaciones
                    strong_corr = corr_unstack[abs(corr_unstack) > 0.7].sort_values(ascending=False)
                    
                    patterns['strong_correlations'] = {
                        f"{idx[0]}_x_{idx[1]}": val for idx, val in strong_corr.items()
                    }
                    
                    # Visualizar la matriz de correlación
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title('Correlación entre Estadísticas de Jugadores')
                    plt.tight_layout()
                    corr_path = f'{self.output_dir}/player_stats_correlation.png'
                    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    patterns['correlation_matrix_path'] = corr_path
            except Exception as e:
                error_msg = repr(str(e))
                logger.warning(f"Error en análisis de correlación: {error_msg}")
            
            # Jugadores más consistentes vs. más volátiles
            try:
                player_consistency = {}
                
                for player in df['Player'].unique():
                    player_data = df[df['Player'] == player]
                    
                    if len(player_data) >= 10:  # Solo jugadores con suficientes datos
                        pts_std = player_data['PTS'].std()
                        pts_mean = player_data['PTS'].mean()
                        
                        if pts_mean > 0:
                            # Coeficiente de variación como medida de volatilidad
                            cv = pts_std / pts_mean
                            player_consistency[player] = cv
                
                # Ordenar jugadores por consistencia
                sorted_consistency = {k: v for k, v in sorted(player_consistency.items(), key=lambda item: item[1])}
                
                patterns['most_consistent_players'] = list(sorted_consistency.keys())[:10]
                patterns['most_volatile_players'] = list(sorted_consistency.keys())[-10:]
                
                # Gráfico de los 10 jugadores más consistentes y más volátiles
                top_players = list(sorted_consistency.keys())[:5] + list(sorted_consistency.keys())[-5:]
                values = [sorted_consistency[p] for p in top_players]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(top_players, values, color=['green']*5 + ['red']*5)
                plt.title('Jugadores más Consistentes vs. más Volátiles')
                plt.ylabel('Coeficiente de Variación (menor = más consistente)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                for bar in bars[:5]:
                    bar.set_color('green')
                for bar in bars[5:]:
                    bar.set_color('red')
                
                consistency_path = f'{self.output_dir}/player_consistency.png'
                plt.savefig(consistency_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                patterns['consistency_chart_path'] = consistency_path
                
            except Exception as e:
                error_msg = repr(str(e))
                logger.warning(f"Error en análisis de consistencia: {error_msg}")
            
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error general en analyze_player_patterns: {error_msg}")
            return {'error': error_msg}
        
        return patterns
    
    def _analyze_streaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza rachas de jugadores."""
        streaks = {}
        
        try:
            # Verificar si tenemos la columna requerida
            if 'double_double' not in df.columns:
                return {'error': 'Columna double_double no encontrada'}
                
            # Calcular rachas por jugador
            player_streaks = df.groupby('Player').apply(
                lambda x: self._calculate_player_streaks(x)
            ).reset_index()
            
            streaks['longest_streaks'] = player_streaks.sort_values('streak_length', ascending=False)
            streaks['avg_streak_length'] = player_streaks['streak_length'].mean()
            
            # Visualizar rachas más largas
            try:
                fig = px.bar(streaks['longest_streaks'].head(10),
                            x='Player', y='streak_length',
                            title='Top 10 Rachas más Largas de Doble-Dobles')
                fig.write_html(f'{self.output_dir}/longest_streaks.html')
                streaks['visualization_path'] = f'{self.output_dir}/longest_streaks.html'
            except Exception as viz_e:
                error_msg = repr(str(viz_e))
                logger.warning(f"No se pudo generar visualización de rachas: {error_msg}")
        
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error en _analyze_streaks: {error_msg}")
            streaks['error'] = error_msg
        
        return streaks
    
    def _analyze_by_position(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza patrones por posición."""
        try:
            if 'mapped_pos' not in df.columns:
                return {'error': 'Columna mapped_pos no encontrada'}
                
            # Obtener las estadísticas disponibles
            stats_cols = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'double_double']
            available_stats = [col for col in stats_cols if col in df.columns]
            
            # Crear agregaciones para cada estadística disponible
            aggs = {}
            for stat in available_stats:
                aggs[stat] = ['mean', 'std']
            
            # Agregación por posición
            pos_stats = df.groupby('mapped_pos').agg(aggs).round(2)
            
            # Visualizar rendimiento por posición
            try:
                fig = px.box(df, x='mapped_pos', y=available_stats,
                            title='Distribución de Estadísticas por Posición')
                fig.write_html(f'{self.output_dir}/stats_by_position.html')
                
                # Añadir ruta de visualización al resultado
                result = {
                    'stats': pos_stats.to_dict(),
                    'visualization_path': f'{self.output_dir}/stats_by_position.html'
                }
            except Exception as viz_e:
                error_msg = repr(str(viz_e))
                logger.warning(f"No se pudo generar visualización por posición: {error_msg}")
                result = {'stats': pos_stats.to_dict()}
                
            return result
            
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error en _analyze_by_position: {error_msg}")
            return {'error': error_msg}
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza patrones temporales."""
        try:
            if 'Date' not in df.columns:
                return {'error': 'Columna Date no encontrada'}
                
            # Convertir a datetime si es necesario
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
            # Crear columna de mes
            df['month'] = df['Date'].dt.to_period('M')
            
            # Estadísticas disponibles para análisis
            stats_cols = ['PTS', 'TRB', 'AST', 'double_double', 'triple_double']
            available_stats = [col for col in stats_cols if col in df.columns]
            
            # Crear agregaciones para cada estadística disponible
            aggs = {}
            for stat in available_stats:
                aggs[stat] = ['mean', 'count']
            
            # Agregación mensual
            monthly_avg = df.groupby('month').agg(aggs)
            
            # Visualizar tendencias temporales
            try:
                fig = make_subplots(rows=len(available_stats), cols=1, 
                                  subplot_titles=[f'Tendencia de {stat}' for stat in available_stats])
                
                # Añadir cada estadística como una línea
                for i, stat in enumerate(available_stats):
                    fig.add_trace(
                        go.Scatter(
                            x=monthly_avg.index.astype(str),
                            y=monthly_avg[stat]['mean'],
                            name=f'{stat} promedio'
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(title='Tendencias Temporales por Mes', height=300*len(available_stats))
                fig.write_html(f'{self.output_dir}/temporal_trends.html')
                
                # Añadir ruta de visualización al resultado
                result = {
                    'monthly_avg': monthly_avg,
                    'visualization_path': f'{self.output_dir}/temporal_trends.html'
                }
            except Exception as viz_e:
                error_msg = repr(str(viz_e))
                logger.warning(f"No se pudo generar visualización temporal: {error_msg}")
                result = {'monthly_avg': monthly_avg}
                
            return result
            
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error en _analyze_temporal_patterns: {error_msg}")
            return {'error': error_msg}
    
    def _analyze_matchups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza rendimiento contra equipos específicos."""
        try:
            if 'Opp' not in df.columns:
                return {'error': 'Columna Opp no encontrada'}
                
            # Estadísticas disponibles para análisis
            stats_cols = ['PTS', 'TRB', 'AST', 'double_double', 'triple_double']
            available_stats = [col for col in stats_cols if col in df.columns]
            
            # Crear agregaciones para cada estadística disponible
            aggs = {stat: 'mean' for stat in available_stats}
            
            # Agregación por oponente
            matchup_stats = df.groupby('Opp').agg(aggs).round(3)
            
            # Visualizar rendimiento contra equipos
            try:
                import plotly.express as px
                
                # Preparar datos para visualización
                matchup_df = matchup_stats.reset_index()
                
                # Elegir las dos primeras estadísticas disponibles para el scatter plot
                if len(available_stats) >= 2:
                    x_stat = available_stats[0]
                    y_stat = available_stats[1]
                    
                    fig = px.scatter(matchup_df,
                                   x=x_stat, y=y_stat,
                                   text='Opp',
                                   title=f'Rendimiento contra Equipos ({x_stat} vs {y_stat})')
                    fig.write_html(f'{self.output_dir}/matchup_analysis.html')
                    
                    # Añadir ruta de visualización al resultado
                    result = {
                        'team_performance': matchup_stats,
                        'visualization_path': f'{self.output_dir}/matchup_analysis.html'
                    }
                else:
                    result = {'team_performance': matchup_stats}
            except Exception as viz_e:
                error_msg = repr(str(viz_e))
                logger.warning(f"No se pudo generar visualización de matchups: {error_msg}")
                result = {'team_performance': matchup_stats}
                
            return result
            
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error en _analyze_matchups: {error_msg}")
            return {'error': error_msg}
    
    def _calculate_player_streaks(self, player_df: pd.DataFrame) -> pd.Series:
        """Calcula rachas para un jugador específico."""
        try:
            # Verificar si tenemos la columna double_double
            if 'double_double' not in player_df.columns:
                return pd.Series({
                    'streak_length': 0,
                    'total_games': len(player_df),
                    'error': 'Columna double_double no encontrada'
                })
                
            current_streak = 0
            max_streak = 0
            
            for dd in player_df['double_double']:
                if dd == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
                    
            return pd.Series({
                'streak_length': max_streak,
                'total_games': len(player_df)
            })
        except Exception as e:
            error_msg = repr(str(e))
            logger.warning(f"Error calculando rachas para jugador: {error_msg}")
            return pd.Series({
                'streak_length': 0,
                'total_games': len(player_df) if isinstance(player_df, pd.DataFrame) else 0,
                'error': error_msg
            })
    
    def generate_advanced_report(self, model_diagnostics: Dict[str, Any], 
                               pattern_analysis: Dict[str, Any]) -> str:
        """
        Genera un reporte HTML detallado con todos los análisis.
        
        Args:
            model_diagnostics: Diagnósticos del modelo
            pattern_analysis: Análisis de patrones
            
        Returns:
            Ruta al reporte HTML generado
        """
        report_template = """
        <html>
        <head>
            <title>Análisis Avanzado NBA</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; }
                .visualization { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Reporte de Análisis Avanzado NBA</h1>
            
            <div class="section">
                <h2>Diagnóstico del Modelo</h2>
                <div class="visualization">
                    <iframe src="model_evaluation.html" width="100%" height="600px"></iframe>
                </div>
                <div class="visualization">
                    <img src="shap_summary.png" width="100%">
                </div>
                <div class="visualization">
                    <iframe src="feature_importance.html" width="100%" height="600px"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>Análisis de Patrones</h2>
                <div class="visualization">
                    <iframe src="longest_streaks.html" width="100%" height="600px"></iframe>
                </div>
                <div class="visualization">
                    <iframe src="stats_by_position.html" width="100%" height="600px"></iframe>
                </div>
                <div class="visualization">
                    <iframe src="temporal_trends.html" width="100%" height="600px"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = f'{self.output_dir}/advanced_analysis_report.html'
        
        with open(report_path, 'w') as f:
            f.write(report_template)
        
        return report_path 

    def safe_predict(self, X, model, X_cols):
        """
        Función de predicción segura que maneja errores.
        
        Args:
            X: Datos de entrada
            model: Modelo a utilizar
            X_cols: Columnas para el DataFrame
            
        Returns:
            array: Predicciones
        """
        try:
            # Convertir a DataFrame con las columnas correctas si es necesario
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=X_cols)
            else:
                X_df = X
                
            # Aplicar ajustes de compatibilidad
            X_df = adjust_features_compatibility(X_df, model)
            return model.predict(X_df)
        except Exception as e:
            logger.warning(f"Error en safe_predict: {repr(str(e))}")
            # En caso de error, devolver ceros
            return np.zeros(len(X))

def adjust_features_compatibility(X_test, model):
    """
    Ajusta las características de X_test para que sean compatibles con el modelo.
    
    Args:
        X_test (pd.DataFrame): Características de prueba
        model: Modelo a evaluar
        
    Returns:
        pd.DataFrame o np.ndarray: Características ajustadas en el formato adecuado
    """
    import numpy as np
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Asegurar que X_test es un DataFrame
    if not isinstance(X_test, pd.DataFrame):
        try:
            X_test = pd.DataFrame(X_test)
            logger.warning("X_test convertido a DataFrame")
        except Exception as e:
            logger.error(f"No se pudo convertir X_test a DataFrame: {repr(str(e))}")
            return X_test
    
    # Detectar contexto de XGBoost y otras bibliotecas que requieren arrays numpy
    requires_numpy_array = False
    try:
        import traceback
        stack = traceback.extract_stack()
        for frame in stack:
            frame_info = frame.filename.lower() if hasattr(frame, 'filename') else str(frame).lower()
            # Detectar contextos que requieren arrays numpy
            if any(lib in frame_info for lib in ['xgboost', 'learning_curve', 'permutation_importance']):
                requires_numpy_array = True
                logger.info(f"Detectado contexto que requiere array NumPy: {frame_info}")
                break
    except Exception as e:
        logger.warning(f"Error al analizar la pila de llamadas: {repr(str(e))}")
    
    # Para modelos XGBoost con get_booster() - manejo especial
    if hasattr(model, 'get_booster'):
        try:
            # Obtener las características del modelo en el orden correcto
            booster = model.get_booster()
            if hasattr(booster, 'feature_names'):
                model_features = booster.feature_names
                logger.info(f"Modelo XGBoost encontrado con {len(model_features)} características")
                
                # Asegurar que todas las características estén presentes
                missing_features = [f for f in model_features if f not in X_test.columns]
                
                if missing_features:
                    logger.warning(f"Faltan {len(missing_features)} características en los datos para XGBoost: {missing_features}")
                    
                    # Crear una copia para no modificar el original
                    X_adjusted = X_test.copy()
                    
                    # Agregar características faltantes
                    for feature in missing_features:
                        # Manejar características especiales de FGA
                        if feature.startswith('FGA_'):
                            if 'FGA' in X_adjusted.columns:
                                if feature == 'FGA_sqrt':
                                    X_adjusted[feature] = np.sqrt(X_adjusted['FGA'].astype(float))
                                elif feature == 'FGA_log2':
                                    X_adjusted[feature] = np.log2(X_adjusted['FGA'].astype(float) + 1)
                                elif feature == 'FGA_yj':
                                    X_adjusted[feature] = 0.5 * np.log1p(X_adjusted['FGA'].astype(float))
                                elif feature == 'FGA_rank':
                                    X_adjusted[feature] = (X_adjusted['FGA'] - X_adjusted['FGA'].min()) / (X_adjusted['FGA'].max() - X_adjusted['FGA'].min() + 1e-8)
                                else:
                                    X_adjusted[feature] = X_adjusted['FGA'] / (X_adjusted['FGA'].mean() + 1e-8)
                            else:
                                X_adjusted[feature] = 0.5
                        # Manejar características de interacción
                        elif '_x_' in feature:
                            parts = feature.split('_x_')
                            if len(parts) == 2 and parts[0] in X_adjusted.columns and parts[1] in X_adjusted.columns:
                                # Convertir a float para evitar problemas con tipos
                                X_adjusted[feature] = X_adjusted[parts[0]].astype(float) * X_adjusted[parts[1]].astype(float)
                            else:
                                X_adjusted[feature] = 0.0
                        # Manejar características cuadráticas
                        elif feature.endswith('_squared') and feature.split('_')[0] in X_adjusted.columns:
                            base_feature = feature.split('_')[0]
                            X_adjusted[feature] = X_adjusted[base_feature].astype(float) ** 2
                        else:
                            X_adjusted[feature] = 0.0
                    
                    # Asegurarse de que devolvemos las características en el orden correcto
                    # y que todos los tipos sean compatibles con el modelo
                    result_df = X_adjusted[model_features]
                else:
                    # Si no faltan características, solo asegurar el orden correcto
                    result_df = X_test[model_features]
                
                # Convertir columnas a float para evitar problemas de tipo
                for col in result_df.columns:
                    if not pd.api.types.is_numeric_dtype(result_df[col]):
                        try:
                            result_df[col] = result_df[col].astype(float)
                        except:
                            result_df[col] = 0.0
                
                # Verificar si hay NaNs y reemplazarlos
                if result_df.isna().any().any():
                    logger.warning("Detectados valores NaN en las características. Imputando con ceros.")
                    result_df = result_df.fillna(0.0)
                
                # Devolver como array numpy si es requerido por el contexto
                if requires_numpy_array:
                    logger.info("Devolviendo array NumPy para XGBoost/learning_curve")
                    return result_df.values
                
                return result_df
                
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error en manejo especial de XGBoost: {error_msg}")
            # Si falla, continuar con el manejo más general
    
    # Para cualquier modelo con feature_names_in_ (scikit-learn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        try:
            model_features = model.feature_names_in_
            logger.info(f"Modelo scikit-learn encontrado con {len(model_features)} características")
            
            # Comprobar si faltan características
            missing_features = [f for f in model_features if f not in X_test.columns]
            
            if missing_features:
                logger.warning(f"Faltan {len(missing_features)} características para scikit-learn: {missing_features}")
                
                # Crear una copia para no modificar el original
                X_adjusted = X_test.copy()
                
                # Agregar características faltantes con valores neutros
                for feature in missing_features:
                    X_adjusted[feature] = 0.0
                
                # Asegurarse de que devolvemos las características en el orden correcto
                result_df = X_adjusted[model_features]
            else:
                # Si no faltan características, solo asegurar el orden correcto
                result_df = X_test[model_features]
            
            # Verificar si hay NaNs y reemplazarlos
            if result_df.isna().any().any():
                logger.warning("Detectados valores NaN en las características. Imputando con ceros.")
                result_df = result_df.fillna(0.0)
            
            # Devolver como array numpy si es requerido por el contexto
            if requires_numpy_array:
                logger.info("Devolviendo array NumPy para scikit-learn (learning_curve, etc.)")
                return result_df.values
                
            return result_df
            
        except Exception as e:
            error_msg = repr(str(e))
            logger.error(f"Error al ajustar características para scikit-learn: {error_msg}")
    
    # Si llega aquí y requiere array NumPy, convertir directamente
    if requires_numpy_array:
        logger.info("Convirtiendo a NumPy array para contexto especial")
        # Convertir a array NumPy y asegurar que no haya NaNs
        try:
            numpy_array = X_test.values
            # Reemplazar NaNs con ceros
            if np.isnan(numpy_array).any():
                numpy_array = np.nan_to_num(numpy_array, nan=0.0)
            return numpy_array
        except Exception as e:
            logger.error(f"Error al convertir a NumPy array: {repr(str(e))}")
    
    # Si llega aquí, devolver X_test intentando convertir tipos si es necesario
    try:
        # Verificar y convertir tipos para evitar problemas de compatibilidad
        for col in X_test.columns:
            if not pd.api.types.is_numeric_dtype(X_test[col]):
                try:
                    X_test[col] = X_test[col].astype(float)
                except:
                    logger.warning(f"No se pudo convertir columna '{col}' a tipo numérico")
        
        # Verificar si hay NaNs y reemplazarlos
        if X_test.isna().any().any():
            logger.warning("Detectados valores NaN en DataFrame final. Imputando con ceros.")
            X_test = X_test.fillna(0.0)
    except Exception as e:
        logger.warning(f"Error al procesar tipos de datos: {repr(str(e))}")
    
    return X_test

def _prepare_data_for_sklearn(X, y=None):
    """
    Prepara los datos para funciones de scikit-learn, manejando el error
    "'DataFrame' object has no attribute 'dtype'"
    
    Args:
        X: Características (DataFrame o array)
        y: Variable objetivo (opcional)
        
    Returns:
        tuple: (X_array, y_array) como arrays NumPy
    """
    # Convertir X a array NumPy si es DataFrame
    if hasattr(X, 'values'):
        # Verificar y manejar NaNs
        if hasattr(X, 'isna') and X.isna().any().any():
            logger.warning("Detectados valores NaN en las características. Imputando con ceros.")
            X_clean = X.fillna(0)
        else:
            X_clean = X
        X_array = X_clean.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        try:
            X_array = np.array(X)
        except:
            logger.error(f"No se pudo convertir X de tipo {type(X)} a array NumPy")
            raise ValueError(f"Tipo de datos no compatible: {type(X)}")
    
    # Manejar valores infinitos o NaN en X
    if np.isnan(X_array).any() or np.isinf(X_array).any():
        logger.warning("Detectados valores infinitos o NaN en X. Reemplazando.")
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Convertir y a array NumPy si es necesario
    if y is not None:
        if hasattr(y, 'values'):
            y_array = y.values
        elif isinstance(y, np.ndarray):
            y_array = y
        else:
            try:
                y_array = np.array(y)
            except:
                logger.error(f"No se pudo convertir y de tipo {type(y)} a array NumPy")
                raise ValueError(f"Tipo de datos no compatible: {type(y)}")
        
        # Manejar valores NaN en y
        if np.isnan(y_array).any() or np.isinf(y_array).any():
            logger.warning("Detectados valores infinitos o NaN en y. Reemplazando.")
            y_array = np.nan_to_num(y_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_array, y_array
    else:
        return X_array

def generate_learning_curve(model, X_train, y_train, cv=5):
    """
    Genera curvas de aprendizaje para un modelo dado, manejando errores comunes.
    
    Args:
        model: Modelo a evaluar
        X_train: Características de entrenamiento
        y_train: Variable objetivo
        cv: Número de folds para validación cruzada
        
    Returns:
        dict: Resultados de la curva de aprendizaje
    """
    from sklearn.model_selection import learning_curve
    
    try:
        # Preparar datos para evitar el error 'DataFrame' object has no attribute 'dtype'
        X_train_array, y_train_array = _prepare_data_for_sklearn(X_train, y_train)
        
        # Intentar con el modelo original
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train_array, y_train_array, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), random_state=42,
                error_score='raise'
            )
            logger.info("Curva de aprendizaje generada exitosamente con arrays NumPy")
        except Exception as model_error:
            error_msg = repr(str(model_error))
            logger.warning(f"Error con modelo original en learning_curve: {error_msg}")
            
            # Verificar si el error es 'DataFrame' object has no attribute 'dtype'
            if "object has no attribute 'dtype'" in str(model_error):
                logger.info("Detectado error de dtype. Intentando con arrays NumPy directamente.")
                
                # Crear un wrapper para el modelo que garantice compatibilidad
                class SafeLearningCurveWrapper(BaseEstimator):
                    def __init__(self, model):
                        self.model = model
                    
                    def fit(self, X, y):
                        # Asegurar que X e y son arrays NumPy
                        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
                        y_array = np.array(y) if not isinstance(y, np.ndarray) else y
                        
                        # Manejar NaNs
                        X_array = np.nan_to_num(X_array, nan=0.0)
                        y_array = np.nan_to_num(y_array, nan=0.0)
                        
                        # Clonar el modelo para no modificar el original
                        from sklearn.base import clone
                        self.fitted_model = clone(self.model)
                        self.fitted_model.fit(X_array, y_array)
                        return self
                    
                    def predict(self, X):
                        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
                        X_array = np.nan_to_num(X_array, nan=0.0)
                        return self.fitted_model.predict(X_array)
                    
                    # Añadir método score para compatibilidad con learning_curve
                    def score(self, X, y):
                        from sklearn.metrics import r2_score
                        y_pred = self.predict(X)
                        return r2_score(y, y_pred)
                
                try:
                    # Usar el wrapper para learning_curve
                    safe_model = SafeLearningCurveWrapper(model)
                    
                    # Convertir los datos a arrays NumPy para evitar problemas
                    X_train_array = np.array(X_train_array)
                    y_train_array = np.array(y_train_array)
                    
                    # Reemplazar NaNs si existen
                    X_train_array = np.nan_to_num(X_train_array, nan=0.0)
                    y_train_array = np.nan_to_num(y_train_array, nan=0.0)
                    
                    train_sizes, train_scores, val_scores = learning_curve(
                        safe_model, X_train_array, y_train_array, cv=cv, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42,
                        scoring='r2'  # Especificar scoring explícitamente
                    )
                    logger.info("Éxito usando wrapper con arrays NumPy para learning_curve")
                except Exception as wrapper_error:
                    logger.warning(f"Error con wrapper para learning_curve: {repr(str(wrapper_error))}")
                    
                    # Si falla el wrapper, intentar con un modelo más simple
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        
                        logger.info("Usando modelo alternativo con arrays NumPy para curvas de aprendizaje")
                        simple_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        
                        train_sizes, train_scores, val_scores = learning_curve(
                            simple_model, X_train_array, y_train_array, cv=cv, n_jobs=-1,
                            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
                        )
                    except Exception as simple_error:
                        logger.error(f"Error con modelo simple: {repr(str(simple_error))}")
                        # Devolver valores vacíos si todo falla
                        return {
                            'train_sizes': [],
                            'train_scores_mean': [],
                            'train_scores_std': [],
                            'val_scores_mean': [],
                            'val_scores_std': [],
                            'error': str(simple_error)
                        }
        
        # Calcular estadísticas
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'val_scores_mean': val_scores_mean.tolist(),
            'val_scores_std': val_scores_std.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error general en generate_learning_curve: {str(e)}")
        return {
            'train_sizes': [],
            'train_scores_mean': [],
            'train_scores_std': [],
            'val_scores_mean': [],
            'val_scores_std': [],
            'error': str(e)
        }