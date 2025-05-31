#!/usr/bin/env python3
"""
Script de Prueba para Entrenamiento del Modelo Double Double NBA con Features Especializadas
===========================================================================================

Este script prueba el entrenamiento del modelo Double Double usando:
- Features especializadas del DoubleDoubleFeatureEngineer
- Modelo con stacking completo (7 modelos: XGB, LGB, RF, ET, GB, CB, NN)
- Validación de uso de features especializadas
- Exportación a JSON con métricas completas
- Visualizaciones comprehensivas de análisis
- Métricas de overfitting
"""

import sys
import warnings
import time
import json
import os
from pathlib import Path

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append('.')

# Imports principales
import pandas as pd
import numpy as np
from datetime import datetime

# Imports para visualización
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.players.double_double.dd_model import create_double_double_model

# Configurar matplotlib para mejor calidad
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [20, 15]
sns.set_style("whitegrid")

def print_header(title: str):
    """Imprimir encabezado con formato"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_section(title: str):
    """Imprimir sección con formato"""
    print(f"\n--- {title} ---")

class NumpyEncoder(json.JSONEncoder):
    """Encoder para manejar arrays numpy y otros tipos en JSON"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def save_comprehensive_json(results: dict, model, test_data: pd.DataFrame, 
                           predictions: np.ndarray, probabilities: np.ndarray,
                           train_predictions: np.ndarray = None, train_probabilities: np.ndarray = None,
                           json_path: str = "json/comprehensive_analysis.json"):
    """Guardar análisis comprehensivo en JSON con manejo de arrays numpy"""
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Preparar datos para JSON
    comprehensive_data = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "total_models": len(results.get('individual_models', {})),
            "stacking_enabled": True,
            "features_used": len(results.get('feature_columns', [])),
            "specialized_features": results.get('specialized_features_used', 0),
            "training_samples": results.get('training_samples', 0),
            "validation_samples": results.get('validation_samples', 0)
        },
        "performance_metrics": {
            "individual_models": {},
            "stacking_model": results.get('stacking_metrics', {}),
            "overfitting_analysis": {}
        },
        "predictions_analysis": {
            "validation_predictions": predictions,
            "validation_probabilities": probabilities[:, 1],
            "validation_actuals": test_data['double_double'].values,
            "prediction_distribution": np.bincount(predictions).tolist(),
            "probability_stats": {
                "min": float(probabilities[:, 1].min()),
                "max": float(probabilities[:, 1].max()),
                "mean": float(probabilities[:, 1].mean()),
                "std": float(probabilities[:, 1].std())
            }
        },
        "feature_analysis": {
            "feature_columns": results.get('feature_columns', []),
            "feature_importance": getattr(model, 'feature_importance', {}),
            "top_features": []
        },
        "data_distribution": {
            "target_distribution": test_data['double_double'].value_counts().to_dict(),
            "target_percentage": (test_data['double_double'].value_counts(normalize=True) * 100).to_dict()
        }
    }
    
    # Agregar métricas de modelos individuales
    if 'individual_models' in results:
        for name, model_info in results['individual_models'].items():
            if 'val_metrics' in model_info:
                comprehensive_data['performance_metrics']['individual_models'][name] = model_info['val_metrics']
    
    # Análisis de overfitting si tenemos predicciones de entrenamiento
    if train_predictions is not None and train_probabilities is not None:
        # Calcular métricas de entrenamiento
        train_accuracy = (train_predictions == test_data['double_double'].values[:len(train_predictions)]).mean()
        val_accuracy = (predictions == test_data['double_double'].values[-len(predictions):]).mean()
        
        overfitting_gap = train_accuracy - val_accuracy
        
        comprehensive_data['performance_metrics']['overfitting_analysis'] = {
            "train_accuracy": float(train_accuracy),
            "validation_accuracy": float(val_accuracy),
            "overfitting_gap": float(overfitting_gap),
            "overfitting_severity": "High" if overfitting_gap > 0.1 else "Medium" if overfitting_gap > 0.05 else "Low",
            "train_probability_mean": float(train_probabilities[:, 1].mean()),
            "val_probability_mean": float(probabilities[:, 1].mean())
        }
    
    # Top features importance
    if hasattr(model, 'feature_importance') and model.feature_importance:
        top_features_info = model.get_feature_importance(top_n=15)
        comprehensive_data['feature_analysis']['top_features'] = top_features_info
    
    # Guardar JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    print(f"📄 JSON comprehensivo guardado: {json_path}")
    return comprehensive_data

def create_comprehensive_visualization(results: dict, model, test_data: pd.DataFrame,
                                     predictions: np.ndarray, probabilities: np.ndarray,
                                     plot_path: str = "plots/comprehensive_analysis.png"):
    """Crear visualización comprehensiva como la imagen anexada"""
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # Configurar figura con múltiples subplots
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Datos reales vs predicciones
    y_true = test_data['double_double'].values[-len(predictions):]
    y_prob = probabilities[:, 1]
    
    # 1. Scatter Plot: Predicciones vs Reales
    ax1 = fig.add_subplot(gs[0, 0])
    scatter_colors = ['red' if pred != true else 'blue' for pred, true in zip(predictions, y_true)]
    ax1.scatter(y_true, predictions, c=scatter_colors, alpha=0.6)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('Valores Reales')
    ax1.set_ylabel('Predicciones')
    ax1.set_title('Predicciones vs Reales')
    ax1.text(0.05, 0.95, f'AUC: {results.get("stacking_metrics", {}).get("roc_auc", 0):.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 2. Distribución de Probabilidades
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(y_prob, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(y_prob.mean(), color='red', linestyle='--', label=f'Media: {y_prob.mean():.3f}')
    ax2.set_xlabel('Probabilidad Predicha')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Probabilidades')
    ax2.legend()
    
    # 3. Curva ROC
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Tasa Falsos Positivos')
    ax3.set_ylabel('Tasa Verdaderos Positivos')
    ax3.set_title('Curva ROC')
    ax3.legend(loc="lower right")
    
    # 4. Distribución de Residuos
    ax4 = fig.add_subplot(gs[0, 3])
    residuos = y_prob - y_true
    ax4.hist(residuos, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Residuos')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Residuos')
    
    # 5. Matriz de Confusión
    ax5 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_true, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_xlabel('Predicciones')
    ax5.set_ylabel('Valores Reales')
    ax5.set_title('Matriz de Confusión')
    
    # 6. Comparación de Modelos (Accuracy)
    ax6 = fig.add_subplot(gs[1, 1])
    if 'individual_models' in results:
        model_names = []
        accuracies = []
        for name, model_info in results['individual_models'].items():
            if 'val_metrics' in model_info:
                model_names.append(name.replace('_', ' ').title())
                accuracies.append(model_info['val_metrics'].get('accuracy', 0))
        
        # Agregar stacking
        if 'stacking_metrics' in results:
            model_names.append('Stacking')
            accuracies.append(results['stacking_metrics'].get('accuracy', 0))
        
        bars = ax6.bar(range(len(model_names)), accuracies, color='lightgreen', alpha=0.7)
        ax6.set_xticks(range(len(model_names)))
        ax6.set_xticklabels(model_names, rotation=45, ha='right')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Comparación de Modelos')
        
        # Agregar valores en las barras
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 7. Feature Importance (Top 10)
    ax7 = fig.add_subplot(gs[1, 2:])
    if hasattr(model, 'feature_importance') and model.feature_importance:
        # Obtener importancia promedio
        if 'average' in model.feature_importance:
            importance_data = model.feature_importance['average']
            features = importance_data['feature_names'][:15]
            importances = importance_data['importances'][:15]
            
            y_pos = np.arange(len(features))
            bars = ax7.barh(y_pos, importances, color='gold', alpha=0.7)
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels(features)
            ax7.set_xlabel('Importancia')
            ax7.set_title('Top 15 Features Más Importantes')
            ax7.invert_yaxis()
    
    # 8. Curva Precision-Recall
    ax8 = fig.add_subplot(gs[2, 0])
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax8.plot(recall, precision, color='purple', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
    ax8.set_xlabel('Recall')
    ax8.set_ylabel('Precision')
    ax8.set_title('Curva Precision-Recall')
    ax8.legend()
    
    # 9. Distribución por Clase
    ax9 = fig.add_subplot(gs[2, 1])
    class_counts = np.bincount(y_true)
    pred_counts = np.bincount(predictions)
    
    x = np.arange(len(class_counts))
    width = 0.35
    
    ax9.bar(x - width/2, class_counts, width, label='Real', alpha=0.7)
    ax9.bar(x + width/2, pred_counts, width, label='Predicho', alpha=0.7)
    ax9.set_xlabel('Clase')
    ax9.set_ylabel('Conteo')
    ax9.set_title('Distribución Real vs Predicha')
    ax9.set_xticks(x)
    ax9.set_xticklabels(['No DD', 'DD'])
    ax9.legend()
    
    # 10. Accuracy por Rango de Confianza
    ax10 = fig.add_subplot(gs[2, 2])
    confidence_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    range_accuracies = []
    range_counts = []
    
    for min_conf, max_conf in confidence_ranges:
        mask = (y_prob >= min_conf) & (y_prob < max_conf)
        if mask.sum() > 0:
            acc = (predictions[mask] == y_true[mask]).mean()
            range_accuracies.append(acc)
            range_counts.append(mask.sum())
        else:
            range_accuracies.append(0)
            range_counts.append(0)
    
    range_labels = [f'{r[0]:.1f}-{r[1]:.1f}' for r in confidence_ranges]
    bars = ax10.bar(range_labels, range_accuracies, color='coral', alpha=0.7)
    ax10.set_xlabel('Rango de Confianza')
    ax10.set_ylabel('Accuracy')
    ax10.set_title('Accuracy por Rango de Confianza')
    
    # Agregar conteos en las barras
    for bar, count in zip(bars, range_counts):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 11. Métricas por Modelo (Heatmap)
    ax11 = fig.add_subplot(gs[2, 3])
    if 'individual_models' in results:
        metrics_data = []
        model_names_clean = []
        
        for name, model_info in results['individual_models'].items():
            if 'val_metrics' in model_info:
                metrics = model_info['val_metrics']
                metrics_data.append([
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0), 
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0),
                    metrics.get('roc_auc', 0)
                ])
                model_names_clean.append(name.replace('_', ' ').title())
        
        if 'stacking_metrics' in results:
            stacking = results['stacking_metrics']
            metrics_data.append([
                stacking.get('accuracy', 0),
                stacking.get('precision', 0),
                stacking.get('recall', 0), 
                stacking.get('f1_score', 0),
                stacking.get('roc_auc', 0)
            ])
            model_names_clean.append('Stacking')
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data, 
                                    columns=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                                    index=model_names_clean)
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax11)
            ax11.set_title('Heatmap de Métricas por Modelo')
    
    # 12. Estadísticas de Distribución (Bottom Left)
    ax12 = fig.add_subplot(gs[3, :2])
    stats_text = f"""
ESTADÍSTICAS GENERALES:
• Total Muestras: {len(test_data):,}
• Features Especializadas: {results.get('specialized_features_used', 0)}
• Total Features: {len(results.get('feature_columns', []))}
• Accuracy Final: {results.get('stacking_metrics', {}).get('accuracy', 0):.3f}
• ROC AUC Final: {results.get('stacking_metrics', {}).get('roc_auc', 0):.3f}
• F1 Score Final: {results.get('stacking_metrics', {}).get('f1_score', 0):.3f}

DISTRIBUCIÓN TARGET:
• No Double-Double: {(y_true == 0).sum():,} ({(y_true == 0).mean()*100:.1f}%)
• Double-Double: {(y_true == 1).sum():,} ({(y_true == 1).mean()*100:.1f}%)

CALIDAD PREDICCIONES:
• Probabilidad Media: {y_prob.mean():.3f}
• Desv. Estándar: {y_prob.std():.3f}
• Confianza Alta (>0.8): {(y_prob > 0.8).sum():,} ({(y_prob > 0.8).mean()*100:.1f}%)
"""
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    # 13. Overfitting Analysis (Bottom Right)
    ax13 = fig.add_subplot(gs[3, 2:])
    if hasattr(model, 'cv_scores') and model.cv_scores:
        cv_data = []
        cv_models = []
        
        for name, scores in model.cv_scores.items():
            if isinstance(scores, dict) and 'scores' in scores:
                cv_data.extend(scores['scores'])
                cv_models.extend([name] * len(scores['scores']))
        
        if cv_data:
            cv_df = pd.DataFrame({'Model': cv_models, 'CV_Score': cv_data})
            sns.boxplot(data=cv_df, x='Model', y='CV_Score', ax=ax13)
            ax13.set_title('Cross-Validation Scores por Modelo')
            ax13.tick_params(axis='x', rotation=45)
    
    # Título general
    fig.suptitle('ANÁLISIS COMPREHENSIVO - MODELO DOUBLE DOUBLE NBA', fontsize=20, fontweight='bold')
    
    # Guardar figura
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Visualización comprehensiva guardada: {plot_path}")

def main():
    """Función principal del script de prueba con validación de correcciones"""
    
    print_header("PRUEBA DE CORRECCIONES DEL MODELO DOUBLE DOUBLE NBA")
    print("🔧 Validando correcciones para solucionar F1 Score = 0.000")
    print("🎯 Enfoque: Reducir regularización excesiva y mejorar manejo del desbalance")
    
    start_time = time.time()
    
    try:
        # Cargar datos
        print_section("1. CARGA DE DATOS")
        
        # Configurar paths de datos
        data_paths = {
            'players': 'data/players.csv',
            'teams': 'data/teams.csv', 
            'biometrics': 'data/height.csv'
        }
        
        # Inicializar data loader con paths correctos
        loader = NBADataLoader(
            game_data_path=data_paths['players'],
            biometrics_path=data_paths['biometrics'],
            teams_path=data_paths['teams']
        )
        
        print("📊 Cargando datos de juegos NBA...")
        
        # Cargar datos usando la función load_data
        merged_data, teams_data = loader.load_data()
        
        # Filtrar datos válidos y usar una muestra representativa
        valid_data = merged_data[merged_data['double_double'].notna()].copy()
        
        # Usar datos más recientes (últimos 6000 registros para mejores resultados)
        if len(valid_data) > 6000:
            df_combined = valid_data.tail(6000).copy()
        else:
            df_combined = valid_data.copy()
            
        print(f"✅ Datos cargados: {len(df_combined)} juegos totales")
        
        # Verificar distribución de double doubles ANTES del entrenamiento
        dd_distribution = df_combined['double_double'].value_counts()
        dd_percentage = df_combined['double_double'].value_counts(normalize=True) * 100
        
        print(f"\n📈 DISTRIBUCIÓN DE DOUBLE DOUBLES (ORIGINAL):")
        print(f"   No DD: {dd_distribution.get(0, 0)} ({dd_percentage.get(0, 0):.1f}%)")
        print(f"   DD:    {dd_distribution.get(1, 0)} ({dd_percentage.get(1, 0):.1f}%)")
        print(f"   Ratio de desbalance: {dd_distribution.get(0, 0) / max(dd_distribution.get(1, 1), 1):.1f}:1")
        
        # Verificar si el desbalance es extremo
        if dd_percentage.get(1, 0) < 5:
            print("⚠️  ALERTA: Desbalance extremo detectado (< 5% de double doubles)")
            print("🔧 Las correcciones implementadas deberían solucionar esto")
        
        # División cronológica manual para mejor control
        print_section("2. DIVISIÓN DE DATOS")
        df_sorted = df_combined.sort_values('Date').reset_index(drop=True)
        
        # 80% para entrenamiento, 20% para validación
        split_idx = int(len(df_sorted) * 0.8)
        train_data = df_sorted.iloc[:split_idx].copy()
        test_data = df_sorted.iloc[split_idx:].copy()
        
        print(f"📊 División cronológica:")
        print(f"   Entrenamiento: {len(train_data)} juegos")
        print(f"   Validación:    {len(test_data)} juegos")
        
        # Verificar distribución en cada set
        train_dd = train_data['double_double'].value_counts(normalize=True) * 100
        test_dd = test_data['double_double'].value_counts(normalize=True) * 100
        
        print(f"\n   Distribución en entrenamiento - DD: {train_dd.get(1, 0):.1f}%")
        print(f"   Distribución en validación    - DD: {test_dd.get(1, 0):.1f}%")
        
        # Crear y configurar modelo
        print_section("3. CREACIÓN DEL MODELO CON CORRECCIONES")
        print("🔧 Configurando modelo con correcciones implementadas:")
        print("   ✅ Regularización balanceada (no extrema)")
        print("   ✅ Manejo automático del desbalance de clases")
        print("   ✅ Umbral de predicción ajustado (0.3 en lugar de 0.5)")
        print("   ✅ Pesos de clase mejorados en red neuronal")
        print("   ✅ Mayor capacidad de los modelos")
        
        # Crear modelo sin optimización bayesiana para prueba rápida
        model = create_double_double_model(
            use_gpu=True,
            optimize_hyperparams=False,  # Deshabilitado para prueba rápida
            bayesian_n_calls=0
        )
        
        # Entrenar modelo
        print_section("4. ENTRENAMIENTO DEL MODELO")
        print("🚀 Iniciando entrenamiento con datos preparados...")
        
        training_start = time.time()
        results = model.train(train_data, validation_split=0.2)
        training_time = time.time() - training_start
        
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        # Verificar que se usaron features especializadas
        print_section("5. VALIDACIÓN DE FEATURES ESPECIALIZADAS")
        feature_columns = results.get('feature_columns', [])
        specialized_count = results.get('specialized_features_used', 0)
        total_generated = results.get('total_features_generated', 0)
        specialized_percentage = results.get('specialized_percentage', 0)
        
        print(f"📊 Features utilizadas: {len(feature_columns)}")
        print(f"📊 Features especializadas: {specialized_count}")
        print(f"📊 Total generadas: {total_generated}")
        print(f"📊 Porcentaje especializado: {specialized_percentage:.1f}%")
        
        if specialized_percentage < 80:
            print("⚠️  ADVERTENCIA: Porcentaje de features especializadas bajo")
        else:
            print("✅ Excelente uso de features especializadas")
        
        # Validar resultados del entrenamiento
        print_section("6. ANÁLISIS DE RESULTADOS DE ENTRENAMIENTO")
        
        # Métricas de modelos individuales
        individual_models = results.get('individual_models', {})
        print(f"🤖 Modelos entrenados: {len(individual_models)}")
        
        for name, model_info in individual_models.items():
            if 'val_metrics' in model_info:
                metrics = model_info['val_metrics']
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                auc = metrics.get('roc_auc', 0)
                print(f"   {name:15s}: ACC={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
                
                # VALIDACIÓN CRÍTICA: Verificar si F1 > 0
                if f1 == 0.0:
                    print(f"      ❌ PROBLEMA: {name} tiene F1=0.000")
                elif f1 > 0.1:
                    print(f"      ✅ MEJORADO: {name} tiene F1={f1:.3f}")
        
        # Métricas de stacking
        stacking_metrics = results.get('stacking_metrics', {})
        if stacking_metrics:
            acc = stacking_metrics.get('accuracy', 0)
            f1 = stacking_metrics.get('f1_score', 0)
            auc = stacking_metrics.get('roc_auc', 0)
            precision = stacking_metrics.get('precision', 0)
            recall = stacking_metrics.get('recall', 0)
            
            print(f"\n🏆 MODELO STACKING:")
            print(f"   Accuracy:  {acc:.3f}")
            print(f"   F1 Score:  {f1:.3f}")
            print(f"   AUC:       {auc:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            
            # VALIDACIÓN CRÍTICA DEL STACKING
            if f1 == 0.0:
                print("   ❌ PROBLEMA CRÍTICO: Stacking sigue con F1=0.000")
                print("   🔧 Necesita ajustes adicionales en umbral o pesos")
            elif f1 > 0.0 and f1 < 0.1:
                print("   🟡 MEJORA PARCIAL: F1 > 0 pero aún bajo")
                print("   🔧 Progreso positivo, puede necesitar ajuste fino")
            elif f1 >= 0.1:
                print("   ✅ CORRECCIÓN EXITOSA: F1 Score mejorado significativamente")
                
        # Predicciones en datos de validación
        print_section("7. PREDICCIONES EN DATOS DE VALIDACIÓN")
        
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
        
        # Análisis de predicciones
        pred_distribution = np.bincount(predictions)
        prob_stats = {
            'min': probabilities[:, 1].min(),
            'max': probabilities[:, 1].max(),
            'mean': probabilities[:, 1].mean(),
            'std': probabilities[:, 1].std()
        }
        
        print(f"📊 Distribución de predicciones:")
        print(f"   No DD predichos: {pred_distribution[0] if len(pred_distribution) > 0 else 0}")
        print(f"   DD predichos:    {pred_distribution[1] if len(pred_distribution) > 1 else 0}")
        
        print(f"\n📊 Estadísticas de probabilidades:")
        print(f"   Media:     {prob_stats['mean']:.3f}")
        print(f"   Std:       {prob_stats['std']:.3f}")
        print(f"   Min:       {prob_stats['min']:.3f}")
        print(f"   Max:       {prob_stats['max']:.3f}")
        
        # VALIDACIÓN CRÍTICA DE PREDICCIONES
        dd_predicted = pred_distribution[1] if len(pred_distribution) > 1 else 0
        if dd_predicted == 0:
            print("   ❌ PROBLEMA: No se predijeron double doubles")
            print("   🔧 Considerar reducir más el umbral de predicción")
        else:
            print(f"   ✅ MEJORA: Se predijeron {dd_predicted} double doubles")
            
        # Calcular métricas finales en test
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        y_true = test_data['double_double'].values
        test_accuracy = accuracy_score(y_true, predictions)
        test_f1 = f1_score(y_true, predictions, zero_division=0)
        test_precision = precision_score(y_true, predictions, zero_division=0)
        test_recall = recall_score(y_true, predictions, zero_division=0)
        test_auc = roc_auc_score(y_true, probabilities[:, 1])
        
        print_section("8. MÉTRICAS FINALES EN DATOS DE VALIDACIÓN")
        print(f"📊 Accuracy:  {test_accuracy:.3f}")
        print(f"📊 F1 Score:  {test_f1:.3f}")
        print(f"📊 Precision: {test_precision:.3f}")
        print(f"📊 Recall:    {test_recall:.3f}")
        print(f"📊 AUC:       {test_auc:.3f}")
        
        # EVALUACIÓN FINAL DE LAS CORRECCIONES
        print_section("9. EVALUACIÓN DE CORRECCIONES")
        
        if test_f1 == 0.0:
            print("❌ LAS CORRECCIONES NO SOLUCIONARON EL PROBLEMA")
            print("🔧 Acciones recomendadas:")
            print("   - Reducir más el umbral de predicción (< 0.3)")
            print("   - Aumentar más los pesos de la clase minoritaria")
            print("   - Usar técnicas de resampling (SMOTE)")
            print("   - Ajustar parámetros de regularización")
            
        elif test_f1 > 0.0 and test_f1 < 0.1:
            print("🟡 CORRECCIONES PARCIALMENTE EXITOSAS")
            print("🔧 F1 Score mejorado pero aún bajo")
            print("🎯 Ajustes adicionales recomendados")
            
        elif test_f1 >= 0.1:
            print("✅ CORRECCIONES EXITOSAS")
            print("🎉 F1 Score significativamente mejorado")
            print("🎯 Modelo funcionando correctamente")
            
        # Generar visualizaciones
        print_section("10. GENERACIÓN DE VISUALIZACIONES")
        
        try:
            create_comprehensive_visualization(results, model, test_data, predictions, probabilities)
            print("✅ Visualización comprehensiva creada")
        except Exception as e:
            print(f"⚠️  Error creando visualización: {str(e)}")
        
        # Guardar JSON
        print_section("11. EXPORTACIÓN DE RESULTADOS")
        
        try:
            json_data = save_comprehensive_json(results, model, test_data, predictions, probabilities)
            print("✅ Resultados exportados a JSON")
        except Exception as e:
            print(f"⚠️  Error guardando JSON: {str(e)}")
        
        # Resumen final
        total_time = time.time() - start_time
        print_section("RESUMEN FINAL")
        print(f"⏱️  Tiempo total: {total_time:.2f} segundos")
        print(f"🎯 F1 Score final: {test_f1:.3f}")
        print(f"🎯 AUC final: {test_auc:.3f}")
        
        if test_f1 > 0.0:
            print("🎉 ÉXITO: Las correcciones mejoraron el modelo")
        else:
            print("🔧 TRABAJO PENDIENTE: Se requieren ajustes adicionales")
            
        return {
            'success': test_f1 > 0.0,
            'f1_score': test_f1,
            'auc_score': test_auc,
            'model': model,
            'results': results
        }
        
    except Exception as e:
        print(f"\n❌ ERROR EN EJECUCIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    try:
        result = main()
        if result['success']:
            print("\n🎯 PRUEBA CON FEATURES ESPECIALIZADAS EXITOSA")
            exit(0)
        else:
            print("\n❌ PRUEBA FALLÓ")
            exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Script interrumpido por el usuario")
        exit(1)
    except Exception as e:
        print(f"\n💥 ERROR FATAL: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 