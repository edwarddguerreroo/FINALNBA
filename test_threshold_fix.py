#!/usr/bin/env python3
"""
Script de prueba rápida para verificar la corrección del threshold
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.players.double_double.dd_model import create_double_double_model
import pandas as pd
import numpy as np

def test_threshold_fix():
    """Prueba rápida del threshold corregido"""
    
    print("=== PRUEBA DE CORRECCIÓN DE THRESHOLD ===")
    
    # Crear datos de prueba simulados
    np.random.seed(42)
    n_samples = 1000
    
    # Simular features
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'Date': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    })
    
    # Simular target con desbalance (8.5% positivos)
    y = np.random.choice([0, 1], size=n_samples, p=[0.915, 0.085])
    
    # Crear DataFrame completo
    df_test = X.copy()
    df_test['double_double'] = y
    
    print(f"Datos de prueba creados: {n_samples} muestras")
    print(f"Distribución target: {np.mean(y):.3f} positivos")
    
    # Crear modelo simple (sin optimización para prueba rápida)
    model = create_double_double_model(
        use_gpu=False,
        optimize_hyperparams=False
    )
    
    print("Modelo creado, iniciando entrenamiento...")
    
    # Entrenar modelo
    try:
        results = model.train(df_test, validation_split=0.2)
        print("✅ Entrenamiento completado")
        
        # Verificar threshold
        threshold = getattr(model, 'optimal_threshold', None)
        print(f"Threshold óptimo: {threshold}")
        
        # Hacer predicciones
        predictions = model.predict(df_test)
        probabilities = model.predict_proba(df_test)
        
        print(f"Predicciones positivas: {np.sum(predictions)}/{len(predictions)} ({np.sum(predictions)/len(predictions)*100:.1f}%)")
        print(f"Probabilidades - Min: {probabilities[:, 1].min():.4f}, Max: {probabilities[:, 1].max():.4f}")
        
        if np.sum(predictions) > 0:
            print("✅ CORRECCIÓN EXITOSA: El modelo está prediciendo double-doubles")
        else:
            print("❌ PROBLEMA PERSISTE: El modelo no predice double-doubles")
            
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_threshold_fix() 