import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sequences(model_type):
    """Carga las secuencias generadas para un tipo de modelo específico."""
    try:
        data_path = Path('data/data')
        sequences = np.load(data_path / f'{model_type}_sequences.npy')
        targets = np.load(data_path / f'{model_type}_targets.npy')
        categorical = np.load(data_path / f'{model_type}_categorical.npy')
        line_values = np.load(data_path / f'{model_type}_line_values.npy')
        
        return sequences, targets, categorical, line_values
    except Exception as e:
        logger.error(f"Error cargando secuencias para {model_type}: {e}")
        return None, None, None, None

def analyze_feature_importance(sequences, targets, feature_names, model_type):
    """Analiza la importancia de las características usando múltiples métodos."""
    results = {}
    
    # 1. Correlación con el target
    correlations = []
    for i in range(sequences.shape[2]):  # Para cada característica
        feature_values = sequences[:, :, i].mean(axis=1)  # Promedio a través de la secuencia
        correlation = np.corrcoef(feature_values, targets)[0, 1]
        correlations.append(abs(correlation))
    
    results['correlation'] = dict(zip(feature_names, correlations))
    
    # 2. Importancia basada en Random Forest
    X = sequences.reshape(sequences.shape[0], -1)  # Aplanar las secuencias
    if model_type in ['win_predictor', 'double_double_predictor', 'triple_double_predictor']:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, targets)
    feature_importance = model.feature_importances_
    
    # Reshape la importancia para que coincida con las características originales
    importance_per_feature = feature_importance.reshape(-1, sequences.shape[2]).mean(axis=0)
    results['random_forest'] = dict(zip(feature_names, importance_per_feature))
    
    # 3. Mutual Information
    if model_type in ['win_predictor', 'double_double_predictor', 'triple_double_predictor']:
        mi_scores = mutual_info_classif(X, targets)
    else:
        mi_scores = mutual_info_regression(X, targets)
    
    mi_per_feature = mi_scores.reshape(-1, sequences.shape[2]).mean(axis=0)
    results['mutual_info'] = dict(zip(feature_names, mi_per_feature))
    
    return results

def plot_feature_importance(results, model_type, output_dir):
    """Genera gráficos de importancia de características."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method, importance in results.items():
        plt.figure(figsize=(12, 6))
        features = list(importance.keys())
        scores = list(importance.values())
        
        # Ordenar por importancia
        sorted_idx = np.argsort(scores)
        features = [features[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importancia')
        plt.title(f'Importancia de Características - {method} - {model_type}')
        plt.tight_layout()
        
        # Guardar gráfico
        plt.savefig(output_dir / f'{model_type}_{method}_importance.png')
        plt.close()

def main():
    # Definir características base
    base_features = ['PTS', 'TRB', 'AST', '3P', 'team_score', 'opp_score']
    
    # Tipos de modelos a analizar
    model_types = [
        'pts_predictor',
        'trb_predictor',
        'ast_predictor',
        '3p_predictor',
        'win_predictor',
        'total_points_predictor',
        'team_points_predictor',
        'double_double_predictor',
        'triple_double_predictor'
    ]
    
    # Directorio para guardar resultados
    output_dir = Path('../../data/feature_importance')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analizar cada modelo
    for model_type in model_types:
        logger.info(f"\nAnalizando características para {model_type}")
        
        # Cargar secuencias
        sequences, targets, categorical, line_values = load_sequences(model_type)
        if sequences is None:
            continue
        
        # Analizar importancia de características
        results = analyze_feature_importance(sequences, targets, base_features, model_type)
        
        # Guardar resultados
        with open(output_dir / f'{model_type}_feature_importance.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generar gráficos
        plot_feature_importance(results, model_type, output_dir)
        
        logger.info(f"Análisis completado para {model_type}")
    
    logger.info("\n¡Análisis de características completado!")

if __name__ == "__main__":
    main() 