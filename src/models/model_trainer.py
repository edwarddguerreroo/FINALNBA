import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from .points_model import PointsModel
from .rebounds_model import ReboundsModel
from .assists_model import AssistsModel
from .threes_model import ThreesModel
from .double_double_model import DoubleDoubleModel

from ..preprocessing.data_loader import NBADataLoader
from ..preprocessing.feature_engineering.players_features import PlayersFeatures
from ..preprocessing.utils.features_selector import FeaturesSelector

logger = logging.getLogger(__name__)

class NBAModelTrainer:
    """
    Entrenador principal que coordina el entrenamiento de todos los modelos específicos para NBA.
    Maneja la carga de datos, generación de características y entrenamiento de modelos separados.
    """
    
    def __init__(self, data_paths: Dict[str, str], output_dir: str = "models"):
        """
        Inicializa el entrenador de modelos NBA.
        
        Args:
            data_paths (dict): Diccionario con rutas a los archivos de datos
                - 'game_data': Ruta a datos de partidos
                - 'biometrics': Ruta a datos biométricos  
                - 'teams': Ruta a datos de equipos
            output_dir (str): Directorio para guardar modelos entrenados
        """
        self.data_paths = data_paths
        self.output_dir = output_dir
        self.models = {}
        self.data_loader = None
        self.feature_generator = None
        self.merged_data = None
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar modelos específicos
        self._initialize_models()
        
    def _initialize_models(self):
        """Inicializa todos los modelos específicos."""
        self.models = {
            'points': PointsModel(),
            'rebounds': ReboundsModel(), 
            'assists': AssistsModel(),
            'threes': ThreesModel(),
            'double_double': DoubleDoubleModel(target='double_double'),
            'triple_double': DoubleDoubleModel(target='triple_double')
        }
        
        logger.info(f"Inicializados {len(self.models)} modelos específicos")
    
    def get_trained_models(self):
        """
        Obtiene los modelos entrenados.
        
        Returns:
            dict: Diccionario con los modelos entrenados
        """
        return self.models
    
    def get_processed_data(self):
        """
        Obtiene los datos procesados.
        
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        if self.merged_data is None:
            raise ValueError("Los datos no han sido cargados")
        return self.merged_data
    
    def load_and_prepare_data(self, regenerate_features: bool = False):
        """
        Carga y prepara todos los datos necesarios.
        
        Args:
            regenerate_features (bool): Si regenerar características desde cero
        """
        logger.info("Iniciando carga y preparación de datos")
        
        # Cargar datos
        self.data_loader = NBADataLoader(
            game_data_path=self.data_paths['game_data'],
            biometrics_path=self.data_paths['biometrics'],
            teams_path=self.data_paths['teams']
        )
        
        self.merged_data, teams_data = self.data_loader.load_data()
        logger.info(f"Datos cargados: {len(self.merged_data)} registros de jugadores")
        
        # Generar características
        if regenerate_features or not hasattr(self, 'feature_generator'):
            logger.info("Generando características específicas para NBA")
            self.feature_generator = PlayersFeatures(
                self.merged_data,
                window_sizes=[3, 5, 10, 20],
                enable_correlation_analysis=True
            )
            
            # Generar todas las características
            self.merged_data = self.feature_generator.generate_features()
            logger.info(f"Características generadas: {self.merged_data.shape[1]} columnas")
        
        # Validar que tenemos las columnas necesarias
        required_targets = ['PTS', 'TRB', 'AST', '3P', 'double_double', 'triple_double']
        missing_targets = [col for col in required_targets if col not in self.merged_data.columns]
        
        if missing_targets:
            logger.warning(f"Columnas objetivo faltantes: {missing_targets}")
        
        logger.info("Preparación de datos completada")
    
    def train_all_models(self, test_size: float = 0.2, use_time_split: bool = True) -> Dict:
        """
        Entrena todos los modelos específicos.
        
        Args:
            test_size (float): Proporción de datos para test
            use_time_split (bool): Si usar división temporal
            
        Returns:
            dict: Resultados del entrenamiento por modelo
        """
        if self.merged_data is None:
            raise ValueError("Datos no han sido cargados. Ejecutar load_and_prepare_data() primero")
        
        logger.info("Iniciando entrenamiento de todos los modelos")
        
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Entrenando modelo para {model_name}")
                
                # Preparar datos específicos para este modelo
                X_train, X_test, y_train, y_test = model.prepare_data(
                    self.merged_data, 
                    test_size=test_size, 
                    time_split=use_time_split
                )
                
                # Entrenar modelos
                model.train_models(X_train, y_train, use_scaling=True)
                
                # Validar modelos
                validation_scores = model.validate_models(X_test, y_test)
                
                # Obtener el mejor modelo
                best_model_name, best_model, best_score = model.get_best_model()
                
                # Guardar modelo
                model_path = os.path.join(self.output_dir, f"{model_name}_model.pkl")
                model.save_model(model_path, best_model_name)
                
                # Recopilar resultados
                training_results[model_name] = {
                    'best_model': best_model_name,
                    'best_score': best_score,
                    'validation_scores': validation_scores,
                    'features_count': len(model.feature_columns),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'model_path': model_path
                }
                
                logger.info(f"Modelo {model_name} entrenado - Mejor: {best_model_name} (Score: {best_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error entrenando modelo {model_name}: {str(e)}")
                training_results[model_name] = {'error': str(e)}
        
        # Guardar resumen de entrenamiento
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        self._save_training_summary(training_results, summary_path)
        
        logger.info("Entrenamiento de todos los modelos completado")
        return training_results
    
    def generate_predictions_report(self, player_name: str = "Anthony Edward´s", n_recent_games: int = 5) -> Dict:
        """
        Genera un reporte completo de predicciones para un jugador o análisis general.
        
        Args:
            player_name (str): Nombre del jugador específico (None para análisis general)
            n_recent_games (int): Número de juegos recientes a considerar
            
        Returns:
            dict: Reporte completo de predicciones y análisis
        """
        if self.merged_data is None:
            raise ValueError("Datos no han sido cargados")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_games': len(self.merged_data),
                'unique_players': self.merged_data['Player'].nunique() if 'Player' in self.merged_data.columns else 0,
                'date_range': (
                    self.merged_data['Date'].min().isoformat() if 'Date' in self.merged_data.columns else None,
                    self.merged_data['Date'].max().isoformat() if 'Date' in self.merged_data.columns else None
                )
            }
        }
        
        # Análisis específico por jugador
        if player_name:
            report['player_analysis'] = self._generate_player_analysis(player_name, n_recent_games)
        
        # Análisis general por cada modelo
        report['model_analysis'] = {}
        
        for model_name, model in self.models.items():
            try:
                analysis = {}
                
                if model_name == 'points':
                    analysis = model.analyze_scoring_patterns(self.merged_data)
                elif model_name == 'rebounds':
                    analysis = model.analyze_rebounding_patterns(self.merged_data)
                elif model_name == 'assists':
                    analysis = model.analyze_playmaking_patterns(self.merged_data)
                elif model_name == 'threes':
                    analysis = model.analyze_three_point_patterns(self.merged_data)
                elif model_name in ['double_double', 'triple_double']:
                    analysis = model.analyze_double_double_patterns(self.merged_data)
                
                # Añadir importancia de características
                if hasattr(model, 'feature_importance') and model.feature_importance:
                    importance_summary = model.get_feature_importance_summary()
                    if not importance_summary.empty:
                        analysis['top_features'] = importance_summary.head(10).to_dict('records')
                
                report['model_analysis'][model_name] = analysis
                
            except Exception as e:
                logger.error(f"Error generando análisis para {model_name}: {str(e)}")
                report['model_analysis'][model_name] = {'error': str(e)}
        
        return report
    
    def _generate_player_analysis(self, player_name: str, n_games: int) -> Dict:
        """
        Genera análisis específico para un jugador.
        
        Args:
            player_name (str): Nombre del jugador
            n_games (int): Número de juegos recientes
            
        Returns:
            dict: Análisis específico del jugador
        """
        player_data = self.merged_data[self.merged_data['Player'] == player_name].copy()
        
        if len(player_data) == 0:
            return {'error': f'Jugador {player_name} no encontrado'}
        
        analysis = {
            'player_name': player_name,
            'total_games': len(player_data),
            'recent_games_analyzed': min(n_games, len(player_data))
        }
        
        # Contexto por cada modelo
        for model_name, model in self.models.items():
            try:
                context = model.get_prediction_context(player_name, self.merged_data, n_games)
                analysis[f'{model_name}_context'] = context
            except Exception as e:
                logger.error(f"Error obteniendo contexto de {model_name} para {player_name}: {str(e)}")
                analysis[f'{model_name}_context'] = {'error': str(e)}
        
        return analysis
    
    def predict_for_player(self, player_name: str, upcoming_game_features: Dict) -> Dict:
        """
        Realiza predicciones para un jugador en un próximo partido.
        
        Args:
            player_name (str): Nombre del jugador
            upcoming_game_features (dict): Características del próximo partido
            
        Returns:
            dict: Predicciones para todas las estadísticas
        """
        predictions = {
            'player': player_name,
            'predictions': {},
            'confidence_scores': {}
        }
        
        # Crear DataFrame con las características del próximo partido
        # (Esta función necesitaría implementación adicional para preparar correctamente las características)
        
        for model_name, model in self.models.items():
            try:
                if not model.is_fitted:
                    logger.warning(f"Modelo {model_name} no ha sido entrenado")
                    continue
                
                # Para modelos de clasificación, obtener probabilidades
                if model.model_type == 'classification':
                    # pred, proba = model.predict_with_probability(X_features)
                    # predictions['predictions'][model_name] = pred[0]
                    # predictions['confidence_scores'][model_name] = proba[0][1]  # Probabilidad de clase positiva
                    pass  # Implementación pendiente
                else:
                    # pred = model.predict(X_features)
                    # predictions['predictions'][model_name] = pred[0]
                    pass  # Implementación pendiente
                    
            except Exception as e:
                logger.error(f"Error prediciendo {model_name} para {player_name}: {str(e)}")
                predictions['predictions'][model_name] = {'error': str(e)}
        
        return predictions
    
    def _save_training_summary(self, results: Dict, filepath: str):
        """
        Guarda un resumen del entrenamiento en formato JSON.
        
        Args:
            results (dict): Resultados del entrenamiento
            filepath (str): Ruta donde guardar el resumen
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': len(results),
            'successful_models': len([r for r in results.values() if 'error' not in r]),
            'failed_models': len([r for r in results.values() if 'error' in r]),
            'results': results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Resumen de entrenamiento guardado en {filepath}")
    
    def load_trained_models(self, models_dir: str = None):
        """
        Carga modelos previamente entrenados.
        
        Args:
            models_dir (str): Directorio con los modelos (None para usar output_dir)
        """
        models_dir = models_dir or self.output_dir
        
        for model_name in self.models.keys():
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            
            if os.path.exists(model_path):
                try:
                    self.models[model_name].load_model(model_path)
                    logger.info(f"Modelo {model_name} cargado desde {model_path}")
                except Exception as e:
                    logger.error(f"Error cargando modelo {model_name}: {str(e)}")
            else:
                logger.warning(f"Archivo de modelo no encontrado: {model_path}")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Genera una comparación de todos los modelos entrenados.
        
        Returns:
            pd.DataFrame: Comparación de métricas por modelo
        """
        comparison_data = []
        
        for model_name, model in self.models.items():
            if not model.is_fitted or not model.validation_scores:
                continue
            
            for algorithm, scores in model.validation_scores.items():
                if 'error' in scores:
                    continue
                
                row = {
                    'target': model_name,
                    'algorithm': algorithm,
                    'model_type': model.model_type,
                    'features_count': len(model.feature_columns)
                }
                
                # Añadir métricas específicas según el tipo de modelo
                if model.model_type == 'regression':
                    row.update({
                        'rmse': scores.get('rmse', None),
                        'mae': scores.get('mae', None),
                        'mse': scores.get('mse', None)
                    })
                else:  # classification
                    row.update({
                        'accuracy': scores.get('accuracy', None)
                    })
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def export_predictions_for_betting(self, output_file: str = None) -> str:
        """
        Exporta predicciones en formato optimizado para apuestas.
        
        Args:
            output_file (str): Archivo de salida (None para generar automáticamente)
            
        Returns:
            str: Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"betting_predictions_{timestamp}.json")
        
        # Generar predicciones para apuestas
        betting_data = {
            'generated_at': datetime.now().isoformat(),
            'models_info': {},
            'high_confidence_predictions': [],
            'player_recommendations': {}
        }
        
        # Información de modelos
        for model_name, model in self.models.items():
            if model.is_fitted:
                best_model_name, _, best_score = model.get_best_model()
                betting_data['models_info'][model_name] = {
                    'best_algorithm': best_model_name,
                    'score': best_score,
                    'features_used': len(model.feature_columns)
                }
        
        # Guardar archivo
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(betting_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Predicciones para apuestas exportadas a {output_file}")
        return output_file 