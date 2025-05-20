"""
Módulo de generación de secuencias para el sistema de predicción de NBA

Este módulo contiene las herramientas necesarias para:
- Generar secuencias de características a partir de datos históricos
- Preparar datasets para entrenamiento de modelos predictivos
- Analizar y seleccionar líneas de apuestas óptimas
"""

from .sequences_generator import (
    SequenceGenerator,
    NBASequenceDataset,
    NBASequenceDatasetWithLines,
    create_data_loaders,
    create_data_loaders_from_splits,
    save_sequences,
    load_sequences,
    prepare_target_specific_sequences,
    prepare_all_target_sequences
)

__all__ = [
    'SequenceGenerator',
    'NBASequenceDataset',
    'NBASequenceDatasetWithLines',
    'create_data_loaders',
    'create_data_loaders_from_splits',
    'save_sequences',
    'load_sequences',
    'prepare_target_specific_sequences',
    'prepare_all_target_sequences'
] 