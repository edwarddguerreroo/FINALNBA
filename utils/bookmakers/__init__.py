"""
Módulo de integración con casas de apuestas
-------------------------------------------
Este módulo proporciona funcionalidades para obtener, analizar y procesar datos 
de odds (cuotas) de casas de apuestas para identificar oportunidades de valor.
Permite la integración con APIs de bookmakers y el análisis comparativo entre nuestras 
predicciones y las probabilidades implícitas en las cuotas del mercado.
"""

from .bookmakers_data_fetcher import BookmakersDataFetcher

__all__ = ['BookmakersDataFetcher']