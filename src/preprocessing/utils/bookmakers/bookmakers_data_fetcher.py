import pandas as pd
import numpy as np
import requests
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BookmakersDataFetcher:
    """
    Clase para obtener y gestionar datos de odds de diferentes casas de apuestas.
    """
    
    def __init__(
        self,
        api_keys: Dict[str, str] = None,
        odds_data_dir: str = "data/bookmakers",
        cache_expiry: int = 12  # Horas
    ):
        self.api_keys = api_keys or {}
        self.odds_data_dir = Path(odds_data_dir)
        self.cache_expiry = cache_expiry
        
        # Asegurar que el directorio de datos existe
        self.odds_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Diccionario de casas de apuestas conocidas
        self.supported_bookmakers = {
            # Casas de apuestas principales
            'draftkings': 'DraftKings',
            'fanduel': 'FanDuel',
            'betmgm': 'BetMGM',
            'caesars': 'Caesars',
            'pointsbet': 'PointsBet',
            'bet365': 'Bet365',
            'wynn': 'Wynn',
            'pinnacle': 'Pinnacle',
            
            # Otras casas menos conocidas
            'barstool': 'Barstool',
            'unibet': 'Unibet',
            'foxbet': 'FOX Bet',
            'betrivers': 'BetRivers',
            'williamhill': 'William Hill',
            'twinspires': 'TwinSpires',
            'betfred': 'Betfred',
            'bovada': 'Bovada',
            'mybookie': 'MyBookie',
            'betonline': 'BetOnline'
        }

    def load_odds_from_api(
        self,
        sport: str = 'basketball_nba',
        markets: List[str] = None,
        bookmakers: List[str] = None,
        api_provider: str = 'odds_api',
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Carga odds desde un proveedor de API (The Odds API o similar)
        
        Args:
            sport: Deporte a consultar
            markets: Lista de mercados (h2h, spreads, totals, player_props)
            bookmakers: Lista de casas de apuestas a incluir
            api_provider: Proveedor de API ('odds_api', 'sportsdata_io', etc.)
            force_refresh: Si forzar la actualización ignorando caché
            
        Returns:
            Datos de odds en formato JSON
        """
        # Mercados por defecto si no se especifican
        if markets is None:
            markets = ['h2h', 'spreads', 'totals', 'player_props']
            
        # Casas de apuestas por defecto
        if bookmakers is None:
            bookmakers = list(self.supported_bookmakers.keys())
        
        # Archivo de caché para esta consulta
        cache_key = f"{sport}_{'-'.join(markets)}_{'-'.join(sorted(bookmakers))}"
        cache_file = self.odds_data_dir / f"{cache_key}.json"
        
        # Verificar si tenemos datos en caché y son recientes
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verificar si los datos son recientes
                last_update = datetime.fromisoformat(cached_data.get('last_update', '2000-01-01'))
                if datetime.now() - last_update < timedelta(hours=self.cache_expiry):
                    logger.info(f"Usando datos de odds en caché para {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error al leer caché: {e}")
        
        # Si llegamos aquí, necesitamos obtener nuevos datos
        api_key = self.api_keys.get(api_provider)
        if not api_key:
            logger.error(f"No se encontró API key para {api_provider}")
            return {'success': False, 'error': f"API key no configurada para {api_provider}"}
            
        # Diferentes implementaciones según el proveedor
        if api_provider == 'odds_api':
            odds_data = self._fetch_from_odds_api(api_key, sport, markets, bookmakers)
        elif api_provider == 'sportsdata_io':
            odds_data = self._fetch_from_sportsdata_io(api_key, sport, markets, bookmakers)
        else:
            logger.error(f"Proveedor de API no soportado: {api_provider}")
            return {'success': False, 'error': f"Proveedor no soportado: {api_provider}"}
        
        # Verificar si fue exitoso
        if odds_data.get('success', False):
            # Añadir timestamp
            odds_data['last_update'] = datetime.now().isoformat()
            
            # Guardar en caché
            try:
                with open(cache_file, 'w') as f:
                    json.dump(odds_data, f)
                logger.info(f"Datos de odds guardados en caché para {cache_key}")
            except Exception as e:
                logger.warning(f"Error al guardar caché: {e}")
        
        return odds_data
    
    def _fetch_from_odds_api(
        self, 
        api_key: str, 
        sport: str, 
        markets: List[str], 
        bookmakers: List[str]
    ) -> Dict[str, Any]:
        """
        Obtiene datos desde The Odds API
        """
        base_url = "https://api.the-odds-api.com/v4/sports"
        all_data = {'success': True, 'data': []}
        
        for market in markets:
            try:
                # Diferentes endpoints según el mercado
                if market == 'player_props':
                    url = f"{base_url}/{sport}/odds?apiKey={api_key}&markets=player_props&bookmakers={','.join(bookmakers)}"
                else:
                    url = f"{base_url}/{sport}/odds?apiKey={api_key}&markets={market}&bookmakers={','.join(bookmakers)}"
                
                logger.info(f"Consultando {market} odds para {sport}")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    market_data = response.json()
                    all_data['data'].extend(market_data)
                    
                    # Respetar límites de API
                    remaining = response.headers.get('X-Remaining-Requests', 0)
                    logger.info(f"Solicitudes restantes: {remaining}")
                    
                    if int(remaining) < 1:
                        logger.warning("Límite de API alcanzado. Algunas odds podrían faltar.")
                        break
                else:
                    logger.error(f"Error al obtener odds: {response.status_code} - {response.text}")
                    
                # Esperar para no saturar la API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error consultando {market} odds: {str(e)}")
                all_data['success'] = False
                all_data['error'] = str(e)
        
        return all_data
    
    def _fetch_from_sportsdata_io(
        self, 
        api_key: str, 
        sport: str, 
        markets: List[str], 
        bookmakers: List[str]
    ) -> Dict[str, Any]:
        """
        Obtiene datos desde SportsData.io
        """
        # Implementación específica para SportsData.io
        # ...
        
        # Placeholder
        return {'success': False, 'error': 'SportsData.io no implementado aún'}
    
    def load_odds_from_file(
        self,
        file_path: str,
        format: str = 'csv'
    ) -> Dict[str, Any]:
        """
        Carga datos de odds desde un archivo local
        
        Args:
            file_path: Ruta al archivo de datos
            format: Formato del archivo ('csv', 'json', 'excel')
            
        Returns:
            Dict con datos de odds
        """
        try:
            if format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif format.lower() == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif format.lower() in ['excel', 'xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Formato no soportado: {format}")
                return {'success': False, 'error': f"Formato no soportado: {format}"}
            
            # Convertir DataFrame a formato estándar
            data = self._standardize_odds_data(df)
            return {'success': True, 'data': data, 'source': 'file', 'file_path': file_path}
            
        except Exception as e:
            logger.error(f"Error al cargar archivo {file_path}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _standardize_odds_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convierte un DataFrame a formato estándar para odds
        """
        # Aquí se implementaría la lógica de estandarización
        # Placeholder
        return df.to_dict(orient='records')
        
    def merge_odds_with_player_data(
        self,
        player_data: pd.DataFrame,
        odds_data: Dict[str, Any],
        target: str = 'PTS',
        line_values: List[float] = None
    ) -> pd.DataFrame:
        """
        Combina datos de jugadores con odds de casas de apuestas
        
        Args:
            player_data: DataFrame con datos de jugadores
            odds_data: Datos de odds (de API o archivo)
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Valores de línea específicos a buscar
            
        Returns:
            DataFrame con datos combinados
        """
        if not odds_data.get('success', False) or not odds_data.get('data'):
            logger.error("No hay datos de odds válidos para combinar")
            return player_data
        
        # Crear copia para no modificar el original
        df = player_data.copy()
        
        # Líneas por defecto según el target si no se especifican
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
        
        # Procesar datos de odds
        try:
            # Diferentes procesamiento según el formato de datos
            if 'source' in odds_data and odds_data['source'] == 'file':
                df = self._merge_file_odds(df, odds_data['data'], target, line_values)
            else:
                # Asumir formato API
                df = self._merge_api_odds(df, odds_data['data'], target, line_values)
            
            return df
            
        except Exception as e:
            logger.error(f"Error al combinar datos de odds: {str(e)}")
            return player_data
    
    def _merge_file_odds(
        self,
        df: pd.DataFrame,
        odds_data: List[Dict[str, Any]],
        target: str,
        line_values: List[float]
    ) -> pd.DataFrame:
        """
        Combina datos de un archivo de odds con datos de jugadores
        """
        # Implementación específica para archivos
        # ...
        
        # Placeholder
        return df
    
    def _merge_api_odds(
        self,
        df: pd.DataFrame,
        odds_data: List[Dict[str, Any]],
        target: str,
        line_values: List[float]
    ) -> pd.DataFrame:
        """
        Combina datos de API de odds con datos de jugadores
        """
        # Implementación específica para datos de API
        # ...
        
        # Placeholder
        return df
    
    def extract_player_props(
        self,
        odds_data: Dict[str, Any],
        player_name: Optional[str] = None,
        team_name: Optional[str] = None,
        prop_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrae props específicas de jugador desde datos de odds completos
        
        Args:
            odds_data: Datos de odds completos
            player_name: Nombre de jugador para filtrar (opcional)
            team_name: Nombre de equipo para filtrar (opcional)
            prop_type: Tipo de prop para filtrar (points, rebounds, assists, threes)
            
        Returns:
            Lista filtrada de props de jugador
        """
        if not odds_data.get('success', False) or not odds_data.get('data'):
            return []
        
        props = []
        
        # Mapeo de prop_type a representaciones comunes en las APIs
        prop_type_map = {
            'points': ['points', 'pts', 'player points', 'total points'],
            'rebounds': ['rebounds', 'reb', 'player rebounds', 'total rebounds', 'trb'],
            'assists': ['assists', 'ast', 'player assists', 'total assists'],
            'threes': ['threes', '3pt', 'three pointers', '3 pointers', '3p']
        }
        
        # Término de búsqueda basado en prop_type
        search_terms = []
        if prop_type and prop_type.lower() in prop_type_map:
            search_terms = prop_type_map[prop_type.lower()]
        
        # Procesar datos
        try:
            for game in odds_data['data']:
                # Verificar si el juego tiene datos de player_props
                if 'bookmakers' not in game:
                    continue
                
                for bookmaker in game['bookmakers']:
                    bookmaker_name = bookmaker['key']
                    
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key', '')
                        
                        # Verificar si es un prop de jugador
                        if 'player' not in market_key.lower():
                            continue
                        
                        # Si se especificó prop_type, verificar que coincida
                        if prop_type and not any(term in market_key.lower() for term in search_terms):
                            continue
                        
                        for outcome in market.get('outcomes', []):
                            # Verificar filtros
                            if player_name and player_name.lower() not in outcome.get('name', '').lower():
                                continue
                                
                            if team_name and team_name.lower() not in outcome.get('team', '').lower():
                                continue
                            
                            # Añadir a resultados
                            props.append({
                                'player': outcome.get('name', ''),
                                'team': outcome.get('team', ''),
                                'prop_type': market_key,
                                'line': outcome.get('point'),
                                'over_price': outcome.get('price') if outcome.get('name', '').lower().endswith('over') else None,
                                'under_price': outcome.get('price') if outcome.get('name', '').lower().endswith('under') else None,
                                'bookmaker': bookmaker_name,
                                'game_time': game.get('commence_time'),
                                'home_team': game.get('home_team'),
                                'away_team': game.get('away_team')
                            })
            
            return props
            
        except Exception as e:
            logger.error(f"Error al extraer props de jugador: {str(e)}")
            return []
    
    def create_odds_columns(
        self,
        df: pd.DataFrame,
        props_data: List[Dict[str, Any]],
        target: str = 'PTS'
    ) -> pd.DataFrame:
        """
        Crea columnas con odds para cada bookmaker en el DataFrame
        
        Args:
            df: DataFrame con datos de jugadores
            props_data: Lista de props extraídas
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            
        Returns:
            DataFrame con columnas de odds añadidas
        """
        # Mapeo de targets a tipos de props
        target_to_prop = {
            'PTS': ['points', 'pts', 'player points', 'total points'],
            'TRB': ['rebounds', 'reb', 'player rebounds', 'total rebounds', 'trb'],
            'AST': ['assists', 'ast', 'player assists', 'total assists'],
            '3P': ['threes', '3pt', 'three pointers', '3 pointers', '3p']
        }
        
        # Asegurar que tenemos prop_type para el target
        if target not in target_to_prop:
            logger.error(f"Target no soportado para odds: {target}")
            return df
            
        prop_types = target_to_prop[target]
        
        # Crear copia para no modificar el original
        result_df = df.copy()
        
        # Agrupar props por jugador
        player_props = {}
        for prop in props_data:
            # Verificar si es el tipo correcto de prop
            if not any(pt in prop['prop_type'].lower() for pt in prop_types):
                continue
                
            player = prop['player']
            if player not in player_props:
                player_props[player] = []
            
            player_props[player].append(prop)
        
        # Para cada jugador en el DataFrame
        for idx, row in result_df.iterrows():
            player_name = row['Player'] if 'Player' in row else None
            if not player_name or player_name not in player_props:
                continue
            
            # Procesar props para este jugador
            player_odds = player_props[player_name]
            
            # Para cada línea y bookmaker
            for prop in player_odds:
                line = prop.get('line')
                if line is None:
                    continue
                    
                bookmaker = prop.get('bookmaker', 'unknown').lower()
                over_price = prop.get('over_price')
                under_price = prop.get('under_price')
                
                # Crear nombres de columna
                if over_price is not None:
                    col_name = f"{target}_over_{line}_odds_{bookmaker}"
                    result_df.at[idx, col_name] = over_price
                
                if under_price is not None:
                    col_name = f"{target}_under_{line}_odds_{bookmaker}"
                    result_df.at[idx, col_name] = under_price
        
        return result_df
    
    def simulate_bookmaker_data(
        self,
        df: pd.DataFrame,
        target: str = 'PTS',
        line_values: List[float] = None,
        bookmakers: List[str] = None,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Genera datos de odds simulados para pruebas cuando no hay datos reales disponibles
        
        Args:
            df: DataFrame con datos de jugadores
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Valores de línea específicos a simular
            bookmakers: Lista de casas de apuestas a simular
            noise_level: Nivel de ruido para variación entre casas
            
        Returns:
            DataFrame con columnas de odds simuladas
        """
        # Valores por defecto
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
                
        if bookmakers is None:
            bookmakers = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']
        
        # Crear copia para no modificar el original
        result_df = df.copy()
        
        # Columna target binaria para cada línea
        for line in line_values:
            line_col = f"{target}_over_{line}"
            
            # Si la columna no existe, crearla
            if line_col not in result_df.columns:
                try:
                    result_df[line_col] = (result_df[target] > line).astype(int)
                except:
                    logger.warning(f"No se pudo crear columna {line_col}")
                    continue
        
        # Para cada jugador, línea y bookmaker, generar odds
        for idx, row in result_df.iterrows():
            for line in line_values:
                line_col = f"{target}_over_{line}"
                
                # Saltarse si no tenemos la columna
                if line_col not in result_df.columns:
                    continue
                
                # Obtener probabilidad real (usamos datos históricos)
                try:
                    # Verificar si tenemos columna de probabilidad histórica
                    prob_col = f"{line_col}_prob_10"  # Probabilidad en 10 juegos
                    
                    if prob_col in result_df.columns:
                        true_prob = result_df.at[idx, prob_col]
                    else:
                        # Si no tenemos histórico, usar un valor aleatorio
                        true_prob = np.random.uniform(0.3, 0.7)
                    
                    # Generar odds para cada bookmaker con ruido
                    for bm in bookmakers:
                        # Añadir ruido específico de bookmaker
                        bm_noise = np.random.normal(0, noise_level)
                        implied_prob = np.clip(true_prob + bm_noise, 0.05, 0.95)
                        
                        # Convertir a precio europeo (1/p)
                        over_price = round(1 / implied_prob, 2)
                        under_price = round(1 / (1 - implied_prob), 2)
                        
                        # Guardar en DataFrame
                        result_df.at[idx, f"{target}_over_{line}_odds_{bm}"] = over_price
                        result_df.at[idx, f"{target}_under_{line}_odds_{bm}"] = under_price
                        
                except Exception as e:
                    logger.warning(f"Error al simular odds para idx {idx}, line {line}: {str(e)}")
        
        return result_df
        
    def compare_bookmaker_odds(
        self,
        df: pd.DataFrame,
        target: str = 'PTS',
        line_values: List[float] = None,
        min_samples: int = 5
    ) -> pd.DataFrame:
        """
        Analiza las diferencias entre odds de distintas casas de apuestas
        
        Args:
            df: DataFrame con datos de jugadores incluyendo odds
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Valores de línea específicos a analizar
            min_samples: Mínimo de muestras para incluir un bookmaker
            
        Returns:
            DataFrame con análisis de diferencias
        """
        # Valores por defecto
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
        
        # Resultados
        comparison_data = []
        
        # Para cada línea
        for line in line_values:
            # Detectar columnas de odds para esta línea
            over_cols = [col for col in df.columns 
                        if f"{target}_over_{line}_odds_" in col]
            
            under_cols = [col for col in df.columns 
                        if f"{target}_under_{line}_odds_" in col]
            
            if not over_cols and not under_cols:
                logger.info(f"No se encontraron columnas de odds para {target} línea {line}")
                continue
            
            # Calcular estadísticas para OVER
            over_stats = {}
            if over_cols:
                for col in over_cols:
                    # Extraer nombre de bookmaker
                    bm = col.split(f"{target}_over_{line}_odds_")[1]
                    
                    # Calcular estadísticas
                    values = df[col].dropna()
                    if len(values) >= min_samples:
                        over_stats[bm] = {
                            'count': len(values),
                            'mean': values.mean(),
                            'min': values.min(),
                            'max': values.max(),
                            'std': values.std(),
                            'implied_prob': 1 / values.mean()
                        }
            
            # Calcular estadísticas para UNDER
            under_stats = {}
            if under_cols:
                for col in under_cols:
                    # Extraer nombre de bookmaker
                    bm = col.split(f"{target}_under_{line}_odds_")[1]
                    
                    # Calcular estadísticas
                    values = df[col].dropna()
                    if len(values) >= min_samples:
                        under_stats[bm] = {
                            'count': len(values),
                            'mean': values.mean(),
                            'min': values.min(),
                            'max': values.max(),
                            'std': values.std(),
                            'implied_prob': 1 / values.mean()
                        }
            
            # Encontrar mejores casas
            best_over_bm = None
            best_over_odds = 0
            
            for bm, stats in over_stats.items():
                if stats['mean'] > best_over_odds:
                    best_over_odds = stats['mean']
                    best_over_bm = bm
            
            best_under_bm = None
            best_under_odds = 0
            
            for bm, stats in under_stats.items():
                if stats['mean'] > best_under_odds:
                    best_under_odds = stats['mean']
                    best_under_bm = bm
            
            # Guardar análisis
            comparison_data.append({
                'target': target,
                'line': line,
                'over_bookmakers': len(over_stats),
                'under_bookmakers': len(under_stats),
                'best_over_bookmaker': best_over_bm,
                'best_over_odds': best_over_odds,
                'best_under_bookmaker': best_under_bm,
                'best_under_odds': best_under_odds,
                'over_stats': over_stats,
                'under_stats': under_stats,
                'juice': sum(1/stats['mean'] for bm, stats in over_stats.items()) / len(over_stats) + 
                         sum(1/stats['mean'] for bm, stats in under_stats.items()) / len(under_stats) if over_stats and under_stats else None
            })
        
        return pd.DataFrame(comparison_data) 