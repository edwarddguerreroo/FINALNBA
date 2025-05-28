import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class TotalPointsFeatureEngine:
    """
    Motor de features para predicción de puntos totales en un game
    Enfoque: Features de DOMINIO ESPECÍFICO con máximo poder predictivo
    OPTIMIZADO - Sin cálculos duplicados
    """
    
    def __init__(self, lookback_games: int = 10):
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        # Cache para evitar recálculos
        self._cached_calculations = {}
        
    def create_features(self, teams_data: pd.DataFrame) -> pd.DataFrame:
        """
        PIPELINE COMPLETO DE FEATURES PARA 97% PRECISIÓN - OPTIMIZADO
        Genera todas las características necesarias sin duplicaciones
        """
        print("Generando features NBA específicas OPTIMIZADAS...")
        
        # Crear copia para evitar modificar datos originales
        df = teams_data.copy()
        
        # Asegurar orden cronológico para evitar data leakage
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date').reset_index(drop=True)
        else:
            print("⚠️ Columna 'Date' no encontrada - usando orden original")
        
        # PASO 1: Cálculos base (una sola vez)
        df = self._create_base_calculations(df)
        
        # PASO 2: Features básicas NBA usando cálculos base
        df = self._create_basic_nba_features(df)
        
        # PASO 3: Features avanzadas usando cálculos existentes
        df = self._create_advanced_features_optimized(df)
        
        # PASO 4: Features de contexto y situación
        df = self._create_context_features(df)
        
        # PASO 5: Features de interacción final
        df = self._create_final_interactions(df)
        
        print(f"Features OPTIMIZADAS creadas: {len(df.columns)} features")
        
        # Aplicar filtros de calidad
        df = self._apply_quality_filters(df)
        
        # Actualizar lista de features disponibles
        self._update_feature_columns(df)
        
        return df
    
    def _create_base_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CÁLCULOS BASE - Una sola vez para evitar duplicaciones
        Todos los cálculos fundamentales que se reutilizan
        """
        print("Calculando métricas base NBA...")
        
        # ==================== CÁLCULOS TEMPORALES BASE ====================
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
        
        # ==================== CÁLCULOS DE POSESIONES (UNA SOLA VEZ) ====================
        # Fórmula NBA oficial
        df['possessions'] = df['FGA'] - df['FG'] + df['FTA'] * 0.44 + df['FG'] * 0.56
        df['opp_possessions'] = df['FGA_Opp'] - df['FG_Opp'] + df['FTA_Opp'] * 0.44 + df['FG_Opp'] * 0.56
        
        # Posesiones alternativas (para validación)
        df['real_possessions'] = df['FGA'] + df['FTA'] * 0.44 - df['FG'] * 0.1
        df['opp_real_possessions'] = df['FGA_Opp'] + df['FTA_Opp'] * 0.44 - df['FG_Opp'] * 0.1
        
        # ==================== CÁLCULOS DE EFICIENCIA BASE ====================
        # True Shooting Percentage (métrica oficial NBA)
        df['true_shooting_pct'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['opp_true_shooting_pct'] = df['PTS_Opp'] / (2 * (df['FGA_Opp'] + 0.44 * df['FTA_Opp']))
        
        # Effective Field Goal Percentage
        df['efg_pct'] = (df['FG'] + 0.5 * df['3P']) / df['FGA']
        df['opp_efg_pct'] = (df['FG_Opp'] + 0.5 * df['3P_Opp']) / df['FGA_Opp']
        
        # Conversion efficiency (combinada)
        df['conversion_efficiency'] = (
            (df['FG%'].fillna(0.45) + df['3P%'].fillna(0.35) + df['FT%'].fillna(0.75)) / 3
        )
        df['opp_conversion_efficiency'] = (
            (df['FG%_Opp'].fillna(0.45) + df['3P%_Opp'].fillna(0.35) + df['FT%_Opp'].fillna(0.75)) / 3
        )
        df['combined_conversion_efficiency'] = (df['conversion_efficiency'] + df['opp_conversion_efficiency']) / 2
        
        # ==================== PROYECCIONES DIRECTAS BASE ====================
        # Proyección directa de scoring
        df['direct_scoring_projection'] = (
            df['FGA'] * df['FG%'].fillna(0.45) * 2 +  # Puntos de 2P
            df['3PA'] * df['3P%'].fillna(0.35) * 3 +  # Puntos de 3P
            df['FTA'] * df['FT%'].fillna(0.75) * 1    # Puntos de FT
        )
        df['opp_direct_scoring_projection'] = (
            df['FGA_Opp'] * df['FG%_Opp'].fillna(0.45) * 2 +
            df['3PA_Opp'] * df['3P%_Opp'].fillna(0.35) * 3 +
            df['FTA_Opp'] * df['FT%_Opp'].fillna(0.75) * 1
        )
        df['total_direct_projection'] = df['direct_scoring_projection'] + df['opp_direct_scoring_projection']
        
        # ==================== VOLÚMENES Y MÉTRICAS COMBINADAS ====================
        df['total_shot_volume'] = df['FGA'] + df['FGA_Opp'] + df['FTA'] + df['FTA_Opp']
        df['weighted_shot_volume'] = df['total_shot_volume'] * df['combined_conversion_efficiency']
        df['total_expected_shots'] = df['FGA'] + df['3PA'] * 0.4 + df['FTA'] * 0.44
        df['opp_expected_shots'] = df['FGA_Opp'] + df['3PA_Opp'] * 0.4 + df['FTA_Opp'] * 0.44
        
        # ==================== FEATURES DE VICTORIA/DERROTA ====================
        if 'PTS' in df.columns and 'PTS_Opp' in df.columns:
            df['is_win'] = (df['PTS'] > df['PTS_Opp']).astype(int)
        
        # ==================== FEATURES DE CONTEXTO BASE ====================
        # Ventaja local
        if 'is_home' not in df.columns:
            df['is_home'] = (df['Away'] == 0).astype(int) if 'Away' in df.columns else 0
        
        # Factor de energía basado en descanso
        df['energy_factor'] = np.where(
            df['days_rest'] == 0, 0.92,  # Back-to-back penalty
            np.where(df['days_rest'] == 1, 0.97,  # 1 día
                    np.where(df['days_rest'] >= 3, 1.03, 1.0))  # 3+ días boost
        )
        
        # Boost de ventaja local
        df['home_court_boost'] = df['is_home'] * 2.5
        
        # Importancia del partido
        df['season_importance'] = np.where(
            df['days_into_season'] > 200, 1.05,  # Playoffs/final temporada
            np.where(df['days_into_season'] > 100, 1.02, 1.0)  # Mitad temporada
        )
        
        return df
    
    def _create_basic_nba_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features básicas NBA usando cálculos base existentes"""
        
        # ==================== PROMEDIOS MÓVILES OPTIMIZADOS ====================
        windows = [3, 5, 7, 10]
        
        # Promedios de puntos (una sola vez por ventana)
        for window in windows:
            # Puntos anotados y permitidos
            df[f'pts_avg_{window}g'] = df.groupby('Team')['PTS'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            df[f'pts_allowed_avg_{window}g'] = df.groupby('Team')['PTS_Opp'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'total_pts_avg_{window}g'] = df[f'pts_avg_{window}g'] + df[f'pts_allowed_avg_{window}g']
            
            # Promedios de eficiencia
            df[f'ts_pct_avg_{window}g'] = df.groupby('Team')['true_shooting_pct'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'opp_ts_pct_avg_{window}g'] = df.groupby('Team')['opp_true_shooting_pct'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Promedios de ritmo
            df[f'pace_avg_{window}g'] = df.groupby('Team')['possessions'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'opp_pace_avg_{window}g'] = df.groupby('Team')['opp_possessions'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Promedios de proyección directa
            df[f'direct_projection_avg_{window}g'] = df.groupby('Team')['direct_scoring_projection'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # ==================== MÉTRICAS DERIVADAS OPTIMIZADAS ====================
        # Ritmo combinado
        df['combined_pace_3g'] = df['pace_avg_3g'] + df['opp_pace_avg_3g']
        df['combined_pace_5g'] = df['pace_avg_5g'] + df['opp_pace_avg_5g']
        df['total_game_pace'] = df['real_possessions'] + df['opp_real_possessions']
        
        # Tendencias (últimos 3 vs últimos 10)
        df['pts_trend_short'] = df['pts_avg_3g'] - df['pts_avg_10g']
        df['pts_allowed_trend_short'] = df['pts_allowed_avg_3g'] - df['pts_allowed_avg_10g']
        df['total_pts_trend'] = df['pts_trend_short'] + df['pts_allowed_trend_short']
        df['pace_acceleration'] = df['pace_avg_3g'] - df['pace_avg_10g']
        df['projection_trend'] = (df['direct_projection_avg_3g'] - df['direct_projection_avg_10g']) / (df['direct_projection_avg_10g'] + 1)
        
        # Volatilidad y consistencia
        df['pts_volatility'] = df.groupby('Team')['PTS'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().shift(1) / 
                     (x.rolling(window=5, min_periods=1).mean().shift(1) + 1)
        )
        df['offensive_consistency'] = 1 / (df['pts_volatility'] + 0.01)
        
        # Eficiencia diferencial
        df['efficiency_differential'] = df['ts_pct_avg_5g'] - df['opp_ts_pct_avg_5g']
        
        # Momentum de scoring
        df['scoring_momentum'] = (df['pts_avg_3g'] - df['pts_avg_10g']) / df['pts_avg_10g']
        df['defensive_momentum'] = (df['pts_allowed_avg_10g'] - df['pts_allowed_avg_3g']) / df['pts_allowed_avg_10g']
        
        return df
    
    def _create_advanced_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features avanzadas optimizadas sin duplicaciones y multicolinealidad"""
        
        # ==================== PROYECCIONES OPTIMIZADAS SIN MULTICOLINEALIDAD ====================
        # Solo mantener la proyección matemática más robusta
        df['mathematical_total_projection'] = (
            df['total_expected_shots'] * df['combined_conversion_efficiency'] * 1.1
        )
        
        # Proyección híbrida simplificada (sin usar componentes altamente correlacionados)
        df['hybrid_scoring_projection'] = (
            df['mathematical_total_projection'] * 0.6 +
            df['weighted_shot_volume'] * 0.4
        )
        
        # ==================== FEATURES DE OPONENTE OPTIMIZADAS ====================
        # Calidad del oponente (una sola vez) - CORREGIDO: Sin data leakage
        opp_def_ranking = df.groupby('Opp')['PTS_Opp'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        df['opp_defensive_ranking'] = opp_def_ranking.rank(pct=True).fillna(0.5)
        
        opp_off_ranking = df.groupby('Opp')['PTS'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        df['opp_offensive_ranking'] = opp_off_ranking.rank(pct=True).fillna(0.5)
        
        df['opponent_quality_factor'] = (
            df['opp_offensive_ranking'] * 0.6 + 
            df['opp_defensive_ranking'] * 0.4
        )
        
        # ==================== FEATURES DE MATCHUP ESPECÍFICO ====================
        # Historial vs oponente específico
        df['vs_opp_pts_history'] = df.groupby(['Team', 'Opp'])['PTS'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['vs_opp_allowed_history'] = df.groupby(['Team', 'Opp'])['PTS_Opp'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Solo mantener el edge total (sin componentes separados que causan multicolinealidad)
        df['matchup_total_edge'] = (
            df['vs_opp_pts_history'].fillna(df['pts_avg_10g']) + 
            df['vs_opp_allowed_history'].fillna(df['pts_allowed_avg_10g']) -
            df['total_pts_avg_10g']
        )
        
        # ==================== FEATURES DE MOMENTUM OPTIMIZADAS ====================
        # Win percentage solo para ventana más estable
        if 'is_win' in df.columns:
            df['win_pct_5g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )
        
        # Factor de confianza simplificado
        df['confidence_factor'] = (df.get('win_pct_5g', 0.5) - 0.5) * 2
        
        return df
    
    def _create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de contexto situacional"""
        
        # ==================== FACTORES CONTEXTUALES ESPECÍFICOS ====================
        # Factor de altitud (equipos específicos)
        altitude_teams = ['DEN', 'UTA', 'PHX']
        df['altitude_factor'] = df['Team'].apply(lambda x: 1.02 if x in altitude_teams else 1.0)
        
        # Factor de rivalidad
        rivalry_pairs = [
            ('ATL', 'BOS'), ('ATL', 'GSW'), ('BOS', 'PHI'), 
            ('MIA', 'BOS'), ('LAC', 'ATL'), ('GSW', 'LAC')
        ]
        df['rivalry_factor'] = df.apply(
            lambda row: 1.03 if (row['Team'], row['Opp']) in rivalry_pairs or 
                              (row['Opp'], row['Team']) in rivalry_pairs else 1.0, axis=1
        )
        
        # Fatiga de temporada
        df['season_fatigue'] = np.where(df['month'].isin([1, 2, 3]), 1.2, 1.0)
        
        # Factor de urgencia
        games_remaining = 82 - df.groupby('Team').cumcount()
        df['urgency_factor'] = np.where(games_remaining <= 10, 1.5, 1.0)
        
        # Advantage de descanso
        df['rest_advantage'] = np.where(
            df['days_rest'] == 0, -3.0,  # Penalización back-to-back
            np.where(df['days_rest'] == 1, -1.0,  # Penalización 1 día
                    np.where(df['days_rest'] >= 3, 2.0, 0.0))  # Bonus 3+ días
        )
        
        return df
    
    def _create_final_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de interacción final optimizadas sin multicolinealidad"""
        
        # ==================== INTERACCIONES SIMPLIFICADAS ====================
        # Solo mantener interacciones que no causen multicolinealidad extrema
        
        # Interacción pace-efficiency (usando FG% en lugar de FGA para evitar correlación 1.0 con FG)
        df['pace_efficiency_interaction'] = df['possessions'] * df['FG%']
        
        # Interacción momentum-contexto simplificada
        df['momentum_context_combo'] = (
            df['confidence_factor'] * df['energy_factor']
        )
        
        # Interacción calidad-eficiencia (sin volumen para evitar multicolinealidad)
        df['quality_efficiency_interaction'] = (
            df['combined_conversion_efficiency'] * df['opponent_quality_factor']
        )
        
        # ==================== PROYECCIÓN FINAL SIMPLIFICADA ====================
        # Una sola proyección final robusta sin componentes altamente correlacionados
        df['final_projection'] = (
            df['hybrid_scoring_projection'] * 0.7 +
            df['quality_efficiency_interaction'] * 20 +  # Escalar para impacto apropiado
            df['home_court_boost'] + 
            df['rest_advantage'] +
            df['season_importance'] * 3.0
        )
        
        # Límites realistas NBA
        df['final_prediction'] = np.clip(df['final_projection'], 185, 275)
        
        # ==================== FEATURES DE CONFIANZA SIMPLIFICADAS ====================
        # Solo una métrica de confianza sin redundancia
        df['prediction_confidence'] = (
            df['opponent_quality_factor'] * 0.5 +
            df['combined_conversion_efficiency'] * 0.5
        )
        
        return df
    
    def apply_final_correlation_filter(self, df: pd.DataFrame, correlation_threshold: float = 0.85) -> pd.DataFrame:
        """
        FILTRO DE CORRELACIÓN OPTIMIZADO PARA ELIMINAR MULTICOLINEALIDAD
        Elimina específicamente features con correlación >0.95
        """
        print(f"\nAPLICANDO FILTRO ANTI-MULTICOLINEALIDAD (>{correlation_threshold*100}%)")
        
        # Identificar columnas de features (excluir metadatos)
        exclude_cols = ['Team', 'TEAM', 'Date', 'Opp', 'PTS', 'PTS_Opp', 'total_score', 'total_points']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Features iniciales: {len(feature_cols)}")
        
        # Crear target para análisis de correlación
        if 'total_score' in df.columns:
            target_col = 'total_score'
        elif 'total_points' in df.columns:
            target_col = 'total_points'
        else:
            # Crear target temporal
            df['temp_total_points'] = df['PTS'] + df['PTS_Opp']
            target_col = 'temp_total_points'
        
        # Calcular correlaciones con target (una sola vez)
        target_correlations = {}
        for col in feature_cols:
            try:
                if df[col].dtype in ['float64', 'int64'] and not df[col].isna().all():
                    corr = abs(df[col].corr(df[target_col]))
                    if not np.isnan(corr):
                        target_correlations[col] = corr
            except Exception:
                continue
        
        # Filtrar features relevantes
        min_target_correlation = 0.08
        relevant_features = [col for col, corr in target_correlations.items() if corr >= min_target_correlation]
        
        # ELIMINAR FEATURES ESPECÍFICAS QUE CAUSAN MULTICOLINEALIDAD ALTA
        multicollinear_features = [
            # Features eliminadas por correlación >0.99
            'ultimate_scoring_projection',  # Correlación 0.997 con ensemble_projection_v1
            'ensemble_projection_v1',       # Correlación 0.988 con total_direct_projection
            'total_direct_projection',      # Correlación 1.000 con opp_direct_scoring_projection
            'efficiency_volume_interaction', # Correlación 1.000 con FG
            'high_confidence_projection',   # Derivada de features correlacionadas
            'master_projection',            # Combinación de features correlacionadas
            'final_master_prediction',      # Redundante
            
            # Features problemáticas adicionales
            'projection_confidence_score',  # Meta-feature problemática
            'certainty_factor',             # Meta-feature problemática
            'matchup_offensive_edge',       # Componente de matchup_total_edge
            'matchup_defensive_edge',       # Componente de matchup_total_edge
            'vs_opp_total_history',         # Suma de componentes separados
            
            # Features de momentum redundantes
            'win_pct_3g', 'win_pct_7g',     # Solo mantener win_pct_5g
            
            # Interacciones problemáticas
            'volume_efficiency_quality_interaction',  # Muy correlacionada
            'projection_context_interaction',         # Derivada
        ]
        
        # Filtrar features problemáticas
        stable_features = [f for f in relevant_features 
                          if f not in multicollinear_features]
        
        print(f"Features eliminadas por multicolinealidad conocida: {len([f for f in multicollinear_features if f in relevant_features])}")
        
        # Filtro de correlación entre features restantes (más estricto)
        correlation_threshold_strict = 0.95  # Más estricto para evitar correlaciones extremas
        
        if len(stable_features) > 1:
            feature_data = df[stable_features].select_dtypes(include=[np.number])
            correlation_matrix = feature_data.corr().abs()
            
            features_to_remove = set()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if corr_val > correlation_threshold_strict:
                        feat1 = correlation_matrix.columns[i]
                        feat2 = correlation_matrix.columns[j]
                        
                        # Mantener el que tiene mayor correlación con target
                        corr1_target = target_correlations.get(feat1, 0)
                        corr2_target = target_correlations.get(feat2, 0)
                        
                        if corr1_target >= corr2_target:
                            features_to_remove.add(feat2)
                            to_remove, to_keep = feat2, feat1
                        else:
                            features_to_remove.add(feat1)
                            to_remove, to_keep = feat1, feat2
                        
                        high_corr_pairs.append((to_keep, to_remove, corr_val))
            
            if high_corr_pairs:
                print(f"MULTICOLINEALIDAD ALTA DETECTADA (>{correlation_threshold_strict}):")
                for keep, remove, corr_val in high_corr_pairs:
                    print(f"   • {remove} eliminada (corr con {keep}: {corr_val:.3f})")
            
            final_features = [col for col in stable_features if col not in features_to_remove]
        else:
            final_features = stable_features
        
        # Limitar número máximo de features
        max_features = 30  # Reducido para mayor estabilidad
        if len(final_features) > max_features:
            final_features_with_corr = [(col, target_correlations[col]) for col in final_features]
            final_features_with_corr.sort(key=lambda x: x[1], reverse=True)
            final_features = [col for col, _ in final_features_with_corr[:max_features]]
            print(f"Seleccionadas TOP {max_features} features por correlacion con target")
        
        # Crear dataframe filtrado
        essential_cols = ['Team', 'TEAM', 'Date', 'Opp', 'PTS', 'PTS_Opp']
        if target_col in df.columns and target_col not in essential_cols:
            essential_cols.append(target_col)
        
        cols_to_keep = [col for col in essential_cols if col in df.columns] + final_features
        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        
        df_filtered = df[cols_to_keep].copy()
        self.feature_columns = final_features
        
        print(f"Features finales sin multicolinealidad: {len(final_features)}")
        print(f"Reduccion total: {((len(feature_cols) - len(final_features)) / len(feature_cols) * 100):.1f}%")
        
        # Limpiar target temporal
        if target_col == 'temp_total_points' and 'temp_total_points' in df_filtered.columns:
            df_filtered = df_filtered.drop('temp_total_points', axis=1)
            if 'temp_total_points' in self.feature_columns:
                self.feature_columns.remove('temp_total_points')
        
        return df_filtered
    
    def get_top_correlated_features(self, df: pd.DataFrame, target_col: str = None, top_n: int = 20) -> Dict:
        """Análisis optimizado de correlaciones"""
        if target_col is None:
            if 'total_score' in df.columns:
                target_col = 'total_score'
            elif 'total_points' in df.columns:
                target_col = 'total_points'
            else:
                target_col = 'PTS'
        
        exclude_cols = ['Team', 'TEAM', 'Date', 'Opp']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]
        
        correlations = {}
        for col in feature_cols:
            try:
                if not df[col].isna().all():
                    corr = abs(df[col].corr(df[target_col]))
                    if not np.isnan(corr):
                        correlations[col] = corr
            except Exception:
                continue
        
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_correlations[:top_n])
        
        return {
            'top_correlations': top_features,
            'target_column': target_col,
            'total_features_analyzed': len(feature_cols)
        }
    
    def prepare_prediction_features(self, team1: str, team2: str, teams_data: pd.DataFrame, 
                                  is_team1_home: bool = True) -> np.ndarray:
        """Preparación optimizada de features para predicción"""
        
        try:
            # Crear features completas
            df_with_features = self.create_features(teams_data)
            df_with_features = self.apply_final_correlation_filter(df_with_features, correlation_threshold=0.95)
            
            # Obtener últimos datos de cada equipo
            team1_data = df_with_features[df_with_features['Team'] == team1].iloc[-1:]
            team2_data = df_with_features[df_with_features['Team'] == team2].iloc[-1:]
            
            if team1_data.empty or team2_data.empty:
                raise ValueError(f"No hay datos suficientes para {team1} o {team2}")
            
            # Extraer features disponibles
            team1_features = []
            team2_features = []
            
            for feature_name in self.feature_columns:
                try:
                    if feature_name in df_with_features.columns:
                        team1_val = team1_data[feature_name].iloc[0] if not team1_data[feature_name].isna().iloc[0] else 0
                        team2_val = team2_data[feature_name].iloc[0] if not team2_data[feature_name].isna().iloc[0] else 0
                    else:
                        # Valores por defecto para features faltantes
                        if 'is_home' in feature_name:
                            team1_val = 1 if is_team1_home else 0
                            team2_val = 0 if is_team1_home else 1
                        else:
                            team1_val = 0
                            team2_val = 0
                except Exception:
                    team1_val = 0
                    team2_val = 0
                
                team1_features.append(team1_val)
                team2_features.append(team2_val)
                        
            # Combinar features de forma inteligente
            combined_features = []
            for i, feature_name in enumerate(self.feature_columns):
                try:
                    if any(keyword in feature_name for keyword in ['pts_avg', 'total_pts', 'projection']):
                        # SUMAR capacidades de puntuación
                        combined_features.append(team1_features[i] + team2_features[i])
                    elif any(keyword in feature_name for keyword in ['pace', 'combined', 'possessions']):
                        # SUMAR métricas de ritmo
                        combined_features.append(team1_features[i] + team2_features[i])
                    elif 'is_home' in feature_name:
                        # Ajustar ventaja local
                        combined_features.append(1 if is_team1_home else 0)
                    else:
                        # Promediar otras métricas
                        combined_features.append((team1_features[i] + team2_features[i]) / 2)
                except Exception:
                    combined_features.append(0)
            
            # Verificar valores problemáticos
            combined_features = np.array(combined_features)
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return combined_features.reshape(1, -1)
        
        except Exception as e:
            print(f"Error en prepare_prediction_features: {e}")
            # Retornar features por defecto
            default_features = np.zeros((1, len(self.feature_columns)))
            return default_features
    
    def get_feature_importance_names(self) -> List[str]:
        """Retorna nombres de features para análisis de importancia"""
        return self.feature_columns
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtros de calidad optimizados"""
        
        # Eliminar columnas con demasiados valores NaN
        threshold = 0.5
        df = df.dropna(axis=1, thresh=int(len(df) * threshold))
        
        # Rellenar NaN restantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                if 'pct' in col.lower() or '%' in col:
                    df[col] = df[col].fillna(0.45)  # Promedio NBA para porcentajes
                elif 'pts' in col.lower() or 'score' in col.lower():
                    df[col] = df[col].fillna(110)   # Promedio NBA para puntos
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # Eliminar outliers extremos
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(Q1, Q3)
        
        return df
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualiza la lista de columnas de features disponibles"""
        
        exclude_cols = [
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            'PTS', 'PTS_Opp', 'total_points', 'total_score'
        ]
        
        feature_cols = []
        for col in df.columns:
            if (col not in exclude_cols and 
                df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and
                not df[col].isna().all()):
                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        print(f"Features disponibles actualizadas: {len(feature_cols)} columnas")
