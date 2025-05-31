"""
TEST DE DEBUGGING - FEATURES ULTRA PREDICTIVAS
===============================================
Script para verificar que las nuevas features funcionan correctamente
"""

import pandas as pd
import numpy as np
import sys
import traceback
from datetime import datetime, timedelta

# Agregar src al path
sys.path.append('src')

def create_test_data():
    """Crear datos de prueba realistas para NBA"""
    print("🔧 Creando datos de prueba...")
    
    teams = ['LAL', 'GSW', 'BOS', 'MIA', 'PHX']
    dates = pd.date_range('2023-10-15', periods=50, freq='2D')
    
    data = []
    for i, date in enumerate(dates):
        for j, team in enumerate(teams):
            # Crear oponente aleatorio diferente al equipo
            opponents = [t for t in teams if t != team]
            opp = np.random.choice(opponents)
            
            # Datos simulados realistas
            pts = np.random.normal(110, 15)
            pts_opp = np.random.normal(108, 15)
            is_win = 1 if pts > pts_opp else 0
            
            data.append({
                'Team': team,
                'Opp': opp,
                'Date': date,
                'is_win': is_win,
                'PTS': max(80, int(pts)),
                'PTS_Opp': max(80, int(pts_opp)),
                'FG': np.random.randint(35, 50),
                'FGA': np.random.randint(80, 100),
                'FG%': np.random.uniform(0.35, 0.55),
                '3P': np.random.randint(8, 18),
                '3PA': np.random.randint(25, 45),
                '3P%': np.random.uniform(0.25, 0.45),
                'FT': np.random.randint(10, 25),
                'FTA': np.random.randint(15, 30),
                'FT%': np.random.uniform(0.70, 0.90),
                'TRB': np.random.randint(40, 55),
                'AST': np.random.randint(18, 30),
                'has_overtime': 1 if np.random.random() < 0.05 else 0,
                'days_into_season': i * 2 + 15,
                'Away': 1 if np.random.random() < 0.5 else 0
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Team', 'Date']).reset_index(drop=True)
    
    # Calcular días de descanso
    df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
    df['rest_advantage'] = np.random.uniform(-0.3, 0.3, len(df))
    
    print(f"✅ Datos creados: {len(df)} filas, {len(df.columns)} columnas")
    return df

def test_basic_features(df):
    """Test de features básicas necesarias"""
    print("\n🔧 Testeando features básicas...")
    
    try:
        from src.models.teams.is_win.features_is_win import IsWinFeatureEngineer
        
        fe = IsWinFeatureEngineer()
        
        # Crear algunas features básicas primero
        fe._create_temporal_features(df)
        fe._create_contextual_features(df)
        fe._create_performance_features_historical(df)
        fe._create_win_features_historical(df)
        
        print(f"✅ Features básicas creadas. Total columnas: {len(df.columns)}")
        
        # Verificar que tenemos las columnas necesarias
        required_cols = ['team_win_rate_10g', 'team_win_rate_15g', 'team_win_rate_20g']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Columnas faltantes: {missing_cols}")
        else:
            print("✅ Todas las columnas requeridas están presentes")
            
        return True
        
    except Exception as e:
        print(f"❌ Error en features básicas: {e}")
        traceback.print_exc()
        return False

def test_ultra_predictive_features(df):
    """Test específico de las features ultra predictivas"""
    print("\n🔥 TESTEANDO FEATURES ULTRA PREDICTIVAS...")
    
    try:
        from src.models.teams.is_win.features_is_win import IsWinFeatureEngineer
        
        fe = IsWinFeatureEngineer()
        
        print("📋 Columnas antes de ultra features:")
        cols_before = set(df.columns)
        print(f"   Total: {len(cols_before)}")
        
        # Ejecutar la función problemática
        fe._create_ultra_predictive_features(df)
        
        print("📋 Columnas después de ultra features:")
        cols_after = set(df.columns)
        new_cols = cols_after - cols_before
        print(f"   Total: {len(cols_after)}")
        print(f"   Nuevas: {len(new_cols)}")
        
        # Mostrar las nuevas features creadas
        print("\n🎯 NUEVAS FEATURES ULTRA PREDICTIVAS CREADAS:")
        for i, col in enumerate(sorted(new_cols), 1):
            print(f"   {i:2d}. {col}")
        
        # Verificar features específicas importantes
        expected_features = [
            'opponent_elite_quality_10g',
            'clutch_dna_score', 
            'pressure_performance_index',
            'season_mastery_score',
            'h2h_win_rate',
            'revenge_factor_elite',
            'energy_management_score',
            'situational_mastery_index',
            'championship_dna_score'
        ]
        
        print("\n🔍 VERIFICANDO FEATURES CLAVE:")
        for feature in expected_features:
            if feature in df.columns:
                non_null = df[feature].notna().sum()
                mean_val = df[feature].mean() if non_null > 0 else 0
                print(f"   ✅ {feature}: {non_null}/{len(df)} valores, mean={mean_val:.3f}")
            else:
                print(f"   ❌ {feature}: NO CREADA")
        
        # Verificar que no hay valores problemáticos
        print("\n🧪 VERIFICANDO CALIDAD DE DATOS:")
        for col in new_cols:
            if col in df.columns:
                has_inf = np.isinf(df[col]).sum()
                has_nan = df[col].isna().sum()
                if has_inf > 0 or has_nan > 0:
                    print(f"   ⚠️  {col}: {has_inf} inf, {has_nan} NaN")
        
        print("✅ Features ultra predictivas creadas exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en ultra features: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline(df):
    """Test del pipeline completo de features"""
    print("\n🚀 TESTEANDO PIPELINE COMPLETO...")
    
    try:
        from src.models.teams.is_win.features_is_win import IsWinFeatureEngineer
        
        fe = IsWinFeatureEngineer()
        
        print("📋 Columnas iniciales:", len(df.columns))
        
        # Ejecutar pipeline completo
        feature_cols = fe.generate_all_features(df)
        
        print(f"✅ Pipeline completo ejecutado!")
        print(f"📊 Total features generadas: {len(feature_cols)}")
        print(f"📊 Total columnas en DataFrame: {len(df.columns)}")
        
        # Mostrar estadísticas de las top features
        print("\n📈 ESTADÍSTICAS DE TOP FEATURES:")
        ultra_features = [col for col in df.columns if any(keyword in col for keyword in [
            'opponent_elite', 'clutch_dna', 'pressure_performance', 'season_mastery',
            'h2h_', 'revenge_factor', 'energy_management', 'situational_mastery',
            'championship_dna'
        ])]
        
        for feature in ultra_features[:10]:  # Top 10
            if feature in df.columns:
                stats = df[feature].describe()
                print(f"   {feature}:")
                print(f"     Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                print(f"     Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en pipeline completo: {e}")
        traceback.print_exc()
        return False

def main():
    """Función principal de testing"""
    print("="*60)
    print("🧪 TEST DE DEBUGGING - FEATURES ULTRA PREDICTIVAS")
    print("="*60)
    
    # Crear datos de prueba
    df = create_test_data()
    
    # Test 1: Features básicas
    if not test_basic_features(df.copy()):
        print("❌ FALLO en features básicas")
        return
    
    # Test 2: Features ultra predictivas específicas
    df_test = df.copy()
    if not test_ultra_predictive_features(df_test):
        print("❌ FALLO en features ultra predictivas")
        return
    
    # Test 3: Pipeline completo
    df_full = df.copy()
    if not test_full_pipeline(df_full):
        print("❌ FALLO en pipeline completo")
        return
    
    print("\n" + "="*60)
    print("🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
    print("🔥 Las features ultra predictivas están funcionando correctamente!")
    print("="*60)

if __name__ == "__main__":
    main() 