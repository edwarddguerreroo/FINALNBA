import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from main import DataLoader

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_sequence_generation.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TestSequenceGeneration')

def main():
    """
    Función principal para probar la generación de secuencias
    """
    logger.info("="*50)
    logger.info("INICIANDO PRUEBA DE GENERACIÓN DE SECUENCIAS")
    logger.info("="*50)
    
    # Crear instancia de DataLoader
    data_loader = DataLoader(data_dir="data")
    
    # Definir targets específicos para probar
    player_targets = ['PTS', 'TRB', 'AST']
    team_targets = ['Win', 'Total_Points_Over_Under']
    all_targets = player_targets + team_targets
    
    # Configurar directorio de salida para las secuencias
    output_dir = Path("data/sequences_test")
    
    # Intentar crear el directorio si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Targets de jugadores: {player_targets}")
    logger.info(f"Targets de equipos: {team_targets}")
    logger.info(f"Directorio de salida: {output_dir}")
    
    # Resultados para cada tipo de datos
    player_results = {}
    team_results = {}
    
    try:
        # PRUEBA 1: Secuencias de jugadores
        logger.info("\n" + "="*50)
        logger.info("PRUEBA 1: GENERACIÓN DE SECUENCIAS DE JUGADORES")
        logger.info("="*50)
        
        # Cargar datos de jugadores
        data_path = Path("data/processed/players_features.csv")
        if data_path.exists():
            logger.info(f"Cargando datos de jugadores desde {data_path}")
            player_df = pd.read_csv(data_path)
            logger.info(f"Datos de jugadores cargados. Shape: {player_df.shape}")
            
            # Generar secuencias para jugadores
            logger.info("Ejecutando generate_sequences para jugadores...")
            player_results = data_loader.generate_sequences(
                data_df=player_df,
                targets=player_targets,
                output_dir=os.path.join(output_dir, "player"),
                save=True
            )
        else:
            logger.warning(f"No se encontró {data_path}. Omitiendo prueba de jugadores.")
        
        # PRUEBA 2: Secuencias de equipos
        logger.info("\n" + "="*50)
        logger.info("PRUEBA 2: GENERACIÓN DE SECUENCIAS DE EQUIPOS")
        logger.info("="*50)
        
        # Cargar datos de equipos
        teams_path = Path("data/processed/teams_features.csv")
        if teams_path.exists():
            logger.info(f"Cargando datos de equipos desde {teams_path}")
            team_df = pd.read_csv(teams_path)
            logger.info(f"Datos de equipos cargados. Shape: {team_df.shape}")
            
            # Generar secuencias para equipos
            logger.info("Ejecutando generate_sequences para equipos...")
            team_results = data_loader.generate_sequences(
                data_df=team_df,
                targets=team_targets,
                output_dir=os.path.join(output_dir, "team"),
                save=True
            )
        else:
            logger.warning(f"No se encontró {teams_path}. Omitiendo prueba de equipos.")
        
        # Si los datos no existen, crear datos sintéticos para demostración
        if not player_results and not team_results:
            logger.warning("No se encontraron datos reales. Creando datos sintéticos para demostración.")
            
            # Crear datos simulados
            player_df = create_sample_player_data()
            team_df = create_sample_team_data()
            
            # Generar secuencias con datos simulados
            logger.info("Ejecutando generate_sequences con datos sintéticos...")
            player_results = data_loader.generate_sequences(
                data_df=player_df,
                targets=player_targets,
                output_dir=os.path.join(output_dir, "player_simulated"),
                save=True
            )
            
            team_results = data_loader.generate_sequences(
                data_df=team_df,
                targets=team_targets,
                output_dir=os.path.join(output_dir, "team_simulated"),
                save=True
            )
        
        # Combinar resultados
        all_results = {**player_results, **team_results}
        
        # Mostrar resultados
        logger.info("\n" + "="*50)
        logger.info("RESULTADOS DE LA GENERACIÓN DE SECUENCIAS")
        logger.info("="*50)
        
        # Resultados de jugadores
        if player_results:
            logger.info("\nRESULTADOS DE JUGADORES:")
            for target, (seq_path, map_path) in player_results.items():
                if seq_path:
                    logger.info(f"Target {target}: [OK]")
                    logger.info(f"  - Secuencias: {seq_path}")
                    logger.info(f"  - Mapeo: {map_path}")
                else:
                    logger.info(f"Target {target}: [FAILED] (falló)")
        
        # Resultados de equipos
        if team_results:
            logger.info("\nRESULTADOS DE EQUIPOS:")
            for target, (seq_path, map_path) in team_results.items():
                if seq_path:
                    logger.info(f"Target {target}: [OK]")
                    logger.info(f"  - Secuencias: {seq_path}")
                    logger.info(f"  - Mapeo: {map_path}")
                else:
                    logger.info(f"Target {target}: [FAILED] (falló)")
        
        # Resumen final
        total_targets = len(player_targets) + len(team_targets)
        successful_targets = len([t for t, (s, _) in all_results.items() if s])
        logger.info(f"\nProcesados {successful_targets}/{total_targets} targets exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar secuencias: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        return False
    
    logger.info("Prueba de generación de secuencias completada")
    return True

def create_sample_player_data(num_players=10, games_per_player=20):
    """
    Crea un DataFrame con datos simulados de jugadores para probar la generación de secuencias
    """
    logger.info(f"Creando datos simulados para {num_players} jugadores con {games_per_player} partidos cada uno")
    
    # Lista para almacenar datos
    data = []
    
    # Crear datos para cada jugador
    for player_id in range(1, num_players + 1):
        player_name = f"Player_{player_id}"
        
        # Crear partidos para este jugador
        for game_id in range(1, games_per_player + 1):
            # Fecha del partido (aumentando cada 3 días)
            date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=game_id*3)
            
            # Simular estadísticas básicas
            pts = np.random.normal(15, 5)  # media 15, desviación 5
            trb = np.random.normal(6, 2)
            ast = np.random.normal(4, 2)
            mp = np.random.normal(30, 5)
            fg = np.random.normal(6, 2)
            fga = fg + np.random.normal(6, 2)
            fg_pct = fg / max(1, fga)
            
            # Simular triples
            _3p = np.random.normal(2, 1)
            _3pa = _3p + np.random.normal(3, 1)
            _3p_pct = _3p / max(1, _3pa)
            
            # Otros stats
            stl = np.random.normal(1, 0.5)
            blk = np.random.normal(0.5, 0.5)
            tov = np.random.normal(2, 1)
            
            # Simular características derivadas
            pts_per_minute = pts / max(1, mp)
            trb_per_minute = trb / max(1, mp)
            ast_per_minute = ast / max(1, mp)
            
            # Crear registro
            record = {
                'Player': player_name,
                'Date': date,
                'Team': f"Team_{player_id % 5 + 1}",
                'Opp': f"Team_{(player_id + 2) % 5 + 1}",
                'PTS': max(0, pts),
                'TRB': max(0, trb),
                'AST': max(0, ast),
                'MP': max(0, mp),
                'FG': max(0, fg),
                'FGA': max(0, fga),
                'FG%': max(0, min(1, fg_pct)),
                '3P': max(0, _3p),
                '3PA': max(0, _3pa),
                '3P%': max(0, min(1, _3p_pct)),
                'STL': max(0, stl),
                'BLK': max(0, blk),
                'TOV': max(0, tov),
                'PTS_mean_3': max(0, pts + np.random.normal(0, 2)),
                'TRB_mean_3': max(0, trb + np.random.normal(0, 1)),
                'AST_mean_3': max(0, ast + np.random.normal(0, 1)),
                'PTS_mean_5': max(0, pts + np.random.normal(0, 3)),
                'TRB_mean_5': max(0, trb + np.random.normal(0, 1.5)),
                'AST_mean_5': max(0, ast + np.random.normal(0, 1.5)),
                'PTS_per_minute': max(0, pts_per_minute),
                'TRB_per_minute': max(0, trb_per_minute),
                'AST_per_minute': max(0, ast_per_minute),
                'is_home': game_id % 2,  # Alternar local/visitante
                'Pos': np.random.choice(['G', 'F', 'C', 'G-F', 'F-C'])
            }
            
            data.append(record)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Ordenar por jugador y fecha
    df = df.sort_values(['Player', 'Date'])
    
    logger.info(f"Datos simulados de jugadores creados. Shape: {df.shape}")
    return df

def create_sample_team_data(num_teams=5, games_per_team=40):
    """
    Crea un DataFrame con datos simulados de equipos para probar la generación de secuencias
    """
    logger.info(f"Creando datos simulados para {num_teams} equipos con {games_per_team} partidos cada uno")
    
    # Lista para almacenar datos
    data = []
    
    # Crear datos para cada equipo
    for team_id in range(1, num_teams + 1):
        team_name = f"Team_{team_id}"
        
        # Crear partidos para este equipo
        for game_id in range(1, games_per_team + 1):
            # Fecha del partido (aumentando cada 3 días)
            date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=game_id*3)
            
            # Equipo oponente
            opp_id = (team_id + game_id) % num_teams + 1
            opp_name = f"Team_{opp_id}"
            
            # Simular estadísticas básicas
            pts = np.random.normal(105, 10)  # media 105, desviación 10
            pts_opp = np.random.normal(100, 10)
            total_points = pts + pts_opp
            
            # Determinar si es victoria
            is_win = pts > pts_opp
            win_probability = 0.5 + np.random.normal(0, 0.2)
            
            # Simular tiros
            fg = np.random.normal(40, 5)
            fga = np.random.normal(85, 7)
            fg_pct = fg / max(1, fga)
            
            # Simular efectividad
            offensive_rating = np.random.normal(110, 5)
            defensive_rating = np.random.normal(105, 5)
            efficiency_diff = offensive_rating - defensive_rating
            
            # Crear registro
            record = {
                'Team': team_name,
                'Date': date,
                'Opp': opp_name,
                'PTS': max(0, pts),
                'PTS_Opp': max(0, pts_opp),
                'total_points': max(0, total_points),
                'is_win': int(is_win),
                'win_probability': max(0, min(1, win_probability)),
                'FG': max(0, fg),
                'FGA': max(0, fga),
                'FG%': max(0, min(1, fg_pct)),
                'offensive_rating': max(0, offensive_rating),
                'defensive_rating': max(0, defensive_rating),
                'efficiency_diff': efficiency_diff,
                'pace': np.random.normal(100, 5),
                'possessions': np.random.normal(100, 5),
                'points_per_possession': max(0, pts) / max(1, np.random.normal(100, 5)),
                'PTS_mean_3': max(0, pts + np.random.normal(0, 5)),
                'PTS_mean_5': max(0, pts + np.random.normal(0, 7)),
                'PTS_Opp_mean_3': max(0, pts_opp + np.random.normal(0, 5)),
                'PTS_Opp_mean_5': max(0, pts_opp + np.random.normal(0, 7)),
                'total_points_mean_5': max(0, total_points + np.random.normal(0, 10)),
                'win_rate_10': max(0, min(1, 0.5 + np.random.normal(0, 0.15))),
                'is_home': game_id % 2,  # Alternar local/visitante
                'home_advantage': np.random.normal(3, 1) if game_id % 2 else 0,
                'Win': int(is_win),
                'Total_Points_Over_Under': int(total_points > 210),  # Línea arbitraria
            }
            
            data.append(record)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Ordenar por equipo y fecha
    df = df.sort_values(['Team', 'Date'])
    
    logger.info(f"Datos simulados de equipos creados. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    main() 