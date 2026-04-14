import fastf1
import pandas as pd
import numpy as np
import os

# Configuración de Cache para funcionamiento offline
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'f1_cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

COMPOUNDS_PHYSICS = {
    'SOFT': {'friccion_base': 11.5, 'k_disipacion': 0.20, 'desgaste_base': 0.045, 'ventana_temp': (95, 105), 'max_life': 25},
    'MEDIUM': {'friccion_base': 8.0, 'k_disipacion': 0.17, 'desgaste_base': 0.030, 'ventana_temp': (90, 100), 'max_life': 40},
    'HARD': {'friccion_base': 7.6, 'k_disipacion': 0.16, 'desgaste_base': 0.020, 'ventana_temp': (85, 95), 'max_life': 65}
}

CIRCUITOS_CONFIG = {
    'Hungría': {'abrasion': 0.65, 'speed_factor': 0.9, 'base_lap_time': 80.0, 'event_name': 'Hungary', 'total_laps': 70},
    'Silverstone': {'abrasion': 1.25, 'speed_factor': 1.4, 'base_lap_time': 90.0, 'event_name': 'Great Britain', 'total_laps': 52},
    'Spa': {'abrasion': 1.25, 'speed_factor': 1.5, 'base_lap_time': 105.0, 'event_name': 'Belgium', 'total_laps': 44},
    'Monza': {'abrasion': 0.8, 'speed_factor': 1.15, 'base_lap_time': 81.5, 'event_name': 'Italy', 'total_laps': 53, 'pit_loss': 25.0},
    'Interlagos': {'abrasion': 1.1, 'speed_factor': 1.2, 'base_lap_time': 71.0, 'event_name': 'Brazil', 'total_laps': 71},

    'Suzuka': {'abrasion': 1.3, 'speed_factor': 1.3, 'base_lap_time': 90.0, 'event_name': 'Japan', 'total_laps': 53, 'pit_loss': 22.0},
    'Zandvoort': {'abrasion': 1.0, 'speed_factor': 1.1, 'base_lap_time': 71.0, 'event_name': 'Dutch', 'total_laps': 72, 'pit_loss': 21.0},
    'Baku': {'abrasion': 0.7, 'speed_factor': 1.25, 'base_lap_time': 103.0, 'event_name': 'Azerbaijan', 'total_laps': 51, 'pit_loss': 20.0},
    'Singapur': {'abrasion': 0.8, 'speed_factor': 0.9, 'base_lap_time': 95.0, 'event_name': 'Singapore', 'total_laps': 62, 'pit_loss': 28.0},
    'Las Vegas': {'abrasion': 0.6, 'speed_factor': 1.4, 'base_lap_time': 94.0, 'event_name': 'Las Vegas', 'total_laps': 50, 'pit_loss': 20.0}
}

def get_track_severity(track_name):
    """Retorna factores de severidad basados en telemetría o config."""
    config = CIRCUITOS_CONFIG.get(track_name, {'abrasion': 1.0, 'speed_factor': 1.0})
    return config

def obtener_datos_aumentados(year=2023, track_name='Silverstone', session_type='R'):
    """
    Obtiene datos reales y los aumenta con perfiles sintéticos.
    """
    config = CIRCUITOS_CONFIG.get(track_name, {'event_name': 'Silverstone'})
    event = config['event_name']
    
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        laps = session.laps.pick_quicklaps()
    except:
        # Fallback sincronizado con la pista elegida
        print(f"Aviso: Usando generador sintético para {track_name}")
        return generar_data_sintetica_pura(track_name)

    # Limpieza y Normalización de combustible
    # Suponemos una pérdida de peso lineal (aprox 0.03s por kg de combustible)
    # y una carga inicial de 100kg que baja a 0 en 50 vueltas.
    data = []
    
    # Perfiles de temperatura para aumentar
    temps_pista = [25, 35, 45]
    
    for t_base in temps_pista:
        for idx, row in laps.iterrows():
            # Inyectar ruido gaussiano en temperatura
            t_ruido = t_base + np.random.normal(0, 0.5)
            
            # Normalización de combustible (aislar degradación)
            # LapTime_norm = LapTime_real - (Combustible * factor)
            fuel_effect = (row['LapNumber'] / 50.0) * 2.0 # Efecto inverso
            lap_time_norm = row['LapTime'].total_seconds() + fuel_effect
            
            data.append({
                'LapNumber': row['LapNumber'],
                'TyreLife': row['TyreLife'],
                'Compound': row['Compound'],
                'TrackTemp': t_ruido,
                'LapTime': lap_time_norm
            })
            
    return pd.DataFrame(data)

def generar_data_sintetica_pura(track_name='Silverstone'):
    """Generador que incluye múltiples compuestos para entrenamiento calibrado por pista."""
    np.random.seed(42)
    data = []
    track_config = CIRCUITOS_CONFIG.get(track_name, {'base_lap_time': 90.0, 'abrasion': 1.0})
    
    # Generamos datos para cada compuesto
    for comp, physics in COMPOUNDS_PHYSICS.items():
        vueltas = np.tile(np.arange(1, 41), 3) 
        temps = np.repeat([25, 35, 45], 40)
        
        for i in range(len(vueltas)):
            v = vueltas[i]
            t = temps[i]
            
            # Modelo: Base Pista + Offset Compuesto + Degradación (con multiplicador)
            offset_comp = 0.0 if comp == 'SOFT' else (1.5 if comp == 'MEDIUM' else 3.0)
            base_lap_time = track_config['base_lap_time'] + offset_comp
            
            # La degradación base se multiplica por el factor de abrasión real de la pista
            # 1. Warm-up Effect (Goma fría): Pico de lentitud en vuelta 1, se optimiza en 3-4
            warmup = 2.0 * np.exp(-(v - 0.5) / 1.8) if v < 5 else 0.0
            
            # 2. Degradación cuadrática (empieza plana, sube después)
            deg_factor = 0.006 if comp == 'SOFT' else (0.0035 if comp == 'MEDIUM' else 0.0015)
            # Elongamos la vida útil inicial (curva en U/J)
            deg = deg_factor * (max(0, v - 3)**2) * track_config['abrasion']
            
            termico = 0.001 * (t - 30)**2
            noise = np.random.normal(0, 0.02)
            
            data.append({
                'TyreLife': v,
                'Compound': comp,
                'TrackTemp': t + np.random.normal(0, 0.5),
                'LapTime': base_lap_time + warmup + deg + termico + noise
            })
            
    return pd.DataFrame(data)
