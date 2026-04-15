import os

import fastf1
import numpy as np
import pandas as pd

# Configuración de Cache para funcionamiento offline
cache_dir = os.path.join(os.path.dirname(__file__), "..", "f1_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)


COMPOUNDS_PHYSICS = {
    "SOFT": {"friccion_base": 11.5, "k_disipacion": 0.20, "desgaste_base": 0.045, "ventana_temp": (95, 105), "max_life": 25},
    "MEDIUM": {"friccion_base": 8.0, "k_disipacion": 0.17, "desgaste_base": 0.030, "ventana_temp": (90, 100), "max_life": 40},
    "HARD": {"friccion_base": 7.6, "k_disipacion": 0.16, "desgaste_base": 0.020, "ventana_temp": (85, 95), "max_life": 65},
}

CIRCUITOS_CONFIG = {
    "Monaco": {"abrasion": 0.5, "speed_factor": 0.7, "base_lap_time": 75.0, "event_name": "Monaco", "total_laps": 78, "pit_loss": 20.0},
    "Hungría": {"abrasion": 0.65, "speed_factor": 0.9, "base_lap_time": 80.0, "event_name": "Hungary", "total_laps": 70},
    "Silverstone": {"abrasion": 1.25, "speed_factor": 1.4, "base_lap_time": 90.0, "event_name": "Great Britain", "total_laps": 52},
    "Spa": {"abrasion": 1.25, "speed_factor": 1.5, "base_lap_time": 105.0, "event_name": "Belgium", "total_laps": 44},
    "Monza": {"abrasion": 0.8, "speed_factor": 1.15, "base_lap_time": 81.5, "event_name": "Italy", "total_laps": 53, "pit_loss": 25.0},
    "Interlagos": {"abrasion": 1.1, "speed_factor": 1.2, "base_lap_time": 71.0, "event_name": "Brazil", "total_laps": 71},
    "Suzuka": {"abrasion": 1.3, "speed_factor": 1.3, "base_lap_time": 90.0, "event_name": "Japan", "total_laps": 53, "pit_loss": 22.0},
    "Zandvoort": {"abrasion": 1.0, "speed_factor": 1.1, "base_lap_time": 71.0, "event_name": "Dutch", "total_laps": 72, "pit_loss": 21.0},
    "Baku": {"abrasion": 0.7, "speed_factor": 1.25, "base_lap_time": 103.0, "event_name": "Azerbaijan", "total_laps": 51, "pit_loss": 20.0},
    "Singapur": {"abrasion": 0.8, "speed_factor": 0.9, "base_lap_time": 95.0, "event_name": "Singapore", "total_laps": 62, "pit_loss": 28.0},
    "Las Vegas": {"abrasion": 0.6, "speed_factor": 1.4, "base_lap_time": 94.0, "event_name": "Las Vegas", "total_laps": 50, "pit_loss": 20.0},
}


def get_track_severity(track_name):
    """Retorna factores de severidad basados en telemetría o config."""
    config = CIRCUITOS_CONFIG.get(track_name, {"abrasion": 1.0, "speed_factor": 1.0})
    return config


def _smooth_closed_path(points, n=420):
    """
    Convierte una lista de puntos en un trazado suave y cerrado.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        raise ValueError("Se necesitan al menos 4 puntos para construir el trazado.")

    # Eliminar repetidos consecutivos
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-9:
            cleaned.append(p)
    pts = np.asarray(cleaned, dtype=float)

    # Cerrar el circuito
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1] if s[-1] > 0 else 1.0
    grid = np.linspace(0, total, n)

    x = np.interp(grid, s, pts[:, 0])
    y = np.interp(grid, s, pts[:, 1])

    x = x - np.mean(x)
    y = y - np.mean(y)

    span = max(np.ptp(x), np.ptp(y))
    if span > 0:
        x = x / span
        y = y / span

    return x, y


# Trazados estilizados por circuito. Sirven como fallback visual.
TRACK_SHAPES_RAW = {
    "Monaco": [(-0.95, 0.00), (-0.75, 0.35), (-0.55, 0.55), (-0.15, 0.58), (0.10, 0.40), (0.25, 0.10), (0.05, -0.05), (-0.05, -0.30), (0.20, -0.55), (0.58, -0.42), (0.92, -0.15), (0.80, 0.20), (0.55, 0.45), (0.15, 0.70), (-0.35, 0.72)],
    "Hungría": [(-0.85, 0.10), (-0.75, 0.45), (-0.40, 0.62), (0.10, 0.65), (0.58, 0.48), (0.80, 0.10), (0.72, -0.28), (0.30, -0.52), (-0.10, -0.60), (-0.55, -0.45), (-0.82, -0.10)],
    "Silverstone": [(-0.95, 0.00), (-0.70, 0.28), (-0.35, 0.36), (0.05, 0.20), (0.30, 0.42), (0.68, 0.34), (0.92, 0.02), (0.65, -0.25), (0.22, -0.12), (-0.10, -0.35), (-0.55, -0.42), (-0.88, -0.18)],
    "Spa": [(-0.95, -0.05), (-0.72, 0.30), (-0.35, 0.42), (0.02, 0.22), (0.20, 0.55), (0.45, 0.72), (0.75, 0.40), (0.92, 0.05), (0.72, -0.20), (0.25, -0.15), (-0.10, -0.35), (-0.55, -0.42), (-0.82, -0.18)],
    "Monza": [(-0.95, 0.00), (-0.55, 0.02), (-0.15, 0.05), (0.25, 0.06), (0.55, 0.05), (0.90, 0.00), (0.60, -0.20), (0.20, -0.25), (-0.20, -0.22), (-0.62, -0.15)],
    "Interlagos": [(-0.88, 0.05), (-0.60, 0.32), (-0.30, 0.10), (0.05, 0.40), (0.35, 0.22), (0.58, 0.50), (0.85, 0.18), (0.68, -0.20), (0.25, -0.30), (-0.05, -0.55), (-0.45, -0.42), (-0.75, -0.18)],
    "Suzuka": [(-0.85, 0.10), (-0.55, 0.35), (-0.25, 0.20), (0.05, 0.45), (0.35, 0.25), (0.55, 0.55), (0.82, 0.30), (0.72, -0.05), (0.40, -0.25), (0.10, -0.52), (-0.30, -0.45), (-0.65, -0.20)],
    "Zandvoort": [(-0.90, 0.15), (-0.65, 0.35), (-0.35, 0.45), (0.00, 0.35), (0.25, 0.55), (0.58, 0.42), (0.82, 0.10), (0.70, -0.25), (0.35, -0.40), (0.05, -0.55), (-0.35, -0.50), (-0.70, -0.20)],
    "Baku": [(-0.95, 0.00), (-0.20, 0.02), (0.10, 0.25), (0.18, 0.55), (0.36, 0.70), (0.58, 0.58), (0.68, 0.20), (0.80, -0.10), (0.92, -0.15), (0.50, -0.28), (0.05, -0.25), (-0.45, -0.18), (-0.85, -0.12)],
    "Singapur": [(-0.92, 0.15), (-0.68, 0.30), (-0.40, 0.18), (-0.15, 0.42), (0.15, 0.28), (0.42, 0.48), (0.70, 0.25), (0.82, -0.08), (0.50, -0.25), (0.18, -0.38), (-0.10, -0.52), (-0.50, -0.45), (-0.80, -0.18)],
    "Las Vegas": [(-0.92, 0.00), (-0.58, 0.12), (-0.18, 0.10), (0.20, 0.12), (0.60, 0.10), (0.92, 0.00), (0.58, -0.18), (0.18, -0.20), (-0.20, -0.18), (-0.62, -0.12)],
}


def _resample_track_from_points(points, n=700):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        return None

    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
            keep.append(i)
    pts = pts[keep]

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return None

    grid = np.linspace(0, total, n)
    x = np.interp(grid, s, pts[:, 0])
    y = np.interp(grid, s, pts[:, 1])

    x = x - np.mean(x)
    y = y - np.mean(y)

    span = max(np.ptp(x), np.ptp(y))
    if span > 0:
        x = x / span
        y = y / span

    return x, y


def obtener_track_geometry(track_name, year=2023, session_type="R"):
    """
    Devuelve geometría visual para el trazado de pista.
    Primero intenta usar la telemetría de posición de una vuelta rápida.
    Si falla, usa un trazado estilizado.
    """
    config = CIRCUITOS_CONFIG.get(track_name, {"event_name": track_name})
    event = config["event_name"]

    corners = []
    source = "fallback"

    # Intento 1: usar posición real de una vuelta rápida
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load(telemetry=True, weather=False, messages=False)
        circuit_info = session.get_circuit_info()

        if hasattr(circuit_info, "corners") and circuit_info.corners is not None:
            try:
                corners = circuit_info.corners.to_dict("records")
            except Exception:
                corners = []

        fast_lap = session.laps.pick_fastest()
        if fast_lap is not None:
            pos = fast_lap.get_pos_data()
            if pos is not None and {"X", "Y"}.issubset(pos.columns):
                pts = pos[["X", "Y"]].dropna().to_numpy()
                resampled = _resample_track_from_points(pts, n=800)
                if resampled is not None:
                    x, y = resampled
                    return {
                        "x": x,
                        "y": y,
                        "corners": corners,
                        "source": "fastf1_telemetry",
                        "track_name": track_name,
                    }

        source = "fastf1_fallback"
    except Exception:
        source = "fallback"

    # Intento 2: fallback estilizado
    base = TRACK_SHAPES_RAW.get(track_name, TRACK_SHAPES_RAW["Silverstone"])
    x, y = _smooth_closed_path(base, n=520)

    return {
        "x": x,
        "y": y,
        "corners": corners,
        "source": source,
        "track_name": track_name,
    }


def obtener_datos_aumentados(year=2023, track_name="Silverstone", session_type="R"):
    """
    Obtiene datos reales y los aumenta con perfiles sintéticos.
    """
    config = CIRCUITOS_CONFIG.get(track_name, {"event_name": "Silverstone"})
    event = config["event_name"]

    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        laps = session.laps.pick_quicklaps()
    except Exception:
        print(f"Aviso: Usando generador sintético para {track_name}")
        return generar_data_sintetica_pura(track_name)

    data = []

    temps_pista = [25, 35, 45]

    for t_base in temps_pista:
        for _, row in laps.iterrows():
            t_ruido = t_base + np.random.normal(0, 0.5)
            fuel_effect = (row["LapNumber"] / 50.0) * 2.0
            lap_time_norm = row["LapTime"].total_seconds() + fuel_effect

            data.append(
                {
                    "LapNumber": row["LapNumber"],
                    "TyreLife": row["TyreLife"],
                    "Compound": row["Compound"],
                    "TrackTemp": t_ruido,
                    "LapTime": lap_time_norm,
                }
            )

    return pd.DataFrame(data)


def generar_data_sintetica_pura(track_name="Silverstone"):
    """Generador que incluye múltiples compuestos para entrenamiento calibrado por pista."""
    np.random.seed(42)
    data = []
    track_config = CIRCUITOS_CONFIG.get(track_name, {"base_lap_time": 90.0, "abrasion": 1.0})

    for comp, physics in COMPOUNDS_PHYSICS.items():
        vueltas = np.tile(np.arange(1, 41), 3)
        temps = np.repeat([25, 35, 45], 40)

        for i in range(len(vueltas)):
            v = vueltas[i]
            t = temps[i]

            offset_comp = 0.0 if comp == "SOFT" else (1.5 if comp == "MEDIUM" else 3.0)
            base_lap_time = track_config["base_lap_time"] + offset_comp

            warmup = 2.0 * np.exp(-(v - 0.5) / 1.8) if v < 5 else 0.0
            deg_factor = 0.006 if comp == "SOFT" else (0.0035 if comp == "MEDIUM" else 0.0015)
            deg = deg_factor * (max(0, v - 3) ** 2) * track_config["abrasion"]
            termico = 0.001 * (t - 30) ** 2
            noise = np.random.normal(0, 0.02)

            data.append(
                {
                    "TyreLife": v,
                    "Compound": comp,
                    "TrackTemp": t + np.random.normal(0, 0.5),
                    "LapTime": base_lap_time + warmup + deg + termico + noise,
                }
            )

    return pd.DataFrame(data)