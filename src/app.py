import os
import time
import base64
import mimetypes
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from extractor_datos import (
    obtener_datos_aumentados,
    obtener_track_geometry,
    COMPOUNDS_PHYSICS,
    CIRCUITOS_CONFIG,
)
from modelo_ml import PredictorDegradacion
from strategy import EngineEstrategia


# =========================
# Helpers visuales
# =========================
def format_time(total_seconds):
    """Formatea segundos en MM:SS.ms o HH:MM:SS.ms si supera la hora."""
    if total_seconds is None or not np.isfinite(total_seconds):
        return "N/A"

    if total_seconds < 3600:
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        ms = int((total_seconds % 1) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{ms:03d}"
    else:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        ms = int((total_seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def format_gap(delta_seconds):
    if delta_seconds is None or not np.isfinite(delta_seconds):
        return "N/A"
    sign = "+" if delta_seconds >= 0 else "-"
    return f"{sign}{abs(delta_seconds):.3f}s"


def asset_path(*parts):
    return Path(__file__).resolve().parent.joinpath(*parts)


def _safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )


def image_to_data_uri(path):
    """Convierte una imagen local a data URI para usarla en Plotly o HTML."""
    if not path or not os.path.exists(path):
        return None
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_or_placeholder(path, title, subtitle=""):
    if path and os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div class="card-box" style="text-align:center; min-height:180px; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:48px; line-height:1;">🏎️</div>
                <div style="font-weight:700; color:#e2e8f0; margin-top:8px;">{title}</div>
                <div class="small-muted">{subtitle if subtitle else "Placeholder para imagen"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================
# Textos académicos externos
# =========================
ACADEMIC_TEXTS_DIR = asset_path("academic_texts")

DEFAULT_ACADEMIC_TEXTS = {
    "least_squares": """
**Ajuste de Curvas / Mínimos Cuadrados**

En este módulo se ajusta una función a los datos de telemetría para representar la degradación del neumático. El método de mínimos cuadrados busca el polinomio que minimiza el error cuadrático entre los valores observados y los valores estimados.

En el simulador, esto permite construir una curva de ritmo por vuelta que luego se usa para comparar estrategias de carrera.

Aplicación en el proyecto:
- transformar datos discretos de telemetría en una función continua;
- estimar la evolución del lap time;
- alimentar el motor de estrategia con un modelo numérico interpretable.
""",
    "rk45": """
**Runge-Kutta / EDO térmica**

La temperatura del neumático se modela mediante una ecuación diferencial ordinaria. El método de Runge-Kutta aproxima la solución paso a paso, capturando la evolución térmica durante el stint.

En el simulador, esta temperatura afecta la degradación, el agarre y las penalizaciones por sobrecalentamiento.

Aplicación en el proyecto:
- resolver la dinámica térmica vuelta a vuelta;
- estimar ventanas óptimas de trabajo térmico;
- conectar física del compuesto con la estrategia.
""",
    "simpson": """
**Integración numérica / Regla de Simpson**

La integración numérica permite calcular el tiempo total acumulado a partir de una función de tiempo por vuelta. Simpson 1/3 aproxima el área bajo la curva con mayor precisión que una suma simple cuando la función presenta variaciones suaves.

En el simulador, se usa para estimar el costo temporal de un stint o una carrera completa.

Aplicación en el proyecto:
- obtener tiempo total continuo;
- comparar estrategias con distinta cantidad de paradas;
- cuantificar impacto de la degradación en el resultado final.
""",
    "newton": """
**Newton-Raphson / Punto de cruce**

Newton-Raphson se usa para encontrar raíces de una ecuación no lineal. En este contexto, ayuda a ubicar el crossover point: el instante en el que una estrategia deja de ser mejor que otra.

En el simulador, la raíz de la diferencia entre dos funciones de tiempo representa el momento de cambio de estrategia.

Aplicación en el proyecto:
- determinar el punto óptimo de parada;
- comparar tiempo de pista vs. tiempo de boxes;
- resolver condiciones de empate entre dos estrategias.
""",
    "ml": """
**Modelo de Machine Learning**

El aprendizaje automático complementa el modelado matemático al capturar relaciones no lineales difíciles de representar con una sola ecuación. Aquí, el modelo se alimenta con variables como vida del neumático, temperatura y compuesto.

En el simulador, ML actúa como predictor del tiempo de vuelta y se integra con los métodos numéricos para construir una solución híbrida.

Aplicación en el proyecto:
- mejorar la estimación del ritmo real;
- combinar física e inferencia estadística;
- reforzar la validación con datos de telemetría.
""",
    "global": """
**Lectura general del simulador**

Este laboratorio une modelado matemático, simulación y predicción de datos para resolver un problema realista de ingeniería: elegir la mejor estrategia de carrera bajo restricciones físicas, térmicas y operativas.

Los métodos numéricos no aparecen como contenido aislado, sino como herramientas que resuelven partes concretas del sistema: ajuste, integración, resolución de EDO, búsqueda de raíces y validación comparativa.
""",
}


def ensure_academic_texts_dir():
    ACADEMIC_TEXTS_DIR.mkdir(parents=True, exist_ok=True)


def load_text_asset(key: str) -> str:
    ensure_academic_texts_dir()
    file_path = ACADEMIC_TEXTS_DIR / f"{key}.md"
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return DEFAULT_ACADEMIC_TEXTS.get(key, "")


def render_explanation_block(title: str, body: str, icon: str = "🧠"):
    st.markdown(
        f"""
        <div class="explain-box">
            <div class="explain-title">{icon} {title}</div>
            <div class="explain-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def academic_method_card(title: str, short_desc: str, detail_key: str, icon: str = "📘"):
    with st.expander(f"{icon} {title}", expanded=False):
        st.markdown(f"<div class='small-muted'>{short_desc}</div>", unsafe_allow_html=True)
        st.markdown(load_text_asset(detail_key))
        st.caption("Podés reemplazar este texto por un archivo .md en academic_texts/ si querés editarlo sin tocar el código.")


# =========================
# Estilos
# =========================
def aplicar_estilos():
    bg_path = asset_path("assets", "f1_bg.png")
    bg_uri = image_to_data_uri(bg_path) if os.path.exists(bg_path) else ""
    bg_css = (
        f"background-image: url('{bg_uri}'); background-size: cover; background-position: center; "
        f"background-attachment: fixed; background-blend-mode: overlay;"
        if bg_uri
        else ""
    )

    style = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .stApp {
            background-color: #0b0b0f;
            color: #ffffff;
            __BG_CSS__
        }

        html, body, [class*="css"] {
            font-family: 'Inter', 'Titillium Web', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: transparent !important; border-bottom: none !important;}
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f14 0%, #15151e 100%) !important;
            border-right: 2px solid #e10600 !important;
            box-shadow: 5px 0 25px rgba(0,0,0,0.5);
        }

        [data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .st-emotion-cache-10lo6u4 {
            color: #e10600 !important;
            font-family: 'Titillium Web', sans-serif !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            border-left: 4px solid #e10600;
            padding-left: 12px;
            margin-top: 25px;
            margin-bottom: 15px;
        }

        [data-testid="stSidebar"] label {
            color: #a1a1aa !important;
            font-weight: 700 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 4px;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"],
        [data-testid="stSidebar"] div[data-baseweb="input"] {
            background-color: #1a1a24 !important;
            border: 1px solid #33333e !important;
            border-radius: 4px !important;
        }

        [data-testid="stSidebar"] .stSlider {
            padding-bottom: 15px;
        }

        [data-testid="stSidebar"] .stExpander {
            background-color: rgba(26, 26, 36, 0.6) !important;
            border: 1px solid #33333e !important;
            border-radius: 6px !important;
            margin-bottom: 10px;
        }

        [data-testid="stSidebar"] .stExpander summary {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        div[data-testid="metric-container"] {
            background-color: rgba(21, 21, 30, 0.85);
            border: 1px solid #e10600;
            padding: 15px 20px;
            border-radius: 8px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(225, 6, 0, 0.2);
            backdrop-filter: blur(8px);
        }

        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #ff1801;
            box-shadow: 0 8px 15px -3px rgba(225, 6, 0, 0.4);
        }

        div[data-testid="stMetricLabel"] {
            color: #a1a1aa !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        .card-box {
            background: rgba(25, 25, 35, 0.8);
            border: 1px solid #33333e;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .small-muted {
            color: #a1a1aa;
            font-size: 0.85rem;
        }

        .explain-box {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.92));
            border: 1px solid rgba(225, 6, 0, 0.35);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
            margin: 6px 0 12px 0;
        }

        .explain-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 1.02rem;
            letter-spacing: 0.3px;
            margin-bottom: 8px;
        }

        .explain-body {
            color: #d1d5db;
            font-size: 0.93rem;
            line-height: 1.55;
        }

        .stButton button {
            background-color: #e10600 !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 4px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.2s !important;
        }

        .stButton button:hover {
            background-color: #ff1801 !important;
            box-shadow: 0 0 15px rgba(225, 6, 0, 0.4) !important;
            transform: translateY(-1px) !important;
        }

        hr {
            border-top: 1px solid #33333e !important;
            margin: 1.5rem 0 !important;
        }

        .method-chip {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            margin: 0 6px 6px 0;
            background: rgba(225, 6, 0, 0.12);
            border: 1px solid rgba(225, 6, 0, 0.35);
            color: #fff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.4px;
        }
        </style>
    """

    style = style.replace("__BG_CSS__", bg_css)
    st.markdown(style, unsafe_allow_html=True)


# =========================
# Catálogo placeholders
# =========================
TEAM_DRIVERS = {
    "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
    "Red Bull": ["Max Verstappen", "Yuki Tsunoda"],
    "McLaren": ["Lando Norris", "Oscar Piastri"],
    "Mercedes": ["George Russell", "Kimi Antonelli"],
    "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
    "Alpine": ["Pierre Gasly", "Franco Colapinto"],
    "Williams": ["Alex Albon", "Carlos Sainz"],
    "Haas": ["Esteban Ocon", "Oliver Bearman"],
    "RB": ["Isack Hadjar", "Liam Lawson"],
    "Sauber": ["Nico Hülkenberg", "Gabriel Bortoleto"],
}

DEFAULT_TEAM_IMAGE = asset_path("assets", "teams", "placeholder_team.png")
DEFAULT_DRIVER_IMAGE = asset_path("assets", "drivers", "placeholder_driver.png")


def get_team_image(team_name):
    safe = _safe_name(team_name)
    candidates = [
        asset_path("assets", "teams", f"{safe}.png"),
        asset_path("assets", "teams", f"{safe}.jpg"),
        asset_path("assets", "teams", f"{safe}.jpeg"),
        asset_path("assets", "teams", f"{safe}.webp"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return str(p)
    return str(DEFAULT_TEAM_IMAGE) if os.path.exists(DEFAULT_TEAM_IMAGE) else None


def get_driver_image(driver_name):
    safe = _safe_name(driver_name)
    candidates = [
        asset_path("assets", "drivers", f"{safe}.png"),
        asset_path("assets", "drivers", f"{safe}.jpg"),
        asset_path("assets", "drivers", f"{safe}.jpeg"),
        asset_path("assets", "drivers", f"{safe}.webp"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return str(p)
    return str(DEFAULT_DRIVER_IMAGE) if os.path.exists(DEFAULT_DRIVER_IMAGE) else None


# =========================
# Reglas de estrategia
# =========================
def normalizar_compuestos_por_regla(tipo_estrategia, lista_compuestos):
    """
    Regla:
    - 1 parada: los 2 stints no pueden ser iguales.
    - 2 paradas: tiene que haber al menos 2 compuestos distintos.
    """
    compuestos = list(lista_compuestos)
    disponibles = list(COMPOUNDS_PHYSICS.keys())

    if len(disponibles) < 2:
        return compuestos, False

    if tipo_estrategia == "1 Parada":
        if len(compuestos) >= 2 and compuestos[0] == compuestos[1]:
            base = compuestos[0]
            alternativo = next((c for c in disponibles if c != base), base)
            compuestos[1] = alternativo
            return compuestos, True

    if tipo_estrategia == "2 Paradas":
        if len(set(compuestos)) < 2:
            base = compuestos[0]
            alternativo = next((c for c in disponibles if c != base), base)
            if len(compuestos) > 1:
                compuestos[1] = alternativo
            if len(compuestos) > 2:
                compuestos[2] = base
            return compuestos, True

    return compuestos, False


# =========================
# Track / animación
# =========================
def _smooth_closed_path(points, n=420):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        raise ValueError("Se necesitan al menos 4 puntos para construir el trazado.")

    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-9:
            cleaned.append(p)
    pts = np.asarray(cleaned, dtype=float)

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

    # Recentrado sin deformar la proporción natural del circuito.
    x = x - np.mean(x)
    y = y - np.mean(y)

    span_x = np.ptp(x)
    span_y = np.ptp(y)
    if max(span_x, span_y) > 0:
        scale = max(span_x, span_y)
        x = x / scale
        y = y / scale

    return x, y


def _track_quality_ok(x, y):
    """Valida que la geometría sea lo bastante limpia para usarla en el mapa."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 120 or len(y) < 120:
        return False
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return False

    span_x = float(np.ptp(x))
    span_y = float(np.ptp(y))
    if span_x <= 0 or span_y <= 0:
        return False

    bbox = max(span_x, span_y)
    aspect = max(span_x, span_y) / max(min(span_x, span_y), 1e-9)
    if aspect > 8.0:
        return False

    pts = np.column_stack([x, y])
    steps = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    if len(steps) == 0:
        return False

    median_step = float(np.median(steps))
    p95_step = float(np.percentile(steps, 95))
    if median_step <= 0:
        return False
    if p95_step > median_step * 12.0:
        return False

    closure = float(np.linalg.norm(pts[0] - pts[-1]))
    if closure > bbox * 0.8:
        return False

    return True


def _build_path_from_corners(corners, n=520):
    """Construye un trazado limpio a partir de las esquinas del circuito."""
    if not corners:
        return None

    try:
        df = pd.DataFrame(corners)
    except Exception:
        return None

    if not {"X", "Y"}.issubset(df.columns):
        return None

    if "Number" in df.columns:
        try:
            df = df.sort_values("Number")
        except Exception:
            pass
    elif "Letter" in df.columns:
        try:
            df = df.sort_values("Letter")
        except Exception:
            pass

    pts = df[["X", "Y"]].dropna().to_numpy()
    if len(pts) < 4:
        return None

    return _smooth_closed_path(pts, n=n)


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

    # Recentrado sin deformar la proporción natural del circuito.
    x = x - np.mean(x)
    y = y - np.mean(y)

    span_x = np.ptp(x)
    span_y = np.ptp(y)
    if max(span_x, span_y) > 0:
        scale = max(span_x, span_y)
        x = x / scale
        y = y / scale

    return x, y


def _track_quality_ok(x, y):
    """Valida que la geometría sea lo bastante limpia para usarla en el mapa."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 120 or len(y) < 120:
        return False
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return False

    span_x = float(np.ptp(x))
    span_y = float(np.ptp(y))
    if span_x <= 0 or span_y <= 0:
        return False

    bbox = max(span_x, span_y)
    aspect = max(span_x, span_y) / max(min(span_x, span_y), 1e-9)
    if aspect > 8.0:
        return False

    pts = np.column_stack([x, y])
    steps = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    if len(steps) == 0:
        return False

    median_step = float(np.median(steps))
    p95_step = float(np.percentile(steps, 95))
    if median_step <= 0:
        return False
    if p95_step > median_step * 12.0:
        return False

    closure = float(np.linalg.norm(pts[0] - pts[-1]))
    if closure > bbox * 0.8:
        return False

    return True


def _build_path_from_corners(corners, n=520):
    """Construye un trazado limpio a partir de las esquinas del circuito."""
    if not corners:
        return None

    try:
        df = pd.DataFrame(corners)
    except Exception:
        return None

    if not {"X", "Y"}.issubset(df.columns):
        return None

    if "Number" in df.columns:
        try:
            df = df.sort_values("Number")
        except Exception:
            pass
    elif "Letter" in df.columns:
        try:
            df = df.sort_values("Letter")
        except Exception:
            pass

    pts = df[["X", "Y"]].dropna().to_numpy()
    if len(pts) < 4:
        return None

    return _smooth_closed_path(pts, n=n)


def obtener_track_geometry(track_name, year=2023, session_type="R"):
    """
    Devuelve geometría visual para el trazado de pista.
    Prioriza FastF1 si la geometría es razonable; si no, usa un fallback estilizado.
    """
    config = CIRCUITOS_CONFIG.get(track_name, {"event_name": track_name})
    event = config["event_name"]

    corners = []
    source = "fallback"

    try:
        import fastf1

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
                    if _track_quality_ok(x, y):
                        return {
                            "x": x,
                            "y": y,
                            "corners": corners,
                            "source": "fastf1_telemetry",
                            "track_name": track_name,
                        }
                    source = "fastf1_rejected"

        # Si la telemetría no fue buena, intentamos con las esquinas del circuito.
        corner_path = _build_path_from_corners(corners, n=620)
        if corner_path is not None:
            x, y = corner_path
            if _track_quality_ok(x, y):
                return {
                    "x": x,
                    "y": y,
                    "corners": corners,
                    "source": "fastf1_corners",
                    "track_name": track_name,
                }
            source = "corners_rejected"

        source = source if source != "fallback" else "fastf1_fallback"
    except Exception:
        source = "fallback"

    # Fallback estilizado: estable, limpio y consistente para defensa / demostración.
    base = TRACK_SHAPES_RAW.get(track_name, TRACK_SHAPES_RAW["Silverstone"])
    x, y = _smooth_closed_path(base, n=520)

    return {
        "x": x,
        "y": y,
        "corners": corners,
        "source": source,
        "track_name": track_name,
    }

def get_path_arrays(track_name):
    geom = obtener_track_geometry(track_name)
    x = np.asarray(geom["x"], dtype=float)
    y = np.asarray(geom["y"], dtype=float)
    return geom, x, y


def get_car_position_on_path(x_path, y_path, progress_0_1):
    progress_0_1 = float(np.clip(progress_0_1, 0.0, 1.0))
    idx = int(progress_0_1 * (len(x_path) - 1))
    idx = max(0, min(idx, len(x_path) - 1))
    return x_path[idx], y_path[idx], idx


def get_lap_state(cumulative_times, sim_clock, total_laps):
    if cumulative_times is None or len(cumulative_times) == 0:
        return 1.0, 0, 0.0, 0.0

    total_time = float(cumulative_times[-1])
    sim_clock = float(np.clip(sim_clock, 0.0, total_time))

    lap_idx = int(np.searchsorted(cumulative_times, sim_clock, side="right"))
    lap_idx = max(0, min(lap_idx, len(cumulative_times) - 1))

    prev_time = 0.0 if lap_idx == 0 else float(cumulative_times[lap_idx - 1])
    curr_time = float(cumulative_times[lap_idx])
    denom = max(1e-9, curr_time - prev_time)
    lap_fraction = float(np.clip((sim_clock - prev_time) / denom, 0.0, 1.0))

    current_lap_float = float(lap_idx + 1 + lap_fraction)
    progress_race = current_lap_float / float(total_laps)
    progress_race = float(np.clip(progress_race, 0.0, 1.0))

    return current_lap_float, lap_idx, lap_fraction, progress_race


def elapsed_at_race_fraction(cumulative_times, total_laps, fraction_0_1):
    if cumulative_times is None or len(cumulative_times) == 0:
        return 0.0
    fraction_0_1 = float(np.clip(fraction_0_1, 0.0, 1.0))
    target_lap = fraction_0_1 * float(total_laps)
    laps = np.arange(1, len(cumulative_times) + 1)
    return float(np.interp(target_lap, laps, cumulative_times))


def build_track_figure(track_name, track_geom, strategy_states):
    x_path = np.asarray(track_geom["x"], dtype=float)
    y_path = np.asarray(track_geom["y"], dtype=float)

    x_span = max(float(np.ptp(x_path)), 1.0)
    y_span = max(float(np.ptp(y_path)), 1.0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_path,
            y=y_path,
            mode="lines",
            line=dict(color="#94a3b8", width=7),
            line_shape="spline",
            name="Circuito",
            hoverinfo="skip",
        )
    )

    for st_data in strategy_states:
        x_car, y_car, idx = get_car_position_on_path(
            x_path, y_path, st_data["display_progress"]
        )

        trail_x = x_path[: idx + 1]
        trail_y = y_path[: idx + 1]

        fig.add_trace(
            go.Scatter(
                x=trail_x,
                y=trail_y,
                mode="lines",
                line=dict(color=st_data["color"], width=6, dash=st_data["dash"]),
                line_shape="spline",
                name=f"{st_data['label']} - progreso",
                hoverinfo="skip",
                opacity=0.85,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_car],
                y=[y_car],
                mode="markers",
                marker=dict(size=42, color="rgba(0,0,0,0)", line=dict(width=0)),
                name=st_data["label"],
                hovertemplate=(
                    f"{st_data['label']}<br>"
                    f"Vuelta: {st_data['current_lap_float']:.2f}<br>"
                    f"Progreso carrera: {st_data['progress_text']}<br>"
                    f"Gap: {st_data['gap_text']}<extra></extra>"
                ),
            )
        )

        icon_path = st_data.get("icon_path")
        icon_uri = image_to_data_uri(icon_path)

        if icon_uri:
            fig.add_layout_image(
                dict(
                    source=icon_uri,
                    xref="x",
                    yref="y",
                    x=x_car,
                    y=y_car,
                    sizex=x_span * 0.05,
                    sizey=y_span * 0.05,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                    sizing="contain",
                    opacity=1.0,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x_car],
                    y=[y_car],
                    mode="markers+text",
                    marker=dict(
                        symbol=st_data["symbol"],
                        size=22,
                        color=st_data["color"],
                        line=dict(color="white", width=2),
                    ),
                    text=[st_data["short_label"]],
                    textposition="top center",
                    name=st_data["label"],
                    hovertemplate=(
                        f"{st_data['label']}<br>"
                        f"Vuelta: {st_data['current_lap_float']:.2f}<br>"
                        f"Progreso carrera: {st_data['progress_text']}<br>"
                        f"Gap: {st_data['gap_text']}<extra></extra>"
                    ),
                )
            )

        fig.add_annotation(
            x=x_car,
            y=y_car,
            text=st_data["short_label"],
            showarrow=False,
            yshift=18,
            font=dict(color=st_data["color"], size=11, family="Inter"),
        )

    fig.add_annotation(
        x=x_path[0],
        y=y_path[0],
        text="START / FINISH",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-20,
        font=dict(color="#e10600", size=10),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        height=540,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision="track_map",
        transition={"duration": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, fixedrange=True)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)

    return fig


def build_comparison_gap_figure(result_a, result_b):
    if len(result_a["cumulative_times"]) == 0 or len(result_b["cumulative_times"]) == 0:
        return None

    laps = np.arange(1, min(len(result_a["cumulative_times"]), len(result_b["cumulative_times"])) + 1)
    gap = result_a["cumulative_times"][:len(laps)] - result_b["cumulative_times"][:len(laps)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=gap,
            mode="lines",
            line=dict(width=3, color="#e10600"),
            name="Gap A - B",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        uirevision="gap_compare",
        transition={"duration": 0},
    )
    fig.update_xaxes(title_text="Vuelta")
    fig.update_yaxes(title_text="Diferencia acumulada (s)")
    return fig


def add_method_help_button(key: str, label: str, default_body: str):
    with st.popover(label, use_container_width=False):
        st.markdown(default_body)
        st.caption("La explicación se carga desde academic_texts/ si existe el archivo correspondiente.")


# =========================
# Estado inicial
# =========================
if "vuelta_actual" not in st.session_state:
    st.session_state.vuelta_actual = 1.0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "sim_clock" not in st.session_state:
    st.session_state.sim_clock = 0.0
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()
if "mode_compare" not in st.session_state:
    st.session_state.mode_compare = False
if "show_academic_panel" not in st.session_state:
    st.session_state.show_academic_panel = True
if "show_method_glossary" not in st.session_state:
    st.session_state.show_method_glossary = False

st.set_page_config(page_title="F1 Strategy Wall", layout="wide", page_icon="F1")
aplicar_estilos()


# =========================
# UI principal
# =========================
logo_path = asset_path("assets", "F1_logo.svg")
logo_uri = image_to_data_uri(logo_path) if os.path.exists(logo_path) else ""

if logo_uri:
    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 5px; margin-top: 15px;'>
            <img src='{logo_uri}' width='150' alt='F1 Logo'/>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<h1 style='text-align: center; color: #ffffff; margin-bottom: 0; text-transform: uppercase; letter-spacing: 2px; font-weight: 800; font-family: \"Titillium Web\", sans-serif;'>STRATEGY CONTROL WALL</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #e10600; font-weight: 600; margin-top: 0; letter-spacing: 1px;'>ENTERPRISE RACE ENGINEERING SUITE</p>",
    unsafe_allow_html=True,
)

# =========================
# Sidebar
# =========================
if logo_uri:
    st.sidebar.markdown(
        f"""
        <div style='text-align: center; padding: 10px 0 20px 0;'>
            <img src='{logo_uri}' width='120' style='filter: drop-shadow(0 0 5px rgba(225, 6, 0, 0.2));'/>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.header("Race Strategy")

track_name = st.sidebar.selectbox("Circuito", list(CIRCUITOS_CONFIG.keys()))
track_settings = CIRCUITOS_CONFIG.get(track_name, {})
track_temp = st.sidebar.slider("Temperatura de Pista (°C)", 15, 60, 35)
total_race_laps = st.sidebar.number_input(
    "Vueltas Totales",
    30,
    90,
    int(track_settings.get("total_laps", 50)),
)
pit_loss = st.sidebar.slider(
    "Costo de Pit Stop (s)",
    15.0,
    30.0,
    float(track_settings.get("pit_loss", 22.0)),
)

st.sidebar.markdown("---")
modo = st.sidebar.radio("Modo", ["1 Estrategia", "Comparar 2 Estrategias"], horizontal=False)

st.sidebar.markdown("### Panel académico")
st.session_state.show_academic_panel = st.sidebar.checkbox(
    "Mostrar explicación de métodos",
    value=st.session_state.show_academic_panel,
)
st.session_state.show_method_glossary = st.sidebar.checkbox(
    "Mostrar glosario resumido",
    value=st.session_state.show_method_glossary,
)


def build_strategy_sidebar(strategy_key, title, default_indices):
    with st.sidebar.expander(title, expanded=True):
        team = st.selectbox(
            "Scuderia",
            list(TEAM_DRIVERS.keys()),
            index=default_indices.get("team_idx", 0),
            key=f"{strategy_key}_team",
        )
        drivers = TEAM_DRIVERS.get(team, ["Piloto 1", "Piloto 2"])
        driver = st.selectbox(
            "Piloto",
            drivers,
            index=min(default_indices.get("driver_idx", 0), len(drivers) - 1),
            key=f"{strategy_key}_driver",
        )

        team_image = get_team_image(team)

        st.caption("Imágenes opcionales: `assets/teams/` y `assets/drivers/`")
        c1, c2 = st.columns(2)
        with c1:
            image_or_placeholder(
                team_image,
                team,
                "Imagen del auto / scuderia",
            )
        with c2:
            image_or_placeholder(
                get_driver_image(driver),
                driver,
                "Imagen del piloto",
            )

        st.markdown("**Estrategia de compuestos**")
        tipo_estrategia = st.radio(
            "Tipo de Estrategia",
            ["1 Parada", "2 Paradas"],
            horizontal=True,
            key=f"{strategy_key}_tipo",
        )

        if tipo_estrategia == "1 Parada":
            s1 = st.selectbox(
                "Stint 1",
                list(COMPOUNDS_PHYSICS.keys()),
                index=default_indices.get("s1_idx", 1),
                key=f"{strategy_key}_s1",
            )
            s2 = st.selectbox(
                "Stint 2",
                list(COMPOUNDS_PHYSICS.keys()),
                index=default_indices.get("s2_idx", 2),
                key=f"{strategy_key}_s2",
            )
            lista_compuestos = [s1, s2]
        else:
            s1 = st.selectbox(
                "Stint 1",
                list(COMPOUNDS_PHYSICS.keys()),
                index=default_indices.get("s1_idx", 0),
                key=f"{strategy_key}_s1",
            )
            s2 = st.selectbox(
                "Stint 2",
                list(COMPOUNDS_PHYSICS.keys()),
                index=default_indices.get("s2_idx", 1),
                key=f"{strategy_key}_s2",
            )
            s3 = st.selectbox(
                "Stint 3",
                list(COMPOUNDS_PHYSICS.keys()),
                index=default_indices.get("s3_idx", 2),
                key=f"{strategy_key}_s3",
            )
            lista_compuestos = [s1, s2, s3]

        lista_compuestos, corregido = normalizar_compuestos_por_regla(tipo_estrategia, lista_compuestos)
        if corregido:
            st.warning(
                "Regla aplicada: la estrategia debe usar al menos 2 compuestos distintos. "
                "Se ajustó automáticamente uno de los stints."
            )

    return {
        "strategy_key": strategy_key,
        "team": team,
        "driver": driver,
        "team_image": team_image,
        "tipo_estrategia": tipo_estrategia,
        "lista_compuestos": lista_compuestos,
    }


if modo == "1 Estrategia":
    st.session_state.mode_compare = False
    strategy_inputs = [
        build_strategy_sidebar(
            "A",
            "Estrategia principal",
            {"team_idx": 0, "driver_idx": 0, "s1_idx": 1, "s2_idx": 2, "s3_idx": 0},
        )
    ]
else:
    st.session_state.mode_compare = True
    strategy_inputs = [
        build_strategy_sidebar(
            "A",
            "Estrategia A",
            {"team_idx": 0, "driver_idx": 0, "s1_idx": 1, "s2_idx": 2, "s3_idx": 0},
        ),
        build_strategy_sidebar(
            "B",
            "Estrategia B",
            {"team_idx": 1, "driver_idx": 1, "s1_idx": 0, "s2_idx": 1, "s3_idx": 2},
        ),
    ]

st.sidebar.markdown("---")
st.sidebar.subheader("Reproducción Live Timing")
col_play, col_stop = st.sidebar.columns(2)
if col_play.button("Iniciar Simulación"):
    st.session_state.playing = True
if col_stop.button("Detener"):
    st.session_state.playing = False

if st.sidebar.button("Reiniciar simulación"):
    st.session_state.playing = False
    st.session_state.sim_clock = 0.0
    st.session_state.vuelta_actual = 1.0
    st.session_state.last_tick = time.time()
    st.rerun()

lap_slider_value = st.sidebar.slider(
    "Vuelta Actual",
    1.0,
    float(total_race_laps),
    value=float(st.session_state.vuelta_actual),
    step=0.1,
)
if not st.session_state.playing:
    st.session_state.vuelta_actual = lap_slider_value

# =========================
# Backend pipeline caching
# =========================
@st.cache_data(show_spinner=False)
def cargar_datos_entrenamiento(track_name):
    return obtener_datos_aumentados(track_name=track_name)


@st.cache_resource(show_spinner=False)
def entrenar_modelo_ia(track_name):
    df = cargar_datos_entrenamiento(track_name)
    predictor = PredictorDegradacion()
    base_time = CIRCUITOS_CONFIG.get(track_name, {}).get("base_lap_time", 90.0)
    predictor.entrenar(df, track_base_time=base_time)
    return predictor


@st.cache_data(show_spinner=False)
def get_sim_data(track_name, track_temp, total_race_laps, pit_loss, lista_compuestos):
    predictor = entrenar_modelo_ia(track_name)
    engine = EngineEstrategia(predictor, track_name=track_name)
    engine.pit_loss = pit_loss

    resultado_opt = engine.optimizador_determinista(list(lista_compuestos), total_race_laps, track_temp)
    vueltas_parada = resultado_opt["vueltas_optimas"]
    tiempos_race = resultado_opt["race_trace"]
    comp_trace = resultado_opt["compounds_trace"]

    piecewise_temps, piecewise_life = [], []
    for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
        comp = lista_compuestos[i]
        duracion = p_lap - (vueltas_parada[i - 1] if i > 0 else 0)
        engine.compound = comp
        _, _, t_sector, l_sector = engine.simular_stint(duracion, track_temp)
        piecewise_temps.extend(t_sector.tolist())
        piecewise_life.extend(l_sector.tolist())

    cumulative_times = np.cumsum(tiempos_race) if len(tiempos_race) > 0 else np.array([])

    return {
        "resultado_opt": resultado_opt,
        "vueltas_parada": vueltas_parada,
        "tiempos_race": tiempos_race,
        "comp_trace": comp_trace,
        "piecewise_temps": piecewise_temps,
        "piecewise_life": piecewise_life,
        "laps_range": np.arange(1, total_race_laps + 1),
        "cumulative_times": cumulative_times,
        "total_time": float(resultado_opt["tiempo_total"]) if np.isfinite(resultado_opt["tiempo_total"]) else float("inf"),
        "lista_compuestos": list(lista_compuestos),
    }


# =========================
# Cargar estrategias
# =========================
strategy_results = []
for s in strategy_inputs:
    sim_data = get_sim_data(
        track_name,
        track_temp,
        int(total_race_laps),
        float(pit_loss),
        tuple(s["lista_compuestos"]),
    )
    strategy_results.append({**s, **sim_data})

track_geom = get_path_arrays(track_name)[0]
track_x = np.asarray(track_geom["x"], dtype=float)
track_y = np.asarray(track_geom["y"], dtype=float)

finite_totals = [r["total_time"] for r in strategy_results if np.isfinite(r["total_time"])]
reference_total_time = max(finite_totals) if finite_totals else 1.0
if reference_total_time <= 0:
    reference_total_time = 1.0

animation_speed = reference_total_time / 120.0 if reference_total_time > 0 else 1.0
animation_speed = max(1.0, animation_speed)


# =========================
# Panel académico superior
# =========================
if st.session_state.show_academic_panel:
    st.markdown("## Laboratorio de modelado y simulación")
    st.markdown(
        """
        Esta vista agrega una lectura académica del simulador: cada bloque numérico se explica como parte de un laboratorio de Modelado y Simulación, no solo como estrategia de Fórmula 1.
        """
    )

    st.tabs(["Visión general", "Métodos numéricos", "Machine Learning", "Validación"])
    tab1, tab2, tab3, tab4 = st.tabs(["Visión general", "Métodos numéricos", "Machine Learning", "Validación"])

    with tab1:
        render_explanation_block("Lectura general", load_text_asset("global"), icon="🎯")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<span class="method-chip">Modelado matemático</span>', unsafe_allow_html=True)
        with c2:
            st.markdown('<span class="method-chip">Simulación numérica</span>', unsafe_allow_html=True)
        with c3:
            st.markdown('<span class="method-chip">Optimización</span>', unsafe_allow_html=True)
        with c4:
            st.markdown('<span class="method-chip">Validación con datos</span>', unsafe_allow_html=True)

    with tab2:
        academic_method_card(
            "Ajuste de curvas / Mínimos cuadrados",
            "Convierte datos discretos de telemetría en una función continua de degradación y ritmo.",
            "least_squares",
            icon="📈",
        )
        academic_method_card(
            "Runge-Kutta / EDO térmica",
            "Aproxima la evolución de la temperatura del neumático a lo largo de la vuelta y del stint.",
            "rk45",
            icon="🌡️",
        )
        academic_method_card(
            "Integración numérica / Simpson",
            "Calcula el tiempo total acumulado como área bajo la curva de lap time.",
            "simpson",
            icon="∫",
        )
        academic_method_card(
            "Newton-Raphson / Crossover point",
            "Ubica el instante en que pararse en boxes empieza a ser más rentable que continuar.",
            "newton",
            icon="🧭",
        )

    with tab3:
        academic_method_card(
            "Aprendizaje automático híbrido",
            "Usa temperatura, compuesto y vida del neumático para predecir el lap time con un modelo estadístico.",
            "ml",
            icon="🤖",
        )
        st.info(
            "En el proyecto, ML no reemplaza al modelado físico: lo complementa. La idea es que la predicción aprenda patrones que después se interpretan dentro del motor numérico."
        )

    with tab4:
        st.markdown("### Cómo se valida el simulador")
        st.write(
            "Se comparan estrategias simuladas con escenarios de referencia y se observa la desviación entre tiempos predichos y tiempos esperados. El objetivo no es solo acertar una vuelta de parada, sino medir consistencia, sensibilidad térmica y estabilidad numérica."
        )
        st.markdown(
            "**Indicadores sugeridos:** RMSE de lap times, error en crossover lap, sensibilidad a temperatura de pista, y estabilidad frente a cambios de compuesto."
        )

    st.divider()


# =========================
# Vista dinámica
# =========================
@st.fragment(run_every="250ms")
def render_live_timing_view():
    now = time.time()
    dt = now - st.session_state.last_tick
    st.session_state.last_tick = now

    if st.session_state.playing:
        st.session_state.sim_clock = min(
            reference_total_time,
            st.session_state.sim_clock + dt * animation_speed,
        )
        st.session_state.vuelta_actual = max(
            1.0,
            min(
                float(total_race_laps),
                (st.session_state.sim_clock / reference_total_time) * float(total_race_laps),
            ),
        )
    else:
        st.session_state.sim_clock = (
            float(st.session_state.vuelta_actual) / float(total_race_laps)
        ) * reference_total_time

    st.divider()

    if any(not np.isfinite(r["total_time"]) for r in strategy_results):
        st.error(
            "ESTRATEGIA NO VIABLE: Ninguna combinación cumple con el Límite de Seguridad Estructural (70% vida)."
        )
        return

    if len(strategy_results) == 1:
        r = strategy_results[0]
        p_text = " y ".join([f"Vuelta {v}" for v in r["vueltas_parada"]]) if r["vueltas_parada"] else "sin parada"
        st.success(
            f"ESTRATEGIA ÓPTIMA: Parada en {p_text} | Tiempo de Carrera: {format_time(r['total_time'])}"
        )
    else:
        st.success("COMPARACIÓN ACTIVADA: ambas estrategias se simulan en paralelo sobre el mismo trazado.")

    color_palette = ["#e10600", "#ffeb00"]
    symbol_palette = ["circle", "diamond"]

    strategy_states = []
    common_fraction = float(np.clip(st.session_state.sim_clock / reference_total_time, 0.0, 1.0))

    for i, r in enumerate(strategy_results):
        current_sim_clock = min(st.session_state.sim_clock, r["total_time"])
        current_lap_float, current_lap_idx, lap_fraction, progress_race = get_lap_state(
            r["cumulative_times"],
            current_sim_clock,
            int(total_race_laps),
        )

        common_elapsed = elapsed_at_race_fraction(
            r["cumulative_times"],
            int(total_race_laps),
            common_fraction,
        )

        strategy_states.append(
            {
                "label": f"Estrategia {chr(65 + i)}",
                "short_label": f"{chr(65 + i)}",
                "current_lap_float": current_lap_float,
                "current_lap_idx": current_lap_idx,
                "lap_fraction": lap_fraction,
                "progress_race": progress_race,
                "common_elapsed": common_elapsed,
                "color": color_palette[i % len(color_palette)],
                "symbol": symbol_palette[i % len(symbol_palette)],
                "dash": "solid" if i == 0 else "dash",
                "display_progress": progress_race,
                "gap_text": "N/A",
                "progress_text": f"{progress_race * 100:.1f}%",
                "icon_path": r.get("team_image"),
            }
        )

    if len(strategy_results) > 1:
        gap_common = strategy_states[0]["common_elapsed"] - strategy_states[1]["common_elapsed"]

        for i, state in enumerate(strategy_states):
            other_elapsed = strategy_states[1 - i]["common_elapsed"]
            gap_vs_other = state["common_elapsed"] - other_elapsed

            avg_lap = max(strategy_results[i]["total_time"] / max(int(total_race_laps), 1), 1e-9)
            visual_offset = np.clip((-gap_vs_other / avg_lap) * 0.35, -0.16, 0.16)

            state["display_progress"] = float(np.clip(state["progress_race"] + visual_offset, 0.0, 1.0))
            state["gap_text"] = format_gap(gap_vs_other)
            state["progress_text"] = f"{state['display_progress'] * 100:.1f}%"

        leader_idx = 0 if gap_common < 0 else 1 if gap_common > 0 else 0
        st.info(
            f"Gap actual en el mismo progreso de carrera: **{format_gap(gap_common)}**. "
            f"Delante ahora: **Estrategia {chr(65 + leader_idx)}** "
            f"({strategy_results[leader_idx]['driver']} / {strategy_results[leader_idx]['team']})."
        )
        st.metric(
            "Diferencia en carrera",
            format_gap(gap_common),
            delta=("A adelante" if gap_common < 0 else "B adelante" if gap_common > 0 else "Igualados"),
        )
    else:
        strategy_states[0]["gap_text"] = "N/A"
        strategy_states[0]["progress_text"] = f"{strategy_states[0]['progress_race'] * 100:.1f}%"

    st.subheader("Trazado del circuito y progreso")
    st.caption(
        "Este gráfico muestra el estado del sistema de simulación sobre la geometría del circuito. La posición del auto no es solo visual: sirve para interpretar el avance relativo de cada estrategia respecto del tiempo acumulado."
    )
    st.caption(f"Fuente geométrica cargada: {track_geom.get('source', 'fallback')}")
    track_fig = build_track_figure(
        track_name=track_name,
        track_geom=track_geom,
        strategy_states=strategy_states,
    )
    st.plotly_chart(track_fig, use_container_width=True, key="track_map")

    with st.expander("Explicación del gráfico del circuito", expanded=False):
        st.markdown(load_text_asset("global"))
        st.write(
            "Aquí el trazado ayuda a contextualizar el problema: el método numérico no se usa de forma abstracta, sino dentro de un sistema que transforma variables físicas y temporales en una decisión de estrategia."
        )

    if len(strategy_results) == 1:
        r = strategy_results[0]
        v_act = float(st.session_state.vuelta_actual)
        idx = max(0, min(int(v_act) - 1, int(total_race_laps) - 1))

        stint_start = 0
        for p in r["vueltas_parada"]:
            if int(v_act) > p:
                stint_start = p
            else:
                break

        if idx > stint_start:
            pace_delta = r["tiempos_race"][idx] - r["tiempos_race"][idx - 1]
        else:
            pace_delta = 0.0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Goma Actual", r["comp_trace"][idx])
        with m2:
            st.metric("Temperatura Goma", f"{r['piecewise_temps'][idx]:.1f}°C")
        with m3:
            st.metric(
                "Lap Time",
                format_time(r["tiempos_race"][idx]),
                delta=f"{pace_delta:+.3f}s",
                delta_color="inverse",
            )
        with m4:
            st.metric("Vida Restante", f"{r['piecewise_life'][idx]:.1f}%")

        prog = v_act / float(total_race_laps)
        st.progress(min(1.0, prog))
        st.markdown(
            f"<p style='text-align: center; color: #00d4ff; font-weight: bold;'>Progreso: Vuelta {v_act:.1f} / {total_race_laps} ({prog*100:.1f}%)</p>",
            unsafe_allow_html=True,
        )
    else:
        col_a, col_b = st.columns(2)

        gap_common = strategy_states[0]["common_elapsed"] - strategy_states[1]["common_elapsed"]

        with col_a:
            r = strategy_results[0]
            state = strategy_states[0]
            st.markdown(
                f"<div class='card-box'>"
                f"<div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>Estrategia A</div>"
                f"<div class='small-muted'>{r['team']} · {r['driver']}</div>"
                f"<div class='small-muted'>Paradas: {', '.join([f'V{v}' for v in r['vueltas_parada']]) if r['vueltas_parada'] else 'sin parada'}</div>"
                f"<div style='margin-top:8px; color:#00d4ff; font-weight:700;'>Tiempo total: {format_time(r['total_time'])}</div>"
                f"<div class='small-muted'>Gap vs B: {format_gap(gap_common)}</div>"
                f"<div class='small-muted'>Progreso: {state['progress_text']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            idx = min(state["current_lap_idx"], len(r["comp_trace"]) - 1)
            st.metric("Compuesto actual", r["comp_trace"][idx])
            st.metric("Temperatura goma", f"{r['piecewise_temps'][idx]:.1f}°C")
            st.metric("Vida restante", f"{r['piecewise_life'][idx]:.1f}%")
            st.metric("Vuelta actual", f"{state['current_lap_float']:.2f}")

        with col_b:
            r = strategy_results[1]
            state = strategy_states[1]
            st.markdown(
                f"<div class='card-box'>"
                f"<div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>Estrategia B</div>"
                f"<div class='small-muted'>{r['team']} · {r['driver']}</div>"
                f"<div class='small-muted'>Paradas: {', '.join([f'V{v}' for v in r['vueltas_parada']]) if r['vueltas_parada'] else 'sin parada'}</div>"
                f"<div style='margin-top:8px; color:#00d4ff; font-weight:700;'>Tiempo total: {format_time(r['total_time'])}</div>"
                f"<div class='small-muted'>Gap vs A: {format_gap(-gap_common)}</div>"
                f"<div class='small-muted'>Progreso: {state['progress_text']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            idx = min(state["current_lap_idx"], len(r["comp_trace"]) - 1)
            st.metric("Compuesto actual", r["comp_trace"][idx])
            st.metric("Temperatura goma", f"{r['piecewise_temps'][idx]:.1f}°C")
            st.metric("Vida restante", f"{r['piecewise_life'][idx]:.1f}%")
            st.metric("Vuelta actual", f"{state['current_lap_float']:.2f}")

    st.divider()

    # =========================
    # Gráficos
    # =========================
    st.subheader("Ritmo Segmentado (Pace Analysis)")
    st.caption(
        "Este gráfico muestra el tiempo de vuelta por tramo. En términos de modelado y simulación, es la salida principal del sistema numérico sobre la variable lap time."
    )
    fig1 = go.Figure()

    if len(strategy_results) == 1:
        r = strategy_results[0]
        for p_lap in r["vueltas_parada"]:
            fig1.add_vline(
                x=p_lap,
                line_width=2,
                line_dash="dash",
                line_color="#ff4b4b",
                opacity=0.6,
            )
            fig1.add_annotation(
                x=p_lap,
                y=0.05,
                yref="paper",
                text="<b>BOX</b>",
                showarrow=False,
                font=dict(color="#ff4b4b", size=12),
                bgcolor="rgba(15, 23, 42, 0.8)",
            )

        s_l = 1
        for i, p_lap in enumerate(r["vueltas_parada"] + [int(total_race_laps)]):
            laps = np.arange(s_l, p_lap + 1)
            vals = r["tiempos_race"][s_l - 1:p_lap]
            comp_name = r["lista_compuestos"][i] if i < len(r["lista_compuestos"]) else "N/A"
            fig1.add_trace(
                go.Scatter(
                    x=laps,
                    y=vals,
                    name=f"Stint {i+1}: {comp_name}",
                    line=dict(
                        color={"SOFT": "#ff3333", "MEDIUM": "#ffff33", "HARD": "#f0f0f0"}.get(comp_name, "#00d4ff"),
                        width=4,
                    ),
                )
            )
            s_l = p_lap + 1

        y_int = np.interp(float(st.session_state.vuelta_actual), r["laps_range"], r["tiempos_race"])
        fig1.add_trace(
            go.Scatter(
                x=[float(st.session_state.vuelta_actual)],
                y=[y_int],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#00d4ff",
                    line=dict(color="white", width=2),
                ),
                name="Posición Actual",
            )
        )

    else:
        for i, r in enumerate(strategy_results):
            style = color_palette[i]
            dash = "solid" if i == 0 else "dash"

            for p_lap in r["vueltas_parada"]:
                fig1.add_vline(
                    x=p_lap,
                    line_width=1,
                    line_dash="dot",
                    line_color=style,
                    opacity=0.25,
                )

            s_l = 1
            for j, p_lap in enumerate(r["vueltas_parada"] + [int(total_race_laps)]):
                laps = np.arange(s_l, p_lap + 1)
                vals = r["tiempos_race"][s_l - 1:p_lap]
                comp_name = r["lista_compuestos"][j] if j < len(r["lista_compuestos"]) else "N/A"
                fig1.add_trace(
                    go.Scatter(
                        x=laps,
                        y=vals,
                        name=f"Estrategia {chr(65+i)} - Stint {j+1} ({comp_name})",
                        line=dict(color=style, width=4, dash=dash),
                    )
                )
                s_l = p_lap + 1

            x_current = float(np.clip(strategy_states[i]["current_lap_float"], 1.0, float(total_race_laps)))
            y_current = np.interp(x_current, r["laps_range"], r["tiempos_race"])
            fig1.add_trace(
                go.Scatter(
                    x=[x_current],
                    y=[y_current],
                    mode="markers",
                    marker=dict(
                        symbol=symbol_palette[i],
                        size=16,
                        color=style,
                        line=dict(color="white", width=2),
                    ),
                    name=f"Estrategia {chr(65+i)} actual",
                )
            )

    fig1.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision="pace_live",
        transition={"duration": 0},
    )
    fig1.update_yaxes(title_text="Lap Time (s)")
    fig1.update_xaxes(title_text="Vuelta")
    st.plotly_chart(fig1, use_container_width=True, key="f_ritmo")

    with st.expander("Explicación académica del gráfico de ritmo", expanded=False):
        st.markdown(load_text_asset("least_squares"))
        st.write(
            "En este gráfico, cada punto sintetiza la salida de un modelo que combina ajuste de curvas, efecto térmico y penalización por combustible. Es útil para discutir convergencia, sensibilidad y interpretación física."
        )

    if len(strategy_results) > 1:
        gap_fig = build_comparison_gap_figure(strategy_results[0], strategy_results[1])
        if gap_fig is not None:
            st.subheader("Diferencia acumulada entre estrategias")
            st.caption(
                "La diferencia acumulada muestra qué estrategia va adelante para un mismo nivel relativo de carrera. Matemáticamente, es una comparación de funciones acumuladas."
            )
            st.plotly_chart(gap_fig, use_container_width=True, key="gap_compare")
            with st.expander("Explicación académica de la comparación acumulada", expanded=False):
                st.markdown(
                    """
                    La curva de diferencia acumulada permite leer el problema como una comparación entre dos funciones de costo. En el lenguaje de la materia, no se trata solo de un gráfico de carrera, sino de una herramienta para analizar cuál curva total minimiza mejor el tiempo bajo las mismas condiciones de simulación.
                    """
                )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolución Térmica")
        st.caption(
            "Este gráfico representa la solución numérica de la EDO térmica. La temperatura afecta el comportamiento del neumático y modifica la salida del modelo de tiempo por vuelta."
        )
        fig_termica = go.Figure()

        if len(strategy_results) == 1:
            r = strategy_results[0]
            for p_lap in r["vueltas_parada"]:
                fig_termica.add_vline(x=p_lap, line_width=1, line_dash="dash", line_color="#ff4b4b", opacity=0.4)

            s_l = 1
            for i, p_lap in enumerate(r["vueltas_parada"] + [int(total_race_laps)]):
                l = np.arange(s_l, p_lap + 1)
                v = r["piecewise_temps"][s_l - 1:p_lap]
                comp_name = r["lista_compuestos"][i] if i < len(r["lista_compuestos"]) else "N/A"
                fig_termica.add_trace(
                    go.Scatter(
                        x=l,
                        y=v,
                        name=f"Stint {i+1} ({comp_name})",
                        line=dict(color={"SOFT": "#ff3333", "MEDIUM": "#ffff33", "HARD": "#f0f0f0"}.get(comp_name, "#00d4ff"), width=3),
                    )
                )
                s_l = p_lap + 1
        else:
            for i, r in enumerate(strategy_results):
                color = color_palette[i]
                for p_lap in r["vueltas_parada"]:
                    fig_termica.add_vline(x=p_lap, line_width=1, line_dash="dash", line_color=color, opacity=0.2)

                s_l = 1
                for j, p_lap in enumerate(r["vueltas_parada"] + [int(total_race_laps)]):
                    l = np.arange(s_l, p_lap + 1)
                    v = r["piecewise_temps"][s_l - 1:p_lap]
                    fig_termica.add_trace(
                        go.Scatter(
                            x=l,
                            y=v,
                            name=f"Estrategia {chr(65+i)} - Stint {j+1}",
                            line=dict(color=color, width=3, dash="solid" if i == 0 else "dash"),
                        )
                    )
                    s_l = p_lap + 1

        fig_termica.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            uirevision="temp_live",
            transition={"duration": 0},
        )
        st.plotly_chart(fig_termica, use_container_width=True, key="f_temp_live")

        with st.popover("Ver explicación de la EDO térmica"):
            st.markdown(load_text_asset("rk45"))
            st.write(
                "La integración paso a paso permite observar cómo una ecuación diferencial se traduce en una trayectoria térmica concreta dentro del simulador."
            )

    with col2:
        st.subheader("Vida útil (%)")
        st.caption(
            "Este gráfico representa la evolución del desgaste mecánico del compuesto y muestra dónde el sistema entra en una zona de riesgo estructural."
        )
        fig_desgaste = go.Figure()

        if len(strategy_results) == 1:
            r = strategy_results[0]
            for p_lap in r["vueltas_parada"]:
                fig_desgaste.add_vline(x=p_lap, line_width=1, line_dash="dash", line_color="#ff4b4b", opacity=0.4)

            fig_desgaste.add_hrect(
                y0=0,
                y1=30,
                fillcolor="#ff4b4b",
                opacity=0.15,
                layer="below",
                line_width=0,
            )
            fig_desgaste.add_annotation(
                x=total_race_laps / 2,
                y=15,
                text="ZONA DE RIESGO ESTRUCTURAL",
                showarrow=False,
                font=dict(color="#ff4b4b", size=10),
                opacity=0.8,
            )

            s_l = 1
            for i, p_lap in enumerate(r["vueltas_parada"] + [int(total_race_laps)]):
                l = np.arange(s_l, p_lap + 1)
                v = r["piecewise_life"][s_l - 1:p_lap]
                comp_name = r["lista_compuestos"][i] if i < len(r["lista_compuestos"]) else "N/A"
                fig_desgaste.add_trace(
                    go.Scatter(
                        x=l,
                        y=v,
                        name=f"Stint {i+1} ({comp_name})",
                        line=dict(color={"SOFT": "#ff3333", "MEDIUM": "#ffff33", "HARD": "#f0f0f0"}.get(comp_name, "#00d4ff"), width=3),
                    )
                )
                s_l = p_lap + 1
        else:
            for i, r in enumerate(strategy_results):
                color = color_palette[i]
                for p_lap in r["vueltas_parada"]:
                    fig_desgaste.add_vline(x=p_lap, line_width=1, line_dash="dash", line_color=color, opacity=0.2)

                s_l = 1
                for j, p_lap in enumerate(r["vueltas_parada"] + [int(total_race_laps)]):
                    l = np.arange(s_l, p_lap + 1)
                    v = r["piecewise_life"][s_l - 1:p_lap]
                    fig_desgaste.add_trace(
                        go.Scatter(
                            x=l,
                            y=v,
                            name=f"Estrategia {chr(65+i)} - Stint {j+1}",
                            line=dict(color=color, width=3, dash="solid" if i == 0 else "dash"),
                        )
                    )
                    s_l = p_lap + 1

            fig_desgaste.add_hrect(
                y0=0,
                y1=30,
                fillcolor="#ff4b4b",
                opacity=0.12,
                layer="below",
                line_width=0,
            )

        fig_desgaste.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            uirevision="wear_live",
            transition={"duration": 0},
        )
        fig_desgaste.update_yaxes(range=[0, 105], title_text="Vida útil %")
        fig_desgaste.update_xaxes(title_text="Vuelta")
        st.plotly_chart(fig_desgaste, use_container_width=True, key="grafico_desgaste_fijo")

        with st.popover("Ver explicación del desgaste y Simpson"):
            st.markdown(load_text_asset("simpson"))
            st.write(
                "El desgaste no se presenta solo como un porcentaje: en la lectura de la materia funciona como una variable de estado que afecta el comportamiento global del sistema."
            )

    st.divider()

    st.subheader("Bloque de lectura técnica rápida")
    st.write(
        "Estos bloques te sirven para la defensa oral y para reforzar la justificación académica del trabajo. Cada uno conecta un método numérico con su función dentro del simulador."
    )
    c1, c2 = st.columns(2)
    with c1:
        academic_method_card(
            "Ajuste de curvas",
            "Convierte telemetría ruidosa en una representación continua del comportamiento del neumático.",
            "least_squares",
            icon="📈",
        )
        academic_method_card(
            "Runge-Kutta",
            "Resuelve la ecuación térmica de forma numérica y estable.",
            "rk45",
            icon="🌡️",
        )
    with c2:
        academic_method_card(
            "Simpson",
            "Integra el ritmo para obtener tiempo total de carrera o de stint.",
            "simpson",
            icon="∫",
        )
        academic_method_card(
            "Newton-Raphson",
            "Localiza el punto de cruce entre dos estrategias.",
            "newton",
            icon="🧭",
        )


render_live_timing_view()
