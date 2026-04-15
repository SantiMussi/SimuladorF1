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


def aplicar_estilos():
    bg_path = asset_path("assets", "f1_bg.png")
    bg_uri = image_to_data_uri(bg_path) if os.path.exists(bg_path) else ""
    bg_css = f"background-image: url('{bg_uri}'); background-size: cover; background-position: center; background-attachment: fixed; background-blend-mode: overlay;" if bg_uri else ""
    
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;800&display=swap');
        
        /* Global App Styles */
        .stApp {{ 
            background-color: #0b0b0f; 
            color: #ffffff; 
            {bg_css} 
        }}
        
        html, body, [class*="css"] {{ 
            font-family: 'Titillium Web', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
        }}

        /* Hide Streamlit elements */
        #MainMenu {{visibility: hidden;}} 
        footer {{visibility: hidden;}}
        header {{background-color: transparent !important; border-bottom: none !important;}}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

        /* Sidebar Styling - F1 Pit Wall Look */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0f0f14 0%, #15151e 100%) !important;
            border-right: 2px solid #e10600 !important;
            box-shadow: 5px 0 25px rgba(0,0,0,0.5);
        }}
        
        [data-testid="stSidebarUserContent"] {{
            padding-top: 1rem;
        }}

        /* Sidebar Headers */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .st-emotion-cache-10lo6u4 {{
            color: #e10600 !important;
            font-family: 'Titillium Web', sans-serif !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            border-left: 4px solid #e10600;
            padding-left: 12px;
            margin-top: 25px;
            margin-bottom: 15px;
        }}

        /* Sidebar Labels */
        [data-testid="stSidebar"] label {{
            color: #a1a1aa !important;
            font-weight: 700 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 4px;
        }}

        /* Sidebar Widgets */
        [data-testid="stSidebar"] div[data-baseweb="select"],
        [data-testid="stSidebar"] div[data-baseweb="input"] {{
            background-color: #1a1a24 !important;
            border: 1px solid #33333e !important;
            border-radius: 4px !important;
        }}
        
        [data-testid="stSidebar"] .stSlider {{
            padding-bottom: 15px;
        }}

        /* Sidebar Expanders */
        [data-testid="stSidebar"] .stExpander {{
            background-color: rgba(26, 26, 36, 0.6) !important;
            border: 1px solid #33333e !important;
            border-radius: 6px !important;
            margin-bottom: 10px;
        }}
        
        [data-testid="stSidebar"] .stExpander summary {{
            color: #ffffff !important;
            font-weight: 700 !important;
        }}

        /* Radio buttons in sidebar */
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
            font-size: 0.85rem;
        }}

        /* Metrics */
        div[data-testid="metric-container"] {{
            background-color: rgba(21, 21, 30, 0.85); 
            border: 1px solid #e10600; 
            padding: 15px 20px; 
            border-radius: 8px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
            box-shadow: 0 4px 6px -1px rgba(225, 6, 0, 0.2);
            backdrop-filter: blur(8px);
        }}
        div[data-testid="metric-container"]:hover {{ 
            transform: translateY(-2px); 
            border-color: #ff1801; 
            box-shadow: 0 8px 15px -3px rgba(225, 6, 0, 0.4); 
        }}
        div[data-testid="stMetricLabel"] {{ 
            color: #a1a1aa !important; 
            font-size: 0.85rem !important; 
            font-weight: 600 !important; 
            text-transform: uppercase; 
            letter-spacing: 1.2px; 
        }}
        div[data-testid="stMetricValue"] {{ 
            color: #ffffff !important; 
            font-weight: 800 !important; 
        }}

        /* Cards and general boxes */
        .card-box {{
            background: rgba(25, 25, 35, 0.8);
            border: 1px solid #33333e;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        .small-muted {{
            color: #a1a1aa;
            font-size: 0.85rem;
        }}

        /* Custom buttons */
        .stButton button {{
            background-color: #e10600 !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 4px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.2s !important;
        }}
        
        .stButton button:hover {{
            background-color: #ff1801 !important;
            box-shadow: 0 0 15px rgba(225, 6, 0, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        /* Divider */
        hr {{
            border-top: 1px solid #33333e !important;
            margin: 1.5rem 0 !important;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )


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


def image_to_data_uri(path):
    """Convierte una imagen local a data URI para usarla en Plotly."""
    if not path or not os.path.exists(path):
        return None
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
    """
    Devuelve:
    - current_lap_float: vuelta equivalente actual (1-based)
    - current_lap_idx: índice 0-based
    - lap_fraction: fracción dentro de la vuelta actual
    - progress_race: progreso total en 0..1
    """
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


def build_track_figure(track_name, track_geom, strategy_states, compare_mode=False):
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

        # Hover invisible para que siga mostrando info aunque el ícono sea una imagen
        fig.add_trace(
            go.Scatter(
                x=[x_car],
                y=[y_car],
                mode="markers",
                marker=dict(
                    size=42,
                    color="rgba(0,0,0,0)",
                    line=dict(width=0),
                ),
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
            # Fallback si no hay imagen
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

    # =========================
    # Estados para el mapa
    # =========================
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

    # Gap entre estrategias en el mismo punto relativo de carrera
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
    track_fig = build_track_figure(
        track_name=track_name,
        track_geom=track_geom,
        strategy_states=strategy_states,
        compare_mode=(len(strategy_results) > 1),
    )
    st.plotly_chart(track_fig, use_container_width=True, key="track_map")

    # =========================
    # Panel de métricas
    # =========================
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

    if len(strategy_results) > 1:
        gap_fig = build_comparison_gap_figure(strategy_results[0], strategy_results[1])
        if gap_fig is not None:
            st.subheader("Diferencia acumulada entre estrategias")
            st.plotly_chart(gap_fig, use_container_width=True, key="gap_compare")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolución Térmica")
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

    with col2:
        st.subheader("Vida útil (%)")
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


render_live_timing_view()