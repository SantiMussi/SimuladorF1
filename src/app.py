import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from extractor_datos import obtener_datos_aumentados, COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG
from modelo_ml import PredictorDegradacion
from strategy import EngineEstrategia

def format_time(total_seconds):
    """Formatea segundos en MM:SS.ms o HH:MM:SS.ms si supera la hora."""
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

def aplicar_estilos():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp { background-color: #0b0f19; color: #e2e8f0; }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
        div[data-testid="metric-container"] {
            background-color: #151a28; border: 1px solid #1e253c; padding: 15px 20px; border-radius: 12px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="metric-container"]:hover { transform: translateY(-2px); border-color: #00d4ff; }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.9rem !important; font-weight: 600 !important; }
        div[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700 !important; }
        [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
        </style>
    """, unsafe_allow_html=True)

# --- Configuración Inicial ---
if 'vuelta_actual' not in st.session_state:
    st.session_state.vuelta_actual = 1.0
if 'playing' not in st.session_state:
    st.session_state.playing = False

st.set_page_config(page_title="F1 Strategy Wall", layout="wide", page_icon="🏎️")
aplicar_estilos()

# --- Título ---
st.markdown("<h1 style='text-align: center; color: #00d4ff; margin-bottom: 0;'>🏎️ F1 STRATEGY CONTROL WALL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-weight: 500; margin-top: 0;'>Enterprise Race Engineering Suite</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🛠️ Race Strategy")
track_name = st.sidebar.selectbox("🏎️ Circuito", list(CIRCUITOS_CONFIG.keys()))
track_settings = CIRCUITOS_CONFIG.get(track_name, {})
track_temp = st.sidebar.slider("🌡️ Temperatura de Pista (°C)", 15, 60, 35)
total_race_laps = st.sidebar.number_input("🏁 Vueltas Totales", 30, 90, track_settings.get('total_laps', 50))
pit_loss = st.sidebar.slider("⏱️ Costo de Pit Stop (s)", 15.0, 30.0, track_settings.get('pit_loss', 22.0))

st.sidebar.markdown("---")
tipo_estrategia = st.sidebar.radio("Tipo de Estrategia", ["1 Parada", "2 Paradas"], horizontal=True)
if tipo_estrategia == "1 Parada":
    s1 = st.sidebar.selectbox("Stint 1", list(COMPOUNDS_PHYSICS.keys()), index=1)
    s2 = st.sidebar.selectbox("Stint 2", list(COMPOUNDS_PHYSICS.keys()), index=2)
    lista_compuestos = [s1, s2]
else:
    s1 = st.sidebar.selectbox("Stint 1", list(COMPOUNDS_PHYSICS.keys()), index=0)
    s2 = st.sidebar.selectbox("Stint 2", list(COMPOUNDS_PHYSICS.keys()), index=1)
    s3 = st.sidebar.selectbox("Stint 3", list(COMPOUNDS_PHYSICS.keys()), index=2)
    lista_compuestos = [s1, s2, s3]

st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Reproducción Live Timing")
col_play, col_stop = st.sidebar.columns(2)
if col_play.button("▶ Iniciar Simulación"):
    st.session_state.playing = True
if col_stop.button("⏸ Detener"):
    st.session_state.playing = False

st.session_state.vuelta_actual = st.sidebar.slider("📍 Vuelta Actual", 1.0, float(total_race_laps), value=st.session_state.vuelta_actual, step=0.1)

# --- Backend Pipeline Caching ---
@st.cache_data
def cargar_datos_entrenamiento(track_name):
    return obtener_datos_aumentados(track_name=track_name)

@st.cache_resource
def entrenar_modelo_ia(track_name):
    df = cargar_datos_entrenamiento(track_name)
    predictor = PredictorDegradacion()
    base_time = CIRCUITOS_CONFIG.get(track_name, {}).get('base_lap_time', 90.0)
    predictor.entrenar(df, track_base_time=base_time)
    return predictor

@st.cache_data
def get_sim_data(track_name, track_temp, total_race_laps, pit_loss, lista_compuestos):
    predictor = entrenar_modelo_ia(track_name)
    engine = EngineEstrategia(predictor, track_name=track_name)
    engine.pit_loss = pit_loss
    resultado_opt = engine.optimizador_determinista(list(lista_compuestos), total_race_laps, track_temp)
    vueltas_parada = resultado_opt['vueltas_optimas']
    tiempos_race = resultado_opt['race_trace']
    comp_trace = resultado_opt['compounds_trace']
    piecewise_temps, piecewise_life = [], []
    for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
        comp = lista_compuestos[i]
        duracion = p_lap - (vueltas_parada[i-1] if i > 0 else 0)
        engine.compound = comp
        _, _, t_sector = engine.simular_stint(duracion, track_temp)
        piecewise_temps.extend(t_sector.tolist())
        phys_comp = COMPOUNDS_PHYSICS.get(comp, {'max_life': 40})
        track_abr = CIRCUITOS_CONFIG.get(track_name, {'abrasion': 1.0})['abrasion']
        m_life_ef = phys_comp['max_life'] * (1.0 / track_abr)
        l_sector = [max(0, 100 * (1 - (v / m_life_ef)**3)) for v in range(1, duracion + 1)]
        piecewise_life.extend(l_sector)
    return {
        'resultado_opt': resultado_opt, 'vueltas_parada': vueltas_parada,
        'tiempos_race': tiempos_race, 'comp_trace': comp_trace,
        'piecewise_temps': piecewise_temps, 'piecewise_life': piecewise_life,
        'laps_range': np.arange(1, total_race_laps + 1)
    }

data = get_sim_data(track_name, track_temp, total_race_laps, pit_loss, tuple(lista_compuestos))
laps_range = data['laps_range']
resultado_opt = data['resultado_opt']
vueltas_parada = data['vueltas_parada']
tiempos_race = data['tiempos_race']
comp_trace = data['comp_trace']
piecewise_temps = data['piecewise_temps']
piecewise_life = data['piecewise_life']
col_map = {'SOFT': '#ff3333', 'MEDIUM': '#ffff33', 'HARD': '#f0f0f0'}

# --- FRAGMENTO DE VISUALIZACIÓN DINÁMICA ---
@st.fragment(run_every="50ms")
def render_live_timing_view():
    if st.session_state.playing and st.session_state.vuelta_actual < total_race_laps:
        st.session_state.vuelta_actual += 0.1
    elif st.session_state.vuelta_actual >= total_race_laps:
        st.session_state.playing = False

    v_act = st.session_state.vuelta_actual
    idx = max(0, min(int(v_act) - 1, total_race_laps - 1))

    st.divider()
    p_text = " y ".join([f"Vuelta {v}" for v in vueltas_parada])
    st.success(f"🏁 **ESTRATEGIA ÓPTIMA:** Parada en {p_text} | Tiempo de Carrera: {format_time(resultado_opt['tiempo_total'])}")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Goma Actual", comp_trace[idx])
    with m2: st.metric("🌡️ Temp. Goma", f"{piecewise_temps[idx]:.1f}°C")
    with m3: st.metric("⏱️ Lap Time", format_time(tiempos_race[idx]))
    with m4: st.metric("📉 Vida Restante", f"{piecewise_life[idx]:.1f}%")

    prog = v_act / total_race_laps
    st.progress(min(1.0, prog))
    st.markdown(f"<p style='text-align: center; color: #00d4ff; font-weight: bold;'>🏁 Progreso: Vuelta {v_act:.1f} / {total_race_laps} ({prog*100:.1f}%)</p>", unsafe_allow_html=True)

    st.divider()
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        st.subheader("⏱️ Ritmo Segmentado")
        fig1 = go.Figure()
        s_l = 1
        for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
            laps = np.arange(s_l, p_lap + 1)
            vals = tiempos_race[s_l-1:p_lap]
            fig1.add_trace(go.Scatter(x=laps, y=vals, name=lista_compuestos[i], line=dict(color=col_map.get(lista_compuestos[i]), width=4)))
            s_l = p_lap + 1
        y_int = np.interp(v_act, laps_range, tiempos_race)
        fig1.add_trace(go.Scatter(x=[v_act], y=[y_int], mode='markers', marker=dict(symbol='star', size=18, color='#00d4ff', line=dict(color='white', width=2)), name='Posición'))
        fig1.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True, key="f1")

    with r1c2:
        st.subheader("📊 Tiempo Acumulado")
        fig2 = go.Figure()
        acum = np.cumsum(tiempos_race)
        for i, v in enumerate(vueltas_parada): acum[v:] += pit_loss
        fig2.add_trace(go.Scatter(x=laps_range, y=acum, fill='tozeroy', line=dict(color='#00d4ff')))
        fig2.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True, key="f2")

    with r2c1:
        st.subheader("🌡️ Evolución Térmica")
        fig3 = go.Figure()
        s_l = 1
        for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
            l = np.arange(s_l, p_lap + 1)
            v = piecewise_temps[s_l-1:p_lap]
            fig3.add_trace(go.Scatter(x=l, y=v, name=f"Stint {i+1}", line=dict(color=col_map.get(lista_compuestos[i]), width=3)))
            s_l = p_lap + 1
        fig3.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True, key="f3")

    with r2c2:
        st.subheader("📉 Desgaste Mecánico (%)")
        fig4 = go.Figure()
        s_l = 1
        for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
            l = np.arange(s_l, p_lap + 1)
            v = piecewise_life[s_l-1:p_lap]
            fig4.add_trace(go.Scatter(x=l, y=v, name=f"Stint {i+1}", line=dict(color=col_map.get(lista_compuestos[i]), width=3, dash='dash')))
            s_l = p_lap + 1
        fig4.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True, key="f4")

render_live_timing_view()
