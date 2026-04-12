import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from extractor_datos import obtener_datos_aumentados, COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG
from modelo_ml import PredictorDegradacion
from strategy import EngineEstrategia

def aplicar_estilos():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Tema Dark-Premium */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Ocultar UI por defecto */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Tarjetas de Métricas (KPIs) */
        div[data-testid="metric-container"] {
            background-color: #151a28;
            border: 1px solid #1e253c;
            padding: 15px 20px;
            border-radius: 12px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #00d4ff;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        }

        /* Ajustes de tipografía en métricas */
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Configuración de Estética Premium ---
st.set_page_config(page_title="F1 Strategy Wall", layout="wide", page_icon="🏎️")
aplicar_estilos()

# --- Título Superior ---
st.markdown("<h1 style='text-align: center; color: #00d4ff; margin-bottom: 0;'>🏎️ F1 STRATEGY CONTROL WALL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-weight: 500; margin-top: 0;'>Enterprise Race Engineering Suite</p>", unsafe_allow_html=True)

# --- Sidebar: Estrategia ---
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

current_lap_ui = st.sidebar.slider("📍 Vuelta Actual", 1, total_race_laps, 1)

# --- Backend Pipeline ---
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

predictor = entrenar_modelo_ia(track_name)
engine = EngineEstrategia(predictor, track_name=track_name)
engine.pit_loss = pit_loss

# OPTIMIZACIÓN MULTI-STINT
resultado_opt = engine.optimizador_determinista(lista_compuestos, total_race_laps, track_temp)
vueltas_parada = resultado_opt['vueltas_optimas']
tiempos_race = resultado_opt['race_trace']
comp_trace = resultado_opt['compounds_trace']

# RECOPILACIÓN PIECEWISE (Temp y Vida Útil)
piecewise_temps = []
piecewise_life = []

start_idx = 0
for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
    comp = lista_compuestos[i]
    duracion = p_lap - (vueltas_parada[i-1] if i > 0 else 0)
    
    # Simulamos sector térmico
    engine.compound = comp
    _, _, t_sector = engine.simular_stint(duracion, track_temp)
    piecewise_temps.extend(t_sector.tolist())
    
    # Calculamos sector de vida útil
    phys_comp = COMPOUNDS_PHYSICS.get(comp, {'max_life': 40})
    track_abr = CIRCUITOS_CONFIG.get(track_name, {'abrasion': 1.0})['abrasion']
    m_life_ef = phys_comp['max_life'] * (1.0 / track_abr)
    l_sector = [max(0, 100 * (1 - (v / m_life_ef)**3)) for v in range(1, duracion + 1)]
    piecewise_life.extend(l_sector)

# --- UI: Veredicto ---
st.divider()
p_text = " y ".join([f"Vuelta {v}" for v in vueltas_parada])
st.success(f"🏁 **ESTRATEGIA ÓPTIMA:** Parada en {p_text} | Tiempo de Carrera: {resultado_opt['tiempo_total']:.2f}s")

# --- GRID DE MÉTRICAS ---
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Goma Actual", comp_trace[current_lap_ui-1])
with m2: st.metric("🌡️ Temp. Goma", f"{piecewise_temps[current_lap_ui-1]:.1f}°C")
with m3: st.metric("⏱️ Lap Time", f"{tiempos_race[current_lap_ui-1]:.3f}s")
with m4: st.metric("📉 Vida Restante", f"{piecewise_life[current_lap_ui-1]:.1f}%")

st.divider()

# --- GRID DE GRÁFICOS (2x2) ---
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

col_map = {'SOFT': '#ff3333', 'MEDIUM': '#ffff33', 'HARD': '#f0f0f0'}

# --- FILA 1, COL 1: RACE TRACE ---
with row1_col1:
    st.subheader("⏱️ Ritmo Segmentado")
    fig1 = go.Figure()
    s_l = 1
    for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
        laps = np.arange(s_l, p_lap + 1)
        vals = tiempos_race[s_l-1:p_lap]
        fig1.add_trace(go.Scatter(x=laps, y=vals, name=lista_compuestos[i], line=dict(color=col_map.get(lista_compuestos[i]), width=4)))
        s_l = p_lap + 1
    fig1.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig1, use_container_width=True)

# --- FILA 1, COL 2: PROYECCIÓN ACUMULADA ---
with row1_col2:
    st.subheader("📊 Tiempo Acumulado")
    fig2 = go.Figure()
    acumulado = np.cumsum(tiempos_race)
    # Inyectamos el pit loss en los saltos para que la proyección sea real
    for i, v in enumerate(vueltas_parada):
        acumulado[v:] += pit_loss
    fig2.add_trace(go.Scatter(x=np.arange(1, total_race_laps + 1), y=acumulado, name="Race Progress", fill='tozeroy', line=dict(color='#00d4ff')))
    fig2.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# --- FILA 2, COL 1: EVOLUCIÓN TÉRMICA ---
with row2_col1:
    st.subheader("🌡️ Evolución Térmica (RK45)")
    fig3 = go.Figure()
    s_l = 1
    for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
        laps = np.arange(s_l, p_lap + 1)
        vals = piecewise_temps[s_l-1:p_lap]
        fig3.add_trace(go.Scatter(x=laps, y=vals, name=f"Stint {i+1}", line=dict(color=col_map.get(lista_compuestos[i]), width=3)))
        # Línea de ventana ideal
        win = COMPOUNDS_PHYSICS.get(lista_compuestos[i], {'ventana_temp': (90,110)})['ventana_temp']
        fig3.add_hrect(y0=win[0], y1=win[1], fillcolor="green", opacity=0.05, line_width=0)
        s_l = p_lap + 1
    fig3.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

# --- FILA 2, COL 2: DESGASTE MECÁNICO ---
with row2_col2:
    st.subheader("📉 Desgaste Mecánico (%)")
    fig4 = go.Figure()
    s_l = 1
    for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
        laps = np.arange(s_l, p_lap + 1)
        vals = piecewise_life[s_l-1:p_lap]
        fig4.add_trace(go.Scatter(x=laps, y=vals, name=f"Stint {i+1}", line=dict(color=col_map.get(lista_compuestos[i]), width=3, dash='dash')))
        s_l = p_lap + 1
    fig4.add_hline(y=30, line_dash="dash", line_color="red") # Umbral de Cliff
    fig4.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig4, use_container_width=True)
