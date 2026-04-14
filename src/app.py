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

def aplicar_estilos():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp { background-color: #0b0f19; color: #e2e8f0; }
        html, body, [class*="css"] { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
        }
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}
        header {background-color: transparent !important; border-bottom: none !important;}
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
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

st.set_page_config(page_title="F1 Strategy Wall", layout="wide", page_icon="F1")
aplicar_estilos()

# --- Título ---
st.markdown("<h1 style='text-align: center; color: #00d4ff; margin-bottom: 0;'>F1 STRATEGY CONTROL WALL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-weight: 500; margin-top: 0;'>Enterprise Race Engineering Suite</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Race Strategy")
track_name = st.sidebar.selectbox("Circuito", list(CIRCUITOS_CONFIG.keys()))
track_settings = CIRCUITOS_CONFIG.get(track_name, {})
track_temp = st.sidebar.slider("Temperatura de Pista (°C)", 15, 60, 35)
total_race_laps = st.sidebar.number_input("Vueltas Totales", 30, 90, track_settings.get('total_laps', 50))
pit_loss = st.sidebar.slider("Costo de Pit Stop (s)", 15.0, 30.0, track_settings.get('pit_loss', 22.0))

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
st.sidebar.subheader("Reproducción Live Timing")
col_play, col_stop = st.sidebar.columns(2)
if col_play.button("Iniciar Simulación"):
    st.session_state.playing = True
if col_stop.button("Detener"):
    st.session_state.playing = False

st.session_state.vuelta_actual = st.sidebar.slider("Vuelta Actual", 1.0, float(total_race_laps), value=st.session_state.vuelta_actual, step=0.1)

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
        _, _, t_sector, l_sector = engine.simular_stint(duracion, track_temp)
        piecewise_temps.extend(t_sector.tolist())
        piecewise_life.extend(l_sector.tolist())
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
    if not np.isfinite(resultado_opt['tiempo_total']):
        st.error("ESTRATEGIA NO VIABLE: Ninguna combinación cumple con el Límite de Seguridad Estructural (70% vida).")
        return

    p_text = " y ".join([f"Vuelta {v}" for v in vueltas_parada])
    st.success(f"ESTRATEGIA ÓPTIMA: Parada en {p_text} | Tiempo de Carrera: {format_time(resultado_opt['tiempo_total'])}")

    # --- CÁLCULO DE MÉTRICAS AVANZADAS ---
    # Pace Delta: Tiempo_Vuelta_Actual - Mejor_Tiempo_Del_Stint_Actual
    stint_start = 0
    for p in vueltas_parada:
        if int(v_act) > p:
            stint_start = p
        else:
            break
    
    # 2. Delta respecto a la vuelta anterior (Live Pace)
    # Si es la primera vuelta del stint, el delta es 0.0
    if idx > stint_start:
        pace_delta = tiempos_race[idx] - tiempos_race[idx-1]
    else:
        pace_delta = 0.0

    # 3. Fuel Effect (Opcional, si querés mantenerlo en la métrica 5)
    fuel_benefit = -(v_act * 0.04)

    # 4. Renderizado de métricas
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Goma Actual", comp_trace[idx])
    with m2: st.metric("Temperatura Goma", f"{piecewise_temps[idx]:.1f}°C")
    with m3: 
        st.metric(
            "Lap Time", 
            format_time(tiempos_race[idx]), 
            delta=f"{pace_delta:+.3f}s", 
            delta_color="inverse"
        )
    with m4: st.metric("Vida Restante", f"{piecewise_life[idx]:.1f}%")

    prog = v_act / total_race_laps
    st.progress(min(1.0, prog))
    st.markdown(f"<p style='text-align: center; color: #00d4ff; font-weight: bold;'>Progreso: Vuelta {v_act:.1f} / {total_race_laps} ({prog*100:.1f}%)</p>", unsafe_allow_html=True)

    st.divider()
    
    # --- REESTRUCTURACIÓN DE LA GRILLA DE GRÁFICOS ---
    # Fila 1: Ritmo Segmentado (100% Ancho)
    st.subheader("Ritmo Segmentado (Pace Analysis)")
    fig1 = go.Figure()
    
    # Agregar líneas verticales de PIT STOP primero para que estén al fondo
    for p_lap in vueltas_parada:
        fig1.add_vline(x=p_lap, line_width=2, line_dash="dash", line_color="#ff4b4b", opacity=0.6)
        fig1.add_annotation(x=p_lap, y=0.05, yref="paper", text="<b>BOX</b>", showarrow=False, 
                           font=dict(color="#ff4b4b", size=12), bgcolor="rgba(15, 23, 42, 0.8)")

    s_l = 1
    for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
        laps = np.arange(s_l, p_lap + 1)
        vals = tiempos_race[s_l-1:p_lap]
        fig1.add_trace(go.Scatter(x=laps, y=vals, name=f"Stint {i+1}: {lista_compuestos[i]}", 
                                 line=dict(color=col_map.get(lista_compuestos[i]), width=4)))
        s_l = p_lap + 1
        
    y_int = np.interp(v_act, laps_range, tiempos_race)
    fig1.add_trace(go.Scatter(x=[v_act], y=[y_int], mode='markers', 
                             marker=dict(symbol='star', size=18, color='#00d4ff', 
                             line=dict(color='white', width=2)), name='Posición Actual'))
    
    fig1.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', 
                      height=400, margin=dict(l=20, r=20, t=20, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig1.update_yaxes(title_text="Lap Time (s)")
    fig1.update_xaxes(title_text="Vuelta")
    st.plotly_chart(fig1, width="stretch", key="f_ritmo")

    # Fila 2: Evolución Térmica y Desgaste Mecánico (50/50)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolución Térmica")
        fig_termica = go.Figure()
        
        for p_lap in vueltas_parada:
            fig_termica.add_vline(x=p_lap, line_width=1, line_dash="dash", line_color="#ff4b4b", opacity=0.4)

        s_l = 1
        for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
            l = np.arange(s_l, p_lap + 1)
            v = piecewise_temps[s_l-1:p_lap]
            fig_termica.add_trace(go.Scatter(x=l, y=v, name=f"Stint {i+1}", 
                                             line=dict(color=col_map.get(lista_compuestos[i]), width=3)))
            s_l = p_lap + 1
        fig_termica.update_layout(template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19', 
                                  height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_termica, width="stretch", key="f_temp_live")

    with col2:
        st.subheader("Desgaste Mecánico (%)")
        if len(piecewise_life) > 0:
            fig_desgaste = go.Figure()

            # Líneas de PIT STOP
            for p_lap in vueltas_parada:
                fig_desgaste.add_vline(x=p_lap, line_width=1, line_dash="dash", line_color="#ff4b4b", opacity=0.4)

            # LÍMITE DE SEGURIDAD ESTRUCTURAL (70%)
            # Mostramos una franja roja para indicar peligro por debajo del 30% de vida restante
            fig_desgaste.add_hrect(y0=0, y1=30, fillcolor="#ff4b4b", opacity=0.15, layer="below", line_width=0)
            fig_desgaste.add_annotation(x=total_race_laps/2, y=15, text="ZONA DE RIESGO ESTRUCTURAL", 
                                       showarrow=False, font=dict(color="#ff4b4b", size=10), opacity=0.8)

            s_l = 1
            for i, p_lap in enumerate(vueltas_parada + [total_race_laps]):
                l = np.arange(s_l, p_lap + 1)
                v = piecewise_life[s_l-1:p_lap]
                fig_desgaste.add_trace(go.Scatter(
                    x=l, y=v, 
                    name=f"Stint {i+1}", 
                    line=dict(color=col_map.get(lista_compuestos[i]), width=3)
                ))
                s_l = p_lap + 1
                
            fig_desgaste.update_layout(
                template="plotly_dark", 
                paper_bgcolor='#0b0f19', 
                plot_bgcolor='#0b0f19', 
                height=300, 
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            fig_desgaste.update_yaxes(range=[0, 105], title_text="Vida útil %")
            fig_desgaste.update_xaxes(title_text="Vuelta")
            
            st.plotly_chart(fig_desgaste, width="stretch", key="grafico_desgaste_fijo")

render_live_timing_view()
