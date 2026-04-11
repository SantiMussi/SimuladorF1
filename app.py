import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Importamos nuestro backend pesado
from src.extractor_datos import obtener_telemetria_limpia
from src.modelo_ml import entrenar_modelo_degradacion
from src.strategy import MotorEstrategia
from src.metodos_numericos import *

# Configuración de página
st.set_page_config(page_title="F1 Pit Optimizer", page_icon="🏎️", layout="wide")

#Cargar CSS externo (Si no existe, no rompe la app)
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass # Si todavía no creaste el CSS, sigue de largo

#Header
st.markdown("""
<div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:0.25rem">
  <span style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:2.2rem; letter-spacing:0.08em; text-transform:uppercase; color:#fff">
    🏎 F1 PIT OPTIMIZER
  </span>
  <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#E53935; letter-spacing:0.2em; padding-bottom:0.2rem">
    v2.4.1 · RACE MODE
  </span>
</div>
<div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#8B949E; letter-spacing:0.12em; margin-bottom:0.75rem">
  STRATEGY SIMULATION SYSTEM · NUMERICAL METHODS ENGINE
</div>
<hr>
""", unsafe_allow_html=True)

# ─── INICIALIZAR ESTADO (Persistencia de IA y Datos) ───
if 'simulacion_ejecutada' not in st.session_state:
    st.session_state.simulacion_ejecutada = False
    st.session_state.resultados = None
    st.session_state.modelo = None
    st.session_state.cols = None
    st.session_state.df = None
    st.session_state.precision_modelo = 0.0
    st.session_state.is_demo = False

# Sidebar 
with st.sidebar:
    st.header("⚙️ Parámetros")
    mapa_gp = {"Bahréin": "Bahrain", "Monza": "Monza", "Silverstone": "Silverstone"}
    gp_seleccionado = st.selectbox("Gran Premio", list(mapa_gp.keys()))
    compuesto = st.selectbox("Compuesto Inicial", ["Blando", "Medio", "Duro"])
    
    st.markdown("---")
    temp_pista = st.slider("Temp. de Pista (°C)", 20.0, 60.0, 35.0, step=1.0)
    vuelta_actual = st.slider("Vuelta Actual (Stint)", 1, 40, 15)
    
    st.markdown("<br>", unsafe_allow_html=True)
    btn_simular = st.button("▶  Ejecutar Simulación", type="primary", use_container_width=True)

# ─── LOGICA DE CONTROL DE ESTADO ───
# Si el usuario hace clic en simular o no hay modelo cargado, entrenamos.
if btn_simular:
    with st.spinner(f"📡 Descargando telemetría de {gp_seleccionado}..."):
        df = obtener_telemetria_limpia(2024, mapa_gp[gp_seleccionado], 'R', 'VER', temp_pista)
        st.session_state.df = df
        
    with st.spinner("🧠 Entrenando Inteligencia Artificial..."):
        modelo, _, cols, error, precision = entrenar_modelo_degradacion(df)
        st.session_state.modelo = modelo
        st.session_state.cols = cols
        st.session_state.precision_modelo = precision
        st.session_state.is_demo = 'IsDemo' in df.columns
        st.session_state.simulacion_ejecutada = True

# Si ya tenemos un modelo (fue entrenado antes o recién), recalculamos el stint
# Esto se dispara AUTOMÁTICAMENTE si el usuario mueve el slider, porque Streamlit recarga el script
# pero como el modelo está en session_state, no volvemos a entrar al bloque 'if btn_simular'
if st.session_state.simulacion_ejecutada:
    # Usamos el modelo y columnas guardadas en el estado
    motor = MotorEstrategia(st.session_state.modelo, st.session_state.cols)
    
    # Recalculamos SOLO la matemática del stint (Simpson, Newton-Raphson, etc.)
    # Esto es ligero y permite interactividad fluida con el slider
    st.session_state.resultados = motor.analizar_stint(compuesto, vuelta_actual, temp_pista, umbral_perdida=1.5)

# ─── INTERFAZ DINÁMICA ───
if st.session_state.simulacion_ejecutada:
    if st.session_state.get('is_demo', False):
        st.warning("⚠️ **MODO DEMO ACTIVADO**: Los servidores de F1 no responden. Se están usando datos simulados realistas para el análisis.")
    
    res = st.session_state.resultados
    
    # Métricas actualizadas con la matemática real
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Temp. de Pista", value=f"{temp_pista} °C", delta=f"R2 IA: {st.session_state.precision_modelo*100:.1f}%", delta_color="normal")
    with col2:
        st.metric(label="Lap Time Actual", value=f"{res['tiempo_actual']:.2f}s", delta=f"+{res['delta_tiempo']:.2f}s (Deg.)", delta_color="inverse")
    with col3:
        # Mostramos la temperatura calculada por RK4
        st.metric(label="Temp. Neumático (RK4)", value=f"{res['temp_goma']:.1f} °C", delta="90°C - 110°C", delta_color="off")
    with col4:
        # El veredicto de Newton-Raphson
        st.metric(label="Crossover Point", value=f"Vuelta {res['crossover_point']}", delta="Vía Newton-Raphson", delta_color="off")
    with col5:
        # El resultado de la integración de Simpson 1/3
        st.metric(label="Tiempo Total Stint", value=f"{res['tiempo_total']:.1f}s", delta="Vía Simpson 1/3", delta_color="off")

    st.markdown("---")

    # Gráfico Plotly con eje dual (Tiempos y Temperatura)
    st.subheader("📊 Análisis Híbrido: Degradación Predicha vs. Evolución Térmica")
    
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Línea principal del modelo (Eje Y Primario)
    color_linea = '#E53935' if compuesto == 'Blando' else '#FDD835' if compuesto == 'Medio' else '#FFFFFF'
    fig.add_trace(go.Scatter(
        x=res['grafico_x'], y=res['grafico_y'], 
        mode='lines', name=f'Lap Time ({compuesto})',
        line=dict(color=color_linea, width=4)
    ), secondary_y=False)
    
    # Perfil térmico RK4 (Eje Y Secundario)
    # Re-calculamos brevemente para el gráfico completo
    from src.extractor_datos import CONSTANTES_NEUMATICOS
    try:
        motor = MotorEstrategia(None, []) # Solo para acceder al mapa
        comp_en = motor.mapa_compuestos.get(compuesto, 'SOFT')
        edo = crear_edo_temperatura(temp_pista, comp_en)
        v_perfil, t_perfil = runge_kutta_4(edo, t0=0, y0=80.0, t_end=60, h=1)
        
        fig.add_trace(go.Scatter(
            x=v_perfil, y=t_perfil,
            mode='lines', name='Temp. RK4 (°C)',
            line=dict(color='rgba(0, 255, 255, 0.4)', width=2, dash='dot')
        ), secondary_y=True)
        
        # Banda de temperatura óptima
        fig.add_hrect(y0=90, y1=110, secondary_y=True, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Banda Óptima", annotation_position="top right")
    except:
        pass

    # Marcador de donde estamos ahora
    fig.add_trace(go.Scatter(
        x=[vuelta_actual], y=[res['tiempo_actual']],
        mode='markers', name='Posición Actual',
        marker=dict(color='#2196F3', size=14, symbol='x', line=dict(width=2, color="white"))
    ), secondary_y=False)
    
    # Línea vertical del Crossover Point
    fig.add_vline(x=res['crossover_point'], line_width=2, line_dash="dash", line_color="#FF9800")
    fig.add_annotation(x=res['crossover_point'], y=max(res['grafico_y']), text="CRITICAL STOP", showarrow=False, yshift=15, font=dict(color="#FF9800", weight="bold"))

    fig.update_layout(
        xaxis_title="Vueltas de Uso (Tyre Life)", 
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Tiempo de Vuelta (s)", secondary_y=False)
    fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=True, range=[60, 130])
    
    st.plotly_chart(fig, use_container_width=True)

else:
    # Pantalla de espera antes de simular
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Temperatura de Pista", value="-")
    col2.metric(label="Lap Time Actual", value="-")
    col3.metric(label="Temperatura Neumático (RK4)", value="-")
    col4.metric(label="Crossover Point", value="-")
    
    st.markdown("---")
    st.subheader("📊 Curvas de Degradación")
    st.info("Ajustá los parámetros en el panel lateral y hacé clic en 'Ejecutar Simulación' para visualizar el análisis de Lab 1.")