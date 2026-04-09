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

# ─── INICIALIZAR ESTADO (Para no mostrar errores antes de simular) ───
if 'simulacion_ejecutada' not in st.session_state:
    st.session_state.simulacion_ejecutada = False
    st.session_state.resultados = None

#Sidebar 
with st.sidebar:
    st.header("⚙️ Parámetros")
    # Usamos los nombres en inglés internamente para que FastF1 no se queje
    mapa_gp = {"Bahréin": "Bahrain", "Monza": "Monza", "Silverstone": "Silverstone"}
    gp_seleccionado = st.selectbox("Gran Premio", list(mapa_gp.keys()))
    compuesto = st.selectbox("Compuesto Inicial", ["Blando", "Medio", "Duro"])
    
    st.markdown("---")
    temp_pista = st.slider("Temp. de Pista (°C)", 20.0, 60.0, 35.0, step=1.0)
    vuelta_actual = st.slider("Vuelta Actual (Stint)", 1, 40, 15) # Clave para ver la evolución!
    
    st.markdown("<br>", unsafe_allow_html=True)
    btn_simular = st.button("▶  Ejecutar Simulación", type="primary", use_container_width=True)

# ─── LÓGICA DE SIMULACIÓN ───
if btn_simular:
    with st.spinner(f"📡 Descargando telemetría de {gp_seleccionado}..."):
        df = obtener_telemetria_limpia(2024, mapa_gp[gp_seleccionado], 'R', 'VER', temp_pista)
        
    with st.spinner("🧠 Entrenando Inteligencia Artificial..."):
        modelo, poly, cols, error, precision = entrenar_modelo_degradacion(df)
        motor = MotorEstrategia(modelo, poly, cols)
        
    with st.spinner("🧮 Aplicando Simpson 1/3 y Newton-Raphson..."):
        st.session_state.resultados = motor.analizar_stint(compuesto, vuelta_actual, temp_pista, umbral_perdida=2.5)
        st.session_state.precision_modelo = precision
        st.session_state.simulacion_ejecutada = True

# ─── INTERFAZ DINÁMICA ───
if st.session_state.simulacion_ejecutada:
    res = st.session_state.resultados
    
    #Métricas actualizadas con la matemática real
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Temperatura de Pista", value=f"{temp_pista} °C", delta=f"Precisión IA: {st.session_state.precision_modelo*100:.1f}%", delta_color="normal")
    with col2:
        st.metric(label="Lap Time Actual", value=f"{res['tiempo_actual']:.2f}s", delta=f"+{res['delta_tiempo']:.2f}s (Degradación)", delta_color="inverse")
    with col3:
        # Mostramos la temperatura calculada por RK4
        st.metric(label="Temperatura Neumático (RK4)", value=f"{res['temp_goma']:.1f} °C", delta="Óptimo: 90°C - 110°C", delta_color="off")
    with col4:
        # El veredicto de Newton-Raphson
        st.metric(label="Crossover Point", value=f"Vuelta {res['crossover_point']}", delta="Vía Newton-Raphson", delta_color="off")

    st.markdown("---")

    # Gráfico Plotly
    st.subheader("📊 Curva Predictiva de Degradación (T(v))")
    
    fig = go.Figure()
    
    # Línea principal del modelo
    color_linea = '#E53935' if compuesto == 'Blando' else '#FDD835' if compuesto == 'Medio' else '#FFFFFF'
    fig.add_trace(go.Scatter(
        x=res['grafico_x'], y=res['grafico_y'], 
        mode='lines+markers', name=f'{compuesto}',
        line=dict(color=color_linea, width=3)
    ))
    
    # Marcador de donde estamos ahora
    fig.add_trace(go.Scatter(
        x=[vuelta_actual], y=[res['tiempo_actual']],
        mode='markers', name='Posición Actual',
        marker=dict(color='blue', size=12, symbol='star')
    ))
    
    # Línea vertical del Crossover Point (Donde hay que parar)
    fig.add_vline(x=res['crossover_point'], line_width=2, line_dash="dash", line_color="cyan")
    fig.add_annotation(x=res['crossover_point'], y=max(res['grafico_y']), text="PIT STOP ÓPTIMO", showarrow=False, yshift=10, font=dict(color="cyan"))

    fig.update_layout(
        xaxis_title="Vueltas de Uso", 
        yaxis_title="Tiempo de Vuelta (Segundos)", 
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
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