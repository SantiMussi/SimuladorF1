import streamlit as st
import pandas as pd
import numpy as np

# Configuración de página
st.set_page_config(page_title="F1 Pit Optimizer", page_icon="🏎️", layout="wide")

#Cargar CSS externo
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

#Sidebar 
with st.sidebar:
    st.header("Parámetros")
    gran_premio = st.selectbox("Gran Premio", ["Bahrein", "Monza", "Silverstone"])
    compuesto = st.selectbox("Compuesto Inicial", ["Blando", "Medio", "Duro"])
    temp_pista = st.slider("Temp. de Pista (°C)", 20, 60, 42)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("▶  Ejecutar Simulación", type="primary", use_container_width=True)

#Métricas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Temperatura de Pista", value=f"{temp_pista} °C", delta="↑ 2 °C")
with col2:
    st.metric(label="Lap Time Delta", value="1:34.520", delta="-0.150s", delta_color="inverse")
with col3:
    st.metric(label="Desgaste · RK4", value="12 %", delta="Crítico > 75%", delta_color="off")
with col4:
    st.metric(label="Crossover Point", value="Vuelta 18", delta="Newton-Raphson", delta_color="off")

st.markdown("---")

#Placeholder gráficos
st.subheader("Curvas de Degradación")
st.info("Los gráficos interactivos (Plotly) aparecerán acá · comparativa 1 vs 2 paradas.")