import streamlit as st
import pandas as pd
import numpy as np

# Configuración básica (DEBE ser la primera línea de Streamlit)
st.set_page_config(page_title="F1 Pit Optimizer", page_icon="🏎️", layout="wide")

# Ocultar menú de Streamlit y pie de página para que parezca una app real
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# Encabezado
st.title("🏎️ F1 Pit Stop optimizer")
st.markdown("---")

# Panel lateral
with st.sidebar:
    st.header("⚙️ Parámetros de Simulación")
    gran_premio = st.selectbox("Gran Premxio", ["Bahrein", "Monza", "Silverstone"])
    compuesto = st.selectbox("Compuesto Inicial", ["Blando", "Medio", "Duro"])
    temp_pista = st.slider("Temp. de Pista Inicial (°C)", 20, 60, 42)
    st.button("Ejecutar Simulación", type="primary", use_container_width=True)

# Dashboard de Métricas Principales (El "chiche" visual)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Temperatura Actual", value=f"{temp_pista} °C", delta="2 °C")
with col2:
    st.metric(label="Lap Time Delta", value="1:34.520", delta="-0.150s", delta_color="inverse")
with col3:
    st.metric(label="Desgaste (RK4)", value="12%", delta="Crítico a >75%", delta_color="off")
with col4:
    st.metric(label="Crossover Point", value="Vuelta 18", delta="Óptimo Newton-Raphson")

st.markdown("---")

# Espacio reservado para los gráficos interactivos
st.subheader("📈 Curvas de Degradación (Próximamente con Plotly)")
st.info("Acá se renderizarán los gráficos interactivos comparando las estrategias de 1 vs 2 paradas.")