import streamlit as st
import pandas as pd
import numpy as np

# Configuración básica de la página
st.set_page_config(page_title="Optimizador Pit Stops F1", page_icon="🏎️", layout="wide")

st.title("Optimizador de Pit Stops - Fórmula 1")
st.markdown("### Laboratorio de Modelado y Simulación")

st.success("¡El entorno virtual y todas las librerías están configuradas correctamente!")

# Panel lateral para los futuros controles
st.sidebar.header("Configuración del Modelo")
st.sidebar.selectbox("Seleccionar Gran Premio", ["Bahrein", "Monza", "Silverstone"])
st.sidebar.selectbox("Compuesto Inicial", ["Blando", "Medio", "Duro"])

st.write("Próximamente aca van a ir las gráficas de degradación y la curva del Crossover Point.")