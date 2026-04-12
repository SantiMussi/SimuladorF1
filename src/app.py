import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from extractor_datos import obtener_datos_aumentados, COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG
from modelo_ml import PredictorDegradacion
from strategy import EngineEstrategia

# --- Configuración de Estética Premium ---
st.set_page_config(page_title="F1 Strategy Wall", layout="wide", page_icon="🏎️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Share Tech Mono', monospace;
    }
    .stMetric {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #333;
    }
    .main {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# --- Título Superior ---
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>🏎️ F1 STRATEGY CONTROL WALL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Hybrid AI & Numerical Methods Engine</p>", unsafe_allow_html=True)

# --- Sidebar: Parámetros ---
st.sidebar.header("🛠️ Race Setup")
# Inyectamos el selector de pista primero para determinar los defaults
track_name = st.sidebar.selectbox("🏎️ Circuito", list(CIRCUITOS_CONFIG.keys()))
track_settings = CIRCUITOS_CONFIG.get(track_name, {})

track_temp = st.sidebar.slider("🌡️ Temperatura de Pista (°C)", 15, 60, 35)
total_race_laps = st.sidebar.number_input("🏁 Vueltas Totales", 30, 90, track_settings.get('total_laps', 50))
pit_loss = st.sidebar.slider("⏱️ Costo de Pit Stop (s)", 15.0, 30.0, track_settings.get('pit_loss', 22.0))
compound_name = st.sidebar.selectbox("🛞 Compuesto", list(COMPOUNDS_PHYSICS.keys()))

# Slider Interactive "ESTÁS ACÁ"
current_lap_ui = st.sidebar.slider("📍 Vuelta Actual", 1, total_race_laps, 1)

# --- Backend Logic (Optimización de Caché) ---
@st.cache_data
def cargar_datos_entrenamiento(track_name):
    return obtener_datos_aumentados(track_name=track_name)

@st.cache_data
def obtener_datos_reales_referencia(track_name):
    # Genera una referencia realista basada en el track seleccionado
    config = CIRCUITOS_CONFIG.get(track_name, {'base_lap_time': 90.0})
    laps_ref = np.arange(1, 46)
    tiempos_ref = config['base_lap_time'] + (laps_ref * 0.040) # Ritmo base + degradación lineal suave
    return laps_ref, tiempos_ref

@st.cache_resource(show_spinner="Entrenando motor de IA con nueva física...")
def entrenar_modelo_ia(track_name):
    df = cargar_datos_entrenamiento(track_name)
    predictor = PredictorDegradacion()
    # Obtenemos el tiempo base de referencia para el anclaje del intercept
    base_time = CIRCUITOS_CONFIG.get(track_name, {}).get('base_lap_time', 90.0)
    predictor.entrenar(df, track_base_time=base_time)
    return predictor

# El motor de estrategia es liviano, no necesita caché recursivo si sus parámetros cambian
predictor = entrenar_modelo_ia(track_name)
engine = EngineEstrategia(predictor, track_name=track_name, compound=compound_name)
engine.pit_loss = pit_loss

# Simulación completa del stint
laps, tiempos, temps = engine.simular_stint(total_race_laps, track_temp)
resultado = engine.calcular_estrategia_optima(laps, tiempos, total_race_laps)
crossover_lap = resultado['crossover_lap']

# --- LÓGICA DE UX: Semáforo de Neumáticos ---
# --- LÓGICA DE UX: Semáforo de Neumáticos (Senior Strategist Sync) ---
def get_tire_status(lap, crossover, tiempos, current_temp, compound):
    # Ventana de trabajo según compuesto
    window = COMPOUNDS_PHYSICS.get(compound, {'ventana_temp': (90, 110)})['ventana_temp']
    t_ideal = tiempos[0]
    t_actual = tiempos[lap-1]
    
    # 0. Estado Térmico: Si está por debajo de la ventana
    if current_temp < window[0]:
        return "🔵 CALENTANDO", "cyan", f"Neumático por debajo de {window[0]}°C. Evitar ataques."
    
    # 1. Condición Crítica: Solo si perdemos más del 2% del ritmo ideal
    if t_actual > t_ideal * 1.02:
        return "🔴 AGOTADO", "red", "PÉRDIDA CRÍTICA. El ritmo ha caído más de un 2%."
    
    # 2. Sincronización con Crossover
    if lap < crossover - 2: 
        return "🟢 ÓPTIMO", "green", "El neumático está en ventana de trabajo."
    if lap < crossover: 
        return "🟡 DESGASTE", "yellow", "Degradación detectada. Preparar estrategia."
    
    return "🟠 CROSSOVER", "orange", "¡VENTANA DE PARADA! Punto óptimo alcanzado."

status_text, status_color, status_desc = get_tire_status(
    current_lap_ui, crossover_lap, tiempos, temps[current_lap_ui-1], compound_name
)

# --- UI: Veredicto de Estrategia ---
st.divider()

net_gain = resultado.get('net_gain', 0)
distancia_box = int(crossover_lap) - current_lap_ui

# Solo hay crossover real si el ahorro neto es positivo (superó el costo del pit stop)
if crossover_lap >= total_race_laps or net_gain <= 0:
    st.info(f"🏎️ **ESTRATEGIA ÓPTIMA: 1 STINT** - No se recomienda parar. El ahorro de ritmo no compensa los {pit_loss}s de pit stop.")
else:
    veredicto_msg = f"Ahorro neto estimado: **{net_gain:.2f}s** | Ventaja competitiva en **Vuelta {int(crossover_lap)}**."
    
    if distancia_box > 4:
        st.success(f"✅ **MANTENER RITMO:** {veredicto_msg}")
    elif 0 <= distancia_box <= 4:
        st.success(f"🟢 **WINDOW OPEN:** La parada ahora es beneficiosa en tiempo total. {veredicto_msg}")
    else:
        st.error(f"⚠️ **BOX BOX BOX!** Has pasado el punto óptimo. {veredicto_msg}")

# Validación de Calibración
base_ref = CIRCUITOS_CONFIG.get(track_name, {}).get('base_lap_time', 90.0)
discrepancia = abs(tiempos[0] - base_ref)

if discrepancia > 2.0:
    st.sidebar.error(f"⚠️ **MODELO DESCALIBRADO**")
    st.sidebar.write(f"Desfase detectado: {discrepancia:.1f}s. El motor térmico está penalizando demasiado.")

# Validación Senior: Alerta si el crossover es demasiado temprano
if crossover_lap < 20 and track_name in ['Monza', 'Spa', 'Silverstone']:
    st.sidebar.warning(f"🚨 **ALERTA ESTRATEGIA:** Crossover en Vuelta {int(crossover_lap)} es demasiado agresivo para {track_name}. Verificar Setup de Degradación.")
elif crossover_lap < 15:
    st.sidebar.warning(f"🚨 **ALERTA ESTRATEGIA:** Punto de parada prematuro detectado.")


# --- Visualización: Grid de Métricas ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Estado Goma", status_text)
with m2:
    st.metric("🌡️ Temp. Neumático", f"{temps[current_lap_ui-1]:.1f}°C")
with m3:
    st.metric("⏱️ Lap Time Actual", f"{tiempos[current_lap_ui-1]:.3f}s")
with m4:
    st.metric("🛠️ Costo Pit", f"{pit_loss}s")

# --- GRÁFICOS INTERACTIVOS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏱️ Análisis de Ritmo (Degradación)")
    fig_ritmo = go.Figure()
    
    # Curva de Degradación IA
    fig_ritmo.add_trace(go.Scatter(x=laps, y=tiempos, name="Simulación IA", 
                                  line=dict(color='#00d4ff', width=4)))
    
    # REFERENCIA REAL (Telemetría de Referencia Sincronizada)
    laps_ref, tiempos_ref = obtener_datos_reales_referencia(track_name)
    fig_ritmo.add_trace(go.Scatter(x=laps_ref, y=tiempos_ref, name=f"Referencia {track_name}", 
                                  line=dict(color='rgba(255, 255, 255, 0.2)', width=2, dash='dot')))
    
    # Marcador "ESTÁS ACÁ" con cambio de color dinámico y pendiente
    # Solo mostrar marcador si la degradación es notable (pendiente positiva)
    slope = (tiempos[current_lap_ui-1] - tiempos[0]) / (current_lap_ui + 1)
    is_crossover = abs(current_lap_ui - crossover_lap) < 3 and slope > 0.02
    marker_color = "#ff8c00" if is_crossover else "white" 
    
    fig_ritmo.add_trace(go.Scatter(x=[current_lap_ui], y=[tiempos[current_lap_ui-1]],
                                  mode='markers+text', name="Posición Actual",
                                  text=["BOX!" if is_crossover else "ESTÁS ACÁ"], 
                                  textposition="top center",
                                  marker=dict(color=marker_color, size=15, symbol='star',
                                             line=dict(color='cyan', width=2))))
    
    # Línea de Crossover
    fig_ritmo.add_vline(x=crossover_lap, line_dash="dash", line_color="orange",
                       annotation_text="Punto de Parada Óptimo")
    
    fig_ritmo.update_layout(template="plotly_dark", xaxis_title="Vuelta", yaxis_title="Segundos")
    st.plotly_chart(fig_ritmo, use_container_width=True)

with col2:
    st.subheader("🌡️ Evolución Térmica (RK45)")
    fig_temp = go.Figure()
    
    # Franja Sombreada
    window = COMPOUNDS_PHYSICS.get(compound_name, {'ventana_temp': (90, 110)})['ventana_temp']
    fig_temp.add_hrect(y0=window[0], y1=window[1], fillcolor="green", opacity=0.15, line_width=0,
                      annotation_text=f"VENTANA {compound_name}", annotation_position="top left")
    
    line_color = "red" if temps[current_lap_ui-1] > window[1] else "#ff4b4b"
    fig_temp.add_trace(go.Scatter(x=laps, y=temps, name="Temp. Core", line=dict(color=line_color, width=3)))
    fig_temp.add_trace(go.Scatter(x=[current_lap_ui], y=[temps[current_lap_ui-1]], mode='markers', marker=dict(color='white', size=12)))
    
    fig_temp.update_layout(template="plotly_dark", xaxis_title="Vuelta", yaxis_title="Grados Celsius")
    st.plotly_chart(fig_temp, use_container_width=True)

# --- NUEVO GRÁFICO: Tyre Life % (Cliff Effect) ---
st.subheader("📉 Desgaste Mecánico (Tyre Life %)")
fig_life = go.Figure()

# Cálculo de Tyre Life % dinámico por compuesto Y PISTA
# Obtenemos max_life base y aplicamos multiplicador por abrasión de pista
phys_comp = COMPOUNDS_PHYSICS.get(compound_name, {'max_life': 40})
m_life_base = phys_comp['max_life']

# Factor de Pista: Baja abrasión (Monza=0.8) -> Más vida (x1.25)
# Alta abrasión (Interlagos=1.1) -> Menos vida (x0.9)
track_abrasion = CIRCUITOS_CONFIG.get(track_name, {'abrasion': 1.0})['abrasion']
m_life_efectiva = m_life_base * (1.0 / track_abrasion)

tyre_life_pct = [max(0, 100 * (1 - (v / m_life_efectiva)**3)) for v in laps]

fig_life.add_trace(go.Scatter(x=laps, y=tyre_life_pct, name="Tyre Health %",
                             line=dict(color='#00ff88', width=4, shape='spline'),
                             fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'))

# Punto actual
fig_life.add_trace(go.Scatter(x=[current_lap_ui], y=[tyre_life_pct[current_lap_ui-1]],
                             mode='markers', marker=dict(color='white', size=12)))

# Umbral Crítico (30%)
fig_life.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="UMBRAL CRÍTICO (30%)")
fig_life.add_vline(x=crossover_lap, line_dash="dash", line_color="orange")

fig_life.update_layout(template="plotly_dark", xaxis_title="Vuelta", yaxis_title="Vida Útil %", yaxis_range=[0, 105])
st.plotly_chart(fig_life, use_container_width=True)

# --- Comparación Estratégica ---
st.subheader("📊 Comparativa de Estrategia")
c1, c2 = st.columns(2)

# Cálculo de tiempo perdido (Delta con vuelta 1)
tiempo_perdido = np.sum(tiempos[:current_lap_ui]) - (tiempos[0] * current_lap_ui)
tiempo_ganado_potencial = tiempos[current_lap_ui-1] - tiempos[0]

with c1:
    st.error(f"⌛ **Tiempo perdido por desgaste:** +{tiempo_perdido:.2f}s acumulados")
with c2:
    st.info(f"🚀 **Ventaja de neumático nuevo:** -{tiempo_ganado_potencial:.2f}s por vuelta")

st.markdown(f"**Análisis Final:** {status_desc}")

# --- PANEL DE ESTRATEGIA EXPERTA (Monza Special) ---
if track_name == 'Monza':
    st.divider()
    with st.expander("📊 Recomendaciones Tácticas: GP de Italia", expanded=True):
        st.markdown("""
        **Estrategia Principal: 1 Parada (Medium 🟡 → Hard ⚪)**
        *   **Ventana Óptima:** Vueltas 22 - 28.
        *   **Clave:** Minimizar el tiempo en el pit lane (25s de pérdida).
        *   **Riesgo Térmico:** Cuidado con la tracción en las chicanes (Prima Variante).
        
        **Estrategia Alternativa: 1 Parada (Soft 🔴 → Hard ⚪)**
        *   **Ventana:** Vueltas 14 - 20. Ideal para ganar posiciones al inicio pero con riesgo de graining.
        """)
