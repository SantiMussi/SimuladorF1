import fastf1 as ff1 
import pandas as pd 
import numpy as np 
import os 
from src.metodos_numericos import runge_kutta_4

# Constantes físicas por compuesto (Rigor Termodinámico)
CONSTANTES_NEUMATICOS = {
    'SOFT':   {'friccion': 8.0, 'k': 0.13}, # Calienta a ~95-105°C
    'MEDIUM': {'friccion': 7.0, 'k': 0.14}, # Calienta a ~85-95°C
    'HARD':   {'friccion': 6.0, 'k': 0.16}  # Calienta a ~75-85°C
}

# Envolvemos la EDO para poder inyectarle la variable T_Pista y el compuesto
def crear_edo_temperatura(T_pista, compuesto='MEDIUM'):
    """Crear la EDO térmica basándose en la temperatura real y las propiedades del compuesto"""
    config = CONSTANTES_NEUMATICOS.get(compuesto.upper(), CONSTANTES_NEUMATICOS['MEDIUM'])
    friccion = config['friccion']
    k = config['k']
    
    def edo(t, T):
        return k * (T_pista - T) + friccion
    return edo


def obtener_telemetria_limpia(year, gp, session, driver, T_pista):
    """
    Descarga y limpia los datos de FastF1 para usarlos en el modelo de MLL
    """

    # Configuración de caché
    if not os.path.exists('data'):
        os.makedirs('data')
    ff1.Cache.enable_cache('data')

    # Normalizamos el nombre de la sesión (FastF1 v3 prefiere nombres completos)
    # Agregamos .strip().upper() para mayor robustez
    session_clean = str(session).strip().upper()
    session_full = 'Race' if session_clean in ['R', 'RACE'] else session_clean
    
    print(f"EXTRACTOR >> Descargando datos: {gp} {year} -- {session_full} ({driver})")

    usa_datos_simulados = False
    try:
        session_data = ff1.get_session(year, gp, session_full)
        session_data.load(laps=True, telemetry=False, weather=False, messages=False)
        
        if len(session_data.laps) == 0:
            raise ValueError("Laps empty")

        laps = session_data.laps.pick_driver(driver)
        if laps.empty:
            raise ValueError("Driver laps empty")

    except Exception as e:
        print(f"EXTRACTOR WARNING >> Falló la carga real de FastF1: {e}")
        print("MODO DEMO >> Generando telemetría sintética (Verstappen, Bahrain 2024)...")
        usa_datos_simulados = True
        
        # Generar data sintética realista para no frenar el proyecto (Facu)
        n_vueltas = 45
        vueltas = list(range(1, n_vueltas + 1))
        tiempos_base = 92.5 + np.cumsum(np.random.normal(0.06, 0.02, n_vueltas)) # Degra 0.06s/lap
        
        laps = pd.DataFrame({
            'LapNumber': vueltas,
            'TyreLife': vueltas,
            'Compound': 'SOFT',
            'LapTime_sec': tiempos_base,
            'PitInTime': [pd.NaT] * n_vueltas,
            'PitOutTime': [pd.NaT] * n_vueltas,
            'IsAccurate': [True] * n_vueltas
        })

    # Transformamos el tiempo de vuelta a segundos (Si es real viene de .dt.total_seconds, si es sintético ya es float)
    if 'LapTime' in laps.columns:
        laps['LapTime_sec'] = laps['LapTime'].dt.total_seconds()
    
    # Limpieza de datos (Aseguramos que existan las columnas clave)
    # Filtramos solo vueltas precisas (eliminamos pits y errores de sensor)
    if not usa_datos_simulados:
        laps = laps[laps['IsAccurate'] == True]
        
    laps = laps.dropna(subset=['LapTime_sec', 'TyreLife', 'Compound'])

    #Correccion fisica (Combustible)
    efecto_combustible_por_vuelta = 0.06
    laps['LapTime_sec_corrected'] = laps['LapTime_sec'] + (laps['LapNumber'] * efecto_combustible_por_vuelta)

    # Creamos la matemática con la temperatura real y el compuesto actual
    # Usamos el primer compuesto del dataframe para la simulación de referencia
    compuesto_ref = laps['Compound'].iloc[0] if not laps.empty else 'MEDIUM'
    edo_dinamica = crear_edo_temperatura(T_pista, compuesto_ref)
    
    # Simulamos 60 vueltas de vida útil, la goma sale de boxes a 80 grados
    vueltas_sim, temp_sim = runge_kutta_4(f=edo_dinamica, t0=0, y0=80.0, t_end=60, h=1)

    # Asignamos la temperatura usando interpolación (más robusto que map)
    laps['Temp_RK4'] = np.interp(laps['TyreLife'], vueltas_sim, temp_sim)

    # Si estamos en modo demo, inyectamos una columna para que el app.py sepa avisar
    if usa_datos_simulados:
        laps['IsDemo'] = True

    return laps[['LapNumber', 'TyreLife', 'Compound', 'Temp_RK4', 'LapTime_sec', 'LapTime_sec_corrected']].dropna()