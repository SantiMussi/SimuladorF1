import fastf1 as ff1 
import pandas as pd 
import numpy as np 
import os 
from metodos_numericos import runge_kutta_4


#Envolvemos la EDO para poder inyectarle la variable T_Pista
def crear_edo_temperatura(T_pista):
    """Crear la EDO termica basandose en la temperatura real de la pista"""
    def edo(t, T):
        friccion = 8.5
        k = 0.12
        return k * (T_pista - T) + friccion
    return edo


def obtener_telemetria_limpia(year, gp, session, driver, T_pista):
    """
    Descarga y limpia los datos de FastF1 para usarlos en el modelo de MLL
    """

    if not os.path.exists('data'):
        os.makedirs('data')
    
    #Configuramos la cache para q no tarde 10 años y cada vez q lo corremos
    ff1.Cache.enable_cache('data')

    print(f"EXTRACTOR >> Descargando datos: {gp} {year} -- {session} ({driver})")

    session_data = ff1.get_session(year, gp, session)
    session_data.load(telemetry=False, weather=False)

    #Filtramos solo las vueltas del piloto elegido 
    laps = session_data.laps.pick_drivers(driver)

    #Limpieza de datos

    # Borramos las vueltas sin tiempo registrado
    laps = laps.dropna(subset=['LapTime', 'TyreLife', 'Compound'])

    # Filtramos vueltas donde entro o salio de boxes (Son mas lentas y ensucian el modelo)
    laps = laps[laps['PitOutTime'].isnull() & laps['PitInTime'].isnull()]

    # Filtramos vueltas anuladas por limites de pista (TrackLimits)
    laps = laps[laps['IsAccurate'] == True].copy()

    #Transformamos el tiempo de vuelta a segundos (Sickit-Learn necesita floats)
    laps['LapTime_sec'] = laps['LapTime'].dt.total_seconds()

    #Correccion fisica (Combustible)
    efecto_combustible_por_vuelta = 0.06
    laps['LapTime_sec_corrected'] = laps['LapTime_sec'] + (laps['LapNumber'] * efecto_combustible_por_vuelta)

    #Creamos la matematica con la temperatura real
    edo_dinamica = crear_edo_temperatura(T_pista)

    #Simulamos 60 vueltas de vida util, la goma sale de boxes a 80 grados
    vueltas_sim, temp_sim = runge_kutta_4(f=edo_dinamica, t0=0, y0=80.0, t_end = 60, h=1)

    #Mapeamos
    mapa_temperaturas = dict(zip(np.round(vueltas_sim, 1), temp_sim))

    laps['Temp_RK4'] = laps['TyreLife'].map(mapa_temperaturas)

    #Nos queda solo con las columnas que le importan a nuestro modelo
    df_limpio = laps[['LapNumber', 'TyreLife', 'Compound', 'Temp_RK4', 'LapTime_sec', 'LapTime_sec_corrected']]
    
    return df_limpio.dropna()