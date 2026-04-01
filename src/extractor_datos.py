import fastf1 as ff1 
import pandas as pd 
import os 

def obtener_telemetria_limpia(year, gp, session, driver):
    """
    Descarga y limpia los datos de FastF1 para usarlos en el modelo de MLL
    """

    if not os.path.exists('data'):
        os.makedirs('data')
    
    #Configuramos la cache para q no tarde 10 a;os y cada vez q lo corremos
    ff1.Cache.enable_cache('data')

    print(f"EXTRACTOR >> Descargando datos: {gp} {year} -- {session} ({driver})")

    session_data = ff1.get_session(year, gp, session)
    session_data.load(telemetry=False, weather=False)

    #Filtramos solo las vueltas del piloto elegido
    laps = session_data.laps.pick_driver(driver)

    #Limpieza de datos

    # Borramos las vueltas sin tiempo registrado
    laps = laps.dropna(subset=['LapTime', 'TyreLife', 'Compound'])

    # Filtramos vueltas donde entro o salio de boxes (Son mas lentas y ensucian el modelo)
    laps = laps[laps['PitOutTime'].isnull() & laps['PitInTime'].isnull()]

    # Filtramos vueltas anuladas por limites de pista (TrackLimits)
    laps = laps[laps['IsAccurate'] == True]

    #Transformamos el tiempo de vuelta a segundos (Sickit-Learn necesita floats)
    laps['LapTime_sec'] = laps=['LapTime'].dt.total_seconds()

    #Nos queda solo con las columnas que le importan a nuestro modelo
    df_limpio = laps[['LapNumber', 'TyreLife', 'Compound', 'LapTime_sec']]

    return df_limpio


#Test, dsp sacarlo
if __name__ == "__main__":
    #Probamos con TUTUTURU MAX VERSTAPPEEEN
    df= obtener_telemetria_limpia(2024, 'Bahrain', 'R', 'VER')
    print('Datos limpios y listos para sickit-learn')
    print(df.head(10))