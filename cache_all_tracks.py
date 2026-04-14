import os
import sys
import fastf1

# Añadir src al path para importar extractor_datos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from extractor_datos import CIRCUITOS_CONFIG, obtener_datos_aumentados

def pre_cargar_todo():
    print("🚀 Iniciando pre-carga de datos para exposición offline...")
    print(f"📂 Los datos se guardarán en: {os.path.abspath('f1_cache')}")
    
    years = [2023, 2024] # Podemos bajar los últimos dos años por seguridad
    tracks = list(CIRCUITOS_CONFIG.keys())
    
    total = len(tracks) * len(years)
    count = 0
    
    for track in tracks:
        for year in years:
            count += 1
            print(f"[{count}/{total}] Cargando {track} ({year})...", end="", flush=True)
            try:
                # Esto internamente llama a fastf1.get_session y .load()
                # Lo que llena la caché de fastf1
                df = obtener_datos_aumentados(year=year, track_name=track)
                print(" ✅ OK")
            except Exception as e:
                print(f" ❌ Error: {e}")

    print("\n✨ ¡Todo listo! Ya puedes desconectar el internet.")
    print("TIP: Abre la app de Streamlit al menos una vez con internet para que se bajen los assets del navegador.")

if __name__ == "__main__":
    pre_cargar_todo()
