import os
import sys
import fastf1

# Añadir src al path para importar extractor_datos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import urllib.request
from extractor_datos import CIRCUITOS_CONFIG, obtener_datos_aumentados, obtener_track_geometry

FONTS = {
    "titillium_400": "https://fonts.gstatic.com/s/titilliumweb/v19/NaPecZTIAOhVxoMyOr9n_E7fRMQ.ttf",
    "titillium_600": "https://fonts.gstatic.com/s/titilliumweb/v19/NaPDcZTIAOhVxoMyOr9n_E7ffBzCKIw.ttf",
    "titillium_700": "https://fonts.gstatic.com/s/titilliumweb/v19/NaPDcZTIAOhVxoMyOr9n_E7ffHjDKIw.ttf",
    "inter_400": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf",
    "inter_500": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuI6fMZg.ttf",
    "inter_600": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuGKYMZg.ttf",
    "inter_700": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuFuYMZg.ttf",
}

def pre_cargar_todo():
    print("🚀 Iniciando pre-carga de datos para exposición offline...")
    print(f"📂 Los datos se guardarán en: {os.path.abspath('f1_cache')}")
    
    # 1. Cargar fuentes para que la UI no dependa de Google Fonts
    print("\n📦 Descargando fuentes para uso local...")
    fonts_dir = os.path.join(os.path.dirname(__file__), "src", "assets", "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    for name, url in FONTS.items():
        print(f"   📥 {name}... ", end="", flush=True)
        try:
            target = os.path.join(fonts_dir, f"{name}.ttf")
            if not os.path.exists(target):
                urllib.request.urlretrieve(url, target)
                print("✅")
            else:
                print("🆗 (ya existe)")
        except Exception as e:
            print(f"❌ Error al descargar {name}: {e}")

    # 2. Cargar Datos F1
    print("\n🏎️ Descargando datos de carreras (pista y telemetría)...")
    years = [2023, 2024]
    tracks = list(CIRCUITOS_CONFIG.keys())
    
    total = len(tracks) * len(years)
    count = 0
    
    for track in tracks:
        for year in years:
            count += 1
            print(f"[{count}/{total}] Procesando {track} ({year})...")
            
            print(f"   ⏱️ Registros de vueltas... ", end="", flush=True)
            try:
                obtener_datos_aumentados(year=year, track_name=track)
                print("✅ OK")
            except Exception as e:
                print(f"❌ Error: {e}")
                
            print(f"   🗺️ Geometría de trazado... ", end="", flush=True)
            try:
                obtener_track_geometry(track_name=track, year=year)
                print("✅ OK")
            except Exception as e:
                print(f"❌ Error: {e}")

    print("\n✨ ¡Todo listo! El simulador está preparado para uso 100% OFFLINE.")
    print("TIP: Al arrancar la app de Streamlit por primera vez en local, ya se buscarán las fuentes en src/assets/fonts/.")

if __name__ == "__main__":
    pre_cargar_todo()


