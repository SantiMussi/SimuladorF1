from strategy import EngineFisico, EngineEstrategia, EngineEstocastico
import numpy as np

try:
    fisico = EngineFisico("Monza", "SOFT")
    estrategia = EngineEstrategia(fisico, 50, 35.0)
    estocastico = EngineEstocastico(estrategia)
    print("Módulos instanciados correctamente.")
    
    perfil = fisico.simular_perfil_termico(50, 35.0)
    print("Perfil térmico generado.")
    
    lap, tiempo, c1, c2 = estrategia.calcular_crossover_optimo()
    print(f"Crossover óptimo: {lap}")
    
    cliff = estrategia.detectar_cliff_critico(0, "SOFT")
    print(f"Cliff crítico: {cliff}")
    
    lap_mc, hist = estocastico.simular_montecarlo(int(lap))
    print(f"Montecarlo: {lap_mc}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
