import numpy as np
from metodos_numericos import rk45_step, least_squares_poly, newton_raphson, romberg_integration, poly_derivative
from extractor_datos import COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG, get_track_severity

class EngineEstrategia:
    def __init__(self, modelo_ml, track_name='Silverstone', compound='SOFT'):
        self.ml = modelo_ml
        self.pit_loss = 22.0 # Segundos promedio de un pit stop
        self.track_name = track_name
        self.compound = compound
        
    def T_v(self, lap, temp):
        """
        Función de tiempo de vuelta (T) alineada con la predicción pura de la IA.
        """
        # Predicción directa del modelo ML calibrado pasándole la pista actual
        t_pred = self.ml.predecir_lap(lap, temp, self.compound, track_name=self.track_name)
        
        # Penalización térmica SI el neumático se sale de ventana
        window = COMPOUNDS_PHYSICS.get(self.compound, {'ventana_temp': (90, 110)})['ventana_temp']
        overheat_penalty = 0
        if temp > window[1]:
            overheat_penalty = (temp - window[1]) * 0.4 # 0.4s por cada grado de exceso
            
        return t_pred + overheat_penalty

    def edo_temperatura(self, t, T_neumatico, T_pista_base):
        """
        EDO Térmica Dinámica: Calibrada para RK45.
        """
        phys = COMPOUNDS_PHYSICS.get(self.compound, {'friccion_base': 10.0, 'k_disipacion': 0.12})
        track_config = get_track_severity(self.track_name)
        
        # Fricción escala con la severidad de la pista
        friccion_real = phys['friccion_base'] * track_config['speed_factor']
        k = phys['k_disipacion']
        
        return friccion_real - k * (T_neumatico - T_pista_base)

    def simular_stint(self, max_laps, T_pista):
        """
        Genera el perfil del stint con evolución térmica.
        """
        laps = []
        tiempos = []
        temps = []
        
        T_actual = 80.0 # Temperatura inicial
        h = 0.2
        
        for lap in range(1, max_laps + 1):
            tiempo_sim = 0
            while tiempo_sim < 1.0:
                T_actual, tiempo_sim, h, success = rk45_step(
                    lambda t, y: self.edo_temperatura(t, y, T_pista),
                    tiempo_sim, T_actual, h
                )
            
            lap_time = self.T_v(lap, T_actual)
            laps.append(lap)
            tiempos.append(lap_time)
            temps.append(T_actual)
            
        return np.array(laps), np.array(tiempos), np.array(temps)

    def optimizador_determinista(self, compounds, total_race_laps, track_temp):
        """
        Busca las vueltas de parada óptimas para una secuencia de compuestos dada (1 o 2 paradas).
        Implementa el reseteo de 'edad de la goma' en cada stint.
        """
        n_paradas = len(compounds) - 1
        mejor_tiempo = float('inf')
        vueltas_optimas = []
        
        # Pre-calcular perfiles de lap time para cada compuesto posible
        # (Esto optimiza el cálculo al no repetir RK45)
        perfiles = {}
        for c in set(compounds):
            # Cambiamos temporalmente el compuesto del motor para simular
            original_compound = self.compound
            self.compound = c
            _, tiempos, _ = self.simular_stint(total_race_laps, track_temp)
            perfiles[c] = tiempos
            self.compound = original_compound

        if n_paradas == 1:
            # Estrategia de 1 Parada: Stint A -> Stint B
            for p1 in range(5, total_race_laps - 4):
                t_stint1 = np.sum(perfiles[compounds[0]][:p1])
                t_stint2 = np.sum(perfiles[compounds[1]][:total_race_laps - p1])
                
                tiempo_total = t_stint1 + self.pit_loss + t_stint2
                
                if tiempo_total < mejor_tiempo:
                    mejor_tiempo = tiempo_total
                    vueltas_optimas = [p1]
        
        elif n_paradas == 2:
            # Estrategia de 2 Paradas: Stint A -> Stint B -> Stint C
            # Usamos bucles anidados según el requerimiento
            for p1 in range(1, total_race_laps - 2):
                t_stint1 = np.sum(perfiles[compounds[0]][:p1])
                for p2 in range(p1 + 1, total_race_laps - 1):
                    t_stint2 = np.sum(perfiles[compounds[1]][:p2 - p1])
                    t_stint3 = np.sum(perfiles[compounds[2]][:total_race_laps - p2])
                    
                    tiempo_total = t_stint1 + self.pit_loss + t_stint2 + self.pit_loss + t_stint3
                    
                    if tiempo_total < mejor_tiempo:
                        mejor_tiempo = tiempo_total
                        vueltas_optimas = [p1, p2]

        # Construir el 'Race Trace' óptimo por segmentos
        race_trace = []
        compounds_trace = []
        current_lap = 1
        
        periodos = vueltas_optimas + [total_race_laps]
        for i, fin_lap in enumerate(periodos):
            comp = compounds[i]
            # Extraer el trozo de tiempo correspondiente (reseteando edad)
            duracion = fin_lap - (periodos[i-1] if i > 0 else 0)
            segmento = perfiles[comp][:duracion]
            race_trace.extend(segmento.tolist())
            compounds_trace.extend([comp] * duracion)
            
        return {
            'vueltas_optimas': vueltas_optimas,
            'tiempo_total': mejor_tiempo,
            'race_trace': np.array(race_trace),
            'compounds_trace': compounds_trace
        }

    def calcular_estrategia_optima(self, laps_sim, tiempos_sim, total_race_laps):
        """
        Algoritmo Integral de Tiempo Neto:
        Compara el tiempo total de carrera terminando con el neumático actual
        vs el tiempo total parando en cada vuelta posible.
        """
        # 1. Ajuste Polinomial para Stint 1 (Compuesto Actual)
        beta = least_squares_poly(laps_sim, tiempos_sim, degree=2)
        
        # --- CORRECCIÓN REGLA FIA: Crossover Asimétrico ---
        # Si elegimos SOFT o MEDIUM, el Stint 2 es HARD. Si elegimos HARD, es SOFT.
        op_compound = 'HARD' if self.compound in ['SOFT', 'MEDIUM'] else 'SOFT'
        laps_ref = np.arange(1, total_race_laps + 1)
        tiempos_op = np.array([self.ml.predecir_lap(v, 100.0, op_compound, track_name=self.track_name) for v in laps_ref])
        beta_op = least_squares_poly(laps_ref, tiempos_op, degree=2)

        def RaceTimeAt(v):
            return np.polyval(beta, v)

        def RaceTimeAtOP(v):
            return np.polyval(beta_op, v)

        # 2. Calcular tiempo total SIN PARAR
        tiempo_sin_parar = np.sum([RaceTimeAt(v) for v in range(1, total_race_laps + 1)])
        
        mejor_tiempo_con_parada = float('inf')
        mejor_vuelta_parada = -1
        
        # 3. Barrido de Crossover
        track_data = CIRCUITOS_CONFIG.get(self.track_name, {})
        p_loss = track_data.get('pit_loss', 25.0)
        
        for x in range(5, total_race_laps - 4):
            # Tiempo del primer stint (Compuesto Original)
            t_stint1 = np.sum([RaceTimeAt(v) for v in range(1, x + 1)])
            
            # Tiempo del segundo stint (Compuesto OPUESTO)
            vueltas_restantes = total_race_laps - x
            t_stint2 = np.sum([RaceTimeAtOP(v) for v in range(1, vueltas_restantes + 1)])
            
            tiempo_total = t_stint1 + p_loss + t_stint2
            
            if tiempo_total < mejor_tiempo_con_parada:
                mejor_tiempo_con_parada = tiempo_total
                mejor_vuelta_parada = x
                
        # 4. VALIDACIÓN DE AHORRO NETO ESTRICTO
        ahorro_neto = tiempo_sin_parar - mejor_tiempo_con_parada
        
        # Si el ahorro neto no supera el beneficio del pit, no paramos
        if ahorro_neto <= 0:
            crossover_final = total_race_laps
        else:
            crossover_final = mejor_vuelta_parada

        return {
            'crossover_lap': int(crossover_final),
            'total_time_no_pit': tiempo_sin_parar,
            'total_time_pit': mejor_tiempo_con_parada,
            'net_gain': ahorro_neto,
            'poly_coeffs': beta
        }
