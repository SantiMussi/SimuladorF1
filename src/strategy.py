import numpy as np
from metodos_numericos import rk45_step, least_squares_poly, newton_raphson, romberg_integration, poly_derivative
from extractor_datos import COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG, get_track_severity

class EngineEstrategia:
    def __init__(self, modelo_ml, track_name='Silverstone', compound='SOFT'):
        self.ml = modelo_ml
        self.pit_loss = 22.0 # Segundos promedio de un pit stop
        self.track_name = track_name
        self.compound = compound
        
    def T_v(self, lap, temp, current_absolute_lap=None):
        """
        Función de tiempo de vuelta con penalización térmica y beneficio de combustible topado.
        """
        t_pred = self.ml.predecir_lap(lap, temp, self.compound, track_name=self.track_name)
        
        # Penalización térmica
        window = COMPOUNDS_PHYSICS.get(self.compound, {'ventana_temp': (90, 110)})['ventana_temp']
        overheat_penalty = 0
        if temp > window[1]:
            overheat_penalty = (temp - window[1]) * 0.4 

        # Beneficio de combustible (0.04s por vuelta, topado a 3.5s para realismo)
        fuel_benefit = 0
        if current_absolute_lap:
            fuel_benefit = min(3.5, current_absolute_lap * 0.04)
            
        return t_pred + overheat_penalty - fuel_benefit

    def edo_temperatura(self, t, T_neumatico, T_pista_base, lap_progress):
        phys = COMPOUNDS_PHYSICS.get(self.compound, {'friccion_base': 10.0, 'k_disipacion': 0.12})
        track_config = get_track_severity(self.track_name)
        
        # EFECTO MASA: A menos goma, más temperatura (efecto térmico de la delgadez del caucho)
        # lap_progress es un valor de 0 a 1 (vuelta_actual / max_life)
        thermal_mass_factor = 1.0 + (lap_progress * 0.5) # Aumenta la fricción efectiva un 50% al final
        
        friccion_real = phys['friccion_base'] * track_config['speed_factor'] * thermal_mass_factor
        k = phys['k_disipacion']
        
        return friccion_real - k * (T_neumatico - T_pista_base)

    def simular_stint(self, max_laps, T_pista, start_race_lap=1):
        """
        Genera el perfil del stint integrando el Factor de Carga Dinámico por peso de combustible.
        """
        laps, tiempos, temps, wear_trace = [], [], [], []
        T_actual = 80.0 
        h = 0.2
        
        # Parámetros del circuito
        track_data = CIRCUITOS_CONFIG.get(self.track_name, {'abrasion': 1.0, 'total_laps': 70})
        abrasion = track_data['abrasion']
        total_laps_gp = track_data['total_laps']
        
        # Parámetros del compuesto
        phys = COMPOUNDS_PHYSICS.get(self.compound, {'max_life': 40, 'ventana_temp': (90, 110)})
        vida_efectiva = phys['max_life'] * (1.0 / abrasion)
        ventana_max = phys['ventana_temp'][1]
        
        uso_acumulado = 0.0 
        
        for lap_idx in range(max_laps):
            current_lap = lap_idx + 1
            absolute_lap = start_race_lap + lap_idx 
            
            # --- FACTOR DE CARGA DINÁMICO (REALISMO) ---
            # El auto es más pesado al principio (mayor desgaste) y más liviano al final.
            # Rango: 1.05 (pesado) -> 0.95 (liviano)
            load_factor = 1.05 - (0.1 * (absolute_lap / total_laps_gp))
            
            lap_progress = current_lap / max_laps
            tiempo_sim = 0.0
            while tiempo_sim < 1.0:
                T_actual, tiempo_sim, h, success = rk45_step(
                    lambda t, y: self.edo_temperatura(t, y, T_pista, lap_progress),
                    tiempo_sim, T_actual, h
                )
            
            # El estrés base se ve afectado por el peso del auto
            stres_base = 1.0 * load_factor
            
            # Acoplamiento térmico: el calor extremo anula el beneficio del peso bajo
            if T_actual > ventana_max:
                factor_estres = stres_base + (T_actual - ventana_max) * 0.02
            else:
                factor_estres = stres_base
                
            uso_acumulado += factor_estres
            
            # Degradación No Lineal (Cliff de rendimiento)
            usage_ratio = uso_acumulado / vida_efectiva
            desgaste_mecanico = min(100.0, (usage_ratio ** 1.4) * 100.0)
            vida_restante = 100.0 - desgaste_mecanico
            
            # Cálculo de tiempo con combustible dinámico
            lap_time = self.T_v(current_lap, T_actual, current_absolute_lap=absolute_lap)
            
            laps.append(current_lap)
            tiempos.append(lap_time)
            temps.append(T_actual)
            wear_trace.append(max(0.0, vida_restante))
            
        return np.array(laps), np.array(tiempos), np.array(temps), np.array(wear_trace)

    def optimizador_determinista(self, compounds, total_race_laps, track_temp):
        """
        Busca las vueltas de parada óptimas para una secuencia de compuestos dada (1 o 2 paradas).
        Implementa un Límite de Seguridad Estructural estricto: ninguna goma puede superar el 70% de su vida efectiva.
        """
        # Basado en la abrasión del circuito y la vida teórica del compuesto
        abrasion = CIRCUITOS_CONFIG[self.track_name]['abrasion']
        limite_vueltas_seguras = {}
        for c in set(compounds):
            vida_maxima = COMPOUNDS_PHYSICS[c]['max_life']
            # Vida efectiva ajustada por la abrasión del circuito
            vida_efectiva = vida_maxima * (1.0 / abrasion)
            # Límite estricto del 70% de vida útil (reserva del 30% estructural)
            limite_vueltas_seguras[c] = vida_efectiva * 0.7

        n_paradas = len(compounds) - 1
        mejor_tiempo = float('inf')
        vueltas_optimas = []
        
        # Pre-calcular perfiles de lap time para cada compuesto
        perfiles = {}
        for c in set(compounds):
            original_compound = self.compound
            self.compound = c
            _, tiempos, _, _ = self.simular_stint(total_race_laps, track_temp)
            perfiles[c] = tiempos
            self.compound = original_compound

        if n_paradas == 1:
            # Estrategia de 1 Parada: Stint A -> Stint B
            for p1 in range(1, total_race_laps):
                stint1_len = p1
                stint2_len = total_race_laps - p1
                
                # VERIFICACIÓN DE SEGURIDAD ESTRUCTURAL: Antes de calcular tiempos
                if stint1_len > limite_vueltas_seguras[compounds[0]] or stint2_len > limite_vueltas_seguras[compounds[1]]:
                    continue

                # Usamos una aproximación del beneficio promedio por vuelta para el tiempo total
                # Beneficio = 0.04 * vuelta_absoluta
                beneficio_fuel_s1 = np.sum(np.arange(1, stint1_len + 1)) * 0.04
                beneficio_fuel_s2 = np.sum(np.arange(p1 + 1, total_race_laps + 1)) * 0.04

                t_stint1 = np.sum(perfiles[compounds[0]][:stint1_len]) - beneficio_fuel_s1
                t_stint2 = np.sum(perfiles[compounds[1]][:stint2_len]) - beneficio_fuel_s2
                
                tiempo_total = t_stint1 + self.pit_loss + t_stint2
                
                if tiempo_total < mejor_tiempo:
                    mejor_tiempo = tiempo_total
                    vueltas_optimas = [p1]
        
        elif n_paradas == 2:
            # Estrategia de 2 Paradas: Stint A -> Stint B -> Stint C
            for p1 in range(1, total_race_laps - 1):
                # VERIFICACIÓN DE SEGURIDAD: Stint 1
                if p1 > limite_vueltas_seguras[compounds[0]]:
                    continue
                
                t_stint1 = np.sum(perfiles[compounds[0]][:p1])
                for p2 in range(p1 + 1, total_race_laps):
                    stint2_len = p2 - p1
                    stint3_len = total_race_laps - p2
                    
                    # VERIFICACIÓN DE SEGURIDAD: Stint 2 y Stint 3
                    if stint2_len > limite_vueltas_seguras[compounds[1]] or stint3_len > limite_vueltas_seguras[compounds[2]]:
                        continue

                    beneficio_fuel_s1 = np.sum(np.arange(1, p1 + 1)) * 0.04
                    beneficio_fuel_s2 = np.sum(np.arange(p1 + 1, p2 + 1)) * 0.04
                    beneficio_fuel_s3 = np.sum(np.arange(p2 + 1, total_race_laps + 1)) * 0.04

                    t_stint1 = np.sum(perfiles[compounds[0]][:p1]) - beneficio_fuel_s1
                    t_stint2 = np.sum(perfiles[compounds[1]][:stint2_len]) - beneficio_fuel_s2
                    t_stint3 = np.sum(perfiles[compounds[2]][:stint3_len]) - beneficio_fuel_s3
                    
                    tiempo_total = t_stint1 + self.pit_loss + t_stint2 + self.pit_loss + t_stint3
                    
                    if tiempo_total < mejor_tiempo:
                        mejor_tiempo = tiempo_total
                        vueltas_optimas = [p1, p2]

        # Manejo de caso donde ninguna combinación es segura
        if mejor_tiempo == float('inf'):
            return {
                'vueltas_optimas': [],
                'tiempo_total': float('inf'),
                'race_trace': np.array([]),
                'compounds_trace': []
            }

        # Construir el 'Race Trace' óptimo final
        race_trace = []
        compounds_trace = []
        periodos = vueltas_optimas + [total_race_laps]
        current_absolute_lap = 1
        for i, fin_lap in enumerate(periodos):
            comp = compounds[i]
            duracion = fin_lap - (periodos[i-1] if i > 0 else 0)
            segmento_base = perfiles[comp][:duracion]
            
            # Aplicar efecto de combustible vuelta a vuelta
            for v_stint in range(duracion):
                t_final = segmento_base[v_stint] - (current_absolute_lap * 0.04)
                race_trace.append(t_final)
                current_absolute_lap += 1
                
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
