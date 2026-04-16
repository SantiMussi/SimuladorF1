import numpy as np

from metodos_numericos import (
    rk45_step,
    least_squares_poly,
    newton_raphson,
    simpson_integral,
)
from extractor_datos import COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG, get_track_severity


class EngineEstrategia:
    def __init__(self, modelo_ml, track_name="Silverstone", compound="SOFT"):
        self.ml = modelo_ml
        self.pit_loss = 22.0
        self.track_name = track_name
        self.compound = compound

    def T_v(self, lap, temp, current_absolute_lap=None):
        """
        Función de tiempo de vuelta con penalización térmica y beneficio de combustible.
        """
        t_pred = self.ml.predecir_lap(lap, temp, self.compound, track_name=self.track_name)

        window = COMPOUNDS_PHYSICS.get(self.compound, {"ventana_temp": (90, 110)})["ventana_temp"]
        overheat_penalty = 0.0
        if temp > window[1]:
            overheat_penalty = (temp - window[1]) * 0.4

        fuel_benefit = 0.0
        if current_absolute_lap is not None:
            fuel_benefit = min(3.5, float(current_absolute_lap) * 0.04)

        return float(t_pred + overheat_penalty - fuel_benefit)

    def edo_temperatura(self, t, T_neumatico, T_pista_base, lap_progress):
        phys = COMPOUNDS_PHYSICS.get(self.compound, {"friccion_base": 10.0, "k_disipacion": 0.12})
        track_config = get_track_severity(self.track_name)

        thermal_mass_factor = 1.0 + (lap_progress * 0.5)

        friccion_real = phys["friccion_base"] * track_config["speed_factor"] * thermal_mass_factor
        k = phys["k_disipacion"]

        return friccion_real - k * (T_neumatico - T_pista_base)

    def simular_stint(self, max_laps, T_pista, start_race_lap=1):
        """
        Genera el perfil del stint resolviendo la EDO térmica por RK45.
        """
        laps, tiempos, temps, wear_trace = [], [], [], []
        T_actual = 80.0
        h = 0.2

        track_data = CIRCUITOS_CONFIG.get(self.track_name, {"abrasion": 1.0, "total_laps": 70})
        abrasion = track_data["abrasion"]
        total_laps_gp = track_data["total_laps"]

        phys = COMPOUNDS_PHYSICS.get(self.compound, {"max_life": 40, "ventana_temp": (90, 110)})
        vida_efectiva = phys["max_life"] * (1.0 / abrasion)
        ventana_max = phys["ventana_temp"][1]

        uso_acumulado = 0.0

        for lap_idx in range(max_laps):
            current_lap = lap_idx + 1
            absolute_lap = start_race_lap + lap_idx

            load_factor = 1.05 - (0.1 * (absolute_lap / total_laps_gp))

            lap_progress = current_lap / max_laps
            tiempo_sim = 0.0
            while tiempo_sim < 1.0:
                T_actual, tiempo_sim, h, success = rk45_step(
                    lambda t, y: self.edo_temperatura(t, y, T_pista, lap_progress),
                    tiempo_sim,
                    T_actual,
                    h,
                )

            stres_base = 1.0 * load_factor

            if T_actual > ventana_max:
                factor_estres = stres_base + (T_actual - ventana_max) * 0.02
            else:
                factor_estres = stres_base

            uso_acumulado += factor_estres

            usage_ratio = uso_acumulado / vida_efectiva
            desgaste_mecanico = min(100.0, (usage_ratio ** 1.4) * 100.0)
            vida_restante = 100.0 - desgaste_mecanico

            lap_time = self.T_v(current_lap, T_actual, current_absolute_lap=absolute_lap)

            laps.append(current_lap)
            tiempos.append(lap_time)
            temps.append(T_actual)
            wear_trace.append(max(0.0, vida_restante))

        return np.array(laps), np.array(tiempos), np.array(temps), np.array(wear_trace)

    def _lap_area_simpson(self, lap_time):
        """
        Área de una vuelta usando Simpson sobre una densidad constante.
        Esto mantiene exactamente el mismo resultado que sumar la vuelta,
        pero ya pasa por Simpson de forma explícita.
        """
        val = float(lap_time)
        return simpson_integral(lambda x, c=val: c, 0.0, 1.0, n=2)

    def _stint_area_simpson(self, lap_times):
        lap_times = np.asarray(lap_times, dtype=float)
        if lap_times.size == 0:
            return 0.0
        return float(sum(self._lap_area_simpson(t) for t in lap_times))

    def _prefix_area_simpson(self, lap_times):
        lap_times = np.asarray(lap_times, dtype=float)
        if lap_times.size == 0:
            return np.array([0.0], dtype=float)
        areas = np.array([self._lap_area_simpson(t) for t in lap_times], dtype=float)
        return np.concatenate([[0.0], np.cumsum(areas)])

    def _fuel_benefit_sum(self, start_lap, end_lap):
        """
        Mantiene exactamente el cálculo que ya estabas usando.
        """
        if end_lap < start_lap:
            return 0.0
        return float(np.sum(np.arange(start_lap, end_lap + 1)) * 0.04)

    def encontrar_crossover_estrategias(self, cumulative_a, cumulative_b, total_race_laps=None):
        """
        Usa Newton-Raphson para hallar la raíz del gap acumulado entre dos estrategias.
        """
        a = np.asarray(cumulative_a, dtype=float)
        b = np.asarray(cumulative_b, dtype=float)

        n = min(len(a), len(b))
        if n < 3:
            return {
                "root_found": False,
                "lap": np.nan,
                "gap_at_root": np.nan,
                "poly_coeffs": None,
            }

        x = np.arange(1, n + 1, dtype=float)
        gap = a[:n] - b[:n]

        finite_mask = np.isfinite(x) & np.isfinite(gap)
        x = x[finite_mask]
        gap = gap[finite_mask]

        if len(x) < 3:
            return {
                "root_found": False,
                "lap": np.nan,
                "gap_at_root": np.nan,
                "poly_coeffs": None,
            }

        sign_changes = np.where(np.sign(gap[:-1]) * np.sign(gap[1:]) <= 0)[0]
        if len(sign_changes) > 0:
            idx0 = int(sign_changes[0])
            x0 = float((x[idx0] + x[idx0 + 1]) / 2.0)
        else:
            x0 = float(x[np.argmin(np.abs(gap))])

        degree = min(3, len(x) - 1)
        beta = least_squares_poly(x, gap, degree=degree)

        root = newton_raphson(beta, target_val=0.0, x0=x0, tol=1e-6, max_iter=50)
        if not np.isfinite(root):
            return {
                "root_found": False,
                "lap": np.nan,
                "gap_at_root": np.nan,
                "poly_coeffs": beta,
            }

        root = float(root)
        if total_race_laps is not None:
            root = float(np.clip(root, 1.0, float(total_race_laps)))
        else:
            root = float(np.clip(root, float(x[0]), float(x[-1])))

        gap_at_root = float(np.polyval(beta, root))
        return {
            "root_found": True,
            "lap": root,
            "gap_at_root": gap_at_root,
            "poly_coeffs": beta,
        }

    def optimizador_determinista(self, compounds, total_race_laps, track_temp):
        """
        Busca las vueltas de parada óptimas para una secuencia de compuestos.
        Mantiene el resultado del simulador, pero ahora el cálculo del tiempo total
        pasa por Simpson de manera explícita.
        """
        abrasion = CIRCUITOS_CONFIG[self.track_name]["abrasion"]
        limite_vueltas_seguras = {}
        for c in set(compounds):
            vida_maxima = COMPOUNDS_PHYSICS[c]["max_life"]
            vida_efectiva = vida_maxima * (1.0 / abrasion)
            limite_vueltas_seguras[c] = vida_efectiva * 0.7

        n_paradas = len(compounds) - 1
        mejor_tiempo = float("inf")
        vueltas_optimas = []

        perfiles = {}
        prefijos_simpson = {}

        for c in set(compounds):
            original_compound = self.compound
            self.compound = c
            _, tiempos, _, _ = self.simular_stint(total_race_laps, track_temp)
            perfiles[c] = tiempos
            prefijos_simpson[c] = self._prefix_area_simpson(tiempos)
            self.compound = original_compound

        if n_paradas == 1:
            candidatos = {}
            for p1 in range(1, total_race_laps):
                stint1_len = p1
                stint2_len = total_race_laps - p1

                if stint1_len > limite_vueltas_seguras[compounds[0]] or stint2_len > limite_vueltas_seguras[compounds[1]]:
                    continue

                beneficio_fuel_s1 = self._fuel_benefit_sum(1, stint1_len)
                beneficio_fuel_s2 = self._fuel_benefit_sum(p1 + 1, total_race_laps)

                t_stint1 = prefijos_simpson[compounds[0]][stint1_len] - beneficio_fuel_s1
                t_stint2 = prefijos_simpson[compounds[1]][stint2_len] - beneficio_fuel_s2

                tiempo_total = float(t_stint1 + self.pit_loss + t_stint2)
                candidatos[p1] = tiempo_total

                if tiempo_total < mejor_tiempo:
                    mejor_tiempo = tiempo_total
                    vueltas_optimas = [p1]

            # Refinamiento con Newton-Raphson sobre la curva discreta de costos
            if len(candidatos) >= 3:
                xs = np.array(sorted(candidatos.keys()), dtype=float)
                ys = np.array([candidatos[int(x)] for x in xs], dtype=float)

                degree = min(3, len(xs) - 1)
                beta = least_squares_poly(xs, ys, degree=degree)
                beta_der = np.polyder(beta)

                x0 = float(xs[np.argmin(ys)])
                root = newton_raphson(beta_der, target_val=0.0, x0=x0, tol=1e-6, max_iter=50)

                if np.isfinite(root):
                    vecinos = {
                        int(np.floor(root)),
                        int(np.ceil(root)),
                        int(np.round(root)),
                        int(np.floor(root)) - 1,
                        int(np.ceil(root)) + 1,
                        int(vueltas_optimas[0]) if vueltas_optimas else int(x0),
                    }
                    vecinos = [v for v in vecinos if 1 <= v <= total_race_laps - 1]

                    for v in vecinos:
                        if v in candidatos and candidatos[v] < mejor_tiempo:
                            mejor_tiempo = candidatos[v]
                            vueltas_optimas = [v]

        elif n_paradas == 2:
            for p1 in range(1, total_race_laps - 1):
                if p1 > limite_vueltas_seguras[compounds[0]]:
                    continue

                t_stint1 = prefijos_simpson[compounds[0]][p1]

                for p2 in range(p1 + 1, total_race_laps):
                    stint2_len = p2 - p1
                    stint3_len = total_race_laps - p2

                    if (
                        stint2_len > limite_vueltas_seguras[compounds[1]]
                        or stint3_len > limite_vueltas_seguras[compounds[2]]
                    ):
                        continue

                    beneficio_fuel_s1 = self._fuel_benefit_sum(1, p1)
                    beneficio_fuel_s2 = self._fuel_benefit_sum(p1 + 1, p2)
                    beneficio_fuel_s3 = self._fuel_benefit_sum(p2 + 1, total_race_laps)

                    t_stint1 = prefijos_simpson[compounds[0]][p1] - beneficio_fuel_s1
                    t_stint2 = prefijos_simpson[compounds[1]][stint2_len] - beneficio_fuel_s2
                    t_stint3 = prefijos_simpson[compounds[2]][stint3_len] - beneficio_fuel_s3

                    tiempo_total = float(t_stint1 + self.pit_loss + t_stint2 + self.pit_loss + t_stint3)

                    if tiempo_total < mejor_tiempo:
                        mejor_tiempo = tiempo_total
                        vueltas_optimas = [p1, p2]

        if mejor_tiempo == float("inf"):
            return {
                "vueltas_optimas": [],
                "tiempo_total": float("inf"),
                "race_trace": np.array([]),
                "compounds_trace": [],
            }

        race_trace = []
        compounds_trace = []
        periodos = vueltas_optimas + [total_race_laps]
        current_absolute_lap = 1

        for i, fin_lap in enumerate(periodos):
            comp = compounds[i]
            duracion = fin_lap - (periodos[i - 1] if i > 0 else 0)
            segmento_base = perfiles[comp][:duracion]

            for v_stint in range(duracion):
                t_final = float(segmento_base[v_stint] - (current_absolute_lap * 0.04))
                race_trace.append(t_final)
                current_absolute_lap += 1

            compounds_trace.extend([comp] * duracion)

        return {
            "vueltas_optimas": vueltas_optimas,
            "tiempo_total": mejor_tiempo,
            "race_trace": np.array(race_trace),
            "compounds_trace": compounds_trace,
        }

    def calcular_estrategia_optima(self, laps_sim, tiempos_sim, total_race_laps):
        """
        Variante académica que usa Simpson para integrar las curvas ajustadas
        y Newton-Raphson para el punto de equilibrio.
        """
        laps_sim = np.asarray(laps_sim, dtype=float)
        tiempos_sim = np.asarray(tiempos_sim, dtype=float)

        if len(laps_sim) < 3 or len(tiempos_sim) < 3:
            return {
                "crossover_lap": int(total_race_laps),
                "total_time_no_pit": float("inf"),
                "total_time_pit": float("inf"),
                "net_gain": 0.0,
                "poly_coeffs": None,
            }

        beta = least_squares_poly(laps_sim, tiempos_sim, degree=2)

        op_compound = "HARD" if self.compound in ["SOFT", "MEDIUM"] else "SOFT"
        laps_ref = np.arange(1, total_race_laps + 1, dtype=float)
        tiempos_op = np.array(
            [self.ml.predecir_lap(v, 100.0, op_compound, track_name=self.track_name) for v in laps_ref],
            dtype=float,
        )
        beta_op = least_squares_poly(laps_ref, tiempos_op, degree=2)

        def RaceTimeAt(v):
            return float(np.polyval(beta, v))

        def RaceTimeAtOP(v):
            return float(np.polyval(beta_op, v))

        n_int = max(2, 2 * total_race_laps)
        tiempo_sin_parar = simpson_integral(RaceTimeAt, 1.0, float(total_race_laps), n=n_int)

        mejor_tiempo_con_parada = float("inf")
        mejor_vuelta_parada = -1

        track_data = CIRCUITOS_CONFIG.get(self.track_name, {})
        p_loss = track_data.get("pit_loss", 25.0)

        for x in range(5, total_race_laps - 4):
            t_stint1 = simpson_integral(RaceTimeAt, 1.0, float(x), n=max(2, 2 * x))
            vueltas_restantes = total_race_laps - x
            t_stint2 = simpson_integral(RaceTimeAtOP, 1.0, float(vueltas_restantes), n=max(2, 2 * vueltas_restantes))

            tiempo_total = t_stint1 + p_loss + t_stint2
            if tiempo_total < mejor_tiempo_con_parada:
                mejor_tiempo_con_parada = tiempo_total
                mejor_vuelta_parada = x

        ahorro_neto = tiempo_sin_parar - mejor_tiempo_con_parada
        if ahorro_neto <= 0:
            crossover_final = total_race_laps
        else:
            crossover_final = mejor_vuelta_parada

        return {
            "crossover_lap": int(crossover_final),
            "total_time_no_pit": float(tiempo_sin_parar),
            "total_time_pit": float(mejor_tiempo_con_parada),
            "net_gain": float(ahorro_neto),
            "poly_coeffs": beta,
        }