import numpy as np
import pandas as pd
from src.metodos_numericos import newton_raphson, simpson_1_3, biseccion, runge_kutta_4
from src.modelo_ml import entrenar_modelo_degradacion
from src.extractor_datos import obtener_telemetria_limpia

class MotorEstrategia:
    """Orquesta la IA y los metodos numericos para tomar decisiones de carrera"""
    def __init__(self, modelo, columnas_entrenamiento):
        self.modelo = modelo
        self.columnas_x = columnas_entrenamiento

        self.mapa_compuestos = {'Blando': 'SOFT', 'Medio': 'MEDIUM', 'Duro': 'HARD'}

    def crear_funcion_desgaste(self, compuesto_es, temp_pista):
        """ Envuelve la predicción de la IA en una función matemática continua T(v).
        Ahora integra el perfil térmico calculado por RK4 como feature de entrada.
        """
        compuesto_en = self.mapa_compuestos.get(compuesto_es, 'SOFT')
        
        # Pre-calculamos el perfil térmico con RK4 para asegurar consistencia física
        from src.extractor_datos import crear_edo_temperatura
        edo = crear_edo_temperatura(temp_pista, compuesto_en)
        # Simulamos hasta 80 vueltas con paso fino para una interpolación estable
        v_perfil, temp_perfil = runge_kutta_4(edo, t0=0, y0=80.0, t_end=80.0, h=0.5)
        
        def T(v):
            # Evitamos que el modelo intente calcular tiempo de vueltas negativas
            v = max(1.0, float(v))

            # Obtenemos la temperatura física para esa vuelta exacta (Interpolación)
            t_fisica = np.interp(v, v_perfil, temp_perfil)

            datos = pd.DataFrame({'TyreLife': [v], 'Temp_RK4': [t_fisica]})

            for col in self.columnas_x:
                if col not in datos.columns:
                    datos[col] = 1 if col == f'Compound_{compuesto_en}' else 0

            datos = datos[self.columnas_x]
            
            # La IA predice (el pipeline se encarga del escalado y poly features)
            return self.modelo.predict(datos)[0]
        
        return T

    def calcular_tiempo_stint(self, T, n_vueltas):
        """ Utiliza la Regla de Simpson 1/3 para integrar la función T(v) 
        y obtener el tiempo total acumulado (Rigor Académico).
        """
        if n_vueltas < 1: return 0
        # n debe ser par para Simpson 1/3
        return simpson_1_3(T, 1, n_vueltas, n=max(2, int(n_vueltas) if int(n_vueltas)%2==0 else int(n_vueltas)+1))

    def analizar_stint(self, compuesto_es, vuelta_actual, temp_pista, umbral_perdida = 1.5):
        """Calcula todo lo que necesita el app.py para llenar las metricas y graficos"""
        # Pasamos temp_pista para que T(v) conozca el perfil térmico
        T = self.crear_funcion_desgaste(compuesto_es, temp_pista)

        # Tiempos deltas
        tiempo_ideal = T(1) # Vuelta de qualy (referencia)
        tiempo_actual = T(vuelta_actual)
        delta = tiempo_actual - tiempo_ideal # Cuanto tiempo perdemos por desgaste

        # Crossover Point (Newton Raphson con Derivada Numérica por Diferencias Centrales)
        def P(v):
            return T(v) - tiempo_ideal - umbral_perdida
        
        def dP(v):
            h = 1e-4 # Paso para la derivada numérica
            return (P(v+h) - P(v- h)) / (2*h)

        try:
            # Buscamos la raíz donde la pérdida de pace cruza el umbral
            # x0 dinámico según compuesto para mejorar la convergencia
            x0_map = {'Blando': 15.0, 'Medio': 25.0, 'Duro': 35.0}
            punto_inicio = x0_map.get(compuesto_es, 20.0)
            
            crossover = max(1, int(round(newton_raphson(P, dP, x0=punto_inicio))))
        except Exception:
            # Fallback robusto a Bisección. 
            # Verificamos si existe un cambio de signo en [1, 60]
            if P(1) * P(60) < 0:
                crossover = max(1, int(round(biseccion(P, 1, 60))))
            else:
                # Si P(60) sigue siendo negativo, la vida útil supera las 60 vueltas
                crossover = 60

        # Tiempo Total Acumulado (Integración por Simpson 1/3)
        tiempo_total_stint = self.calcular_tiempo_stint(T, vuelta_actual)

        # Termodinámica con RK4 para la métrica instantánea
        from src.extractor_datos import crear_edo_temperatura
        compuesto_en = self.mapa_compuestos.get(compuesto_es, 'SOFT')
        edo = crear_edo_temperatura(temp_pista, compuesto_en)
        _, temps = runge_kutta_4(edo, t0=0, y0=80.0, t_end=vuelta_actual, h=1)
        temp_goma_rk4 = temps[-1]

        # Generar datos para el grafico de Plotly
        vueltas_proyeccion = list(range(1, 61))
        tiempos_proyeccion = [T(v) for v in vueltas_proyeccion]

        return {
            "tiempo_ideal": tiempo_ideal,
            "tiempo_actual": tiempo_actual,
            "delta_tiempo": delta,
            "crossover_point": crossover,
            "temp_goma": temp_goma_rk4,
            "tiempo_total": tiempo_total_stint,
            "grafico_x": vueltas_proyeccion,
            "grafico_y": tiempos_proyeccion
        }
