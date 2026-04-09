import numpy as np
import pandas as pd
from src.metodos_numericos import newton_raphson, simpson_1_3, biseccion, runge_kutta_4
from src.modelo_ml import entrenar_modelo_degradacion
from src.extractor_datos import obtener_telemetria_limpia

class MotorEstrategia:
    """Orquesta la IA y los metodos numericos para tomar decisiones de carrera"""
    def __init__(self, modelo, poly, columnas_entrenamiento):
        self.modelo = modelo
        self.poly = poly
        self.columnas_x = columnas_entrenamiento

        self.mapa_compuestos = {'Blando': 'SOFT', 'Medio': 'MEDIUM', 'Duro': 'HARD'}

    def crear_funcion_desgaste(self, compuesto_es):
        """ Envuelve la prediccion de la IA en una funcion matematica continua (Tv)"""
        compuesto_en = self.mapa_compuestos.get(compuesto_es, 'SOFT')
        
        def T(v):
            #Evitamos que el modelo intente calcular tiempo de vueltas negativas
            v = max(1.0, float(v))

            datos = pd.DataFrame({'TyreLife': [v]})

            for col in self.columnas_x:
                if col not in datos.columns:
                    datos[col] = 1 if col == f'Compound_{compuesto_en}' else 0

            datos = datos[self.columnas_x]
            datos_poly = self.poly.transform(datos)
            #La ia predice
            return self.modelo.predict(datos_poly)[0]
        
        return T

    def analizar_stint(self, compuesto_es, vuelta_actual, temp_pista, umbral_perdida = 2.5):
        """Calcula todo lo que necesita el app.py para llenar las metricas y graficos"""
        T = self.crear_funcion_desgaste(compuesto_es)

        # Tiempos deltas
        tiempo_ideal = T(1) #Vuelta de qualy
        tiempo_actual = T(vuelta_actual)
        delta = tiempo_actual - tiempo_ideal #Cuanto tiempo perdemos por desgaste

        # Crossover Point (Newton Raphson)
        def P(v):
            return T(v) - tiempo_ideal - umbral_perdida
        
        def dP(v):
            h = 1e-4
            return (P(v+h) - P(v- h)) / (2*h)

        try:
            crossover = max(1, int(round(newton_raphson(P, dP, x0=10.0))))
        except Exception:
            crossover = max(1, int(round(biseccion(P, 1, 60))))

        #Termodinamica con RK4
        #Importamos la EDO del extractor de datos
        from src.extractor_datos import crear_edo_temperatura
        edo = crear_edo_temperatura(temp_pista)
        _, temps = runge_kutta_4(edo, t0=0, y0=80.0, t_end=vuelta_actual, h=1)
        temp_goma_rk4 = temps[-1]

        #Generar datos para el grafico de Plotly
        vueltas_proyeccion = list(range(1, 61))
        tiempos_proyeccion = [T(v) for v in vueltas_proyeccion]

        return {
            "tiempo_ideal": tiempo_ideal,
            "tiempo_actual": tiempo_actual,
            "delta_tiempo": delta,
            "crossover_point": crossover,
            "temp_goma": temp_goma_rk4,
            "grafico_x": vueltas_proyeccion,
            "grafico_y": tiempos_proyeccion
        }
