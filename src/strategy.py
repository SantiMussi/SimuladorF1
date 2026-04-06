import numpy as np
import pandas as pd
from metodos_numericos import newton_raphson, simpson_1_3, biseccion
from modelo_ml import entrenar_modelo_degradacion
from extractor_datos import obtener_telemetria_limpia

class MotorEstrategia:
    """Orquesta la IA y los metodos numericos para tomar decisiones de carrera"""
    def __init__(self, modelo, poly, columnas_entrenamiento):
        self.modelo = modelo
        self.poly = poly
        self.columnas_x = columnas_entrenamiento

    def crear_funcion_desgaste(self, compuesto):
        """ Envuelve la prediccion de la IA en una funcion matematica continua (Tv)"""
        def T(v):
            #Evitamos que el modelo intente calcular tiempo de vueltas negativas
            v = max(1.0, float(v))

            datos = pd.DataFrame({'TyreLife': [v]})

            for col in self.columnas_x:
                if col not in datos.columns:
                    datos[col] = 1 if col == f'Compound_{compuesto}' else 0

            datos = datos[self.columnas_x]
            datos_poly = self.poly.transform(datos)
            return self.modelo.predict(datos_poly)[0]
        
        return T

    # TODO: Calculador crossoverPoint (Newton-Raphson)
    #TODO: evaluar stint con simpson
    #TODO: Test breve