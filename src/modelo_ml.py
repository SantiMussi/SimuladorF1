import pandas as pd 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def entrenar_modelo_degradacion(df):
    """
    Entrena el modelo de ML (Regresión Polinomial) para predecir el tiempo 
    de vuelta basándose en el desgaste del neumático y el compuesto.
    """

    #Transformamos el texto en numeros
    #Convierte 'Compound' en columnas como "COMPOUND_SOFT" (1 o 0)
    df_ml = pd.get_dummies(df, columns=["Compound"], drop_first=False)

    #Definir que miramos (X) y que queremos predecir (Y)
    columnas_x = [col for col in df_ml.columns if col in ['TyreLife', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']]
    X = df_ml[columnas_x]

    #Prediccion: tiempo de vuelta en segundos
    y = df_ml['LapTime_sec']

    #Separar 80% para entrenar y 20% para evaluar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- NUEVO: Magia Polinomial (Grado 2 = Curva Parabólica) ---
    # Esto transforma "Vueltas" en "Vueltas al cuadrado" para capturar la degradación acelerada
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    #Crear y entrenar la ia (Regresión Lineal sobre las características polinomialicas)
    modelo = LinearRegression()
    modelo.fit(X_train_poly, y_train)

    #Evaluar a la IA con el 20% restante
    predicciones_test = modelo.predict(X_test_poly)
    error = mean_absolute_error(y_test, predicciones_test)
    precision = r2_score(y_test, predicciones_test)

    return modelo, poly, columnas_x, error, precision

#TEST
if __name__ == "__main__":
    from extractor_datos import obtener_telemetria_limpia

    df = obtener_telemetria_limpia(2024, 'Bahrain', 'R', 'VER')

    print('Entrenando modelo de MLL Polinomial..')
    modelo, poly, columnas_entrenamiento, error, precision = entrenar_modelo_degradacion(df)
    print('Modelo entrenado!')
    
    print('\nRESULTADOS DE LA EVALUACION:')
    print(f'Error Promedio (MAE): Le erra por {error:.3f} segundos')
    print(f'Precision (R2): {precision * 100:.1f}%\n')

    #Le hacemos la pregunta imposible a la ia (Extrapolación)
    datos_futuros = pd.DataFrame({'TyreLife': [25]})
    
    for col in columnas_entrenamiento:
        if col not in datos_futuros.columns:
            datos_futuros[col] = 1 if col == 'Compound_SOFT' else 0

    datos_futuros = datos_futuros[columnas_entrenamiento]

    #Tenemos que transformar los datos futuros al mismo formato polinomial antes de predecir
    datos_futuros_poly = poly.transform(datos_futuros)

    #La magic de la ia
    prediccion = modelo.predict(datos_futuros_poly)
    print('PREDICCION DE LA IA')
    print(f'Con un neumatico SOFT con 25 vueltas, el tiempo estimado es {prediccion[0]:.3f} segundos')