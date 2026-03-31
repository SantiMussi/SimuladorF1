from altair import NumericArrayMarkPropDef
import numpy as np 

# Integracion numerica

def simspon_1_3(f, a, b, n):
    """ Regla de Simspon 1/3 para calcular areas bajo la curva.
    En el simulador >> Se usa para intregrar la curva de degradacion 
    y obtener el tiempo total acumulado en un stint (conjunto de vueltas)

    f: Funcion a integrar (ej>> funcion de tiempo de vuelta)
    a: limite inferior (ej: vuelta inicial)
    b: limite superior (vuelta final)
    n: numero de subintervalos (debe ser par)
    return: valor de la integral (tiempo acumulado)
    """

    if  n%2 != 0:
        n+=1 #Asegurar q n sea par

    h = (b - a ) / n
    suma = f(a) + f(b)

    for i in range(1,n):
        x_i = a + i * h
        if i%2 == 0:
            suma += 2 * f(x_i)
        else:
            suma += 4  * f(x_i)

    return (h/3) * suma

