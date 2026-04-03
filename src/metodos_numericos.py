import numpy as np 

# INTEGRACION NUMERICA (Simpson)
def simpson_1_3(f, a, b, n):
    """ Regla de Simpson 1/3 para calcular areas bajo la curva.
    En el simulador >> Se usa para intregrar la curva de degradacion 
    y obtener el tiempo total acumulado en un stint (conjunto de vueltas)

    f: Funcion a integrar (ej>> funcion de tiempo de vuelta)
    a: limite inferior (ej: vuelta inicial)
    b: limite superior (vuelta final)
    n: numero de subintervalos (debe ser par)
    return: valor de la integral (tiempo acumulado)
    """

    if n % 2 != 0:
        n += 1 # Asegurar q n sea par

    h = (b - a) / n
    suma = f(a) + f(b)

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            suma += 2 * f(x_i)
        else:
            suma += 4 * f(x_i)

    return (h / 3) * suma

# ECUACIONES DIFERENCIALES ORDINARIAS (EDO)
def runge_kutta_4(f, t0, y0, t_end, h):
    """
    Método de Runge-Kutta de 4to orden (RK4).
    En el simulador >> Predice la evolución térmica del neumático vuelta a vuelta.
    
    f: EDO que rige el cambio de temperatura (dy/dt = f(t, y))
    t0: Tiempo/Vuelta inicial
    y0: Temperatura inicial del neumático
    t_end: Tiempo/Vuelta final a simular
    h: Tamaño del paso (ej. 1 vuelta)
    return: Listas de tiempos y temperaturas calculadas
    """
    t_values = [t0]
    y_values = [y0]
    
    t = t0
    y = y0
    
    while t < t_end:
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        
        y = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        
        t_values.append(t)
        y_values.append(y)
        
    return np.array(t_values), np.array(y_values)

# BÚSQUEDA DE RAÍCES (CROSSOVER POINT) (Newton  Raphson)
def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    """
    Encuentra la raíz de una función usando su derivada.
    En el simulador >> Encuentra la vuelta exacta donde el tiempo perdido por 
    desgaste supera al tiempo que se pierde haciendo un Pit Stop (Crossover).
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10:
            raise ValueError("La derivada es muy cercana a cero. Cambiar método a Bisección.")
            
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new
            
        x = x_new
        
    raise ValueError("El método no convergió después de las iteraciones máximas.")

def biseccion(f, a, b, tol=1e-5, max_iter=100):
    """
    Método de bisección. Plan B robusto para encontrar raíces.
    En el simulador >> Se ejecuta automáticamente si Newton-Raphson falla.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("El intervalo no contiene una raíz (los signos no son opuestos).")
        
    for _ in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            return c
            
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
            
    return (a + b) / 2

# INTERPOLACIÓN (lagrange)
def lagrange(x_puntos, y_puntos, x_interp):
    """
    Polinomio interpolador de Lagrange.
    En el simulador >> Rellena "huecos" de datos en la telemetría 
    (si la API de F1 pierde algún paquete de datos).
    """
    n = len(x_puntos)
    resultado = 0.0
    
    for i in range(n):
        termino = y_puntos[i]
        for j in range(n):
            if i != j:
                termino *= (x_interp - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        resultado += termino
        
    return resultado