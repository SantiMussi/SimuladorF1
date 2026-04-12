import numpy as np

def rk45_step(f, t, y, h, tol=1e-6):
    """
    Implementación de Runge-Kutta-Fehlberg (Paso Adaptativo).
    """
    # Coeficientes de Butcher para RKF45
    c2, a21 = 1/4, 1/4
    c3, a31, a32 = 3/8, 3/32, 9/32
    c4, a41, a42, a43 = 12/13, 1932/2197, -7200/2197, 7296/2197
    c5, a51, a52, a53, a54 = 1, 439/216, -8, 3680/513, -845/4104
    c6, a61, a62, a63, a64, a65 = 1/2, -8/27, 2, -3544/2565, 1859/4104, -11/40

    # Pesos para orden 4 y 5
    b = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    bst = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]

    k1 = h * f(t, y)
    k2 = h * f(t + c2*h, y + a21*k1)
    k3 = h * f(t + c3*h, y + a31*k1 + a32*k2)
    k4 = h * f(t + c4*h, y + a41*k1 + a42*k2 + a43*k3)
    k5 = h * f(t + c5*h, y + a51*k1 + a52*k2 + a53*k3 + a54*k4)
    k6 = h * f(t + c6*h, y + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)

    # Estimación de orden 4 y 5
    y_next = y + b[0]*k1 + b[2]*k3 + b[3]*k4 + b[4]*k5
    y_star = y + bst[0]*k1 + bst[2]*k3 + bst[3]*k4 + bst[4]*k5 + bst[5]*k6

    # Error y ajuste de paso
    te = np.abs(y_next - y_star)
    if te <= tol:
        h_next = h * 0.84 * (tol / te)**0.25 if te > 0 else h * 2
        return y_next, t + h, h_next, True
    else:
        h_new = h * 0.84 * (tol / te)**0.25
        return y, t, h_new, False

def least_squares_poly(x, y, degree=3):
    """
    Ajuste polinomial mediante Mínimos Cuadrados (Ecuaciones Normales).
    """
    X = np.vander(x, degree + 1)
    # Resolver (X^T * X) * beta = X^T * y
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    return beta

def evaluate_poly(beta, x):
    return np.polyval(beta, x)

def poly_derivative(beta):
    return np.polyder(beta)

def newton_raphson(beta, target_val, x0, tol=1e-5, max_iter=100):
    """
    Encuentra la raíz de f(x) - target_val = 0 utilizando la forma analítica del polinomio.
    """
    beta_target = beta.copy()
    beta_target[-1] -= target_val # f(x) - target_val
    
    beta_der = poly_derivative(beta_target)
    
    x = x0
    for _ in range(max_iter):
        fx = np.polyval(beta_target, x)
        dfx = np.polyval(beta_der, x)
        
        if abs(dfx) < 1e-9: break
        
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def romberg_integration(f_poly, a, b, steps=5):
    """
    Integración de Romberg para calcular tiempo total.
    """
    R = np.zeros((steps, steps))
    h = b - a
    R[0, 0] = 0.5 * h * (np.polyval(f_poly, a) + np.polyval(f_poly, b))
    
    for i in range(1, steps):
        h /= 2
        sum_f = sum(np.polyval(f_poly, a + k * h) for k in range(1, 2**i, 2))
        R[i, 0] = 0.5 * R[i-1, 0] + h * sum_f
        
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
def simpson_integral(f_poly, a, b, n=50):
    """
    Integración por Regla de Simpson (1/3) para mayor precisión en sectores.
    """
    if n % 2 == 1: n += 1 # n debe ser par
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.polyval(f_poly, x)
    
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S

def montecarlo_simulation(f_poly, a, b, samples=1000):
    """
    Simulación de Montecarlo para validar variabilidad del ritmo en el sector.
    """
    x_rand = np.random.uniform(a, b, samples)
    y_vals = np.polyval(f_poly, x_rand)
    return np.mean(y_vals) * (b - a)
