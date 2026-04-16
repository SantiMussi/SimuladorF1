import numpy as np


def _to_array(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def rk45_step(f, t, y, h, tol=1e-6):
    """
    Implementación de Runge-Kutta-Fehlberg (4,5) con paso adaptativo.
    Soporta estados escalares o vectoriales.
    """
    y_arr = np.asarray(y, dtype=float)
    scalar_input = y_arr.ndim == 0
    if scalar_input:
        y_arr = y_arr.reshape(1)

    def call_f(tt, yy):
        try:
            out = f(tt, yy if not scalar_input else float(np.asarray(yy).reshape(-1)[0]))
        except Exception:
            out = f(tt, float(np.asarray(yy).reshape(-1)[0]))
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            out = out.reshape(1)
        return out

    # Coeficientes de Butcher para RKF45
    c2, a21 = 1 / 4, 1 / 4
    c3, a31, a32 = 3 / 8, 3 / 32, 9 / 32
    c4, a41, a42, a43 = 12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197
    c5, a51, a52, a53, a54 = 1, 439 / 216, -8, 3680 / 513, -845 / 4104
    c6, a61, a62, a63, a64, a65 = 1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40

    # Pesos para orden 4 y 5
    b = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0], dtype=float)
    bst = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55], dtype=float)

    k1 = h * call_f(t, y_arr)
    k2 = h * call_f(t + c2 * h, y_arr + a21 * k1)
    k3 = h * call_f(t + c3 * h, y_arr + a31 * k1 + a32 * k2)
    k4 = h * call_f(t + c4 * h, y_arr + a41 * k1 + a42 * k2 + a43 * k3)
    k5 = h * call_f(t + c5 * h, y_arr + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    k6 = h * call_f(t + c6 * h, y_arr + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)

    y_next = y_arr + b[0] * k1 + b[2] * k3 + b[3] * k4 + b[4] * k5
    y_star = y_arr + bst[0] * k1 + bst[2] * k3 + bst[3] * k4 + bst[4] * k5 + bst[5] * k6

    te = float(np.max(np.abs(y_next - y_star)))

    if te <= tol:
        h_next = h * 2.0 if te == 0 else h * 0.84 * (tol / te) ** 0.25
        result = y_next
        return (float(result[0]), t + h, h_next, True) if scalar_input else (result, t + h, h_next, True)

    h_new = h * 0.84 * (tol / max(te, 1e-15)) ** 0.25
    return (float(y_arr[0]), t, h_new, False) if scalar_input else (y_arr, t, h_new, False)


def least_squares_poly(x, y, degree=3):
    """
    Ajuste polinomial mediante mínimos cuadrados.
    Devuelve coeficientes en formato compatible con np.polyval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < degree + 1:
        raise ValueError("No hay suficientes puntos para ajustar el polinomio.")

    X = np.vander(x, degree + 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def evaluate_poly(beta, x):
    return np.polyval(np.asarray(beta, dtype=float), x)


def poly_derivative(beta):
    return np.polyder(np.asarray(beta, dtype=float))


def newton_raphson(func_or_coeffs, target_val=0.0, x0=0.0, df=None, tol=1e-6, max_iter=100):
    """
    Newton-Raphson general.

    Formas admitidas:
    - newton_raphson(coef_pol, target_val, x0)
    - newton_raphson(f, x0=..., df=df_callable)
    """
    # Caso 1: se pasa una función
    if callable(func_or_coeffs):
        f = func_or_coeffs

        def g(x):
            return float(f(x) - target_val)

        if df is None:
            def dg(x, eps=1e-6):
                return float((g(x + eps) - g(x - eps)) / (2 * eps))
        else:
            def dg(x):
                return float(df(x))

        x = float(x0)
        for _ in range(max_iter):
            fx = g(x)
            dfx = dg(x)
            if abs(dfx) < 1e-12:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return float(x_new)
            x = float(x_new)
        return float(x)

    # Caso 2: se pasan coeficientes polinomiales
    beta = np.asarray(func_or_coeffs, dtype=float)
    beta_target = beta.copy()
    beta_target[-1] -= float(target_val)
    beta_der = np.polyder(beta_target)

    x = float(x0)
    for _ in range(max_iter):
        fx = float(np.polyval(beta_target, x))
        dfx = float(np.polyval(beta_der, x))
        if abs(dfx) < 1e-12:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return float(x_new)
        x = float(x_new)
    return float(x)


def simpson_integral(f_or_coeffs, a, b, n=50):
    """
    Regla de Simpson 1/3.
    Admite:
    - una función f(x)
    - un vector de coeficientes polinomiales
    """
    n = int(n)
    if n < 2:
        n = 2
    if n % 2 == 1:
        n += 1

    x = np.linspace(a, b, n + 1)

    if callable(f_or_coeffs):
        try:
            y = np.asarray(f_or_coeffs(x), dtype=float)
            if y.shape != x.shape:
                raise ValueError
        except Exception:
            y = np.array([float(f_or_coeffs(float(xi))) for xi in x], dtype=float)
    else:
        coeffs = np.asarray(f_or_coeffs, dtype=float)
        y = np.polyval(coeffs, x)

    S = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return float((b - a) * S / (3.0 * n))


def romberg_integration(f_or_coeffs, a, b, steps=5):
    """
    Integración Romberg.
    Devuelve la mejor aproximación final.
    """
    steps = max(1, int(steps))
    R = np.zeros((steps, steps), dtype=float)

    def f(x):
        if callable(f_or_coeffs):
            return float(f_or_coeffs(x))
        return float(np.polyval(np.asarray(f_or_coeffs, dtype=float), x))

    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, steps):
        h /= 2.0
        sum_f = sum(f(a + k * h) for k in range(1, 2**i, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_f

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return float(R[steps - 1, steps - 1])


def montecarlo_simulation(f_or_coeffs, a, b, samples=1000):
    """
    Estimación Monte Carlo simple del área bajo la curva.
    """
    samples = max(1, int(samples))
    x_rand = np.random.uniform(a, b, samples)

    if callable(f_or_coeffs):
        try:
            y_vals = np.asarray(f_or_coeffs(x_rand), dtype=float)
        except Exception:
            y_vals = np.array([float(f_or_coeffs(float(xi))) for xi in x_rand], dtype=float)
    else:
        coeffs = np.asarray(f_or_coeffs, dtype=float)
        y_vals = np.polyval(coeffs, x_rand)

    return float(np.mean(y_vals) * (b - a))