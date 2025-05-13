from scipy.special import ive, iv
from scipy.interpolate import interp1d
import numpy as np

def k(a, da, t, q, sigma=2):
    return 0.5 * (q - 0.5 * sigma - da(t))

def psi(a, da, t, z, tau, q, sigma=2):
    epsilon = 1e-10  # Small value to avoid division by zero
    kk = k(a, da, t, q, sigma)
    ratio = 2 * np.sqrt(a(t) * z) / (sigma * (t - tau + epsilon))

    if ratio <= 700:
        term1 = 1. / (sigma * (t - tau + epsilon)) * np.exp(- (a(t) + z) / (sigma * (t - tau + epsilon)))
        term2 = (a(t) / z)**(0.5 * (q - sigma) / sigma)
        term3 = da(t) - (a(t) / (t - tau + epsilon)) + kk
        term4 = iv(q / sigma - 1, ratio)
        term5 = (np.sqrt(a(t) * z) / (t - tau + epsilon)) * iv(q / sigma, ratio)
    else:
        term1 = 1. / (sigma * (t - tau + epsilon))
        term2 = (a(t) / z)**(0.5 * (q - sigma) / sigma)
        term3 = da(t) - (a(t) / (t - tau + epsilon)) + kk
        term4 = ive(q / sigma - 1, (a(t) + z) / (sigma * (t - tau + epsilon)))
        term5 = (np.sqrt(a(t) * z) / (t - tau + epsilon)) * ive(q / sigma, (a(t) + z) / (sigma * (t - tau + epsilon)))

    return term1 * term2 * (term3 * term4 + term5)

def fpt(a, da, q, z, sigma=2, dt=0.1, T_max=2):
    g = [0]
    T = [0]
    try:
        g.append(-2 * psi(a, da, dt, z, 0, q, sigma))
    except ZeroDivisionError:
        g.append(0)  # Handle division by zero gracefully
    T.append(dt)

    for n in range(2, int(T_max / dt) + 2):
        try:
            s = -2 * psi(a, da, n * dt, z, 0, q, sigma)
        except ZeroDivisionError:
            s = 0  # Handle division by zero gracefully

        for j in range(1, n):
            if a(j * dt) <= 0:  # Skip invalid `a` values
                continue
            try:
                s += 2 * dt * g[j] * psi(a, da, n * dt, a(j * dt), j * dt, q, sigma)
            except ZeroDivisionError:
                continue

        g.append(s)
        T.append(n * dt)

    g = np.asarray(g)
    T = np.asarray(T)

    gt = interp1d(T, g, fill_value="extrapolate")
    return gt

