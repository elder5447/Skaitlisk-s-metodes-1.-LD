import numpy as np
import matplotlib.pyplot as plt

ALPHA0_TABULAI = [k * np.pi / 20 for k in range(1, 11)]
ALPHA0_GRAFIKAM = np.linspace(0.02 * np.pi, 0.999 * np.pi / 2, 200)
N_INTEGRACIJAI = 2000
ALPHA0_UZD45 = np.pi / 3
N_KONVERGENCEI = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
N_KONVERGENCEI_FIT = [4, 8, 16, 32, 64, 128]
PERIODI_RADIT = 1.5


def integrandis_g(u, alpha0):
    u = np.asarray(u, dtype=float)
    sauc = np.cos(alpha0 - u**2) - np.cos(alpha0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 2 * u / np.sqrt(sauc)
    return np.where(np.abs(u) < 1e-12, 2 / np.sqrt(np.sin(alpha0)), g)


def J_integralis(alpha0, N=2000, formula=1):
    if alpha0 <= 0: raise ValueError("alpha0 jābūt > 0")
    if formula == 1 and N < 4: raise ValueError("1. formulai nepieciešams N >= 4")
    if formula == 2 and N < 6: raise ValueError("2. formulai nepieciešams N >= 6")
    u_max = np.sqrt(alpha0)
    u = np.linspace(0, u_max, N)
    h = u_max / (N - 1)
    f = integrandis_g(u, alpha0)
    if formula == 1:
        S = 1.5 * f[1] + np.sum(f[2:N-2]) + 1.5 * f[N-2]
    elif formula == 2:
        S = (23 / 12) * f[1] + (7 / 12) * f[2] + np.sum(f[3:N-3]) + (7 / 12) * f[N-3] + (23 / 12) * f[N-2]
    else:
        raise ValueError("formula jābūt 1 vai 2")
    return h * S


def alpha_no_t(alpha0, n_fine=4000, n_query=3000, periodi=PERIODI_RADIT):
    x_max = np.sqrt(alpha0)
    x = np.linspace(0, x_max, n_fine + 1)
    g_vals = integrandis_g(x, alpha0)
    dx = x[1] - x[0]
    G = np.zeros_like(x)
    G[1:] = np.cumsum(0.5 * (g_vals[:-1] + g_vals[1:]) * dx)

    ceturt_period = G[-1] / np.sqrt(2)
    puse_period = 2 * ceturt_period
    pilns_period = 2 * puse_period

    def t_no_alpha(alpha):
        alpha = np.atleast_1d(alpha).astype(float)
        rez = np.empty_like(alpha)
        m1 = alpha >= 0
        y1 = np.sqrt(np.clip(alpha0 - alpha[m1], 0, None))
        rez[m1] = np.interp(y1, x, G) / np.sqrt(2)
        m2 = ~m1
        y2 = np.sqrt(np.clip(alpha0 + alpha[m2], 0, None))
        rez[m2] = puse_period - np.interp(y2, x, G) / np.sqrt(2)
        return rez

    alpha_reste = np.linspace(alpha0, -alpha0, 2 * n_fine + 1)
    t_reste = t_no_alpha(alpha_reste)
    kart = np.argsort(t_reste)
    t_kart, alpha_kart = t_reste[kart], alpha_reste[kart]
    t_vaic = np.linspace(0, periodi * pilns_period, n_query)
    alpha_vaic = np.empty_like(t_vaic)

    for i, t in enumerate(t_vaic):
        tm = t % pilns_period
        s = tm if tm <= puse_period else pilns_period - tm
        alpha_vaic[i] = np.interp(s, t_kart, alpha_kart)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t_vaic, alpha_vaic, lw=1.8)
    plt.axhline(0, color="gray", lw=0.7)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\alpha(t)$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return t_vaic, alpha_vaic, pilns_period


alpha_no_t(ALPHA0_UZD45)