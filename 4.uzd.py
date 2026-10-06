import numpy as np
import matplotlib.pyplot as plt

ALPHA0 = np.pi / 3
N_REF = 100000
N_LISTS = {
    "trapecveida":  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "Simpsona 1/3": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "Simpsona 3/8": [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536],
    "Bula likums":  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
}


def integrandis_g(u, alpha0):
    u = np.asarray(u, dtype=float)
    alpha = alpha0 - u**2
    saucejs = np.cos(alpha) - np.cos(alpha0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vertiba = 2 * u / np.sqrt(np.where(saucejs > 0, saucejs, np.nan))
    vertiba = np.where(u < 1e-10, 2.0 / np.sqrt(np.sin(alpha0)), vertiba)
    return vertiba


def trapezoid(y, h):
    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)


def simpson(y, h):         
    i = np.arange(0, len(y) - 2, 2)
    return np.sum(h / 3 * (y[i] + 4 * y[i + 1] + y[i + 2]))


def simpson38(y, h):        
    i = np.arange(0, len(y) - 3, 3)
    return np.sum(3 * h / 8 * (y[i] + 3 * y[i + 1] + 3 * y[i + 2] + y[i + 3]))


def boole(y, h):              
    i = np.arange(0, len(y) - 4, 4)
    return np.sum(2 * h / 45 * (7*y[i] + 32*y[i+1] + 12*y[i+2] + 32*y[i+3] + 7*y[i+4]))


METODES = {"trapecveida": trapezoid, "Simpsona 1/3": simpson,
           "Simpsona 3/8": simpson38, "Bula likums": boole}


def T_tilde(alpha0, N, metode):
    u_max = np.sqrt(alpha0)
    u = np.linspace(0.0, u_max, N + 1)
    y = integrandis_g(u, alpha0)
    h = u_max / N
    J = metode(y, h)
    return (np.sqrt(2) / np.pi) * J


T_ref = T_tilde(ALPHA0, N_REF, simpson)

plt.figure(figsize=(7, 5.5))
PLATO_SLIEKSNIS = 1e-9  #precizitātes slieksnis.
gamma_rezultati = {}
print(f"{'metode':>14} {'gamma (globāls fits)':>22}")
for nosaukums, f in METODES.items():
    N_list = N_LISTS[nosaukums]
    kludas = np.array([abs(T_tilde(ALPHA0, N, f) - T_ref) for N in N_list])
    plt.loglog(N_list, kludas, "o-", label=nosaukums)

    tiri = kludas > PLATO_SLIEKSNIS
    N_tiri = np.array(N_list)[tiri]
    E_tiri = kludas[tiri]

    # globāls pieskaņojums visiem "tīrajiem" punktiem
    gamma_global, _ = np.polyfit(np.log(N_tiri), np.log(E_tiri), 1)

    gamma_rezultati[nosaukums] = gamma_global
    print(f"{nosaukums:>14} {gamma_global:>22.3f}")

plt.xlabel("N (apakšintervālu skaits)")
plt.ylabel(r"$|\tilde T_N - \tilde T_{\mathrm{ref}}|$")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()
