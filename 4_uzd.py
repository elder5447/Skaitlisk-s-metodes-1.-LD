import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

ALPHA0 = np.pi / 3
N_REF = 100000
# Katrai metodei savs N saraksts (mazākais solis, kas apmierina tās
# integrēšanas prasību), lai pirms mašīnprecīzijas platō būtu pēc iespējas
# vairāk "tīru" punktu precīzai gamma noteikšanai:
N_LISTS = {
    "trapecveida":   [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "Simpsona 1/3":  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "Simpsona 3/8":  [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536],
    "Bula likums":   [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    # Gausa-Ležandra konverģē eksponenciāli (ne pēc pakāpes likuma), tāpēc
    # jau ar dažiem desmitiem punktu sasniedz mašīnprecīzijas platō - garam
    # N sarakstam šeit nav jēgas:
    "Gausa kvadratūra": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}


def integrandis_g(u, alpha0):
    """Desingularizētais integrandis (sk. svarsts.py, 2. uzdevums)."""
    u = np.asarray(u, dtype=float)
    alpha = alpha0 - u**2
    saucejs = np.cos(alpha) - np.cos(alpha0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vertiba = 2 * u / np.sqrt(np.where(saucejs > 0, saucejs, np.nan))
    vertiba = np.where(u < 1e-10, 2.0 / np.sqrt(np.sin(alpha0)), vertiba)
    return vertiba


def trapezoid(y, h):
    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)


def simpson(y, h):           # vajag N (=len(y)-1) pāra
    i = np.arange(0, len(y) - 2, 2)
    return np.sum(h / 3 * (y[i] + 4 * y[i + 1] + y[i + 2]))


def simpson38(y, h):         # vajag N dalāmu ar 3
    i = np.arange(0, len(y) - 3, 3)
    return np.sum(3 * h / 8 * (y[i] + 3 * y[i + 1] + 3 * y[i + 2] + y[i + 3]))


def boole(y, h):              # vajag N dalāmu ar 4
    i = np.arange(0, len(y) - 4, 4)
    return np.sum(2 * h / 45 * (7*y[i] + 32*y[i+1] + 12*y[i+2] + 32*y[i+3] + 7*y[i+4]))


def gauss_legendre(f, a, b, N):
    """N-punktu Gausa-Ležandra kvadratūra intervālā [a, b] (scipy.integrate.fixed_quad).
    Atšķirībā no pārējām metodēm šī NAV kompozītformula uz vienmērīga režģa -
    N ir Gausa mezglu skaits, kas tiek novietoti optimāli (nevis vienmērīgi)."""
    J, _ = integrate.fixed_quad(f, a, b, n=N)
    return J


METODES = {"trapecveida": trapezoid, "Simpsona 1/3": simpson,
           "Simpsona 3/8": simpson38, "Bula likums": boole,
           "Gausa kvadratūra": gauss_legendre}


def T_tilde(alpha0, N, metode):
    """Kompozītformulas (trapecveida/Simpsona/Bula) - vienmērīgs režģis ar N apakšintervāliem."""
    u_max = np.sqrt(alpha0)
    u = np.linspace(0.0, u_max, N + 1)
    y = integrandis_g(u, alpha0)
    h = u_max / N
    J = metode(y, h)
    return (np.sqrt(2) / np.pi) * J


def T_tilde_gauss(alpha0, N):
    """Gausa-Ležandra kvadratūra - integrē g(u) tieši ar N mezgliem uz [0, u_max]."""
    u_max = np.sqrt(alpha0)
    f = lambda u: integrandis_g(u, alpha0)
    J = gauss_legendre(f, 0.0, u_max, N)
    return (np.sqrt(2) / np.pi) * J


T_ref = T_tilde(ALPHA0, N_REF, simpson)

# 1.grafiks

plt.figure(figsize=(7, 5.5))
PLATO_SLIEKSNIS = 1e-9   # zem šī kļūda uzskatāma par mašīnprecīzijas troksni
gamma_rezultati = {}
print(f"{'metode':>14} {'gamma (globāls fits)':>22}")
for nosaukums, f in METODES.items():
    N_list = N_LISTS[nosaukums]
    if nosaukums == "Gausa kvadratūra":
        kludas = np.array([abs(T_tilde_gauss(ALPHA0, N) - T_ref) for N in N_list])
    else:
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
# plt.title(rf"Kvadratūras formulu salīdzinājums, $\alpha_0=\pi/3$")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("metozu_salidzinajums.png", dpi=150)
plt.show()
print("\nGrafiks saglabāts: metozu_salidzinajums.png")

# 2.grafiks

plt.figure(figsize=(7, 5.5))

PLATO_SLIEKSNIS = 1e-9
gamma_rezultati = {}

for nosaukums in ["trapecveida", "Simpsona 1/3",
                  "Simpsona 3/8", "Bula likums"]:

    f = METODES[nosaukums]
    N_list = N_LISTS[nosaukums]

    kludas = np.array([
        abs(T_tilde(ALPHA0, N, f) - T_ref)
        for N in N_list
    ])

    plt.loglog(
        N_list,
        kludas,
        "o-",
        label=nosaukums
    )

    # Izmanto tikai punktus virs mašīnprecizitātes trokšņa
    tiri = kludas > PLATO_SLIEKSNIS

    N_tiri = np.array(N_list)[tiri]
    E_tiri = kludas[tiri]

    gamma_global, _ = np.polyfit(
        np.log(N_tiri),
        np.log(E_tiri),
        1
    )

    gamma_rezultati[nosaukums] = gamma_global

    print(
        f"{nosaukums:>16} "
        f"gamma = {gamma_global: .3f}"
    )

plt.xlabel("N (apakšintervālu skaits)")
plt.ylabel(r"$|\tilde T_N-\tilde T_{\mathrm{ref}}|$")
plt.title(rf"Kompozīto kvadratūras formulu konverģence, $\alpha_0=\pi/3$")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()

plt.savefig(
    "kompozito_metozu_salidzinajums.png",
    dpi=150
)

plt.show()


# 3. grafiks

plt.figure(figsize=(7, 5.5))

N_list = N_LISTS["Gausa kvadratūra"]

kludas_gauss = np.array([
    abs(T_tilde_gauss(ALPHA0, N) - T_ref)
    for N in N_list
])

plt.semilogy(
    N_list,
    kludas_gauss,
    "o-",
    label="Gausa-Ležandra"
)

plt.xlabel("N (Gausa mezglu skaits)")
plt.ylabel(r"$|\tilde T_N-\tilde T_{\mathrm{ref}}|$")

plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()

plt.savefig(
    "gausa_kvadratūras_konvergence.png",
    dpi=150
)

plt.show()

print("\nGrafiki saglabāti:")
print("  kompozito_metozu_salidzinajums.png")
print("  gausa_kvadratūras_konvergence.png")