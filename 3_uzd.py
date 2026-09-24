import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

l = 1
g = 9.81

def omega(a, a0):
    return 1/(np.sqrt(2*g/l) * np.sqrt(np.cos(a) - np.cos(a0)))


def f4_1_15(a, a0):
    w = omega(a, a0)
    da = a[1, :] - a[0, :]
    return 4 * da * (
        3/2 * w[1]
        + np.sum(w[2:-2], axis=0)
        + 3/2 * w[-2]
    )


def f4_1_16(a, a0):
    w = omega(a, a0)
    da = a[1, :] - a[0, :]
    return 4 * da * (
        23/12 * w[1]
        + 7/12 * w[2]
        + np.sum(w[3:-3], axis=0)
        + 7/12 * w[-3]
        + 23/12 * w[-2]
    )


ome = lambda a, a0: 1 / (
    np.sqrt(2*g/l) * np.sqrt(np.cos(a) - np.cos(a0))
)

dim = 2*np.pi*np.sqrt(l/g)

a0 = np.linspace(np.pi/200, np.pi/2, 20)

n = 10000000
a = np.linspace(0, 1, n)[:, None] * a0[None, :]

f15 = f4_1_15(a, a0) / dim
f16 = f4_1_16(a, a0) / dim


# integrate.quad()
quad_results = np.array([
    4 * integrate.quad(ome, 0, x, args=(x,))[0]
    for x in a0
])

quad_results = quad_results / dim

print("quad:")
print(np.round(quad_results, 5))


plt.style.use("seaborn-v0_8-whitegrid")

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(8, 10),
    dpi=110,
    sharex=True
)


# f4.1.15
ax1.axhline(
    1,
    color="#c0392b",
    linestyle="--",
    linewidth=1.5,
    label=r"$\tilde{T}$ = 1 ($\alpha_0 \to 0$)"
)

ax1.plot(
    a0, f15,
    marker="o",
    markersize=5,
    linewidth=1.8,
    color="#2e86ab",
    label=r"$f_{4.1.15}$"
)

ax1.set_ylabel(r"$\tilde{T}$", fontsize=12)
ax1.legend(frameon=True, fontsize=10, loc="upper left")
ax1.tick_params(labelsize=10)


# f4.1.16
ax2.axhline(
    1,
    color="#c0392b",
    linestyle="--",
    linewidth=1.5,
    label=r"$\tilde{T}$ = 1 ($\alpha_0 \to 0$)"
)

ax2.plot(
    a0, f16,
    marker="o",
    markersize=5,
    linewidth=1.8,
    color="#2e9e5b",
    label=r"$f_{4.1.16}$"
)

ax2.set_ylabel(r"$\tilde{T}$", fontsize=12)
ax2.legend(frameon=True, fontsize=10, loc="upper left")
ax2.tick_params(labelsize=10)

# Gauss-Konrod
ax3.axhline(
    1,
    color="#c0392b",
    linestyle="--",
    linewidth=1.5,
    label=r"$\tilde{T}$ = 1 ($\alpha_0 \to 0$)"
)

ax3.plot(
    a0, quad_results,
    marker="o",
    markersize=5,
    linewidth=1.8,
    color="#8e44ad",
    label=r"$Gausa-Kronroda$"
)

ax3.set_xlabel(r"$\alpha_0$ (rad)", fontsize=12)
ax3.set_ylabel(r"$\tilde{T}$", fontsize=12)
ax3.legend(frameon=True, fontsize=10, loc="upper left")
ax3.tick_params(labelsize=10)


fig.tight_layout()

print(a0)
print(np.round(f15, 4))
print(np.round(f16, 4))
print(np.round(quad_results, 4))
print(abs(f15 - f16))

plt.savefig("3_uzd.png")
plt.show()