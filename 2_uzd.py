import numpy as np
from scipy import integrate

l = 1
g = 9.81

def omega(a, a0):
    return 1/(np.sqrt(2*g/l) * np.sqrt(np.cos(a) - np.cos(a0)))


def f4_1_15(a, a0):
    w = omega(a, a0)
    da = a[1, :] - a[0, :]
    return 4 * da * (3/2 * w[1] + np.sum(w[2:-2], axis=0) + 3/2 * w[-2])

def f4_1_16(a, a0):
    w = omega(a, a0)
    da = a[1, :] - a[0, :]
    return 4 * da * (23/12 * w[1] + 7/12 * w[2] + np.sum(w[3:-3], axis=0) + 7/12 * w[-3] + 23/12 * w[-2])

dim = 2*np.pi*np.sqrt(l/g)

a0 = np.linspace(np.pi/20, np.pi/2, 10)

n = 500000
a = np.linspace(0, a0, n)

print(a0)
print(np.round(f4_1_15(a, a0)/dim, 5))
print(np.round(f4_1_16(a, a0)/dim, 5))

ome = lambda a, a0: 1/(np.sqrt(2*g/l) * np.sqrt(np.cos(a) - np.cos(a0)))

quad_results = np.array([
    4 * integrate.quad(ome, 0, x, args=(x,))[0]
    for x in a0
])

print(np.round(quad_results / dim, 5))
