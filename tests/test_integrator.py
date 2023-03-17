import os, sys
sys.path.insert(0, os.path.abspath("../scr"))

import numpy as np
import matplotlib.pyplot as plt

from brownian_integrator import brownian_integrator
from gls_spline import gls_spline


@np.vectorize
def G0(x):
    if np.abs(x) > 0.5:
        return 2 * (np.abs(x) - 1) ** 2 - 1
    else:
        return -2 * x ** 2


def G(x, q, dG=4, k=2, delta_x=1):
    return dG * G0(x / delta_x) + 0.5 * k * (x - q) ** 2


deltaG = 6
k = 3
delta_x = 1.5

x_knots = np.linspace(-6, 6, 150)
y_knots = deltaG * G0(x_knots/delta_x)

x_eval = np.linspace(-6, 6, 1000)/delta_x
y_gsl = gls_spline(x_knots, y_knots, x_eval)

plt.plot(x_eval, y_gsl, label='spline approximation')
plt.plot(x_eval, deltaG * G0(x_eval/delta_x), label='true potential')
plt.plot(x_knots, y_knots, 'ob')
plt.ylim(-6, 10)
plt.xlim(-3, 3)
plt.legend()
plt.show()

q = brownian_integrator(
        x0=1,
        q0=1,
        Dx=1,
        Dq=1,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=int(1e9),
        dt=5e-4,
        fs=10
)

N = 200
x_values = np.linspace(-3, 3, N)
y_values = np.linspace(-6.5, 6.5, N)
x, y = np.meshgrid(x_values, y_values)
pot = G(x, y, dG=deltaG, k=k, delta_x=delta_x)

L = len(pot[0, :])
y_proj = np.zeros(L)
for i in range(L):
    y_proj[i] = -np.log(np.trapz(np.exp(-pot[i, :])))
y_proj = y_proj - min(y_proj)

counts, bins = np.histogram(q, bins=100, density=True)
bolt_inv = -np.log(counts)
bolt_inv = bolt_inv - min(bolt_inv)
bin_middle = bins[1:] - (bins[1:] - bins[:-1])/2
plt.plot(bin_middle, bolt_inv, label='boltz inversion')
plt.plot(y_values, y_proj, label='true pmf')
plt.ylim(0, 6)
plt.xlim(-3, 3)
plt.legend()
plt.show()
