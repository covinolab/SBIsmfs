import pytest
import numpy as np
from sbi_smfs.simulator.brownian_integrator import brownian_integrator


@np.vectorize
def G0(x):
    if np.abs(x) > 0.5:
        return 2 * (np.abs(x) - 1) ** 2 - 1
    else:
        return -2 * x**2


def G(x, q, dG=4, k=2, delta_x=1):
    return dG * G0(x / delta_x) + 0.5 * k * (x - q) ** 2


def test_integrator_error():
    deltaG = 6
    k = 3
    delta_x = 1.5
    x_knots = np.linspace(-6, 6, 150)
    y_knots = deltaG * G0(x_knots / delta_x)

    q = brownian_integrator(
        x0=10,
        q0=10,
        Dx=1,
        Dq=1,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=10,
        dt=5e-4,
        fs=1,
    )
    assert q is None


def test_integrator_saving():
    deltaG = 6
    k = 3
    delta_x = 1.5
    x_knots = np.linspace(-6, 6, 150)
    y_knots = deltaG * G0(x_knots / delta_x)

    num_steps = 1e4
    saving_freq = 10

    q = brownian_integrator(
        x0=1,
        q0=1,
        Dx=1,
        Dq=1,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=num_steps,
        dt=5e-4,
        fs=saving_freq,
    )
    assert len(q) == num_steps // saving_freq


@pytest.mark.parametrize("deltaG, k, delta_x", [(6, 3, 1.5), (4, 2, 1.0)])
def test_integrator_pmf(deltaG: float, k: float, delta_x: float):
    x_knots = np.linspace(-6, 6, 150)
    y_knots = deltaG * G0(x_knots / delta_x)

    q = brownian_integrator(
        x0=1,
        q0=1,
        Dx=1,
        Dq=1,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=int(5e8),
        dt=5e-3,
        fs=10,
    )

    bins = np.linspace(-3.0, 3.0, 200)
    counts, bins = np.histogram(q, bins=bins, density=True)
    pmf_sim = -np.log(counts)
    pmf_sim = pmf_sim - min(pmf_sim)
    bin_center = bins[1:] - (bins[1:] - bins[:-1]) / 2

    x_values = np.linspace(-3, 3, len(bin_center))
    y_values = bin_center
    x, y = np.meshgrid(x_values, y_values)
    pot = G(x, y, dG=deltaG, k=k, delta_x=delta_x)

    L = len(pot[0, :])
    y_proj = np.zeros(L)
    for i in range(L):
        y_proj[i] = -np.log(np.trapz(np.exp(-pot[i, :])))
    y_proj = y_proj - min(y_proj)
    average_error = np.mean(y_proj - pmf_sim)
    assert np.isclose(average_error, 0, atol=0.1)


def test_integrator_rng_generator():
    deltaG = 6
    k = 3
    delta_x = 1.5
    x_knots = np.linspace(-6, 6, 150)
    y_knots = deltaG * G0(x_knots / delta_x)

    num_steps = 1e2
    saving_freq = 1

    q1 = brownian_integrator(
        x0=1,
        q0=1,
        Dx=1,
        Dq=1,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=num_steps,
        dt=5e-4,
        fs=saving_freq,
    )

    q2 = brownian_integrator(
        x0=1,
        q0=1,
        Dx=1,
        Dq=1,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=num_steps,
        dt=5e-4,
        fs=saving_freq,
    )

    assert not (q1 == q2).all()
