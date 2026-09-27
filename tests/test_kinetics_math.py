"""Формулы кинетики осаждения (`paraphin/equations/Kinetics_math.py`): аналитические решения и предельные случаи."""
import math

import numpy as np
import pytest

from paraphin.constants import k_B, g, R
from paraphin.equations.Kinetics_math import (relax_exp, wall_fraction, brownian_diffusivity, shear_diffusivity,
                                              stokes_velocity, leveque_velocity, coagulation_kernel,
                                              smoluchowski_step, floc_size, langmuir_constant, langmuir_eq,
                                              perm_kozeny_carman, perm_power, perm_damage, perm_surface,
                                              hawkins_skin)


def test_relaxation_is_exact():
    """Точная экспонента: два полушага дают один шаг, пределы k -> 0 и k -> inf."""
    x, eq, k, dt = 0.03, 0.10, 2.5e-3, 400.0
    half = relax_exp(relax_exp(x, eq, k, dt / 2), eq, k, dt / 2)
    assert np.isclose(half, relax_exp(x, eq, k, dt), rtol=1e-14)
    assert relax_exp(x, eq, 0.0, dt) == x
    assert np.isclose(relax_exp(x, eq, 1e6, dt), eq, rtol=1e-14)


def test_wall_fraction_splits_supersaturation():
    """Стенкам и объему вместе достается 1 - exp(-(k_w + k_b)*dt), в отношении k_w : k_b."""
    kw, kb, dt = 0.02, 0.06, 30.0
    total = 1.0 - math.exp(-(kw + kb) * dt)
    assert np.isclose(wall_fraction(kw, kb, dt) + wall_fraction(kb, kw, dt), total, rtol=1e-14)
    assert np.isclose(wall_fraction(kw, kb, dt) / wall_fraction(kb, kw, dt), kw / kb, rtol=1e-12)
    assert wall_fraction(0.0, 0.0, dt) == 0.0
    assert np.isclose(wall_fraction(kw, 0.0, dt), 1.0 - math.exp(-kw * dt), rtol=1e-14)


def test_smoluchowski_constant_kernel():
    """dN/dt = -K*N^2/2 с постоянным ядром: N(t) = N0/(1 + K*N0*t/2), шаги складываются точно."""
    T_K, mu = 343.15, 5e-3
    k = coagulation_kernel(T_K, mu, 1.0)
    assert np.isclose(k, 8 * k_B * T_K / (3 * mu), rtol=1e-14)
    n0, t = 1e18, 3600.0
    n = n0
    for _ in range(10):
        n = smoluchowski_step(n, k, t / 10)
    assert np.isclose(n, n0 / (1 + k * n0 * t / 2), rtol=1e-12)
    # Реакционно-лимитированная агрегация (W > 1) медленнее диффузионной
    assert smoluchowski_step(n0, coagulation_kernel(T_K, mu, 100.0), t) > smoluchowski_step(n0, k, t)


def test_floc_size_fractal():
    rho, d0 = 1150.0, 1e-7
    m0 = rho * math.pi * d0 ** 3 / 6
    assert np.isclose(floc_size(m0, d0, 2.0, rho), d0)
    assert np.isclose(floc_size(8 * m0, d0, 3.0, rho), 2 * d0, rtol=1e-12)  # компактная частица
    assert np.isclose(floc_size(4 * m0, d0, 2.0, rho), 2 * d0, rtol=1e-12)  # рыхлый агрегат растет быстрее
    assert floc_size(0.1 * m0, d0, 2.0, rho) == d0                          # не мельче первичной


def test_particle_transport_reference_values():
    """Стокс-Эйнштейн и Стокс - по определению; перенос к стенке растет с коэффициентом диффузии как D^(2/3)."""
    T_K, mu, d = 293.15, 3e-3, 15e-6
    assert np.isclose(brownian_diffusivity(T_K, mu, d), k_B * T_K / (3 * math.pi * mu * d), rtol=1e-14)
    a, drho = 7.5e-6, 262.0
    assert np.isclose(stokes_velocity(a, drho, mu), 2 * a * a * drho * g / (9 * mu), rtol=1e-14)
    assert stokes_velocity(a, -drho, mu) == stokes_velocity(a, drho, mu)
    um, r, dd = 1e-4, 12e-6, 2e-15
    assert np.isclose(leveque_velocity(um, r, 8 * dd) / leveque_velocity(um, r, dd), 4.0, rtol=1e-12)
    assert shear_diffusivity(0.5, 0.0, 10.0, a) == 0.0
    assert np.isclose(shear_diffusivity(0.5, 0.1, 10.0, a), 0.5 * 0.01 * 10.0 * a * a, rtol=1e-14)


def test_langmuir():
    g_max, k = 0.12, 200.0
    assert langmuir_eq(g_max, k, 0.0) == 0.0
    assert np.isclose(langmuir_eq(g_max, k, 1e6), g_max, rtol=1e-6)
    assert np.isclose(langmuir_eq(g_max, k, 1.0 / k), g_max / 2, rtol=1e-14)  # половина при K*c = 1
    t_ref = 343.15
    assert np.isclose(langmuir_constant(k, -2e4, t_ref, t_ref, R), k, rtol=1e-14)
    # Экзотермическая адсорбция (dH < 0) при охлаждении растет
    assert langmuir_constant(k, -2e4, 318.15, t_ref, R) > k > langmuir_constant(k, -2e4, 363.15, t_ref, R)


@pytest.mark.parametrize('m', [0.30, 0.27, 0.2])
def test_permeability_correlations(m):
    m0 = 0.3
    kc = perm_kozeny_carman(m, m0)
    assert (kc == 1.0) if m == m0 else (0.0 < kc < (m / m0) ** 3)
    assert np.isclose(perm_power(m, m0, 10.0), (m / m0) ** 10, rtol=1e-14)
    assert np.isclose(perm_surface(m, m0, 0.0, 10.0), kc, rtol=1e-14)
    assert perm_surface(m, m0, 0.05, 10.0) < kc
    assert perm_damage(0.0, 1.0, 2.0) == 1.0
    assert perm_damage(1.2, 1.0, 2.0) == 0.0


def test_hawkins_skin():
    assert hawkins_skin(1.0, 1.0, 5.0, 0.1) == 0.0
    assert np.isclose(hawkins_skin(1.0, 0.25, 5.0, 0.1), 3 * math.log(50), rtol=1e-14)
    assert hawkins_skin(1.0, 2.0, 5.0, 0.1) < 0.0  # стимулированная зона
