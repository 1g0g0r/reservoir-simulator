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


def test_wettability_mix_limits():
    """omega = 0 - водосмачиваемые ОФП базовой модели, omega = 1 - нефтесмачиваемый набор."""
    from paraphin.constants import S_min, S_max
    from paraphin.utils.math_utils import pf_o, pf_w, pf_o_mix, pf_w_mix
    for s in np.linspace(0.0, 1.0, 41):
        assert pf_o_mix(s, 0.0, 0.1, 0.8, 3.0) == pf_o(s)
        assert pf_w_mix(s, 0.0, 0.1, 0.8, 1.5) == pf_w(s)
    s = 0.5
    assert np.isclose(pf_o_mix(s, 1.0, 0.1, 0.8, 3.0), ((0.8 - s) / 0.7) ** 3, rtol=1e-14)
    assert np.isclose(pf_w_mix(s, 1.0, 0.1, 0.8, 1.5), ((s - 0.1) / 0.7) ** 1.5, rtol=1e-14)


def test_ltne_exchange_conserves_energy_and_relaxes():
    """Теплообмен флюид - порода за шаг: энергия C_f*T + C_s*T_s сохраняется, разность убывает по экспоненте."""
    from paraphin.equations.Thermal_ltne import exchange_step
    c_f, c_s, h, dt = 1.2e6, 2.0e6, 5e3, 100.0
    t_f, t_s = exchange_step(20.0, 70.0, c_f, c_s, h, dt)
    assert np.isclose(c_f * t_f + c_s * t_s, c_f * 20.0 + c_s * 70.0, rtol=1e-14)
    assert np.isclose(t_f - t_s, -50.0 * math.exp(-h * (1 / c_f + 1 / c_s) * dt), rtol=1e-12)
    t_f, t_s = exchange_step(20.0, 70.0, c_f, c_s, 1e12, dt)  # мгновенный обмен - равновесие
    assert np.isclose(t_f, t_s, atol=1e-10)


def test_ltne_schumann_breakthrough():
    """Шуман (J Franklin Inst 1929, 208:405): холодный флюид входит в горячий слой без теплопроводности. Перенос
    против потока с числом Куранта 1 - точный сдвиг, поэтому ошибка только от расщепления с точным обменом за шаг
    (первый порядок). Температура на выходе сверяется с решением Шумана (Anzelius):
        T_f = 1 - exp(-z)*int_0^y exp(-s)*I0(2*sqrt(s*z)) ds,  y = h*L/(c_f*u),  z = h*(t - L/u)/c_s."""
    from scipy.special import i0
    from scipy.integrate import quad
    from paraphin.equations.Thermal_ltne import exchange_step
    L, u, c_f, c_s, h = 1.0, 1e-3, 1.0e6, 2.0e6, 5000.0  # y = 5: развитый тепловой фронт
    n = 400
    dt = L / n / u
    y = h * L / (c_f * u)
    tf, ts = np.zeros(n), np.zeros(n)
    checks = {2000.0: None, 3000.0: None, 4000.0: None}
    for step in range(1, int(4000.0 / dt) + 1):
        tf[1:] = tf[:-1].copy()
        tf[0] = 1.0
        for k in range(n):
            tf[k], ts[k] = exchange_step(tf[k], ts[k], c_f, c_s, h, dt)
        t = step * dt
        if t in checks:
            checks[t] = tf[-1]
    for t, num in checks.items():
        z = h * (t - L / u) / c_s
        val, _ = quad(lambda s: math.exp(-s) * i0(2.0 * math.sqrt(s * z)), 0.0, y, limit=200)
        exact = 1.0 - math.exp(-z) * val
        assert 0.2 < exact < 0.9 and abs(num - exact) < 1e-3, (t, num, exact)


def test_langmuir_film_step():
    """Пленочная кинетика: корень в [0, G_max), сходимость к изотерме при постоянной c, линейный предел совпадает
    с кинетикой твердой фазы (k = A/(G_max*K)), при выпуклой изотерме скорость постоянна почти до насыщения."""
    from paraphin.equations.Kinetics_math import langmuir_film_step
    g_max, c = 2.0, 0.01
    for k_l in (1.0, 100.0, 1e4):
        g = 0.0
        for _ in range(4000):
            g = langmuir_film_step(g, g_max, k_l, c, 0.5)
            assert 0.0 <= g < g_max
        assert np.isclose(g, langmuir_eq(g_max, k_l, c), rtol=1e-8)
    # Линейный предел: неявный Эйлер твердой фазы с k = A/(G_max*K)
    k_l, a_dt, g0 = 1e-3, 0.3, 1e-7
    film = langmuir_film_step(g0, g_max, k_l, c, a_dt)
    k_dt = a_dt / (g_max * k_l)  # (k*dt) твердой фазы
    solid = (g0 + k_dt * langmuir_eq(g_max, k_l, c)) / (1.0 + k_dt)
    assert np.isclose(film, solid, rtol=1e-3)
    # Выпуклая изотерма (K*c = 100): до 90 % насыщения скорость не ниже 90 % начальной
    k_l, a_dt = 1e4, 1e-3
    g, steps = 0.0, []
    while g < 0.9 * langmuir_eq(g_max, k_l, c):
        g_new = langmuir_film_step(g, g_max, k_l, c, a_dt)
        steps.append(g_new - g)
        g = g_new
    assert min(steps) > 0.9 * steps[0]


def test_langmuir_ldf_step():
    """Кинетика твердой фазы, неявная по концентрации: при малой доле пути f - явная запись, при f = 1 -
    равновесие закрытой ячейки (масса сохраняется, G = G_eq(c_new)), c_new не отрицательна и не колеблется."""
    from paraphin.equations.Kinetics_math import langmuir_ldf_step
    g_max, k_l = 4.8, 4.5e4
    for c, a, g in ((0.009, 160.0, 3.2), (0.009, 5.0, 3.2), (0.0, 5.0, 4.0), (0.02, 1e-3, 0.5)):
        # малый шаг: f*dG_eq/dc/a << 1 (у c = 0 изотерма крутая, dG_eq/dc = G_max*K)
        f = 1e-4 * a / (g_max * k_l)
        explicit = f * (langmuir_eq(g_max, k_l, c) - g)
        assert np.isclose(langmuir_ldf_step(g, g_max, k_l, c, a, f) - g, explicit, rtol=1e-3)
        g_new = langmuir_ldf_step(g, g_max, k_l, c, a, 1.0)
        c_new = c - (g_new - g) / a
        assert c_new >= 0.0
        assert np.isclose(g_new, langmuir_eq(g_max, k_l, c_new), rtol=1e-9)
    # Закрытая ячейка с малым запасом нефти: монотонный подход к равновесию без колебаний
    g, c, a = 3.2, 0.009, 5.0
    total = g + a * c
    steps = []
    for _ in range(50):
        g_new = langmuir_ldf_step(g, g_max, k_l, c, a, 0.9)
        c -= (g_new - g) / a
        steps.append(g_new - g)
        g = g_new
        assert c >= 0.0 and np.isclose(g + a * c, total, rtol=1e-12)
    assert all(s >= -1e-15 for s in steps)
    assert langmuir_ldf_step(g, g_max, k_l, c, 0.0, 0.5) == g  # нет нефти - нет обмена


def test_ema_network():
    """Эффективная среда Киркпатрика: одинаковые горла - та же проводимость; доля p открытых одинаковых горл -
    g*(p - 2/z)/(1 - 2/z) и ноль ниже порога протекания 2/z; при z -> inf - среднее арифметическое (пучок)."""
    from paraphin.equations.Kinetics_math import ema_conductance
    g = np.full(5, 2.0)
    w = np.full(5, 0.2)
    assert np.isclose(ema_conductance(g, w, 0.0, 4.0), 2.0, rtol=1e-10)
    for z, p in ((4.0, 0.8), (6.0, 0.5), (3.0, 0.9)):
        gm = ema_conductance(g, w * p, 1.0 - p, z)
        assert np.isclose(gm, 2.0 * (p - 2.0 / z) / (1.0 - 2.0 / z), rtol=1e-9)
    assert ema_conductance(g, w * 0.45, 0.55, 4.0) == 0.0  # p = 0.45 < 2/z = 0.5
    g2 = np.array([1.0, 3.0, 8.0])
    w2 = np.array([0.5, 0.3, 0.2])
    assert np.isclose(ema_conductance(g2, w2, 0.0, 1e9), np.sum(g2 * w2), rtol=1e-6)
    # Разброс проводимостей при конечном z снижает эффективную ниже среднего (сеть чувствительнее пучка)
    assert ema_conductance(g2, w2, 0.0, 4.0) < np.sum(g2 * w2)
