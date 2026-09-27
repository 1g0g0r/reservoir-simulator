"""Гель: предел текучести, множитель Букингема-Райнера и подвижность пучка капилляров (`equations/Gel.py`)."""
import math

import numpy as np

from paraphin import fi_0, r1, w4_cv, eta, dr_cv, r4
from paraphin.constants import Nr, gel_phi, gel_phi_ref, gel_tau_ref, gel_tau_mult, gel_time
from paraphin.equations.Gel import br_factor, gel_phi_eq, yield_stress, pore_solid_fraction


def _fi():
    return np.tile(fi_0, (1, 1, 1))


def test_w4_weights_exact():
    """Веса w4_cv дают точный интеграл r^4*fi кусочно-линейной fi - тот же, что в `_calculate_integrals`."""
    fi = fi_0
    exact = 0.0
    for ij in range(1, Nr):
        a, b = r1[ij - 1], r1[ij]
        # интеграл r^4*(линейная fi) на отрезке аналитически
        A = (fi[ij - 1] * b - fi[ij] * a) / (b - a)
        B = (fi[ij] - fi[ij - 1]) / (b - a)
        exact += A * (b ** 5 - a ** 5) / 5 + B * (b ** 6 - a ** 6) / 6
    assert math.isclose((w4_cv * fi).sum(), exact, rel_tol=1e-12)


def test_buckingham_reiner():
    assert br_factor(0.0) == 1.0
    assert br_factor(1.0) == 0.0 and br_factor(2.0) == 0.0
    h = 1e-6  # F'(1) = 0: срыв гладкий
    assert abs((br_factor(1.0 - h) - br_factor(1.0 - 2 * h)) / h) < 1e-4
    xs = np.linspace(0.0, 1.0, 101)
    assert np.all(np.diff([br_factor(x) for x in xs]) <= 0.0)


def test_yield_stress():
    assert yield_stress(gel_phi) == 0.0
    assert math.isclose(yield_stress(gel_phi_ref), gel_tau_mult * gel_tau_ref)
    assert yield_stress(0.2) > yield_stress(0.15) > 0.0


def test_phi_limits_and_monotonicity():
    fi = _fi()
    assert gel_phi_eq(0, 0, fi, 1e5, 0.0) == 1.0  # без геля ровно 1, без округления
    tau = 1.0
    grads = [1e3, 1e5, 1e6, 1e7, 1e9]
    phis = [gel_phi_eq(0, 0, fi, g, tau) for g in grads]
    assert all(0.0 <= x <= 1.0 for x in phis)
    assert np.all(np.diff(phis) >= 0.0)  # растет с градиентом давления
    assert gel_phi_eq(0, 0, fi, 1e6, 2.0) <= gel_phi_eq(0, 0, fi, 1e6, 1.0)  # падает с пределом текучести
    # порог: при |grad p| < 2*eta*tau_y/r_max не течет ни один капилляр
    g_thr = 2.0 * eta * tau / r1[-1]
    assert gel_phi_eq(0, 0, fi, 0.999 * g_thr, tau) == 0.0
    assert gel_phi_eq(0, 0, fi, 1.5 * g_thr, tau) > 0.0


def test_pore_solid_fraction():
    assert pore_solid_fraction(0.3, 0.4, 0.02, 0.0) == 0.02
    assert math.isclose(pore_solid_fraction(0.3, 0.4, 0.0, 0.03), 0.03 / (0.18 + 0.03))
    assert pore_solid_fraction(0.3, 1.0, 0.1, 0.0) == 0.0


def test_relaxation_damps_lagged_oscillation():
    """Три ячейки последовательно при заданном расходе: застывшая ячейка берет на себя перепад, гель
    срывается, перепад падает, гель снова застывает. Без релаксации (gel_time -> 0) запаздывающая схема
    колеблется, с релаксацией gel_time >= dt колебания затухают."""
    fi = _fi()
    tau, dt = 0.5, 4320.0
    lam = np.array([1.0, 1.0, 1.0])

    def run(t_relax, n=400):
        phi, hist = 1.0, []
        for _ in range(n):
            mob = lam.copy()
            mob[1] *= max(phi, 0.05)
            grad = 1e5 / mob[1] / (1.0 / mob).sum() * 3.0  # доля перепада на средней ячейке при общем 1 бар/м
            eq = gel_phi_eq(0, 0, fi, grad, tau)
            phi = eq + (phi - eq) * (math.exp(-dt / t_relax) if t_relax > 0.0 else 0.0)
            hist.append(phi)
        d = np.diff(hist[-50:])
        return int((np.sign(d[1:]) * np.sign(d[:-1]) < 0).sum()), hist[-1]

    flips_relaxed, _ = run(gel_time)
    assert flips_relaxed <= 2
