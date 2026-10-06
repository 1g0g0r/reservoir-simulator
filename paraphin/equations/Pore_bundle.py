"""Пучок капилляров: интегралы по сетке радиусов и неявный перенос функции пор fi по оси радиусов.

Общие кирпичи ядер детального состава (`Qp_m_k_fi.calc_qp_m_k_fi_2`, `Deposition`). Прежнее ядро
(`Qp_m_k_fi.calc_qp_m_k_fi`) пользуется своими `_calculate_integrals`, `_blocking`, `_update_fi`: их порядок
сложения держит побитовую регрессию. Интегралы точные по отрезкам кусочно-линейных fi и профилей (как
`_calculate_integrals`); профили u, b - строки скретча `rows[i]` (своя на каждый i: гонка в prange).
"""
from numba import njit

from paraphin.geometry import r1, r2, r3, r4, r5, r6, w2_cv, dr_cv
from paraphin.constants import Nr


@njit(cache=True)
def moments(fi, i, j):
    """Интегралы r*fi, r^2*fi, r^4*fi по кусочно-линейной fi (точно по отрезкам, как `_calculate_integrals`)."""
    rfi, r2fi, r4fi = 0.0, 0.0, 0.0
    for ij in range(1, Nr):
        dr = r1[ij] - r1[ij - 1]
        a = (fi[i, j, ij - 1] * r1[ij] - fi[i, j, ij] * r1[ij - 1]) / dr
        b = (fi[i, j, ij] - fi[i, j, ij - 1]) / dr
        rfi += (r2[ij] - r2[ij - 1]) * a / 2 + (r3[ij] - r3[ij - 1]) * b / 3
        r2fi += (r3[ij] - r3[ij - 1]) * a / 3 + (r4[ij] - r4[ij - 1]) * b / 4
        r4fi += (r5[ij] - r5[ij - 1]) * a / 5 + (r6[ij] - r6[ij - 1]) * b / 6
    return rfi, r2fi, r4fi


@njit(cache=True)
def int_r_u_fi(fi, i, j, u):
    """Интеграл r*u*fi по кусочно-линейным u и fi (u - строка скретча)."""
    s = 0.0
    for ij in range(1, Nr):
        dr = r1[ij] - r1[ij - 1]
        af = (fi[i, j, ij - 1] * r1[ij] - fi[i, j, ij] * r1[ij - 1]) / dr
        bf = (fi[i, j, ij] - fi[i, j, ij - 1]) / dr
        au = (u[ij - 1] * r1[ij] - u[ij] * r1[ij - 1]) / dr
        bu = (u[ij] - u[ij - 1]) / dr
        s += ((r2[ij] - r2[ij - 1]) * af * au / 2 + (r3[ij] - r3[ij - 1]) * (af * bu + bf * au) / 3
              + (r4[ij] - r4[ij - 1]) * bf * bu / 4)
    return s


@njit(cache=True)
def weighted(fi, i, j, b, weights, scale):
    """sum(weights*scale*b*fi) - объем каналов или пробок, блокируемых за единицу времени (единицы int r^2*fi)."""
    s = 0.0
    for ij in range(Nr):
        s += min(weights[ij] * scale, w2_cv[ij]) * b[ij] * fi[i, j, ij]
    return s


@njit(cache=True)
def update_fi_rows(new_fi, fi, i, j, u, b, a_tdma, b_tdma, dt):
    """Неявный перенос fi по оси радиусов со скоростью u(r) и неявным блокированием b(r)*fi (как `_update_fi`).
    Схема против потока по знаку u в каждом узле, поэтому годится и для сужения (u < 0), и для выноса (u > 0)."""
    d = 1.0 / dt + abs(u[0]) / dr_cv[0] + b[0]
    e = min(u[1], 0.0) / dr_cv[0]
    a_tdma[0] = -e / d
    b_tdma[0] = fi[i, j, 0] / dt / d
    for ij in range(1, Nr):
        c = -max(u[ij - 1], 0.0) / dr_cv[ij]
        d = 1.0 / dt + abs(u[ij]) / dr_cv[ij] + b[ij]
        e = min(u[ij + 1], 0.0) / dr_cv[ij] if ij + 1 < Nr else 0.0
        den = c * a_tdma[ij - 1] + d
        a_tdma[ij] = -e / den
        b_tdma[ij] = (fi[i, j, ij] / dt - c * b_tdma[ij - 1]) / den
    new_fi[i, j, Nr - 1] = b_tdma[Nr - 1]
    for ij in range(Nr - 2, -1, -1):
        new_fi[i, j, ij] = new_fi[i, j, ij + 1] * a_tdma[ij] + b_tdma[ij]
