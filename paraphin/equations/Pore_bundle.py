"""Пучок капилляров: интегралы по сетке радиусов и неявный перенос функции пор fi по оси радиусов.

Общие кирпичи ядер кольматации (`Qp_m_k_fi.calc_qp_m_k_fi`, `Deposition`). Интегралы точные по отрезкам
кусочно-линейных fi и профилей (как `Qp_m_k_fi._calculate_integrals`); профили u, b - строки скретча `rows[i]`
(своя на каждый i: гонка в prange).
"""
from numba import njit

from paraphin.geometry import r1, w1_cv, w2_cv, w4_cv, dr_cv
from paraphin.constants import Nr

_INV_DR_CV = 1.0 / dr_cv  # прогонка `update_fi_rows`: умножение вместо деления
# Веса билинейной формы int r*u*fi по отрезку [a, b], r = a + s*dr, phi0 = 1 - s, phi1 = s (`int_r_u_fi`,
# `Qp_m_k_fi._calculate_integrals`)
_A, _DR = r1[:-1], r1[1:] - r1[:-1]
Q00 = _DR * (_A / 3 + _DR / 12)  # int r*phi0*phi0 dr
Q01 = _DR * (_A / 6 + _DR / 12)  # int r*phi0*phi1 dr
Q11 = _DR * (_A / 3 + _DR / 4)   # int r*phi1*phi1 dr


@njit(cache=True)
def moments(fi, i, j):
    """Интегралы r*fi, r^2*fi, r^4*fi по кусочно-линейной fi - скалярные произведения с узловыми весами (точно по
    отрезкам, без делений: `geometry.w1_cv`, `w2_cv`, `w4_cv`)."""
    rfi, r2fi, r4fi = 0.0, 0.0, 0.0
    for ij in range(Nr):
        f = fi[i, j, ij]
        rfi += w1_cv[ij] * f
        r2fi += w2_cv[ij] * f
        r4fi += w4_cv[ij] * f
    return rfi, r2fi, r4fi


@njit(cache=True)
def int_r_u_fi(fi, i, j, u):
    """Интеграл r*u*fi по кусочно-линейным u и fi (u - строка скретча) - билинейная форма по соседним узлам с весами
    Q00, Q01, Q11 (точно по отрезкам, без делений)."""
    s = 0.0
    for ij in range(1, Nr):
        f0, f1, u0, u1 = fi[i, j, ij - 1], fi[i, j, ij], u[ij - 1], u[ij]
        s += f0 * (Q00[ij - 1] * u0 + Q01[ij - 1] * u1) + f1 * (Q01[ij - 1] * u0 + Q11[ij - 1] * u1)
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
    """Неявный перенос fi по оси радиусов со скоростью u(r) и неявным блокированием b(r)*fi.

    Поток через грань ij+1/2 расщеплен по знаку скорости: F = max(u_ij, 0)*fi_ij + min(u_ij+1, 0)*fi_ij+1. Это схема
    против потока в каждом узле, поэтому она годится и для сужения (u < 0), и для выноса (u > 0), и на стыке r_pass
    (слева u = 0, справа u < 0) не теряет поток из узла n_pass в n_pass-1: сузившиеся до r_pass капилляры попадают под
    блокирование, а не исчезают из баланса. Блокирование b*fi неявное - на диагонали: при явной записи большое b*dt
    уводит fi в минус. Матрица - M-матрица (диагональ 1/dt + |u|/dr + b, внедиагональные <= 0), поэтому прогонка
    устойчива и fi >= 0 без зажима. Поток делится на ширину контрольного объема узла `dr_cv`, а не на общий шаг:
    сохраняется sum(fi*dr_cv). На правой границе (r_max) соседа нет - условие fi = 0: капилляров шире r_max нет.
    """
    inv_dt = 1.0 / dt
    d = inv_dt + abs(u[0]) * _INV_DR_CV[0] + b[0]
    e = min(u[1], 0.0) * _INV_DR_CV[0]
    a_tdma[0] = -e / d
    b_tdma[0] = fi[i, j, 0] * inv_dt / d
    for ij in range(1, Nr):
        c = -max(u[ij - 1], 0.0) * _INV_DR_CV[ij]
        d = inv_dt + abs(u[ij]) * _INV_DR_CV[ij] + b[ij]
        e = min(u[ij + 1], 0.0) * _INV_DR_CV[ij] if ij + 1 < Nr else 0.0
        inv = 1.0 / (c * a_tdma[ij - 1] + d)
        a_tdma[ij] = -e * inv
        b_tdma[ij] = (fi[i, j, ij] * inv_dt - c * b_tdma[ij - 1]) * inv
    new_fi[i, j, Nr - 1] = b_tdma[Nr - 1]
    for ij in range(Nr - 2, -1, -1):
        new_fi[i, j, ij] = new_fi[i, j, ij + 1] * a_tdma[ij] + b_tdma[ij]
