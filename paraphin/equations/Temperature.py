"""Решение уравнения температуры по явной схеме."""
import numpy as np
from numba import njit

from paraphin.constants import volume, h, ro_w, ro_f, ro_ff, ro_o, ro_p, init_T, K_ff
from paraphin.utils.math_utils import erfc


@njit(cache=True)
def temperature_equation(i, j, T, m, S, C_o, C_w, C_f, C_ff, C_p, Wps, qp, cells_T_eq, t,
                         lam_o, lam_w, grad_p, new_T, new_m, new_S, dt) -> None:
    """Вычисление температуры по явной схеме.

    Помимо перетоков `cells_T_eq` учитывает изменение теплоемкости смеси за шаг, теплоту
    кристаллизации парафина (`qp`) и потери через кровлю и подошву пласта (метод Ловерье).

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    psi = _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f)
    psi_next = _psi(i, j, new_m, new_S, Wps, C_w, C_o, C_p, C_f)
    derivative_add = T[i, j] * volume * (psi_next - psi) / dt
    T_losses = _top_bottom_heat_losses(i, j, t, T, lam_o, lam_w, C_o, C_w, C_f, C_ff, grad_p)

    new_T[i, j] += T[i, j] + dt / psi_next / volume * (cells_T_eq[i, j] - derivative_add - T_losses * volume + qp[i, j] * ro_p * C_p[i, j] * volume)


@njit(cache=True)
def temperature_well(well, T, m, S, C_o, C_w, C_f, C_p, Wps, new_T, dt) -> None:
    """Учет скважины в уравнении энергии.

    У нагнетательной берется температура закачиваемой воды, у добывающей - температура ячейки.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    i, j = well.i, well.j

    # У нагнетательной берется температура закачки, у добывающей - температура ячейки
    Twell = well.T if well.is_injector == 1 else T[i, j]

    multiplier = dt / _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f) / volume

    new_T[i, j] -= (C_o[i, j] * ro_o * well.q[0] + C_w[i, j] * ro_w * well.q[1]) * multiplier * Twell


@njit(cache=True)
def _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f):
    # TODO уточнить энергию осевшего на порах парафина
    return (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) +
                                     ro_p * C_p[i, j] * Wps[i, j])) + (1.0 - m[i, j]) * ro_f * C_f[i, j])


@njit(cache=True)
def _top_bottom_heat_losses(i, j, t, T, lam_o, lam_w, C_o, C_w, C_f, C_ff, grad_p):
    """Потери тепла через кровлю и подошву пласта по методу Ловерье, [Вт/м^3].

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    t_loss = 0.0
    # Подвижности уже посчитаны calc_mobility: mobility_o(k, S, mu_o) - это ровно lam_o[i, j]
    V_o = grad_p[i, j] * lam_o[i, j]
    V_w = grad_p[i, j] * lam_w[i, j]

    teta = 4.0 * K_ff * t / (C_f[i, j] * ro_f) / h / h
    ksi = 4.0 * K_ff / (V_o * C_o[i, j] * ro_o + V_w * C_w[i, j] * ro_w) / h

    if teta > ksi:
        erfc_argument = ksi / np.sqrt((C_f[i, j] * ro_f) / (C_ff[i, j] * ro_ff) * (teta - ksi)) * 0.5
        t_loss = (T[i, j] - init_T) * erfc(erfc_argument)

    return t_loss
