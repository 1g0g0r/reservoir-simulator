"""Множитель средней скорости нефти в капилляре"""
import numpy as np
from numba import njit

from paraphin.constants import Nx, Ny, hx, hy, eta


@njit(cache=True)
def calc_Um_r2(i, j, p, grad_p, Um_r2, mu_o):
    """Средняя скорость нефти в капилляре без множителя r^2 и модуль градиента давления.
    Градиент - односторонними разностями на границах, центральными во внутренних точках.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    # Односторонние разности на границах и центральные разности для внутренних точек
    if i == 0:
        df_dx = (p[i + 1, j] - p[i, j]) / hx
    elif i == Nx - 1:
        df_dx = (p[i, j] - p[i - 1, j]) / hx
    else:
        df_dx = (p[i + 1, j] - p[i - 1, j]) / (2.0 * hx)

    if j == 0:
        df_dy = (p[i, j + 1] - p[i, j]) / hy
    elif j == Ny - 1:
        df_dy = (p[i, j] - p[i, j - 1]) / hy
    else:
        df_dy = (p[i, j + 1] - p[i, j - 1]) / (2.0 * hy)

    grad_p[i, j] = np.sqrt(df_dx * df_dx + df_dy * df_dy)
    Um_r2[i, j] = grad_p[i, j] / mu_o[i, j] * 0.125 / eta
