"""Множитель средней скорости нефти в капилляре"""
import numpy as np
from numba import njit

from paraphin import eta
from paraphin.constants import Nx, Ny, hx, hy


@njit(cache=True)
def calc_Um_r2(i, j, p, grad_p, Um_r2, mu_o):
    """Средняя скорость нефти в капилляре без множителя r^2 и модуль градиента давления.
    Градиент - односторонними разностями на границах, центральными во внутренних точках.
    Коэффициент извилистости `eta` согласует пучок капилляров с k_0, см. `paraphin/__init__.py`.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    # Односторонние разности на границах и центральные разности для внутренних точек.
    # В один ряд ячеек (одномерный керн) соседа нет: без этой ветки индекс уходил за массив.
    if Nx == 1:
        df_dx = 0.0
    elif i == 0:
        df_dx = (p[i + 1, j] - p[i, j]) / hx
    elif i == Nx - 1:
        df_dx = (p[i, j] - p[i - 1, j]) / hx
    else:
        df_dx = (p[i + 1, j] - p[i - 1, j]) / (2.0 * hx)

    if Ny == 1:
        df_dy = 0.0
    elif j == 0:
        df_dy = (p[i, j + 1] - p[i, j]) / hy
    elif j == Ny - 1:
        df_dy = (p[i, j] - p[i, j - 1]) / hy
    else:
        df_dy = (p[i, j + 1] - p[i, j - 1]) / (2.0 * hy)

    grad_p[i, j] = np.sqrt(df_dx * df_dx + df_dy * df_dy)
    Um_r2[i, j] = grad_p[i, j] / mu_o[i, j] * 0.125 / eta
