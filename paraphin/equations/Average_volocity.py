"""Множитель средней скорости нефти в капилляре"""
import numpy as np
from numba import njit

from paraphin.constants import Nx, Ny, hx, hy, eta


@njit
def calc_Um_r2(i, j, p, grad_p, Um_r2, mu_o) -> None:
    """Вычисление средней скорости в капилляре без множителя r^2.

    Parameters
    ----------
    i, j: int
        Индексы текущей ячейки, [-]
    p: numpy.ndarray((Nx, Ny)
		Давление, [Па]
    grad_p: numpy.ndarray((Nx, Ny)
        Поле перепада давления, [Па/м]
	Um_r2: numpy.ndarray((Nx, Ny)
        Средняя скорость в капилляре без множителя r^2, [1/(с*м)]
	mu_o: numpy.ndarray((Nx, Ny)
		Вязкость нефти, [Па*с]
    """
    df_dx, df_dy = 0.0, 0.0

    # Односторонние разности на границах и центральные разности для внутренних точек
    if i == 0:
        df_dx = (p[i + 1, j] - p[i, j]) / hx
    elif i == Nx - 1:
        df_dx = (p[i, j] - p[i - 1, j]) / hx
    else:
        df_dx = (p[i + 1, j] - p[i - 1, j]) / (2 * hx)

    if j == 0:
        df_dy = (p[i, j + 1] - p[i, j]) / hy
    elif j == Ny - 1:
        df_dy = (p[i, j] - p[i, j - 1]) / hy
    else:
        df_dy = (p[i, j + 1] - p[i, j - 1]) / (2 * hy)

    grad_p[i, j] = np.sqrt(df_dx * df_dx + df_dy * df_dy)
    Um_r2[i, j] = grad_p[i, j] / mu_o[i, j] * 0.125 / eta
