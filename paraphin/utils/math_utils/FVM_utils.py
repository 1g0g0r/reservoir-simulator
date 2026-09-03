"""Вспомогательные процедуры для реализации решения методом конечных объемов."""
import numpy as np
from numba import njit, prange

from paraphin.constants import data_type, ro_o, ro_p, K_o, K_f, K_w, K_p, Nx, Ny, hx, hy, h
from .phase_f import pf_o, pf_w

# Обход соседей ячейки: вправо, влево, вверх, вниз. Смещение индексов, расстояние между центрами
# и площадь грани. Одни и те же таблицы нужны и сборке матрицы давления, и расчету перетоков,
# поэтому лежат здесь, а не копией в каждом из двух модулей.
DI = np.array([1, -1, 0, 0])
DJ = np.array([0, 0, 1, -1])
HIJ = np.array([hx, hx, hy, hy])
AREA = np.array([hy * h, hy * h, hx * h, hx * h])


@njit(parallel=True, cache=True)
def calc_mobility(k, S, mu_o, mu_w, lam_o, lam_w) -> None:
    """Подвижности фаз k*pf/mu во всех ячейках.

    Считаются один раз за шаг, до сборки матрицы давления. Раньше `mid_Ko_Kw`, `up_ko` и `up_kw`
    пересчитывали их для обеих ячеек на каждой из четырех граней - по 8 вычислений `pf_o`/`pf_w`
    на ячейку вместо одного.
    """
    for i in prange(Nx):
        for j in range(Ny):
            lam_o[i, j] = mobility_o(k[i, j], S[i, j], mu_o[i, j])
            lam_w[i, j] = mobility_w(k[i, j], S[i, j], mu_w[i, j])


@njit(cache=True)
def up_fraction(lam_up: data_type, lam_sum: data_type) -> data_type:
    """Доля фазы в суммарной подвижности ячейки вверх по потоку. Ноль при нулевой подвижности."""
    return lam_up / lam_sum if lam_sum > 0.0 else 0.0


@njit(cache=True)
def mid(x: data_type, y: data_type) -> data_type:
    """Среднее гармоническое. Ноль при нулевой сумме: иначе 0/0 даст NaN и испортит всю матрицу."""
    s = x + y
    return 2.0 * x * y / s if s > 0.0 else 0.0


@njit(cache=True)
def up_T(p_i: data_type, T_i: data_type, p_j: data_type, T_j: data_type) -> data_type:
    """Температура ячейки вверх по потоку."""
    return T_i if p_i >= p_j else T_j


@njit(cache=True)
def up_wp(p_i: data_type, Wp_i: data_type, Wps_i: data_type,
          p_j: data_type, Wp_j: data_type, Wps_j: data_type) -> data_type:
    """Массовая концентрация парафина (растворенного и взвешенного) вверх по потоку."""
    if p_i >= p_j:
        return ro_o * Wp_i + ro_p * Wps_i

    return ro_o * Wp_j + ro_p * Wps_j


@njit(cache=True)
def lam_heat(S: data_type, m: data_type, Wps: data_type) -> data_type:
    """Эффективная теплопроводность ячейки, [Вт/(м*С)].

    Отдельно от осреднения по грани: величина у своей ячейки одна на все четыре грани, и раньше
    `mid_lam` пересчитывала ее четырежды. Осреднение по грани - `mid(lam_heat_i, lam_heat_j)`.
    """
    return m * (S * K_w + (1.0 - S) * ((1.0 - Wps) * K_o + Wps * K_p)) + (1.0 - m) * K_f


@njit(cache=True)
def mobility_o(k: data_type, s: data_type, mu_o: data_type) -> data_type:
    """Подвижность нефти k*pf_o/mu_o в одной ячейке, [м^2/(Па*с)].

    Имя не K_o: так называется теплопроводность нефти в constants.py.
    """
    return k * pf_o(s) / mu_o


@njit(cache=True)
def mobility_w(k: data_type, s: data_type, mu_w: data_type) -> data_type:
    """Подвижность воды k*pf_w/mu_w в одной ячейке, [м^2/(Па*с)]."""
    return k * pf_w(s) / mu_w
