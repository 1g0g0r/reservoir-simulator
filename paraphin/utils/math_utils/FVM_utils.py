"""Вспомогательные процедуры для реализации решения методом конечных объемов."""
import numpy as np
from numba import njit, prange

from paraphin.constants import data_type, K_o, K_f, K_w, K_p, Nx, Ny, hx, hy, h, init_m
from .phase_f import pf_o, pf_w, pf_o_mix, pf_w_mix

# Обход соседей ячейки: вправо, влево, вверх, вниз. Смещение индексов, расстояние между центрами и площадь грани.
# Одни и те же таблицы нужны и сборке матрицы давления, и расчету перетоков, поэтому лежат здесь.
DI = np.array([1, -1, 0, 0])
DJ = np.array([0, 0, 1, -1])
HIJ = np.array([hx, hx, hy, hy])
AREA = np.array([hy * h, hy * h, hx * h, hx * h])


@njit(parallel=True, cache=True)
def calc_mobility(k, S, m, Wp, Wps, mu_o, mu_w, lam_o, lam_w, lam_h) -> None:
    """Поячеечные свойства, общие для сборки матрицы давления и для перетоков.

    Считаются один раз за шаг, до сборки матрицы давления. Раньше `mid_Ko_Kw`, `up_ko` и `up_kw`
    пересчитывали подвижности для обеих ячеек на каждой из четырех граней - по 8 вычислений
    `pf_o`/`pf_w` на ячейку вместо одного. Эффективная теплопроводность `lam_h` попала сюда по той
    же причине: `flows_in_cells` считала ее пять раз на ячейку (свою и по разу на каждого соседа),
    хотя величина у ячейки одна. Исключение - фиктивная ячейка за границей: там своя
    насыщенность и температура из ГУ, и `lam_heat` для нее считается на месте.
    """
    for i in prange(Nx):
        for j in range(Ny):
            lam_o[i, j] = mobility_o(k[i, j], S[i, j], mu_o[i, j])
            lam_w[i, j] = mobility_w(k[i, j], S[i, j], mu_w[i, j])
            w_sum = Wp[i, j] + Wps[i, j]
            lam_h[i, j] = lam_heat(S[i, j], m[i, j], 1.0 - w_sum, w_sum)


@njit(parallel=True, cache=True)
def calc_mobility_w(k, S, m, Wp, Wps, mu_o, mu_w, lam_o, lam_w, lam_h, g_ads, g_max, s_min_ow, s_max_ow, n_o_ow,
                    n_w_ow) -> None:
    """`calc_mobility` при смене смачиваемости (`wettability`): ОФП - смесь водо- и нефтесмачиваемых наборов
    по доле покрытия поверхности адсорбированными асфальтенами omega = min(G/G_max, 1) (`pf_o_mix`, `pf_w_mix`).
    g_ads, g_max - поля адсорбированных асфальтенов и предельной адсорбции (`kx['ga']`, `kx['gmax']`)."""
    for i in prange(Nx):
        for j in range(Ny):
            omega = min(g_ads[i, j] / g_max[i, j], 1.0) if g_max[i, j] > 0.0 else 0.0
            s = S[i, j]
            lam_o[i, j] = k[i, j] * pf_o_mix(s, omega, s_min_ow, s_max_ow, n_o_ow) / mu_o[i, j]
            lam_w[i, j] = k[i, j] * pf_w_mix(s, omega, s_min_ow, s_max_ow, n_w_ow) / mu_w[i, j]
            w_sum = Wp[i, j] + Wps[i, j]
            lam_h[i, j] = lam_heat(S[i, j], m[i, j], 1.0 - w_sum, w_sum)


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
def lam_heat(S: data_type, m: data_type, Wo: data_type, Wsum: data_type) -> data_type:
    """Эффективная теплопроводность ячейки, [Вт/(м*С)].

        L = m*S_w*K_w + m*S_o*(w_o*K_o + (w_p + w_ps)*K_p) + (m_0 - m)*K_p + (1 - m_0)*K_f

    `m_0` здесь - *начальная* пористость `init_m`, а не предыдущий временной слой: слагаемое
    (m_0 - m) описывает выведенный из фильтрации объем (осевший парафин и содержимое
    заблокированных каналов), свойства которого приняты равными свойствам парафина. Прежняя
    запись (1 - m)*K_f отдавала этот объем породе.

    Отдельно от осреднения по грани: величина у своей ячейки одна на все четыре грани, и раньше
    `mid_lam` пересчитывала ее четырежды. Осреднение по грани - `mid(lam_heat_i, lam_heat_j)`.
    """
    return (m * (S * K_w + (1.0 - S) * (Wo * K_o + Wsum * K_p))
            + (init_m - m) * K_p + (1.0 - init_m) * K_f)


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
