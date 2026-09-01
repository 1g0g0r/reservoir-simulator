"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
import numpy as np
from numba import njit

from paraphin.constants import Nx, Ny, hx, hy, h, ro_w, ro_o, ro_p
from paraphin.utils import mid, up_T, up_wp, mid_lam, up_fraction, apply_bc, get_bound
from paraphin.utils.math_utils.FVM_utils import _K_o, _K_w

# Смещения соседей и геометрия граней - те же константы, что и в сборке матрицы давления.
# Раньше этот список создавался заново на каждой ячейке; на сетке 25x25 его аллокация была
# основной статьей расхода явного цикла.
_DI   = np.array([1, -1, 0, 0])
_DJ   = np.array([0, 0, 1, -1])
_HIJ  = np.array([hx, hx, hy, hy])
_AREA = np.array([hy * h, hy * h, hx * h, hx * h])


@njit(cache=True)
def flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, lam_o, lam_w, m, Wp, Wps,
                   C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out) -> None:
    """Вычисление потоков в ячейках.

    Parameters
    ----------
    i, j: int
        Индексы текущей ячейки, [-]
    boundary_conditions: ti.field(4, 3, 2)
        Граничные условия: Граница -> Поле -> Тип, Значение
    p: numpy.ndarray(Nx, Ny)
        Давление, [Па]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    T: numpy.ndarray(Nx, Ny)
        Температура, [C]
    k: numpy.ndarray(Nx, Ny)
        Проницаемость, [м^2]
    mu_o: numpy.ndarray(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: numpy.ndarray(Nx, Ny)
        Вязкость воды, [Па*с]
    lam_o, lam_w: numpy.ndarray(Nx, Ny)
        Подвижности фаз k*pf/mu, посчитанные `calc_mobility` до цикла, [м^2/(Па*с)]
    m: numpy.ndarray(Nx, Ny)
        Пористость, [-]
    Wp: numpy.ndarray(Nx, Ny)
        Массовая доля растворенного парафина в нефти, [-]
    Wps: numpy.ndarray(Nx, Ny)
        Массовая доля взвешенного парафина в нефти, [-]
    C_o: numpy.ndarray(Nx, Ny)
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: numpy.ndarray(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_p: numpy.ndarray(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
    cells_T_eq: numpy.ndarray(Nx, Ny)
        Сумма величин перетоков тепла в уравнении энергии
    cells_S_eq: numpy.ndarray(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    cells_Wp_eq: numpy.ndarray(Nx, Ny)
        Перетоки нефти в ячейках, [Па*м]
    cells_Q_out: numpy.ndarray(Nx, Ny)
        Суммарный исходящий объемный поток через грани ячейки, [м^3/с]. Нужен для оценки CFL
    """
    Co = ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j]
    lo_ij, lw_ij = lam_o[i, j], lam_w[i, j]
    lam_ij = lo_ij + lw_ij
    p_c, S_c, T_c = p[i, j], S[i, j], T[i, j]

    _s, _wp, _t, _q_out = 0.0, 0.0, 0.0, 0.0

    for idx in range(4):
        i1 = i + _DI[idx]
        j1 = j + _DJ[idx]
        hij = _HIJ[idx]
        areaij = _AREA[idx]

        # Свойства соседа читаются внутри ветки, а не по переиспользованному i1, j1: при сборке
        # parfor numba типизировала бы индекс, переприсвоенный в обеих ветках, как float64.
        if (0 <= i1 < Nx) and (0 <= j1 < Ny):
            p_ij, S_ij, T_ij = p[i1, j1], S[i1, j1], T[i1, j1]
            lo_n, lw_n = lam_o[i1, j1], lam_w[i1, j1]
            m_n, Wp_n, Wps_n = m[i1, j1], Wp[i1, j1], Wps[i1, j1]
            Co_ij = ro_o * C_o[i1, j1] * (1.0 - Wps_n) + ro_p * Wps_n * C_p[i1, j1]
        else:
            # За границей все свойства берутся из самой ячейки, по ГУ меняется только поле
            bound = get_bound(i1, j1)
            hij *= 0.5
            p_ij = apply_bc(boundary_conditions, bound, 0, p, i, j, hij)
            S_ij = apply_bc(boundary_conditions, bound, 1, S, i, j, hij)
            T_ij = apply_bc(boundary_conditions, bound, 2, T, i, j, hij)
            lo_n = _K_o(k[i, j], S_ij, mu_o[i, j])
            lw_n = _K_w(k[i, j], S_ij, mu_w[i, j])
            m_n, Wp_n, Wps_n = m[i, j], Wp[i, j], Wps[i, j]
            Co_ij = ro_o * C_o[i, j] * (1.0 - Wps_n) + ro_p * Wps_n * C_p[i, j]

        value = (p_ij - p_c) / hij * areaij * mid(lam_ij, lo_n + lw_n)

        # Доли фаз берутся вверх по потоку
        if p_c >= p_ij:
            lo_up, lw_up = lo_ij, lw_ij
        else:
            lo_up, lw_up = lo_n, lw_n
        lam_up = lo_up + lw_up

        up_kw_ij = up_fraction(lw_up, lam_up) * value
        up_ko_ij = up_fraction(lo_up, lam_up) * value
        up_T_ij = up_T(p_c, T_c, p_ij, T_ij)

        _wp += up_ko_ij * up_wp(p_c, Wp[i, j], Wps[i, j], p_ij, Wp_n, Wps_n)
        _s += up_kw_ij
        _t += (T_ij - T_c) / hij * areaij * mid_lam(S_c, m[i, j], Wps[i, j], S_ij, m_n, Wps_n)
        _t += C_w[i, j] * ro_w * up_T_ij * up_kw_ij
        _t += mid(Co, Co_ij) * up_T_ij * up_ko_ij

        # value > 0 - приток в ячейку, value < 0 - отток. Для CFL нужен суммарный отток.
        if value < 0.0:
            _q_out -= value

    cells_S_eq[i, j] = _s
    cells_Wp_eq[i, j] = _wp
    cells_T_eq[i, j] = _t
    cells_Q_out[i, j] = _q_out
