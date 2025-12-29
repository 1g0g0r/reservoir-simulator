"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
from numba import njit

from paraphin.constants import Nx, Ny, hx, hy, h, ro_w, ro_o, ro_p
from paraphin.utils import up_kw, up_ko, mid_Ko_Kw, mid, up_T, up_wp, mid_lam, apply_bc, get_bound


@njit
def flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, m, Wp, Wps, C_o, C_w, C_p,
                   cells_T_eq, cells_Wp_eq, cells_S_eq) -> None:
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
    """
    Co = ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j]

    arr = [[i + 1, j, hx, hy * h], [i - 1, j, hx, hy * h], [i, j + 1, hy, hx * h], [i, j - 1, hy, hx * h]]
    _s, _wp, _t, p_ij, S_ij, T_ij = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for idx in range(4):
        i1, j1, hij, areaij = arr[idx]
        if (0 <= i1 < Nx) and (0 <= j1 < Ny):
            p_ij, S_ij, T_ij = p[i1, j1], S[i1, j1], T[i1, j1]
        else:
            bound = get_bound(i1, j1)
            i1, j1, hij = i, j, hij * 0.5
            p_ij = apply_bc(boundary_conditions, bound,0, p, i, j, hij)
            S_ij = apply_bc(boundary_conditions, bound,1, S, i, j, hij)
            T_ij = apply_bc(boundary_conditions, bound,2, T, i, j, hij)

        Co_ij = ro_o * C_o[i1, j1] * (1.0 - Wps[i1, j1]) + ro_p * Wps[i1, j1] * C_p[i1, j1]
        value = (p_ij - p[i, j]) / hij * areaij * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                                            k[i1, j1], S_ij, mu_o[i1, j1], mu_w[i1, j1])
        up_kw_ij = up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                         k[i1, j1], S_ij, p_ij, mu_o[i1, j1], mu_w[i1, j1]) * value
        up_ko_ij = up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                         k[i1, j1], S_ij, p_ij, mu_o[i1, j1], mu_w[i1, j1]) * value
        up_T_ij = up_T(p[i, j], T[i, j], p_ij, T_ij)

        _wp += up_ko_ij * up_wp(p[i, j], Wp[i, j], Wps[i, j], p_ij, Wp[i1, j1], Wps[i1, j1])
        _s += up_kw_ij
        _t += (T_ij - T[i, j]) / hij * areaij * mid_lam(S[i, j], m[i, j], Wps[i, j], S_ij, m[i1, j1], Wps[i1, j1])
        _t += C_w[i, j] * ro_w * up_T_ij * up_kw_ij
        _t += mid(Co, Co_ij) * up_T_ij * up_ko_ij

    cells_S_eq[i, j] = _s
    cells_Wp_eq[i, j] = _wp
    cells_T_eq[i, j] = _t
