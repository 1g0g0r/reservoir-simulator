"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
from numba import njit

from paraphin.constants import Nx, Ny, ro_w, ro_o, ro_p
from paraphin.utils import (mid, up_T, up_wp, lam_heat, up_fraction, apply_bc, get_bound,
                            mobility_o, mobility_w, DI, DJ, HIJ, AREA)


@njit(cache=True)
def flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, lam_o, lam_w, m, Wp, Wps,
                   C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out) -> None:
    """Перетоки массы и энергии через четыре грани ячейки.

    Заполняет `cells_S_eq`, `cells_Wp_eq`, `cells_T_eq` и `cells_Q_out` для ячейки (i, j).
    Проводимость грани - гармоническое среднее суммарных подвижностей соседей; доли фаз,
    температура и концентрация парафина берутся вверх по потоку. За границей области все свойства
    берутся из самой ячейки, по граничному условию меняется только само поле.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    Co = ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j]
    lo_ij, lw_ij = lam_o[i, j], lam_w[i, j]
    lam_ij = lo_ij + lw_ij
    p_c, S_c, T_c = p[i, j], S[i, j], T[i, j]
    m_c, Wp_c, Wps_c = m[i, j], Wp[i, j], Wps[i, j]
    lam_heat_c = lam_heat(S_c, m_c, Wps_c)

    _s, _wp, _t, _q_out = 0.0, 0.0, 0.0, 0.0

    for idx in range(4):
        i1 = i + DI[idx]
        j1 = j + DJ[idx]
        hij = HIJ[idx]
        areaij = AREA[idx]

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
            lo_n = mobility_o(k[i, j], S_ij, mu_o[i, j])
            lw_n = mobility_w(k[i, j], S_ij, mu_w[i, j])
            m_n, Wp_n, Wps_n = m_c, Wp_c, Wps_c
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

        _wp += up_ko_ij * up_wp(p_c, Wp_c, Wps_c, p_ij, Wp_n, Wps_n)
        _s += up_kw_ij
        _t += (T_ij - T_c) / hij * areaij * mid(lam_heat_c, lam_heat(S_ij, m_n, Wps_n))
        _t += C_w[i, j] * ro_w * up_T_ij * up_kw_ij
        _t += mid(Co, Co_ij) * up_T_ij * up_ko_ij

        # value > 0 - приток в ячейку, value < 0 - отток. Для CFL нужен суммарный отток.
        if value < 0.0:
            _q_out -= value

    cells_S_eq[i, j] = _s
    cells_Wp_eq[i, j] = _wp
    cells_T_eq[i, j] = _t
    cells_Q_out[i, j] = _q_out
