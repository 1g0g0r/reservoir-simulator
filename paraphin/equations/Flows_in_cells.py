"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
from numba import njit

from paraphin.constants import Nx, Ny, ro_w
from paraphin.utils import (mid, up_value, lam_heat, up_fraction, apply_bc, get_bound,
                            mobility_o, mobility_w, DI, DJ, HIJ, AREA)
from .Temperature import c_oil, h_oil


@njit(cache=True)
def flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, lam_o, lam_w, m, Wo, Wp, Wps,
                   C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq,
                   cells_Q_out, cells_Qo_out, cells_T_out) -> None:
    """Перетоки массы и энергии через четыре грани ячейки.

    Заполняет `cells_S_eq`, `cells_Wp_eq`, `cells_T_eq` и три накопителя оттока для оценки шага по
    Куранту. Проводимость грани - гармоническое среднее суммарных подвижностей соседей; доли фаз,
    температура, суммарная доля парафина и объемная энтальпия нефти берутся вверх по потоку.
    Гармоническое среднее здесь не годится: если в соседней (промытой) ячейке доля равна нулю,
    оно обнуляет поток. За границей области все свойства берутся из самой ячейки, по граничному
    условию меняется только само поле.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    Co = c_oil(i, j, Wo, Wp, Wps, C_o, C_p)
    Ho = h_oil(i, j, T[i, j], Wo, Wp, Wps, C_o, C_p)
    lo_ij, lw_ij = lam_o[i, j], lam_w[i, j]
    lam_ij = lo_ij + lw_ij
    p_c, S_c, T_c = p[i, j], S[i, j], T[i, j]
    m_c, Wo_c, Wsum_c = m[i, j], Wo[i, j], Wp[i, j] + Wps[i, j]
    lam_heat_c = lam_heat(S_c, m_c, Wo_c, Wsum_c)

    _s, _wp, _t, _q_out, _qo_out, _t_out = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
            m_n, Wo_n, Wsum_n = m[i1, j1], Wo[i1, j1], Wp[i1, j1] + Wps[i1, j1]
            Co_ij = c_oil(i1, j1, Wo, Wp, Wps, C_o, C_p)
            Ho_ij = h_oil(i1, j1, T_ij, Wo, Wp, Wps, C_o, C_p)
        else:
            # За границей все свойства берутся из самой ячейки, по ГУ меняется только поле
            bound = get_bound(i1, j1)
            hij *= 0.5
            p_ij = apply_bc(boundary_conditions, bound, 0, p, i, j, hij)
            S_ij = apply_bc(boundary_conditions, bound, 1, S, i, j, hij)
            T_ij = apply_bc(boundary_conditions, bound, 2, T, i, j, hij)
            lo_n = mobility_o(k[i, j], S_ij, mu_o[i, j])
            lw_n = mobility_w(k[i, j], S_ij, mu_w[i, j])
            m_n, Wo_n, Wsum_n = m_c, Wo_c, Wsum_c
            Co_ij = Co
            Ho_ij = Ho + Co * (T_ij - T_c)  # за границей состав тот же, отличается только T

        value = (p_ij - p_c) / hij * areaij * mid(lam_ij, lo_n + lw_n)

        # Доли фаз берутся вверх по потоку
        if p_c >= p_ij:
            lo_up, lw_up = lo_ij, lw_ij
        else:
            lo_up, lw_up = lo_n, lw_n
        lam_up = lo_up + lw_up

        up_kw_ij = up_fraction(lw_up, lam_up) * value
        up_ko_ij = up_fraction(lo_up, lam_up) * value
        up_T_ij = up_value(p_c, T_c, p_ij, T_ij)
        up_Co_ij = up_value(p_c, Co, p_ij, Co_ij)
        up_Ho_ij = up_value(p_c, Ho, p_ij, Ho_ij)
        lam_heat_ij = mid(lam_heat_c, lam_heat(S_ij, m_n, Wo_n, Wsum_n)) * areaij / hij

        _wp += up_ko_ij * up_value(p_c, Wsum_c, p_ij, Wsum_n)
        _s += up_kw_ij
        _t += lam_heat_ij * (T_ij - T_c)
        _t += C_w[i, j] * ro_w * up_T_ij * up_kw_ij
        _t += up_Ho_ij * up_ko_ij  # энтальпия нефти: c_o*T плюс скрытая теплота растворенного парафина

        # value > 0 - приток в ячейку, value < 0 - отток. Для CFL нужен суммарный отток:
        # по насыщенности - объемный, по переносу парафина - нефтяной фазы, по температуре -
        # конвективный тепловой плюс кондуктивный (он идет по обеим граням).
        if value < 0.0:
            _q_out -= value
            _qo_out -= up_ko_ij
            _t_out -= C_w[i, j] * ro_w * up_kw_ij + up_Co_ij * up_ko_ij
        _t_out += lam_heat_ij

    cells_S_eq[i, j] = _s
    cells_Wp_eq[i, j] = _wp
    cells_T_eq[i, j] = _t
    cells_Q_out[i, j] = _q_out
    cells_Qo_out[i, j] = _qo_out
    cells_T_out[i, j] = _t_out
