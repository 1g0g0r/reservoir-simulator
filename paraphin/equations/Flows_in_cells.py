"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
from numba import njit

from paraphin.constants import Nx, Ny, ro_w, wax_components
from paraphin.utils import (mid, lam_heat, up_fraction, apply_bc, get_bound,
                            mobility_o, mobility_w, DI, DJ, HIJ, AREA)
from .Temperature import c_oil, h_oil
from .Wp_balance import _wp_saturated
from .Thermo_wax import sle_hl_boundary


@njit(cache=True)
def flows_in_cells(i, j, boundary_conditions, bc_Wc, p, S, T, k, mu_o, mu_w, lam_o, lam_w, lam_h, m, Wo, Wp, Wps,
                   Hl, C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out, Fo_row):
    """Перетоки массы и энергии через четыре грани ячейки.

    Заполняет `cells_S_eq`, `cells_Wp_eq`, `cells_T_eq`, `cells_Q_out` и возвращает еще два оттока
    для условий Куранта - нефтяной фазы и тепловой. Массивами они не заводятся: за пределами своей
    итерации цикла их никто не читает, в отличие от `cells_Q_out`, который нужен `_calc_dt` для
    ячеек со скважинами.

    Проводимость грани - гармоническое среднее суммарных подвижностей соседей. Доли фаз,
    температура, суммарная доля парафина и объемная энтальпия нефти берутся вверх по потоку:
    гармоническое среднее для них не годится - если в соседней (промытой водой) ячейке доля равна
    нулю, оно обнуляет поток. За границей области все свойства берутся из самой ячейки, по
    граничному условию меняется только само поле; состав втекающей нефти задается отдельным ГУ
    Дирихле `DataField.Paraffin` (без него - как в ячейке).

    При детальном составе (флаг `wax_components`) грань дополнительно записывает поток нефтяной фазы в
    `Fo_row` - по нему `components_equation` переносит каждый компонент, - а скрытую теплоту несет не
    `Wp`, а `Hl` (растворенный парафин, взвешенный по удельной теплоте групп). Без флага обе ветки
    выбрасываются на компиляции, и цикл по граням - ровно прежний.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    lo_ij, lw_ij = lam_o[i, j], lam_w[i, j]
    lam_ij = lo_ij + lw_ij
    p_c, T_c = p[i, j], T[i, j]
    m_c, Wo_c, Wp_c = m[i, j], Wo[i, j], Wp[i, j]
    Wsum_c = Wp_c + Wps[i, j]
    C_o_c, C_p_c = C_o[i, j], C_p[i, j]
    Co = c_oil(Wo_c, Wsum_c, C_o_c, C_p_c)
    if wax_components:
        Ho = h_oil(Co, T_c, Hl[i, j])
    else:
        Ho = h_oil(Co, T_c, Wp_c)
    lam_heat_c = lam_h[i, j]
    cw_ro_w = C_w[i, j] * ro_w

    # Грань с оттоком (value < 0) равносильна p_c > p_ij, то есть вверх по потоку на ней всегда сама ячейка.
    # Поэтому отток нефтяной фазы и конвективный тепловой отток - это один и тот же объемный отток `_q_out`, умноженный на константы ячейки.
    # Отдельных накопителей в цикле по граням они не требуют: лишние живые значения там выбивают цикл из векторизации,
    # и каждое обходится дороже, чем вся арифметика, которую экономит.
    fo_c = up_fraction(lo_ij, lam_ij)
    out_heat_coef = cw_ro_w * up_fraction(lw_ij, lam_ij) + Co * fo_c

    _s, _wp, _t, _q_out, _cond = 0.0, 0.0, 0.0, 0.0, 0.0

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
            Wo_n, Wp_n = Wo[i1, j1], Wp[i1, j1]
            Wsum_n = Wp_n + Wps[i1, j1]
            if wax_components:
                Hl_n = Hl[i1, j1]
            else:
                Hl_n = Wp_n
            C_o_n, C_p_n = C_o[i1, j1], C_p[i1, j1]
            lam_heat_n = lam_h[i1, j1]
        else:
            # За границей все свойства берутся из самой ячейки, по ГУ меняется только поле
            bound = get_bound(i1, j1)
            hij *= 0.5
            p_ij = apply_bc(boundary_conditions, bound, 0, p, i, j, hij)
            S_ij = apply_bc(boundary_conditions, bound, 1, S, i, j, hij)
            T_ij = apply_bc(boundary_conditions, bound, 2, T, i, j, hij)
            lo_n = mobility_o(k[i, j], S_ij, mu_o[i, j])
            lw_n = mobility_w(k[i, j], S_ij, mu_w[i, j])
            # Состав втекающей нефти: по ГУ Дирихле (прокачка нефти через керн), иначе - как в самой ячейке.
            # Делится на растворенный и взвешенный по температуре фиктивной ячейки. Присваивания - поштучно,
            # не кортежем: кортеж в ветке внутри parfor ломает вывод типов индексов (см. выше).
            Wo_n = Wo_c
            Wp_n = Wp_c
            Wsum_n = Wsum_c
            if wax_components:
                Hl_n = Hl[i, j]
            else:
                Hl_n = Wp_c
            if boundary_conditions[bound, 3, 0] == 1:
                Wsum_n = boundary_conditions[bound, 3, 1]
                if wax_components:
                    Hl_n = sle_hl_boundary(bc_Wc[bound, :], T_ij, p_ij)
                else:
                    Wp_n = _wp_saturated(Wsum_n, T_ij)
                    Hl_n = Wp_n
                Wo_n = 1.0 - Wsum_n
            C_o_n, C_p_n = C_o_c, C_p_c
            # У фиктивной ячейки своя насыщенность и температура из ГУ, готового значения нет
            lam_heat_n = lam_heat(S_ij, m_c, Wo_c, Wsum_c)

        value = (p_ij - p_c) / hij * areaij * mid(lam_ij, lo_n + lw_n)

        # Все переносимые величины берутся вверх по потоку одной веткой: доли фаз, температура, суммарная доля парафина
        # и объемная энтальпия нефти. Энтальпия соседа считается только когда он и есть верх по потоку.
        if p_c >= p_ij:
            lo_up, lw_up = lo_ij, lw_ij
            T_up, Ho_up, Wsum_up = T_c, Ho, Wsum_c
        else:
            lo_up, lw_up = lo_n, lw_n
            T_up = T_ij
            Ho_up = h_oil(c_oil(Wo_n, Wsum_n, C_o_n, C_p_n), T_ij, Hl_n)
            Wsum_up = Wsum_n

        lam_up = lo_up + lw_up
        up_kw_ij = up_fraction(lw_up, lam_up) * value
        up_ko_ij = up_fraction(lo_up, lam_up) * value
        lam_heat_ij = mid(lam_heat_c, lam_heat_n) * areaij / hij

        _wp += up_ko_ij * Wsum_up
        if wax_components:
            Fo_row[idx] = up_ko_ij
        _s += up_kw_ij
        _t += lam_heat_ij * (T_ij - T_c) + cw_ro_w * T_up * up_kw_ij + Ho_up * up_ko_ij

        # value > 0 - приток в ячейку, value < 0 - отток
        if value < 0.0:
            _q_out -= value
        _cond += lam_heat_ij  # кондукция идет по всем граням, а не только по граням с оттоком

    cells_S_eq[i, j] = _s
    cells_Wp_eq[i, j] = _wp
    cells_T_eq[i, j] = _t
    cells_Q_out[i, j] = _q_out

    return fo_c * _q_out, out_heat_coef * _q_out + _cond
