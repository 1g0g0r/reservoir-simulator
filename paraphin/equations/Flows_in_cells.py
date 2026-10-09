"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
from numba import njit

from paraphin.constants import Nx, Ny, ro_w
from paraphin.layout import NC_T
from paraphin.utils import (PARAFFIN, DIRICHLET, mid, lam_heat, up_fraction, apply_bc, get_bound,
                            mobility_o, mobility_w, DI, DJ, HIJ, AREA)
from .Temperature import c_oil, h_oil
from .Thermo_wax import sle_hl_boundary, single_split
from paraphin.layout import N_W
from paraphin.oil_composition import WAX_L_REL


@njit(cache=True)
def flows_in_cells(i, j, boundary_conditions, bc_Wc, p, S, T, k, mu_o, mu_w, lam_o, lam_w, lam_h, m, Wp, Wps,
                   Hl, C_o, C_w, C_p, cells_T_eq, cells_S_eq, cells_Q_out, cells_W_eq, Fo):
    """Перетоки массы и энергии через четыре грани ячейки.

    Заполняет `cells_S_eq`, `cells_T_eq`, `cells_Q_out`, потоки нефтяной фазы через грани `Fo[i, j, :]` (по ним
    `components_equation` переносит каждый компонент нефти; при одном переносимом компоненте вместо них - его приток
    `cells_W_eq`, скалярным накопителем: запись по граням выбивает цикл из векторизации) и возвращает еще два оттока -
    нефтяной фазы и
    тепловой: первый - запас ограничителя осаждения и условие Куранта, второй - условие Куранта по температуре.

    Проводимость грани - гармоническое среднее суммарных подвижностей соседей. Доли фаз,
    температура и объемная энтальпия нефти берутся вверх по потоку: гармоническое среднее для них
    не годится - если в соседней (промытой водой) ячейке доля равна нулю, оно обнуляет поток. За
    границей области все свойства берутся из самой ячейки, по граничному условию меняется только
    само поле; состав втекающей нефти задается отдельным ГУ Дирихле `DataField.Paraffin` и
    `Solver.bc_Wc` (без него - как в ячейке). Скрытую теплоту несет носитель `Hl` - растворенный
    парафин, взвешенный по удельной теплоте групп (при одной группе это сам `Wp`).

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    lo_ij, lw_ij = lam_o[i, j], lam_w[i, j]
    lam_ij = lo_ij + lw_ij
    p_c, T_c = p[i, j], T[i, j]
    m_c = m[i, j]
    Wsum_c = Wp[i, j] + Wps[i, j]
    Wo_c = 1.0 - Wsum_c
    C_o_c, C_p_c = C_o[i, j], C_p[i, j]
    Co = c_oil(Wo_c, Wsum_c, C_o_c, C_p_c)
    Ho = h_oil(Co, T_c, Hl[i, j])
    lam_heat_c = lam_h[i, j]
    cw_ro_w = C_w[i, j] * ro_w

    # Грань с оттоком (value < 0) равносильна p_c > p_ij, то есть вверх по потоку на ней всегда сама ячейка.
    # Поэтому отток нефтяной фазы и конвективный тепловой отток - это один и тот же объемный отток `_q_out`, умноженный на константы ячейки.
    # Отдельных накопителей в цикле по граням они не требуют: лишние живые значения там выбивают цикл из векторизации,
    # и каждое обходится дороже, чем вся арифметика, которую экономит.
    fo_c = up_fraction(lo_ij, lam_ij)
    out_heat_coef = cw_ro_w * up_fraction(lw_ij, lam_ij) + Co * fo_c

    _s, _t, _q_out, _cond, _w = 0.0, 0.0, 0.0, 0.0, 0.0

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
            Wsum_n = Wp[i1, j1] + Wps[i1, j1]
            Wo_n = 1.0 - Wsum_n
            Hl_n = Hl[i1, j1]
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
            Wsum_n = Wsum_c
            Hl_n = Hl[i, j]
            if boundary_conditions[bound, PARAFFIN].type == DIRICHLET:
                Wsum_n = boundary_conditions[bound, PARAFFIN].value
                if N_W == 1:  # замкнутая форма: общий `sle_hl_boundary` здесь утяжеляет весь цикл по граням
                    Hl_n = WAX_L_REL[0] * single_split(Wsum_n, T_ij, p_ij)[0]
                else:
                    Hl_n = sle_hl_boundary(bc_Wc, bound, T_ij, p_ij)
                Wo_n = 1.0 - Wsum_n
            C_o_n, C_p_n = C_o_c, C_p_c
            # У фиктивной ячейки своя насыщенность и температура из ГУ, готового значения нет
            lam_heat_n = lam_heat(S_ij, m_c, Wo_c, Wsum_c)

        value = (p_ij - p_c) / hij * areaij * mid(lam_ij, lo_n + lw_n)

        # Все переносимые величины берутся вверх по потоку одной веткой: доли фаз, температура и объемная энтальпия
        # нефти. Энтальпия соседа считается только когда он и есть верх по потоку.
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

        if NC_T == 1:  # переносится один компонент - сразу его приток
            _w += up_ko_ij * Wsum_up
        else:
            Fo[i, j, idx] = up_ko_ij
        _s += up_kw_ij
        _t += lam_heat_ij * (T_ij - T_c) + cw_ro_w * T_up * up_kw_ij + Ho_up * up_ko_ij

        # value > 0 - приток в ячейку, value < 0 - отток
        if value < 0.0:
            _q_out -= value
        _cond += lam_heat_ij  # кондукция идет по всем граням, а не только по граням с оттоком

    cells_S_eq[i, j] = _s
    cells_T_eq[i, j] = _t
    cells_Q_out[i, j] = _q_out
    if NC_T == 1:
        cells_W_eq[i, j] = _w

    return fo_c * _q_out, out_heat_coef * _q_out + _cond
