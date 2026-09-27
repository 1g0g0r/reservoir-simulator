"""Перенос компонентов нефтяной фазы и их равновесие: группы парафина, асфальтены, смолы (флаг `wax_components`).

Компоненты (массовые доли в нефтяной фазе, индексы - `paraphin.oil_composition`):
    Wc[..., :N_w]   группы н-алканов, растворенные + взвешенные кристаллы;
    Wc[..., IA_D]   растворенные (пептизированные) асфальтены;
    Wc[..., IA_F]   флокулы асфальтенов;
    Wc[..., I_R]    смолы.
Остаток 1 - sum(Wc) - ненормальные насыщенные и ароматика в неизменной пропорции (SARA).
Все компоненты движутся с нефтью с одной скоростью (обзор, разд. 3.2: уравнение неразрывности по
компоненту со стоком в отложения), поэтому уравнение сохранения для каждого одно и то же:

    (m*S_o*w_c)^new = (m*S_o*w_c) + dt/|V|*[sum_f F_f*w_c,up + q_o*w_c] - dt*s_c,          (1)

F_f - объемный поток нефтяной фазы через грань f (`flows_in_cells` пишет их в `Fo_row`), w_c,up - доля
вверх по потоку, q_o - дебит нефти скважины. Стоки s_c - осадок в порах (`calc_qp_m_k_fi[_2]`), поделенный
на ro_o, как в `wp_equation`:
    группа парафина k:  ro_p/ro_o*(q_p1 + q_p2) * w_ps,k/sum(w_ps)  - оседают взвешенные кристаллы;
    флокулы:            ro_ad/ro_o*(1 - f_r)*q_pa;
    смолы:              ro_ad/ro_o*f_r*q_pa  (соосаждение пептизирующих смол).
После переноса - равновесие: группы парафина делятся на растворенные и взвешенные (`Thermo_wax.sle_split`),
асфальтены релаксируют к равновесной доле флокул (`Asphaltene.floc_relax`). Отложения копятся в `Dep`,
[кг/м^3 породы]: sum_k Dep_k/ro_p + (Dep_af + Dep_r)/ro_ad = m0 - m.
"""
from numba import njit

from paraphin.constants import Nx, Ny, volume, ro_o, ro_p, ro_asph_dep, resin_in_deposit, asphaltenes
from paraphin.oil_composition import N_W, NC, IA_D, IA_F, I_R, F_SAT_REST
from paraphin.utils import get_bound, DI, DJ
from .Asphaltene import asph_soluble, floc_relax
from .Thermo_wax import sle_split

_RO_P_O = ro_p / ro_o
_RO_AD_O = ro_asph_dep / ro_o


@njit(cache=True)
def components_equation(i, j, _paraphin, boundary_conditions, bc_Wc, Fo_row, p, T, m, S, new_m, new_S,
                        Wc, new_Wc, Ws, new_Ws, src_Qo, new_qp1, new_qp2, new_qpa, new_Wp, new_Wps, new_Hl,
                        Dep, dt) -> None:
    """Перенос (1), стоки в отложения, равновесие. Пишет new_Wc, new_Ws, суммы new_Wp, new_Wps и new_Hl.

    Fo_row: numpy.ndarray(4)
        Потоки нефтяной фазы через грани ячейки из `flows_in_cells` этой же итерации (строка скретча `Fo[i]`).
    bc_Wc: numpy.ndarray(4, NC)
        Состав нефти, втекающей через границу с ГУ Дирихле `DataField.Paraffin`.
    Dep: numpy.ndarray(Nx, Ny, NC)
        Накопленные отложения по компонентам, [кг/м^3 породы]; обновляется на месте.
    """
    if not _paraphin:
        return None

    mso = m[i, j] * (1.0 - S[i, j])
    mso_new = new_m[i, j] * (1.0 - new_S[i, j])

    # Стоки в отложения
    qw = new_qp1[i, j] + new_qp2[i, j]
    ws_sum = 0.0
    for k in range(N_W):
        ws_sum += Ws[i, j, k]
    s_af = _RO_AD_O * (1.0 - resin_in_deposit) * new_qpa[i, j]
    s_r = _RO_AD_O * resin_in_deposit * new_qpa[i, j]
    for k in range(N_W):
        s_k = _RO_P_O * qw * Ws[i, j, k] / ws_sum if ws_sum > 0.0 else 0.0
        Dep[i, j, k] += dt * ro_o * s_k
        new_Wc[i, j, k] = mso * Wc[i, j, k] - dt * s_k
    Dep[i, j, IA_F] += dt * ro_o * s_af
    Dep[i, j, I_R] += dt * ro_o * s_r
    new_Wc[i, j, IA_D] = mso * Wc[i, j, IA_D]
    new_Wc[i, j, IA_F] = mso * Wc[i, j, IA_F] - dt * s_af
    new_Wc[i, j, I_R] = mso * Wc[i, j, I_R] - dt * s_r

    # Перенос: поток через грань и дебит скважины (у добывающей - свой состав, у нагнетательной q_o = 0)
    coef = dt / volume
    q_well = coef * src_Qo[i, j]
    for c in range(NC):
        new_Wc[i, j, c] += q_well * Wc[i, j, c]
    for idx in range(4):
        f = coef * Fo_row[idx]
        if f > 0.0:
            i1 = i + DI[idx]
            j1 = j + DJ[idx]
            if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                for c in range(NC):
                    new_Wc[i, j, c] += f * Wc[i1, j1, c]
            else:
                bound = get_bound(i1, j1)
                if boundary_conditions[bound, 3, 0] == 1:
                    for c in range(NC):
                        new_Wc[i, j, c] += f * bc_Wc[bound, c]
                else:
                    for c in range(NC):
                        new_Wc[i, j, c] += f * Wc[i, j, c]
        elif f < 0.0:
            for c in range(NC):
                new_Wc[i, j, c] += f * Wc[i, j, c]

    if mso_new <= 1e-12:  # ячейка промыта водой: нефтяной фазы нет, переносить нечего
        for c in range(NC):
            new_Wc[i, j, c] = 0.0
        for k in range(N_W):
            new_Ws[i, j, k] = 0.0
        new_Wp[i, j] = 0.0
        new_Wps[i, j] = 0.0
        new_Hl[i, j] = 0.0
        return None

    wax_total = 0.0
    for c in range(NC):
        new_Wc[i, j, c] = min(max(new_Wc[i, j, c] / mso_new, 0.0), 1.0)
        if c < N_W:
            wax_total += new_Wc[i, j, c]

    # Равновесие групп парафина при температуре текущего слоя (как в `wp_equation`) и новом давлении
    w_dis, w_sus, hl = sle_split(new_Wc[i, j, :N_W], T[i, j], p[i, j], new_Ws[i, j, :])
    new_Wp[i, j] = w_dis
    new_Wps[i, j] = w_sus
    new_Hl[i, j] = hl

    # Асфальтены: равновесная доля флокул по параметру растворимости мальтенов и релаксация к ней
    if asphaltenes:
        a_tot = new_Wc[i, j, IA_D] + new_Wc[i, j, IA_F]
        w_res = new_Wc[i, j, I_R]
        rest = max(1.0 - wax_total - a_tot - w_res, 0.0)
        w_max = asph_soluble(rest * F_SAT_REST + w_dis, rest * (1.0 - F_SAT_REST), w_res, p[i, j], T[i, j])
        af_eq = max(a_tot - w_max, 0.0)
        af = min(max(floc_relax(new_Wc[i, j, IA_F], af_eq, dt), 0.0), a_tot)
        new_Wc[i, j, IA_F] = af
        new_Wc[i, j, IA_D] = a_tot - af
