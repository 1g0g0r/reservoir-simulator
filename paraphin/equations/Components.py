"""Перенос компонентов нефтяной фазы и их равновесие: группы парафина, асфальтены, смолы.

Компоненты (массовые доли в нефтяной фазе, индексы - `paraphin/layout.py`):
    Wc[..., :N_w]   группы н-алканов, растворенные + взвешенные кристаллы;
    Wc[..., IA_D]   растворенные (пептизированные) асфальтены;
    Wc[..., IA_F]   флокулы асфальтенов;
    Wc[..., I_R]    смолы.
Остаток 1 - sum(Wc) - ненормальные насыщенные и ароматика в неизменной пропорции (SARA). Однокомпонентная модель
парафина - частный случай: одна группа (`wax_characterization = 'single'`). Без асфальтенов переносятся только
группы парафина (`layout.NC_T`): доли асфальтенов и смол тогда ни на что не влияют.
Все компоненты движутся с нефтью с одной скоростью (обзор, разд. 3.2: уравнение неразрывности по
компоненту со стоком в отложения), поэтому уравнение сохранения для каждого одно и то же:

    (m*S_o*w_c)^new = (m*S_o*w_c) + dt/|V|*[sum_f F_f*w_c,up + q_o*w_c] - dt*s_c,          (1)

F_f - объемный поток нефтяной фазы через грань f (`flows_in_cells` пишет их в `Fo`), w_c,up - доля
вверх по потоку, q_o - дебит нефти скважины. Стоки s_c - осадок в порах (`calc_qp_m_k_fi`, `Deposition`), поделенный
на ro_o (доли массовые, плотность фазы одна):
    группа парафина k:  ro_p/ro_o*(q_p1 + q_p2) * w_ps,k/sum(w_ps)  - оседают взвешенные кристаллы;
    флокулы:            ro_ad/ro_o*(1 - f_r)*q_pa;
    смолы:              ro_ad/ro_o*f_r*q_pa  (соосаждение пептизирующих смол).
После переноса - равновесие: группы парафина делятся на растворенные и взвешенные (`Thermo_wax.sle_split`),
асфальтены релаксируют к равновесной доле флокул (`Asphaltene.floc_relax`). Отложения копятся в `Dep`,
[кг/м^3 породы]: sum_k Dep_k/ro_p + (Dep_af + Dep_r)/ro_ad = m0 - m.
"""
import math

from numba import njit

from paraphin.constants import (Nx, Ny, volume, ro_o, ro_p, ro_asph, ro_asph_dep, resin_in_deposit, asphaltenes,
                                wax_kinetics, asph_aggregation, adsorption, deposition_kinetics)
from paraphin.kinetics_params import K_CRYST, K_DISS, AGG_D0, AGG_W
from paraphin.layout import N_W, NC_T, IA_D, IA_F, I_R, IS0, IN_F, LEAN
from paraphin.oil_composition import F_SAT_REST, WAX_L_REL
from paraphin.utils import get_bound, DI, DJ, PARAFFIN, DIRICHLET
from .Asphaltene import asph_soluble, floc_relax
from .Thermo_wax import sle_split, single_split
from .Kinetics_math import relax_exp, coagulation_kernel, smoluchowski_step

_RO_P_O = ro_p / ro_o
_RO_AD_O = ro_asph_dep / ro_o


@njit(cache=True)
def components_equation(i, j, _paraphin, boundary_conditions, bc_Wc, cells_W_eq, Fo, p, T, m, S, new_m, new_S,
                        Wc, new_Wc, Ws, new_Ws, src_Qo, new_qp1, new_qp2, new_Wp, new_Wps, new_Hl,
                        Dep, kin, kx, new_kx, mu_p, dt) -> None:
    """Перенос (1), стоки в отложения, равновесие. Пишет new_Wc, new_Ws, суммы new_Wp, new_Wps и new_Hl.

    cells_W_eq, Fo: numpy.ndarray(Nx, Ny), numpy.ndarray(Nx, Ny, 4)
        Из `flows_in_cells` этого шага: потоки нефтяной фазы через грани `Fo[i, j, :]` или, при одном переносимом
        компоненте (`NC_T = 1`), сразу его приток через грани sum_f F_f*w_up - `cells_W_eq[i, j]`.
    bc_Wc: numpy.ndarray(4, NC)
        Состав нефти, втекающей через границу с ГУ Дирихле `DataField.Paraffin`.
    Dep: numpy.ndarray(Nx, Ny, NC)
        Накопленные отложения по компонентам, [кг/м^3 породы]; обновляется на месте.
    kin, kx, new_kx
        Параметры и поля кинетических моделей (`paraphin/kinetics_params.py`); без флагов кинетики не читаются.

    Флаги кинетики (`equations/Deposition.py`) добавляют:
      - стоки стеночной кристаллизации и старения (из растворенного парафина, по долям групп `KX_WSH`,
        `kx.gsh`) и адсорбции (из растворенных асфальтенов и смол);
      - отрицательные стоки - вынос: осадок возвращается во взвесь и флокулы по своему составу;
      - `wax_kinetics`: взвесь групп - переносимое состояние Wc[IS0 + k], релаксирующее к равновесию
        с k_cryst (кристаллизация) или k_diss (растворение), d w_s/dt = k*(w_s^eq - w_s), точно за шаг;
      - `asph_aggregation`: число флокул Wc[IN_F] - новые первичные частицы из выпадения и броуновская
        коагуляция Смолуховского dN/dt = -K*N^2/2, K = 8*k_B*T/(3*mu*W), точно за шаг N/(1 + K*N*dt/2).
    """
    if not _paraphin:
        return None

    mso = m[i, j] * (1.0 - S[i, j])
    mso_new = new_m[i, j] * (1.0 - new_S[i, j])

    if LEAN:
        # Однокомпонентная модель - (1) для единственной группы без циклов по компонентам и отложениям: приток через
        # грани собран в `flows_in_cells`, сток - кристаллы q_p1 + q_p2, равновесие - замкнутой формой, взвесь по
        # группам Ws не нужна. Общий путь здесь вдвое замедлял шаг однокомпонентного расчета. Сумма группы - из Wc, а не
        # аргументами Wp, Wps: два лишних аргумента-массива замедляли проход по ячейкам на 40 % (75x75)
        w = Wc[i, j, 0]
        s_k = _RO_P_O * (new_qp1[i, j] + new_qp2[i, j])
        Dep[i, j, 0] += dt * ro_o * s_k
        if mso_new <= 1e-12:  # ячейка промыта водой: нефтяной фазы нет
            new_Wc[i, j, 0] = 0.0
            new_Wp[i, j] = 0.0
            new_Wps[i, j] = 0.0
            new_Hl[i, j] = 0.0
            return None
        w = (mso * w + dt * ((cells_W_eq[i, j] + src_Qo[i, j] * w) / volume - s_k)) / mso_new
        w = min(max(w, 0.0), 1.0)
        new_Wc[i, j, 0] = w
        w_dis, w_sus = single_split(w, T[i, j], p[i, j])
        new_Wp[i, j] = w_dis
        new_Wps[i, j] = w_sus
        new_Hl[i, j] = WAX_L_REL[0] * w_dis
        return None

    # Стоки в отложения
    qw = new_qp1[i, j] + new_qp2[i, j]
    ws_sum = 0.0
    for k in range(N_W):
        ws_sum += Ws[i, j, k]
    dep_w_sum = 0.0
    if deposition_kinetics:
        for k in range(N_W):
            dep_w_sum += Dep[i, j, k]
    for k in range(N_W):
        if deposition_kinetics and qw < 0.0:
            # вынос: возвращается осадок в своем составе
            s_cap = _RO_P_O * qw * Dep[i, j, k] / dep_w_sum if dep_w_sum > 0.0 else 0.0
        else:
            s_cap = _RO_P_O * qw * Ws[i, j, k] / ws_sum if ws_sum > 0.0 else 0.0
        s_k = s_cap
        if deposition_kinetics:
            s_k += _RO_P_O * (new_kx[i, j].qw * new_kx[i, j].wsh[k]
                              + new_kx[i, j].qg * new_kx[i, j].gsh[k])
        Dep[i, j, k] += dt * ro_o * s_k
        new_Wc[i, j, k] = mso * Wc[i, j, k] - dt * s_k
        if wax_kinetics:  # захват уносит взвесь, вынос ее возвращает
            new_Wc[i, j, IS0 + k] = mso * Wc[i, j, IS0 + k] - dt * s_cap
    s_af = 0.0
    if NC_T > N_W:  # асфальтены и смолы переносятся (без асфальтенов осадка у них нет, qpa = 0)
        qpa = new_kx[i, j].qpa
        if deposition_kinetics and qpa < 0.0:
            dep_a = Dep[i, j, IA_F] + Dep[i, j, I_R]
            f_af = Dep[i, j, IA_F] / dep_a if dep_a > 0.0 else 1.0 - resin_in_deposit
            s_af = _RO_AD_O * f_af * qpa
            s_r = _RO_AD_O * (1.0 - f_af) * qpa
        else:
            s_af = _RO_AD_O * (1.0 - resin_in_deposit) * qpa
            s_r = _RO_AD_O * resin_in_deposit * qpa
        Dep[i, j, IA_F] += dt * ro_o * s_af
        Dep[i, j, I_R] += dt * ro_o * s_r
        new_Wc[i, j, IA_D] = mso * Wc[i, j, IA_D]
        new_Wc[i, j, IA_F] = mso * Wc[i, j, IA_F] - dt * s_af
        new_Wc[i, j, I_R] = mso * Wc[i, j, I_R] - dt * s_r
    if adsorption:
        s_ada = _RO_AD_O * new_kx[i, j].qada
        s_adr = _RO_AD_O * new_kx[i, j].qadr
        new_kx[i, j].ga = kx[i, j].ga + dt * ro_o * s_ada
        new_kx[i, j].gr = kx[i, j].gr + dt * ro_o * s_adr
        new_Wc[i, j, IA_D] -= dt * s_ada
        new_Wc[i, j, I_R] -= dt * s_adr
    if asph_aggregation:
        # Осевшие и вынесенные флокулы уносят (возвращают) число флокул со средней массой флокулы
        n_now = Wc[i, j, IN_F]
        w_now = Wc[i, j, IA_F]
        new_Wc[i, j, IN_F] = mso * n_now - (dt * s_af * n_now / w_now if w_now > 0.0 else 0.0)

    # Перенос: поток через грань и дебит скважины (у добывающей - свой состав, у нагнетательной q_o = 0)
    coef = dt / volume
    q_well = coef * src_Qo[i, j]
    for c in range(NC_T):
        new_Wc[i, j, c] += q_well * Wc[i, j, c]
    if NC_T == 1:  # приток единственного компонента уже собран по граням в `flows_in_cells`
        new_Wc[i, j, 0] += coef * cells_W_eq[i, j]
    for idx in range(4 if NC_T > 1 else 0):
        f = coef * Fo[i, j, idx]
        if f > 0.0:
            i1 = i + DI[idx]
            j1 = j + DJ[idx]
            if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                for c in range(NC_T):
                    new_Wc[i, j, c] += f * Wc[i1, j1, c]
            else:
                bound = get_bound(i1, j1)
                if boundary_conditions[bound, PARAFFIN].type == DIRICHLET:
                    for c in range(NC_T):
                        new_Wc[i, j, c] += f * bc_Wc[bound, c]
                else:
                    for c in range(NC_T):
                        new_Wc[i, j, c] += f * Wc[i, j, c]
        elif f < 0.0:
            for c in range(NC_T):
                new_Wc[i, j, c] += f * Wc[i, j, c]

    if mso_new <= 1e-12:  # ячейка промыта водой: нефтяной фазы нет, переносить нечего
        for c in range(NC_T):
            new_Wc[i, j, c] = 0.0
        for k in range(N_W):
            new_Ws[i, j, k] = 0.0
        new_Wp[i, j] = 0.0
        new_Wps[i, j] = 0.0
        new_Hl[i, j] = 0.0
        return None

    wax_total = 0.0
    for c in range(NC_T):
        v = max(new_Wc[i, j, c] / mso_new, 0.0)
        new_Wc[i, j, c] = min(v, 1.0) if c < IN_F else v  # число флокул - не доля
        if c < N_W:
            wax_total += new_Wc[i, j, c]

    if wax_kinetics:
        # Равновесная взвесь - цель релаксации; фактическая взвесь - переносимое состояние
        sle_split(new_Wc[i, j, :N_W], T[i, j], p[i, j], new_Ws[i, j, :])
        w_dis, w_sus, hl = 0.0, 0.0, 0.0
        for k in range(N_W):
            eq = new_Ws[i, j, k]
            new_kx[i, j].weq[k] = eq
            s_k = min(new_Wc[i, j, IS0 + k], new_Wc[i, j, k])
            s_k = relax_exp(s_k, eq, kin[K_CRYST] if s_k < eq else kin[K_DISS], dt)
            new_Wc[i, j, IS0 + k] = s_k
            new_Ws[i, j, k] = s_k
            dis = new_Wc[i, j, k] - s_k
            w_dis += dis
            w_sus += s_k
            hl += WAX_L_REL[k] * dis
    else:
        # Равновесие групп парафина при температуре текущего слоя (энергия считается после) и новом давлении
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
        af_pre = new_Wc[i, j, IA_F]
        af = min(max(floc_relax(af_pre, af_eq, dt), 0.0), a_tot)
        new_Wc[i, j, IA_F] = af
        new_Wc[i, j, IA_D] = a_tot - af
        if asph_aggregation:
            d0 = kin[AGG_D0]
            m0 = ro_asph * math.pi * d0 * d0 * d0 / 6.0
            n = new_Wc[i, j, IN_F]
            if af > af_pre:
                n += (af - af_pre) / m0              # выпавшее - новые первичные частицы
            elif af_pre > 0.0:
                n *= af / af_pre                     # растворение уменьшает число пропорционально массе
            if n <= 0.0 and af > 0.0:
                n = af / m0
            k_coag = coagulation_kernel(T[i, j] + 273.15, mu_p[i, j], kin[AGG_W])
            n_vol = smoluchowski_step(n * ro_o, k_coag, dt)  # число флокул в 1 м^3 нефти после коагуляции
            new_Wc[i, j, IN_F] = min(n_vol / ro_o, af / m0) if af > 0.0 else 0.0
