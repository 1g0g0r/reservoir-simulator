"""Детальный состав нефти: характеризация по SARA и SCN, свойства групп парафинов, асфальтенов и смол.

Считается один раз при импорте (как `eta` в `paraphin/__init__.py`), результат - константы уровня
модуля: njit-функции уравнений читают их как литералы. Работает только при `wax_components = True`,
но считается всегда: массивы маленькие, а импорт от флага не зависит.

Состав нефти (обзор `твт_статья/full_review.pdf`, разд. 1.2-1.3):
    - насыщенные углеводороды: н-алканы C17-C60 (парафины и воски, кристаллизуются) и остальная
      часть (изо- и циклоалканы, растворитель);
    - ароматические - растворитель, повышают параметр растворимости мальтенов;
    - смолы - пептизаторы асфальтенов (Leontaritis & Mansoori, SPE 16258, 1987), соосаждаются с ними;
    - асфальтены - коллоидная фракция (модель Йена-Маллинса: Mullins et al., Energy Fuels 2012,
      26:3986), выпадают при падении параметра растворимости мальтенов (Hirschberg et al., SPEJ 1984).

Н-алканы разбиваются на группы по числу атомов углерода (SCN: Katz & Firoozabadi, JPT 1978;
укрупнение - Whitson, SPEJ 1983). Внутри плюс-фракции мольные доли убывают экспоненциально,
z_n ~ exp(-scn_slope*n) (Pedersen et al., Energy Fuels 1991, 5:924). Свойства групп:
    M_k   - среднечисловая молярная масса, [г/моль];
    Tm_k  - температура плавления по Won (FPE 1986, 30:265): Tm = 374.5 + 0.02617*M - 20172/M, [K],
            осредненная по массе и сдвинутая на `wax_Tm_shift` (эффективный параметр, см. constants.py);
    dH_k  - эффективная теплота в законе растворимости `wax_alpha_eff`, одна на все группы, [Дж/моль];
    L_k   - калориметрическая удельная теплота плавления по Won: dH = 0.1426*M*Tm, [кал/моль] - она идет в
            уравнение энергии (сверка с измерениями: Broadhurst, J. Res. NBS 1962, 66A:241);
    dv_k  - скачок мольного объема при плавлении wax_dv_frac*M/ro_wax_liq (поправка Пойнтинга), [м^3/моль].

Режим 'single' - одна группа ровно с параметрами прежней модели (MW, Tm, alpha, latent_heat): тогда
многокомпонентная машинерия воспроизводит прежнюю термодинамику и проверяет перенос, стоки и
скрытую теплоту независимо от характеризации.

Запуск `python -m paraphin.oil_composition` печатает таблицу групп.
"""
import math

import numpy as np

from paraphin.constants import (data_type, init_Wp, init_Wps, init_T, MW, M_o, Tm_K, alpha, latent_heat,
                                kal_to_J, R, ro_o, wax_characterization, scn_first, scn_last, scn_bounds,
                                scn_slope, wax_alpha_eff, wax_Tm_shift, wax_dv_frac, ro_wax_liq,
                                sara_aromatics, sara_resins, sara_asphaltenes, P_bubble, Rs_bubble,
                                T_sc_gas, P_sc_gas, v_gas, delta_gas, P_onset_asph, v_asph, ro_asph,
                                delta_sat, delta_aro, delta_res, c_oil_comp, beta_oil, P_ref_wax)

WAX_TOTAL = init_Wp + init_Wps  # суммарная доля парафина в нефтяной фазе, [-]


def won_tm(M):
    """Температура плавления н-алкана по молярной массе (Won, 1986), [K]."""
    return 374.5 + 0.02617 * M - 20172.0 / M


def won_dh(M):
    """Теплота плавления н-алкана (Won, 1986), [Дж/моль]."""
    return 0.1426 * M * won_tm(M) * kal_to_J


def scn_distribution(slope=scn_slope, n_first=scn_first, n_last=scn_last, total=WAX_TOTAL):
    """Массовые доли н-алканов C_n в нефтяной фазе при мольных долях z_n ~ exp(-slope*n).

    Returns
    -------
    n, M, w: numpy.ndarray
        Число атомов углерода, молярная масса [г/моль], массовая доля в нефтяной фазе [-]
    """
    n = np.arange(n_first, n_last + 1)
    M = 14.027 * n + 2.016
    w = np.exp(-slope * n) * M
    w *= total / w.sum()
    return n, M, w


def lump_groups(n, M, w, bounds=scn_bounds, tm_shift=wax_Tm_shift):
    """Укрупнение SCN в группы: доля, среднечисловая M, массово осредненная Tm Won (+ сдвиг) и
    удельная калориметрическая теплота плавления группы.

    Returns
    -------
    w_k, M_k, Tm_k, L_k: numpy.ndarray
        Доля группы [-], молярная масса [г/моль], эффективная Tm [K], теплота плавления [Дж/кг]
    """
    edges = [n[0]] + [b + 1 for b in bounds] + [n[-1] + 1]
    w_k, M_k, Tm_k, L_k = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (n >= lo) & (n < hi)
        wk = w[s].sum()
        w_k.append(wk)
        M_k.append(wk / (w[s] / M[s]).sum())
        Tm_k.append((w[s] * won_tm(M[s])).sum() / wk + tm_shift)
        L_k.append((w[s] * won_dh(M[s]) / (M[s] * 1e-3)).sum() / wk)
    return np.array(w_k), np.array(M_k), np.array(Tm_k), np.array(L_k)


if wax_characterization == 'single':
    WAX_W0 = np.array([WAX_TOTAL], data_type)
    WAX_M = np.array([MW], data_type)
    WAX_TM_K = np.array([Tm_K], data_type)
    WAX_DH = np.array([alpha], data_type)
    WAX_L_REL = np.array([1.0], data_type)
    SCN_N, SCN_M, SCN_W = np.array([0]), np.array([MW]), np.array([WAX_TOTAL])
else:
    SCN_N, SCN_M, SCN_W = scn_distribution()
    _w, _M, _Tm, _L = lump_groups(SCN_N, SCN_M, SCN_W)
    WAX_W0 = _w.astype(data_type)
    WAX_M = _M.astype(data_type)
    WAX_TM_K = _Tm.astype(data_type)
    WAX_DH = np.full(_w.size, wax_alpha_eff, data_type)
    WAX_L_REL = (_L / latent_heat).astype(data_type)  # удельная теплота группы в долях прежней latent_heat

N_W = int(WAX_W0.size)          # число групп парафина
WAX_DV = (wax_dv_frac * WAX_M * 1e-3 / ro_wax_liq).astype(data_type)  # dv = v_L - v_S, [м^3/моль]
WAX_DH_R = (WAX_DH / R).astype(data_type)   # dH/R, [K]
WAX_DV_R = (WAX_DV / R).astype(data_type)   # dv/R, [м^3*K/Дж]

# Индексы компонентов в массиве Wc (Nx, Ny, NC): группы парафина, асфальтены растворенные и флокулы, смолы
IA_D = N_W
IA_F = N_W + 1
I_R = N_W + 2
NC = N_W + 3

# --- Нефть без парафина: растворитель, SARA ----------------------------------------------------------------
SAT0 = 1.0 - sara_aromatics - sara_resins - sara_asphaltenes  # насыщенные вместе с парафином
REST0 = 1.0 - WAX_TOTAL - sara_asphaltenes - sara_resins       # «остаток»: ненормальные насыщенные + ароматика
if SAT0 < WAX_TOTAL - 1e-12:
    raise ValueError(f'SARA: насыщенных {SAT0:.4f} меньше, чем парафина {WAX_TOTAL:.4f}')
# Доля насыщенных в остатке неизменна: остаток переносится как одно целое
F_SAT_REST = (SAT0 - WAX_TOTAL) / REST0 if REST0 > 0.0 else 0.0

# Растворенный газ: мольное содержание в граммах дегазированной нефти при P_b, [моль/г]
N_GAS_B = Rs_bubble * P_sc_gas / (R * T_sc_gas) / (ro_o * 1e3)
V_M = M_o * 1e-3 / ro_o  # мольный объем мальтенов, [м^3/моль]
V_LIQ = 1.0 / (ro_o * 1e3)  # объем грамма жидкой нефти, [м^3/г]


def gas_moles(p):
    """Растворенный газ на грамм дегазированной нефти, [моль/г]: линейно по давлению до P_b."""
    return N_GAS_B * min(max(p, 0.0) / P_bubble, 1.0)


def delta_maltene_py(w_sat, w_aro, w_res, p, t):
    """Параметр растворимости мальтенов (все, кроме асфальтенов и кристаллов), [МПа^0.5].

    Жидкие псевдокомпоненты смешиваются по массовым долям (плотности отличаются меньше чем на 15 %),
    результат масштабируется плотностью (правило «одной трети»: параметр растворимости пропорционален
    плотности, Buckley et al., Pet. Sci. Technol. 1998, 16:251; Wang & Buckley, Energy Fuels 2001,
    15:1004), затем смешивается с растворенным газом по объемным долям. Та же формула в njit -
    `equations/Asphaltene.delta_maltene`.
    """
    w_liq = w_sat + w_aro + w_res
    d_liq = (w_sat * delta_sat + w_aro * delta_aro + w_res * delta_res) / w_liq
    d_liq *= 1.0 + c_oil_comp * (p - P_ref_wax) - beta_oil * (t - 25.0)
    vg = gas_moles(p) * v_gas
    vl = w_liq * V_LIQ
    phi_g = vg / (vg + vl)
    return (1.0 - phi_g) * d_liq + phi_g * delta_gas


def asph_volume_fraction(w_a):
    """Объемная доля асфальтенов в нефтяной фазе по массовой."""
    va = w_a / ro_asph
    return va / (va + (1.0 - w_a) / ro_o)


def _calibrate_delta_a() -> float:
    """Параметр растворимости асфальтенов, при котором в точке (P_onset_asph, init_T) начальная доля
    асфальтенов ровно насыщающая (Hirschberg et al., 1984): ln phi_a0 = v_a/v_m - 1 - v_a*dd^2/(R*T)."""
    if sara_asphaltenes <= 0.0:
        return 21.0
    w_sat = REST0 * F_SAT_REST + WAX_TOTAL  # при пластовой температуре весь парафин растворен
    w_aro = REST0 * (1.0 - F_SAT_REST)
    d_m = delta_maltene_py(w_sat, w_aro, sara_resins, P_onset_asph, init_T)
    rt = R * (init_T + 273.15)
    arg = rt * (v_asph / V_M - 1.0 - math.log(asph_volume_fraction(sara_asphaltenes))) / v_asph
    return d_m + math.sqrt(arg) * 1e-3  # Па^0.5 -> МПа^0.5


DELTA_ASPH = _calibrate_delta_a()  # параметр растворимости асфальтенов, [МПа^0.5]
# Индекс коллоидной неустойчивости начальной нефти: < 0.7 - асфальтены устойчивы, > 0.9 - неустойчивы
# (Yen, Yin & Asomaning, SPE 65376, 2001)
CII0 = (SAT0 + sara_asphaltenes) / (sara_aromatics + sara_resins)


def initial_components() -> np.ndarray:
    """Начальный состав нефтяной фазы по компонентам Wc: группы парафина (суммарно init_Wp + init_Wps),
    асфальтены (все растворены - равновесие ставит `Solver.initialize`), флокулы, смолы."""
    wc = np.zeros(NC, data_type)
    wc[:N_W] = WAX_W0
    wc[IA_D] = sara_asphaltenes
    wc[I_R] = sara_resins
    return wc


if __name__ == '__main__':
    print(f"Характеризация '{wax_characterization}': {N_W} групп, парафина {WAX_TOTAL:.4f}, "
          f'SARA: насыщенные {SAT0:.4f} (из них остаток {REST0 * F_SAT_REST:.4f}), ароматика {sara_aromatics}, '
          f'смолы {sara_resins}, асфальтены {sara_asphaltenes}; CII = {CII0:.2f}')
    print('| группа | w, масс. | M, г/моль | Tm эфф., C | Tm Won, C | L, кДж/кг | dv, см^3/моль |')
    for k in range(N_W):
        print(f'| {k + 1} | {WAX_W0[k]:.4f} | {WAX_M[k]:.1f} | {WAX_TM_K[k] - 273.15:.1f} | '
              f'{won_tm(WAX_M[k]) - 273.15:.1f} | {WAX_L_REL[k] * latent_heat / 1e3:.1f} | {WAX_DV[k] * 1e6:.2f} |')
    print(f'Газ при P_b: {N_GAS_B * 1e3:.3f} моль/кг; delta_a = {DELTA_ASPH:.2f} МПа^0.5')
