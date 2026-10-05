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
                                scn_slope, scn_gamma_alpha, wax_alpha_eff, wax_Tm_shift, wax_dv_frac, ro_wax_liq,
                                sara_aromatics, sara_resins, sara_asphaltenes, P_bubble, Rs_bubble,
                                T_sc_gas, P_sc_gas, v_gas, delta_gas, P_onset_asph, v_asph, ro_asph, asph_curve,
                                delta_sat, delta_aro, delta_res, c_oil_comp, beta_oil, P_ref_wax,
                                wax_kinetics, asph_aggregation)

WAX_TOTAL = init_Wp + init_Wps  # суммарная доля парафина в нефтяной фазе, [-]


def won_tm(M):
    """Температура плавления н-алкана по молярной массе (Won, 1986), [K]."""
    return 374.5 + 0.02617 * M - 20172.0 / M


def won_dh(M):
    """Теплота плавления н-алкана (Won, 1986), [Дж/моль]."""
    return 0.1426 * M * won_tm(M) * kal_to_J


def scn_distribution(slope=scn_slope, n_first=scn_first, n_last=scn_last, total=WAX_TOTAL, alpha=None):
    """Массовые доли н-алканов C_n в нефтяной фазе при мольных долях z_n ~ exp(-slope*n) или, если задана
    форма alpha, по гамма-распределению Уитсона с тем же масштабом (`thermo.characterization`).

    Returns
    -------
    n, M, w: numpy.ndarray
        Число атомов углерода, молярная масса [г/моль], массовая доля в нефтяной фазе [-]
    """
    n = np.arange(n_first, n_last + 1)
    M = 14.027 * n + 2.016
    if alpha is None:
        w = np.exp(-slope * n) * M
    else:
        from paraphin.thermo.characterization import gamma_scn_fractions
        w = gamma_scn_fractions(M, alpha, M[0] - 7.0135, 14.027 / slope) * M
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


def group_properties(slope=scn_slope, alpha_eff=wax_alpha_eff, tm_shift=wax_Tm_shift, bounds=scn_bounds,
                     n_first=scn_first, n_last=scn_last, total=WAX_TOTAL, dv_frac=wax_dv_frac):
    """Свойства групп при заданных параметрах характеризации - для калибровки и рисунков (`experiments/calibrate.py`).

    Returns
    -------
    dict: w, M [г/моль], Tm [K], dH [Дж/моль], dv [м^3/моль], L [Дж/кг]
    """
    n, M, w = scn_distribution(slope, n_first, n_last, total)
    wk, Mk, tmk, lk = lump_groups(n, M, w, bounds, tm_shift)
    return {'w': wk, 'M': Mk, 'Tm': tmk, 'dH': np.full(wk.size, alpha_eff), 'dv': dv_frac * Mk * 1e-3 / ro_wax_liq,
            'L': lk}


def sle_split_np(w, M, tm, dh, dv, t_c, dp=0.0, n_g=0.0, m_o=M_o, x=None):
    """Растворенные доли групп - numpy-эталон njit-ядра `equations/Thermo_wax.sle_split` с параметрами
    аргументами (тот же жадный набор насыщенных групп). dp = P - P_ref, [Па]; n_g - газ, [моль/г];
    x - готовые мольные доли насыщения групп (уравнение состояния), тогда tm, dh, dv, t_c, dp не нужны."""
    if x is None:
        t = t_c + 273.15
        x = np.minimum(1.0, np.exp(-dh / R * (1.0 / t - 1.0 / tm) - dv * dp / (R * t)))
    a = (1.0 - w.sum()) / m_o + n_g + (w / M).sum()
    b = 1.0
    sat = np.zeros(w.size, bool)
    while True:
        cand = (~sat) & (w > 0.0) & (x < 1.0)
        if not cand.any():
            break
        thr = np.where(cand, np.where(x > 0.0, w / (M * np.maximum(x, 1e-300)), np.inf), -np.inf)
        k = int(thr.argmax())
        if thr[k] * b <= a:
            break
        sat[k] = True
        a -= w[k] / M[k]
        b -= x[k]
    return np.where(sat, x * (a / b) * M, w)


def wat_np(w, M, tm, dh, dv, dp=0.0, n_g=0.0, m_o=M_o):
    """Температура начала кристаллизации в замкнутой форме, [C] (эталон `Thermo_wax.wat_cell`)."""
    n_l = (1.0 - w.sum()) / m_o + n_g + (w / M).sum()
    ok = w > 0.0
    t_k = (dh[ok] + dv[ok] * dp) / R / (dh[ok] / (R * tm[ok]) - np.log(w[ok] / M[ok] / n_l))
    return float(t_k.max()) - 273.15


if wax_characterization == 'single':
    WAX_W0 = np.array([WAX_TOTAL], data_type)
    WAX_M = np.array([MW], data_type)
    WAX_TM_K = np.array([Tm_K], data_type)
    WAX_DH = np.array([alpha], data_type)
    WAX_L_REL = np.array([1.0], data_type)
    SCN_N, SCN_M, SCN_W = np.array([0]), np.array([MW]), np.array([WAX_TOTAL])
else:
    SCN_N, SCN_M, SCN_W = scn_distribution(alpha=scn_gamma_alpha if wax_characterization == 'gamma' else None)
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
NCB = N_W + 3  # «массовые» компоненты: по ним сходятся балансы массы, их сумма с остатком - единица
# Дальше - переносимые с нефтью величины, которые компонентами не являются (включаются флагами кинетики):
#   Wc[..., IS0 + k] - взвешенные кристаллы группы k, часть Wc[..., k] (`wax_kinetics`: взвесь не равновесная);
#   Wc[..., IN_F]    - число флокул асфальтенов на килограмм нефти (`asph_aggregation`), [1/кг].
IS0 = NCB
IN_F = IS0 + (N_W if wax_kinetics else 0)
NC = IN_F + (1 if asph_aggregation else 0)

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


_W_SAT0 = REST0 * F_SAT_REST + WAX_TOTAL  # насыщенные исходной нефти: при init_T весь парафин растворен
_W_ARO0 = REST0 * (1.0 - F_SAT_REST)


def _calibrate_delta_a(v_a=v_asph) -> float:
    """Параметр растворимости асфальтенов, при котором в точке (P_onset_asph, init_T) начальная доля
    асфальтенов ровно насыщающая (Hirschberg et al., 1984): ln phi_a0 = v_a/v_m - 1 - v_a*dd^2/(R*T)."""
    if sara_asphaltenes <= 0.0:
        return 21.0
    d_m = delta_maltene_py(_W_SAT0, _W_ARO0, sara_resins, P_onset_asph, init_T)
    rt = R * (init_T + 273.15)
    arg = rt * (v_a / V_M - 1.0 - math.log(asph_volume_fraction(sara_asphaltenes))) / v_a
    return d_m + math.sqrt(arg) * 1e-3  # Па^0.5 -> МПа^0.5


def hirschberg_precipitated(v_a, p_list):
    """Равновесно выпавшие асфальтены исходной нефти при init_T, [доля массы нефти]: w_a0 - w_a^max(p) по (1)
    `equations/Asphaltene.py`, delta_a - по P_onset_asph при этом v_a."""
    d_a = _calibrate_delta_a(v_a)
    rt = R * (init_T + 273.15)
    out = []
    for p in p_list:
        dd = (d_a - delta_maltene_py(_W_SAT0, _W_ARO0, sara_resins, p, init_T)) * 1e3
        phi = min(1.0, math.exp(v_a / V_M - 1.0 - v_a * dd * dd / rt))
        out.append(max(0.0, sara_asphaltenes - phi * ro_asph / (phi * ro_asph + (1.0 - phi) * ro_o)))
    return np.array(out)


def _fit_v_asph() -> float:
    """Мольный объем асфальтенов по кривой выпавших `asph_curve` (МНК по ln v_a; delta_a держит P_onset_asph);
    без кривой - v_asph из констант."""
    if asph_curve is None or sara_asphaltenes <= 0.0:
        return v_asph
    from scipy.optimize import least_squares

    p, w = np.asarray(asph_curve, float).T
    fit = least_squares(lambda x: hirschberg_precipitated(math.exp(x[0]), p) - w, [math.log(v_asph)],
                        bounds=([math.log(0.1e-3)], [math.log(10e-3)]))
    return math.exp(fit.x[0])


V_ASPH = _fit_v_asph()  # мольный объем асфальтенов в (1), [м^3/моль]
DELTA_ASPH = _calibrate_delta_a(V_ASPH)  # параметр растворимости асфальтенов, [МПа^0.5]
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
    print(f'Газ при P_b: {N_GAS_B * 1e3:.3f} моль/кг; delta_a = {DELTA_ASPH:.2f} МПа^0.5, '
          f'v_a = {V_ASPH * 1e3:.3f} л/моль' + (' (по asph_curve)' if asph_curve is not None else ''))
