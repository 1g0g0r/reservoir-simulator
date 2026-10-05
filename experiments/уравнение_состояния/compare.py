"""Модели на уравнении состояния (`paraphin/thermo`) против опытов и моделей других авторов.

    python experiments/уравнение_состояния/compare.py [--plot]  ->  results.json, figures/*.png, сравнение_EOS.md

A. Синтетические смеси н-алканов C18-C36 в н-декане (da Silva et al., 2017; `experiments/data/dasilva2017_sle.json`):
   WDT и кривые выпадения. Проверяется сама термодинамика - PR + multi-solid (плавление по Вону и по Coutinho с
   твердо-твердым переходом), идеальные multi-solid и твердый раствор, твердый раствор UNIQUAC (Coutinho) - без
   характеризации нефти, и сравнивается с теми же моделями у авторов.
B. Нефть Жетыбая (Li et al., 2024): кривая выпадения и WAT по ДСК. Текущая модель (идеальный раствор с
   эффективными dH, сдвигом Tm) против моделей с калориметрическими свойствами без подбора и UNIQUAC с подобранной
   характеризацией (наклон SCN-распределения и последний SCN).
C. WAT от давления: дегазированная нефть (Sandyga et al., 2020: 0.15-0.23 C/МПа) и живая (Pan et al., 1997).
D. Асфальтены: Hirschberg (Флори-Хаггинс) против модели твердой фазы Nghiem при тех же P_onset и P_b.
E. Асфальтены Nghiem на кривой выпавших (Tabzar et al., 2018; `experiments/data/tabzar2018_asphaltene.json`):
   калибровка по одной точке начала осаждения против подбора по всей кривой (`asph_curve`).
F. PVT нефти модели по уравнению состояния (`paraphin/thermo/pvt.py`) против линейного R_s симулятора.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'исходная_модель'))

import core_flood as cf  # noqa: E402
from paraphin import oil_composition as oc  # noqa: E402
from paraphin.constants import (P_ref_wax, P_bubble, P_onset_asph, init_T, R, ro_o, ro_asph, v_asph,  # noqa: E402
                                sara_asphaltenes, sara_resins, scn_first, Rs_bubble, ro_wax_liq, ro_p)
from paraphin.thermo import tables as tb  # noqa: E402
from paraphin.thermo.characterization import (coutinho_nalkane, critical_props, nalkane_sg,  # noqa: E402
                                              nalkane_tb)
from paraphin.thermo.eos import PengRobinson, bubble_pressure, flash  # noqa: E402
from paraphin.thermo.pvt import pvt_point  # noqa: E402
from paraphin.thermo.solids import (IdealSolidSolutionWax, MultiSolidWax, NghiemAsphaltene,  # noqa: E402
                                    UniquacSolidWax)

DATA = json.loads((ROOT / 'experiments' / 'data' / 'dasilva2017_sle.json').read_text(encoding='utf-8'))
TABZAR = json.loads((ROOT / 'experiments' / 'data' / 'tabzar2018_asphaltene.json').read_text(encoding='utf-8'))
FIGURES = HERE / 'figures'
P_ATM = 101325.0
PSI = 6894.757
# Цвет и стиль закреплены за моделью во всех рисунках (палитра рисунков experiments/)
MODELS = {'eff': ('текущая: идеальный раствор, эфф. dH и Tm', '-', '#2a78d6'),
          'pr': ('PR + multi-solid, Tm и dH Вона', '--', '#eb6834'),
          'ideal': ('идеальный multi-solid, Tm и dH Вона', ':', '#1baf7a'),
          'ss': ('идеальный твердый раствор, Tm и dH Вона', '-.', '#e87ba4'),
          'ms_c': ('PR + multi-solid, Coutinho (с переходом)', (0, (5, 1, 1, 1)), '#8a5cc2'),
          'uq': ('PR + UNIQUAC, Coutinho', '-', '#1a1a1a')}


def _rms(model, exp):
    return float(np.sqrt(np.mean((np.asarray(model) - np.asarray(exp)) ** 2)))


# --- A. Синтетические смеси (da Silva et al., 2017) ----------------------------------------------------------
def synthetic(t_curve):
    out = {}
    for name, mix in DATA['mixtures'].items():
        n = np.array(sorted(int(k) for k in mix['wt']))
        w = np.array([mix['wt'][str(k)] for k in n]) / 100.0
        M = 14.027 * n + 2.016
        eos = PengRobinson(*critical_props(nalkane_tb(M), nalkane_sg(M)))
        z = (w / M) / np.sum(w / M)
        f = n != DATA['solvent']
        tm, dh = oc.won_tm(M), oc.won_dh(M)
        ms = MultiSolidWax(eos, np.where(f)[0], tm[f], dh[f])
        ss = IdealSolidSolutionWax(f, tm, dh)
        t_m, dh_m, t_tr, dh_tr = coutinho_nalkane(n)
        ms_c = MultiSolidWax(eos, np.where(f)[0], t_m[f], dh_m[f], t_tr=t_tr[f], dh_tr=dh_tr[f])
        ms_cp = MultiSolidWax(eos, np.where(f)[0], t_m[f], dh_m[f], t_tr=t_tr[f], dh_tr=dh_tr[f], mw=M[f])  # + dCp
        uq, uq_id = UniquacSolidWax(ms_c, n[f]), UniquacSolidWax(ms_c, n[f], liquid='ideal')
        m_solv = float(M[~f][0])

        def solid_pct(s):
            return 100.0 * np.sum(s * M) / np.sum(z * M)

        def curves(t_c):
            t = t_c + 273.15
            return {'pr': solid_pct(ms.precipitate(t, P_ATM, z)[0]),
                    'ss': solid_pct(ss.precipitate(t, z)[0]),
                    'ideal': 100.0 * (w[f].sum() - oc.sle_split_np(w[f], M[f], tm[f], dh[f], 0.0, t_c,
                                                                   m_o=m_solv).sum()),
                    'ms_c': solid_pct(ms_c.precipitate(t, P_ATM, z)[0]),
                    'ms_cp': solid_pct(ms_cp.precipitate(t, P_ATM, z)[0]),
                    'uq': solid_pct(uq.precipitate(t, P_ATM, z)[0]),
                    'uq_id': solid_pct(uq_id.precipitate(t, P_ATM, z)[0])}

        t_exp, s_exp = np.array(mix['solid_wt']).T
        at_exp = [curves(t) for t in t_exp]
        grid = [curves(t) for t in t_curve]
        keys = ('pr', 'ss', 'ideal', 'ms_c', 'ms_cp', 'uq', 'uq_id')
        out[name] = {
            'wdt': {'pr': ms.wat(P_ATM, z) - 273.15, 'ss': ss.wat(z) - 273.15,
                    'ideal': oc.wat_np(w[f], M[f], tm[f], dh[f], 0.0 * M[f], m_o=m_solv),
                    'ms_c': ms_c.wat(P_ATM, z) - 273.15, 'ms_cp': ms_cp.wat(P_ATM, z) - 273.15,
                    'uq': uq.wat(P_ATM, z) - 273.15,
                    'uq_id': uq_id.wat(P_ATM, z) - 273.15},
            'rms': {k: _rms([c[k] for c in at_exp], s_exp) for k in keys},
            'aad': {k: float(np.mean(np.abs(np.array([c[k] for c in at_exp]) - s_exp))) for k in keys},
            'curves': {k: [c[k] for c in grid] for k in keys},
        }
    return out


# --- B. Нефть Жетыбая (Li et al., 2024) ----------------------------------------------------------------------
def zhetybai(t_curve):
    w0 = oc.WAX_W0
    eos, n, mw, _, iw, _ = tb.oil_fluid(0.0, gas=False)
    z = n / n.sum()
    g = np.arange(iw, iw + oc.N_W)
    is_f = np.zeros(eos.nc, bool)
    is_f[g] = True
    tm_all, dh_all = np.full(eos.nc, np.nan), np.full(eos.nc, np.nan)
    tm_all[g], dh_all[g] = tb.WAX_TM_WON, tb.WAX_DH_WON
    ms = MultiSolidWax(eos, g, tb.WAX_TM_WON, tb.WAX_DH_WON, tb.WAX_DV_EOS, P_ref_wax)
    ss = IdealSolidSolutionWax(is_f, tm_all, dh_all)
    ms_c, uq = _coutinho_models(eos, g, oc.WAX_M, tb.WAX_DV_EOS)

    def wax_pct(model, t):
        return 100.0 * np.sum(model.precipitate(t, P_ref_wax, z)[0] * mw) * n.sum()

    def curves(t_c):
        t = t_c + 273.15
        return {'eff': 100.0 * (w0.sum() - oc.sle_split_np(w0, oc.WAX_M, oc.WAX_TM_K, oc.WAX_DH, oc.WAX_DV, t_c).sum()),
                'ideal': 100.0 * (w0.sum() - oc.sle_split_np(w0, oc.WAX_M, tb.WAX_TM_WON, tb.WAX_DH_WON, 0.0, t_c).sum()),
                'pr': wax_pct(ms, t), 'ss': 100.0 * np.sum(ss.precipitate(t, z)[0] * mw) * n.sum(),
                'ms_c': wax_pct(ms_c, t), 'uq': wax_pct(uq, t)}

    exp = np.array(cf.LI_PRECIPITATION)
    fit = exp[:, 0] >= cf.LI_FIT_T_MIN
    at_exp = [curves(t) for t in exp[:, 0]]
    grid = [curves(t) for t in t_curve]
    keys = ('eff', 'ideal', 'pr', 'ss', 'ms_c', 'uq')
    return {
        'wat': {'eff': oc.wat_np(w0, oc.WAX_M, oc.WAX_TM_K, oc.WAX_DH, oc.WAX_DV),
                'ideal': oc.wat_np(w0, oc.WAX_M, tb.WAX_TM_WON, tb.WAX_DH_WON, 0.0 * w0),
                'pr': ms.wat(P_ref_wax, z) - 273.15, 'ss': ss.wat(z) - 273.15,
                'ms_c': ms_c.wat(P_ref_wax, z) - 273.15, 'uq': uq.wat(P_ref_wax, z) - 273.15},
        'rms_fit': {k: _rms([c[k] for c, ok in zip(at_exp, fit) if ok], exp[fit, 1]) for k in keys},
        'rms_all': {k: _rms([c[k] for c in at_exp], exp[:, 1]) for k in keys},
        'curves': {k: [c[k] for c in grid] for k in keys},
        'uq_fit': zhetybai_uniquac_fit(exp[fit]),
    }


def _coutinho_models(eos, g, m_wax, dv):
    """PR + multi-solid и PR + UNIQUAC с плавлением групп по Coutinho (n по средней молярной массе группы)."""
    nc = (np.asarray(m_wax) - 2.016) / 14.027
    t_m, dh_m, t_tr, dh_tr = coutinho_nalkane(nc)
    ms_c = MultiSolidWax(eos, g, t_m, dh_m, dv, P_ref_wax, t_tr, dh_tr)
    return ms_c, UniquacSolidWax(ms_c, nc)


def zhetybai_uniquac_fit(exp):
    """UNIQUAC с подобранной характеризацией: наклон SCN-распределения и последний SCN (сетка) по кривой Li et al.;
    плавление и неидеальность - без подбора. Сколько дает одна характеризация без эффективных параметров."""
    best = None
    for s in np.linspace(0.02, 0.2, 37):
        for n_last in range(42, 61, 2):
            nn, M, w = oc.scn_distribution(s, scn_first, n_last)
            wk, mk = oc.lump_groups(nn, M, w)[:2]
            eos, n, mw, _, iw, _ = tb.oil_fluid(0.0, gas=False, w_wax=wk, m_wax=mk)
            uq = _coutinho_models(eos, np.arange(iw, iw + wk.size), mk, mk * 1e-3 * (1.0 / ro_wax_liq - 1.0 / ro_p))[1]
            z = n / n.sum()
            c = [100.0 * np.sum(uq.precipitate(t + 273.15, P_ref_wax, z)[0] * mw) * n.sum() for t in exp[:, 0]]
            r = _rms(c, exp[:, 1])
            if best is None or r < best['rms']:
                best = {'rms': r, 'scn_slope': float(s), 'scn_last': n_last, 'wat': uq.wat(P_ref_wax, z) - 273.15}
    return best


# --- C. WAT от давления --------------------------------------------------------------------------------------
def wat_pressure(p_dead, p_live):
    w0 = oc.WAX_W0
    eos, n, _, _, iw, _ = tb.oil_fluid(0.0, gas=False)
    ms = MultiSolidWax(eos, np.arange(iw, iw + oc.N_W), tb.WAX_TM_WON, tb.WAX_DH_WON, tb.WAX_DV_EOS, P_ref_wax)
    z = n / n.sum()
    dead_pr = [ms.wat(p, z) - 273.15 for p in p_dead]
    # без скачка объема (вариант пакета research): твердое с объемом жидкости
    ms0 = MultiSolidWax(eos, np.arange(iw, iw + oc.N_W), tb.WAX_TM_WON, tb.WAX_DH_WON)
    dead_pr0 = [ms0.wat(p, z) - 273.15 for p in p_dead]
    dead_eff = [oc.wat_np(w0, oc.WAX_M, oc.WAX_TM_K, oc.WAX_DH, oc.WAX_DV, p - P_ref_wax) for p in p_dead]

    kij = tb.calibrate_kij_gas()
    eos, n, _, ig, iw, _ = tb.oil_fluid(kij, gas=True)
    g = np.arange(iw, iw + oc.N_W)
    ms = MultiSolidWax(eos, g, tb.WAX_TM_WON, tb.WAX_DH_WON, tb.WAX_DV_EOS, P_ref_wax)

    def supersat(t, p):  # самая пересыщенная группа жидкости живой нефти при всем растворенном парафине
        x = tb.live_liquid(eos, n, ig, t, p)[0]
        return float(np.max((ms.liquid_ln_fugacity(t, p, x) - ms.solid_ln_fugacity(t, p))[g]))

    live_pr = [brentq(supersat, 250.0, 450.0, args=(p,), xtol=1e-4) - 273.15 for p in p_live]
    live_eff = [oc.wat_np(w0, oc.WAX_M, oc.WAX_TM_K, oc.WAX_DH, oc.WAX_DV, p - P_ref_wax, oc.gas_moles(p))
                for p in p_live]
    sel = np.asarray(p_dead) <= 20e6
    slope = {k: float(np.polyfit(np.asarray(p_dead)[sel] / 1e6, np.asarray(v)[sel], 1)[0])
             for k, v in (('pr', dead_pr), ('pr_no_dv', dead_pr0), ('eff', dead_eff))}
    return {'kij_gas': kij, 'dead': {'pr': dead_pr, 'eff': dead_eff}, 'slope_dead': slope,
            'live': {'pr': live_pr, 'eff': live_eff}}


# --- D. Асфальтены -------------------------------------------------------------------------------------------
def asphaltenes(p_grid):
    w_sat = oc.REST0 * oc.F_SAT_REST + oc.WAX_TOTAL  # при пластовой температуре парафин растворен
    w_aro = oc.REST0 * (1.0 - oc.F_SAT_REST)
    rt = R * (init_T + 273.15)

    def hirschberg(p):
        dd = (oc.DELTA_ASPH - oc.delta_maltene_py(w_sat, w_aro, sara_resins, p, init_T)) * 1e3
        phi = min(1.0, np.exp(v_asph / oc.V_M - 1.0 - v_asph * dd * dd / rt))
        return phi * ro_asph / (phi * ro_asph + (1.0 - phi) * ro_o)

    def nghiem():
        return np.exp(tb.build_tables(np.array([float(init_T)]), np.asarray(p_grid), wax=False, asph=True,
                                      gas=True)[2][0])

    w_h = np.array([hirschberg(p) for p in p_grid])
    w_n = nghiem()
    kij, tb.eos_kij_asph_gas = tb.eos_kij_asph_gas, 0.0  # без газа-осадителя: колокола нет
    try:
        w_n0 = nghiem()
    finally:
        tb.eos_kij_asph_gas = kij
    w_all = {'hirschberg': w_h, 'nghiem': w_n, 'nghiem_kij0': w_n0}
    return {'w_max': {k: w.tolist() for k, w in w_all.items()},
            'precipitated': {k: np.clip(1.0 - w / sara_asphaltenes, 0.0, 1.0).tolist() for k, w in w_all.items()}}


# --- E. Асфальтены Nghiem на кривой выпавших (Tabzar et al., 2018) -------------------------------------------
def tabzar(p_curve_psig):
    """Флюид Tabzar (16 компонентов, Table 3), 212 F. Во всех вариантах f_s* - по P_onset = 5000 psia, kij метан-C7+
    - по давлению насыщения 2050 psia. Варианты: одна точка с параметрами авторов (kij асфальтены-легкие 0.4,
    V_s = 0.8 л/моль); V_s по кривой при kij 0.4; kij и V_s по кривой (сетка по kij с шагом 0.005, V_s - МНК)."""
    d = TABZAR
    t = (d['T_F'] - 32.0) / 1.8 + 273.15
    c = d['components']
    z = np.array([x['mol'] for x in c])
    z /= z.sum()
    mw = np.array([x['MW'] for x in c])
    tc, pc, om = (np.array([x[k] for x in c]) for k in ('Tc', 'Pc_atm', 'omega'))
    light = np.array([x['light'] for x in c])
    ia = next(i for i, x in enumerate(c) if x.get('asphaltene'))
    heavy = (mw > 100.0) & (np.arange(len(c)) != ia)
    ic1 = [x['name'] for x in c].index('C1')
    p_b, p_on = d['P_bubble_psia'] * PSI, d['P_onset_psia'] * PSI
    p_exp, w_exp = np.array(d['precipitated']).T

    def eos_of(k, kh):
        kij = np.zeros((len(c), len(c)))
        kij[ia, light] = kij[light, ia] = k
        kij[ic1, heavy] = kij[heavy, ic1] = kh
        return PengRobinson(tc, pc * P_ATM, om, kij)

    def kh_of(k):
        return brentq(lambda kh: np.log(bubble_pressure(eos_of(k, kh), t, z) / p_b), -0.4, 0.2, xtol=1e-6)

    def v_bar(eos):  # парциальный мольный объем асфальтенов при P_onset, [л/моль]
        h = 1e-3 * p_on
        return 1e3 * (R * t * (eos.ln_phi(t, p_on + h, z, 'liquid')[ia] - eos.ln_phi(t, p_on - h, z, 'liquid')[ia])
                      / (2.0 * h) + R * t / p_on)

    def curve(k, v_s, p_psig, kh):  # выпавшие, % масс. живой нефти; v_s в л/моль
        eos = eos_of(k, kh)
        m = NghiemAsphaltene(eos, ia, v_s * 1e-3)
        m.calibrate_onset(t, p_on, z)
        out = []
        for p in np.asarray(p_psig, float) * PSI + P_ATM:
            beta, x, _ = flash(eos, t, p, z, vapor_only=True)
            if not 0.0 < beta < 1.0:
                beta, x = 0.0, z
            out.append(100.0 * m.precipitate(t, p, x) * (1.0 - beta) * mw[ia] / np.sum(z * mw))
        return np.array(out)

    def fit_vs(k):
        kh = kh_of(k)
        vb = v_bar(eos_of(k, kh))
        f = least_squares(lambda q: curve(k, q[0] * vb, p_exp, kh) - w_exp, [1.01], bounds=([0.9], [1.2]),
                          diff_step=1e-5)
        return {'kij': float(k), 'kh': kh, 'v_bar': vb, 'V_s': float(f.x[0] * vb), 'rms': _rms(f.fun, 0.0)}

    k_a, vs_a = d['kij_asph_light_authors'], d['V_s_authors_l_mol']
    kh_a = kh_of(k_a)
    onset = {'kij': k_a, 'kh': kh_a, 'V_s': vs_a, 'v_bar': v_bar(eos_of(k_a, kh_a))}
    onset['rms'] = _rms(curve(k_a, vs_a, p_exp, kh_a), w_exp)
    scan = [fit_vs(k) for k in np.arange(0.0, 0.5001, 0.005)]
    variants = {'onset': onset, 'vs': fit_vs(k_a), 'curve': min(scan, key=lambda v: v['rms'])}
    for v in variants.values():
        v['at_exp'] = curve(v['kij'], v['V_s'], p_exp, v['kh']).tolist()
        v['curve'] = curve(v['kij'], v['V_s'], p_curve_psig, v['kh']).tolist()
    return {'variants': variants, 'scan': [{k: s[k] for k in ('kij', 'V_s', 'rms')} for s in scan],
            'p_b_pr_kij0': bubble_pressure(eos_of(0.0, 0.0), t, z) / PSI,
            'asph_wt': 100.0 * z[ia] * mw[ia] / np.sum(z * mw)}


# --- F. PVT нефти модели ------------------------------------------------------------------------------------
def pvt(p_grid):
    kij = tb.calibrate_kij_gas()
    eos, n, mw, ig = tb.oil_fluid(kij, gas=True)[:4]
    rows = [pvt_point(eos, n, mw, ig, init_T + 273.15, p) for p in p_grid]
    out = {k: [float(r[k]) for r in rows] for k in rows[0]}
    out['Rs_linear'] = [Rs_bubble * min(p / P_bubble, 1.0) for p in p_grid]
    out['kij_gas'] = kij
    return out


def main():
    t_syn = np.arange(-10.0, 45.01, 1.0)
    t_li = np.arange(-20.0, 75.01, 1.0)
    p_dead = [0.1e6, 5e6, 10e6, 20e6, 30e6]
    p_live = [0.1e6, 2e6, 4e6, 6e6, 8e6, P_bubble, 12e6, 15e6, 20e6, 30e6]
    p_asph = np.linspace(1e6, 25e6, 49)
    p_tz = np.linspace(0.0, 5000.0, 101)
    p_pvt = [0.5e6, 2e6, 4e6, 6e6, 8e6, P_bubble, 12e6, 20e6, 30e6]
    import time
    res = {'t_syn': t_syn.tolist(), 't_li': t_li.tolist(), 'p_dead': p_dead, 'p_live': p_live,
           'p_asph': p_asph.tolist(), 'P_onset_asph': P_onset_asph, 'P_bubble': P_bubble,
           'p_tabzar_psig': p_tz.tolist(), 'p_pvt': p_pvt}
    for key, job in (('synthetic', lambda: synthetic(t_syn)), ('zhetybai', lambda: zhetybai(t_li)),
                     ('pressure', lambda: wat_pressure(p_dead, p_live)), ('asph', lambda: asphaltenes(p_asph)),
                     ('tabzar', lambda: tabzar(p_tz)), ('pvt', lambda: pvt(p_pvt))):
        t0 = time.perf_counter()
        res[key] = job()
        print(f'{key}: {time.perf_counter() - t0:.0f} с', flush=True)
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


def _line(ax, x, y, key, **kw):
    label, ls, color = MODELS[key]
    ax.plot(x, y, ls=ls, color=color, lw=1.8, label=kw.pop('label', label), **kw)


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0), sharey=True)
    for ax, (name, m) in zip(axes, r['synthetic'].items()):
        t, s = np.array(DATA['mixtures'][name]['solid_wt']).T
        ax.plot(t, s, 'o', mfc='white', mec='k', ms=4, label='опыт')
        for k in ('pr', 'ss', 'ms_c', 'uq'):
            _line(ax, r['t_syn'], m['curves'][k], k)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('T, °C')
    axes[0].set_ylabel('твердое, % масс.')
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / 'synthetic.png', dpi=200)
    plt.close(fig)

    z = r['zhetybai']
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    exp = np.array(cf.LI_PRECIPITATION)
    ax.plot(exp[:, 0], exp[:, 1], 'o', mfc='white', mec='k', ms=4, label='опыт (ДСК, Li et al., 2024)')
    for k in ('eff', 'pr', 'ss', 'ms_c', 'uq'):
        _line(ax, r['t_li'], z['curves'][k], k)
    ax.set_xlabel('T, °C')
    ax.set_ylabel('выпавший парафин, % масс. нефти')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'zhetybai.png', dpi=200)
    plt.close(fig)

    pr = r['pressure']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    p = np.array(r['p_dead']) / 1e6
    a1.fill_between(p, 0.15 * (p - p[0]), 0.23 * (p - p[0]), color='0.85', label='Sandyga et al. (2020): 0.15–0.23 °C/МПа')
    for k in ('eff', 'pr'):
        _line(a1, p, np.array(pr['dead'][k]) - pr['dead'][k][0], k)
    a1.set_title('дегазированная нефть', fontsize=9)
    p = np.array(r['p_live']) / 1e6
    for k in ('eff', 'pr'):
        _line(a2, p, np.array(pr['live'][k]) - pr['live'][k][0], k, marker='o', ms=3)
    a2.axvline(r['P_bubble'] / 1e6, color='0.5', lw=0.8, ls='--')
    a2.text(r['P_bubble'] / 1e6, a2.get_ylim()[1] * 0.9, ' P_b', fontsize=8, color='0.3')
    a2.set_title('живая нефть (газ из flash / линейный Rs)', fontsize=9)
    for ax in (a1, a2):
        ax.set_xlabel('P, МПа')
    a1.set_ylabel('WAT(P) − WAT(0.1 МПа), °C')
    a1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'pressure.png', dpi=200)
    plt.close(fig)

    a = r['asph']
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    p = np.array(r['p_asph']) / 1e6
    _line(ax, p, 100.0 * np.array(a['precipitated']['hirschberg']), 'eff', label='Hirschberg (текущая)')
    _line(ax, p, 100.0 * np.array(a['precipitated']['nghiem']), 'pr', label='Nghiem на PR, kij асфальтены–газ 0.4')
    ax.plot(p, 100.0 * np.array(a['precipitated']['nghiem_kij0']), ls=':', color=MODELS['pr'][2], lw=1.2,
            label='Nghiem на PR, kij = 0')
    for x, name in ((r['P_bubble'], 'P_b'), (r['P_onset_asph'], 'P_onset')):
        ax.axvline(x / 1e6, color='0.5', lw=0.8, ls='--')
        ax.text(x / 1e6, ax.get_ylim()[1] * 0.9, ' ' + name, fontsize=8, color='0.3')
    ax.set_xlabel('P, МПа')
    ax.set_ylabel('выпало асфальтенов, %')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'asphaltenes.png', dpi=200)
    plt.close(fig)

    tz = r['tabzar']
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    p_exp, w_exp = np.array(TABZAR['precipitated']).T
    ax.plot(p_exp, w_exp, 'o', mfc='white', mec='k', ms=5, label='опыт (Tabzar et al., 2018)', zorder=3)
    on = tz['variants']['onset']['at_exp']
    for key, label, ls, color in (
            ('onset', f'одна точка, kij и V_s авторов ({min(on):.0f}–{max(on):.0f} %, выше рисунка)', ':',
             MODELS['pr'][2]),
            ('vs', 'V_s по кривой, kij = 0.4', '--', MODELS['pr'][2]),
            ('curve', 'kij и V_s по кривой', '-', MODELS['uq'][2])):
        ax.plot(r['p_tabzar_psig'], tz['variants'][key]['curve'], ls=ls, color=color, lw=1.8, label=label)
    ax.axvline(TABZAR['P_bubble_psia'] - 14.7, color='0.5', lw=0.8, ls='--')
    ax.set_ylim(0.0, 2.0)
    ax.text(TABZAR['P_bubble_psia'] - 14.7, 1.85, ' P_b', fontsize=8, color='0.3')
    ax.set_xlabel('P, psig')
    ax.set_ylabel('выпало асфальтенов, % масс. нефти')
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIGURES / 'tabzar.png', dpi=200)
    plt.close(fig)


def _asph_window(p, frac):
    p, frac = np.asarray(p) / 1e6, np.asarray(frac)
    on = p[frac > 1e-3]
    i = int(np.argmax(frac))
    return 100.0 * frac[i], p[i], (on.min(), on.max()) if on.size else (np.nan, np.nan)


def report(r):
    syn, z, pr, a = r['synthetic'], r['zhetybai'], r['pressure'], r['asph']
    mx = DATA['mixtures']
    rows_wdt = '\n'.join(
        f"| {n} | {mx[n]['wdt_exp']:.2f} | {m['wdt']['pr']:.2f} | {m['wdt']['ideal']:.2f} | {m['wdt']['ss']:.2f} | "
        f"{m['wdt']['ms_c']:.2f} | {m['wdt']['uq']:.2f} | {m['wdt']['uq_id']:.2f} | "
        f"{mx[n]['wdt_models']['PR+Multisolid']:.2f} | {mx[n]['wdt_models']['PR+UNIQUAC']:.2f} |"
        for n, m in syn.items())
    rows_rms = '\n'.join(
        f"| {n} | {m['rms']['pr']:.1f} | {m['rms']['ideal']:.1f} | {m['rms']['ss']:.1f} | {m['rms']['ms_c']:.1f} | "
        f"{m['rms']['uq']:.1f} | {m['rms']['uq_id']:.1f} | {m['aad']['uq']:.2f} | {mx[n]['aad_models']['PR+UNIQUAC']:.2f} |"
        for n, m in syn.items())

    def dev(key):  # отклонения WDT от опыта по смесям, [C]
        return np.array([m['wdt'][key] - mx[n]['wdt_exp'] for n, m in syn.items()])

    d_pr = dev('pr').mean()
    d_auth = np.mean([mx[n]['wdt_models']['PR+Multisolid'] - mx[n]['wdt_exp'] for n in syn])
    d_auth_uq = np.array([mx[n]['wdt_models']['PR+UNIQUAC'] - mx[n]['wdt_exp'] for n in syn])
    d_eos = np.mean([m['wdt']['pr'] - m['wdt']['ideal'] for m in syn.values()])
    rng = lambda v: f'{v.min():+.1f}…{v.max():+.1f}'  # noqa: E731
    rms_rng = lambda k: f"{min(m['rms'][k] for m in syn.values()):.1f}–{max(m['rms'][k] for m in syn.values()):.1f}"  # noqa: E731
    names = {'eff': 'текущая (идеальный раствор, эфф. dH = 41.9 кДж/моль, сдвиг Tm +52.4 K; подбор по этой кривой)',
             'pr': 'PR + multi-solid, Tm и dH Вона', 'ideal': 'идеальный multi-solid, Tm и dH Вона',
             'ss': 'идеальный твердый раствор, Tm и dH Вона', 'ms_c': 'PR + multi-solid, Coutinho с переходом',
             'uq': 'PR + UNIQUAC, Coutinho'}
    rows_li = '\n'.join(f"| {names[k]} | {z['wat'][k]:.1f} | {z['rms_fit'][k]:.2f} | {z['rms_all'][k]:.2f} |"
                        for k in ('eff', 'pr', 'ideal', 'ss', 'ms_c', 'uq'))
    uf = z['uq_fit']
    tz = r['tabzar']
    tv = tz['variants']
    p_exp, w_exp = np.array(TABZAR['precipitated']).T
    rows_tz = '\n'.join(
        f"| {label} | {v['kij']:.3f} | {v['V_s']:.4f} | {v['V_s'] - v['v_bar']:+.4f} | "
        + ' | '.join(f'{x:.2f}' for x in v['at_exp']) + f" | {v['rms']:.3f} |"
        for label, v in (('одна точка, kij и V_s авторов', tv['onset']), ('V_s по кривой, kij авторов', tv['vs']),
                         ('kij и V_s по кривой', tv['curve'])))
    pv = r['pvt']
    rows_pvt = '\n'.join(
        f"| {p / 1e6:.1f} | {pv['Rs'][i]:.1f} | {pv['Rs_linear'][i]:.1f} | {pv['Bo'][i]:.4f} | "
        + (f"{pv['Bg'][i]:.4f} | {pv['rho_g'][i]:.1f} | {pv['mu_g'][i] * 1e6:.1f} | {pv['free_gas'][i]:.3f} |"
           if pv['free_gas'][i] > 0.0 else '— | — | — | 0 |')
        for i, p in enumerate(r['p_pvt']))
    rel = np.abs(np.array(pv['Rs']) / np.array(pv['Rs_linear']) - 1.0)
    i_mid = int(np.argmax(rel))
    rs_dev = rel[i_mid]
    i6 = r['p_pvt'].index(6e6)
    i_b = r['p_live'].index(r['P_bubble'])
    drop = {k: pr['live'][k][i_b] - pr['live'][k][0] for k in ('eff', 'pr')}
    rise = {k: (pr['live'][k][-1] - pr['live'][k][i_b]) / ((r['p_live'][-1] - r['P_bubble']) / 1e6) for k in ('eff', 'pr')}
    win = {k: _asph_window(r['p_asph'], a['precipitated'][k]) for k in ('hirschberg', 'nghiem', 'nghiem_kij0')}
    rows_asph = '\n'.join(f"| {label} | {win[k][0]:.1f} | {win[k][1]:.1f} | {win[k][2][0]:.1f}–{win[k][2][1]:.1f} |"
                          for k, label in (('hirschberg', 'Hirschberg (Флори–Хаггинс), текущая'),
                                           ('nghiem', 'Nghiem на PR, kij асфальтены–газ 0.4 (Tabzar et al., 2018)'),
                                           ('nghiem_kij0', 'Nghiem на PR, kij = 0')))
    text = f"""# Модели на уравнении состояния: сравнение с опытами и моделями других авторов

Воспроизвести: `python experiments/уравнение_состояния/compare.py` (несколько минут) → `results.json`,
рисунки `figures/*.png` и этот файл. Модели — `paraphin/thermo` (флаги `wax_eos`, `asph_nghiem`); описание —
`docs/модель_АСПО.md`, разд. 3.4 и 4.5.

Во всех рисунках цвет и стиль закреплены за моделью: синяя сплошная — текущая модель симулятора, оранжевая
штриховая — уравнение состояния Пенга–Робинсона с multi-solid (или Nghiem для асфальтенов), зеленая
пунктирная — идеальный multi-solid, розовая штрихпунктирная — идеальный твердый раствор, фиолетовая — PR +
multi-solid с плавлением по Coutinho, черная сплошная — PR + твердый раствор UNIQUAC; кружки — опыт.

## A. Синтетические смеси н-алканов в н-декане (da Silva et al., 2017)

Пять смесей C18–C36 в н-декане (64–66 % масс. растворителя; Dauphin et al., 1999; Fleming et al., 2017): одна
непрерывная (Bim 0) и четыре бимодальные. Составы — Table 1 статьи, температура исчезновения парафина (WDT) —
Table 3, доля твердого — оцифровка рис. 4b–8b (около 0.7 % масс., 0.7 °C; `experiments/data/dasilva2017_sle.json`).
Здесь проверяется сама термодинамика без характеризации нефти. Свойства компонентов — те же, что в симуляторе:
Tc, Pc по Riazi–Daubert, ω по Kesler–Lee, kij = 0; плавление — по Won или по Coutinho (SPE 78324, 2002:
температура и теплота плавления и перехода ротаторной фазы в орторомбическую, `coutinho_nalkane`). Твердый
раствор UNIQUAC — предсказательный, по Coutinho: параметры решетки и энергии взаимодействия из теплот сублимации
н-алканов, без подбора (`solids.UniquacSolidWax`). У авторов свойства по Marano & Holder, плавление и переходы
по Coutinho et al. (2006), kij по Pan et al. (1997).

Таблица A1. WDT, °C.

| Смесь | опыт | PR + multi-solid, Вон | идеальный multi-solid, Вон | идеальный тв. раствор, Вон | PR + multi-solid, Coutinho | PR + UNIQUAC | UNIQUAC, идеальная жидкость | PR + multi-solid (авторы) | PR + UNIQUAC (авторы) |
|---|---|---|---|---|---|---|---|---|---|
{rows_wdt}

Таблица A2. Отклонение доли твердого от опыта, % масс.: СКО по моделям, у UNIQUAC также среднее абсолютное
(AAD) — в нем авторы приводят свою модель (Table 5).

| Смесь | PR + multi-solid, Вон | идеальный multi-solid, Вон | идеальный тв. раствор, Вон | PR + multi-solid, Coutinho | PR + UNIQUAC | UNIQUAC, идеальная жидкость | AAD PR + UNIQUAC | AAD PR + UNIQUAC (авторы) |
|---|---|---|---|---|---|---|---|---|
{rows_rms}

![](figures/synthetic.png)

Рис. A. Доля твердого в синтетических смесях: опыт (da Silva et al., 2017) и модели с параметрами симулятора.

Multi-solid на PR с плавлением по Вону занижает WDT в среднем на {-d_pr:.1f} °C, у авторов — на {-d_auth:.1f} °C.
Неидеальность жидкости по PR сдвигает WDT вверх лишь на {d_eos:.1f} °C относительно идеального раствора: смесь
н-алканов почти атермична. Значит, разница с авторами — в свойствах плавления. Корреляция Вона не содержит
твердо-твердого перехода н-алканов (ротаторная фаза), и полная теплота перехода у нее меньше калориметрической:
у C24 65 кДж/моль против 54.9 + 31.3 = 86 кДж/моль плавления и переходов (Broadhurst, 1962, табл. 3).

Переход по Coutinho сокращает занижение WDT multi-solid до {rng(dev('ms_c'))} °C и СКО доли твердого до
{rms_rng('ms_c')} % масс. (по Вону {rng(dev('pr'))} °C и {rms_rng('pr')} %). Скачок теплоемкости по Pedersen et al.
(1991) поверх перехода ухудшает WDT до {rng(dev('ms_cp'))} °C: корреляция выведена для multi-solid с плавлением
по Вону и в паре с переходом учитывает одно и то же дважды, поэтому в модели он не включен (аргумент `mw`
`MultiSolidWax` остается для проверки).

Твердый раствор UNIQUAC с жидкостью по PR — лучшая модель и здесь: WDT {rng(dev('uq'))} °C (у авторов
{rng(d_auth_uq)} °C), СКО доли твердого {rms_rng('uq')} % масс. Хуже всего он описывает бимодальные смеси с
глубоким провалом между модами (Bim 9 и Bim 13): у них возможно расслоение на два твердых раствора, а здесь
раствор один. С идеальной жидкостью WDT ниже на 1–2 °C ({rng(dev('uq_id'))} °C).

## B. Нефть Жетыбая (Li et al., 2024): кривая выпадения и WAT по ДСК

Таблица B. WAT (ДСК — 45.65 °C) и СКО кривой выпадения, % масс. нефти.

| Модель | WAT, °C | СКО 10–45 °C | СКО −20…45 °C |
|---|---|---|---|
{rows_li}

![](figures/zhetybai.png)

Рис. B. Кривая выпадения нефти Жетыбая: опыт и модели. Группы те же (SCN C17–C60, `scn_slope` = 0.07).

Все модели с калориметрическими свойствами плавления завышают WAT на
{min(z['wat'][k] for k in ('pr', 'ideal', 'ss', 'ms_c', 'uq')) - 45.65:.0f}–{max(z['wat'][k] for k in ('pr', 'ideal', 'ss', 'ms_c', 'uq')) - 45.65:.0f} °C.
Уравнение состояния картину не исправляет: PR добавляет к идеальному multi-solid еще
{z['wat']['pr'] - z['wat']['ideal']:.1f} °C. Переход по Coutinho и твердый раствор UNIQUAC, лучшие на
синтетических смесях, здесь дают СКО {z['rms_fit']['ms_c']:.1f} и {z['rms_fit']['uq']:.1f} % масс. (multi-solid по
Вону — {z['rms_fit']['pr']:.1f}): переход делает твердое устойчивее и повышает WAT, а она у н-алканов и так выше,
чем у «парафина» этой нефти.
Подбор одной характеризации — наклона SCN-распределения и последнего SCN — при UNIQUAC без эффективных параметров
дает СКО {uf['rms']:.1f} % масс. (наклон {uf['scn_slope']:.3f}, последний SCN {uf['scn_last']}, WAT
{uf['wat']:.0f} °C) против {z['rms_fit']['eff']:.2f} % у текущей модели.
Пологая кривая ДСК (1.5 % при 40 °C и 24.9 % при −20 °C) у нефти с 25 % «парафина» указывает, что в эту долю входят
изо- и циклоалканы с меньшей теплотой плавления, а не только н-алканы, а распределение н-алканов не измерено.
Для такой нефти эффективные параметры (текущая модель) остаются нужны, и в симуляторе по умолчанию остается она.
Уравнение состояния полезно там, где модель должна предсказывать, а не воспроизводить: давление, газ, состав.

## C. WAT от давления

Таблица C. Наклон WAT дегазированной нефти на 0.1–20 МПа и ход WAT живой нефти.

| Модель | dWAT/dP, °C/МПа | WAT(P_b) − WAT(0.1 МПа), °C | dWAT/dP выше P_b, °C/МПа |
|---|---|---|---|
| опыт: Sandyga et al. (2020), дегазированная нефть | 0.15–0.23 | — | — |
| текущая (Пойнтинг с dv = 0.028 v_L, подбор по Sandyga; газ линейно по Rs) | {pr['slope_dead']['eff']:.3f} | {drop['eff']:.1f} | {rise['eff']:.3f} |
| PR + multi-solid, dv = 0.13 v_L по плотностям расплава и кристаллов, без подбора | {pr['slope_dead']['pr']:.3f} | {drop['pr']:.1f} | {rise['pr']:.3f} |
| PR + multi-solid без скачка объема (вариант пакета research) | {pr['slope_dead']['pr_no_dv']:.3f} | — | — |

![](figures/pressure.png)

Рис. C. Сдвиг WAT с давлением: дегазированная нефть (слева, серая полоса — регрессии Sandyga et al., 2020) и
живая нефть с R_s = 40 м³/м³, P_b = 9.5 МПа (справа).

В пакете research фугитивность твердого бралась от чистой жидкости при системном давлении. Так твердое молча
получает объем жидкости, и наклон выходит на порядок меньше измеренного ({pr['slope_dead']['pr_no_dv']:.3f} °C/МПа).
С поправкой Пойнтинга на скачок объема (Pan et al., 1997) и физическим dv = v_L − v_S по плотностям 780 и
900 кг/м³ наклон {pr['slope_dead']['pr']:.2f} °C/МПа — того же порядка, что опыт, без подбора. Эффективная модель
попадает в опыт подбором dv при заниженной dH, модель на PR — с калориметрической dH и физическим dv.

Растворенный газ понижает WAT живой нефти к P_b на {-drop['pr']:.1f} °C по PR и на {-drop['eff']:.1f} °C по линейному Rs.
У Pan et al. (1997) снижение доходило до 15 K; оно растет с газосодержанием, а здесь R_s всего 40 м³/м³. Газ по PR
(kij газ–нефть = {pr['kij_gas']:.4f}, подобран по P_b) действует на группы слабее, чем в идеальном растворе: метан
повышает коэффициенты фугитивности тяжелых компонентов и частично снимает эффект разбавления.

## D. Асфальтены: Hirschberg против Nghiem

Обе модели откалиброваны по одной точке: при {init_T} °C и P_onset = {r['P_onset_asph'] / 1e6:.0f} МПа нефть ровно
насыщена асфальтенами; P_b = {r['P_bubble'] / 1e6:.1f} МПа.

Таблица D. Выпадение асфальтенов при {init_T} °C.

| Модель | максимум, % от содержания | при P, МПа | интервал выпадения, МПа |
|---|---|---|---|
{rows_asph}

![](figures/asphaltenes.png)

Рис. D. Доля выпавших асфальтенов в зависимости от давления.

Обе модели дают колокол с максимумом у давления насыщения: выше P_b асфальтены выпадают при расширении нефти,
ниже — растворяются вновь, потому что уходит газ-осадитель. У Nghiem колокол держится на kij асфальтены–газ:
при kij = 0 выпадение растет монотонно до {win['nghiem_kij0'][0]:.0f} % при {win['nghiem_kij0'][1]:.0f} МПа и ниже
P_b не растворяется, что противоречит опытам (Burke et al., 1990; Tabzar et al., 2018). Значение 0.4 взято у
Tabzar et al. (2018), где оно подобрано по замерам выпадения. Амплитуды колокола у моделей разные:
{win['hirschberg'][0]:.0f} и {win['nghiem'][0]:.0f} % содержания. Одна точка начала осаждения их не задает: нужна
кривая количества выпавших асфальтенов от давления (гравиметрия или HPM на глубинной пробе). Если она есть, ее
задает константа `asph_curve`: по ней подбираются v_a у Hirschberg или V_s и kij асфальтены–газ у Nghiem (давление
начала осаждения по-прежнему держит P_onset_asph). Для Жетыбая и Узеня кривой нет; проверка — в разделе E.

## E. Асфальтены Nghiem: одна точка против кривой выпавших (Tabzar et al., 2018)

Живая нефть Tabzar et al. (2018) по данным Jamaluddin et al.: 16 компонентов (Table 3), асфальтены — часть фракции
C31+ (C31B+, {tz['asph_wt']:.1f} % масс. живой нефти), 212 °F (100 °C), давление начала осаждения 5000 psia,
насыщения 2050 psia; выпавшие асфальтены при четырех давлениях (Table 4). PR с kij = 0 дает давление насыщения
{tz['p_b_pr_kij0']:.0f} psia, поэтому во всех вариантах kij метан–C7+ подобран по 2050 psia (у авторов давление
насыщения модели 2009 psia), а f_s* — по давлению начала осаждения.

Таблица E. Выпавшие асфальтены, % масс. живой нефти, и параметры. v̄_a — парциальный мольный объем асфальтенов
в нефти при давлении начала осаждения, л/моль.

| Вариант | kij асфальтены–легкие | V_s, л/моль | V_s − v̄_a | {' | '.join(f'{p:.0f} psig' for p in p_exp)} | СКО |
|---|---|---|---|---|---|---|---|---|
| опыт | — | — | — | {' | '.join(f'{x:.3f}' for x in w_exp)} | — |
{rows_tz}

![](figures/tabzar.png)

Рис. E. Выпадение асфальтенов из живой нефти при 212 °F: опыт (Tabzar et al., 2018, Table 4) и модель Nghiem на
PR с калибровкой по одной точке и по кривой.

Количество выпавших задает не сам V_s, а его разность с парциальным объемом асфальтенов в нефти: снижение давления
на ΔP от точки начала осаждения пересыщает нефть в exp[(v̄_a − V_s)ΔP/RT] раз. При V_s = 0.8 л/моль авторов эта
разность на нашем PR (без сдвига объема, со своими kij) другая, и выпадает {min(tv['onset']['at_exp']):.0f}–{max(tv['onset']['at_exp']):.0f} %
вместо 0.4–1 %: параметры одной модели не переносятся в другую, и одна точка начала осаждения количества не задает.
V_s, подобранный по кривой, отличается от v̄_a на {100 * (tv['curve']['V_s'] / tv['curve']['v_bar'] - 1):.1f} %. kij
асфальтены–легкие задает форму колокола ниже P_b: при 0.4 растворение ниже давления насыщения слишком быстрое
(при 1000 psig выпавших нет), лучшее значение — {tv['curve']['kij']:.3f}, и тогда кривая воспроизводит все четыре
точки со СКО {tv['curve']['rms']:.3f} % масс. Ниже 1000 psig замеров нет, и рост выпадения у атмосферного давления
на рис. E — экстраполяция: при малом kij уход газа почти не растворяет асфальтены, и снова преобладает расширение
нефти. В симуляторе тот же подбор делает `tables.fit_nghiem_curve` по `asph_curve`.

## F. PVT нефти модели по уравнению состояния

Газ, растворитель и группы парафина на PR (`paraphin/thermo/pvt.py`, `python -m paraphin.thermo.pvt`); kij
газ–нефть = {pv['kij_gas']:.4f} подобран по P_b = {r['P_bubble'] / 1e6:.1f} МПа при {init_T} °C. Вязкость газа —
Lee, Gonzalez & Eakin (1966). V_g/V_L — объем выделившегося газа на объем нефти при пластовых условиях: в
симуляторе газ растворен всегда, и это поле ('Free gas' в выгрузке при `wax_eos` и `wax_pressure`) показывает,
где модель держит в нефти газ, который выделился бы.

Таблица F. PVT при {init_T} °C.

| P, МПа | R_s по PR, м³/м³ | R_s линейный (симулятор) | B_o | B_g | ρ_g, кг/м³ | μ_g, мкПа·с | V_g/V_L |
|---|---|---|---|---|---|---|---|
{rows_pvt}

Линейный R_s симулятора выше R_s по PR не больше чем на {100 * rs_dev:.0f} % (при
{r['p_pvt'][i_mid] / 1e6:g} МПа, где газа в нефти мало): для растворенного газа в уравнениях равновесия парафина и
асфальтенов линейного закона достаточно. Свободный газ ниже P_b быстро набирает объем: при 6 МПа
V_g/V_L = {pv['free_gas'][i6]:.2f}, при 2 МПа — {pv['free_gas'][1]:.1f}. Там, где давление падает заметно ниже P_b,
модель с растворенным газом неточна по подвижности фаз; это ограничение постановки, а не термодинамики.

## Выводы

1. Порт термодинамики из пакета research верен: числа совпадают с исходным кодом до 1e-8 (`tests/test_thermo_eos.py`).
2. На синтетических смесях multi-solid на PR с корреляциями Вона занижает WDT на 7–10 °C. Твердо-твердый переход
   по Coutinho сокращает занижение до {rng(dev('ms_c'))} °C, твердый раствор UNIQUAC с жидкостью по PR дает
   {rng(dev('uq'))} °C и СКО доли твердого {rms_rng('uq')} % масс. — на уровне лучшей модели авторов.
3. На нефти Жетыбая ни переход, ни UNIQUAC, ни подбор характеризации не заменяют эффективные параметры
   (СКО {uf['rms']:.1f} % против {z['rms_fit']['eff']:.2f} %): нужна хроматограмма н-алканов и учет изо- и
   циклоалканов.
4. Скачок объема при кристаллизации обязателен в multi-solid на уравнении состояния; с физическим dv наклон
   dWAT/dP совпадает с опытом по порядку без подбора. Поправка внесена в `paraphin/thermo/solids.py`.
5. Колокол выпадения асфальтенов по Nghiem появляется только с kij асфальтены–газ. Калибровка по одной точке
   начала осаждения количества не задает (на опыте Tabzar — ошибка на порядок), калибровка по кривой выпавших
   (`asph_curve`) воспроизводит опыт со СКО {tv['curve']['rms']:.2f} % масс.
6. PVT по уравнению состояния подтверждает линейный R_s симулятора (отклонение до {100 * abs(rs_dev):.0f} %).
"""
    (HERE / 'сравнение_EOS.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    # --plot - только рисунки и отчет по готовому results.json
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8')) if '--plot' in sys.argv else main()
    plot(res)
    report(res)
    print((HERE / 'сравнение_EOS.md').read_text(encoding='utf-8'))
