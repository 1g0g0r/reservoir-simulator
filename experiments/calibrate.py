"""Калибровка параметров детального состава нефти по опубликованным измерениям.

    python experiments/calibrate.py

Печатает таблицы и пишет `experiments/results/calibration.json` (его читают `docs/make_model_figures.py` и текст docx).
Найденные значения переносятся в `paraphin/constants.py` вручную, с комментарием, - как Tm и alpha.

A. SCN-распределение и эффективные параметры растворимости групп (scn_slope, wax_alpha_eff, wax_Tm_shift) -
   по кривой выпадения нефти Жетыбая (Li et al., Processes 2024, 12:421, рис. 3б; `core_flood.LI_PRECIPITATION`)
   в интервале 10-45 C, как и прежние Tm, alpha.
B. Скачок объема при плавлении wax_dv_frac - по росту WAT с давлением у Sandyga et al. (JPEPT 2020, 10:2541;
   регрессии `validation_sandyga2020.WAT_P_FIT`): средний наклон на 0.1-20 МПа.
C. Вязкость и гель (visc_D, gel_phi, gel_tau_ref, gel_n) - по вязкости той же нефти при охлаждении и 150 1/с
   (Li et al., 2024, рис. 4а; `validation_li2024.VISCOSITY`): бингамовское разложение mu = mu_p + tau_y/gamma.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'исходная_модель'))

import core_flood as cf  # noqa: E402
from paraphin import oil_composition as oc  # noqa: E402
from paraphin.constants import (MW, M_o, Tm, alpha, R, ro_o, ro_p, gel_phi_ref, scn_bounds, scn_first,  # noqa: E402
                                scn_last, scn_slope, P_ref_wax, wax_alpha_eff, wax_Tm_shift, wax_dv_frac)

OUT = Path(__file__).resolve().parent / 'results' / 'calibration.json'
GAMMA_LI = 150.0  # скорость сдвига реометра Li et al. (2024), [1/с]

# Данные те же, что в `experiments/исходная_модель/validation_li2024.py` и `validation_sandyga2020.py` (там же - оцифровка и
# точность); сюда скопированы, потому что те модули при импорте читают результаты прогонов керна.
# Вязкость нефти Жетыбая при охлаждении, 150 1/с (Li et al., 2024, рис. 4а): (T, C; мПа*с)
LI_VISCOSITY = [(24.4, 443.0), (25.6, 321.0), (28.2, 150.0), (30.9, 73.0), (33.8, 47.5), (36.5, 38.0),
                (40.0, 33.2), (45.0, 27.1), (50.0, 23.29), (55.0, 21.0), (70.0, 14.3), (80.0, 11.2),
                (90.0, 10.53), (100.0, 8.7)]
VISC_FIT_T_MIN = 40.0  # выше аномальной точки нефть ньютоновская - по этим точкам жидкая основа
# Регрессии Sandyga et al. (2020) P = a*ln(WAT) - b (P в МПа, WAT в C) для 10, 30 и 60 % парафина в керосине
SANDYGA_WAT_P = {0.10: (175.75, 569.44), 0.30: (197.45, 693.84), 0.60: (207.79, 793.29)}


def precipitation(props, t_c, dp=0.0, n_g=0.0):
    """Выпавший парафин (взвесь), доля нефти, по группам `props` при температурах t_c."""
    return np.array([props['w'].sum() - oc.sle_split_np(props['w'], props['M'], props['Tm'], props['dH'],
                                                         props['dv'], t, dp, n_g).sum() for t in np.atleast_1d(t_c)])


def wat(props, dp=0.0, n_g=0.0):
    return oc.wat_np(props['w'], props['M'], props['Tm'], props['dH'], props['dv'], dp, n_g)


# --- A. Группы по кривой выпадения ---------------------------------------------------------------------------

WAT_DSC = 45.65      # температура начала кристаллизации по ДСК (Li et al., 2024, рис. 3а), [C]
WAT_WEIGHT = 0.002   # вес невязки WAT в подгонке: 1 C ~ 0.2 % выпавшего - порядок точности оцифровки кривой


def _fit_groups(slope, bounds, t, prec):
    """Эффективные wax_alpha_eff, wax_Tm_shift групп по точкам кривой выпадения и WAT по ДСК."""
    def residual(x):
        g = oc.group_properties(slope, x[0] * 1e3, x[1], bounds=bounds, total=cf.LI_W)
        return np.append(precipitation(g, t) - prec, WAT_WEIGHT * (wat(g) - WAT_DSC))
    best = None
    for x0 in ((55.0, 25.0), (40.0, 45.0), (70.0, 10.0), (47.0, 41.0)):
        res = least_squares(residual, x0, bounds=([5.0, -40.0], [200.0, 80.0]))
        if best is None or res.cost < best.cost:
            best = res
    g = oc.group_properties(slope, best.x[0] * 1e3, best.x[1], bounds=bounds, total=cf.LI_W)
    dev = precipitation(g, t) - prec
    return best.x[0] * 1e3, best.x[1], g, dev


def fit_scn():
    pts = np.array([p for p in cf.LI_PRECIPITATION if p[0] >= cf.LI_FIT_T_MIN])
    t, prec = pts[:, 0], pts[:, 1] / 100.0

    scan = []
    for bounds in ((24, 34), scn_bounds, (20, 24, 28, 34, 42)):
        for slope in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10):
            a, sh, g, dev = _fit_groups(slope, bounds, t, prec)
            scan.append({'groups': len(bounds) + 1, 'slope': slope, 'alpha': a, 'shift': sh,
                         'max': float(np.abs(dev).max()), 'rms': float(np.sqrt(np.mean(dev ** 2))), 'wat': wat(g)})
    print('A. Подгонка групп по кривой Li (10-45 C) и WAT по ДСК при разном числе групп и наклоне SCN:')
    print('| групп | scn_slope | alpha_eff, кДж/моль | сдвиг Tm, K | макс. откл., % | СКО, % | WAT, C |')
    for row in scan:
        print(f"| {row['groups']} | {row['slope']:.2f} | {row['alpha'] / 1e3:.1f} | {row['shift']:.1f} | "
              f"{100 * row['max']:.2f} | {100 * row['rms']:.2f} | {row['wat']:.1f} |")

    alpha_fit, shift_fit, g, _ = _fit_groups(scn_slope, scn_bounds, t, prec)
    t_all = np.array([p[0] for p in cf.LI_PRECIPITATION])
    groups_curve = precipitation(g, t_all)
    by_group = np.array([[g['w'][k] - oc.sle_split_np(g['w'], g['M'], g['Tm'], g['dH'], g['dv'], tt)[k]
                          for k in range(g['w'].size)] for tt in t_all])
    legacy_curve = cf.LI_W - cf.w_saturated(cf.LI_W, t_all, MW, M_o, Tm, alpha)
    fit_pts = t_all >= cf.LI_FIT_T_MIN
    exp = np.array([p[1] for p in cf.LI_PRECIPITATION]) / 100.0
    rms = lambda c: float(np.sqrt(np.mean((c[fit_pts] - exp[fit_pts]) ** 2)))  # noqa: E731
    print(f'A. scn_slope = {scn_slope}, {len(scn_bounds) + 1} группы: wax_alpha_eff = {alpha_fit / 1e3:.2f} кДж/моль, '
          f'wax_Tm_shift = {shift_fit:.2f} K, WAT = {wat(g):.2f} C (ДСК {WAT_DSC} C; прежняя (6.1) - '
          f'{cf.cloud_point(cf.LI_W, MW, M_o, Tm, alpha):.1f} C); СКО {100 * rms(groups_curve):.2f} % против '
          f'{100 * rms(legacy_curve):.2f} % у прежней')
    print('| T, C | опыт, % | группы, % | по группам, % | прежняя (6.1), % |')
    for i, tt in enumerate(t_all):
        print(f'| {tt:.1f} | {100 * exp[i]:.1f} | {100 * groups_curve[i]:.2f} | '
              f'{" / ".join(f"{100 * x:.2f}" for x in by_group[i])} | {100 * legacy_curve[i]:.2f} |')
    n, M, w = oc.scn_distribution(scn_slope, scn_first, scn_last, cf.LI_W)
    return {'scan': scan, 'alpha': alpha_fit, 'shift': shift_fit, 'wat': wat(g), 'T': t_all.tolist(),
            'exp': (100 * exp).tolist(), 'groups': (100 * groups_curve).tolist(), 'by_group': (100 * by_group).tolist(),
            'legacy': (100 * legacy_curve).tolist(), 'rms_groups': rms(groups_curve), 'rms_legacy': rms(legacy_curve),
            'wat_legacy': cf.cloud_point(cf.LI_W, MW, M_o, Tm, alpha),
            'scn_n': n.tolist(), 'scn_w': w.tolist(), 'scn_M': M.tolist(),
            'group_w': g['w'].tolist(), 'group_M': g['M'].tolist(), 'group_Tm': (g['Tm'] - 273.15).tolist(),
            'group_L': g['L'].tolist()}


# --- B. Давление ----------------------------------------------------------------------------------------------

def fit_dv(a_scn):
    # Средний наклон WAT(P) регрессий Sandyga на 0.1-20 МПа
    slopes = {c: float((np.exp((20.0 + b) / a) - np.exp((0.1 + b) / a)) / 19.9) for c, (a, b) in SANDYGA_WAT_P.items()}
    target = float(np.mean(list(slopes.values())))

    def props(frac):
        g = oc.group_properties(scn_slope, a_scn['alpha'], a_scn['shift'], total=cf.LI_W, dv_frac=frac)
        return g

    def slope(frac):
        g = props(frac)
        return (wat(g, 20e6 - P_ref_wax) - wat(g, 0.1e6 - P_ref_wax)) / 19.9

    frac = brentq(lambda f: slope(f) - target, 1e-4, 0.5)
    g = props(frac)
    p = np.linspace(0.1, 35.0, 70)
    wat_poynting = [wat(g, pp * 1e6 - P_ref_wax) for pp in p]
    n_gas_b = oc.N_GAS_B
    wat_gas = [wat(g, pp * 1e6 - P_ref_wax, n_gas_b * min(pp * 1e6 / oc.P_bubble, 1.0)) for pp in p]
    sandyga = {str(c): [float(np.exp((pp + b) / a) - np.exp((0.1 + b) / a)) for pp in p] for c, (a, b) in SANDYGA_WAT_P.items()}
    print(f'B. Наклон WAT(P) у Sandyga (0.1-20 МПа): ' + ', '.join(f'{100 * c:.0f}%: {s:.3f}' for c, s in slopes.items()) +
          f' C/МПа, среднее {target:.3f} -> wax_dv_frac = {frac:.4f} (Клапейрон-Клаузиус для группы C41+: '
          f'dWAT/dP = WAT*dv/dH = {(wat(g) + 273.15) * g["dv"][-1] / g["dH"][-1] * 1e6:.3f} C/МПа)')
    print(f'B. С газом (P_b = {oc.P_bubble / 1e6:.1f} МПа): WAT при 0.1 / P_b / 20 МПа = {wat_gas[0]:.1f} / '
          f'{wat(g, oc.P_bubble - P_ref_wax, n_gas_b):.1f} / {wat_gas[int(np.argmin(np.abs(p - 20)))]:.1f} C')
    return {'target': target, 'slopes': slopes, 'dv_frac': frac, 'P': p.tolist(), 'wat_poynting': wat_poynting,
            'wat_gas': wat_gas, 'sandyga_shift': sandyga}


# --- C. Вязкость и гель ----------------------------------------------------------------------------------------

def fit_rheology(a_scn):
    t, mu = np.array(LI_VISCOSITY).T
    hot = t >= VISC_FIT_T_MIN

    def arrhenius(x, tt):
        return x[0] * np.exp(x[1] * 1e3 / R * (1.0 / (tt + 273.15) - 1.0 / 298.15))

    base = least_squares(lambda x: np.log(arrhenius(x, t[hot])) - np.log(mu[hot]), [40.0, 25.0])
    mu_l = arrhenius(base.x, t)
    g = oc.group_properties(scn_slope, a_scn['alpha'], a_scn['shift'], total=cf.LI_W)
    w_ps = precipitation(g, t)
    phi = (w_ps / ro_p) / (w_ps / ro_p + (1.0 - w_ps) / ro_o)
    cold = ~hot

    def model(x, tt_mu_l, ph):
        d, phi_gel, tau_ref, n = x
        tau = np.where(ph > phi_gel, tau_ref * np.clip((ph - phi_gel) / (gel_phi_ref - phi_gel), 0.0, None) ** n, 0.0)
        return tt_mu_l * np.exp(d * ph) + tau / GAMMA_LI * 1e3  # [мПа*с]

    fit = least_squares(lambda x: np.log(model(x, mu_l[cold], phi[cold])) - np.log(mu[cold]),
                        [4.0, 0.05, 10.0, 3.5], bounds=([0.0, 0.005, 0.01, 1.0], [30.0, 0.09, 1e4, 8.0]))
    d, phi_gel, tau_ref, n = fit.x
    pr_full = mu_l * np.exp(18.12 * phi + 405.1 * phi ** 4 / np.sqrt(GAMMA_LI) + 7.876e6 * phi ** 4 / GAMMA_LI)
    pr_d = mu_l * np.exp(18.12 * phi)
    kd = mu_l * (1.0 - np.minimum(phi, 0.99 * 0.6) / 0.6) ** (-1.5)
    bingham = model(fit.x, mu_l, phi)
    print(f'C. Жидкая основа: mu(25 C) = {base.x[0]:.1f} мПа*с, E = {base.x[1]:.1f} кДж/моль')
    print(f'C. Бингамовское разложение при {GAMMA_LI:.0f} 1/с: visc_D = {d:.2f}, gel_phi = {phi_gel:.4f}, '
          f'gel_tau_ref = {tau_ref:.2f} Па при phi = {gel_phi_ref}, gel_n = {n:.2f}; СКО ln mu {np.sqrt(np.mean(fit.fun ** 2)):.3f}')
    print('| T, C | опыт | Кригер-Догерти | P-R, только D | P-R полная | D + tau_y | phi_s, % | tau_y, Па |')
    for i in range(t.size):
        tau_i = max(0.0, (bingham[i] - mu_l[i] * np.exp(d * phi[i])) * GAMMA_LI * 1e-3)
        print(f'| {t[i]:.1f} | {mu[i]:.1f} | {kd[i]:.1f} | {pr_d[i]:.1f} | {pr_full[i]:.3g} | {bingham[i]:.1f} | '
              f'{100 * phi[i]:.1f} | {tau_i:.2f} |')
    return {'T': t.tolist(), 'mu': mu.tolist(), 'mu_l': mu_l.tolist(), 'kd': kd.tolist(), 'pr_d': pr_d.tolist(),
            'pr_full': pr_full.tolist(), 'bingham': bingham.tolist(), 'phi': phi.tolist(), 'visc_D': d,
            'gel_phi': phi_gel, 'gel_tau_ref': tau_ref, 'gel_n': n, 'mu25': base.x[0], 'E': base.x[1] * 1e3}


def main():
    a = fit_scn()
    b = fit_dv(a)
    c = fit_rheology(a)
    print('\nЗначения для constants.py:')
    print(f'  scn_slope = {scn_slope}; wax_alpha_eff = {a["alpha"]:.4g}; wax_Tm_shift = {a["shift"]:.3g} '
          f'(сейчас {wax_alpha_eff:.4g}, {wax_Tm_shift})')
    print(f'  wax_dv_frac = {b["dv_frac"]:.3g} (сейчас {wax_dv_frac})')
    print(f'  visc_D = {c["visc_D"]:.3g}; gel_phi = {c["gel_phi"]:.3g}; gel_tau_ref = {c["gel_tau_ref"]:.3g}; '
          f'gel_n = {c["gel_n"]:.3g}')
    OUT.write_text(json.dumps({'scn': a, 'pressure': b, 'rheology': c}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print('Записано:', OUT)


if __name__ == '__main__':
    main()
