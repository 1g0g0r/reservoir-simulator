"""Сравнение с опытами Li et al. (2024): вязкость нефти Жетыбая и кольматация низкопроницаемого керна.

Li X., Zhao L., Fei R., Wang J., Liu S., Li M., Han S., Zhou F. Cooling Damage Characterization and
Chemical-Enhanced Oil Recovery in Low-Permeable and High-Waxy Oil Reservoirs // Processes. 2024. V. 12.
P. 421. doi:10.3390/pr12020421 (resources/литература/парафиновое/li2024_zhetybai_cooling_damage.pdf).

По кривой выпадения этой же нефти подобраны Tm, alpha в constants.py (`core_flood.py --fit`), поэтому
здесь модель проверяется на той же нефти, но по другим измерениям:
  A. вязкость при охлаждении (рис. 4а) - жидкая основа по Аррениусу плюс множитель Кригера-Догерти;
  B. прокачка нефти через керн 2 (табл. 1: 18.81 мД, 16.9%, 5 см) при 90, 65, 45 и 25 C, k/k0 от PV (рис. 5).
Комментарии и выводы - VALIDATION.md.

    python твт_статья/validation_li2024.py          # A и B (прогоны керна ~20 шт., правят constants.py)
    python твт_статья/validation_li2024.py --plot   # только рисунок по сохраненному json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_flood as cf  # noqa: E402

sys.path.insert(0, str(cf.ROOT))
from paraphin.constants import Tm, alpha, MW, M_o, ro_p, phi_max  # noqa: E402

OUT = cf.ROOT / 'outputs' / 'data' / 'validation_li2024.json'
FIG = cf.ROOT / 'outputs' / 'figures' / 'validation' / 'li2024.png'

# --- Данные Li et al. (2024) ---------------------------------------------------------------------------
RO_OIL = 883.0  # плотность нефти при 20 C, [кг/м^3] (разд. 2.1)
# Рис. 4а, кривая охлаждения при 150 1/с: (T, C; мПа*с). Оцифровка с растра 300 dpi, точность ~3 мПа*с
# (маркер) и ~0.3 C; точки 50 и 90 C - значения из текста (23.29 и 10.53 мПа*с). Аномальная точка 35.6 C.
VISCOSITY = [(24.4, 443.0), (25.6, 321.0), (28.2, 150.0), (30.9, 73.0), (33.8, 47.5), (36.5, 38.0),
             (40.0, 33.2), (45.0, 27.1), (50.0, 23.29), (55.0, 21.0), (70.0, 14.3), (80.0, 11.2),
             (90.0, 10.53), (100.0, 8.7)]
VISC_FIT_T_MIN = 40.0  # выше аномальной точки нефть ньютоновская - по этим точкам жидкая основа

# Рис. 5: k(t)/K от прокачанных PV. Керн один, охлаждался ступенями 90 -> 65 -> 45 -> 25 C; на каждой
# ступени нефть той же температуры прокачивалась с 0.1 мл/мин до стабилизации перепада, K - проницаемость
# в начале ступени (все кривые начинаются с 1). Оцифровка маркеров по цвету с растра 300 dpi (~0.01);
# точки 45 C при 1-2.5 PV закрыты другими маркерами и прочитаны визуально (~0.02).
K_PV = {
    90: [(0.0, 1.0), (0.5, 0.868), (0.75, 0.826), (1.0, 0.795), (1.25, 0.784), (1.5, 0.768), (1.75, 0.768),
         (2.0, 0.759), (2.25, 0.768), (2.5, 0.768), (2.75, 0.768), (3.0, 0.768), (3.25, 0.776), (3.5, 0.784),
         (3.75, 0.768), (4.0, 0.760), (4.25, 0.768), (4.5, 0.768), (4.75, 0.784), (5.0, 0.768)],
    65: [(0.0, 1.0), (0.25, 0.932), (0.5, 0.826), (0.75, 0.734), (1.0, 0.634), (1.25, 0.564), (1.5, 0.520),
         (1.75, 0.487), (2.0, 0.440), (2.25, 0.432), (2.5, 0.434), (2.75, 0.431), (3.0, 0.431), (3.25, 0.417),
         (3.5, 0.423), (3.75, 0.420), (4.0, 0.417), (4.25, 0.420), (4.5, 0.426), (4.75, 0.426), (5.0, 0.426)],
    45: [(0.0, 1.0), (0.25, 0.905), (0.5, 0.762), (0.75, 0.649), (1.0, 0.54), (1.25, 0.47), (1.5, 0.415),
         (1.75, 0.365), (2.0, 0.345), (2.25, 0.33), (2.5, 0.34), (2.75, 0.342), (3.0, 0.331), (3.25, 0.331),
         (3.5, 0.331), (3.75, 0.331), (4.0, 0.331), (4.25, 0.331), (4.5, 0.331), (4.75, 0.331), (5.0, 0.331)],
    25: [(0.0, 1.0), (0.25, 0.815), (0.5, 0.696), (0.75, 0.617), (1.0, 0.543), (1.25, 0.490), (1.5, 0.431),
         (1.75, 0.376), (2.0, 0.342), (2.25, 0.314), (2.5, 0.292), (2.75, 0.267), (3.0, 0.236), (3.25, 0.209),
         (3.5, 0.186), (3.75, 0.167), (4.0, 0.150), (4.25, 0.136), (4.5, 0.122), (4.75, 0.110), (5.0, 0.111)],
}

# Керн 2 (табл. 1) и режим (разд. 2.2.3). Сечение - квадрат той же площади, что круг диаметром 2.449 см.
CORE = dict(
    title='Zhetybai crude, core 2 (Li et al., 2024)',
    k0=18.81e-3, porosity=0.169, length=0.05, side=float(np.sqrt(np.pi / 4.0) * 0.02449),
    q=0.1e-6 / 60.0, T=25.0,
    w=cf.LI_W, MW=MW, M_o=M_o, Tm=Tm, dH=alpha, ro_o=RO_OIL, ro_p=ro_p,
    exp=K_PV[25],
)
# Поры. Распределение для этого керна не измерено (есть только спектры ЯМР T2 без поверхностной
# релаксивности). Берется форма полевого fi_0 (sigma_r = 0.4), а масштаб - из sqrt(k0/m0) относительно
# керна опыта 1 Sutton & Roberts, на котором калибровались d_p и L_k: тогда извилистость та же (5.2).
BEREA = cf.EXPERIMENTS[1]
SCALE = float(np.sqrt((CORE['k0'] / CORE['porosity']) / (BEREA['k0'] / cf.POROSITY)))
PORES = {'r_m': repr(12e-6 * SCALE), 'r_max': repr(40e-6 * SCALE)}
# d_p, L_k - калибровка по опыту 1 Sutton & Roberts (`core_flood.py`, outputs/data/core_flood.json)
CALIBRATION = json.loads(cf.RESULT.read_text(encoding='utf-8'))['best']['1']
D_P, L_K = CALIBRATION['d_p'], CALIBRATION['lk']

D_GRID = (1e-6, 2e-6, 3e-6, 5e-6)       # после переноса: r_pass = d_p/(2*gamma) = 1.25-6.25 мкм
LK_GRID = (3e-5, 1e-4, 3e-4, 1e-3)
# Чувствительность к масштабу пор при d_p, L_k с Berea: от пор Berea (12 мкм, извилистость 20) до
# масштабированных по sqrt(k0/m0) (3.15 мкм, извилистость 5.2). k0 при этом один - меняется извилистость.
R_M_GRID = (8e-6, 6e-6, 4.5e-6)


# --- A. Вязкость ---------------------------------------------------------------------------------------

def kd_multiplier(w_ps):
    """Множитель Кригера-Догерти по объемной доле кристаллов - как `calc_mu_o`."""
    v_p = w_ps / ro_p
    phi = np.minimum(v_p / (v_p + (1.0 - w_ps) / RO_OIL), 0.99 * phi_max)
    return (1.0 - phi / phi_max) ** (-2.5 * phi_max)


def viscosity():
    """Жидкая основа - Аррениус по точкам выше аномальной, затем модель (6.1)-(6.2) + Кригер-Догерти."""
    t, mu = np.array(VISCOSITY).T
    hot = t >= VISC_FIT_T_MIN

    def residual(x):
        return np.log(x[0] * np.exp(x[1] * 1e3 / cf.R_GAS * (1.0 / (t[hot] + 273.15) - 1.0 / 298.15))) - np.log(mu[hot])

    res = least_squares(residual, [40.0, 30.0])
    mu25, e_act = res.x[0], res.x[1] * 1e3
    mu_l = mu25 * np.exp(e_act / cf.R_GAS * (1.0 / (t + 273.15) - 1.0 / 298.15))
    w_ps = cf.LI_W - cf.w_saturated(cf.LI_W, t, MW, M_o, Tm, alpha)
    model = mu_l * kd_multiplier(w_ps)
    print(f'A. Жидкая основа: mu(25 C) = {mu25:.1f} мПа*с, E = {e_act / 1e3:.1f} кДж/моль '
          f'(СКО ln mu выше {VISC_FIT_T_MIN:.0f} C: {np.sqrt(np.mean(res.fun ** 2)):.3f})')
    print('| T, C | опыт, мПа*с | жидкая основа | модель (+КД) | w_ps, % | опыт / модель |')
    for row in zip(t, mu, mu_l, model, w_ps):
        print(f'| {row[0]:.1f} | {row[1]:.1f} | {row[2]:.1f} | {row[3]:.1f} | {100 * row[4]:.1f} | {row[1] / row[3]:.2f} |')
    return dict(T=t.tolist(), mu=mu.tolist(), mu_liquid=mu_l.tolist(), mu_model=model.tolist(),
                w_ps=w_ps.tolist(), mu25=float(mu25), E=float(e_act))


# --- B. Керн ---------------------------------------------------------------------------------------------

def rms_ratio(result):
    """СКО от k(25 C)/k(90 C) при том же PV: так вычитается повреждение, которое есть и выше WAT
    (у Li - накопление смол и асфальтенов в горлах), а модель описывает только парафиновое."""
    pv90, k90 = np.array(K_PV[90]).T
    pts = [(pv, k / np.interp(pv, pv90, k90)) for pv, k in K_PV[25]]
    return cf.rms(result, pts)


def _thin(res, n=250):
    idx = np.unique(np.linspace(0, len(res['pv']) - 1, n).astype(int))
    return {'pv': [res['pv'][i] for i in idx], 'k': [res['k'][i] for i in idx]}


def core_runs(mu25):
    exp = dict(CORE, mu=mu25 * 1e-3)
    for t_c in (45.0, 65.0, 90.0):
        w_sat = float(cf.w_saturated(cf.LI_W, t_c, MW, M_o, Tm, alpha))
        print(f'B. {t_c:.0f} C: взвеси по (6.1)-(6.2) {100 * (cf.LI_W - w_sat):.2f}% - модель дает k/k0 = 1 '
              f'(WAT модели {cf.cloud_point(cf.LI_W, MW, M_o, Tm, alpha):.1f} C)')
    print(f'B. 25 C: взвеси {100 * (cf.LI_W - cf.w_saturated(cf.LI_W, 25.0, MW, M_o, Tm, alpha)):.1f}%, '
          f'масштаб пор {SCALE:.3f} (r_m = {12 * SCALE:.2f} мкм)')

    runs = {}

    def run(name, d_p, lk, extra):
        res = cf.run_case(3, d_p, lk, exp=exp, extra=extra)
        res['rms_ratio'] = rms_ratio(res)
        at = '/'.join(f'{np.interp(x, res["pv"], res["k"]):.2f}' for x in (0.25, 1.0, 3.0, 5.0))
        print(f'| {name} | {d_p * 1e6:.1f} | {lk * 1e3:.2f} | {res["rms"]:.3f} | {res["rms_ratio"]:.3f} | {at} | '
              f'{res["plugged"] if res["plugged"] is not None else "—"} | {res["eta"]:.2f} |', flush=True)
        runs[name] = dict(_thin(res), rms=res['rms'], rms_ratio=res['rms_ratio'], d_p=d_p, lk=lk,
                          plugged=res['plugged'], eta=res['eta'],
                          m_profile=res['m_profile'], k_profile=res['k_profile'])
        return res

    print('| Вариант | d_p, мкм | L_k, мм | СКО | СКО отн. 90 C | k/k0 при 0.25/1/3/5 PV | закупорка, PV | eta |')
    run('перенос с Berea, поры масштабированы', D_P, L_K, PORES)
    run('перенос с Berea, поры Berea', D_P, L_K, {})
    for r_m in R_M_GRID:
        run(f'перенос с Berea, r_m = {r_m * 1e6:.1f} мкм', D_P, L_K,
            {'r_m': repr(r_m), 'r_max': repr(40e-6 * r_m / 12e-6)})
    for d_p in D_GRID:
        for lk in LK_GRID:
            run(f'подбор d_p = {d_p * 1e6:.0f} мкм, L_k = {lk * 1e3:.2f} мм', d_p, lk, PORES)
    fits = {k: v for k, v in runs.items() if k.startswith('подбор')}
    best = min(fits, key=lambda k: fits[k]['rms'])
    best_ratio = min(fits, key=lambda k: fits[k]['rms_ratio'])
    print(f'Лучшее по k/k0: {best} (СКО {fits[best]["rms"]:.3f}); '
          f'по k(25)/k(90): {best_ratio} (СКО {fits[best_ratio]["rms_ratio"]:.3f})')
    return runs, best, best_ratio


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={'width_ratios': (1, 1.1)})
    v = data['viscosity']
    ax1.semilogy(v['T'], v['mu'], 'ko', label='Li et al., охлаждение')
    ax1.semilogy(v['T'], v['mu_liquid'], 'k:', label='жидкая основа (Аррениус)')
    ax1.semilogy(v['T'], v['mu_model'], 'k-', label='модель: + Кригер-Догерти')
    ax1.set_xlabel('T, °C')
    ax1.set_ylabel('μ, мПа·с')
    ax1.legend(fontsize=8)
    ax1.set_title('а) вязкость')

    for t_c, marker in zip((90, 65, 45, 25), ('^', 's', 'D', 'o')):
        pv, k = np.array(K_PV[t_c]).T
        ax2.plot(pv, k, marker, mfc='none', color='k', label=f'опыт {t_c} °C')
    ax2.axhline(1.0, color='0.5', ls=':', label='модель 45-90 °C (взвеси нет)')
    styles = {'перенос с Berea, поры Berea': ('k-', 'перенос d_p, L_k с Berea, r_m = 12 мкм'),
              'перенос с Berea, r_m = 6.0 мкм': ('k-.', 'то же, r_m = 6 мкм'),
              'перенос с Berea, поры масштабированы': ('k--', 'то же, r_m = 3.15 мкм (по √(k/m))'),
              data['best']: ('k:', 'подбор при r_m = 3.15 мкм: ' + data['best'].split('подбор ')[-1])}
    for name, (style, label) in styles.items():
        run = data['runs'][name]
        pv, k = np.array(run['pv']), np.array(run['k'])
        if run['plugged'] is not None:  # драйвер дописывает точку (PV_END, 0) - на графике ее нет, закупорка - крест
            keep = pv <= run['plugged']
            pv, k = pv[keep], k[keep]
            ax2.plot(pv[-1], k[-1], 'kx')
        ax2.plot(pv, k, style, label=label)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('прокачано, PV')
    ax2.set_ylabel('k/k₀')
    ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax2.set_title('б) керн 2, 18.8 мД')
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f'Рисунок: {FIG}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1:2] == ['--plot']:
        plot(json.loads(OUT.read_text(encoding='utf-8')))
        return
    visc = viscosity()
    runs, best, best_ratio = core_runs(visc['mu25'])
    data = dict(viscosity=visc, runs=runs, best=best, best_ratio=best_ratio, scale=SCALE, K_PV=K_PV)
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(f'Записано: {OUT}')
    plot(data)


if __name__ == '__main__':
    main()
