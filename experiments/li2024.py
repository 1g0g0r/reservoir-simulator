"""Керн 2 нефти Жетыбая (Li et al., 2024): ступенчатое охлаждение 90 -> 65 -> 45 -> 25 C с прокачкой нефти.

    python experiments/li2024.py           # подбор прогонами (least_squares, кэш) -> results + params.json
    python experiments/li2024.py --quick   # только итоговый набор из params.json (4 прогона, ~3 мин)
    python experiments/li2024.py --plot    # только рисунок

Опыт: один керн, на каждой ступени нефть той же температуры прокачивается до стабилизации перепада, k/k0
нормирован на начало ступени (`data/li2024.json`). Расчет повторяет протокол целиком в одном прогоне (режим
'stages' в `core_runner.py`): к 25 C керн уже поврежден предыдущими ступенями. Прежнее сравнение
(`experiments/исходная_модель/VALIDATION.md`) начинало каждую ступень с чистого керна.

Повреждение выше WAT (0.77 при 90 C, 0.43 при 65 C, 0.33 при 45 C) парафином не объясняется: парафин при этих
температурах растворен. Li et al. связывают его со смолами и асфальтенами (5.52 и 0.91 % масс.). Баланс массы
исключает равномерный слой адсорбции (docs/кинетика_осаждения.md, разд. 13.7): удержание идет в горлах, где
малый объем дает большую потерю проводимости (обзор 4.1-4.2). Поэтому удержанные по Ленгмюру (K(T) по
Вант-Гоффу) смолы и асфальтены умножают проводимость пучка на функцию повреждения k/k0 = 1 - sigma/sigma_max
(Civan 2015), а парафин по-прежнему сужает и затыкает каналы пучка (d_p, L_k по Sutton & Roberts).

Подбор (семейство «выше WAT»): предельное удержание G_s, константа Ленгмюра K_ref, энтальпия dH и скорость
k_ads - по ступеням 90, 65 и 45 C; ступень 25 C (парафин и гель) - прогноз. Начальная точка - равновесный подбор
плато (`plateau_fit`: три плато при трех параметрах Ленгмюра подбираются точно), затем least_squares прогонами
всего протокола (`dynamic_fit`): удержание идет фронтом от входа, подвод асфальтенов ограничен, а при 45 C
добавляется парафин - равновесная оценка этого не видит. Проверки переносимости - аналитические, по двум плато:
линейная изотерма по 90 и 45 C предсказывает для 65 C отношение 0.62 против 0.43, а по 90 и 65 C с физически
допустимой dH - полную закупорку при 45 C. Удержание, следовательно, близко к насыщению, а его температурная
зависимость сильнее адсорбционной (dH ~ -100 кДж/моль).
"""
import json
import math
import sys

import numpy as np
from scipy.optimize import brentq, least_squares

from common import (load, core_constants, run_many, rms_k, noise_floor, mode_from_argv, load_params, save_params,
                    save_results, load_results, FIGURES)

DATA = load('li2024')
STAGES = [[90.0, 5.0], [65.0, 5.0], [45.0, 5.0], [25.0, 5.0]]
R = 8.31446261815324
RO_AD = 1200.0     # плотность удержанного материала асфальтены + смолы, [кг/м^3] (constants.ro_asph_dep)
SIGMA_MAX = 0.02   # sigma_max в функции повреждения, доля m0 (масштаб; вместе с G_max подбирается их отношение)


def exp_dict() -> dict:
    from paraphin.constants import MW, M_o, Tm, alpha, ro_p
    c, o = DATA['core'], DATA['oil']
    return dict(length=c['length'], side=float(np.sqrt(np.pi / 4.0) * c['diameter']), porosity=c['porosity'],
                k0=c['k0_darcy'], T=STAGES[0][0], P_out=101325.0, q=c['q'], pv_end=sum(p for _, p in STAGES),
                stages=STAGES, w=o['wax'], MW=MW, M_o=M_o, Tm=Tm, dH=alpha, ro_o=o['density'], ro_p=ro_p,
                mu=46.6e-3)


def constants(flags: dict) -> dict:
    """Керн Li: вязкость жидкой основы - Аррениус по точкам выше 40 C (mu(25 C) = 46.6 мПа*с, E = 21.3 кДж/моль,
    `experiments/calibrate.py`), опорная точка - 25 C, а не температура первой ступени."""
    c = core_constants(exp_dict())
    c.update({'mu_o_ref': '46.6e-3', 'T_mu_ref': '25.0', 'E_activation': '21342.75'})
    c.update(flags)
    return c


SCN = {'wax_components': 'True', 'wax_viscosity': '1', 'gelation': 'True', 'gel_time': '300.0',
       'asphaltenes': 'True'}
# Удержание смол и асфальтенов в горлах (адсорбция + функция повреждения), парафин - пучок капилляров с d_p, L_k
# по Sutton & Roberts, гель с объемным пределом текучести
RETENTION = dict(SCN, adsorption='True')


def cumulative_plateaus():
    """Плато ступеней в опыте и накопленная проницаемость (произведение плато): k/k0 от исходного керна."""
    plateaus = {t: DATA['k_pv'][str(int(t))][-1][1] for t, _ in STAGES}
    cum, out = 1.0, {}
    for t, _ in STAGES:
        cum *= plateaus[t]
        out[t] = cum
    return plateaus, out


def surface_area(m0: float) -> float:
    """Удельная поверхность породы - пучка fi_0, [1/м] (`paraphin.surf_0` при m0)."""
    from paraphin import fi_0, w1_cv, w2_cv
    return 2.0 * m0 * float((w1_cv * fi_0).sum()) / float((w2_cv * fi_0).sum())


def sigma_rel(t_c, g_s, k_ref, d_h, m0, resin_ratio=0.5, t_ref=70.0):
    """Равновесное удержание при t_c в долях sigma_max*m0 (функция повреждения: k/k0 = 1 - это)."""
    a_v = surface_area(m0)
    k_l = k_ref * math.exp(-d_h / R * (1.0 / (t_c + 273.15) - 1.0 / (t_ref + 273.15)))
    c_a, c_r = DATA['oil']['asphaltenes'], DATA['oil']['resins']
    g = g_s * a_v * (k_l * c_a / (1.0 + k_l * c_a) + resin_ratio * k_l * c_r / (1.0 + k_l * c_r))
    return g / RO_AD / (SIGMA_MAX * m0)


D_H_B = -5.0e4  # [Дж/моль]: плато 90 и 65 C совместимы с Ленгмюром только при -dH >= 43.5 кДж/моль (docs/)


def plateau_fit(temps, d_h_fixed=None):
    """G_s, K_ref (и dH, если не задана) по накопленным плато ступеней temps: равновесие Ленгмюра, функция
    повреждения линейна. При двух плато и трех параметрах задача недоопределена - тогда dH фиксируется."""
    m0 = DATA['core']['porosity']
    _, cum = cumulative_plateaus()

    def unpack(x):
        return 10 ** x[0], 10 ** x[1], (d_h_fixed if d_h_fixed is not None else -x[2] * 1e4)

    def resid(x):
        g_s, k_ref, d_h = unpack(x)
        return [sigma_rel(t, g_s, k_ref, d_h, m0) - (1.0 - cum[t]) for t in temps]

    best = None
    for x0 in ([-5.0, 0.0, 3.0], [-4.0, 1.0, 5.0], [-5.5, 2.0, 2.0], [-3.5, -1.0, 6.0]):
        x0 = x0[:2] if d_h_fixed is not None else x0
        lo = [-9.0, -4.0] + ([] if d_h_fixed is not None else [0.0])
        hi = [0.0, 6.0] + ([] if d_h_fixed is not None else [15.0])
        res = least_squares(resid, x0, bounds=(lo, hi))
        if best is None or res.cost < best.cost:
            best = res
    g_s, k_ref, d_h = unpack(best.x)
    pred = {t: 1.0 - sigma_rel(t, g_s, k_ref, d_h, m0) for t, _ in STAGES}
    return dict(ads_gmax=g_s, ads_K=k_ref, ads_dH=d_h, cumulative_model=pred, cost=float(best.cost))


def stage_curves(res):
    """Разбивка прогона 'stages' на кривые ступеней: {T: (pv от начала ступени, k/k0 ступени)}."""
    pv, k, st = np.array(res['pv']), np.array(res['k']), np.array(res['stage'])
    out = {}
    for n, (t, _) in enumerate(STAGES):
        sel = st == n
        if sel.any():
            x = pv[sel] - pv[sel][0]
            out[t] = (x, k[sel])
    return out


def stage_rms(res):
    curves, out = stage_curves(res), {}
    for t, _ in STAGES:
        if t in curves:
            x, k = curves[t]
            out[t] = rms_k({'pv': x.tolist(), 'k': k.tolist(), 'plugged': None}, DATA['k_pv'][str(int(t))])
    return out


FIT_STAGES = STAGES[:3]  # калибровка - ступени выше WAT; 25 C (парафин) - прогноз того же набора
# Параметры динамического подбора: lg G_s, lg K_ref, -dH/10^4 [Дж/моль], lg k_ads(T_ref); шаги производных и границы
FIT_STEP = np.array([0.02, 0.02, 0.05, 0.02])
FIT_LO, FIT_HI = [-7.0, -2.0, 0.0, -6.0], [-2.0, 6.0, 15.0, -1.0]


def fit_kin(x) -> dict:
    return {'ADS_GMAX': 10 ** x[0], 'ADS_K': 10 ** x[1], 'ADS_DH': -x[2] * 1e4, 'ADS_RATE': 10 ** x[3],
            'PERM_SMAX': SIGMA_MAX, 'PERM_BETA': 1.0, 'PERM_GAMMA': 1.0}


def fit_case(x, stages) -> dict:
    return {'mode': 'stages', 'exp': dict(exp_dict(), stages=stages), 'kin': fit_kin([float(v) for v in x])}


def stage_residuals(res, stages):
    """Модель минус опыт во всех точках ступеней stages (кроме PV = 0). История прорежена, и последняя точка
    ступени может лечь чуть раньше ее конца - там берется последнее значение; после закупорки модель - ноль."""
    curves, out = stage_curves(res), []
    for t, _ in stages:
        pts = np.array([p for p in DATA['k_pv'][str(int(t))] if p[0] > 0.0])
        if t in curves:
            x, k = curves[t]
            model = np.interp(pts[:, 0], x, k)
            if res['plugged'] is not None and t == list(curves)[-1]:  # ступень, на которой керн закупорен
                model = np.where(pts[:, 0] <= x[-1] + 1e-12, model, 0.0)
        else:
            model = np.zeros(len(pts))
        out.append(model - pts[:, 1])
    return np.concatenate(out)


def dynamic_fit(x0, max_nfev=14):
    """Подбор G_s, K_ref, dH, k_ads прогонами ступенчатого протокола (а не аналитическим равновесием): удержание
    идет фронтом от входа, подвод асфальтенов ограничен, а при 45 C к нему добавляется парафин - равновесная
    оценка плато этого не видит. Якобиан - разностный, все его прогоны идут параллельно (`run_many`), параметры
    кинетики меняются без перекомпиляции. Прогоны кэшируются: повторный запуск подбор не пересчитывает."""
    consts = constants(RETENTION)

    def runs(xs):
        res = run_many([('exp_li_retA', consts, fit_case(x, FIT_STAGES)) for x in xs])
        return [stage_residuals(r, FIT_STAGES) for r in res]

    def fun(x):
        r = runs([x])[0]
        print(f'  подбор: x = {np.round(x, 3).tolist()}, СКО {np.sqrt(np.mean(r ** 2)):.4f}', flush=True)
        return r

    def jac(x):
        xs = [np.array(x)] + [np.array(x) + FIT_STEP[n] * np.eye(len(x))[n] for n in range(len(x))]
        rs = runs(xs)
        return np.array([(rs[n + 1] - rs[0]) / FIT_STEP[n] for n in range(len(x))]).T

    # Остановка по относительному изменению СКО 1e-3: мельче шумового порога опыта (0.005-0.007) на два порядка
    return least_squares(fun, x0, jac=jac, bounds=(FIT_LO, FIT_HI), max_nfev=max_nfev, x_scale=FIT_STEP * 5,
                         ftol=1e-3, xtol=1e-3, gtol=1e-6)


def run(mode: str = 'full') -> dict:
    if mode == 'plot':
        out = load_results('li2024')
        plot(out)
        return out
    plateaus, cum = cumulative_plateaus()
    fit_all = plateau_fit([90.0, 65.0, 45.0])
    fit_cross = plateau_fit([90.0, 45.0])
    fit_b = plateau_fit([90.0, 65.0], d_h_fixed=D_H_B)
    print('плато ступеней', plateaus, 'накопленная k', {t: round(v, 3) for t, v in cum.items()})
    print('равновесный подбор по трем плато:', {k: v for k, v in fit_all.items() if k != 'cumulative_model'})
    print('по 90 и 45 C -> прогноз 65 C: накопленная k', round(fit_cross['cumulative_model'][65.0], 3),
          'опыт', round(cum[65.0], 3), flush=True)
    print(f'по 90 и 65 C при dH = {D_H_B / 1e3:.0f} кДж/моль -> накопленная k при 45 C',
          round(fit_b['cumulative_model'][45.0], 3), 'опыт', round(cum[45.0], 3), flush=True)

    base = {'mode': 'stages', 'exp': exp_dict()}
    x_eq = np.array([math.log10(fit_all['ads_gmax']), math.log10(fit_all['ads_K']), -fit_all['ads_dH'] / 1e4,
                     math.log10(4e-4)])
    if mode == 'full':
        fit = dynamic_fit(x_eq)
        x_best, nfev = [float(v) for v in fit.x], int(fit.nfev)
        save_params('li2024', {'x': x_best, 'kin': fit_kin(x_best),
                               'x_names': 'lg ADS_GMAX [кг/м^2], lg ADS_K, -ADS_DH/1e4 [Дж/моль], lg ADS_RATE [1/с]'})
    else:
        x_best, nfev = load_params('li2024')['x'], 0
    res = run_many([('exp_li_legacy', constants({}), dict(base, kin={})),
                    ('exp_li_scn', constants(SCN), dict(base, kin={})),
                    ('exp_li_retA', constants(RETENTION), fit_case(x_eq, STAGES)),
                    ('exp_li_retA', constants(RETENTION), fit_case(x_best, STAGES))])
    out = {'plateaus': plateaus, 'cumulative': cum, 'fit_all': fit_all, 'fit_cross': fit_cross, 'fit_b': fit_b,
           'noise': {str(int(t)): noise_floor(DATA['k_pv'][str(int(t))]) for t, _ in STAGES},
           'legacy': res[0], 'scn': res[1],
           'equilibrium': dict(x=x_eq.tolist(), kin=fit_kin(x_eq), rms=stage_rms(res[2]), result=res[2]),
           'dynamic': dict(x=x_best, kin=fit_kin(x_best), rms=stage_rms(res[3]), result=res[3],
                           nfev=nfev, calibrated=[t for t, _ in FIT_STAGES])}
    for name, key in (('прежняя', 'legacy'), ('4 группы + гель', 'scn')):
        print(name, {t: round(v, 3) for t, v in stage_rms(out[key]).items()}, flush=True)
    for name, key in (('удержание, равновесный подбор плато', 'equilibrium'), ('удержание, подбор прогонами', 'dynamic')):
        e = out[key]
        print(name, {t: round(v, 3) for t, v in e['rms'].items()},
              {k: float(f'{v:.4g}') for k, v in e['kin'].items()}, flush=True)
    save_results('li2024', out)
    plot(out)
    return out


def summary(out) -> list:
    """Строки сводной таблицы: (вариант, СКО по ступеням 90, 65, 45, 25 C)."""
    temps = [str(float(t)) for t, _ in STAGES]
    get = lambda d, t: d.get(t, d.get(float(t), float('nan'))) if isinstance(d, dict) else float('nan')
    rows = [('шумовой порог опыта', [out['noise'][str(int(float(t)))] for t in temps])]
    for name, key in (('прежняя модель', 'legacy'), ('4 группы парафина + гель', 'scn')):
        r = stage_rms(out[key])
        rows.append((name, [r.get(float(t), float('nan')) for t in temps]))
    for name, key in (('удержание, плато по равновесию', 'equilibrium'), ('удержание, подбор прогонами', 'dynamic')):
        rows.append((name, [get(out[key]['rms'], t) for t in temps]))
    return rows


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, (t, _) in zip(axes, STAGES):
        pv, k = np.array(DATA['k_pv'][str(int(t))]).T
        ax.plot(pv, k, 'o', mfc='white', mec='k', ms=4, label='опыт')
        for r, style, label in ((out['legacy'], '-', 'прежняя'), (out['scn'], '--', '4 группы + гель'),
                                (out['equilibrium']['result'], ':', 'удержание, плато по равновесию'),
                                (out['dynamic']['result'], '-.', 'удержание, подбор прогонами')):
            curves = stage_curves(r)
            if float(t) in curves or t in curves:
                x, y = curves[float(t)] if float(t) in curves else curves[t]
                ax.plot(x, y, style, label=label)
        ax.set_title(f'{t:.0f} °C', fontsize=9)
        ax.set_xlabel('PV ступени')
        ax.set_xlim(0, 5.1)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('k / k₀ ступени')
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / 'li2024.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
