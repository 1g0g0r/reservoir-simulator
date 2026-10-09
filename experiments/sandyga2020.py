"""Охлаждение керна с прокачкой раствора парафина в керосине (Sandyga et al., 2020).

    python experiments/sandyga2020.py           # подбор по сетке (прогоны кэшируются) -> results + params.json
    python experiments/sandyga2020.py --quick   # только итоговый набор из params.json
    python experiments/sandyga2020.py --plot    # только рисунок

Опыт (`data/sandyga2020.json`): 20 % парафина C20-C40 в керосине, песчаник 3 x 5 см, пористость 9 %, 0.5 см^3/мин,
весь стенд охлаждается от 40 C со скоростью 1 C/ч. Градиент давления плавно растет до 34 C и затем за 1.2 C - в 44
раза; томография после опыта: пористость 9 -> 2.1 %, все классы пор потеряли 76-87 % объема. Упрощенная модель с WAT
пористой среды (33.8 C) угадывает начало роста, но дает лишь 4.1 раза и пористость 0.96: захват кристаллов
сортирует поры по размеру (узкие затыкаются, широкие почти не тронуты), а объем кристаллов мал.

Механизмы кинетики (`docs/кинетика_осаждения.md`), которые меняют именно это:
  - кристаллизация на стенках пор (`wax_kinetics`, разд. 13.2): пересыщение раствора при охлаждении уходит на
    стенки всех каналов одинаково - как на томографии;
  - гель-отложение (`deposit_aging`, разд. 13.9): отложение - гель с долей парафина C0, проводящие каналы
    сужаются по объему геля; проводящая пористость и есть то, что видит томография;
  - гель в поре со статическим порогом 2 % (Létoffé et al. 1995).
Подбор: k_wall, C0 и k_cryst по росту градиента (сетка, затем least_squares), пористость после опыта и потери по
классам пор - проверка. WAT пористой среды - тоже параметр: авторы дают 33.8 C, но градиент в опыте растет уже с
34.5 C (в 1.4 раза при 34.5 C, в 2 раза при 34.0 C), а на такой крутой кривой сдвиг начала на 0.5 C меняет
расхождение в разы. Перебираются 33.8, 34.3 и 34.8 C (константа Tm - копия пакета на каждое значение).

Шумовой порог: у кривой почти скачок (в 3.5 раза за 0.3 C), и ошибка оцифровки температуры 0.1 C дает там
ошибку lg(grad) 0.2; вместе с ошибкой 0.3 МПа/м по градиенту это и есть нижняя граница СКО lg (`noise_log`).
"""
import importlib.util
import itertools
import sys

import numpy as np
from scipy.optimize import brentq, least_squares

from common import (load, core_constants, run_many, mode_from_argv, load_params, save_params, save_results,
                    load_results, FIGURES, ROOT, DARCY)

DATA = load('sandyga2020')
SOL, CORE = DATA['solution'], DATA['core']
SINGLE = {'wax_characterization': "'single'"}
GEL2 = dict(SINGLE, gelation='True', wax_viscosity='1', gel_time='300.0', gel_phi='0.02')
KINETICS = dict(GEL2, wax_kinetics='True', deposit_aging='True')
GRID = list(itertools.product((1e-3, 1e-2), (0.1, 0.3, 1.0), (1e-3,)))  # k_wall, C0, k_cryst
WATS = (33.8, 34.3, 34.8)  # WAT пористой среды, C: по авторам и по началу роста градиента
# Модель порового пространства: пучок капилляров или сеть пор и горл (z = 6) с отношением горла к поре
# (docs/кинетика_осаждения.md, разд. 13.13): у сети проводимость рушится на пороге протекания - кандидат на рост
# градиента в 44 раза за 1.2 C
PORE_VARIANTS = (None, 0.3, 0.4)
FIT_STEP = np.array([0.05, 0.05, 0.05])  # шаги производных по lg k_wall, lg C0, lg k_cryst
FIT_LO, FIT_HI = [-6.0, -2.0, -5.0], [0.0, 0.0, 0.0]
POINTS = (35.0, 34.0, 33.5, 33.0, 32.8)
# Скорости охлаждения керна, C/ч, для прогноза: в опыте 1 C/ч, кинетика кристаллизации дает зависимость от скорости
COOLING_RATES = (0.25, 0.5, 2.0, 4.0)  # температуры критерия «в пределах 1.25 раза»
T_ERR, G_ERR = 0.1, 0.3                 # точность оцифровки: C и МПа/м


def _cf():
    spec = importlib.util.spec_from_file_location('core_flood', ROOT / 'experiments' / 'исходная_модель' / 'core_flood.py')
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def kosugi(r, r_m, sigma):
    return np.exp(-0.5 * (np.log(r / r_m) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma * r)


def bins_volume(r, fi):
    """Объем r^2*fi в классах томографии (диаметр d +- 5 мкм)."""
    r, fi = np.asarray(r), np.asarray(fi)
    dr = r[1] - r[0]
    return np.array([np.sum((r ** 2 * fi)[(r >= d / 2 * 1e-6 - 2.5e-6 + 1e-12) & (r < d / 2 * 1e-6 + 2.5e-6)]) * dr
                     for d in DATA['pores']['d_um']])


def pore_fit():
    """Kosugi по томографии до опыта."""
    r = np.linspace(0.05e-6, 60e-6, 4000)
    before = np.array(DATA['pores']['before']) / np.sum(DATA['pores']['before'])

    def shares(r_m, sigma):
        b = bins_volume(r, kosugi(r, r_m, sigma))
        return b / b.sum()

    res = least_squares(lambda x: shares(x[0] * 1e-6, x[1]) - before, [12.0, 0.4], bounds=([3, 0.1], [40, 1.5]))
    return res.x[0] * 1e-6, res.x[1]


def core_k0(r_m, sigma):
    """k0 пучка с извилистостью керна Berea, [м^2]."""
    r = np.linspace(0.0, 40e-6 * r_m / 12e-6, 31)[1:]
    fi = kosugi(r, r_m, sigma)
    return CORE['porosity'] * np.sum(r ** 4 * fi) / (8.0 * DATA['berea_tortuosity'] ** 2 * np.sum(r ** 2 * fi))


def exp_case(wat=None):
    """Постановка опыта; wat - WAT пористой среды, C (по умолчанию - по авторам, 33.8 C): по ней ставится Tm."""
    cf = _cf()
    r_m, sigma = pore_fit()
    wat = SOL['WAT_core'] if wat is None else wat
    tm = brentq(lambda t: float(cf.w_saturated(SOL['wax'], wat, SOL['MW_wax'], SOL['M_kerosene'], t,
                                               SOL['dH'])) - SOL['wax'] + 1e-9, wat + 0.1, 200.0)
    area = np.pi * CORE['diameter'] ** 2 / 4.0
    pv0 = CORE['length'] * area * CORE['porosity']
    exp = dict(length=CORE['length'], side=float(np.sqrt(area)), porosity=CORE['porosity'],
               k0=float(core_k0(r_m, sigma) / DARCY), T=CORE['T_start'], T_end=CORE['T_end'], cooling=CORE['cooling_C_per_s'],
               q=CORE['q'], P_out=1900 * 6894.757, plugged=1e-3,
               pv_end=CORE['q'] * (CORE['T_start'] - CORE['T_end']) / CORE['cooling_C_per_s'] / pv0,
               w=SOL['wax'], MW=SOL['MW_wax'], M_o=SOL['M_kerosene'], dH=SOL['dH'], ro_o=SOL['ro_solution'],
               ro_p=SOL['ro_wax'], mu=SOL['mu_40'], Tm=float(tm))
    extra = {'r_m': repr(float(r_m)), 'sigma_r': repr(float(sigma)), 'r_max': repr(float(40e-6 * r_m / 12e-6))}
    return exp, extra


def curve(res):
    t, k = np.array(res['T_hist']), np.array(res['k'])
    order = np.argsort(t, kind='stable')
    return t[order], 1.0 / np.maximum(k[order], 1e-12)


def noise_log():
    """Нижняя граница СКО lg(grad/grad0) от точности оцифровки по обеим осям."""
    t, g = np.array(DATA['gradient']['points']).T
    slope = np.gradient(np.log10(g), t)
    s = np.sqrt((G_ERR / (g * np.log(10))) ** 2 + (T_ERR * slope) ** 2)
    return float(np.sqrt(np.mean(s ** 2)))


def pore_loss(res):
    """Доля объема, потерянная классами пор томографии: модель (проводящие каналы у входа, в середине и у выхода
    керна - среднее) и опыт."""
    r, fi0 = res['r'], res['fi0']
    b0 = bins_volume(r, fi0)
    model = np.mean([1.0 - bins_volume(r, f) / np.maximum(b0, 1e-300) for f in res['fi_end']], axis=0)
    exp = 1.0 - np.array(DATA['pores']['after']) / np.array(DATA['pores']['before'])
    return model.tolist(), exp.tolist()


def metrics(res):
    t_exp, g_exp = np.array(DATA['gradient']['points']).T
    ratio_exp = g_exp / g_exp[0]
    t, g = curve(res)
    model = np.interp(t_exp, t, g)
    at = [float(np.interp(x, t, g)) for x in POINTS]
    exp_at = [float(np.interp(-x, -t_exp, ratio_exp)) for x in POINTS]
    loss_model, loss_exp = pore_loss(res)
    return dict(rms_log=float(np.sqrt(np.mean((np.log10(model) - np.log10(ratio_exp)) ** 2))),
                rms_k=float(np.sqrt(np.mean((1.0 / model - 1.0 / ratio_exp) ** 2))),
                at=at, exp_at=exp_at, max_factor=float(max(max(a / e, e / a) for a, e in zip(at, exp_at))),
                m_conductive=float(np.mean(res['m_conductive_profile'])), m_total=float(np.mean(res['m_profile'])),
                pore_loss=loss_model, pore_loss_exp=loss_exp)


def kin(k_wall, c0, k_cr, net=None):
    out = {'K_WALL': k_wall, 'AGE_C0': c0, 'K_CRYST': k_cr, 'AGE_RATE': 0.0}
    if net is not None:
        out.update(NET_Z=6.0, NET_GAMMA=net)
    return out


def kin_case(wat, p, net=None, rate=None):
    """Прогон с кристаллизацией на стенках и гель-отложением (net - горло сети пор или None - пучок); rate -
    скорость охлаждения, C/ч, если не как в опыте (прогноз)."""
    exp, extra = exp_case(wat)
    if rate is not None:
        exp['pv_end'] *= exp['cooling'] / (rate / 3600.0)  # тот же интервал температур
        exp['cooling'] = rate / 3600.0
    base = dict(core_constants(exp, dt=2.0), **extra)
    flags = dict(KINETICS, pore_network='True') if net is not None else KINETICS
    # копия - на набор констант, а Time_end зависит от скорости
    name = f'exp_sd_{"net" if net is not None else "kin"}_{wat:g}' + ('' if rate is None else f'_r{rate:g}')
    return (name, dict(base, **flags), {'exp': exp, 'mode': 'ramp', 'kin': kin(*p, net)})


def residuals(res):
    t_exp, g_exp = np.array(DATA['gradient']['points']).T
    t, g = curve(res)
    return np.log10(np.interp(t_exp, t, g)) - np.log10(g_exp / g_exp[0])


def refine(wat, p0, net=None, max_nfev=10):
    """least_squares по lg k_wall, lg C0, lg k_cryst при заданной WAT; якобиан - параллельными прогонами."""
    def runs(xs):
        return [residuals(r) for r in run_many([kin_case(wat, tuple(10.0 ** np.asarray(x)), net) for x in xs])]

    def fun(x):
        r = runs([x])[0]
        print(f'  подбор (WAT {wat} C): x = {np.round(x, 3).tolist()}, СКО lg {np.sqrt(np.mean(r ** 2)):.4f}', flush=True)
        return r

    def jac(x):
        xs = [np.array(x)] + [np.array(x) + FIT_STEP[n] * np.eye(len(x))[n] for n in range(len(x))]
        rs = runs(xs)
        return np.array([(rs[n + 1] - rs[0]) / FIT_STEP[n] for n in range(len(x))]).T

    fit = least_squares(fun, np.log10(p0), jac=jac, bounds=(FIT_LO, FIT_HI), max_nfev=max_nfev,
                        x_scale=FIT_STEP * 5, ftol=1e-3, xtol=1e-3)
    return tuple(float(v) for v in 10.0 ** fit.x)


def run(mode: str = 'full') -> dict:
    if mode == 'plot':
        out = load_results('sandyga2020')
        plot(out)
        return out
    exp, extra = exp_case()
    base = dict(core_constants(exp, dt=2.0), **extra)
    case = {'exp': exp, 'mode': 'ramp'}
    head = run_many([('exp_sd_legacy', base, dict(case, kin={})), ('exp_sd_gel2', dict(base, **GEL2), dict(case, kin={}))])
    if mode == 'full':
        grid = [(wat, p, net) for net in PORE_VARIANTS for wat in WATS for p in GRID]
        res = run_many([kin_case(wat, p, net) for wat, p, net in grid])
        table = [dict(wat=wat, k_wall=p[0], c0=p[1], k_cryst=p[2], net=net, rms_log=metrics(r)['rms_log'])
                 for (wat, p, net), r in zip(grid, res)]
        start_pt = min(table, key=lambda e: e['rms_log'])
        wat, net = start_pt['wat'], start_pt['net']
        p_best = refine(wat, (start_pt['k_wall'], start_pt['c0'], start_pt['k_cryst']), net)
        # тот же подбор при WAT по авторам - для сравнения, что дает сдвиг начала
        start_ref = min((e for e in table if e['wat'] == SOL['WAT_core']), key=lambda e: e['rms_log'])
        net_ref = start_ref['net']
        p_ref = refine(SOL['WAT_core'], (start_ref['k_wall'], start_ref['c0'], start_ref['k_cryst']), net_ref)
        save_params('sandyga2020', {'best': [wat, *p_best], 'net': net, 'wat_authors': [SOL['WAT_core'], *p_ref],
                                    'net_authors': net_ref,
                                    'kin': 'WAT [C], K_WALL [1/с], AGE_C0, K_CRYST [1/с]; AGE_RATE = 0; net - горло '
                                           'сети пор (z = 6) или null - пучок'})
    else:
        params = load_params('sandyga2020')
        wat, p_best, net = params['best'][0], tuple(params['best'][1:]), params.get('net')
        p_ref, net_ref, table = tuple(params['wat_authors'][1:]), params.get('net_authors'), []
    r_best, r_ref, *r_rates = run_many([kin_case(wat, p_best, net), kin_case(SOL['WAT_core'], p_ref, net_ref)]
                                       + [kin_case(wat, p_best, net, rate) for rate in COOLING_RATES])
    out = {'legacy': dict(result=head[0], **metrics(head[0])), 'gel2': dict(result=head[1], **metrics(head[1])),
           'grid': table, 'noise_log': noise_log(), 'porosity_exp': CORE['porosity_after'] / CORE['porosity'],
           'wat_authors': dict(wat=SOL['WAT_core'], k_wall=p_ref[0], c0=p_ref[1], k_cryst=p_ref[2], net=net_ref,
                               result=r_ref, **metrics(r_ref)),
           'kinetics_best': dict(wat=wat, k_wall=p_best[0], c0=p_best[1], k_cryst=p_best[2], net=net, result=r_best,
                                 **metrics(r_best))}
    out['best'] = {k: v for k, v in out['kinetics_best'].items() if k != 'result'}
    out['cooling'] = cooling_rates(dict(zip(COOLING_RATES, r_rates)), r_best, p_best[2])
    out['wat_rate'] = wat_rate(wat, p_best[2])
    print(f'шумовой порог СКО lg {out["noise_log"]:.3f}; пористость после опыта {out["porosity_exp"]:.3f}', flush=True)
    for name, e in (('прежняя', out['legacy']), ('гель 2 %', out['gel2']),
                    (f'кинетика, WAT {SOL["WAT_core"]} C', out['wat_authors']),
                    (f'кинетика, WAT {wat} C (подбор)', out['kinetics_best'])):
        print(f'{name}: СКО lg {e["rms_log"]:.3f}, grad/grad0 {[round(x, 1) for x in e["at"]]} '
              f'(опыт {[round(x, 1) for x in e["exp_at"]]}), проводящая пористость {e["m_conductive"]:.2f}, '
              f'полная {e["m_total"]:.2f}', flush=True)
    save_results('sandyga2020', out)
    plot(out)
    return out


def cooling_rates(runs: dict, r_best, k_cr: float) -> dict:
    """Прогноз: рост градиента при разных скоростях охлаждения (параметры подбора при 1 C/ч).

    Показатели: температура, при которой градиент вырос в 2 и в 10 раз, и рост к концу охлаждения (32.8 C).
    Чем быстрее охлаждение, тем дальше пересыщение отстает от равновесия (k_cryst конечна) и тем ниже температура
    заметного роста; число Дамкелера Da = k_cryst*dT_W/v_cool - отношение времени охлаждения на интервале
    выпадения к времени кристаллизации."""
    rate_exp = CORE['cooling_C_per_s'] * 3600.0
    runs = {**runs, rate_exp: r_best}
    out = {}
    for rate in sorted(runs):
        t, g = curve(runs[rate])
        order = np.argsort(-t)  # по убыванию температуры - по времени

        def at_ratio(x):
            above = np.nonzero(g[order] >= x)[0]
            return float(t[order][above[0]]) if above.size else None
        out[f'{rate:g}'] = dict(rate=rate, t2=at_ratio(2.0), t10=at_ratio(10.0), final=float(g[order][-1]),
                                damkohler=float(k_cr / (rate / 3600.0)),  # на 1 C интервала выпадения
                                curve=[t.tolist(), g.tolist()])
    return out


def _appearance(cf, w, tm, eq, v, k_cr, thr):
    """Температура появления кристаллов в объеме при охлаждении со скоростью v [C/с]: доля кристаллов
    релаксирует к равновесной с k_cr (как `relax_exp` в `components_equation`), прибор их видит с доли thr."""
    dt, t, ws = 0.02 / v, eq + 0.5, 0.0
    while t > eq - 30.0:
        t -= v * dt
        w_eq = w - float(cf.w_saturated(w, t, SOL['MW_wax'], SOL['M_kerosene'], tm, SOL['dH']))
        if w_eq > ws:
            ws = w_eq + (ws - w_eq) * np.exp(-k_cr * dt)
        if ws >= thr:
            return t  # шаг 0.02 C - точнее оцифровки
    return eq - 30.0


def wat_rate(wat: float, k_cr: float) -> dict:
    """Проверка кинетики без керна: WAT раствора от скорости охлаждения (Struchkov, Rogachev, 2017).

    Подбираются порог видимости кристаллов реометром thr (доля массы раствора) и сдвиг равновесной WAT 30 %-го
    раствора dT: идеальный раствор с Tm по 20 %-му раствору дает для 30 %-го равновесную WAT выше опыта. Запаздывание
    WAT при первом порядке кинетики ~ sqrt(thr*v/k_cr), поэтому опыт задает отношение thr/k_cr: проверка в том,
    что при k_cr подбора по керну порог правдоподобен (десятые доли процента кристаллов). Затем той же моделью -
    объемная WAT 20 %-го раствора при скорости реометра против измеренной (30 C) и при 1 C/ч керна."""
    cf, rr = _cf(), DATA['wat_cooling_rate']
    v_exp, wat_exp = np.array(rr['points']).T
    tm = brentq(lambda t: float(cf.w_saturated(SOL['wax'], wat, SOL['MW_wax'], SOL['M_kerosene'], t, SOL['dH']))
                - SOL['wax'] + 1e-9, wat + 0.1, 200.0)
    eq30 = cf.cloud_point(rr['w'], SOL['MW_wax'], SOL['M_kerosene'], tm, SOL['dH'])

    out = {}
    for name, k in (('sandyga', k_cr), ('li2024', 10.0 ** load_params('li2024_cold')['x'][2])):
        # модель ступенчатая (шаг по температуре), поэтому порог - перебором, сдвиг при нем - средняя невязка
        best = None
        for thr in 10.0 ** np.arange(-5.0, -0.95, 0.05):
            m = np.array([_appearance(cf, rr['w'], tm, eq30, v, k, thr) for v in v_exp])
            shift = float(np.mean(wat_exp - m))
            rms = float(np.sqrt(np.mean((m + shift - wat_exp) ** 2)))
            if best is None or rms < best['rms']:
                best = dict(k_cryst=k, thr=float(thr), shift=shift, model=(m + shift).tolist(), rms=rms)
        out[name] = best
    s = out['sandyga']
    w20 = lambda v: _appearance(cf, SOL['wax'], tm, wat, v, k_cr, s['thr'])
    rate_core = CORE['cooling_C_per_s']
    out.update(v=v_exp.tolist(), wat_exp=wat_exp.tolist(), eq30=eq30, eq20=wat,
               bulk20_rheometer=w20(rr['bulk_rate']), bulk20_core=w20(rate_core),
               bulk20_shift=w20(rate_core) - w20(rr['bulk_rate']),
               exp_span=float(wat_exp[0] - wat_exp[-1]))
    print(f'WAT от скорости охлаждения: k_cryst {k_cr:.2e} 1/с -> порог {s["thr"] * 100:.2f} % кристаллов, '
          f'СКО {s["rms"]:.2f} C; k_cryst Li {out["li2024"]["k_cryst"]:.2e} -> порог '
          f'{out["li2024"]["thr"] * 100:.1f} %, СКО {out["li2024"]["rms"]:.2f} C; 20 %-й раствор: '
          f'{out["bulk20_rheometer"]:.1f} C при {rr["bulk_rate"]} C/с (опыт {SOL["WAT_bulk"]} C), '
          f'{out["bulk20_core"]:.1f} C при 1 C/ч', flush=True)
    return out


def summary(out) -> list:
    """Строки сводной таблицы: (вариант, СКО lg(grad/grad0), СКО k/k0, наибольшее расхождение в 5 точках, раз,
    проводящая пористость)."""
    pore = lambda e: 'пучок' if e.get('net') is None else f'сеть, горло {e["net"]:g}'
    rows = [('шумовой порог опыта', out['noise_log'], None, None, out['porosity_exp'])]
    for name, e in (('упрощенная модель', out['legacy']), ('гель, порог 2 %', out['gel2']),
                    (f'стенки + гель-отложение, {pore(out["wat_authors"])}, WAT {out["wat_authors"]["wat"]:g} °C '
                     f'(авторы)', out['wat_authors']),
                    (f'стенки + гель-отложение, {pore(out["best"])}, WAT {out["best"]["wat"]:g} °C (подбор)',
                     out['best'])):
        rows.append((name, e['rms_log'], e.get('rms_k'), e['max_factor'], e['m_conductive']))
    return rows


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    best = out['kinetics_best']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={'width_ratios': [1.35, 1]})
    t_exp, g_exp = np.array(DATA['gradient']['points']).T
    ax.plot(t_exp, g_exp / g_exp[0], 'o', mfc='white', mec='k', ms=4.5, label='опыт')
    for e, style, label in ((out['legacy'], '-', 'упрощенная модель'), (out['gel2'], '--', 'гель, порог 2 %'),
                            (out['wat_authors'], ':', f'стенки + гель, WAT {out["wat_authors"]["wat"]:g} °C'),
                            (best, '-.', f'стенки + гель, WAT {best["wat"]:g} °C (подбор)')):
        t, g = curve(e['result'])
        ax.plot(t, g, ls=style, label=f'{label}: СКО lg {e["rms_log"]:.2f}')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('T, °C')
    ax.set_ylabel('∇p / ∇p(40 °C)')
    ax.legend(fontsize=6.8, handlelength=3.0)
    d = np.array(DATA['pores']['d_um'])
    w = 2.2
    ax2.bar(d - w, best['pore_loss_exp'], width=2 * w, color='0.75', edgecolor='k', label='томография')
    ax2.bar(d + w, best['pore_loss'], width=2 * w, color='#2a78d6', label='модель (проводящие каналы)')
    ax2.plot(d, out['legacy']['pore_loss'], 'x', color='k', label='упрощенная модель')
    ax2.set_xlabel('диаметр пор, мкм')
    ax2.set_ylabel('потерянная доля объема класса')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'sandyga2020.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
