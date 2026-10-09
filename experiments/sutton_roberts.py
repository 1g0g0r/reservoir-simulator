"""Керновые опыты Sutton & Roberts (1974): прокачка парафинистой нефти через керн Berea ниже точки помутнения.

    python experiments/sutton_roberts.py           # подбор по сеткам (прогоны кэшируются) -> results + params.json
    python experiments/sutton_roberts.py --quick   # только итоговые наборы из params.json (~5 мин)
    python experiments/sutton_roberts.py --plot    # только рисунок по results/sutton_roberts.json

Данные - `data/sutton_roberts_{1,2}.json` (опыт, керн, кривые моделей Ring et al. 1994 и Wang & Civan 2005).
Упрощенная модель: d_p = 15 мкм и L_k = 0.3 мм подобраны по опыту 1 (СКО 0.048), опыт 2 - прогноз (СКО 0.246,
закупорка к 4.4 PV вместо плато ~0.3). К ней добавляются механизмы кинетики (`docs/кинетика_осаждения.md`):

  - новое ядро без дополнительных механизмов - что меняет само ядро (скорости по текущему слою);
  - пучок капилляров + вынос отложений (entrainment): равновесие осаждение - вынос дает плато k/k0, как в опыте 2;
  - модель глубинной фильтрации (Civan 2015; обзор 3.6) со степенной проницаемостью и выносом - так опыты
    моделировали Wang & Civan (2005).

Подбор - общий на семейство (оба опыта одним набором, решение пользователя), плюс набор на каждый опыт отдельно
(так подбирали Wang & Civan: коэффициент осаждения у них свой на опыт) и перекрестная проверка: параметры по
одному опыту - прогноз другого. Цель - СКО k/k0 не больше max(0.03, шумовой порог опыта). Кривые чужих моделей
оцениваются той же метрикой по тем же точкам.
"""
import itertools
import sys

import numpy as np

from common import (load, core_constants, run_many, rms_k, noise_floor, mode_from_argv, load_params, save_params,
                    save_results, load_results, lsq, k_residuals, FIGURES)

EXPS = {n: load(f'sutton_roberts_{n}') for n in (1, 2)}
PV_END = 5.2
# Детальный состав с одной группой ('single') - ровно прежняя термодинамика, через новый перенос компонентов
SINGLE = {'wax_characterization': "'single'"}
# Новое ядро без механизмов: snowball с нулевым коэффициентом (флаг только включает ядро `Deposition.py`)
KERNEL = dict(SINGLE, snowball='True')
ENTRAINMENT = dict(KERNEL, entrainment='True')
FILTRATION = dict(SINGLE, deposition_model="'filtration'", perm_model="'power'", entrainment='True')
# Сетки подбора. tau_w в этом керне 0.8-2.6 Па (каналы 12-40 мкм), поэтому ent_tau - в этом диапазоне, ent_rate -
# от «вынос не успевает» до «мгновенный». Показатель n степенного закона: 8-19 по обзору (разд. 4.2) и ~6 по
# парам (m/m0, k/k0) кернов He et al. (2020, `he2020.py`).
ENT_GRID = list(itertools.product((0.3, 0.6, 1.0, 1.5, 2.0), (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)))
FILT_GRID = list(itertools.product((0.005, 0.01, 0.02, 0.04), (0.0, 30.0, 300.0), (6.0, 10.0, 16.0)))
FILT_UCR = 1e-5

# Неизотермическая постановка опыта (режим 'thermal'): нефть входит горячей (54.4 C), охлаждается только выходной
# конец керна (Wang & Civan 2005, с. 320) - кристаллы выпадают в самом керне, а не приходят взвесью. Параметры
# подбираются least_squares прогонами: диаметр кристалла d_p (порог блокирования d_p/(2*gamma), `D_CRYST`) и
# множитель броуновской диффузии (`DIFF_MULT`: вязкость нефти опыта 2 не измерена, 3 мПа*с - допущение Ring),
# у сети пор и горл еще отношение горла к поре. Семейства: общий набор на оба опыта и свой на каждый.
LSQ_FAMILIES = {
    'rate_bundle': dict(flags=KERNEL, names=('D_CRYST', 'DIFF_MULT'), mode='rate',
                        x0=(-4.824, 0.0), lo=(-5.7, -3.0), hi=(-4.3, 3.0), step=(0.02, 0.05), log=(True, True),
                        title='пучок, изотермический керн'),
    'rate_entrainment': dict(flags=ENTRAINMENT, names=('D_CRYST', 'DIFF_MULT', 'ENT_TAU', 'ENT_RATE'), mode='rate',
                             x0=(-4.824, 0.0, 1.5, -3.5), lo=(-5.7, -3.0, 0.1, -6.0), hi=(-4.3, 3.0, 5.0, -1.0),
                             step=(0.02, 0.05, 0.05, 0.05), log=(True, True, False, True),
                             title='пучок + вынос, изотермический керн'),
    'thermal_bundle': dict(flags=dict(SINGLE, snowball='True'), names=('D_CRYST', 'DIFF_MULT'), mode='thermal',
                           x0=(-4.824, 0.0), lo=(-5.7, -3.0), hi=(-4.3, 3.0), step=(0.02, 0.05), log=(True, True),
                           title='пучок, неизотермический керн'),
    'thermal_network': dict(flags=dict(SINGLE, pore_network='True'), names=('D_CRYST', 'DIFF_MULT', 'NET_GAMMA'),
                            mode='thermal',
                            x0=(-5.1, -1.0, 0.4), lo=(-5.7, -3.0, 0.2), hi=(-4.3, 3.0, 0.9), step=(0.02, 0.05, 0.02),
                            log=(True, True, False), title='сеть пор и горл, неизотермический керн'),
}


def lsq_kin(fam, x):
    f = LSQ_FAMILIES[fam]
    kin = {'SNOW_A': 0.0, 'NET_Z': 6.0}
    for name, v, is_log in zip(f['names'], x, f['log']):
        kin[name] = float(10.0 ** v) if is_log else float(v)
    return kin


# Общий набор кинетики (d_p, предел прочности отложения, скорость выноса) и своя вязкость каждой нефти: у обеих
# принята 3 мПа*с (Ring et al.), не измерена - множитель броуновской диффузии у каждой нефти свой (D ~ 1/mu)
VISC_X0 = (-4.806, 1.54, -3.527, 0.0, 0.0)  # lg d_p, tau_кр, lg скорость выноса, lg множитель опыта 1 и 2
VISC_LO, VISC_HI = (-5.7, 0.1, -6.0, -3.0, -3.0), (-4.3, 5.0, -1.0, 3.0, 3.0)
VISC_STEP = (0.02, 0.05, 0.05, 0.05, 0.05)


def visc_kin(x, n):
    return {'SNOW_A': 0.0, 'D_CRYST': float(10 ** x[0]), 'ENT_TAU': float(x[1]), 'ENT_RATE': float(10 ** x[2]),
            'DIFF_MULT': float(10 ** x[3 if n == 1 else 4])}


def fit_visc():
    jobs_of = lambda x: [job('visc', n, ENTRAINMENT, visc_kin(x, n)) for n in (1, 2)]
    resid = lambda res: np.concatenate([k_residuals(r, EXPS[n]['k_pv']) for r, n in zip(res, (1, 2))])
    x, best, _ = lsq('пучок + вынос, общая кинетика и своя вязкость нефти', jobs_of, resid, VISC_X0, VISC_STEP,
                     VISC_LO, VISC_HI)
    return x, dict(zip((1, 2), best))


def fit_lsq(fam, exps=(1, 2), x0=None):
    """least_squares семейства по опытам exps (оба - общий набор). Возвращает x, прогоны по опытам, СКО."""
    f = LSQ_FAMILIES[fam]
    jobs_of = lambda x: [job(fam, n, f['flags'], lsq_kin(fam, x), mode=f['mode']) for n in exps]
    resid = lambda res: np.concatenate([k_residuals(r, EXPS[n]['k_pv']) for r, n in zip(res, exps)])
    x, best, _ = lsq(f'{f["title"]}, опыт {"+".join(map(str, exps))}', jobs_of, resid,
                     f['x0'] if x0 is None else x0, f['step'], f['lo'], f['hi'])
    return x, dict(zip(exps, best))


def exp_dict(n: int) -> dict:
    e = EXPS[n]
    c, o = e['core'], e['oil']
    return dict(length=c['length'], side=c['side'], porosity=c['porosity'], k0=c['k0_darcy'], T=c['T'],
                T_hot=c['T_hot'], P_out=c['P_out'], mu=c['mu_oil'], q=e['q'], pv_end=PV_END,
                w=o['w'], MW=o['MW'], M_o=o['M_o'], ro_o=o['ro_o'], ro_p=o['ro_p'], Tm=o['Tm'], dH=o['dH'])


def job(tag: str, n: int, flags: dict, kin: dict = None, mode: str = 'rate'):
    exp = exp_dict(n)
    constants = dict(core_constants(exp), **flags)
    return (f'exp_sr{n}_{tag}', constants, {'exp': exp, 'mode': mode, 'kin': kin or {'SNOW_A': 0.0}})


def ent_kin(tau, rate):
    return {'SNOW_A': 0.0, 'ENT_TAU': tau, 'ENT_RATE': rate}


def filt_kin(kd, ke, n_pow):
    return {'FILT_KD': kd, 'FILT_KE': ke, 'FILT_UCR': FILT_UCR, 'PERM_N': n_pow, 'SNOW_A': 0.0}


def rms(res, n):
    return rms_k(res, EXPS[n]['k_pv'])


def fit_grid(tag, flags, grid, make_kin):
    """Все наборы сетки на оба опыта: общий набор (min СКО по двум опытам), свой на каждый опыт, перекрестный
    прогноз (набор по опыту 1 -> опыт 2 и наоборот)."""
    jobs, keys = [], []
    for params in grid:
        for n in (1, 2):
            jobs.append(job(tag, n, flags, make_kin(*params)))
            keys.append((params, n))
    res = dict(zip(keys, run_many(jobs)))
    table = []
    for params in grid:
        r1, r2 = rms(res[(params, 1)], 1), rms(res[(params, 2)], 2)
        table.append(dict(params=list(params), rms1=r1, rms2=r2, joint=float(np.sqrt((r1 ** 2 + r2 ** 2) / 2))))
    best = min(table, key=lambda r: r['joint'])
    own = {n: min(table, key=lambda r: r[f'rms{n}']) for n in (1, 2)}
    cross = {'1to2': own[1]['rms2'], '2to1': own[2]['rms1']}
    return dict(table=table, best=best, own=own, cross=cross,
                curves={n: res[(tuple(best['params']), n)] for n in (1, 2)},
                own_curves={n: res[(tuple(own[n]['params']), n)] for n in (1, 2)})


def final_sets(tag, flags, sets, make_kin):
    """Прогоны по итоговым наборам (режим --quick): {'joint': params, 'own1': params, 'own2': params}."""
    jobs, keys = [], []
    for name in ('joint', 'own1', 'own2'):
        for n in (1, 2):
            jobs.append(job(tag, n, flags, make_kin(*sets[name])))
            keys.append((name, n))
    res = dict(zip(keys, run_many(jobs)))
    entry = lambda name: dict(params=list(sets[name]), rms1=rms(res[(name, 1)], 1), rms2=rms(res[(name, 2)], 2))
    best = entry('joint')
    best['joint'] = float(np.sqrt((best['rms1'] ** 2 + best['rms2'] ** 2) / 2))
    own = {1: entry('own1'), 2: entry('own2')}
    return dict(best=best, own=own, cross={'1to2': own[1]['rms2'], '2to1': own[2]['rms1']},
                curves={n: res[('joint', n)] for n in (1, 2)}, own_curves={n: res[(f'own{n}', n)] for n in (1, 2)})


def others():
    """СКО кривых чужих моделей (оцифровка) по тем же точкам и той же метрикой."""
    out = {}
    for n in (1, 2):
        for name, pts in EXPS[n]['models'].items():
            x, y = np.array(pts).T
            out.setdefault(name, {})[n] = rms_k({'pv': x.tolist(), 'k': y.tolist(), 'plugged': None}, EXPS[n]['k_pv'])
    return out


def run(mode: str = 'full') -> dict:
    if mode == 'plot':
        out = load_results('sutton_roberts')
        plot(out)
        return out
    base = run_many([job('legacy', n, {}) for n in (1, 2)] + [job('kernel', n, KERNEL) for n in (1, 2)])
    out = {'legacy': {n: base[n - 1] for n in (1, 2)}, 'kernel': {n: base[n + 1] for n in (1, 2)},
           'noise': {n: noise_floor(EXPS[n]['k_pv']) for n in (1, 2)}, 'others': others()}
    for n in (1, 2):
        print(f'опыт {n}: шум {out["noise"][n]:.3f}; упрощенная модель СКО {rms(out["legacy"][n], n):.3f}; '
              f'новое ядро {rms(out["kernel"][n], n):.3f}', flush=True)
    if mode == 'full':
        ent = fit_grid('ent', ENTRAINMENT, ENT_GRID, ent_kin)
        filt = fit_grid('filt', FILTRATION, FILT_GRID, filt_kin)
        save_params('sutton_roberts', {
            'entrainment': {'joint': ent['best']['params'], 'own1': ent['own'][1]['params'],
                            'own2': ent['own'][2]['params'], 'kin': 'ENT_TAU [Па], ENT_RATE [1/с]'},
            'filtration': {'joint': filt['best']['params'], 'own1': filt['own'][1]['params'],
                           'own2': filt['own'][2]['params'], 'kin': 'FILT_KD [1/с], FILT_KE [1/м], PERM_N'}})
    else:
        p = load_params('sutton_roberts')
        ent = final_sets('ent', ENTRAINMENT, p['entrainment'], ent_kin)
        filt = final_sets('filt', FILTRATION, p['filtration'], filt_kin)
    out['entrainment'], out['filtration'] = ent, filt
    # Неизотермическая постановка, подбор least_squares: общий набор и свой на каждый опыт
    p_lsq = {} if mode == 'full' else load_params('sutton_roberts_lsq')
    out['lsq'] = {}
    for fam in LSQ_FAMILIES:
        e = {}
        for key, exps in (('joint', (1, 2)), ('own1', (1,)), ('own2', (2,))):
            if mode == 'full':
                x, runs = fit_lsq(fam, exps, None if key == 'joint' else e['joint']['x'])
            else:
                x = p_lsq[fam][key]
                runs = dict(zip(exps, run_many([job(fam, n, LSQ_FAMILIES[fam]['flags'], lsq_kin(fam, x),
                                                    mode=LSQ_FAMILIES[fam]['mode'])
                                                for n in exps])))
            e[key] = dict(x=x, kin=lsq_kin(fam, x), rms={n: rms(runs[n], n) for n in exps}, curves=runs)
        # перекрестный прогноз: набор по одному опыту - на другом
        fm = LSQ_FAMILIES[fam]['mode']
        cross = run_many([job(fam, 2, LSQ_FAMILIES[fam]['flags'], lsq_kin(fam, e['own1']['x']), mode=fm),
                          job(fam, 1, LSQ_FAMILIES[fam]['flags'], lsq_kin(fam, e['own2']['x']), mode=fm)])
        e['cross'] = {'1to2': rms(cross[0], 2), '2to1': rms(cross[1], 1)}
        out['lsq'][fam] = e
        print(f'{LSQ_FAMILIES[fam]["title"]}: общий {e["joint"]["kin"]} СКО {e["joint"]["rms"][1]:.3f} / '
              f'{e["joint"]["rms"][2]:.3f}; свой {e["own1"]["rms"][1]:.3f} / {e["own2"]["rms"][2]:.3f}; '
              f'перекрестно 1->2 {e["cross"]["1to2"]:.3f}, 2->1 {e["cross"]["2to1"]:.3f}', flush=True)
    # общая кинетика и своя вязкость каждой нефти
    if mode == 'full':
        xv, runs_v = fit_visc()
    else:
        xv = p_lsq['visc']
        runs_v = dict(zip((1, 2), run_many([job('visc', n, ENTRAINMENT, visc_kin(xv, n)) for n in (1, 2)])))
    out['visc'] = dict(x=xv, kin={n: visc_kin(xv, n) for n in (1, 2)}, rms={n: rms(runs_v[n], n) for n in (1, 2)},
                       curves=runs_v)
    print(f'общая кинетика + своя вязкость: {out["visc"]["kin"]}, СКО {out["visc"]["rms"][1]:.3f} / '
          f'{out["visc"]["rms"][2]:.3f}', flush=True)
    if mode == 'full':
        save_params('sutton_roberts_lsq', dict({fam: {k: out['lsq'][fam][k]['x'] for k in ('joint', 'own1', 'own2')}
                                                for fam in LSQ_FAMILIES}, visc=xv))
    for name, f in (('пучок + вынос', ent), ('глубинная фильтрация', filt)):
        b = f['best']
        print(f'{name}, общий набор {b["params"]}: СКО {b["rms1"]:.3f} / {b["rms2"]:.3f}; свой на опыт: '
              f'{f["own"][1]["rms1"]:.3f} / {f["own"][2]["rms2"]:.3f}; перекрестно 1->2 {f["cross"]["1to2"]:.3f}, '
              f'2->1 {f["cross"]["2to1"]:.3f}', flush=True)
    print('чужие модели:', {k: {n: round(v, 3) for n, v in d.items()} for k, d in out['others'].items()}, flush=True)
    save_results('sutton_roberts', out)
    plot(out)
    return out


def summary(out) -> list:
    """Строки сводной таблицы: (опыт, вариант, СКО опыт 1, СКО опыт 2, что подобрано)."""
    g = lambda d, n: d[n] if n in d else d[str(n)]
    rows = [('шумовой порог опыта', g(out['noise'], 1), g(out['noise'], 2), '-')]
    for name, d in out['others'].items():
        rows.append((name, g(d, 1), g(d, 2), 'по публикации'))
    rows.append(('упрощенная модель (пучок)', rms(g(out['legacy'], 1), 1), rms(g(out['legacy'], 2), 2), 'd_p, L_k по опыту 1'))
    for key, name in (('entrainment', 'пучок + вынос'), ('filtration', 'глубинная фильтрация')):
        f = out[key]
        rows.append((f'{name}, общий набор', f['best']['rms1'], f['best']['rms2'], 'оба опыта'))
        rows.append((f'{name}, свой набор на опыт', g(f['own'], 1)['rms1'], g(f['own'], 2)['rms2'], 'каждый опыт'))
        rows.append((f'{name}, перекрестный прогноз', f['cross']['2to1'], f['cross']['1to2'], 'по другому опыту'))
    if 'visc' in out:
        v = out['visc']
        rows.append(('пучок + вынос, общая кинетика и своя вязкость нефти', g(v['rms'], 1), g(v['rms'], 2),
                     'оба опыта (множитель диффузии - у каждого)'))
    for fam, e in out.get('lsq', {}).items():
        name = LSQ_FAMILIES[fam]['title']
        rows.append((f'{name}, общий набор', g(e['joint']['rms'], 1), g(e['joint']['rms'], 2), 'оба опыта'))
        rows.append((f'{name}, свой набор на опыт', g(e['own1']['rms'], 1), g(e['own2']['rms'], 2), 'каждый опыт'))
        rows.append((f'{name}, перекрестный прогноз', e['cross']['2to1'], e['cross']['1to2'], 'по другому опыту'))
    return rows


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    g = lambda d, n: d[n] if n in d else d[str(n)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, n in zip(axes, (1, 2)):
        pv, k = np.array(EXPS[n]['k_pv']).T
        ax.plot(pv, k, 'o', mfc='white', mec='k', ms=4.5, label='опыт')
        series = [(g(out['legacy'], n), '-', 'упрощенная модель'),
                  (g(out['entrainment']['curves'], n), '-.', 'пучок + вынос (общий набор)'),
                  (g(out['filtration']['own_curves'], n), (0, (5, 1, 1, 1)), 'глубинная фильтрация (свой набор)')]
        if 'lsq' in out and 'rate_entrainment' in out['lsq']:
            e = out['lsq']['rate_entrainment']
            series += [(g(e[f'own{n}']['curves'], n), (0, (1, 1)), 'пучок + вынос, least_squares (свой набор)')]
        if 'visc' in out:
            series += [(g(out['visc']['curves'], n), '--', 'пучок + вынос, общая кинетика, своя вязкость')]
        if 'lsq' in out and 'thermal_network' in out['lsq']:
            e = out['lsq']['thermal_network']
            series += [(g(e[f'own{n}']['curves'], n), (0, (3, 1, 1, 1, 1, 1)), 'сеть пор и горл, горячий вход')]
        for r, style, label in series:
            ax.plot(r['pv'], r['k'], ls=style, label=f'{label}: {rms(r, n):.3f}')
        for (name, pts), ls in zip(EXPS[n]['models'].items(), (':', (0, (4, 2)))):
            x, y = np.array(pts).T
            ax.plot(x, y, ls=ls, lw=1.0, color='0.4',
                    label=f'{name}: {rms_k({"pv": x.tolist(), "k": y.tolist(), "plugged": None}, EXPS[n]["k_pv"]):.3f}')
        ax.set_title(f'опыт {n}', fontsize=9)
        ax.set_xlabel('прокачано, PV')
        ax.set_xlim(0, 5.1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6.3, handlelength=3.0)
    axes[0].set_ylabel('k / k₀')
    fig.tight_layout()
    fig.savefig(FIGURES / 'sutton_roberts.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
