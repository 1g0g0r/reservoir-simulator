"""Общий набор параметров кинетики осаждения на все керновые опыты с парафином (совместная калибровка).

    python experiments/unified.py           # подбор (часы; прогоны кэшируются) -> results/unified.json, params.json
    python experiments/unified.py --quick   # только прогоны с сохраненным набором и перекрестные прогнозы

Подборы по семействам (`sutton_roberts.py`, `li2024.py`, `sandyga2020.py`) дают каждому опыту свой набор: 15
параметров на четыре кривые. Здесь у всех кернов одна модель - одни и те же механизмы (пучок капилляров, перенос
кристаллов к стенкам, кристаллизация в объеме и на стенках, вынос отложений; удержание смол и асфальтенов - там,
где они есть в нефти) и одни и те же числа. Различаются только величины, которые физически зависят от системы и в
опыте не измерены:

  - вязкость нефтей Sutton & Roberts (у Ring et al. принята 3 мПа*с для обеих): при заданном расходе она входит
    только в броуновскую диффузию D_p ~ 1/mu, поэтому подбирается в пределах 1-30 мПа*с;
  - константа кристаллизации в объеме: у нефти (Li et al.) и у модельного раствора парафина в керосине (Sandyga et
    al.) своя - зарождение в нефти гетерогенное, на асфальтенах и смолах. Значение для раствора независимо
    проверяется опытом Struchkov & Rogachev (WAT от скорости охлаждения, `sandyga2020.wat_rate`).

Множителя броуновской диффузии нет (DIFF_MULT = 1): у Li et al. вязкость измерена. Проверена и отвергнута гипотеза,
что подбор по ступени 25 C (DIFF_MULT = 0.10) отражает затрудненную диффузию в гелированной нефти - диффузия по
кажущейся вязкости mu_o = mu_p/Phi (флаг `gel_hindered_diffusion`, обобщенное соотношение Стокса - Эйнштейна): Phi в
керне близок к 1, и СКО ступени 25 C с флагом 0.22 против 0.23 без него (`hypothesis_checks`). Удержание
смол и асфальтенов (Li et al., 90-45 C) - из подбора `li2024.py`: другой нефти с ними нет. WAT пористой среды у
Sandyga et al. - из `params.json` (обоснование - `sandyga2020.py`), отложение - сплошной кристалл (C0 = 1).

Невязки опытов нормируются на их цель max(0.05, шумовой порог) по k/k0, у Sandyga et al. - на порог оцифровки
lg(grad/grad0). Перекрестная проверка: подбор без одного опыта и прогноз этого опыта; его собственный параметр
(вязкость, k_cr) берется априорным.
"""
import math
import sys

import numpy as np

import li2024 as li
import sandyga2020 as sd
import sutton_roberts as sr
from common import (run_many, lsq, k_residuals, load_params, save_params, save_results, mode_from_argv,
                    fit_metrics, k_model_at)

MU_RING = 3e-3  # вязкость нефтей Sutton & Roberts по Ring et al. (1994), [Па*с]: к ней отнесен DIFF_MULT
FLAGS = {'entrainment': 'True'}
# x: lg D_CRYST [м], ENT_TAU [Па], lg ENT_RATE [1/с], lg mu_SR1, lg mu_SR2 [Па*с], lg K_CRYST нефти, lg K_WALL,
# lg K_CRYST раствора [1/с]
NAMES = ('lg D_CRYST', 'ENT_TAU', 'lg ENT_RATE', 'lg mu SR1', 'lg mu SR2', 'lg K_CRYST нефть', 'lg K_WALL',
         'lg K_CRYST раствор')
# старт - прежние подборы: d_p, вынос и вязкости - «общая кинетика, своя вязкость» Sutton & Roberts, k_cr нефти -
# ступень 25 C Li et al., k_wall и k_cr раствора - Sandyga et al.
X0 = (-4.812, 1.649, -3.704, -3.0, -2.004, -1.798, -2.967, -3.435)
LO = (-5.7, 0.1, -6.0, -3.0, -3.0, -5.0, -6.0, -5.0)
HI = (-4.3, 5.0, -1.0, math.log10(3e-2), math.log10(3e-2), -0.5, 0.0, -1.0)
STEP = (0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
SIG_K, SIG_LG = 0.05, sd.noise_log()  # цели: k/k0 и lg(grad/grad0)
EXPS = ('sr1', 'sr2', 'li', 'sd')


def _li_ret():
    p = load_params('li2024')
    return p[p['best']]['kin']


def jobs_of(x, exps=EXPS, rate=None):
    """Задания опытов exps при наборе x; rate - скорость охлаждения керна Sandyga et al., C/ч (прогноз)."""
    lg_d, tau, lg_ent, lg_mu1, lg_mu2, lg_kco, lg_kw, lg_kcs = x
    ent = {'ENT_TAU': float(tau), 'ENT_RATE': float(10 ** lg_ent)}
    jobs = []
    for n, lg_mu in ((1, lg_mu1), (2, lg_mu2)):
        if f'sr{n}' in exps:
            kin = dict(ent, SNOW_A=0.0, D_CRYST=float(10 ** lg_d), DIFF_MULT=MU_RING / 10 ** lg_mu)
            jobs.append(sr.job('uni', n, dict(sr.ENTRAINMENT, **FLAGS), kin))
    if 'li' in exps:
        gel_time = load_params('li2024_cold')['gel_time']
        _, consts, case = li.cold_job(gel_time, [lg_d, 0.0, lg_kco], _li_ret())
        case['kin'].update(ent, K_WALL=float(10 ** lg_kw))
        jobs.append(('exp_li_uni', dict(consts, **FLAGS), case))
    if 'sd' in exps:
        wat = load_params('sandyga2020')['best'][0]
        _, consts, case = sd.kin_case(wat, (10 ** lg_kw, 1.0, 10 ** lg_kcs), rate=rate)
        case['kin'].update(ent, D_CRYST=float(10 ** lg_d))
        jobs.append(('exp_sd_uni' + ('' if rate is None else f'_r{rate:g}'), dict(consts, **FLAGS), case))
    return jobs


def residuals(res, exps=EXPS):
    """Нормированные невязки по опытам: k/k0 - на SIG_K, lg(grad/grad0) - на SIG_LG."""
    out = {}
    for e, r in zip(exps, res):
        if e in ('sr1', 'sr2'):
            out[e] = k_residuals(r, sr.EXPS[int(e[-1])]['k_pv']) / SIG_K
        elif e == 'li':
            out[e] = li.stage_residuals(r, li.STAGES) / SIG_K
        else:
            out[e] = sd.residuals(r) / SIG_LG
    return out


def scores(res, exps=EXPS) -> dict:
    """Метрики каждого опыта: k/k0 у кернов с прокачкой (у Li et al. - по ступеням), у Sandyga et al. - lg."""
    out = {}
    for e, r in zip(exps, res):
        if e in ('sr1', 'sr2'):
            out[e] = fit_metrics(*k_model_at(r, sr.EXPS[int(e[-1])]['k_pv']))
        elif e == 'li':
            stages = {}
            for t, _ in li.STAGES:
                pts = np.array([p for p in li.DATA['k_pv'][str(int(t))] if p[0] > 0.0])
                stages[f'{t:.0f}'] = fit_metrics(pts[:, 1] + li.stage_residuals(r, [[t, 0.0]]), pts[:, 1])
            out[e] = stages
        else:
            m = sd.metrics(r)
            _, g_exp = np.array(sd.DATA['gradient']['points']).T
            lg_exp = np.log10(g_exp / g_exp[0])
            out[e] = dict(fit_metrics(lg_exp + sd.residuals(r), lg_exp), rms_k=m['rms_k'], max_factor=m['max_factor'],
                          m_conductive=m['m_conductive'])
    return out


def fit(exps=EXPS, fixed=None, x0=X0, max_nfev=10):
    """Совместный подбор по опытам exps; fixed - {индекс: значение} закрепленных параметров (перекрестная проверка)."""
    fixed = fixed or {}
    free = [i for i in range(len(X0)) if i not in fixed]

    def full(z):
        x = list(x0)
        for i, v in fixed.items():
            x[i] = v
        for i, v in zip(free, z):
            x[i] = float(v)
        return x
    resid = lambda res: np.concatenate(list(residuals(res, exps).values()))
    z, best, err = lsq(f'общий набор по {"+".join(exps)}', lambda z: jobs_of(full(z), exps), resid,
                       [x0[i] for i in free], [STEP[i] for i in free], [LO[i] for i in free], [HI[i] for i in free],
                       max_nfev=max_nfev)
    return full(z), best, err


def hypothesis_checks() -> dict:
    """Проверка гипотезы о затрудненной диффузии в геле на ступени 25 C керна Li et al.: подбор по семейству
    (DIFF_MULT = 0.10) против DIFF_MULT = 1 без флага и с флагом `gel_hindered_diffusion`."""
    pl, pc = load_params('li2024'), load_params('li2024_cold')
    x = list(pc['x'])
    variants = (('подбор по ступени 25 C', x, {}), ('DIFF_MULT = 1', [x[0], 0.0, x[2]], {}),
                ('DIFF_MULT = 1, диффузия по кажущейся вязкости', [x[0], 0.0, x[2]], {'gel_hindered_diffusion': 'True'}))
    jobs = []
    for tag, xx, flags in variants:
        _, consts, case = li.cold_job(pc['gel_time'], xx, pl[pl['best']]['kin'])
        jobs.append((f'probe_li_{len(jobs)}', dict(consts, **flags), case))
    out = {}
    for (tag, _, _), r in zip(variants, run_many(jobs)):
        out[tag] = dict(rms={f'{t:.0f}': v for t, v in li.stage_rms(r).items()}, plugged=r['plugged'])
        print(f'  {tag}:', {t: round(v, 3) for t, v in out[tag]['rms'].items()}, 'закупорка', r['plugged'], flush=True)
    return out


# Перекрестная проверка: опыт-прогноз и его собственный параметр, который без опыта не определен - априорный
def _loo_fixed(target, x):
    return {'sr2': {4: math.log10(MU_RING)},  # вязкость нефти 2 - как у Ring et al.
            'li': {5: x[7]},                  # k_cr нефти - как у раствора
            'sd': {7: x[5]}}[target]          # k_cr раствора - как у нефти


def run(mode: str = 'full') -> dict:
    checks = hypothesis_checks()
    if mode == 'full':
        x, best, _ = fit()
        save_params('unified', {'x': x, 'names': NAMES})
    else:
        x = load_params('unified')['x']
        best = run_many(jobs_of(x))
    out = {'x': x, 'names': NAMES, 'scores': scores(best), 'curves': dict(zip(EXPS, best)), 'loo': {},
           'checks': checks}
    print('общий набор:', dict(zip(NAMES, np.round(x, 3).tolist())), flush=True)
    for e, s in out['scores'].items():
        print(f'  {e}:', {t: round(v['rmse'], 3) for t, v in s.items()} if e == 'li' else s, flush=True)
    # Прогноз опыта Sandyga et al. при других скоростях охлаждения тем же набором (статья о термических процессах)
    rates = run_many([jobs_of(x, ('sd',), rate)[0] for rate in sd.COOLING_RATES])
    out['sd_cooling'] = sd.cooling_rates(dict(zip(sd.COOLING_RATES, rates)), best[EXPS.index('sd')], 10 ** x[7])
    # Проверка k_cr раствора без керна: WAT от скорости охлаждения (Struchkov & Rogachev 2017) и объемная WAT 20 %-го
    # раствора при скорости реометра Sandyga et al. (1 C/мин)
    out['wat_rate'] = sd.wat_rate(load_params('sandyga2020')['best'][0], 10 ** x[7])
    saved = {} if mode == 'full' else load_params('unified').get('loo', {})
    for target in ('sr2', 'li', 'sd'):
        others = tuple(e for e in EXPS if e != target)
        if mode == 'full':
            xl, _, _ = fit(others, _loo_fixed(target, x), x0=x, max_nfev=5)
        else:
            xl = saved[target]
        pred = run_many(jobs_of(xl, (target,)))
        s = scores(pred, (target,))[target]
        out['loo'][target] = dict(x=xl, scores=s, curve=pred[0])
        print(f'  прогноз {target} без него в подборе:',
              {t: round(v['rmse'], 3) for t, v in s.items()} if target == 'li' else s, flush=True)
    if mode == 'full':
        save_params('unified', {'x': x, 'names': NAMES, 'loo': {t: e['x'] for t, e in out['loo'].items()}})
    save_results('unified', out)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
