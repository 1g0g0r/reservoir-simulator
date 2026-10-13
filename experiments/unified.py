"""Общий набор параметров кинетики осаждения на все керновые опыты с парафином (совместная калибровка).

    python experiments/unified.py           # подбор (часы; прогоны кэшируются) -> results/unified.json, params.json
    python experiments/unified.py --quick   # только прогоны с сохраненным набором и перекрестные прогнозы
    python experiments/unified.py --net     # то же для сети пор и горл (+ отношение горла к поре) -> ключ 'net'

Подборы по семействам (`sutton_roberts.py`, `li2024.py`, `sandyga2020.py`) дают каждому опыту свой набор. Здесь у
всех кернов одна модель - одни и те же механизмы (пучок капилляров, перенос кристаллов к стенкам, рост захвата с
осадком - snowball, кристаллизация в объеме и на стенках, вынос отложений; удержание смол и асфальтенов - там, где они
есть в нефти) и одни и те же числа. Диаметр кристалла один на все керны и уточняется только в пределах измерений в
нефти (1-10 мкм, `common.LG_D`).
Различаются только величины, которые физически зависят от системы и в опыте не измерены:

  - вязкость нефтей Sutton & Roberts (у Ring et al. принята 3 мПа*с для обеих): при заданном расходе она входит
    только в броуновскую диффузию D_p ~ 1/mu, поэтому подбирается в пределах 1-30 мПа*с;
  - константа кристаллизации в объеме: у нефти (Li et al.) и у модельного раствора парафина в керосине (Sandyga et
    al.) своя - зарождение в нефти гетерогенное, на асфальтенах и смолах. Значение для раствора независимо
    проверяется опытом Struchkov & Rogachev (WAT от скорости охлаждения, `sandyga2020.wat_rate`).

Множителя броуновской диффузии нет (DIFF_MULT = 1): у Li et al. вязкость измерена. Гипотеза, что малый DIFF_MULT
подбора по ступени 25 C отражает затрудненную диффузию в гелированной нефти - диффузия по кажущейся вязкости
mu_o = mu_p/Phi (флаг `gel_hindered_diffusion`, обобщенное соотношение Стокса - Эйнштейна), - проверяется
(`hypothesis_checks`). Удержание смол и асфальтенов (Li et al., 90-45 C) - из подбора `li2024.py`: другой нефти с ними
нет. WAT пористой среды у Sandyga et al. - из `params.json` (обоснование - `sandyga2020.py`), отложение - сплошной
кристалл (C0 = 1).

Второй вариант (`--net`) - та же модель на сети пор и горл (`pore_network`) с еще одним общим параметром, отношением
горла к поре. Лучший из двух по сумме квадратов нормированных невязок - `params.json: unified.best`.

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
from common import (run_many, lsq, k_residuals, load_params, save_params, save_results, load_results, mode_from_argv,
                    fit_metrics, k_model_at, LG_D)

MU_RING = 3e-3  # вязкость нефтей Sutton & Roberts по Ring et al. (1994), [Па*с]: к ней отнесен DIFF_MULT
FLAGS = {'entrainment': 'True', 'snowball': 'True'}
FLAGS_NET = dict(FLAGS, pore_network='True')
# x: lg SNOW_A, ENT_TAU [Па], lg ENT_RATE [1/с], lg mu_SR1, lg mu_SR2 [Па*с], lg K_CRYST нефти, lg K_WALL,
# lg K_CRYST раствора [1/с], lg D_CRYST [м] (в пределах измерений); у сети десятый - NET_GAMMA. Индексы 1-7 те же, что
# до отказа от d_p на месте 0 (их читают `_loo_fixed` и `make_article.py`)
NAMES = ('lg SNOW_A', 'ENT_TAU', 'lg ENT_RATE', 'lg mu SR1', 'lg mu SR2', 'lg K_CRYST нефть', 'lg K_WALL',
         'lg K_CRYST раствор', 'lg D_CRYST')
# старт - прежние подборы: вынос и вязкости - «общая кинетика, своя вязкость» Sutton & Roberts, k_cr нефти -
# ступень 25 C Li et al., k_wall и k_cr раствора - Sandyga et al.; SNOW_A - умолчание 10 (`kinetics_params`)
X0 = (1.0, 1.649, -3.704, -3.0, -2.004, -1.798, -2.967, -3.435, LG_D['x0'])
LO = (-1.0, 0.1, -6.0, -3.0, -3.0, -5.0, -6.0, -5.0, LG_D['lo'])
HI = (3.0, 5.0, -1.0, math.log10(3e-2), math.log10(3e-2), -0.5, 0.0, -1.0, LG_D['hi'])
STEP = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, LG_D['step'])
NET = dict(name='NET_GAMMA', x0=0.4, lo=0.2, hi=0.9, step=0.02)  # десятый параметр сети (`he2020.py`: 0.40-0.49)
SIG_K, SIG_LG = 0.05, sd.noise_log()  # цели: k/k0 и lg(grad/grad0)
MAX_NFEV = 25  # итераций основного подбора: 10 не хватало (подбор останавливался по лимиту)
EXPS = ('sr1', 'sr2', 'li', 'sd')


def _li_ret():
    p = load_params('li2024')
    return p[p['best']]['kin']


def _box(n: int):
    """Границы и шаги для набора длины n: у сети к девяти общим параметрам добавляется NET_GAMMA."""
    ext = n > len(NAMES)
    return [list(b) + ([NET[k]] if ext else []) for b, k in ((LO, 'lo'), (HI, 'hi'), (STEP, 'step'))]


def jobs_of(x, exps=EXPS, rate=None):
    """Задания опытов exps при наборе x (9 параметров - пучок, 10 - сеть пор и горл); rate - скорость охлаждения керна
    Sandyga et al., C/ч (прогноз)."""
    lg_snow, tau, lg_ent, lg_mu1, lg_mu2, lg_kco, lg_kw, lg_kcs, lg_d = x[:len(NAMES)]
    net = len(x) > len(NAMES)
    flags, tag = (FLAGS_NET, '_net') if net else (FLAGS, '')  # своя копия пакета: другой набор констант
    ent = {'ENT_TAU': float(tau), 'ENT_RATE': float(10 ** lg_ent), 'SNOW_A': float(10 ** lg_snow),
           'D_CRYST': float(10 ** lg_d)}
    if net:
        ent.update(NET_Z=6.0, NET_GAMMA=float(x[len(NAMES)]))
    jobs = []
    for n, lg_mu in ((1, lg_mu1), (2, lg_mu2)):
        if f'sr{n}' in exps:
            kin = dict(ent, DIFF_MULT=MU_RING / 10 ** lg_mu)
            jobs.append(sr.job('uni' + tag, n, dict(sr.ENTRAINMENT, **flags), kin))
    if 'li' in exps:
        gel_time = load_params('li2024_cold')['gel_time']
        _, consts, case = li.cold_job(gel_time, [0.0, lg_kco, lg_snow, lg_d], _li_ret())
        case['kin'].update(ent, K_WALL=float(10 ** lg_kw))
        jobs.append(('exp_li_uni' + tag, dict(consts, **flags), case))
    if 'sd' in exps:
        wat = load_params('sandyga2020')['best'][0]
        _, consts, case = sd.kin_case(wat, (10 ** lg_kw, 1.0, 10 ** lg_kcs), rate=rate)
        case['kin'].update(ent)
        jobs.append(('exp_sd_uni' + tag + ('' if rate is None else f'_r{rate:g}'), dict(consts, **flags), case))
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


def cost(res, exps=EXPS) -> float:
    """Сумма квадратов нормированных невязок - критерий выбора между пучком и сетью."""
    return float(sum(np.sum(v ** 2) for v in residuals(res, exps).values()))


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
    lo, hi, step = _box(len(x0))
    free = [i for i in range(len(x0)) if i not in fixed]

    def full(z):
        x = list(x0)
        for i, v in fixed.items():
            x[i] = v
        for i, v in zip(free, z):
            x[i] = float(v)
        return x
    resid = lambda res: np.concatenate(list(residuals(res, exps).values()))
    z, best, err = lsq(f'общий набор по {"+".join(exps)}', lambda z: jobs_of(full(z), exps), resid,
                       [x0[i] for i in free], [step[i] for i in free], [lo[i] for i in free], [hi[i] for i in free],
                       max_nfev=max_nfev)
    return full(z), best, err


def hypothesis_checks() -> dict:
    """Проверка гипотезы о затрудненной диффузии в геле на ступени 25 C керна Li et al.: подбор по семейству
    (малый DIFF_MULT) против DIFF_MULT = 1 без флага и с флагом `gel_hindered_diffusion`."""
    pl, pc = load_params('li2024'), load_params('li2024_cold')
    x = list(pc['x'])  # lg DIFF_MULT, lg K_CRYST, lg SNOW_A (`li2024.COLD_NAMES`)
    variants = (('подбор по ступени 25 C', x, {}), ('DIFF_MULT = 1', [0.0] + x[1:], {}),
                ('DIFF_MULT = 1, диффузия по кажущейся вязкости', [0.0] + x[1:], {'gel_hindered_diffusion': 'True'}))
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


def _start() -> list:
    """Старт подбора: X0, но d_p - средний lg подборов семейств (Sutton & Roberts, ступень 25 C Li et al.): по d_p
    целевая функция многоэкстремальна, а один подбор общего набора - часы."""
    x0 = list(X0)
    try:
        x0[8] = float(np.mean([load_params('sutton_roberts_lsq')[sr.MAIN_FAMILY]['joint'][0],
                               load_params('li2024_cold')['x'][3]]))
    except (SystemExit, KeyError, IndexError):
        pass
    return x0


def _saved() -> dict:
    try:
        return load_params('unified')
    except SystemExit:
        return {}


def run(mode: str = 'full', net: bool = False) -> dict:
    """Подбор и прогнозы одного варианта: пучок (net = False) или сеть пор и горл. Сеть пишется ключом 'net' в тот же
    results/unified.json и params.json: unified, рядом с пучком."""
    saved = _saved()
    own = saved.get('net', {}) if net else saved
    names = NAMES + ((NET['name'],) if net else ())
    if mode != 'full' and list(own.get('names', [])) != list(names):
        raise SystemExit('params.json: unified - набор прежнего формата (с подбором d_p) или нет набора: сначала '
                         'полный подбор')
    checks = {} if net else hypothesis_checks()
    if mode == 'full':
        # старт - сохраненный оптимум того же формата (повторный подбор продолжает его), иначе `_start`
        x0 = own['x'] if list(own.get('names', [])) == list(names) else _start() + ([NET['x0']] if net else [])
        x, best, _ = fit(x0=x0, max_nfev=MAX_NFEV)
    else:
        x = own['x']
        best = run_many(jobs_of(x))
    out = {'x': x, 'names': names, 'scores': scores(best), 'curves': dict(zip(EXPS, best)), 'loo': {},
           'checks': checks, 'cost': cost(best)}
    print(f'общий набор{" (сеть пор и горл)" if net else ""}:', dict(zip(names, np.round(x, 3).tolist())),
          f'сумма квадратов невязок {out["cost"]:.2f}', flush=True)
    for e, s in out['scores'].items():
        print(f'  {e}:', {t: round(v['rmse'], 3) for t, v in s.items()} if e == 'li' else s, flush=True)
    # Прогноз опыта Sandyga et al. при других скоростях охлаждения тем же набором (статья о термических процессах)
    rates = run_many([jobs_of(x, ('sd',), rate)[0] for rate in sd.COOLING_RATES])
    out['sd_cooling'] = sd.cooling_rates(dict(zip(sd.COOLING_RATES, rates)), best[EXPS.index('sd')], 10 ** x[7])
    # Проверка k_cr раствора без керна: WAT от скорости охлаждения (Struchkov & Rogachev 2017) и объемная WAT 20 %-го
    # раствора при скорости реометра Sandyga et al. (1 C/мин)
    out['wat_rate'] = sd.wat_rate(load_params('sandyga2020')['best'][0], 10 ** x[7])
    for target in ('sr2', 'li', 'sd'):
        others = tuple(e for e in EXPS if e != target)
        if mode == 'full':
            xl, _, _ = fit(others, _loo_fixed(target, x), x0=x, max_nfev=5)
        else:
            xl = own['loo'][target]
        pred = run_many(jobs_of(xl, (target,)))
        s = scores(pred, (target,))[target]
        out['loo'][target] = dict(x=xl, scores=s, curve=pred[0])
        print(f'  прогноз {target} без него в подборе:',
              {t: round(v['rmse'], 3) for t, v in s.items()} if target == 'li' else s, flush=True)
    try:
        prev = load_results('unified')
    except SystemExit:
        prev = {}
    save_results('unified', dict(prev, net=out) if net else dict(out, **{k: prev[k] for k in ('net',) if k in prev}))
    if mode == 'full':
        entry = {'x': x, 'names': names, 'cost': out['cost'], 'loo': {t: e['x'] for t, e in out['loo'].items()}}
        params = dict(saved, net=entry) if net else dict(entry, **{k: saved[k] for k in ('net',) if k in saved})
        if 'cost' in params and 'cost' in params.get('net', {}):
            params['best'] = 'net' if params['net']['cost'] < params['cost'] else 'bundle'
        save_params('unified', params)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv(), net='--net' in sys.argv[1:])
