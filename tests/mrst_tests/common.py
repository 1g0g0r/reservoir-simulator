"""Общее для сравнений paraphin с MRST (`tests/mrst_tests/test_*.py`, описание - README.md рядом).

Каждый тест сравнивает расчет paraphin с эталоном MRST из `data/<тест>.npz`. Эталон снимается один раз командой
`python -m tests.mrst_tests.<тест> --mrst`: `run_mrst` пишет постановку outputs/mrst/<имя>.case.json и запускает
MATLAB-код своего теста `mrst_<тест без test_>.m`:
    octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests'); mrst_thermal('<имя>.case.json', '<имя>.json')"
(лог - outputs/mrst/<имя>.log), затем `save_reference` сжимает результаты до суточных накопленных объемов и полей.
paraphin считается в копиях пакета (`tests/_patched_copy.py`) рабочим процессом `worker.py`; прогоны одного
pytest-сеанса переиспользуются (два теста двухфазной задачи считают paraphin один раз).
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / 'data'
RUNS = ROOT / 'outputs' / 'mrst'
FIG = RUNS / 'figures'
OCTAVE = os.environ.get('OCTAVE', r'C:\Program Files\GNU Octave\Octave-10.3.0\mingw64\bin\octave-cli.exe')
MRST_ROOT = os.environ.get('MRST_ROOT', r'C:\Program Files\GNU Octave\Octave-10.3.0\mrst-2024a')

KEYS = ('inj', 'w', 'o')  # закачка, отбор воды, отбор нефти
CODES = {'paraphin': 'paraphin (IMPES)', 'incomp': 'MRST incomp (IMPES)', 'ad': 'MRST ad-blackoil (неявная)',
         'thermal': 'MRST geothermal (неявная)'}
# Цвет и маркер: paraphin - линия, MRST - полые маркеры поверх нее
STYLE = {'paraphin': ('#2a78d6', None), 'incomp': ('#eb6834', 'o'), 'ad': ('#1baf7a', 's'), 'thermal': ('#1baf7a', 's')}
# Двухфазная постановка start.py без парафина и теплообмена; закачка при пластовой температуре - вязкости постоянны
PATCH_TWO_PHASE = {'init_Wp': '0.0', 'heat_losses': '0', 'sol_time_step': '1e30', 'max_eta': '2.0'}
GRIDS = (20, 40, 75)             # сетки двухфазной задачи
VARIANTS = ('native', 'flooded')  # начальное состояние: как start.py; ячейка нагнетательной промыта (S = S_max)
DAYS = 2000                       # до обводненности ~0.98 (критерий останова max_eta)
FIELD_DAYS = (250, 500, 1000, 2000)


def physics(**extra) -> dict:
    """Постановка из constants.py репозитория - те же числа, с которыми считают копии paraphin. Она же хранится в
    эталоне: разошлась с текущей - эталон устарел."""
    from paraphin.constants import (Pw, Po, rw, init_p, init_k, init_m, S_min, S_max, n_power, ro_w, ro_o,
                                    X_min, X_max, Y_min, Y_max, h, init_T)
    from paraphin.utils.math_utils.fluids_correlations import calc_mu_o, calc_mu_w
    return dict(Lx=X_max - X_min, Ly=Y_max - Y_min, h=h, k=init_k, m=init_m, mu_w=float(calc_mu_w(init_T)),
                mu_o=float(calc_mu_o(init_T, 0.0)), ro_w=ro_w, ro_o=ro_o, n=n_power, S_min=S_min, S_max=S_max,
                rw=rw, p_inj=Pw, p_prod=Po, wi_mult=0.25, p0=init_p, **extra)


def mrst_case(phys, solver, n, days, dt_days, field_days, flooded=False, inj_q=None, perm=None, **extra) -> dict:
    """Постановка для mrst_*.m (читает mrst_start.m): числа СИ, решатель, сетка, шаг, номера шагов с полями."""
    case = dict(phys, mrst_root=MRST_ROOT, solver=solver, nx=n, ny=n, flooded_injector=flooded,
                inj_control='rate' if inj_q else 'bhp', inj_val=inj_q or phys['p_inj'],
                dt=dt_days * 86400.0, n_steps=round(days / dt_days), field_steps=[round(d / dt_days) for d in field_days],
                **extra)
    if perm is not None:
        case['perm'] = np.asarray(perm).ravel(order='F').tolist()  # ячейка (i, j) - элемент i + j*nx
    return case


def run_mrst(name, case, script) -> None:
    """Один прогон MRST в Octave функцией script (mrst_*.m); готовый outputs/mrst/<имя>.json не пересчитывается."""
    if (RUNS / f'{name}.json').exists():
        return
    path = RUNS / f'{name}.case.json'
    path.write_text(json.dumps(case), encoding='utf-8')
    cmd = f"addpath('{HERE.as_posix()}'); {script}('{path.as_posix()}', '{(RUNS / name).as_posix()}.json')"
    with open(RUNS / f'{name}.log', 'w', encoding='utf-8') as log:
        subprocess.run([OCTAVE, '--no-gui', '--eval', cmd], stdout=log, stderr=subprocess.STDOUT, check=True)
    print('готов', name, flush=True)


def save_reference(test, phys, cases: dict) -> None:
    """Прогоны MRST кодом теста mrst_<тест без test_>.m параллельно (старты разнесены: startup.m пишет настройки
    MRST) и эталон data/<тест>.npz."""
    script = 'mrst_' + test.removeprefix('test_')
    RUNS.mkdir(parents=True, exist_ok=True)
    order = sorted(cases, key=lambda name: -cases[name]['nx'] ** 2 * cases[name]['n_steps'])  # длинные - первыми
    with ThreadPoolExecutor(max(1, min(len(order), (os.cpu_count() or 2) - 1))) as pool:
        futures = []
        for name in order:
            futures.append(pool.submit(run_mrst, name, cases[name], script))
            time.sleep(3)
        for f in futures:
            f.result()
    data = {'physics': json.dumps(phys)}
    for name in cases:
        d = from_raw(name)
        for key, value in d.items():
            if key in ('p', 'sw', 'T'):
                value = value.astype(np.float32)
            if not key.startswith('q_') and key != 'eta':
                data[f'{name}__{key}'] = value
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / f'{test}.npz', **data)
    print('эталон:', DATA / f'{test}.npz')


def load_reference(test, phys) -> dict:
    path = DATA / f'{test}.npz'
    if not path.exists():
        pytest.skip(f'нет эталона {path.name}: python -m tests.mrst_tests.{test} --mrst')
    ref = np.load(path)
    stored = json.loads(str(ref['physics']))
    changed = {k: (stored.get(k), v) for k, v in phys.items() if stored.get(k) is None or not np.isclose(stored[k], v)}
    if changed:
        pytest.skip(f'эталон {path.name} снят при других константах {changed}: python -m tests.mrst_tests.{test} --mrst')
    runs = defaultdict(dict)
    for key in ref.files:
        if '__' in key:
            name, field = key.split('__')
            runs[name][field] = ref[key].astype(float)
    return {name: with_rates(d) for name, d in runs.items()}


_DONE = {}


def paraphin_runs(jobs: dict) -> dict:
    """Прогоны paraphin {имя: (копия, подстановка constants.py, опции worker.py)}. Копии - параллельно, прогоны одной
    копии - подряд: два процесса сразу писали бы в один кеш numba. В сеансе pytest каждый прогон - один раз."""
    from tests._patched_copy import make_copy
    RUNS.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for name, job in jobs.items():
        if name not in _DONE:
            groups[job[0]].append((name, job))

    def run_group(items):
        for name, (copy, patch, opts) in items:
            root = make_copy(copy, patch)
            opts_path = RUNS / f'{name}.opts.json'
            opts_path.write_text(json.dumps(opts), encoding='utf-8')
            env = dict(os.environ, PYTHONPATH=str(root), GOMP_SPINCOUNT='0', NUMBA_NUM_THREADS='2')
            done = subprocess.run([sys.executable, str(HERE / 'worker.py'), str(RUNS / f'{name}.json'), str(opts_path)],
                                  cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if done.returncode:
                raise RuntimeError(f'paraphin {name}:\n{done.stderr[-3000:]}')
            _DONE[name] = from_raw(name)

    if groups:
        with ThreadPoolExecutor(len(groups)) as pool:
            list(pool.map(run_group, groups.values()))
    return {name: _DONE[name] for name in jobs}


def from_raw(name) -> dict:
    """Прогон из json: накопленные объемы на конец суток, [м^3], забойное давление нагнетательной, [Па], и
    температура добывающей, [K], на конец суток, поля в сутках field_t."""
    r = json.loads((RUNS / f'{name}.json').read_text(encoding='utf-8'))
    t = np.concatenate([[0.0], np.asarray(r['t'], float)])
    days = np.arange(1, int(round(t[-1] / 86400.0)) + 1) * 86400.0
    d = {'steps': np.float64(len(r['t'])), 'elapsed': np.float64(r['elapsed'])}
    for key, raw in zip(KEYS, ('q_inj', 'qw_prod', 'qo_prod')):
        acc = np.concatenate([[0.0], np.cumsum(np.asarray(r[raw], float) * np.diff(t))])
        d['Q_' + key] = np.interp(days, t, acc)
    for key in ('bhp_inj', 'T_prod'):
        d[key] = np.interp(days, t[1:], np.asarray(r[key], float))
    for key in ('p', 'sw', 'T'):
        if key in r:
            d[key] = np.atleast_2d(np.asarray(r[key], float))
    return with_rates(d)


def with_rates(d) -> dict:
    """Суточные дебиты, [м^3/сут], и обводненность из накопленных объемов."""
    for key in KEYS:
        d['q_' + key] = np.diff(d['Q_' + key], prepend=0.0)
    q = d['q_w'] + d['q_o']
    d['eta'] = np.divide(d['q_w'], q, out=np.zeros_like(q), where=q > 0)
    return d


def metrics(r) -> dict:
    """Показатели прогона: прорыв воды (обводненность > 1 %), обводненность 0.98, объемы, КИН."""
    from paraphin.constants import X_min, X_max, Y_min, Y_max, h, init_m, S_min
    oil_in_place = (X_max - X_min) * (Y_max - Y_min) * h * init_m * (1.0 - S_min)

    def first_day(mask):
        idx = np.flatnonzero(mask)
        return int(idx[0]) + 1 if idx.size else None
    return {'t_bt': first_day(r['eta'] > 0.01), 't98': first_day(r['eta'] >= 0.98),
            'Q_inj_10': float(r['Q_inj'][min(9, r['Q_inj'].size - 1)]), 'Qo': float(r['Q_o'][-1]), 'Q_inj': float(r['Q_inj'][-1]),
            'KIN': float(r['Q_o'][-1] / oil_in_place), 'steps': int(r['steps']), 'elapsed': float(r['elapsed'])}


def differences(r, ref) -> dict:
    """Расхождение прогона MRST r с paraphin ref. Дебиты - относительная норма L2 суточных значений; поля - по
    сохраненным суткам; от 30 сут - после промывки ячейки нагнетательной (см. README, соглашение о подвижности)."""
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    m, m_ref = metrics(r), metrics(ref)
    d = {'q_inj': rel(r['q_inj'], ref['q_inj']), 'q_o': rel(r['q_o'], ref['q_o']),
         'eta_mean': float(np.abs(r['eta'] - ref['eta']).mean()),
         'Q_inj_10': m['Q_inj_10'] / m_ref['Q_inj_10'] - 1.0, 'Qo': m['Qo'] / max(m_ref['Qo'], 1e-30) - 1.0,
         'q_inj_1': float(r['q_inj'][0] / ref['q_inj'][0] - 1.0),
         't_bt': None if m['t_bt'] is None or m_ref['t_bt'] is None else m['t_bt'] - m_ref['t_bt'],
         'bhp_inj_30': rel(r['bhp_inj'][30:], ref['bhp_inj'][30:]),
         'bhp_inj_1': float(r['bhp_inj'][0] / ref['bhp_inj'][0] - 1.0) if ref['bhp_inj'][0] else 0.0,
         'T_prod_max': float(np.abs(r['T_prod'] - ref['T_prod']).max()) if r['T_prod'].any() else 0.0}
    ds, dp = r['sw'] - ref['sw'], (r['p'] - ref['p']) / 1e5
    d.update(dS_mean=np.abs(ds).mean(axis=1).tolist(), dS_max=np.abs(ds).max(axis=1).tolist(),
             dp_rms=np.sqrt((dp ** 2).mean(axis=1)).tolist(), dp_max=np.abs(dp).max(axis=1).tolist())
    if 'T' in r:
        dT = r['T'] - ref['T']
        d.update(dT_mean=np.abs(dT).mean(axis=1).tolist(), dT_max=np.abs(dT).max(axis=1).tolist())
    return d


def check(name, d, limits: dict) -> list:
    """Нарушения допусков: limits {показатель: предел модуля}; у полей проверяются все сохраненные сутки."""
    bad = []
    for key, limit in limits.items():
        values = d[key] if isinstance(d[key], list) else [d[key]]
        if any(v is None or abs(v) > limit for v in values):
            bad.append(f'{name}: {key} = {values} > {limit}')
    return bad


def assert_limits(diffs, limits) -> None:
    bad = [msg for name, d in diffs.items() for msg in check(name, d, limits)]
    assert not bad, 'расхождение с MRST больше допуска:\n' + '\n'.join(bad)


def two_phase_cases(solver, dt_days) -> dict:
    """Двухфазная задача start.py на всех сетках и в обоих вариантах начального состояния."""
    phys = physics()
    return {f'{solver}_n{n}_{v}': mrst_case(phys, solver, n, DAYS, dt_days, FIELD_DAYS, flooded=v == 'flooded')
            for n in GRIDS for v in VARIANTS}


def two_phase_paraphin() -> dict:
    return paraphin_runs({f'paraphin_n{n}_{v}': (f'mrst_n{n}', {'Nx, Ny': f'{n}, {n}', **PATCH_TWO_PHASE},
                                                  {'days': DAYS, 'field_days': FIELD_DAYS, 'flooded': v == 'flooded'})
                          for n in GRIDS for v in VARIANTS})


def plot_two_phase(test, runs, solver) -> None:
    """Дебиты и обводненность на всех сетках, профили и изолинии S_w и p на 75x75."""
    from paraphin.constants import Pw, Po, S_min, S_max
    for n in GRIDS:
        for v in VARIANTS:
            names = [f'paraphin_n{n}_{v}', f'{solver}_n{n}_{v}']
            plot_series(f'{test}_rates_n{n}_{v}.png', [
                ('Приемистость нагнетательной, м³/сут', 'q_inj', names, 1.0),
                ('Дебит нефти, м³/сут', 'q_o', names, 1.0),
                ('Обводненность, доли', 'eta', names, 1.0)], runs)
    n = max(GRIDS)
    fields = [('sw', 1.0, np.linspace(S_min + 0.02, S_max - 0.02, 9).round(2), 'S_w'),
              ('p', 1e-5, np.arange(Po / 1e5 + 5, Pw / 1e5, 5), 'p, бар')]
    for v in VARIANTS:
        ref, other = f'paraphin_n{n}_{v}', f'{solver}_n{n}_{v}'
        plot_isolines(f'{test}_isolines_n{n}_{v}.png', runs, ref, other, n, FIELD_DAYS[:3], fields, f'{n}x{n}, {v}')
        plot_profiles(f'{test}_profiles_n{n}_{v}.png', runs, [ref, other], n, FIELD_DAYS[:3],
                      [('sw', 1.0, 'S_w'), ('p', 1e-5, 'p, бар')], f'{n}x{n}, {v}')
        plot_maps(f'{test}_maps_n{n}_{v}.png', runs, ref, other, n, FIELD_DAYS.index(500),
                  [('sw', 1.0, 'Blues', 'S_w'), ('p', 1e-5, 'Purples', 'p, бар')], f'{n}x{n}, {v}, 500 сут')


def report(test, runs, pairs, field_days) -> dict:
    """Таблицы показателей и расхождений pairs {прогон MRST: прогон paraphin} -> outputs/mrst/<тест>.md и stdout."""
    diffs = {name: differences(runs[name], runs[ref]) for name, ref in pairs.items()}
    out = [f'# {test}\n', '| прогон | прорыв, сут | обв. 0.98, сут | закачка за 10 сут, м³ | Qнефти, м³ | Qзакачки, м³ '
           '| КИН, % | шагов | время, с |', '|---|---|---|---|---|---|---|---|---|']
    for name in sorted(set(pairs) | set(pairs.values())):
        m = metrics(runs[name])
        out.append(f"| {name} | {m['t_bt']} | {m['t98']} | {m['Q_inj_10']:.0f} | {m['Qo']:.0f} | {m['Q_inj']:.0f} "
                   f"| {100 * m['KIN']:.2f} | {m['steps']} | {m['elapsed']:.0f} |")
    out += ['\n| MRST − paraphin | L2 q_закачки | L2 q_нефти | ср. Δобв. | Δзакачки 10 сут | ΔQнефти | Δпрорыв, сут '
            '| L2 p_заб (>30 сут) | max ΔT_доб, K |', '|---|---|---|---|---|---|---|---|---|']
    for name, d in diffs.items():
        out.append(f"| {name} | {d['q_inj']:.2%} | {d['q_o']:.2%} | {d['eta_mean']:.4f} | {d['Q_inj_10']:+.1%} "
                   f"| {d['Qo']:+.2%} | {d['t_bt']} | {d['bhp_inj_30']:.2%} | {d['T_prod_max']:.2f} |")
    out += ['\n| MRST − paraphin | сут | ср. ΔS | max ΔS | ср.кв. Δp, бар | max Δp, бар | ср. ΔT, K | max ΔT, K |',
            '|---|---|---|---|---|---|---|---|']
    for name, d in diffs.items():
        for k, day in enumerate(field_days):
            t = f"{d['dT_mean'][k]:.3f} | {d['dT_max'][k]:.3f}" if 'dT_mean' in d else '- | -'
            out.append(f"| {name} | {day} | {d['dS_mean'][k]:.4f} | {d['dS_max'][k]:.3f} | {d['dp_rms'][k]:.3f} "
                       f"| {d['dp_max'][k]:.3f} | {t} |")
    text = '\n'.join(out) + '\n'
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f'{test}.md').write_text(text, encoding='utf-8')
    print('\n' + text)
    return diffs


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.color': '#e4e3df', 'axes.edgecolor': '#52514e',
                         'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2})
    FIG.mkdir(parents=True, exist_ok=True)
    return plt


def _code(name):
    return name.split('_n')[0]


def _draw(ax, x, y, name, k=0):
    """paraphin - сплошной линией, прогоны MRST - полыми маркерами поверх (около 40 на кривую, со сдвигом k, чтобы
    маркеры разных прогонов не ложились друг на друга): видны обе кривые даже при полном совпадении."""
    code = _code(name)
    color, marker = STYLE[code]
    if marker is None:
        ax.plot(x, y, color=color, lw=2, label=CODES[code], zorder=1)
    else:
        step = max(1, len(x) // 40)
        ax.plot(x, y, ls='none', marker=marker, ms=5, mfc='none', mew=1.3, color=color, label=CODES[code],
                markevery=(k * step // 3 % step, step), zorder=2)


def plot_series(file, panels, runs, xlim=None) -> None:
    """Кривые по суткам: panels [(заголовок, ключ, [прогоны], множитель)]."""
    plt = _plt()
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), layout='constrained', squeeze=False)
    for ax, (title, key, names, scale) in zip(axs[0], panels):
        for k, name in enumerate(names):
            y = np.asarray(runs[name][key]) * scale
            x = np.arange(1, y.size + 1)
            if xlim:  # маркеры - по видимой части
                y, x = y[x <= xlim[1]], x[x <= xlim[1]]
            _draw(ax, x, y, name, k)
        ax.set_title(title, loc='left')
        ax.set_xlabel('t, сут')
    axs[0, 0].legend(frameon=False)
    fig.savefig(FIG / file, dpi=150)
    plt.close(fig)


def plot_profiles(file, runs, names, n, field_days, rows, title) -> None:
    """Профили полей по диагонали от нагнетательной (0, 0) к добывающей: строки rows [(ключ, множитель, подпись)],
    столбцы - сохраненные сутки."""
    plt = _plt()
    dist = np.sqrt(2.0) * 200.0 / n * (np.arange(n) + 0.5)
    diag = np.arange(n) * (n + 1)  # ячейка (i, i)
    fig, axs = plt.subplots(len(rows), len(field_days), figsize=(3.6 * len(field_days), 3.2 * len(rows)),
                            layout='constrained', sharex=True, squeeze=False)
    for col, day in enumerate(field_days):
        for row, (key, scale, label) in enumerate(rows):
            for k, name in enumerate(names):
                _draw(axs[row, col], dist, runs[name][key][col][diag] * scale, name, k)
            axs[row, col].set_title(f'{day} сут: {label}', loc='left')
        axs[-1, col].set_xlabel('расстояние по диагонали, м')
    axs[0, 0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha='left')
    fig.savefig(FIG / file, dpi=150)
    plt.close(fig)


def plot_isolines(file, runs, ref, other, n, field_days, rows, title) -> None:
    """Изолинии полей: на каждой панели paraphin (ref) - сплошные с подписями, MRST (other) - штриховые на тех же
    уровнях. Строки rows [(ключ, множитель, уровни, подпись)], столбцы - сохраненные сутки."""
    plt = _plt()
    from matplotlib.lines import Line2D
    xc = (np.arange(n) + 0.5) * 200.0 / n
    c_ref, c_other = STYLE[_code(ref)][0], STYLE[_code(other)][0]
    fig, axs = plt.subplots(len(rows), len(field_days), figsize=(3.4 * len(field_days), 3.4 * len(rows)),
                            layout='constrained', squeeze=False)
    for col, day in enumerate(field_days):
        for row, (key, scale, levels, label) in enumerate(rows):
            ax = axs[row, col]
            for name, color, ls in ((ref, c_ref, '-'), (other, c_other, '--')):
                z = runs[name][key][col].reshape(n, n, order='F').T * scale
                cs = ax.contour(xc, xc, z, levels=levels, colors=color, linestyles=ls, linewidths=1.3)
                if name == ref:
                    ax.clabel(cs, fontsize=6, fmt='%g')
            ax.set_title(f'{day} сут: {label}', loc='left')
            ax.set_aspect('equal')
            ax.grid(False)
    for ax in axs[-1]:
        ax.set_xlabel('x, м')
    for ax in axs[:, 0]:
        ax.set_ylabel('y, м')
    handles = [Line2D([], [], color=c_ref, ls='-', lw=1.3), Line2D([], [], color=c_other, ls='--', lw=1.3)]
    fig.legend(handles, [CODES[_code(ref)], CODES[_code(other)]], loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle(title, x=0.01, ha='left')
    fig.savefig(FIG / file, dpi=150)
    plt.close(fig)


def plot_maps(file, runs, ref, other, n, k, rows, title) -> None:
    """Карты полей в k-е сохраненные сутки: строки rows [(ключ, множитель, cmap, подпись)], столбцы - paraphin (ref),
    MRST (other) в одной шкале и их разность (other - ref) в симметричной шкале."""
    plt = _plt()
    fig, axs = plt.subplots(len(rows), 3, figsize=(11, 3.4 * len(rows)), layout='constrained', squeeze=False)
    ext = (0, 200, 0, 200)
    for row, (key, scale, cmap, label) in enumerate(rows):
        a, b = (runs[name][key][k].reshape(n, n, order='F').T * scale for name in (ref, other))
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        for ax, z, name in ((axs[row, 0], a, ref), (axs[row, 1], b, other)):
            im = ax.imshow(z, origin='lower', extent=ext, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f'{CODES[_code(name)]}: {label}', loc='left', fontsize=8)
        fig.colorbar(im, ax=axs[row, 1], label=label)
        lim = max(np.abs(b - a).max(), 1e-12)
        im = axs[row, 2].imshow(b - a, origin='lower', extent=ext, cmap='coolwarm', vmin=-lim, vmax=lim)
        axs[row, 2].set_title('MRST − paraphin', loc='left', fontsize=8)
        fig.colorbar(im, ax=axs[row, 2], label='Δ' + label)
    for ax in axs.flat:
        ax.grid(False)
        ax.set_xlabel('x, м')
        ax.set_ylabel('y, м')
    fig.suptitle(title, x=0.01, ha='left')
    fig.savefig(FIG / file, dpi=150)
    plt.close(fig)
