"""Сеточная сходимость расчета для модели из `paraphin/constants.py` (с `start.py`: скважины, ГУ).

Модель - то, что записано в `constants.py`: флаги, `init_Wp`, `Time_end` и остальное. Прогоны идут в копиях пакета
(`tests/_patched_copy.py`), в которых подставляется только варьируемая константа, поэтому репозиторий не правится.
Результат каждого уровня кешируется в `outputs/data/grid_study/<хеш модели>/`: ключ - хеш всех `.py` пакета (в том числе
`constants.py`), `start.py` и подстановок `--set`, поэтому правка модели не подхватывает старые прогоны.

    python run_grid_study.py                       # сетки Nx = Ny: 50, 70, 100
    python run_grid_study.py --grid 20 30 40       # свои сетки
    python run_grid_study.py --nr 15 21 31 45      # сетка радиусов пор (блок кольматации, fi, m, k)
    python run_grid_study.py --cfl 0.8 0.4 0.2     # число Куранта (шаг по времени)
    python run_grid_study.py --vary 'Nx, Ny' '20, 40' '30, 60'   # любая константа: имя и ее значения
    python run_grid_study.py --set 'Time_end=day_to_sec*100' --grid 20 30 40   # разовая подстановка поверх constants.py
    python run_grid_study.py --workers 2 --force   # два прогона параллельно, пересчитать, не беря готовые

Анализ: ряды по времени (дебиты и обводненность скважин, КИН), средние по полю (давление, насыщенность, температура,
пористость, парафин) и минимум проницаемости на общей сетке времени. Для каждого уровня - отклонение от самого мелкого
(наибольшее и среднеквадратичное, в долях наибольшего значения ряда) и по трем самым мелким уровням - наблюдаемый порядок
сходимости (разность ряда на соседних уровнях ~ h^p). Значения в конкретных ячейках (`k [10,10]`) не сравниваются: на
разных сетках это разные точки. Итог - таблица в консоли и `outputs/data/grid_study/<хеш>/summary_<константа>.json`.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tests._patched_copy import make_copy  # noqa: E402

STUDY = ROOT / 'outputs' / 'data' / 'grid_study'
# Предустановки: константа, ее значение для уровня, шаг h уровня (по нему считается порядок)
PRESETS = {
    'grid': ('Nx, Ny', lambda n: f'{n}, {n}', lambda n: 1.0 / float(n)),
    'nr': ('Nr', lambda n: str(n), lambda n: 1.0 / float(n)),
    'cfl': ('CFL_target', lambda x: repr(float(x)), lambda x: float(x)),
}
DEFAULT_GRIDS = ('50', '70', '100')
FIELDS_MEAN = ('Pressure', 'Saturation', 'Temperature', 'm', 'Wp')  # средние по полю
FIELDS_MIN = ('k ',)  # минимум по полю (ключ выгрузки - с пробелом)


def model_tag(sets: dict) -> str:
    """Хеш модели: все `.py` пакета (с `constants.py`), start.py и подстановки `--set`."""
    h = hashlib.sha256()
    for path in sorted((ROOT / 'paraphin').rglob('*.py')) + [ROOT / 'start.py']:
        h.update(path.relative_to(ROOT).as_posix().encode() + path.read_bytes())
    h.update(json.dumps(sets, sort_keys=True).encode())
    return h.hexdigest()[:12]


def _file_name(name: str, value: str) -> str:
    return re.sub(r'[^\w.=-]+', '_', f'{name}={value}') + '.pkl'


def run_level(tag: str, name: str, value: str, sets: dict, force: bool = False, threads: int = None) -> Path:
    """Один прогон `start.py` с константой `name = value`. Возвращает pkl с результатом (из кеша, если он есть)."""
    target = STUDY / tag / _file_name(name, value)
    if target.is_file() and not force:
        print(f'=== {name} = {value}: взято из {target.relative_to(ROOT)}', flush=True)
        return target
    print(f'=== {name} = {value} ===', flush=True)
    root = make_copy('gs_' + re.sub(r'\W+', '_', f'{name}_{value}'), dict(sets, **{name: value}))
    shutil.copy2(ROOT / 'start.py', root / 'start.py')
    data = root / 'outputs' / 'data'
    if data.is_dir():
        shutil.rmtree(data)  # результат прошлого прогона не должен сойти за новый
    env = dict(os.environ, PYTHONPATH=str(root))
    if threads:  # несколько прогонов делят ядра: крутящиеся потоки отнимают кванты у соседей (CLAUDE.md)
        env.update(NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), GOMP_SPINCOUNT='0')
    start = perf_counter()
    result = subprocess.run([sys.executable, 'start.py'], cwd=root, env=env)
    produced = list(data.glob('*_processed_data.pkl'))
    if result.returncode != 0 or len(produced) != 1:
        raise SystemExit(f'{name} = {value}: прогон не удался (код {result.returncode}), результатов {len(produced)}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(produced[0], target)
    print(f'--- {name} = {value}: {perf_counter() - start:.0f} с -> {target.relative_to(ROOT)}', flush=True)
    return target


def series(data: dict) -> dict:
    """Ряды по времени, не зависящие от размера сетки: показатели скважин, КИН, средние и минимум по полю."""
    out = {}
    for group in ('Wells', 'Wells_accumulated'):
        for key, values in data.get(group, {}).items():
            out[f'{group}/{key}'] = np.asarray(values, dtype=float)
    if 'KIN' in data.get('Other params', {}):
        out['KIN'] = np.asarray(data['Other params']['KIN'], dtype=float)
    for key in FIELDS_MEAN + FIELDS_MIN:
        if key in data and np.ndim(data[key]) >= 3:
            field = np.asarray(data[key], dtype=float).reshape(len(data['Time']), -1)
            if key in FIELDS_MEAN:
                out[f'{key} (среднее)'] = field.mean(axis=1)
            else:
                out[f'{key.strip()} (минимум)'] = field.min(axis=1)
    return out


def observed_order(h: tuple, d12: float, d23: float) -> float:
    """Порядок p по трем уровням h1 > h2 > h3 и разностям решений d12 = |u1 - u2|, d23 = |u2 - u3|.

    Если u(h) = u0 + C*h^p, то d12/d23 = (h1^p - h2^p)/(h2^p - h3^p); уравнение решается бисекцией по p (отношения
    шагов могут быть разными, например сетки 50, 70, 100). nan - если ряды не лежат на степенном законе
    (колебания, выход на шум) или p вне [0.05, 12]."""
    h1, h2, h3 = h
    if not (d12 > 0 and d23 > 0 and h1 > h2 > h3 > 0):
        return float('nan')

    def ratio(p):
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p)

    lo, hi = 0.05, 12.0
    if not ratio(lo) <= d12 / d23 <= ratio(hi):
        return float('nan')
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ratio(mid) < d12 / d23 else (lo, mid)
    return 0.5 * (lo + hi)


def analyze(levels: list) -> dict:
    """levels - [(h, имя уровня, данные расчета)]. Сравнение с самым мелким уровнем на общей сетке времени."""
    levels = sorted(levels, key=lambda lv: -lv[0])  # от грубого к мелкому
    t_end = min(float(np.max(data['Time'])) for _, _, data in levels)
    grid = np.linspace(0.0, t_end, 400)
    curves = []
    for _, _, data in levels:
        time = np.asarray(data['Time'], dtype=float)
        curves.append({key: np.interp(grid, time, values) for key, values in series(data).items()})
    result = {}
    for key in (k for k in curves[-1] if all(k in c for c in curves)):
        ref = curves[-1][key]
        scale = max(float(np.max(np.abs(ref))), 1e-30)
        rows = {}
        for (_, label, _), c in zip(levels[:-1], curves[:-1]):
            diff = c[key] - ref
            rows[label] = dict(max=float(np.max(np.abs(diff)) / scale), rms=float(np.sqrt(np.mean(diff ** 2)) / scale))
        order = float('nan')
        if len(levels) >= 3:
            def dist(a, b):
                return float(np.sqrt(np.mean((curves[a][key] - curves[b][key]) ** 2)))
            order = observed_order(tuple(lv[0] for lv in levels[-3:]), dist(-3, -2), dist(-2, -1))
        result[key] = dict(levels=rows, order=order, reference=levels[-1][1], t_end=t_end)
    return result


def print_table(result: dict) -> None:
    first = next(iter(result.values()))
    labels = list(first['levels'])
    print(f'\nОтклонение от самого мелкого уровня ({first["reference"]}): наибольшее / среднеквадратичное, '
          'в долях наибольшего значения')
    print(f'{"величина":<36}' + ''.join(f'{label:>20}' for label in labels) + f'{"порядок":>10}')
    for key, row in result.items():
        cells = ''
        for label in labels:
            cell = row['levels'][label]
            cells += f'{cell["max"]:.1e} / {cell["rms"]:.1e}'.rjust(20)
        order = '-' if np.isnan(row['order']) else f'{row["order"]:.2f}'
        print(f'{key:<36}{cells}{order:>10}')


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    what = parser.add_mutually_exclusive_group()
    what.add_argument('--grid', nargs='+', metavar='N', help='сетки Nx = Ny')
    what.add_argument('--nr', nargs='+', metavar='N', help='сетка радиусов пор Nr')
    what.add_argument('--cfl', nargs='+', metavar='X', help='целевое число Куранта CFL_target')
    what.add_argument('--vary', nargs='+', metavar='ИМЯ ЗНАЧЕНИЕ', help='константа и ее значения (выражения Python)')
    parser.add_argument('--set', action='append', default=[], metavar='ИМЯ=ВЫРАЖЕНИЕ', help='подстановка поверх constants.py')
    parser.add_argument('--workers', type=int, default=1, help='прогонов параллельно (по умолчанию 1)')
    parser.add_argument('--force', action='store_true', help='пересчитать, не беря готовые результаты')
    args = parser.parse_args()

    sets = dict(item.split('=', 1) for item in args.set)
    if args.vary:
        name, values = args.vary[0], args.vary[1:]

        def step_of(value):  # шаг ~ 1 / первое число значения (сетка '20, 40' -> 1/20)
            return 1.0 / float(re.search(r'\d+\.?\d*(?:e[-+]?\d+)?', value).group())
        levels = [(name, v, step_of(v)) for v in values]
    else:
        preset = next((p for p in PRESETS if getattr(args, p)), 'grid')
        name, value_of, step_of = PRESETS[preset]
        levels = [(name, value_of(v), step_of(v)) for v in getattr(args, preset) or DEFAULT_GRIDS]
    if len(levels) < 2:
        parser.error('нужно хотя бы два уровня')

    tag = model_tag(sets)
    threads = max(1, (os.cpu_count() or 1) // args.workers) if args.workers > 1 else None
    print(f'модель {tag}: constants.py' + (f' + {sets}' if sets else '') + f'; {name}: ' + ', '.join(v for _, v, _ in levels))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(lambda lv: run_level(tag, lv[0], lv[1], sets, args.force, threads), levels))

    loaded = []
    for (_, value, step), path in zip(levels, paths):
        with open(path, 'rb') as f:
            loaded.append((step, value, pickle.load(f)[1]))
    result = analyze(loaded)
    print_table(result)
    suffix = re.sub(r'\W+', '_', name)
    summary = STUDY / tag / f'summary_{suffix}.json'
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nитог -> {summary.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
