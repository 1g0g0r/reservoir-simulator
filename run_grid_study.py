"""Проверка сеточной сходимости на базовом варианте расчета.

Считает базовый вариант (теплообмен по Винсому-Вестервельду, Wp = 0.20) на последовательности
сеток и складывает результаты под разными именами: `case_name` размер сетки не различает,
поэтому без переименования прогоны затерли бы файлы друг друга.

Сетка 50x50 обычно уже посчитана основным прогоном (`твт_статья_АСПО/однокомпонентная_модель/run_cases.py`) - тогда ее файл просто
копируется под новым именем, а не считается заново. Ключ --force считает заново все.

Флаг --nr переключает сходимость на сетку радиусов пор (`Nr` в constants.py) вместо Nx/Ny -
она влияет на блок кольматации (`fi`, `m`, `k`) и не зависит от сетки по x/y.

    python run_grid_study.py             # Nx=Ny: 50 (из готового), 70, 100
    python run_grid_study.py 70          # только 70x70
    python run_grid_study.py --force     # пересчитать все, включая 50x50
    python run_grid_study.py --nr        # Nr: 15, 21, 31, 45
    python run_grid_study.py --nr 21 61  # только заданные Nr
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tests._patched_copy import make_copy  # noqa: E402
from paraphin.constants import case_name  # noqa: E402

DATA = ROOT / 'outputs' / 'data'

GRIDS = (50, 70, 100)
NR_GRID = (15, 21, 31, 45)
BASE_CASE = f'{case_name}_processed_data.pkl'  # имя, под которым базовый вариант пишет результат


def target_name(n: int) -> str:
    return f'grid{n}_processed_data.pkl'


def nr_target_name(n: int) -> str:
    return f'nr{n}_processed_data.pkl'


def _run(label: str, values: dict, target: str) -> None:
    """Один прогон базового варианта в копии пакета с подстановкой `values` (`tests/_patched_copy.py`)."""
    print(f'\n=== {label} ===', flush=True)
    root = make_copy(target.split('_')[0], values)
    shutil.copy2(ROOT / 'start.py', root / 'start.py')

    tt = perf_counter()
    result = subprocess.run([sys.executable, 'start.py'], cwd=root, env=dict(os.environ, PYTHONPATH=str(root)))
    print(f'--- {label}: {perf_counter() - tt:.0f} с, код возврата {result.returncode}', flush=True)

    produced = root / 'outputs' / 'data' / BASE_CASE
    if produced.is_file():
        DATA.mkdir(parents=True, exist_ok=True)
        produced.replace(DATA / target)
        print(f'--- результат сохранен как {target}', flush=True)
    else:
        print(f'--- ВНИМАНИЕ: {BASE_CASE} не создан, результат потерян', flush=True)


def run_grid(n: int) -> None:
    """Один прогон базового варианта на сетке n x n."""
    _run(f'Сетка {n}x{n}', {'Nx, Ny': f'{n}, {n}'}, target_name(n))


def run_nr(n: int) -> None:
    """Один прогон базового варианта с сеткой радиусов пор Nr = n."""
    _run(f'Nr = {n}', {'Nr': str(n)}, nr_target_name(n))


def main() -> None:
    args = sys.argv[1:]
    force = '--force' in args
    nr_mode = '--nr' in args
    numbers = [int(a) for a in args if a.isdigit()] or list(NR_GRID if nr_mode else GRIDS)
    run, target = (run_nr, nr_target_name) if nr_mode else (run_grid, target_name)

    # Прогоны идут в копиях пакета и пишут в свои outputs, поэтому результат основного прогона в DATA не затирается
    ready = DATA / BASE_CASE
    for n in numbers:
        # Базовую сетку не пересчитываем, если основной прогон ее уже посчитал.
        # Для Nr аналогичного готового результата нет - там пересчет всегда.
        if not nr_mode and n == 50 and not force and ready.is_file():
            shutil.copy2(ready, DATA / target(n))
            print(f'\n=== Сетка 50x50: взята из готового {BASE_CASE} ===', flush=True)
            continue
        run(n)


if __name__ == '__main__':
    main()
