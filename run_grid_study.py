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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
CONSTANTS = ROOT / 'paraphin' / 'constants.py'
DATA = ROOT / 'outputs' / 'data'

GRIDS = (50, 70, 100)
NR_GRID = (15, 21, 31, 45)
BASE_CASE = 'Wp=0.2_processed_data.pkl'  # имя, под которым базовый вариант пишет результат


def target_name(n: int) -> str:
    return f'grid{n}_processed_data.pkl'


def nr_target_name(n: int) -> str:
    return f'nr{n}_processed_data.pkl'


def _run(label: str, patched: str, target: str) -> None:
    """Один прогон базового варианта с уже подготовленным текстом constants.py."""
    print(f'\n=== {label} ===', flush=True)
    CONSTANTS.write_text(patched, encoding='utf-8')

    tt = perf_counter()
    result = subprocess.run([sys.executable, 'start.py'], cwd=ROOT)
    print(f'--- {label}: {perf_counter() - tt:.0f} с, код возврата {result.returncode}', flush=True)

    produced = DATA / BASE_CASE
    if produced.is_file():
        produced.replace(DATA / target)
        print(f'--- результат сохранен как {target}', flush=True)
    else:
        print(f'--- ВНИМАНИЕ: {BASE_CASE} не создан, результат потерян', flush=True)


def run_grid(n: int, original: str) -> None:
    """Один прогон базового варианта на сетке n x n."""
    patched = re.sub(r'^Nx, Ny = .*$', f'Nx, Ny = {n}, {n}', original, count=1, flags=re.M)
    _run(f'Сетка {n}x{n}', patched, target_name(n))


def run_nr(n: int, original: str) -> None:
    """Один прогон базового варианта с сеткой радиусов пор Nr = n."""
    patched = re.sub(r'^Nr = .*$', f'Nr = {n}', original, count=1, flags=re.M)
    _run(f'Nr = {n}', patched, nr_target_name(n))


def main() -> None:
    args = sys.argv[1:]
    force = '--force' in args
    nr_mode = '--nr' in args
    numbers = [int(a) for a in args if a.isdigit()] or list(NR_GRID if nr_mode else GRIDS)
    run, target = (run_nr, nr_target_name) if nr_mode else (run_grid, target_name)
    original = CONSTANTS.read_text(encoding='utf-8')

    # Каждый прогон пишет в один и тот же `Wp=0.05_processed_data.pkl`, то есть затирает результат
    # базового варианта еще до того, как мы переименуем свой. Поэтому базовый файл откладывается
    # в сторону на время теста и возвращается на место в finally.
    ready = DATA / BASE_CASE
    backup = DATA / f'{BASE_CASE}.backup'
    if ready.is_file():
        shutil.copy2(ready, backup)

    try:
        for n in numbers:
            # Базовую сетку не пересчитываем, если основной прогон ее уже посчитал.
            # Для Nr аналогичного готового результата нет - там пересчет всегда.
            if not nr_mode and n == 50 and not force and backup.is_file():
                shutil.copy2(backup, DATA / target(n))
                print(f'\n=== Сетка 50x50: взята из готового {BASE_CASE} ===', flush=True)
                continue
            run(n, original)
    finally:
        CONSTANTS.write_text(original, encoding='utf-8')
        if backup.is_file():
            backup.replace(ready)
            print(f'{BASE_CASE} возвращен на место.')
        print('constants.py восстановлен в исходный вид.')


if __name__ == '__main__':
    main()
