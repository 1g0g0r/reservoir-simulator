"""Проверка сеточной сходимости на базовом варианте расчета.

Считает базовый вариант (теплообмен по Винсому-Вестервельду, Wp = 0.05) на последовательности
сеток и складывает результаты под разными именами: `case_name` размер сетки не различает,
поэтому без переименования прогоны затерли бы файлы друг друга.

Сетка 50x50 обычно уже посчитана основным прогоном (`run_cases.py`) - тогда ее файл просто
копируется под новым именем, а не считается заново. Ключ --force считает заново все.

    python твт_статья/run_grid_study.py             # 50 (из готового), 70, 100
    python твт_статья/run_grid_study.py 70          # только 70x70
    python твт_статья/run_grid_study.py --force     # пересчитать все, включая 50x50
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
CONSTANTS = ROOT / 'paraphin' / 'constants.py'
DATA = ROOT / 'outputs' / 'data'

GRIDS = (50, 70, 100)
BASE_CASE = 'Wp=0.05_processed_data.pkl'  # имя, под которым базовый вариант пишет результат


def target_name(n: int) -> str:
    return f'grid{n}_processed_data.pkl'


def run_grid(n: int, original: str) -> None:
    """Один прогон базового варианта на сетке n x n."""
    print(f'\n=== Сетка {n}x{n} ===', flush=True)
    patched = re.sub(r'^Nx, Ny = .*$', f'Nx, Ny = {n}, {n}', original, count=1, flags=re.M)
    CONSTANTS.write_text(patched, encoding='utf-8')

    tt = perf_counter()
    result = subprocess.run([sys.executable, 'start.py'], cwd=ROOT)
    print(f'--- {n}x{n}: {perf_counter() - tt:.0f} с, код возврата {result.returncode}', flush=True)

    produced = DATA / BASE_CASE
    if produced.is_file():
        produced.replace(DATA / target_name(n))
        print(f'--- результат сохранен как {target_name(n)}', flush=True)
    else:
        print(f'--- ВНИМАНИЕ: {BASE_CASE} не создан, результат потерян', flush=True)


def main() -> None:
    args = sys.argv[1:]
    force = '--force' in args
    numbers = [int(a) for a in args if a.isdigit()] or list(GRIDS)
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
            # Базовую сетку не пересчитываем, если основной прогон ее уже посчитал
            if n == 50 and not force and backup.is_file():
                shutil.copy2(backup, DATA / target_name(n))
                print(f'\n=== Сетка 50x50: взята из готового {BASE_CASE} ===', flush=True)
                continue
            run_grid(n, original)
    finally:
        CONSTANTS.write_text(original, encoding='utf-8')
        if backup.is_file():
            backup.replace(ready)
            print(f'{BASE_CASE} возвращен на место.')
        print('constants.py восстановлен в исходный вид.')


if __name__ == '__main__':
    main()
