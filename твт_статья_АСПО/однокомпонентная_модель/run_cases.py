"""Прогон всех вариантов расчета для статьи.

Флаги `init_Wp`, `heat_losses` - константы уровня модуля: numba вшивает их в машинный код,
поэтому переключить вариант внутри одного процесса нельзя. Каждый вариант считается `start.py` в своей копии
пакета с поправленным `constants.py` (`tests/_patched_copy.py`), результат копируется в `outputs/data`.

Запуск: python твт_статья_АСПО/однокомпонентная_модель/run_cases.py [номера вариантов через пробел]
        python твт_статья_АСПО/однокомпонентная_модель/run_cases.py        # все варианты
        python твт_статья_АСПО/однокомпонентная_модель/run_cases.py 2 3    # только 2-й и 3-й
"""
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from make_figures import main as main_figures  # noqa: E402
from paraphin.constants import init_Wp  # noqa: E402
from tests._patched_copy import make_copy  # noqa: E402

DATA = ROOT / 'outputs' / 'data'

# (номер, описание, Wp, heat_losses (0 - нет, 1 - Ловерье, 2 - Винсом-Вестервельд), имя файла)
CASES = [
    (1, f'базовый: перетоки по Винсому-Вестервельду, Wp = {init_Wp}', init_Wp, 2, f'Wp={init_Wp}'),
    (2, f'без перетоков тепла, Wp = {init_Wp}', init_Wp, 0, f'Wp={init_Wp}_noheat'),
    (3, f'без перетоков тепла, без парафина', 0.0, 0, 'Wp=0.0_noheat'),
    (4, f'перетоки по схеме Ловерье, Wp = {init_Wp}', init_Wp, 1, f'Wp={init_Wp}_lauwerier'),
    (5, 'перетоки по Винсому-Вестервельду, без парафина', 0.0, 2, 'Wp=0.0'),
]


def main() -> None:
    numbers = [int(a) for a in sys.argv[1:]] or [c[0] for c in CASES]

    try:
        for number, title, init_wp, heat, name in CASES:
            if number not in numbers:
                continue

            print(f'\n=== Вариант {number}: {title} -> {name}_processed_data.pkl ===', flush=True)
            root = make_copy(f'run_cases_{number}', {'init_Wp': repr(init_wp), 'heat_losses': str(heat)})
            shutil.copy2(ROOT / 'start.py', root / 'start.py')

            tt = perf_counter()
            result = subprocess.run([sys.executable, 'start.py'], cwd=root)
            print(f'--- вариант {number} занял {perf_counter() - tt:.0f} с, код возврата {result.returncode}', flush=True)
            DATA.mkdir(parents=True, exist_ok=True)
            for src in (root / 'outputs' / 'data').glob(f'{name}_*'):
                shutil.copy2(src, DATA / src.name)
    finally:
        main_figures()


if __name__ == '__main__':
    main()
