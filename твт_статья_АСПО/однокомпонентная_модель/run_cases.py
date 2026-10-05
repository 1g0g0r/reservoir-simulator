"""Прогон всех вариантов расчета для статьи.

Флаги `init_Wp`, `heat_losses` - константы уровня модуля: numba вшивает их в машинный код,
поэтому переключить вариант внутри одного процесса нельзя. Скрипт правит `constants.py`,
запускает `start.py` отдельным процессом и восстанавливает файл в исходный вид.

Запуск: python твт_статья_АСПО/однокомпонентная_модель/run_cases.py [номера вариантов через пробел]
        python твт_статья_АСПО/однокомпонентная_модель/run_cases.py        # все варианты
        python твт_статья_АСПО/однокомпонентная_модель/run_cases.py 2 3    # только 2-й и 3-й
"""
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from make_figures import main as main_figures
from paraphin.constants import init_Wp

ROOT = Path(__file__).resolve().parents[2]
CONSTANTS = ROOT / 'paraphin' / 'constants.py'

# (номер, описание, Wp, heat_losses (0 - нет, 1 - Ловерье, 2 - Винсом-Вестервельд), имя файла)
CASES = [
    (1, f'базовый: перетоки по Винсому-Вестервельду, Wp = {init_Wp}', init_Wp, 2, f'Wp={init_Wp}'),
    (2, f'без перетоков тепла, Wp = {init_Wp}', init_Wp, 0, f'Wp={init_Wp}_noheat'),
    (3, f'без перетоков тепла, без парафина', 0.0, 0, 'Wp=0.0_noheat'),
    (4, f'перетоки по схеме Ловерье, Wp = {init_Wp}', init_Wp, 1, f'Wp={init_Wp}_lauwerier'),
    (5, 'перетоки по Винсому-Вестервельду, без парафина', 0.0, 2, 'Wp=0.0'),
]


def patch_constants(text: str, init_wp: float, heat: int) -> str:
    """Подстановка значений констант в исходный текст `constants.py`."""
    text = re.sub(r'^init_Wp\s*=.*$', f'init_Wp  = {init_wp}', text, count=1, flags=re.M)
    text = re.sub(r'^heat_losses:\s*int\s*=.*$', f'heat_losses: int = {heat}', text, count=1, flags=re.M)
    return text


def main() -> None:
    numbers = [int(a) for a in sys.argv[1:]] or [c[0] for c in CASES]
    original = CONSTANTS.read_text(encoding='utf-8')

    try:
        for number, title, init_wp, heat, name in CASES:
            if number not in numbers:
                continue

            print(f'\n=== Вариант {number}: {title} -> {name}_processed_data.pkl ===', flush=True)
            CONSTANTS.write_text(patch_constants(original, init_wp, heat), encoding='utf-8')

            tt = perf_counter()
            result = subprocess.run([sys.executable, 'start.py'], cwd=ROOT)
            print(f'--- вариант {number} занял {perf_counter() - tt:.0f} с, код возврата {result.returncode}', flush=True)
    finally:
        main_figures()
        CONSTANTS.write_text(original, encoding='utf-8')
        print('\nconstants.py восстановлен в исходный вид.')


if __name__ == '__main__':
    main()
