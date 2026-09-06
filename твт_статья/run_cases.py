"""Прогон всех вариантов расчета для статьи.

Флаги `init_Wp`, `heat_losses`, `vinsome_westerveld` - константы уровня модуля: numba вшивает их
в машинный код, поэтому переключить вариант внутри одного процесса нельзя. Скрипт правит
`constants.py`, запускает `start.py` отдельным процессом и восстанавливает файл в исходный вид.

Запуск: python твт_статья/run_cases.py [номера вариантов через пробел]
        python твт_статья/run_cases.py        # все варианты
        python твт_статья/run_cases.py 2 3    # только 2-й и 3-й
"""
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
CONSTANTS = ROOT / 'paraphin' / 'constants.py'

# (номер, описание, init_Wp, heat_losses, vinsome_westerveld, ожидаемое имя файла)
CASES = [
    (1, 'базовый: перетоки по Винсому-Вестервельду, Wp = 0.05', 0.05, True, True, 'Wp=0.05'),
    (2, 'без перетоков тепла, Wp = 0.05', 0.05, False, True, 'Wp=0.05_noheat'),
    (3, 'без перетоков тепла, без парафина', 0.0, False, True, 'Wp=0.0_noheat'),
    (4, 'перетоки по схеме Ловерье, Wp = 0.05', 0.05, True, False, 'Wp=0.05_lauwerier'),
    # Четвертая клетка плана 2x2. Без нее вклад теплообмена измеряется только при наличии
    # парафина, и проверить аддитивность эффектов нечем: разложение через три варианта -
    # тождество, а не результат.
    (5, 'перетоки по Винсому-Вестервельду, без парафина', 0.0, True, True, 'Wp=0.0'),
]


def patch_constants(text: str, init_wp: float, heat: bool, vw: bool) -> str:
    """Подстановка значений трех констант в исходный текст `constants.py`."""
    text = re.sub(r'^init_Wp\s*=.*$', f'init_Wp  = {init_wp}', text, count=1, flags=re.M)
    text = re.sub(r'^heat_losses\s*=.*$', f'heat_losses = {heat}', text, count=1, flags=re.M)
    text = re.sub(r'^vinsome_westerveld\s*=.*$', f'vinsome_westerveld = {vw}', text, count=1, flags=re.M)
    return text


def main() -> None:
    numbers = [int(a) for a in sys.argv[1:]] or [c[0] for c in CASES]
    original = CONSTANTS.read_text(encoding='utf-8')

    try:
        for number, title, init_wp, heat, vw, name in CASES:
            if number not in numbers:
                continue

            print(f'\n=== Вариант {number}: {title} -> {name}_processed_data.pkl ===', flush=True)
            CONSTANTS.write_text(patch_constants(original, init_wp, heat, vw), encoding='utf-8')

            tt = perf_counter()
            result = subprocess.run([sys.executable, 'start.py'], cwd=ROOT)
            print(f'--- вариант {number} занял {perf_counter() - tt:.0f} с, код возврата {result.returncode}',
                  flush=True)
    finally:
        CONSTANTS.write_text(original, encoding='utf-8')
        print('\nconstants.py восстановлен в исходный вид.')


if __name__ == '__main__':
    main()
