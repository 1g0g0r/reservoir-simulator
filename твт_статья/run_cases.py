"""Прогон всех вариантов расчета для статьи.

Флаги `init_Wp`, `heat_losses` - константы уровня модуля: numba вшивает их в машинный код,
поэтому переключить вариант внутри одного процесса нельзя. Скрипт правит `constants.py`,
запускает `start.py` отдельным процессом и восстанавливает файл в исходный вид.

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

# Содержание парафина 0.20, а не 0.05: под исправленной моделью равновесия (6.1)-(6.2) предел
# растворимости при 20 С (температура закачки, самая холодная точка задачи) равен ~0.082 -
# при Wp = 0.05 парафин физически не может выпасть нигде в пласте. Wp = 0.20 дает порог
# кристаллизации ~30 С, оставляя зазор и от температуры закачки, и от Tm = 52.9 С.

# (номер, описание, init_Wp, heat_losses (0 - нет, 1 - Ловерье, 2 - Винсом-Вестервельд), имя файла)
CASES = [
    (1, 'базовый: перетоки по Винсому-Вестервельду, Wp = 0.20', 0.20, 2, 'Wp=0.2'),
    (2, 'без перетоков тепла, Wp = 0.20', 0.20, 0, 'Wp=0.2_noheat'),
    (3, 'без перетоков тепла, без парафина', 0.0, 0, 'Wp=0.0_noheat'),
    (4, 'перетоки по схеме Ловерье, Wp = 0.20', 0.20, 1, 'Wp=0.2_lauwerier'),
    # Четвертая клетка плана 2x2. Без нее вклад теплообмена измеряется только при наличии
    # парафина, и проверить аддитивность эффектов нечем: разложение через три варианта - тождество, а не результат.
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
            print(f'--- вариант {number} занял {perf_counter() - tt:.0f} с, код возврата {result.returncode}',
                  flush=True)
    finally:
        CONSTANTS.write_text(original, encoding='utf-8')
        print('\nconstants.py восстановлен в исходный вид.')


if __name__ == '__main__':
    main()
