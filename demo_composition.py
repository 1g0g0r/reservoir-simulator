"""Демонстрационный расчет детального состава нефти: группы парафинов и восков, асфальтены, смолы,
давление в WAT, гелеобразование - все механизмы включены.

    python demo_composition.py              # все варианты: механизмы добавляются по одному (VARIANTS)
    python demo_composition.py full         # только полная

Флаги - константы уровня модуля, поэтому варианты считаются в копиях пакета с поправленным constants.py
(`tests/_patched_copy.py`): репозиторий и его кеш numba не трогаются. Результаты копируются в
`outputs/data/demo_<вариант>_processed_data.pkl` (+ `_final_composition.npz`); рисунки -
`python docs/make_model_figures.py`.

Постановка - пятиточечный элемент заводнения (как `start.py`), но давление пласта 12 МПа выше давления
начала осаждения асфальтенов (11 МПа), которое выше давления насыщения (9.5 МПа, Узень XIII): у добывающей
скважины (7 МПа) давление проходит оба порога. PVT-значения - литературные оценки (см. constants.py).
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tests._patched_copy import make_copy  # noqa: E402

COMMON = {'Nx, Ny': '50, 50', 'Time_end': 'day_to_sec * 365 * 5.0', 'Pw': '170 * bar_to_pa', 'Po': '70 * bar_to_pa'}
FULL = {'wax_components': 'True', 'wax_pressure': 'True', 'asphaltenes': 'True', 'gelation': 'True',
        'wax_viscosity': '1', 'pressure_viscosity': 'True'}
# Варианты добавляют механизмы по одному - разложение разницы в КИН между прежней и полной моделью
WAX = {'wax_components': 'True', 'wax_pressure': 'True', 'wax_viscosity': '1'}
VARIANTS = {
    'legacy': dict(COMMON),
    'wax': dict(COMMON, **WAX),                                    # группы парафина, давление в WAT, вязкость P-R
    'asph': dict(COMMON, **WAX, asphaltenes='True'),               # + асфальтены и смолы
    'nogel': dict(COMMON, **dict(FULL, gelation='False')),         # + вязкость от давления
    'full': dict(COMMON, **FULL),                                  # + гель
}

RUNNER = '''import sys
sys.path.insert(0, sys.argv[1])
import start
start.solve()
'''


def run(name: str) -> None:
    root = make_copy(f'demo_{name}', VARIANTS[name])
    shutil.copy2(ROOT / 'start.py', root / 'start.py')
    (root / 'run_demo.py').write_text(RUNNER, encoding='utf-8')
    print(f'--- вариант {name}: {root}')
    subprocess.run([sys.executable, 'run_demo.py', str(root)], cwd=root, check=True)

    out = ROOT / 'outputs' / 'data'
    out.mkdir(parents=True, exist_ok=True)
    for src in (root / 'outputs' / 'data').glob('Wp=*'):
        dst = out / src.name.replace(src.name.split('_')[0], f'demo_{name}', 1)
        shutil.copy2(src, dst)
        print('  ->', dst)


if __name__ == '__main__':
    for variant in (sys.argv[1:] or list(VARIANTS)):
        run(variant)
