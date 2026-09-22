"""Модуль решения задачи двухфазной неизотермической фильтрации с учетом кольматаци пласта парафином."""
import hashlib
import warnings
from pathlib import Path

import numpy as np
from numba.core.errors import NumbaWarning

# Функции, получающие скважины (numba jitclass), на диск не кешируются - numba предупреждает об
# этом на каждом запуске. Сделать с этим нечего, кроме отказа от jitclass, поэтому глушим
# конкретно это сообщение: остальные предупреждения numba остаются видимыми.
warnings.filterwarnings('ignore', message='.*Cannot cache compiled function', category=NumbaWarning)

from paraphin.constants import data_type, Nx, Ny, Nr, D, gamma, r_m, sigma_r


def _drop_stale_numba_cache() -> None:
    """Сброс дискового кеша numba при правке любого исходника пакета.

    Горячие функции помечены njit(cache=True) - без этого компиляция всего графа занимает 25 секунд
    при каждом запуске, что больше самого расчета. Но numba инвалидирует кеш по mtime файла с самой
    функцией, а все, что она вызывает, вшивается в ее машинный код: значения из constants.py - как
    литералы, njit-функции других модулей - телом. Поменяв Nx или dt, без этой проверки мы считали бы
    по старой сетке; поправив `calc_qp_m_k_fi`, гоняли бы старое тело внутри кешированной
    `_equations_loop` из solver.py (проверено: правка не подхватывалась, пока не сброшен кеш).
    Поэтому кеш сбрасывается по хешу всех .py пакета.
    """
    pkg = Path(__file__).parent
    digest = hashlib.md5(b''.join(path.read_bytes() for path in sorted(pkg.rglob('*.py')))).hexdigest()
    stamp = pkg / '__pycache__' / 'constants_hash.txt'
    if stamp.is_file() and stamp.read_text(encoding='ascii') == digest:
        return None

    for cached in pkg.rglob('__pycache__/*.nb[ic]'):
        cached.unlink(missing_ok=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest, encoding='ascii')


_drop_stale_numba_cache()

N = Nx * Ny  # размер матрицы давления

# Сетка радиусов равномерная. Сгущение у r_pass и r_max проверено и точности не дает: ошибка
# сидит там, где масса r^2*fi и r^4*fi (15-30 мкм), и сгущение к краям только отбирает оттуда узлы.
# Помогает лишь рост Nr. Схема при этом шага не предполагает - см. `dr_cv`.
r_max = 40e-6
r = np.linspace(0.0, r_max, Nr)
dr_cv = np.empty(Nr, data_type)  # ширина контрольного объема узла: знаменатель в `_update_fi`
dr_cv[1:-1] = (r[2:] - r[:-2]) * 0.5
dr_cv[0], dr_cv[-1] = (r[1] - r[0]) * 0.5, (r[-1] - r[-2]) * 0.5

# Начальная функция пор по размерам - логнормальная модель Косуги (Kosugi K. // Water Resour.
# Res. 1996. V. 32. No 9. P. 2697): r_m - медианный радиус, sigma_r - СКО ln r, мода в r_m*exp(-sigma_r^2).
# sigma_r = 0.4 и r_max/r_m = 3.3 - как у упаковок шаров (Ghanbarian, 2020,
# resources/литература/ghanbarian2020.pdf, табл. 1); на r_max плотность 0.3% максимума, ниже
# r_pass лежит ~1% капилляров. Нормировка на единичный интеграл: sum(fi_0*dr_cv) = 1.
fi_0 = np.zeros(Nr, data_type)
fi_0[1:] = np.exp(-0.5 * (np.log(r[1:] / r_m) / sigma_r) ** 2) / (np.sqrt(2 * np.pi) * sigma_r * r[1:])
fi_0 /= (fi_0 * dr_cv).sum()

# массивы радиусов пор в необходимых степенях
r1 = r
r2 = r * r
r3 = r2 * r
r4 = r3 * r
r5 = r4 * r
r6 = r5 * r
cbrt_r1 = np.cbrt(r1)

# r < r_pass - частица не проходит горло и блокирует капилляр (Ub), r >= r_pass - оседает на
# стенке и сужает его (Ur < 0). Сужение схема учитывает переносом fi по оси радиусов.
r_pass = D * 0.5 / gamma
n_pass = int(np.searchsorted(r1, r_pass, side='left'))  # первый узел с r >= r_pass


if __name__ == '__main__':
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=fi_0, mode='lines'))
    fig.show()
