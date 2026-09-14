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
    """Сброс дискового кеша numba при правке констант.

    Горячие функции помечены njit(cache=True) - без этого компиляция всего графа занимает 25 секунд
    при каждом запуске, что больше самого расчета. Но numba инвалидирует кеш по mtime файла с самой
    функцией, а значения из constants.py вшиваются в машинный код как константы: поменяв Nx или dt,
    без этой проверки мы считали бы по старой сетке. Поэтому кеш сбрасывается по хешу констант.
    """
    pkg = Path(__file__).parent
    digest = hashlib.md5(b''.join((pkg / name).read_bytes()
                                  for name in ('constants.py', '__init__.py'))).hexdigest()
    stamp = pkg / '__pycache__' / 'constants_hash.txt'
    if stamp.is_file() and stamp.read_text(encoding='ascii') == digest:
        return None

    for cached in pkg.rglob('__pycache__/*.nb[ic]'):
        cached.unlink(missing_ok=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest, encoding='ascii')


_drop_stale_numba_cache()

N = Nx * Ny  # размер матрицы давления

r = np.linspace(0, 40 * 1e-6, Nr, endpoint=True)
# fi_0 = np.array([0.0, 0.013, 0.023, 0.031, 0.035, 0.034, 0.027, 0.021, 0.016, 0.018, 0.025, 0.032, 0.041, 0.052, 0.061, 0.073, 0.082, 0.086, 0.081, 0.07, 0.059, 0.048, 0.035, 0.024, 0.013, 0])

# Начальная функция пор по размерам - логнормальная модель Косуги (Kosugi K. Lognormal
# Distribution Model for Unsaturated Soil Hydraulic Properties // Water Resour. Res. 1996.
# V. 32. No 9. P. 2697):
#   fi_0(r) = 1/(sqrt(2*pi)*sigma_r*r) * exp(-ln^2(r/r_m) / (2*sigma_r^2)),
# r_m - медианный радиус (геометрическое среднее), sigma_r - стандартное отклонение ln r;
# мода лежит в r_m*exp(-sigma_r^2). В узле r = 0 плотность равна нулю (предел).
# Параметры подобраны под сетку [0, r_max]: sigma_r = 0.4 - верх диапазона, снятого по
# имбибиционным кривым упаковок шаров (0.32-0.43: Ghanbarian, 2020,
# resources/литература/ghanbarian2020.pdf, табл. 1), r_max/r_m = 3.3 - как в той же таблице
# (2.5-3.7). При этом на r_max распределение выходит в нуль (0.3% максимума, у прежнего
# гауссова было 0.6%), а ниже r_pass = 5 мкм лежит 0.8% капилляров - столько же, сколько прежде,
# так что блокирование каналов ведет себя как раньше. Нормировка - на единичную сумму по
# узлам: в уравнения fi входит только отношениями интегралов, а тест сохранения
# (tests/test_fi_update.py) сравнивает суммы.
fi_0 = np.zeros(Nr, data_type)
fi_0[1:] = np.exp(-0.5 * (np.log(r[1:] / r_m) / sigma_r) ** 2) / (np.sqrt(2 * np.pi) * sigma_r * r[1:])
fi_0 /= fi_0.sum()

# массивы радиусов пор в необходимых степенях
r1 = r
r2 = r * r
r3 = r2 * r
r4 = r3 * r
r5 = r4 * r
r6 = r5 * r
cbrt_r1 = np.cbrt(r1)

# Критерий прохождения частицы через горло капилляра:
#   r <  r_pass - частица не пролезает и затыкает капилляр целиком (Ub), осадку взяться неоткуда;
#   r >= r_pass - частица проходит и оседает на стенке, капилляр сужается (Ur < 0).
# Сужение со временем схема учитывает не изменением r, а переносом fi по оси радиусов.
# Границу считаем один раз здесь, чтобы не проверять условие на каждом узле в циклах по Nr.
r_pass = D * 0.5 / gamma
n_pass = int(np.searchsorted(r1, r_pass, side='left'))  # первый узел с r >= r_pass


if __name__ == '__main__':
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=fi_0, mode='lines'))
    fig.show()
