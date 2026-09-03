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

from paraphin.constants import data_type, Nx, Ny, Nr, D, gamma


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
_sigma = 2.0 * np.pi
_m = np.max(r) / 2

# TODO перейти на лог-нормальное распределение
fi_0_np = np.exp(-0.5 * ((r - _m) / 1e-6 / _sigma)**2) / _sigma
fi_0_np[0] = fi_0_np[-1] = 0.0
fi_0 = fi_0_np / np.sum(fi_0_np)

# массивы радиусов пор в необходимых степенях
r1 = r
r2 = r * r
r3 = r2 * r
r4 = r3 * r
r5 = r4 * r
r6 = r5 * r
cbrt_r1 = np.cbrt(r1)

# Критерий прохождения частицы через горло капилляра. У капилляра радиуса r горло имеет радиус
# gamma*r, частица диаметра D проходит его при 2*gamma*r > D, то есть при r > D/(2*gamma):
#   r <= r_pass - частица не пролезает и затыкает капилляр целиком (Ub), осадку взяться неоткуда;
#   r >= r_pass - частица проходит и оседает на стенке, капилляр сужается (Ur < 0).
# Сужение со временем схема учитывает не изменением r (сетка r1 фиксирована), а переносом fi по оси
# радиусов: d(fi)/dt + d(Ur*fi)/dr = -Ub, см. `_update_fi`. Сузившийся капилляр сползает по сетке
# влево и, перейдя r_pass, сам переходит из режима сужения в режим блокирования - поэтому критерий
# и проверяется по узловому r1[ij], это и есть текущий просвет.
# Границы диапазонов считаем один раз здесь, чтобы не проверять условие на каждом узле в поячеечных
# циклах по Nr. Значений два, а не одно, только ради узла, попавшего ровно на r_pass: в исходных
# условиях стояли <= и >=, то есть такой узел принадлежит обоим режимам. На штатной сетке
# (r_pass = 5.0 мкм, шаг 1.33 мкм) точного попадания нет и оба равны 4.
r_pass = D * 0.5 / gamma
n_block = int(np.searchsorted(r1, r_pass, side='right'))  # число узлов с r <= r_pass
n_narrow = int(np.searchsorted(r1, r_pass, side='left'))  # первый узел с r >= r_pass


if __name__ == '__main__':
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=fi_0_np, mode='lines'))
    fig.show()
