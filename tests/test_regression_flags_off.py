"""Регрессия «все новые флаги выключены»: расчет побитово совпадает с эталоном до детального состава нефти.

Эталон и прогон - `tests/regression_baseline.py`. Побитовое совпадение возможно только на той же
платформе и с теми же версиями numba/llvmlite/numpy: на другой машине или после обновления библиотек
тест пропускается с сообщением (сравнивать там нечего - меняется сама машинная арифметика).
"""
import ast

import numpy as np
import pytest

from paraphin.constants import (wax_components, wax_pressure, asphaltenes, gelation, wax_viscosity,
                                pressure_viscosity)
from tests.regression_baseline import BASELINE, metadata, run


def test_flags_off_bitwise():
    if not BASELINE.is_file():
        pytest.skip(f'нет эталона {BASELINE}')
    if wax_components or wax_pressure or asphaltenes or gelation or wax_viscosity != 0 or pressure_viscosity:
        pytest.skip('в constants.py включены новые механизмы - регрессия осмысленна только при выключенных')

    ref = np.load(BASELINE, allow_pickle=False)
    ref_meta = ast.literal_eval(str(ref['__meta__']))
    if ref_meta != metadata():
        pytest.skip(f'эталон снят в другом окружении: {ref_meta} != {metadata()}')

    fields = run()
    diffs = []
    for name in ref.files:
        if name == '__meta__':
            continue
        a, b = ref[name], fields[name]
        if a.shape != b.shape:
            diffs.append(f'{name}: форма {a.shape} != {b.shape}')
        elif not np.array_equal(a, b):
            diffs.append(f'{name}: max|разность| = {np.abs(a - b).max():.3e}')

    assert not diffs, 'Расчет разошелся с эталоном:\n' + '\n'.join(diffs)
