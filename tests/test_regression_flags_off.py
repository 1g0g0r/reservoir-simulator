"""Регрессия модели по умолчанию: один псевдокомпонент парафина (`wax_characterization = 'single'`), механизмы выключены.

Эталоны и прогон - `tests/regression_baseline.py`.
1. `test_flags_off_bitwise` - расчет побитово совпадает с эталоном своего окружения (`regression_flags_off_<хеш>.npz`).
   Побитовое совпадение возможно только на той же платформе и с теми же версиями numba/llvmlite/numpy, поэтому эталон
   у каждого окружения свой (`baseline_path`); без эталона своего окружения тест пропускается.
2. `test_reduces_to_legacy` - однокомпонентная модель как частный случай общей модели АСПО воспроизводит прежний
   однокомпонентный код (`legacy_single_<хеш>.npz`, снят до объединения моделей; годится эталон любого окружения).
   Побитово они совпасть не обязаны: общий путь переносит компоненты по потокам граней в другом порядке сложения, а
   ограничитель осаждения подводом берет запас за вычетом оттока за шаг. Многосеточный PCG останавливается сразу под
   `p_pcg_rtol`, поэтому разный порядок сложения дает давление другим на ~5e-11, а взвесь Wps - малая разность близких
   w и w_sat - расходится сильнее всех полей, ~4e-9 от своего максимума.
"""
import ast

import numpy as np
import pytest

from paraphin.constants import (wax_pressure, asphaltenes, gelation, wax_viscosity, pressure_viscosity,
                                deposition_kinetics)
from paraphin.layout import N_W
from tests.regression_baseline import DATA, baseline_path, legacy_path, metadata, run

if N_W > 1 or wax_pressure or asphaltenes or gelation or wax_viscosity != 0 or pressure_viscosity or deposition_kinetics:
    pytest.skip('в constants.py включены механизмы - регрессия осмысленна только для модели по умолчанию',
                allow_module_level=True)


@pytest.fixture(scope='module')
def fields():
    return run()


def test_flags_off_bitwise(fields):
    path = baseline_path()
    if not path.is_file():
        pytest.skip(f'нет эталона для окружения {metadata()}: см. tests/regression_baseline.py')

    ref = np.load(path, allow_pickle=False)
    assert ast.literal_eval(str(ref['__meta__'])) == metadata()

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


def test_reduces_to_legacy(fields):
    refs = [legacy_path()] + sorted(DATA.glob('legacy_single_*.npz'))
    ref_path = next((path for path in refs if path.is_file()), None)
    if ref_path is None:
        pytest.skip('нет эталона прежнего однокомпонентного кода')
    ref = np.load(ref_path)
    assert np.isclose(fields['KIN'], ref['KIN'], rtol=1e-9)
    for name in ('p', 'S', 'T', 'm', 'k', 'Wp', 'Wps'):
        scale = np.abs(ref[name]).max()
        assert np.abs(fields[name] - ref[name]).max() <= 1e-8 * scale, name
