"""Двухфазная задача start.py: paraphin (IMPES) против полностью неявной схемы MRST ad-blackoil (как mrst_test.m).

MRST ad-blackoil (TwoPhaseOilWaterModel): давление и насыщенность неявно, фазовые подвижности на грани - против
потока (у paraphin суммарная - гармоническое среднее, доли фаз - против потока), неявный Эйлер с шагом 1 сут.
Сетки 20, 40, 75; варианты native (как start.py) и flooded (ячейка нагнетательной промыта). Расхождения - от схемы
по пространству и времени и соглашения о подвижности закачки; допуски те же, что у IMPES MRST.

MRST-часть - mrst_two_phase_implicit.m (MATLAB/Octave; без аргументов - ручной запуск, возвращает результат).

    python -m pytest tests/mrst_tests/test_two_phase_implicit.py -s       # ~10 мин; таблицы и рисунки - outputs/mrst
    python -m tests.mrst_tests.test_two_phase_implicit --mrst              # эталон: 6 прогонов octave-cli, ~50 мин
"""
import argparse

import pytest

from tests.mrst_tests import common as c

TEST = 'test_two_phase_implicit'
SOLVER, DT = 'ad', 1.0  # шаг MRST, [сут]


@pytest.fixture(scope='module')
def diffs():
    runs = c.load_reference(TEST, c.physics())
    runs.update(c.two_phase_paraphin())
    pairs = {name: name.replace(SOLVER, 'paraphin', 1) for name in runs if name.startswith(SOLVER)}
    d = c.report(TEST, runs, pairs, c.FIELD_DAYS)
    c.plot_two_phase(TEST, runs, SOLVER)
    return d


def test_rates(diffs):
    """Суточные приемистость и дебит нефти (норма L2) и накопленная нефть за 2000 сут."""
    c.assert_limits(diffs, {'q_inj': 0.02, 'q_o': 0.03, 'Qo': 0.005})


def test_water_cut(diffs):
    """Обводненность: средняя разность и время прорыва (крутой фронт - поточечная разность не годится)."""
    c.assert_limits(diffs, {'eta_mean': 0.005, 't_bt': 10})


def test_saturation_field(diffs):
    c.assert_limits(diffs, {'dS_mean': 0.005, 'dS_max': 0.08})


def test_pressure_field(diffs):
    """Давление, [бар], при перепаде между скважинами 100 бар."""
    c.assert_limits(diffs, {'dp_rms': 1.0, 'dp_max': 2.0})


def test_injector_mobility_convention(diffs):
    """В нагнетательной paraphin берет k/mu_w, MRST - суммарную подвижность ячейки k*(k_rw/mu_w + k_ro/mu_o) <= k/mu_w:
    пока ячейка не промыта (первые сутки native), MRST закачивает меньше; с промытой ячейкой соглашения совпадают."""
    for name, d in diffs.items():
        if name.endswith('native'):
            assert d['q_inj_1'] < -0.01, f'{name}: приемистость первых суток {d["q_inj_1"]:+.1%}'
        else:
            assert abs(d['q_inj_1']) < 0.08, f'{name}: приемистость первых суток {d["q_inj_1"]:+.1%}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mrst', action='store_true', help='снять эталон MRST в Octave')
    if parser.parse_args().mrst:
        c.save_reference(TEST, c.physics(), c.two_phase_cases(SOLVER, DT))
