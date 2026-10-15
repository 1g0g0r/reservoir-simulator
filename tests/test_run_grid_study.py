"""Анализ сеточной сходимости (`run_grid_study.py`): порядок по трем уровням и отклонение от самого мелкого."""
import numpy as np

from run_grid_study import analyze, observed_order


def test_observed_order_recovers_power_law():
    for p in (1.0, 2.0, 0.5):
        h = (1 / 50, 1 / 70, 1 / 100)  # шаги с разным отношением: порядок все равно находится
        u = [3.0 + 5.0 * x ** p for x in h]
        assert abs(observed_order(h, abs(u[0] - u[1]), abs(u[1] - u[2])) - p) < 1e-6


def test_observed_order_nan_without_power_law():
    assert np.isnan(observed_order((0.04, 0.02, 0.01), 1.0, 0.0))  # на самом мелком уровне разности нет
    assert np.isnan(observed_order((0.04, 0.02, 0.01), 1e-9, 1.0))  # разность растет при измельчении


def _level(n, time, p=2):
    """Расчет на сетке n: КИН = 0.5*t + C*h^p с разной выгрузкой времени у разных уровней."""
    t = np.asarray(time, dtype=float)
    return 1.0 / n, str(n), {'Time': t, 'Wells': {'Producer_oil': 2.0 + t * 0 + 10 * n ** -p},
                             'Other params': {'KIN': 0.5 * t / 100 + 3 * n ** -p}}


def test_analyze_order_and_deviation():
    levels = [_level(20, np.linspace(0, 100, 50)), _level(40, np.linspace(0, 120, 77)),
              _level(80, np.linspace(0, 100, 31))]
    result = analyze(levels)
    assert set(result) == {'Wells/Producer_oil', 'KIN'}
    assert abs(result['KIN']['order'] - 2.0) < 1e-3
    assert result['KIN']['reference'] == '80' and set(result['KIN']['levels']) == {'20', '40'}
    assert result['KIN']['levels']['20']['max'] > result['KIN']['levels']['40']['max'] > 0
    assert result['KIN']['t_end'] == 100.0  # общий отрезок времени - по самому короткому расчету
