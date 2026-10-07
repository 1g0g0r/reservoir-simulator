"""Нагнетательная на заданном дебите (режим `q` в add_well): paraphin против MRST incomp и ad-blackoil.

Постановка start.py с закомментированным вариантом закачки по дебиту: нагнетательная - 100 м^3/сут, добывающая - на
Po, сетка 40x40, 2000 сут. Проверяются режим скважины с неизвестным забойным давлением (`upd_q_and_eta`:
P_заб = P_ячейки + q/J) и тот же расчет течения при другом граничном режиме. Пока ячейка нагнетательной не промыта,
MRST берет в ней суммарную подвижность, а paraphin - k/mu_w (README.md), поэтому забойное давление MRST в первые
сутки выше; дальше оно совпадает.

MRST-часть - mrst_rate_control.m (MATLAB/Octave; без аргументов - ручной запуск, возвращает результат).

    python -m pytest tests/mrst_tests/test_rate_control.py -s       # ~2 мин
    python -m tests.mrst_tests.test_rate_control --mrst              # эталон: 2 прогона octave-cli, ~25 мин
"""
import argparse

import pytest

from tests.mrst_tests import common as c

TEST = 'test_rate_control'
N, Q_INJ = 40, 100.0 / 86400.0  # [м^3/с]
DT = {'incomp': 0.25, 'ad': 1.0}


def cases() -> dict:
    phys = c.physics(inj_q=Q_INJ)
    return {f'{s}_n{N}_rate': c.mrst_case(phys, s, N, c.DAYS, dt, c.FIELD_DAYS, inj_q=Q_INJ) for s, dt in DT.items()}


@pytest.fixture(scope='module')
def diffs():
    runs = c.load_reference(TEST, c.physics(inj_q=Q_INJ))
    runs.update(c.paraphin_runs({f'paraphin_n{N}_rate': (
        f'mrst_n{N}', {'Nx, Ny': f'{N}, {N}', **c.PATCH_TWO_PHASE},
        {'days': c.DAYS, 'field_days': c.FIELD_DAYS, 'inj_q': Q_INJ})}))
    d = c.report(TEST, runs, {name: f'paraphin_n{N}_rate' for name in cases()}, c.FIELD_DAYS)
    names = [f'paraphin_n{N}_rate', *cases()]
    c.plot_series(f'{TEST}_rates.png', [('Забойное давление нагнетательной, бар', 'bhp_inj', names, 1e-5),
                                        ('Дебит нефти, м³/сут', 'q_o', names, 1.0),
                                        ('Обводненность, доли', 'eta', names, 1.0)], runs)
    fields = [('sw', 1.0, [0.2, 0.3, 0.4, 0.5, 0.6, 0.68], 'S_w'), ('p', 1e-5, list(range(55, 150, 5)), 'p, бар')]
    for name in cases():
        c.plot_isolines(f'{TEST}_isolines_{name}.png', runs, names[0], name, N, c.FIELD_DAYS[:3], fields,
                        f'{N}x{N}, закачка 100 м³/сут')
        c.plot_maps(f'{TEST}_maps_{name}.png', runs, names[0], name, N, c.FIELD_DAYS.index(500),
                    [('sw', 1.0, 'Blues', 'S_w'), ('p', 1e-5, 'Purples', 'p, бар')], f'{N}x{N}, закачка 100 м³/сут, 500 сут')
    return d


def test_rates(diffs):
    """Закачка задана и совпадает тождественно; дебит нефти и накопленная нефть."""
    c.assert_limits(diffs, {'q_inj': 1e-6, 'q_o': 0.03, 'Qo': 0.005})


def test_water_cut(diffs):
    c.assert_limits(diffs, {'eta_mean': 0.005, 't_bt': 10})


def test_injector_bhp(diffs):
    """Забойное давление нагнетательной после промывки ее ячейки (от 30 сут, норма L2)."""
    c.assert_limits(diffs, {'bhp_inj_30': 0.02})


def test_injector_mobility_convention(diffs):
    """Первые сутки: MRST закачивает в непромытую ячейку с суммарной подвижностью - забойное выше."""
    for name, d in diffs.items():
        assert d['bhp_inj_1'] > 0.0, f'{name}: забойное давление первых суток {d["bhp_inj_1"]:+.1%}'


def test_fields(diffs):
    c.assert_limits(diffs, {'dS_mean': 0.005, 'dS_max': 0.08, 'dp_rms': 1.0, 'dp_max': 2.0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mrst', action='store_true', help='снять эталон MRST в Octave')
    if parser.parse_args().mrst:
        c.save_reference(TEST, c.physics(inj_q=Q_INJ), cases())
