"""Неизотермическая фильтрация: закачка холодной воды, paraphin против MRST geothermal (GeothermalModel).

geothermal считает одну фазу - воду (его двухфазный режим - вода и пар одного компонента, без ОФП нефть-вода),
поэтому и paraphin здесь однофазный: S_max = 1, начальная S = S_max (нефти нет), парафина и теплообмена с кровлей
и подошвой нет. Вода 20 C закачивается в пласт 70 C по схеме start.py (скважины на забойном давлении), сетки 20 и
40, 1000 сут. Вязкость воды - Андраде, как calc_mu_w (в MRST - та же формула через fluid.muW(p, T)), поэтому
приемистость падает по мере охлаждения призабойной зоны. Теплоемкости и теплопроводности - из constants.py:
φ*ρ_w*c_w + (1-φ)*ρ_f*c_f и φ*K_w + (1-φ)*K_f у обоих.

Известное расхождение физики: энтальпия MRST h = u + p/ρ, то есть в его уравнении энергии есть нагрев трением
-v·∇p, которого нет у paraphin. Вода, прошедшая весь перепад, нагревается на (Pw - Po)/(ρ_w*c_w) ≈ 2.4 K
(`dT_friction`), от этого допуски по температуре.

MRST-часть - mrst_thermal.m (MATLAB/Octave; без аргументов - ручной запуск, возвращает результат).

    python -m pytest tests/mrst_tests/test_thermal.py -s       # ~3 мин
    python -m tests.mrst_tests.test_thermal --mrst              # эталон: 2 прогона octave-cli, ~1 ч
"""
import argparse

import pytest

from tests.mrst_tests import common as c

TEST = 'test_thermal'
GRIDS, DAYS, DT = (20, 40), 1000, 1.0
FIELD_DAYS = (100, 250, 500, 1000)
PATCH = {**c.PATCH_TWO_PHASE, 'S_max': '1.0', 'init_S': 'S_max'}


def physics() -> dict:
    from paraphin.constants import init_T, Twater, c_w, c_f, K_w, K_f, ro_f
    return c.physics(T0=init_T + 273.15, T_inj=Twater + 273.15, Cp_w=c_w, lambda_w=K_w, lambda_R=K_f, rho_R=ro_f,
                     Cp_R=c_f)


def dT_friction() -> float:
    from paraphin.constants import Pw, Po, ro_w, c_w
    return (Pw - Po) / (ro_w * c_w)


def cases() -> dict:
    return {f'thermal_n{n}': c.mrst_case(physics(), 'thermal', n, DAYS, DT, FIELD_DAYS) for n in GRIDS}


@pytest.fixture(scope='module')
def result():
    from paraphin.constants import Twater
    runs = c.load_reference(TEST, physics())
    runs.update(c.paraphin_runs({f'paraphin_n{n}_thermal': (
        f'mrst_thermal_n{n}', {'Nx, Ny': f'{n}, {n}', **PATCH},
        {'days': DAYS, 'field_days': FIELD_DAYS, 'T_inj': Twater}) for n in GRIDS}))
    d = c.report(TEST, runs, {f'thermal_n{n}': f'paraphin_n{n}_thermal' for n in GRIDS}, FIELD_DAYS)
    for n in GRIDS:
        names = [f'paraphin_n{n}_thermal', f'thermal_n{n}']
        for name in names:  # температура для рисунков - в °C
            runs[name]['T_prod_C'] = runs[name]['T_prod'] - 273.15
            runs[name]['T_C'] = runs[name]['T'] - 273.15
        c.plot_series(f'{TEST}_rates_n{n}.png', [('Приемистость нагнетательной, м³/сут', 'q_inj', names, 1.0),
                                                 ('Температура на добывающей, °C', 'T_prod_C', names, 1.0)], runs)
        fields = [('T_C', 1.0, list(range(25, 70, 5)), 'T, °C'), ('p', 1e-5, list(range(55, 150, 5)), 'p, бар')]
        c.plot_isolines(f'{TEST}_isolines_n{n}.png', runs, names[0], names[1], n, FIELD_DAYS[:3], fields,
                        f'{n}x{n}, закачка воды 20 °C в пласт 70 °C')
        c.plot_profiles(f'{TEST}_profiles_n{n}.png', runs, names, n, FIELD_DAYS[:3],
                        [('T_C', 1.0, 'T, °C'), ('p', 1e-5, 'p, бар')], f'{n}x{n}')
        c.plot_maps(f'{TEST}_maps_n{n}.png', runs, names[0], names[1], n, FIELD_DAYS.index(250),
                    [('T_C', 1.0, 'Oranges_r', 'T, °C'), ('p', 1e-5, 'Purples', 'p, бар')], f'{n}x{n}, 250 сут')
    return runs, d


def test_injectivity(result):
    """Приемистость: вязкость воды растет с охлаждением одинаково у обоих (Андраде)."""
    c.assert_limits(result[1], {'q_inj': 0.05})


def test_production_temperature(result):
    """Температура на добывающей: расхождение не больше нагрева трением у MRST и запаса на схему."""
    c.assert_limits(result[1], {'T_prod_max': dT_friction() + 1.0})


def test_temperature_field(result):
    """Поле температуры, [K]: среднее расхождение и наибольшее (у фронта - сдвиг на ячейку дает десятки K)."""
    c.assert_limits(result[1], {'dT_mean': 1.0, 'dT_max': 15.0})


def test_pressure_field(result):
    c.assert_limits(result[1], {'dp_rms': 2.0, 'dp_max': 4.0})


def test_frictional_heating(result):
    """До прихода холодного фронта (50 сут) добывающая у paraphin ровно при init_T, у MRST - теплее, но не больше
    чем на нагрев трением: так видно, что расхождение по температуре - физика MRST, а не схема."""
    runs, _ = result
    from paraphin.constants import init_T
    for n in GRIDS:
        heat = runs[f'thermal_n{n}']['T_prod'][49] - (init_T + 273.15)
        assert abs(runs[f'paraphin_n{n}_thermal']['T_prod'][49] - (init_T + 273.15)) < 1e-6
        assert 0.0 < heat <= dT_friction(), f'{n}x{n}: нагрев на добывающей в MRST {heat:.3f} K'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mrst', action='store_true', help='снять эталон MRST в Octave')
    if parser.parse_args().mrst:
        c.save_reference(TEST, physics(), cases())
