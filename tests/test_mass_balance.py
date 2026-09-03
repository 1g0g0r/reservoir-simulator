"""Проверки шага по времени целиком: закон сохранения массы и отсутствие гонок в prange.

Обе проверки короткие по модельному времени, но гоняют полный `upd_time_step`, поэтому ловят то,
чего не видит поячеечная проверка уравнения давления: рассогласование матрицы со скважинами,
утечку через границы и порчу общих буферов параллельным циклом.
"""
import numpy as np

from paraphin.constants import Nx, Ny, Pw, Po, rw, Twater, volume, day_to_sec
from paraphin.solver import Solver

# Гнать надо до момента, когда парафин начнет осаждаться (около 80 суток на штатной постановке):
# до этого блок кольматации не вызывается и проверять в нем нечего. Время, а не число шагов,
# потому что dt в constants.py меняют.
T_TEST = 100.0 * day_to_sec


def _make_solver():
    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25, is_injector=True, T=Twater)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.initialize()
    solver.upd_time_step(0.0)

    return solver


def _run(t_end=T_TEST):
    """Прогон до t_end. Возвращает поля на конце и худшую невязку баланса скважин за прогон."""
    solver = _make_solver()
    t, worst = 0.0, 0.0
    while t < t_end:
        step_dt = solver.dt
        # Источник в правой части уравнения давления, ровно в той форме, в какой его собирает
        # _fill_matrix_and_rhs. Считается до шага: после него m_0 и Wo_0 уже сдвинуты.
        src = -((solver.m - solver.m_0)
                + solver.m_0 * solver.S_0 * (solver.Wo - solver.Wo_0) / solver.Wo).sum() / step_dt * volume
        t += step_dt
        solver.upd_time_step(t)
        q_in = -solver.wells[0].q[2]  # дебит нагнетательной отрицателен
        q_out = solver.wells[1].q[2]
        worst = max(worst, abs((q_out - q_in - src) / q_in))

    return solver, worst


def test_mass_balance():
    """Границы непроницаемы, флюиды несжимаемы: отбор минус закачка равен источнику в правой части.

    При выключенном парафине источник тождественно нулевой и дебиты обязаны просто совпасть.
    Расхождение означает, что матрица давления разошлась со скважинами или решатель не сошелся.
    """
    _, worst = _run()
    assert worst < 1e-6, f'баланс закачки и отбора не сходится: относительная невязка {worst:.3e}'


def test_no_race_in_parallel_loop():
    """Два одинаковых прогона обязаны совпасть побитово.

    Единственный источник расхождения - гонка в `prange`: общий на все ячейки скратч-буфер, в
    который пишут разные потоки. Так была поймана прогонка для `fi` по общим `a_tdma`/`b_tdma`:
    поля расходились, `m` и `k` прыгали на порядки, давление уходило за забойные.

    Проверка имеет смысл только при включенном парафине: при `init_Wp + init_Wps = 0` весь блок
    кольматации не вызывается и параллельному циклу нечего делить.
    """
    a, _ = _run()
    b, _ = _run()
    for name in ('p', 'S', 'T', 'm', 'k', 'Wp', 'Wps', 'fi'):
        fa, fb = getattr(a, name), getattr(b, name)
        assert np.array_equal(fa, fb), (f'поле {name} разошлось между двумя одинаковыми прогонами, '
                                        f'максимум |разности| {np.abs(fa - fb).max():.3e}')


if __name__ == '__main__':
    test_mass_balance()
    test_no_race_in_parallel_loop()
    print('OK')
