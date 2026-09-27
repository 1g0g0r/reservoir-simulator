"""Проверки шага по времени целиком: законы сохранения и отсутствие гонок в prange.

Все проверки короткие по модельному времени, но гоняют полный `upd_time_step`, поэтому ловят то,
чего не видит поячеечная проверка уравнения давления: рассогласование матрицы со скважинами,
утечку через границы и порчу общих буферов параллельным циклом.
"""
import numpy as np

from paraphin.constants import Nx, Ny, Pw, Po, rw, day_to_sec, bar_to_pa, ro_o, ro_p, volume
from paraphin.solver import Solver

# Время, а не число шагов, потому что dt в constants.py меняют.
T_TEST = 100.0 * day_to_sec

# Температура закачки берется не из constants.py, а заведомо ниже точки помутнения (при штатных
# init_Wp = 0.25 и эффективных Tm, alpha она равна ~39 C по (6.1)-(6.2)). Иначе парафин не выпадает,
# блок кольматации не вызывается вовсе, и проверки порового баланса и гонок в prange становятся
# пустыми: проверять было бы нечего.
T_INJECTION = 5.0

# Заданный дебит для проверки режима: того же порядка, что дает штатная постановка на забойных
# давлениях (~78 м^3/сут), иначе перепад получается нефизичным и проверять нечего.
Q_SET = 50.0 / day_to_sec


def _make_solver():
    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25, is_injector=True, T=T_INJECTION)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.initialize()
    solver.upd_time_step(0.0)

    return solver


def _run(t_end=T_TEST):
    """Прогон до t_end. Возвращает поля на конце и худшие невязки двух тождеств за прогон.

    Тождества: закачка равна отбору (правая часть уравнения давления - только дебиты) и
    (q_p1 + q_p2)*dt = m - m^new (нормировка скоростей потери порового объема).
    """
    solver = _make_solver()
    t, worst_q, worst_m = 0.0, 0.0, 0.0
    while t < t_end:
        step_dt = solver.dt
        m_before = solver.m.copy()
        t += step_dt
        solver.upd_time_step(t)

        # Дебиты знаковые: q > 0 - закачка, q < 0 - отбор
        q_in, q_out = solver.wells[0].q[2], -solver.wells[1].q[2]
        worst_q = max(worst_q, abs((q_out - q_in) / q_in))

        dm = m_before - solver.m
        residual = np.abs((solver.qp1 + solver.qp2) * step_dt - dm).max()
        worst_m = max(worst_m, residual / max(np.abs(dm).max(), 1e-30))

    return solver, worst_q, worst_m


def test_mass_balance():
    """Границы непроницаемы, флюиды несжимаемы: отбор обязан в точности равняться закачке.

    Правая часть уравнения давления содержит только дебиты - производные по времени сокращаются
    при сложении балансов фаз, а не баланса воды с балансом отдельного компонента. Скважины при
    этом неявны по давлению, поэтому равенство суммарных дебитов выполняется тождественно.
    Расхождение означает, что матрица давления разошлась со скважинами или решатель не сошелся.
    """
    _, worst_q, _ = _run()
    assert worst_q < 1e-6, f'баланс закачки и отбора не сходится: относительная невязка {worst_q:.3e}'


def test_pore_volume_balance():
    """(q_p1 + q_p2)*dt = m - m^new: скорости потери порового объема нормированы на убыль пористости.

    Тождество связывает пористость с обеими скоростями: осадок на стенках и пробки в горлах
    блокированных каналов - кристаллы, вместе они дают ровно то, что теряет поровое пространство
    (объем блокированного канала остается тупиковой пористостью). На него опираются и уравнение
    насыщенности, и уравнение баланса парафина.
    """
    solver, _, worst_m = _run()
    assert solver.Wps.max() > 0.0, 'парафин не выпал: проверка порового баланса ничего не проверяет'
    assert solver.m.min() < solver.m.max(), 'пористость не изменилась: кольматация не запускалась'
    assert worst_m < 1e-10, f'сумма q_p1 + q_p2 разошлась с убылью пористости: {worst_m:.3e}'
    assert solver.fi.min() >= 0.0, f'функция пор по размерам ушла в минус: {solver.fi.min():.3e}'


def test_paraffin_mass_balance():
    """Глобальный баланс массы парафина: начальная масса = в фазе + осело + добыто.

    Осевший парафин складывается из стоков `wp_equation` теми же множителями, что стоят в ней:
    осадок q_p1 и пробки q_p2 - чистый парафин (ro_p). Нефть блокированных каналов остается в
    пористости и считается в `in_place`. Добытый - через дебит нефтяной фазы добывающей скважины и
    долю парафина в ее ячейке на начало шага, как в `_wells_loop`. Закачивается только вода, приток
    парафина извне нулевой.

    Схема консервативна, единственный неконсервативный элемент - зажим доли в нуле в `wp_equation`:
    он срабатывает, когда сток за шаг превышает запас, и создает массу. Именно его ловит проверка,
    вместе с лагом стоков (q_p1, q_p2 прошлого слоя вместо нового).
    """
    solver = _make_solver()
    producer = solver.wells[solver._producer]
    i_p, j_p = producer.i, producer.j

    def in_place():
        return (solver.m * (1.0 - solver.S) * (solver.Wp + solver.Wps)).sum() * ro_o * volume

    mass_0 = in_place()
    deposited, produced, t = 0.0, 0.0, 0.0
    while t < T_TEST:
        step_dt = solver.dt
        w_sum = solver.Wp + solver.Wps  # доли на начало шага - с ними считаются стоки и добыча
        t += step_dt
        solver.upd_time_step(t)

        # После обмена слоев qp1, qp2 - те, что дали убыль пористости на этом шаге
        deposited += (ro_p * (solver.qp1 + solver.qp2) * volume).sum() * step_dt
        produced += -producer.q[0] * ro_o * w_sum[i_p, j_p] * step_dt

    assert deposited > 0.0, 'парафин не осел: баланс ничего не проверяет'
    residual = abs(mass_0 - in_place() - deposited - produced) / mass_0
    assert residual < 1e-8, f'баланс массы парафина не сходится: относительная невязка {residual:.3e}'


def test_oil_phase_composition():
    """Замыкающее соотношение w_o + w_p + w_ps = 1 при неотрицательных долях.

    Разделение парафина на растворенный и взвешенный алгебраическое (предел растворимости), а не
    отдельное уравнение, поэтому соотношение обязано выполняться точно, а не накапливать невязку.
    """
    solver, _, _ = _run()
    worst = np.abs(solver.Wo + solver.Wp + solver.Wps - 1.0).max()
    assert worst < 1e-12, f'w_o + w_p + w_ps отличается от 1 на {worst:.3e}'
    assert solver.Wp.min() >= 0.0, f'отрицательная доля растворенного парафина: {solver.Wp.min():.3e}'
    assert solver.Wps.min() >= 0.0, f'отрицательная доля взвешенного парафина: {solver.Wps.min():.3e}'


def test_rate_control():
    """Режим заданного дебита: скважина выдает ровно заданный расход, забойное давление - результат.

    Дебит входит в правую часть уравнения давления известной величиной, поэтому равенство
    `q[2] == q_set` обязано выполняться с машинной точностью, а не с точностью решателя. Забойное
    давление восстанавливается из формулы Писмана и обязано быть выше пластового в ячейке -
    иначе закачка шла бы против перепада.
    """
    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, q=Q_SET, rw=rw, mult=0.25,
                    is_injector=True, T=T_INJECTION)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.initialize()
    solver.upd_time_step(0.0)

    t, worst_set, worst_bal = 0.0, 0.0, 0.0
    while t < 10.0 * day_to_sec:
        t += solver.dt
        solver.upd_time_step(t)
        worst_set = max(worst_set, abs((solver.wells[0].q[2] - Q_SET) / Q_SET))
        worst_bal = max(worst_bal, abs((-solver.wells[1].q[2] - Q_SET) / Q_SET))

    assert worst_set < 1e-14, f'заданный дебит не выдерживается: невязка {worst_set:.3e}'
    assert worst_bal < 1e-6, f'отбор разошелся с заданной закачкой: невязка {worst_bal:.3e}'

    p_bh, p_cell = solver.wells[0].p, solver.p[0, 0]
    assert p_bh > p_cell, (f'забойное давление нагнетательной {p_bh / bar_to_pa:.2f} бар не выше '
                           f'пластового {p_cell / bar_to_pa:.2f} бар')


def test_rate_control_producer():
    """Дебит задается положительным для любой скважины, знак ставит `add_well` по `is_injector`.

    Проверяется именно добывающая: у нее знак меняется на противоположный, то есть внутри
    `q[2] == -q_set`. Забойное давление обязано быть ниже пластового - иначе отбор шел бы
    против перепада.
    """
    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25,
                    is_injector=True, T=T_INJECTION)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, q=Q_SET, rw=rw, mult=0.25,
                    is_injector=False)
    solver.initialize()
    solver.upd_time_step(0.0)

    t, worst_set, worst_bal = 0.0, 0.0, 0.0
    while t < 10.0 * day_to_sec:
        t += solver.dt
        solver.upd_time_step(t)
        worst_set = max(worst_set, abs((-solver.wells[1].q[2] - Q_SET) / Q_SET))
        worst_bal = max(worst_bal, abs((solver.wells[0].q[2] - Q_SET) / Q_SET))

    assert worst_set < 1e-14, f'заданный отбор не выдерживается: невязка {worst_set:.3e}'
    assert worst_bal < 1e-6, f'закачка разошлась с заданным отбором: невязка {worst_bal:.3e}'

    p_bh, p_cell = solver.wells[1].p, solver.p[Nx - 1, Ny - 1]
    assert p_bh < p_cell, (f'забойное давление добывающей {p_bh / bar_to_pa:.2f} бар не ниже '
                           f'пластового {p_cell / bar_to_pa:.2f} бар')


def test_no_race_in_parallel_loop():
    """Два одинаковых прогона обязаны совпасть побитово.

    Единственный источник расхождения - гонка в `prange`: общий на все ячейки скратч-буфер, в
    который пишут разные потоки. Так была поймана прогонка для `fi` по общим `a_tdma`/`b_tdma`:
    поля расходились, `m` и `k` прыгали на порядки, давление уходило за забойные.

    Проверка имеет смысл только при включенном парафине: при `init_Wp + init_Wps = 0` весь блок
    кольматации не вызывается и параллельному циклу нечего делить.
    """
    a, _, _ = _run()
    b, _, _ = _run()
    for name in ('p', 'S', 'T', 'm', 'k', 'Wp', 'Wps', 'fi'):
        fa, fb = getattr(a, name), getattr(b, name)
        assert np.array_equal(fa, fb), (f'поле {name} разошлось между двумя одинаковыми прогонами, '
                                        f'максимум |разности| {np.abs(fa - fb).max():.3e}')


def test_neumann_pressure_bc():
    """Приток через границу с ненулевым Нейманом по давлению уходит в добывающую скважину без невязки.

    Фиктивная ячейка за границей - P + g*h/2 (`apply_bc`), поток g*A*lam от давления не зависит. Матрица
    давления обязана заложить тот же поток: раньше она брала g*val без множителя h/2, то есть приток в
    ~1/h раз больше того, что видят перетоки, и отбор скважины расходился с притоком через границу.
    """
    from paraphin.constants import hx, h
    from paraphin.utils import Bound, DataField, TypeBC

    grad = 1e5  # [Па/м], давление за левой границей выше - нефть и вода втекают
    solver = Solver()
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.add_bc(field=DataField.Pressure, bound=Bound.Left, type_bc=TypeBC.Neumann, value=grad)
    solver.initialize()
    solver.upd_time_step(0.0)

    inflow = grad * hx * h * float((solver.lam_o[:, 0] + solver.lam_w[:, 0]).sum())
    outflow = -solver.wells[0].q[2]
    assert abs(outflow - inflow) / inflow < 1e-6, f'отбор {outflow:.4e} не равен притоку через границу {inflow:.4e}'


def test_paraffin_inflow_bc():
    """ГУ Дирихле `DataField.Paraffin` задает состав втекающей нефти, а не берет его из самой ячейки.

    Без него керн, через который прокачивают нефть, получал бы на входе ту долю парафина, что осталась в
    первой ячейке после осаждения, то есть подвод взвеси иссякал бы сам собой.
    """
    from paraphin.constants import init_p, init_S, init_k, init_m, init_T, c_o, c_w, c_p, data_type
    from paraphin.equations import flows_in_cells
    from paraphin.oil_composition import NC
    from paraphin.utils import Bound, DataField, calc_mu_o, calc_mu_w, calc_mobility

    def field(value):
        return np.full((Nx, Ny), value, data_type)

    w_cell, w_bc = 0.3, 0.1
    p, S, T, k, m = field(init_p), field(init_S), field(init_T), field(init_k), field(init_m)
    Wo, Wp, Wps = field(1.0 - w_cell), field(w_cell), field(0.0)
    mu_o, mu_w = field(calc_mu_o(init_T, 0.0)), field(calc_mu_w(init_T))
    lam_o, lam_w, lam_h = field(0.0), field(0.0), field(0.0)
    calc_mobility(k, S, m, Wo, Wp, Wps, mu_o, mu_w, lam_o, lam_w, lam_h)

    def inflow(w_boundary=None):
        """Приток парафина в угловую ячейку через левую границу с повышенным давлением."""
        bc = np.zeros((4, 4, 2), data_type)
        bc[Bound.Left.value, DataField.Pressure.value] = (1, init_p + bar_to_pa)
        if w_boundary is not None:
            bc[Bound.Left.value, DataField.Paraffin.value] = (1, w_boundary)
        cells_wp = field(0.0)
        # bc_Wc и Fo_row нужны только детальному составу (`wax_components`); носитель скрытой теплоты - сам Wp
        flows_in_cells(0, 0, bc, np.zeros((4, NC), data_type), p, S, T, k, mu_o, mu_w, lam_o, lam_w, lam_h, m,
                       Wo, Wp, Wps, Wp, field(c_o), field(c_w), field(c_p), field(0.0), cells_wp, field(0.0),
                       field(0.0), np.zeros(4, data_type))
        return cells_wp[0, 0]

    assert inflow() > 0.0, 'через левую границу с повышенным давлением нефть обязана втекать'
    assert np.isclose(inflow(w_bc), inflow() * w_bc / w_cell, rtol=1e-12), \
        'доля парафина во втекающей нефти взята не из ГУ'


if __name__ == '__main__':
    test_mass_balance()
    test_pore_volume_balance()
    test_oil_phase_composition()
    test_rate_control()
    test_rate_control_producer()
    test_no_race_in_parallel_loop()
    test_paraffin_inflow_bc()
    test_neumann_pressure_bc()
    print('OK')
