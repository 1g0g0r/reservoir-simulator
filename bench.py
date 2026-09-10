"""Замер стадий шага по времени. Базовая линия для правок, см. PERFORMANCE_FINDINGS.md.

Запуск: python bench.py [число_шагов] [прогрев_суток]

Стадии меряются обертками вокруг функций в `paraphin.solver`, то есть внутри настоящего шага,
а не повторными вызовами из Python на замороженном состоянии: списки аргументов у них длинные и
меняются, дублировать их здесь - гарантированно отстать от кода. Побочный эффект: в стадию входит
упаковка jitclass со скважинами на границе Python/numba, которой в самом шаге нет.

Прогрев в сутках нужен для режима с парафином: кольматация включается примерно с 80 суток, до этого
блок не вызывается и профиль показывает не то, что интересно.
"""
from collections import defaultdict
from statistics import median
from sys import argv
from time import perf_counter

import paraphin.solver as solver_module
from paraphin.constants import Nx, Ny, Nr, init_Wp, init_Wps, Pw, Po, rw, Twater, day_to_sec
from paraphin.solver import Solver

_STAGES = ('calc_mobility', 'calc_pressure', '_update_wells_data', '_wells_loop',
           '_equations_loop', '_calc_dt', '_swap_time_steps', 'save_fields')
_NAMES = {'calc_mobility': 'подвижности', 'calc_pressure': 'давление (сборка + СЛАУ)',
          '_update_wells_data': 'данные скважин', '_wells_loop': 'скважины в уравнениях',
          '_equations_loop': 'явные уравнения по ячейкам', '_calc_dt': 'подбор шага по CFL',
          '_swap_time_steps': 'обмен временных слоев', 'save_fields': 'сохранение полей'}


def _make_solver() -> Solver:
    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25, is_injector=True, T=Twater)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.initialize()

    return solver


def _instrument(acc: dict) -> None:
    """Обертка-таймер на каждую стадию шага."""
    for name in _STAGES:
        origin = getattr(solver_module, name)

        def timed(*args, _f=origin, _n=name, **kwargs):
            start = perf_counter()
            result = _f(*args, **kwargs)
            acc[_n].append((perf_counter() - start) * 1e6)

            return result

        setattr(solver_module, name, timed)


def main(n_steps: int = 200, warmup_days: float = 0.0) -> None:
    acc = defaultdict(list)
    _instrument(acc)

    start = perf_counter()
    solver = _make_solver()
    solver.upd_time_step(0.0)
    print(f'сетка {Nx}x{Ny}, Nr={Nr}, init_Wp={init_Wp}, init_Wps={init_Wps}, '
          f'парафин {"включен" if solver._paraphin else "выключен"}')
    print(f'компиляция и прогрев: {perf_counter() - start:.2f} c')

    t = 0.0
    while t < warmup_days * day_to_sec:
        t += solver.dt
        solver.upd_time_step(t)

    acc.clear()
    steps = []
    for _ in range(n_steps):
        t += solver.dt
        start = perf_counter()
        solver.upd_time_step(t)
        steps.append((perf_counter() - start) * 1e6)

    avg = sum(steps) / len(steps)
    print(f'модельное время {t / day_to_sec:.1f} сут, шагов {n_steps}\n')
    print(f'{"стадия":<30}{"медиана":>10}{"на шаг":>10}{"доля":>8}')
    print('-' * 58)
    for name in _STAGES:
        if not acc[name]:
            continue
        vals = acc[name]
        print(f'{_NAMES[name]:<30}{median(sorted(vals)):>10.1f}{sum(vals) / n_steps:>10.1f}'
              f'{sum(vals) / n_steps / avg * 100:>7.1f}%')
    print('-' * 58)
    print(f'{"шаг целиком":<30}{median(sorted(steps)):>10.1f}{avg:>10.1f}{100.0:>7.1f}%')


if __name__ == '__main__':
    main(int(argv[1]) if len(argv) > 1 else 500,
         float(argv[2]) if len(argv) > 2 else 0.0)
