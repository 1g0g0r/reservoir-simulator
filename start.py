"""Модуль запуска всего расчета."""
from sys import stdout
from time import perf_counter

import numpy as np
from tqdm import tqdm

from paraphin.constants import Time_end, dt, day_to_sec, Pw, Po, rw, Twater, Ny, Nx
from paraphin.solver import Solver


def solve():
    """Запуск расчета."""
    times = np.linspace(0, Time_end, int(Time_end / dt + 1))
    solution = Solver()

    # Создание скважин
    solution.add_well(name='Injector', i=0,    j=0,    p=Pw, rw=rw, is_injector=True, T=Twater)
    solution.add_well(name='Producer', i=Nx-1, j=Ny-1, p=Po, rw=rw, is_injector=False)

    tt = perf_counter()
    solution.initialize()  # Задание начальных условий из файла const.py
    solution.upd_time_step(0)  # При первом запуске компилируются модули
    print('Время компиляции:', perf_counter() - tt)

    with tqdm(iterable=times[1:], ncols=90, desc='Решение задачи', file=stdout, smoothing=0.05,
              bar_format="{l_bar}{bar}[{elapsed}/{remaining}]  {n_fmt}/{total_fmt}{postfix}   ") as pbar:
        for t in pbar:
            solution.upd_time_step(t)
            pbar.set_postfix(день=t / day_to_sec)
            # if t >= day_to_sec * 1679.:  #solution.wells[1].eta >= 0.97:
            #     print('KIN:', solution.KIN)
            #     break
            # ti.profiler.print_kernel_profiler_info()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
