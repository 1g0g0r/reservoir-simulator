"""Модуль запуска всего расчета."""
from sys import stdout

import numpy as np
from tqdm import tqdm

from paraphin.constants import Time_end, dt, day_to_sec, Pw, Po, rw, Twater, Ny, Nx
from paraphin.solver import Solver


def solve():
    """Запуск расчета."""
    n_times = int(Time_end / dt + 1)
    times = np.linspace(0, Time_end, n_times)
    solution = Solver()

    # Создание скважин
    solution.add_well(name='Injector', i=0,    j=0,    p=Pw, rw=rw, is_injector=True, T=Twater)
    solution.add_well(name='Producer', i=Nx-1, j=Ny-1, p=Po, rw=rw, is_injector=False)

    solution.initialize()  # Задание начальных условий из файла const.py
    solution.upd_time_step(0)  # при первом запуске компилируются модули

    with tqdm(iterable=times[1:], ncols=90, desc='Решение задачи', file=stdout, smoothing=0.05,
              bar_format="{l_bar}{bar}[{elapsed}/{rate_fmt}]  {n_fmt}/{total_fmt}{postfix}   ") as pbar:
        for t in pbar:
            solution.upd_time_step(t)
            pbar.set_postfix(день=t / day_to_sec)


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
