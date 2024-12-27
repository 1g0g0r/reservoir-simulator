"""Модуль запуска всего расчета."""
from sys import stdout

import numpy as np
from tqdm import tqdm

from paraphin.constants import Time_end, dt, sol_time_step, day_to_sec
from paraphin.solver import Solver


def solve():
    """Запуск расчета."""
    times = np.linspace(0, Time_end, int(Time_end / dt + 1))
    pbar = tqdm(iterable=times, ncols=110, desc='Парафин считается', file=stdout)

    iter = 0
    sol = Solver()
    sol.initialize()  # Задание начальных условий
    for t in times:
        sol.upd_time_step(t / day_to_sec, iter)
        pbar.set_postfix(день=t / day_to_sec)
        pbar.update(1)

        if t >= iter * sol_time_step:  # or np.isclose(t, Time_end):
            sol.save_results(t)
            iter += 1


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
