"""Модуль запуска всего расчета."""
from sys import stdout

import numpy as np
from tqdm import tqdm

from paraphin.constants import Time_end, dt, sol_time_step
from paraphin.solver import Solver
from paraphin.utils.vizualization import visualize_solution


def solve():
    """Запуск расчета."""
    iter = 0
    sol = Solver()
    sol.initialize()  # Задание начальных условий

    times = np.linspace(0, Time_end, int(Time_end / dt + 1))
    for t in tqdm(iterable=times, ncols=100, desc='Парафин считается', file=stdout):
        sol.upd_time_step(t)

        if t >= iter * sol_time_step or np.isclose(t, Time_end):
            sol.save_results(t)
            iter += 1


if __name__ == '__main__':
    solve()
    visualize_solution()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
