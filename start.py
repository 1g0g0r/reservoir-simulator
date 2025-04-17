"""Модуль запуска всего расчета."""
from sys import stdout

import numpy as np
from tqdm import tqdm

from paraphin.constants import Time_end, dt, day_to_sec, Pw, Po, Twater, Ny, Nx
from paraphin.solver import Solver


def solve():
    """Запуск расчета."""
    times = np.linspace(0, Time_end, int(Time_end / dt + 1))
    pbar = tqdm(iterable=times, ncols=70, desc='Решение задачи', file=stdout)

    sol = Solver()
    # Задание начальных условий из файла const.py
    sol.initialize()
    # Создание скважин
    sol.add_well(name='inj',  i=0,    j=0,    p=Pw, type_well='inj', T=Twater)
    sol.add_well(name='prod', i=Nx-1, j=Ny-1, p=Po, type_well='prod')

    for t in times:
        sol.upd_time_step(t)
        pbar.set_postfix(день=t / day_to_sec)
        pbar.update()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
