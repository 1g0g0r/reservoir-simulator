"""Модуль запуска всего расчета."""
from sys import stdout

import numpy as np
from tqdm import tqdm

from paraphin.constants import Time_end, dt, day_to_sec, Pw, Po, Twater, Ny, Nx, outputs_path
from paraphin.solver import Solver


def profile_with_snakeviz(func):
    """For launch, you should install Snakeviz and enter the command in the terminal: snakeviz plugin.prof"""
    import cProfile
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profile_file = outputs_path / f"{func.__name__}.prof"
        profiler.dump_stats(profile_file)
        return result
    return wrapper


# @profile_with_snakeviz
def solve():
    """Запуск расчета."""
    times = np.linspace(0, Time_end, int(Time_end / dt + 1))
    pbar = tqdm(iterable=times, ncols=90, desc='Решение задачи', file=stdout, smoothing=0)

    sol = Solver()
    sol.initialize()  # Задание начальных условий из файла const.py

    # Создание скважин
    sol.add_well(name='inj',  i=0,    j=0,    p=Pw, is_injector=True, T=Twater)
    sol.add_well(name='prod', i=Nx-1, j=Ny-1, p=Po, is_injector=False)

    for t in times:
        sol.upd_time_step(t)
        pbar.set_postfix(день=t / day_to_sec)
        pbar.update()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
