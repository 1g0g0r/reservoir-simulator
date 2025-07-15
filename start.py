"""Модуль запуска всего расчета."""
from paraphin.constants import Pw, Po, rw, Twater, Ny, Nx
from paraphin.solver import Solver


def solve():
    """Запуск расчета."""
    solution = Solver()

    # Создание скважин
    solution.add_well(name='Injector', i=0,    j=0,    p=Pw, rw=rw, is_injector=True, T=Twater)
    solution.add_well(name='Producer', i=Nx-1, j=Ny-1, p=Po, rw=rw, is_injector=False)

    solution.start()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
