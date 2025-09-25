"""Модуль запуска всего расчета."""
from paraphin.constants import Pw, Po, rw, Twater, Ny, Nx
from paraphin.solver import Solver, Bound, TypeBC, DataField


def solve():
    """Запуск расчета."""
    solution = Solver()

    # Создание скважин
    # solution.add_well(name='Injector', i=0,    j=0,    p=Pw, rw=rw, mult=0.25, is_injector=True, T=Twater)
    # solution.add_well(name='Producer', i=Nx-1, j=Ny-1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solution.add_bc(DataField.Pressure,    Bound.Left, TypeBC.Dirichlet, Pw)
    solution.add_bc(DataField.Saturation,  Bound.Left, TypeBC.Dirichlet, 1)
    solution.add_bc(DataField.Temperature, Bound.Left, TypeBC.Dirichlet, Twater)

    solution.start()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
