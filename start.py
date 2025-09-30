"""Модуль запуска всего расчета."""
# Идея использовать taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
from paraphin.constants import Pw, Po, rw, Twater, Ny, Nx, S_max, day_to_sec
from paraphin.solver import Solver, Bound, TypeBC, DataField


def solve():
    """Запуск расчета."""
    solver = Solver()

    # Добавление скважин
    solver.add_well(name='Injector', i=0,    j=0,    p=Pw, rw=rw, mult=0.25, is_injector=True, T=Twater)
    solver.add_well(name='Producer', i=Nx-1, j=Ny-1, p=Po, rw=rw, mult=0.25, is_injector=False)

    # Добавление ГУ на границе
    # solver.add_bc(field=DataField.Pressure,    bound=Bound.Left,  type_bc=TypeBC.Dirichlet, value=Pw)
    # solver.add_bc(field=DataField.Pressure,    bound=Bound.Right, type_bc=TypeBC.Dirichlet, value=Po)
    # solver.add_bc(field=DataField.Saturation,  bound=Bound.Left,  type_bc=TypeBC.Dirichlet, value=S_max)
    # solver.add_bc(field=DataField.Temperature, bound=Bound.Left,  type_bc=TypeBC.Dirichlet, value=Twater)

    solver.start()


if __name__ == '__main__':
    solve()
